//! Ordinary Differential Equation (ODE) solver.
//!
//! Solves first-order and second-order ODEs:
//!
//! - **Full separable:** `dy/dx = f(x) * g(y)` → `∫ 1/g(y) dy = ∫ f(x) dx`,
//!   solved for `y` when that has one solution
//! - **Simple separable:** `a(x)·dy/dx = f(x)` (no y dependence)
//! - **First-order linear (variable coefficient):** `a(x)·y' + P(x)*y = Q(x)`
//!   → `y = (1/μ) * [∫ Q(x)*μ dx + C1]` where `μ = exp(∫ P(x) dx)`
//!   (after dividing by `a`)
//! - **First-order linear constant-coefficient:** `y' + a*y = f(x)`
//!   → `y = e^(-ax) * ∫ f(x)*e^(ax) dx`
//! - **Exact** `M + N·y' = 0` (`∂M/∂y = ∂N/∂x`, decided as a rational
//!   function) and with an **integrating factor** `μ(x)` or `μ(y)`
//! - **Bernoulli:** `y' + P·y = Q·yⁿ`
//! - **Second-order linear constant-coefficient:** `y'' + b*y' + c*y = 0`
//!   → characteristic equation `r² + b*r + c = 0`, solution based on roots
//! - **Cauchy–Euler:** `a·x²y'' + b·x·y' + c·y = g(x)` (for `x > 0`; a
//!   forcing term through `x = eᵗ` and the constant-coefficient solvers)
//! - **Variation of parameters** for `y'' + b·y' + c·y = g(x)`
//! - **Homogeneous coefficient:** `y' = f(y/x)` — substitution `v = y/x`
//! - **nth-order reducible:** `F(y, y', y'') = 0` (no `x`) — substitution
//!   `p(y) = y'`; `F(x, y', y'') = 0` (no `y`) — `p(x) = y'`, `y = ∫ p dx`
//! - **nth-order linear constant-coefficient:** `Σ a_k y^(k) = g(x)` for any
//!   order via the characteristic polynomial (repeated roots → `x^k e^{rx}`,
//!   complex pairs → `e^{ax}(cos bx, sin bx)`), with undetermined coefficients
//!   for `poly × exp × {sin, cos}` forcing (resonance handled)
//! - **Clairaut:** `y = x·y' + f(y')` → `y = C·x + f(C)`
//! - **Riccati:** `y' = q₀ + q₁·y + q₂·y²` given a particular solution
//!   ([`solve_riccati`])
//! - **Constant-coefficient systems:** `ẋ = A·x` → `x(t) = exp(A·t)·c`
//!   via the eigenvectors (real modes for complex pairs) or the exact matrix
//!   exponential of the Jordan form, with initial values via
//!   [`solve_ode_system_ivp`]
//! - **Non-homogeneous systems:** `ẋ = A·x + b(t)` → variation of parameters
//!
//! # Implicit solutions
//!
//! A solution that could not be solved for `y` is returned as an expression
//! `G(x, y, C1, …)` whose zero set is the family (`G = 0`); it still
//! contains `y`.  [`dsolve`] prefers an explicit solution found by a later
//! method to an implicit one found by an earlier one.
//!
//! # Branches
//!
//! Where the general solution has several families, [`OdeResult::solution`]
//! is the first and [`OdeResult::branches`] holds the others, as SymPy's
//! list of solutions: a Bernoulli equation whose `1/(1−n)` has an even
//! denominator gives `y = ±v^{1/(1−n)}` with `v` the solution of the linear
//! equation (`y' = y³`: `(C1 − 2x)^{−1/2}` and `−(C1 − 2x)^{−1/2}`), a
//! separable equation every explicit root of `∫ dy/g(y) = ∫ f dx + C1`, and
//! `y'' = F(y')` (no `y`) one family per family of `y'` (`y'' = y'³`:
//! `C2 ∓ √(C1 − 2x)`).  Before 0.30 only the first family was returned.
//! [`Ex::solve_ode`](crate::api::expr::Ex::solve_ode) returns the first
//! family, `Ex::solve_ode_all` all of them, and `Ex::solve_ode_ivp` fits the
//! initial conditions on each family in turn (`y' = y³`, `y(0) = −1` is on
//! the negative one).
//!
//! A formula is a solution where it is real on the principal branch:
//! `y' = √y` gives `(x/2 + C1/2)²`, a solution for `x + C1 ≥ 0` only (for
//! `x + C1 < 0` its derivative is negative while `√y ≥ 0`; the global
//! solution is `0` there, the parabola after), and the singular solution
//! `y = 0` is not included; `solve_ode_ivp` keeps only constants for which
//! the ODE holds at the initial point (`y(0) = 1` gives `C1 = 2`, not the
//! other root `C1 = −2`).  Cauchy–Euler solutions are for `x > 0` (`ln x`,
//! `x^r`).

use crate::api::expr::Ex;
use crate::base::arena::Arena;
use crate::base::combinatorics::binomial;
use crate::base::node::{ExprId, ExprNode, SymbolId};
use crate::base::numeric::Q;
use crate::domains::matrix::Matrix;
use num_traits::One;
use num_traits::Signed;

/// Most families [`try_full_separable`] returns (roots of `∫ dy/g(y) =
/// ∫ f dx + C1`); more stay implicit.
const MAX_BRANCHES: usize = 8;

/// An ODE representation: f(x, y, y', y'', ...) = 0
/// For now, we support limited forms detected by pattern matching.
pub struct OdeResult {
    /// The general solution y = ... (may contain constants C1, C2)
    pub solution: ExprId,
    /// Names of the arbitrary constants
    pub constants: Vec<ExprId>,
    /// The other families of the general solution, in the same constants:
    /// with `solution` they form the complete set (`y′ = y³` has
    /// `(C1 − 2x)^(−1/2)` and `−(C1 − 2x)^(−1/2)`).  Empty when `solution`
    /// is the whole family.  See *Branches* in the module docs.
    pub branches: Vec<ExprId>,
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

    // Try to detect the ODE type.  An implicit solution (one still in
    // `func`) is kept while the later methods get a chance to find an
    // explicit one: the integrating factor of `y′ = y − y²` gave
    // `x − ln|y| + ln|y − 1| − C1`, Bernoulli gives `y` itself.
    let mut implicit: Option<OdeResult> = None;
    macro_rules! attempt {
        ($result:expr) => {
            if let Some(result) = $result {
                if contains_sym(arena, result.solution, func_sym) {
                    if implicit.is_none() {
                        implicit = Some(result);
                    }
                } else {
                    return Some(result);
                }
            }
        };
    }

    // Type 1: Second-order constant-coefficient: a*y'' + b*y' + c*y = 0
    attempt!(try_second_order_const_coeff(
        arena, expr, func, var, func_sym, var_sym
    ));

    // Type 1b: Second-order CC nonhomogeneous: a*y'' + b*y' + c*y = f(x)
    attempt!(try_second_order_cc_nonhomogeneous(
        arena, expr, func, var, func_sym, var_sym
    ));

    // Type 1c: Euler-Cauchy: a·x²·y'' + b·x·y' + c·y = 0
    attempt!(try_euler_cauchy(arena, expr, func, var, func_sym, var_sym));

    // Type 1e: nth-order linear constant-coefficient (any order ≥ 2),
    // forcing = poly × exp × {sin, cos} via undetermined coefficients.
    attempt!(try_nth_order_linear_const_coeff(
        arena, expr, func, var, func_sym
    ));

    // Type 1d: Variation of parameters: y'' + p·y' + q·y = g(x) (fallback)
    attempt!(try_variation_of_parameters(
        arena, expr, func, var, func_sym, var_sym
    ));

    // Type 1f: Clairaut: y = x·y' + f(y')
    attempt!(try_clairaut(arena, expr, func, var, func_sym, var_sym));

    // Type 2: General first-order linear (variable P(x)): y' + P(x)*y = Q(x)
    attempt!(try_first_order_linear_general(
        arena, expr, func, var, func_sym, var_sym
    ));

    // Type 2b: Exact first-order ODE: M(x,y) + N(x,y)·y' = 0 with ∂M/∂y = ∂N/∂x
    attempt!(try_exact_ode(arena, expr, func, var, func_sym, var_sym));

    // Type 2c: Non-exact ODE with integrating factor μ(x) or μ(y)
    attempt!(try_integrating_factor_ode(
        arena, expr, func, var, func_sym, var_sym
    ));

    // Type 2d: Bernoulli: y' + P(x)·y = Q(x)·y^n (n ≠ 0, 1)
    attempt!(try_bernoulli(arena, expr, func, var, func_sym, var_sym));

    // Type 2e: Homogeneous coefficient: y' = f(y/x)
    attempt!(try_homogeneous_coefficient(
        arena, expr, func, var, func_sym, var_sym
    ));

    // Type 2f: nth-order reducible: F(y, y', y'') = 0, no explicit x
    attempt!(try_nth_order_reducible(
        arena, expr, func, var, func_sym, var_sym
    ));

    // Type 3: Full separable: y' = f(x)*g(y)
    attempt!(try_full_separable(
        arena, expr, func, var, func_sym, var_sym
    ));

    // Type 4: Simple separable: y' = f(x) (no y dependence) — fallback
    attempt!(try_simple_separable(
        arena, expr, func, var, func_sym, var_sym
    ));

    implicit
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
        // a(x)·y′ + f(x) = 0 with a(x) free of y (before, only a = 1:
        // `2y′ = e⁻ˣ` and `(1 + x²)y′ = x²` were not solved at all).
        let mut deriv_coeffs: Vec<ExprId> = Vec::new();
        let mut other_terms: Vec<ExprId> = Vec::new();

        for &child in children {
            let (coeff, term) = arena.as_coeff_term(child);
            if term == dy_dx {
                deriv_coeffs.push(ode_ratio_to_expr(arena, &coeff));
            } else if !contains_sym(arena, child, func_sym) {
                other_terms.push(child);
            } else if let ExprNode::Mul(factors) = arena.node(term).clone()
                && factors.iter().filter(|&&f| f == dy_dx).count() == 1
                && factors
                    .iter()
                    .all(|&f| f == dy_dx || !contains_sym(arena, f, func_sym))
            {
                let mut rest: Vec<ExprId> =
                    factors.iter().copied().filter(|&f| f != dy_dx).collect();
                rest.push(ode_ratio_to_expr(arena, &coeff));
                deriv_coeffs.push(arena.mul(&rest));
            } else {
                // Term contains y — not simple separable
                return None;
            }
        }

        if !deriv_coeffs.is_empty() {
            let a = arena.add(&deriv_coeffs);
            let a = crate::transforms::eval::eval(arena, a);
            if arena.is_zero_structural(a) {
                return None;
            }
            // a·y' + other_terms = 0 → y' = -other_terms/a → y = -∫ other_terms/a dx + C
            let rhs = if other_terms.is_empty() {
                arena.zero
            } else {
                let sum = arena.add(&other_terms);
                let neg = arena.neg(sum);
                let q = arena.div(neg, a);
                crate::transforms::eval::eval(arena, q)
            };
            let integral = integrate_forms(arena, rhs, var);
            let c1 = arena.symbol("C1");
            let solution = arena.add(&[integral, c1]);
            return Some(OdeResult {
                solution,
                constants: vec![c1],
                branches: Vec::new(),
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
            branches: Vec::new(),
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
    let roots = crate::transforms::solve::solve(arena, char_eq, r_var);

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
                    branches: Vec::new(),
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
                    branches: Vec::new(),
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
                branches: Vec::new(),
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
fn solve_characteristic_equation(arena: &mut Arena, b: Q, c: Q, var: ExprId) -> Option<OdeResult> {
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
    let roots = crate::transforms::solve::solve(arena, char_eq, r_var);

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
                    branches: Vec::new(),
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
                    branches: Vec::new(),
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
                branches: Vec::new(),
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
            let mut rhs_coeffs: Vec<Q> = Vec::new();
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
                find_particular_polynomial(&b, &c, &rhs_coeffs)
                    .map(|pc| build_polynomial_expr(arena, &pc, var))
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
        let rhs_expr = crate::transforms::eval::eval(arena, rhs_expr);
        try_undetermined_trig_exp(arena, rhs_expr, &b, &c, var)?
    };

    // Homogeneous solution via the characteristic equation
    let homo_result = solve_characteristic_equation(arena, b, c, var)?;

    // General solution = homogeneous + particular
    let solution = arena.add(&[homo_result.solution, y_p]);

    Some(OdeResult {
        solution,
        constants: homo_result.constants,
        branches: Vec::new(),
    })
}

/// Solve the undetermined-coefficients linear system for a polynomial
/// particular solution of  y'' + b·y' + c·y = rhs(x).
///
/// `rhs_coeffs[j]` is the coefficient of x^j on the right-hand side,
/// in ascending degree order.
///
/// Returns ascending-order coefficients of y_p, or `None` on failure.
fn find_particular_polynomial(b: &Q, c: &Q, rhs_coeffs: &[Q]) -> Option<Vec<Q>> {
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
            let factor2 = Ratio::from_integer(BigInt::from(((j + 2) * (j + 1)) as i64)) * a_j2;
            let factor1 = Ratio::from_integer(BigInt::from((j + 1) as i64)) * b.clone() * a_j1;
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
                Ratio::from_integer(BigInt::from(((j + 2) * (j + 1)) as i64)) * bb[j + 1].clone()
            } else {
                Ratio::zero()
            };
            let denom = Ratio::from_integer(BigInt::from((j + 1) as i64)) * b.clone();
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
            let denom = Ratio::from_integer(BigInt::from(((j + 1) * (j + 2)) as i64));
            result.push(r_j.clone() / denom);
        }
        Some(result)
    }
}

/// Build an arena polynomial expression from ascending-order rational
/// coefficients: `coeffs[j]` is the coefficient of `var^j`.
fn build_polynomial_expr(arena: &mut Arena, coeffs: &[Q], var: ExprId) -> ExprId {
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
        let integral = crate::transforms::integrate::integrate(arena, f, var);
        let c1 = arena.symbol("C1");
        let solution = arena.add(&[integral, c1]);
        return Some(OdeResult {
            solution,
            constants: vec![c1],
            branches: Vec::new(),
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
        let combined_arg = crate::transforms::eval::eval(arena, combined_arg);
        arena.exp(combined_arg)
    } else {
        arena.mul(&[neg_f, exp_ax])
    };
    let integrand = crate::transforms::eval::eval(arena, integrand);
    let integral = crate::transforms::integrate::integrate(arena, integrand, var);

    let c1 = arena.symbol("C1");
    let inner = arena.add(&[integral, c1]);
    let solution = arena.mul(&[exp_neg_ax, inner]);

    Some(OdeResult {
        solution,
        constants: vec![c1],
        branches: Vec::new(),
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

    // g(y) = k·y: y = C1·exp(k·∫f(x)dx).  Before, `exp(k·∫f dx + C1)`,
    // which is never zero or negative: the solutions y ≤ 0 were missing.
    {
        let (coeff, base) = arena.as_coeff_term(g_y);
        if base == func {
            let coeff_id = ode_ratio_to_expr(arena, &coeff);
            let scaled_fx = arena.mul(&[coeff_id, f_x]);
            let integral_fx = crate::transforms::integrate::integrate(arena, scaled_fx, var);
            let c1 = arena.symbol("C1");
            let growth = arena.exp(integral_fx);
            let solution = arena.mul(&[c1, growth]);
            return Some(OdeResult {
                solution,
                constants: vec![c1],
                branches: Vec::new(),
            });
        }
    }

    // General case: ∫ dy/g(y) = ∫ f(x) dx + C1, solved for y (every
    // explicit root is a family: `solution` and `branches`), else the
    // implicit `∫ dy/g(y) − ∫ f(x) dx − C1` (= 0).
    // Before, `∫ dy/g(y)` was never attempted (a formal `Integral` was
    // returned even for `y′ = 1 + y²`).
    let neg_one = arena.int(-1);
    let inv_gy = arena.pow(g_y, neg_one);
    let inv_gy = crate::transforms::eval::eval(arena, inv_gy);
    let lhs_attempt = integrate_forms(arena, inv_gy, func);
    let lhs_integral = if integral_failed(arena, lhs_attempt) {
        arena.intern(ExprNode::Integral(inv_gy, func))
    } else {
        lhs_attempt
    };
    let rhs_integral = crate::transforms::integrate::integrate(arena, f_x, var);
    let c1 = arena.symbol("C1");
    let neg_rhs = arena.neg(rhs_integral);
    let neg_c1 = arena.neg(c1);
    let implicit = arena.add(&[lhs_integral, neg_rhs, neg_c1]);
    let implicit = crate::transforms::eval::eval(arena, implicit);
    if !integral_failed(arena, implicit) {
        let solutions = crate::transforms::solve::solve(arena, implicit, func);
        if !solutions.is_empty()
            && solutions.len() <= MAX_BRANCHES
            && solutions
                .iter()
                .all(|s| !contains_sym(arena, s.value, func_sym))
        {
            let mut families: Vec<ExprId> = solutions
                .iter()
                .map(|s| crate::transforms::eval::eval(arena, s.value))
                .collect();
            let solution = families.remove(0);
            return Some(OdeResult {
                solution,
                constants: vec![c1],
                branches: families,
            });
        }
    }
    Some(OdeResult {
        solution: implicit,
        constants: vec![c1],
        branches: Vec::new(),
    })
}

/// Collect multiplicative factors from an expression.
/// If `expr` is `Mul(a, b, c)`, return `[a, b, c]`.
/// If `expr` is `Neg(inner)`, return factors of inner with a `-1` prepended.
/// Otherwise return `[expr]`.
fn collect_mul_factors(arena: &mut Arena, expr: ExprId) -> Vec<ExprId> {
    let mut factors = Vec::new();
    let mut expr = expr;
    loop {
        match arena.node(expr).clone() {
            ExprNode::Mul(children) => {
                factors.extend(children.iter().copied());
                return factors;
            }
            ExprNode::Neg(inner) => {
                factors.push(arena.int(-1));
                expr = inner;
            }
            _ => {
                factors.push(expr);
                return factors;
            }
        }
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
    if integral_failed(arena, int_px) {
        // Integration of P(x) failed — we can't compute the integrating factor
        return None;
    }

    let mu = exp_of_log_sum(arena, int_px);

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
    // Step 1: F_partial = ∫ M dx  (treating y as constant)
    let integral_m = crate::transforms::integrate::integrate(arena, m_expr, var);
    if integral_failed(arena, integral_m) {
        return None; // Integration of M w.r.t. x failed
    }

    // Step 2: g'(y) = N − ∂(∫M dx)/∂y
    let d_intm_dy = crate::transforms::diff::diff(arena, integral_m, func);
    let g_prime = arena.sub(n_expr, d_intm_dy);
    let g_prime = crate::transforms::eval::eval(arena, g_prime);
    let g_prime = crate::transforms::expand::expand(arena, g_prime);
    let g_prime = crate::transforms::eval::eval(arena, g_prime);

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

    // Try to solve F(x,y) = C1 for y explicitly.
    let f_minus_c1 = arena.sub(potential, c1);
    let solutions = crate::transforms::solve::solve(arena, f_minus_c1, func);

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
    let v = arena.symbol("__v");

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

    // Try to solve for y explicitly.
    let solutions = crate::transforms::solve::solve(arena, implicit, func);
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
    let stripped = crate::transforms::subs::subs(arena, expr, d2y_dx2, d2_placeholder);
    let stripped = crate::transforms::subs::subs(arena, stripped, dy_dx, d1_placeholder);
    let has_x = contains_sym(arena, stripped, var_sym);
    let has_y = contains_sym(arena, stripped, _func_sym);
    if has_x && has_y {
        return None; // neither reduction applies
    }
    let p = arena.symbol("__p");
    let p_sym = match arena.node(p) {
        ExprNode::Symbol(s) => *s,
        _ => return None,
    };
    let c2 = arena.symbol("C2");

    // The reduced first-order solution `p = f(·, C1)` must be explicit and
    // carry its constant.  Before, an implicit one (still in `p`, and
    // without its constant) was integrated as if it were `p`: `y″ + 2y′²
    // = 0` gave `−1/4·ln(−2·__p²·(C2 + x))`.
    let usable = |arena: &Arena, r: &OdeResult| {
        !contains_sym(arena, r.solution, p_sym)
            && r.constants.len() == 1
            && expr_contains(arena, r.solution, r.constants[0])
            && !crate::base::walk::has_unevaluated(arena, r.solution)
    };

    // F(x, y′, y″) = 0 (no y): p(x) = y′ gives F(x, p, p′) = 0, and
    // y = ∫ p dx + C2 is explicit.
    if !has_y {
        let dp_dx = arena.intern(ExprNode::Derivative(p, var));
        let reduced = crate::transforms::subs::subs(arena, expr, d2y_dx2, dp_dx);
        let reduced = crate::transforms::subs::subs(arena, reduced, dy_dx, p);
        let reduced = crate::transforms::eval::eval(arena, reduced);
        if let Some(pr) = dsolve(arena, reduced, p, var)
            && usable(arena, &pr)
        {
            // One family of `y` per family of `p` (`y″ = y′³`: `p = ±(C1 −
            // 2x)^(−1/2)`; before 0.30 only the first).
            let antiderivative = |arena: &mut Arena, p_family: ExprId| {
                let integral = crate::transforms::integrate::integrate(arena, p_family, var);
                (!integral_failed(arena, integral)).then(|| {
                    let s = arena.add(&[integral, c2]);
                    crate::transforms::eval::eval(arena, s)
                })
            };
            if let Some(solution) = antiderivative(arena, pr.solution) {
                let mut branches = Vec::new();
                for &b in &pr.branches {
                    match antiderivative(arena, b) {
                        Some(y_b) => branches.push(y_b),
                        None => tracing::debug!("ode: a family of y' has no closed antiderivative"),
                    }
                }
                return Some(OdeResult {
                    solution,
                    constants: vec![pr.constants[0], c2],
                    branches,
                });
            }
        }
        if has_x {
            return None;
        }
    }

    // Now perform the reduction: let p = dy/dx, then d²y/dx² = p·dp/dy
    let dp_dy = arena.intern(ExprNode::Derivative(p, func));
    let p_dp_dy = arena.mul(&[p, dp_dy]);

    // Substitute: y'' → p·dp/dy,  y' → p
    let reduced = crate::transforms::subs::subs(arena, expr, d2y_dx2, p_dp_dy);
    let reduced = crate::transforms::subs::subs(arena, reduced, dy_dx, p);
    let reduced = crate::transforms::eval::eval(arena, reduced);
    // A factor `p` common to every term is divided out (it only carries
    // the constant solutions `y′ = 0`): `y·y″ − y′²` becomes `y·p′ − p`,
    // linear in `p(y)`.
    let reduced = divide_common_factor(arena, reduced, p).unwrap_or(reduced);

    // Now `reduced` is a first-order ODE in p(y) with independent var = y.
    let p_result = dsolve(arena, reduced, p, func)?;
    if !usable(arena, &p_result) {
        return None;
    }

    // p_result.solution gives p = f(y, C1).
    // Now solve dy/dx = p(y) — this is separable: ∫ dy/p(y) = x + C2.
    let p_sol = p_result.solution;

    // Set up: dy/dx - p_sol = 0  →  ∫ 1/p_sol dy = x + C2
    // We need to integrate 1/p_sol w.r.t. y.
    let neg_one_id = arena.int(-1);
    let inv_p = arena.pow(p_sol, neg_one_id);
    let inv_p = crate::transforms::eval::eval(arena, inv_p);
    let lhs_integral = crate::transforms::integrate::integrate(arena, inv_p, func);

    if integral_failed(arena, lhs_integral) {
        return None; // Can't integrate 1/p(y)
    }

    // Implicit solution: ∫ dy/p(y) = x + C2
    let rhs = arena.add(&[var, c2]);
    let implicit = arena.sub(lhs_integral, rhs);
    let implicit = crate::transforms::eval::eval(arena, implicit);

    // Try to solve explicitly for y.
    let solutions = crate::transforms::solve::solve(arena, implicit, func);
    if solutions.len() == 1 {
        let sol = crate::transforms::eval::eval(arena, solutions[0].value);
        let mut constants = p_result.constants;
        constants.push(c2);
        return Some(OdeResult {
            solution: sol,
            constants,
            branches: Vec::new(),
        });
    }

    // Return implicit form.
    let mut constants = p_result.constants;
    constants.push(c2);
    Some(OdeResult {
        solution: implicit,
        constants,
        branches: Vec::new(),
    })
}

/// Does `sym` occur free in `expr`?  Bound occurrences (a `Sum` index, a
/// `RootOf` variable; see `walk::binder`) do not count.  Up to 0.28 this
/// was a structural walk with no visited set, exponential on a DAG with
/// shared sub-expressions.
fn contains_sym(arena: &Arena, expr: ExprId, sym: SymbolId) -> bool {
    crate::base::walk::has_free_symbol(arena, expr, sym)
}

/// `∫ e d(var)`, trying equivalent forms of `e` until one integrates: as
/// given, with products of exponentials merged (`powsimp`: variation of
/// parameters builds `x·e⁻ˣ·e⁻ˣ/x²/e⁻²ˣ`, which is `1/x`), as one
/// fraction (`ratsimp`), and expanded.  The first attempt's result (with its
/// formal `Integral`) when none does.
fn integrate_forms(arena: &mut Arena, e: ExprId, var: ExprId) -> ExprId {
    let first = crate::transforms::integrate::integrate(arena, e, var);
    if !integral_failed(arena, first) {
        return first;
    }
    let merged = crate::simplify::powsimp::powsimp(arena, e);
    let merged = crate::transforms::eval::eval(arena, merged);
    let fraction = crate::simplify::ratsimp::ratsimp(arena, merged);
    let fraction = crate::transforms::eval::eval(arena, fraction);
    let expanded = crate::transforms::expand::expand(arena, merged);
    let expanded = crate::transforms::eval::eval(arena, expanded);
    for form in [merged, fraction, expanded] {
        if form == e {
            continue;
        }
        let r = crate::transforms::integrate::integrate(arena, form, var);
        if !integral_failed(arena, r) {
            return r;
        }
    }
    first
}

/// Is `e` zero as a rational function of its symbols and opaque
/// subexpressions (`ratsimp`)?  Catches `∂M/∂y − ∂N/∂x` of quotients
/// that `expand` leaves as unreduced fractions.
fn rational_zero(arena: &mut Arena, e: ExprId) -> bool {
    let r = crate::simplify::ratsimp::ratsimp(arena, e);
    let r = crate::transforms::eval::eval(arena, r);
    arena.is_zero_structural(r)
}

/// `expr / p` when `p` is a factor of every term of `expand(expr)`
/// (`None` otherwise).
fn divide_common_factor(arena: &mut Arena, expr: ExprId, p: ExprId) -> Option<ExprId> {
    let expanded = crate::transforms::expand::expand(arena, expr);
    let terms: Vec<ExprId> = match arena.node(expanded) {
        ExprNode::Add(c) => c.to_vec(),
        _ => vec![expanded],
    };
    let has_factor = |arena: &Arena, t: ExprId| -> bool {
        let is_power_of_p = |f: ExprId| -> bool {
            f == p
                || matches!(arena.node(f), ExprNode::Pow(b, e) if *b == p
                    && arena.as_num(*e).is_some_and(|r| r.is_integer() && r.is_positive()))
        };
        match arena.node(t) {
            ExprNode::Mul(fs) => fs.iter().any(|&f| is_power_of_p(f)),
            ExprNode::Neg(inner) => match arena.node(*inner) {
                ExprNode::Mul(fs) => fs.iter().any(|&f| is_power_of_p(f)),
                _ => is_power_of_p(*inner),
            },
            _ => is_power_of_p(t),
        }
    };
    if terms.is_empty() || !terms.iter().all(|&t| has_factor(arena, t)) {
        return None;
    }
    let quotient = arena.div(expanded, p);
    let quotient = crate::transforms::eval::eval(arena, quotient);
    let quotient = crate::transforms::expand::expand(arena, quotient);
    Some(crate::transforms::eval::eval(arena, quotient))
}

/// Did an antiderivative fail anywhere in `id`?  The integrator returns a
/// formal `Integral` node for the part it cannot do, not necessarily at
/// the top: `∫ 2·sin(y)/cos(y) dy` came back as `2·Integral(…)`, which
/// the top-level checks let through into an integrating factor (and a
/// `Piecewise` of nested integrals was returned as the "solution").
fn integral_failed(arena: &Arena, id: ExprId) -> bool {
    let mut stack = vec![id];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        let node = arena.node(n);
        if matches!(node, ExprNode::Integral(..)) {
            return true;
        }
        node.for_each_child(|c| stack.push(c));
    }
    false
}

/// `exp(Σ cᵢ·ln(fᵢ))` → `Π fᵢ^{cᵢ}` for integrating factors.
///
/// Integrating factors are only needed up to a constant factor, so
/// `ln|f|` is treated as `ln f` (the sign is absorbed into the constant
/// of integration).  Terms that are not logarithms stay inside `exp`.
fn exp_of_log_sum(arena: &mut Arena, integral: ExprId) -> ExprId {
    let terms: Vec<ExprId> = match arena.node(integral).clone() {
        ExprNode::Add(c) => c.to_vec(),
        _ => vec![integral],
    };
    let mut factors: Vec<ExprId> = Vec::new();
    let mut leftover: Vec<ExprId> = Vec::new();
    for t in terms {
        let (coeff, term) = arena.as_coeff_term(t);
        let inner = match arena.node(term).clone() {
            ExprNode::Ln(inner) => Some(inner),
            _ => None,
        };
        match inner {
            Some(inner) => {
                let base = match arena.node(inner).clone() {
                    ExprNode::Abs(a) => a,
                    _ => inner,
                };
                if coeff.is_one() {
                    factors.push(base);
                } else {
                    let c_id = ode_ratio_to_expr(arena, &coeff);
                    factors.push(arena.pow(base, c_id));
                }
            }
            None => leftover.push(t),
        }
    }
    if factors.is_empty() {
        return arena.exp(integral);
    }
    if !leftover.is_empty() {
        let rest = arena.add(&leftover);
        factors.push(arena.exp(rest));
    }
    let prod = arena.mul(&factors);
    crate::transforms::eval::eval(arena, prod)
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers for complex roots and undetermined coefficients
// ═══════════════════════════════════════════════════════════════════════════

/// Convert a `Ratio<BigInt>` to an arena `ExprId`.
fn ode_ratio_to_expr(arena: &mut Arena, r: &Q) -> ExprId {
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
    b: &Q,
    disc: &Q,
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
    let beta_expr = crate::transforms::eval::eval(arena, beta_expr);

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
        branches: Vec::new(),
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
    b: &Q,
    c: &Q,
    var: ExprId,
) -> Option<ExprId> {
    use num_traits::Zero;

    let (coeff_r, term) = arena.as_coeff_term(rhs);
    let node = arena.node(term).clone();

    match node {
        ExprNode::Sin(inner) => {
            let (omega, constant) = extract_linear_numeric(arena, inner, var)?;
            if !constant.is_zero() {
                return None;
            }
            let zero_r = num_rational::Ratio::<num_bigint::BigInt>::zero();
            try_trig_particular(arena, &coeff_r, &zero_r, &omega, b, c, var)
        }
        ExprNode::Cos(inner) => {
            let (omega, constant) = extract_linear_numeric(arena, inner, var)?;
            if !constant.is_zero() {
                return None;
            }
            let zero_r = num_rational::Ratio::<num_bigint::BigInt>::zero();
            try_trig_particular(arena, &zero_r, &coeff_r, &omega, b, c, var)
        }
        ExprNode::Exp(inner) => {
            let (r_val, constant) = extract_linear_numeric(arena, inner, var)?;
            if !constant.is_zero() {
                return None;
            }
            try_exp_particular(arena, &coeff_r, &r_val, b, c, var)
        }
        _ => None,
    }
}

/// Extract the numeric coefficient and constant from a linear expression.
/// Returns `Some((a, b))` where `expr = a·var + b`, both rational.
fn extract_linear_numeric(arena: &Arena, expr: ExprId, var: ExprId) -> Option<(Q, Q)> {
    let poly = crate::poly::polybridge::expr_to_poly(arena, expr, var)?;
    if poly.degree()? != 1 {
        return None;
    }
    Some((poly.coeff(1), poly.coeff(0)))
}

/// Compute a particular solution for `y'' + b·y' + c·y = P·sin(ωx) + Q·cos(ωx)`
/// via the method of undetermined coefficients.
fn try_trig_particular(
    arena: &mut Arena,
    p: &Q,
    q: &Q,
    omega: &Q,
    b: &Q,
    c: &Q,
    var: ExprId,
) -> Option<ExprId> {
    use num_traits::Zero;

    let omega_sq = omega * omega;
    let d = c - &omega_sq; // c − ω²
    let bw = b * omega; // b·ω

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
        if omega.is_zero() {
            return None;
        }
        let two_omega = num_rational::Ratio::from_integer(num_bigint::BigInt::from(2)) * omega;
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
    coeff_r: &Q,
    r: &Q,
    b: &Q,
    c: &Q,
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
        let deriv_val = num_rational::Ratio::from_integer(num_bigint::BigInt::from(2)) * r + b;
        if !deriv_val.is_zero() {
            // Single root: y_p = A·x·exp(rx) where A = R / (2r + b)
            let a_val = coeff_r / &deriv_val;
            let a_id = ode_ratio_to_expr(arena, &a_val);
            let x_exp = arena.mul(&[var, exp_rx]);
            Some(arena.mul(&[a_id, x_exp]))
        } else {
            // Double root: y_p = A·x²·exp(rx) where A = R / 2
            let two_r_val = num_rational::Ratio::from_integer(num_bigint::BigInt::from(2));
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
    let mut q_x_terms: Vec<(ExprId, Q)> = Vec::new();

    for &child in &children {
        let (coeff, term) = arena.as_coeff_term(child);
        if term == dy_dx {
            dy_coeff += coeff;
        } else if !contains_sym(arena, child, func_sym) {
            return None; // Free terms not allowed in standard Bernoulli
        } else if let Some(px) = extract_coeff_of_func(arena, child, func, func_sym, var_sym) {
            p_x_terms.push(px);
        } else if let Some((qx, n)) = extract_bernoulli_term(arena, child, func, func_sym, var_sym)
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
        crate::transforms::eval::eval(arena, s)
    };

    // Q_raw·y^n appears on the LHS: y' + P·y + Q_raw·y^n = 0
    // So actual Q in y' + P·y = Q·y^n is −Q_raw
    let q_sum: Vec<ExprId> = q_x_terms.iter().map(|(qx, _)| *qx).collect();
    let q_raw = if q_sum.len() == 1 {
        q_sum[0]
    } else {
        arena.add(&q_sum)
    };
    let neg_q_raw = arena.neg(q_raw);
    let q_x = if dy_coeff.is_one() {
        neg_q_raw
    } else {
        let s = arena.mul(&[inv_dy_id, neg_q_raw]);
        crate::transforms::eval::eval(arena, s)
    };

    if contains_sym(arena, p_x, func_sym) || contains_sym(arena, q_x, func_sym) {
        return None;
    }

    // Substitution: v = y^(1−n)
    // Transformed ODE: v' + (1−n)·P·v = (1−n)·Q
    let one_minus_n = num_rational::Ratio::<num_bigint::BigInt>::one() - &n_val;
    let one_minus_n_id = ode_ratio_to_expr(arena, &one_minus_n);

    let new_p = arena.mul(&[one_minus_n_id, p_x]);
    let new_p = crate::transforms::eval::eval(arena, new_p);
    let new_q = arena.mul(&[one_minus_n_id, q_x]);
    let new_q = crate::transforms::eval::eval(arena, new_q);

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
    let v_result = try_first_order_linear_general(arena, linear_expr, v, var, v_sym, var_sym)
        .or_else(|| try_simple_separable(arena, linear_expr, v, var, v_sym, var_sym))?;

    // Recover y = v^(1/(1−n)); for an even denominator (`y⁻² = v`) the
    // negative root `−v^(1/(1−n))` is a family too (before 0.30 it was
    // missing: `y′ = y³` with `y(0) = −1` had no solution).
    let inv_one_minus_n = num_rational::Ratio::<num_bigint::BigInt>::one() / &one_minus_n;
    let inv_id = ode_ratio_to_expr(arena, &inv_one_minus_n);
    let solution = arena.pow(v_result.solution, inv_id);
    let solution = crate::transforms::eval::eval(arena, solution);
    let two = num_bigint::BigInt::from(2);
    let branches = if inv_one_minus_n.denom() % &two == num_bigint::BigInt::from(0) {
        let negative = arena.neg(solution);
        vec![crate::transforms::eval::eval(arena, negative)]
    } else {
        Vec::new()
    };

    Some(OdeResult {
        solution,
        constants: v_result.constants,
        branches,
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
) -> Option<(ExprId, Q)> {
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
        if !coeff.is_one()
            && term != expr
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
        && let Some((inner_qx, n)) = extract_bernoulli_term(arena, inner, func, func_sym, _var_sym)
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
///
/// A forcing term `g(x)` is handled by `x = eᵗ` (for `x > 0`, like the
/// homogeneous solutions): `a·Y″ + (b − a)·Y′ + c·Y = g(eᵗ)` has constant
/// coefficients, and its solution with `t = ln x` is the answer.  Before,
/// every forced Cauchy–Euler equation was refused.
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
    let mut forcing: Vec<ExprId> = Vec::new();

    for &child in &children {
        let (coeff, term) = arena.as_coeff_term(child);
        if term == func {
            c_coeff += coeff;
        } else if !contains_sym(arena, child, func_sym) {
            forcing.push(child);
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
                if contains_sym(arena, of, func_sym) || contains_sym(arena, of, var_sym) {
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

    if !forcing.is_empty() {
        return euler_cauchy_forced(arena, &a_coeff, &b_coeff, &c_coeff, &forcing, var, var_sym);
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
        let roots = crate::transforms::solve::solve(arena, char_eq, r_var);

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
                branches: Vec::new(),
            })
        } else {
            None
        }
    } else if disc.is_zero() {
        // Repeated root: r = −p/2
        let two_r = num_rational::Ratio::<num_bigint::BigInt>::from_integer(2.into());
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
            branches: Vec::new(),
        })
    } else {
        // Complex roots α ± βi: α = −p/2, β = √(−disc)/2
        let two_r = num_rational::Ratio::<num_bigint::BigInt>::from_integer(2.into());
        let alpha = -(&p) / &two_r;
        let neg_disc = -disc;
        let neg_disc_id = ode_ratio_to_expr(arena, &neg_disc);
        let half = arena.rational(1, 2);
        let sqrt_neg_disc = arena.pow(neg_disc_id, half);
        let two_expr = arena.int(2);
        let beta = arena.div(sqrt_neg_disc, two_expr);
        let beta = crate::transforms::eval::eval(arena, beta);

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
            branches: Vec::new(),
        })
    }
}

/// `a·x²·y″ + b·x·y′ + c·y + Σ forcing = 0` by `x = eᵗ` (see
/// [`try_euler_cauchy`]).
fn euler_cauchy_forced(
    arena: &mut Arena,
    a: &Q,
    b: &Q,
    c: &Q,
    forcing: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
) -> Option<OdeResult> {
    let t = arena.symbol("__t");
    let big_y = arena.symbol("__Y");
    let (t_sym, y_sym) = match (arena.node(t), arena.node(big_y)) {
        (ExprNode::Symbol(ts), ExprNode::Symbol(ys)) => (*ts, *ys),
        _ => return None,
    };
    let d1 = arena.intern(ExprNode::Derivative(big_y, t));
    let d2 = arena.intern(ExprNode::Derivative(d1, t));
    let exp_t = arena.exp(t);
    let ln_x = arena.ln(var);
    let mut parts = Vec::with_capacity(3 + forcing.len());
    let a_id = ode_ratio_to_expr(arena, a);
    parts.push(arena.mul(&[a_id, d2]));
    let bma = ode_ratio_to_expr(arena, &(b - a));
    parts.push(arena.mul(&[bma, d1]));
    let c_id = ode_ratio_to_expr(arena, c);
    parts.push(arena.mul(&[c_id, big_y]));
    for &f in forcing {
        // ln x = t first (x > 0), then x = eᵗ.
        let g = crate::transforms::subs::subs(arena, f, ln_x, t);
        let g = crate::transforms::subs::subs(arena, g, var, exp_t);
        parts.push(crate::transforms::eval::eval(arena, g));
    }
    let ode_t = arena.add(&parts);
    let ode_t = crate::transforms::eval::eval(arena, ode_t);
    if contains_sym(arena, ode_t, var_sym) {
        return None;
    }
    let res = dsolve(arena, ode_t, big_y, t)?;
    if contains_sym(arena, res.solution, y_sym)
        || res.constants.len() != 2
        || crate::base::walk::has_unevaluated(arena, res.solution)
    {
        return None;
    }
    // e^{k·t + m} = x^k·e^m, then t = ln x.
    let mut replacements: Vec<(ExprId, ExprId)> = Vec::new();
    for id in crate::base::walk::post_order_ids(arena, res.solution) {
        let ExprNode::Exp(arg) = arena.node(id).clone() else {
            continue;
        };
        let Some(coeffs) = arena.coefficients_of(arg, t) else {
            continue;
        };
        if coeffs.len() != 2 || contains_sym(arena, coeffs[1], t_sym) {
            continue;
        }
        let x_k = arena.pow(var, coeffs[1]);
        let rest = arena.exp(coeffs[0]);
        let r = arena.mul(&[x_k, rest]);
        let r = crate::transforms::eval::eval(arena, r);
        replacements.push((id, r));
    }
    let solution = crate::transforms::subs::subs_map(arena, res.solution, &replacements);
    let solution = crate::transforms::subs::subs(arena, solution, t, ln_x);
    let solution = crate::transforms::eval::eval(arena, solution);
    if contains_sym(arena, solution, t_sym) {
        return None;
    }
    Some(OdeResult {
        solution,
        constants: res.constants,
        branches: Vec::new(),
    })
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
    let g_x = crate::transforms::eval::eval(arena, g_x);

    // Solve homogeneous equation
    let homo = solve_characteristic_equation(arena, b.clone(), c.clone(), var)?;
    let (y1, y2) = extract_fundamental_solutions(arena, homo.solution, &homo.constants)?;

    // Wronskian via symbolic differentiation
    let y1_prime = crate::transforms::diff::diff(arena, y1, var);
    let y2_prime = crate::transforms::diff::diff(arena, y2, var);
    let w_term1 = arena.mul(&[y1, y2_prime]);
    let w_term2 = arena.mul(&[y2, y1_prime]);
    let wronskian = arena.sub(w_term1, w_term2);
    let wronskian = crate::transforms::eval::eval(arena, wronskian);
    let wronskian = crate::transforms::expand::expand(arena, wronskian);
    let wronskian = crate::transforms::eval::eval(arena, wronskian);

    // If W still depends on var (e.g. cos²+sin² unsimplified), use Abel's
    // identity: W(x) = W(0)·exp(−b·x).
    let wronskian = if contains_sym(arena, wronskian, var_sym) {
        let w0 = crate::transforms::subs::subs(arena, wronskian, var, arena.zero);
        let w0 = crate::transforms::eval::eval(arena, w0);
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
    let integrand1 = crate::transforms::eval::eval(arena, integrand1);
    let integral1 = integrate_forms(arena, integrand1, var);
    if integral_failed(arena, integral1) {
        return None;
    }

    let y1_g = arena.mul(&[y1, g_x]);
    let integrand2 = arena.div(y1_g, wronskian);
    let integrand2 = crate::transforms::eval::eval(arena, integrand2);
    let integral2 = integrate_forms(arena, integrand2, var);
    if integral_failed(arena, integral2) {
        return None;
    }

    let term1 = arena.mul(&[y1, integral1]);
    let neg_term1 = arena.neg(term1);
    let term2 = arena.mul(&[y2, integral2]);
    let y_p = arena.add(&[neg_term1, term2]);
    let y_p = crate::transforms::eval::eval(arena, y_p);

    let solution = arena.add(&[homo.solution, y_p]);
    let solution = crate::transforms::eval::eval(arena, solution);

    Some(OdeResult {
        solution,
        constants: homo.constants,
        branches: Vec::new(),
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

    let y1 = crate::transforms::subs::subs(arena, homo_solution, c1, one);
    let y1 = crate::transforms::subs::subs(arena, y1, c2, zero);
    let y1 = crate::transforms::eval::eval(arena, y1);

    let y2 = crate::transforms::subs::subs(arena, homo_solution, c1, zero);
    let y2 = crate::transforms::subs::subs(arena, y2, c2, one);
    let y2 = crate::transforms::eval::eval(arena, y2);

    if y1 == arena.zero || y2 == arena.zero {
        return None;
    }

    Some((y1, y2))
}

// ═══════════════════════════════════════════════════════════════════════════
// nth-order linear constant-coefficient ODEs: Σ a_k y^(k) = g(x)
// ═══════════════════════════════════════════════════════════════════════════

/// Highest derivative order recognised by the nth-order solver.
const MAX_ODE_ORDER: usize = 12;

/// A linear constant-coefficient ODE `Σ a_k·y^(k) + g(x) = 0`.
struct LinearCcOde {
    /// `a_0 … a_n` (ascending derivative order), `a_n ≠ 0`.
    coeffs: Vec<Q>,
    /// The forcing terms `g(x)` as they appear in the zero-form expression.
    forcing: Vec<ExprId>,
}

/// Derivative chain `[y, y', y'', …]` up to [`MAX_ODE_ORDER`].
fn derivative_chain(arena: &mut Arena, func: ExprId, var: ExprId) -> Vec<ExprId> {
    let mut chain = vec![func];
    for k in 1..=MAX_ODE_ORDER {
        let d = arena.intern(ExprNode::Derivative(chain[k - 1], var));
        chain.push(d);
    }
    chain
}

/// Recognise `Σ a_k·y^(k) + g(x) = 0` with rational `a_k`.
fn extract_linear_cc(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
) -> Option<LinearCcOde> {
    use num_traits::Zero;
    let chain = derivative_chain(arena, func, var);
    let children: Vec<ExprId> = match arena.node(expr).clone() {
        ExprNode::Add(c) => c.to_vec(),
        _ => vec![expr],
    };
    let mut coeffs = vec![Q::zero(); MAX_ODE_ORDER + 1];
    let mut forcing = Vec::new();
    for child in children {
        let (coeff, term) = arena.as_coeff_term(child);
        if let Some(k) = chain.iter().position(|&d| d == term) {
            coeffs[k] += coeff;
        } else if !contains_sym(arena, child, func_sym) {
            forcing.push(child);
        } else {
            return None;
        }
    }
    while coeffs.len() > 1 && coeffs.last().is_some_and(|c| c.is_zero()) {
        coeffs.pop();
    }
    if coeffs.len() < 2 {
        return None; // no derivative at all
    }
    Some(LinearCcOde { coeffs, forcing })
}

/// Gaussian rational `re + im·i` — exact arithmetic for the complex
/// exponential shift used by undetermined coefficients.
#[derive(Clone, Debug, PartialEq)]
struct CQ {
    re: Q,
    im: Q,
}

impl CQ {
    fn new(re: Q, im: Q) -> Self {
        Self { re, im }
    }
    fn real(re: Q) -> Self {
        Self {
            re,
            im: num_traits::Zero::zero(),
        }
    }
    fn is_zero(&self) -> bool {
        num_traits::Zero::is_zero(&self.re) && num_traits::Zero::is_zero(&self.im)
    }
    fn add(&self, o: &CQ) -> CQ {
        CQ::new(&self.re + &o.re, &self.im + &o.im)
    }
    fn sub(&self, o: &CQ) -> CQ {
        CQ::new(&self.re - &o.re, &self.im - &o.im)
    }
    fn mul(&self, o: &CQ) -> CQ {
        CQ::new(
            &self.re * &o.re - &self.im * &o.im,
            &self.re * &o.im + &self.im * &o.re,
        )
    }
    fn div(&self, o: &CQ) -> CQ {
        let denom = &o.re * &o.re + &o.im * &o.im;
        let num = self.mul(&CQ::new(o.re.clone(), -o.im.clone()));
        CQ::new(num.re / &denom, num.im / &denom)
    }
    fn scale(&self, r: &Q) -> CQ {
        CQ::new(&self.re * r, &self.im * r)
    }
}

/// Evaluate `p^(j)(λ) / j!` for a rational polynomial `p` (ascending
/// coefficients) at a Gaussian rational `λ`.
fn shifted_coefficient(p: &[Q], j: usize, lambda: &CQ) -> CQ {
    // p^(j)(λ)/j! = Σ_{k≥j} C(k, j) a_k λ^{k-j}
    let mut acc = CQ::real(num_traits::Zero::zero());
    let mut lambda_pow = CQ::real(num_traits::One::one());
    for (k, a_k) in p.iter().enumerate().skip(j) {
        let binom = Q::from_integer(binomial(k as u64, j as u64));
        let term = lambda_pow.scale(&(a_k * binom));
        acc = acc.add(&term);
        lambda_pow = lambda_pow.mul(lambda);
    }
    acc
}

/// Trigonometric flavour of a forcing term.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum TrigKind {
    None,
    Cos,
    Sin,
}

/// One forcing term `c·x^d·e^{ax}·{1 | cos(bx) | sin(bx)}`.
struct ForcingTerm {
    coeff: Q,
    degree: usize,
    a: Q,
    b: Q,
    kind: TrigKind,
}

/// Parse a single additive forcing term (free of `y`).
fn parse_forcing_term(arena: &mut Arena, term: ExprId, var: ExprId) -> Option<ForcingTerm> {
    use num_traits::{Signed, Zero};
    let (mut coeff, t) = arena.as_coeff_term(term);
    let factors: Vec<ExprId> = if t == arena.one {
        Vec::new()
    } else {
        match arena.node(t).clone() {
            ExprNode::Mul(c) => c.to_vec(),
            _ => vec![t],
        }
    };
    let mut degree = 0usize;
    let mut a = Q::zero();
    let mut b = Q::zero();
    let mut kind = TrigKind::None;
    let mut seen_exp = false;
    for f in factors {
        if f == var {
            degree += 1;
            continue;
        }
        match arena.node(f).clone() {
            ExprNode::Pow(base, exp) if base == var => {
                let n = arena.as_num(exp)?.clone();
                if !n.is_integer() || n.is_negative() {
                    return None;
                }
                degree += usize::try_from(n.to_integer()).ok()?;
            }
            ExprNode::Exp(inner) => {
                if seen_exp {
                    return None;
                }
                let (rate, c) = extract_linear_numeric(arena, inner, var)?;
                if !c.is_zero() {
                    return None;
                }
                seen_exp = true;
                a = rate;
            }
            ExprNode::Sin(inner) => {
                if kind != TrigKind::None {
                    return None;
                }
                let (freq, c) = extract_linear_numeric(arena, inner, var)?;
                if !c.is_zero() || freq.is_zero() {
                    return None;
                }
                kind = TrigKind::Sin;
                if freq.is_negative() {
                    coeff = -coeff;
                    b = -freq;
                } else {
                    b = freq;
                }
            }
            ExprNode::Cos(inner) => {
                if kind != TrigKind::None {
                    return None;
                }
                let (freq, c) = extract_linear_numeric(arena, inner, var)?;
                if !c.is_zero() || freq.is_zero() {
                    return None;
                }
                kind = TrigKind::Cos;
                b = freq.abs();
            }
            _ => {
                let r = arena.as_num(f)?;
                coeff *= r.clone();
            }
        }
    }
    Some(ForcingTerm {
        coeff,
        degree,
        a,
        b,
        kind,
    })
}

/// Build `Σ c_k·x^k` from rational coefficients.
fn rat_poly_expr(arena: &mut Arena, coeffs: &[Q], var: ExprId) -> ExprId {
    build_polynomial_expr(arena, coeffs, var)
}

/// Particular solution of `Σ a_k y^(k) = q(x)·e^{ax}·{1|cos bx|sin bx}` via the
/// complex exponential shift: with `λ = a + bi`, `y_p = Re/Im(w(x)·e^{λx})`
/// where `w` solves the triangular system `Σ_j (p^(j)(λ)/j!)·w^(j) = q`.
/// Resonance (λ a root of multiplicity `s`) is handled by the extra
/// factor `x^s` implicit in the degree of `w`.
fn particular_for_group(
    arena: &mut Arena,
    p: &[Q],
    q: &[Q],
    a: &Q,
    b: &Q,
    kind: TrigKind,
    var: ExprId,
) -> Option<ExprId> {
    use num_traits::Zero;
    let lambda = CQ::new(a.clone(), b.clone());
    let n = p.len() - 1;
    // c_j = p^(j)(λ)/j!
    let c: Vec<CQ> = (0..=n)
        .map(|j| shifted_coefficient(p, j, &lambda))
        .collect();
    let s = c.iter().position(|cj| !cj.is_zero())?;
    let d = q.len().checked_sub(1)?;
    let m = d + s;
    let mut w = vec![CQ::real(Q::zero()); m + 1];
    // Solve from the top degree down: for k = d..0,
    //   Σ_{j≥s} c_j·(k+j)!/k!·w_{k+j} = q_k
    for k in (0..=d).rev() {
        let mut rhs = CQ::real(q[k].clone());
        for j in (s + 1)..=n {
            if k + j > m {
                continue;
            }
            let fact = falling_factorial_rat(k + j, j);
            rhs = rhs.sub(&c[j].mul(&w[k + j]).scale(&fact));
        }
        let lead = c[s].scale(&falling_factorial_rat(k + s, s));
        w[k + s] = rhs.div(&lead);
    }
    let wr: Vec<Q> = w.iter().map(|z| z.re.clone()).collect();
    let wi: Vec<Q> = w.iter().map(|z| z.im.clone()).collect();
    let wr_x = rat_poly_expr(arena, &wr, var);
    let wi_x = rat_poly_expr(arena, &wi, var);
    let exp_ax = if a.is_zero() {
        arena.one
    } else {
        let a_id = ode_ratio_to_expr(arena, a);
        let ax = arena.mul(&[a_id, var]);
        arena.exp(ax)
    };
    let body = match kind {
        TrigKind::None => wr_x,
        TrigKind::Cos | TrigKind::Sin => {
            let b_id = ode_ratio_to_expr(arena, b);
            let bx = arena.mul(&[b_id, var]);
            let cos_bx = arena.cos(bx);
            let sin_bx = arena.sin(bx);
            if kind == TrigKind::Cos {
                // Re(w e^{ibx}) = wr cos - wi sin
                let t1 = arena.mul(&[wr_x, cos_bx]);
                let t2 = arena.mul(&[wi_x, sin_bx]);
                arena.sub(t1, t2)
            } else {
                // Im(w e^{ibx}) = wr sin + wi cos
                let t1 = arena.mul(&[wr_x, sin_bx]);
                let t2 = arena.mul(&[wi_x, cos_bx]);
                arena.add(&[t1, t2])
            }
        }
    };
    let y_p = arena.mul(&[exp_ax, body]);
    let y_p = crate::transforms::expand::expand(arena, y_p);
    Some(crate::transforms::eval::eval(arena, y_p))
}

/// `n·(n-1)·…·(n-j+1)` as a rational.
fn falling_factorial_rat(n: usize, j: usize) -> Q {
    let mut r = Q::from_integer(1.into());
    for i in 0..j {
        r *= Q::from_integer(((n - i) as i64).into());
    }
    r
}

/// Fundamental solutions of `Σ a_k y^(k) = 0` from the square-free
/// factorisation of the characteristic polynomial: `x^j e^{rx}` for a
/// real root of multiplicity `> j`, and `x^j e^{αx} cos(βx)`,
/// `x^j e^{αx} sin(βx)` for a complex pair `α ± βi`.
fn homogeneous_basis_cc(arena: &mut Arena, coeffs: &[Q], var: ExprId) -> Option<Vec<ExprId>> {
    let p = crate::poly::Poly::from_coeffs(coeffs.to_vec());
    let r_sym = arena.symbol("__r_cc");
    let mut basis: Vec<ExprId> = Vec::new();
    for (factor, mult) in p.squarefree_factors() {
        let f_expr = crate::poly::polybridge::poly_to_expr(arena, &factor, r_sym);
        let roots = crate::transforms::solve::solve(arena, f_expr, r_sym);
        if roots.is_empty() {
            return None;
        }
        let i_unit = arena.i_unit;
        let mut used = vec![false; roots.len()];
        for i in 0..roots.len() {
            if used[i] {
                continue;
            }
            used[i] = true;
            let root = roots[i].value;
            let is_complex = crate::base::walk::contains(arena, root, i_unit);
            if !is_complex {
                for j in 0..mult {
                    basis.push(mode_real(arena, root, j, var));
                }
                continue;
            }
            let (re, im) = arena.as_real_imag_expr(root);
            let re = crate::transforms::eval::eval(arena, re);
            let im = crate::transforms::eval::eval(arena, im);
            if arena.is_zero_structural(im) {
                for j in 0..mult {
                    basis.push(mode_real(arena, re, j, var));
                }
                continue;
            }
            // Mark the conjugate partner as consumed.
            let neg_im = arena.neg(im);
            let neg_im = crate::transforms::eval::eval(arena, neg_im);
            for (k, other) in roots.iter().enumerate() {
                if used[k] {
                    continue;
                }
                let (re2, im2) = arena.as_real_imag_expr(other.value);
                let re2 = crate::transforms::eval::eval(arena, re2);
                let im2 = crate::transforms::eval::eval(arena, im2);
                if re2 == re && im2 == neg_im {
                    used[k] = true;
                    break;
                }
            }
            let beta = if arena
                .as_num(im)
                .is_some_and(num_traits::Signed::is_negative)
            {
                neg_im
            } else {
                im
            };
            for j in 0..mult {
                let (c, s) = mode_complex(arena, re, beta, j, var);
                basis.push(c);
                basis.push(s);
            }
        }
    }
    Some(basis)
}

/// `x^j·e^{r·x}` (with `e^{0}` folded away).
fn mode_real(arena: &mut Arena, r: ExprId, j: usize, var: ExprId) -> ExprId {
    let x_pow = match j {
        0 => arena.one,
        1 => var,
        _ => {
            let e = arena.int(j as i64);
            arena.pow(var, e)
        }
    };
    if arena.is_zero_structural(r) {
        return x_pow;
    }
    let rx = arena.mul(&[r, var]);
    let exp_rx = arena.exp(rx);
    let m = arena.mul(&[x_pow, exp_rx]);
    crate::transforms::eval::eval(arena, m)
}

/// `(x^j e^{αx} cos βx, x^j e^{αx} sin βx)`.
fn mode_complex(
    arena: &mut Arena,
    alpha: ExprId,
    beta: ExprId,
    j: usize,
    var: ExprId,
) -> (ExprId, ExprId) {
    let envelope = mode_real(arena, alpha, j, var);
    let bx = arena.mul(&[beta, var]);
    let cos_bx = arena.cos(bx);
    let sin_bx = arena.sin(bx);
    let c = arena.mul(&[envelope, cos_bx]);
    let s = arena.mul(&[envelope, sin_bx]);
    (
        crate::transforms::eval::eval(arena, c),
        crate::transforms::eval::eval(arena, s),
    )
}

/// Solve `Σ a_k y^(k) + g(x) = 0` of any order `≥ 2` with rational
/// coefficients, where `g` is a sum of `poly·exp·{sin|cos}` terms (or
/// empty).  Homogeneous part via the characteristic polynomial (repeated
/// and complex roots handled), particular part via undetermined
/// coefficients with resonance handling.
fn try_nth_order_linear_const_coeff(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
) -> Option<OdeResult> {
    use num_traits::Zero;
    let ode = extract_linear_cc(arena, expr, func, var, func_sym)?;
    let n = ode.coeffs.len() - 1;
    if n < 2 {
        return None;
    }
    tracing::debug!(order = n, "ode: nth-order linear constant-coefficient");

    // Forcing: L[y] = -g(x).  Parse and group by (a, b, kind).
    let mut groups: Vec<((Q, Q, TrigKind), Vec<Q>)> = Vec::new();
    for &term in &ode.forcing {
        let ft = parse_forcing_term(arena, term, var)?;
        let key = (ft.a.clone(), ft.b.clone(), ft.kind);
        let entry = match groups.iter_mut().find(|(k, _)| *k == key) {
            Some(e) => e,
            None => {
                groups.push((key, Vec::new()));
                groups.last_mut()?
            }
        };
        if entry.1.len() <= ft.degree {
            entry.1.resize(ft.degree + 1, Q::zero());
        }
        entry.1[ft.degree] -= ft.coeff; // move to the right-hand side
    }

    let basis = homogeneous_basis_cc(arena, &ode.coeffs, var)?;
    if basis.len() != n {
        return None;
    }
    let mut constants = Vec::with_capacity(n);
    let mut terms = Vec::with_capacity(n + groups.len());
    for (i, &b) in basis.iter().enumerate() {
        let c = arena.symbol(&format!("C{}", i + 1));
        constants.push(c);
        terms.push(arena.mul(&[c, b]));
    }
    for ((a, b, kind), q) in &groups {
        if q.iter().all(Zero::is_zero) {
            continue;
        }
        let y_p = particular_for_group(arena, &ode.coeffs, q, a, b, *kind, var)?;
        terms.push(y_p);
    }
    let solution = arena.add(&terms);
    let solution = crate::transforms::eval::eval(arena, solution);
    Some(OdeResult {
        solution,
        constants,
        branches: Vec::new(),
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Clairaut: y = x·y' + f(y')
// ═══════════════════════════════════════════════════════════════════════════

/// Recognise the Clairaut form `y = x·y' + f(y')` and return `f(p)` with
/// `p` standing for `y'`.
fn clairaut_f(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
    var_sym: SymbolId,
    p: ExprId,
) -> Option<ExprId> {
    let dy_dx = arena.intern(ExprNode::Derivative(func, var));
    let d2y_dx2 = arena.intern(ExprNode::Derivative(dy_dx, var));
    if !expr_contains(arena, expr, dy_dx) || expr_contains(arena, expr, d2y_dx2) {
        return None;
    }
    let g = crate::transforms::subs::subs(arena, expr, dy_dx, p);
    // Linear in y with a constant coefficient.
    let coeffs = crate::transforms::solve::symbolic_poly_coeffs(arena, g, func)?;
    if coeffs.len() != 2 {
        return None;
    }
    let c_y = coeffs[1];
    let c_0 = coeffs[0];
    let p_sym = match arena.node(p) {
        ExprNode::Symbol(s) => *s,
        _ => return None,
    };
    if contains_sym(arena, c_y, var_sym) || contains_sym(arena, c_y, p_sym) {
        return None;
    }
    // g / c_y = y + c_0/c_y  must equal  y - x·p - f(p)  ⇒  f = -c_0/c_y - x·p
    let ratio = arena.div(c_0, c_y);
    let neg_ratio = arena.neg(ratio);
    let xp = arena.mul(&[var, p]);
    let f = arena.sub(neg_ratio, xp);
    let f = crate::transforms::eval::eval(arena, f);
    let f = crate::transforms::expand::expand(arena, f);
    let f = crate::transforms::eval::eval(arena, f);
    if contains_sym(arena, f, var_sym) || contains_sym(arena, f, func_sym) {
        return None;
    }
    Some(f)
}

/// Solve a Clairaut equation `y = x·y' + f(y')`: the general solution is
/// the family of lines `y = C·x + f(C)`.  (The singular envelope solution
/// is not returned.)
fn try_clairaut(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
    var_sym: SymbolId,
) -> Option<OdeResult> {
    let p = arena.symbol("__clairaut_p");
    let f = clairaut_f(arena, expr, func, var, func_sym, var_sym, p)?;
    tracing::debug!("ode: Clairaut form recognised");
    let c1 = arena.symbol("C1");
    let f_c = crate::transforms::subs::subs(arena, f, p, c1);
    let cx = arena.mul(&[c1, var]);
    let solution = arena.add(&[cx, f_c]);
    let solution = crate::transforms::eval::eval(arena, solution);
    Some(OdeResult {
        solution,
        constants: vec![c1],
        branches: Vec::new(),
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Riccati: y' = q0(x) + q1(x)·y + q2(x)·y²
// ═══════════════════════════════════════════════════════════════════════════

/// Extract `(q0, q1, q2)` from a Riccati equation in zero form,
/// `c·y' - (q0 + q1·y + q2·y²) = 0` with `c` a nonzero constant.
/// Requires `q2 ≠ 0` and `q0 ≠ 0` (otherwise the equation is Bernoulli /
/// linear) and coefficients free of `y` and `y'`.
fn riccati_coeffs(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
) -> Option<(ExprId, ExprId, ExprId)> {
    let dy_dx = arena.intern(ExprNode::Derivative(func, var));
    let d2y_dx2 = arena.intern(ExprNode::Derivative(dy_dx, var));
    if expr_contains(arena, expr, d2y_dx2) {
        return None;
    }
    let (rest, c) = extract_m_n(arena, expr, func, var)?;
    if crate::base::walk::contains(arena, c, func) || crate::base::walk::contains(arena, c, var) {
        return None;
    }
    if crate::base::walk::contains(arena, rest, dy_dx) {
        return None;
    }
    // y' = -rest/c
    let neg_rest = arena.neg(rest);
    let rhs = arena.div(neg_rest, c);
    let rhs = crate::transforms::eval::eval(arena, rhs);
    let coeffs = crate::transforms::solve::symbolic_poly_coeffs(arena, rhs, func)?;
    if coeffs.len() != 3 {
        return None;
    }
    let (q0, q1, q2) = (coeffs[0], coeffs[1], coeffs[2]);
    if arena.is_zero_structural(q0) || arena.is_zero_structural(q2) {
        return None;
    }
    Some((q0, q1, q2))
}

/// Solve a Riccati equation `y' = q0(x) + q1(x)·y + q2(x)·y²` given a known
/// particular solution `y_p(x)`.
///
/// The substitution `y = y_p + 1/v` turns the equation into the linear
/// ODE `v' + (q1 + 2·q2·y_p)·v = -q2`, which is solved with [`dsolve`];
/// the result is `y = y_p + 1/v`.
///
/// Returns `None` if `expr` is not of Riccati form, if `particular` does
/// not satisfy it, or if the linear equation for `v` cannot be solved.
pub fn solve_riccati(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    particular: ExprId,
) -> Option<OdeResult> {
    let (q0, q1, q2) = riccati_coeffs(arena, expr, func, var)?;
    // Verify the particular solution.
    if !checkodesol(arena, expr, particular, func, var) {
        // Try the derivative form directly: y_p' - (q0 + q1 y_p + q2 y_p²).
        let yp_prime = crate::transforms::diff::diff(arena, particular, var);
        let yp2 = arena.mul(&[particular, particular]);
        let t1 = arena.mul(&[q1, particular]);
        let t2 = arena.mul(&[q2, yp2]);
        let rhs = arena.add(&[q0, t1, t2]);
        let residual = arena.sub(yp_prime, rhs);
        let residual = crate::transforms::eval::eval(arena, residual);
        let residual = crate::transforms::expand::expand(arena, residual);
        let residual = crate::transforms::eval::eval(arena, residual);
        if !arena.is_zero_structural(residual) {
            return None;
        }
    }
    let _ = q0;
    let v = arena.symbol("__riccati_v");
    let dv = arena.intern(ExprNode::Derivative(v, var));
    // v' + (q1 + 2 q2 y_p) v + q2 = 0
    let two = arena.int(2);
    let two_q2_yp = arena.mul(&[two, q2, particular]);
    let coeff = arena.add(&[q1, two_q2_yp]);
    let coeff = crate::transforms::eval::eval(arena, coeff);
    let coeff_v = arena.mul(&[coeff, v]);
    let lin = arena.add(&[dv, coeff_v, q2]);
    let lin = crate::transforms::eval::eval(arena, lin);
    let v_res = dsolve(arena, lin, v, var)?;
    if crate::base::walk::contains(arena, v_res.solution, v) {
        return None; // implicit
    }
    let neg_one = arena.neg_one;
    let inv_v = arena.pow(v_res.solution, neg_one);
    let solution = arena.add(&[particular, inv_v]);
    let solution = crate::transforms::eval::eval(arena, solution);
    Some(OdeResult {
        solution,
        constants: v_res.constants,
        branches: Vec::new(),
    })
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
fn expr_contains(arena: &Arena, haystack: ExprId, needle: ExprId) -> bool {
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
    // Derivative nodes y, y', y'', … up to the highest order present.
    let chain = derivative_chain(arena, func, var);
    let max_order = (1..chain.len())
        .rev()
        .find(|&k| expr_contains(arena, ode_expr, chain[k]))
        .unwrap_or(0);

    // Derivatives of the solution.
    let mut sol_derivs = vec![solution];
    for k in 1..=max_order {
        let d = crate::transforms::diff::diff(arena, sol_derivs[k - 1], var);
        sol_derivs.push(d);
    }

    // Substitute the highest derivative first (more specific), down to func.
    let mut result = ode_expr;
    for k in (0..=max_order).rev() {
        result = crate::transforms::subs::subs(arena, result, chain[k], sol_derivs[k]);
    }

    // Evaluate and simplify
    result = crate::transforms::eval::eval(arena, result);
    result = crate::transforms::expand::expand(arena, result);
    result = crate::transforms::eval::eval(arena, result);

    if result == arena.zero {
        return true;
    }

    // Try another round of expand + eval for stubborn expressions
    result = crate::transforms::expand::expand(arena, result);
    result = crate::transforms::eval::eval(arena, result);
    if result == arena.zero {
        return true;
    }

    // Last resort: the full simplifier.
    let simplified = crate::simplify::simplify_engine::unified_simplify(
        arena,
        result,
        &crate::simplify::simplify_engine::SimplifyOpts::default(),
    );
    arena.is_zero_structural(simplified.expr)
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
/// 4. Otherwise (repeated or defective eigenvalues, eigenvalues the
///    eigenvector route cannot split into real and imaginary parts) uses
///    the exact matrix exponential `e^{At}·c` from the Jordan form
///    ([`Matrix::matrix_exp_t`]).
///
/// Before 0.30 step 4 was the Taylor polynomial of `e^{At}` to degree 12
/// ([`Matrix::exp_series`]), returned as if it were the solution: every
/// defective matrix (`[[−1, 1], [0, −1]]`) and several diagonalisable ones
/// got a polynomial that does not satisfy the system.
///
/// # Returns
///
/// `Some(vec)` with the solution vector, or `None` if the matrix is not
/// square, empty, contains entries that depend on `t_var`, or its
/// eigenvalues have no closed form.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::matrix::Matrix;
/// let ctx = Context::new();
/// let t = ctx.symbol("t");
/// let a = Matrix::new(vec![
///     vec![ctx.int(0), ctx.int(1)],
///     vec![ctx.int(-2), ctx.int(-3)],
/// ]).unwrap();
/// let sol = symplex::ode::solve_ode_system(&a, &t).unwrap();
/// assert_eq!(sol.len(), 2);
/// ```
pub fn solve_ode_system(a_matrix: &Matrix, t_var: &Ex) -> Option<Vec<Ex>> {
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

    // Otherwise the exact matrix exponential.
    solve_ode_system_expm(a_matrix, t_var, n)
}

/// Solve the initial-value problem `dx/dt = A·x`, `x(0) = x0`.
///
/// Computes the general solution with [`solve_ode_system`], evaluates it
/// at `t = 0`, and determines the constants `C1, …, Cn` from the linear
/// system `x(0) = x0` via [`linsolve`](crate::api::expr_solve_ext::linsolve).
///
/// # Errors
///
/// - [`SymplexError::InvalidArgument`](crate::base::errors::SymplexError::InvalidArgument) if `A` is not square or `x0` has
///   the wrong length.
/// - [`SymplexError::ComputationFailed`](crate::base::errors::SymplexError::ComputationFailed) if the general solution cannot be
///   found or the constants cannot be determined.
/// - [`SymplexError::NoSolution`](crate::base::errors::SymplexError::NoSolution) if the initial data is contradictory.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// let ctx = Context::new();
/// let t = ctx.symbol("t");
/// // x' = y, y' = -x  with x(0) = 1, y(0) = 0  →  x = cos t, y = -sin t
/// let a = matrix![ctx, [0, 1], [-1, 0]];
/// let sol = symplex::ode::solve_ode_system_ivp(&a, &t, &[ctx.int(1), ctx.int(0)]).unwrap();
/// assert_eq!(format!("{}", sol[0].simplify()), "cos(t)");
/// assert_eq!(format!("{}", sol[1].simplify()), "-sin(t)");
/// ```
pub fn solve_ode_system_ivp(
    a_matrix: &Matrix,
    t_var: &Ex,
    x0: &[Ex],
) -> Result<Vec<Ex>, crate::base::errors::SymplexError> {
    use crate::base::errors::SymplexError;
    let n = a_matrix.nrows();
    if !a_matrix.is_square() || n == 0 {
        return Err(SymplexError::InvalidArgument {
            operation: "solve_ode_system_ivp",
            reason: "coefficient matrix must be square and non-empty".into(),
        });
    }
    if x0.len() != n {
        return Err(SymplexError::InvalidArgument {
            operation: "solve_ode_system_ivp",
            reason: format!("expected {n} initial values, got {}", x0.len()),
        });
    }
    let general =
        solve_ode_system(a_matrix, t_var).ok_or_else(|| SymplexError::ComputationFailed {
            operation: "solve_ode_system_ivp",
            reason: "could not solve the homogeneous system".into(),
        })?;
    let ctx = t_var.context();
    let constants: Vec<Ex> = (1..=n).map(|i| ctx.symbol(&format!("C{i}"))).collect();
    let zero = ctx.int(0);
    let eqs: Vec<Ex> = general
        .iter()
        .zip(x0)
        .map(|(g, v)| (g.subs(t_var, &zero).eval() - v).eval())
        .collect();
    let sol = crate::api::expr_solve_ext::linsolve(&eqs, &constants).map_err(|e| {
        SymplexError::ComputationFailed {
            operation: "solve_ode_system_ivp",
            reason: format!("could not fit initial values: {e}"),
        }
    })?;
    let pairs = match sol {
        crate::api::expr_solve_ext::LinearSolution::Inconsistent => {
            return Err(SymplexError::NoSolution {
                operation: "solve_ode_system_ivp",
                reason: "initial values are inconsistent with the general solution".into(),
            });
        }
        crate::api::expr_solve_ext::LinearSolution::Unique(p) => p,
        crate::api::expr_solve_ext::LinearSolution::Parametric { solution, free } => solution
            .into_iter()
            .filter(|(c, _)| !free.contains(c))
            .collect(),
    };
    Ok(general
        .iter()
        .map(|g| {
            let mut e = g.clone();
            for (c, v) in &pairs {
                e = e.subs(c, v);
            }
            e.eval().simplify()
        })
        .collect())
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
/// The matrix exponentials `e^{±At}` are the exact ones from the Jordan
/// form ([`Matrix::matrix_exp_t`]); an antiderivative the integrator cannot
/// find stays a formal `Integral`.  Before 0.30 they were Taylor
/// polynomials of degree 12 ([`Matrix::exp_series`]) whenever the exact
/// exponential of the symbolic `∓A·t` failed — which was always — so the
/// "particular solution" did not satisfy the system even for constant
/// `b(t)`.
///
/// # Returns
///
/// `None` if the homogeneous part cannot be solved, the matrix
/// exponential has no closed form, or dimensions mismatch.
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
    let neg_a = a_matrix.scale(&t_var.context().int(-1));
    let exp_neg_at = neg_a.matrix_exp_t(t_var).ok()?;

    let b_col = Matrix::col_vector(b_vec.to_vec()).ok()?;
    let integrand_matrix = exp_neg_at.matmul(&b_col).ok()?.eval();

    // Integrate each component w.r.t. t
    let mut integrated = Vec::with_capacity(n);
    for i in 0..n {
        integrated.push(integrand_matrix.get(i, 0).integrate(t_var).eval());
    }
    let integrated_col = Matrix::col_vector(integrated).ok()?;

    // Multiply by exp(At)
    let exp_at = a_matrix.matrix_exp_t(t_var).ok()?;
    let particular = exp_at.matmul(&integrated_col).ok()?.eval();

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
    let ctx = t_var.context();
    (0..n)
        .map(|i| {
            let ci = ctx.symbol(&format!("C{}", i + 1));
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

/// `e^{At}·c` with the exact matrix exponential ([`Matrix::matrix_exp_t`],
/// from the Jordan form), `c = (C1, …, Cn)`.  `None` when the Jordan form
/// is not available in closed form.
fn solve_ode_system_expm(a_matrix: &Matrix, t_var: &Ex, n: usize) -> Option<Vec<Ex>> {
    let ctx = t_var.context();
    let exp_m = a_matrix.matrix_exp_t(t_var).ok()?;
    let constants: Vec<Ex> = (1..=n).map(|i| ctx.symbol(&format!("C{i}"))).collect();
    let c_vec = Matrix::col_vector(constants).ok()?;
    let result = exp_m.matmul(&c_vec).ok()?;
    Some((0..n).map(|i| result.get(i, 0).eval()).collect())
}

/// Eigenvector-based exact solver for diagonalisable constant-coefficient
/// systems, with the eigenvectors of [`Matrix::eigenvects`] (computed in
/// the eigenvalue's number field):
/// - **Real λ** with eigenvector `v`: the mode `C·exp(λt)·v`;
/// - **Complex λ = α + βi** (`β > 0`; the conjugate gives nothing new) with
///   `v = u + i·w`: the real modes `exp(αt)·(cos(βt)·u − sin(βt)·w)` and
///   `exp(αt)·(sin(βt)·u + cos(βt)·w)`, with `u`, `w` the real and
///   imaginary parts of the entries ([`Ex::as_real_imag`]).
///
/// `None` when `A` is not diagonalisable, a part cannot be separated, or
/// the modes do not number `n` (the caller falls back to the matrix
/// exponential).  Before 0.30 the eigenvectors came from `null(A − λI)`,
/// which is empty for every complex or irrational `λ` (its entries are not
/// recognised as zero), and the real part of an entry was taken by
/// substituting `I → 0`, which is wrong for `1/(1 − I)` (real part `1/2`).
fn solve_ode_system_eigen(a_matrix: &Matrix, t_var: &Ex, n: usize) -> Option<Vec<Ex>> {
    let ctx = t_var.context();
    let i_unit = ctx.i_unit();
    let eigen = a_matrix.eigenvects().ok()?;
    if eigen.iter().map(|(_, _, vs)| vs.len()).sum::<usize>() != n
        || eigen.iter().any(|(_, m, vs)| *m != vs.len())
    {
        return None;
    }
    // Real and imaginary part of a constant, both free of `I`.
    let split = |e: &Ex| -> Option<(Ex, Ex)> {
        let (re, im) = e.as_real_imag();
        let (re, im) = (re.eval().simplify(), im.eval().simplify());
        let clean = |p: &Ex| {
            !p.contains(&i_unit) && !p.to_string().contains("re(") && !p.to_string().contains("im(")
        };
        (clean(&re) && clean(&im)).then_some((re, im))
    };
    let mut solution: Vec<Ex> = (0..n).map(|_| ctx.int(0)).collect();
    let mut const_idx = 1_usize;
    for (ev, _, vectors) in &eigen {
        let (alpha, beta) = split(ev)?;
        if beta.is_zero_structural() {
            let exp_ev_t = if alpha.is_zero_structural() {
                ctx.int(1)
            } else {
                (&alpha * t_var).exp()
            };
            for v in vectors {
                let ci = ctx.symbol(&format!("C{const_idx}"));
                const_idx += 1;
                for (row, sol_row) in solution.iter_mut().enumerate() {
                    let (vr, vi) = split(v.get(row, 0))?;
                    if !vi.is_zero_structural() {
                        return None;
                    }
                    if !vr.is_zero_structural() {
                        *sol_row = &*sol_row + &(&ci * &(&exp_ev_t * &vr));
                    }
                }
            }
            continue;
        }
        // One of each conjugate pair: β > 0.
        let b = beta.eval_f64().ok()?;
        if b < 0.0 {
            continue;
        }
        let exp_alpha_t = if alpha.is_zero_structural() {
            ctx.int(1)
        } else {
            (&alpha * t_var).exp()
        };
        let cos_bt = (&beta * t_var).cos();
        let sin_bt = (&beta * t_var).sin();
        for v in vectors {
            let c_a = ctx.symbol(&format!("C{const_idx}"));
            let c_b = ctx.symbol(&format!("C{}", const_idx + 1));
            const_idx += 2;
            for (row, sol_row) in solution.iter_mut().enumerate() {
                let (u, w) = split(v.get(row, 0))?;
                let m1 = &exp_alpha_t * &(&(&cos_bt * &u) - &(&sin_bt * &w));
                let m2 = &exp_alpha_t * &(&(&sin_bt * &u) + &(&cos_bt * &w));
                *sol_row = &*sol_row + &(&(&c_a * &m1) + &(&c_b * &m2));
            }
        }
    }
    if const_idx != n + 1 {
        return None;
    }
    Some(solution.into_iter().map(|s| s.eval()).collect())
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
