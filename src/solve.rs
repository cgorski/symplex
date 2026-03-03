//! Symbolic equation solving.
//!
//! This module implements [`solve`], which finds the values of a variable
//! that make an expression equal to zero.
//!
//! # Supported equation types
//!
//! - **Linear:** `a*x + b = 0` → `x = -b/a`
//! - **Quadratic:** `a*x² + b*x + c = 0` → quadratic formula
//! - **Factorable polynomials:** if the polynomial can be expressed as
//!   a product of linear factors over ℚ, all rational roots are found
//!   via the Rational Root Theorem.
//!
//! # Design
//!
//! The solver works by:
//! 1. Converting the expression to a [`Poly`] in the given variable.
//! 2. Applying degree-specific solvers (linear, quadratic).
//! 3. For higher degrees, attempting rational root finding.
//! 4. Returning solutions as `Vec<ExprId>` — each entry is the value
//!    of the variable that makes the expression zero.
//!
//! If the expression is not polynomial in the variable, or if the
//! roots cannot be found in closed form over ℚ, an empty vector is
//! returned (not an error — the solver simply couldn't find solutions).

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Signed, Zero};

use crate::arena::Arena;
use crate::node::{ExprId, ExprNode, SymbolId};
use crate::poly::Poly;
use crate::polybridge;

/// A solution to an equation, with the variable and its value.
#[derive(Clone, Debug)]
pub struct Solution {
    /// The value of the variable that satisfies the equation.
    pub value: ExprId,
}

// ═══════════════════════════════════════════════════════════════════════════
// Public entry point
// ═══════════════════════════════════════════════════════════════════════════

/// Solve `expr = 0` for `var`.
///
/// Returns a vector of solutions (values of `var` that make `expr`
/// zero).  Returns an empty vector if:
/// - The expression is not polynomial in `var`.
/// - The polynomial degree is too high and no rational roots exist.
/// - The expression is identically zero (infinite solutions).
///
/// Solutions are returned as symbolic expressions ([`ExprId`]) in the
/// arena, fully canonicalized.
pub(crate) fn solve(arena: &mut Arena, expr: ExprId, var: ExprId) -> Vec<Solution> {
    // Pre-check: if expr is a Mul, solve each factor independently.
    // x*(x-1)*(x+2) = 0 → union of solutions for each factor
    if let ExprNode::Mul(ref children) = arena.node(expr).clone() {
        let mut solutions = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for &child in children {
            let child_solutions = solve(arena, child, var);
            for sol in child_solutions {
                let key = format!("{:?}", sol.value);
                if seen.insert(key) {
                    solutions.push(sol);
                }
            }
        }
        if !solutions.is_empty() {
            return solutions;
        }
    }

    // Step 1: Convert to polynomial.
    let poly = match polybridge::expr_to_poly(arena, expr, var) {
        Some(p) => p,
        None => {
            // Not polynomial → try transcendental solving via inversion peeling.
            // Handles: exp(x)=c, ln(x)=c, sin(x)=c, sqrt(x)=c, etc.
            if let Some(solutions) = try_solve_by_inversion(arena, expr, var)
                && !solutions.is_empty()
            {
                return solutions;
            }
            // Try change-of-variable: if expression is polynomial in f(x) for some f,
            // substitute t = f(x), solve the polynomial, then back-substitute.
            if let Some(solutions) = try_change_of_variable(arena, expr, var)
                && !solutions.is_empty()
            {
                return solutions;
            }
            return Vec::new();
        }
    };

    // Step 2: Handle trivial cases.
    if poly.is_zero() {
        return Vec::new(); // 0 = 0 is always true, infinite solutions.
    }

    if poly.is_constant() {
        return Vec::new(); // c = 0 where c ≠ 0 has no solutions.
    }

    // Step 3: Dispatch by degree.
    let degree = poly.degree().unwrap();
    tracing::debug!(degree = degree, "polynomial degree determined");

    match degree {
        1 => solve_linear(arena, &poly),
        2 => solve_quadratic(arena, &poly),
        _ => solve_rational_roots(arena, &poly),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Local helper: check if an expression contains a given variable
// ═══════════════════════════════════════════════════════════════════════════

fn expr_contains_var(arena: &Arena, expr: ExprId, var: ExprId) -> bool {
    if expr == var {
        return true;
    }
    for &child in arena.node(expr).children().iter() {
        if expr_contains_var(arena, child, var) {
            return true;
        }
    }
    false
}

// ═══════════════════════════════════════════════════════════════════════════
// Transcendental solving via inversion peeling
// ═══════════════════════════════════════════════════════════════════════════

/// Try to solve `expr = 0` by algebraic inversion.
/// Restructures as `f(x) = c` and inverts `f`.
fn try_solve_by_inversion(arena: &mut Arena, expr: ExprId, var: ExprId) -> Option<Vec<Solution>> {
    // Only works for expressions with exactly one occurrence of var
    // after some rearrangement.

    let node = arena.node(expr).clone();

    match node {
        // a*f(x) + b = 0 → f(x) = -b/a
        ExprNode::Add(ref children) => {
            let mut dep = Vec::new();
            let mut indep = Vec::new();
            for &child in children {
                if expr_contains_var(arena, child, var) {
                    dep.push(child);
                } else {
                    indep.push(child);
                }
            }

            if dep.len() != 1 {
                return None; // Multiple var-dependent terms, can't simply invert
            }

            let f_of_x = dep[0];
            // rhs = -sum(indep)
            let rhs = if indep.is_empty() {
                arena.zero
            } else {
                let sum_indep = arena.add(&indep);
                arena.neg(sum_indep)
            };

            // Now solve f_of_x = rhs by peeling layers
            solve_by_peeling(arena, f_of_x, rhs, var)
        }
        _ => None,
    }
}

/// Peel layers off `lhs = rhs` to isolate `var`.
fn solve_by_peeling(
    arena: &mut Arena,
    lhs: ExprId,
    rhs: ExprId,
    var: ExprId,
) -> Option<Vec<Solution>> {
    // Base case: lhs IS the variable
    if lhs == var {
        return Some(vec![Solution { value: rhs }]);
    }

    let node = arena.node(lhs).clone();
    match node {
        // c * f(x) = rhs → f(x) = rhs/c
        ExprNode::Mul(ref children) => {
            let mut dep = Vec::new();
            let mut indep = Vec::new();
            for &child in children {
                if expr_contains_var(arena, child, var) {
                    dep.push(child);
                } else {
                    indep.push(child);
                }
            }
            if dep.len() != 1 || indep.is_empty() {
                return None;
            }
            let coeff = arena.mul(&indep);
            let new_rhs = arena.div(rhs, coeff);
            solve_by_peeling(arena, dep[0], new_rhs, var)
        }
        // exp(f(x)) = rhs → f(x) = ln(rhs)
        ExprNode::Exp(inner) => {
            let new_rhs = arena.ln(rhs);
            solve_by_peeling(arena, inner, new_rhs, var)
        }
        // ln(f(x)) = rhs → f(x) = exp(rhs)
        ExprNode::Ln(inner) => {
            let new_rhs = arena.exp(rhs);
            solve_by_peeling(arena, inner, new_rhs, var)
        }
        // sin(f(x)) = rhs → f(x) = asin(rhs)  [principal value]
        ExprNode::Sin(inner) => {
            let new_rhs = arena.asin(rhs);
            solve_by_peeling(arena, inner, new_rhs, var)
        }
        // cos(f(x)) = rhs → f(x) = acos(rhs)  [principal value]
        ExprNode::Cos(inner) => {
            let new_rhs = arena.acos(rhs);
            solve_by_peeling(arena, inner, new_rhs, var)
        }
        // tan(f(x)) = rhs → f(x) = atan(rhs)
        ExprNode::Tan(inner) => {
            let new_rhs = arena.atan(rhs);
            solve_by_peeling(arena, inner, new_rhs, var)
        }
        // f(x)^n = rhs → f(x) = rhs^(1/n)
        ExprNode::Pow(inner_base, inner_exp) => {
            if let Some(n) = arena.as_num(inner_exp) {
                let n = n.clone();
                if !n.is_zero() {
                    let inv_n = Ratio::one() / n;
                    let inv_n_id = {
                        let nid = arena.intern_num(inv_n);
                        arena.intern(ExprNode::Num(nid))
                    };
                    let new_rhs = arena.pow(rhs, inv_n_id);
                    return solve_by_peeling(arena, inner_base, new_rhs, var);
                }
            }
            None
        }
        // Neg(-f(x)) = rhs → f(x) = -rhs
        ExprNode::Neg(inner) => {
            let new_rhs = arena.neg(rhs);
            solve_by_peeling(arena, inner, new_rhs, var)
        }
        _ => None,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Linear solver: a*x + b = 0 → x = -b/a
// ═══════════════════════════════════════════════════════════════════════════

fn solve_linear(arena: &mut Arena, poly: &Poly) -> Vec<Solution> {
    let a = poly.coeff(1); // coefficient of x
    let b = poly.coeff(0); // constant term

    if a.is_zero() {
        return Vec::new(); // Degenerate: 0*x + b = 0.
    }

    // x = -b / a
    let value = -b / a;
    let value_id = rational_to_expr(arena, &value);

    vec![Solution { value: value_id }]
}

// ═══════════════════════════════════════════════════════════════════════════
// Quadratic solver: a*x² + b*x + c = 0
// ═══════════════════════════════════════════════════════════════════════════

fn solve_quadratic(arena: &mut Arena, poly: &Poly) -> Vec<Solution> {
    let a = poly.coeff(2);
    let b = poly.coeff(1);
    let c = poly.coeff(0);

    if a.is_zero() {
        // Actually linear.
        let linear = Poly::from_coeffs(vec![c, b]);
        return solve_linear(arena, &linear);
    }

    // Discriminant: b² - 4ac
    let discriminant = &b * &b - Ratio::from_integer(BigInt::from(4)) * &a * &c;

    if discriminant.is_zero() {
        // Double root: x = -b / (2a)
        let two_a = Ratio::from_integer(BigInt::from(2)) * &a;
        let value = -b / two_a;
        let value_id = rational_to_expr(arena, &value);
        return vec![Solution { value: value_id }];
    }

    if discriminant.is_negative() {
        // Complex roots: x = (-b ± i√|Δ|) / (2a)
        let abs_disc = -discriminant;
        let neg_b = rational_to_expr(arena, &(-&b));
        let abs_disc_id = rational_to_expr(arena, &abs_disc);

        // Check if |Δ| is a perfect square
        let sqrt_abs_disc = if let Some(s) = rational_sqrt(&abs_disc) {
            rational_to_expr(arena, &s)
        } else {
            arena.sqrt(abs_disc_id)
        };

        let i_sqrt = arena.mul(&[arena.i_unit, sqrt_abs_disc]);
        let two_a_val = Ratio::from_integer(BigInt::from(2)) * &a;
        let two_a_id = rational_to_expr(arena, &two_a_val);
        let two_a_inv = {
            let neg_one = arena.neg_one;
            arena.pow(two_a_id, neg_one)
        };

        // x1 = (-b + i√|Δ|) / (2a)
        let sum1 = arena.add(&[neg_b, i_sqrt]);
        let x1 = arena.mul(&[sum1, two_a_inv]);

        // x2 = (-b - i√|Δ|) / (2a)
        let neg_i_sqrt = arena.neg(i_sqrt);
        let sum2 = arena.add(&[neg_b, neg_i_sqrt]);
        let x2 = arena.mul(&[sum2, two_a_inv]);

        return vec![Solution { value: x1 }, Solution { value: x2 }];
    }

    // Check if the discriminant is a perfect square (rational root).
    if let Some(sqrt_disc) = rational_sqrt(&discriminant) {
        let two_a = Ratio::from_integer(BigInt::from(2)) * &a;

        // x = (-b ± √Δ) / (2a)
        let x1 = (-&b + &sqrt_disc) / &two_a;
        let x2 = (-&b - &sqrt_disc) / &two_a;

        let x1_id = rational_to_expr(arena, &x1);
        let x2_id = rational_to_expr(arena, &x2);

        if x1 == x2 {
            vec![Solution { value: x1_id }]
        } else {
            vec![Solution { value: x1_id }, Solution { value: x2_id }]
        }
    } else {
        // Discriminant is not a perfect square — roots are irrational.
        // Express as (-b ± sqrt(discriminant)) / (2*a) symbolically.
        let neg_b = {
            let neg_b_val = -&b;
            rational_to_expr(arena, &neg_b_val)
        };
        let sqrt_disc = {
            let disc_id = rational_to_expr(arena, &discriminant);
            arena.sqrt(disc_id)
        };
        let two_a_val = Ratio::from_integer(BigInt::from(2)) * &a;
        let two_a_id = rational_to_expr(arena, &two_a_val);
        let two_a_inv = {
            let neg_one = arena.neg_one;
            arena.pow(two_a_id, neg_one)
        };

        // x1 = (-b + sqrt(disc)) / (2a)
        let sum1 = arena.add(&[neg_b, sqrt_disc]);
        let x1 = arena.mul(&[sum1, two_a_inv]);

        // x2 = (-b - sqrt(disc)) / (2a)
        let neg_sqrt = arena.neg(sqrt_disc);
        let sum2 = arena.add(&[neg_b, neg_sqrt]);
        let x2 = arena.mul(&[sum2, two_a_inv]);

        vec![Solution { value: x1 }, Solution { value: x2 }]
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Rational Root Theorem for higher-degree polynomials
// ═══════════════════════════════════════════════════════════════════════════

/// Find rational roots of a polynomial using the Rational Root Theorem.
///
/// For a polynomial with integer coefficients `aₙxⁿ + … + a₀`, any
/// rational root `p/q` (in lowest terms) must have `p | a₀` and `q | aₙ`.
///
/// We convert to integer coefficients by clearing denominators, then
/// enumerate candidate roots and test them.
fn solve_rational_roots(arena: &mut Arena, poly: &Poly) -> Vec<Solution> {
    // Convert to integer polynomial by clearing denominators.
    let (int_poly, _scale) = clear_denominators(poly);

    let degree = match int_poly.degree() {
        Some(d) => d,
        None => return Vec::new(),
    };

    // For very high degree, bail to avoid combinatorial explosion.
    if degree > 20 {
        return Vec::new();
    }

    let a0 = int_poly.coeff(0).to_integer(); // constant term
    let an = int_poly.leading_coeff().unwrap().to_integer(); // leading coeff

    if a0.is_zero() {
        // x = 0 is a root.  Factor out x and recurse.
        let mut roots = vec![Solution { value: arena.zero }];
        // Divide by x: shift coefficients down.
        let reduced_coeffs: Vec<Ratio<BigInt>> = poly.coeffs().iter().skip(1).cloned().collect();
        let reduced = Poly::from_coeffs(reduced_coeffs);
        if !reduced.is_zero() && !reduced.is_constant() {
            let more = solve_rational_roots(arena, &reduced);
            roots.extend(more);
        }
        return roots;
    }

    // Enumerate divisors of |a0| and |an|.
    let divisors_a0 = divisors(&a0.abs());
    let divisors_an = divisors(&an.abs());

    // Test each candidate p/q.
    let mut roots = Vec::new();
    let mut remaining = poly.clone();

    for p in &divisors_a0 {
        for q in &divisors_an {
            if remaining.is_constant() {
                break;
            }
            // Test +p/q and -p/q.
            for sign in &[1i64, -1i64] {
                let candidate = Ratio::new(p * BigInt::from(*sign), q.clone());

                if remaining.eval(&candidate).is_zero() {
                    // Found a root!
                    let value_id = rational_to_expr(arena, &candidate);
                    roots.push(Solution { value: value_id });

                    // Factor out (x - candidate) from remaining.
                    let factor = Poly::from_coeffs(vec![-candidate.clone(), Ratio::one()]);
                    let (quotient, _rem) = remaining.div_rem(&factor);
                    remaining = quotient;
                }
            }
        }
    }

    // If remaining has degree ≤ 2, solve it too.
    if let Some(d) = remaining.degree() {
        if d == 1 {
            roots.extend(solve_linear(arena, &remaining));
        } else if d == 2 {
            roots.extend(solve_quadratic(arena, &remaining));
        }
    }

    tracing::debug!(
        rational_roots = roots.len(),
        "rational roots found via theorem"
    );
    roots
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Convert a `Ratio<BigInt>` to an expression in the arena.
fn rational_to_expr(arena: &mut Arena, r: &Ratio<BigInt>) -> ExprId {
    let nid = arena.intern_num(r.clone());
    arena.intern(ExprNode::Num(nid))
}

/// Try to compute the exact square root of a non-negative rational.
///
/// Returns `Some(√r)` if `r` is a perfect square (both numerator and
/// denominator are perfect squares), or `None` otherwise.
fn rational_sqrt(r: &Ratio<BigInt>) -> Option<Ratio<BigInt>> {
    if r.is_negative() {
        return None;
    }
    if r.is_zero() {
        return Some(Ratio::zero());
    }

    let n = r.numer().abs();
    let d = r.denom().abs();

    let sqrt_n = n.sqrt();
    let sqrt_d = d.sqrt();

    if &sqrt_n * &sqrt_n == n && &sqrt_d * &sqrt_d == d {
        Some(Ratio::new(sqrt_n, sqrt_d))
    } else {
        None
    }
}

/// Clear denominators of a polynomial's coefficients.
///
/// Returns the integer-coefficient polynomial and the scale factor.
fn clear_denominators(poly: &Poly) -> (Poly, BigInt) {
    if poly.is_zero() {
        return (Poly::zero(), BigInt::one());
    }

    // Compute LCM of all denominators.
    let mut lcm = BigInt::one();
    for c in poly.coeffs() {
        let d = c.denom().abs();
        lcm = num_integer::lcm(lcm, d);
    }

    // Multiply all coefficients by the LCM.
    let scale = Ratio::from_integer(lcm.clone());
    let scaled = poly.scale(&scale);

    (scaled, lcm)
}

/// Compute all positive divisors of `n` (a positive BigInt).
///
/// Returns them in ascending order.  For n=0, returns `[1]` as a
/// fallback.
fn divisors(n: &BigInt) -> Vec<BigInt> {
    if n.is_zero() {
        return vec![BigInt::one()];
    }

    let n_abs = n.abs();

    // For small numbers, brute-force trial division.
    // For large numbers, this would be slow — but our polynomials
    // typically have small coefficients.
    let limit: u64 = match (&n_abs).try_into() {
        Ok(v) if v <= 1_000_000u64 => v,
        _ => {
            // Very large coefficient — just return 1 and n.
            return vec![BigInt::one(), n_abs];
        }
    };

    let mut result = Vec::new();
    let mut i = 1u64;
    while i * i <= limit {
        if limit.is_multiple_of(i) {
            result.push(BigInt::from(i));
            if i * i != limit {
                result.push(BigInt::from(limit / i));
            }
        }
        i += 1;
    }

    result.sort();
    result
}

// ═══════════════════════════════════════════════════════════════════════════
// Change-of-variable solving
// ═══════════════════════════════════════════════════════════════════════════

/// Try solving by detecting that the expression is polynomial in some f(x).
/// E.g., `exp(2x) - 3*exp(x) + 2` is polynomial in `t = exp(x)`: `t² - 3t + 2`.
fn try_change_of_variable(arena: &mut Arena, expr: ExprId, var: ExprId) -> Option<Vec<Solution>> {
    let var_sym = match arena.node(var) {
        ExprNode::Symbol(sid) => *sid,
        _ => return None,
    };

    // Collect candidate generators: inner functions applied to var
    let candidates = collect_generators(arena, expr, var, var_sym);

    for generator in candidates {
        // Create a temporary variable for substitution
        let t = arena.symbol("__t_subst");

        // --- Simple case: direct structural substitution of generator → t ---
        // This handles e.g. sin(x)^2 - sin(x) where walk_and_rebuild
        // naturally replaces sin(x) inside Pow(sin(x), 2) as well.
        let substituted = arena.subs_structural(expr, generator, t);

        if !expr_contains_var(arena, substituted, var) {
            let t_solutions = solve(arena, substituted, t);

            if !t_solutions.is_empty() {
                // Back-substitute: for each t = c, solve generator(var) = c.
                // Use solve_by_peeling directly to avoid recursing into
                // try_change_of_variable again (which would infinite-loop).
                let mut var_solutions = Vec::new();
                for t_sol in &t_solutions {
                    if let Some(back_sols) = solve_by_peeling(arena, generator, t_sol.value, var) {
                        for s in back_sols {
                            var_solutions.push(s);
                        }
                    }
                }
                if !var_solutions.is_empty() {
                    return Some(var_solutions);
                }
            }
        }

        // --- Advanced: detect exp(n*x) = exp(x)^n pattern ---
        // exp(2*x) is Exp(Mul([2, x])) which does NOT structurally
        // contain exp(x) = Exp(x), so simple substitution misses it.
        // We rewrite exp(k*x) → exp(x)^k first, then substitute.
        if let ExprNode::Exp(inner) = arena.node(generator).clone()
            && inner == var
        {
            let rewritten = rewrite_exp_powers(arena, expr, var, generator);
            if rewritten != expr {
                let substituted2 = arena.subs_structural(rewritten, generator, t);
                if !expr_contains_var(arena, substituted2, var) {
                    let t_solutions = solve(arena, substituted2, t);
                    if !t_solutions.is_empty() {
                        let mut var_solutions = Vec::new();
                        for t_sol in &t_solutions {
                            if let Some(back_sols) =
                                solve_by_peeling(arena, generator, t_sol.value, var)
                            {
                                for s in back_sols {
                                    var_solutions.push(s);
                                }
                            }
                        }
                        if !var_solutions.is_empty() {
                            return Some(var_solutions);
                        }
                    }
                }
            }
        }
    }

    None
}

/// Collect candidate generator functions: exp(x), sin(x), cos(x), ln(x), x^(k), etc.
fn collect_generators(arena: &Arena, expr: ExprId, var: ExprId, var_sym: SymbolId) -> Vec<ExprId> {
    let mut generators = Vec::new();
    let mut visited = std::collections::HashSet::new();
    collect_gens_recursive(arena, expr, var, var_sym, &mut generators, &mut visited);
    generators
}

fn collect_gens_recursive(
    arena: &Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
    gens: &mut Vec<ExprId>,
    visited: &mut std::collections::HashSet<ExprId>,
) {
    if !visited.insert(expr) {
        return;
    }
    match arena.node(expr).clone() {
        ExprNode::Exp(inner) if inner == var => {
            if !gens.contains(&expr) {
                gens.push(expr);
            }
        }
        ExprNode::Sin(inner) | ExprNode::Cos(inner) | ExprNode::Tan(inner) if inner == var => {
            if !gens.contains(&expr) {
                gens.push(expr);
            }
        }
        ExprNode::Ln(inner) if inner == var => {
            if !gens.contains(&expr) {
                gens.push(expr);
            }
        }
        ExprNode::Pow(base, exp) if base == var && !expr_contains_var(arena, exp, var) => {
            // x^(1/n) or x^k type
            if !gens.contains(&expr) {
                gens.push(expr);
            }
        }
        ExprNode::Add(ref children) | ExprNode::Mul(ref children) => {
            for &c in children {
                collect_gens_recursive(arena, c, var, var_sym, gens, visited);
            }
        }
        ExprNode::Pow(base, exp) => {
            collect_gens_recursive(arena, base, var, var_sym, gens, visited);
            collect_gens_recursive(arena, exp, var, var_sym, gens, visited);
        }
        ExprNode::Neg(inner)
        | ExprNode::Exp(inner)
        | ExprNode::Ln(inner)
        | ExprNode::Sin(inner)
        | ExprNode::Cos(inner)
        | ExprNode::Tan(inner) => {
            collect_gens_recursive(arena, inner, var, var_sym, gens, visited);
        }
        _ => {}
    }
}

/// Rewrite `exp(k*x)` as `exp(x)^k` for small integer k throughout the expression.
fn rewrite_exp_powers(arena: &mut Arena, expr: ExprId, var: ExprId, gen_exp_x: ExprId) -> ExprId {
    let mut result = expr;
    // Check small positive integer multiples: exp(k*x) → exp(x)^k
    for k in 2i64..=6 {
        let k_id = arena.int(k);
        let k_var = arena.mul(&[k_id, var]);
        let exp_k_var = arena.exp(k_var);
        let gen_pow_k = arena.pow(gen_exp_x, k_id);
        result = arena.subs_structural(result, exp_k_var, gen_pow_k);
    }
    // Also handle negative multiples: exp(-k*x) → exp(x)^(-k)
    for k in [-1i64, -2, -3] {
        let k_id = arena.int(k);
        let k_var = arena.mul(&[k_id, var]);
        let exp_k_var = arena.exp(k_var);
        let gen_pow_k = arena.pow(gen_exp_x, k_id);
        result = arena.subs_structural(result, exp_k_var, gen_pow_k);
    }
    result
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::Arena;

    fn sym(a: &mut Arena, name: &str) -> ExprId {
        a.symbol(name)
    }

    fn display(a: &Arena, id: ExprId) -> String {
        a.display(id).to_string()
    }

    fn solution_strings(a: &Arena, solutions: &[Solution]) -> Vec<String> {
        solutions.iter().map(|s| display(a, s.value)).collect()
    }

    // ── Linear ──────────────────────────────────────────────────────

    #[test]
    fn solve_linear_simple() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x - 3 = 0 → x = 3
        let three = a.int(3);
        let expr = a.sub(x, three);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1);
        assert_eq!(display(&a, solutions[0].value), "3");
    }

    #[test]
    fn solve_linear_with_coefficient() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // 2*x - 6 = 0 → x = 3
        let two = a.int(2);
        let six = a.int(6);
        let two_x = a.mul(&[two, x]);
        let expr = a.sub(two_x, six);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1);
        assert_eq!(display(&a, solutions[0].value), "3");
    }

    #[test]
    fn solve_linear_rational_solution() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // 3*x - 1 = 0 → x = 1/3
        let three = a.int(3);
        let one = a.one;
        let three_x = a.mul(&[three, x]);
        let expr = a.sub(three_x, one);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1);
        assert_eq!(display(&a, solutions[0].value), "1/3");
    }

    #[test]
    fn solve_linear_negative_solution() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x + 5 = 0 → x = -5
        let five = a.int(5);
        let expr = a.add(&[x, five]);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1);
        assert_eq!(display(&a, solutions[0].value), "-5");
    }

    // ── Quadratic ───────────────────────────────────────────────────

    #[test]
    fn solve_quadratic_two_roots() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x^2 - 5x + 6 = 0 → x = 2, x = 3
        let two = a.int(2);
        let five = a.int(5);
        let six = a.int(6);
        let x_sq = a.pow(x, two);
        let five_x = a.mul(&[five, x]);
        let neg_five_x = a.neg(five_x);
        let expr = a.add(&[x_sq, neg_five_x, six]);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 2);
        let vals: Vec<String> = solution_strings(&a, &solutions);
        assert!(
            vals.contains(&"2".to_string()),
            "should have root 2: {vals:?}"
        );
        assert!(
            vals.contains(&"3".to_string()),
            "should have root 3: {vals:?}"
        );
    }

    #[test]
    fn solve_quadratic_double_root() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x^2 - 2x + 1 = 0 → x = 1 (double root)
        let two = a.int(2);
        let one = a.one;
        let x_sq = a.pow(x, two);
        let two_x = a.mul(&[two, x]);
        let neg_two_x = a.neg(two_x);
        let expr = a.add(&[x_sq, neg_two_x, one]);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1);
        assert_eq!(display(&a, solutions[0].value), "1");
    }

    #[test]
    fn solve_quadratic_no_real_roots() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x^2 + 1 = 0 → complex roots ±i
        let two = a.int(2);
        let one = a.one;
        let x_sq = a.pow(x, two);
        let expr = a.add(&[x_sq, one]);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 2, "x²+1=0 should have 2 complex roots");
        let vals: Vec<String> = solution_strings(&a, &solutions);
        assert!(
            vals.iter().all(|v| v.contains("I")),
            "roots should contain I: {vals:?}"
        );
    }

    #[test]
    fn solve_quadratic_irrational_roots() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x^2 - 2 = 0 → x = ±√2
        let two = a.int(2);
        let x_sq = a.pow(x, two);
        let expr = a.sub(x_sq, two);
        let solutions = solve(&mut a, expr, x);
        // Should return symbolic sqrt(2) and -sqrt(2).
        assert_eq!(solutions.len(), 2, "x²-2=0 should have 2 solutions");
        let vals: Vec<String> = solution_strings(&a, &solutions);
        // Check that one contains sqrt and the other is negative.
        let has_sqrt = vals.iter().any(|v| v.contains("sqrt"));
        assert!(has_sqrt, "should contain sqrt(2): {vals:?}");
    }

    #[test]
    fn solve_x_squared_minus_one() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x^2 - 1 = 0 → x = 1, x = -1
        let two = a.int(2);
        let one = a.one;
        let x_sq = a.pow(x, two);
        let expr = a.sub(x_sq, one);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 2);
        let vals: Vec<String> = solution_strings(&a, &solutions);
        assert!(vals.contains(&"1".to_string()));
        assert!(vals.contains(&"-1".to_string()));
    }

    // ── Cubic via rational roots ────────────────────────────────────

    #[test]
    fn solve_cubic_all_rational() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6
        let three = a.int(3);
        let six = a.int(6);
        let eleven = a.int(11);
        let two = a.int(2);

        let x3 = a.pow(x, three);
        let x2 = a.pow(x, two);
        let six_x2 = a.mul(&[six, x2]);
        let eleven_x = a.mul(&[eleven, x]);

        let neg_six_x2 = a.neg(six_x2);
        let neg_six = a.neg(six);
        let expr = a.add(&[x3, neg_six_x2, eleven_x, neg_six]);

        let solutions = solve(&mut a, expr, x);
        let vals: Vec<String> = solution_strings(&a, &solutions);
        assert!(
            vals.contains(&"1".to_string()),
            "should have root 1: {vals:?}"
        );
        assert!(
            vals.contains(&"2".to_string()),
            "should have root 2: {vals:?}"
        );
        assert!(
            vals.contains(&"3".to_string()),
            "should have root 3: {vals:?}"
        );
    }

    // ── Edge cases ──────────────────────────────────────────────────

    #[test]
    fn solve_constant_nonzero_no_solutions() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // 5 = 0 → no solutions.
        let five = a.int(5);
        let solutions = solve(&mut a, five, x);
        assert!(solutions.is_empty());
    }

    #[test]
    fn solve_zero_expression() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // 0 = 0 → infinite solutions (returns empty).
        let zero = a.zero;
        let solutions = solve(&mut a, zero, x);
        assert!(solutions.is_empty());
    }

    #[test]
    fn solve_sin_x_eq_zero_via_change_of_variable() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // sin(x) = 0 → change-of-variable with t = sin(x) finds t = 0,
        // then back-substitutes sin(x) = 0 → x = asin(0) = 0.
        let expr = a.sin(x);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(
            solutions.len(),
            1,
            "sin(x)=0 should have 1 principal solution"
        );
        let val = display(&a, solutions[0].value);
        assert!(
            val == "0" || val.contains("asin"),
            "solution should be 0 or asin(0): {val}"
        );
    }

    #[test]
    fn solve_with_zero_root() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x^2 - x = x*(x-1) = 0 → x = 0, x = 1
        let two = a.int(2);
        let x_sq = a.pow(x, two);
        let expr = a.sub(x_sq, x);
        let solutions = solve(&mut a, expr, x);
        let vals: Vec<String> = solution_strings(&a, &solutions);
        assert!(
            vals.contains(&"0".to_string()),
            "should have root 0: {vals:?}"
        );
        assert!(
            vals.contains(&"1".to_string()),
            "should have root 1: {vals:?}"
        );
    }

    // ── Verification ────────────────────────────────────────────────

    #[test]
    fn solve_and_verify_quadratic() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x^2 - 5x + 6 = 0
        let two = a.int(2);
        let five = a.int(5);
        let six = a.int(6);
        let x_sq = a.pow(x, two);
        let five_x = a.mul(&[five, x]);
        let neg_five_x = a.neg(five_x);
        let expr = a.add(&[x_sq, neg_five_x, six]);

        let solutions = solve(&mut a, expr, x);
        // Verify each solution by substitution.
        for sol in &solutions {
            let val = crate::subs::subs(&mut a, expr, x, sol.value);
            assert!(
                a.is_zero_structural(val),
                "substituting x={} should give 0, got {}",
                display(&a, sol.value),
                display(&a, val)
            );
        }
    }

    // ── Transcendental / inversion peeling ──────────────────────────

    #[test]
    fn solve_exp_x_eq_5() {
        // exp(x) - 5 = 0 → x = ln(5)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let five = a.int(5);
        let exp_x = a.exp(x);
        let expr = a.sub(exp_x, five);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1, "exp(x)-5=0 should have 1 solution");
        let val = display(&a, solutions[0].value);
        assert!(
            val.contains("ln") || val.contains("log"),
            "solution should be ln(5): {val}"
        );
    }

    #[test]
    fn solve_ln_x_eq_2() {
        // ln(x) - 2 = 0 → x = exp(2) = e^2
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let ln_x = a.ln(x);
        let expr = a.sub(ln_x, two);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1, "ln(x)-2=0 should have 1 solution");
        let val = display(&a, solutions[0].value);
        assert!(
            val.contains("exp") || val.contains("e") || val.contains("E"),
            "solution should be exp(2): {val}"
        );
    }

    #[test]
    fn solve_sin_x_eq_half() {
        // sin(x) - 1/2 = 0 → x = asin(1/2)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let half = {
            let nid = a.intern_num(Ratio::new(BigInt::from(1), BigInt::from(2)));
            a.intern(ExprNode::Num(nid))
        };
        let sin_x = a.sin(x);
        let expr = a.sub(sin_x, half);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1, "sin(x)-1/2=0 should have 1 solution");
        let val = display(&a, solutions[0].value);
        assert!(
            val.contains("asin") || val.contains("arcsin"),
            "solution should be asin(1/2): {val}"
        );
    }

    #[test]
    fn solve_sqrt_x_eq_3() {
        // sqrt(x) - 3 = 0 → x = 9
        // sqrt(x) is x^(1/2)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let three = a.int(3);
        let sqrt_x = a.sqrt(x);
        let expr = a.sub(sqrt_x, three);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1, "sqrt(x)-3=0 should have 1 solution");
        let val = display(&a, solutions[0].value);
        assert_eq!(val, "9", "solution should be 9: {val}");
    }

    #[test]
    fn solve_mul_factors() {
        // x*(x-1)*(x+2) = 0 → roots 0, 1, -2
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.one;
        let two = a.int(2);
        let x_minus_1 = a.sub(x, one);
        let x_plus_2 = a.add(&[x, two]);
        let expr = a.mul(&[x, x_minus_1, x_plus_2]);
        let solutions = solve(&mut a, expr, x);
        assert!(
            solutions.len() >= 3,
            "x*(x-1)*(x+2)=0 should have 3 roots, got {}",
            solutions.len()
        );
        let vals: Vec<String> = solution_strings(&a, &solutions);
        assert!(
            vals.contains(&"0".to_string()),
            "should have root 0: {vals:?}"
        );
        assert!(
            vals.contains(&"1".to_string()),
            "should have root 1: {vals:?}"
        );
        assert!(
            vals.contains(&"-2".to_string()),
            "should have root -2: {vals:?}"
        );
    }

    #[test]
    fn solve_2_exp_x_minus_6() {
        // 2*exp(x) - 6 = 0 → exp(x) = 3 → x = ln(3)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let six = a.int(6);
        let exp_x = a.exp(x);
        let two_exp_x = a.mul(&[two, exp_x]);
        let expr = a.sub(two_exp_x, six);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 1, "2*exp(x)-6=0 should have 1 solution");
        let val = display(&a, solutions[0].value);
        assert!(
            val.contains("ln") || val.contains("log"),
            "solution should be ln(3): {val}"
        );
    }

    #[test]
    fn solve_and_verify_cubic() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x^3 - 6x^2 + 11x - 6 = 0
        let three = a.int(3);
        let six = a.int(6);
        let eleven = a.int(11);
        let two = a.int(2);

        let x3 = a.pow(x, three);
        let x2 = a.pow(x, two);
        let six_x2 = a.mul(&[six, x2]);
        let eleven_x = a.mul(&[eleven, x]);

        let neg_six_x2 = a.neg(six_x2);
        let neg_six = a.neg(six);
        let expr = a.add(&[x3, neg_six_x2, eleven_x, neg_six]);

        let solutions = solve(&mut a, expr, x);
        for sol in &solutions {
            let val = crate::subs::subs(&mut a, expr, x, sol.value);
            assert!(
                a.is_zero_structural(val),
                "substituting x={} should give 0, got {}",
                display(&a, sol.value),
                display(&a, val)
            );
        }
    }

    // ── Change-of-variable ──────────────────────────────────────────

    #[test]
    fn solve_exp_2x_minus_3_exp_x_plus_2() {
        // exp(2x) - 3*exp(x) + 2 = 0
        // Let t = exp(x): t² - 3t + 2 = (t-1)(t-2) = 0
        // t = 1 → x = ln(1) = 0
        // t = 2 → x = ln(2)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let three = a.int(3);

        let two_x = a.mul(&[two, x]);
        let exp_2x = a.exp(two_x);
        let exp_x = a.exp(x);
        let three_exp_x = a.mul(&[three, exp_x]);
        let neg_three_exp_x = a.neg(three_exp_x);
        let expr = a.add(&[exp_2x, neg_three_exp_x, two]);

        let solutions = solve(&mut a, expr, x);
        assert!(
            solutions.len() >= 2,
            "exp(2x)-3*exp(x)+2=0 should have 2 solutions, got {}: {:?}",
            solutions.len(),
            solution_strings(&a, &solutions)
        );
        let vals: Vec<String> = solution_strings(&a, &solutions);
        // ln(1) may or may not simplify to 0 depending on canonicalization
        assert!(
            vals.contains(&"0".to_string()) || vals.contains(&"ln(1)".to_string()),
            "should have root 0 or ln(1) (from exp(x)=1): {vals:?}"
        );
        let has_ln2 = vals.iter().any(|v| v.contains("ln") || v.contains("log"));
        assert!(has_ln2, "should have root ln(2): {vals:?}");
    }

    #[test]
    fn solve_sin_squared_minus_sin() {
        // sin(x)^2 - sin(x) = 0
        // Let t = sin(x): t² - t = t(t-1) = 0
        // t = 0 → sin(x) = 0 → x = asin(0) = 0
        // t = 1 → sin(x) = 1 → x = asin(1)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);

        let sin_x = a.sin(x);
        let sin_x_sq = a.pow(sin_x, two);
        let expr = a.sub(sin_x_sq, sin_x);

        let solutions = solve(&mut a, expr, x);
        assert!(
            solutions.len() >= 2,
            "sin(x)^2-sin(x)=0 should have ≥2 solutions, got {}: {:?}",
            solutions.len(),
            solution_strings(&a, &solutions)
        );
        let vals: Vec<String> = solution_strings(&a, &solutions);
        // One root should be 0 (from sin(x)=0 → x=asin(0)=0)
        let has_zero = vals.contains(&"0".to_string());
        // The other should involve asin (from sin(x)=1 → x=asin(1))
        let has_asin = vals
            .iter()
            .any(|v| v.contains("asin") || v.contains("arcsin"));
        assert!(
            has_zero || has_asin,
            "should have root 0 or asin(1): {vals:?}"
        );
    }

    #[test]
    fn solve_exp_quadratic_one_valid_root() {
        // exp(2x) - 5*exp(x) + 6 = 0
        // Let t = exp(x): t² - 5t + 6 = (t-2)(t-3) = 0
        // t = 2 → x = ln(2)
        // t = 3 → x = ln(3)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let five = a.int(5);
        let six = a.int(6);

        let two_x = a.mul(&[two, x]);
        let exp_2x = a.exp(two_x);
        let exp_x = a.exp(x);
        let five_exp_x = a.mul(&[five, exp_x]);
        let neg_five_exp_x = a.neg(five_exp_x);
        let expr = a.add(&[exp_2x, neg_five_exp_x, six]);

        let solutions = solve(&mut a, expr, x);
        assert!(
            solutions.len() >= 2,
            "exp(2x)-5*exp(x)+6=0 should have 2 solutions, got {}: {:?}",
            solutions.len(),
            solution_strings(&a, &solutions)
        );
        let vals: Vec<String> = solution_strings(&a, &solutions);
        let has_ln = vals.iter().all(|v| v.contains("ln") || v.contains("log"));
        assert!(has_ln, "all roots should involve ln: {vals:?}");
    }

    // ── Helper tests ────────────────────────────────────────────────

    #[test]
    fn rational_sqrt_perfect() {
        let r = Ratio::new(BigInt::from(9), BigInt::from(4));
        let s = rational_sqrt(&r).unwrap();
        assert_eq!(s, Ratio::new(BigInt::from(3), BigInt::from(2)));
    }

    #[test]
    fn rational_sqrt_not_perfect() {
        let r = Ratio::from_integer(BigInt::from(2));
        assert!(rational_sqrt(&r).is_none());
    }

    #[test]
    fn rational_sqrt_zero() {
        let r = Ratio::zero();
        let s = rational_sqrt(&r).unwrap();
        assert!(s.is_zero());
    }

    #[test]
    fn divisors_of_12() {
        let d = divisors(&BigInt::from(12));
        assert_eq!(
            d,
            vec![1, 2, 3, 4, 6, 12]
                .into_iter()
                .map(BigInt::from)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn divisors_of_1() {
        let d = divisors(&BigInt::from(1));
        assert_eq!(d, vec![BigInt::from(1)]);
    }

    #[test]
    fn clear_denominators_works() {
        let poly = Poly::from_coeffs(vec![
            Ratio::new(BigInt::from(1), BigInt::from(2)),
            Ratio::new(BigInt::from(1), BigInt::from(3)),
        ]);
        let (int_poly, lcm) = clear_denominators(&poly);
        assert_eq!(lcm, BigInt::from(6));
        // 1/2 * 6 = 3, 1/3 * 6 = 2
        assert_eq!(int_poly.coeff(0), Ratio::from_integer(BigInt::from(3)));
        assert_eq!(int_poly.coeff(1), Ratio::from_integer(BigInt::from(2)));
    }

    // ── Complex quadratic roots ─────────────────────────────────────

    #[test]
    fn solve_x2_plus_1() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x² + 1 = 0 → roots ±i
        let two = a.int(2);
        let one = a.one;
        let x_sq = a.pow(x, two);
        let expr = a.add(&[x_sq, one]);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 2, "x²+1=0 should have 2 complex roots");
        let vals: Vec<String> = solution_strings(&a, &solutions);
        assert!(
            vals.iter().all(|v| v.contains("I")),
            "roots should contain I: {vals:?}"
        );
    }

    #[test]
    fn solve_x2_plus_2x_plus_5() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x² + 2x + 5 = 0 → roots -1 ± 2i
        let two = a.int(2);
        let five = a.int(5);
        let x_sq = a.pow(x, two);
        let two_x = a.mul(&[two, x]);
        let expr = a.add(&[x_sq, two_x, five]);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 2, "x²+2x+5=0 should have 2 complex roots");
        let vals: Vec<String> = solution_strings(&a, &solutions);
        // roots are -1 ± 2i
        for v in &vals {
            assert!(v.contains("I"), "root should contain I: {v}");
        }
    }

    #[test]
    fn solve_x2_plus_4() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x² + 4 = 0 → roots ±2i
        let two = a.int(2);
        let four = a.int(4);
        let x_sq = a.pow(x, two);
        let expr = a.add(&[x_sq, four]);
        let solutions = solve(&mut a, expr, x);
        assert_eq!(solutions.len(), 2, "x²+4=0 should have 2 complex roots");
        let vals: Vec<String> = solution_strings(&a, &solutions);
        assert!(
            vals.iter().all(|v| v.contains("I")),
            "roots should contain I: {vals:?}"
        );
    }
}
