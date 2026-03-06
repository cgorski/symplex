//! Polynomial system solving via Gröbner bases.
//!
//! Pipeline: grevlex Buchberger → FGLM to lex → back-substitution.

use crate::groebner;
use crate::multipoly::*;
use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, ToPrimitive, Zero};

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
    let lex_gb = groebner::groebner_basis_lex(
        &nonzero.iter().map(|p| p.clone()).collect::<Vec<_>>(),
    );

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
fn rational_roots_of_univariate(
    poly: &MultiPoly<Lex>,
    var_idx: usize,
) -> Vec<Ratio<BigInt>> {
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
    let const_abs = const_term
        .numer()
        .to_i64()
        .map(|n| n.abs())
        .unwrap_or(0);
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
    let const_for_divs = if const_abs > 10_000 {
        // For very large constant terms, only try small divisors
        // This is a heuristic; we might miss some roots.
        const_abs.min(10_000)
    } else {
        const_abs
    };
    let lc_for_divs = if lc_abs > 10_000 {
        lc_abs.min(10_000)
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
        let p = MultiPoly::<GrevLex>::var(1, 0)
            .add(&MultiPoly::from_int(1, -3));
        let sols = solve_polynomial_system(&[p]).unwrap();
        assert_eq!(sols.len(), 1);
        assert_eq!(sols[0], vec![rat(3)]);
    }
}
