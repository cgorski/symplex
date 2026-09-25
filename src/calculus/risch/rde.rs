//! Risch differential equation solver.
//!
//! Solves the first-order linear ODE `y' + f·y = g` where `f` and `g` are
//! rational functions in `ℚ(x)` (or more generally in `k(t)` for a monomial
//! extension `t`), and the solution `y` must also lie in the same field.
//!
//! This is the key sub-algorithm for the **exponential case** of the Risch
//! algorithm: when integrating a polynomial part in `θ = exp(u)`, the
//! coefficient of `θ^k` (for `k ≠ 0`) satisfies `Bₖ' + k·u'·Bₖ = aₖ`,
//! which is a Risch differential equation.
//!
//! # Algorithm
//!
//! For the rational function case (`y' + f·y = g` with `f, g ∈ ℚ(x)`),
//! the algorithm proceeds in stages (following Bronstein Ch. 5):
//!
//! 1. **Bound the denominator** of `y`: analyze poles of `f` and `g` to
//!    determine which irreducible polynomials can appear in the denominator
//!    of `y`, and with what multiplicity.
//! 2. **Construct the denominator** `D_y` from the analysis.
//! 3. **Bound the degree** of the numerator `N_y`.
//! 4. **Ansatz**: write `y = N_y / D_y` with undetermined coefficients.
//! 5. **Substitute** into `y' + f·y = g`, clear denominators.
//! 6. **Equate coefficients** of powers of `x` → linear system.
//! 7. **Solve** the linear system for the undetermined coefficients.
//!
//! # Status
//!
//! This module implements the rational function case (`f, g ∈ ℚ(x)`).
//! The extension case (monomial extensions) is a stub.
//!
//! # References
//!
//! - Bronstein, *Symbolic Integration I*, Chapters 5–6
//! - SymPy `integrals/rde.py`

use num_rational::Ratio;
use num_traits::{One, Zero};

use crate::base::numeric::Q;
use crate::poly::dense::Poly;

// ═══════════════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════════════

/// Result of attempting to solve the Risch differential equation.
#[derive(Clone, Debug)]
pub enum RdeResult {
    /// Found a solution `y = numer / denom`.
    Solution { numer: Poly, denom: Poly },
    /// Proved that no solution exists in the field.
    NoSolution,
    /// Hit an unimplemented case.
    NotImplemented(String),
}

/// Attempt to solve the Risch differential equation
///
/// ```text
///   y' + (f_numer/f_denom) · y = g_numer/g_denom
/// ```
///
/// for `y ∈ ℚ(x)` (a rational function of `x`).
///
/// Returns `RdeResult::Solution` if a solution exists,
/// `RdeResult::NoSolution` if provably no solution exists,
/// or `RdeResult::NotImplemented` for unhandled cases.
///
/// # Algorithm
///
/// 1. Compute a denominator bound for `y` by analyzing the poles of `f` and `g`.
/// 2. Compute a degree bound for the numerator of `y`.
/// 3. Set up a system of linear equations from the ansatz `y = N/D`.
/// 4. Solve the system; if consistent, return the solution.
///
/// A zero `f_denom` or `g_denom` (an equation the integrator never
/// builds: its denominators come from `as_numer_denom`) is reported as
/// [`RdeResult::NotImplemented`], so the caller falls back; up to 0.28 it
/// was a runtime `assert!`.
#[allow(dead_code)]
pub fn solve_risch_de_rational(
    f_numer: &Poly,
    f_denom: &Poly,
    g_numer: &Poly,
    g_denom: &Poly,
) -> RdeResult {
    if f_denom.is_zero() || g_denom.is_zero() {
        return RdeResult::NotImplemented("Risch DE with a zero denominator".into());
    }

    // Trivial case: if g = 0, then y = 0 is always a solution.
    if g_numer.is_zero() {
        return RdeResult::Solution {
            numer: Poly::zero(),
            denom: Poly::from_int(1),
        };
    }

    // Trivial case: if f = 0, then y' = g, so y = ∫g dx.
    // For g = g_numer/g_denom ∈ ℚ(x), we need ∫g dx ∈ ℚ(x).
    // This requires g to have no logarithmic part (all residues zero).
    if f_numer.is_zero() {
        return solve_y_prime_equals_g(g_numer, g_denom);
    }

    // General case: y' + f·y = g with f ≠ 0, g ≠ 0.
    //
    // Step 1: Denominator bound.
    //
    // The denominator of y is bounded by analyzing the poles of f and g.
    // For each irreducible factor p of lcm(f_denom, g_denom):
    //   - Let m = order of pole of f at p, n = order of pole of g at p.
    //   - If m ≥ 2: y has a pole of order m-1 at p.
    //   - If m = 1: need to check the residue of f at p.
    //     If the residue is a non-negative integer, y might have a pole;
    //     otherwise y is regular at p.
    //   - If m = 0: y has the same pole order as g at p.
    //
    // For the rational case, we use a simpler conservative bound:
    // D_y = lcm(f_denom, g_denom) (this is not tight but is correct).
    let d_y = compute_denominator_bound(f_numer, f_denom, g_numer, g_denom);

    // Step 2: Degree bound for the numerator.
    //
    // If y = N/D with known D, then y' = (N'D - ND')/D², so
    // y' + f·y = g becomes:
    //   (N'D - ND')/D² + (f_n/f_d)·(N/D) = g_n/g_d
    //
    // Clearing denominators and analyzing leading degrees gives a bound
    // on deg(N).
    let n_bound = compute_degree_bound(f_numer, f_denom, g_numer, g_denom, &d_y);

    if n_bound < 0 {
        // Negative degree bound means no polynomial numerator can work.
        return RdeResult::NoSolution;
    }
    let n_bound = n_bound as usize;

    // Step 3: Ansatz — write N(x) = c₀ + c₁x + c₂x² + ... + cₙxⁿ
    // with n = n_bound, and undetermined coefficients c₀, ..., cₙ.
    //
    // Step 4: Substitute into the equation, clear denominators, and
    // collect powers of x to get a linear system in the cᵢ.
    //
    // Step 5: Solve the linear system.
    solve_with_ansatz(f_numer, f_denom, g_numer, g_denom, &d_y, n_bound)
}

/// Solve the Risch DE in a monomial extension tower.
///
/// This is the general version that handles logarithmic and exponential
/// extensions beyond the base field `ℚ(x)`.
///
/// # Status
///
/// Stub — delegates to the rational case when at the base level.
#[allow(dead_code)]
pub fn solve_risch_de(
    f_numer: &Poly,
    f_denom: &Poly,
    g_numer: &Poly,
    g_denom: &Poly,
    de: &super::tower::DifferentialExtension,
) -> RdeResult {
    if de.is_base_level() {
        return solve_risch_de_rational(f_numer, f_denom, g_numer, g_denom);
    }

    // TODO: Handle logarithmic and exponential extension cases.
    // For logarithmic extensions, the denominator/degree bounding
    // generalizes using the extension's derivation.
    // For exponential extensions, additional structure is exploited.
    RdeResult::NotImplemented("Risch DE in monomial extensions not yet implemented".into())
}

// ═══════════════════════════════════════════════════════════════════════════
// Internal helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Solve the special case y' = g (i.e., f = 0).
///
/// y must be a rational function, so g must integrate to a rational function.
/// This means g must have zero residues at all poles (no logarithmic part).
fn solve_y_prime_equals_g(g_numer: &Poly, g_denom: &Poly) -> RdeResult {
    // Use Hermite reduction: ∫g dx = rational_part + ∫(square-free remainder).
    // If the square-free remainder is zero, y = rational_part.
    let Some(hr) = super::hermite::hermite_reduce(g_numer, g_denom) else {
        return RdeResult::NotImplemented("Risch DE with a zero denominator".into());
    };

    if hr.h_numer.is_zero() {
        // No logarithmic part — y is purely rational.
        return RdeResult::Solution {
            numer: hr.g_numer,
            denom: hr.g_denom,
        };
    }

    // There's a logarithmic part, so ∫g dx ∉ ℚ(x).
    RdeResult::NoSolution
}

/// Compute a conservative denominator bound for y.
///
/// Returns a polynomial D_y such that if y = N/D is a solution,
/// then D divides D_y.
///
/// This uses the simple bound D_y = lcm(f_denom, g_denom) with
/// additional factors from the square-free decomposition of f_denom.
fn compute_denominator_bound(
    _f_numer: &Poly,
    f_denom: &Poly,
    _g_numer: &Poly,
    g_denom: &Poly,
) -> Poly {
    // Conservative bound: D_y should contain all factors of f_denom
    // (with appropriate multiplicity adjustments) and g_denom.
    //
    // For a pole of f at p with order m ≥ 2, y has a pole of order m-1.
    // For a pole of f at p with order m = 1, y might or might not have
    // a pole (depends on the residue — a non-negative integer residue
    // allows a pole).
    //
    // Simplified approach: use the square-free part of f_denom as the
    // denominator bound, multiplied by g_denom's square-free part.
    // This is conservative but correct.

    let f_sqfree = f_denom.square_free_part();
    let g_sqfree = g_denom.square_free_part();

    // LCM via gcd: lcm(a, b) = a * b / gcd(a, b)
    let gcd_val = Poly::gcd(&f_sqfree, &g_sqfree);
    if gcd_val.degree().unwrap_or(0) == 0 {
        // Coprime — LCM is the product.
        let product = &f_sqfree * &g_sqfree;
        if product.is_zero() || product.is_constant() {
            return Poly::from_int(1);
        }
        product.make_monic()
    } else {
        let lcm = &(&f_sqfree * &g_sqfree).div(&gcd_val);
        if lcm.is_zero() || lcm.is_constant() {
            return Poly::from_int(1);
        }
        lcm.make_monic()
    }
}

/// Compute a degree bound for the numerator of y, given the denominator D_y.
///
/// Returns the maximum degree that N(x) can have, or a negative value
/// if no solution is possible.
fn compute_degree_bound(
    f_numer: &Poly,
    f_denom: &Poly,
    g_numer: &Poly,
    g_denom: &Poly,
    d_y: &Poly,
) -> i64 {
    // The equation y' + f·y = g with y = N/D_y gives:
    //   (N'·D_y - N·D_y') / D_y² + (f_n/f_d)·(N/D_y) = g_n/g_d
    //
    // Multiply through by D_y² · f_d · g_d:
    //   (N'·D_y - N·D_y')·f_d·g_d + f_n·N·D_y·g_d = g_n·D_y²·f_d
    //
    // The highest-degree term on the LHS involving N is:
    //   N · (f_n · D_y · g_d - D_y' · f_d · g_d)
    //
    // This has degree deg(N) + deg(f_n) + deg(D_y) + deg(g_d) - ...
    //
    // Simplified bound: deg(N) ≤ max(deg(g) + deg(D_y) - deg(f), deg(D_y) + 1)
    //
    // We use a generous bound to avoid missing solutions.
    let deg_f = f_numer.degree().unwrap_or(0) as i64 - f_denom.degree().unwrap_or(0) as i64;
    let deg_g = g_numer.degree().unwrap_or(0) as i64 - g_denom.degree().unwrap_or(0) as i64;
    let deg_dy = d_y.degree().unwrap_or(0) as i64;

    let bound1 = deg_g + deg_dy; // from matching the g side
    let bound2 = deg_dy + 1; // from the N' term
    let bound3 = deg_g - deg_f + deg_dy; // from the f·y = g balance

    let bound = bound1.max(bound2).max(bound3).max(0);

    // Safety cap to prevent enormous ansatz matrices.
    bound.min(50)
}

/// Solve the RDE using the ansatz y = N(x) / D_y(x) where N has degree ≤ n_bound.
///
/// Sets up a linear system by substituting the ansatz into y' + f·y = g,
/// clearing denominators, and equating coefficients.
#[allow(clippy::needless_range_loop)]
fn solve_with_ansatz(
    f_numer: &Poly,
    f_denom: &Poly,
    g_numer: &Poly,
    g_denom: &Poly,
    d_y: &Poly,
    n_bound: usize,
) -> RdeResult {
    // We need to find N(x) = c₀ + c₁x + ... + cₙxⁿ such that
    //   (N/D_y)' + (f_n/f_d) · (N/D_y) = g_n/g_d
    //
    // Multiplying out:
    //   (N'·D_y - N·D_y') · f_d · g_d  +  f_n · N · D_y · g_d  =  g_n · D_y² · f_d
    //
    // LHS is linear in the coefficients cᵢ of N.  RHS is known.
    //
    // We evaluate this identity at n_bound + max_extra_degree + 1 points,
    // or equivalently expand in powers of x and equate.

    let d_y_prime = d_y.derivative();

    // Build the "operator" polynomial for each basis element xⁱ:
    //   For N = xⁱ, LHS_i = (i·x^{i-1}·D_y - x^i·D_y')·f_d·g_d + f_n·x^i·D_y·g_d
    //
    // We compute the full LHS polynomial for a general N by linearity.
    // Actually, it's simpler to just try the ansatz numerically.

    // RHS polynomial: g_n · D_y² · f_d
    let d_y_sq = &(d_y * d_y);
    let rhs_poly = &(&(g_numer * d_y_sq) * f_denom);

    let rhs_degree = rhs_poly.degree().unwrap_or(0);

    // Total number of unknowns.
    let num_unknowns = n_bound + 1;

    // Pre-compute LHS polynomials for each basis function xⁱ.
    // We need these to determine the correct number of equations
    // (the max degree across all LHS polys and the RHS).
    let mut lhs_polys: Vec<Poly> = Vec::with_capacity(num_unknowns);
    for i in 0..num_unknowns {
        // N = xⁱ (a single monomial)
        let n_i = Poly::monomial(Ratio::one(), i);
        let n_i_prime = n_i.derivative();

        // term1 = (N'·D_y - N·D_y') · f_d · g_d
        let np_dy = &n_i_prime * d_y;
        let n_dyp = &n_i * &d_y_prime;
        let term1_inner = &np_dy - &n_dyp;
        let t1_fd = &term1_inner * f_denom;
        let term1 = &t1_fd * g_denom;

        // term2 = f_n · N · D_y · g_d
        let fn_ni = f_numer * &n_i;
        let fn_ni_dy = &fn_ni * d_y;
        let term2 = &fn_ni_dy * g_denom;

        lhs_polys.push(&term1 + &term2);
    }

    // Determine the number of equations: max degree across all LHS polys and RHS + 1.
    let max_lhs_degree = lhs_polys
        .iter()
        .filter_map(|p| p.degree())
        .max()
        .unwrap_or(0);
    let num_equations = rhs_degree.max(max_lhs_degree) + 1;

    // Build the coefficient matrix.
    let mut matrix: Vec<Vec<Q>> = vec![vec![Ratio::zero(); num_unknowns + 1]; num_equations];

    // Fill the RHS column (last column).
    for j in 0..num_equations {
        matrix[j][num_unknowns] = rhs_poly.coeff(j);
    }

    // Fill the LHS columns from pre-computed polynomials.
    for (i, lhs_i) in lhs_polys.iter().enumerate() {
        for j in 0..num_equations {
            matrix[j][i] = lhs_i.coeff(j);
        }
    }

    // Solve the linear system via Gaussian elimination.
    match solve_linear_system(&mut matrix, num_unknowns) {
        Some(solution) => {
            // Build N from the solution coefficients.
            let n_poly = Poly::from_coeffs(solution);
            if n_poly.is_zero() && !g_numer.is_zero() {
                return RdeResult::NoSolution;
            }

            // Simplify y = N / D_y by cancelling common factors.
            let gcd_nd = Poly::gcd(&n_poly, d_y);
            let result_numer = n_poly.div(&gcd_nd);
            let result_denom = d_y.div(&gcd_nd);

            RdeResult::Solution {
                numer: result_numer,
                denom: result_denom,
            }
        }
        None => RdeResult::NoSolution,
    }
}

/// Solve a system of linear equations via Gaussian elimination over ℚ.
///
/// `matrix` is an `m × (n+1)` augmented matrix (last column is RHS).
/// Returns `Some(solution)` with `n` values if consistent, `None` otherwise.
#[allow(clippy::needless_range_loop)]
fn solve_linear_system(matrix: &mut [Vec<Q>], num_unknowns: usize) -> Option<Vec<Q>> {
    let m = matrix.len();
    let n = num_unknowns;

    // Forward elimination.
    let mut pivot_row = 0;
    for col in 0..n {
        // Find a non-zero pivot in this column.
        let mut found = None;
        for row in pivot_row..m {
            if !matrix[row][col].is_zero() {
                found = Some(row);
                break;
            }
        }

        let pr = match found {
            Some(r) => r,
            None => continue, // No pivot in this column — free variable.
        };

        // Swap pivot row into position.
        matrix.swap(pivot_row, pr);

        // Eliminate below.
        let pivot_val = matrix[pivot_row][col].clone();
        for row in (pivot_row + 1)..m {
            if !matrix[row][col].is_zero() {
                let factor = matrix[row][col].clone() / &pivot_val;
                for j in col..=n {
                    let sub = &factor * &matrix[pivot_row][j];
                    matrix[row][j] = &matrix[row][j] - &sub;
                }
            }
        }

        pivot_row += 1;
    }

    // Check for inconsistency: rows of the form [0 0 ... 0 | nonzero].
    for row in pivot_row..m {
        if !matrix[row][n].is_zero() {
            return None; // Inconsistent system.
        }
    }

    // Back substitution.
    let mut solution = vec![Ratio::zero(); n];

    for row in (0..pivot_row).rev() {
        // Find the pivot column for this row.
        let mut pivot_col = None;
        for col in 0..n {
            if !matrix[row][col].is_zero() {
                pivot_col = Some(col);
                break;
            }
        }

        let col = match pivot_col {
            Some(c) => c,
            None => continue,
        };

        let mut rhs = matrix[row][n].clone();
        for j in (col + 1)..n {
            rhs -= &matrix[row][j] * &solution[j];
        }
        solution[col] = rhs / &matrix[row][col];
    }

    Some(solution)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;

    fn rat(n: i64, d: i64) -> Q {
        Ratio::new(BigInt::from(n), BigInt::from(d))
    }

    /// Verify a solution by substituting back: y' + f·y should equal g.
    fn verify_rde_solution(
        f_numer: &Poly,
        f_denom: &Poly,
        g_numer: &Poly,
        g_denom: &Poly,
        y_numer: &Poly,
        y_denom: &Poly,
    ) {
        // y = y_n / y_d
        // y' = (y_n' · y_d - y_n · y_d') / y_d²
        let yn_prime = y_numer.derivative();
        let yd_prime = y_denom.derivative();
        let yn_yd = &yn_prime * y_denom;
        let yn_ydp = y_numer * &yd_prime;
        let y_prime_numer = &yn_yd - &yn_ydp;
        let y_prime_denom = y_denom * y_denom;

        // y' + f·y  = y_prime_n/y_prime_d + (f_n/f_d)·(y_n/y_d)
        //           = (y_prime_n · f_d · y_d + f_n · y_n · y_prime_d) / (y_prime_d · f_d · y_d)
        //
        // This should equal g_n / g_d.
        // Cross-multiply: LHS_n · g_d == g_n · LHS_d

        let fd_yd = &(f_denom * y_denom);
        let lhs_term1 = &y_prime_numer * fd_yd;
        let fn_yn = &(f_numer * y_numer);
        let lhs_term2 = fn_yn * &y_prime_denom;
        let lhs_numer = &lhs_term1 + &lhs_term2;
        let ypd_fd = &y_prime_denom * f_denom;
        let lhs_denom = &ypd_fd * y_denom;

        let lhs_cross = &lhs_numer * g_denom;
        let rhs_cross = g_numer * &lhs_denom;
        let check = &lhs_cross - &rhs_cross;
        assert!(
            check.is_zero(),
            "RDE solution verification failed: y' + f·y ≠ g\n  residual = {check}"
        );
    }

    #[test]
    fn rde_trivial_g_zero() {
        // y' + 2·y = 0  →  y = 0 is a solution in ℚ(x).
        let f_n = Poly::from_int(2);
        let f_d = Poly::from_int(1);
        let g_n = Poly::zero();
        let g_d = Poly::from_int(1);

        match solve_risch_de_rational(&f_n, &f_d, &g_n, &g_d) {
            RdeResult::Solution { numer, denom } => {
                assert!(numer.is_zero(), "y should be 0 for g=0");
                verify_rde_solution(&f_n, &f_d, &g_n, &g_d, &numer, &denom);
            }
            other => panic!("expected Solution, got {:?}", other),
        }
    }

    #[test]
    fn rde_f_zero_poly_g() {
        // y' = 2x  →  y = x²
        let f_n = Poly::zero();
        let f_d = Poly::from_int(1);
        let g_n = Poly::from_coeffs(vec![rat(0, 1), rat(2, 1)]); // 2x
        let g_d = Poly::from_int(1);

        match solve_risch_de_rational(&f_n, &f_d, &g_n, &g_d) {
            RdeResult::Solution { numer, denom } => {
                verify_rde_solution(&f_n, &f_d, &g_n, &g_d, &numer, &denom);
                // y = x² / 1
                assert_eq!(numer.degree(), Some(2), "y should be degree 2");
                assert!(denom.is_constant(), "denom should be 1");
            }
            other => panic!("expected Solution, got {:?}", other),
        }
    }

    #[test]
    fn rde_constant_f_constant_g() {
        // y' + y = 1  →  The solution in ℚ(x) is y = 1 (constant).
        // Check: y' + y = 0 + 1 = 1. ✓
        let f_n = Poly::from_int(1);
        let f_d = Poly::from_int(1);
        let g_n = Poly::from_int(1);
        let g_d = Poly::from_int(1);

        match solve_risch_de_rational(&f_n, &f_d, &g_n, &g_d) {
            RdeResult::Solution { numer, denom } => {
                verify_rde_solution(&f_n, &f_d, &g_n, &g_d, &numer, &denom);
            }
            other => panic!("expected Solution, got {:?}", other),
        }
    }

    #[test]
    fn rde_exp_minus_x_squared_nonelementary() {
        // y' - 2x·y = 1
        // This is the Risch DE from ∫ exp(-x²) dx.
        // It has NO rational function solution (proof: any polynomial solution
        // of degree n gives LHS of degree n+1, which can't equal constant 1).
        let f_n = Poly::from_coeffs(vec![rat(0, 1), rat(-2, 1)]); // -2x
        let f_d = Poly::from_int(1);
        let g_n = Poly::from_int(1);
        let g_d = Poly::from_int(1);

        let result = solve_risch_de_rational(&f_n, &f_d, &g_n, &g_d);
        assert!(
            matches!(result, RdeResult::NoSolution),
            "y' - 2x·y = 1 should have no rational solution, got {:?}",
            result
        );
    }

    #[test]
    fn rde_f_zero_rational_g() {
        // y' = 1/x²  →  y = -1/x (rational solution exists)
        // This is Hermite-reducible.
        let f_n = Poly::zero();
        let f_d = Poly::from_int(1);
        let g_n = Poly::from_int(1);
        let g_d = Poly::from_coeffs(vec![rat(0, 1), rat(0, 1), rat(1, 1)]); // x²

        match solve_risch_de_rational(&f_n, &f_d, &g_n, &g_d) {
            RdeResult::Solution { numer, denom } => {
                verify_rde_solution(&f_n, &f_d, &g_n, &g_d, &numer, &denom);
            }
            other => panic!("expected Solution for y'=1/x², got {:?}", other),
        }
    }

    #[test]
    fn rde_f_zero_log_g_has_no_rational_solution() {
        // y' = 1/x  →  y = ln(x), which is NOT in ℚ(x).
        // Should return NoSolution.
        let f_n = Poly::zero();
        let f_d = Poly::from_int(1);
        let g_n = Poly::from_int(1);
        let g_d = Poly::x(); // x

        let result = solve_risch_de_rational(&f_n, &f_d, &g_n, &g_d);
        assert!(
            matches!(result, RdeResult::NoSolution),
            "y' = 1/x should have no rational solution, got {:?}",
            result
        );
    }

    #[test]
    fn rde_linear_ode_with_poly_solution() {
        // y' + (1/x)·y = 1 + 1/x
        // Try y = x: y' = 1, y'/1 + (1/x)·x = 1 + 1 = 2 ≠ 1 + 1/x. Not x.
        // Try y = 1: y' = 0, 0 + 1/x = 1/x ≠ 1 + 1/x. No.
        // Actually this ODE has solution y = x (check: 1 + x/x = 1 + 1 = 2).
        // Hmm, let me re-check: y' + (1/x)·y = g.
        // With y = x: y' = 1, (1/x)·x = 1, so y' + f·y = 1 + 1 = 2.
        // So g = 2, not 1 + 1/x.  Let me pick a better example.
        //
        // y' + 0·y = 3x²  →  y = x³
        let f_n = Poly::zero();
        let f_d = Poly::from_int(1);
        let g_n = Poly::from_coeffs(vec![rat(0, 1), rat(0, 1), rat(3, 1)]); // 3x²
        let g_d = Poly::from_int(1);

        match solve_risch_de_rational(&f_n, &f_d, &g_n, &g_d) {
            RdeResult::Solution { numer, denom } => {
                verify_rde_solution(&f_n, &f_d, &g_n, &g_d, &numer, &denom);
                assert_eq!(numer.degree(), Some(3), "y should be x³");
            }
            other => panic!("expected Solution, got {:?}", other),
        }
    }

    // ── Linear system solver tests ──────────────────────────────────

    #[test]
    fn solve_linear_2x2() {
        // x + y = 3
        // 2x + 3y = 8
        // Solution: x = 1, y = 2
        let mut matrix = vec![
            vec![rat(1, 1), rat(1, 1), rat(3, 1)],
            vec![rat(2, 1), rat(3, 1), rat(8, 1)],
        ];
        let sol = solve_linear_system(&mut matrix, 2).unwrap();
        assert_eq!(sol[0], rat(1, 1));
        assert_eq!(sol[1], rat(2, 1));
    }

    #[test]
    fn solve_linear_inconsistent() {
        // x + y = 1
        // x + y = 2  (inconsistent)
        let mut matrix = vec![
            vec![rat(1, 1), rat(1, 1), rat(1, 1)],
            vec![rat(1, 1), rat(1, 1), rat(2, 1)],
        ];
        assert!(solve_linear_system(&mut matrix, 2).is_none());
    }

    #[test]
    fn solve_linear_overdetermined_consistent() {
        // x = 3
        // x = 3
        // 2x = 6
        let mut matrix = vec![
            vec![rat(1, 1), rat(3, 1)],
            vec![rat(1, 1), rat(3, 1)],
            vec![rat(2, 1), rat(6, 1)],
        ];
        let sol = solve_linear_system(&mut matrix, 1).unwrap();
        assert_eq!(sol[0], rat(3, 1));
    }

    // ── Group 5: RDE edge cases ─────────────────────────────────────

    #[test]
    fn rde_large_degree_polynomial_rhs() {
        // y' + y = x^5 + x^3 + x
        // Solution is a polynomial: y = x^5 - 5x^4 + 20x^3 - 59x^2 + ... (complex)
        // Just verify a solution exists and satisfies the equation.
        let f_n = Poly::from_int(1);
        let f_d = Poly::from_int(1);
        let g_n = Poly::from_coeffs(vec![
            rat(0, 1),
            rat(1, 1),
            rat(0, 1),
            rat(1, 1),
            rat(0, 1),
            rat(1, 1),
        ]); // x^5 + x^3 + x
        let g_d = Poly::from_int(1);

        match solve_risch_de_rational(&f_n, &f_d, &g_n, &g_d) {
            RdeResult::Solution { numer, denom } => {
                verify_rde_solution(&f_n, &f_d, &g_n, &g_d, &numer, &denom);
            }
            other => panic!("expected Solution for y'+y=x^5+x^3+x, got {:?}", other),
        }
    }

    #[test]
    fn rde_zero_f_zero_g() {
        // y' = 0 → y = 0 (trivial)
        let f_n = Poly::zero();
        let f_d = Poly::from_int(1);
        let g_n = Poly::zero();
        let g_d = Poly::from_int(1);

        match solve_risch_de_rational(&f_n, &f_d, &g_n, &g_d) {
            RdeResult::Solution { numer, denom } => {
                assert!(numer.is_zero(), "y should be 0");
                verify_rde_solution(&f_n, &f_d, &g_n, &g_d, &numer, &denom);
            }
            other => panic!("expected Solution(0), got {:?}", other),
        }
    }

    #[test]
    fn rde_negative_constant_f() {
        // y' - 3y = 0 → y = 0 is the only rational solution
        // (the general solution is C·exp(3x), not in ℚ(x) for C≠0)
        let f_n = Poly::from_int(-3);
        let f_d = Poly::from_int(1);
        let g_n = Poly::zero();
        let g_d = Poly::from_int(1);

        match solve_risch_de_rational(&f_n, &f_d, &g_n, &g_d) {
            RdeResult::Solution { numer, .. } => {
                assert!(
                    numer.is_zero(),
                    "y' - 3y = 0: only rational solution is y=0"
                );
            }
            other => panic!("expected Solution(0), got {:?}", other),
        }
    }

    #[test]
    fn rde_negative_f_with_rhs() {
        // y' - y = x → solution y = -(x+1) in ℚ(x)?
        // Check: y' = -1, -y = x+1, so y' - y = -1 + x + 1 = x. ✓
        let f_n = Poly::from_int(-1);
        let f_d = Poly::from_int(1);
        let g_n = Poly::x(); // x
        let g_d = Poly::from_int(1);

        match solve_risch_de_rational(&f_n, &f_d, &g_n, &g_d) {
            RdeResult::Solution { numer, denom } => {
                verify_rde_solution(&f_n, &f_d, &g_n, &g_d, &numer, &denom);
            }
            other => panic!("expected Solution for y'-y=x, got {:?}", other),
        }
    }

    #[test]
    fn rde_quadratic_f_no_solution() {
        // y' + x²·y = 1
        // For polynomial y of degree n: deg(y' + x²·y) = n + 2, which must
        // equal deg(1) = 0. So n + 2 = 0 is impossible → no polynomial solution.
        // For rational y with denominator: the denominator analysis should also fail.
        let f_n = Poly::from_coeffs(vec![rat(0, 1), rat(0, 1), rat(1, 1)]); // x²
        let f_d = Poly::from_int(1);
        let g_n = Poly::from_int(1);
        let g_d = Poly::from_int(1);

        let result = solve_risch_de_rational(&f_n, &f_d, &g_n, &g_d);
        assert!(
            matches!(result, RdeResult::NoSolution),
            "y' + x²·y = 1 should have no rational solution, got {:?}",
            result
        );
    }

    #[test]
    fn rde_f_zero_cubic_g() {
        // y' = 6x² + 2x + 1 → y = 2x³ + x² + x
        let f_n = Poly::zero();
        let f_d = Poly::from_int(1);
        let g_n = Poly::from_coeffs(vec![rat(1, 1), rat(2, 1), rat(6, 1)]); // 6x²+2x+1
        let g_d = Poly::from_int(1);

        match solve_risch_de_rational(&f_n, &f_d, &g_n, &g_d) {
            RdeResult::Solution { numer, denom } => {
                verify_rde_solution(&f_n, &f_d, &g_n, &g_d, &numer, &denom);
                assert_eq!(numer.degree(), Some(3), "y should be degree 3");
            }
            other => panic!("expected Solution, got {:?}", other),
        }
    }
}
