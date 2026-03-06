//! Control systems analysis: state-space models, transfer functions,
//! stability analysis, and controller design utilities.
//!
//! # State-Space Models
//!
//! A linear time-invariant (LTI) system in state-space form:
//!     ẋ = Ax + Bu
//!     y = Cx + Du
//!
//! # Transfer Functions
//!
//! G(s) = num(s) / den(s) — a ratio of polynomials in the Laplace variable s.

use crate::matrix::Matrix;
use crate::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// State-Space Model
// ═══════════════════════════════════════════════════════════════════════════

/// A linear time-invariant (LTI) state-space model.
///
/// ẋ = Ax + Bu
/// y = Cx + Du
///
/// where:
/// - `x` is the state vector (n×1)
/// - `u` is the input vector (m×1)
/// - `y` is the output vector (p×1)
/// - `A` is the state matrix (n×n)
/// - `B` is the input matrix (n×m)
/// - `C` is the output matrix (p×n)
/// - `D` is the feedthrough matrix (p×m)
#[derive(Clone, Debug)]
pub struct StateSpace {
    /// State matrix (n×n)
    pub a: Matrix,
    /// Input matrix (n×m)
    pub b: Matrix,
    /// Output matrix (p×n)
    pub c: Matrix,
    /// Feedthrough matrix (p×m)
    pub d: Matrix,
}

impl StateSpace {
    /// Create a new state-space model from the four system matrices.
    ///
    /// # Panics
    ///
    /// Panics if dimensions are inconsistent:
    /// - A must be n×n (square)
    /// - B must be n×m
    /// - C must be p×n
    /// - D must be p×m
    pub fn new(a: Matrix, b: Matrix, c: Matrix, d: Matrix) -> Self {
        let n = a.nrows();
        assert_eq!(a.ncols(), n, "A must be square: got {}×{}", a.nrows(), a.ncols());
        assert_eq!(b.nrows(), n, "B must have {} rows (same as A), got {}", n, b.nrows());
        assert_eq!(c.ncols(), n, "C must have {} cols (same as A), got {}", n, c.ncols());
        let m = b.ncols();
        let p = c.nrows();
        assert_eq!(d.nrows(), p, "D must have {} rows (same as C), got {}", p, d.nrows());
        assert_eq!(d.ncols(), m, "D must have {} cols (same as B), got {}", m, d.ncols());
        StateSpace { a, b, c, d }
    }

    /// Number of states (dimension of the state vector).
    pub fn num_states(&self) -> usize {
        self.a.nrows()
    }

    /// Number of inputs (dimension of the input vector).
    pub fn num_inputs(&self) -> usize {
        self.b.ncols()
    }

    /// Number of outputs (dimension of the output vector).
    pub fn num_outputs(&self) -> usize {
        self.c.nrows()
    }

    /// Poles of the system (eigenvalues of A).
    ///
    /// The `var` parameter is the symbolic variable used to compute
    /// the characteristic polynomial. The returned expressions are
    /// the roots of `det(var·I - A) = 0`.
    pub fn poles(&self, var: &Ex) -> Vec<Ex> {
        self.a.eigenvals(var)
    }

    /// Characteristic polynomial: det(sI - A).
    ///
    /// Returns a polynomial expression in the given variable `s`.
    pub fn char_poly(&self, s: &Ex) -> Ex {
        let n = self.num_states();
        let si = Matrix::identity(n).scale(s);
        let si_minus_a = si.sub(&self.a);
        si_minus_a.det()
    }

    /// Controllability matrix: \[B, AB, A²B, ..., Aⁿ⁻¹B\].
    ///
    /// For an n-state, m-input system, this is an n×(n·m) matrix.
    /// The system is controllable if and only if this matrix has rank n.
    pub fn controllability_matrix(&self) -> Matrix {
        let n = self.num_states();
        let mut cols: Vec<Matrix> = vec![self.b.clone()];
        let mut ab = self.a.matmul(&self.b);
        for _ in 1..n {
            cols.push(ab.clone());
            ab = self.a.matmul(&ab);
        }
        let refs: Vec<&Matrix> = cols.iter().collect();
        Matrix::hstack(&refs)
    }

    /// Observability matrix: \[C; CA; CA²; ...; CAⁿ⁻¹\].
    ///
    /// For an n-state, p-output system, this is an (n·p)×n matrix.
    /// The system is observable if and only if this matrix has rank n.
    pub fn observability_matrix(&self) -> Matrix {
        let n = self.num_states();
        let mut rows: Vec<Matrix> = vec![self.c.clone()];
        let mut ca = self.c.matmul(&self.a);
        for _ in 1..n {
            rows.push(ca.clone());
            ca = ca.matmul(&self.a);
        }
        let refs: Vec<&Matrix> = rows.iter().collect();
        Matrix::vstack(&refs)
    }

    /// Check controllability: rank(controllability_matrix) == n.
    ///
    /// Returns `true` if the system is fully state controllable.
    pub fn is_controllable(&self) -> bool {
        self.controllability_matrix().rank() == self.num_states()
    }

    /// Check observability: rank(observability_matrix) == n.
    ///
    /// Returns `true` if the system is fully observable.
    pub fn is_observable(&self) -> bool {
        self.observability_matrix().rank() == self.num_states()
    }

    /// Check stability: all eigenvalues have negative real part.
    ///
    /// Returns `Some(true)` if all poles are in the open left half-plane.
    /// Returns `Some(false)` if any pole can be shown to have non-negative real part.
    /// Returns `None` if stability cannot be determined symbolically.
    pub fn is_stable(&self) -> Option<bool> {
        let s = crate::var("__s_stability");
        let poles = self.poles(&s);
        for pole in &poles {
            // Try real evaluation first
            if let Ok(val) = pole.eval_f64() {
                if val >= 0.0 {
                    return Some(false);
                }
            } else if let Ok((re, _im)) = pole.eval_complex64() {
                if re >= 0.0 {
                    return Some(false);
                }
            } else {
                return None; // Can't determine
            }
        }
        Some(true)
    }

    /// Discretize using zero-order hold (ZOH).
    ///
    /// Aᵈ = eᴬᵈᵗ (matrix exponential)
    /// Bᵈ = A⁻¹(eᴬᵈᵗ - I)B — computed via series to avoid inverting A.
    ///
    /// Uses series approximation for the matrix exponential.
    /// `dt` is the sampling period (symbolic or numeric).
    /// `order` is the Taylor series truncation order.
    ///
    /// The discrete-time B matrix is computed as:
    ///   Bᵈ = (I·dt + A·dt²/2! + A²·dt³/3! + ...)B
    /// which avoids requiring A to be invertible.
    pub fn discretize_zoh(&self, dt: &Ex, order: usize) -> StateSpace {
        let n = self.num_states();
        let a_dt = self.a.scale(dt);
        let exp_a_dt = a_dt.exp_series(order);

        // Bᵈ = (I·dt + A·dt²/2! + A²·dt³/3! + ...)B
        let ident = Matrix::identity(n);
        let mut b_sum = ident.scale(dt);
        let mut a_power = Matrix::identity(n);
        for k in 2..=order {
            a_power = a_power.matmul(&self.a);
            let factorial: i64 = (1..=k as i64).product();
            let coeff = crate::rational(1, factorial);
            let dt_power = dt.powi(k as i64);
            let term = a_power.scale(&(&coeff * &dt_power));
            b_sum = b_sum.add(&term);
        }
        let b_d = b_sum.matmul(&self.b);

        StateSpace::new(exp_a_dt, b_d, self.c.clone(), self.d.clone())
    }

    // ── Advanced control: Riccati & pole placement ─────────────────────

    /// Set up the continuous-time algebraic Riccati equation (CARE).
    ///
    /// Returns the residual matrix: AᵀP + PA − PBR⁻¹BᵀP + Q
    /// which should equal zero when P is the solution.
    ///
    /// Returns `None` if R is singular.
    pub fn riccati_residual(&self, p: &Matrix, q: &Matrix, r: &Matrix) -> Option<Matrix> {
        let at = self.a.transpose();
        let r_inv = r.inv()?;
        let bt = self.b.transpose();

        let term1 = at.matmul(p);           // AᵀP
        let term2 = p.matmul(&self.a);       // PA
        let term3 = p.matmul(&self.b)        // PBR⁻¹BᵀP
            .matmul(&r_inv)
            .matmul(&bt)
            .matmul(p);

        let residual = term1.add(&term2).sub(&term3).add(q);
        Some(residual)
    }

    /// Pole placement via Ackermann's formula (single-input systems only).
    ///
    /// Given desired pole locations, computes feedback gain K such that
    /// the eigenvalues of (A − BK) equal the desired poles.
    ///
    /// Only works for single-input (m=1) controllable systems.
    /// The number of desired poles must equal the number of states.
    ///
    /// Returns `None` if the system is not single-input, not controllable,
    /// or the controllability matrix is singular.
    pub fn ackermann(&self, desired_poles: &[Ex]) -> Option<Matrix> {
        if self.num_inputs() != 1 {
            return None;
        }
        if !self.is_controllable() {
            return None;
        }

        let n = self.num_states();
        assert_eq!(
            desired_poles.len(),
            n,
            "ackermann: need {} desired poles, got {}",
            n,
            desired_poles.len()
        );

        let ctrb = self.controllability_matrix();
        let ctrb_inv = ctrb.inv()?;

        // Build the desired characteristic polynomial:
        // p(s) = (s − p₁)(s − p₂)···(s − pₙ)
        // poly_coeffs[0] is the leading coefficient (1),
        // poly_coeffs[k] is the coefficient of s^(n-k).
        let mut poly_coeffs: Vec<Ex> = vec![crate::int(1)];
        for pole in desired_poles {
            let neg_pole = -(pole.clone());
            let prev = poly_coeffs;
            poly_coeffs = vec![crate::int(0); prev.len() + 1];
            for (i, c) in prev.into_iter().enumerate() {
                let c_neg_pole = &c * &neg_pole;
                poly_coeffs[i] = poly_coeffs[i].clone() + c;
                poly_coeffs[i + 1] = poly_coeffs[i + 1].clone() + c_neg_pole;
            }
        }

        // Evaluate p(A) = poly_coeffs[0]·Aⁿ + poly_coeffs[1]·Aⁿ⁻¹ + ··· + poly_coeffs[n]·I
        let mut p_a = Matrix::zeros(n, n);
        for (i, coeff) in poly_coeffs.iter().enumerate() {
            let power = (poly_coeffs.len() - 1 - i) as u32;
            let a_power = self.a.powi(power);
            p_a = p_a.add(&a_power.scale(coeff));
        }

        // K = eₙᵀ · C⁻¹ · p(A)
        // where eₙᵀ is the last standard basis row vector,
        // so eₙᵀ · C⁻¹ is the last row of C⁻¹.
        let last_row: Vec<Ex> = (0..n)
            .map(|j| ctrb_inv.get(n - 1, j).clone())
            .collect();
        let last_row_mat = Matrix::new(vec![last_row]); // 1×n

        let k = last_row_mat.matmul(&p_a); // 1×n
        Some(k)
    }
}

impl std::fmt::Display for StateSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "StateSpace(n={}, m={}, p={})\nA = {}\nB = {}\nC = {}\nD = {}",
            self.num_states(),
            self.num_inputs(),
            self.num_outputs(),
            self.a,
            self.b,
            self.c,
            self.d
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Transfer Function
// ═══════════════════════════════════════════════════════════════════════════

/// A transfer function G(s) = num(s) / den(s).
///
/// Numerator and denominator are represented as symbolic expressions
/// in the Laplace variable s. This representation supports symbolic
/// manipulation: series/parallel connections, feedback loops, pole/zero
/// analysis, and evaluation.
#[derive(Clone, Debug)]
pub struct TransferFunction {
    /// Numerator polynomial in s.
    pub num: Ex,
    /// Denominator polynomial in s.
    pub den: Ex,
    /// The Laplace variable (typically "s").
    pub var: Ex,
}

impl TransferFunction {
    /// Create a new transfer function G(s) = num / den.
    ///
    /// # Arguments
    ///
    /// * `num` - Numerator polynomial expression
    /// * `den` - Denominator polynomial expression
    /// * `var` - The Laplace variable (e.g., `symplex::var("s")`)
    pub fn new(num: Ex, den: Ex, var: Ex) -> Self {
        TransferFunction { num, den, var }
    }

    /// Create a transfer function from coefficient slices.
    ///
    /// Coefficients are in ascending power order:
    /// `coeffs[0] + coeffs[1]*s + coeffs[2]*s² + ...`
    ///
    /// # Example
    ///
    /// ```
    /// use symplex::control::TransferFunction;
    /// let s = symplex::var("s");
    /// // G(s) = 1 / (s² + 3s + 2)
    /// let g = TransferFunction::from_coeffs(&[1], &[2, 3, 1], &s);
    /// ```
    pub fn from_coeffs(num_coeffs: &[i64], den_coeffs: &[i64], var: &Ex) -> Self {
        let build_poly = |coeffs: &[i64]| -> Ex {
            let mut result = crate::int(0);
            for (i, &c) in coeffs.iter().enumerate() {
                if c != 0 {
                    let term = if i == 0 {
                        crate::int(c)
                    } else {
                        &crate::int(c) * &var.powi(i as i64)
                    };
                    result = &result + &term;
                }
            }
            result
        };
        Self::new(build_poly(num_coeffs), build_poly(den_coeffs), var.clone())
    }

    /// Poles: roots of the denominator polynomial.
    ///
    /// Returns an empty vector if the solver cannot find the roots.
    pub fn poles(&self) -> Vec<Ex> {
        self.den.solve_or_empty(&self.var)
    }

    /// Zeros: roots of the numerator polynomial.
    ///
    /// Returns an empty vector if the solver cannot find the roots.
    pub fn zeros(&self) -> Vec<Ex> {
        self.num.solve_or_empty(&self.var)
    }

    /// DC gain: G(0) = num(0) / den(0).
    ///
    /// Evaluates the transfer function at s = 0, giving the
    /// steady-state gain for a step input.
    pub fn dc_gain(&self) -> Ex {
        let zero = crate::int(0);
        let num_0 = self.num.subs(&self.var, &zero).eval();
        let den_0 = self.den.subs(&self.var, &zero).eval();
        &num_0 / &den_0
    }

    /// Series connection: G₁(s) · G₂(s).
    ///
    /// The series (cascade) connection multiplies the transfer functions:
    /// G_series = G₁ · G₂ = (num₁·num₂) / (den₁·den₂)
    pub fn series(&self, other: &TransferFunction) -> TransferFunction {
        TransferFunction {
            num: &self.num * &other.num,
            den: &self.den * &other.den,
            var: self.var.clone(),
        }
    }

    /// Parallel connection: G₁(s) + G₂(s).
    ///
    /// The parallel connection adds the transfer functions:
    /// G_parallel = (num₁·den₂ + num₂·den₁) / (den₁·den₂)
    pub fn parallel(&self, other: &TransferFunction) -> TransferFunction {
        TransferFunction {
            num: &(&self.num * &other.den) + &(&other.num * &self.den),
            den: &self.den * &other.den,
            var: self.var.clone(),
        }
    }

    /// Negative unity feedback: G(s) / (1 + G(s)).
    ///
    /// The closed-loop transfer function with unity negative feedback:
    /// G_cl = num / (den + num)
    pub fn feedback(&self) -> TransferFunction {
        TransferFunction {
            num: self.num.clone(),
            den: &self.den + &self.num,
            var: self.var.clone(),
        }
    }

    /// Feedback with controller H(s): G(s) / (1 + G(s)·H(s)).
    ///
    /// The closed-loop transfer function with feedback element H(s):
    /// G_cl = (num_G · den_H) / (den_G · den_H + num_G · num_H)
    pub fn feedback_with(&self, h: &TransferFunction) -> TransferFunction {
        TransferFunction {
            num: &self.num * &h.den,
            den: &(&self.den * &h.den) + &(&self.num * &h.num),
            var: self.var.clone(),
        }
    }

    /// Evaluate the transfer function at a specific value of s.
    ///
    /// Substitutes `s_val` for the Laplace variable and evaluates:
    /// G(s_val) = num(s_val) / den(s_val)
    pub fn eval_at(&self, s_val: &Ex) -> Ex {
        let num = self.num.subs(&self.var, s_val).eval();
        let den = self.den.subs(&self.var, s_val).eval();
        &num / &den
    }
}

impl std::fmt::Display for TransferFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}) / ({})", self.num, self.den)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Routh-Hurwitz Stability Criterion
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the Routh array for stability analysis.
///
/// Given coefficients of a characteristic polynomial ordered from
/// highest degree to lowest:
///
///   aₙsⁿ + aₙ₋₁sⁿ⁻¹ + ... + a₁s + a₀
///
/// passed as `coeffs = [aₙ, aₙ₋₁, ..., a₁, a₀]`,
/// this function builds the Routh array and returns it as a vector of rows.
///
/// The first two rows are formed from the even-indexed and odd-indexed
/// coefficients respectively. Subsequent rows are computed as:
///
/// ```text
/// routh[i][j] = (routh[i-1][0] * routh[i-2][j+1] - routh[i-2][0] * routh[i-1][j+1]) / routh[i-1][0]
/// ```
///
/// # Panics
///
/// Panics if `coeffs` is empty.
pub fn routh_array(coeffs: &[Ex]) -> Vec<Vec<Ex>> {
    assert!(!coeffs.is_empty(), "routh_array: coefficients must not be empty");

    let n = coeffs.len();
    if n == 1 {
        return vec![vec![coeffs[0].clone()]];
    }

    // Number of columns in the Routh table
    let num_cols = (n + 1) / 2;

    // Build first row: even-indexed coefficients (a_n, a_{n-2}, a_{n-4}, ...)
    let mut row0: Vec<Ex> = Vec::with_capacity(num_cols);
    for i in (0..n).step_by(2) {
        row0.push(coeffs[i].clone());
    }
    // Pad with zeros if needed
    while row0.len() < num_cols {
        row0.push(crate::int(0));
    }

    // Build second row: odd-indexed coefficients (a_{n-1}, a_{n-3}, ...)
    let mut row1: Vec<Ex> = Vec::with_capacity(num_cols);
    for i in (1..n).step_by(2) {
        row1.push(coeffs[i].clone());
    }
    // Pad with zeros if needed
    while row1.len() < num_cols {
        row1.push(crate::int(0));
    }

    let mut table: Vec<Vec<Ex>> = vec![row0, row1];

    // Build subsequent rows
    let total_rows = n;
    for i in 2..total_rows {
        let prev = &table[i - 1];
        let prev2 = &table[i - 2];
        let pivot = &prev[0];

        let mut new_row: Vec<Ex> = Vec::with_capacity(num_cols);
        for j in 0..(num_cols - 1) {
            let prev2_j1 = if j + 1 < prev2.len() {
                prev2[j + 1].clone()
            } else {
                crate::int(0)
            };
            let prev_j1 = if j + 1 < prev.len() {
                prev[j + 1].clone()
            } else {
                crate::int(0)
            };
            // routh[i][j] = (pivot * prev2[j+1] - prev2[0] * prev[j+1]) / pivot
            let numerator = &(pivot * &prev2_j1) - &(&prev2[0] * &prev_j1);
            let entry = (&numerator / pivot).eval();
            new_row.push(entry);
        }
        // Last column is always zero (or not needed), pad if row is too short
        if new_row.is_empty() {
            new_row.push(crate::int(0));
        }
        table.push(new_row);
    }

    table
}

/// Check Routh-Hurwitz stability: all first-column entries must be positive
/// (or all negative) for stability.
///
/// Given coefficients of a characteristic polynomial from highest degree
/// to lowest (`coeffs = [aₙ, aₙ₋₁, ..., a₁, a₀]`), builds the Routh
/// array and checks for sign changes in the first column.
///
/// Returns `Some(true)` if stable (no sign changes), `Some(false)` if
/// unstable (sign changes detected), or `None` if stability cannot be
/// determined symbolically.
pub fn is_routh_stable(coeffs: &[Ex]) -> Option<bool> {
    if coeffs.is_empty() {
        return Some(false);
    }

    let table = routh_array(coeffs);

    // Collect first-column entries
    let first_col: Vec<&Ex> = table.iter().map(|row| &row[0]).collect();

    // Try to evaluate each entry numerically
    let mut values: Vec<f64> = Vec::with_capacity(first_col.len());
    for entry in &first_col {
        if let Ok(val) = entry.eval_f64() {
            values.push(val);
        } else {
            return None; // Can't evaluate symbolically
        }
    }

    // Check for sign changes
    if values.is_empty() {
        return None;
    }

    // All entries must be of the same sign (all positive or all negative)
    let first_sign = values[0] > 0.0;
    for &val in &values[1..] {
        if val == 0.0 {
            // A zero in the first column indicates marginal stability or
            // requires special handling — treat as unstable for simplicity
            return Some(false);
        }
        if (val > 0.0) != first_sign {
            return Some(false); // Sign change → unstable
        }
    }

    Some(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_space_basic_construction() {
        let a = Matrix::new(vec![
            vec![crate::int(0), crate::int(1)],
            vec![crate::int(-2), crate::int(-3)],
        ]);
        let b = Matrix::new(vec![
            vec![crate::int(0)],
            vec![crate::int(1)],
        ]);
        let c = Matrix::new(vec![
            vec![crate::int(1), crate::int(0)],
        ]);
        let d = Matrix::new(vec![
            vec![crate::int(0)],
        ]);
        let ss = StateSpace::new(a, b, c, d);
        assert_eq!(ss.num_states(), 2);
        assert_eq!(ss.num_inputs(), 1);
        assert_eq!(ss.num_outputs(), 1);
    }

    #[test]
    fn transfer_function_basic_display() {
        let s = crate::var("s");
        let tf = TransferFunction::new(
            crate::int(1),
            &s * &s + &s * 3 + 2,
            s,
        );
        let display = format!("{tf}");
        assert!(display.contains("/"), "Display should show fraction: {display}");
    }

    #[test]
    fn routh_array_row_count() {
        // s^2 + 3s + 2 → 3 coefficients → 3 rows
        let coeffs = vec![crate::int(1), crate::int(3), crate::int(2)];
        let table = routh_array(&coeffs);
        assert_eq!(table.len(), 3);
    }
}
