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

use crate::domains::matrix::Matrix;
use crate::prelude::*;

/// Ascending coefficients of `expr` viewed as a polynomial in `var`,
/// allowing symbolic (var-free) coefficients.
///
/// [`Ex::coeffs`] only handles rational coefficients; transfer functions
/// such as `K·ωₙ² / (s² + 2ζωₙ s + ωₙ²)` need the symbolic variant.
/// Returns `None` if `var` occurs in a non-polynomial position.
fn poly_coeffs_symbolic(expr: &Ex, var: &Ex) -> Option<Vec<Ex>> {
    let var_id = expr.checked_id(var);
    let mut inner = expr.inner.write();
    let ids =
        crate::transforms::solve::symbolic_poly_coeffs(&mut inner.arena, expr.raw_id(), var_id)?;
    drop(inner);
    Some(ids.into_iter().map(|id| expr.wrap(id)).collect())
}

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
    /// Extract a [`Context`] from this model's state matrix.
    fn ctx(&self) -> crate::api::context::Context {
        self.a.get(0, 0).context()
    }

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
        assert_eq!(
            a.ncols(),
            n,
            "A must be square: got {}×{}",
            a.nrows(),
            a.ncols()
        );
        assert_eq!(
            b.nrows(),
            n,
            "B must have {} rows (same as A), got {}",
            n,
            b.nrows()
        );
        assert_eq!(
            c.ncols(),
            n,
            "C must have {} cols (same as A), got {}",
            n,
            c.ncols()
        );
        let m = b.ncols();
        let p = c.nrows();
        assert_eq!(
            d.nrows(),
            p,
            "D must have {} rows (same as C), got {}",
            p,
            d.nrows()
        );
        assert_eq!(
            d.ncols(),
            m,
            "D must have {} cols (same as B), got {}",
            m,
            d.ncols()
        );
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

    /// Poles of the system (eigenvalues of A), repeated with multiplicity.
    ///
    /// Returns an empty vector if the eigenvalue solver cannot find any
    /// root of the characteristic polynomial.
    pub fn poles(&self) -> Vec<Ex> {
        self.a.eigenvals().unwrap_or_default()
    }

    /// Characteristic polynomial: det(sI - A).
    ///
    /// Returns a polynomial expression in the given variable `s`.  The
    /// determinant is undefined when `A` is not square (a model built as a
    /// struct literal, bypassing [`StateSpace::new`]); such a model yields
    /// NaN — use [`try_char_poly`](Self::try_char_poly) for the error.
    pub fn char_poly(&self, s: &Ex) -> Ex {
        self.try_char_poly(s).unwrap_or_else(|_| self.ctx().nan())
    }

    /// Characteristic polynomial `det(sI − A)` in the variable `s`.
    ///
    /// # Errors
    ///
    /// Returns an error if `A` is not square.
    pub fn try_char_poly(&self, s: &Ex) -> Result<Ex, SymplexError> {
        let n = self.num_states();
        let si = Matrix::identity(&self.ctx(), n).scale(s);
        si.sub(&self.a)?.det()
    }

    /// Transfer function `G(s) = C (sI − A)⁻¹ B + D` of a single-input,
    /// single-output system.
    ///
    /// The result has denominator `det(sI − A)` and numerator
    /// `C·adj(sI − A)·B + D·det(sI − A)`, both expanded polynomials in
    /// `s`; common factors are **not** cancelled.
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::InvalidArgument`] if the system is not SISO.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let s = ctx.symbol("s");
    /// // ẋ = [[0, 1], [-2, -3]] x + [0, 1] u,  y = [1, 0] x
    /// let ss = StateSpace::new(
    ///     matrix![ctx, [0, 1], [-2, -3]],
    ///     matrix![ctx, [0], [1]],
    ///     matrix![ctx, [1, 0]],
    ///     matrix![ctx, [0]],
    /// );
    /// let g = ss.to_transfer_function(&s).unwrap();
    /// // G(s) = 1 / (s² + 3s + 2)
    /// assert_eq!(g.num, ctx.int(1));
    /// assert_eq!(g.den, &s.powi(2) + &s * 3 + 2);
    /// ```
    pub fn to_transfer_function(&self, s: &Ex) -> Result<TransferFunction, SymplexError> {
        if self.num_inputs() != 1 || self.num_outputs() != 1 {
            return Err(SymplexError::InvalidArgument {
                operation: "StateSpace::to_transfer_function",
                reason: format!(
                    "requires a SISO system, got {} input(s) and {} output(s)",
                    self.num_inputs(),
                    self.num_outputs()
                ),
            });
        }
        let n = self.num_states();
        let si_minus_a = Matrix::identity(&self.ctx(), n).scale(s).sub(&self.a)?;
        let den = si_minus_a.det()?.expand();
        let adj = si_minus_a.adjugate()?;
        let c_adj_b = self.c.matmul(&adj)?.matmul(&self.b)?;
        let num = (c_adj_b.get(0, 0) + &(self.d.get(0, 0) * &den)).expand();
        Ok(TransferFunction::new(num, den, s.clone()))
    }

    /// Controllability matrix: \[B, AB, A²B, ..., Aⁿ⁻¹B\].
    ///
    /// For an n-state, m-input system, this is an n×(n·m) matrix.
    /// The system is controllable if and only if this matrix has rank n.
    ///
    /// # Errors
    ///
    /// Returns an error if the shapes of `A` and `B` are inconsistent (a
    /// model built as a struct literal, bypassing [`StateSpace::new`]).
    pub fn controllability_matrix(&self) -> Result<Matrix, SymplexError> {
        let n = self.num_states();
        let mut cols: Vec<Matrix> = vec![self.b.clone()];
        let mut ab = self.a.matmul(&self.b)?;
        for _ in 1..n {
            cols.push(ab.clone());
            ab = self.a.matmul(&ab)?;
        }
        let refs: Vec<&Matrix> = cols.iter().collect();
        Matrix::hstack(&refs)
    }

    /// Observability matrix: \[C; CA; CA²; ...; CAⁿ⁻¹\].
    ///
    /// For an n-state, p-output system, this is an (n·p)×n matrix.
    /// The system is observable if and only if this matrix has rank n.
    ///
    /// # Errors
    ///
    /// Returns an error if the shapes of `A` and `C` are inconsistent (a
    /// model built as a struct literal, bypassing [`StateSpace::new`]).
    pub fn observability_matrix(&self) -> Result<Matrix, SymplexError> {
        let n = self.num_states();
        let mut rows: Vec<Matrix> = vec![self.c.clone()];
        let mut ca = self.c.matmul(&self.a)?;
        for _ in 1..n {
            rows.push(ca.clone());
            ca = ca.matmul(&self.a)?;
        }
        let refs: Vec<&Matrix> = rows.iter().collect();
        Matrix::vstack(&refs)
    }

    /// Check controllability: rank(controllability_matrix) == n.
    ///
    /// Returns `true` if the system is fully state controllable.  A model
    /// with inconsistent shapes is not a valid system and is reported as
    /// not controllable.
    pub fn is_controllable(&self) -> bool {
        self.controllability_matrix()
            .is_ok_and(|m| m.rank() == self.num_states())
    }

    /// Check observability: rank(observability_matrix) == n.
    ///
    /// Returns `true` if the system is fully observable.  A model with
    /// inconsistent shapes is not a valid system and is reported as not
    /// observable.
    pub fn is_observable(&self) -> bool {
        self.observability_matrix()
            .is_ok_and(|m| m.rank() == self.num_states())
    }

    /// Check stability: all eigenvalues have negative real part.
    ///
    /// Returns `Some(true)` if all poles are in the open left half-plane.
    /// Returns `Some(false)` if any pole can be shown to have non-negative real part.
    /// Returns `None` if stability cannot be determined symbolically.
    pub fn is_stable(&self) -> Option<bool> {
        let poles = self.poles();
        if poles.len() < self.num_states() {
            return None; // not all eigenvalues found
        }
        for pole in &poles {
            // Try real evaluation first
            if let Ok(val) = pole.eval_f64() {
                if val >= 0.0 {
                    return Some(false);
                }
            } else if let Ok(z) = pole.eval_complex64() {
                if z.re >= 0.0 {
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
    ///
    /// # Errors
    ///
    /// Returns an error if the shapes of `A` and `B` are inconsistent (a
    /// model built as a struct literal, bypassing [`StateSpace::new`]).
    pub fn discretize_zoh(&self, dt: &Ex, order: usize) -> Result<StateSpace, SymplexError> {
        let n = self.num_states();
        let a_dt = self.a.scale(dt);
        let exp_a_dt = a_dt.exp_series(order)?;

        // Bᵈ = (I·dt + A·dt²/2! + A²·dt³/3! + ...)B
        let ctx = self.ctx();
        let ident = Matrix::identity(&ctx, n);
        let mut b_sum = ident.scale(dt);
        let mut a_power = Matrix::identity(&ctx, n);
        for k in 2..=order {
            a_power = a_power.matmul(&self.a)?;
            let factorial: i64 = (1..=k as i64).product();
            let coeff = self.ctx().rational(1, factorial);
            let dt_power = dt.powi(k as i64);
            let term = a_power.scale(&(&coeff * &dt_power));
            b_sum = b_sum.add(&term)?;
        }
        let b_d = b_sum.matmul(&self.b)?;

        // Discretisation preserves every shape, so the result is as
        // well-formed as `self`.
        Ok(StateSpace {
            a: exp_a_dt,
            b: b_d,
            c: self.c.clone(),
            d: self.d.clone(),
        })
    }

    // ── Advanced control: Riccati & pole placement ─────────────────────

    /// Set up the continuous-time algebraic Riccati equation (CARE).
    ///
    /// Returns the residual matrix: AᵀP + PA − PBR⁻¹BᵀP + Q
    /// which should equal zero when P is the solution.
    ///
    /// # Errors
    ///
    /// Returns an error if `R` is singular, or if `P` is not n×n, `Q` is not
    /// n×n or `R` is not m×m.
    pub fn riccati_residual(
        &self,
        p: &Matrix,
        q: &Matrix,
        r: &Matrix,
    ) -> Result<Matrix, SymplexError> {
        let at = self.a.transpose();
        let r_inv = r.inv()?;
        let bt = self.b.transpose();

        let term1 = at.matmul(p)?; // AᵀP
        let term2 = p.matmul(&self.a)?; // PA
        let term3 = p.matmul(&self.b)?.matmul(&r_inv)?.matmul(&bt)?.matmul(p)?; // PBR⁻¹BᵀP

        term1.add(&term2)?.sub(&term3)?.add(q)
    }

    /// Pole placement via Ackermann's formula (single-input systems only).
    ///
    /// Given desired pole locations, computes feedback gain K such that
    /// the eigenvalues of (A − BK) equal the desired poles.
    ///
    /// Only works for single-input (m=1) controllable systems.
    /// The number of desired poles must equal the number of states.
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::InvalidArgument`] if the system is not
    /// single-input or the number of desired poles is not `n`, and
    /// [`SymplexError::ComputationFailed`] if the system is not controllable.
    pub fn ackermann(&self, desired_poles: &[Ex]) -> Result<Matrix, SymplexError> {
        if self.num_inputs() != 1 {
            return Err(SymplexError::InvalidArgument {
                operation: "StateSpace::ackermann",
                reason: format!(
                    "requires a single-input system, got {} inputs",
                    self.num_inputs()
                ),
            });
        }
        let n = self.num_states();
        if desired_poles.len() != n {
            return Err(SymplexError::InvalidArgument {
                operation: "StateSpace::ackermann",
                reason: format!("need {n} desired poles, got {}", desired_poles.len()),
            });
        }

        let ctrb = self.controllability_matrix()?;
        if ctrb.rank() != n {
            return Err(SymplexError::ComputationFailed {
                operation: "StateSpace::ackermann",
                reason: "system is not controllable".into(),
            });
        }
        let ctrb_inv = ctrb.inv()?;

        // Build the desired characteristic polynomial:
        // p(s) = (s − p₁)(s − p₂)···(s − pₙ)
        // poly_coeffs[0] is the leading coefficient (1),
        // poly_coeffs[k] is the coefficient of s^(n-k).
        let mut poly_coeffs: Vec<Ex> = vec![self.ctx().int(1)];
        for pole in desired_poles {
            let neg_pole = -(pole.clone());
            let prev = poly_coeffs;
            poly_coeffs = vec![self.ctx().int(0); prev.len() + 1];
            for (i, c) in prev.into_iter().enumerate() {
                let c_neg_pole = &c * &neg_pole;
                poly_coeffs[i] = poly_coeffs[i].clone() + c;
                poly_coeffs[i + 1] = poly_coeffs[i + 1].clone() + c_neg_pole;
            }
        }

        // Evaluate p(A) = poly_coeffs[0]·Aⁿ + poly_coeffs[1]·Aⁿ⁻¹ + ··· + poly_coeffs[n]·I
        let mut p_a = Matrix::zeros(&self.ctx(), n, n);
        for (i, coeff) in poly_coeffs.iter().enumerate() {
            let power = (poly_coeffs.len() - 1 - i) as u32;
            let a_power = self.a.powi(power)?;
            p_a = p_a.add(&a_power.scale(coeff))?;
        }

        // K = eₙᵀ · C⁻¹ · p(A)
        // where eₙᵀ is the last standard basis row vector,
        // so eₙᵀ · C⁻¹ is the last row of C⁻¹.
        let last_row: Vec<Ex> = (0..n).map(|j| ctrb_inv.get(n - 1, j).clone()).collect();
        let last_row_mat = Matrix::new(vec![last_row])?; // 1×n

        last_row_mat.matmul(&p_a) // 1×n
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
    /// Extract a [`Context`] from this transfer function's numerator.
    fn ctx(&self) -> crate::api::context::Context {
        self.num.context()
    }

    /// Create a new transfer function G(s) = num / den.
    ///
    /// # Arguments
    ///
    /// * `num` - Numerator polynomial expression
    /// * `den` - Denominator polynomial expression
    /// * `var` - The Laplace variable (e.g., `Context::new().symbol("s")`)
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
    /// use symplex::prelude::*;
    /// use symplex::control::TransferFunction;
    /// let ctx = Context::new();
    /// let s = ctx.symbol("s");
    /// // G(s) = 1 / (s² + 3s + 2)
    /// let g = TransferFunction::from_coeffs(&[1], &[2, 3, 1], &s);
    /// ```
    pub fn from_coeffs(num_coeffs: &[i64], den_coeffs: &[i64], var: &Ex) -> Self {
        let ctx = var.context();
        let build_poly = |coeffs: &[i64]| -> Ex {
            let mut result = ctx.int(0);
            for (i, &c) in coeffs.iter().enumerate() {
                if c != 0 {
                    let term = if i == 0 {
                        ctx.int(c)
                    } else {
                        &ctx.int(c) * &var.powi(i as i64)
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
        let zero = self.ctx().int(0);
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

    /// State-space realisation in **controllable canonical form**.
    ///
    /// For a proper transfer function with monic-normalised denominator
    /// `sⁿ + aₙ₋₁sⁿ⁻¹ + … + a₀` and numerator `bₙsⁿ + … + b₀`:
    ///
    /// ```text
    /// A = [ 0    1    0  …  0   ]   B = [0]   C = [b₀−a₀bₙ, …, bₙ₋₁−aₙ₋₁bₙ]   D = [bₙ]
    ///     [ 0    0    1  …  0   ]       [0]
    ///     [ …                  ]       […]
    ///     [−a₀ −a₁ −a₂ … −aₙ₋₁]       [1]
    /// ```
    ///
    /// Coefficients may be symbolic.
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::InvalidArgument`] if numerator or denominator
    /// is not a polynomial in the Laplace variable, the denominator is
    /// constant, or the transfer function is improper (`deg num > deg den`).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let s = ctx.symbol("s");
    /// let g = TransferFunction::from_coeffs(&[1], &[2, 3, 1], &s); // 1/(s²+3s+2)
    /// let ss = g.to_state_space().unwrap();
    /// assert_eq!(ss.a, matrix![ctx, [0, 1], [-2, -3]]);
    /// assert_eq!(ss.b, matrix![ctx, [0], [1]]);
    /// assert_eq!(ss.c, matrix![ctx, [1, 0]]);
    /// // Round trip
    /// let back = ss.to_transfer_function(&s).unwrap();
    /// assert_eq!(back.den, g.den.expand());
    /// assert_eq!(back.num, g.num);
    /// ```
    pub fn to_state_space(&self) -> Result<StateSpace, SymplexError> {
        let ctx = self.ctx();
        let inv = |reason: String| SymplexError::InvalidArgument {
            operation: "TransferFunction::to_state_space",
            reason,
        };
        let den = poly_coeffs_symbolic(&self.den, &self.var)
            .ok_or_else(|| inv("denominator is not a polynomial in the Laplace variable".into()))?;
        let num = poly_coeffs_symbolic(&self.num, &self.var)
            .ok_or_else(|| inv("numerator is not a polynomial in the Laplace variable".into()))?;
        let n = den.len().saturating_sub(1);
        if n == 0 {
            return Err(inv("denominator must have degree ≥ 1".into()));
        }
        if num.len() > den.len() {
            return Err(inv(format!(
                "improper transfer function: numerator degree {} > denominator degree {}",
                num.len() - 1,
                n
            )));
        }
        // Normalise so the denominator is monic.
        let lead = &den[n];
        let a_coef: Vec<Ex> = den.iter().take(n).map(|c| (c / lead).eval()).collect();
        let mut b_coef: Vec<Ex> = num.iter().map(|c| (c / lead).eval()).collect();
        b_coef.resize(n + 1, ctx.zero());
        let bn = b_coef[n].clone();

        let a = Matrix::from_fn(n, n, |i, j| {
            if i + 1 < n {
                if j == i + 1 { ctx.one() } else { ctx.zero() }
            } else {
                -&a_coef[j]
            }
        });
        let b = Matrix::from_fn(n, 1, |i, _| if i + 1 == n { ctx.one() } else { ctx.zero() });
        let c = Matrix::from_fn(1, n, |_, j| (&b_coef[j] - &(&a_coef[j] * &bn)).eval());
        let d = Matrix::new(vec![vec![bn]])?;
        Ok(StateSpace::new(a, b, c, d))
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
    assert!(
        !coeffs.is_empty(),
        "routh_array: coefficients must not be empty"
    );

    let n = coeffs.len();
    if n == 1 {
        return vec![vec![coeffs[0].clone()]];
    }

    // Number of columns in the Routh table
    let num_cols = n.div_ceil(2);

    // Build first row: even-indexed coefficients (a_n, a_{n-2}, a_{n-4}, ...)
    let mut row0: Vec<Ex> = Vec::with_capacity(num_cols);
    for i in (0..n).step_by(2) {
        row0.push(coeffs[i].clone());
    }
    // Pad with zeros if needed
    while row0.len() < num_cols {
        row0.push(coeffs[0].context().int(0));
    }

    // Build second row: odd-indexed coefficients (a_{n-1}, a_{n-3}, ...)
    let mut row1: Vec<Ex> = Vec::with_capacity(num_cols);
    for i in (1..n).step_by(2) {
        row1.push(coeffs[i].clone());
    }
    // Pad with zeros if needed
    while row1.len() < num_cols {
        row1.push(coeffs[0].context().int(0));
    }

    let mut table: Vec<Vec<Ex>> = vec![row0, row1];

    // Build subsequent rows
    let total_rows = n;
    for i in 2..total_rows {
        let prev = &table[i - 1];
        let prev2 = &table[i - 2];
        let mut pivot = prev[0].clone();

        // Epsilon method: if pivot is zero, check if entire row is zero
        let pivot_is_zero = pivot.eval_f64().map(|v| v.abs() < 1e-30).unwrap_or(false);

        if pivot_is_zero {
            // Epsilon method: replace zero pivot with small ε to preserve
            // sign information. This handles both the "only pivot is zero"
            // case and the "entire row is zero" (auxiliary polynomial) case.
            pivot = coeffs[0].context().rational(1, 1_000_000_000);
        }

        let mut new_row: Vec<Ex> = Vec::with_capacity(num_cols);
        for j in 0..(num_cols - 1) {
            let prev2_j1 = if j + 1 < prev2.len() {
                prev2[j + 1].clone()
            } else {
                coeffs[0].context().int(0)
            };
            let prev_j1 = if j + 1 < prev.len() {
                prev[j + 1].clone()
            } else {
                coeffs[0].context().int(0)
            };
            // routh[i][j] = (pivot * prev2[j+1] - prev2[0] * prev[j+1]) / pivot
            let numerator = &(&pivot * &prev2_j1) - &(&prev2[0] * &prev_j1);
            let entry = (&numerator / &pivot).eval();
            new_row.push(entry);
        }
        // Last column is always zero (or not needed), pad if row is too short
        if new_row.is_empty() {
            new_row.push(coeffs[0].context().int(0));
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
        let ctx = crate::api::context::Context::new();
        let a = Matrix::new(vec![
            vec![ctx.int(0), ctx.int(1)],
            vec![ctx.int(-2), ctx.int(-3)],
        ])
        .unwrap();
        let b = Matrix::new(vec![vec![ctx.int(0)], vec![ctx.int(1)]]).unwrap();
        let c = Matrix::new(vec![vec![ctx.int(1), ctx.int(0)]]).unwrap();
        let d = Matrix::new(vec![vec![ctx.int(0)]]).unwrap();
        let ss = StateSpace::new(a, b, c, d);
        assert_eq!(ss.num_states(), 2);
        assert_eq!(ss.num_inputs(), 1);
        assert_eq!(ss.num_outputs(), 1);
    }

    #[test]
    fn transfer_function_basic_display() {
        let ctx = crate::api::context::Context::new();
        let s = ctx.symbol("s");
        let tf = TransferFunction::new(ctx.int(1), &s * &s + &s * 3 + 2, s);
        let display = format!("{tf}");
        assert!(
            display.contains("/"),
            "Display should show fraction: {display}"
        );
    }

    #[test]
    fn routh_array_row_count() {
        let ctx = crate::api::context::Context::new();
        // s^2 + 3s + 2 → 3 coefficients → 3 rows
        let coeffs = vec![ctx.int(1), ctx.int(3), ctx.int(2)];
        let table = routh_array(&coeffs);
        assert_eq!(table.len(), 3);
    }
}
