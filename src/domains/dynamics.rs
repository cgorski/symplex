//! Lagrangian dynamics for robotic systems.
//!
//! Derives the equations of motion from kinetic and potential energy:
//!
//! Given T(q, q̇) and V(q), the Euler-Lagrange equations are:
//!     d/dt(∂L/∂q̇ᵢ) - ∂L/∂qᵢ = τᵢ
//!
//! which yields the manipulator equation:
//!     M(q)q̈ + C(q, q̇)q̇ + G(q) = τ
//!
//! This module provides functions to extract M, C, and G symbolically.
//! Each coordinate travels with its velocity and acceleration as a
//! [`GeneralizedCoordinate`]; [`manipulator_equation`] returns the three
//! terms as a [`ManipulatorEquation`].

use crate::domains::matrix::Matrix;
use crate::prelude::*;

fn invalid(operation: &'static str, reason: String) -> SymplexError {
    SymplexError::invalid_argument(operation, reason)
}

/// One generalized coordinate `qᵢ` together with its velocity `q̇ᵢ` and
/// acceleration `q̈ᵢ`.
///
/// The Lagrangian formalism treats the three as *independent* symbols for
/// partial differentiation; [`total_time_derivative`] reassembles `d/dt`
/// from them by the chain rule.  Keeping them in one value rules out a
/// transposed `(q̇, q)` pair or an acceleration list of the wrong length.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::dynamics::GeneralizedCoordinate;
///
/// let ctx = Context::new();
/// let q = ctx.symbol("q");
/// let qd = ctx.symbol("qd");
/// let qdd = ctx.symbol("qdd");
///
/// let coord = GeneralizedCoordinate { q: &q, q_dot: &qd, q_ddot: &qdd };
/// assert_eq!(format!("{}", coord.q_ddot), "qdd");
/// ```
#[derive(Clone, Copy, Debug)]
pub struct GeneralizedCoordinate<'a> {
    /// `qᵢ` — the coordinate (a joint angle, a displacement, …).
    pub q: &'a Ex,
    /// `q̇ᵢ` — its generalized velocity.
    pub q_dot: &'a Ex,
    /// `q̈ᵢ` — its generalized acceleration.
    pub q_ddot: &'a Ex,
}

/// Compute the total time derivative of an expression.
///
/// Given an expression that depends on generalized coordinates q(t) and
/// their velocities q̇(t), computes d/dt using the chain rule:
///
/// d/dt f(q, q̇) = Σᵢ (∂f/∂qᵢ)·q̇ᵢ + Σᵢ (∂f/∂q̇ᵢ)·q̈ᵢ
///
/// Since q and q̇ are treated as independent symbols for the purpose of
/// partial differentiation, we apply the chain rule manually.
///
/// # Parameters
/// - `expr`: The expression to differentiate with respect to time
/// - `coords`: The generalized coordinates, each with its velocity q̇ᵢ and
///   acceleration q̈ᵢ
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::dynamics::{total_time_derivative, GeneralizedCoordinate};
///
/// let ctx = Context::new();
/// let q = ctx.symbol("q");
/// let qd = ctx.symbol("qd");
/// let qdd = ctx.symbol("qdd");
///
/// // d/dt(q) = qd
/// let coords = [GeneralizedCoordinate { q: &q, q_dot: &qd, q_ddot: &qdd }];
/// let result = total_time_derivative(&q, &coords);
/// let val = result.subs(&qd, &ctx.int(7)).eval().eval_f64().unwrap();
/// assert!((val - 7.0).abs() < 1e-12);
/// ```
pub fn total_time_derivative(expr: &Ex, coords: &[GeneralizedCoordinate<'_>]) -> Ex {
    // d/dt f = Σᵢ (∂f/∂qᵢ)·q̇ᵢ + Σᵢ (∂f/∂q̇ᵢ)·q̈ᵢ
    let mut result = expr.context().int(0);

    for coord in coords {
        // ∂f/∂qᵢ · q̇ᵢ
        let df_dqi = expr.diff(coord.q);
        result = &result + &(&df_dqi * coord.q_dot);

        // ∂f/∂q̇ᵢ · q̈ᵢ
        let df_dqi_dot = expr.diff(coord.q_dot);
        result = &result + &(&df_dqi_dot * coord.q_ddot);
    }

    result
}

/// Compute the Euler-Lagrange equations of motion.
///
/// Given kinetic energy T(q, q̇) and potential energy V(q), computes:
/// d/dt(∂L/∂q̇ᵢ) - ∂L/∂qᵢ = τᵢ
///
/// where L = T - V is the Lagrangian.
///
/// Returns a vector of expressions, one per generalized coordinate.
/// Each expression equals the generalized force τᵢ on the left-hand side
/// of the equation of motion.
///
/// # Parameters
/// - `kinetic_energy`: T(q, q̇), the kinetic energy
/// - `potential_energy`: V(q), the potential energy
/// - `coords`: The generalized coordinates, each with its velocity q̇ᵢ and
///   acceleration q̈ᵢ; one equation is returned per entry, in order
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::dynamics::{euler_lagrange, GeneralizedCoordinate};
///
/// let ctx = Context::new();
/// let m = ctx.symbol("m");
/// let q = ctx.symbol("q");
/// let qd = ctx.symbol("qd");
/// let qdd = ctx.symbol("qdd");
///
/// // Free particle: T = ½m·q̇², V = 0
/// let half = ctx.rational(1, 2);
/// let ke = &half * &m * &qd.powi(2);
/// let pe = ctx.int(0);
/// let coords = [GeneralizedCoordinate { q: &q, q_dot: &qd, q_ddot: &qdd }];
/// let eqs = euler_lagrange(&ke, &pe, &coords);
/// // Should give m·q̈
/// assert_eq!(eqs.len(), 1);
/// ```
pub fn euler_lagrange(
    kinetic_energy: &Ex,
    potential_energy: &Ex,
    coords: &[GeneralizedCoordinate<'_>],
) -> Vec<Ex> {
    let lagrangian = kinetic_energy - potential_energy;
    let mut equations = Vec::with_capacity(coords.len());

    for coord in coords {
        // ∂L/∂q̇ᵢ
        let dl_dqi_dot = lagrangian.diff(coord.q_dot);

        // d/dt(∂L/∂q̇ᵢ)
        let dt_dl_dqi_dot = total_time_derivative(&dl_dqi_dot, coords);

        // ∂L/∂qᵢ
        let dl_dqi = lagrangian.diff(coord.q);

        // Euler-Lagrange: d/dt(∂L/∂q̇ᵢ) - ∂L/∂qᵢ
        let eq_i = &dt_dl_dqi_dot - &dl_dqi;
        equations.push(eq_i.eval());
    }

    equations
}

/// Extract the mass (inertia) matrix M(q) from kinetic energy.
///
/// For a system where T = ½ q̇ᵀ M(q) q̇, the mass matrix entries are:
///
/// M_ij = ∂²T / (∂q̇ᵢ ∂q̇ⱼ)
///
/// The resulting matrix is symmetric for physical systems.
///
/// # Parameters
/// - `kinetic_energy`: T(q, q̇), the kinetic energy expression
/// - `qdot_vars`: The velocity variables [q̇₁, q̇₂, ...]
///
/// # Errors
///
/// Returns [`SymplexError::InvalidArgument`] if `qdot_vars` is empty.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::dynamics::mass_matrix;
///
/// let ctx = Context::new();
/// let m = ctx.symbol("m");
/// let qd = ctx.symbol("qd");
///
/// // T = ½m·q̇²  →  M = [[m]]
/// let half = ctx.rational(1, 2);
/// let ke = &half * &m * &qd.powi(2);
/// let mm = mass_matrix(&ke, &[&qd]).unwrap();
/// assert_eq!(mm.shape(), (1, 1));
/// ```
pub fn mass_matrix(kinetic_energy: &Ex, qdot_vars: &[&Ex]) -> Result<Matrix, SymplexError> {
    if qdot_vars.is_empty() {
        return Err(invalid(
            "dynamics::mass_matrix",
            "need at least one velocity variable".into(),
        ));
    }
    let n = qdot_vars.len();
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = Vec::with_capacity(n);
        for j in 0..n {
            // M_ij = ∂²T / ∂q̇ᵢ∂q̇ⱼ
            let m_ij = kinetic_energy.diff(qdot_vars[i]).diff(qdot_vars[j]);
            row.push(m_ij);
        }
        rows.push(row);
    }
    // n ≥ 1 rows of n entries each.
    Ok(Matrix::from_rows_unchecked(rows).eval())
}

/// Compute Christoffel symbols of the first kind from the mass matrix.
///
/// The Christoffel symbols are defined as:
///
/// Γᵢⱼₖ = ½(∂Mᵢⱼ/∂qₖ + ∂Mᵢₖ/∂qⱼ - ∂Mⱼₖ/∂qᵢ)
///
/// Returns a 3D structure indexed as `christoffel[i][j][k]`.
///
/// # Parameters
/// - `mass_mat`: The mass matrix M(q), an n×n [`Matrix`]
/// - `q_vars`: The generalized coordinate variables [q₁, q₂, ...]
///
/// # Errors
///
/// Returns [`SymplexError::InvalidArgument`] if `mass_mat` is not
/// `n×n` for `n = q_vars.len()` (in particular if `q_vars` is empty).
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::dynamics::{mass_matrix, christoffel_symbols};
///
/// let ctx = Context::new();
/// let m = ctx.symbol("m");
/// let qd = ctx.symbol("qd");
/// let q = ctx.symbol("q");
///
/// let half = ctx.rational(1, 2);
/// let ke = &half * &m * &qd.powi(2);
/// let mm = mass_matrix(&ke, &[&qd]).unwrap();
/// let cs = christoffel_symbols(&mm, &[&q]).unwrap();
/// assert_eq!(cs.len(), 1);
/// assert_eq!(cs[0].len(), 1);
/// assert_eq!(cs[0][0].len(), 1);
/// ```
pub fn christoffel_symbols(
    mass_mat: &Matrix,
    q_vars: &[&Ex],
) -> Result<Vec<Vec<Vec<Ex>>>, SymplexError> {
    let n = q_vars.len();
    if n == 0 || mass_mat.shape() != (n, n) {
        return Err(invalid(
            "dynamics::christoffel_symbols",
            format!(
                "mass matrix shape {:?} does not match {n} coordinates",
                mass_mat.shape()
            ),
        ));
    }

    let half = q_vars[0].context().rational(1, 2);

    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let mut plane = Vec::with_capacity(n);
        for j in 0..n {
            let mut row = Vec::with_capacity(n);
            for k in 0..n {
                // Γᵢⱼₖ = ½(∂Mᵢⱼ/∂qₖ + ∂Mᵢₖ/∂qⱼ - ∂Mⱼₖ/∂qᵢ)
                let dm_ij_dqk = mass_mat.get(i, j).diff(q_vars[k]);
                let dm_ik_dqj = mass_mat.get(i, k).diff(q_vars[j]);
                let dm_jk_dqi = mass_mat.get(j, k).diff(q_vars[i]);

                let gamma = &half * &(&(&dm_ij_dqk + &dm_ik_dqj) - &dm_jk_dqi);
                row.push(gamma);
            }
            plane.push(row);
        }
        result.push(plane);
    }

    Ok(result)
}

/// Compute the Coriolis matrix C(q, q̇).
///
/// The Coriolis matrix combines centrifugal and Coriolis effects:
///
/// Cᵢⱼ = Σₖ Γᵢⱼₖ · q̇ₖ
///
/// where Γᵢⱼₖ are the Christoffel symbols of the first kind.
///
/// # Parameters
/// - `mass_mat`: The mass matrix M(q)
/// - `q_vars`: Generalized coordinate variables [q₁, q₂, ...]
/// - `qdot_vars`: Generalized velocity variables [q̇₁, q̇₂, ...]
///
/// # Errors
///
/// Returns [`SymplexError::InvalidArgument`] if `q_vars` and `qdot_vars`
/// differ in length, or if `mass_mat` is not `n×n` for `n = q_vars.len()`
/// (in particular if `q_vars` is empty).
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::dynamics::{mass_matrix, coriolis_matrix};
///
/// let ctx = Context::new();
/// let m = ctx.symbol("m");
/// let q = ctx.symbol("q");
/// let qd = ctx.symbol("qd");
///
/// let half = ctx.rational(1, 2);
/// let ke = &half * &m * &qd.powi(2);
/// let mm = mass_matrix(&ke, &[&qd]).unwrap();
/// let c = coriolis_matrix(&mm, &[&q], &[&qd]).unwrap();
/// assert_eq!(c.shape(), (1, 1));
/// ```
pub fn coriolis_matrix(
    mass_mat: &Matrix,
    q_vars: &[&Ex],
    qdot_vars: &[&Ex],
) -> Result<Matrix, SymplexError> {
    let n = q_vars.len();
    if qdot_vars.len() != n {
        return Err(invalid(
            "dynamics::coriolis_matrix",
            format!(
                "q_vars and qdot_vars must have the same length, got {n} and {}",
                qdot_vars.len()
            ),
        ));
    }

    // Also rejects an empty `q_vars`, so `christoffel` is n×n×n with n ≥ 1.
    let christoffel = christoffel_symbols(mass_mat, q_vars)?;

    let mut rows = Vec::with_capacity(n);
    for christoffel_i in christoffel.iter().take(n) {
        let mut row = Vec::with_capacity(n);
        for christoffel_ij in christoffel_i.iter().take(n) {
            // C_ij = Σₖ Γᵢⱼₖ · q̇ₖ
            let mut c_ij = qdot_vars[0].context().int(0);
            for k in 0..n {
                c_ij = &c_ij + &(&christoffel_ij[k] * qdot_vars[k]);
            }
            row.push(c_ij);
        }
        rows.push(row);
    }

    Ok(Matrix::from_rows_unchecked(rows).eval())
}

/// Compute the gravity vector g(q) = ∂V/∂q.
///
/// Each element gᵢ = ∂V/∂qᵢ is the generalized gravitational force
/// acting on the i-th coordinate.
///
/// # Parameters
/// - `potential_energy`: V(q), the potential energy expression
/// - `q_vars`: Generalized coordinate variables [q₁, q₂, ...]
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::dynamics::gravity_vector;
///
/// let ctx = Context::new();
/// let m = ctx.symbol("m");
/// let g = ctx.symbol("g");
/// let l = ctx.symbol("L");
/// let q = ctx.symbol("q");
///
/// // V = m·g·L·cos(q)  →  g(q) = ∂V/∂q = -m·g·L·sin(q)
/// let pe = &m * &g * &l * &q.cos();
/// let gv = gravity_vector(&pe, &[&q]);
/// assert_eq!(gv.len(), 1);
/// ```
pub fn gravity_vector(potential_energy: &Ex, q_vars: &[&Ex]) -> Vec<Ex> {
    q_vars
        .iter()
        .map(|qi| potential_energy.diff(qi).eval())
        .collect()
}

/// The three terms of the manipulator equation
///
/// ```text
/// M(q)·q̈ + C(q, q̇)·q̇ + G(q) = τ
/// ```
///
/// as returned by [`manipulator_equation`], for `n` generalized coordinates.
/// `G` is the gradient of the potential, so for the same energies the
/// left-hand side evaluates to exactly what [`euler_lagrange`] returns.
#[derive(Clone, Debug)]
pub struct ManipulatorEquation {
    /// `M(q)` — the `n×n` mass (inertia) matrix, `Mᵢⱼ = ∂²T/∂q̇ᵢ∂q̇ⱼ`
    /// ([`mass_matrix`]).
    pub mass: Matrix,
    /// `C(q, q̇)` — the `n×n` Coriolis/centrifugal matrix built from the
    /// Christoffel symbols of the first kind, `Cᵢⱼ = Σₖ Γᵢⱼₖ·q̇ₖ`; it
    /// multiplies `q̇` ([`coriolis_matrix`]).
    pub coriolis: Matrix,
    /// `G(q)` — the `n` generalized gravity forces, `Gᵢ = ∂V/∂qᵢ`
    /// ([`gravity_vector`]).
    pub gravity: Vec<Ex>,
}

/// Compute the full manipulator equation components: M(q), C(q, q̇), G(q).
///
/// Given kinetic energy T(q, q̇) and potential energy V(q), returns
/// the mass matrix, Coriolis matrix, and gravity vector such that:
///
/// M(q)q̈ + C(q, q̇)q̇ + G(q) = τ
///
/// This is a convenience function that calls [`mass_matrix`],
/// [`coriolis_matrix`], and [`gravity_vector`].
///
/// # Parameters
/// - `kinetic_energy`: T(q, q̇)
/// - `potential_energy`: V(q)
/// - `q_vars`: Generalized coordinate variables [q₁, q₂, ...]
/// - `qdot_vars`: Generalized velocity variables [q̇₁, q̇₂, ...]
///
/// # Returns
///
/// A [`ManipulatorEquation`] holding the `n×n` [`mass`](ManipulatorEquation::mass)
/// matrix, the `n×n` [`coriolis`](ManipulatorEquation::coriolis) matrix and the
/// `n`-element [`gravity`](ManipulatorEquation::gravity) vector.
///
/// # Errors
///
/// Returns [`SymplexError::InvalidArgument`] if `q_vars` and `qdot_vars`
/// differ in length or are empty.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::dynamics::manipulator_equation;
///
/// let ctx = Context::new();
/// let m_val = ctx.symbol("m");
/// let q = ctx.symbol("q");
/// let qd = ctx.symbol("qd");
///
/// let half = ctx.rational(1, 2);
/// let ke = &half * &m_val * &qd.powi(2);
/// let pe = ctx.int(0);
/// let eq = manipulator_equation(&ke, &pe, &[&q], &[&qd]).unwrap();
/// assert_eq!(eq.mass.shape(), (1, 1));
/// assert_eq!(eq.coriolis.shape(), (1, 1));
/// assert_eq!(eq.gravity.len(), 1);
/// ```
pub fn manipulator_equation(
    kinetic_energy: &Ex,
    potential_energy: &Ex,
    q_vars: &[&Ex],
    qdot_vars: &[&Ex],
) -> Result<ManipulatorEquation, SymplexError> {
    let m = mass_matrix(kinetic_energy, qdot_vars)?;
    let c = coriolis_matrix(&m, q_vars, qdot_vars)?;
    let g = gravity_vector(potential_energy, q_vars);
    Ok(ManipulatorEquation {
        mass: m.eval(),
        coriolis: c.eval(),
        gravity: g.into_iter().map(|e| e.eval()).collect(),
    })
}
