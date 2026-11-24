//! ODE System Solver (constant-coefficient systems: ẋ = Ax) (part of [`ode`](super)).

use super::*;

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
    solve_ode_system_with_form(a_matrix, t_var).map(|(sol, _)| sol)
}

/// Is the general solution of [`solve_ode_system_with_form`] `e^{At}·c`
/// (`c = (C1, …, Cn)`, so `x(0) = c`) or a sum of eigenmodes?
#[derive(Clone, Copy, PartialEq, Eq)]
enum SystemForm {
    /// `e^{At}·c`: diagonal `A` or the exact matrix exponential.
    ExpTimesConstants,
    /// `Σ Cᵢ·(modeᵢ)` from the eigenvectors.
    Modes,
}

/// [`solve_ode_system`] together with the shape of the solution.
fn solve_ode_system_with_form(a_matrix: &Matrix, t_var: &Ex) -> Option<(Vec<Ex>, SystemForm)> {
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
        return Some((
            solve_ode_system_diagonal(a_matrix, t_var, n),
            SystemForm::ExpTimesConstants,
        ));
    }

    // Try eigenvalue-based exact solution
    if let Some(sol) = solve_ode_system_eigen(a_matrix, t_var, n) {
        return Some((sol, SystemForm::Modes));
    }

    // Otherwise the exact matrix exponential.
    solve_ode_system_expm(a_matrix, t_var, n).map(|sol| (sol, SystemForm::ExpTimesConstants))
}

/// Solve the initial-value problem `dx/dt = A·x`, `x(0) = x0`.
///
/// Computes the general solution with [`solve_ode_system`], evaluates it
/// at `t = 0`, and determines the constants `C1, …, Cn` from the linear
/// system `x(0) = x0` via [`linsolve`](crate::api::expr_solve_ext::linsolve).
/// When the general solution is `e^{At}·c` (a diagonal `A`, or the exact
/// matrix exponential, e.g. for eigenvalues that are `RootOf`s of an
/// irreducible quartic) the solution is `e^{At}·x0` directly: elimination
/// over the `RootOf` entries did not finish (4×4 matrices with irreducible
/// characteristic polynomials hung).
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
    let (general, form) = solve_ode_system_with_form(a_matrix, t_var).ok_or_else(|| {
        SymplexError::ComputationFailed {
            operation: "solve_ode_system_ivp",
            reason: "could not solve the homogeneous system".into(),
        }
    })?;
    let ctx = t_var.context();
    let constants: Vec<Ex> = (1..=n).map(|i| ctx.symbol(&format!("C{i}"))).collect();
    if form == SystemForm::ExpTimesConstants {
        let pairs: Vec<(&Ex, &Ex)> = constants.iter().zip(x0).collect();
        return Ok(general
            .iter()
            .map(|g| {
                let e = g.subs_map(&pairs).eval();
                let rootof = {
                    let inner = e.inner.read();
                    has_rootof(&inner.arena, e.raw_id())
                };
                if rootof { e } else { e.simplify() }
            })
            .collect());
    }
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

    // Integrate each component w.r.t. t (in the equivalent forms of
    // [`integrate_forms`]: `e^{−2t}·cos 2t·cos t/t` integrates with the
    // cosines written as exponentials, to `Ei` at complex points).
    let mut integrated = Vec::with_capacity(n);
    for i in 0..n {
        let entry = integrand_matrix.get(i, 0);
        let t_id = entry.checked_id(t_var);
        let id = {
            let mut inner = entry.inner.write();
            integrate_forms(&mut inner.arena, entry.raw_id(), t_id)
        };
        integrated.push(entry.wrap(id).eval());
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
        // Free of `I` and of unevaluated `re`/`im` nodes (up to 0.48 found
        // by printing and searching for "re(" and "im(", which a function
        // name ending in `re` or `im` also matched).
        let clean = |p: &Ex| {
            if p.contains(&i_unit) {
                return false;
            }
            let inner = p.inner.read();
            !crate::base::walk::post_order_ids(&inner.arena, p.raw_id())
                .iter()
                .any(|&id| matches!(inner.arena.node(id), ExprNode::Re(_) | ExprNode::Im(_)))
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

/// The roots of `e = 0` in `var` the solver claims
/// ([`known_real_roots`](crate::transforms::solve::known_real_roots): none
/// when it finds none), as `Solution`s.
pub(super) fn known_roots(
    arena: &mut Arena,
    e: ExprId,
    var: ExprId,
) -> Vec<crate::transforms::solve::Solution> {
    crate::transforms::solve::known_real_roots(arena, e, var)
        .into_iter()
        .map(|value| crate::transforms::solve::Solution { value })
        .collect()
}
