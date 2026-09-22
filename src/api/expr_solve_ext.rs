//! Extended solving methods on [`Ex`]: general solutions, systems, recurrences, IVPs.
//!
//! This module hosts the 0.2 solving additions:
//!
//! - [`Ex::solve_general`] — full periodic solution families for trig
//!   equations, with a fresh integer parameter.
//! - [`linsolve`] / [`linsolve_matrix`] — symbolic linear systems
//!   (unique, parametric, or inconsistent) via reduced row-echelon form.
//! - [`solve_numeric_system`] — damped Newton iteration for square
//!   nonlinear systems with a symbolic Jacobian.
//! - [`Ex::solve_ode_ivp`] — initial-value problems on top of the
//!   general ODE solver.

use crate::api::context::Context;
use crate::api::eq::Equation;
use crate::api::expr::Ex;
use crate::base::assumptions::Assumption;
use crate::base::dense_f64;
use crate::base::errors::SymplexError;
use crate::base::node::SymbolId;
use crate::domains::matrix::Matrix;

// ═══════════════════════════════════════════════════════════════════════════
// Zero-form conversion
// ═══════════════════════════════════════════════════════════════════════════

/// Types that can be viewed as an equation `expr = 0`.
///
/// Implemented for [`Ex`] (the expression itself is the zero form) and
/// [`Equation`] (`lhs - rhs`), so solver entry points accept either.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::polysys::ZeroForm;
///
/// let ctx = Context::new();
/// let x = ctx.symbol("x");
/// let eq = Equation::new(&x + 1, ctx.int(3));
/// assert_eq!(format!("{}", eq.to_zero_form()), "x - 2");
/// ```
pub trait ZeroForm {
    /// Return the expression that equals zero when the equation holds.
    fn to_zero_form(&self) -> Ex;
}

impl ZeroForm for Ex {
    fn to_zero_form(&self) -> Ex {
        self.clone()
    }
}

impl ZeroForm for Equation {
    fn to_zero_form(&self) -> Ex {
        self.to_expr()
    }
}

impl<T: ZeroForm> ZeroForm for &T {
    fn to_zero_form(&self) -> Ex {
        (*self).to_zero_form()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Linear systems
// ═══════════════════════════════════════════════════════════════════════════

/// Result of solving a linear system with [`linsolve`].
///
/// Values are always given in the order of the `vars` slice passed to the
/// solver.
#[derive(Debug, Clone)]
pub enum LinearSolution {
    /// Exactly one solution: `(variable, value)` pairs.
    Unique(Vec<(Ex, Ex)>),
    /// Infinitely many solutions.  `solution` gives every variable — pivot
    /// variables are expressed in terms of the `free` variables, and each
    /// free variable maps to itself.
    Parametric {
        /// `(variable, value)` pairs for every unknown.
        solution: Vec<(Ex, Ex)>,
        /// The free (parameter) variables, a subset of the unknowns.
        free: Vec<Ex>,
    },
    /// No solution: elimination produced a row `0 = c` with `c ≠ 0`.
    ///
    /// This is reported as a variant rather than an `Err` because it is a
    /// legitimate mathematical outcome; `Err` is reserved for malformed
    /// input (non-linear equations, empty input, shape mismatch).
    Inconsistent,
}

impl LinearSolution {
    /// `true` for [`LinearSolution::Unique`].
    #[must_use]
    #[allow(dead_code)] // public API; reachable once re-exported from lib.rs
    pub fn is_unique(&self) -> bool {
        matches!(self, LinearSolution::Unique(_))
    }

    /// `true` for [`LinearSolution::Inconsistent`].
    #[must_use]
    #[allow(dead_code)] // public API; reachable once re-exported from lib.rs
    pub fn is_inconsistent(&self) -> bool {
        matches!(self, LinearSolution::Inconsistent)
    }

    /// The `(variable, value)` pairs, or `None` if inconsistent.
    #[must_use]
    #[allow(dead_code)] // public API; reachable once re-exported from lib.rs
    pub fn pairs(&self) -> Option<&[(Ex, Ex)]> {
        match self {
            LinearSolution::Unique(p) => Some(p),
            LinearSolution::Parametric { solution, .. } => Some(solution),
            LinearSolution::Inconsistent => None,
        }
    }

    /// Look up the value of a particular variable.
    #[must_use]
    #[allow(dead_code)] // public API; reachable once re-exported from lib.rs
    pub fn get(&self, var: &Ex) -> Option<Ex> {
        self.pairs()?
            .iter()
            .find(|(v, _)| v == var)
            .map(|(_, val)| val.clone())
    }
}

/// Solve a system of linear equations symbolically.
///
/// Each element of `eqs` is either an [`Ex`] (meaning `expr = 0`) or an
/// [`Equation`].  Coefficients may be symbolic; the solver performs
/// reduced row-echelon elimination, preferring numeric pivots and
/// treating a symbolic pivot as nonzero (generic solution).
///
/// Under-determined systems return [`LinearSolution::Parametric`],
/// over-determined but consistent systems return
/// [`LinearSolution::Unique`], and contradictory systems return
/// [`LinearSolution::Inconsistent`].
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] if `eqs` or `vars` is empty, or an
/// equation is not linear in the unknowns.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::polysys::linsolve;
///
/// let ctx = Context::new();
/// let (x, y, a, b) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("a"), ctx.symbol("b"));
/// // a*x + y = 1,  x - y = b
/// let sol = linsolve(&[&a * &x + &y - 1, &x - &y - &b], &[x.clone(), y.clone()]).unwrap();
/// let xv = sol.get(&x).unwrap();
/// // x = (1 + b) / (a + 1)
/// let residual = (&xv * (&a + 1) - (&b + 1)).simplify();
/// assert!(residual.is_zero_structural(), "x = {xv}");
///
/// // Under-determined: x + y = 1
/// let sol = linsolve(&[&x + &y - 1], &[x.clone(), y.clone()]).unwrap();
/// assert!(matches!(sol, symplex::polysys::LinearSolution::Parametric { .. }));
///
/// // Over-determined but consistent (three equations, two unknowns): `Unique`,
/// // not an error.  A contradictory third equation gives `Inconsistent`.
/// use symplex::polysys::LinearSolution;
/// let sol = linsolve(&[&x - 1, &y - 2, &x + &y - 3], &[x.clone(), y.clone()]).unwrap();
/// assert!(matches!(sol, LinearSolution::Unique(_)));
/// let sol = linsolve(&[&x - 1, &y - 2, &x + &y - 4], &[x.clone(), y.clone()]).unwrap();
/// assert!(matches!(sol, LinearSolution::Inconsistent));
/// ```
pub fn linsolve<E: ZeroForm>(eqs: &[E], vars: &[Ex]) -> Result<LinearSolution, SymplexError> {
    let zero_forms: Vec<Ex> = eqs.iter().map(ZeroForm::to_zero_form).collect();
    let result = crate::domains::linalg::linsolve_symbolic(&zero_forms, vars)?;
    Ok(wrap_linear_result(result, vars))
}

/// Convert the backend result into the public enum.
fn wrap_linear_result(
    result: crate::domains::linalg::SymbolicLinearResult,
    vars: &[Ex],
) -> LinearSolution {
    if result.inconsistent {
        return LinearSolution::Inconsistent;
    }
    let solution: Vec<(Ex, Ex)> = vars
        .iter()
        .cloned()
        .zip(result.values.iter().cloned())
        .collect();
    if result.free.is_empty() {
        LinearSolution::Unique(solution)
    } else {
        let free = result.free.iter().map(|&i| vars[i].clone()).collect();
        LinearSolution::Parametric { solution, free }
    }
}

/// Solve `A·x = b` for a coefficient matrix `a` (m×n) and right-hand side
/// `b` (m×1), with symbolic entries allowed.
///
/// The unknowns are named `x1, …, xn` in the context of `a`.  Unlike
/// [`Matrix::solve`], this handles rectangular, singular and inconsistent
/// systems, reporting free variables where appropriate.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] if `b` is not a column vector with
/// as many rows as `a`.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::polysys::{linsolve_matrix, LinearSolution};
///
/// let ctx = Context::new();
/// let a = matrix![ctx, [1, 1], [1, -1]];
/// let b = Matrix::col_vector(vec![ctx.int(3), ctx.int(1)]);
/// match linsolve_matrix(&a, &b).unwrap() {
///     LinearSolution::Unique(pairs) => {
///         assert_eq!(format!("{}", pairs[0].1), "2");
///         assert_eq!(format!("{}", pairs[1].1), "1");
///     }
///     other => panic!("expected unique solution, got {other:?}"),
/// }
/// ```
#[allow(dead_code)] // public API; reachable once re-exported from lib.rs
pub fn linsolve_matrix(a: &Matrix, b: &Matrix) -> Result<LinearSolution, SymplexError> {
    let m = a.nrows();
    let n = a.ncols();
    if b.ncols() != 1 || b.nrows() != m {
        return Err(SymplexError::InvalidArgument {
            operation: "linsolve_matrix",
            reason: format!(
                "right-hand side must be {m}×1, got {}×{}",
                b.nrows(),
                b.ncols()
            ),
        });
    }
    let ctx = a.get(0, 0).context();
    let unknowns: Vec<Ex> = (1..=n).map(|i| ctx.symbol(&format!("x{i}"))).collect();
    let rows: Vec<Vec<Ex>> = (0..m)
        .map(|i| (0..n).map(|j| a.get(i, j).clone()).collect())
        .collect();
    let rhs: Vec<Ex> = (0..m).map(|i| b.get(i, 0).clone()).collect();
    let result = crate::domains::linalg::rref_solve(rows, rhs, &unknowns)?;
    Ok(wrap_linear_result(result, &unknowns))
}

// ═══════════════════════════════════════════════════════════════════════════
// General (periodic) solutions
// ═══════════════════════════════════════════════════════════════════════════

/// Result of [`Ex::solve_general`]: solution families plus the integer
/// parameters they are expressed with.
#[derive(Debug, Clone)]
pub struct GeneralSolution {
    /// Solution expressions.  Families of periodic solutions mention the
    /// symbols in `parameters`; non-periodic solutions do not.
    pub solutions: Vec<Ex>,
    /// Fresh integer-assumed parameter symbols (`n`, `n1`, …) that appear
    /// in `solutions`.  Empty when no periodic family was produced.
    pub parameters: Vec<Ex>,
}

impl GeneralSolution {
    /// Substitute a concrete integer for every parameter, giving one
    /// representative of each family.
    #[must_use]
    pub fn instance(&self, k: i64) -> Vec<Ex> {
        self.solutions
            .iter()
            .map(|s| {
                let mut e = s.clone();
                for p in &self.parameters {
                    e = e.subs_i64(p, k);
                }
                e.eval()
            })
            .collect()
    }
}

/// Create a symbol whose name is not yet used anywhere in the context.
///
/// Tries `base`, then `base1`, `base2`, … and attaches the given
/// assumptions.
pub(crate) fn fresh_symbol(ctx: &Context, base: &str, assumptions: &[Assumption]) -> Ex {
    let name = {
        let inner = ctx.inner.read();
        let taken = |name: &str| -> bool {
            let n = inner.arena.symbols.len();
            (0..n).any(|i| inner.arena.symbols.name(SymbolId(i as u32)) == name)
        };
        if !taken(base) {
            base.to_string()
        } else {
            let mut k = 1usize;
            loop {
                let candidate = format!("{base}{k}");
                if !taken(&candidate) {
                    break candidate;
                }
                k += 1;
            }
        }
    };
    ctx.symbol_with(&name, assumptions)
}

impl Ex {
    /// Solve `self = 0` for `var`, returning **general** solution families.
    ///
    /// Unlike [`solve`](Ex::solve), which returns only principal branches,
    /// this expresses periodic solutions with a fresh integer parameter
    /// (`n`, or `n1`, `n2`, … if `n` is already in use), exposed through
    /// [`GeneralSolution::parameters`]:
    ///
    /// - `sin(x) = c` → `asin(c) + 2πn`, `π − asin(c) + 2πn`
    /// - `cos(x) = c` → `±acos(c) + 2πn`
    /// - `tan(x) = c` → `atan(c) + πn`
    ///
    /// Linear arguments (`sin(a·x + b) = c`) and change-of-variable forms
    /// (`sin²x − sin x = 0`) are supported.  Non-periodic equations return
    /// the same solutions as `solve` with an empty parameter list.
    ///
    /// # Errors
    ///
    /// Same as [`solve`](Ex::solve): `InfiniteSolutions` for identities,
    /// `NoSolution` for contradictions, `ComputationFailed` when nothing
    /// applies.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let eq = &x.sin() - &ctx.rational(1, 2);
    /// let fam = eq.solve_general(&x).unwrap();
    /// assert_eq!(fam.solutions.len(), 2);
    /// assert_eq!(fam.parameters.len(), 1);
    /// // Every member of every family satisfies the equation.
    /// for k in -2..=2 {
    ///     for s in fam.instance(k) {
    ///         let residual = eq.subs(&x, &s).eval_f64().unwrap();
    ///         assert!(residual.abs() < 1e-12);
    ///     }
    /// }
    /// ```
    pub fn solve_general(&self, var: &Ex) -> Result<GeneralSolution, SymplexError> {
        let var_id = self.checked_id(var);
        let ctx = self.context();
        let param = fresh_symbol(&ctx, "n", &[Assumption::Integer]);
        let param_id = self.checked_id(&param);

        let outcome = {
            let mut inner = self.inner.write();
            crate::transforms::solve::solve_general(
                &mut inner.arena,
                self.raw_id(),
                var_id,
                param_id,
            )
        };
        match outcome {
            crate::transforms::solve::SolveOutcome::Solutions(solutions) => {
                if solutions.is_empty() {
                    let is_poly = {
                        let inner = self.inner.read();
                        crate::poly::polybridge::expr_to_poly(&inner.arena, self.raw_id(), var_id)
                            .is_some()
                    };
                    if !is_poly {
                        return Err(SymplexError::ComputationFailed {
                            operation: "solve_general",
                            reason: "expression is not polynomial in the given variable and transcendental solver could not find solutions".into(),
                        });
                    }
                    return Ok(GeneralSolution {
                        solutions: Vec::new(),
                        parameters: Vec::new(),
                    });
                }
                let solutions: Vec<Ex> = solutions
                    .into_iter()
                    .map(|s| self.wrap(s.value).eval())
                    .collect();
                let uses_param = solutions.iter().any(|s| s.contains(&param));
                Ok(GeneralSolution {
                    solutions,
                    parameters: if uses_param { vec![param] } else { Vec::new() },
                })
            }
            crate::transforms::solve::SolveOutcome::Identity => {
                Err(SymplexError::InfiniteSolutions {
                    operation: "solve_general",
                    reason: format!(
                        "equation is an identity (0 = 0): every value of {var} is a solution"
                    ),
                })
            }
            crate::transforms::solve::SolveOutcome::NoSolution(reason) => {
                Err(SymplexError::NoSolution {
                    operation: "solve_general",
                    reason,
                })
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Numeric multivariate solving (Newton)
// ═══════════════════════════════════════════════════════════════════════════

/// Options for [`solve_numeric_system_with`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)] // public API; reachable once re-exported from lib.rs
pub struct NewtonOpts {
    /// Convergence threshold on the residual norm `‖F(x)‖∞`.
    pub tol: f64,
    /// Maximum number of Newton iterations.
    pub max_iter: usize,
    /// Enable backtracking line search (halve the step until the residual
    /// decreases).  Disable for pure Newton steps.
    pub damping: bool,
}

impl Default for NewtonOpts {
    fn default() -> Self {
        Self {
            tol: 1e-12,
            max_iter: 100,
            damping: true,
        }
    }
}

/// Callable scalar function of `k` variables, compiled when possible and
/// falling back to exact substitution + `eval_f64` otherwise.
enum Evaluator {
    Compiled(crate::output::lambdify::CompiledFn),
    Symbolic(Ex, Vec<Ex>),
}

impl Evaluator {
    fn new(expr: &Ex, vars: &[Ex], names: &[&str]) -> Self {
        match expr.compile(names) {
            Ok(f) => Evaluator::Compiled(f),
            Err(_) => Evaluator::Symbolic(expr.clone(), vars.to_vec()),
        }
    }

    fn call(&self, x: &[f64]) -> Result<f64, SymplexError> {
        match self {
            Evaluator::Compiled(f) => Ok(f.call(x)),
            Evaluator::Symbolic(expr, vars) => {
                let mut e = expr.clone();
                for (v, &xv) in vars.iter().zip(x) {
                    let val = float_to_ex(v, xv)?;
                    e = e.subs(v, &val);
                }
                e.eval_f64()
            }
        }
    }
}

/// Exact rational literal for an `f64` (dyadic), in the context of `like`.
fn float_to_ex(like: &Ex, x: f64) -> Result<Ex, SymplexError> {
    let r = num_rational::Ratio::<num_bigint::BigInt>::from_float(x).ok_or_else(|| {
        SymplexError::ComputationFailed {
            operation: "solve_numeric_system",
            reason: format!("non-finite value {x} encountered"),
        }
    })?;
    let mut inner = like.inner.write();
    let nid = inner.arena.intern_num(r);
    let id = inner.arena.intern(crate::base::node::ExprNode::Num(nid));
    drop(inner);
    Ok(like.wrap(id))
}

/// Solve the dense linear system `a·x = b` with partial pivoting
/// ([`dense_f64::solve_partial_pivot`]).  Returns `None` if the matrix is
/// numerically singular or has a non-finite entry.
fn gauss_solve(a: Vec<Vec<f64>>, b: Vec<f64>) -> Option<Vec<f64>> {
    let flat = dense_f64::flatten(&a);
    if flat.iter().any(|v| !v.is_finite()) {
        return None;
    }
    dense_f64::solve_partial_pivot(&flat, b.len(), &b)
}

fn inf_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |m, &x| m.max(x.abs()))
}

/// Newton's method for the square nonlinear system `eqs = 0` in `vars`,
/// starting from `x0`, with default options (`tol = 1e-12`,
/// `max_iter = 100`, damping on).
///
/// See [`solve_numeric_system_with`].
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::polysys::solve_numeric_system;
///
/// let ctx = Context::new();
/// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
/// // Intersection of the unit circle with y = x, near (1, 1)
/// let eqs = [&x.powi(2) + &y.powi(2) - 1, &y - &x];
/// let sol = solve_numeric_system(&eqs, &[x, y], &[1.0, 1.0]).unwrap();
/// let r = std::f64::consts::FRAC_1_SQRT_2;
/// assert!((sol[0] - r).abs() < 1e-10 && (sol[1] - r).abs() < 1e-10);
/// ```
#[allow(dead_code)] // public API; reachable once re-exported from lib.rs
pub fn solve_numeric_system(eqs: &[Ex], vars: &[Ex], x0: &[f64]) -> Result<Vec<f64>, SymplexError> {
    solve_numeric_system_with(eqs, vars, x0, &NewtonOpts::default())
}

/// Newton's method with explicit [`NewtonOpts`].
///
/// The Jacobian is computed symbolically ([`crate::matrix::jacobian`]) and
/// compiled to native closures where possible.  Each step solves
/// `J·Δ = −F` by Gaussian elimination with partial pivoting; with
/// `damping` on, the step is halved until the residual norm decreases.
///
/// # Errors
///
/// - [`SymplexError::InvalidArgument`] if the system is not square or
///   `x0` has the wrong length.
/// - [`SymplexError::ComputationFailed`] if the Jacobian becomes singular
///   or the iteration does not converge within `max_iter` steps; the
///   message reports the final residual norm and iterate.
#[allow(dead_code)] // public API; reachable once re-exported from lib.rs
pub fn solve_numeric_system_with(
    eqs: &[Ex],
    vars: &[Ex],
    x0: &[f64],
    opts: &NewtonOpts,
) -> Result<Vec<f64>, SymplexError> {
    let n = vars.len();
    if n == 0 || eqs.len() != n {
        return Err(SymplexError::InvalidArgument {
            operation: "solve_numeric_system",
            reason: format!(
                "system must be square and non-empty: {} equations, {} unknowns",
                eqs.len(),
                n
            ),
        });
    }
    if x0.len() != n {
        return Err(SymplexError::InvalidArgument {
            operation: "solve_numeric_system",
            reason: format!("initial guess has {} entries, expected {n}", x0.len()),
        });
    }
    for v in vars {
        let _ = eqs[0].checked_id(v);
    }
    for e in eqs {
        let _ = eqs[0].checked_id(e);
    }

    let names: Vec<String> = vars.iter().map(|v| format!("{v}")).collect();
    let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();

    let f_refs: Vec<&Ex> = eqs.iter().collect();
    let v_refs: Vec<&Ex> = vars.iter().collect();
    let jac = crate::domains::matrix::jacobian(&f_refs, &v_refs);

    let f_eval: Vec<Evaluator> = eqs
        .iter()
        .map(|e| Evaluator::new(e, vars, &name_refs))
        .collect();
    let j_eval: Vec<Vec<Evaluator>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| Evaluator::new(jac.get(i, j), vars, &name_refs))
                .collect()
        })
        .collect();

    let residual = |x: &[f64]| -> Result<Vec<f64>, SymplexError> {
        f_eval.iter().map(|f| f.call(x)).collect()
    };

    let mut x = x0.to_vec();
    let mut fx = residual(&x)?;
    let mut norm = inf_norm(&fx);

    for _iter in 0..opts.max_iter {
        if norm < opts.tol {
            return Ok(x);
        }
        if !norm.is_finite() {
            break;
        }
        let mut jm = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                jm[i][j] = j_eval[i][j].call(&x)?;
            }
        }
        let neg_f: Vec<f64> = fx.iter().map(|v| -v).collect();
        let dx = match gauss_solve(jm, neg_f) {
            Some(d) => d,
            None => {
                return Err(SymplexError::ComputationFailed {
                    operation: "solve_numeric_system",
                    reason: format!("Jacobian is singular at x = {x:?} (residual norm {norm:.3e})"),
                });
            }
        };

        // Backtracking line search.
        let mut step = 1.0;
        loop {
            let x_new: Vec<f64> = x.iter().zip(&dx).map(|(a, d)| a + step * d).collect();
            let f_new = residual(&x_new)?;
            let norm_new = inf_norm(&f_new);
            if !opts.damping || norm_new < norm || step < 1e-10 {
                x = x_new;
                fx = f_new;
                norm = norm_new;
                break;
            }
            step *= 0.5;
        }
    }

    if norm < opts.tol {
        return Ok(x);
    }
    Err(SymplexError::ComputationFailed {
        operation: "solve_numeric_system",
        reason: format!(
            "did not converge within {} iterations: residual norm {norm:.3e} at x = {x:?}",
            opts.max_iter
        ),
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// ODE initial-value problems
// ═══════════════════════════════════════════════════════════════════════════

/// One initial condition of an ODE initial-value problem:
/// **`y^(order)(x) = value`**, i.e. the `order`-th derivative of the
/// unknown function, evaluated at the point `x`, equals `value`.
///
/// `order = 0` is the plain `y(x) = value`.  Used by
/// [`Ex::solve_ode_ivp`].
///
/// ```
/// use symplex::prelude::*;
///
/// let ctx = Context::new();
/// // y'(0) = 1
/// let ic = InitialCondition { order: 1, x: ctx.int(0), value: ctx.int(1) };
/// assert_eq!(ic.order, 1);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct InitialCondition {
    /// Derivative order `k` of the condition `y^(k)(x) = value`.
    pub order: usize,
    /// The point at which the derivative is prescribed.
    pub x: Ex,
    /// The prescribed value of `y^(order)` at `x`.
    pub value: Ex,
}

impl Ex {
    /// Solve the ODE `self = 0` for `func(var)` subject to initial
    /// conditions.
    ///
    /// Each [`InitialCondition`] `{ order: k, x: x0, value }` means
    /// `d^k func / d var^k (x0) = value` (`k = 0` is `func(x0) = value`).
    /// The general solution is found with [`solve_ode`](Ex::solve_ode),
    /// then the integration constants `C1, C2, …` are determined by
    /// substituting the conditions and solving the resulting (usually
    /// linear) system with [`linsolve`]; nonlinear constant equations are
    /// handled one at a time with [`solve`](Ex::solve).  Constants not
    /// pinned down by the conditions remain in the result.
    ///
    /// # Errors
    ///
    /// - [`SymplexError::ComputationFailed`] if the ODE cannot be solved
    ///   or the constants cannot be determined.
    /// - [`SymplexError::NoSolution`] if the initial conditions are
    ///   contradictory.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// // y'' + y = 0, y(0) = 0, y'(0) = 1  →  y = sin(x)
    /// let ode = &y.formal_diff(&x).formal_diff(&x) + &y;
    /// let sol = ode
    ///     .solve_ode_ivp(
    ///         &y,
    ///         &x,
    ///         &[
    ///             InitialCondition { order: 0, x: ctx.int(0), value: ctx.int(0) },
    ///             InitialCondition { order: 1, x: ctx.int(0), value: ctx.int(1) },
    ///         ],
    ///     )
    ///     .unwrap();
    /// assert_eq!(format!("{}", sol.simplify()), "sin(x)");
    /// ```
    pub fn solve_ode_ivp(
        &self,
        func: &Ex,
        var: &Ex,
        ics: &[InitialCondition],
    ) -> Result<Ex, SymplexError> {
        let func_id = self.checked_id(func);
        let var_id = self.checked_id(var);
        for ic in ics {
            let _ = self.checked_id(&ic.x);
            let _ = self.checked_id(&ic.value);
        }

        let (general, constants): (Ex, Vec<Ex>) = {
            let mut inner = self.inner.write();
            match crate::calculus::ode::dsolve(&mut inner.arena, self.raw_id(), func_id, var_id) {
                Some(res) => {
                    let sol = res.solution;
                    let consts = res.constants.clone();
                    drop(inner);
                    (
                        self.wrap(sol),
                        consts.into_iter().map(|c| self.wrap(c)).collect(),
                    )
                }
                None => {
                    drop(inner);
                    return Err(SymplexError::ComputationFailed {
                        operation: "solve_ode_ivp",
                        reason: "could not find the general solution of the ODE".into(),
                    });
                }
            }
        };
        if general.has_unevaluated() {
            return Err(SymplexError::ComputationFailed {
                operation: "solve_ode_ivp",
                reason: format!("general solution contains unevaluated forms: {general}"),
            });
        }
        // Implicit solutions (still mentioning `func`) cannot be fitted.
        if general.contains(func) {
            return Err(SymplexError::ComputationFailed {
                operation: "solve_ode_ivp",
                reason: format!("general solution is implicit in {func}: {general}"),
            });
        }
        apply_initial_conditions(&general, &constants, var, ics, "solve_ode_ivp")
    }
}

impl Ex {
    /// Solve the Riccati equation `self = 0`, i.e.
    /// `y' = q₀(x) + q₁(x)·y + q₂(x)·y²`, given a known particular
    /// solution `particular`.
    ///
    /// The substitution `y = y_p + 1/v` reduces the equation to the linear
    /// ODE `v' + (q₁ + 2·q₂·y_p)·v = −q₂`; the result is `y_p + 1/v` with the
    /// integration constant `C1`.
    ///
    /// # Errors
    ///
    /// - [`SymplexError::InvalidArgument`] if `self` is not a Riccati
    ///   equation in `func` or `particular` does not satisfy it.
    /// - [`SymplexError::ComputationFailed`] if the linear equation for
    ///   `v` cannot be solved in closed form.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// // y' = y² - 2/x² has the particular solution y = 1/x
    /// let ode = &y.formal_diff(&x) - &y.powi(2) + &(&ctx.int(2) / &x.powi(2));
    /// let sol = ode.solve_riccati(&y, &x, &(&ctx.int(1) / &x)).unwrap();
    /// assert!(sol.contains(&ctx.symbol("C1")));
    /// assert!(ode.check_ode_solution(&sol, &y, &x));
    /// ```
    pub fn solve_riccati(&self, func: &Ex, var: &Ex, particular: &Ex) -> Result<Ex, SymplexError> {
        let func_id = self.checked_id(func);
        let var_id = self.checked_id(var);
        let part_id = self.checked_id(particular);
        let mut inner = self.inner.write();
        match crate::calculus::ode::solve_riccati(
            &mut inner.arena,
            self.raw_id(),
            func_id,
            var_id,
            part_id,
        ) {
            Some(res) => {
                let sol = res.solution;
                drop(inner);
                let sol = self.wrap(sol);
                if sol.has_unevaluated() {
                    return Err(SymplexError::ComputationFailed {
                        operation: "solve_riccati",
                        reason: format!(
                            "linear equation for the substitution could not be solved in closed form: {sol}"
                        ),
                    });
                }
                Ok(sol)
            }
            None => {
                drop(inner);
                Err(SymplexError::InvalidArgument {
                    operation: "solve_riccati",
                    reason: format!(
                        "not a Riccati equation in {func}, or {particular} is not a particular solution"
                    ),
                })
            }
        }
    }
}

/// Fit integration constants to initial conditions `y^(order)(x) = value`.
pub(crate) fn apply_initial_conditions(
    general: &Ex,
    constants: &[Ex],
    var: &Ex,
    ics: &[InitialCondition],
    operation: &'static str,
) -> Result<Ex, SymplexError> {
    if ics.is_empty() || constants.is_empty() {
        return Ok(general.clone());
    }
    // Build one equation per initial condition.
    let mut eqs: Vec<Ex> = Vec::with_capacity(ics.len());
    for ic in ics {
        let mut d = general.clone();
        for _ in 0..ic.order {
            d = d.diff(var);
        }
        let at = d.subs(var, &ic.x).eval();
        eqs.push((&at - &ic.value).eval());
    }
    fit_constants(general, constants, &eqs, operation)
}

/// Solve `eqs = 0` for `constants` (linear first, then one-at-a-time) and
/// substitute into `general`.
pub(crate) fn fit_constants(
    general: &Ex,
    constants: &[Ex],
    eqs: &[Ex],
    operation: &'static str,
) -> Result<Ex, SymplexError> {
    // Only constants that actually appear matter.
    let present: Vec<Ex> = constants
        .iter()
        .filter(|c| general.contains(c) || eqs.iter().any(|e| e.contains(c)))
        .cloned()
        .collect();
    if present.is_empty() {
        return Ok(general.clone());
    }
    match linsolve(eqs, &present) {
        Ok(LinearSolution::Inconsistent) => Err(SymplexError::NoSolution {
            operation,
            reason: "initial conditions are contradictory".into(),
        }),
        Ok(LinearSolution::Unique(pairs)) => {
            let mut sol = general.clone();
            for (c, v) in &pairs {
                sol = sol.subs(c, v);
            }
            // Substituting algebraic constants can swell the expression;
            // bound the work before simplifying.
            crate::domains::matrix::budget_check([&sol], operation)?;
            let sol = sol.eval();
            crate::domains::matrix::budget_check([&sol], operation)?;
            Ok(sol.simplify())
        }
        Ok(LinearSolution::Parametric { solution, free }) => {
            let mut sol = general.clone();
            for (c, v) in &solution {
                if !free.contains(c) {
                    sol = sol.subs(c, v);
                }
            }
            crate::domains::matrix::budget_check([&sol], operation)?;
            Ok(sol.eval().simplify())
        }
        Err(_) => {
            // Nonlinear in the constants: solve sequentially.
            let mut sol = general.clone();
            let mut remaining: Vec<Ex> = eqs.to_vec();
            let mut unsolved: Vec<Ex> = present.clone();
            while let Some(pos) = remaining.iter().position(|e| !e.is_zero_structural()) {
                let eq = remaining.remove(pos);
                let eq = eq.eval();
                if eq.is_zero_structural() {
                    continue;
                }
                let target = unsolved
                    .iter()
                    .position(|c| eq.contains(c))
                    .ok_or_else(|| SymplexError::NoSolution {
                        operation,
                        reason: format!("initial conditions are contradictory: {eq} = 0"),
                    })?;
                let c = unsolved.remove(target);
                let roots = eq.solve(&c).map_err(|e| SymplexError::ComputationFailed {
                    operation,
                    reason: format!("could not solve for {c}: {e}"),
                })?;
                let value =
                    roots
                        .first()
                        .cloned()
                        .ok_or_else(|| SymplexError::ComputationFailed {
                            operation,
                            reason: format!("no value of {c} satisfies {eq} = 0"),
                        })?;
                sol = sol.subs(&c, &value);
                remaining = remaining
                    .iter()
                    .map(|e| e.subs(&c, &value).eval())
                    .collect();
            }
            Ok(sol.eval().simplify())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_symbol_avoids_existing_names() {
        let ctx = Context::new();
        let n = ctx.symbol("n");
        let fresh = fresh_symbol(&ctx, "n", &[Assumption::Integer]);
        assert_ne!(fresh, n);
        assert_eq!(format!("{fresh}"), "n1");
        let fresh2 = fresh_symbol(&ctx, "n", &[Assumption::Integer]);
        assert_eq!(format!("{fresh2}"), "n2");
    }

    #[test]
    fn linsolve_inconsistent() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let sol = linsolve(&[&x - 1, &x - 2], std::slice::from_ref(&x)).unwrap();
        assert!(sol.is_inconsistent());
    }

    #[test]
    fn linsolve_rejects_nonlinear() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let r = linsolve(&[x.powi(2) - 1], std::slice::from_ref(&x));
        assert!(matches!(r, Err(SymplexError::InvalidArgument { .. })));
    }

    #[test]
    fn linsolve_accepts_equations() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let y = ctx.symbol("y");
        let e1 = Equation::new(&x + &y, ctx.int(3));
        let e2 = Equation::new(&x - &y, ctx.int(1));
        let sol = linsolve(&[e1, e2], &[x.clone(), y.clone()]).unwrap();
        assert_eq!(format!("{}", sol.get(&x).unwrap()), "2");
        assert_eq!(format!("{}", sol.get(&y).unwrap()), "1");
    }

    #[test]
    fn gauss_solve_basic() {
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let b = vec![3.0, 5.0];
        let x = gauss_solve(a, b).unwrap();
        assert!((x[0] - 0.8).abs() < 1e-12 && (x[1] - 1.4).abs() < 1e-12);
    }

    #[test]
    fn gauss_solve_singular() {
        let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        assert!(gauss_solve(a, vec![1.0, 2.0]).is_none());
    }
}
