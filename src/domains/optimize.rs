//! Numerical optimisation and root bracketing.
//!
//! Plain `f64` routines that need nothing but a closure:
//!
//! | Task | Routines |
//! |------|----------|
//! | Bracketed root finding | [`brent_root`] (Brent–Dekker), [`bisect`] |
//! | Root polishing from a point | [`newton_root`] |
//! | Derivative-free local minimisation | [`nelder_mead`] |
//! | Bracketed scalar minimisation | [`minimize_scalar`] (Brent), [`golden_section`] |
//! | Global minimisation in a box | [`differential_evolution`] (DE/rand/1/bin + Nelder–Mead polish) |
//! | Least-squares fitting | [`poly_fit`], [`poly_fit_exact`], [`linear_fit`] |
//! | Helpers | [`trapezoid`], [`eval_poly`] |
//!
//! and convenience methods on [`Ex`] that compile an expression with
//! [`Ex::compile`] and hand the resulting closure to the matching routine:
//! [`Ex::find_root_bracket`], [`Ex::minimize_numeric`],
//! [`Ex::minimize_scalar_numeric`], [`Ex::minimize_global_numeric`] and
//! [`Ex::poly_fit_points`] (exact rational least squares).
//!
//! # Conventions
//!
//! * Every routine is deterministic — [`differential_evolution`] draws its
//!   random numbers from a local SplitMix64 generator seeded by
//!   [`DeOpts::seed`] — and bounded by an explicit iteration budget.
//! * Nothing panics.  Bad input (empty vectors, a bracket without a sign
//!   change, non-finite bounds, `degree ≥ len`, …) is reported as
//!   [`SymplexError::InvalidArgument`]; running out of iterations or hitting
//!   a non-finite function value is [`SymplexError::ComputationFailed`].
//!   The minimisers that return a [`MinimizeResult`] report an exhausted
//!   budget through [`MinimizeResult::converged`] instead of an error, so
//!   the best point found is never thrown away.
//! * Polynomial coefficients are always in **ascending** degree:
//!   `[c₀, c₁, …, c_d]` represents `c₀ + c₁·x + … + c_d·x^d`.
//!
//! ```
//! use symplex::optimize::{brent_root, nelder_mead, MinimizeOpts, RootOpts};
//!
//! // √2 as the root of x² − 2 on [0, 2].
//! let r = brent_root(|x| x * x - 2.0, 0.0, 2.0, &RootOpts::default()).unwrap();
//! assert!((r - 2f64.sqrt()).abs() < 1e-12);
//!
//! // Minimum of (x − 1)² + (y + 2)² by Nelder–Mead.
//! let bowl = |p: &[f64]| (p[0] - 1.0).powi(2) + (p[1] + 2.0).powi(2);
//! let m = nelder_mead(bowl, &[0.0, 0.0], &MinimizeOpts::default()).unwrap();
//! assert!(m.converged);
//! assert!((m.x[0] - 1.0).abs() < 1e-6 && (m.x[1] + 2.0).abs() < 1e-6);
//! ```

use crate::api::context::Context;
use crate::api::expr::Ex;
use crate::base::errors::SymplexError;
use crate::base::node::ExprNode;
use crate::output::lambdify::CompiledFn;
use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Zero};

// ═══════════════════════════════════════════════════════════════════════════
// Shared helpers
// ═══════════════════════════════════════════════════════════════════════════

fn invalid(operation: &'static str, reason: String) -> SymplexError {
    SymplexError::InvalidArgument { operation, reason }
}

fn failed(operation: &'static str, reason: String) -> SymplexError {
    SymplexError::ComputationFailed { operation, reason }
}

/// `NaN` objective values are treated as "worse than anything" so that a
/// minimiser steps away from them instead of propagating the NaN.
fn nan_to_inf(v: f64) -> f64 {
    if v.is_nan() { f64::INFINITY } else { v }
}

fn check_endpoints(op: &'static str, a: f64, b: f64) -> Result<(), SymplexError> {
    if !a.is_finite() || !b.is_finite() {
        return Err(invalid(
            op,
            format!("interval endpoints must be finite, got [{a}, {b}]"),
        ));
    }
    Ok(())
}

/// Validate an interval for the scalar minimisers: finite endpoints with
/// positive width.  A reversed interval is accepted and returned ordered.
fn check_interval(op: &'static str, a: f64, b: f64) -> Result<(f64, f64), SymplexError> {
    check_endpoints(op, a, b)?;
    if a == b {
        return Err(invalid(
            op,
            format!("interval must have positive width, got [{a}, {b}]"),
        ));
    }
    Ok(if a < b { (a, b) } else { (b, a) })
}

// ═══════════════════════════════════════════════════════════════════════════
// Root finding
// ═══════════════════════════════════════════════════════════════════════════

/// Options for the scalar root finders [`brent_root`], [`bisect`] and
/// [`newton_root`].
///
/// ```
/// use symplex::optimize::RootOpts;
///
/// let opts = RootOpts::default();
/// assert_eq!(opts.xtol, 2e-12);
/// assert_eq!(opts.rtol, 4.0 * f64::EPSILON);
/// assert_eq!(opts.max_iter, 100);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct RootOpts {
    /// Absolute tolerance on the root location (default `2e-12`).
    pub xtol: f64,
    /// Relative tolerance on the root location (default `4·ε`).
    ///
    /// The bracketing methods stop once the bracket width is at most
    /// `xtol + rtol·|x|`; Newton stops once the step is that small.
    pub rtol: f64,
    /// Maximum number of iterations (default `100`).  Each iteration costs
    /// one function evaluation (plus one derivative evaluation for Newton);
    /// the bracketing methods also evaluate the two endpoints up front.
    pub max_iter: usize,
}

impl Default for RootOpts {
    fn default() -> Self {
        Self {
            xtol: 2e-12,
            rtol: 4.0 * f64::EPSILON,
            max_iter: 100,
        }
    }
}

fn check_root_opts(op: &'static str, opts: &RootOpts) -> Result<(), SymplexError> {
    let bad = |t: f64| t < 0.0 || !t.is_finite();
    if bad(opts.xtol) || bad(opts.rtol) {
        return Err(invalid(
            op,
            format!(
                "tolerances must be finite and non-negative, got xtol = {}, rtol = {}",
                opts.xtol, opts.rtol
            ),
        ));
    }
    Ok(())
}

/// Validate a root bracket.  `Ok(Some(x))` when an endpoint is an exact
/// zero, `Ok(None)` for a proper sign change.
fn check_bracket(
    op: &'static str,
    a: f64,
    b: f64,
    fa: f64,
    fb: f64,
) -> Result<Option<f64>, SymplexError> {
    if !fa.is_finite() || !fb.is_finite() {
        return Err(invalid(
            op,
            format!(
                "function is not finite at the bracket endpoints: f({a}) = {fa}, f({b}) = {fb}"
            ),
        ));
    }
    if fa == 0.0 {
        return Ok(Some(a));
    }
    if fb == 0.0 {
        return Ok(Some(b));
    }
    if (fa > 0.0) == (fb > 0.0) {
        return Err(invalid(
            op,
            format!("f(a) and f(b) must have opposite signs: f({a}) = {fa}, f({b}) = {fb}"),
        ));
    }
    Ok(None)
}

/// Find a root of `f` in the bracket `[a, b]` by the Brent–Dekker method.
///
/// Each step chooses between inverse quadratic interpolation, the secant
/// step and bisection, so convergence is superlinear on smooth functions
/// while never being slower than bisection.  The bracket must satisfy
/// `f(a)·f(b) < 0`; an exact zero at an endpoint is returned immediately.
/// The result is within `xtol + rtol·|x|` of a sign change of `f`.
///
/// # Errors
///
/// * [`SymplexError::InvalidArgument`] if an endpoint or the function value
///   there is not finite, if `f(a)` and `f(b)` have the same sign, or if the
///   tolerances are negative.
/// * [`SymplexError::ComputationFailed`] if `f` returns a non-finite value
///   inside the bracket or the tolerance is not met within
///   [`RootOpts::max_iter`] iterations.
///
/// # Examples
///
/// ```
/// use symplex::optimize::{brent_root, RootOpts};
///
/// let root = brent_root(|x| x.cos() - x, 0.0, 1.0, &RootOpts::default()).unwrap();
/// assert!((root.cos() - root).abs() < 1e-12);
///
/// // No sign change → error, not a bogus answer.
/// assert!(brent_root(|x| x * x + 1.0, -1.0, 1.0, &RootOpts::default()).is_err());
/// ```
pub fn brent_root(
    f: impl Fn(f64) -> f64,
    a: f64,
    b: f64,
    opts: &RootOpts,
) -> Result<f64, SymplexError> {
    const OP: &str = "brent_root";
    check_root_opts(OP, opts)?;
    check_endpoints(OP, a, b)?;
    let (mut a, mut b) = (a, b);
    let (mut fa, mut fb) = (f(a), f(b));
    if let Some(root) = check_bracket(OP, a, b, fa, fb)? {
        return Ok(root);
    }
    // Invariant: `b` is the best iterate, `c` brackets the root with `b`,
    // `a` is the previous iterate; `d` is the last step, `e` the one before.
    let mut c = a;
    let mut fc = fa;
    let mut d = b - a;
    let mut e = d;
    for _ in 0..opts.max_iter {
        if (fb > 0.0) == (fc > 0.0) {
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }
        if fc.abs() < fb.abs() {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }
        let tol1 = 0.5 * (opts.xtol + opts.rtol * b.abs());
        let xm = 0.5 * (c - b);
        if xm.abs() <= tol1 || fb == 0.0 {
            return Ok(b);
        }
        if e.abs() >= tol1 && fa.abs() > fb.abs() {
            // Try interpolation: secant if only two points, else inverse
            // quadratic through (a, fa), (b, fb), (c, fc).
            let s = fb / fa;
            let (mut p, mut q) = if a == c {
                (2.0 * xm * s, 1.0 - s)
            } else {
                let q = fa / fc;
                let r = fb / fc;
                (
                    s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0)),
                    (q - 1.0) * (r - 1.0) * (s - 1.0),
                )
            };
            if p > 0.0 {
                q = -q;
            }
            p = p.abs();
            let min1 = 3.0 * xm * q - (tol1 * q).abs();
            let min2 = (e * q).abs();
            if 2.0 * p < min1.min(min2) {
                e = d;
                d = p / q;
            } else {
                d = xm;
                e = d;
            }
        } else {
            d = xm;
            e = d;
        }
        a = b;
        fa = fb;
        b += if d.abs() > tol1 { d } else { tol1.copysign(xm) };
        fb = f(b);
        if !fb.is_finite() {
            return Err(failed(OP, format!("f({b}) = {fb} is not finite")));
        }
    }
    Err(failed(
        OP,
        format!(
            "did not converge within {} iterations; bracket [{}, {}] has width {:.3e}",
            opts.max_iter,
            b.min(c),
            b.max(c),
            (c - b).abs()
        ),
    ))
}

/// Find a root of `f` in the bracket `[a, b]` by bisection.
///
/// Linear convergence (one bit per iteration), but bullet-proof: only the
/// sign of `f` is used.  The same bracket rules and error conditions as
/// [`brent_root`] apply.  With the default `max_iter = 100` the method can
/// resolve any bracket down to the default tolerance.
///
/// # Examples
///
/// ```
/// use symplex::optimize::{bisect, RootOpts};
///
/// let r = bisect(|x| x * x - 2.0, 0.0, 2.0, &RootOpts::default()).unwrap();
/// assert!((r - 2f64.sqrt()).abs() < 1e-11);
/// ```
pub fn bisect(
    f: impl Fn(f64) -> f64,
    a: f64,
    b: f64,
    opts: &RootOpts,
) -> Result<f64, SymplexError> {
    const OP: &str = "bisect";
    check_root_opts(OP, opts)?;
    check_endpoints(OP, a, b)?;
    let (fa, fb) = (f(a), f(b));
    if let Some(root) = check_bracket(OP, a, b, fa, fb)? {
        return Ok(root);
    }
    let (mut lo, mut hi, mut flo) = (a, b, fa);
    for _ in 0..opts.max_iter {
        let mid = lo + 0.5 * (hi - lo);
        let fm = f(mid);
        if !fm.is_finite() {
            return Err(failed(OP, format!("f({mid}) = {fm} is not finite")));
        }
        if fm == 0.0 {
            return Ok(mid);
        }
        if (fm > 0.0) == (flo > 0.0) {
            lo = mid;
            flo = fm;
        } else {
            hi = mid;
        }
        let mid = lo + 0.5 * (hi - lo);
        if (hi - lo).abs() <= opts.xtol + opts.rtol * mid.abs() {
            return Ok(mid);
        }
    }
    Err(failed(
        OP,
        format!(
            "did not converge within {} iterations; bracket [{}, {}] has width {:.3e}",
            opts.max_iter,
            lo.min(hi),
            lo.max(hi),
            (hi - lo).abs()
        ),
    ))
}

/// Newton's method for a root of `f` starting from `x0`, using the
/// derivative `df`.
///
/// Stops when the Newton step is smaller than `xtol + rtol·|x|` or `f(x)`
/// is exactly zero.  Divergence is detected and reported instead of
/// looping: a non-finite iterate or function value, a vanishing (or
/// non-finite) derivative, and exhaustion of the iteration budget all
/// yield [`SymplexError::ComputationFailed`].
///
/// # Errors
///
/// * [`SymplexError::InvalidArgument`] if `x0` is not finite or the
///   tolerances are negative.
/// * [`SymplexError::ComputationFailed`] on divergence or non-convergence.
///
/// # Examples
///
/// ```
/// use symplex::optimize::{newton_root, RootOpts};
///
/// let r = newton_root(|x| x * x * x - 2.0, |x| 3.0 * x * x, 1.0, &RootOpts::default()).unwrap();
/// assert!((r - 2f64.cbrt()).abs() < 1e-12);
///
/// // atan(x) from x₀ = 2 diverges: the iterates blow up and the call fails cleanly.
/// let d = newton_root(f64::atan, |x| 1.0 / (1.0 + x * x), 2.0, &RootOpts::default());
/// assert!(d.is_err());
/// ```
pub fn newton_root(
    f: impl Fn(f64) -> f64,
    df: impl Fn(f64) -> f64,
    x0: f64,
    opts: &RootOpts,
) -> Result<f64, SymplexError> {
    const OP: &str = "newton_root";
    check_root_opts(OP, opts)?;
    if !x0.is_finite() {
        return Err(invalid(
            OP,
            format!("initial guess must be finite, got {x0}"),
        ));
    }
    let mut x = x0;
    for _ in 0..opts.max_iter {
        let fx = f(x);
        if !fx.is_finite() {
            return Err(failed(
                OP,
                format!("f({x}) = {fx} is not finite; the iteration diverged"),
            ));
        }
        if fx == 0.0 {
            return Ok(x);
        }
        let dfx = df(x);
        if !dfx.is_finite() || dfx == 0.0 {
            return Err(failed(
                OP,
                format!("derivative f'({x}) = {dfx} vanishes or is not finite"),
            ));
        }
        let step = fx / dfx;
        let x_new = x - step;
        if !x_new.is_finite() {
            return Err(failed(
                OP,
                format!("iterate became non-finite after the step {step:e} from x = {x}"),
            ));
        }
        if step.abs() <= opts.xtol + opts.rtol * x_new.abs() {
            return Ok(x_new);
        }
        x = x_new;
    }
    Err(failed(
        OP,
        format!(
            "did not converge within {} iterations; last iterate x = {x}, |f(x)| = {:.3e}",
            opts.max_iter,
            f(x).abs()
        ),
    ))
}

// ═══════════════════════════════════════════════════════════════════════════
// Minimisation
// ═══════════════════════════════════════════════════════════════════════════

/// Options for [`nelder_mead`], [`minimize_scalar`] and [`golden_section`].
///
/// ```
/// use symplex::optimize::MinimizeOpts;
///
/// let opts = MinimizeOpts::default();
/// assert_eq!(opts.xtol, 1e-8);
/// assert_eq!(opts.ftol, 1e-12);
/// assert_eq!(opts.max_iter, 0);       // automatic: 200·n
/// assert_eq!(opts.initial_step, 0.0); // automatic: 5 % of |x₀ᵢ|, or 0.00025
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct MinimizeOpts {
    /// Absolute tolerance on the location of the minimum (default `1e-8`).
    ///
    /// Nelder–Mead stops when every simplex vertex is within `xtol` of the
    /// best one (in the max norm) *and* the `ftol` criterion holds.  The
    /// bracketed scalar minimisers stop when the bracket has shrunk to
    /// `xtol + √ε·|x|`; asking for more than `√ε·|x|` is pointless because
    /// the objective is flat to rounding on that scale.
    pub xtol: f64,
    /// Absolute tolerance on the objective value (default `1e-12`).
    ///
    /// Nelder–Mead requires every vertex value to be within `ftol` of the
    /// best one.  Not used by the bracketed scalar minimisers.
    pub ftol: f64,
    /// Maximum number of iterations.  `0` (the default) selects `200·n`,
    /// where `n` is the number of variables.
    ///
    /// Reaching the budget is not an error for [`nelder_mead`] — the result
    /// carries [`MinimizeResult::converged`]` == false` — but it is for the
    /// scalar minimisers, which have no way to report partial success.
    pub max_iter: usize,
    /// Nelder–Mead initial simplex edge length.  `0.0` (the default) uses
    /// the SciPy convention: vertex `i` perturbs coordinate `i` of `x0` by
    /// 5 % of its value, or by `0.00025` when that coordinate is zero.  A
    /// positive value is used as an absolute perturbation for every
    /// coordinate.  Ignored by the scalar minimisers.
    pub initial_step: f64,
}

impl Default for MinimizeOpts {
    fn default() -> Self {
        Self {
            xtol: 1e-8,
            ftol: 1e-12,
            max_iter: 0,
            initial_step: 0.0,
        }
    }
}

impl MinimizeOpts {
    fn effective_max_iter(&self, n: usize) -> usize {
        if self.max_iter == 0 {
            200usize.saturating_mul(n)
        } else {
            self.max_iter
        }
    }
}

fn check_minimize_opts(op: &'static str, opts: &MinimizeOpts) -> Result<(), SymplexError> {
    let bad = |t: f64| t < 0.0 || !t.is_finite();
    if bad(opts.xtol) || bad(opts.ftol) || bad(opts.initial_step) {
        return Err(invalid(
            op,
            format!(
                "xtol, ftol and initial_step must be finite and non-negative, got {}, {}, {}",
                opts.xtol, opts.ftol, opts.initial_step
            ),
        ));
    }
    Ok(())
}

/// Outcome of a multivariate minimisation ([`nelder_mead`],
/// [`differential_evolution`] and the `Ex` wrappers).
#[derive(Clone, Debug, PartialEq)]
pub struct MinimizeResult {
    /// Location of the best point found.
    pub x: Vec<f64>,
    /// Objective value at [`x`](Self::x).
    pub fun: f64,
    /// Iterations performed: Nelder–Mead steps, or generations for
    /// differential evolution.
    pub iterations: usize,
    /// Total number of objective evaluations (including any polishing).
    pub evaluations: usize,
    /// `true` if the stopping criterion was met before the iteration budget
    /// ran out.  When `false`, `x` is still the best point seen.
    pub converged: bool,
}

/// Minimise `f` by the Nelder–Mead downhill-simplex method starting from `x0`.
///
/// Uses the standard reflect / expand / contract / shrink steps.  For `n ≤ 2`
/// the classic coefficients `(1, 2, ½, ½)` are used; for `n > 2` the
/// dimension-adaptive coefficients `(1, 1 + 2/n, ¾ − 1/(2n), 1 − 1/n)`
/// are used, which markedly improve behaviour in higher dimensions.  The
/// iteration stops when all vertices are within [`MinimizeOpts::xtol`] of
/// the best vertex and all objective values within [`MinimizeOpts::ftol`]
/// of the best value.  `NaN` objective values are treated as `+∞`, so the
/// simplex simply moves away from regions where `f` is undefined.
///
/// Exhausting [`MinimizeOpts::max_iter`] is **not** an error: the best
/// vertex is returned with [`MinimizeResult::converged`]` == false`.
///
/// # Errors
///
/// * [`SymplexError::InvalidArgument`] if `x0` is empty or contains a
///   non-finite entry, if `f(x0)` is not finite, or if the options are
///   negative.
/// * [`SymplexError::ComputationFailed`] if `f` returns `−∞` (the objective
///   is unbounded below).
///
/// # Examples
///
/// ```
/// use symplex::optimize::{nelder_mead, MinimizeOpts};
///
/// // Rosenbrock's banana function; minimum f = 0 at (1, 1).
/// let rosen = |p: &[f64]| (1.0 - p[0]).powi(2) + 100.0 * (p[1] - p[0] * p[0]).powi(2);
/// let opts = MinimizeOpts { max_iter: 2000, ..MinimizeOpts::default() };
/// let r = nelder_mead(rosen, &[-1.2, 1.0], &opts).unwrap();
/// assert!(r.converged);
/// assert!((r.x[0] - 1.0).abs() < 1e-4 && (r.x[1] - 1.0).abs() < 1e-4);
/// assert!(r.fun < 1e-8);
/// ```
pub fn nelder_mead(
    mut f: impl FnMut(&[f64]) -> f64,
    x0: &[f64],
    opts: &MinimizeOpts,
) -> Result<MinimizeResult, SymplexError> {
    const OP: &str = "nelder_mead";
    let n = x0.len();
    if n == 0 {
        return Err(invalid(OP, "initial point must not be empty".into()));
    }
    if x0.iter().any(|v| !v.is_finite()) {
        return Err(invalid(
            OP,
            format!("initial point must be finite, got {x0:?}"),
        ));
    }
    check_minimize_opts(OP, opts)?;
    let max_iter = opts.effective_max_iter(n);

    let mut evaluations = 0usize;
    let mut eval = |x: &[f64]| -> f64 {
        evaluations += 1;
        nan_to_inf(f(x))
    };

    let f0 = eval(x0);
    if !f0.is_finite() {
        return Err(invalid(
            OP,
            format!("f(x0) = {f0} is not finite at x0 = {x0:?}"),
        ));
    }

    // Initial simplex: x0 plus one perturbed vertex per coordinate.
    let mut vertices: Vec<(Vec<f64>, f64)> = Vec::with_capacity(n + 1);
    vertices.push((x0.to_vec(), f0));
    for i in 0..n {
        let mut p = x0.to_vec();
        p[i] = if opts.initial_step > 0.0 {
            p[i] + opts.initial_step
        } else if p[i] != 0.0 {
            p[i] * 1.05
        } else {
            0.000_25
        };
        let fp = eval(&p);
        vertices.push((p, fp));
    }

    let nf = n as f64;
    let (rho, chi, psi, sigma) = if n > 2 {
        (1.0, 1.0 + 2.0 / nf, 0.75 - 0.5 / nf, 1.0 - 1.0 / nf)
    } else {
        (1.0, 2.0, 0.5, 0.5)
    };

    // `(1 + t)·xbar − t·xw`: the point on the line through the centroid and
    // the worst vertex, parameterised so that t = rho is the reflection.
    let along = |xbar: &[f64], xw: &[f64], t: f64| -> Vec<f64> {
        xbar.iter()
            .zip(xw)
            .map(|(c, w)| (1.0 + t) * c - t * w)
            .collect()
    };

    let mut iterations = 0usize;
    let mut converged = false;
    loop {
        vertices.sort_by(|a, b| a.1.total_cmp(&b.1));
        if vertices[0].1 == f64::NEG_INFINITY {
            return Err(failed(
                OP,
                format!(
                    "objective is unbounded below: f = -inf at {:?}",
                    vertices[0].0
                ),
            ));
        }
        if simplex_converged(&vertices, opts.xtol, opts.ftol) {
            converged = true;
            break;
        }
        if iterations >= max_iter {
            break;
        }
        iterations += 1;

        let mut xbar = vec![0.0; n];
        for (v, _) in &vertices[..n] {
            for (c, xi) in xbar.iter_mut().zip(v) {
                *c += xi;
            }
        }
        for c in &mut xbar {
            *c /= nf;
        }
        let xw = vertices[n].0.clone();
        let fw = vertices[n].1;
        let f_best = vertices[0].1;
        let f_second_worst = vertices[n - 1].1;

        let xr = along(&xbar, &xw, rho);
        let fr = eval(&xr);
        if fr < f_best {
            let xe = along(&xbar, &xw, rho * chi);
            let fe = eval(&xe);
            vertices[n] = if fe < fr { (xe, fe) } else { (xr, fr) };
        } else if fr < f_second_worst {
            vertices[n] = (xr, fr);
        } else {
            let mut shrink = false;
            if fr < fw {
                // Outside contraction.
                let xc = along(&xbar, &xw, psi * rho);
                let fc = eval(&xc);
                if fc <= fr {
                    vertices[n] = (xc, fc);
                } else {
                    shrink = true;
                }
            } else {
                // Inside contraction.
                let xcc = along(&xbar, &xw, -psi);
                let fcc = eval(&xcc);
                if fcc < fw {
                    vertices[n] = (xcc, fcc);
                } else {
                    shrink = true;
                }
            }
            if shrink {
                let best = vertices[0].0.clone();
                for (v, fv) in vertices.iter_mut().skip(1) {
                    for (xj, bj) in v.iter_mut().zip(&best) {
                        *xj = bj + sigma * (*xj - bj);
                    }
                    *fv = eval(v);
                }
            }
        }
    }

    let (x, fun) = vertices.swap_remove(0);
    Ok(MinimizeResult {
        x,
        fun,
        iterations,
        evaluations,
        converged,
    })
}

/// Nelder–Mead stopping test on a simplex sorted by objective value.
fn simplex_converged(vertices: &[(Vec<f64>, f64)], xtol: f64, ftol: f64) -> bool {
    let Some(((x0, f0), rest)) = vertices.split_first() else {
        return true;
    };
    let dx = rest
        .iter()
        .flat_map(|(x, _)| x.iter().zip(x0).map(|(a, b)| (a - b).abs()))
        .fold(0.0_f64, f64::max);
    let df = rest
        .iter()
        .map(|(_, fv)| (fv - f0).abs())
        .fold(0.0_f64, f64::max);
    dx <= xtol && df <= ftol
}

/// Evaluate `f(x)` for a scalar minimiser, rejecting non-finite values.
fn eval_finite(op: &'static str, f: &impl Fn(f64) -> f64, x: f64) -> Result<f64, SymplexError> {
    let v = f(x);
    if v.is_finite() {
        Ok(v)
    } else {
        Err(failed(op, format!("f({x}) = {v} is not finite")))
    }
}

/// Minimise a scalar function on `[a, b]` by Brent's method.
///
/// Combines golden-section steps with successive parabolic interpolation
/// (Brent's `localmin`), giving superlinear convergence on smooth functions
/// and golden-section behaviour otherwise.  Returns `(x_min, f_min)`.  On a
/// bracket containing several local minima the method converges to one of
/// them; which one depends on the bracket.  A reversed interval is
/// accepted.
///
/// # Errors
///
/// * [`SymplexError::InvalidArgument`] if an endpoint is not finite, the
///   interval has zero width, or the options are negative.
/// * [`SymplexError::ComputationFailed`] if `f` returns a non-finite value
///   or the tolerance is not met within the iteration budget (default
///   `200`).
///
/// # Examples
///
/// ```
/// use symplex::optimize::{minimize_scalar, MinimizeOpts};
///
/// let (x, fx) = minimize_scalar(|x| (x - 1.0).powi(2) + 3.0, -5.0, 5.0, &MinimizeOpts::default()).unwrap();
/// assert!((x - 1.0).abs() < 1e-6);
/// assert!((fx - 3.0).abs() < 1e-12);
/// ```
pub fn minimize_scalar(
    f: impl Fn(f64) -> f64,
    a: f64,
    b: f64,
    opts: &MinimizeOpts,
) -> Result<(f64, f64), SymplexError> {
    const OP: &str = "minimize_scalar";
    let (mut a, mut b) = check_interval(OP, a, b)?;
    check_minimize_opts(OP, opts)?;
    let max_iter = opts.effective_max_iter(1);
    let cgold = 0.5 * (3.0 - 5.0_f64.sqrt());
    let sqrt_eps = f64::EPSILON.sqrt();

    // x: best point; w: second best; v: previous w.
    let mut x = a + cgold * (b - a);
    let mut w = x;
    let mut v = x;
    let mut fx = eval_finite(OP, &f, x)?;
    let mut fw = fx;
    let mut fv = fx;
    let mut d = 0.0_f64; // last step
    let mut e = 0.0_f64; // step before last

    for _ in 0..max_iter {
        let xm = 0.5 * (a + b);
        let tol1 = sqrt_eps * x.abs() + opts.xtol / 3.0;
        let tol2 = 2.0 * tol1;
        if (x - xm).abs() <= tol2 - 0.5 * (b - a) {
            return Ok((x, fx));
        }
        let golden = if e.abs() > tol1 {
            // Parabola through (x, fx), (v, fv), (w, fw).
            let r = (x - w) * (fx - fv);
            let mut q = (x - v) * (fx - fw);
            let mut p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            }
            q = q.abs();
            let e_prev = e;
            e = d;
            if p.abs() >= (0.5 * q * e_prev).abs() || p <= q * (a - x) || p >= q * (b - x) {
                true
            } else {
                d = p / q;
                let u = x + d;
                if u - a < tol2 || b - u < tol2 {
                    d = tol1.copysign(xm - x);
                }
                false
            }
        } else {
            true
        };
        if golden {
            e = if x >= xm { a - x } else { b - x };
            d = cgold * e;
        }
        let u = if d.abs() >= tol1 {
            x + d
        } else {
            x + tol1.copysign(d)
        };
        let fu = eval_finite(OP, &f, u)?;
        if fu <= fx {
            if u >= x {
                a = x;
            } else {
                b = x;
            }
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || w == x {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        }
    }
    Err(failed(
        OP,
        format!("did not converge within {max_iter} iterations; bracket [{a}, {b}], best x = {x}"),
    ))
}

/// Minimise a scalar function on `[a, b]` by golden-section search.
///
/// Shrinks the bracket by the golden ratio each iteration using only
/// function comparisons; linear convergence, but immune to the parabolic
/// mis-steps of [`minimize_scalar`] on badly behaved functions.  Returns
/// `(x_min, f_min)`.  Same argument rules and errors as
/// [`minimize_scalar`].
///
/// # Examples
///
/// ```
/// use symplex::optimize::{golden_section, MinimizeOpts};
///
/// // x·ln x has its minimum at x = 1/e.
/// let (x, fx) = golden_section(|x| x * x.ln(), 0.1, 2.0, &MinimizeOpts::default()).unwrap();
/// assert!((x - (-1.0f64).exp()).abs() < 1e-6);
/// assert!((fx + (-1.0f64).exp()).abs() < 1e-12);
/// ```
pub fn golden_section(
    f: impl Fn(f64) -> f64,
    a: f64,
    b: f64,
    opts: &MinimizeOpts,
) -> Result<(f64, f64), SymplexError> {
    const OP: &str = "golden_section";
    let (mut a, mut b) = check_interval(OP, a, b)?;
    check_minimize_opts(OP, opts)?;
    let max_iter = opts.effective_max_iter(1);
    let inv_phi = 0.5 * (5.0_f64.sqrt() - 1.0);
    let sqrt_eps = f64::EPSILON.sqrt();

    let mut x1 = b - inv_phi * (b - a);
    let mut x2 = a + inv_phi * (b - a);
    let mut f1 = eval_finite(OP, &f, x1)?;
    let mut f2 = eval_finite(OP, &f, x2)?;
    for _ in 0..max_iter {
        let mid = 0.5 * (a + b);
        if (b - a).abs() <= opts.xtol + sqrt_eps * mid.abs() {
            return Ok(if f1 <= f2 { (x1, f1) } else { (x2, f2) });
        }
        if f1 < f2 {
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = b - inv_phi * (b - a);
            f1 = eval_finite(OP, &f, x1)?;
        } else {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = a + inv_phi * (b - a);
            f2 = eval_finite(OP, &f, x2)?;
        }
    }
    Err(failed(
        OP,
        format!("did not converge within {max_iter} iterations; bracket [{a}, {b}]"),
    ))
}

// ═══════════════════════════════════════════════════════════════════════════
// Differential evolution
// ═══════════════════════════════════════════════════════════════════════════

/// SplitMix64: a tiny, fast, well-distributed 64-bit generator.  Used so
/// that [`differential_evolution`] is reproducible from a `u64` seed
/// without pulling in a dependency.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)` with 53 random bits.
    fn next_f64(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / (1u64 << 53) as f64;
        (self.next_u64() >> 11) as f64 * SCALE
    }

    /// Uniform in `0..n` (`0` when `n == 0`).
    fn below(&mut self, n: usize) -> usize {
        // The modulo bias is < 2⁻⁴⁰ for any realistic population size.
        (self.next_u64() % (n as u64).max(1)) as usize
    }

    /// Uniform index in `0..n` that is not in `excluded`.
    ///
    /// `excluded` must hold distinct values `< n` and is sorted in place;
    /// the caller guarantees `excluded.len() < n`.
    fn below_excluding(&mut self, n: usize, excluded: &mut [usize]) -> usize {
        excluded.sort_unstable();
        let mut r = self.below(n.saturating_sub(excluded.len()));
        for &e in excluded.iter() {
            if r >= e {
                r += 1;
            }
        }
        r
    }

    fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i + 1);
            items.swap(i, j);
        }
    }
}

/// Options for [`differential_evolution`].
///
/// ```
/// use symplex::optimize::DeOpts;
///
/// let opts = DeOpts::default();
/// assert_eq!(opts.population, 0); // automatic: max(15·n, 8)
/// assert_eq!(opts.max_generations, 300);
/// assert_eq!(opts.crossover, 0.7);
/// assert_eq!(opts.differential_weight, 0.8);
/// assert_eq!(opts.tol, 1e-8);
/// assert_eq!(opts.seed, 0);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct DeOpts {
    /// Population size.  `0` (the default) selects `max(15·n, 8)` for `n`
    /// variables.  Explicit values must be at least `4`.
    pub population: usize,
    /// Maximum number of generations (default `300`).
    pub max_generations: usize,
    /// Crossover probability `CR ∈ [0, 1]` (default `0.7`): the chance that
    /// each coordinate of a trial vector is taken from the mutant rather
    /// than the parent.  One coordinate is always taken from the mutant.
    pub crossover: f64,
    /// Differential weight `F > 0` (default `0.8`) scaling the difference
    /// vector in `x_{r1} + F·(x_{r2} − x_{r3})`.
    pub differential_weight: f64,
    /// Convergence tolerance (default `1e-8`).  The run stops once the
    /// standard deviation of the population's objective values is at most
    /// `tol·(1 + |mean|)`.
    pub tol: f64,
    /// Seed of the internal SplitMix64 generator (default `0`).  Identical
    /// seeds and inputs give bit-identical results.
    pub seed: u64,
}

impl Default for DeOpts {
    fn default() -> Self {
        Self {
            population: 0,
            max_generations: 300,
            crossover: 0.7,
            differential_weight: 0.8,
            tol: 1e-8,
            seed: 0,
        }
    }
}

/// Population-convergence test for differential evolution.
fn population_converged(energies: &[f64], tol: f64) -> bool {
    let n = energies.len() as f64;
    if n == 0.0 {
        return true;
    }
    let mean = energies.iter().sum::<f64>() / n;
    let var = energies
        .iter()
        .map(|e| (e - mean) * (e - mean))
        .sum::<f64>()
        / n;
    var.sqrt() <= tol * (1.0 + mean.abs())
}

/// Global minimisation of `f` over the box `bounds` by differential
/// evolution (strategy `DE/rand/1/bin`), followed by a Nelder–Mead polish
/// of the best member.
///
/// The population is initialised by Latin-hypercube sampling; each
/// generation builds one trial vector per member from three other distinct
/// members (`x_{r1} + F·(x_{r2} − x_{r3})`, binomial crossover), clips it
/// to the box, and replaces the member when the trial is no worse.  Every
/// point at which `f` is evaluated — including during the polish — lies
/// inside `bounds`.  The run is deterministic for a given
/// [`DeOpts::seed`].
///
/// [`MinimizeResult::iterations`] is the number of generations,
/// [`MinimizeResult::evaluations`] counts all objective calls including the
/// polish, and [`MinimizeResult::converged`] reports whether the
/// population-spread criterion ([`DeOpts::tol`]) was met before
/// [`DeOpts::max_generations`] ran out.
///
/// # Errors
///
/// * [`SymplexError::InvalidArgument`] if `bounds` is empty, a bound is not
///   finite or reversed, the population is smaller than 4, `crossover` is
///   outside `[0, 1]`, or `differential_weight`/`tol` are not positive and
///   finite.
/// * [`SymplexError::ComputationFailed`] if `f` has no finite value
///   anywhere the search looked.
///
/// # Examples
///
/// ```
/// use symplex::optimize::{differential_evolution, DeOpts};
///
/// // Rastrigin's function: many local minima, global minimum 0 at the origin.
/// let rastrigin = |p: &[f64]| {
///     10.0 * p.len() as f64
///         + p.iter()
///             .map(|x| x * x - 10.0 * (2.0 * std::f64::consts::PI * x).cos())
///             .sum::<f64>()
/// };
/// let bounds = [(-5.12, 5.12), (-5.12, 5.12)];
/// let r = differential_evolution(rastrigin, &bounds, &DeOpts::default()).unwrap();
/// assert!(r.fun < 1e-6, "f = {}", r.fun);
/// assert!(r.x.iter().all(|x| x.abs() < 1e-3));
/// ```
pub fn differential_evolution(
    mut f: impl FnMut(&[f64]) -> f64,
    bounds: &[(f64, f64)],
    opts: &DeOpts,
) -> Result<MinimizeResult, SymplexError> {
    const OP: &str = "differential_evolution";
    let n = bounds.len();
    if n == 0 {
        return Err(invalid(OP, "bounds must not be empty".into()));
    }
    for &(lo, hi) in bounds {
        if !lo.is_finite() || !hi.is_finite() || lo > hi {
            return Err(invalid(
                OP,
                format!(
                    "each bound must be a finite (lo, hi) pair with lo <= hi, got ({lo}, {hi})"
                ),
            ));
        }
    }
    if !(0.0..=1.0).contains(&opts.crossover) {
        return Err(invalid(
            OP,
            format!("crossover must lie in [0, 1], got {}", opts.crossover),
        ));
    }
    if !opts.differential_weight.is_finite() || opts.differential_weight <= 0.0 {
        return Err(invalid(
            OP,
            format!(
                "differential_weight must be positive and finite, got {}",
                opts.differential_weight
            ),
        ));
    }
    if !opts.tol.is_finite() || opts.tol < 0.0 {
        return Err(invalid(
            OP,
            format!("tol must be finite and non-negative, got {}", opts.tol),
        ));
    }
    let np = if opts.population == 0 {
        (15 * n).max(8)
    } else {
        opts.population
    };
    if np < 4 {
        return Err(invalid(
            OP,
            format!("population must be at least 4, got {np}"),
        ));
    }

    let mut rng = SplitMix64::new(opts.seed);
    let mut evaluations = 0usize;
    let mut eval = |x: &[f64]| -> f64 {
        evaluations += 1;
        nan_to_inf(f(x))
    };

    // Latin-hypercube initialisation: every coordinate is stratified into
    // `np` equal slices, each used exactly once.
    let mut pop = vec![vec![0.0; n]; np];
    let mut perm: Vec<usize> = (0..np).collect();
    for (j, &(lo, hi)) in bounds.iter().enumerate() {
        rng.shuffle(&mut perm);
        for (member, &slice) in pop.iter_mut().zip(&perm) {
            let u = (slice as f64 + rng.next_f64()) / np as f64;
            member[j] = lo + u * (hi - lo);
        }
    }
    let mut energies: Vec<f64> = pop.iter().map(|m| eval(m)).collect();
    let mut best = argmin(&energies);

    let mut generations = 0usize;
    let mut converged = false;
    let mut trial = vec![0.0; n];
    loop {
        if population_converged(&energies, opts.tol) {
            converged = true;
            break;
        }
        if generations >= opts.max_generations {
            break;
        }
        generations += 1;
        for i in 0..np {
            let r1 = rng.below_excluding(np, &mut [i]);
            let r2 = rng.below_excluding(np, &mut [i, r1]);
            let r3 = rng.below_excluding(np, &mut [i, r1, r2]);
            let j_rand = rng.below(n);
            for (j, &(lo, hi)) in bounds.iter().enumerate() {
                let v = if j == j_rand || rng.next_f64() < opts.crossover {
                    pop[r1][j] + opts.differential_weight * (pop[r2][j] - pop[r3][j])
                } else {
                    pop[i][j]
                };
                // Bounds were validated finite with lo <= hi, so clamp cannot panic.
                trial[j] = v.clamp(lo, hi);
            }
            let ft = eval(&trial);
            if ft <= energies[i] {
                pop[i].copy_from_slice(&trial);
                energies[i] = ft;
                if ft < energies[best] {
                    best = i;
                }
            }
        }
    }
    let de_evaluations = evaluations;

    let f_best = energies[best];
    if !f_best.is_finite() {
        return Err(failed(
            OP,
            format!("objective has no finite value in the box after {generations} generations"),
        ));
    }

    // Local polish, evaluating only inside the box.
    let mut clipped = vec![0.0; n];
    let polish = nelder_mead(
        |x: &[f64]| {
            for ((c, &xi), &(lo, hi)) in clipped.iter_mut().zip(x).zip(bounds) {
                *c = xi.clamp(lo, hi);
            }
            f(&clipped)
        },
        &pop[best],
        &MinimizeOpts::default(),
    )
    .map_err(|e| failed(OP, format!("Nelder–Mead polish failed: {e}")))?;

    let (x, fun) = if polish.fun < f_best {
        let x = polish
            .x
            .iter()
            .zip(bounds)
            .map(|(&xi, &(lo, hi))| xi.clamp(lo, hi))
            .collect();
        (x, polish.fun)
    } else {
        (pop.swap_remove(best), f_best)
    };
    Ok(MinimizeResult {
        x,
        fun,
        iterations: generations,
        evaluations: de_evaluations + polish.evaluations,
        converged,
    })
}

/// Index of the smallest value (`0` for an empty slice).
fn argmin(values: &[f64]) -> usize {
    values.iter().enumerate().fold(
        0usize,
        |best, (i, &v)| {
            if v < values[best] { i } else { best }
        },
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// Least-squares fitting
// ═══════════════════════════════════════════════════════════════════════════

/// Least-squares polynomial fit of degree `degree` to the samples
/// `(xs[i], ys[i])`.
///
/// Returns the coefficients in **ascending** degree, `[c₀, c₁, …, c_d]`,
/// so that `ys[i] ≈ c₀ + c₁·xs[i] + … + c_d·xs[i]^d` (evaluate with
/// [`eval_poly`]).  The Vandermonde matrix is column-scaled and factored
/// by Householder QR, which is backward stable; the normal equations are
/// never formed.  With `degree + 1 == xs.len()` the fit interpolates.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] if the slices differ in length,
/// `degree >= xs.len()`, or any sample is not finite;
/// [`SymplexError::ComputationFailed`] if the Vandermonde matrix is
/// numerically rank deficient (fewer than `degree + 1` distinct
/// abscissae).
///
/// # Examples
///
/// ```
/// use symplex::optimize::{eval_poly, poly_fit};
///
/// let xs: Vec<f64> = (0..6).map(f64::from).collect();
/// let ys: Vec<f64> = xs.iter().map(|x| 1.0 + 2.0 * x + 3.0 * x * x).collect();
/// let c = poly_fit(&xs, &ys, 2).unwrap();
/// assert!((c[0] - 1.0).abs() < 1e-9 && (c[1] - 2.0).abs() < 1e-9 && (c[2] - 3.0).abs() < 1e-9);
/// assert!((eval_poly(&c, 10.0) - 321.0).abs() < 1e-7);
///
/// // Not enough points for the requested degree.
/// assert!(poly_fit(&[0.0, 1.0], &[0.0, 1.0], 2).is_err());
/// ```
pub fn poly_fit(xs: &[f64], ys: &[f64], degree: usize) -> Result<Vec<f64>, SymplexError> {
    const OP: &str = "poly_fit";
    let m = xs.len();
    if m != ys.len() {
        return Err(invalid(
            OP,
            format!(
                "xs and ys must have the same length, got {m} and {}",
                ys.len()
            ),
        ));
    }
    if degree >= m {
        return Err(invalid(
            OP,
            format!(
                "degree {degree} needs at least {} points, got {m}",
                degree + 1
            ),
        ));
    }
    if xs.iter().chain(ys).any(|v| !v.is_finite()) {
        return Err(invalid(OP, "all samples must be finite".into()));
    }
    let ncols = degree + 1;
    let mut a: Vec<Vec<f64>> = xs
        .iter()
        .map(|&x| {
            let mut p = 1.0;
            (0..ncols)
                .map(|_| {
                    let v = p;
                    p *= x;
                    v
                })
                .collect()
        })
        .collect();
    // Equilibrate the columns: brings the condition number within a
    // modest factor of the best diagonal scaling.
    let mut scale = vec![1.0; ncols];
    for (j, s) in scale.iter_mut().enumerate() {
        let norm = a.iter().map(|row| row[j] * row[j]).sum::<f64>().sqrt();
        if norm > 0.0 && norm.is_finite() {
            *s = norm;
            for row in &mut a {
                row[j] /= norm;
            }
        }
    }
    let c = lstsq_householder(a, ys.to_vec(), ncols).ok_or_else(|| {
        failed(
            OP,
            format!("Vandermonde matrix is rank deficient: fewer than {ncols} distinct abscissae"),
        )
    })?;
    Ok(c.iter().zip(&scale).map(|(c, s)| c / s).collect())
}

/// Least-squares solution of the overdetermined system `a·x = b`
/// (`a` is `m × n` with `m ≥ n`) via Householder QR.  `None` if `a` is
/// numerically rank deficient.
fn lstsq_householder(mut a: Vec<Vec<f64>>, mut b: Vec<f64>, n: usize) -> Option<Vec<f64>> {
    let m = a.len();
    if m < n || b.len() != m {
        return None;
    }
    for k in 0..n {
        let norm = (k..m).map(|i| a[i][k] * a[i][k]).sum::<f64>().sqrt();
        if norm == 0.0 || !norm.is_finite() {
            return None;
        }
        // Householder vector v = x − α·e₁ with α chosen to avoid cancellation.
        let alpha = if a[k][k] > 0.0 { -norm } else { norm };
        let mut v: Vec<f64> = (k..m).map(|i| a[i][k]).collect();
        v[0] -= alpha;
        let vnorm2: f64 = v.iter().map(|x| x * x).sum();
        if vnorm2 == 0.0 {
            continue;
        }
        // Apply H = I − 2vvᵀ/‖v‖² to the trailing block of `a` and to `b`:
        // A ← A − (2/‖v‖²)·v·(vᵀA).
        let scale = 2.0 / vnorm2;
        let w: Vec<f64> = (k..n)
            .map(|j| v.iter().zip(k..m).map(|(vi, i)| vi * a[i][j]).sum::<f64>())
            .collect();
        for (vi, i) in v.iter().zip(k..m) {
            for (wj, entry) in w.iter().zip(a[i][k..].iter_mut()) {
                *entry -= scale * vi * wj;
            }
        }
        let s: f64 = v.iter().zip(k..m).map(|(vi, i)| vi * b[i]).sum();
        let factor = scale * s;
        for (vi, i) in v.iter().zip(k..m) {
            b[i] -= factor * vi;
        }
    }
    // Back-substitution on the leading n × n block (R).
    let r_max = (0..n).map(|k| a[k][k].abs()).fold(0.0_f64, f64::max);
    let threshold = r_max * f64::EPSILON * m as f64;
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let diag = a[r][r];
        if !diag.is_finite() || diag.abs() <= threshold {
            return None;
        }
        let s = b[r] - ((r + 1)..n).map(|c| a[r][c] * x[c]).sum::<f64>();
        x[r] = s / diag;
    }
    Some(x)
}

/// Exact least-squares polynomial fit over ℚ.
///
/// Solves the normal equations `AᵀA·c = Aᵀy` for the Vandermonde matrix
/// `A` with exact rational Gaussian elimination, so the returned
/// coefficients (in **ascending** degree) are the exact least-squares
/// solution — for consistent data, the exact interpolating polynomial.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] if `degree >= points.len()`;
/// [`SymplexError::ComputationFailed`] if the normal matrix is singular
/// (fewer than `degree + 1` distinct abscissae).
///
/// # Examples
///
/// ```
/// use num_bigint::BigInt;
/// use num_rational::Ratio;
/// use symplex::optimize::poly_fit_exact;
///
/// let q = |p: i64, d: i64| Ratio::new(BigInt::from(p), BigInt::from(d));
/// // y = x²/3 − x/2 + 1/7 sampled at x = 0, 1, 2, 3, 4 (five points, degree 2).
/// let pts = [
///     (q(0, 1), q(1, 7)),
///     (q(1, 1), q(-1, 42)),
///     (q(2, 1), q(10, 21)),
///     (q(3, 1), q(23, 14)),
///     (q(4, 1), q(73, 21)),
/// ];
/// let c = poly_fit_exact(&pts, 2).unwrap();
/// assert_eq!(c, vec![q(1, 7), q(-1, 2), q(1, 3)]);
/// ```
pub fn poly_fit_exact(
    points: &[(Ratio<BigInt>, Ratio<BigInt>)],
    degree: usize,
) -> Result<Vec<Ratio<BigInt>>, SymplexError> {
    const OP: &str = "poly_fit_exact";
    let m = points.len();
    if degree >= m {
        return Err(invalid(
            OP,
            format!(
                "degree {degree} needs at least {} points, got {m}",
                degree + 1
            ),
        ));
    }
    let ncols = degree + 1;
    // Power sums S_p = Σ xᵖ (p ≤ 2d) and moments T_j = Σ xʲ·y (j ≤ d).
    let mut power_sums = vec![Ratio::<BigInt>::zero(); 2 * degree + 1];
    let mut moments = vec![Ratio::<BigInt>::zero(); ncols];
    for (x, y) in points {
        let mut pow = Ratio::<BigInt>::one();
        for (p, s) in power_sums.iter_mut().enumerate() {
            *s += &pow;
            if let Some(t) = moments.get_mut(p) {
                *t += &pow * y;
            }
            if p + 1 < 2 * degree + 1 {
                pow *= x;
            }
        }
    }
    let normal: Vec<Vec<Ratio<BigInt>>> = (0..ncols)
        .map(|j| (0..ncols).map(|k| power_sums[j + k].clone()).collect())
        .collect();
    solve_exact(normal, moments).ok_or_else(|| {
        failed(
            OP,
            format!("normal equations are singular: fewer than {ncols} distinct abscissae"),
        )
    })
}

/// Exact Gaussian elimination for the square system `a·x = b` over ℚ.
/// `None` if the matrix is singular.
fn solve_exact(
    mut a: Vec<Vec<Ratio<BigInt>>>,
    mut b: Vec<Ratio<BigInt>>,
) -> Option<Vec<Ratio<BigInt>>> {
    let n = b.len();
    if a.len() != n || a.iter().any(|row| row.len() != n) {
        return None;
    }
    for col in 0..n {
        let pivot = (col..n).find(|&r| !a[r][col].is_zero())?;
        a.swap(col, pivot);
        b.swap(col, pivot);
        let pivot_row = a[col].clone();
        let pivot_b = b[col].clone();
        for r in (col + 1)..n {
            if a[r][col].is_zero() {
                continue;
            }
            let factor = &a[r][col] / &pivot_row[col];
            for (entry, p) in a[r].iter_mut().zip(&pivot_row).skip(col) {
                *entry -= &factor * p;
            }
            b[r] -= &factor * &pivot_b;
        }
    }
    let mut x = vec![Ratio::<BigInt>::zero(); n];
    for r in (0..n).rev() {
        let mut s = b[r].clone();
        for c in (r + 1)..n {
            s -= &a[r][c] * &x[c];
        }
        x[r] = s / &a[r][r];
    }
    Some(x)
}

/// Least-squares straight line `y ≈ slope·x + intercept`.
///
/// Returns `(slope, intercept)`.  Equivalent to [`poly_fit`] with
/// `degree = 1`; needs at least two samples with distinct abscissae.
///
/// # Examples
///
/// ```
/// use symplex::optimize::linear_fit;
///
/// let xs = [0.0, 1.0, 2.0, 3.0];
/// let ys = [1.0, 4.0, 7.0, 10.0]; // y = 3x + 1
/// let (slope, intercept) = linear_fit(&xs, &ys).unwrap();
/// assert!((slope - 3.0).abs() < 1e-12 && (intercept - 1.0).abs() < 1e-12);
/// ```
pub fn linear_fit(xs: &[f64], ys: &[f64]) -> Result<(f64, f64), SymplexError> {
    let c = poly_fit(xs, ys, 1)?;
    match c.as_slice() {
        [intercept, slope] => Ok((*slope, *intercept)),
        _ => Err(failed(
            "linear_fit",
            format!("expected two coefficients, got {}", c.len()),
        )),
    }
}

/// Trapezoidal-rule integral of the samples `ys` at abscissae `xs`.
///
/// `Σ ½·(xs[i+1] − xs[i])·(ys[i] + ys[i+1])`; fewer than two samples give
/// `0`.  The abscissae need not be evenly spaced.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] if the slices differ in length.
///
/// # Examples
///
/// ```
/// use symplex::optimize::trapezoid;
///
/// let xs: Vec<f64> = (0..=1000).map(|i| i as f64 / 1000.0).collect();
/// let ys: Vec<f64> = xs.iter().map(|x| x * x).collect();
/// assert!((trapezoid(&ys, &xs).unwrap() - 1.0 / 3.0).abs() < 1e-6);
/// ```
pub fn trapezoid(ys: &[f64], xs: &[f64]) -> Result<f64, SymplexError> {
    if ys.len() != xs.len() {
        return Err(invalid(
            "trapezoid",
            format!(
                "ys and xs must have the same length, got {} and {}",
                ys.len(),
                xs.len()
            ),
        ));
    }
    Ok(xs
        .windows(2)
        .zip(ys.windows(2))
        .map(|(x, y)| 0.5 * (x[1] - x[0]) * (y[0] + y[1]))
        .sum())
}

/// Evaluate a polynomial given by **ascending** coefficients at `x`
/// (Horner's rule).
///
/// ```
/// use symplex::optimize::eval_poly;
///
/// assert_eq!(eval_poly(&[1.0, 2.0, 3.0], 2.0), 17.0); // 1 + 2·2 + 3·4
/// assert_eq!(eval_poly(&[], 5.0), 0.0);
/// ```
#[must_use]
pub fn eval_poly(coeffs_ascending: &[f64], x: f64) -> f64 {
    coeffs_ascending
        .iter()
        .rev()
        .fold(0.0, |acc, &c| acc * x + c)
}

// ═══════════════════════════════════════════════════════════════════════════
// Ex conveniences
// ═══════════════════════════════════════════════════════════════════════════

/// Validate `vars` (same context, all symbols, covering every free symbol
/// of `expr`) and compile `expr` as a function of them, in order.
fn compile_in(
    expr: &Ex,
    vars: &[&Ex],
    operation: &'static str,
) -> Result<CompiledFn, SymplexError> {
    if vars.is_empty() {
        return Err(invalid(
            operation,
            "at least one variable is required".into(),
        ));
    }
    let ids: Vec<_> = vars.iter().map(|v| expr.checked_id(*v)).collect();
    let non_symbol = {
        let inner = expr.inner.read();
        ids.iter()
            .position(|&id| !matches!(inner.arena.node(id), ExprNode::Symbol(_)))
    };
    if let Some(i) = non_symbol {
        return Err(invalid(
            operation,
            format!("variables must be symbols, got `{}`", vars[i]),
        ));
    }
    if let Some(extra) = expr.free_symbols().into_iter().find(|s| !vars.contains(&s)) {
        return Err(SymplexError::FreeSymbol {
            name: format!("{extra}"),
        });
    }
    let names: Vec<String> = vars.iter().map(|v| format!("{v}")).collect();
    let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
    expr.compile(&name_refs)
}

impl Ex {
    /// Numerically find a root of this expression in `var` inside the
    /// bracket `[a, b]` by [`brent_root`] with default [`RootOpts`].
    ///
    /// The expression is compiled with [`compile`](Self::compile) first, so
    /// evaluation is fast and the usual compile-time checks apply.
    ///
    /// # Errors
    ///
    /// * [`SymplexError::InvalidArgument`] if `var` is not a symbol, or the
    ///   bracket is invalid (non-finite, or no sign change).
    /// * [`SymplexError::FreeSymbol`] if the expression contains a symbol
    ///   other than `var`.
    /// * [`SymplexError::NotImplemented`] if the expression cannot be
    ///   compiled to `f64` arithmetic.
    /// * [`SymplexError::ComputationFailed`] if the iteration does not
    ///   converge or meets a non-finite value.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let r = (&x.powi(2) - 2).find_root_bracket(&x, 0.0, 2.0).unwrap();
    /// assert!((r - 2f64.sqrt()).abs() < 1e-12);
    ///
    /// // A transcendental equation: cos x = x.
    /// let r = (x.cos() - &x).find_root_bracket(&x, 0.0, 1.0).unwrap();
    /// assert!((r - 0.739_085_133_215_160_6).abs() < 1e-12);
    ///
    /// // Another free symbol → FreeSymbol, not a silent NaN.
    /// let a = ctx.symbol("a");
    /// assert!(matches!(
    ///     (&x.powi(2) - &a).find_root_bracket(&x, 0.0, 2.0),
    ///     Err(SymplexError::FreeSymbol { .. })
    /// ));
    /// ```
    pub fn find_root_bracket(&self, var: &Ex, a: f64, b: f64) -> Result<f64, SymplexError> {
        self.find_root_bracket_with(var, a, b, &RootOpts::default())
    }

    /// [`find_root_bracket`](Self::find_root_bracket) with explicit
    /// [`RootOpts`].
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::optimize::RootOpts;
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let opts = RootOpts { xtol: 1e-6, ..RootOpts::default() };
    /// let r = (x.exp() - 3).find_root_bracket_with(&x, 0.0, 2.0, &opts).unwrap();
    /// assert!((r - 3f64.ln()).abs() < 1e-6);
    /// ```
    pub fn find_root_bracket_with(
        &self,
        var: &Ex,
        a: f64,
        b: f64,
        opts: &RootOpts,
    ) -> Result<f64, SymplexError> {
        let f = compile_in(self, &[var], "find_root_bracket")?;
        brent_root(|x| f.call(&[x]), a, b, opts)
    }

    /// Minimise this expression numerically over `vars` from the starting
    /// point `x0` by [`nelder_mead`] with default [`MinimizeOpts`].
    ///
    /// `x0[i]` is the initial value of `vars[i]`.
    ///
    /// # Errors
    ///
    /// * [`SymplexError::InvalidArgument`] if `vars` is empty, a variable is
    ///   not a symbol, or `x0.len() != vars.len()`.
    /// * [`SymplexError::FreeSymbol`] if the expression contains a symbol
    ///   not listed in `vars`.
    /// * [`SymplexError::NotImplemented`] if the expression cannot be
    ///   compiled to `f64` arithmetic.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// let bowl = (&x - 1).powi(2) + (&y + 2).powi(2);
    /// let r = bowl.minimize_numeric(&[&x, &y], &[0.0, 0.0]).unwrap();
    /// assert!(r.converged);
    /// assert!((r.x[0] - 1.0).abs() < 1e-6 && (r.x[1] + 2.0).abs() < 1e-6);
    /// assert!(r.fun < 1e-12);
    /// ```
    pub fn minimize_numeric(
        &self,
        vars: &[&Ex],
        x0: &[f64],
    ) -> Result<MinimizeResult, SymplexError> {
        self.minimize_numeric_with(vars, x0, &MinimizeOpts::default())
    }

    /// [`minimize_numeric`](Self::minimize_numeric) with explicit
    /// [`MinimizeOpts`].
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::optimize::MinimizeOpts;
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// let rosen = (1 - &x).powi(2) + 100 * (&y - &x.powi(2)).powi(2);
    /// let opts = MinimizeOpts { max_iter: 2000, ..MinimizeOpts::default() };
    /// let r = rosen.minimize_numeric_with(&[&x, &y], &[-1.2, 1.0], &opts).unwrap();
    /// assert!((r.x[0] - 1.0).abs() < 1e-4 && (r.x[1] - 1.0).abs() < 1e-4);
    /// ```
    pub fn minimize_numeric_with(
        &self,
        vars: &[&Ex],
        x0: &[f64],
        opts: &MinimizeOpts,
    ) -> Result<MinimizeResult, SymplexError> {
        const OP: &str = "minimize_numeric";
        if x0.len() != vars.len() {
            return Err(invalid(
                OP,
                format!(
                    "initial point has {} entries, expected {}",
                    x0.len(),
                    vars.len()
                ),
            ));
        }
        let f = compile_in(self, vars, OP)?;
        nelder_mead(|x| f.call(x), x0, opts)
    }

    /// Minimise this expression in the single variable `var` over `[a, b]`
    /// by Brent's method ([`minimize_scalar`]) with default
    /// [`MinimizeOpts`].  Returns `(x_min, f_min)`.
    ///
    /// # Errors
    ///
    /// As for [`find_root_bracket`](Self::find_root_bracket) plus the
    /// interval rules of [`minimize_scalar`].
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// // x·ln x has its minimum −1/e at x = 1/e.
    /// let (xm, fm) = (&x * x.ln()).minimize_scalar_numeric(&x, 0.1, 2.0).unwrap();
    /// assert!((xm - (-1.0f64).exp()).abs() < 1e-6);
    /// assert!((fm + (-1.0f64).exp()).abs() < 1e-12);
    /// ```
    pub fn minimize_scalar_numeric(
        &self,
        var: &Ex,
        a: f64,
        b: f64,
    ) -> Result<(f64, f64), SymplexError> {
        let f = compile_in(self, &[var], "minimize_scalar_numeric")?;
        minimize_scalar(|x| f.call(&[x]), a, b, &MinimizeOpts::default())
    }

    /// Globally minimise this expression over the box `bounds` (one
    /// `(lo, hi)` pair per entry of `vars`) by
    /// [`differential_evolution`].
    ///
    /// # Errors
    ///
    /// As for [`minimize_numeric`](Self::minimize_numeric), with
    /// `bounds.len()` playing the role of `x0.len()`, plus the option and
    /// bound rules of [`differential_evolution`].
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::optimize::DeOpts;
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// // Himmelblau's function has four global minima with f = 0.
    /// let h = (&x.powi(2) + &y - 11).powi(2) + (&x + &y.powi(2) - 7).powi(2);
    /// let r = h.minimize_global_numeric(&[&x, &y], &[(-5.0, 5.0), (-5.0, 5.0)], &DeOpts::default()).unwrap();
    /// assert!(r.fun < 1e-8, "f = {}", r.fun);
    /// ```
    pub fn minimize_global_numeric(
        &self,
        vars: &[&Ex],
        bounds: &[(f64, f64)],
        opts: &DeOpts,
    ) -> Result<MinimizeResult, SymplexError> {
        const OP: &str = "minimize_global_numeric";
        if bounds.len() != vars.len() {
            return Err(invalid(
                OP,
                format!("got {} bounds for {} variables", bounds.len(), vars.len()),
            ));
        }
        let f = compile_in(self, vars, OP)?;
        differential_evolution(|x| f.call(x), bounds, opts)
    }

    /// Exact least-squares polynomial of degree `degree` in `var` through
    /// the rational points `(x, y)`.
    ///
    /// Each coordinate is constant-folded with [`eval`](Self::eval) and
    /// must then be a rational literal (`ctx.int`, `ctx.rational`,
    /// `sqrt(4)`, …).  The fit is computed by [`poly_fit_exact`], so the
    /// result is the exact least-squares polynomial — the interpolating
    /// polynomial when `degree + 1 == points.len()` or the data are
    /// consistent.
    ///
    /// # Errors
    ///
    /// * [`SymplexError::InvalidArgument`] if a coordinate is not a rational
    ///   literal after evaluation, or `degree >= points.len()`.
    /// * [`SymplexError::ComputationFailed`] if the normal equations are
    ///   singular (fewer than `degree + 1` distinct abscissae).
    ///
    /// # Panics
    ///
    /// Panics if `var` or a point belongs to a different context than
    /// `ctx` (the standard cross-context guard).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// // Five samples of x²/3 − x/2 + 1/7.
    /// let pts = [
    ///     (ctx.int(0), ctx.rational(1, 7)),
    ///     (ctx.int(1), ctx.rational(-1, 42)),
    ///     (ctx.int(2), ctx.rational(10, 21)),
    ///     (ctx.int(3), ctx.rational(23, 14)),
    ///     (ctx.int(4), ctx.rational(73, 21)),
    /// ];
    /// let p = Ex::poly_fit_points(&ctx, &pts, &x, 2).unwrap();
    /// let expected = &x.powi(2) * ctx.rational(1, 3) - &x * ctx.rational(1, 2) + ctx.rational(1, 7);
    /// assert!((&p - &expected).expand().is_zero_structural(), "{p}");
    ///
    /// // A symbolic coordinate is rejected.
    /// let a = ctx.symbol("a");
    /// assert!(Ex::poly_fit_points(&ctx, &[(ctx.int(0), a), (ctx.int(1), ctx.int(1))], &x, 1).is_err());
    /// ```
    pub fn poly_fit_points(
        ctx: &Context,
        points: &[(Ex, Ex)],
        var: &Ex,
        degree: usize,
    ) -> Result<Ex, SymplexError> {
        const OP: &str = "poly_fit_points";
        let _ = ctx.own_id(var);
        let to_ratio = |e: &Ex| -> Result<Ratio<BigInt>, SymplexError> {
            let _ = var.checked_id(e);
            e.eval().as_rational().ok_or_else(|| {
                invalid(
                    OP,
                    format!("point coordinate `{e}` is not a rational literal"),
                )
            })
        };
        let mut pts: Vec<(Ratio<BigInt>, Ratio<BigInt>)> = Vec::with_capacity(points.len());
        for (px, py) in points {
            pts.push((to_ratio(px)?, to_ratio(py)?));
        }
        let coeffs = poly_fit_exact(&pts, degree)?;
        let mut terms: Vec<Ex> = Vec::with_capacity(coeffs.len());
        for (i, c) in coeffs.into_iter().enumerate() {
            if c.is_zero() {
                continue;
            }
            let power =
                i64::try_from(i).map_err(|_| invalid(OP, format!("degree {i} is too large")))?;
            terms.push(ctx.from_ratio(c) * var.powi(power));
        }
        Ok(ctx.sum(&terms))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix_is_deterministic_and_in_range() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..100 {
            let u = a.next_f64();
            assert_eq!(u, b.next_f64());
            assert!((0.0..1.0).contains(&u));
            let k = a.below(7);
            assert_eq!(k, b.below(7));
            assert!(k < 7);
        }
        assert_eq!(SplitMix64::new(0).below(0), 0);
    }

    #[test]
    fn below_excluding_never_returns_excluded() {
        let mut rng = SplitMix64::new(7);
        for _ in 0..1000 {
            let i = rng.below(10);
            let r1 = rng.below_excluding(10, &mut [i]);
            assert_ne!(r1, i);
            let r2 = rng.below_excluding(10, &mut [i, r1]);
            assert!(r2 != i && r2 != r1);
            let r3 = rng.below_excluding(10, &mut [i, r1, r2]);
            assert!(r3 != i && r3 != r1 && r3 != r2 && r3 < 10);
        }
    }

    #[test]
    fn shuffle_is_a_permutation() {
        let mut rng = SplitMix64::new(3);
        let mut v: Vec<usize> = (0..20).collect();
        rng.shuffle(&mut v);
        let mut sorted = v.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..20).collect::<Vec<_>>());
        assert_ne!(v, sorted, "20 elements should not stay in order");
    }

    #[test]
    fn householder_solves_square_system() {
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let x = lstsq_householder(a, vec![3.0, 5.0], 2).unwrap();
        assert!((x[0] - 0.8).abs() < 1e-12 && (x[1] - 1.4).abs() < 1e-12);
    }

    #[test]
    fn householder_detects_rank_deficiency() {
        let a = vec![vec![1.0, 2.0], vec![2.0, 4.0], vec![3.0, 6.0]];
        assert!(lstsq_householder(a, vec![1.0, 2.0, 3.0], 2).is_none());
    }

    #[test]
    fn exact_solver_basic_and_singular() {
        let q = |n: i64| Ratio::from_integer(BigInt::from(n));
        let a = vec![vec![q(2), q(1)], vec![q(1), q(3)]];
        let x = solve_exact(a, vec![q(3), q(5)]).unwrap();
        assert_eq!(x[0], Ratio::new(BigInt::from(4), BigInt::from(5)));
        assert_eq!(x[1], Ratio::new(BigInt::from(7), BigInt::from(5)));
        let s = vec![vec![q(1), q(2)], vec![q(2), q(4)]];
        assert!(solve_exact(s, vec![q(1), q(2)]).is_none());
    }

    #[test]
    fn argmin_picks_first_smallest() {
        assert_eq!(argmin(&[3.0, 1.0, 1.0, 2.0]), 1);
        assert_eq!(argmin(&[]), 0);
        assert_eq!(argmin(&[f64::INFINITY, f64::INFINITY]), 0);
    }
}
