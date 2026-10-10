//! Vector calculus: gradient, divergence, curl, Laplacian (in Cartesian,
//! cylindrical and spherical coordinates), directional derivatives, line
//! integrals and scalar potentials.
//!
//! Scalar fields are [`Ex`]; vector fields are `n×1` column-vector
//! [`Matrix`] values whose components are expressed in the **coordinate
//! basis** of the chosen [`CoordinateSystem`] (e.g. `(F_r, F_φ, F_z)` for
//! cylindrical coordinates).
//!
//! # Coordinate conventions
//!
//! | System | `vars` order | Scale factors `(h₁, h₂, h₃)` |
//! |--------|--------------|------------------------------|
//! | [`Cartesian`](CoordinateSystem::Cartesian) | `(x, y, z)` (any dimension) | `(1, 1, 1)` |
//! | [`Cylindrical`](CoordinateSystem::Cylindrical) | `(r, φ, z)` | `(1, r, 1)` |
//! | [`Spherical`](CoordinateSystem::Spherical) | `(r, θ, φ)` — **physics convention**: `θ` polar angle from `+z`, `φ` azimuth | `(1, r, r·sin θ)` |
//!
//! The Cartesian functions ([`gradient`], [`divergence`], [`curl`],
//! [`laplacian`]) work in any dimension (curl: 3-D only).  The `_in`
//! variants take an explicit coordinate system; the curvilinear systems
//! are 3-D and require exactly three variables.
//!
//! # Errors
//!
//! Every operator returns a `Result`.  Wrong field dimensions / variable
//! counts (an empty variable list, a curvilinear system without exactly
//! three variables, a field that is not an `n×1` column with
//! `n == vars.len()`) are [`SymplexError::InvalidArgument`]; mathematical
//! failure (e.g. a non-conservative field passed to [`scalar_potential`])
//! is [`SymplexError::ComputationFailed`].

use crate::api::expr::Ex;
use crate::base::errors::SymplexError;
use crate::domains::matrix::{Matrix, all3, ex_is_zero};

pub use crate::domains::matrix_decomp::hessian;

/// Orthogonal coordinate systems supported by the vector-calculus operators.
///
/// See the [module documentation](self) for variable order and conventions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CoordinateSystem {
    /// Cartesian `(x, y, z, …)`; scale factors all `1`.  Any dimension.
    Cartesian,
    /// Cylindrical `(r, φ, z)`; scale factors `(1, r, 1)`.
    Cylindrical,
    /// Spherical `(r, θ, φ)` in the physics convention (`θ` polar from
    /// `+z`, `φ` azimuthal); scale factors `(1, r, r·sin θ)`.
    Spherical,
}

impl CoordinateSystem {
    /// Lamé scale factors `hᵢ` for the given variables.
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::InvalidArgument`] if `vars` is empty, or a
    /// curvilinear system is used with `vars.len() != 3`.
    pub fn scale_factors(self, vars: &[&Ex]) -> Result<Vec<Ex>, SymplexError> {
        self.scale_factors_for("CoordinateSystem::scale_factors", vars)
    }

    /// [`scale_factors`](Self::scale_factors), reporting errors as `op`.
    fn scale_factors_for(self, op: &'static str, vars: &[&Ex]) -> Result<Vec<Ex>, SymplexError> {
        let Some(&first) = vars.first() else {
            return Err(SymplexError::invalid_argument(op, "vars must be non-empty"));
        };
        let one = first.context().one();
        let need_three = |coords: &str| {
            if vars.len() == 3 {
                Ok(())
            } else {
                Err(SymplexError::invalid_argument(
                    op,
                    format!(
                        "{coords} coordinates need exactly 3 variables, got {}",
                        vars.len()
                    ),
                ))
            }
        };
        match self {
            CoordinateSystem::Cartesian => Ok(vec![one; vars.len()]),
            CoordinateSystem::Cylindrical => {
                need_three("cylindrical (r, phi, z)")?;
                Ok(vec![one.clone(), first.clone(), one])
            }
            CoordinateSystem::Spherical => {
                need_three("spherical (r, theta, phi)")?;
                Ok(vec![one, first.clone(), first * &vars[1].sin()])
            }
        }
    }
}

/// `field` must be an `n×1` column with `n == vars.len()` (so `vars` is
/// non-empty: a matrix has at least one row).
fn check_field(field: &Matrix, vars: &[&Ex], op: &'static str) -> Result<(), SymplexError> {
    if field.ncols() != 1 || field.nrows() != vars.len() {
        return Err(SymplexError::invalid_argument(
            op,
            format!(
                "field must be an n×1 column vector with n == vars.len() (got {}×{} and {} variables)",
                field.nrows(),
                field.ncols(),
                vars.len()
            ),
        ));
    }
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// Differential operators
// ═══════════════════════════════════════════════════════════════════════════

/// Gradient of a scalar field in Cartesian coordinates:
/// `∇f = [∂f/∂x₁, …, ∂f/∂xₙ]ᵀ` (an `n×1` column vector).
///
/// # Errors
///
/// Returns [`SymplexError::InvalidArgument`] if `vars` is empty.
pub fn gradient(f: &Ex, vars: &[&Ex]) -> Result<Matrix, SymplexError> {
    gradient_in(f, vars, CoordinateSystem::Cartesian)
}

/// Gradient in the given coordinate system:
/// `(∇f)ᵢ = (1/hᵢ) ∂f/∂qᵢ`.
///
/// # Errors
///
/// Returns [`SymplexError::InvalidArgument`] if `vars` is empty or a
/// curvilinear system is used with `vars.len() != 3`.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::vector::{gradient_in, CoordinateSystem};
///
/// let ctx = Context::new();
/// let (r, th, ph) = (ctx.symbol("r"), ctx.symbol("theta"), ctx.symbol("phi"));
/// // ∇(r² sin θ) = (2r sin θ, r cos θ, 0) in spherical coordinates
/// let f = &r.powi(2) * &th.sin();
/// let g = gradient_in(&f, &[&r, &th, &ph], CoordinateSystem::Spherical).unwrap().simplify();
/// assert_eq!(g[(0, 0)], &(&r * 2) * &th.sin());
/// assert_eq!(g[(1, 0)], &r * &th.cos());
/// assert!(g[(2, 0)].is_zero_structural());
/// assert!(gradient_in(&f, &[&r, &th], CoordinateSystem::Spherical).is_err());
/// ```
pub fn gradient_in(f: &Ex, vars: &[&Ex], cs: CoordinateSystem) -> Result<Matrix, SymplexError> {
    let h = cs.scale_factors_for("gradient", vars)?;
    let partials: Vec<Ex> = vars
        .iter()
        .zip(h.iter())
        .map(|(v, hi)| {
            let d = f.diff(v);
            if hi.is_one_structural() { d } else { &d / hi }
        })
        .collect();
    Matrix::col_vector(partials)
}

/// Divergence of a vector field in Cartesian coordinates:
/// `∇·F = Σᵢ ∂Fᵢ/∂xᵢ`.
///
/// # Errors
///
/// Returns [`SymplexError::InvalidArgument`] if `field` is not an `n×1`
/// column vector with `n == vars.len()`.
pub fn divergence(field: &Matrix, vars: &[&Ex]) -> Result<Ex, SymplexError> {
    divergence_in(field, vars, CoordinateSystem::Cartesian)
}

/// Divergence in the given coordinate system:
/// `∇·F = (1/(h₁h₂h₃)) Σᵢ ∂/∂qᵢ ( (h₁h₂h₃/hᵢ) Fᵢ )`.
///
/// # Errors
///
/// Same conditions as [`divergence`], plus 3 variables for curvilinear
/// systems.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::vector::{divergence_in, CoordinateSystem};
///
/// let ctx = Context::new();
/// let (r, th, ph) = (ctx.symbol("r"), ctx.symbol("theta"), ctx.symbol("phi"));
/// // Radial field F = r r̂ has divergence 3 in spherical coordinates
/// let f = Matrix::col_vector(vec![r.clone(), ctx.int(0), ctx.int(0)]).unwrap();
/// let d = divergence_in(&f, &[&r, &th, &ph], CoordinateSystem::Spherical).unwrap().simplify();
/// assert_eq!(d, ctx.int(3));
/// ```
pub fn divergence_in(
    field: &Matrix,
    vars: &[&Ex],
    cs: CoordinateSystem,
) -> Result<Ex, SymplexError> {
    check_field(field, vars, "divergence")?;
    let h = cs.scale_factors_for("divergence", vars)?;
    let ctx = vars[0].context();
    if cs == CoordinateSystem::Cartesian {
        let mut sum = ctx.zero();
        for (i, var) in vars.iter().enumerate() {
            sum = &sum + &field.get(i, 0).diff(var);
        }
        return Ok(sum);
    }
    let jac = &(&h[0] * &h[1]) * &h[2];
    let mut sum = ctx.zero();
    for (i, var) in vars.iter().enumerate() {
        let weight = &jac / &h[i];
        let term = (&weight * field.get(i, 0)).diff(var);
        sum = &sum + &term;
    }
    Ok(&sum / &jac)
}

/// Curl of a 3-D vector field in Cartesian coordinates: `∇×F`.
///
/// # Errors
///
/// Returns [`SymplexError::InvalidArgument`] if `field` is not `3×1` or
/// `vars.len() != 3`.
pub fn curl(field: &Matrix, vars: &[&Ex]) -> Result<Matrix, SymplexError> {
    curl_in(field, vars, CoordinateSystem::Cartesian)
}

/// Curl in the given coordinate system:
///
/// ```text
/// (∇×F)₁ = (1/(h₂h₃)) [ ∂(h₃F₃)/∂q₂ − ∂(h₂F₂)/∂q₃ ]   (and cyclic)
/// ```
///
/// # Errors
///
/// Returns [`SymplexError::InvalidArgument`] if `field` is not `3×1` or
/// `vars.len() != 3`.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::vector::{curl_in, CoordinateSystem};
///
/// let ctx = Context::new();
/// let (r, ph, z) = (ctx.symbol("r"), ctx.symbol("phi"), ctx.symbol("z"));
/// // Rigid rotation F = r φ̂ has curl 2 ẑ in cylindrical coordinates
/// let f = Matrix::col_vector(vec![ctx.int(0), r.clone(), ctx.int(0)]).unwrap();
/// let c = curl_in(&f, &[&r, &ph, &z], CoordinateSystem::Cylindrical).unwrap().simplify();
/// assert_eq!(c, matrix![ctx, [0], [0], [2]]);
/// assert!(curl_in(&f, &[&r, &ph], CoordinateSystem::Cartesian).is_err());
/// ```
pub fn curl_in(field: &Matrix, vars: &[&Ex], cs: CoordinateSystem) -> Result<Matrix, SymplexError> {
    if field.shape() != (3, 1) || vars.len() != 3 {
        return Err(SymplexError::invalid_argument(
            "curl",
            format!(
                "curl needs a 3×1 field and 3 variables (got {}×{} and {})",
                field.nrows(),
                field.ncols(),
                vars.len()
            ),
        ));
    }

    let h = cs.scale_factors_for("curl", vars)?;
    let hf: Vec<Ex> = (0..3).map(|i| &h[i] * field.get(i, 0)).collect();
    // Component i uses the cyclic pair (j, k).
    let comp = |j: usize, k: usize| {
        let num = &hf[k].diff(vars[j]) - &hf[j].diff(vars[k]);
        let denom = &h[j] * &h[k];
        if denom.is_one_structural() {
            num
        } else {
            &num / &denom
        }
    };
    Matrix::col_vector(vec![comp(1, 2), comp(2, 0), comp(0, 1)])
}

/// Laplacian of a scalar field in Cartesian coordinates:
/// `∇²f = Σᵢ ∂²f/∂xᵢ²`.
///
/// # Errors
///
/// Returns [`SymplexError::InvalidArgument`] if `vars` is empty.
pub fn laplacian(f: &Ex, vars: &[&Ex]) -> Result<Ex, SymplexError> {
    laplacian_in(f, vars, CoordinateSystem::Cartesian)
}

/// Laplacian `∇²f = ∇·(∇f)` in the given coordinate system.
///
/// # Errors
///
/// Same conditions as [`gradient_in`].
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::vector::{laplacian_in, CoordinateSystem};
///
/// let ctx = Context::new();
/// let (r, th, ph) = (ctx.symbol("r"), ctx.symbol("theta"), ctx.symbol("phi"));
/// // 1/r is harmonic away from the origin
/// let f = ctx.int(1) / &r;
/// let l = laplacian_in(&f, &[&r, &th, &ph], CoordinateSystem::Spherical).unwrap().simplify();
/// assert!(l.is_zero_structural());
/// // ∇²(r²) = 6 in spherical coordinates
/// let l2 = laplacian_in(&r.powi(2), &[&r, &th, &ph], CoordinateSystem::Spherical).unwrap().simplify();
/// assert_eq!(l2, ctx.int(6));
/// ```
pub fn laplacian_in(f: &Ex, vars: &[&Ex], cs: CoordinateSystem) -> Result<Ex, SymplexError> {
    let grad = gradient_in(f, vars, cs)?;
    divergence_in(&grad, vars, cs)
}

/// Directional derivative `∇f · d` of `f` along the (Cartesian) direction
/// vector `d`.
///
/// `d` is **not** normalised; pass a unit vector for the classical
/// directional derivative.
///
/// # Errors
///
/// Returns [`SymplexError::InvalidArgument`] if `direction` is not an
/// `n×1` column vector with `n == vars.len()`.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::vector::directional_derivative;
///
/// let ctx = Context::new();
/// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
/// let f = &x.powi(2) + &y.powi(2);
/// let d = matrix![ctx, [1], [1]];
/// assert_eq!(directional_derivative(&f, &[&x, &y], &d).unwrap().expand(), &x * 2 + &y * 2);
/// assert!(directional_derivative(&f, &[&x], &d).is_err());
/// ```
pub fn directional_derivative(
    f: &Ex,
    vars: &[&Ex],
    direction: &Matrix,
) -> Result<Ex, SymplexError> {
    check_field(direction, vars, "directional_derivative")?;
    let grad = gradient(f, vars)?;
    crate::domains::matrix::dot(&grad, direction)
}

// ═══════════════════════════════════════════════════════════════════════════
// Field classification & potentials
// ═══════════════════════════════════════════════════════════════════════════

/// Is the 3-D vector field conservative (curl-free)?  Three-valued.
///
/// Returns `Some(false)` for anything that is not a `3×1` field with three
/// variables or if a curl component is provably non-zero, `Some(true)` if
/// every component simplifies to zero, and `None` if some component's
/// value cannot be decided symbolically.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::vector::{gradient, is_conservative};
///
/// let ctx = Context::new();
/// let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
/// let f = gradient(&(&(&x * &y) + &z.powi(2)), &[&x, &y, &z]).unwrap();
/// assert_eq!(is_conservative(&f, &[&x, &y, &z]), Some(true));
/// let rot = Matrix::col_vector(vec![-&y, x.clone(), ctx.int(0)]).unwrap();
/// assert_eq!(is_conservative(&rot, &[&x, &y, &z]), Some(false));
/// ```
pub fn is_conservative(field: &Matrix, vars: &[&Ex]) -> Option<bool> {
    // `curl` fails exactly on the shapes that are not a 3-D field.
    let Ok(c) = curl(field, vars) else {
        return Some(false);
    };
    all3(c.iter().map(|e| identically_zero(e, vars)))
}

/// Is the field component `e` identically zero as a function of `vars`?
/// Three-valued.
///
/// [`ex_is_zero`] asks whether `e` vanishes for *every* value of its
/// symbols, so it cannot refute a component such as `x·sin x − cos x`,
/// which is zero at some points: a point of `vars` (only those are
/// substituted) where `e` is a certified nonzero finite constant proves
/// that `e` is not identically zero.  (Before 0.41 `is_conservative` and
/// `is_solenoidal` returned `None` for such fields: 193 of 200 random
/// rotational fields.)
fn identically_zero(e: &Ex, vars: &[&Ex]) -> Option<bool> {
    let decided = ex_is_zero(e);
    if decided.is_some() {
        return decided;
    }
    let ctx = e.context();
    const POINTS: [[(i64, i64); 3]; 3] = [
        [(1, 3), (2, 7), (5, 11)],
        [(7, 5), (-3, 4), (9, 7)],
        [(-2, 9), (11, 6), (-5, 13)],
    ];
    for point in POINTS {
        let values: Vec<Ex> = (0..vars.len())
            .map(|i| {
                let (n, d) = point[i % 3];
                ctx.rational(n + i as i64 / 3, d)
            })
            .collect();
        let pairs: Vec<(&Ex, &Ex)> = vars.iter().copied().zip(values.iter()).collect();
        let at = e.subs_map(&pairs).eval();
        if !at.free_symbols().is_empty() {
            return None; // depends on parameters: leave it undecided
        }
        if at
            .eval_complex64()
            .is_ok_and(|z| z.re.is_finite() && z.im.is_finite() && z.norm() > 0.0)
            && ex_is_zero(&at) == Some(false)
        {
            return Some(false);
        }
    }
    None
}

/// Alias of [`is_conservative`] (irrotational ⇔ curl-free).
pub fn is_irrotational(field: &Matrix, vars: &[&Ex]) -> Option<bool> {
    is_conservative(field, vars)
}

/// Is the vector field solenoidal (divergence-free)?  Three-valued, like
/// [`is_conservative`]: `Some(false)` for a field that is not an `n×1`
/// column with `n == vars.len()`.
pub fn is_solenoidal(field: &Matrix, vars: &[&Ex]) -> Option<bool> {
    divergence(field, vars).map_or(Some(false), |d| identically_zero(&d, vars))
}

/// Scalar potential `φ` with `∇φ = F` for a conservative Cartesian field.
///
/// Works in any dimension.  The potential is built by successive
/// integration: `φ = ∫F₁ dx₁ + ∫(F₂ − ∂φ₁/∂x₂) dx₂ + …`, and the result is
/// verified by re-differentiation.  The additive constant is zero.  An
/// antiderivative that splits on a parameter (`∫ x·cos(xy) dx`, a
/// `Piecewise` on `y ≠ 0`) contributes its generic branch, as the
/// potential is generic in the other coordinates anyway (before 0.41 the
/// `Piecewise` made the next integration fail: `∇(sin(xy))` had no
/// potential).
///
/// # Errors
///
/// - [`SymplexError::InvalidArgument`] if `field` is not an `n×1` column
///   vector with `n == vars.len()`.
/// - [`SymplexError::ComputationFailed`] if the field is not conservative
///   or an antiderivative could not be found in closed form.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::vector::{gradient, scalar_potential};
///
/// let ctx = Context::new();
/// let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
/// let phi = &(&x * &y) * &z + &x.powi(2);
/// let f = gradient(&phi, &[&x, &y, &z]).unwrap();
/// let recovered = scalar_potential(&f, &[&x, &y, &z]).unwrap();
/// assert!((&recovered - &phi).expand().is_zero_structural());
///
/// // y x̂ − x ŷ is rotational: no potential
/// let rot = Matrix::col_vector(vec![y.clone(), -&x, ctx.int(0)]).unwrap();
/// assert!(scalar_potential(&rot, &[&x, &y, &z]).is_err());
/// ```
pub fn scalar_potential(field: &Matrix, vars: &[&Ex]) -> Result<Ex, SymplexError> {
    if field.ncols() != 1 || field.nrows() != vars.len() || vars.is_empty() {
        return Err(SymplexError::InvalidArgument {
            operation: "scalar_potential",
            reason: format!(
                "field must be an n×1 column vector with n == vars.len() (got {}×{} and {} variables)",
                field.nrows(),
                field.ncols(),
                vars.len()
            ),
        });
    }
    let ctx = vars[0].context();
    let mut phi = ctx.zero();
    for (i, var) in vars.iter().enumerate() {
        let residual = (field.get(i, 0) - &phi.diff(var)).expand();
        if residual.is_zero_structural() {
            continue;
        }
        let anti = residual
            .try_integrate(var)
            .map_err(|e| SymplexError::ComputationFailed {
                operation: "scalar_potential",
                reason: format!("could not integrate component {i} ({residual}): {e}"),
            })?;
        let anti = generic_branches(&anti, var).ok_or_else(|| SymplexError::ComputationFailed {
            operation: "scalar_potential",
            reason: format!("the antiderivative of component {i} is piecewise in {var}: {anti}"),
        })?;
        phi = (&phi + &anti).expand();
    }
    // Verify ∇φ = F; a mismatch means the field is not conservative.
    for (i, var) in vars.iter().enumerate() {
        let diff = (&phi.diff(var) - field.get(i, 0)).expand().simplify();
        if !diff.is_zero_structural() {
            return Err(SymplexError::ComputationFailed {
                operation: "scalar_potential",
                reason: format!(
                    "field is not conservative: ∂φ/∂{} − F_{} = {} ≠ 0",
                    var,
                    i + 1,
                    diff
                ),
            });
        }
    }
    Ok(phi)
}

/// `e` with every `Piecewise` whose first condition is free of `var`
/// (a condition on the other coordinates, `y ≠ 0`) replaced by its first,
/// generic branch; `None` if a `Piecewise` in `var` remains.  Iterative
/// (post-order), no recursion over the expression tree.
fn generic_branches(e: &Ex, var: &Ex) -> Option<Ex> {
    use crate::base::node::ExprNode;
    use crate::base::walk;
    let var_id = e.checked_id(var);
    let mut inner = e.inner.write();
    let arena = &mut inner.arena;
    let root = e.raw_id();
    let post = walk::post_order_ids(arena, root);
    let mut cache: rustc_hash::FxHashMap<_, _> = rustc_hash::FxHashMap::default();
    for &id in &post {
        let rebuilt = walk::rebuild_with_cache(arena, id, &cache);
        let new = match arena.node(rebuilt).clone() {
            ExprNode::Piecewise(pairs) => {
                let &(value, cond) = pairs.first()?;
                if walk::contains(arena, cond, var_id) {
                    return None;
                }
                value
            }
            _ => rebuilt,
        };
        cache.insert(id, new);
    }
    let r = cache.get(&root).copied().unwrap_or(root);
    let r = crate::transforms::eval::eval(arena, r);
    drop(inner);
    Some(e.wrap(r))
}

// ═══════════════════════════════════════════════════════════════════════════
// Line integrals
// ═══════════════════════════════════════════════════════════════════════════

/// `∫ₐᵇ g dt` by the definite integrator ([`Ex::integrate_definite`]):
/// an antiderivative with a jump in `[a, b]` is handled there, and an
/// integral without a closed form stays a `DefiniteIntegral` node that
/// evaluates numerically.  (Before 0.41 this was `F(b) − F(a)` of the
/// indefinite integral, which left unevaluable `Subs(Integral(…))` forms.)
///
/// With a symbolic bound the definite integrator may leave a closed-form
/// integral unevaluated (`∫₀ᵀ √(4t² + 1) dt`); then `F(b) − F(a)` of a
/// closed-form antiderivative is used, as before.
///
/// `t` is the real stand-in of [`real_parameter`]; an integral left
/// unevaluated is shown in the caller's parameter `t_user`.
fn definite(g: &Ex, t: &Ex, t_user: &Ex, a: &Ex, b: &Ex) -> Ex {
    let d = g.integrate_definite(t, a, b);
    if d.has_unevaluated() && !(a.is_constant() && b.is_constant()) {
        let anti = g.integrate(t);
        if !anti.has_unevaluated() {
            return (&anti.subs(t, b) - &anti.subs(t, a)).eval();
        }
    }
    if d.is_definite_integral() && t != t_user {
        return g.subs(t, t_user).definite_integral_node(t_user, a, b);
    }
    d
}

/// The curve parameter as a real symbol: `t` itself when it is declared
/// real, otherwise a fresh real stand-in.  The parameter runs over the
/// real interval `[a, b]`, and a complex `t` keeps the speed
/// `√(cos²t)` from becoming `|cos t|` (before 0.41 the arc length of
/// `(sin t, 0)` over `[0, π]` stayed an unevaluated integral).
///
/// The stand-in has a reserved-looking name (`__t`, `__t_1`, …: the first
/// one absent from `avoid`), so repeated calls reuse it.
fn real_parameter(t: &Ex, avoid: &[&Ex]) -> Ex {
    if t.is_real() == Some(true) {
        return t.clone();
    }
    let ctx = t.context();
    let real = crate::base::assumptions::Assumptions::default()
        .with(crate::base::assumptions::Assumption::Real);
    let mut k = 0usize;
    loop {
        let name = if k == 0 {
            "__t".to_string()
        } else {
            format!("__t_{k}")
        };
        let plain = ctx.symbol(&name);
        if !avoid.iter().any(|e| e.contains(&plain)) {
            match plain.is_real() {
                Some(true) => return plain,
                None => return ctx.declare_symbol(&name, real),
                Some(false) => {} // declared non-real elsewhere: skip it
            }
        }
        k += 1;
    }
}

/// `e` on the curve: every variable replaced by its curve component
/// simultaneously, so a curve parametrised by one of the coordinates
/// (`t = y`) is not substituted twice.  (Before 0.41 the replacements
/// were made one after the other: the curve `(y, 2y)` with parameter `y`
/// became `(2y, 2y)`.)
fn on_curve(e: &Ex, vars: &[&Ex], curve: &[Ex]) -> Ex {
    let pairs: Vec<(&Ex, &Ex)> = vars.iter().copied().zip(curve.iter()).collect();
    e.subs_map(&pairs)
}

/// Scalar line integral `∫_C f ds = ∫ₐᵇ f(r(t)) ‖r′(t)‖ dt`.
///
/// `curve` gives the components of `r(t)` in the same order as `vars`.
/// The result may contain an unevaluated `Integral` node if no closed
/// form is found (check with [`Ex::has_unevaluated`]).
///
/// # Errors
///
/// Returns [`SymplexError::InvalidArgument`] if `vars` is empty or
/// `curve.len() != vars.len()`.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::vector::line_integral_scalar;
///
/// let ctx = Context::new();
/// let (x, y, t) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("t"));
/// // Arc length of the unit circle: ∫ 1 ds over t ∈ [0, 2π] = 2π
/// let circle = [t.cos(), t.sin()];
/// let len = line_integral_scalar(&ctx.int(1), &[&x, &y], &circle, &t, &ctx.int(0), &(ctx.pi() * 2)).unwrap();
/// assert_eq!(len.simplify(), ctx.pi() * 2);
/// ```
pub fn line_integral_scalar(
    f: &Ex,
    vars: &[&Ex],
    curve: &[Ex],
    t: &Ex,
    a: &Ex,
    b: &Ex,
) -> Result<Ex, SymplexError> {
    if vars.is_empty() || curve.len() != vars.len() {
        return Err(SymplexError::invalid_argument(
            "line_integral_scalar",
            format!(
                "need a non-empty variable list and one curve component per variable (got {} components, {} variables)",
                curve.len(),
                vars.len()
            ),
        ));
    }
    let ctx = t.context();
    let f_on_curve = on_curve(f, vars, curve);
    let mut avoid: Vec<&Ex> = vec![&f_on_curve, a, b];
    avoid.extend(curve.iter());
    let tr = real_parameter(t, &avoid);
    let f_on_curve = f_on_curve.subs(t, &tr);
    let mut speed_sq = ctx.zero();
    for c in curve {
        speed_sq += c.diff(t).subs(t, &tr).powi(2);
    }
    let integrand = (&f_on_curve * &speed_sq.simplify().sqrt()).simplify();
    Ok(definite(&integrand, &tr, t, a, b))
}

/// Vector line integral (work) `∫_C F·dr = ∫ₐᵇ F(r(t)) · r′(t) dt`.
///
/// `field` is an `n×1` column vector in the variables `vars`; `curve`
/// gives `r(t)` component-wise.  The result may contain an unevaluated
/// `Integral` node if no closed form is found.
///
/// # Errors
///
/// Returns [`SymplexError::InvalidArgument`] if `field` is not `n×1` with
/// `n == vars.len() == curve.len()`.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::vector::line_integral_vector;
///
/// let ctx = Context::new();
/// let (x, y, t) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("t"));
/// // Circulation of F = (−y, x) around the unit circle = 2π
/// let f = Matrix::col_vector(vec![-&y, x.clone()]).unwrap();
/// let circle = [t.cos(), t.sin()];
/// let w = line_integral_vector(&f, &[&x, &y], &circle, &t, &ctx.int(0), &(ctx.pi() * 2)).unwrap();
/// assert_eq!(w.simplify(), ctx.pi() * 2);
/// ```
pub fn line_integral_vector(
    field: &Matrix,
    vars: &[&Ex],
    curve: &[Ex],
    t: &Ex,
    a: &Ex,
    b: &Ex,
) -> Result<Ex, SymplexError> {
    check_field(field, vars, "line_integral_vector")?;
    if curve.len() != vars.len() {
        return Err(SymplexError::invalid_argument(
            "line_integral_vector",
            format!(
                "curve has {} components but {} variables",
                curve.len(),
                vars.len()
            ),
        ));
    }
    let ctx = t.context();
    let mut integrand = ctx.zero();
    for (i, c) in curve.iter().enumerate() {
        let fi = on_curve(field.get(i, 0), vars, curve);
        integrand += &fi * &c.diff(t);
    }
    let tr = real_parameter(t, &[&integrand, a, b]);
    Ok(definite(&integrand.subs(t, &tr).simplify(), &tr, t, a, b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::context::Context;

    #[test]
    fn cartesian_operators_unchanged() {
        let ctx = Context::new();
        let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
        let f = &(&x.powi(2) * &y) + &z.powi(3);
        let g = gradient(&f, &[&x, &y, &z]).unwrap();
        assert_eq!(g.get(0, 0), &(&(&x * 2) * &y));
        assert_eq!(g.get(2, 0), &(&z.powi(2) * 3));
        assert_eq!(laplacian(&f, &[&x, &y, &z]).unwrap(), &y * 2 + &z * 6);
        let field = Matrix::col_vector(vec![x.clone(), y.clone(), z.clone()]).unwrap();
        assert_eq!(divergence(&field, &[&x, &y, &z]).unwrap(), ctx.int(3));
        assert_eq!(is_conservative(&field, &[&x, &y, &z]), Some(true));
        assert_eq!(is_irrotational(&field, &[&x, &y, &z]), Some(true));
        assert_eq!(is_solenoidal(&field, &[&x, &y, &z]), Some(false));
        // Undecidable: curl component `a` with no sign information.
        let a = ctx.symbol("a");
        let unknown = Matrix::col_vector(vec![ctx.int(0), ctx.int(0), &a * &x]).unwrap();
        assert_eq!(is_conservative(&unknown, &[&x, &y, &z]), None);
    }

    #[test]
    fn cylindrical_identities() {
        let ctx = Context::new();
        let (r, ph, z) = (ctx.symbol("r"), ctx.symbol("phi"), ctx.symbol("z"));
        let vars = [&r, &ph, &z];
        // ∇²(r²) = 4 (since r² = x² + y²)
        let l = laplacian_in(&r.powi(2), &vars, CoordinateSystem::Cylindrical)
            .unwrap()
            .simplify();
        assert_eq!(l, ctx.int(4));
        // ln r is harmonic in 2D
        let l2 = laplacian_in(&r.ln(), &vars, CoordinateSystem::Cylindrical)
            .unwrap()
            .simplify();
        assert!(l2.is_zero_structural());
        // div(r r̂) = 2
        let f = Matrix::col_vector(vec![r.clone(), ctx.int(0), ctx.int(0)]).unwrap();
        assert_eq!(
            divergence_in(&f, &vars, CoordinateSystem::Cylindrical)
                .unwrap()
                .simplify(),
            ctx.int(2)
        );
        // curl of a gradient vanishes
        let g = gradient_in(
            &(&r.powi(2) * &ph.cos()),
            &vars,
            CoordinateSystem::Cylindrical,
        )
        .unwrap();
        let c = curl_in(&g, &vars, CoordinateSystem::Cylindrical)
            .unwrap()
            .simplify();
        for i in 0..3 {
            assert!(
                c.get(i, 0).is_zero_structural(),
                "curl grad component {i} = {}",
                c.get(i, 0)
            );
        }
        // φ̂/r has zero curl (away from the axis)
        let vortex = Matrix::col_vector(vec![ctx.int(0), ctx.int(1) / &r, ctx.int(0)]).unwrap();
        let cv = curl_in(&vortex, &vars, CoordinateSystem::Cylindrical)
            .unwrap()
            .simplify();
        assert!(cv.get(2, 0).is_zero_structural());
    }

    #[test]
    fn spherical_identities() {
        let ctx = Context::new();
        let (r, th, ph) = (ctx.symbol("r"), ctx.symbol("theta"), ctx.symbol("phi"));
        let vars = [&r, &th, &ph];
        assert_eq!(
            CoordinateSystem::Spherical.scale_factors(&vars).unwrap(),
            vec![ctx.one(), r.clone(), &r * &th.sin()]
        );
        // ∇²(1/r) = 0, ∇²(r²) = 6
        let l = laplacian_in(&(ctx.int(1) / &r), &vars, CoordinateSystem::Spherical)
            .unwrap()
            .simplify();
        assert!(l.is_zero_structural(), "{l}");
        let l2 = laplacian_in(&r.powi(2), &vars, CoordinateSystem::Spherical)
            .unwrap()
            .simplify();
        assert_eq!(l2, ctx.int(6));
        // ∇²(r cos θ) = 0 (it is z)
        let l3 = laplacian_in(&(&r * &th.cos()), &vars, CoordinateSystem::Spherical)
            .unwrap()
            .simplify();
        assert!(l3.is_zero_structural(), "{l3}");
        // div(r̂/r²) = 0
        let coulomb =
            Matrix::col_vector(vec![ctx.int(1) / r.powi(2), ctx.int(0), ctx.int(0)]).unwrap();
        let d = divergence_in(&coulomb, &vars, CoordinateSystem::Spherical)
            .unwrap()
            .simplify();
        assert!(d.is_zero_structural(), "{d}");
        // curl of gradient vanishes
        let g = gradient_in(
            &(&r.powi(3) * &(&th.sin() * &ph.cos())),
            &vars,
            CoordinateSystem::Spherical,
        )
        .unwrap();
        let c = curl_in(&g, &vars, CoordinateSystem::Spherical)
            .unwrap()
            .simplify();
        for i in 0..3 {
            assert!(
                c.get(i, 0).is_zero_structural(),
                "component {i} = {}",
                c.get(i, 0)
            );
        }
    }

    #[test]
    fn directional_and_potential() {
        let ctx = Context::new();
        let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
        let f = &x * &y;
        let d = Matrix::col_vector(vec![ctx.int(2), ctx.int(-1)]).unwrap();
        assert_eq!(
            directional_derivative(&f, &[&x, &y], &d).unwrap().expand(),
            &y * 2 - &x
        );

        let phi = &(&x.powi(2) * &y) + &(&y * &z.sin()) + &z.powi(3);
        let field = gradient(&phi, &[&x, &y, &z]).unwrap();
        let back = scalar_potential(&field, &[&x, &y, &z]).unwrap();
        assert!(
            (&back - &phi).expand().simplify().is_zero_structural(),
            "{back}"
        );
        let rot = Matrix::col_vector(vec![-&y, x.clone(), ctx.int(0)]).unwrap();
        assert!(scalar_potential(&rot, &[&x, &y, &z]).is_err());
        assert!(scalar_potential(&rot, &[&x, &y]).is_err());
        // 2-D works too
        let p2 = scalar_potential(
            &Matrix::col_vector(vec![y.clone(), x.clone()]).unwrap(),
            &[&x, &y],
        )
        .unwrap();
        assert!((&p2 - &(&x * &y)).expand().is_zero_structural());
    }

    #[test]
    fn line_integrals() {
        let ctx = Context::new();
        let (x, y, t) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("t"));
        // Length of the segment (0,0)→(3,4) is 5
        let seg = [&t * 3, &t * 4];
        let len = line_integral_scalar(&ctx.int(1), &[&x, &y], &seg, &t, &ctx.int(0), &ctx.int(1))
            .unwrap();
        assert_eq!(len.simplify(), ctx.int(5));
        // ∫ (x + y) ds along the same segment = 5 · ∫₀¹ 7t dt = 35/2
        let s = line_integral_scalar(&(&x + &y), &[&x, &y], &seg, &t, &ctx.int(0), &ctx.int(1))
            .unwrap();
        assert_eq!(s.simplify(), ctx.rational(35, 2));
        // Work of a conservative field depends only on endpoints: F = ∇(xy) along the segment = 12
        let f = Matrix::col_vector(vec![y.clone(), x.clone()]).unwrap();
        let w = line_integral_vector(&f, &[&x, &y], &seg, &t, &ctx.int(0), &ctx.int(1)).unwrap();
        assert_eq!(w.simplify(), ctx.int(12));
        // Circulation of (−y, x) around the unit circle = 2π
        let rot = Matrix::col_vector(vec![-&y, x.clone()]).unwrap();
        let circle = [t.cos(), t.sin()];
        let circ = line_integral_vector(&rot, &[&x, &y], &circle, &t, &ctx.int(0), &(ctx.pi() * 2))
            .unwrap();
        assert_eq!(circ.simplify(), ctx.pi() * 2);
    }

    #[test]
    fn cylindrical_needs_three_vars() {
        let ctx = Context::new();
        let (r, ph) = (ctx.symbol("r"), ctx.symbol("phi"));
        let err = gradient_in(&r, &[&r, &ph], CoordinateSystem::Cylindrical).unwrap_err();
        assert!(err.to_string().contains("exactly 3 variables"), "{err}");
    }
}
