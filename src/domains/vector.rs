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
//! # Shape preconditions
//!
//! Wrong field dimensions / variable counts are programming errors and
//! **panic** (like slice indexing), consistent with the rest of this
//! module.  Mathematical failure (e.g. a non-conservative field passed to
//! [`scalar_potential`]) returns `Err`.

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
    /// # Panics
    ///
    /// Panics if a curvilinear system is used with `vars.len() != 3`.
    pub fn scale_factors(self, vars: &[&Ex]) -> Vec<Ex> {
        match self {
            CoordinateSystem::Cartesian => {
                let one = vars[0].context().one();
                vec![one; vars.len()]
            }
            CoordinateSystem::Cylindrical => {
                assert_eq!(
                    vars.len(),
                    3,
                    "cylindrical coordinates need exactly 3 variables (r, phi, z)"
                );
                let one = vars[0].context().one();
                vec![one.clone(), vars[0].clone(), one]
            }
            CoordinateSystem::Spherical => {
                assert_eq!(
                    vars.len(),
                    3,
                    "spherical coordinates need exactly 3 variables (r, theta, phi)"
                );
                let one = vars[0].context().one();
                vec![one, vars[0].clone(), vars[0] * &vars[1].sin()]
            }
        }
    }
}

fn check_field(field: &Matrix, vars: &[&Ex], op: &str) {
    assert_eq!(
        field.nrows(),
        vars.len(),
        "{op}: field dimension ({}) must match variable count ({})",
        field.nrows(),
        vars.len()
    );
    assert_eq!(field.ncols(), 1, "{op}: field must be a column vector");
}

// ═══════════════════════════════════════════════════════════════════════════
// Differential operators
// ═══════════════════════════════════════════════════════════════════════════

/// Gradient of a scalar field in Cartesian coordinates:
/// `∇f = [∂f/∂x₁, …, ∂f/∂xₙ]ᵀ` (an `n×1` column vector).
pub fn gradient(f: &Ex, vars: &[&Ex]) -> Matrix {
    gradient_in(f, vars, CoordinateSystem::Cartesian)
}

/// Gradient in the given coordinate system:
/// `(∇f)ᵢ = (1/hᵢ) ∂f/∂qᵢ`.
///
/// # Panics
///
/// Panics if `vars` is empty or a curvilinear system is used with
/// `vars.len() != 3`.
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
/// let g = gradient_in(&f, &[&r, &th, &ph], CoordinateSystem::Spherical).simplify();
/// assert_eq!(g[(0, 0)], &(&r * 2) * &th.sin());
/// assert_eq!(g[(1, 0)], &r * &th.cos());
/// assert!(g[(2, 0)].is_zero_structural());
/// ```
pub fn gradient_in(f: &Ex, vars: &[&Ex], cs: CoordinateSystem) -> Matrix {
    assert!(!vars.is_empty(), "gradient: vars must be non-empty");
    let h = cs.scale_factors(vars);
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
/// # Panics
///
/// Panics if `field` is not an `n×1` column vector with `n == vars.len()`.
pub fn divergence(field: &Matrix, vars: &[&Ex]) -> Ex {
    divergence_in(field, vars, CoordinateSystem::Cartesian)
}

/// Divergence in the given coordinate system:
/// `∇·F = (1/(h₁h₂h₃)) Σᵢ ∂/∂qᵢ ( (h₁h₂h₃/hᵢ) Fᵢ )`.
///
/// # Panics
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
/// let f = Matrix::col_vector(vec![r.clone(), ctx.int(0), ctx.int(0)]);
/// let d = divergence_in(&f, &[&r, &th, &ph], CoordinateSystem::Spherical).simplify();
/// assert_eq!(d, ctx.int(3));
/// ```
pub fn divergence_in(field: &Matrix, vars: &[&Ex], cs: CoordinateSystem) -> Ex {
    check_field(field, vars, "divergence");
    let ctx = vars[0].context();
    if cs == CoordinateSystem::Cartesian {
        let mut sum = ctx.zero();
        for (i, var) in vars.iter().enumerate() {
            sum = &sum + &field.get(i, 0).diff(var);
        }
        return sum;
    }
    let h = cs.scale_factors(vars);
    let jac = &(&h[0] * &h[1]) * &h[2];
    let mut sum = ctx.zero();
    for (i, var) in vars.iter().enumerate() {
        let weight = &jac / &h[i];
        let term = (&weight * field.get(i, 0)).diff(var);
        sum = &sum + &term;
    }
    &sum / &jac
}

/// Curl of a 3-D vector field in Cartesian coordinates: `∇×F`.
///
/// # Panics
///
/// Panics if `field` is not `3×1` or `vars.len() != 3`.
pub fn curl(field: &Matrix, vars: &[&Ex]) -> Matrix {
    curl_in(field, vars, CoordinateSystem::Cartesian)
}

/// Curl in the given coordinate system:
///
/// ```text
/// (∇×F)₁ = (1/(h₂h₃)) [ ∂(h₃F₃)/∂q₂ − ∂(h₂F₂)/∂q₃ ]   (and cyclic)
/// ```
///
/// # Panics
///
/// Panics if `field` is not `3×1` or `vars.len() != 3`.
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
/// let f = Matrix::col_vector(vec![ctx.int(0), r.clone(), ctx.int(0)]);
/// let c = curl_in(&f, &[&r, &ph, &z], CoordinateSystem::Cylindrical).simplify();
/// assert_eq!(c, matrix![ctx, [0], [0], [2]]);
/// ```
pub fn curl_in(field: &Matrix, vars: &[&Ex], cs: CoordinateSystem) -> Matrix {
    assert_eq!(field.nrows(), 3, "curl requires 3D vector field");
    assert_eq!(vars.len(), 3, "curl requires 3 variables");
    assert_eq!(field.ncols(), 1, "field must be a column vector");

    let h = cs.scale_factors(vars);
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
pub fn laplacian(f: &Ex, vars: &[&Ex]) -> Ex {
    laplacian_in(f, vars, CoordinateSystem::Cartesian)
}

/// Laplacian `∇²f = ∇·(∇f)` in the given coordinate system.
///
/// # Panics
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
/// let l = laplacian_in(&f, &[&r, &th, &ph], CoordinateSystem::Spherical).simplify();
/// assert!(l.is_zero_structural());
/// // ∇²(r²) = 6 in spherical coordinates
/// let l2 = laplacian_in(&r.powi(2), &[&r, &th, &ph], CoordinateSystem::Spherical).simplify();
/// assert_eq!(l2, ctx.int(6));
/// ```
pub fn laplacian_in(f: &Ex, vars: &[&Ex], cs: CoordinateSystem) -> Ex {
    let grad = gradient_in(f, vars, cs);
    divergence_in(&grad, vars, cs)
}

/// Directional derivative `∇f · d` of `f` along the (Cartesian) direction
/// vector `d`.
///
/// `d` is **not** normalised; pass a unit vector for the classical
/// directional derivative.
///
/// # Panics
///
/// Panics if `direction` is not an `n×1` column vector with
/// `n == vars.len()`.
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
/// assert_eq!(directional_derivative(&f, &[&x, &y], &d).expand(), &x * 2 + &y * 2);
/// ```
pub fn directional_derivative(f: &Ex, vars: &[&Ex], direction: &Matrix) -> Ex {
    check_field(direction, vars, "directional_derivative");
    let grad = gradient(f, vars);
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
/// let f = gradient(&(&(&x * &y) + &z.powi(2)), &[&x, &y, &z]);
/// assert_eq!(is_conservative(&f, &[&x, &y, &z]), Some(true));
/// let rot = Matrix::col_vector(vec![-&y, x.clone(), ctx.int(0)]);
/// assert_eq!(is_conservative(&rot, &[&x, &y, &z]), Some(false));
/// ```
pub fn is_conservative(field: &Matrix, vars: &[&Ex]) -> Option<bool> {
    if field.nrows() != 3 || vars.len() != 3 || field.ncols() != 1 {
        return Some(false);
    }
    let c = curl(field, vars);
    all3(c.iter().map(ex_is_zero))
}

/// Alias of [`is_conservative`] (irrotational ⇔ curl-free).
pub fn is_irrotational(field: &Matrix, vars: &[&Ex]) -> Option<bool> {
    is_conservative(field, vars)
}

/// Is the vector field solenoidal (divergence-free)?  Three-valued, like
/// [`is_conservative`].
pub fn is_solenoidal(field: &Matrix, vars: &[&Ex]) -> Option<bool> {
    ex_is_zero(&divergence(field, vars))
}

/// Scalar potential `φ` with `∇φ = F` for a conservative Cartesian field.
///
/// Works in any dimension.  The potential is built by successive
/// integration: `φ = ∫F₁ dx₁ + ∫(F₂ − ∂φ₁/∂x₂) dx₂ + …`, and the result is
/// verified by re-differentiation.  The additive constant is zero.
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
/// let f = gradient(&phi, &[&x, &y, &z]);
/// let recovered = scalar_potential(&f, &[&x, &y, &z]).unwrap();
/// assert!((&recovered - &phi).expand().is_zero_structural());
///
/// // y x̂ − x ŷ is rotational: no potential
/// let rot = Matrix::col_vector(vec![y.clone(), -&x, ctx.int(0)]);
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

// ═══════════════════════════════════════════════════════════════════════════
// Line integrals
// ═══════════════════════════════════════════════════════════════════════════

/// Evaluate `∫ₐᵇ g dt` via an antiderivative and substitution.
///
/// Kept local (rather than calling `Ex::definite_integral`) so this module
/// is independent of the definite-integration API surface.
fn definite(g: &Ex, t: &Ex, a: &Ex, b: &Ex) -> Ex {
    let anti = g.integrate(t);
    (&anti.subs(t, b) - &anti.subs(t, a)).eval()
}

/// Scalar line integral `∫_C f ds = ∫ₐᵇ f(r(t)) ‖r′(t)‖ dt`.
///
/// `curve` gives the components of `r(t)` in the same order as `vars`.
/// The result may contain an unevaluated `Integral` node if no closed
/// form is found (check with [`Ex::has_unevaluated`]).
///
/// # Panics
///
/// Panics if `curve.len() != vars.len()` or `vars` is empty.
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
/// let len = line_integral_scalar(&ctx.int(1), &[&x, &y], &circle, &t, &ctx.int(0), &(ctx.pi() * 2));
/// assert_eq!(len.simplify(), ctx.pi() * 2);
/// ```
pub fn line_integral_scalar(f: &Ex, vars: &[&Ex], curve: &[Ex], t: &Ex, a: &Ex, b: &Ex) -> Ex {
    assert!(
        !vars.is_empty(),
        "line_integral_scalar: vars must be non-empty"
    );
    assert_eq!(
        curve.len(),
        vars.len(),
        "line_integral_scalar: curve has {} components but {} variables",
        curve.len(),
        vars.len()
    );
    let ctx = t.context();
    let mut f_on_curve = f.clone();
    let mut speed_sq = ctx.zero();
    for (v, c) in vars.iter().zip(curve.iter()) {
        f_on_curve = f_on_curve.subs(v, c);
        speed_sq += c.diff(t).powi(2);
    }
    let integrand = (&f_on_curve * &speed_sq.simplify().sqrt()).simplify();
    definite(&integrand, t, a, b)
}

/// Vector line integral (work) `∫_C F·dr = ∫ₐᵇ F(r(t)) · r′(t) dt`.
///
/// `field` is an `n×1` column vector in the variables `vars`; `curve`
/// gives `r(t)` component-wise.  The result may contain an unevaluated
/// `Integral` node if no closed form is found.
///
/// # Panics
///
/// Panics if `field` is not `n×1` with `n == vars.len() == curve.len()`.
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
/// let f = Matrix::col_vector(vec![-&y, x.clone()]);
/// let circle = [t.cos(), t.sin()];
/// let w = line_integral_vector(&f, &[&x, &y], &circle, &t, &ctx.int(0), &(ctx.pi() * 2));
/// assert_eq!(w.simplify(), ctx.pi() * 2);
/// ```
pub fn line_integral_vector(
    field: &Matrix,
    vars: &[&Ex],
    curve: &[Ex],
    t: &Ex,
    a: &Ex,
    b: &Ex,
) -> Ex {
    check_field(field, vars, "line_integral_vector");
    assert_eq!(
        curve.len(),
        vars.len(),
        "line_integral_vector: curve has {} components but {} variables",
        curve.len(),
        vars.len()
    );
    let ctx = t.context();
    let mut integrand = ctx.zero();
    for (i, c) in curve.iter().enumerate() {
        let mut fi = field.get(i, 0).clone();
        for (v, cc) in vars.iter().zip(curve.iter()) {
            fi = fi.subs(v, cc);
        }
        integrand += &fi * &c.diff(t);
    }
    definite(&integrand.simplify(), t, a, b)
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
        let g = gradient(&f, &[&x, &y, &z]);
        assert_eq!(g.get(0, 0), &(&(&x * 2) * &y));
        assert_eq!(g.get(2, 0), &(&z.powi(2) * 3));
        assert_eq!(laplacian(&f, &[&x, &y, &z]), &y * 2 + &z * 6);
        let field = Matrix::col_vector(vec![x.clone(), y.clone(), z.clone()]);
        assert_eq!(divergence(&field, &[&x, &y, &z]), ctx.int(3));
        assert_eq!(is_conservative(&field, &[&x, &y, &z]), Some(true));
        assert_eq!(is_irrotational(&field, &[&x, &y, &z]), Some(true));
        assert_eq!(is_solenoidal(&field, &[&x, &y, &z]), Some(false));
        // Undecidable: curl component `a` with no sign information.
        let a = ctx.symbol("a");
        let unknown = Matrix::col_vector(vec![ctx.int(0), ctx.int(0), &a * &x]);
        assert_eq!(is_conservative(&unknown, &[&x, &y, &z]), None);
    }

    #[test]
    fn cylindrical_identities() {
        let ctx = Context::new();
        let (r, ph, z) = (ctx.symbol("r"), ctx.symbol("phi"), ctx.symbol("z"));
        let vars = [&r, &ph, &z];
        // ∇²(r²) = 4 (since r² = x² + y²)
        let l = laplacian_in(&r.powi(2), &vars, CoordinateSystem::Cylindrical).simplify();
        assert_eq!(l, ctx.int(4));
        // ln r is harmonic in 2D
        let l2 = laplacian_in(&r.ln(), &vars, CoordinateSystem::Cylindrical).simplify();
        assert!(l2.is_zero_structural());
        // div(r r̂) = 2
        let f = Matrix::col_vector(vec![r.clone(), ctx.int(0), ctx.int(0)]);
        assert_eq!(
            divergence_in(&f, &vars, CoordinateSystem::Cylindrical).simplify(),
            ctx.int(2)
        );
        // curl of a gradient vanishes
        let g = gradient_in(
            &(&r.powi(2) * &ph.cos()),
            &vars,
            CoordinateSystem::Cylindrical,
        );
        let c = curl_in(&g, &vars, CoordinateSystem::Cylindrical).simplify();
        for i in 0..3 {
            assert!(
                c.get(i, 0).is_zero_structural(),
                "curl grad component {i} = {}",
                c.get(i, 0)
            );
        }
        // φ̂/r has zero curl (away from the axis)
        let vortex = Matrix::col_vector(vec![ctx.int(0), ctx.int(1) / &r, ctx.int(0)]);
        let cv = curl_in(&vortex, &vars, CoordinateSystem::Cylindrical).simplify();
        assert!(cv.get(2, 0).is_zero_structural());
    }

    #[test]
    fn spherical_identities() {
        let ctx = Context::new();
        let (r, th, ph) = (ctx.symbol("r"), ctx.symbol("theta"), ctx.symbol("phi"));
        let vars = [&r, &th, &ph];
        assert_eq!(
            CoordinateSystem::Spherical.scale_factors(&vars),
            vec![ctx.one(), r.clone(), &r * &th.sin()]
        );
        // ∇²(1/r) = 0, ∇²(r²) = 6
        let l = laplacian_in(&(ctx.int(1) / &r), &vars, CoordinateSystem::Spherical).simplify();
        assert!(l.is_zero_structural(), "{l}");
        let l2 = laplacian_in(&r.powi(2), &vars, CoordinateSystem::Spherical).simplify();
        assert_eq!(l2, ctx.int(6));
        // ∇²(r cos θ) = 0 (it is z)
        let l3 = laplacian_in(&(&r * &th.cos()), &vars, CoordinateSystem::Spherical).simplify();
        assert!(l3.is_zero_structural(), "{l3}");
        // div(r̂/r²) = 0
        let coulomb = Matrix::col_vector(vec![ctx.int(1) / r.powi(2), ctx.int(0), ctx.int(0)]);
        let d = divergence_in(&coulomb, &vars, CoordinateSystem::Spherical).simplify();
        assert!(d.is_zero_structural(), "{d}");
        // curl of gradient vanishes
        let g = gradient_in(
            &(&r.powi(3) * &(&th.sin() * &ph.cos())),
            &vars,
            CoordinateSystem::Spherical,
        );
        let c = curl_in(&g, &vars, CoordinateSystem::Spherical).simplify();
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
        let d = Matrix::col_vector(vec![ctx.int(2), ctx.int(-1)]);
        assert_eq!(
            directional_derivative(&f, &[&x, &y], &d).expand(),
            &y * 2 - &x
        );

        let phi = &(&x.powi(2) * &y) + &(&y * &z.sin()) + &z.powi(3);
        let field = gradient(&phi, &[&x, &y, &z]);
        let back = scalar_potential(&field, &[&x, &y, &z]).unwrap();
        assert!(
            (&back - &phi).expand().simplify().is_zero_structural(),
            "{back}"
        );
        let rot = Matrix::col_vector(vec![-&y, x.clone(), ctx.int(0)]);
        assert!(scalar_potential(&rot, &[&x, &y, &z]).is_err());
        assert!(scalar_potential(&rot, &[&x, &y]).is_err());
        // 2-D works too
        let p2 =
            scalar_potential(&Matrix::col_vector(vec![y.clone(), x.clone()]), &[&x, &y]).unwrap();
        assert!((&p2 - &(&x * &y)).expand().is_zero_structural());
    }

    #[test]
    fn line_integrals() {
        let ctx = Context::new();
        let (x, y, t) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("t"));
        // Length of the segment (0,0)→(3,4) is 5
        let seg = [&t * 3, &t * 4];
        let len = line_integral_scalar(&ctx.int(1), &[&x, &y], &seg, &t, &ctx.int(0), &ctx.int(1));
        assert_eq!(len.simplify(), ctx.int(5));
        // ∫ (x + y) ds along the same segment = 5 · ∫₀¹ 7t dt = 35/2
        let s = line_integral_scalar(&(&x + &y), &[&x, &y], &seg, &t, &ctx.int(0), &ctx.int(1));
        assert_eq!(s.simplify(), ctx.rational(35, 2));
        // Work of a conservative field depends only on endpoints: F = ∇(xy) along the segment = 12
        let f = Matrix::col_vector(vec![y.clone(), x.clone()]);
        let w = line_integral_vector(&f, &[&x, &y], &seg, &t, &ctx.int(0), &ctx.int(1));
        assert_eq!(w.simplify(), ctx.int(12));
        // Circulation of (−y, x) around the unit circle = 2π
        let rot = Matrix::col_vector(vec![-&y, x.clone()]);
        let circle = [t.cos(), t.sin()];
        let circ = line_integral_vector(&rot, &[&x, &y], &circle, &t, &ctx.int(0), &(ctx.pi() * 2));
        assert_eq!(circ.simplify(), ctx.pi() * 2);
    }

    #[test]
    #[should_panic(expected = "exactly 3 variables")]
    fn cylindrical_needs_three_vars() {
        let ctx = Context::new();
        let (r, ph) = (ctx.symbol("r"), ctx.symbol("phi"));
        let _ = gradient_in(&r, &[&r, &ph], CoordinateSystem::Cylindrical);
    }
}
