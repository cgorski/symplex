//! Robotics kinematics helpers: DH parameters, forward kinematics, Jacobian,
//! and closed-form inverse kinematics of the planar 2-link arm.
//!
//! # Denavit-Hartenberg Convention
//!
//! Each joint in a serial robot arm is described by four parameters, in the
//! *standard* (distal) DH convention `Tᵢ = Rz(θᵢ)·Tz(dᵢ)·Tx(aᵢ)·Rx(αᵢ)`:
//! - θ (theta): joint angle, rotation about zᵢ₋₁ (variable for revolute joints)
//! - d: link offset, translation along zᵢ₋₁ (variable for prismatic joints)
//! - a: link length, translation along xᵢ
//! - α (alpha): link twist, rotation about xᵢ
//!
//! These produce a 4×4 homogeneous transformation matrix per joint
//! ([`dh_matrix`]).  Chaining these matrices gives the forward kinematics
//! ([`fk_chain`]).  A chain is a slice of [`DhLink`]s (symbolic `Ex`
//! parameters) or, with compile-time dimension checking, of [`DhParams`]
//! (`Angle`/`Length`); both name the four parameters so `d` and `a` — two
//! lengths — cannot be transposed silently.

use crate::domains::matrix::Matrix;
use crate::prelude::*;
use crate::units::si::{Angle, Length};

/// Build the standard Denavit-Hartenberg transformation matrix for one joint.
///
/// The DH matrix is:
/// ```text
/// T = | cos(θ)  -sin(θ)cos(α)   sin(θ)sin(α)  a·cos(θ) |
///     | sin(θ)   cos(θ)cos(α)  -cos(θ)sin(α)  a·sin(θ) |
///     |   0        sin(α)          cos(α)          d     |
///     |   0          0               0             1     |
/// ```
///
/// Parameters can be symbolic expressions (e.g. a joint variable `theta`)
/// or numeric constants built with [`Context::int`](crate::api::context::Context::int).
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::robotics::dh_matrix;
///
/// // Identity-like DH matrix (all parameters zero)
/// let ctx = Context::new();
/// let zero = ctx.int(0);
/// let t = dh_matrix(&zero, &zero, &zero, &zero);
/// assert_eq!(t.nrows(), 4);
/// assert_eq!(t.ncols(), 4);
/// ```
pub fn dh_matrix(theta: &Ex, d: &Ex, a: &Ex, alpha: &Ex) -> Matrix {
    let cos_theta = theta.cos();
    let sin_theta = theta.sin();
    let cos_alpha = alpha.cos();
    let sin_alpha = alpha.sin();

    let zero = theta.context().int(0);
    let one = theta.context().int(1);

    // Row 0: [ cos(θ), -sin(θ)cos(α), sin(θ)sin(α), a·cos(θ) ]
    let r00 = cos_theta.clone();
    let r01 = -(&sin_theta * &cos_alpha);
    let r02 = &sin_theta * &sin_alpha;
    let r03 = a * &cos_theta;

    // Row 1: [ sin(θ), cos(θ)cos(α), -cos(θ)sin(α), a·sin(θ) ]
    let r10 = sin_theta.clone();
    let r11 = &cos_theta * &cos_alpha;
    let r12 = -(&cos_theta * &sin_alpha);
    let r13 = a * &sin_theta;

    // Row 2: [ 0, sin(α), cos(α), d ]
    let r20 = zero.clone();
    let r21 = sin_alpha;
    let r22 = cos_alpha;
    let r23 = d.clone();

    // Row 3: [ 0, 0, 0, 1 ]
    let r30 = zero.clone();
    let r31 = zero.clone();
    let r32 = zero;
    let r33 = one;

    Matrix::from_rows_unchecked(vec![
        vec![r00, r01, r02, r03],
        vec![r10, r11, r12, r13],
        vec![r20, r21, r22, r23],
        vec![r30, r31, r32, r33],
    ])
}

/// Product of two `n×n` matrices built in this module (3×3 rotations, 4×4
/// homogeneous transforms).  Both factors are literals of the same fixed
/// size, so unlike [`Matrix::matmul`] there is no shape to check.
fn mul_square(a: &Matrix, b: &Matrix) -> Matrix {
    let n = a.nrows();
    // `n` = `a.nrows()` ≥ 1.
    Matrix::from_fn_unchecked(n, n, |i, j| {
        let mut acc = a.get(i, 0) * b.get(0, j);
        for k in 1..n {
            acc += a.get(i, k) * b.get(k, j);
        }
        acc
    })
}

/// The four standard Denavit-Hartenberg parameters of one link, as symbolic
/// expressions.
///
/// [`dh_matrix`] turns a link into its 4×4 transform
/// `Rz(theta)·Tz(d)·Tx(a)·Rx(alpha)`; [`fk_chain`], [`fk_position`] and
/// [`fk_rotation`] take a slice of links.  The fields are borrowed so that a
/// chain can share one `zero` between links (as every planar arm does)
/// without cloning, mirroring the dimension-typed [`DhParams`].
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::robotics::DhLink;
///
/// let ctx = Context::new();
/// let theta = ctx.symbol("theta");
/// let zero = ctx.int(0);
/// let l = ctx.symbol("L");
///
/// // A planar revolute joint: only the angle and the link length are non-zero.
/// let link = DhLink { theta: &theta, d: &zero, a: &l, alpha: &zero };
/// assert_eq!(format!("{}", link.a), "L");
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DhLink<'a> {
    /// θᵢ — joint angle: rotation about the previous z-axis zᵢ₋₁ (the joint
    /// variable of a revolute joint).
    pub theta: &'a Ex,
    /// dᵢ — link offset: translation along the previous z-axis zᵢ₋₁ (the
    /// joint variable of a prismatic joint).
    pub d: &'a Ex,
    /// aᵢ — link length: translation along the new x-axis xᵢ.
    pub a: &'a Ex,
    /// αᵢ — link twist: rotation about the new x-axis xᵢ.
    pub alpha: &'a Ex,
}

/// Chain-multiply a sequence of DH transformation matrices to compute
/// the total forward-kinematics transformation.
///
/// ```text
/// T_total = T₁ · T₂ · T₃ · ... · Tₙ
/// ```
///
/// Starts with a 4×4 identity matrix and multiplies each DH matrix
/// in order.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::robotics::{dh_matrix, fk_chain, DhLink};
///
/// let ctx = Context::new();
/// let theta1 = ctx.symbol("theta1");
/// let zero = ctx.int(0);
/// let l1 = ctx.symbol("L1");
///
/// let links = [DhLink { theta: &theta1, d: &zero, a: &l1, alpha: &zero }];
/// let t = fk_chain(&links);
/// assert_eq!(t.shape(), (4, 4));
/// ```
pub fn fk_chain(links: &[DhLink<'_>]) -> Matrix {
    let ctx = if let Some(first) = links.first() {
        first.theta.context()
    } else {
        crate::api::context::Context::new()
    };
    let mut result = Matrix::identity_unchecked(&ctx, 4);
    for link in links {
        result = mul_square(&result, &dh_matrix(link.theta, link.d, link.a, link.alpha));
    }
    result
}

/// Extract the end-effector position `(x, y, z)` from the forward-kinematics
/// chain.
///
/// These are the top three entries of the last column (column 3) of the
/// 4×4 homogeneous transformation matrix.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::robotics::{fk_position, DhLink};
///
/// let ctx = Context::new();
/// let theta = ctx.symbol("theta");
/// let zero = ctx.int(0);
/// let l = ctx.symbol("L");
///
/// let (x, y, z) = fk_position(&[DhLink { theta: &theta, d: &zero, a: &l, alpha: &zero }]);
/// // x, y, z are symbolic expressions
/// ```
pub fn fk_position(links: &[DhLink<'_>]) -> (Ex, Ex, Ex) {
    let t = fk_chain(links);
    let px = t.get(0, 3).clone().eval();
    let py = t.get(1, 3).clone().eval();
    let pz = t.get(2, 3).clone().eval();
    (px, py, pz)
}

/// Extract the 3×3 rotation submatrix from the forward-kinematics result.
///
/// This is the upper-left 3×3 block of the 4×4 homogeneous transformation
/// matrix, representing the orientation of the end-effector frame.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::robotics::{fk_rotation, DhLink};
///
/// let ctx = Context::new();
/// let theta = ctx.symbol("theta");
/// let zero = ctx.int(0);
/// let l = ctx.symbol("L");
///
/// let r = fk_rotation(&[DhLink { theta: &theta, d: &zero, a: &l, alpha: &zero }]);
/// assert_eq!(r.shape(), (3, 3));
/// ```
pub fn fk_rotation(links: &[DhLink<'_>]) -> Matrix {
    let t = fk_chain(links);
    Matrix::from_fn_unchecked(3, 3, |i, j| t.get(i, j).clone())
}

/// Rotation matrix about the x-axis by angle θ.
///
/// ```text
/// Rx(θ) = | 1    0       0    |
///          | 0  cos(θ)  -sin(θ)|
///          | 0  sin(θ)   cos(θ)|
/// ```
pub fn rot_x(theta: &Ex) -> Matrix {
    let zero = theta.context().int(0);
    let one = theta.context().int(1);
    let c = theta.cos();
    let s = theta.sin();
    Matrix::from_rows_unchecked(vec![
        vec![one, zero.clone(), zero.clone()],
        vec![zero.clone(), c.clone(), -&s],
        vec![zero, s, c],
    ])
}

/// Rotation matrix about the y-axis by angle θ.
///
/// ```text
/// Ry(θ) = |  cos(θ)  0  sin(θ) |
///          |    0     1    0     |
///          | -sin(θ)  0  cos(θ)  |
/// ```
pub fn rot_y(theta: &Ex) -> Matrix {
    let zero = theta.context().int(0);
    let one = theta.context().int(1);
    let c = theta.cos();
    let s = theta.sin();
    Matrix::from_rows_unchecked(vec![
        vec![c.clone(), zero.clone(), s.clone()],
        vec![zero.clone(), one, zero.clone()],
        vec![-&s, zero, c],
    ])
}

/// Rotation matrix about the z-axis by angle θ.
///
/// ```text
/// Rz(θ) = | cos(θ)  -sin(θ)  0 |
///          | sin(θ)   cos(θ)  0 |
///          |   0        0     1 |
/// ```
pub fn rot_z(theta: &Ex) -> Matrix {
    let zero = theta.context().int(0);
    let one = theta.context().int(1);
    let c = theta.cos();
    let s = theta.sin();
    Matrix::from_rows_unchecked(vec![
        vec![c.clone(), -&s, zero.clone()],
        vec![s, c, zero.clone()],
        vec![zero.clone(), zero, one],
    ])
}

/// Skew-symmetric matrix from a 3-vector [a, b, c].
///
/// Returns the 3×3 matrix such that skew(v) · w = v × w:
/// ```text
/// [ω]× = |  0  -c   b |
///         |  c   0  -a |
///         | -b   a   0 |
/// ```
pub fn skew3(a: &Ex, b: &Ex, c: &Ex) -> Matrix {
    let zero = a.context().int(0);
    Matrix::from_rows_unchecked(vec![
        vec![zero.clone(), -c, b.clone()],
        vec![c.clone(), zero.clone(), -a],
        vec![-b, a.clone(), zero],
    ])
}

/// Build a 4×4 homogeneous transformation matrix from a 3×3 rotation
/// matrix and a 3-element position vector.
///
/// ```text
/// T = | R  p |
///     | 0  1 |
/// ```
///
/// # Errors
///
/// Returns [`SymplexError::InvalidArgument`] if `rotation` is not 3×3.
pub fn homogeneous(rotation: &Matrix, position: &[Ex; 3]) -> Result<Matrix, SymplexError> {
    if rotation.shape() != (3, 3) {
        return Err(SymplexError::InvalidArgument {
            operation: "robotics::homogeneous",
            reason: format!(
                "rotation must be 3×3, got {}×{}",
                rotation.nrows(),
                rotation.ncols()
            ),
        });
    }
    let zero = position[0].context().int(0);
    let one = position[0].context().int(1);
    Ok(Matrix::from_rows_unchecked(vec![
        vec![
            rotation.get(0, 0).clone(),
            rotation.get(0, 1).clone(),
            rotation.get(0, 2).clone(),
            position[0].clone(),
        ],
        vec![
            rotation.get(1, 0).clone(),
            rotation.get(1, 1).clone(),
            rotation.get(1, 2).clone(),
            position[1].clone(),
        ],
        vec![
            rotation.get(2, 0).clone(),
            rotation.get(2, 1).clone(),
            rotation.get(2, 2).clone(),
            position[2].clone(),
        ],
        vec![zero.clone(), zero.clone(), zero, one],
    ]))
}

/// Pure translation as a 4×4 homogeneous transformation matrix.
///
/// ```text
/// T = | I  p |
///     | 0  1 |
/// ```
pub fn translation(x: &Ex, y: &Ex, z: &Ex) -> Matrix {
    let zero = x.context().int(0);
    let one = x.context().int(1);
    Matrix::from_rows_unchecked(vec![
        vec![one.clone(), zero.clone(), zero.clone(), x.clone()],
        vec![zero.clone(), one.clone(), zero.clone(), y.clone()],
        vec![zero.clone(), zero.clone(), one, z.clone()],
        vec![zero.clone(), zero.clone(), zero, x.context().int(1)],
    ])
}

/// Euler angle convention for rotation composition.
///
/// Supports common conventions: ZYX (aerospace/Tait-Bryan),
/// ZXZ (classical), XYZ, etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EulerConvention {
    /// Aerospace (yaw-pitch-roll): R = Rz(φ) · Ry(θ) · Rx(ψ)
    ZYX,
    /// Classical: R = Rz(φ) · Rx(θ) · Rz(ψ)
    ZXZ,
    /// Roll-pitch-yaw: R = Rx(φ) · Ry(θ) · Rz(ψ)
    XYZ,
}

/// Rotation matrix from Euler angles with specified convention.
pub fn rot_euler(phi: &Ex, theta: &Ex, psi: &Ex, convention: EulerConvention) -> Matrix {
    match convention {
        // R = Rz(phi) * Ry(theta) * Rx(psi)
        EulerConvention::ZYX => mul_square(&mul_square(&rot_z(phi), &rot_y(theta)), &rot_x(psi)),
        EulerConvention::ZXZ => mul_square(&mul_square(&rot_z(phi), &rot_x(theta)), &rot_z(psi)),
        EulerConvention::XYZ => mul_square(&mul_square(&rot_x(phi), &rot_y(theta)), &rot_z(psi)),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 2-DOF Planar Inverse Kinematics
// ═══════════════════════════════════════════════════════════════════════════

/// Relative tolerance of the reachability tests of
/// [`inverse_kinematics_2dof`], in units of the `f64` epsilon: covers the
/// rounding of the inputs themselves (`0.8` is not `4/5`, so `1 + 0.8` is
/// not `1.8`) and of the few operations that combine them.
const IK_TOLERANCE_ULPS: f64 = 64.0;

/// Is `a = b` within the tolerance, relative to the magnitude `scale` of
/// the terms they were computed from?
fn ik_equal(a: f64, b: f64, scale: f64) -> bool {
    (a - b).abs() <= IK_TOLERANCE_ULPS * f64::EPSILON * scale
}

/// Solve 2-DOF planar inverse kinematics in closed form.
///
/// Given link lengths (`l1`, `l2`) and a target end-effector position
/// (`target_x`, `target_y`), finds all joint angle pairs (θ₁, θ₂)
/// satisfying the forward-kinematics equations:
///
/// ```text
///   l1·cos(θ₁) + l2·cos(θ₁+θ₂) = target_x
///   l1·sin(θ₁) + l2·sin(θ₁+θ₂) = target_y
/// ```
///
/// By the law of cosines `cos θ₂ = (r² − l1² − l2²)/(2·l1·l2)` with
/// `r² = x² + y²`, and then the target is the vector `(l1 + l2·cos θ₂,
/// l2·sin θ₂)` rotated by `θ₁`, which gives `θ₁` by `atan2`.  An interior
/// target has the two solutions `θ₂ = ±acos(…)` (elbow up and down), a
/// target on the boundary of the workspace (`r = |l1 ± l2|`, the arm
/// stretched or folded) has one, an unreachable one none.  Reachability is
/// decided with a tolerance of a few dozen ulps of the inputs, so that
/// `(1.8, 0)` with links `1` and `0.8` counts as the stretched arm.
///
/// Where the solutions form a continuum — the target at the origin with
/// `|l1| = |l2|` (any `θ₁`), or a link of length zero (the angle of the
/// other joint is free) — one representative is returned, with the free
/// angle `0`.
///
/// Angles are in radians, in `(−π, π]`, sorted by `(θ₁, θ₂)`.  Returns
/// an empty vec if the target is unreachable or any input is not a finite
/// number.  (Up to 0.29 this solved a Gröbner system over ℚ, which finds
/// only the solutions whose sines and cosines are rational: a reachable
/// target such as `(1.5, 0.5)` with unit links came back empty.)
///
/// # Examples
///
/// ```
/// use symplex::robotics::inverse_kinematics_2dof;
///
/// let sols = inverse_kinematics_2dof(1.0, 1.0, 1.5, 0.5);
/// assert_eq!(sols.len(), 2);
/// for (t1, t2) in sols {
///     assert!((t1.cos() + (t1 + t2).cos() - 1.5).abs() < 1e-12);
///     assert!((t1.sin() + (t1 + t2).sin() - 0.5).abs() < 1e-12);
/// }
/// assert_eq!(inverse_kinematics_2dof(1.0, 0.8, 1.8, 0.0), vec![(0.0, 0.0)]);
/// assert!(inverse_kinematics_2dof(1.0, 1.0, 2.5, 0.0).is_empty());
/// ```
pub fn inverse_kinematics_2dof(l1: f64, l2: f64, target_x: f64, target_y: f64) -> Vec<(f64, f64)> {
    if ![l1, l2, target_x, target_y].iter().all(|v| v.is_finite()) {
        return vec![];
    }
    // The angles are scale-invariant: normalise to avoid overflow and
    // underflow in the squares.
    let scale = [l1, l2, target_x, target_y]
        .iter()
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    if scale == 0.0 {
        // Two zero-length links at the origin: every pair is a solution.
        return vec![(0.0, 0.0)];
    }
    let (l1, l2, x, y) = (l1 / scale, l2 / scale, target_x / scale, target_y / scale);
    let r2 = x * x + y * y;
    let mut angles: Vec<(f64, f64)> = Vec::new();
    // The angle of the target seen through a link of signed length `l`.
    let direction = |l: f64| (l.signum() * y).atan2(l.signum() * x);
    if l1 == 0.0 || l2 == 0.0 {
        // One free joint: reachable iff r = |l| for the other link.
        let l = if l1 == 0.0 { l2 } else { l1 };
        if l == 0.0 {
            return vec![];
        }
        if !ik_equal(r2, l * l, r2 + l * l) {
            return vec![];
        }
        let phi = direction(l);
        return vec![if l1 == 0.0 { (0.0, phi) } else { (phi, 0.0) }];
    }
    let num = r2 - l1 * l1 - l2 * l2;
    let den = 2.0 * l1 * l2;
    let mag = r2 + l1 * l1 + l2 * l2;
    // cos θ₂ = num/den, classified against ±1 without dividing.
    let cos_t2 = if ik_equal(num, den.abs(), mag) {
        den.signum()
    } else if ik_equal(num, -den.abs(), mag) {
        -den.signum()
    } else if num.abs() > den.abs() {
        return vec![];
    } else {
        num / den
    };
    if cos_t2.abs() == 1.0 {
        // Stretched or folded: θ₂ ∈ {0, π}, one solution.
        let t2 = if cos_t2 > 0.0 {
            0.0
        } else {
            std::f64::consts::PI
        };
        let k1 = l1 + l2 * cos_t2;
        if ik_equal(k1, 0.0, l1.abs() + l2.abs()) {
            // Folded onto the origin (|l1| = |l2|, r = 0): θ₁ is free.
            angles.push((0.0, t2));
        } else {
            angles.push((direction(k1), t2));
        }
    } else {
        let sin_abs = (1.0 - cos_t2 * cos_t2).sqrt();
        for sin_t2 in [sin_abs, -sin_abs] {
            let (k1, k2) = (l1 + l2 * cos_t2, l2 * sin_t2);
            // (x, y) = R(θ₁)·(k1, k2)
            let t1 = (k1 * y - k2 * x).atan2(k1 * x + k2 * y);
            angles.push((t1, sin_t2.atan2(cos_t2)));
        }
    }
    angles.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    angles
}

// ═══════════════════════════════════════════════════════════════════════════
// Dimension-typed robotics API
// ═══════════════════════════════════════════════════════════════════════════

/// The four standard Denavit-Hartenberg parameters of one link, with
/// compile-time dimensional checking: the two angles are [`Angle`], the two
/// lengths are [`Length`].
///
/// Putting an angle in a length slot is a compile error, and the named
/// fields keep the two lengths (`d`, `a`) and the two angles (`theta`,
/// `alpha`) from being transposed.  The fields have the same meaning as on
/// the symbolic [`DhLink`]; `DhLink::from(params)` erases the units.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::units::si::*;
/// use symplex::robotics::{DhLink, DhParams};
///
/// let ctx = Context::new();
/// let theta = Angle::symbol(&ctx, "theta");
/// let l = Length::rational(&ctx, 3, 10);
/// let zero_l = Length::zero(&ctx);
/// let zero_a = Angle::zero(&ctx);
///
/// let params = DhParams { theta: &theta, d: &zero_l, a: &l, alpha: &zero_a };
/// let link = DhLink::from(params);
/// assert_eq!(format!("{}", link.a), "3/10");
/// ```
#[derive(Clone, Copy, Debug)]
pub struct DhParams<'a> {
    /// θᵢ — joint angle: rotation about the previous z-axis zᵢ₋₁ (the joint
    /// variable of a revolute joint).
    pub theta: &'a Angle,
    /// dᵢ — link offset: translation along the previous z-axis zᵢ₋₁ (the
    /// joint variable of a prismatic joint).
    pub d: &'a Length,
    /// aᵢ — link length: translation along the new x-axis xᵢ.
    pub a: &'a Length,
    /// αᵢ — link twist: rotation about the new x-axis xᵢ.
    pub alpha: &'a Angle,
}

impl<'a> From<DhParams<'a>> for DhLink<'a> {
    /// Erase the units: each field becomes its underlying `Ex`.
    fn from(params: DhParams<'a>) -> Self {
        DhLink {
            theta: params.theta.inner(),
            d: params.d.inner(),
            a: params.a.inner(),
            alpha: params.alpha.inner(),
        }
    }
}

fn untyped_links<'a>(dh_params: &[DhParams<'a>]) -> Vec<DhLink<'a>> {
    dh_params.iter().map(|&p| DhLink::from(p)).collect()
}

/// Compute end-effector position from typed DH parameters.
///
/// Returns `(x, y, z)` as `Length` values — compile-time guaranteed to be
/// lengths, not angles or velocities.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::units::si::*;
/// use symplex::robotics::{fk_position_typed, DhParams};
///
/// let ctx = Context::new();
/// let theta1 = Angle::symbol(&ctx, "theta1");
/// let l1 = Length::rational(&ctx, 3, 10);
/// let zero_l = Length::zero(&ctx);
/// let zero_a = Angle::zero(&ctx);
///
/// let (px, py, pz) = fk_position_typed(&[
///     DhParams { theta: &theta1, d: &zero_l, a: &l1, alpha: &zero_a },
/// ]);
/// // px, py, pz are Length — guaranteed at compile time
/// assert_eq!(format!("{}", px.inner()), "3/10*cos(theta1)");
/// assert_eq!(format!("{}", py.inner()), "3/10*sin(theta1)");
/// assert_eq!(format!("{}", pz.inner()), "0");
/// ```
pub fn fk_position_typed(dh_params: &[DhParams<'_>]) -> (Length, Length, Length) {
    let (px, py, pz) = fk_position(&untyped_links(dh_params));
    (
        Length::from_ex(px),
        Length::from_ex(py),
        Length::from_ex(pz),
    )
}

/// Compute the full 4×4 FK transformation matrix from typed DH parameters.
pub fn fk_chain_typed(dh_params: &[DhParams<'_>]) -> Matrix {
    fk_chain(&untyped_links(dh_params))
}

/// Compute the 3×3 rotation matrix from typed DH parameters.
pub fn fk_rotation_typed(dh_params: &[DhParams<'_>]) -> Matrix {
    fk_rotation(&untyped_links(dh_params))
}
