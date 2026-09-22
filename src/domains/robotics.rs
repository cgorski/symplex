//! Robotics kinematics helpers: DH parameters, forward kinematics, Jacobian,
//! and algebraic inverse kinematics.
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

use crate::base::numeric::Q;
use crate::domains::matrix::Matrix;
use crate::poly::multipoly::{GrevLex, MultiPoly};
use crate::poly::polysys;
use crate::prelude::*;
use crate::units::si::{Angle, Length};
use num_traits::ToPrimitive;

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
    Matrix::from_fn(n, n, |i, j| {
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
    let mut result = Matrix::identity(&ctx, 4);
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
    Matrix::from_fn(3, 3, |i, j| t.get(i, j).clone())
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

/// Best rational approximation of an `f64` with denominator ≤ 10⁶
/// (continued-fraction convergents), so `0.1 → 1/10` and `1/3` round-trips.
///
/// Returns `None` for NaN / ±∞.
fn f64_to_ratio(v: f64) -> Option<Q> {
    crate::base::numeric::f64_to_ratio_approx(v, 1_000_000)
}

/// Solve 2-DOF planar inverse kinematics algebraically.
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
/// Uses the sin/cos ring approach with Gröbner bases. Introduces four
/// polynomial variables `(s₁, c₁, s₂, c₂)` representing `sin(θ₁)`,
/// `cos(θ₁)`, `sin(θ₂)`, `cos(θ₂)`, together with Pythagorean
/// constraints `s₁²+c₁²=1` and `s₂²+c₂²=1`.
///
/// The angle-addition identities expand the FK equations:
/// ```text
///   cos(θ₁+θ₂) = c₁·c₂ − s₁·s₂
///   sin(θ₁+θ₂) = s₁·c₂ + c₁·s₂
/// ```
///
/// Returns all solution branches as `(θ₁, θ₂)` pairs in radians.
/// Returns an empty vec if the target is unreachable or any input is
/// not a finite number.
pub fn inverse_kinematics_2dof(l1: f64, l2: f64, target_x: f64, target_y: f64) -> Vec<(f64, f64)> {
    // Variables: 0=s1, 1=c1, 2=s2, 3=c2
    let nv = 4;

    let (Some(rl1), Some(rl2), Some(rtx), Some(rty)) = (
        f64_to_ratio(l1),
        f64_to_ratio(l2),
        f64_to_ratio(target_x),
        f64_to_ratio(target_y),
    ) else {
        return vec![];
    };

    let s1 = MultiPoly::<GrevLex>::var(nv, 0);
    let c1 = MultiPoly::<GrevLex>::var(nv, 1);
    let s2 = MultiPoly::<GrevLex>::var(nv, 2);
    let c2 = MultiPoly::<GrevLex>::var(nv, 3);
    let one = MultiPoly::<GrevLex>::from_int(nv, 1);

    // Pythagorean constraints: s1^2 + c1^2 - 1 = 0, s2^2 + c2^2 - 1 = 0
    let pyth1 = &(&s1 * &s1) + &(&c1 * &c1) - one.clone();
    let pyth2 = &(&s2 * &s2) + &(&c2 * &c2) - one;

    // FK x-equation: l1*c1 + l2*(c1*c2 - s1*s2) - tx = 0
    let fk_x = {
        let term1 = c1.scale(&rl1);
        let cos12 = &(&c1 * &c2) - &(&s1 * &s2); // c1*c2 - s1*s2
        let term2 = cos12.scale(&rl2);
        let target_poly = MultiPoly::<GrevLex>::constant(nv, rtx.clone());
        &(&term1 + &term2) - &target_poly
    };

    // FK y-equation: l1*s1 + l2*(s1*c2 + c1*s2) - ty = 0
    let fk_y = {
        let term1 = s1.scale(&rl1);
        let sin12 = &(&s1 * &c2) + &(&c1 * &s2); // s1*c2 + c1*s2
        let term2 = sin12.scale(&rl2);
        let target_poly = MultiPoly::<GrevLex>::constant(nv, rty.clone());
        &(&term1 + &term2) - &target_poly
    };

    let system = vec![fk_x, fk_y, pyth1, pyth2];

    let solutions = match polysys::solve_polynomial_system(&system) {
        Ok(sols) => sols,
        Err(_) => return vec![],
    };

    // Convert (s1, c1, s2, c2) → (θ₁, θ₂)
    let mut angles = Vec::new();
    for sol in &solutions {
        if sol.len() != 4 {
            continue;
        }
        let s1_val = sol[0].to_f64().unwrap_or(0.0);
        let c1_val = sol[1].to_f64().unwrap_or(0.0);
        let s2_val = sol[2].to_f64().unwrap_or(0.0);
        let c2_val = sol[3].to_f64().unwrap_or(0.0);

        let theta1 = s1_val.atan2(c1_val);
        let theta2 = s2_val.atan2(c2_val);

        angles.push((theta1, theta2));
    }

    // Deduplicate solutions that are numerically close
    angles.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    });
    angles.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-6 && (a.1 - b.1).abs() < 1e-6);

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
