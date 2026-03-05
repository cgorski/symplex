//! Robotics kinematics helpers: DH parameters, forward kinematics, Jacobian.
//!
//! # Denavit-Hartenberg Convention
//!
//! Each joint in a serial robot arm is described by four parameters:
//! - θ (theta): joint angle (variable for revolute joints)
//! - d: link offset along z-axis
//! - a: link length along x-axis
//! - α (alpha): link twist angle
//!
//! These produce a 4×4 homogeneous transformation matrix per joint.
//! Chaining these matrices gives the forward kinematics.

use crate::matrix::Matrix;
use crate::prelude::*;

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
/// or numeric constants built with [`crate::int`].
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::robotics::dh_matrix;
///
/// // Identity-like DH matrix (all parameters zero)
/// let zero = symplex::int(0);
/// let t = dh_matrix(&zero, &zero, &zero, &zero);
/// assert_eq!(t.nrows(), 4);
/// assert_eq!(t.ncols(), 4);
/// ```
pub fn dh_matrix(theta: &Ex, d: &Ex, a: &Ex, alpha: &Ex) -> Matrix {
    let cos_theta = theta.cos();
    let sin_theta = theta.sin();
    let cos_alpha = alpha.cos();
    let sin_alpha = alpha.sin();

    let zero = crate::int(0);
    let one = crate::int(1);

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

    Matrix::new(vec![
        vec![r00, r01, r02, r03],
        vec![r10, r11, r12, r13],
        vec![r20, r21, r22, r23],
        vec![r30, r31, r32, r33],
    ])
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
/// use symplex::robotics::{dh_matrix, fk_chain};
///
/// let theta1 = symplex::var("theta1");
/// let zero = symplex::int(0);
/// let l1 = symplex::var("L1");
///
/// let params = [(&theta1, &zero, &l1, &zero)];
/// let t = fk_chain(&params);
/// assert_eq!(t.shape(), (4, 4));
/// ```
pub fn fk_chain(dh_params: &[(&Ex, &Ex, &Ex, &Ex)]) -> Matrix {
    let mut result = Matrix::identity(4);
    for &(theta, d, a, alpha) in dh_params {
        let ti = dh_matrix(theta, d, a, alpha);
        result = result.matmul(&ti);
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
/// use symplex::robotics::fk_position;
///
/// let theta = symplex::var("theta");
/// let zero = symplex::int(0);
/// let l = symplex::var("L");
///
/// let (x, y, z) = fk_position(&[(&theta, &zero, &l, &zero)]);
/// // x, y, z are symbolic expressions
/// ```
pub fn fk_position(dh_params: &[(&Ex, &Ex, &Ex, &Ex)]) -> (Ex, Ex, Ex) {
    let t = fk_chain(dh_params);
    (
        t.get(0, 3).clone(),
        t.get(1, 3).clone(),
        t.get(2, 3).clone(),
    )
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
/// use symplex::robotics::fk_rotation;
///
/// let theta = symplex::var("theta");
/// let zero = symplex::int(0);
/// let l = symplex::var("L");
///
/// let r = fk_rotation(&[(&theta, &zero, &l, &zero)]);
/// assert_eq!(r.shape(), (3, 3));
/// ```
pub fn fk_rotation(dh_params: &[(&Ex, &Ex, &Ex, &Ex)]) -> Matrix {
    let t = fk_chain(dh_params);
    Matrix::new(vec![
        vec![
            t.get(0, 0).clone(),
            t.get(0, 1).clone(),
            t.get(0, 2).clone(),
        ],
        vec![
            t.get(1, 0).clone(),
            t.get(1, 1).clone(),
            t.get(1, 2).clone(),
        ],
        vec![
            t.get(2, 0).clone(),
            t.get(2, 1).clone(),
            t.get(2, 2).clone(),
        ],
    ])
}

/// Rotation matrix about the x-axis by angle θ.
///
/// ```text
/// Rx(θ) = | 1    0       0    |
///          | 0  cos(θ)  -sin(θ)|
///          | 0  sin(θ)   cos(θ)|
/// ```
pub fn rot_x(theta: &Ex) -> Matrix {
    let zero = crate::int(0);
    let one = crate::int(1);
    let c = theta.cos();
    let s = theta.sin();
    Matrix::new(vec![
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
    let zero = crate::int(0);
    let one = crate::int(1);
    let c = theta.cos();
    let s = theta.sin();
    Matrix::new(vec![
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
    let zero = crate::int(0);
    let one = crate::int(1);
    let c = theta.cos();
    let s = theta.sin();
    Matrix::new(vec![
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
    let zero = crate::int(0);
    Matrix::new(vec![
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
pub fn homogeneous(rotation: &Matrix, position: &[Ex; 3]) -> Matrix {
    assert_eq!(rotation.shape(), (3, 3), "rotation must be 3×3");
    let zero = crate::int(0);
    let one = crate::int(1);
    Matrix::new(vec![
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
    ])
}

/// Pure translation as a 4×4 homogeneous transformation matrix.
///
/// ```text
/// T = | I  p |
///     | 0  1 |
/// ```
pub fn translation(x: &Ex, y: &Ex, z: &Ex) -> Matrix {
    let zero = crate::int(0);
    let one = crate::int(1);
    Matrix::new(vec![
        vec![one.clone(), zero.clone(), zero.clone(), x.clone()],
        vec![zero.clone(), one.clone(), zero.clone(), y.clone()],
        vec![zero.clone(), zero.clone(), one, z.clone()],
        vec![zero.clone(), zero.clone(), zero, crate::int(1)],
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
        EulerConvention::ZYX => {
            // R = Rz(phi) * Ry(theta) * Rx(psi)
            rot_z(phi).matmul(&rot_y(theta)).matmul(&rot_x(psi))
        }
        EulerConvention::ZXZ => {
            rot_z(phi).matmul(&rot_x(theta)).matmul(&rot_z(psi))
        }
        EulerConvention::XYZ => {
            rot_x(phi).matmul(&rot_y(theta)).matmul(&rot_z(psi))
        }
    }
}
