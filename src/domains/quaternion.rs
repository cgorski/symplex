//! Symbolic quaternion algebra for attitude representation.
//!
//! Quaternions are represented as q = w + xi + yj + zk where w, x, y, z
//! are symbolic expressions (`Ex`). This module provides:
//!
//! - Hamilton product (quaternion multiplication)
//! - Conjugate, norm, inverse
//! - Conversion to/from 3×3 rotation matrices
//! - Conversion to/from axis-angle representation
//! - Quaternion kinematics (q̇ = ½q⊗ω)

use crate::domains::matrix::Matrix;
use crate::prelude::*;

/// A symbolic quaternion q = w + xi + yj + zk.
///
/// Components are symbolic expressions, enabling both concrete
/// (q = 1 + 0i + 0j + 0k) and symbolic (q = cos(θ/2) + sin(θ/2)·k̂) use.
#[derive(Clone, Debug)]
pub struct Quaternion {
    /// Scalar (real) part
    pub w: Ex,
    /// i component
    pub x: Ex,
    /// j component
    pub y: Ex,
    /// k component
    pub z: Ex,
}

impl Quaternion {
    /// Create a quaternion from four symbolic expressions.
    pub fn new(w: Ex, x: Ex, y: Ex, z: Ex) -> Self {
        Quaternion { w, x, y, z }
    }

    /// Identity quaternion (1, 0, 0, 0).
    pub fn identity() -> Self {
        Quaternion {
            w: crate::int(1),
            x: crate::int(0),
            y: crate::int(0),
            z: crate::int(0),
        }
    }

    /// Zero quaternion.
    pub fn zero() -> Self {
        Quaternion {
            w: crate::int(0),
            x: crate::int(0),
            y: crate::int(0),
            z: crate::int(0),
        }
    }

    /// Pure quaternion (0, x, y, z) from a 3-vector.
    pub fn from_vector(x: &Ex, y: &Ex, z: &Ex) -> Self {
        Quaternion {
            w: crate::int(0),
            x: x.clone(),
            y: y.clone(),
            z: z.clone(),
        }
    }

    /// Hamilton product: self ⊗ other.
    ///
    /// (a₁ + b₁i + c₁j + d₁k)(a₂ + b₂i + c₂j + d₂k) =
    ///   (a₁a₂ - b₁b₂ - c₁c₂ - d₁d₂) +
    ///   (a₁b₂ + b₁a₂ + c₁d₂ - d₁c₂)i +
    ///   (a₁c₂ - b₁d₂ + c₁a₂ + d₁b₂)j +
    ///   (a₁d₂ + b₁c₂ - c₁b₂ + d₁a₂)k
    pub fn mul(&self, other: &Quaternion) -> Quaternion {
        let a1 = &self.w;
        let b1 = &self.x;
        let c1 = &self.y;
        let d1 = &self.z;
        let a2 = &other.w;
        let b2 = &other.x;
        let c2 = &other.y;
        let d2 = &other.z;

        // w = a1*a2 - b1*b2 - c1*c2 - d1*d2
        let w = &(a1 * a2) - &(b1 * b2) - &(c1 * c2) - &(d1 * d2);

        // x = a1*b2 + b1*a2 + c1*d2 - d1*c2
        let x = &(a1 * b2) + &(b1 * a2) + &(c1 * d2) - &(d1 * c2);

        // y = a1*c2 - b1*d2 + c1*a2 + d1*b2
        let y = &(a1 * c2) - &(b1 * d2) + &(c1 * a2) + &(d1 * b2);

        // z = a1*d2 + b1*c2 - c1*b2 + d1*a2
        let z = &(a1 * d2) + &(b1 * c2) - &(c1 * b2) + &(d1 * a2);

        Quaternion { w, x, y, z }
    }

    /// Quaternion conjugate: q* = w - xi - yj - zk.
    pub fn conjugate(&self) -> Quaternion {
        Quaternion {
            w: self.w.clone(),
            x: -&self.x,
            y: -&self.y,
            z: -&self.z,
        }
    }

    /// Squared norm: |q|² = w² + x² + y² + z².
    pub fn norm_squared(&self) -> Ex {
        &self.w.powi(2) + &self.x.powi(2) + &self.y.powi(2) + &self.z.powi(2)
    }

    /// Norm: |q| = √(w² + x² + y² + z²).
    pub fn norm(&self) -> Ex {
        self.norm_squared().sqrt()
    }

    /// Inverse: q⁻¹ = q* / |q|².
    pub fn inverse(&self) -> Quaternion {
        let conj = self.conjugate();
        let n2 = self.norm_squared();
        Quaternion {
            w: &conj.w / &n2,
            x: &conj.x / &n2,
            y: &conj.y / &n2,
            z: &conj.z / &n2,
        }
    }

    /// Normalize to unit quaternion: q / |q|.
    pub fn normalize(&self) -> Quaternion {
        let n = self.norm();
        Quaternion {
            w: &self.w / &n,
            x: &self.x / &n,
            y: &self.y / &n,
            z: &self.z / &n,
        }
    }

    /// Convert to 3×3 rotation matrix.
    ///
    /// Assumes unit quaternion. The rotation matrix is:
    /// ```text
    /// R = | 1-2(y²+z²)    2(xy-wz)     2(xz+wy)  |
    ///     | 2(xy+wz)      1-2(x²+z²)   2(yz-wx)  |
    ///     | 2(xz-wy)      2(yz+wx)     1-2(x²+y²) |
    /// ```
    pub fn to_rotation_matrix(&self) -> Matrix {
        let one = crate::int(1);
        let two = crate::int(2);

        let xx = self.x.powi(2);
        let yy = self.y.powi(2);
        let zz = self.z.powi(2);
        let xy = &self.x * &self.y;
        let xz = &self.x * &self.z;
        let yz = &self.y * &self.z;
        let wx = &self.w * &self.x;
        let wy = &self.w * &self.y;
        let wz = &self.w * &self.z;

        // R[0][0] = 1 - 2*(y² + z²)
        let r00 = &one - &(&two * &(&yy + &zz));
        // R[0][1] = 2*(xy - wz)
        let r01 = &two * &(&xy - &wz);
        // R[0][2] = 2*(xz + wy)
        let r02 = &two * &(&xz + &wy);

        // R[1][0] = 2*(xy + wz)
        let r10 = &two * &(&xy + &wz);
        // R[1][1] = 1 - 2*(x² + z²)
        let r11 = &one - &(&two * &(&xx + &zz));
        // R[1][2] = 2*(yz - wx)
        let r12 = &two * &(&yz - &wx);

        // R[2][0] = 2*(xz - wy)
        let r20 = &two * &(&xz - &wy);
        // R[2][1] = 2*(yz + wx)
        let r21 = &two * &(&yz + &wx);
        // R[2][2] = 1 - 2*(x² + y²)
        let r22 = &one - &(&two * &(&xx + &yy));

        Matrix::new(vec![
            vec![r00, r01, r02],
            vec![r10, r11, r12],
            vec![r20, r21, r22],
        ]).unwrap()
    }

    /// Create a quaternion from an axis-angle representation.
    ///
    /// q = cos(θ/2) + sin(θ/2)·(axisₓi + axisᵧj + axis_zk)
    /// Assumes axis is a unit vector.
    pub fn from_axis_angle(
        axis_x: &Ex,
        axis_y: &Ex,
        axis_z: &Ex,
        angle: &Ex,
    ) -> Quaternion {
        let two = crate::int(2);
        let half_angle = angle / &two;
        let c = half_angle.cos();
        let s = half_angle.sin();

        Quaternion {
            w: c,
            x: &s * axis_x,
            y: &s * axis_y,
            z: &s * axis_z,
        }
    }

    /// Quaternion derivative for angular velocity.
    ///
    /// q̇ = ½ · q ⊗ ω_quat
    /// where ω_quat = Quaternion::from_vector(ωx, ωy, ωz)
    pub fn angular_velocity_derivative(
        &self,
        omega_x: &Ex,
        omega_y: &Ex,
        omega_z: &Ex,
    ) -> Quaternion {
        let omega_quat = Quaternion::from_vector(omega_x, omega_y, omega_z);
        let product = self.mul(&omega_quat);
        let half = crate::rational(1, 2);
        Quaternion {
            w: &half * &product.w,
            x: &half * &product.x,
            y: &half * &product.y,
            z: &half * &product.z,
        }
    }

    /// Substitute a variable in all four components.
    pub fn subs(&self, old: &Ex, new: &Ex) -> Quaternion {
        Quaternion {
            w: self.w.subs(old, new),
            x: self.x.subs(old, new),
            y: self.y.subs(old, new),
            z: self.z.subs(old, new),
        }
    }

    /// Evaluate all components.
    pub fn eval(&self) -> Quaternion {
        Quaternion {
            w: self.w.eval(),
            x: self.x.eval(),
            y: self.y.eval(),
            z: self.z.eval(),
        }
    }

    /// Simplify all components.
    pub fn simplify(&self) -> Quaternion {
        Quaternion {
            w: self.w.simplify(),
            x: self.x.simplify(),
            y: self.y.simplify(),
            z: self.z.simplify(),
        }
    }

    /// Render this quaternion as LaTeX.
    ///
    /// # Example
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::quaternion::Quaternion;
    /// let q = Quaternion::identity();
    /// let latex = q.to_latex();
    /// assert!(latex.contains(r"\mathbf{i}"));
    /// ```
    pub fn to_latex(&self) -> String {
        format!(
            "{} + {}\\mathbf{{i}} + {}\\mathbf{{j}} + {}\\mathbf{{k}}",
            self.w.to_latex(),
            self.x.to_latex(),
            self.y.to_latex(),
            self.z.to_latex()
        )
    }
}

impl std::fmt::Display for Quaternion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({} + {}i + {}j + {}k)", self.w, self.x, self.y, self.z)
    }
}
