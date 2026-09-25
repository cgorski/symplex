//! Symbolic quaternion algebra for attitude representation.
//!
//! Quaternions are represented as `q = w + x·i + y·j + z·k` where `w, x,
//! y, z` are symbolic expressions ([`Ex`]). This module provides:
//!
//! - Hamilton product (`*`), addition/subtraction, scalar multiplication
//! - Conjugate, norm, inverse, normalisation, dot product
//! - Conversion to/from 3×3 rotation matrices (Shepperd's method)
//! - Conversion to/from axis–angle and Euler angles
//!   ([`EulerConvention`] from the robotics module)
//! - Rotation of vectors, spherical linear interpolation (slerp)
//! - Quaternion exponential, logarithm and powers
//! - Quaternion kinematics (`q̇ = ½ q ⊗ ω`) and differentiation
//!
//! # Conventions
//!
//! * Rotations act on column vectors as `v' = q ⊗ (0, v) ⊗ q*`, which
//!   matches [`to_rotation_matrix`](Quaternion::to_rotation_matrix)
//!   (`v' = R v`) and the right-handed `rot_x/rot_y/rot_z` matrices of
//!   [`robotics`](crate::domains::robotics).
//! * Methods documented as *assuming a unit quaternion* do not normalise
//!   their input; call [`normalize`](Quaternion::normalize) first when the
//!   norm is not provably one.

use crate::api::expr::Ex;
use crate::base::errors::SymplexError;
use crate::domains::matrix::{Matrix, all3, ex_is_positive, ex_is_zero, sqrt_rationalized};
pub use crate::domains::robotics::EulerConvention;

fn invalid(operation: &'static str, reason: impl Into<String>) -> SymplexError {
    SymplexError::invalid_argument(operation, reason)
}

fn failed(operation: &'static str, reason: impl Into<String>) -> SymplexError {
    SymplexError::computation_failed(operation, reason)
}

/// A symbolic quaternion `q = w + x·i + y·j + z·k`.
///
/// Components are symbolic expressions, enabling both concrete
/// (`q = 1 + 0i + 0j + 0k`) and symbolic (`q = cos(θ/2) + sin(θ/2)·k̂`) use.
/// Equality is structural (same canonical expressions); use
/// [`equals`](Self::equals) for a simplifying comparison.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
///
/// let ctx = Context::new();
/// let i = Quaternion::new(ctx.int(0), ctx.int(1), ctx.int(0), ctx.int(0));
/// let j = Quaternion::new(ctx.int(0), ctx.int(0), ctx.int(1), ctx.int(0));
/// let k = Quaternion::new(ctx.int(0), ctx.int(0), ctx.int(0), ctx.int(1));
/// assert_eq!(&i * &j, k);              // i·j = k
/// assert_eq!(&j * &i, -&k);            // j·i = −k
/// assert_eq!(&i * &i, -&Quaternion::identity(&ctx));
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
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

    /// Identity quaternion `(1, 0, 0, 0)`.
    pub fn identity(ctx: &crate::api::context::Context) -> Self {
        Quaternion {
            w: ctx.int(1),
            x: ctx.int(0),
            y: ctx.int(0),
            z: ctx.int(0),
        }
    }

    /// Zero quaternion.
    pub fn zero(ctx: &crate::api::context::Context) -> Self {
        Quaternion {
            w: ctx.int(0),
            x: ctx.int(0),
            y: ctx.int(0),
            z: ctx.int(0),
        }
    }

    /// Pure quaternion `(0, x, y, z)` from three components.
    pub fn from_vector(x: &Ex, y: &Ex, z: &Ex) -> Self {
        Quaternion {
            w: x.context().int(0),
            x: x.clone(),
            y: y.clone(),
            z: z.clone(),
        }
    }

    /// Pure quaternion `(0, v)` from a 3×1 column vector.
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::InvalidArgument`] if `v` is not 3×1.
    pub fn from_vector_matrix(v: &Matrix) -> Result<Self, SymplexError> {
        if v.shape() != (3, 1) {
            return Err(invalid(
                "Quaternion::from_vector_matrix",
                format!(
                    "expected a 3×1 column vector, got {}×{}",
                    v.nrows(),
                    v.ncols()
                ),
            ));
        }
        Ok(Quaternion::from_vector(
            v.get(0, 0),
            v.get(1, 0),
            v.get(2, 0),
        ))
    }

    /// The [`Context`](crate::api::context::Context) this quaternion lives in.
    pub fn context(&self) -> crate::api::context::Context {
        self.w.context()
    }

    /// The scalar part `w`.
    pub fn scalar(&self) -> &Ex {
        &self.w
    }

    /// The vector part `(x, y, z)` as a 3×1 column vector.
    pub fn vector(&self) -> Matrix {
        Matrix::col_vector(vec![self.x.clone(), self.y.clone(), self.z.clone()])
    }

    /// Components as `[w, x, y, z]`.
    pub fn components(&self) -> [&Ex; 4] {
        [&self.w, &self.x, &self.y, &self.z]
    }

    /// Apply `f` to each component.
    pub fn map(&self, mut f: impl FnMut(&Ex) -> Ex) -> Quaternion {
        Quaternion {
            w: f(&self.w),
            x: f(&self.x),
            y: f(&self.y),
            z: f(&self.z),
        }
    }

    // ── Algebra ────────────────────────────────────────────────────────

    /// Hamilton product `self ⊗ other` (also available as the `*` operator).
    ///
    /// ```text
    /// (a₁ + b₁i + c₁j + d₁k)(a₂ + b₂i + c₂j + d₂k) =
    ///   (a₁a₂ − b₁b₂ − c₁c₂ − d₁d₂) +
    ///   (a₁b₂ + b₁a₂ + c₁d₂ − d₁c₂)i +
    ///   (a₁c₂ − b₁d₂ + c₁a₂ + d₁b₂)j +
    ///   (a₁d₂ + b₁c₂ − c₁b₂ + d₁a₂)k
    /// ```
    pub fn mul(&self, other: &Quaternion) -> Quaternion {
        let (a1, b1, c1, d1) = (&self.w, &self.x, &self.y, &self.z);
        let (a2, b2, c2, d2) = (&other.w, &other.x, &other.y, &other.z);
        Quaternion {
            w: &(a1 * a2) - &(b1 * b2) - &(c1 * c2) - &(d1 * d2),
            x: &(a1 * b2) + &(b1 * a2) + &(c1 * d2) - &(d1 * c2),
            y: &(a1 * c2) - &(b1 * d2) + &(c1 * a2) + &(d1 * b2),
            z: &(a1 * d2) + &(b1 * c2) - &(c1 * b2) + &(d1 * a2),
        }
    }

    /// Component-wise sum (also the `+` operator).
    pub fn add(&self, other: &Quaternion) -> Quaternion {
        Quaternion {
            w: &self.w + &other.w,
            x: &self.x + &other.x,
            y: &self.y + &other.y,
            z: &self.z + &other.z,
        }
    }

    /// Component-wise difference (also the `-` operator).
    pub fn sub(&self, other: &Quaternion) -> Quaternion {
        Quaternion {
            w: &self.w - &other.w,
            x: &self.x - &other.x,
            y: &self.y - &other.y,
            z: &self.z - &other.z,
        }
    }

    /// Multiply every component by a scalar.
    pub fn scale(&self, s: &Ex) -> Quaternion {
        self.map(|c| s * c)
    }

    /// Quaternion conjugate `q* = w − xi − yj − zk`.
    pub fn conjugate(&self) -> Quaternion {
        Quaternion {
            w: self.w.clone(),
            x: -&self.x,
            y: -&self.y,
            z: -&self.z,
        }
    }

    /// Four-dimensional dot product `w₁w₂ + x₁x₂ + y₁y₂ + z₁z₂`.
    pub fn dot(&self, other: &Quaternion) -> Ex {
        &(&self.w * &other.w)
            + &(&self.x * &other.x)
            + &(&self.y * &other.y)
            + &(&self.z * &other.z)
    }

    /// Squared norm `|q|² = w² + x² + y² + z²`.
    pub fn norm_squared(&self) -> Ex {
        &self.w.powi(2) + &self.x.powi(2) + &self.y.powi(2) + &self.z.powi(2)
    }

    /// Norm `|q| = √(w² + x² + y² + z²)`.
    pub fn norm(&self) -> Ex {
        self.norm_squared().sqrt()
    }

    /// Is `|q| = 1`?  Three-valued: `None` when the norm cannot be decided
    /// symbolically.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let th = ctx.symbol("theta");
    /// let q = Quaternion::from_axis_angle(&ctx.int(0), &ctx.int(0), &ctx.int(1), &th);
    /// assert_eq!(q.is_unit(), Some(true));
    /// assert_eq!(Quaternion::new(ctx.int(2), ctx.int(0), ctx.int(0), ctx.int(0)).is_unit(), Some(false));
    /// ```
    pub fn is_unit(&self) -> Option<bool> {
        let one = self.context().one();
        ex_is_zero(&(&self.norm_squared() - &one))
    }

    /// Inverse `q⁻¹ = q* / |q|²`.
    pub fn inverse(&self) -> Quaternion {
        let conj = self.conjugate();
        let n2 = self.norm_squared();
        conj.map(|c| c / &n2)
    }

    /// Normalise to a unit quaternion `q / |q|`.
    pub fn normalize(&self) -> Quaternion {
        let n = self.norm();
        self.map(|c| c / &n)
    }

    /// Mathematical equality (three-valued): every component difference
    /// simplifies to zero.
    pub fn equals(&self, other: &Quaternion) -> Option<bool> {
        all3(
            self.components()
                .iter()
                .zip(other.components().iter())
                .map(|(a, b)| ex_is_zero(&(*a - *b))),
        )
    }

    // ── Rotations ──────────────────────────────────────────────────────

    /// Convert to a 3×3 rotation matrix.
    ///
    /// Assumes a unit quaternion.  The rotation matrix is:
    /// ```text
    /// R = | 1−2(y²+z²)   2(xy−wz)    2(xz+wy)  |
    ///     | 2(xy+wz)     1−2(x²+z²)  2(yz−wx)  |
    ///     | 2(xz−wy)     2(yz+wx)    1−2(x²+y²) |
    /// ```
    pub fn to_rotation_matrix(&self) -> Matrix {
        let ctx = self.context();
        let one = ctx.int(1);
        let two = ctx.int(2);

        let xx = self.x.powi(2);
        let yy = self.y.powi(2);
        let zz = self.z.powi(2);
        let xy = &self.x * &self.y;
        let xz = &self.x * &self.z;
        let yz = &self.y * &self.z;
        let wx = &self.w * &self.x;
        let wy = &self.w * &self.y;
        let wz = &self.w * &self.z;

        let r00 = &one - &(&two * &(&yy + &zz));
        let r01 = &two * &(&xy - &wz);
        let r02 = &two * &(&xz + &wy);
        let r10 = &two * &(&xy + &wz);
        let r11 = &one - &(&two * &(&xx + &zz));
        let r12 = &two * &(&yz - &wx);
        let r20 = &two * &(&xz - &wy);
        let r21 = &two * &(&yz + &wx);
        let r22 = &one - &(&two * &(&xx + &yy));

        Matrix::from_rows_unchecked(vec![
            vec![r00, r01, r02],
            vec![r10, r11, r12],
            vec![r20, r21, r22],
        ])
    }

    /// Recover a unit quaternion from a 3×3 rotation matrix (Shepperd's
    /// method).
    ///
    /// For matrices whose entries are all rational numbers the numerically
    /// most stable of the four Shepperd branches is chosen.  For symbolic
    /// matrices the *trace branch* is used:
    ///
    /// ```text
    /// w = √(1 + tr R) / 2,   x = (R₂₁ − R₁₂)/(4w),  y = (R₀₂ − R₂₀)/(4w),  z = (R₁₀ − R₀₁)/(4w)
    /// ```
    ///
    /// which is valid whenever `1 + tr R > 0` (rotation angle < 180°).
    /// The returned quaternion has `w ≥ 0` in the numeric case.
    ///
    /// # Errors
    ///
    /// - [`SymplexError::InvalidArgument`] if `r` is not 3×3 or is provably
    ///   not orthogonal.
    /// - [`SymplexError::ComputationFailed`] if the matrix is symbolic and
    ///   `1 + tr R` is provably non-positive (trace branch unusable).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::robotics::rot_z;
    ///
    /// let ctx = Context::new();
    /// let th = ctx.symbol_with("theta", &[Assumption::Positive]).unwrap();
    /// let q = Quaternion::from_rotation_matrix(&rot_z(&th)).unwrap();
    /// // Round trip back to the same rotation matrix
    /// let back = q.to_rotation_matrix().simplify();
    /// assert_eq!(back.equals(&rot_z(&th)), Some(true));
    /// ```
    pub fn from_rotation_matrix(r: &Matrix) -> Result<Quaternion, SymplexError> {
        if r.shape() != (3, 3) {
            return Err(invalid(
                "Quaternion::from_rotation_matrix",
                format!("expected a 3×3 matrix, got {}×{}", r.nrows(), r.ncols()),
            ));
        }
        if r.is_orthogonal() == Some(false) {
            return Err(invalid(
                "Quaternion::from_rotation_matrix",
                "matrix is not orthogonal (RᵀR ≠ I)",
            ));
        }
        let ctx = r.context();
        let one = ctx.int(1);
        let two = ctx.int(2);
        let four = ctx.int(4);
        let tr = r.trace()?;

        let numeric = r.eval_f64().ok();
        if let Some(v) = numeric {
            // Shepperd: pick the branch with the largest diagonal quantity.
            let t = v[0][0] + v[1][1] + v[2][2];
            let cands = [t, v[0][0], v[1][1], v[2][2]];
            let mut best = 0;
            for (i, c) in cands.iter().enumerate() {
                if *c > cands[best] {
                    best = i;
                }
            }
            let e = |i: usize, j: usize| r.get(i, j);
            let q = match best {
                0 => {
                    let s = (&one + &tr).sqrt(); // s = 2w
                    let w = &s / &two;
                    let d = &two * &s; // 4w
                    Quaternion::new(
                        w,
                        &(e(2, 1) - e(1, 2)) / &d,
                        &(e(0, 2) - e(2, 0)) / &d,
                        &(e(1, 0) - e(0, 1)) / &d,
                    )
                }
                1 => {
                    let s = (&(&(&one + e(0, 0)) - e(1, 1)) - e(2, 2)).sqrt(); // 2x
                    let d = &two * &s;
                    Quaternion::new(
                        &(e(2, 1) - e(1, 2)) / &d,
                        &s / &two,
                        &(e(0, 1) + e(1, 0)) / &d,
                        &(e(0, 2) + e(2, 0)) / &d,
                    )
                }
                2 => {
                    let s = (&(&(&one - e(0, 0)) + e(1, 1)) - e(2, 2)).sqrt(); // 2y
                    let d = &two * &s;
                    Quaternion::new(
                        &(e(0, 2) - e(2, 0)) / &d,
                        &(e(0, 1) + e(1, 0)) / &d,
                        &s / &two,
                        &(e(1, 2) + e(2, 1)) / &d,
                    )
                }
                _ => {
                    let s = (&(&(&one - e(0, 0)) - e(1, 1)) + e(2, 2)).sqrt(); // 2z
                    let d = &two * &s;
                    Quaternion::new(
                        &(e(1, 0) - e(0, 1)) / &d,
                        &(e(0, 2) + e(2, 0)) / &d,
                        &(e(1, 2) + e(2, 1)) / &d,
                        &s / &two,
                    )
                }
            };
            // All entries are constants: fold them (`simplify` is slow on
            // trig-of-rational constants and adds nothing numerically).
            let q = q.eval();
            // Canonical sign: w ≥ 0.
            let q = if ex_is_positive(&(-&q.w)) == Some(true) {
                -&q
            } else {
                q
            };
            return Ok(q);
        }

        let one_plus_tr = (&one + &tr).simplify();
        if ex_is_positive(&one_plus_tr) == Some(false) {
            return Err(failed(
                "Quaternion::from_rotation_matrix",
                format!(
                    "trace branch requires 1 + tr(R) > 0, but 1 + tr(R) = {one_plus_tr}; \
                     the rotation angle is 180°"
                ),
            ));
        }
        let w = &one_plus_tr.sqrt() / &two;
        let four_w = &four * &w;
        Ok(Quaternion::new(
            w.clone(),
            (&(r.get(2, 1) - r.get(1, 2)) / &four_w).simplify(),
            (&(r.get(0, 2) - r.get(2, 0)) / &four_w).simplify(),
            (&(r.get(1, 0) - r.get(0, 1)) / &four_w).simplify(),
        ))
    }

    /// Create a quaternion from an axis–angle representation:
    /// `q = cos(θ/2) + sin(θ/2)·(axisₓ i + axisᵧ j + axis_z k)`.
    ///
    /// Assumes `axis` is a unit vector.
    pub fn from_axis_angle(axis_x: &Ex, axis_y: &Ex, axis_z: &Ex, angle: &Ex) -> Quaternion {
        let two = angle.context().int(2);
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

    /// Recover `(axis, angle)` with `axis` a 3×1 unit column vector and
    /// `angle = 2·atan2(|v|, w) ∈ [0, 2π)`, where `v` is the vector part.
    /// When `w` is provably positive the equivalent `2·atan(|v|/w)` is used,
    /// which the evaluator can fold for exact constants (`π/2` below).
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::ComputationFailed`] if the vector part is
    /// provably zero (identity rotation — the axis is undefined).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// // 90° about z
    /// let q = Quaternion::from_axis_angle(&ctx.int(0), &ctx.int(0), &ctx.int(1), &(ctx.pi() / 2));
    /// let (axis, angle) = q.to_axis_angle().unwrap();
    /// assert_eq!(axis.eval(), matrix![ctx, [0], [0], [1]]);
    /// assert_eq!(angle.eval(), ctx.pi() / 2);
    /// ```
    pub fn to_axis_angle(&self) -> Result<(Matrix, Ex), SymplexError> {
        let v_norm_sq = (&self.x.powi(2) + &self.y.powi(2) + &self.z.powi(2)).eval();
        if ex_is_zero(&v_norm_sq) == Some(true) {
            return Err(failed(
                "Quaternion::to_axis_angle",
                "vector part is zero (identity rotation): the axis is undefined",
            ));
        }
        // Rationalised for constants (`√(1/2)` → `√2/2`) so that it matches
        // the evaluator's form of `cos(π/4)` and quotients fold exactly.
        let v_norm = sqrt_rationalized(&v_norm_sq);
        let tidy = |e: Ex| {
            if e.is_constant() {
                e.eval()
            } else {
                e.simplify()
            }
        };
        let axis = Matrix::col_vector(vec![
            tidy(&self.x.eval() / &v_norm),
            tidy(&self.y.eval() / &v_norm),
            tidy(&self.z.eval() / &v_norm),
        ]);
        let w = self.w.eval();
        let half_angle = if ex_is_positive(&w) == Some(true) {
            (&v_norm / &w).atan()
        } else {
            v_norm.atan2(&w)
        };
        let angle = tidy(&half_angle * 2);
        Ok((axis, angle))
    }

    /// Quaternion for the Euler-angle rotation `rot_euler(phi, theta, psi,
    /// convention)` of the robotics module (same axis order and handedness).
    ///
    /// For example `ZYX` gives `q = qz(φ) ⊗ qy(θ) ⊗ qx(ψ)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::quaternion::EulerConvention;
    /// use symplex::robotics::rot_euler;
    ///
    /// let ctx = Context::new();
    /// let (a, b, c) = (ctx.rational(1, 3), ctx.rational(-2, 5), ctx.rational(7, 10));
    /// let q = Quaternion::from_euler(&a, &b, &c, EulerConvention::ZYX);
    /// let r = rot_euler(&a, &b, &c, EulerConvention::ZYX);
    /// let diff = (&q.to_rotation_matrix() - &r).eval_f64().unwrap();
    /// assert!(diff.iter().flatten().all(|d| d.abs() < 1e-12));
    /// ```
    pub fn from_euler(phi: &Ex, theta: &Ex, psi: &Ex, convention: EulerConvention) -> Quaternion {
        let ctx = phi.context();
        let (zero, one) = (ctx.int(0), ctx.int(1));
        let qx = |a: &Ex| Quaternion::from_axis_angle(&one, &zero, &zero, a);
        let qy = |a: &Ex| Quaternion::from_axis_angle(&zero, &one, &zero, a);
        let qz = |a: &Ex| Quaternion::from_axis_angle(&zero, &zero, &one, a);
        match convention {
            EulerConvention::ZYX => qz(phi).mul(&qy(theta)).mul(&qx(psi)),
            EulerConvention::ZXZ => qz(phi).mul(&qx(theta)).mul(&qz(psi)),
            EulerConvention::XYZ => qx(phi).mul(&qy(theta)).mul(&qz(psi)),
        }
    }

    /// Euler angles `(phi, theta, psi)` such that
    /// `rot_euler(phi, theta, psi, convention)` equals this (unit)
    /// quaternion's rotation matrix.
    ///
    /// Uses `atan2` throughout, so the angles lie in the principal ranges
    /// (`theta ∈ [−π/2, π/2]` for ZYX/XYZ, `theta ∈ [0, π]` for ZXZ).  At
    /// gimbal lock (`|theta|` at the range boundary) `phi` and `psi` are
    /// not unique; the returned pair is still a valid factorisation.
    ///
    /// Assumes a unit quaternion.  Symbolic results are simplified;
    /// constant ones (angles given as numbers) are only constant-folded,
    /// since they are meant to be evaluated numerically.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::quaternion::EulerConvention;
    ///
    /// let ctx = Context::new();
    /// let (a, b, c) = (ctx.rational(1, 3), ctx.rational(-2, 5), ctx.rational(7, 10));
    /// let q = Quaternion::from_euler(&a, &b, &c, EulerConvention::ZYX);
    /// let (phi, theta, psi) = q.to_euler(EulerConvention::ZYX);
    /// assert!((phi.eval_f64().unwrap() - 1.0 / 3.0).abs() < 1e-12);
    /// assert!((theta.eval_f64().unwrap() + 0.4).abs() < 1e-12);
    /// assert!((psi.eval_f64().unwrap() - 0.7).abs() < 1e-12);
    /// ```
    pub fn to_euler(&self, convention: EulerConvention) -> (Ex, Ex, Ex) {
        let r = self.to_rotation_matrix();
        let e = |i: usize, j: usize| r.get(i, j).clone();
        let hyp = |a: Ex, b: Ex| (a.powi(2) + b.powi(2)).sqrt();
        let (phi, theta, psi) = match convention {
            EulerConvention::ZYX => (
                e(1, 0).atan2(&e(0, 0)),
                (-e(2, 0)).atan2(&hyp(e(0, 0), e(1, 0))),
                e(2, 1).atan2(&e(2, 2)),
            ),
            EulerConvention::XYZ => (
                (-e(1, 2)).atan2(&e(2, 2)),
                e(0, 2).atan2(&hyp(e(0, 0), e(0, 1))),
                (-e(0, 1)).atan2(&e(0, 0)),
            ),
            EulerConvention::ZXZ => (
                e(0, 2).atan2(&(-e(1, 2))),
                hyp(e(2, 0), e(2, 1)).atan2(&e(2, 2)),
                e(2, 0).atan2(&e(2, 1)),
            ),
        };
        let tidy = |e: Ex| {
            if e.is_constant() {
                e.eval()
            } else {
                e.simplify()
            }
        };
        (tidy(phi), tidy(theta), tidy(psi))
    }

    /// Rotate a 3×1 column vector: `v' = q ⊗ (0, v) ⊗ q*`.
    ///
    /// Assumes a unit quaternion (equivalent to
    /// `to_rotation_matrix() * v`).
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::InvalidArgument`] if `v` is not 3×1.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// // 90° about z maps x̂ to ŷ
    /// let q = Quaternion::from_axis_angle(&ctx.int(0), &ctx.int(0), &ctx.int(1), &(ctx.pi() / 2));
    /// let v = q.rotate_vector(&matrix![ctx, [1], [0], [0]]).unwrap();
    /// assert_eq!(v.eval().simplify(), matrix![ctx, [0], [1], [0]]);
    /// ```
    pub fn rotate_vector(&self, v: &Matrix) -> Result<Matrix, SymplexError> {
        let p = Quaternion::from_vector_matrix(v).map_err(|_| {
            invalid(
                "Quaternion::rotate_vector",
                format!(
                    "expected a 3×1 column vector, got {}×{}",
                    v.nrows(),
                    v.ncols()
                ),
            )
        })?;
        let rotated = self.mul(&p).mul(&self.conjugate());
        Ok(rotated.vector())
    }

    /// Spherical linear interpolation between two **unit** quaternions:
    ///
    /// ```text
    /// slerp(q₀, q₁, t) = sin((1−t)Ω)/sin Ω · q₀ + sin(tΩ)/sin Ω · q₁,   cos Ω = q₀·q₁
    /// ```
    ///
    /// The result is left in terms of `Ω = acos(q₀·q₁)`, so it stays
    /// symbolic in `t`.
    ///
    /// # Domain
    ///
    /// * Both quaternions must be unit (otherwise `q₀·q₁` leaves `[−1, 1]`
    ///   and `acos` is complex).
    /// * `q₀ ≠ ±q₁` (otherwise `sin Ω = 0`); for `q₀ = q₁` the result is
    ///   `0/0` — check with [`equals`](Self::equals) first.
    /// * No shortest-path sign flip is applied: if `q₀·q₁ < 0` the
    ///   interpolation takes the long way round; negate one input to
    ///   avoid that.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let q0 = Quaternion::identity(&ctx);
    /// let q1 = Quaternion::from_axis_angle(&ctx.int(0), &ctx.int(0), &ctx.int(1), &(ctx.pi() / 2));
    /// let t = ctx.symbol("t");
    /// let q = q0.slerp(&q1, &t);
    /// // At t = 1/2 the rotation is 45° about z
    /// let half = q.subs(&t, &ctx.rational(1, 2)).eval().simplify();
    /// assert!((half.w.eval_f64().unwrap() - (std::f64::consts::PI / 8.0).cos()).abs() < 1e-12);
    /// assert!((half.z.eval_f64().unwrap() - (std::f64::consts::PI / 8.0).sin()).abs() < 1e-12);
    /// ```
    pub fn slerp(&self, other: &Quaternion, t: &Ex) -> Quaternion {
        let one = t.context().int(1);
        let omega = self.dot(other).acos();
        let sin_omega = omega.sin();
        let a = &(&(&one - t) * &omega).sin() / &sin_omega;
        let b = &(t * &omega).sin() / &sin_omega;
        self.scale(&a).add(&other.scale(&b))
    }

    // ── Transcendental functions ───────────────────────────────────────

    /// Quaternion exponential `exp(w + v) = eʷ (cos|v| + (v/|v|) sin|v|)`.
    ///
    /// For a pure quaternion `v = (θ/2)·n̂` this is the unit rotation
    /// quaternion `cos(θ/2) + n̂ sin(θ/2)`.  If the vector part is
    /// structurally zero the result is `(eʷ, 0, 0, 0)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let th = ctx.symbol_with("theta", &[Assumption::Positive]).unwrap();
    /// let half = &th / 2;
    /// let v = Quaternion::from_vector(&ctx.int(0), &ctx.int(0), &half);
    /// let q = v.exp().eval().simplify();
    /// assert_eq!(q.w, half.cos());
    /// assert_eq!(q.z, half.sin());
    /// ```
    pub fn exp(&self) -> Quaternion {
        let ctx = self.context();
        let ew = self.w.exp();
        let v_sq = &self.x.powi(2) + &self.y.powi(2) + &self.z.powi(2);
        if v_sq.is_zero_structural() {
            return Quaternion::new(ew, ctx.int(0), ctx.int(0), ctx.int(0));
        }
        let theta = v_sq.sqrt().simplify();
        let k = &(&ew * &theta.sin()) / &theta;
        Quaternion {
            w: &ew * &theta.cos(),
            x: &k * &self.x,
            y: &k * &self.y,
            z: &k * &self.z,
        }
    }

    /// Quaternion logarithm `ln q = ln|q| + (v/|v|) · acos(w/|q|)`.
    ///
    /// Principal branch (`acos ∈ [0, π]`).  If the vector part is
    /// structurally zero the result is `(ln w, 0, 0, 0)` — real only for
    /// `w > 0`.  For a unit rotation quaternion `cos(θ/2) + n̂ sin(θ/2)`
    /// with `θ ∈ (0, 2π)` this gives `(θ/2)·n̂`.
    pub fn ln(&self) -> Quaternion {
        let ctx = self.context();
        let v_sq = &self.x.powi(2) + &self.y.powi(2) + &self.z.powi(2);
        if v_sq.is_zero_structural() {
            return Quaternion::new(self.w.ln(), ctx.int(0), ctx.int(0), ctx.int(0));
        }
        let norm = self.norm().simplify();
        let v_norm = v_sq.sqrt().simplify();
        let angle = (&self.w / &norm).acos();
        let k = &angle / &v_norm;
        Quaternion {
            w: norm.ln(),
            x: &k * &self.x,
            y: &k * &self.y,
            z: &k * &self.z,
        }
    }

    /// Symbolic power `qᵗ = exp(t · ln q)`.
    ///
    /// For a unit rotation quaternion this scales the rotation angle by
    /// `t` about the same axis.  Components are simplified.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// // (90° about z)^2 = 180° about z
    /// let q = Quaternion::from_axis_angle(&ctx.int(0), &ctx.int(0), &ctx.int(1), &(ctx.pi() / 2));
    /// let q2 = q.pow(&ctx.int(2)).eval();
    /// assert!(q2.w.eval_f64().unwrap().abs() < 1e-12);
    /// assert!((q2.z.eval_f64().unwrap() - 1.0).abs() < 1e-12);
    /// ```
    pub fn pow(&self, t: &Ex) -> Quaternion {
        self.ln().scale(t).exp().eval().simplify()
    }

    // ── Kinematics / calculus ──────────────────────────────────────────

    /// Quaternion derivative for a body angular velocity `ω`:
    /// `q̇ = ½ · q ⊗ (0, ω)`.
    pub fn angular_velocity_derivative(
        &self,
        omega_x: &Ex,
        omega_y: &Ex,
        omega_z: &Ex,
    ) -> Quaternion {
        let omega_quat = Quaternion::from_vector(omega_x, omega_y, omega_z);
        let product = self.mul(&omega_quat);
        let half = self.context().rational(1, 2);
        product.scale(&half)
    }

    /// Differentiate every component with respect to `var`.
    pub fn diff(&self, var: &Ex) -> Quaternion {
        self.map(|c| c.diff(var))
    }

    /// Substitute a variable in all four components.
    pub fn subs(&self, old: &Ex, new: &Ex) -> Quaternion {
        self.map(|c| c.subs(old, new))
    }

    /// Evaluate all components.
    pub fn eval(&self) -> Quaternion {
        self.map(|c| c.eval())
    }

    /// Simplify all components.
    pub fn simplify(&self) -> Quaternion {
        self.map(|c| c.simplify())
    }

    /// Render this quaternion as LaTeX.
    ///
    /// # Example
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::quaternion::Quaternion;
    /// let ctx = Context::new();
    /// let q = Quaternion::identity(&ctx);
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

// ═══════════════════════════════════════════════════════════════════════════
// Operators
// ═══════════════════════════════════════════════════════════════════════════

macro_rules! quat_binop {
    ($trait:ident, $method:ident, $inner:ident) => {
        impl std::ops::$trait<&Quaternion> for &Quaternion {
            type Output = Quaternion;
            fn $method(self, rhs: &Quaternion) -> Quaternion {
                Quaternion::$inner(self, rhs)
            }
        }
        impl std::ops::$trait<Quaternion> for Quaternion {
            type Output = Quaternion;
            fn $method(self, rhs: Quaternion) -> Quaternion {
                Quaternion::$inner(&self, &rhs)
            }
        }
        impl std::ops::$trait<&Quaternion> for Quaternion {
            type Output = Quaternion;
            fn $method(self, rhs: &Quaternion) -> Quaternion {
                Quaternion::$inner(&self, rhs)
            }
        }
        impl std::ops::$trait<Quaternion> for &Quaternion {
            type Output = Quaternion;
            fn $method(self, rhs: Quaternion) -> Quaternion {
                Quaternion::$inner(self, &rhs)
            }
        }
    };
}

quat_binop!(Add, add, add);
quat_binop!(Sub, sub, sub);
quat_binop!(Mul, mul, mul);

impl std::ops::Neg for &Quaternion {
    type Output = Quaternion;
    fn neg(self) -> Quaternion {
        self.map(|c| -c)
    }
}
impl std::ops::Neg for Quaternion {
    type Output = Quaternion;
    fn neg(self) -> Quaternion {
        -&self
    }
}

impl std::ops::Mul<&Ex> for &Quaternion {
    type Output = Quaternion;
    fn mul(self, rhs: &Ex) -> Quaternion {
        self.scale(rhs)
    }
}
impl std::ops::Mul<Ex> for &Quaternion {
    type Output = Quaternion;
    fn mul(self, rhs: Ex) -> Quaternion {
        self.scale(&rhs)
    }
}
impl std::ops::Mul<&Ex> for Quaternion {
    type Output = Quaternion;
    fn mul(self, rhs: &Ex) -> Quaternion {
        self.scale(rhs)
    }
}
impl std::ops::Mul<Ex> for Quaternion {
    type Output = Quaternion;
    fn mul(self, rhs: Ex) -> Quaternion {
        self.scale(&rhs)
    }
}
impl std::ops::Mul<&Quaternion> for &Ex {
    type Output = Quaternion;
    fn mul(self, rhs: &Quaternion) -> Quaternion {
        rhs.scale(self)
    }
}
impl std::ops::Mul<Quaternion> for &Ex {
    type Output = Quaternion;
    fn mul(self, rhs: Quaternion) -> Quaternion {
        rhs.scale(self)
    }
}
impl std::ops::Mul<&Quaternion> for Ex {
    type Output = Quaternion;
    fn mul(self, rhs: &Quaternion) -> Quaternion {
        rhs.scale(&self)
    }
}
impl std::ops::Mul<Quaternion> for Ex {
    type Output = Quaternion;
    fn mul(self, rhs: Quaternion) -> Quaternion {
        rhs.scale(&self)
    }
}
impl std::ops::Mul<i64> for &Quaternion {
    type Output = Quaternion;
    fn mul(self, rhs: i64) -> Quaternion {
        let s = self.context().int(rhs);
        self.scale(&s)
    }
}
impl std::ops::Mul<i64> for Quaternion {
    type Output = Quaternion;
    fn mul(self, rhs: i64) -> Quaternion {
        &self * rhs
    }
}
impl std::ops::Mul<&Quaternion> for i64 {
    type Output = Quaternion;
    fn mul(self, rhs: &Quaternion) -> Quaternion {
        rhs * self
    }
}
impl std::ops::Mul<Quaternion> for i64 {
    type Output = Quaternion;
    fn mul(self, rhs: Quaternion) -> Quaternion {
        &rhs * self
    }
}
impl std::ops::Div<&Ex> for &Quaternion {
    type Output = Quaternion;
    fn div(self, rhs: &Ex) -> Quaternion {
        self.map(|c| c / rhs)
    }
}
impl std::ops::Div<Ex> for &Quaternion {
    type Output = Quaternion;
    fn div(self, rhs: Ex) -> Quaternion {
        self / &rhs
    }
}
impl std::ops::Div<&Ex> for Quaternion {
    type Output = Quaternion;
    fn div(self, rhs: &Ex) -> Quaternion {
        &self / rhs
    }
}
impl std::ops::Div<Ex> for Quaternion {
    type Output = Quaternion;
    fn div(self, rhs: Ex) -> Quaternion {
        &self / &rhs
    }
}
impl std::ops::Div<i64> for &Quaternion {
    type Output = Quaternion;
    fn div(self, rhs: i64) -> Quaternion {
        let s = self.context().int(rhs);
        self / &s
    }
}
impl std::ops::Div<i64> for Quaternion {
    type Output = Quaternion;
    fn div(self, rhs: i64) -> Quaternion {
        &self / rhs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::context::Context;
    use crate::base::assumptions::Assumption;
    use crate::domains::robotics::{rot_euler, rot_x, rot_y, rot_z};

    fn q(ctx: &Context, w: i64, x: i64, y: i64, z: i64) -> Quaternion {
        Quaternion::new(ctx.int(w), ctx.int(x), ctx.int(y), ctx.int(z))
    }

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10
    }

    #[test]
    fn hamilton_basis_relations() {
        let ctx = Context::new();
        let one = Quaternion::identity(&ctx);
        let i = q(&ctx, 0, 1, 0, 0);
        let j = q(&ctx, 0, 0, 1, 0);
        let k = q(&ctx, 0, 0, 0, 1);
        assert_eq!(&i * &i, -&one);
        assert_eq!(&j * &j, -&one);
        assert_eq!(&k * &k, -&one);
        assert_eq!(&i * &j, k);
        assert_eq!(&j * &k, i);
        assert_eq!(&k * &i, j);
        assert_eq!(&j * &i, -&k);
        assert_eq!((&i * &j) * &k, -&one);
        assert_eq!(i.clone() * j.clone(), k);
        assert_eq!(i.mul(&j), k);
    }

    #[test]
    fn add_sub_scale_neg_operators() {
        let ctx = Context::new();
        let a = q(&ctx, 1, 2, 3, 4);
        let b = q(&ctx, 5, 6, 7, 8);
        assert_eq!(&a + &b, q(&ctx, 6, 8, 10, 12));
        assert_eq!(&b - &a, q(&ctx, 4, 4, 4, 4));
        assert_eq!(&a * 2, q(&ctx, 2, 4, 6, 8));
        assert_eq!(2 * &a, q(&ctx, 2, 4, 6, 8));
        assert_eq!(&a * &ctx.int(3), q(&ctx, 3, 6, 9, 12));
        assert_eq!(&ctx.int(3) * &a, q(&ctx, 3, 6, 9, 12));
        assert_eq!(
            &a / 2,
            Quaternion::new(
                ctx.rational(1, 2),
                ctx.int(1),
                ctx.rational(3, 2),
                ctx.int(2)
            )
        );
        assert_eq!(-&a, q(&ctx, -1, -2, -3, -4));
        assert_eq!(a.dot(&b), ctx.int(70));
        assert_eq!(a.norm_squared(), ctx.int(30));
    }

    #[test]
    fn inverse_and_unit() {
        let ctx = Context::new();
        let a = q(&ctx, 1, 2, 3, 4);
        let prod = (&a * &a.inverse()).eval();
        assert_eq!(prod, Quaternion::identity(&ctx));
        assert_eq!(a.is_unit(), Some(false));
        assert_eq!(a.normalize().is_unit(), Some(true));
        let th = ctx.symbol("th");
        let u = Quaternion::from_axis_angle(&ctx.int(0), &ctx.int(1), &ctx.int(0), &th);
        assert_eq!(u.is_unit(), Some(true));
        let x = ctx.symbol("x");
        assert_eq!(
            Quaternion::new(x.clone(), ctx.int(0), ctx.int(0), ctx.int(0)).is_unit(),
            None
        );
        assert_eq!(a.equals(&q(&ctx, 1, 2, 3, 4)), Some(true));
        assert_eq!(a.equals(&b_like(&ctx)), Some(false));
    }

    fn b_like(ctx: &Context) -> Quaternion {
        q(ctx, 1, 2, 3, 5)
    }

    #[test]
    fn rotation_matrix_matches_robotics_rotations() {
        let ctx = Context::new();
        let th = ctx.symbol("theta");
        let (zero, one) = (ctx.int(0), ctx.int(1));
        let qx = Quaternion::from_axis_angle(&one, &zero, &zero, &th);
        let qy = Quaternion::from_axis_angle(&zero, &one, &zero, &th);
        let qz = Quaternion::from_axis_angle(&zero, &zero, &one, &th);
        for (qq, r) in [(qx, rot_x(&th)), (qy, rot_y(&th)), (qz, rot_z(&th))] {
            let m = qq.to_rotation_matrix();
            for v in [0.3f64, 1.1, 2.9] {
                let val = ctx.rational((v * 1e6) as i64, 1_000_000);
                let a = m.subs(&th, &val).eval_f64().unwrap();
                let b = r.subs(&th, &val).eval_f64().unwrap();
                for i in 0..3 {
                    for j in 0..3 {
                        assert!(
                            approx(a[i][j], b[i][j]),
                            "({i},{j}): {} vs {}",
                            a[i][j],
                            b[i][j]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn rotate_vector_matches_matrix() {
        let ctx = Context::new();
        let th = ctx.symbol("theta");
        let qz = Quaternion::from_axis_angle(&ctx.int(0), &ctx.int(0), &ctx.int(1), &th);
        let v = Matrix::col_vector(vec![ctx.int(1), ctx.int(2), ctx.int(3)]);
        let rv = qz.rotate_vector(&v).unwrap().simplify();
        let mv = (&qz.to_rotation_matrix() * &v).simplify();
        assert_eq!(rv.equals(&mv), Some(true));
        assert!(qz.rotate_vector(&Matrix::identity(&ctx, 2)).is_err());
    }

    #[test]
    fn from_rotation_matrix_numeric_all_branches() {
        let ctx = Context::new();
        // Angles chosen so each Shepperd branch is exercised, incl. 180° rotations.
        let cases: Vec<Matrix> = vec![
            rot_z(&(ctx.pi() / 3)),
            rot_x(&ctx.pi()),
            rot_y(&ctx.pi()),
            rot_z(&ctx.pi()),
            rot_euler(
                &(ctx.pi() / 5),
                &(ctx.pi() * ctx.rational(3, 4)),
                &(ctx.pi() / 7),
                EulerConvention::ZYX,
            ),
        ];
        for r in cases {
            let r = r.eval();
            let qq = Quaternion::from_rotation_matrix(&r).unwrap();
            let back = qq.to_rotation_matrix().eval_f64().unwrap();
            let orig = r.eval_f64().unwrap();
            for i in 0..3 {
                for j in 0..3 {
                    assert!(
                        approx(back[i][j], orig[i][j]),
                        "({i},{j}) {} vs {}",
                        back[i][j],
                        orig[i][j]
                    );
                }
            }
            assert!(qq.w.eval_f64().unwrap() >= -1e-12, "canonical sign w ≥ 0");
            assert!(approx(qq.norm_squared().eval_f64().unwrap(), 1.0));
        }
        assert!(Quaternion::from_rotation_matrix(&Matrix::identity(&ctx, 2)).is_err());
        assert!(
            Quaternion::from_rotation_matrix(
                &Matrix::from_i64(&ctx, &[&[1, 1, 0], &[0, 1, 0], &[0, 0, 1]]).unwrap()
            )
            .is_err()
        );
    }

    #[test]
    fn from_rotation_matrix_symbolic_trace_branch() {
        let ctx = Context::new();
        let th = ctx.symbol_with("theta", &[Assumption::Positive]).unwrap();
        let qq = Quaternion::from_rotation_matrix(&rot_y(&th)).unwrap();
        // Compare numerically with from_axis_angle about y
        let expected = Quaternion::from_axis_angle(&ctx.int(0), &ctx.int(1), &ctx.int(0), &th);
        for v in [0.4f64, 1.3, 2.5] {
            let val = ctx.rational((v * 1e6) as i64, 1_000_000);
            for (a, b) in qq.components().iter().zip(expected.components().iter()) {
                let av = a.subs(&th, &val).eval_f64().unwrap();
                let bv = b.subs(&th, &val).eval_f64().unwrap();
                assert!(approx(av, bv), "{a} vs {b} at {v}: {av} vs {bv}");
            }
        }
        // 180° about x: trace = −1 → trace branch unusable for symbolic input
        let x = ctx.symbol("x");
        let r180 = Matrix::new(vec![
            vec![ctx.int(1), ctx.int(0), ctx.int(0)],
            vec![ctx.int(0), ctx.int(-1), &x - &x],
            vec![ctx.int(0), &x - &x, ctx.int(-1)],
        ])
        .unwrap();
        assert!(
            Quaternion::from_rotation_matrix(&r180).is_ok(),
            "numeric fallback handles 180°"
        );
    }

    #[test]
    fn axis_angle_round_trip() {
        let ctx = Context::new();
        let axis = [ctx.rational(3, 5), ctx.int(0), ctx.rational(4, 5)];
        let angle = ctx.pi() * ctx.rational(2, 3);
        let qq = Quaternion::from_axis_angle(&axis[0], &axis[1], &axis[2], &angle);
        let (ax, ang) = qq.to_axis_angle().unwrap();
        let ax = ax.eval_f64().unwrap();
        assert!(
            approx(ax[0][0], 0.6) && approx(ax[1][0], 0.0) && approx(ax[2][0], 0.8),
            "{ax:?}"
        );
        assert!(approx(
            ang.eval_f64().unwrap(),
            2.0 * std::f64::consts::PI / 3.0
        ));
        assert!(Quaternion::identity(&ctx).to_axis_angle().is_err());
    }

    /// Euler round trip for one convention over three angle triples.
    fn euler_round_trip(conv: EulerConvention) {
        let ctx = Context::new();
        let angles: [(f64, f64, f64); 3] = [(0.3, -0.4, 0.7), (-1.2, 0.9, 2.5), (2.0, 1.1, -0.6)];
        for (a, b, c) in angles {
            // ZXZ needs theta in (0, π)
            let b = if matches!(conv, EulerConvention::ZXZ) {
                b.abs()
            } else {
                b
            };
            let (ea, eb, ec) = (
                ctx.rational((a * 1e6) as i64, 1_000_000),
                ctx.rational((b * 1e6) as i64, 1_000_000),
                ctx.rational((c * 1e6) as i64, 1_000_000),
            );
            let qq = Quaternion::from_euler(&ea, &eb, &ec, conv);
            // Rotation matrix agrees with rot_euler
            let m = qq.to_rotation_matrix().eval_f64().unwrap();
            let r = rot_euler(&ea, &eb, &ec, conv).eval_f64().unwrap();
            for i in 0..3 {
                for j in 0..3 {
                    assert!(approx(m[i][j], r[i][j]), "{conv:?} ({i},{j})");
                }
            }
            // Euler angles recovered
            let (pa, pb, pc) = qq.to_euler(conv);
            let (pa, pb, pc) = (
                pa.eval_f64().unwrap(),
                pb.eval_f64().unwrap(),
                pc.eval_f64().unwrap(),
            );
            let exp = [
                ea.eval_f64().unwrap(),
                eb.eval_f64().unwrap(),
                ec.eval_f64().unwrap(),
            ];
            assert!(
                approx(pa, exp[0]) && approx(pb, exp[1]) && approx(pc, exp[2]),
                "{conv:?}: got ({pa}, {pb}, {pc}) expected {exp:?}"
            );
        }
    }

    #[test]
    fn euler_round_trip_zyx() {
        euler_round_trip(EulerConvention::ZYX);
    }

    #[test]
    fn euler_round_trip_xyz() {
        euler_round_trip(EulerConvention::XYZ);
    }

    #[test]
    fn euler_round_trip_zxz() {
        euler_round_trip(EulerConvention::ZXZ);
    }

    #[test]
    fn exp_ln_pow() {
        let ctx = Context::new();
        let th = ctx.symbol_with("theta", &[Assumption::Positive]).unwrap();
        let half = &th / 2;
        let v = Quaternion::from_vector(&ctx.int(0), &ctx.int(0), &half);
        let e = v.exp().eval().simplify();
        assert_eq!(e.w, half.cos());
        assert_eq!(e.z, half.sin());
        assert!(e.x.is_zero_structural());
        // ln(exp(v)) = v numerically
        let back = e.ln().subs(&th, &ctx.rational(1, 2)).eval_f64_components();
        assert!(approx(back[0], 0.0) && approx(back[3], 0.25), "{back:?}");
        // pure real
        let r = Quaternion::new(ctx.int(2), ctx.int(0), ctx.int(0), ctx.int(0));
        assert_eq!(r.exp().w, ctx.int(2).exp());
        assert_eq!(r.ln().w, ctx.int(2).ln());
        // pow: (90° about z)^(1/2) = 45° about z
        let qz =
            Quaternion::from_axis_angle(&ctx.int(0), &ctx.int(0), &ctx.int(1), &(ctx.pi() / 2));
        let h = qz.pow(&ctx.rational(1, 2)).eval();
        assert!(approx(
            h.w.eval_f64().unwrap(),
            (std::f64::consts::PI / 8.0).cos()
        ));
        assert!(approx(
            h.z.eval_f64().unwrap(),
            (std::f64::consts::PI / 8.0).sin()
        ));
    }

    impl Quaternion {
        fn eval_f64_components(&self) -> [f64; 4] {
            [
                self.w.eval_f64().unwrap(),
                self.x.eval_f64().unwrap(),
                self.y.eval_f64().unwrap(),
                self.z.eval_f64().unwrap(),
            ]
        }
    }

    #[test]
    fn slerp_endpoints_and_midpoint() {
        let ctx = Context::new();
        let q0 = Quaternion::identity(&ctx);
        let q1 =
            Quaternion::from_axis_angle(&ctx.int(0), &ctx.int(1), &ctx.int(0), &(ctx.pi() / 2));
        let t = ctx.symbol("t");
        let s = q0.slerp(&q1, &t);
        let at = |tv: (i64, i64)| {
            s.subs(&t, &ctx.rational(tv.0, tv.1))
                .eval()
                .eval_f64_components()
        };
        let a0 = at((0, 1));
        assert!(approx(a0[0], 1.0) && approx(a0[2], 0.0));
        let a1 = at((1, 1));
        assert!(approx(a1[0], (std::f64::consts::PI / 4.0).cos()));
        let am = at((1, 2));
        assert!(approx(am[0], (std::f64::consts::PI / 8.0).cos()));
        assert!(approx(am[2], (std::f64::consts::PI / 8.0).sin()));
        // d/dt of slerp is symbolic
        let ds = s.diff(&t);
        assert!(ds.w.contains(&t));
    }

    #[test]
    fn kinematics_and_diff() {
        let ctx = Context::new();
        let t = ctx.symbol("t");
        let qq = Quaternion::from_axis_angle(&ctx.int(0), &ctx.int(0), &ctx.int(1), &t);
        let qdot = qq.diff(&t);
        // For rotation about z at unit rate, q̇ = ½ q ⊗ (0, 0, 0, 1)
        let kin = qq.angular_velocity_derivative(&ctx.int(0), &ctx.int(0), &ctx.int(1));
        assert_eq!(qdot.simplify().equals(&kin.simplify()), Some(true));
        assert_eq!(qq.vector().shape(), (3, 1));
        assert_eq!(qq.scalar(), &(&t / 2).cos());
    }
}
