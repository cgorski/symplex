//! Integral transforms (Fourier, Mellin) and directional limits on [`Ex`](crate::api::expr::Ex).
//!
//! This module hosts the 0.2 additions to the transform API:
//!
//! * one-sided limits (`Direction`, `limit_dir`, `limit_left`, `limit_right`);
//! * the public Fourier transform API (`fourier_transform`,
//!   `inverse_fourier_transform`, with a selectable [`FourierConvention`]);
//! * the Mellin transform API (`mellin_transform`, `inverse_mellin_transform`);
//! * Laplace helpers (`laplace_initial_value`, `laplace_final_value`,
//!   `laplace_convolution`);
//! * Fourier series on arbitrary intervals (`fourier_series_on`).

use tracing::debug_span;

use crate::api::expr::{Ex, Expr, Numeric};
use crate::base::errors::SymplexError;

pub use crate::calculus::fourier_transform::FourierConvention;
pub use crate::calculus::limit::Direction;

// ═══════════════════════════════════════════════════════════════════════════
// Directional limits
// ═══════════════════════════════════════════════════════════════════════════

impl Expr<Numeric> {
    /// Compute the limit of this expression as `var` approaches `point`
    /// from the given `Direction`.
    ///
    /// * `Direction::Right`: `x → a⁺` (values slightly larger than `a`)
    /// * `Direction::Left`: `x → a⁻` (values slightly smaller than `a`)
    /// * `Direction::Both` (the default): two-sided; both one-sided limits
    ///   must exist and agree.
    ///
    /// For `point = ±∞` the direction is irrelevant.
    ///
    /// If the limit does not exist or cannot be determined, the formal
    /// two-sided `Limit(expr, var, point)` node is returned (check with
    /// [`has_unevaluated`](Self::has_unevaluated)). The node has no
    /// direction slot, so an uncomputable one-sided limit is represented by
    /// the same node as the two-sided one.
    ///
    /// Pathological inputs are cut off by an internal work budget and
    /// likewise come back as the unevaluated node.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let zero = ctx.int(0);
    /// // e^{1/x} → ∞ from the right, → 0 from the left
    /// let f = (1 / &x).exp();
    /// assert_eq!(format!("{}", f.limit_right(&x, &zero)), "oo");
    /// assert_eq!(format!("{}", f.limit_left(&x, &zero)), "0");
    /// // ⌊x⌋ at an integer
    /// assert_eq!(format!("{}", x.floor().limit_left(&x, &ctx.int(2))), "1");
    /// assert_eq!(format!("{}", x.floor().limit_right(&x, &ctx.int(2))), "2");
    /// ```
    #[must_use = "returns the limit value; does not modify in place"]
    pub fn limit_dir(&self, var: &Ex, point: &Ex, dir: Direction) -> Ex {
        match self.try_limit_dir(var, point, dir) {
            Ok(v) => v,
            Err(_) => {
                let var_id = self.checked_id(var);
                let point_id = self.checked_id(point);
                let id = self
                    .inner
                    .write()
                    .arena
                    .intern(crate::base::node::ExprNode::Limit(
                        self.raw_id(),
                        var_id,
                        point_id,
                    ));
                self.wrap(id)
            }
        }
    }

    /// Like [`limit_dir`](Self::limit_dir), but returns `Err` if the limit
    /// does not exist or cannot be computed.
    ///
    /// `±∞` are legitimate limit values and are returned as `Ok`. When the
    /// two one-sided limits of a `Direction::Both` request differ, the
    /// error is `ComputationFailed` with a reason of the form
    /// `"left and right limits differ: left = …, right = …"`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let f = x.sign();
    /// // sign(x) has different one-sided limits at 0
    /// let err = f.try_limit(&x, &ctx.int(0)).unwrap_err();
    /// assert!(err.to_string().contains("differ"));
    /// ```
    pub fn try_limit_dir(&self, var: &Ex, point: &Ex, dir: Direction) -> Result<Ex, SymplexError> {
        let var_id = self.checked_id(var);
        let point_id = self.checked_id(point);
        let _span =
            debug_span!("limit_dir", expr = ?self.raw_id(), var = ?var_id, dir = ?dir).entered();
        let result = {
            let mut inner = self.inner.write();
            crate::calculus::limit::limit_dir(
                &mut inner.arena,
                self.raw_id(),
                var_id,
                point_id,
                dir,
            )
        };
        result.map(|id| self.wrap(id))
    }

    /// Left-hand limit `lim_{var → point⁻}`. Shorthand for
    /// [`limit_dir`](Self::limit_dir) with `Direction::Left`.
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let f = &x.abs() / &x;
    /// assert_eq!(format!("{}", f.limit_left(&x, &ctx.int(0))), "-1");
    /// ```
    #[must_use = "returns the limit value; does not modify in place"]
    pub fn limit_left(&self, var: &Ex, point: &Ex) -> Ex {
        self.limit_dir(var, point, Direction::Left)
    }

    /// Right-hand limit `lim_{var → point⁺}`. Shorthand for
    /// [`limit_dir`](Self::limit_dir) with `Direction::Right`.
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let f = &x * x.ln();
    /// assert_eq!(format!("{}", f.limit_right(&x, &ctx.int(0))), "0");
    /// assert_eq!(format!("{}", x.ln().limit_right(&x, &ctx.int(0))), "-oo");
    /// ```
    #[must_use = "returns the limit value; does not modify in place"]
    pub fn limit_right(&self, var: &Ex, point: &Ex) -> Ex {
        self.limit_dir(var, point, Direction::Right)
    }

    /// Fallible left-hand limit. See [`try_limit_dir`](Self::try_limit_dir).
    pub fn try_limit_left(&self, var: &Ex, point: &Ex) -> Result<Ex, SymplexError> {
        self.try_limit_dir(var, point, Direction::Left)
    }

    /// Fallible right-hand limit. See [`try_limit_dir`](Self::try_limit_dir).
    pub fn try_limit_right(&self, var: &Ex, point: &Ex) -> Result<Ex, SymplexError> {
        self.try_limit_dir(var, point, Direction::Right)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Fourier transform
// ═══════════════════════════════════════════════════════════════════════════

impl Expr<Numeric> {
    /// Fourier transform `F(ω) = ∫_{−∞}^{∞} f(t) e^{−iωt} dt` of this
    /// expression (a function of `t`) as a function of `omega`.
    ///
    /// This is the non-unitary angular-frequency convention; see
    /// [`fourier_transform_with`](Self::fourier_transform_with) for the
    /// others. The transform is computed from a table (`δ`, constants,
    /// `H(t)`, `sign(t)`, `1/t`, `|t|`, rectangular windows, `e^{−a|t|}`,
    /// Gaussians, `tⁿ e^{−at} H(t)`, `cos`/`sin`, `sinc`) together with
    /// linearity, time shift, modulation, scaling, the derivative rule and
    /// `t·f(t) → i F′(ω)`.
    ///
    /// Symbols other than `t` and `omega` are treated as **real**
    /// parameters. Conditions such as `a > 0` in `e^{−a|t|}` are checked
    /// through the assumption system (declare `a` with
    /// `Assumption::Positive`); an unprovable condition is an error.
    ///
    /// # Errors
    ///
    /// * `InvalidArgument` if `t`/`omega` are not distinct symbols.
    /// * `ComputationFailed` if no rule applies, a required sign assumption
    ///   is missing, or the result would need a distribution that cannot be
    ///   represented (e.g. `δ′`). There is no unevaluated node for Fourier
    ///   transforms, so this API is `Result`-only.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let t = ctx.symbol("t");
    /// let w = ctx.symbol("w");
    /// let a = ctx.symbol_with("a", &[Assumption::Positive]);
    ///
    /// // e^{-a|t|}  →  2a/(a² + ω²)
    /// let f = (-&a * t.abs()).exp().fourier_transform(&t, &w).unwrap();
    /// assert_eq!(format!("{f}"), "2*a/(a^2 + w^2)");
    ///
    /// // Rectangular window H(t + 1) − H(t − 1)  →  2 sin(ω)/ω
    /// let rect = (&t + 1).heaviside() - (&t - 1).heaviside();
    /// let r = rect.fourier_transform(&t, &w).unwrap();
    /// assert_eq!(format!("{r}"), "2*sin(w)/w");
    ///
    /// // e^{-2t} H(t)  →  1/(iω + 2)
    /// let g = ((&t * -2).exp() * t.heaviside()).fourier_transform(&t, &w).unwrap();
    /// assert_eq!(g, 1 / (ctx.i_unit() * &w + 2));
    ///
    /// // Unknown sign → Err rather than a guess.
    /// let b = ctx.symbol("b");
    /// assert!((-&b * t.abs()).exp().fourier_transform(&t, &w).is_err());
    /// ```
    pub fn fourier_transform(&self, t: &Ex, omega: &Ex) -> Result<Ex, SymplexError> {
        self.fourier_transform_with(t, omega, FourierConvention::NonUnitaryAngular)
    }

    /// Fourier transform in the given [`FourierConvention`].
    ///
    /// | convention            | `F =`                                  |
    /// |-----------------------|----------------------------------------|
    /// | `NonUnitaryAngular`   | `∫ f(t) e^{−iωt} dt`                   |
    /// | `UnitaryAngular`      | `(1/√(2π)) ∫ f(t) e^{−iωt} dt`         |
    /// | `Ordinary`            | `∫ f(t) e^{−2πiνt} dt` (`omega` is `ν`)  |
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::fourier_transform::FourierConvention;
    ///
    /// let ctx = Context::new();
    /// let t = ctx.symbol("t");
    /// let nu = ctx.symbol("nu");
    /// // Gaussian e^{-πt²} is its own transform in the ordinary convention.
    /// let g = (-(ctx.pi() * t.powi(2))).exp();
    /// let f = g
    ///     .fourier_transform_with(&t, &nu, FourierConvention::Ordinary)
    ///     .unwrap();
    /// assert_eq!(f, (-(ctx.pi() * nu.powi(2))).exp());
    /// ```
    pub fn fourier_transform_with(
        &self,
        t: &Ex,
        omega: &Ex,
        convention: FourierConvention,
    ) -> Result<Ex, SymplexError> {
        let t_id = self.checked_id(t);
        let w_id = self.checked_id(omega);
        let _span = debug_span!("fourier_transform", expr = ?self.raw_id(), t = ?t_id).entered();
        let r = {
            let mut inner = self.inner.write();
            crate::calculus::fourier_transform::fourier_transform_with(
                &mut inner.arena,
                self.raw_id(),
                t_id,
                w_id,
                convention,
            )
        };
        r.map(|id| self.wrap(id))
    }

    /// Inverse Fourier transform `f(t) = (1/2π) ∫ F(ω) e^{iωt} dω` of this
    /// expression (a function of `omega`) as a function of `t`, in the
    /// non-unitary angular convention.
    ///
    /// Handles `δ(ω − ω₀)`, constants, `H(ω)`, `sign(ω)`, `1/ω`,
    /// `1/(iω − a)ⁿ`, `1/(ω² + a²)`, `ω/(ω² + a²)`, Gaussians, `sin(aω)/ω`,
    /// `cos(aω)`/`sin(aω)`, rectangular windows in `ω`, plus linearity,
    /// shift, modulation, scaling and `ωⁿ G(ω) → (−i)ⁿ g⁽ⁿ⁾(t)`.
    ///
    /// # Errors
    ///
    /// As for [`fourier_transform`](Self::fourier_transform).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let t = ctx.symbol("t");
    /// let w = ctx.symbol("w");
    /// // 1/(iω + 3)  →  e^{-3t} H(t)
    /// let big_f = 1 / (ctx.i_unit() * &w + 3);
    /// let f = big_f.inverse_fourier_transform(&w, &t).unwrap();
    /// assert_eq!(format!("{f}"), "exp(-3*t)*H(t)");
    /// // 2/(ω² + 1)  →  e^{-|t|}
    /// let g = (2 / (w.powi(2) + 1)).inverse_fourier_transform(&w, &t).unwrap();
    /// assert_eq!(format!("{g}"), "exp(-abs(t))");
    /// ```
    pub fn inverse_fourier_transform(&self, omega: &Ex, t: &Ex) -> Result<Ex, SymplexError> {
        self.inverse_fourier_transform_with(omega, t, FourierConvention::NonUnitaryAngular)
    }

    /// Inverse Fourier transform in the given [`FourierConvention`]
    /// (see [`fourier_transform_with`](Self::fourier_transform_with)).
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::fourier_transform::FourierConvention;
    ///
    /// let ctx = Context::new();
    /// let t = ctx.symbol("t");
    /// let w = ctx.symbol("w");
    /// let f = (-t.abs()).exp();
    /// let big_f = f
    ///     .fourier_transform_with(&t, &w, FourierConvention::UnitaryAngular)
    ///     .unwrap();
    /// let back = big_f
    ///     .inverse_fourier_transform_with(&w, &t, FourierConvention::UnitaryAngular)
    ///     .unwrap();
    /// assert_eq!(format!("{back}"), "exp(-abs(t))");
    /// ```
    pub fn inverse_fourier_transform_with(
        &self,
        omega: &Ex,
        t: &Ex,
        convention: FourierConvention,
    ) -> Result<Ex, SymplexError> {
        let w_id = self.checked_id(omega);
        let t_id = self.checked_id(t);
        let _span = debug_span!("inverse_fourier_transform", expr = ?self.raw_id(), omega = ?w_id)
            .entered();
        let r = {
            let mut inner = self.inner.write();
            crate::calculus::fourier_transform::inverse_fourier_transform_with(
                &mut inner.arena,
                self.raw_id(),
                w_id,
                t_id,
                convention,
            )
        };
        r.map(|id| self.wrap(id))
    }
}
