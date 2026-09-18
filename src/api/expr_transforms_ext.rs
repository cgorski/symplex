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
