//! Integral transforms (Fourier, Mellin) and directional limits on [`Ex`].
//!
//! This module hosts the 0.2 additions to the transform API:
//!
//! * one-sided limits (`Direction`, `limit_dir`, `limit_left`, `limit_right`);
//! * the public Fourier transform API (`fourier_transform`,
//!   `inverse_fourier_transform`, with a selectable `FourierConvention`);
//! * the Mellin transform API (`mellin_transform`, `inverse_mellin_transform`);
//! * Laplace helpers (`laplace_initial_value`, `laplace_final_value`);
//! * Fourier series on arbitrary intervals (`fourier_series_on`,
//!   `FourierSeries`);
//! * the Z-transform (`z_transform`, `inverse_z_transform`).

use tracing::debug_span;

use crate::api::expr::{BoolEx, Ex, Expr, Numeric};
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

// ═══════════════════════════════════════════════════════════════════════════
// Laplace helpers
// ═══════════════════════════════════════════════════════════════════════════

impl Expr<Numeric> {
    /// Initial value theorem: `f(0⁺) = lim_{s→∞} s·F(s)` for this Laplace
    /// transform `F(s)`.
    ///
    /// An infinite initial value (e.g. `f = 1/√t`, `F = √(π/s)`) is returned
    /// as the extended-real value `oo`.
    ///
    /// # Errors
    ///
    /// `ComputationFailed` if the limit does not exist or cannot be computed.
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let s = ctx.symbol("s");
    /// // F(s) = (s + 2)/(s² + 3s + 5)  →  f(0⁺) = 1
    /// let f = (&s + 2) / (s.powi(2) + 3 * &s + 5);
    /// assert_eq!(format!("{}", f.laplace_initial_value(&s).unwrap()), "1");
    /// // F(s) = 1/(s² + 1) (f = sin t)  →  f(0⁺) = 0
    /// assert_eq!(format!("{}", (1 / (s.powi(2) + 1)).laplace_initial_value(&s).unwrap()), "0");
    /// ```
    pub fn laplace_initial_value(&self, s: &Ex) -> Result<Ex, SymplexError> {
        let sf = s * self;
        sf.try_limit(s, &s.context().infinity())
    }

    /// Final value theorem: `lim_{t→∞} f(t) = lim_{s→0⁺} s·F(s)` for this
    /// Laplace transform `F(s)`.
    ///
    /// The theorem only holds when every pole of `s·F(s)` lies in the open
    /// left half-plane. When `s·F(s)` is a rational function whose poles can
    /// be located, this is verified and a violation is reported as
    /// `Divergent` instead of returning a meaningless number; for
    /// non-rational transforms (e.g. delays `e^{−as}F(s)`) the caller is
    /// responsible for the precondition.
    ///
    /// # Errors
    ///
    /// * `Divergent` if `s·F(s)` has a pole with non-negative real part.
    /// * `ComputationFailed` if the poles cannot be located or the limit
    ///   cannot be computed.
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let s = ctx.symbol("s");
    /// // Step response of a stable first-order system: F(s) = 3/(s(s + 2)) → 3/2
    /// let f = 3 / (&s * (&s + 2));
    /// assert_eq!(format!("{}", f.laplace_final_value(&s).unwrap()), "3/2");
    /// // e^{t} has no final value: the pole at s = 1 is detected.
    /// assert!(matches!(
    ///     (1 / (&s - 1)).laplace_final_value(&s),
    ///     Err(SymplexError::Divergent { .. })
    /// ));
    /// ```
    pub fn laplace_final_value(&self, s: &Ex) -> Result<Ex, SymplexError> {
        let ctx = s.context();
        let sf = (s * self).together();
        let (_, den) = sf.as_numer_denom();
        if den.is_polynomial(s) && !den.free_symbols().is_empty() {
            let poles = den.solve_or_empty(s);
            if poles.is_empty() {
                return Err(SymplexError::ComputationFailed {
                    operation: "laplace_final_value",
                    reason: format!(
                        "cannot locate the poles of s·F(s) (denominator {den}) to verify the final value theorem"
                    ),
                });
            }
            for p in &poles {
                let re = match p.eval_complex64() {
                    Ok((re, _)) => re,
                    Err(_) => match p.re().is_negative() {
                        Some(true) => -1.0,
                        Some(false) => 0.0,
                        None => {
                            return Err(SymplexError::ComputationFailed {
                                operation: "laplace_final_value",
                                reason: format!(
                                    "cannot determine the sign of Re({p}) (a pole of s·F(s))"
                                ),
                            });
                        }
                    },
                };
                if re >= 0.0 {
                    return Err(SymplexError::Divergent {
                        operation: "laplace_final_value",
                        reason: format!(
                            "s·F(s) has a pole at s = {p} with non-negative real part, so f(t) has no finite limit"
                        ),
                    });
                }
            }
        }
        sf.try_limit_right(s, &ctx.int(0))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Fourier series
// ═══════════════════════════════════════════════════════════════════════════

/// The trigonometric Fourier series of a function on an interval `[a, b]`
/// (period `T = b − a`, fundamental frequency `ω₀ = 2π/T`):
///
/// `f(x) ~ a₀/2 + Σ_{k≥1} [a_k cos(kω₀x) + b_k sin(kω₀x)]`
///
/// with `a_k = (2/T) ∫_a^b f(x) cos(kω₀x) dx` and
/// `b_k = (2/T) ∫_a^b f(x) sin(kω₀x) dx`. Produced by
/// `Ex::fourier_series_on`; the first `n` harmonics are stored, further
/// coefficients are computed on demand.
///
/// ```
/// use symplex::prelude::*;
///
/// let ctx = Context::new();
/// let x = ctx.symbol("x");
/// // Sawtooth f(x) = x on [-π, π]: b_k = 2(-1)^{k+1}/k, all a_k = 0
/// let fs = x.fourier_series_on(&x, &(-ctx.pi()), &ctx.pi(), 3).unwrap();
/// assert_eq!(format!("{}", fs.a0), "0");
/// assert_eq!(fs.bn.iter().map(|b| b.to_string()).collect::<Vec<_>>(), ["2", "-1", "2/3"]);
/// assert_eq!(format!("{}", fs.coefficient_b(4)), "-1/2");
/// assert_eq!(fs.truncate(2), 2 * x.sin() - (2 * &x).sin());
/// ```
#[derive(Debug, Clone)]
pub struct FourierSeries {
    /// The expanded function.
    pub function: Ex,
    /// The expansion variable.
    pub var: Ex,
    /// Left end of the interval.
    pub lower: Ex,
    /// Right end of the interval.
    pub upper: Ex,
    /// The period `T = upper − lower`.
    pub period: Ex,
    /// The constant coefficient `a₀` (the series' constant term is `a₀/2`).
    pub a0: Ex,
    /// Cosine coefficients `a₁, …, aₙ`.
    pub an: Vec<Ex>,
    /// Sine coefficients `b₁, …, bₙ`.
    pub bn: Vec<Ex>,
}

impl FourierSeries {
    /// The fundamental angular frequency `ω₀ = 2π/T`.
    #[must_use]
    pub fn omega0(&self) -> Ex {
        let ctx = self.var.context();
        2 * ctx.pi() / &self.period
    }

    /// Number of stored harmonics.
    #[must_use]
    pub fn n_terms(&self) -> usize {
        self.an.len()
    }

    fn raw_coefficient(&self, k: u32, sine: bool) -> Ex {
        let ctx = self.var.context();
        let arg = (ctx.int(i64::from(k)) * self.omega0() * &self.var).eval();
        let kernel = if sine { arg.sin() } else { arg.cos() };
        let integrand = &self.function * kernel;
        let integral = integrand.integrate_definite(&self.var, &self.lower, &self.upper);
        (2 * integral / &self.period).simplify().eval()
    }

    /// Cosine coefficient `a_k` (`k = 0` gives `a₀`). Coefficients beyond
    /// the stored ones are integrated on demand; the result may contain an
    /// unevaluated `Integral` if the integral has no closed form.
    #[must_use]
    pub fn coefficient_a(&self, k: u32) -> Ex {
        if k == 0 {
            return self.a0.clone();
        }
        if let Some(a) = self.an.get(k as usize - 1) {
            return a.clone();
        }
        self.raw_coefficient(k, false)
    }

    /// Sine coefficient `b_k` (`b₀ = 0`).
    #[must_use]
    pub fn coefficient_b(&self, k: u32) -> Ex {
        if k == 0 {
            return self.var.context().int(0);
        }
        if let Some(b) = self.bn.get(k as usize - 1) {
            return b.clone();
        }
        self.raw_coefficient(k, true)
    }

    /// Complex coefficient `c_k = (1/T) ∫_a^b f(x) e^{−ikω₀x} dx` of the
    /// exponential form `f(x) ~ Σ_k c_k e^{ikω₀x}`, for any integer `k`:
    /// `c₀ = a₀/2`, `c_k = (a_k − i b_k)/2`, `c_{−k} = (a_k + i b_k)/2`.
    #[must_use]
    pub fn coefficient_c(&self, k: i64) -> Ex {
        let ctx = self.var.context();
        if k == 0 {
            return (&self.a0 / 2).eval();
        }
        let m = k.unsigned_abs() as u32;
        let a = self.coefficient_a(m);
        let b = self.coefficient_b(m);
        let i = ctx.i_unit();
        let c = if k > 0 {
            (&a - &i * &b) / 2
        } else {
            (&a + &i * &b) / 2
        };
        c.eval()
    }

    /// Partial sum `a₀/2 + Σ_{k=1}^{n} [a_k cos(kω₀x) + b_k sin(kω₀x)]`.
    /// Harmonics beyond the stored ones are computed on demand.
    #[must_use]
    pub fn truncate(&self, n: u32) -> Ex {
        let ctx = self.var.context();
        let mut sum = &self.a0 / 2;
        let w0 = self.omega0();
        for k in 1..=n {
            let arg = (ctx.int(i64::from(k)) * &w0 * &self.var).eval();
            let a = self.coefficient_a(k);
            let b = self.coefficient_b(k);
            if !a.is_zero_structural() {
                sum = &sum + &a * arg.cos();
            }
            if !b.is_zero_structural() {
                sum = &sum + &b * arg.sin();
            }
        }
        sum.eval()
    }
}

impl Expr<Numeric> {
    /// Fourier series of this expression in `var` on the interval
    /// `[lower, upper]`, with the first `n_terms` harmonics computed.
    ///
    /// Coefficients are exact definite integrals
    /// ([`integrate_definite`](Self::integrate_definite)), so piecewise,
    /// `|x|`, `sign` and `Heaviside` inputs (square, sawtooth and triangle
    /// waves) work. Returns the `FourierSeries` with `a0`, `an`, `bn`,
    /// `period` and the `truncate` /
    /// `coefficient_c` helpers.
    ///
    /// # Errors
    ///
    /// * `InvalidArgument` if `var` is not a symbol or the interval is
    ///   degenerate.
    /// * `ComputationFailed` if a coefficient integral has no closed form.
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// // Square wave sign(x) on [-π, π]: b_k = 4/(kπ) for odd k
    /// let sq = x.sign().fourier_series_on(&x, &(-ctx.pi()), &ctx.pi(), 3).unwrap();
    /// assert_eq!(sq.coefficient_b(1), 4 / ctx.pi());
    /// assert_eq!(sq.coefficient_b(3), 4 / (3 * ctx.pi()));
    /// assert!(sq.coefficient_b(2).is_zero_structural());
    /// // Triangle wave |x| on [-π, π]: a₀ = π, a_k = -4/(k²π) for odd k
    /// let tri = x.abs().fourier_series_on(&x, &(-ctx.pi()), &ctx.pi(), 2).unwrap();
    /// assert_eq!(format!("{}", tri.a0), "pi");
    /// assert_eq!(format!("{}", tri.coefficient_a(1)), "-4/pi");
    /// assert_eq!(format!("{}", tri.coefficient_a(2)), "0");
    /// ```
    pub fn fourier_series_on(
        &self,
        var: &Ex,
        lower: &Ex,
        upper: &Ex,
        n_terms: u32,
    ) -> Result<FourierSeries, SymplexError> {
        let ctx = self.context();
        let var_id = self.checked_id(var);
        let _ = self.checked_id(lower);
        let _ = self.checked_id(upper);
        {
            let inner = self.inner.read();
            if !matches!(
                inner.arena.node(var_id),
                crate::base::node::ExprNode::Symbol(_)
            ) {
                return Err(SymplexError::InvalidArgument {
                    operation: "fourier_series_on",
                    reason: "the expansion variable must be a symbol".into(),
                });
            }
        }
        let period = (upper - lower).eval();
        if period.is_zero_structural() || period.is_positive() == Some(false) {
            return Err(SymplexError::InvalidArgument {
                operation: "fourier_series_on",
                reason: format!("the interval [{lower}, {upper}] must have positive length"),
            });
        }
        let _span =
            debug_span!("fourier_series_on", expr = ?self.raw_id(), var = ?var_id).entered();

        let check = |c: Ex, what: String| -> Result<Ex, SymplexError> {
            if c.has_unevaluated() {
                Err(SymplexError::ComputationFailed {
                    operation: "fourier_series_on",
                    reason: format!("the coefficient integral for {what} has no closed form: {c}"),
                })
            } else {
                Ok(c)
            }
        };

        let a0_int = self.integrate_definite(var, lower, upper);
        let a0 = check((2 * a0_int / &period).simplify().eval(), "a_0".into())?;
        let mut series = FourierSeries {
            function: self.clone(),
            var: var.clone(),
            lower: lower.clone(),
            upper: upper.clone(),
            period,
            a0,
            an: Vec::with_capacity(n_terms as usize),
            bn: Vec::with_capacity(n_terms as usize),
        };
        for k in 1..=n_terms {
            let a = check(series.raw_coefficient(k, false), format!("a_{k}"))?;
            let b = check(series.raw_coefficient(k, true), format!("b_{k}"))?;
            series.an.push(a);
            series.bn.push(b);
        }
        let _ = ctx;
        Ok(series)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Z-transform
// ═══════════════════════════════════════════════════════════════════════════

// `Ex::z_transform` / `Ex::inverse_z_transform` live in
// `crate::calculus::z_transform` (pre-existing public API).

// ═══════════════════════════════════════════════════════════════════════════
// Mellin transform
// ═══════════════════════════════════════════════════════════════════════════

impl Expr<Numeric> {
    /// Mellin transform `M{f}(s) = ∫₀^∞ x^{s−1} f(x) dx` of this expression
    /// (a function of `x`) as a function of `s`, together with the
    /// *fundamental strip* `a < Re(s) < b` on which the integral converges,
    /// returned as a boolean condition on `Re(s)`.
    ///
    /// Table: `e^{−x} → Γ(s)`, `e^{−x²} → Γ(s/2)/2`, `1/(1+x) → π/sin(πs)`,
    /// `1/(1+x)^ν → B(s, ν−s)`, `H(1−x)·x^a → 1/(s+a)`, `H(x−1)·x^a →
    /// −1/(s+a)`, `H(1−x)(1−x)^b → B(s, b+1)`, `sin x → Γ(s) sin(πs/2)`,
    /// `cos x → Γ(s) cos(πs/2)`, `ln(1+x) → π/(s sin πs)`; rules: linearity,
    /// `x^a f → F(s+a)`, `f(ax) → a^{−s}F(s)`, `f(x^b) → F(s/b)/|b|`,
    /// `f′ → −(s−1)F(s−1)`, `ln(x) f → F′(s)`.
    ///
    /// Parameters are treated as real; sign conditions (`a > 0` in
    /// `e^{−ax}`) are checked through the assumption system and an
    /// unprovable condition is an error.
    ///
    /// # Errors
    ///
    /// * `InvalidArgument` if `x`/`s` are not distinct symbols.
    /// * `ComputationFailed` if no rule applies, a sign assumption is
    ///   missing, or the strips of two summands do not overlap.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let s = ctx.symbol("s");
    ///
    /// let (f, strip) = (-&x).exp().mellin_transform(&x, &s).unwrap();
    /// assert_eq!(f, s.gamma());
    /// assert_eq!(format!("{strip}"), "re(s) > 0");
    ///
    /// // x² e^{-3x}: power rule and scaling rule
    /// let (g, strip) = (x.powi(2) * (&x * -3).exp()).mellin_transform(&x, &s).unwrap();
    /// assert_eq!(g, ctx.int(3).pow(&(-&s - 2)) * (&s + 2).gamma());
    /// assert_eq!(format!("{strip}"), "re(s) > -2");
    ///
    /// // 1/(1+x)³ → B(s, 3 − s) on 0 < Re s < 3
    /// let (h, strip) = (1 / (1 + &x).powi(3)).mellin_transform(&x, &s).unwrap();
    /// assert_eq!(h, s.beta(&(3 - &s)));
    /// assert_eq!(format!("{strip}"), "re(s) > 0 & 3 > re(s)");
    /// ```
    pub fn mellin_transform(&self, x: &Ex, s: &Ex) -> Result<(Ex, BoolEx), SymplexError> {
        let x_id = self.checked_id(x);
        let s_id = self.checked_id(s);
        let _span = debug_span!("mellin_transform", expr = ?self.raw_id(), x = ?x_id).entered();
        let r = {
            let mut inner = self.inner.write();
            crate::calculus::mellin::mellin_transform(&mut inner.arena, self.raw_id(), x_id, s_id)
        };
        r.map(|(f, cond)| (self.wrap(f), self.wrap_as(cond)))
    }

    /// Inverse Mellin transform of this expression (a function of `s`) as a
    /// function of `x`, by table lookup (`Γ(s/b) → b e^{−x^b}`,
    /// `π/sin(πs) → 1/(1+x)`, `B(s, ν−s) → (1+x)^{−ν}`, `B(s, b+1) →
    /// H(1−x)(1−x)^b`, `1/(s+a) → x^a H(1−x)`, `Γ(s) sin(πs/2) → sin x`, …)
    /// with the shift (`G(s+a) → x^a g`) and scaling (`a^{−s}G → g(ax)`)
    /// rules applied in reverse.
    ///
    /// A Mellin transform determines its function only together with a
    /// strip. Where the table entry is ambiguous (`1/(s + a)` is the
    /// transform of `x^a H(1−x)` on `Re s > −a` and of `−x^a H(x−1)` on
    /// `Re s < −a`) the strip to the **right** of the pole is chosen.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let s = ctx.symbol("s");
    /// assert_eq!(s.gamma().inverse_mellin_transform(&s, &x).unwrap(), (-&x).exp());
    /// let f = (ctx.pi() / (ctx.pi() * &s).sin()).inverse_mellin_transform(&s, &x).unwrap();
    /// assert_eq!(f, 1 / (&x + 1));
    /// // round trip through the shift and scaling rules
    /// let g = x.powi(2) * (&x * -3).exp();
    /// let (big_g, _) = g.mellin_transform(&x, &s).unwrap();
    /// assert_eq!(big_g.inverse_mellin_transform(&s, &x).unwrap(), g);
    /// ```
    pub fn inverse_mellin_transform(&self, s: &Ex, x: &Ex) -> Result<Ex, SymplexError> {
        let s_id = self.checked_id(s);
        let x_id = self.checked_id(x);
        let _span =
            debug_span!("inverse_mellin_transform", expr = ?self.raw_id(), s = ?s_id).entered();
        let r = {
            let mut inner = self.inner.write();
            crate::calculus::mellin::inverse_mellin_transform(
                &mut inner.arena,
                self.raw_id(),
                s_id,
                x_id,
            )
        };
        r.map(|id| self.wrap(id))
    }
}
