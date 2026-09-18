//! Definite / improper / numeric integration methods on [`Ex`].

use tracing::debug_span;

use crate::api::expr::{Ex, Expr, Numeric};
use crate::base::errors::SymplexError;
use crate::base::node::ExprNode;
use crate::calculus::definite::{self, QuadOpts};

impl Expr<Numeric> {
    /// Compute the definite integral `∫_lo^hi self dvar`.
    ///
    /// Unlike a naive `F(hi) − F(lo)`, this locates singularities of the
    /// integrand inside the interval, treats infinite bounds and endpoint
    /// singularities as improper integrals via one-sided limits, applies
    /// symmetry shortcuts, resolves `Abs`/`Sign`/`Heaviside`/`DiracDelta`/
    /// `Piecewise` integrands, and consults a table of classical improper
    /// integrals when no elementary antiderivative exists.
    ///
    /// If the integral is divergent or cannot be evaluated the result is an
    /// **unevaluated** definite-integral node `Integral(f, x, lo, hi)`
    /// (see [`definite_integral_node`](Self::definite_integral_node)) —
    /// never a wrong finite number.  Use
    /// [`try_integrate_definite`](Self::try_integrate_definite) to
    /// distinguish "diverges" from "could not compute".
    ///
    /// If `self` is, or contains, such a node, the inner integrals are
    /// evaluated first (innermost out), so nested integrals can be built up
    /// with repeated calls.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    ///
    /// // ∫₀¹ x² dx = 1/3
    /// let v = x.powi(2).integrate_definite(&x, &ctx.int(0), &ctx.int(1));
    /// assert_eq!(format!("{v}"), "1/3");
    ///
    /// // ∫₀^∞ e^{−x} dx = 1
    /// let v = (-&x).exp().integrate_definite(&x, &ctx.int(0), &ctx.infinity());
    /// assert_eq!(format!("{v}"), "1");
    ///
    /// // ∫₋∞^∞ e^{−x²} dx = √π
    /// let v = (-x.powi(2)).exp().integrate_definite(&x, &ctx.neg_infinity(), &ctx.infinity());
    /// assert_eq!(format!("{v}"), "sqrt(pi)");
    ///
    /// // ∫₋₁¹ x⁻² dx diverges: the result stays unevaluated, bounds intact.
    /// let v = x.powi(-2).integrate_definite(&x, &ctx.int(-1), &ctx.int(1));
    /// assert!(v.has_unevaluated());
    /// assert!(v.is_definite_integral());
    /// assert_eq!(format!("{v}"), "Integral(x^(-2), x, -1, 1)");
    ///
    /// // ∫₀¹ xˣ dx has no closed form; the node still has a numeric value.
    /// let v = x.pow(&x).integrate_definite(&x, &ctx.int(0), &ctx.int(1));
    /// assert!(v.is_definite_integral());
    /// assert!((v.eval_f64().unwrap() - 0.7834305107).abs() < 1e-8);
    /// ```
    #[must_use = "returns the definite integral value"]
    pub fn integrate_definite(&self, var: &Ex, lo: &Ex, hi: &Ex) -> Ex {
        let var_id = self.checked_id(var);
        let lo_id = self.checked_id(lo);
        let hi_id = self.checked_id(hi);
        let _span =
            debug_span!("integrate_definite", expr = ?self.raw_id(), var = ?var_id).entered();
        let mut inner = self.inner.write();
        let id = match definite::integrate_definite(
            &mut inner.arena,
            self.raw_id(),
            var_id,
            lo_id,
            hi_id,
        ) {
            Ok(id) => id,
            Err(_) => definite::unevaluated_definite(
                &mut inner.arena,
                self.raw_id(),
                var_id,
                lo_id,
                hi_id,
            ),
        };
        drop(inner);
        self.wrap(id)
    }

    /// Like [`integrate_definite`](Self::integrate_definite), but returns
    /// `Err` instead of an unevaluated form.
    ///
    /// * [`SymplexError::Divergent`] — the integral was proven to diverge,
    /// * [`SymplexError::ComputationFailed`] — no closed form could be
    ///   established, or the result still contains an unevaluated form
    ///   (a `DefiniteIntegral`, `Integral`, `Limit`, … node),
    /// * [`SymplexError::InvalidArgument`] — `var` is not a symbol.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    ///
    /// let r = x.powi(-2).try_integrate_definite(&x, &ctx.int(-1), &ctx.int(1));
    /// assert!(matches!(r, Err(SymplexError::Divergent { .. })));
    ///
    /// let v = x.ln().try_integrate_definite(&x, &ctx.int(0), &ctx.int(1)).unwrap();
    /// assert_eq!(format!("{v}"), "-1");
    /// ```
    pub fn try_integrate_definite(&self, var: &Ex, lo: &Ex, hi: &Ex) -> Result<Ex, SymplexError> {
        let var_id = self.checked_id(var);
        let lo_id = self.checked_id(lo);
        let hi_id = self.checked_id(hi);
        let _span =
            debug_span!("try_integrate_definite", expr = ?self.raw_id(), var = ?var_id).entered();
        let mut inner = self.inner.write();
        let id =
            definite::integrate_definite(&mut inner.arena, self.raw_id(), var_id, lo_id, hi_id)?;
        let unevaluated = crate::base::walk::has_unevaluated(&inner.arena, id);
        drop(inner);
        if unevaluated {
            return Err(SymplexError::ComputationFailed {
                operation: "integrate_definite",
                reason: "result contains unevaluated forms".into(),
            });
        }
        Ok(self.wrap(id))
    }

    /// Build the formal, unevaluated definite integral `∫_lo^hi self dvar`
    /// **without** attempting to evaluate it.
    ///
    /// Useful for display, LaTeX, and formal manipulation (differentiation
    /// by the Leibniz rule, substitution into the bounds, numeric
    /// evaluation by quadrature).  Only the cheap structural folds are
    /// applied: `lo == hi` gives `0`, an integrand free of `var` over a
    /// finite interval gives `self · (hi − lo)`, and numeric bounds with
    /// `lo > hi` are reordered with a sign flip.  To evaluate the node
    /// later, call [`eval_integrals`](Self::eval_integrals) on it (or on any
    /// expression containing it).
    ///
    /// The integration variable is bound inside the integrand: it is not
    /// reported by [`free_symbols`](Self::free_symbols) and is not touched
    /// by [`subs`](Self::subs); the bounds are in the enclosing scope.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let t = ctx.symbol("t");
    ///
    /// let node = x.sin().definite_integral_node(&x, &ctx.int(0), &t);
    /// assert!(node.is_definite_integral());
    /// assert!(node.has_unevaluated());
    /// assert_eq!(format!("{node}"), "Integral(sin(x), x, 0, t)");
    /// assert_eq!(node.to_latex(), r"\int_{0}^{t} \sin\left(x\right)\, dx");
    ///
    /// // Only `t` is free; `x` is bound.
    /// assert_eq!(node.free_symbols().len(), 1);
    ///
    /// // Leibniz rule: d/dt ∫₀ᵗ sin(x) dx = sin(t).
    /// assert_eq!(format!("{}", node.diff(&t)), "sin(t)");
    ///
    /// // Evaluating recovers the closed form 1 − cos(t).
    /// let v = node.eval_integrals();
    /// assert_eq!(format!("{v}"), "-cos(t) + 1");
    /// ```
    #[must_use = "returns the formal definite integral node"]
    pub fn definite_integral_node(&self, var: &Ex, lo: &Ex, hi: &Ex) -> Ex {
        let var_id = self.checked_id(var);
        let lo_id = self.checked_id(lo);
        let hi_id = self.checked_id(hi);
        let mut inner = self.inner.write();
        let id = inner
            .arena
            .definite_integral(self.raw_id(), var_id, lo_id, hi_id);
        drop(inner);
        self.wrap(id)
    }

    /// Is this expression an unevaluated definite integral node
    /// (`Integral(f, x, lo, hi)`)?
    ///
    /// Only the root node is inspected; use
    /// [`has_unevaluated`](Self::has_unevaluated) to search the whole tree.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// // No closed form → the node comes back.
    /// let v = x.pow(&x).integrate_definite(&x, &ctx.int(0), &ctx.int(1));
    /// assert!(v.is_definite_integral());
    /// // Closed form → a number.
    /// let v = x.integrate_definite(&x, &ctx.int(0), &ctx.int(1));
    /// assert!(!v.is_definite_integral());
    /// ```
    #[must_use]
    pub fn is_definite_integral(&self) -> bool {
        let inner = self.inner.read();
        matches!(
            inner.arena.node(self.raw_id()),
            ExprNode::DefiniteIntegral(..)
        )
    }

    /// Evaluate every formal definite-integral node in this expression,
    /// innermost first.
    ///
    /// This is the "doit" operation for `Integral(f, x, lo, hi)` nodes (the
    /// analogue of [`eval_derivatives`](Self::eval_derivatives) for
    /// `Derivative`): each node is run through the definite integrator and
    /// replaced by its closed form.  Nodes that still cannot be evaluated —
    /// no closed form, or a proven divergence — are left in place, so the
    /// result is never a wrong finite number; check with
    /// [`has_unevaluated`](Self::has_unevaluated).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let t = ctx.symbol("t");
    ///
    /// // d/dt ∫₀¹ sin(t·x) dx = ∫₀¹ x·cos(t·x) dx, then evaluate it.
    /// let node = (&t * &x).sin().definite_integral_node(&x, &ctx.int(0), &ctx.int(1));
    /// let d = node.diff(&t);
    /// assert!(d.is_definite_integral());
    /// let v = d.eval_integrals();
    /// assert!(!v.has_unevaluated(), "{v}");
    ///
    /// // No closed form: the node is returned unchanged.
    /// let n = x.pow(&x).definite_integral_node(&x, &ctx.int(0), &ctx.int(1));
    /// assert!(n.eval_integrals().is_definite_integral());
    /// ```
    #[must_use = "returns the expression with definite integrals evaluated"]
    pub fn eval_integrals(&self) -> Ex {
        let _span = debug_span!("eval_integrals", expr = ?self.raw_id()).entered();
        let id = {
            let mut inner = self.inner.write();
            definite::evaluate_inner_definite(&mut inner.arena, self.raw_id())
        };
        self.wrap(id)
    }

    /// Numerically integrate `self` over `[lo, hi]` with adaptive
    /// Gauss–Kronrod (G7/K15) quadrature using the default [`QuadOpts`].
    ///
    /// Infinite bounds are supported.  The integrand is compiled with
    /// [`compile`](Self::compile), so it must contain no free symbols other
    /// than `var` and only nodes the compiler supports.
    ///
    /// # Errors
    ///
    /// * [`SymplexError::FreeSymbol`] — another free symbol is present,
    /// * [`SymplexError::NotImplemented`] — the integrand contains a node
    ///   that cannot be compiled to `f64` code,
    /// * [`SymplexError::Unevaluable`] — a bound is not a real number,
    /// * [`SymplexError::ComputationFailed`] — the quadrature did not reach
    ///   the requested accuracy (e.g. a divergent integral).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    ///
    /// let v = x.sin().integrate_numeric(&x, &ctx.int(0), &ctx.pi()).unwrap();
    /// assert!((v - 2.0).abs() < 1e-10);
    ///
    /// // ∫₀^∞ e^{−x²} dx = √π/2
    /// let v = (-x.powi(2)).exp().integrate_numeric(&x, &ctx.int(0), &ctx.infinity()).unwrap();
    /// assert!((v - std::f64::consts::PI.sqrt() / 2.0).abs() < 1e-10);
    ///
    /// // A divergent integral is reported as an error, not a number.
    /// assert!(x.powi(-2).integrate_numeric(&x, &ctx.int(-1), &ctx.int(1)).is_err());
    /// ```
    ///
    /// Conditionally convergent or slowly decaying oscillatory tails (e.g.
    /// `sin(x)/x` on `[0, ∞)`) are beyond plain adaptive quadrature and
    /// also return an error; use [`integrate_definite`](Self::integrate_definite)
    /// for those.
    pub fn integrate_numeric(&self, var: &Ex, lo: &Ex, hi: &Ex) -> Result<f64, SymplexError> {
        let opts = QuadOpts::default();
        let (value, err) = self.integrate_numeric_with(var, lo, hi, &opts)?;
        let tol = opts.abs_tol.max(opts.rel_tol * value.abs());
        if err > 1e3 * tol {
            return Err(SymplexError::ComputationFailed {
                operation: "integrate_numeric",
                reason: format!(
                    "quadrature did not converge: estimate {value} with error {err:e} (integral may diverge)"
                ),
            });
        }
        Ok(value)
    }

    /// Numerically integrate with explicit [`QuadOpts`], returning
    /// `(value, error_estimate)`.
    ///
    /// The error estimate is returned even if the tolerance was not met
    /// within `max_subdivisions`; check it before trusting the value.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::definite::QuadOpts;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let opts = QuadOpts { rel_tol: 1e-6, ..QuadOpts::default() };
    /// let (v, err) = (-x.powi(2)).exp()
    ///     .integrate_numeric_with(&x, &ctx.neg_infinity(), &ctx.infinity(), &opts)
    ///     .unwrap();
    /// assert!((v - std::f64::consts::PI.sqrt()).abs() < 1e-6);
    /// assert!(err < 1e-4);
    /// ```
    pub fn integrate_numeric_with(
        &self,
        var: &Ex,
        lo: &Ex,
        hi: &Ex,
        opts: &QuadOpts,
    ) -> Result<(f64, f64), SymplexError> {
        let var_id = self.checked_id(var);
        let lo_id = self.checked_id(lo);
        let hi_id = self.checked_id(hi);

        let mut inner = self.inner.write();
        let arena = &mut inner.arena;

        let var_name = match arena.node(var_id) {
            ExprNode::Symbol(sid) => arena.symbol_name(*sid).to_string(),
            _ => {
                return Err(SymplexError::InvalidArgument {
                    operation: "integrate_numeric",
                    reason: "integration variable must be a symbol".into(),
                });
            }
        };

        // Bounds → f64 (infinities allowed).
        let bound_f64 = |arena: &mut crate::base::arena::Arena, id| -> Result<f64, SymplexError> {
            if id == arena.infinity() {
                return Ok(f64::INFINITY);
            }
            if id == arena.neg_infinity() {
                return Ok(f64::NEG_INFINITY);
            }
            crate::transforms::evalf::eval_const_f64(arena, id).ok_or_else(|| {
                SymplexError::Unevaluable {
                    reason: format!(
                        "integration bound {} is not a real number",
                        arena.display(id)
                    ),
                }
            })
        };
        let a = bound_f64(arena, lo_id)?;
        let b = bound_f64(arena, hi_id)?;

        // Free symbols other than the variable.
        let evaled = crate::transforms::eval::eval(arena, self.raw_id());
        for s in crate::base::walk::free_symbols(arena, evaled) {
            if s != var_id
                && let ExprNode::Symbol(sid) = arena.node(s)
            {
                return Err(SymplexError::FreeSymbol {
                    name: arena.symbol_name(*sid).to_string(),
                });
            }
        }

        let func = crate::output::lambdify::compile(arena, evaled, &[&var_name]).map_err(|e| {
            SymplexError::NotImplemented(format!(
                "integrand {} cannot be compiled for numeric quadrature: {e}",
                arena.display(evaled)
            ))
        })?;
        drop(inner);

        let f = |t: f64| func(&[t]);
        definite::quadrature(&f, a, b, opts)
    }

    /// Residue of `self` at `var = ∞`, defined as
    /// `Res_{z=∞} f(z) = −Res_{t=0} f(1/t)/t²`.
    ///
    /// The sum of all finite residues plus the residue at infinity is
    /// zero for a function meromorphic on the extended plane.  If the
    /// residue cannot be computed, a formal `Residue` node is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let z = ctx.symbol("z");
    /// // f = 1/z has residue 1 at 0, hence −1 at ∞.
    /// let f = &ctx.int(1) / &z;
    /// assert_eq!(format!("{}", f.residue_at_infinity(&z)), "-1");
    /// // Polynomials: Res_{∞} z = −Res_{t=0} 1/t³ = 0.
    /// assert_eq!(format!("{}", z.residue_at_infinity(&z)), "0");
    /// ```
    #[must_use = "returns the residue value; does not modify in place"]
    pub fn residue_at_infinity(&self, var: &Ex) -> Ex {
        let var_id = self.checked_id(var);
        let _span =
            debug_span!("residue_at_infinity", expr = ?self.raw_id(), var = ?var_id).entered();
        let mut inner = self.inner.write();
        match crate::calculus::residue::residue_at_infinity(&mut inner.arena, self.raw_id(), var_id)
        {
            Ok(id) => {
                drop(inner);
                self.wrap(id)
            }
            Err(_) => {
                let inf = inner.arena.infinity();
                let id = inner
                    .arena
                    .intern(ExprNode::Residue(self.raw_id(), var_id, inf));
                drop(inner);
                self.wrap(id)
            }
        }
    }
}
