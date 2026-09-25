//! Numeric expression methods — math functions, calculus, algebra, solving.
//!
//! This module contains all `impl Expr<Numeric>` method blocks.
//! The type definitions live in `expr.rs`.

use num_bigint::BigInt;
use num_complex::Complex64;
use num_traits::Zero;
use tracing::debug_span;

use crate::api::expr::{BoolEx, Ex, Expr, Numeric, SetEx, SetValued};
use crate::base::assumptions::{Assumption, Props};
use crate::base::errors::SymplexError;
use crate::base::interval::Interval;

/// Ascending coefficients of `expr` as a polynomial in `var`, allowing
/// arbitrary `var`-free symbolic coefficients.  The zero polynomial is the
/// empty list, matching the rational-coefficient path in `polybridge`.
pub(crate) fn symbolic_coeffs_of(
    arena: &mut crate::base::arena::Arena,
    expr: crate::base::node::ExprId,
    var: crate::base::node::ExprId,
) -> Option<Vec<crate::base::node::ExprId>> {
    let coeffs = crate::transforms::solve::symbolic_poly_coeffs(arena, expr, var)?;
    if coeffs.len() == 1 && arena.is_zero_structural(coeffs[0]) {
        return Some(vec![]);
    }
    Some(coeffs)
}

// ═══════════════════════════════════════════════════════════════════════════
// impl Expr<Numeric> — Numeric-specific methods
// ═══════════════════════════════════════════════════════════════════════════

impl Expr<Numeric> {
    // ── Math functions ─────────────────────────────────────────────

    /// Raise to a symbolic power: `self ^ exp`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn pow(&self, exp: &Ex) -> Ex {
        let exp_id = self.checked_id(exp);
        let id = self.inner.write().arena.pow(self.raw_id(), exp_id);
        self.wrap(id)
    }

    /// Raise to an integer power: `self ^ n`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn powi(&self, n: i64) -> Ex {
        let mut inner = self.inner.write();
        let exp = inner.arena.int(n);
        let id = inner.arena.pow(self.raw_id(), exp);
        drop(inner);
        self.wrap(id)
    }

    /// Sine: `sin(self)`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn sin(&self) -> Ex {
        let id = self.inner.write().arena.sin(self.raw_id());
        self.wrap(id)
    }

    /// Cosine: `cos(self)`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn cos(&self) -> Ex {
        let id = self.inner.write().arena.cos(self.raw_id());
        self.wrap(id)
    }

    /// Tangent: `tan(self)`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn tan(&self) -> Ex {
        let id = self.inner.write().arena.tan(self.raw_id());
        self.wrap(id)
    }

    /// Natural exponential: `e^self`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn exp(&self) -> Ex {
        let id = self.inner.write().arena.exp(self.raw_id());
        self.wrap(id)
    }

    /// Natural logarithm: `ln(self)`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn ln(&self) -> Ex {
        let id = self.inner.write().arena.ln(self.raw_id());
        self.wrap(id)
    }

    /// Logarithm with arbitrary base: `log_base(self)`.
    ///
    /// Computed as `ln(self) / ln(base)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let result = x.log(&ctx.int(2));
    /// let s = format!("{result}");
    /// assert!(s.contains("ln"), "log should be expressed in terms of ln, got: {s}");
    /// ```
    #[must_use]
    pub fn log(&self, base: &Ex) -> Ex {
        let ln_self = self.ln();
        let ln_base = base.ln();
        &ln_self / &ln_base
    }

    /// Principal square root: `√self`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn sqrt(&self) -> Ex {
        let id = self.inner.write().arena.sqrt(self.raw_id());
        self.wrap(id)
    }

    /// Principal cube root: `self^(1/3)`.
    ///
    /// Like every power, this is the principal branch: `∛(−8) = 2·(−1)^(1/3)
    /// = 1 + √3·i` (SymPy's `cbrt`).  For the real cube root of a negative
    /// real use [`real_root`](Self::real_root).
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn cbrt(&self) -> Ex {
        let id = self.inner.write().arena.cbrt(self.raw_id());
        self.wrap(id)
    }

    /// The real `n`-th root (SymPy's `real_root`).
    ///
    /// For odd `n` and real `self` this is the real `r` with `rⁿ = self`:
    /// `sign(self)·|self|^(1/n)`, so `real_root(−8, 3) = −2` where the
    /// principal [`cbrt`](Self::cbrt) is `1 + √3·i`.  For even `n` it is the
    /// principal root, real exactly when `self ≥ 0`.  When `self` is not
    /// known to be real the odd case is
    /// `Piecewise((sign(x)·|x|^(1/n), im(x) = 0), (x^(1/n), True))`, as in
    /// SymPy.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] for `n = 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// assert_eq!(ctx.int(-8).real_root(3)?, ctx.int(-2));
    /// assert_eq!(ctx.rational(-1, 8).real_root(3)?, ctx.rational(-1, 2));
    /// assert_eq!(ctx.int(-4).real_root(2)?, ctx.int(-4).sqrt());       // 2*I: no real root
    /// let r = ctx.symbol_with("r", &[Assumption::Real]).unwrap();
    /// assert_eq!(format!("{}", r.real_root(3)?), "cbrt(abs(r))*sign(r)");
    /// assert!((ctx.int(-2).real_root(3)?.eval_f64()? + 2f64.cbrt()).abs() < 1e-15);
    /// # Ok::<(), SymplexError>(())
    /// ```
    pub fn real_root(&self, n: u32) -> Result<Ex, SymplexError> {
        if n == 0 {
            return Err(SymplexError::invalid_argument(
                "real_root",
                "the root index n must be at least 1",
            ));
        }
        let ctx = self.context();
        let exp = ctx.rational(1, i64::from(n));
        let principal = self.pow(&exp);
        if n.is_multiple_of(2) {
            return Ok(principal);
        }
        let real_form = &self.sign() * &self.abs().pow(&exp);
        if self.as_rational().is_some() {
            return Ok(real_form.eval());
        }
        if self.is_real_valued() == Some(true) {
            return Ok(real_form);
        }
        let on_real_line = self.im().eq_expr(&ctx.zero());
        Ok(Ex::piecewise(&[
            (&real_form, &on_real_line),
            (&principal, &ctx.bool_true()),
        ]))
    }

    /// Nth root: `self^(1/n)`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn nthroot(&self, n: i64) -> Ex {
        let mut inner = self.inner.write();
        let frac = inner.arena.rational(1, n);
        let id = inner.arena.pow(self.raw_id(), frac);
        drop(inner);
        self.wrap(id)
    }

    /// Absolute value (or complex modulus): `|self|`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn abs(&self) -> Ex {
        let id = self.inner.write().arena.abs(self.raw_id());
        self.wrap(id)
    }

    /// Inverse sine (arcsin): `asin(self)`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn asin(&self) -> Ex {
        let id = self.inner.write().arena.asin(self.raw_id());
        self.wrap(id)
    }

    /// Inverse cosine (arccos): `acos(self)`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn acos(&self) -> Ex {
        let id = self.inner.write().arena.acos(self.raw_id());
        self.wrap(id)
    }

    /// Inverse tangent (arctan): `atan(self)`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn atan(&self) -> Ex {
        let id = self.inner.write().arena.atan(self.raw_id());
        self.wrap(id)
    }

    /// Two-argument arctangent: `atan2(y, x)`.
    ///
    /// Returns the angle in (-π, π] between the positive x-axis and the
    /// point (x, y). Correctly handles all four quadrants.
    #[must_use]
    pub fn atan2(&self, x: &Ex) -> Ex {
        let x_id = self.checked_id(x);
        let id = self.inner.write().arena.atan2(self.raw_id(), x_id);
        self.wrap(id)
    }

    /// Hyperbolic sine: `sinh(self)`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn sinh(&self) -> Ex {
        let id = self.inner.write().arena.sinh(self.raw_id());
        self.wrap(id)
    }

    /// Hyperbolic cosine: `cosh(self)`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn cosh(&self) -> Ex {
        let id = self.inner.write().arena.cosh(self.raw_id());
        self.wrap(id)
    }

    /// Hyperbolic tangent: `tanh(self)`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn tanh(&self) -> Ex {
        let id = self.inner.write().arena.tanh(self.raw_id());
        self.wrap(id)
    }

    /// Inverse hyperbolic sine: `asinh(self)`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn asinh(&self) -> Ex {
        let id = self.inner.write().arena.asinh(self.raw_id());
        self.wrap(id)
    }

    /// Inverse hyperbolic cosine: `acosh(self)`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn acosh(&self) -> Ex {
        let id = self.inner.write().arena.acosh(self.raw_id());
        self.wrap(id)
    }

    /// Inverse hyperbolic tangent: `atanh(self)`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn atanh(&self) -> Ex {
        let id = self.inner.write().arena.atanh(self.raw_id());
        self.wrap(id)
    }

    // ── Reciprocal trig / hyperbolic convenience methods ───────────

    /// Secant: `sec(x) = 1/cos(x)`.
    #[must_use]
    pub fn sec(&self) -> Ex {
        let one_id = self.inner.read().arena.one();
        let one = self.wrap(one_id);
        &one / &self.cos()
    }

    /// Cosecant: `csc(x) = 1/sin(x)`.
    #[must_use]
    pub fn csc(&self) -> Ex {
        let one_id = self.inner.read().arena.one();
        let one = self.wrap(one_id);
        &one / &self.sin()
    }

    /// Cotangent: `cot(x) = cos(x)/sin(x)`.
    #[must_use]
    pub fn cot(&self) -> Ex {
        &self.cos() / &self.sin()
    }

    /// Inverse cotangent: `acot(x) = atan(1/x)`.
    #[must_use]
    pub fn acot(&self) -> Ex {
        let one_id = self.inner.read().arena.one();
        let one = self.wrap(one_id);
        (&one / self).atan()
    }

    /// Inverse secant: `asec(x) = acos(1/x)`.
    #[must_use]
    pub fn asec(&self) -> Ex {
        let one_id = self.inner.read().arena.one();
        let one = self.wrap(one_id);
        (&one / self).acos()
    }

    /// Inverse cosecant: `acsc(x) = asin(1/x)`.
    #[must_use]
    pub fn acsc(&self) -> Ex {
        let one_id = self.inner.read().arena.one();
        let one = self.wrap(one_id);
        (&one / self).asin()
    }

    /// Hyperbolic cotangent: `coth(x) = cosh(x)/sinh(x)`.
    #[must_use]
    pub fn coth(&self) -> Ex {
        &self.cosh() / &self.sinh()
    }

    /// Hyperbolic secant: `sech(x) = 1/cosh(x)`.
    #[must_use]
    pub fn sech(&self) -> Ex {
        let one_id = self.inner.read().arena.one();
        let one = self.wrap(one_id);
        &one / &self.cosh()
    }

    /// Hyperbolic cosecant: `csch(x) = 1/sinh(x)`.
    #[must_use]
    pub fn csch(&self) -> Ex {
        let one_id = self.inner.read().arena.one();
        let one = self.wrap(one_id);
        &one / &self.sinh()
    }

    /// Inverse hyperbolic cotangent: `acoth(x) = atanh(1/x)`.
    #[must_use]
    pub fn acoth(&self) -> Ex {
        let one_id = self.inner.read().arena.one();
        let one = self.wrap(one_id);
        (&one / self).atanh()
    }

    /// Inverse hyperbolic secant: `asech(x) = acosh(1/x)`.
    #[must_use]
    pub fn asech(&self) -> Ex {
        let one_id = self.inner.read().arena.one();
        let one = self.wrap(one_id);
        (&one / self).acosh()
    }

    /// Inverse hyperbolic cosecant: `acsch(x) = asinh(1/x)`.
    #[must_use]
    pub fn acsch(&self) -> Ex {
        let one_id = self.inner.read().arena.one();
        let one = self.wrap(one_id);
        (&one / self).asinh()
    }

    /// Cardinal sine: `sinc(x) = sin(x)/x`, with `sinc(0) = 1` (requires limit).
    #[must_use]
    pub fn sinc(&self) -> Ex {
        &self.sin() / self
    }

    /// Sign function: 1 if positive, -1 if negative, 0 if zero.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn sign(&self) -> Ex {
        let id = self.inner.write().arena.sign(self.raw_id());
        self.wrap(id)
    }

    /// Floor function: `⌊self⌋` (greatest integer ≤ self).
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn floor(&self) -> Ex {
        let id = self.inner.write().arena.floor(self.raw_id());
        self.wrap(id)
    }

    /// Ceiling function: `⌈self⌉` (least integer ≥ self).
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn ceiling(&self) -> Ex {
        let id = self.inner.write().arena.ceiling(self.raw_id());
        self.wrap(id)
    }

    /// Fractional part: `self - floor(self)`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn frac(&self) -> Ex {
        let fl = self.floor();
        self - &fl
    }

    /// Remainder: `self - other * floor(self / other)`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn rem(&self, other: &Ex) -> Ex {
        let quotient = self / other;
        let fl = quotient.floor();
        self - &(other * &fl)
    }

    /// Binary minimum: `min(self, other)`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn min_with(&self, other: &Ex) -> Ex {
        let other_id = self.checked_id(other);
        let mut inner = self.inner.write();
        let ids: smallvec::SmallVec<[crate::base::node::ExprId; 4]> =
            smallvec::smallvec![self.raw_id(), other_id];
        let id = inner.arena.intern(crate::base::node::ExprNode::Min(ids));
        drop(inner);
        self.wrap(id)
    }

    /// Binary maximum: `max(self, other)`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn max_with(&self, other: &Ex) -> Ex {
        let other_id = self.checked_id(other);
        let mut inner = self.inner.write();
        let ids: smallvec::SmallVec<[crate::base::node::ExprId; 4]> =
            smallvec::smallvec![self.raw_id(), other_id];
        let id = inner.arena.intern(crate::base::node::ExprNode::Max(ids));
        drop(inner);
        self.wrap(id)
    }

    /// N-ary minimum of a collection of expressions.
    pub fn min_of(ctx: &crate::api::context::Context, exprs: impl IntoIterator<Item = Ex>) -> Ex {
        let items: Vec<Ex> = exprs.into_iter().collect();
        if items.is_empty() {
            return ctx.infinity();
        }
        let mut inner = items[0].inner.write();
        let ids: smallvec::SmallVec<[crate::base::node::ExprId; 4]> =
            items.iter().map(|e| e.raw_id()).collect();
        let id = inner.arena.intern(crate::base::node::ExprNode::Min(ids));
        drop(inner);
        items[0].wrap(id)
    }

    /// N-ary maximum of a collection of expressions.
    pub fn max_of(ctx: &crate::api::context::Context, exprs: impl IntoIterator<Item = Ex>) -> Ex {
        let items: Vec<Ex> = exprs.into_iter().collect();
        if items.is_empty() {
            return ctx.neg_infinity();
        }
        let mut inner = items[0].inner.write();
        let ids: smallvec::SmallVec<[crate::base::node::ExprId; 4]> =
            items.iter().map(|e| e.raw_id()).collect();
        let id = inner.arena.intern(crate::base::node::ExprNode::Max(ids));
        drop(inner);
        items[0].wrap(id)
    }

    /// Symbolic summation: `Sum(body, var=lower..upper)` (unevaluated node).
    ///
    /// Construction is cheap and does nothing.  `.eval()` runs the symbolic
    /// summation engine (see [`summation`](Self::summation)): small concrete
    /// ranges are enumerated exactly, symbolic and infinite bounds get closed
    /// forms where known.  Use [`summation`](Self::summation) to evaluate
    /// directly without building the node.
    pub fn symbolic_sum(body: &Ex, var: &Ex, lower: &Ex, upper: &Ex) -> Ex {
        let var_id = body.checked_id(var);
        let lower_id = body.checked_id(lower);
        let upper_id = body.checked_id(upper);
        let mut inner = body.inner.write();
        let id = inner.arena.intern(crate::base::node::ExprNode::Sum(
            body.raw_id(),
            var_id,
            lower_id,
            upper_id,
        ));
        drop(inner);
        body.wrap(id)
    }

    /// Symbolic product: `Product(body, var=lower..upper)` (unevaluated node).
    ///
    /// When evaluated (`.eval()`), if `lower` and `upper` are concrete integers,
    /// the product is computed by substituting each integer value for `var` in
    /// `body`.  Use [`product_over`](Self::product_over) for closed forms with
    /// symbolic or infinite bounds.
    pub fn symbolic_product(body: &Ex, var: &Ex, lower: &Ex, upper: &Ex) -> Ex {
        let var_id = body.checked_id(var);
        let lower_id = body.checked_id(lower);
        let upper_id = body.checked_id(upper);
        let mut inner = body.inner.write();
        let id = inner.arena.intern(crate::base::node::ExprNode::Product_(
            body.raw_id(),
            var_id,
            lower_id,
            upper_id,
        ));
        drop(inner);
        body.wrap(id)
    }

    /// Test whether the infinite series `Σ_{k=1}^{∞} self(var)` converges
    /// (absolutely or conditionally).
    ///
    /// Returns `Some(true)` if the series converges, `Some(false)` if it
    /// diverges, or `None` if the tests are inconclusive — never a guess.
    ///
    /// Tests applied: divergence test, rational-function degree test,
    /// exact Stirling growth analysis for hypergeometric-type terms (this
    /// subsumes the ratio, root, p-series and alternating-series tests),
    /// direct comparison for bounded factors (`sin`, `cos`), and the
    /// integral test for log-exp terms such as `1/(k ln² k)`.  See also
    /// [`is_absolutely_convergent`](Self::is_absolutely_convergent).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let k = ctx.symbol("k");
    /// // 1/k² converges (p-series with p=2)
    /// assert_eq!(k.powi(-2).is_convergent(&k), Some(true));
    /// // k!/k^k converges (ratio test, limit 1/e)
    /// assert_eq!((k.factorial() / k.pow(&k)).is_convergent(&k), Some(true));
    /// // k/(k+1) diverges (terms do not tend to zero)
    /// assert_eq!((&k / &(&k + 1)).is_convergent(&k), Some(false));
    /// ```
    #[must_use]
    pub fn is_convergent(&self, var: &Ex) -> Option<bool> {
        let var_id = self.checked_id(var);
        let _span = debug_span!("is_convergent").entered();
        let mut inner = self.inner.write();
        inner.arena.is_convergent_expr(self.raw_id(), var_id)
    }

    /// Attempt closed-form evaluation of a symbolic sum.
    ///
    /// Given a sum `Σ_{var=lower}^{upper} body`, tries to find a closed-form
    /// expression using Faulhaber formulas, geometric series, linearity, etc.
    ///
    /// Returns the closed form if found, or the original sum expression unchanged.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let k = ctx.symbol("k");
    /// let n = ctx.symbol("n");
    /// let s = Ex::symbolic_sum(&k, &k, &ctx.int(1), &n);
    /// let closed = s.closed_form_sum();
    /// // Should give n*(n+1)/2
    /// ```
    #[must_use]
    pub fn closed_form_sum(&self) -> Ex {
        let _span = debug_span!("closed_form_sum").entered();
        let mut inner = self.inner.write();
        let node = inner.arena.node(self.raw_id()).clone();
        if let crate::base::node::ExprNode::Sum(body, var, lower, upper) = node
            && let Some(closed) = inner.arena.eval_sum_symbolic_expr(body, var, lower, upper)
        {
            let result = crate::transforms::eval::eval(&mut inner.arena, closed);
            drop(inner);
            return self.wrap(result);
        }
        drop(inner);
        self.clone()
    }

    // `re`, `im`, `conjugate`, `arg` and the other complex-analysis methods
    // live in `expr_complex.rs`.

    // ── Special functions (native ExprNode variants) ───────────────

    /// Gamma function: Γ(self).
    ///
    /// For positive integer arguments, `.eval()` computes `(n-1)!`.
    /// `Gamma(1/2) = √π`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let result = ctx.int(5).gamma().eval();
    /// assert_eq!(format!("{result}"), "24");
    /// ```
    #[must_use]
    pub fn gamma(&self) -> Ex {
        let id = self.inner.write().arena.gamma(self.raw_id());
        self.wrap(id)
    }

    /// Log-gamma function: ln(Γ(self)).
    ///
    /// For positive integer arguments, `.eval()` computes `ln((n-1)!)`.
    #[must_use]
    pub fn log_gamma(&self) -> Ex {
        let id = self.inner.write().arena.log_gamma(self.raw_id());
        self.wrap(id)
    }

    /// Digamma function: ψ(self) = Γ'(self)/Γ(self).
    #[must_use]
    pub fn digamma(&self) -> Ex {
        let id = self.inner.write().arena.digamma(self.raw_id());
        self.wrap(id)
    }

    /// Error function: erf(self) = 2/√π ∫₀ˢᵉˡᶠ e^(-t²) dt.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let result = ctx.int(0).erf().eval();
    /// assert_eq!(format!("{result}"), "0");
    /// ```
    #[must_use]
    pub fn erf(&self) -> Ex {
        let id = self.inner.write().arena.erf(self.raw_id());
        self.wrap(id)
    }

    /// Complementary error function: erfc(self) = 1 - erf(self).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let result = ctx.int(0).erfc().eval();
    /// assert_eq!(format!("{result}"), "1");
    /// ```
    #[must_use]
    pub fn erfc(&self) -> Ex {
        let id = self.inner.write().arena.erfc(self.raw_id());
        self.wrap(id)
    }

    /// Beta function: B(self, other) = Γ(self)Γ(other)/Γ(self+other).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let result = ctx.int(2).beta(&ctx.int(3)).eval();
    /// assert_eq!(format!("{result}"), "1/12");
    /// ```
    #[must_use]
    pub fn beta(&self, other: &Ex) -> Ex {
        let other_id = self.checked_id(other);
        let id = self.inner.write().arena.beta(self.raw_id(), other_id);
        self.wrap(id)
    }

    /// Creates a `Factorial` node. For non-negative integer arguments,
    /// `.eval()` will compute the exact value using arbitrary-precision
    /// arithmetic.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let result = ctx.int(5).factorial().eval();
    /// assert_eq!(format!("{result}"), "120");
    /// ```
    #[must_use]
    pub fn factorial(&self) -> Ex {
        let id = self.inner.write().arena.factorial(self.raw_id());
        self.wrap(id)
    }

    /// Compute the binomial coefficient C(self, k).
    ///
    /// Creates a `Binomial(self, k)` node. For non-negative integer
    /// arguments, `.eval()` will compute the exact value.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let result = ctx.int(10).binomial(&ctx.int(3)).eval();
    /// assert_eq!(format!("{result}"), "120");
    /// ```
    #[must_use]
    pub fn binomial(&self, k: &Ex) -> Ex {
        let k_id = self.checked_id(k);
        let id = self.inner.write().arena.binomial(self.raw_id(), k_id);
        self.wrap(id)
    }

    // ── Combinatorial functions (Apply-based) ──────────────────────

    /// Double factorial: `self!!`.
    ///
    /// For non-negative integer arguments, `.eval()` computes the exact value.
    /// `0!! = 1`, `1!! = 1`, `(-1)!! = 1`.
    #[must_use]
    pub fn factorial2(&self) -> Ex {
        let id = self.inner.write().arena.factorial2(self.raw_id());
        self.wrap(id)
    }

    /// Subfactorial (derangement count): `!self`.
    ///
    /// For non-negative integer arguments, `.eval()` computes the exact value.
    #[must_use]
    pub fn subfactorial(&self) -> Ex {
        let id = self.inner.write().arena.subfactorial(self.raw_id());
        self.wrap(id)
    }

    /// Rising factorial (Pochhammer symbol): `(self)_n`.
    ///
    /// `rising_factorial(x, n) = x * (x+1) * ... * (x+n-1)`.
    #[must_use]
    pub fn rising_factorial(&self, n: &Ex) -> Ex {
        let n_id = self.checked_id(n);
        let id = self
            .inner
            .write()
            .arena
            .rising_factorial(self.raw_id(), n_id);
        self.wrap(id)
    }

    /// Falling factorial: `self^(n) = self * (self-1) * ... * (self-n+1)`.
    #[must_use]
    pub fn falling_factorial(&self, n: &Ex) -> Ex {
        let n_id = self.checked_id(n);
        let id = self
            .inner
            .write()
            .arena
            .falling_factorial(self.raw_id(), n_id);
        self.wrap(id)
    }

    /// Fibonacci number: `F(self)`.
    ///
    /// For non-negative integer arguments, `.eval()` computes the exact value.
    /// `F(0) = 0`, `F(1) = 1`, `F(n) = F(n-1) + F(n-2)`.
    #[must_use]
    pub fn fibonacci(&self) -> Ex {
        let id = self.inner.write().arena.fibonacci(self.raw_id());
        self.wrap(id)
    }

    /// Lucas number: `L(self)`.
    ///
    /// For non-negative integer arguments, `.eval()` computes the exact value.
    /// `L(0) = 2`, `L(1) = 1`, `L(n) = L(n-1) + L(n-2)`.
    #[must_use]
    pub fn lucas(&self) -> Ex {
        let id = self.inner.write().arena.lucas(self.raw_id());
        self.wrap(id)
    }

    /// Bernoulli number: `B(self)`.
    ///
    /// For non-negative integer arguments, `.eval()` computes the exact value.
    /// `B(0) = 1`, `B(1) = -1/2`, `B(2) = 1/6`.
    #[must_use]
    pub fn bernoulli_number(&self) -> Ex {
        let id = self.inner.write().arena.bernoulli_number(self.raw_id());
        self.wrap(id)
    }

    /// Harmonic number: `H(self) = 1 + 1/2 + ... + 1/self`.
    ///
    /// For non-negative integer arguments, `.eval()` computes the exact value.
    /// `H(0) = 0`.
    #[must_use]
    pub fn harmonic(&self) -> Ex {
        let id = self.inner.write().arena.harmonic(self.raw_id());
        self.wrap(id)
    }

    /// Catalan number: `C(self) = (2n)! / ((n+1)! * n!)`.
    ///
    /// For non-negative integer arguments, `.eval()` computes the exact value.
    #[must_use]
    pub fn catalan_number(&self) -> Ex {
        let id = self.inner.write().arena.catalan_number(self.raw_id());
        self.wrap(id)
    }

    /// Bell number: `B(self)`.
    ///
    /// For non-negative integer arguments, `.eval()` computes the exact value.
    /// `B(0) = 1`, `B(1) = 1`, `B(2) = 2`, `B(3) = 5`.
    #[must_use]
    pub fn bell(&self) -> Ex {
        let id = self.inner.write().arena.bell(self.raw_id());
        self.wrap(id)
    }

    /// Euler number: `E(self)`.
    ///
    /// For non-negative integer arguments, `.eval()` computes the exact value.
    /// Odd indices are 0. `E(0) = 1`, `E(2) = -1`, `E(4) = 5`.
    #[must_use]
    pub fn euler_number(&self) -> Ex {
        let id = self.inner.write().arena.euler_number(self.raw_id());
        self.wrap(id)
    }

    // ── Combinatorial functions — Phase 1 (Apply-based) ────────────

    /// Stirling number of the second kind: `S(self, k)`.
    ///
    /// Counts the number of ways to partition a set of `self` elements
    /// into exactly `k` non-empty subsets.
    ///
    /// For non-negative integer arguments, `.eval()` computes the exact value.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let result = ctx.int(5).stirling2(&ctx.int(3)).eval();
    /// assert_eq!(format!("{result}"), "25");
    /// ```
    #[must_use]
    pub fn stirling2(&self, k: &Ex) -> Ex {
        let k_id = self.checked_id(k);
        let id = self.inner.write().arena.stirling2(self.raw_id(), k_id);
        self.wrap(id)
    }

    /// Signed Stirling number of the first kind: `s(self, k)`.
    ///
    /// Related to the number of permutations of `self` elements with
    /// exactly `k` cycles.  Satisfies `x^{(n)} = Σ_k s(n, k) x^k`
    /// where `x^{(n)}` is the falling factorial.
    ///
    /// For non-negative integer arguments, `.eval()` computes the exact value.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let result = ctx.int(4).stirling1(&ctx.int(2)).eval();
    /// assert_eq!(format!("{result}"), "11");
    /// ```
    #[must_use]
    pub fn stirling1(&self, k: &Ex) -> Ex {
        let k_id = self.checked_id(k);
        let id = self.inner.write().arena.stirling1(self.raw_id(), k_id);
        self.wrap(id)
    }

    /// Number of integer partitions of `self`.
    ///
    /// An integer partition of `n` is a way to write `n` as a sum of
    /// positive integers (order doesn't matter).
    ///
    /// For non-negative integer arguments, `.eval()` computes the exact value.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let result = ctx.int(5).partition_count().eval();
    /// assert_eq!(format!("{result}"), "7");
    /// ```
    #[must_use]
    pub fn partition_count(&self) -> Ex {
        let id = self.inner.write().arena.partition_count(self.raw_id());
        self.wrap(id)
    }

    // ── Special functions (Apply-based) ────────────────────────────

    /// Heaviside step function: 0 for x<0, 1/2 for x=0, 1 for x>0.
    #[must_use]
    pub fn heaviside(&self) -> Ex {
        let id = self.inner.write().arena.heaviside(self.raw_id());
        self.wrap(id)
    }

    /// Dirac delta distribution: 0 for x≠0, symbolic at x=0.
    #[must_use]
    pub fn dirac_delta(&self) -> Ex {
        let id = self.inner.write().arena.dirac_delta(self.raw_id());
        self.wrap(id)
    }

    /// Lambert W function (principal branch): W(x)·exp(W(x)) = x.
    #[must_use]
    pub fn lambertw(&self) -> Ex {
        let id = self.inner.write().arena.lambertw(self.raw_id());
        self.wrap(id)
    }

    // ── Relational operators (return BoolEx) ───────────────────────

    /// Greater than: `self > other`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn gt(&self, other: &Ex) -> BoolEx {
        let other_id = self.checked_id(other);
        let id = self.inner.write().arena.gt(self.raw_id(), other_id);
        self.wrap_as(id)
    }

    /// Greater than or equal: `self >= other`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn ge(&self, other: &Ex) -> BoolEx {
        let other_id = self.checked_id(other);
        let id = self.inner.write().arena.ge(self.raw_id(), other_id);
        self.wrap_as(id)
    }

    /// Less than: `self < other` (implemented as `other > self`).
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn lt(&self, other: &Ex) -> BoolEx {
        let other_id = self.checked_id(other);
        let id = self.inner.write().arena.gt(other_id, self.raw_id());
        self.wrap_as(id)
    }

    /// Less than or equal: `self <= other` (implemented as `other >= self`).
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn le(&self, other: &Ex) -> BoolEx {
        let other_id = self.checked_id(other);
        let id = self.inner.write().arena.ge(other_id, self.raw_id());
        self.wrap_as(id)
    }

    /// Mathematical equality test (boolean-valued): `self == other`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn eq_expr(&self, other: &Ex) -> BoolEx {
        let other_id = self.checked_id(other);
        let id = self.inner.write().arena.eq_(self.raw_id(), other_id);
        self.wrap_as(id)
    }

    /// Not-equal test (boolean-valued): `self != other`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn ne_expr(&self, other: &Ex) -> BoolEx {
        let other_id = self.checked_id(other);
        let id = self.inner.write().arena.ne_(self.raw_id(), other_id);
        self.wrap_as(id)
    }

    /// Piecewise function from `(value, condition)` pairs.
    ///
    /// Returns the value of the first pair whose condition is true.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn piecewise(pairs: &[(&Ex, &BoolEx)]) -> Ex {
        if pairs.is_empty() {
            return crate::api::context::Context::new().zero();
        }
        let first = pairs[0].0;
        let arena_pairs: Vec<(crate::base::node::ExprId, crate::base::node::ExprId)> = pairs
            .iter()
            .map(|(v, c)| (first.checked_id(v), first.checked_id(c)))
            .collect();
        let id = first.inner.write().arena.piecewise(&arena_pairs);
        first.wrap(id)
    }

    // ── Assumption queries ─────────────────────────────────────────

    /// Query whether this expression is positive.
    ///
    /// Returns `Some(true)` if provably positive, `Some(false)` if
    /// provably not positive, or `None` if unknown.
    #[must_use]
    pub fn is_positive(&self) -> Option<bool> {
        let inner = self.inner.read();
        inner
            .assumptions
            .lock()
            .query(&inner.arena, self.raw_id(), Props::POSITIVE)
    }

    /// Query whether this expression is zero.
    ///
    /// Uses layered detection:
    /// 1. Structural identity with 0 (O(1))
    /// 2. Assumption system query
    ///
    /// Returns `Some(true)` if provably zero, `Some(false)` if provably
    /// nonzero, or `None` if unknown.
    #[must_use]
    pub fn is_zero(&self) -> Option<bool> {
        let inner = self.inner.read();
        // Layer 1: structural.
        if inner.arena.is_zero_structural(self.raw_id()) {
            return Some(true);
        }
        // Layer 2: assumption system.
        inner
            .assumptions
            .lock()
            .query(&inner.arena, self.raw_id(), Props::ZERO)
    }

    /// Query any mathematical property via [`Props`].
    ///
    /// Returns `Some(true)` if provably true, `Some(false)` if provably
    /// false, or `None` if unknown.
    #[must_use]
    pub fn query(&self, prop: Props) -> Option<bool> {
        let inner = self.inner.read();
        inner
            .assumptions
            .lock()
            .query(&inner.arena, self.raw_id(), prop)
    }

    /// Query whether this expression is negative.
    ///
    /// Returns `Some(true)` if provably negative, `Some(false)` if
    /// provably not negative, or `None` if unknown.
    #[must_use]
    pub fn is_negative(&self) -> Option<bool> {
        self.query(Props::NEGATIVE)
    }

    /// Query whether this expression is real.
    ///
    /// Returns `Some(true)` if provably real, `Some(false)` if
    /// provably not real, or `None` if unknown.
    #[must_use]
    pub fn is_real(&self) -> Option<bool> {
        self.query(Props::REAL)
    }

    /// Query whether this expression is an integer.
    ///
    /// Returns `Some(true)` if provably an integer, `Some(false)` if
    /// provably not an integer, or `None` if unknown.
    #[must_use]
    pub fn is_integer(&self) -> Option<bool> {
        self.query(Props::INTEGER)
    }

    /// Query whether this expression is nonzero.
    ///
    /// Returns `Some(true)` if provably nonzero, `Some(false)` if
    /// provably zero, or `None` if unknown.
    #[must_use]
    pub fn is_nonzero(&self) -> Option<bool> {
        self.query(Props::NONZERO)
    }

    /// Query whether this expression is finite.
    ///
    /// Returns `Some(true)` if provably finite, `Some(false)` if
    /// provably not finite, or `None` if unknown.
    #[must_use]
    pub fn is_finite(&self) -> Option<bool> {
        self.query(Props::FINITE)
    }

    /// Returns `Some(true)` if this expression is known to be ≥ 0.
    #[must_use]
    pub fn is_nonnegative(&self) -> Option<bool> {
        self.query(Props::NONNEGATIVE)
    }

    /// Returns `Some(true)` if this expression is known to be ≤ 0.
    #[must_use]
    pub fn is_nonpositive(&self) -> Option<bool> {
        self.query(Props::NONPOSITIVE)
    }

    /// Returns `Some(true)` if this expression is known to be imaginary.
    #[must_use]
    pub fn is_imaginary(&self) -> Option<bool> {
        self.query(Props::IMAGINARY)
    }

    /// Returns `Some(true)` if this expression is known to be complex.
    #[must_use]
    pub fn is_complex(&self) -> Option<bool> {
        self.query(Props::COMPLEX)
    }

    /// Returns `Some(true)` if this expression is known to be rational.
    #[must_use]
    pub fn is_rational(&self) -> Option<bool> {
        self.query(Props::RATIONAL)
    }

    /// Returns whether this expression is known to be even.
    #[must_use]
    pub fn is_even(&self) -> Option<bool> {
        self.query(Props::EVEN)
    }

    /// Returns whether this expression is known to be odd.
    #[must_use]
    pub fn is_odd(&self) -> Option<bool> {
        self.query(Props::ODD)
    }

    /// Returns whether this expression is known to be prime.
    #[must_use]
    pub fn is_prime(&self) -> Option<bool> {
        self.query(Props::PRIME)
    }

    /// Returns whether this expression is known to be composite.
    #[must_use]
    pub fn is_composite(&self) -> Option<bool> {
        self.query(Props::COMPOSITE)
    }

    /// Returns whether this expression is known to be algebraic.
    #[must_use]
    pub fn is_algebraic(&self) -> Option<bool> {
        self.query(Props::ALGEBRAIC)
    }

    /// Returns whether this expression is known to be transcendental.
    #[must_use]
    pub fn is_transcendental(&self) -> Option<bool> {
        self.query(Props::TRANSCENDENTAL)
    }

    /// Returns whether this expression is known to be irrational.
    #[must_use]
    pub fn is_irrational(&self) -> Option<bool> {
        self.query(Props::IRRATIONAL)
    }

    /// Returns whether this expression is known to be hermitian.
    #[must_use]
    pub fn is_hermitian(&self) -> Option<bool> {
        self.query(Props::HERMITIAN)
    }

    // ── Assumption mutation ────────────────────────────────────────

    /// Set a mathematical assumption on this expression (must be a symbol).
    ///
    /// Returns `self` for fluent chaining. If this expression is not a
    /// symbol, the assumption is silently ignored.  The assumption is added
    /// to those already declared on the symbol.
    ///
    /// # Errors
    ///
    /// [`SymplexError::ContradictoryAssumptions`] if the new assumption
    /// contradicts those already declared on the symbol once their
    /// consequences are drawn (`Negative` on a `Positive` symbol,
    /// `Irrational` on an `Integer` one, …); the symbol is then left
    /// unchanged.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let t = ctx.symbol("t")
    ///     .assume(Assumption::Positive)?
    ///     .assume(Assumption::Real)?;
    /// assert_eq!(t.is_positive(), Some(true));
    /// assert_eq!(t.is_real(), Some(true));
    /// assert!(matches!(
    ///     t.clone().assume(Assumption::Negative),
    ///     Err(SymplexError::ContradictoryAssumptions { .. })
    /// ));
    /// assert_eq!(t.is_positive(), Some(true));
    /// # Ok::<(), SymplexError>(())
    /// ```
    pub fn assume(self, assumption: Assumption) -> Result<Ex, SymplexError> {
        use crate::base::node::ExprNode;
        let mut inner = self.inner.write();
        if let ExprNode::Symbol(sid) = inner.arena.node(self.raw_id()) {
            let sid = *sid;
            // Checked under the same lock as the update, so neither setter
            // below can meet a contradictory set.
            let a = crate::base::assumptions::Assumptions::declare(
                inner.arena.symbols.name(sid),
                inner.arena.symbol_assumptions(sid),
                &[assumption],
            )?;
            inner.arena.set_symbol_assumptions(sid, a);
            inner
                .assumptions
                .lock()
                .set_symbol_assumptions(self.raw_id(), a);
        }
        drop(inner);
        Ok(self)
    }

    /// Mathematical equality: attempts to determine whether `self == other`
    /// as mathematical objects.
    ///
    /// Three-valued:
    ///
    /// * `Some(true)` — `self − other` is structurally zero, or becomes zero
    ///   after [`expand`](Self::expand) or [`simplify`](Self::simplify), or
    ///   the assumption system proves the difference is zero.
    /// * `Some(false)` — the difference (possibly after expand/simplify)
    ///   is a **nonzero rational constant**, the assumption system proves it
    ///   nonzero (e.g. `x² + 1` for real `x`), or both sides are constants
    ///   (no free symbols) whose 16-digit numeric values differ by more
    ///   than `1e-9` relative.
    /// * `None` — undetermined.  In particular `x.equals(&y)` for distinct
    ///   free symbols is `None`, not `Some(false)`; use
    ///   [`probably_equal`](Self::probably_equal) for a randomized test.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// assert_eq!((&x + 1).powi(2).equals(&(&x.powi(2) + &x * 2 + 1)), Some(true));
    /// assert_eq!((&x.sin().powi(2) + &x.cos().powi(2)).equals(&ctx.one()), Some(true));
    /// assert_eq!(x.equals(&(&x + 1)), Some(false));
    /// assert_eq!(ctx.int(1).equals(&ctx.int(2)), Some(false));
    /// assert_eq!(ctx.pi().equals(&ctx.rational(22, 7)), Some(false));
    /// assert_eq!(x.equals(&ctx.symbol("y")), None);
    /// ```
    #[must_use]
    pub fn equals(&self, other: &Ex) -> Option<bool> {
        // Layer 1: structural identity (same arena node).
        let other_id = self.checked_id(other);
        if self.raw_id() == other_id {
            return Some(true);
        }

        // Layer 2: compute self - other; zero ⇒ equal, nonzero literal ⇒ not.
        let diff = self - other;
        if let Some(known) = Self::equals_from_difference(&diff) {
            return Some(known);
        }

        // Layer 3: expand the difference and check again.
        let expanded = diff.expand();
        if let Some(known) = Self::equals_from_difference(&expanded) {
            return Some(known);
        }

        // Layer 4: simplify the difference (catches trig identities, etc.)
        let simplified = diff.simplify();
        if let Some(known) = Self::equals_from_difference(&simplified) {
            return Some(known);
        }

        // Layer 5: assumption system on the simplified difference.
        if simplified.is_zero() == Some(true) {
            return Some(true);
        }
        if simplified.is_nonzero() == Some(true) {
            return Some(false);
        }

        // Layer 6: both sides are constants — compare 16-digit numeric values.
        if simplified.is_constant()
            && let (Ok(a), Ok(b)) = (self.eval_complex64(), other.eval_complex64())
        {
            let scale = 1.0_f64.max(a.norm()).max(b.norm());
            if (a - b).norm() > 1e-9 * scale {
                return Some(false);
            }
        }

        // Could not determine equality.
        None
    }

    /// `Some(true)` if `diff` is structurally zero, `Some(false)` if it is a
    /// nonzero numeric literal, `None` otherwise.
    fn equals_from_difference(diff: &Ex) -> Option<bool> {
        let inner = diff.inner.read();
        if inner.arena.is_zero_structural(diff.raw_id()) {
            return Some(true);
        }
        match inner.arena.node(diff.raw_id()) {
            crate::base::node::ExprNode::Num(_) => Some(false),
            _ => None,
        }
    }

    // ── Calculus ───────────────────────────────────────────────────

    /// Symbolic differentiation with respect to `var`.
    ///
    /// Computes the derivative using standard rules (linearity, product
    /// rule, chain rule, power rule) for all supported node types.
    ///
    /// `var` should be a symbol expression (created via `ctx.symbol()`).
    /// If `var` does not appear in the expression, the result is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let expr = x.powi(3);
    /// let deriv = expr.diff(&x);
    /// assert_eq!(format!("{deriv}"), "3*x^2");
    /// ```
    #[must_use = "returns the derivative as a new expression"]
    pub fn diff(&self, var: &Ex) -> Ex {
        let var_id = self.checked_id(var);
        let _span = debug_span!("diff", expr = ?self.raw_id(), var = ?var_id).entered();
        let id = self.inner.write().arena.diff_wrt(self.raw_id(), var_id);
        self.wrap(id)
    }

    /// Like [`diff`](Self::diff), but returns `Err` if the result contains
    /// unevaluated forms (e.g. formal `Derivative` nodes).
    pub fn try_diff(&self, var: &Ex) -> Result<Ex, SymplexError> {
        let result = self.diff(var);
        if result.has_unevaluated() {
            Err(SymplexError::ComputationFailed {
                operation: "diff",
                reason: "derivative contains unevaluated forms".into(),
            })
        } else {
            Ok(result)
        }
    }

    /// Create a formal (unevaluated) derivative node.
    ///
    /// Unlike [`diff`](Self::diff) which computes the derivative,
    /// this creates a `Derivative(self, var)` node that represents
    /// "the derivative of self with respect to var" without evaluating it.
    /// This is used for constructing ODEs.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let y = ctx.symbol("y");
    /// let dy_dx = y.formal_diff(&x);
    /// let s = format!("{dy_dx}");
    /// assert!(s.contains("Derivative") || s.contains("d/d"), "got: {s}");
    /// ```
    #[must_use]
    pub fn formal_diff(&self, var: &Ex) -> Ex {
        let var_id = self.checked_id(var);
        let id = self
            .inner
            .write()
            .arena
            .intern(crate::base::node::ExprNode::Derivative(
                self.raw_id(),
                var_id,
            ));
        self.wrap(id)
    }

    /// Differentiate treating certain symbols as dependent on `var`.
    ///
    /// For any symbol `dep` in `dependent_vars`, `d/d(var)(dep)` returns
    /// a formal `Derivative(dep, var)` instead of zero. This enables
    /// implicit differentiation and ODE construction.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let y = ctx.symbol("y");
    /// let expr = &x.powi(2) + &y.powi(2);
    /// let result = expr.diff_with_dependent(&x, &[&y]);
    /// // d/dx(x² + y²) with y depending on x = 2x + 2y·dy/dx
    /// ```
    pub fn diff_with_dependent(&self, var: &Ex, dependent_vars: &[&Ex]) -> Ex {
        let var_id = self.checked_id(var);
        let mut deps = rustc_hash::FxHashSet::default();
        for dep in dependent_vars {
            deps.insert(self.checked_id(dep));
        }
        let id = {
            let mut guard = self.inner.write();
            crate::transforms::diff::diff_with_deps(&mut guard.arena, self.raw_id(), var_id, &deps)
        };
        self.wrap(id)
    }

    /// Concretely evaluate all formal `Derivative` nodes in this expression.
    ///
    /// This is the "doit" operation: each `Derivative(f, x)` node is replaced
    /// by the result of actually differentiating `f` with respect to `x`.
    /// Useful after substituting a solution into an ODE for verification.
    pub fn eval_derivatives(&self) -> Ex {
        let id = {
            let mut guard = self.inner.write();
            crate::transforms::subs::eval_derivatives(&mut guard.arena, self.raw_id())
        };
        self.wrap(id)
    }

    /// Compute the nth derivative with respect to `var`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let f = x.powi(4);
    /// let d3 = f.diff_n(&x, 3);
    /// assert_eq!(format!("{d3}"), "24*x");
    /// ```
    #[must_use]
    pub fn diff_n(&self, var: &Ex, n: usize) -> Ex {
        let mut result = self.clone();
        for _ in 0..n {
            result = result.diff(var);
        }
        result
    }

    /// Compute the indefinite integral with respect to `var`.
    ///
    /// Supports power rule, trigonometric, exponential, linearity,
    /// and constant factor extraction. For integrands that don't match
    /// any known rule, returns an unevaluated `Integral(body, var)` node.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let expr = x.powi(2);
    /// let anti = expr.integrate(&x);
    /// assert_eq!(format!("{anti}"), "1/3*x^3");
    /// ```
    #[must_use = "returns the antiderivative; does not modify in place"]
    pub fn integrate(&self, var: &Ex) -> Ex {
        let var_id = self.checked_id(var);
        let _span = debug_span!("integrate", expr = ?self.raw_id(), var = ?var_id).entered();
        let id = self
            .inner
            .write()
            .arena
            .integrate_expr(self.raw_id(), var_id);
        self.wrap(id)
    }

    /// Like [`integrate`](Self::integrate), but returns `Err` if the result
    /// contains unevaluated forms (e.g. formal `Integral` nodes).
    pub fn try_integrate(&self, var: &Ex) -> Result<Ex, SymplexError> {
        let result = self.integrate(var);
        if result.has_unevaluated() {
            Err(SymplexError::ComputationFailed {
                operation: "integrate",
                reason: "integral contains unevaluated forms".into(),
            })
        } else {
            Ok(result)
        }
    }

    /// Compute the Taylor / Laurent series around `point` with every term
    /// of exponent `< order` in `(var − point)`.
    ///
    /// Poles at `point` give negative powers (`1/sin x = 1/x + x/6 + …`);
    /// `point = ±∞` gives the asymptotic expansion in `1/var` (see
    /// [`series_at_infinity`](Self::series_at_infinity)).  Elementary
    /// functions use closed-form coefficients, so high orders stay fast.
    /// If no Laurent expansion exists (fractional-power or logarithmic
    /// singularity, essential singularity) a formal `Series` node is
    /// returned; see [`try_series`](Self::try_series).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let s = x.exp().series(&x, &ctx.int(0), 4);
    /// assert_eq!(s.to_string(), "1/6*x^3 + 1/2*x^2 + x + 1");
    ///
    /// // ln x about 1
    /// let s = x.ln().series(&x, &ctx.int(1), 3);
    /// assert_eq!(s.to_string(), "x - 1/2*(x - 1)^2 - 1");
    ///
    /// // Laurent expansion at a pole
    /// let s = (&ctx.int(1) / &x.sin()).series(&x, &ctx.int(0), 2);
    /// assert_eq!(s.to_string(), "1/x + 1/6*x");
    /// ```
    #[must_use = "returns the series expansion; does not modify in place"]
    pub fn series(&self, var: &Ex, point: &Ex, order: u32) -> Ex {
        let var_id = self.checked_id(var);
        let point_id = self.checked_id(point);
        let _span = debug_span!("series", expr = ?self.raw_id(), order = order).entered();
        let mut inner = self.inner.write();
        match inner
            .arena
            .series_expr(self.raw_id(), var_id, point_id, order)
        {
            Ok(id) => {
                drop(inner);
                self.wrap(id)
            }
            Err(_) => {
                let order_id = inner.arena.int(order as i64);
                let id = inner.arena.intern(crate::base::node::ExprNode::Series(
                    self.raw_id(),
                    var_id,
                    point_id,
                    order_id,
                ));
                drop(inner);
                self.wrap(id)
            }
        }
    }

    /// Like [`series`](Self::series), but returns `Err` if the result
    /// contains unevaluated forms (e.g. a formal `Series` node).
    pub fn try_series(&self, var: &Ex, point: &Ex, order: u32) -> Result<Ex, SymplexError> {
        let result = self.series(var, point, order);
        if result.has_unevaluated() {
            Err(SymplexError::ComputationFailed {
                operation: "series",
                reason: "could not compute series expansion".into(),
            })
        } else {
            Ok(result)
        }
    }

    /// Compute the Maclaurin series (Taylor series around 0) with every
    /// term of exponent `< order`.
    ///
    /// This is a convenience shorthand for `self.series(var, &zero, order)`
    /// that avoids needing to construct a zero expression manually.
    /// If the series cannot be computed, returns a formal `Series` node.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// assert_eq!(x.sin().maclaurin(&x, 6).to_string(), "1/120*x^5 - 1/6*x^3 + x");
    /// assert_eq!(x.atan().maclaurin(&x, 6).to_string(), "1/5*x^5 - 1/3*x^3 + x");
    /// // sqrt(x)·sin(x) is a Puiseux series: kept as a formal node
    /// assert!((&x.sqrt() * &x.sin()).maclaurin(&x, 4).has_unevaluated());
    /// ```
    #[must_use = "returns the series expansion; does not modify in place"]
    pub fn maclaurin(&self, var: &Ex, order: u32) -> Ex {
        let var_id = self.checked_id(var);
        let mut inner = self.inner.write();
        let zero = inner.arena.zero;
        match inner.arena.series_expr(self.raw_id(), var_id, zero, order) {
            Ok(id) => {
                drop(inner);
                self.wrap(id)
            }
            Err(_) => {
                let order_id = inner.arena.int(order as i64);
                let id = inner.arena.intern(crate::base::node::ExprNode::Series(
                    self.raw_id(),
                    var_id,
                    zero,
                    order_id,
                ));
                drop(inner);
                self.wrap(id)
            }
        }
    }

    /// Like [`maclaurin`](Self::maclaurin), but returns `Err` if the result
    /// contains unevaluated forms (e.g. a formal `Series` node).
    pub fn try_maclaurin(&self, var: &Ex, order: u32) -> Result<Ex, SymplexError> {
        let result = self.maclaurin(var, order);
        if result.has_unevaluated() {
            Err(SymplexError::ComputationFailed {
                operation: "maclaurin",
                reason: "could not compute Maclaurin series".into(),
            })
        } else {
            Ok(result)
        }
    }

    /// Compute the residue of this expression at `var = point`.
    ///
    /// The residue is the coefficient of `1/(x-a)` in the Laurent series
    /// expansion of the function around `a`.  Poles of any order are
    /// handled: the order `m` is detected from the denominator (or by
    /// limits for non-polynomial denominators) and
    /// `Res = 1/(m−1)! · d^{m−1}/dx^{m−1} [(x−a)^m f(x)]` at `x = a`.
    /// Essential singularities and undetectable cases yield a formal
    /// `Residue` node.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// // Res(1/x, x=0) = 1
    /// let f = &ctx.int(1) / &x;
    /// let result = f.residue(&x, &ctx.int(0));
    /// assert_eq!(format!("{result}"), "1");
    ///
    /// // Double pole: Res(e^x / x², x=0) = 1
    /// let g = &x.exp() / &x.powi(2);
    /// assert_eq!(format!("{}", g.residue(&x, &ctx.int(0))), "1");
    /// ```
    #[must_use = "returns the residue value; does not modify in place"]
    pub fn residue(&self, var: &Ex, point: &Ex) -> Ex {
        let var_id = self.checked_id(var);
        let point_id = self.checked_id(point);
        let _span = debug_span!("residue", expr = ?self.raw_id(), var = ?var_id).entered();
        let mut inner = self.inner.write();
        match inner.arena.residue_expr(self.raw_id(), var_id, point_id) {
            Ok(id) => {
                drop(inner);
                self.wrap(id)
            }
            Err(_) => {
                let id = inner.arena.intern(crate::base::node::ExprNode::Residue(
                    self.raw_id(),
                    var_id,
                    point_id,
                ));
                drop(inner);
                self.wrap(id)
            }
        }
    }

    /// Like [`residue`](Self::residue), but returns `Err` if the result
    /// contains unevaluated forms (e.g. a formal `Residue` node).
    pub fn try_residue(&self, var: &Ex, point: &Ex) -> Result<Ex, SymplexError> {
        let result = self.residue(var, point);
        if result.has_unevaluated() {
            Err(SymplexError::ComputationFailed {
                operation: "residue",
                reason: "could not compute residue".into(),
            })
        } else {
            Ok(result)
        }
    }

    /// Compute the Fourier series of this expression over \[-π, π\]
    /// with `n_terms` harmonics.
    ///
    /// Returns the truncated Fourier trigonometric series:
    /// `a₀/2 + Σ_{n=1}^{N} [aₙ cos(nx) + bₙ sin(nx)]`
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let f = ctx.int(1);
    /// let result = f.fourier_series(&x, 2);
    /// // Fourier series of constant 1 should evaluate to ≈ 1
    /// ```
    #[must_use = "returns the Fourier series; does not modify in place"]
    pub fn fourier_series(&self, var: &Ex, n_terms: u32) -> Ex {
        let var_id = self.checked_id(var);
        let _span = debug_span!("fourier_series", expr = ?self.raw_id(), var = ?var_id).entered();
        // Prefer the exact definite-integral coefficients (handles |x|,
        // sign, Heaviside and piecewise inputs); fall back to the
        // antiderivative-based expansion when a coefficient has no closed
        // form.
        let ctx = self.context();
        let pi = ctx.pi();
        if let Ok(series) = self.fourier_series_on(var, &(-&pi), &pi, n_terms) {
            return series.truncate(n_terms);
        }
        let id = self
            .inner
            .write()
            .arena
            .fourier_series_expr(self.raw_id(), var_id, n_terms);
        self.wrap(id)
    }

    /// Compute the Laplace transform L{self}(s) with respect to time variable `t`.
    ///
    /// Uses a table-based approach supporting constants, polynomials in `t`,
    /// exponentials, trigonometric and hyperbolic functions, plus linearity
    /// and the frequency-shift property.
    ///
    /// If the transform cannot be computed, returns a formal
    /// `LaplaceTransform(…)` node (check with [`has_unevaluated`](Self::has_unevaluated)).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let t = ctx.symbol("t");
    /// let s = ctx.symbol("s");
    /// // L{exp(2t)} = 1/(s-2)
    /// let result = (&t * 2).exp().laplace(&t, &s);
    /// ```
    #[must_use = "returns the Laplace transform; does not modify in place"]
    pub fn laplace(&self, t: &Ex, s: &Ex) -> Ex {
        let t_id = self.checked_id(t);
        let s_id = self.checked_id(s);
        let _span = debug_span!("laplace", expr = ?self.raw_id(), t = ?t_id, s = ?s_id).entered();
        let mut inner = self.inner.write();
        match inner
            .arena
            .laplace_transform_expr(self.raw_id(), t_id, s_id)
        {
            Ok(id) => {
                drop(inner);
                self.wrap(id)
            }
            Err(_) => {
                let id = inner
                    .arena
                    .intern(crate::base::node::ExprNode::LaplaceTransform(
                        self.raw_id(),
                        t_id,
                        s_id,
                    ));
                drop(inner);
                self.wrap(id)
            }
        }
    }

    /// Like [`laplace`](Self::laplace), but returns `Err` if the result
    /// contains unevaluated forms (e.g. a formal `LaplaceTransform` node).
    pub fn try_laplace(&self, t: &Ex, s: &Ex) -> Result<Ex, SymplexError> {
        let result = self.laplace(t, s);
        if result.has_unevaluated() {
            Err(SymplexError::ComputationFailed {
                operation: "laplace",
                reason: "could not compute Laplace transform".into(),
            })
        } else {
            Ok(result)
        }
    }

    /// Compute the inverse Laplace transform L⁻¹{self}(t) with respect to
    /// frequency variable `s`.
    ///
    /// Uses partial fraction decomposition followed by table lookup for
    /// each term.
    ///
    /// If the transform cannot be computed, returns a formal
    /// `InverseLaplaceTransform(…)` node (check with [`has_unevaluated`](Self::has_unevaluated)).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let t = ctx.symbol("t");
    /// let s = ctx.symbol("s");
    /// // L⁻¹{1/s} = 1
    /// let result = (&ctx.int(1) / &s).inverse_laplace(&s, &t);
    /// ```
    #[must_use = "returns the inverse Laplace transform; does not modify in place"]
    pub fn inverse_laplace(&self, s: &Ex, t: &Ex) -> Ex {
        let s_id = self.checked_id(s);
        let t_id = self.checked_id(t);
        let _span =
            debug_span!("inverse_laplace", expr = ?self.raw_id(), s = ?s_id, t = ?t_id).entered();
        let mut inner = self.inner.write();
        match inner
            .arena
            .inverse_laplace_transform_expr(self.raw_id(), s_id, t_id)
        {
            Ok(id) => {
                drop(inner);
                self.wrap(id)
            }
            Err(_) => {
                let id = inner
                    .arena
                    .intern(crate::base::node::ExprNode::InverseLaplaceTransform(
                        self.raw_id(),
                        s_id,
                        t_id,
                    ));
                drop(inner);
                self.wrap(id)
            }
        }
    }

    /// Like [`inverse_laplace`](Self::inverse_laplace), but returns `Err` if
    /// the result contains unevaluated forms (e.g. a formal
    /// `InverseLaplaceTransform` node).
    pub fn try_inverse_laplace(&self, s: &Ex, t: &Ex) -> Result<Ex, SymplexError> {
        let result = self.inverse_laplace(s, t);
        if result.has_unevaluated() {
            Err(SymplexError::ComputationFailed {
                operation: "inverse_laplace",
                reason: "could not compute inverse Laplace transform".into(),
            })
        } else {
            Ok(result)
        }
    }

    /// Compute the limit of this expression as `var` approaches `point`.
    ///
    /// Uses direct substitution, L'Hôpital's rule (for 0/0 and ∞/∞),
    /// and series expansion as fallbacks. If the limit cannot be
    /// determined, returns a formal `Limit` node.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// // lim_{x→0} sin(x)/x = 1
    /// let expr = &x.sin() / &x;
    /// let result = expr.limit(&x, &ctx.int(0));
    /// assert_eq!(format!("{result}"), "1");
    /// ```
    #[must_use = "returns the limit value; does not modify in place"]
    pub fn limit(&self, var: &Ex, point: &Ex) -> Ex {
        // Two-sided limit: both one-sided limits must exist and agree.
        // See `limit_dir` for the directional variants.
        self.limit_dir(var, point, crate::calculus::limit::Direction::Both)
    }

    /// Like [`limit`](Self::limit), but returns `Err` if the limit does not
    /// exist or cannot be computed.
    ///
    /// `±∞` are legitimate limit values (`Ok(oo)`), matching the analytic
    /// notion of a limit in the extended reals. When the left and right
    /// limits differ the error reason reads
    /// `"left and right limits differ: left = …, right = …"`.
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// assert_eq!(format!("{}", x.exp().try_limit(&x, &ctx.infinity()).unwrap()), "oo");
    /// assert!((1 / &x).try_limit(&x, &ctx.int(0)).is_err());
    /// ```
    pub fn try_limit(&self, var: &Ex, point: &Ex) -> Result<Ex, SymplexError> {
        self.try_limit_dir(var, point, crate::calculus::limit::Direction::Both)
    }

    // ── Algebra ────────────────────────────────────────────────────

    /// Expand trigonometric functions with composite arguments.
    ///
    /// Applies addition formulas:
    /// - `sin(a + b)` → `sin(a)·cos(b) + cos(a)·sin(b)`
    /// - `cos(a + b)` → `cos(a)·cos(b) - sin(a)·sin(b)`
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// let expr = (&x + &y).sin();
    /// let expanded = expr.expand_trig();
    /// let s = format!("{expanded}");
    /// assert!(s.contains("sin(x)") && s.contains("cos(y)"), "should expand: {s}");
    /// ```
    #[must_use = "returns the expanded form; does not modify in place"]
    pub fn expand_trig(&self) -> Ex {
        let id = self.inner.write().arena.expand_trig_expr(self.raw_id());
        self.wrap(id)
    }

    /// Expand logarithms where the identities hold (SymPy's `expand_log`).
    ///
    /// `ln(a·b) → ln a + ln b` and `ln(a^e) → e·ln a` hold for positive
    /// reals, not for every complex value (`ln((−1)·(−1)) = 0 ≠ 2πi`), so
    /// by default a product splits off only the factors known positive (and
    /// known-negative ones as `ln(−f)`, their sign kept in the rest), and a
    /// power `e·ln a` only for `a > 0` with `e` real, or `−1 < e ≤ 1`
    /// (`ln √x = ½ ln x` for every `x`).
    /// [`expand_log_with(true)`](Self::expand_log_with) applies both
    /// unconditionally.  (Before 0.23 the forced form was the default.)
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// assert_eq!((&x * &y).ln().expand_log(), (&x * &y).ln());   // x = y = −1 would break it
    /// assert_eq!(format!("{}", (&x * 2).ln().expand_log()), "ln(2) + ln(x)");
    /// assert_eq!(format!("{}", x.sqrt().ln().expand_log()), "1/2*ln(x)");
    /// let (p, q) = (ctx.symbol_with("p", &[Assumption::Positive]).unwrap(), ctx.symbol_with("q", &[Assumption::Positive]).unwrap());
    /// assert_eq!(format!("{}", (&p * q.powi(2)).ln().expand_log()), "2*ln(q) + ln(p)");
    /// ```
    #[must_use = "returns the expanded form; does not modify in place"]
    pub fn expand_log(&self) -> Ex {
        let id = self.inner.write().arena.expand_log_expr(self.raw_id());
        self.wrap(id)
    }

    /// Combine logarithms where the identities hold (inverse of
    /// [`expand_log`](Self::expand_log); SymPy's `logcombine`).
    ///
    /// `ln a + ln b → ln(a·b)` and `c·ln a → ln(a^c)` hold for positive
    /// reals, not for every complex value, so by default the logarithms of
    /// known-positive arguments combine with each other and with at most
    /// one other logarithm (`ln 2 + ln x = ln(2x)` for every complex `x`:
    /// `arg 2 = 0`), and `c·ln a` becomes `ln(a^c)` only for `a > 0` with `c`
    /// real, or `−1 < c ≤ 1`.  [`log_combine_with(true)`](Self::log_combine_with)
    /// combines unconditionally.  (Before 0.23 the forced form was the
    /// default.)
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// let e = &x.ln() + &y.ln();
    /// assert_eq!(e.log_combine(), e);                          // x = y = −1 would break it
    /// assert_eq!(format!("{}", (ctx.int(2).ln() + x.ln()).log_combine()), "ln(2*x)");
    /// let p = ctx.symbol_with("p", &[Assumption::Positive]).unwrap();
    /// assert_eq!(format!("{}", (&p.ln() * 3).log_combine()), "ln(p^3)");
    /// ```
    #[must_use = "returns the combined form; does not modify in place"]
    pub fn log_combine(&self) -> Ex {
        let id = self.inner.write().arena.log_combine_expr(self.raw_id());
        self.wrap(id)
    }

    /// Apply trigonometric product-to-sum and double-angle identities.
    ///
    /// - `sin(a)·cos(b) → ½[sin(a+b) + sin(a-b)]`
    /// - `cos(a)·cos(b) → ½[cos(a-b) + cos(a+b)]`
    /// - `sin(a)·sin(b) → ½[cos(a-b) - cos(a+b)]`
    /// - `cos²(x) - sin²(x) → cos(2x)`
    ///
    /// This is the inverse of [`expand_trig`](Self::expand_trig).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let expr = &x.sin() * &x.cos();
    /// let combined = expr.trig_combine();
    /// let s = format!("{combined}");
    /// assert!(s.contains("sin"), "should produce product-to-sum: {s}");
    /// ```
    #[must_use = "returns the combined form; does not modify in place"]
    pub fn trig_combine(&self) -> Ex {
        let id = self.inner.write().arena.trig_combine_expr(self.raw_id());
        self.wrap(id)
    }

    /// Dedicated trigonometric simplification.
    ///
    /// Goes beyond the pattern-based rules in [`simplify`](Self::simplify)
    /// by trying exhaustive Pythagorean replacements (sin²→1−cos² and
    /// cos²→1−sin²) and picking the result with the fewest operations.
    /// Simplify an expression using mathematical assumptions.
    ///
    /// Unlike [`simplify`](Expr::simplify) which performs structural
    /// rewriting, `refine` applies rewrites that are only valid under
    /// certain assumptions (e.g., "x is positive").  Assumptions are
    /// set on symbols via [`assume`](Expr::assume).
    ///
    /// # Rewrites applied
    ///
    /// | Expression | Condition | Result |
    /// |------------|-----------|--------|
    /// | `abs(x)` | x ≥ 0 | x |
    /// | `abs(x)` | x < 0 | −x |
    /// | `sign(x)` | x > 0 | 1 |
    /// | `sign(x)` | x < 0 | −1 |
    /// | `sign(x)` | x = 0 | 0 |
    /// | `floor(x)` | x ∈ ℤ | x |
    /// | `ceiling(x)` | x ∈ ℤ | x |
    /// | `sqrt(x²)` | x > 0 | x |
    /// | `sqrt(x²)` | x ∈ ℝ | abs(x) |
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x").assume(Assumption::Positive).unwrap();
    /// let expr = x.abs();
    /// let refined = expr.refine();
    /// assert_eq!(format!("{refined}"), format!("{x}"));
    /// ```
    /// Render this expression as a 2D Unicode string for terminal display.
    ///
    /// Produces multi-line output with stacked fractions, superscripts,
    /// height-matched parentheses, and graduated fraction bar weights
    /// for nested fractions.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let expr = ctx.rational(1, 2);
    /// let s = expr.pretty();
    /// assert!(s.lines().count() == 3, "fraction should be 3 lines");
    /// ```
    #[must_use = "returns the rendered string; does not modify in place"]
    pub fn pretty(&self) -> String {
        let inner = self.inner.read();
        crate::output::pretty::pretty_print(
            &inner.arena,
            self.raw_id(),
            crate::output::pretty::RenderMode::Unicode,
        )
        .render()
    }

    /// Render this expression as a 2D ASCII string for terminal display.
    ///
    /// Like [`pretty`](Self::pretty) but uses only ASCII characters,
    /// avoiding Unicode box-drawing and bracket pieces that may not
    /// render correctly in all terminals.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let expr = ctx.rational(1, 2);
    /// let s = expr.pretty_ascii();
    /// assert!(s.contains('-'), "ASCII fraction uses dashes");
    /// ```
    #[must_use = "returns the rendered string; does not modify in place"]
    pub fn pretty_ascii(&self) -> String {
        let inner = self.inner.read();
        crate::output::pretty::pretty_print(
            &inner.arena,
            self.raw_id(),
            crate::output::pretty::RenderMode::Ascii,
        )
        .render()
    }

    /// Simplify an expression using assumptions.
    ///
    /// Unlike [`simplify`](Expr::simplify) which performs structural
    /// rewriting, `refine` applies rewrites that are only valid under
    /// certain assumptions (e.g., "x is positive").  Assumptions are
    /// set on symbols via [`assume`](Expr::assume).
    ///
    /// See [`refine_with`](Self::refine_with) for temporary assumptions.
    #[must_use = "returns the refined form; does not modify in place"]
    pub fn refine(&self) -> Ex {
        let mut inner = self.inner.write();
        let crate::api::context::ContextInner {
            ref mut arena,
            ref assumptions,
            ..
        } = *inner;
        let mut assumptions_guard = assumptions.lock();
        let id = crate::simplify::refine::refine_full(arena, &mut assumptions_guard, self.raw_id());
        drop(assumptions_guard);
        drop(inner);
        self.wrap(id)
    }

    /// Simplify an expression using temporary assumptions.
    ///
    /// Like [`refine`](Self::refine), but takes a list of
    /// `(variable, assumption)` pairs that are applied temporarily
    /// for this call only — the symbols' stored assumptions are not
    /// modified.
    ///
    /// This is useful when you want to explore "what if x is positive?"
    /// without permanently changing the symbol's properties.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let expr = x.abs();
    /// // x has no permanent assumptions, but refine_with treats it as positive:
    /// let refined = expr.refine_with(&[(&x, Assumption::Positive)]).unwrap();
    /// assert_eq!(format!("{refined}"), format!("{x}"));
    /// // Original x is unchanged — no permanent assumption was set:
    /// assert!(x.is_positive().is_none());
    /// // Contradictory hypotheses are an error, and change nothing:
    /// assert!(matches!(
    ///     expr.refine_with(&[(&x, Assumption::Positive), (&x, Assumption::Negative)]),
    ///     Err(SymplexError::ContradictoryAssumptions { .. })
    /// ));
    /// ```
    ///
    /// # Errors
    ///
    /// [`SymplexError::ContradictoryAssumptions`] if the temporary
    /// assumptions contradict each other or a symbol's stored ones (`x`
    /// positive and negative); the context is left unchanged.
    ///
    /// # Panics
    ///
    /// Panics if a variable comes from another context (the crate-wide
    /// cross-context logic error).
    pub fn refine_with(
        &self,
        temp_assumptions: &[(&Ex, crate::base::assumptions::Assumption)],
    ) -> Result<Ex, SymplexError> {
        use crate::base::assumptions::{AssumptionCache, Assumptions};
        use crate::base::node::{ExprNode, SymbolId};

        // Extract checked IDs before acquiring the lock.
        let var_ids: Vec<crate::base::node::ExprId> = temp_assumptions
            .iter()
            .map(|(var, _)| self.checked_id(var))
            .collect();

        let mut inner = self.inner.write();
        let crate::api::context::ContextInner {
            ref mut arena,
            ref assumptions,
            ..
        } = *inner;

        // The stored assumptions of each symbol involved (once per symbol,
        // so two hypotheses on one symbol restore to the original), and the
        // hypotheses on it.
        let mut saved: Vec<(SymbolId, Assumptions)> = Vec::new();
        let mut hypotheses: Vec<Vec<crate::base::assumptions::Assumption>> = Vec::new();
        for (i, (_var, assumption)) in temp_assumptions.iter().enumerate() {
            let ExprNode::Symbol(sid) = *arena.node(var_ids[i]) else {
                continue;
            };
            let slot = match saved.iter().position(|(s, _)| *s == sid) {
                Some(k) => k,
                None => {
                    saved.push((sid, arena.symbol_assumptions(sid)));
                    hypotheses.push(Vec::new());
                    saved.len() - 1
                }
            };
            hypotheses[slot].push(*assumption);
        }
        // The temporary sets, all validated before any is applied.
        let temporary: Vec<(SymbolId, Assumptions)> = saved
            .iter()
            .zip(&hypotheses)
            .map(|(&(sid, original), hyps)| {
                Assumptions::declare(arena.symbols.name(sid), original, hyps).map(|a| (sid, a))
            })
            .collect::<Result<_, _>>()?;
        // Apply them and refine with a fresh assumption cache; restore the
        // stored sets whatever happens.
        let refined = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            for (sid, a) in temporary {
                arena.set_symbol_assumptions(sid, a);
            }
            let mut temp_cache = AssumptionCache::new();
            crate::simplify::refine::refine_full(arena, &mut temp_cache, self.raw_id())
        }));
        for (sid, original) in saved {
            arena.set_symbol_assumptions(sid, original);
        }

        // Invalidate the main assumption cache since we temporarily mutated symbols.
        {
            let mut main_cache = assumptions.lock();
            *main_cache = AssumptionCache::new();
        }

        drop(inner);
        match refined {
            Ok(id) => Ok(self.wrap(id)),
            Err(panic) => std::panic::resume_unwind(panic),
        }
    }

    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let expr = &x.sin().powi(2) + &x.cos().powi(2);
    /// assert_eq!(format!("{}", expr.simplify_trig()), "1");
    /// ```
    #[must_use = "returns the simplified form; does not modify in place"]
    pub fn simplify_trig(&self) -> Ex {
        let id = self.inner.write().arena.trigsimp_expr(self.raw_id());
        self.wrap(id)
    }

    /// Apply Fu's trig simplification algorithm.
    ///
    /// Tries 26+ named transforms (TR0–TR14, TRmorrie, TRpower, Pythagorean
    /// substitutions) organized into rule lists, and picks the result with
    /// the lowest `(trig_count, op_count)` measure.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let expr = &x.sin().powi(2) + &x.cos().powi(2);
    /// assert_eq!(format!("{}", expr.fu()), "1");
    /// ```
    #[must_use = "returns the simplified form; does not modify in place"]
    pub fn fu(&self) -> Ex {
        let id = {
            let mut guard = self.inner.write();
            crate::simplify::fu::fu(&mut guard.arena, self.raw_id())
        };
        self.wrap(id)
    }

    /// Linearize trigonometric powers using Chebyshev-type expansion.
    ///
    /// Converts `sin(x)^n` and `cos(x)^n` into linear combinations of
    /// `sin(kx)` and `cos(kx)`. For example, `sin(x)^3` becomes
    /// `3/4·sin(x) - 1/4·sin(3x)`.
    ///
    /// This is Fu's TRpower transform.
    #[must_use = "returns the linearized form; does not modify in place"]
    pub fn trig_power_linearize(&self) -> Ex {
        let id = {
            let mut guard = self.inner.write();
            crate::simplify::fu::tr_power(&mut guard.arena, self.raw_id())
        };
        self.wrap(id)
    }

    /// Apply half-angle factoring to trigonometric expressions.
    ///
    /// Converts patterns like `cos(x) - 1` to `-2·sin²(x/2)` and
    /// `cos(x) + 1` to `2·cos²(x/2)`.
    ///
    /// This is Fu's TR14 transform.
    #[must_use = "returns the transformed form; does not modify in place"]
    pub fn trig_half_angle(&self) -> Ex {
        let id = {
            let mut guard = self.inner.write();
            crate::simplify::fu::tr14(&mut guard.arena, self.raw_id())
        };
        self.wrap(id)
    }

    /// Attempt closed-form evaluation of a hypergeometric sum using Gosper's algorithm.
    ///
    /// Given a symbolic sum `Σ_{k=lower}^{upper} f(k)`, tries to find a
    /// hypergeometric antidifference via Gosper's algorithm (1978).
    /// Returns `Some(result)` with the closed-form expression, or `None`
    /// if the sum is not Gosper-summable.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let k = ctx.symbol("k");
    /// let n = ctx.symbol("n");
    /// let body = ctx.int(2).pow(&k);
    /// let s = Ex::symbolic_sum(&body, &k, &ctx.int(0), &n);
    /// // gosper_sum extracts bounds and body from a Sum node
    /// let _result = s.gosper_sum(&k);
    /// ```
    #[must_use]
    pub fn gosper_sum(&self, var: &Ex) -> Ex {
        let var_id = self.checked_id(var);
        let mut inner = self.inner.write();
        let node = inner.arena.node(self.raw_id()).clone();
        if let crate::base::node::ExprNode::Sum(body, _sum_var, lower, upper) = node {
            let result =
                crate::calculus::gosper::gosper_sum(&mut inner.arena, body, var_id, lower, upper);
            drop(inner);
            if let Some(id) = result {
                return self.wrap(id);
            }
        } else {
            drop(inner);
        }
        self.clone()
    }

    /// Like [`gosper_sum`](Self::gosper_sum), but returns `Err` if the result
    /// contains unevaluated forms (i.e. the sum is not Gosper-summable).
    pub fn try_gosper_sum(&self, var: &Ex) -> Result<Ex, SymplexError> {
        let result = self.gosper_sum(var);
        if result.has_unevaluated() {
            Err(SymplexError::ComputationFailed {
                operation: "gosper_sum",
                reason: "not Gosper-summable".into(),
            })
        } else {
            Ok(result)
        }
    }

    /// Simplify combinatorial expressions (factorials, binomials).
    ///
    /// Cancels common factorial terms in products, detects binomial
    /// coefficient patterns, and simplifies ratios like `n! / (n-1)!` → `n`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let result = ctx.int(5).factorial().eval();
    /// let four_fact = ctx.int(4).factorial().eval();
    /// let ratio = (&result / &four_fact).simplify_combinatorial();
    /// assert_eq!(format!("{ratio}"), "5");
    /// ```
    #[must_use = "returns the simplified form; does not modify in place"]
    pub fn simplify_combinatorial(&self) -> Ex {
        let id = self.inner.write().arena.combsimp_expr(self.raw_id());
        self.wrap(id)
    }

    /// Find a simple closed-form for a numerical expression.
    ///
    /// Tries rational approximations (via continued fractions),
    /// π-multiples, and square roots of small integers within the
    /// given `tolerance`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let expr = ctx.rational(333333, 1000000);
    /// let result = expr.simplify_numeric(1e-5);
    /// assert_eq!(format!("{result}"), "1/3");
    /// ```
    #[must_use = "returns the simplified form; does not modify in place"]
    pub fn simplify_numeric(&self, tolerance: f64) -> Ex {
        let id = self
            .inner
            .write()
            .arena
            .nsimplify_expr(self.raw_id(), tolerance);
        self.wrap(id)
    }

    /// Combine like bases in products with symbolic exponents.
    ///
    /// Extends the numeric power-merging done during canonicalization
    /// to symbolic exponents: `x^a * x^b → x^(a+b)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, a, b) = (ctx.symbol("x"), ctx.symbol("a"), ctx.symbol("b"));
    /// let expr = &x.pow(&a) * &x.pow(&b);
    /// let result = expr.simplify_powers();
    /// let s = format!("{result}");
    /// assert!(s.contains("a + b") || s.contains("b + a"), "should combine: {s}");
    /// ```
    #[must_use = "returns the simplified form; does not modify in place"]
    pub fn simplify_powers(&self) -> Ex {
        let id = self.inner.write().arena.powsimp_expr(self.raw_id());
        self.wrap(id)
    }

    /// Rewrite trigonometric functions as complex exponentials.
    ///
    /// - `sin(x) → (exp(ix) − exp(−ix)) / (2i)`
    /// - `cos(x) → (exp(ix) + exp(−ix)) / 2`
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let result = x.sin().rewrite_as_exp();
    /// let s = format!("{result}");
    /// assert!(s.contains("exp") || s.contains("E"), "should contain exponentials: {s}");
    /// ```
    #[must_use = "returns the rewritten form; does not modify in place"]
    pub fn rewrite_as_exp(&self) -> Ex {
        let id = self.inner.write().arena.rewrite_as_exp_expr(self.raw_id());
        self.wrap(id)
    }

    /// Rewrite complex exponentials as trigonometric functions (Euler's formula).
    ///
    /// `exp(ix) → cos(x) + i·sin(x)`
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let i = ctx.i_unit();
    /// let expr = (&i * &x).exp();
    /// let result = expr.rewrite_as_trig();
    /// let s = format!("{result}");
    /// assert!(s.contains("cos") && s.contains("sin"), "should contain trig: {s}");
    /// ```
    #[must_use = "returns the rewritten form; does not modify in place"]
    pub fn rewrite_as_trig(&self) -> Ex {
        let id = self.inner.write().arena.rewrite_as_trig_expr(self.raw_id());
        self.wrap(id)
    }

    /// Group an expression by powers of `var`.
    ///
    /// Converts the expression to a univariate polynomial in `var`
    /// and rebuilds it, naturally grouping coefficients by power.
    ///
    /// When the expression is not polynomial in `var` (symbolic or
    /// negative exponents), the terms of the sum are grouped by the exact
    /// power of `var` they contain instead:
    /// `y·x^a + z·x^a + x^2 → (y + z)·x^a + x^2`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// let expr = &x * &y + &x.powi(2) + &y;
    /// let collected = expr.collect(&x);
    /// // Terms are grouped by powers of x.
    /// assert_eq!(format!("{collected}"), "x^2 + x*y + y");
    ///
    /// // Symbolic exponents:
    /// let (a, z) = (ctx.symbol("a"), ctx.symbol("z"));
    /// let expr = &y * &x.pow(&a) + &z * &x.pow(&a);
    /// assert_eq!(format!("{}", expr.collect(&x)), "x^a*(y + z)");
    /// ```
    #[must_use = "returns the collected form; does not modify in place"]
    pub fn collect(&self, var: &Ex) -> Ex {
        let var_id = self.checked_id(var);
        let id = {
            let mut inner = self.inner.write();
            let poly = inner.arena.collect_expr(self.raw_id(), var_id);
            if poly == self.raw_id() {
                crate::simplify::factor_terms::collect_powers(&mut inner.arena, poly, var_id)
            } else {
                poly
            }
        };
        self.wrap(id)
    }

    /// Combine every fraction in the expression — at any depth — into a
    /// single quotient `numerator / denominator`.
    ///
    /// Sums are put over a common denominator (the polynomial LCM when the
    /// denominators are univariate, otherwise the product of the distinct
    /// denominators), products multiply numerators and denominators, and
    /// integer powers distribute, so fractions nested inside numerators or
    /// denominators are flattened too.  Function arguments are left alone.
    /// **No common factors are cancelled** — that is
    /// [`ratsimp`](Self::ratsimp), which also normalises the result.
    ///
    /// The pieces are available separately from
    /// [`as_numer_denom`](Self::as_numer_denom); note that a purely numeric
    /// common denominator cannot survive canonicalisation as a quotient
    /// (`(3x + 2)/6` is stored as `1/2*x + 1/3`), while `as_numer_denom`
    /// still reports `(3*x + 2, 6)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// assert_eq!((1 / &x + 1 / &y).together().to_string(), "(x + y)/(x*y)");
    /// // Nested: the inner fraction is flattened as well.
    /// let nested = (&x + 1 / &y) / (&x - 1);
    /// let (n, d) = nested.together().as_numer_denom();
    /// assert_eq!((n.to_string(), d.to_string()), ("x*y + 1".to_string(), "y*(x - 1)".to_string()));
    /// ```
    #[must_use = "returns the combined form; does not modify in place"]
    pub fn together(&self) -> Ex {
        let id = self.inner.write().arena.together_expr(self.raw_id());
        self.wrap(id)
    }

    /// Cancel common polynomial factors in a rational expression.
    ///
    /// Decomposes the expression into numerator and denominator,
    /// converts both to univariate polynomials in `var`, divides out
    /// their GCD, and rebuilds the expression.
    ///
    /// Returns the expression unchanged if it is not a rational
    /// function in `var` or if there is no common factor.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// // (x² - 1) / (x - 1) → x + 1
    /// let expr = (&x.powi(2) - 1) / (&x - 1);
    /// let cancelled = expr.cancel(&x);
    /// assert_eq!(format!("{cancelled}"), "x + 1");
    /// ```
    #[must_use = "returns the cancelled form; does not modify in place"]
    pub fn cancel(&self, var: &Ex) -> Ex {
        let var_id = self.checked_id(var);
        let _span = debug_span!("cancel", expr = ?self.raw_id(), var = ?var_id).entered();
        let id = self.inner.write().arena.cancel_expr(self.raw_id(), var_id);
        self.wrap(id)
    }

    /// Rational simplification: combine fractions and cancel.
    ///
    /// Combines every fraction over a common denominator and cancels the
    /// common polynomial factors, in all variables at once.  Since 0.3 this
    /// is the same normal form as [`ratsimp`](Self::ratsimp).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// // 1/x + 1/x → 2/x after simplify_rational
    /// let expr = &x.powi(-1) + &x.powi(-1);
    /// let simplified = expr.simplify_rational();
    /// assert_eq!(simplified, ctx.int(2) / &x);
    /// ```
    #[must_use = "returns the simplified form; does not modify in place"]
    pub fn simplify_rational(&self) -> Ex {
        self.ratsimp()
    }

    /// Rational-function normal form: a single fraction with common factors
    /// cancelled and an integer-primitive numerator and denominator.
    ///
    /// The expression is read as `P/Q` with `P` and `Q` polynomials over
    /// the free symbols and every maximal non-rational subexpression
    /// (`sin(x)`, `π`, `√x`, … are treated as independent indeterminates,
    /// exactly like SymPy's `cancel`).  `gcd(P, Q)` is divided out with a
    /// heuristic multivariate GCD, denominators are cleared so that both
    /// parts have integer coefficients with no common integer factor, and
    /// the leading coefficient of `Q` is made positive.
    ///
    /// Expressions containing `±∞`, `NaN` or unevaluated nodes are
    /// returned unchanged, as is anything that is already in normal form
    /// (the returned handle is then structurally identical to `self`).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// // (x² − y²)/(x − y) → x + y
    /// let e = (&x.powi(2) - &y.powi(2)) / (&x - &y);
    /// assert_eq!(e.ratsimp(), &x + &y);
    /// // 1/x + 1/y → (x + y)/(x y)
    /// let e = ctx.int(1) / &x + ctx.int(1) / &y;
    /// assert_eq!(e.ratsimp(), (&x + &y) / (&x * &y));
    /// // Opaque subexpressions are indeterminates: sin(x)²/sin(x) → sin(x)
    /// assert_eq!((x.sin().powi(2) / x.sin()).ratsimp(), x.sin());
    /// ```
    #[must_use = "returns the normalised form; does not modify in place"]
    pub fn ratsimp(&self) -> Ex {
        let _span = debug_span!("ratsimp", expr = ?self.raw_id()).entered();
        let id = {
            let mut inner = self.inner.write();
            crate::simplify::ratsimp::ratsimp(&mut inner.arena, self.raw_id())
        };
        self.wrap(id)
    }

    /// Partial fraction decomposition with respect to `var`.
    ///
    /// Decomposes a rational expression into a sum of simpler fractions.
    /// Returns the expression unchanged if it's not a rational function
    /// or if the denominator cannot be factored.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let expr = 1 / (&x.powi(2) - 1);
    /// let decomposed = expr.partial_fractions(&x);
    /// let s = format!("{decomposed}");
    /// // Should be decomposed into simpler fractions
    /// assert!(s != format!("{expr}") || s.contains("1/"), "should decompose: {s}");
    /// ```
    #[must_use = "returns the decomposed form; does not modify in place"]
    pub fn partial_fractions(&self, var: &Ex) -> Ex {
        let var_id = self.checked_id(var);
        let id = self.inner.write().arena.apart_expr(self.raw_id(), var_id);
        self.wrap(id)
    }

    /// Factor a polynomial expression into a product of linear factors.
    ///
    /// Finds rational roots via the equation solver, extracts content
    /// (GCD of coefficients), and handles root multiplicities.
    ///
    /// Returns the expression unchanged if it is not polynomial in `var`
    /// or if no rational roots can be found.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let expr = &x.powi(2) - 1;
    /// let factored = expr.factor(&x);
    /// let s = format!("{factored}");
    /// // Should be factored into (x-1)(x+1) form
    /// assert!(!s.contains("x^2"), "should be factored: {s}");
    /// ```
    #[must_use = "returns the factored form; does not modify in place"]
    pub fn factor(&self, var: &Ex) -> Ex {
        let var_id = self.checked_id(var);
        let _span = debug_span!("factor", expr = ?self.raw_id(), var = ?var_id).entered();
        let id = self.inner.write().arena.factor_expr(self.raw_id(), var_id);
        self.wrap(id)
    }

    /// Factor out the GCD of numeric coefficients from a sum.
    ///
    /// Returns `(gcd, inner)` where `self == gcd * inner` mathematically.
    /// The inner expression has each coefficient divided by the GCD.
    ///
    /// Due to canonicalization (Number×Add distribution), reconstructing
    /// `gcd * inner` may produce the original distributed form. Use the
    /// returned pair directly for display or cancellation.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let y = ctx.symbol("y");
    /// let expr = &x * 4 + &y * 6;
    /// let (gcd, inner) = expr.factor_terms();
    /// assert_eq!(format!("{gcd}"), "2");
    /// // inner is 2x + 3y
    /// ```
    #[must_use]
    pub fn factor_terms(&self) -> (Ex, Ex) {
        let (gcd_id, inner_id) = self
            .inner
            .write()
            .arena
            .factor_terms_pair_expr(self.raw_id());
        (self.wrap(gcd_id), self.wrap(inner_id))
    }

    /// Rationalize the denominator of a fraction containing square roots.
    ///
    /// - `1/√2 → √2/2`
    /// - `1/(1 + √2) → √2 - 1`
    ///
    /// Returns the expression unchanged if the denominator contains
    /// no square roots.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let expr = 1 / &ctx.int(2).sqrt();
    /// let rationalized = expr.rationalize_denom();
    /// let s = format!("{rationalized}");
    /// assert!(s.contains("2"), "should rationalize: {s}");
    /// ```
    #[must_use = "returns the rationalized form; does not modify in place"]
    pub fn rationalize_denom(&self) -> Ex {
        let id = self
            .inner
            .write()
            .arena
            .rationalize_denom_expr(self.raw_id());
        self.wrap(id)
    }

    /// Separate variables in a multiplicative expression.
    ///
    /// Given a list of variables, partitions the top-level factors
    /// by which variables they depend on. Returns a vec of
    /// `(dependent_vars, product_of_factors)` pairs.
    ///
    /// - Factors that depend on none of the listed vars get an empty
    ///   dependency list (i.e. they are "constant" w.r.t. the vars).
    /// - Factors that depend on exactly one var are grouped together.
    /// - Factors that depend on multiple vars form a "mixed" group.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// // 2 * x * y  — each factor depends on different vars
    /// let expr = &x * &y * 2;
    /// let groups = expr.separate_vars(&[&x, &y]);
    /// assert!(groups.len() >= 2, "should separate into multiple groups");
    /// ```
    #[must_use]
    pub fn separate_vars(&self, vars: &[&Ex]) -> Vec<(Vec<Ex>, Ex)> {
        let var_ids: Vec<crate::base::node::ExprId> =
            vars.iter().map(|v| self.checked_id(v)).collect();
        let raw = self
            .inner
            .write()
            .arena
            .separatevars_expr(self.raw_id(), &var_ids);
        raw.into_iter()
            .map(|(dep_ids, prod_id)| {
                let dep_exprs: Vec<Ex> = dep_ids.into_iter().map(|id| self.wrap(id)).collect();
                (dep_exprs, self.wrap(prod_id))
            })
            .collect()
    }

    // ── Polynomial introspection ───────────────────────────────────

    /// Return the degree of this expression as a polynomial in `var`.
    ///
    /// Returns `Some(n)` if the expression is a polynomial of degree `n`
    /// in `var`, or `None` if it is not polynomial (e.g., contains `sin(x)`)
    /// or is the zero polynomial.  Coefficients may be exact numbers or
    /// arbitrary expressions free of `var` (symbolic parameters).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, a) = (ctx.symbol("x"), ctx.symbol("a"));
    /// assert_eq!((&x.powi(3) + &x + 1).degree(&x), Some(3));
    /// assert_eq!((&a * &x.powi(2) + &x * (&a + 1) + 3).degree(&x), Some(2));
    /// assert_eq!(x.sin().degree(&x), None);
    /// ```
    #[must_use]
    pub fn degree(&self, var: &Ex) -> Option<usize> {
        let var_id = self.checked_id(var);
        let mut inner = self.inner.write();
        if let Some(d) = inner.arena.degree_of(self.raw_id(), var_id) {
            return Some(d);
        }
        let coeffs = symbolic_coeffs_of(&mut inner.arena, self.raw_id(), var_id)?;
        if coeffs.is_empty() {
            return None;
        }
        Some(coeffs.len() - 1)
    }

    /// Return the coefficients of this expression as a polynomial in `var`,
    /// in ascending degree order: `[a_0, a_1, a_2, ...]`.
    ///
    /// Returns `None` if the expression is not polynomial in `var`.
    /// Coefficients may be exact numbers or arbitrary expressions free of
    /// `var`; the zero polynomial yields an empty vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, a) = (ctx.symbol("x"), ctx.symbol("a"));
    /// // x^2 + 3*x + 5 → coefficients [5, 3, 1]
    /// let expr = &x.powi(2) + &x * 3 + 5;
    /// let cs = expr.coeffs(&x).unwrap();
    /// let strs: Vec<String> = cs.iter().map(|c| format!("{c}")).collect();
    /// assert_eq!(strs, vec!["5", "3", "1"]);
    ///
    /// // Symbolic parameters are collected too: a*x^2 + (a + 1)*x + 3
    /// let expr = &a * &x.powi(2) + &x * (&a + 1) + 3;
    /// let cs = expr.coeffs(&x).unwrap();
    /// let strs: Vec<String> = cs.iter().map(|c| format!("{c}")).collect();
    /// assert_eq!(strs, vec!["3", "a + 1", "a"]);
    /// ```
    #[must_use]
    pub fn coeffs(&self, var: &Ex) -> Option<Vec<Ex>> {
        let var_id = self.checked_id(var);
        let mut inner = self.inner.write();
        let ids = match inner.arena.coefficients_of(self.raw_id(), var_id) {
            Some(ids) => ids,
            None => symbolic_coeffs_of(&mut inner.arena, self.raw_id(), var_id)?,
        };
        drop(inner);
        Some(ids.into_iter().map(|id| self.wrap(id)).collect())
    }

    /// Extract the coefficient of `var^n` in this expression.
    ///
    /// Returns `None` if the expression is not polynomial in `var`.
    /// Returns the zero expression if the coefficient is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let expr = &x.powi(2) * 3 + &x * 5 + 7;
    /// assert_eq!(format!("{}", expr.coeff(&x, 2).unwrap()), "3");
    /// assert_eq!(format!("{}", expr.coeff(&x, 1).unwrap()), "5");
    /// assert_eq!(format!("{}", expr.coeff(&x, 0).unwrap()), "7");
    /// ```
    #[must_use]
    pub fn coeff(&self, var: &Ex, power: usize) -> Option<Ex> {
        let cs = self.coeffs(var)?;
        if power < cs.len() {
            Some(cs[power].clone())
        } else {
            // Coefficient is zero for powers above the degree.
            let inner = self.inner.read();
            let zero = inner.arena.zero;
            drop(inner);
            Some(self.wrap(zero))
        }
    }

    /// Decompose this expression into `(numerator, denominator)` with
    /// `self == numerator / denominator`, the same way SymPy's
    /// `as_numer_denom` does.
    ///
    /// * A rational literal splits into integers: `3/31` → `(3, 31)`.
    /// * A rational coefficient splits too: `2/3 * x` → `(2*x, 3)`.
    /// * A sum is combined over a common denominator at every depth
    ///   (see [`together`](Self::together)): `x/2 + 1/3` → `(3*x + 2, 6)`,
    ///   `1/x + 1/y` → `(x + y, x*y)`.
    /// * Function arguments are opaque; anything without a denominator is
    ///   `(self, 1)`.
    ///
    /// Nothing is cancelled: `((x² − 1)/(x − 1)).as_numer_denom()` is
    /// `(x^2 - 1, x - 1)`.  Call [`ratsimp`](Self::ratsimp) first when a
    /// reduced fraction is wanted.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// let s = |(n, d): (Ex, Ex)| (n.to_string(), d.to_string());
    /// assert_eq!(s((&x / &y).as_numer_denom()), ("x".into(), "y".into()));
    /// assert_eq!(s(ctx.rational(3, 31).as_numer_denom()), ("3".into(), "31".into()));
    /// assert_eq!(s((&x * 2 / 3).as_numer_denom()), ("2*x".into(), "3".into()));
    /// assert_eq!(s((&x / 2 + ctx.rational(1, 3)).as_numer_denom()), ("3*x + 2".into(), "6".into()));
    /// assert_eq!(s((1 / &x + 1 / &y).as_numer_denom()), ("x + y".into(), "x*y".into()));
    /// ```
    #[must_use]
    pub fn as_numer_denom(&self) -> (Ex, Ex) {
        let mut inner = self.inner.write();
        let (n, d) = inner.arena.as_numer_denom_expr(self.raw_id());
        drop(inner);
        (self.wrap(n), self.wrap(d))
    }

    /// Returns `true` if this expression contains no free symbols
    /// (i.e., it is a constant — a number, π, e, etc.).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// assert!(ctx.int(5).is_constant());
    /// assert!(ctx.pi().is_constant());
    /// assert!(!ctx.symbol("x").is_constant());
    /// ```
    #[must_use]
    pub fn is_constant(&self) -> bool {
        self.free_symbols().is_empty()
    }

    /// Returns `true` if this expression is a polynomial in `var`.
    ///
    /// Equivalent to `self.degree(var).is_some()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// assert!((&x.powi(2) + 1).is_polynomial(&x));
    /// assert!(!x.sin().is_polynomial(&x));
    /// ```
    #[must_use]
    pub fn is_polynomial(&self, var: &Ex) -> bool {
        self.degree(var).is_some()
    }

    /// Compute the polynomial GCD of `self` and `other` with respect to `var`.
    ///
    /// Returns `None` if either expression is not polynomial in `var`.
    #[must_use]
    pub fn poly_gcd(&self, other: &Ex, var: &Ex) -> Option<Ex> {
        let other_id = self.checked_id(other);
        let var_id = self.checked_id(var);
        let mut inner = self.inner.write();
        let id = inner.arena.poly_gcd_expr(self.raw_id(), other_id, var_id)?;
        drop(inner);
        Some(self.wrap(id))
    }

    /// Compute the polynomial LCM of `self` and `other` with respect to `var`.
    ///
    /// Returns `None` if either expression is not polynomial in `var`.
    #[must_use]
    pub fn poly_lcm(&self, other: &Ex, var: &Ex) -> Option<Ex> {
        let other_id = self.checked_id(other);
        let var_id = self.checked_id(var);
        let mut inner = self.inner.write();
        let id = inner.arena.poly_lcm_expr(self.raw_id(), other_id, var_id)?;
        drop(inner);
        Some(self.wrap(id))
    }

    // ── Solve ──────────────────────────────────────────────────────

    /// Solve `self = 0` for the given variable.
    ///
    /// Returns the values of `var` that make this expression zero.
    /// Supports polynomial equations (exact radicals through degree 4,
    /// `RootOf` placeholders beyond, all `n` roots of `a·xⁿ + b`),
    /// symbolic-coefficient linear and quadratic equations, and
    /// transcendental equations (`exp`, `ln`, trig, hyperbolic, `|·|`,
    /// change of variable, Lambert W).  For periodic functions only the
    /// principal branches are returned — use
    /// [`solve_general`](Ex::solve_general) for full solution families.
    ///
    /// Polynomials are first factored exactly over ℤ, so a constant
    /// multiple has the same roots, each distinct root is returned once
    /// whatever its multiplicity, and `RootOf` placeholders only ever refer
    /// to an irreducible factor of degree ≥ 5.
    ///
    /// # Errors
    ///
    /// - [`SymplexError::InfiniteSolutions`] when the equation reduces to
    ///   the identity `0 = 0` (every value of `var` is a solution).
    /// - [`SymplexError::NoSolution`] when the equation is provably
    ///   unsatisfiable: it reduces to a nonzero constant (`1 = 0`), does
    ///   not depend on `var` at all, or violates a range restriction such
    ///   as `exp(x) = 0` or `sin(x) = 2` (no real solution).
    /// - [`SymplexError::ComputationFailed`] when the expression is not
    ///   polynomial in `var` and no transcendental strategy applies.
    ///
    /// `Ok(vec![])` is reserved for genuine equations whose roots could
    /// not be found in the searched domain.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// // Solve x² - 5x + 6 = 0
    /// let expr = &x.powi(2) - &x * 5 + 6;
    /// let solutions = expr.solve(&x).unwrap();
    /// assert_eq!(solutions.len(), 2);
    ///
    /// // 0 = 0 is an identity, 1 = 0 a contradiction
    /// assert!(matches!(
    ///     ctx.int(0).solve(&x),
    ///     Err(SymplexError::InfiniteSolutions { .. })
    /// ));
    /// assert!(matches!(
    ///     ctx.int(1).solve(&x),
    ///     Err(SymplexError::NoSolution { .. })
    /// ));
    /// ```
    pub fn solve(&self, var: &Ex) -> Result<Vec<Ex>, SymplexError> {
        let var_id = self.checked_id(var);
        let _span = debug_span!("solve", expr = ?self.raw_id(), var = ?var_id).entered();
        let mut inner = self.inner.write();
        let outcome =
            crate::transforms::solve::solve_classified(&mut inner.arena, self.raw_id(), var_id);
        match outcome {
            crate::transforms::solve::SolveOutcome::Solutions(solutions) => {
                if !solutions.is_empty() {
                    drop(inner);
                    return Ok(solutions
                        .into_iter()
                        .map(|sol| self.wrap(sol.value))
                        .collect());
                }
                // No solutions found — check if the expression is polynomial.
                // For polynomial expressions, an empty result is valid (no roots).
                // For non-polynomial expressions, report an error.
                let poly =
                    crate::poly::polybridge::expr_to_poly(&inner.arena, self.raw_id(), var_id);
                drop(inner);
                if poly.is_none() {
                    return Err(SymplexError::ComputationFailed {
                        operation: "solve",
                        reason: "expression is not polynomial in the given variable and transcendental solver could not find solutions".into(),
                    });
                }
                Ok(vec![])
            }
            crate::transforms::solve::SolveOutcome::Identity => {
                drop(inner);
                Err(SymplexError::InfiniteSolutions {
                    operation: "solve",
                    reason: format!(
                        "equation is an identity (0 = 0): every value of {var} is a solution"
                    ),
                })
            }
            crate::transforms::solve::SolveOutcome::NoSolution(reason) => {
                drop(inner);
                Err(SymplexError::NoSolution {
                    operation: "solve",
                    reason,
                })
            }
        }
    }

    /// Solve `self = 0` for `var`, returning an empty vector on failure.
    ///
    /// This is a convenience wrapper around [`solve`](Ex::solve) that
    /// returns `vec![]` whenever `solve` returns an error — including the
    /// identity (`0 = 0`) and contradiction (`1 = 0`) cases, which have no
    /// finite list of roots.  Use [`solve`](Ex::solve) for diagnostic
    /// information.
    pub fn solve_or_empty(&self, var: &Ex) -> Vec<Ex> {
        self.solve(var).unwrap_or_default()
    }

    /// Solve `self > 0` for `var`, returning the solution as a set.
    ///
    /// Uses the sign-chart method: finds roots, tests sign in each
    /// region, and returns a union of intervals.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// // x > 0 → (0, ∞)
    /// let result = x.solve_gt(&x);
    /// let s = format!("{result}");
    /// assert!(!s.contains("EmptySet"), "x > 0 should not be empty: {s}");
    /// ```
    pub fn solve_gt(&self, var: &Ex) -> SetEx {
        let var_id = self.checked_id(var);
        let mut inner = self.inner.write();
        match inner.arena.solve_inequality_expr(
            self.raw_id(),
            var_id,
            crate::transforms::inequalities::Relation::Gt,
        ) {
            Ok(id) => {
                drop(inner);
                self.wrap_as::<SetValued>(id)
            }
            Err(_) => {
                let zero = inner.arena.zero;
                let cond = inner.arena.gt(self.raw_id(), zero);
                let id = inner
                    .arena
                    .intern(crate::base::node::ExprNode::ConditionSet(var_id, cond));
                drop(inner);
                self.wrap_as::<SetValued>(id)
            }
        }
    }

    /// Like [`solve_gt`](Ex::solve_gt), but returns `Err` if the result
    /// contains unevaluated forms.
    pub fn try_solve_gt(&self, var: &Ex) -> Result<SetEx, SymplexError> {
        let result = self.solve_gt(var);
        if result.has_unevaluated() {
            Err(SymplexError::ComputationFailed {
                operation: "solve_gt",
                reason: "could not solve inequality".into(),
            })
        } else {
            Ok(result)
        }
    }

    /// Solve `self >= 0` for `var`, returning the solution as a set.
    ///
    /// Like [`solve_gt`](Ex::solve_gt), but includes the roots themselves
    /// (where `self = 0`).
    pub fn solve_ge(&self, var: &Ex) -> SetEx {
        let var_id = self.checked_id(var);
        let mut inner = self.inner.write();
        match inner.arena.solve_inequality_expr(
            self.raw_id(),
            var_id,
            crate::transforms::inequalities::Relation::Ge,
        ) {
            Ok(id) => {
                drop(inner);
                self.wrap_as::<SetValued>(id)
            }
            Err(_) => {
                let zero = inner.arena.zero;
                let cond = inner.arena.ge(self.raw_id(), zero);
                let id = inner
                    .arena
                    .intern(crate::base::node::ExprNode::ConditionSet(var_id, cond));
                drop(inner);
                self.wrap_as::<SetValued>(id)
            }
        }
    }

    /// Like [`solve_ge`](Ex::solve_ge), but returns `Err` if the result
    /// contains unevaluated forms.
    pub fn try_solve_ge(&self, var: &Ex) -> Result<SetEx, SymplexError> {
        let result = self.solve_ge(var);
        if result.has_unevaluated() {
            Err(SymplexError::ComputationFailed {
                operation: "solve_ge",
                reason: "could not solve inequality".into(),
            })
        } else {
            Ok(result)
        }
    }

    /// Solve `self < 0` for `var`, returning the solution as a set.
    ///
    /// Uses the sign-chart method with a strict less-than relation.
    pub fn solve_lt(&self, var: &Ex) -> SetEx {
        let var_id = self.checked_id(var);
        let mut inner = self.inner.write();
        match inner.arena.solve_inequality_expr(
            self.raw_id(),
            var_id,
            crate::transforms::inequalities::Relation::Lt,
        ) {
            Ok(id) => {
                drop(inner);
                self.wrap_as::<SetValued>(id)
            }
            Err(_) => {
                let zero = inner.arena.zero;
                let cond = inner.arena.gt(zero, self.raw_id());
                let id = inner
                    .arena
                    .intern(crate::base::node::ExprNode::ConditionSet(var_id, cond));
                drop(inner);
                self.wrap_as::<SetValued>(id)
            }
        }
    }

    /// Like [`solve_lt`](Ex::solve_lt), but returns `Err` if the result
    /// contains unevaluated forms.
    pub fn try_solve_lt(&self, var: &Ex) -> Result<SetEx, SymplexError> {
        let result = self.solve_lt(var);
        if result.has_unevaluated() {
            Err(SymplexError::ComputationFailed {
                operation: "solve_lt",
                reason: "could not solve inequality".into(),
            })
        } else {
            Ok(result)
        }
    }

    /// Solve `self <= 0` for `var`, returning the solution as a set.
    ///
    /// Like [`solve_lt`](Ex::solve_lt), but includes the roots themselves.
    pub fn solve_le(&self, var: &Ex) -> SetEx {
        let var_id = self.checked_id(var);
        let mut inner = self.inner.write();
        match inner.arena.solve_inequality_expr(
            self.raw_id(),
            var_id,
            crate::transforms::inequalities::Relation::Le,
        ) {
            Ok(id) => {
                drop(inner);
                self.wrap_as::<SetValued>(id)
            }
            Err(_) => {
                let zero = inner.arena.zero;
                let cond = inner.arena.ge(zero, self.raw_id());
                let id = inner
                    .arena
                    .intern(crate::base::node::ExprNode::ConditionSet(var_id, cond));
                drop(inner);
                self.wrap_as::<SetValued>(id)
            }
        }
    }

    /// Like [`solve_le`](Ex::solve_le), but returns `Err` if the result
    /// contains unevaluated forms.
    pub fn try_solve_le(&self, var: &Ex) -> Result<SetEx, SymplexError> {
        let result = self.solve_le(var);
        if result.has_unevaluated() {
            Err(SymplexError::ComputationFailed {
                operation: "solve_le",
                reason: "could not solve inequality".into(),
            })
        } else {
            Ok(result)
        }
    }

    /// Solve `self = 0`, returning solutions as a set.
    ///
    /// This is a set-valued variant of [`solve`](Ex::solve) — instead of
    /// returning a `Vec<Ex>`, it returns a `SetEx`:
    ///
    /// - a `FiniteSet` of the roots when they can be found,
    /// - `UniversalSet` when the equation is the identity `0 = 0`,
    /// - `EmptySet` when the equation is provably unsatisfiable or no
    ///   roots were found.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let poly = &x.powi(2) - &x * 5 + 6;
    /// let result = poly.solve_as_set(&x);
    /// let s = format!("{result}");
    /// // Should contain {2, 3} or similar
    /// assert!(!s.contains("EmptySet"), "solve_as_set: {s}");
    /// assert_eq!(format!("{}", ctx.int(0).solve_as_set(&x)), "UniversalSet");
    /// assert_eq!(format!("{}", ctx.int(1).solve_as_set(&x)), "EmptySet");
    /// ```
    pub fn solve_as_set(&self, var: &Ex) -> SetEx {
        let var_id = self.checked_id(var);
        let id = self
            .inner
            .write()
            .arena
            .solveset_expr(self.raw_id(), var_id);
        self.wrap_as::<SetValued>(id)
    }

    /// Numerical root finding via a safeguarded Newton's method.
    ///
    /// Finds a numerical root of `self = 0` near `initial_guess`.  The
    /// core iteration is Newton's `x_{n+1} = x_n - f(x_n)/f'(x_n)`, made
    /// robust by:
    ///
    /// - **backtracking** — a step that increases `|f|` is halved (up to
    ///   30 times) before being accepted;
    /// - **secant fallback** — when `f'(x)` vanishes, the previous iterate
    ///   provides a secant step (or a small perturbation on the first step);
    /// - **bisection fallback** — once two iterates with opposite signs of
    ///   `f` have been seen, any Newton step that leaves the bracket is
    ///   replaced by the bracket midpoint, guaranteeing progress.
    ///
    /// The expression is compiled to a native closure when possible, so
    /// each iteration is cheap.
    ///
    /// # Arguments
    /// - `var` — the variable to solve for
    /// - `initial_guess` — starting point for iteration
    /// - `max_iterations` — maximum number of Newton steps
    /// - `tolerance` — convergence threshold (stop when `|f(x)| < tolerance`)
    ///
    /// # Errors
    /// Returns `Err` if the method doesn't converge within `max_iterations`
    /// (the message reports the final residual), or if evaluation fails
    /// (e.g. free symbols other than `var`).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// // Solve x - cos(x) = 0 near x=1
    /// let expr = &x - &x.cos();
    /// let root = expr.solve_numeric(&x, 1.0, 50, 1e-12).unwrap();
    /// assert!((root - 0.7390851332).abs() < 1e-8);
    ///
    /// // atan(x) = 0 from x = 3: undamped Newton diverges, the damped
    /// // iteration converges to the root at 0.
    /// let r = x.atan().solve_numeric(&x, 3.0, 100, 1e-12).unwrap();
    /// assert!(r.abs() < 1e-8);
    /// ```
    pub fn solve_numeric(
        &self,
        var: &Ex,
        initial_guess: f64,
        max_iterations: usize,
        tolerance: f64,
    ) -> Result<f64, SymplexError> {
        let _ = self.checked_id(var);
        let deriv = self.diff(var);
        let name = format!("{var}");
        let compiled_f = self.compile(&[name.as_str()]);
        let compiled_fp = deriv.compile(&[name.as_str()]);

        let to_ex = |x: f64| -> Result<Ex, SymplexError> {
            let r = num_rational::Ratio::<num_bigint::BigInt>::from_float(x).ok_or_else(|| {
                SymplexError::ComputationFailed {
                    operation: "solve_numeric",
                    reason: format!("could not approximate x = {x} as rational"),
                }
            })?;
            let mut inner = self.inner.write();
            let nid = inner.arena.intern_num(r);
            let id = inner.arena.intern(crate::base::node::ExprNode::Num(nid));
            drop(inner);
            Ok(self.wrap(id))
        };
        let eval_f = |x: f64| -> Result<f64, SymplexError> {
            match &compiled_f {
                Ok(f) => Ok(f(&[x])),
                Err(_) => self.subs(var, &to_ex(x)?).eval_f64(),
            }
        };
        let eval_fp = |x: f64| -> Result<f64, SymplexError> {
            match &compiled_fp {
                Ok(f) => Ok(f(&[x])),
                Err(_) => deriv.subs(var, &to_ex(x)?).eval_f64(),
            }
        };

        let mut x = initial_guess;
        let mut fx = eval_f(x)?;
        // Bracket [lo, hi] with f(lo) and f(hi) of opposite sign, once known.
        let mut bracket: Option<(f64, f64, f64, f64)> = None; // (lo, f_lo, hi, f_hi)
        let mut prev: Option<(f64, f64)> = None; // previous iterate for secant

        let update_bracket = |bracket: &mut Option<(f64, f64, f64, f64)>, xn: f64, fn_: f64| {
            if let Some((lo, f_lo, hi, f_hi)) = *bracket
                && xn > lo
                && xn < hi
            {
                if (fn_ < 0.0) == (f_lo < 0.0) {
                    *bracket = Some((xn, fn_, hi, f_hi));
                } else {
                    *bracket = Some((lo, f_lo, xn, fn_));
                }
            }
        };

        for _ in 0..max_iterations {
            if fx.abs() < tolerance {
                return Ok(x);
            }
            if !fx.is_finite() {
                break;
            }
            // Establish a bracket from the previous iterate if signs differ.
            if bracket.is_none()
                && let Some((px, pf)) = prev
                && pf.is_finite()
                && (pf < 0.0) != (fx < 0.0)
            {
                bracket = if px < x {
                    Some((px, pf, x, fx))
                } else {
                    Some((x, fx, px, pf))
                };
            }

            let fp = eval_fp(x)?;
            let mut dx = if fp.abs() > 1e-300 && fp.is_finite() {
                -fx / fp
            } else if let Some((px, pf)) = prev
                && (fx - pf).abs() > 1e-300
            {
                // Secant step.
                -fx * (x - px) / (fx - pf)
            } else {
                // Perturb away from the stationary point.
                let h = if x.abs() > 1.0 { 1e-3 * x.abs() } else { 1e-3 };
                if fx > 0.0 { -h } else { h }
            };
            if !dx.is_finite() {
                break;
            }

            // Bisection safeguard: stay inside a known bracket.
            let mut x_new = x + dx;
            if let Some((lo, _, hi, _)) = bracket
                && (x_new <= lo || x_new >= hi)
            {
                x_new = 0.5 * (lo + hi);
                dx = x_new - x;
            }

            // Backtracking: halve the step while |f| does not decrease.
            let mut f_new = eval_f(x_new)?;
            let mut tries = 0;
            while (!f_new.is_finite() || f_new.abs() > fx.abs()) && tries < 30 {
                dx *= 0.5;
                x_new = x + dx;
                f_new = eval_f(x_new)?;
                tries += 1;
            }
            if tries == 30
                && let Some((lo, _, hi, _)) = bracket
            {
                // Newton is stuck; bisect instead.
                x_new = 0.5 * (lo + hi);
                f_new = eval_f(x_new)?;
            }

            prev = Some((x, fx));
            x = x_new;
            fx = f_new;
            update_bracket(&mut bracket, x, fx);
        }

        if fx.abs() < tolerance {
            return Ok(x);
        }
        Err(SymplexError::ComputationFailed {
            operation: "solve_numeric",
            reason: format!(
                "did not converge within {} iterations (last x = {x}, residual |f(x)| = {:.3e})",
                max_iterations,
                fx.abs()
            ),
        })
    }

    /// Solve an ODE represented as `self = 0`.
    ///
    /// `self` should contain formal derivative nodes (created via
    /// [`formal_diff`](Self::formal_diff)). `func` is the dependent
    /// variable (e.g., `y`) and `var` is the independent variable (e.g., `x`).
    ///
    /// Returns the general solution expression. If the ODE cannot be solved,
    /// returns an unevaluated `DSolve(expr, func, var)` node.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let y = ctx.symbol("y");
    /// let dy = y.formal_diff(&x);  // y'
    /// let ode = &dy + &(&y * 2);   // y' + 2y = 0
    /// let sol = ode.solve_ode(&y, &x);
    /// let s = format!("{sol}");
    /// assert!(s.contains("exp"), "solution should contain exp: {s}");
    /// ```
    pub fn solve_ode(&self, func: &Ex, var: &Ex) -> Ex {
        let func_id = self.checked_id(func);
        let var_id = self.checked_id(var);
        let mut inner = self.inner.write();
        match crate::calculus::ode::dsolve(&mut inner.arena, self.raw_id(), func_id, var_id) {
            Some(ode_result) => {
                let sol_id = ode_result.solution;
                drop(inner);
                self.wrap(sol_id)
            }
            None => {
                let id = inner.arena.intern(crate::base::node::ExprNode::DSolve(
                    self.raw_id(),
                    func_id,
                    var_id,
                ));
                drop(inner);
                self.wrap(id)
            }
        }
    }

    /// Like [`solve_ode`](Ex::solve_ode), but returns `Err` if the result
    /// contains unevaluated forms (i.e., the ODE could not be solved).
    pub fn try_solve_ode(&self, func: &Ex, var: &Ex) -> Result<Ex, SymplexError> {
        let result = self.solve_ode(func, var);
        if result.has_unevaluated() {
            Err(SymplexError::ComputationFailed {
                operation: "solve_ode",
                reason: "could not solve ODE".into(),
            })
        } else {
            Ok(result)
        }
    }

    // ── Numeric evaluation ─────────────────────────────────────────

    /// Numeric floating-point evaluation to the given number of decimal
    /// digits.
    ///
    /// Returns the decimal string representation of the evaluated expression.
    /// Every digit shown is right up to the rounding of the last one: the
    /// evaluator keeps an error bound for each sub-expression and
    /// re-evaluates at a higher working precision when cancellation or
    /// amplification has eaten into the requested digits (`exp(10⁻³⁰) − 1`
    /// gives `1e-30`, not `0`).  A value that is zero to the precision
    /// reached is `0`.
    ///
    /// ```
    /// use symplex::prelude::*;
    /// let ctx = Context::new();
    /// let tiny = ctx.parse("exp(1/10^30) - 1").unwrap();
    /// assert_eq!(tiny.eval_decimal(20).unwrap(), "1e-30");
    /// // A quotient by a difference that cancels to 0 has no digits to give.
    /// let q = ctx.parse("1/(sin(1)^2 + cos(1)^2 - 1)").unwrap();
    /// assert!(matches!(q.eval_decimal(20), Err(SymplexError::PrecisionExhausted { .. })));
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::FreeSymbol`] if the expression contains
    /// unbound symbols (e.g., `x.eval_decimal(10)` without substituting a value).
    ///
    /// Returns [`SymplexError::Unevaluable`] if the expression contains
    /// nodes that cannot be evaluated to a finite number (infinity, NaN,
    /// imaginary unit, unevaluated derivatives/integrals, user functions).
    ///
    /// Returns [`SymplexError::PrecisionExhausted`] if the requested
    /// precision exceeds `EvalConfig::max_evalf_precision`, if intermediate
    /// computation produces NaN, or if the requested digits cannot be
    /// certified within twice the initial working precision plus 256 bits
    /// (a division by a quantity that cancels to 0, `sign` of such a
    /// quantity).
    #[must_use = "returns the numerical value as a string"]
    pub fn eval_decimal(&self, digits: u32) -> Result<String, SymplexError> {
        let _span = debug_span!("eval_decimal", expr = ?self.raw_id(), digits = digits).entered();
        // Reduce exact values before numerical evaluation.
        // This ensures e.g. Gamma(5) → 24 (exact) rather than
        // computing 23.9999... via Stirling series.
        let evaled = self.eval();
        let guard = evaled.inner.read();
        crate::transforms::evalf::evalf(&guard.arena, evaled.raw_id(), digits)
    }

    /// Convenience: evaluate to an `f64`.
    ///
    /// Evaluates at the precision of [`eval_decimal`](Ex::eval_decimal)
    /// with 16 digits (128 working bits) and rounds the result once, to
    /// the nearest `f64`. This avoids the common pattern of
    /// `.eval_decimal(15).unwrap().parse::<f64>().unwrap()`.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`eval_decimal`](Ex::eval_decimal), plus a
    /// [`SymplexError::ComputationFailed`] if the value has an imaginary
    /// part above `1e-15` in magnitude.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let val = x.powi(2).subs_i64(&x, 3).eval_f64().unwrap();
    /// assert!((val - 9.0).abs() < 1e-10);
    /// ```
    pub fn eval_f64(&self) -> Result<f64, SymplexError> {
        let _span = debug_span!("eval_f64", expr = ?self.raw_id()).entered();
        // eval() first to reduce exact values (sin(0)→0, Gamma(5)→24, etc.)
        // before numerical computation.
        let evaled = self.eval();
        let guard = evaled.inner.read();
        crate::transforms::evalf::evalf_f64(&guard.arena, evaled.raw_id())
    }

    /// Evaluates the expression to a [`Complex64`] (`re + im·i`).
    ///
    /// Uses 16 decimal digits of precision internally. Returns both the
    /// real and imaginary parts, correctly handling complex expressions
    /// like `sqrt(-1)` → `Complex64::new(0.0, 1.0)`.  Real results have
    /// `im == 0.0`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the expression contains free symbols or if
    /// the arbitrary-precision engine fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let z = (&ctx.int(3) + &(&ctx.int(4) * &ctx.i_unit())).eval_complex64().unwrap();
    /// assert!((z - Complex64::new(3.0, 4.0)).norm() < 1e-12);
    /// assert!((z.norm() - 5.0).abs() < 1e-12);
    /// ```
    pub fn eval_complex64(&self) -> Result<Complex64, SymplexError> {
        let _span = debug_span!("eval_complex64", expr = ?self.raw_id()).entered();
        let evaled = self.eval();
        let guard = evaled.inner.read();
        crate::transforms::evalf::evalf_complex64(&guard.arena, evaled.raw_id())
    }

    // ── Code generation ────────────────────────────────────────────

    /// Compile this expression into a fast numerical function.
    ///
    /// `var_names` specifies the variable-to-index mapping: the returned
    /// function takes `&[f64]` where index 0 corresponds to `var_names[0]`,
    /// etc.  The result is a `CompiledFn`:
    /// `Clone + Send + Sync`, callable like a closure (`f(&[x])`) or via
    /// `f.call(&[x])` / `f.try_call(&[x])`, with `f.arity()` reporting the
    /// expected argument count.
    ///
    /// The expression is constant-folded (`eval()`) and common
    /// subexpressions are shared before lowering to a stack-VM program.
    /// Every numerically evaluable node is supported, including the
    /// special functions (`gamma`, `lgamma`, `digamma`, `erf`, `erfc`,
    /// `lambertw`, `beta`, `factorial`, `binomial`), Bessel functions and
    /// orthogonal polynomials with constant integer order, `fibonacci`,
    /// `lucas`, `harmonic`, `factorial2`, rising/falling factorials,
    /// `min`/`max`/`floor`/`ceiling`/`sign`/`heaviside`/`atan2`, and
    /// `piecewise` with relational and boolean conditions.  `DiracDelta`
    /// evaluates to `0.0` everywhere (its pointwise value away from the
    /// support); `Heaviside(0)` is `0.5`.
    ///
    /// # Errors
    ///
    /// * [`SymplexError::FreeSymbol`] if a symbol is not listed in `var_names`.
    /// * [`SymplexError::NotImplemented`] for nodes with no numerical meaning
    ///   (`ImaginaryUnit`, unevaluated `Integral`/`Derivative`/`Sum`, sets,
    ///   user-defined `Apply` nodes, Bessel/orthogonal-polynomial nodes whose
    ///   order is not a constant integer).  The message names the node.
    /// * [`SymplexError::InvalidArgument`] for duplicate parameter names.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let f = &x.powi(2) + 1;
    /// let func = f.compile(&["x"]).expect("should compile");
    /// assert!((func(&[3.0]) - 10.0).abs() < 1e-10);
    /// assert_eq!(func.arity(), 1);
    ///
    /// // Special functions are supported too.
    /// let g = x.gamma().compile(&["x"]).unwrap();
    /// assert!((g.call(&[5.0]) - 24.0).abs() < 1e-12);
    ///
    /// // Free symbols are an error, not a silent NaN.
    /// let y = ctx.symbol("y");
    /// assert!(matches!((&x + &y).compile(&["x"]), Err(SymplexError::FreeSymbol { .. })));
    /// ```
    pub fn compile(
        &self,
        var_names: &[&str],
    ) -> Result<crate::output::lambdify::CompiledFn, SymplexError> {
        let mut inner = self.inner.write();
        crate::output::lambdify::compile(&mut inner.arena, self.raw_id(), var_names)
    }

    /// Compile several expressions into one vector-valued numerical function.
    ///
    /// All expressions must belong to the same context.  A single
    /// common-subexpression-elimination pass is shared across all outputs,
    /// so this is the efficient way to evaluate gradients, Jacobians, or any
    /// family of expressions with overlapping structure.  The returned
    /// `CompiledFnVec` offers
    /// `call(&args, &mut out)`, `call_vec(&args)`, `try_call`, `arity()` and
    /// `len()`.
    ///
    /// An empty `exprs` slice yields a function with zero outputs.
    ///
    /// # Errors
    ///
    /// Same conditions as [`compile`](Self::compile).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let y = ctx.symbol("y");
    /// let f = &x.powi(2) * &y + &x.sin();
    /// let grad = Ex::compile_many(&[&f.diff(&x), &f.diff(&y)], &["x", "y"]).unwrap();
    /// let g = grad.call_vec(&[1.0, 2.0]);
    /// assert!((g[0] - (4.0 + 1f64.cos())).abs() < 1e-12); // 2xy + cos x
    /// assert!((g[1] - 1.0).abs() < 1e-12);                 // x^2
    /// ```
    pub fn compile_many(
        exprs: &[&Ex],
        var_names: &[&str],
    ) -> Result<crate::output::lambdify::CompiledFnVec, SymplexError> {
        let Some(first) = exprs.first() else {
            return crate::output::lambdify::compile_many_empty(var_names);
        };
        let ids: Vec<crate::base::node::ExprId> =
            exprs.iter().map(|e| first.checked_id(e)).collect();
        let mut inner = first.inner.write();
        crate::output::lambdify::compile_many(&mut inner.arena, &ids, var_names)
    }

    /// Perform common subexpression elimination (CSE).
    ///
    /// Identifies repeated subexpressions and extracts them into named
    /// temporaries (`__cse_0`, `__cse_1`, …), reducing redundant
    /// computation when generating code.
    ///
    /// Returns a list of `(name, value)` bindings and the rewritten
    /// expression where common subexpressions are replaced by their names.
    /// Bindings are ordered by first occurrence (post-order), so a binding
    /// only refers to earlier bindings and the numbering is deterministic.
    /// Trivially cheap nodes (a negated or scaled atom, `x^2`, `x^-1`) are
    /// only extracted when used three or more times; boolean-valued nodes
    /// are never extracted.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let sin_x = x.sin();
    /// let expr = &sin_x.powi(2) + &sin_x;
    /// let (bindings, result) = expr.cse();
    /// // sin(x) may be extracted as a common subexpression
    /// let _ = format!("{result}");
    /// ```
    pub fn cse(&self) -> (Vec<(Ex, Ex)>, Ex) {
        let result = {
            let mut guard = self.inner.write();
            crate::output::cse::cse(&mut guard.arena, self.raw_id())
        };
        let bindings = result
            .bindings
            .into_iter()
            .map(|(name, val)| (self.wrap(name), self.wrap(val)))
            .collect();
        (bindings, self.wrap(result.expr))
    }

    /// Common subexpression elimination across several expressions.
    ///
    /// Temporaries are shared by all inputs, which is what code generators
    /// and [`compile_many`](Self::compile_many) need for gradients and
    /// Jacobians.  Returns the shared `(name, value)` bindings and the
    /// rewritten expressions (in input order).  All expressions must belong
    /// to the same context; an empty input yields empty outputs.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let s = x.sin();
    /// let (bindings, exprs) = Ex::cse_many(&[&(&s + 1), &s.powi(3)]);
    /// assert_eq!(exprs.len(), 2);
    /// assert_eq!(bindings.len(), 1); // sin(x) shared by both
    /// assert_eq!(format!("{}", bindings[0].1), "sin(x)");
    /// ```
    pub fn cse_many(exprs: &[&Ex]) -> (Vec<(Ex, Ex)>, Vec<Ex>) {
        let Some(first) = exprs.first() else {
            return (Vec::new(), Vec::new());
        };
        let ids: Vec<crate::base::node::ExprId> =
            exprs.iter().map(|e| first.checked_id(e)).collect();
        let result = {
            let mut guard = first.inner.write();
            crate::output::cse::cse_multi(&mut guard.arena, &ids)
        };
        let bindings = result
            .bindings
            .into_iter()
            .map(|(name, val)| (first.wrap(name), first.wrap(val)))
            .collect();
        let exprs = result.exprs.into_iter().map(|id| first.wrap(id)).collect();
        (bindings, exprs)
    }

    /// Generate a Rust function body as a string.
    ///
    /// The generated function takes `f64` arguments and returns `f64`.
    /// Uses CSE (common subexpression elimination) for efficient code.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let f = x.powi(2) + 1;
    /// let code = f.to_rust_fn("my_func", &["x"]).unwrap();
    /// assert!(code.contains("pub fn my_func"));
    /// ```
    pub fn to_rust_fn(&self, name: &str, args: &[&str]) -> Result<String, SymplexError> {
        let mut guard = self.inner.write();
        guard.arena.to_rust_fn(self.raw_id(), name, args)
    }

    /// Generate a Rust function body as a string with custom code generation options.
    ///
    /// See [`CodegenOptions`](crate::output::codegen::CodegenOptions) for available
    /// settings (precision, math backend, annotations, CSE toggle).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use symplex::prelude::*;
    /// use symplex::codegen::{CodegenOptions, Precision};
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let f = x.powi(2) + 1;
    /// let opts = CodegenOptions { precision: Precision::F32, ..Default::default() };
    /// let code = f.to_rust_fn_with_options("my_func", &["x"], &opts).unwrap();
    /// assert!(code.contains("f32"));
    /// ```
    pub fn to_rust_fn_with_options(
        &self,
        name: &str,
        args: &[&str],
        options: &crate::output::codegen::CodegenOptions,
    ) -> Result<String, SymplexError> {
        let mut guard = self.inner.write();
        crate::output::codegen::to_rust_fn_with_options(
            &mut guard.arena,
            self.raw_id(),
            name,
            args,
            options,
        )
    }

    /// Generate a self-contained C99 function as a string.
    ///
    /// The output starts with `#include <math.h>`, followed by any
    /// `static inline symplex_*` helper functions the expression needs
    /// (Lambert W, digamma, Bessel functions, orthogonal polynomials,
    /// integer sequences, … — everything `<math.h>` lacks), then the
    /// function itself with `const double tN = …;` temporaries for common
    /// subexpressions.  Functions available in `<math.h>` (`tgamma`,
    /// `lgamma`, `erf`, `erfc`, `fma`, `expm1`, `log1p`, …) are used
    /// directly; integer powers `|n| ≤ 4` of simple operands become repeated
    /// multiplication, other powers use `pow`.  Piecewise expressions become
    /// ternary chains ending in `NAN`.
    ///
    /// # Errors
    ///
    /// Same conditions as [`to_rust_fn`](Self::to_rust_fn):
    /// [`SymplexError::FreeSymbol`] for unbound symbols and
    /// [`SymplexError::NotImplemented`] for nodes without numerical meaning
    /// (the message names the node).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let code = (x.sin().powi(2) + x.lambertw()).to_c_fn("f", &["x"]).unwrap();
    /// assert!(code.contains("#include <math.h>"));
    /// assert!(code.contains("double f(double x) {"));
    /// assert!(code.contains("static inline double symplex_lambert_w0(double x)"));
    /// ```
    pub fn to_c_fn(&self, name: &str, args: &[&str]) -> Result<String, SymplexError> {
        let mut guard = self.inner.write();
        crate::output::codegen::codegen_c::to_c_fn(&mut guard.arena, self.raw_id(), name, args)
    }

    /// Generate a C99 function with custom options.
    ///
    /// Honoured [`CodegenOptions`](crate::output::codegen::CodegenOptions)
    /// fields: `precision` (`double` / `float` with the `f`-suffixed math
    /// functions), `cse`, `inline` (`static inline`), `use_mul_add` (`fma`),
    /// `checked_domain` (`assert` preconditions) and `emit_runtime` (set to
    /// `false` and paste
    /// [`CodegenOptions::c_runtime`](crate::output::codegen::CodegenOptions::c_runtime)
    /// once when several functions share a translation unit).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::matrix::{CodegenOptions, Precision};
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let opts = CodegenOptions { precision: Precision::F32, inline: true, ..Default::default() };
    /// let code = x.exp().to_c_fn_with_options("f", &["x"], &opts).unwrap();
    /// assert!(code.contains("static inline float f(float x) {"));
    /// assert!(code.contains("expf(x)"));
    /// ```
    pub fn to_c_fn_with_options(
        &self,
        name: &str,
        args: &[&str],
        options: &crate::output::codegen::CodegenOptions,
    ) -> Result<String, SymplexError> {
        let mut guard = self.inner.write();
        crate::output::codegen::codegen_c::to_c_fn_with_options(
            &mut guard.arena,
            self.raw_id(),
            name,
            args,
            options,
        )
    }

    // ── Collection reduction ───────────────────────────────────────

    /// Sum a collection of expressions.
    ///
    /// All expressions must belong to the same context. Returns zero
    /// if the iterator is empty (using the context of `ctx`).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let terms: Vec<Ex> = (1..=4).map(|n| ctx.int(n)).collect();
    /// let total = Ex::sum_of(&ctx, terms);
    /// assert_eq!(format!("{total}"), "10");
    /// ```
    pub fn sum_of(ctx: &crate::api::context::Context, exprs: impl IntoIterator<Item = Ex>) -> Ex {
        let items: Vec<Ex> = exprs.into_iter().collect();
        if items.is_empty() {
            return ctx.int(0);
        }
        let mut inner = items[0].inner.write();
        let ids: smallvec::SmallVec<[crate::base::node::ExprId; 8]> =
            items.iter().map(|e| e.raw_id()).collect();
        let id = inner.arena.add(&ids);
        drop(inner);
        items[0].wrap(id)
    }

    /// Multiply a collection of expressions.
    ///
    /// All expressions must belong to the same context. Returns one
    /// if the iterator is empty (using the context of `ctx`).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let factors: Vec<Ex> = (1..=4).map(|n| ctx.int(n)).collect();
    /// let total = Ex::product_of(&ctx, factors);
    /// assert_eq!(format!("{total}"), "24");
    /// ```
    pub fn product_of(
        ctx: &crate::api::context::Context,
        exprs: impl IntoIterator<Item = Ex>,
    ) -> Ex {
        let items: Vec<Ex> = exprs.into_iter().collect();
        if items.is_empty() {
            return ctx.int(1);
        }
        let mut inner = items[0].inner.write();
        let ids: smallvec::SmallVec<[crate::base::node::ExprId; 8]> =
            items.iter().map(|e| e.raw_id()).collect();
        let id = inner.arena.mul(&ids);
        drop(inner);
        items[0].wrap(id)
    }

    // ── Expression walking ─────────────────────────────────────────

    /// Walk the expression bottom-up, applying a user-provided transformation
    /// at each node.
    ///
    /// The closure receives an [`ExprView`](crate::api::expr_view::ExprView) for
    /// each sub-expression — a non-locking, read-only view that supports
    /// identity comparison with [`Ex`] but cannot acquire any locks.
    /// Return `Some(replacement)` to replace it, or `None` to keep it
    /// unchanged.
    ///
    /// The walk is **structural**: it visits every node, bound variables
    /// included, which makes it the tool to rename a bound variable
    /// (`Sum(k*x, x, 0, 3)` with `x → j` is `Sum(k*j, j, 0, 3)`; use
    /// [`subs`](Ex::subs) to replace only free occurrences).  Replacing the
    /// variable of a binder (`Sum`, `Product`, `Integral`, `Derivative`,
    /// `Limit`, `RootOf`, `Subs`, …) by anything but a symbol makes a
    /// meaningless node — `Sum(k*x^2, x, 0, 3)` with `x → 1/3` is
    /// `Sum(1/9*k, 1/3=0..3)`.  SymPy's `xreplace` raises there;
    /// [`try_replace`](Self::try_replace) returns an error.
    ///
    /// # Panics
    ///
    /// Panics if the closure returns an expression built in another
    /// [`Context`](crate::api::context::Context) than `self` — the crate's
    /// cross-context logic error, as when combining two such expressions
    /// with an operator.  (Replacements are built before the call, as in the
    /// example: the context stays write-locked while the closure runs.)
    /// [`try_replace`](Self::try_replace) returns an error instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// let expr = x.powi(2);
    /// let replaced = expr.replace(|e| if e == &x { Some(y.clone()) } else { None });
    /// assert_eq!(format!("{replaced}"), "y^2");
    /// ```
    #[must_use = "returns the transformed expression; does not modify in place"]
    pub fn replace<F>(&self, f: F) -> Ex
    where
        F: Fn(crate::api::expr_view::ExprView<'_>) -> Option<Ex>,
    {
        let self_ctx_id = self.ctx_id;
        let result_id = {
            let mut guard = self.inner.write();
            crate::base::walk::walk_and_rebuild(&mut guard.arena, self.raw_id(), &|arena, id| {
                let view = crate::api::expr_view::ExprView { id, arena };
                f(view).map(|ex| {
                    assert!(
                        self_ctx_id == ex.ctx_id,
                        "replace: returned expression belongs to a different context"
                    );
                    ex.raw_id()
                })
            })
        };
        self.wrap(result_id)
    }

    /// [`replace`](Self::replace), checked: for a closure that may return
    /// an expression from another [`Context`](crate::api::context::Context)
    /// or replace a bound variable by a non-symbol.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if the closure returns an
    /// expression built in another context than `self`, or if the result
    /// would have a binder whose variable is not a symbol
    /// (`Sum(k*x^2, x, 0, 3)` with `x → 1/3`; SymPy's `xreplace` raises
    /// "Invalid limits").  Renaming a bound variable to a symbol is fine.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// let expr = x.powi(2);
    /// let replaced = expr.try_replace(|e| if e == &x { Some(y.clone()) } else { None }).unwrap();
    /// assert_eq!(format!("{replaced}"), "y^2");
    ///
    /// let other = Context::new().symbol("y");
    /// assert!(expr.try_replace(|e| if e == &x { Some(other.clone()) } else { None }).is_err());
    ///
    /// let sum = ctx.parse("Sum(k*x^2, x, 0, 3)").unwrap();
    /// let third = ctx.rational(1, 3);
    /// assert!(sum.try_replace(|e| if e == &x { Some(third.clone()) } else { None }).is_err());
    /// ```
    pub fn try_replace<F>(&self, f: F) -> Result<Ex, SymplexError>
    where
        F: Fn(crate::api::expr_view::ExprView<'_>) -> Option<Ex>,
    {
        let self_ctx_id = self.ctx_id;
        // A foreign replacement keeps the node and is reported after the walk.
        let foreign = std::cell::Cell::new(false);
        let result_id = {
            let mut guard = self.inner.write();
            crate::base::walk::walk_and_rebuild(&mut guard.arena, self.raw_id(), &|arena, id| {
                let view = crate::api::expr_view::ExprView { id, arena };
                let ex = f(view)?;
                if ex.ctx_id == self_ctx_id {
                    Some(ex.raw_id())
                } else {
                    foreign.set(true);
                    None
                }
            })
        };
        if foreign.get() {
            return Err(SymplexError::invalid_argument(
                "Ex::replace",
                "the closure returned an expression from another context",
            ));
        }
        let malformed = {
            let inner = self.inner.read();
            crate::base::walk::malformed_binder(&inner.arena, result_id, self.raw_id())
                .map(|id| inner.arena.display(id).to_string())
        };
        if let Some(node) = malformed {
            return Err(SymplexError::invalid_argument(
                "Ex::replace",
                format!(
                    "the result `{node}` binds a variable that is not a symbol; a bound \
                     variable can only be renamed (use `subs` to substitute a value)"
                ),
            ));
        }
        Ok(self.wrap(result_id))
    }

    // ── Solver utilities ───────────────────────────────────────────

    /// Check whether `val` is a solution of `self = 0` for variable `var`.
    ///
    /// Substitutes `val` for `var`, evaluates, and checks if the result is zero.
    /// Returns `Some(true)` if the residual is zero (structurally or numerically),
    /// `Some(false)` if definitely non-zero, or `None` if the result is ambiguous.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// let ctx = Context::new();
    /// let ctx = &ctx;
    /// symplex::syms!(ctx; x);
    /// let poly = expr!(ctx, x^2 - 4);
    /// assert_eq!(poly.check_solution(&x, &ctx.int(2)), Some(true));
    /// assert_eq!(poly.check_solution(&x, &ctx.int(-2)), Some(true));
    /// assert_eq!(poly.check_solution(&x, &ctx.int(3)), Some(false));
    /// ```
    #[must_use]
    pub fn check_solution(&self, var: &Ex, val: &Ex) -> Option<bool> {
        let substituted = self.subs(var, val).eval().simplify();
        if substituted.is_zero_structural() {
            return Some(true);
        }
        // A surviving nonzero numeric literal is a definite failure.
        let is_nonzero_literal = |e: &Ex| -> bool {
            let inner = e.inner.read();
            inner
                .arena
                .as_num(e.raw_id())
                .is_some_and(|r| !num_traits::Zero::is_zero(r))
        };
        if is_nonzero_literal(&substituted) {
            return Some(false);
        }
        // Try numerical (complex) evaluation.
        if let Ok(z) = substituted.eval_complex64() {
            let mag = z.norm();
            if mag < 1e-10 {
                return Some(true);
            }
            if mag > 1e-6 {
                return Some(false);
            }
        }
        // Try expand + eval
        let expanded = substituted.expand().eval();
        if expanded.is_zero_structural() {
            return Some(true);
        }
        if is_nonzero_literal(&expanded) {
            return Some(false);
        }
        // Assumption system: a provably nonzero residual (e.g. a positive
        // symbol) is a definite failure.
        if expanded.is_zero() == Some(false) {
            return Some(false);
        }
        None
    }

    /// Classify an ODE represented as `self = 0`.
    ///
    /// `self` should contain formal derivative nodes (created via
    /// [`formal_diff`](Self::formal_diff)). `func` is the dependent
    /// variable (e.g., `y`) and `var` is the independent variable (e.g., `x`).
    ///
    /// Returns an [`OdeType`](crate::calculus::ode::OdeType) describing the
    /// recognized ODE class, or [`OdeType::Unknown`](crate::calculus::ode::OdeType::Unknown)
    /// if the form is not recognized.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::ode::OdeType;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let y = ctx.symbol("y");
    /// let dy = y.formal_diff(&x);
    /// let ode = &dy - &x; // y' = x
    /// assert_eq!(ode.classify_ode(&y, &x), OdeType::SimpleSeparable);
    /// ```
    #[must_use]
    pub fn classify_ode(&self, func: &Ex, var: &Ex) -> crate::calculus::ode::OdeType {
        let func_id = self.checked_id(func);
        let var_id = self.checked_id(var);
        let mut guard = self.inner.write();
        crate::calculus::ode::classify_ode(&mut guard.arena, self.raw_id(), func_id, var_id)
    }

    /// Check whether `solution` satisfies the ODE `self = 0`.
    ///
    /// Substitutes the solution for `func` and its derivative for
    /// `Derivative(func, var)`, then evaluates and checks whether the
    /// residual is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let y = ctx.symbol("y");
    /// let dy = y.formal_diff(&x);
    /// let ode = &dy - &x; // y' - x = 0
    /// // Solution: y = x²/2
    /// let sol = &x.powi(2) / 2;
    /// assert!(ode.check_ode_solution(&sol, &y, &x));
    /// ```
    #[must_use]
    pub fn check_ode_solution(&self, solution: &Ex, func: &Ex, var: &Ex) -> bool {
        let solution_id = self.checked_id(solution);
        let func_id = self.checked_id(func);
        let var_id = self.checked_id(var);
        {
            let mut guard = self.inner.write();
            crate::calculus::ode::checkodesol(
                &mut guard.arena,
                self.raw_id(),
                solution_id,
                func_id,
                var_id,
            )
        }
    }

    // ── Set construction from numeric expressions ──────────────────

    /// Create a closed interval `[self, end]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let i = ctx.int(0).closed_interval(&ctx.int(1));
    /// let s = format!("{i}");
    /// assert!(s.contains("[") && s.contains("]"), "closed interval: {s}");
    /// ```
    #[must_use]
    pub fn closed_interval(&self, end: &Ex) -> SetEx {
        let end_id = self.checked_id(end);
        let id = self.inner.write().arena.interval(
            self.raw_id(),
            end_id,
            crate::base::node::INTERVAL_BOTH_CLOSED,
        );
        self.wrap_as(id)
    }

    /// Create an open interval `(self, end)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let i = ctx.int(0).open_interval(&ctx.int(1));
    /// let s = format!("{i}");
    /// assert!(s.contains("(") && s.contains(")"), "open interval: {s}");
    /// ```
    #[must_use]
    pub fn open_interval(&self, end: &Ex) -> SetEx {
        let end_id = self.checked_id(end);
        let id = self.inner.write().arena.interval(
            self.raw_id(),
            end_id,
            crate::base::node::INTERVAL_BOTH_OPEN,
        );
        self.wrap_as(id)
    }

    // ── Evaluation shortcuts (Wave P4) ─────────────────────────────

    /// Substitute values for symbols (simultaneously) and evaluate to `f64`.
    ///
    /// The values may be any [`ToEx`](crate::eq::ToEx) type: integers,
    /// `f64` (converted **exactly** — `0.1` is the dyadic
    /// `3602879701896397/36028797018963968`, which is what you want when
    /// the goal is a numeric answer), `BigInt`, `Ratio<BigInt>`, or `Ex`.
    /// All values in one call must have the same Rust type.
    ///
    /// Substitution goes through [`subs_map`](Self::subs_map) (simultaneous),
    /// then [`eval`](Self::eval), then [`eval_f64`](Self::eval_f64).
    ///
    /// # Errors
    ///
    /// Same as [`eval_f64`](Self::eval_f64): free symbols left unbound, or a
    /// result that is not a real number.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// let f = &x.powi(2) + &y;
    /// assert_eq!(f.eval_f64_with(&[(&x, 3), (&y, 1)]).unwrap(), 10.0);
    /// assert_eq!(f.eval_f64_with(&[(&x, 0.5), (&y, 0.25)]).unwrap(), 0.5);
    /// assert!(f.eval_f64_with(&[(&x, 1.0)]).is_err()); // y unbound
    /// ```
    pub fn eval_f64_with<V: crate::api::expr_ops::ToEx>(
        &self,
        subs: &[(&Ex, V)],
    ) -> Result<f64, SymplexError> {
        let bound = self.subs_map_with(subs).eval();
        if let Some(free) = bound.free_symbols().into_iter().next() {
            return Err(SymplexError::FreeSymbol {
                name: format!("{free}"),
            });
        }
        bound.eval_f64()
    }

    /// Substitute multiple rational values `p/q` and evaluate to `f64`.
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// assert_eq!((&x * 4).eval_f64_with_rational(&[(&x, 1, 2)]).unwrap(), 2.0);
    /// ```
    pub fn eval_f64_with_rational(&self, subs: &[(&Ex, i64, i64)]) -> Result<f64, SymplexError> {
        let ctx = self.context();
        let vals: Vec<Ex> = subs.iter().map(|(_, p, q)| ctx.rational(*p, *q)).collect();
        let pairs: Vec<(&Ex, &Ex)> = subs.iter().map(|(v, _, _)| *v).zip(vals.iter()).collect();
        self.subs_map(&pairs).eval().eval_f64()
    }

    /// Substitute multiple integer values simultaneously.
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// assert_eq!(format!("{}", (&x + &y).subs_map_i64(&[(&x, 1), (&y, 2)])), "3");
    /// ```
    #[must_use = "returns a new expression with substitutions applied"]
    pub fn subs_map_i64(&self, subs: &[(&Ex, i64)]) -> Ex {
        self.subs_map_with(subs)
    }

    /// Substitute values of any [`ToEx`](crate::eq::ToEx) type
    /// simultaneously (no evaluation).
    ///
    /// `f64` values are converted exactly; use
    /// [`Context::from_f64_nice`](crate::context::Context::from_f64_nice)
    /// first if you want `0.1 → 1/10`.
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// let e = (&x * &y).subs_map_with(&[(&x, 0.5), (&y, 4.0)]);
    /// assert_eq!(format!("{e}"), "2");
    /// ```
    #[must_use = "returns a new expression with substitutions applied"]
    pub fn subs_map_with<V: crate::api::expr_ops::ToEx>(&self, subs: &[(&Ex, V)]) -> Ex {
        let ctx = self.context();
        let vals: Vec<Ex> = subs.iter().map(|(_, v)| v.to_ex(&ctx)).collect();
        let pairs: Vec<(&Ex, &Ex)> = subs.iter().map(|(s, _)| *s).zip(vals.iter()).collect();
        self.subs_map(&pairs)
    }

    // ── Special function methods (Wave P5) ─────────────────────────

    /// Bessel function of the first kind: J_order(self).
    pub fn bessel_j(&self, order: &Ex) -> Ex {
        let order_id = self.checked_id(order);
        let id = {
            let mut guard = self.inner.write();
            guard.arena.besselj(order_id, self.raw_id())
        };
        self.wrap(id)
    }

    /// Bessel function of the second kind: Y_order(self).
    pub fn bessel_y(&self, order: &Ex) -> Ex {
        let order_id = self.checked_id(order);
        let id = {
            let mut guard = self.inner.write();
            guard.arena.bessely(order_id, self.raw_id())
        };
        self.wrap(id)
    }

    /// Modified Bessel function of the first kind: I_order(self).
    pub fn bessel_i(&self, order: &Ex) -> Ex {
        let order_id = self.checked_id(order);
        let id = {
            let mut guard = self.inner.write();
            guard.arena.besseli(order_id, self.raw_id())
        };
        self.wrap(id)
    }

    /// Modified Bessel function of the second kind: K_order(self).
    pub fn bessel_k(&self, order: &Ex) -> Ex {
        let order_id = self.checked_id(order);
        let id = {
            let mut guard = self.inner.write();
            guard.arena.besselk(order_id, self.raw_id())
        };
        self.wrap(id)
    }

    /// Legendre polynomial P_n(self).
    pub fn legendre(&self, n: &Ex) -> Ex {
        let n_id = self.checked_id(n);
        let id = {
            let mut guard = self.inner.write();
            guard.arena.legendre(n_id, self.raw_id())
        };
        self.wrap(id)
    }

    /// Chebyshev polynomial of the first kind T_n(self).
    pub fn chebyshev_t(&self, n: &Ex) -> Ex {
        let n_id = self.checked_id(n);
        let id = {
            let mut guard = self.inner.write();
            guard.arena.chebyshev_t(n_id, self.raw_id())
        };
        self.wrap(id)
    }

    /// Chebyshev polynomial of the second kind U_n(self).
    pub fn chebyshev_u(&self, n: &Ex) -> Ex {
        let n_id = self.checked_id(n);
        let id = {
            let mut guard = self.inner.write();
            guard.arena.chebyshev_u(n_id, self.raw_id())
        };
        self.wrap(id)
    }

    /// Hermite polynomial H_n(self) (physicist's convention).
    pub fn hermite(&self, n: &Ex) -> Ex {
        let n_id = self.checked_id(n);
        let id = {
            let mut guard = self.inner.write();
            guard.arena.hermite(n_id, self.raw_id())
        };
        self.wrap(id)
    }

    /// Laguerre polynomial L_n(self).
    pub fn laguerre(&self, n: &Ex) -> Ex {
        let n_id = self.checked_id(n);
        let id = {
            let mut guard = self.inner.write();
            guard.arena.laguerre(n_id, self.raw_id())
        };
        self.wrap(id)
    }

    // ── More special functions (0.9) ───────────────────────────────────

    /// Build a library `Apply(f, args)` node.  Every id in `args` must
    /// already have been validated with `checked_id` (or be `self`).
    fn special_apply(
        &self,
        f: crate::base::libfn::LibFn,
        args: &[crate::base::node::ExprId],
    ) -> Ex {
        let id = self.inner.write().arena.lib_apply(f, args);
        self.wrap(id)
    }

    /// Imaginary error function `erfi(self) = −i·erf(i·self) = (2/√π) ∫₀ˣ e^{t²} dt`.
    ///
    /// Exact: `erfi(0) = 0`, odd; `d/dx erfi(x) = 2e^{x²}/√π`.
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// assert_eq!(format!("{}", x.erfi()), "erfi(x)");
    /// assert_eq!(format!("{}", ctx.int(0).erfi().eval()), "0");
    /// assert!((ctx.rational(7, 10).erfi().eval_f64().unwrap() - 0.94028293383350736168).abs() < 1e-14);
    /// ```
    #[must_use]
    pub fn erfi(&self) -> Ex {
        self.special_apply(crate::base::libfn::LibFn::Erfi, &[self.raw_id()])
    }

    /// Inverse error function `erfinv(self)`: `erf(erfinv(y)) = y` for `|y| < 1`.
    ///
    /// Exact: `erfinv(0) = 0`, `erfinv(±1) = ±∞`, odd;
    /// `d/dy erfinv(y) = (√π/2) e^{erfinv(y)²}`.
    #[must_use]
    pub fn erfinv(&self) -> Ex {
        self.special_apply(crate::base::libfn::LibFn::ErfInv, &[self.raw_id()])
    }

    /// Inverse complementary error function `erfcinv(self) = erfinv(1 − self)`.
    ///
    /// Exact: `erfcinv(1) = 0`, `erfcinv(0) = ∞`, `erfcinv(2) = −∞`.
    #[must_use]
    pub fn erfcinv(&self) -> Ex {
        self.special_apply(crate::base::libfn::LibFn::ErfcInv, &[self.raw_id()])
    }

    /// Generalised exponential integral `E_n(self) = ∫₁^∞ e^{−self·t} t^{−n} dt`
    /// (SymPy `expint(n, x)`; the order comes first in the display).
    ///
    /// Exact: `E_n(0) = 1/(n−1)` for `n > 1`, `E_0(x) = e^{−x}/x`,
    /// `E_n(∞) = 0`; `d/dx E_n(x) = −E_{n−1}(x)`.
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// assert_eq!(format!("{}", x.expint(&ctx.int(2))), "expint(2, x)");
    /// assert_eq!(format!("{}", ctx.int(0).expint(&ctx.int(3)).eval()), "1/2");
    /// ```
    #[must_use]
    pub fn expint(&self, n: &Ex) -> Ex {
        let n_id = self.checked_id(n);
        self.special_apply(crate::base::libfn::LibFn::ExpInt, &[n_id, self.raw_id()])
    }

    /// Exponential integral `E₁(self) = expint(1, self) = ∫_self^∞ e^{−t}/t dt`.
    ///
    /// For `x > 0`, `E₁(x) = −Ei(−x)`.
    #[must_use]
    pub fn e1(&self) -> Ex {
        let one = self.inner.read().arena.one;
        self.special_apply(crate::base::libfn::LibFn::ExpInt, &[one, self.raw_id()])
    }

    /// Hyperbolic sine integral `Shi(self) = ∫₀ˣ sinh(t)/t dt`.
    ///
    /// Exact: `Shi(0) = 0`, odd; `d/dx Shi(x) = sinh(x)/x`.
    #[must_use]
    pub fn shi(&self) -> Ex {
        self.special_apply(crate::base::libfn::LibFn::Shi, &[self.raw_id()])
    }

    /// Hyperbolic cosine integral `Chi(self) = γ + ln x + ∫₀ˣ (cosh(t) − 1)/t dt`.
    ///
    /// Exact: `Chi(0) = −∞`; `d/dx Chi(x) = cosh(x)/x`.
    #[must_use]
    pub fn chi(&self) -> Ex {
        self.special_apply(crate::base::libfn::LibFn::Chi, &[self.raw_id()])
    }

    /// Fresnel sine integral `S(self) = ∫₀ˣ sin(πt²/2) dt`.
    ///
    /// Exact: `S(0) = 0`, `S(±∞) = ±1/2`, odd; `d/dx S(x) = sin(πx²/2)`.
    #[must_use]
    pub fn fresnels(&self) -> Ex {
        self.special_apply(crate::base::libfn::LibFn::FresnelS, &[self.raw_id()])
    }

    /// Fresnel cosine integral `C(self) = ∫₀ˣ cos(πt²/2) dt`.
    ///
    /// Exact: `C(0) = 0`, `C(±∞) = ±1/2`, odd; `d/dx C(x) = cos(πx²/2)`.
    #[must_use]
    pub fn fresnelc(&self) -> Ex {
        self.special_apply(crate::base::libfn::LibFn::FresnelC, &[self.raw_id()])
    }

    /// Lower incomplete gamma function `γ(s, self) = ∫₀ˣ t^{s−1} e^{−t} dt`
    /// (SymPy `lowergamma(s, x)`).
    ///
    /// Exact: `γ(s, 0) = 0`, `γ(s, ∞) = Γ(s)`, `γ(1, x) = 1 − e^{−x}`,
    /// `γ(1/2, x) = √π erf(√x)`, and closed forms for small integer and
    /// half-integer `s`; `∂/∂x γ(s, x) = x^{s−1} e^{−x}`.
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// assert_eq!(x.lowergamma(&ctx.int(1)).eval(), 1 - (-&x).exp());
    /// assert_eq!(format!("{}", x.uppergamma(&ctx.int(1)).eval()), "exp(-x)");
    /// ```
    #[must_use]
    pub fn lowergamma(&self, s: &Ex) -> Ex {
        let s_id = self.checked_id(s);
        self.special_apply(
            crate::base::libfn::LibFn::LowerGamma,
            &[s_id, self.raw_id()],
        )
    }

    /// Upper incomplete gamma function `Γ(s, self) = ∫_x^∞ t^{s−1} e^{−t} dt`
    /// (SymPy `uppergamma(s, x)`).
    ///
    /// Exact: `Γ(s, 0) = Γ(s)`, `Γ(s, ∞) = 0`, `Γ(1, x) = e^{−x}`,
    /// `Γ(0, x) = E₁(x)`, `Γ(1/2, x) = √π erfc(√x)`, and closed forms for
    /// small integer and half-integer `s`; `∂/∂x Γ(s, x) = −x^{s−1} e^{−x}`.
    #[must_use]
    pub fn uppergamma(&self, s: &Ex) -> Ex {
        let s_id = self.checked_id(s);
        self.special_apply(
            crate::base::libfn::LibFn::UpperGamma,
            &[s_id, self.raw_id()],
        )
    }

    /// Polylogarithm `Li_s(self) = Σ_{k≥1} self^k / k^s` (SymPy `polylog(s, z)`).
    ///
    /// Exact: `Li_s(0) = 0`, `Li_s(1) = ζ(s)`, `Li_s(−1) = −η(s)`,
    /// `Li_1(z) = −ln(1 − z)`, `Li_0(z) = z/(1 − z)`, `Li_{−n}(z)` rational,
    /// `Li_2(1/2) = π²/12 − ln²2/2`; `d/dz Li_s(z) = Li_{s−1}(z)/z`.
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let z = ctx.symbol("z");
    /// assert_eq!(format!("{}", z.polylog(&ctx.int(2))), "polylog(2, z)");
    /// assert_eq!(format!("{}", ctx.int(1).polylog(&ctx.int(2)).eval()), "1/6*pi^2");
    /// assert_eq!(z.polylog(&ctx.int(-1)).eval(), &z / (1 - &z).powi(2));
    /// ```
    #[must_use]
    pub fn polylog(&self, s: &Ex) -> Ex {
        let s_id = self.checked_id(s);
        self.special_apply(crate::base::libfn::LibFn::PolyLog, &[s_id, self.raw_id()])
    }

    /// Dirichlet eta function `η(self) = Σ (−1)^{k+1}/k^s = (1 − 2^{1−s}) ζ(s)`.
    ///
    /// Exact: `η(1) = ln 2`, `η(0) = 1/2`, and `η(s)` rewrites through `ζ(s)`
    /// whenever `s` is an integer (so `η(2) = π²/12`).
    #[must_use]
    pub fn dirichlet_eta(&self) -> Ex {
        self.special_apply(crate::base::libfn::LibFn::DirichletEta, &[self.raw_id()])
    }

    /// Airy function of the first kind `Ai(self)`.
    ///
    /// Exact: `Ai(0) = 1/(3^{2/3} Γ(2/3))`, `Ai(±∞) = 0`; `Ai' = airyaiprime`.
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// assert_eq!(format!("{}", x.airyai().diff(&x)), "airyaiprime(x)");
    /// assert_eq!(format!("{}", x.airyaiprime().diff(&x)), "x*airyai(x)");
    /// ```
    #[must_use]
    pub fn airyai(&self) -> Ex {
        self.special_apply(crate::base::libfn::LibFn::AiryAi, &[self.raw_id()])
    }

    /// Airy function of the second kind `Bi(self)`.
    ///
    /// Exact: `Bi(0) = 1/(3^{1/6} Γ(2/3))`, `Bi(−∞) = 0`, `Bi(∞) = ∞`.
    #[must_use]
    pub fn airybi(&self) -> Ex {
        self.special_apply(crate::base::libfn::LibFn::AiryBi, &[self.raw_id()])
    }

    /// Derivative of the Airy function of the first kind `Ai′(self)`.
    ///
    /// Exact: `Ai′(0) = −1/(3^{1/3} Γ(1/3))`; `d/dx Ai′(x) = x·Ai(x)`.
    #[must_use]
    pub fn airyaiprime(&self) -> Ex {
        self.special_apply(crate::base::libfn::LibFn::AiryAiPrime, &[self.raw_id()])
    }

    /// Derivative of the Airy function of the second kind `Bi′(self)`.
    ///
    /// Exact: `Bi′(0) = 3^{1/6}/Γ(1/3)`; `d/dx Bi′(x) = x·Bi(x)`.
    #[must_use]
    pub fn airybiprime(&self) -> Ex {
        self.special_apply(crate::base::libfn::LibFn::AiryBiPrime, &[self.raw_id()])
    }

    /// Complete elliptic integral of the first kind
    /// `K(m) = ∫₀^{π/2} dθ / √(1 − m sin²θ)` with `self = m = k²`.
    ///
    /// Exact: `K(0) = π/2`, `K(1) = z∞`;
    /// `d/dm K = (E(m) − (1 − m)K(m)) / (2m(1 − m))`.
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// assert_eq!(format!("{}", ctx.int(0).elliptic_k().eval()), "1/2*pi");
    /// // K(1/2) = Γ(1/4)² / (4√π)
    /// assert!((ctx.rational(1, 2).elliptic_k().eval_f64().unwrap() - 1.8540746773013719184).abs() < 1e-14);
    /// ```
    #[must_use]
    pub fn elliptic_k(&self) -> Ex {
        self.special_apply(crate::base::libfn::LibFn::EllipticK, &[self.raw_id()])
    }

    /// Complete elliptic integral of the second kind
    /// `E(m) = ∫₀^{π/2} √(1 − m sin²θ) dθ` with `self = m = k²`.
    ///
    /// Exact: `E(0) = π/2`, `E(1) = 1`; `d/dm E = (E(m) − K(m)) / (2m)`.
    #[must_use]
    pub fn elliptic_e(&self) -> Ex {
        self.special_apply(crate::base::libfn::LibFn::EllipticE, &[self.raw_id()])
    }

    /// Incomplete elliptic integral of the first kind
    /// `F(φ | m) = ∫₀^φ dθ / √(1 − m sin²θ)` with `self = φ`.
    ///
    /// Exact: `F(0 | m) = 0`, `F(φ | 0) = φ`, `F(π/2 | m) = K(m)`;
    /// `∂/∂φ F = 1/√(1 − m sin²φ)` (the `m`-derivative stays formal).
    #[must_use]
    pub fn elliptic_f(&self, m: &Ex) -> Ex {
        let m_id = self.checked_id(m);
        self.special_apply(crate::base::libfn::LibFn::EllipticF, &[self.raw_id(), m_id])
    }

    /// Complete elliptic integral of the third kind
    /// `Π(n | m) = ∫₀^{π/2} dθ / ((1 − n sin²θ) √(1 − m sin²θ))` with `self = n`.
    ///
    /// Exact: `Π(0 | m) = K(m)`, `Π(n | 0) = π/(2√(1 − n))`, `Π(n | n) = E(n)/(1 − n)`,
    /// `Π(1 | m) = z∞`; both partial derivatives have closed forms in `K`, `E`, `Π`.
    #[must_use]
    pub fn elliptic_pi(&self, m: &Ex) -> Ex {
        let m_id = self.checked_id(m);
        self.special_apply(
            crate::base::libfn::LibFn::EllipticPi,
            &[self.raw_id(), m_id],
        )
    }

    /// Gegenbauer (ultraspherical) polynomial `C_n^{(a)}(self)`.
    ///
    /// Expands to an explicit polynomial under `eval` for integer `n ≥ 0`;
    /// `C_n^{(1/2)} = P_n`, `C_n^{(1)} = U_n`; `d/dx C_n^{(a)} = 2a C_{n−1}^{(a+1)}`.
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, a) = (ctx.symbol("x"), ctx.symbol("a"));
    /// assert_eq!(format!("{}", x.gegenbauer(&ctx.int(2), &a).eval()), "2*a^2*x^2 + 2*a*x^2 - a");
    /// ```
    #[must_use]
    pub fn gegenbauer(&self, n: &Ex, a: &Ex) -> Ex {
        let n_id = self.checked_id(n);
        let a_id = self.checked_id(a);
        self.special_apply(
            crate::base::libfn::LibFn::Gegenbauer,
            &[n_id, a_id, self.raw_id()],
        )
    }

    /// Jacobi polynomial `P_n^{(a, b)}(self)`.
    ///
    /// Expands to an explicit polynomial under `eval` for integer `n ≥ 0`;
    /// `P_n^{(0,0)} = P_n`; `d/dx P_n^{(a,b)} = (n + a + b + 1)/2 · P_{n−1}^{(a+1, b+1)}`.
    #[must_use]
    pub fn jacobi(&self, n: &Ex, a: &Ex, b: &Ex) -> Ex {
        let n_id = self.checked_id(n);
        let a_id = self.checked_id(a);
        let b_id = self.checked_id(b);
        self.special_apply(
            crate::base::libfn::LibFn::Jacobi,
            &[n_id, a_id, b_id, self.raw_id()],
        )
    }

    /// Associated Legendre function `P_n^m(self)` (Condon–Shortley phase,
    /// as in SymPy: `P_1^1(x) = −√(1 − x²)`).
    ///
    /// Expands under `eval` for integer `n ≥ 0` and integer `m` (zero when
    /// `|m| > n`); `P_n^0 = P_n`;
    /// `d/dx P_n^m = (n x P_n^m − (n + m) P_{n−1}^m) / (x² − 1)`.
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// assert_eq!(format!("{}", x.assoc_legendre(&ctx.int(2), &ctx.int(2)).eval()), "-3*x^2 + 3");
    /// ```
    #[must_use]
    pub fn assoc_legendre(&self, n: &Ex, m: &Ex) -> Ex {
        let n_id = self.checked_id(n);
        let m_id = self.checked_id(m);
        self.special_apply(
            crate::base::libfn::LibFn::AssocLegendre,
            &[n_id, m_id, self.raw_id()],
        )
    }

    /// Generalised (associated) Laguerre polynomial `L_n^{(a)}(self)`.
    ///
    /// Expands under `eval` for integer `n ≥ 0`; `L_n^{(0)} = L_n`;
    /// `d/dx L_n^{(a)} = −L_{n−1}^{(a+1)}`.
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, a) = (ctx.symbol("x"), ctx.symbol("a"));
    /// assert_eq!(format!("{}", x.assoc_laguerre(&ctx.int(1), &a).eval()), "a - x + 1");
    /// ```
    #[must_use]
    pub fn assoc_laguerre(&self, n: &Ex, a: &Ex) -> Ex {
        let n_id = self.checked_id(n);
        let a_id = self.checked_id(a);
        self.special_apply(
            crate::base::libfn::LibFn::AssocLaguerre,
            &[n_id, a_id, self.raw_id()],
        )
    }

    // ── Incomplete beta (0.12) ─────────────────────────────────────────

    /// Generalised incomplete beta function
    /// `B_{(x₁, x₂)}(a, b) = ∫_{x₁}^{x₂} t^{a−1} (1 − t)^{b−1} dt`.
    ///
    /// **Argument order.** `self` is the *upper* limit `x₂`; the node is
    /// stored in SymPy's order `betainc(a, b, x1, x2)`, so
    /// `x2.betainc(&a, &b, &x1)` displays as `betainc(a, b, x1, x2)`.  The
    /// classical incomplete beta `B_x(a, b)` is `x.betainc(&a, &b, &zero)`.
    ///
    /// Exact: `betainc(a, b, x, x) = 0`, `betainc(a, b, 0, 1) = B(a, b)`, and
    /// for positive integers `a`, `b` the integrand is a polynomial, so the
    /// node expands to an explicit polynomial in `x₁`, `x₂`;
    /// `∂/∂x₂ = x₂^{a−1}(1 − x₂)^{b−1}`, `∂/∂x₁ = −x₁^{a−1}(1 − x₁)^{b−1}`
    /// (parameter derivatives stay formal).  Numerically evaluated for
    /// `a, b > 0` and `0 ≤ x₁, x₂ ≤ 1`.
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (a, b, x) = (ctx.symbol("a"), ctx.symbol("b"), ctx.symbol("x"));
    /// let f = x.betainc(&a, &b, &ctx.int(0));
    /// assert_eq!(format!("{f}"), "betainc(a, b, 0, x)");
    /// assert_eq!(format!("{}", x.betainc(&ctx.int(2), &ctx.int(3), &ctx.int(0)).eval()), "1/4*x^4 - 2/3*x^3 + 1/2*x^2");
    /// ```
    #[must_use]
    pub fn betainc(&self, a: &Ex, b: &Ex, x1: &Ex) -> Ex {
        let a_id = self.checked_id(a);
        let b_id = self.checked_id(b);
        let x1_id = self.checked_id(x1);
        self.special_apply(
            crate::base::libfn::LibFn::BetaInc,
            &[a_id, b_id, x1_id, self.raw_id()],
        )
    }

    /// Regularised generalised incomplete beta function
    /// `I_{(x₁, x₂)}(a, b) = B_{(x₁, x₂)}(a, b) / B(a, b)`.
    ///
    /// **Argument order.** As for [`betainc`](Self::betainc): `self` is the
    /// upper limit `x₂` and the node is stored in SymPy's order
    /// `betainc_regularized(a, b, x1, x2)`.  `x.betainc_regularized(&a, &b, &zero)`
    /// is the Beta-distribution CDF `I_x(a, b)`.
    ///
    /// Exact: `I_{(x, x)} = 0`, `I_{(0, 1)}(a, b) = 1`, and the polynomial
    /// expansion (divided by `B(a, b)`) for positive integers `a`, `b`;
    /// `∂/∂x₂ = x₂^{a−1}(1 − x₂)^{b−1} / B(a, b)`.
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// // Beta(2, 3) CDF
    /// let cdf = x.betainc_regularized(&ctx.int(2), &ctx.int(3), &ctx.int(0)).eval();
    /// assert_eq!(format!("{cdf}"), "3*x^4 - 8*x^3 + 6*x^2");
    /// ```
    #[must_use]
    pub fn betainc_regularized(&self, a: &Ex, b: &Ex, x1: &Ex) -> Ex {
        let a_id = self.checked_id(a);
        let b_id = self.checked_id(b);
        let x1_id = self.checked_id(x1);
        self.special_apply(
            crate::base::libfn::LibFn::BetaIncRegularized,
            &[a_id, b_id, x1_id, self.raw_id()],
        )
    }

    // ── Formal power series ────────────────────────────────────────

    /// Compute the formal power series of this expression about `point`.
    ///
    /// Returns a [`FormalPowerSeries`](crate::calculus::formal_series::FormalPowerSeries)
    /// with exact, lazily computed coefficients (`Ex`-valued), a closed-form
    /// general term for elementary functions, truncation, and series
    /// arithmetic (`add`, `mul`, `compose`, `inverse`, `reversion`, …).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let series = x.exp().fps(&x, &ctx.int(0));
    /// assert!(series.has_closed_form());
    /// assert_eq!(series.coefficient(3).to_string(), "1/6");
    /// ```
    #[must_use]
    pub fn fps(&self, var: &Ex, point: &Ex) -> crate::calculus::formal_series::FormalPowerSeries {
        let var_id = self.checked_id(var);
        let point_id = self.checked_id(point);
        let _span = debug_span!("fps", expr = ?self.raw_id()).entered();
        crate::calculus::formal_series::fps(&self.context(), self.raw_id(), var_id, point_id)
    }

    /// Compute the formal power series about 0 (Maclaurin series).
    ///
    /// Convenience shorthand for `self.fps(var, &zero)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let series = x.sin().fps_maclaurin(&x);
    /// assert!(series.has_closed_form());
    /// assert_eq!(series.truncate(6).to_string(), "1/120*x^5 - 1/6*x^3 + x");
    /// ```
    #[must_use]
    pub fn fps_maclaurin(&self, var: &Ex) -> crate::calculus::formal_series::FormalPowerSeries {
        let var_id = self.checked_id(var);
        let zero = self.inner.read().arena.zero;
        crate::calculus::formal_series::fps(&self.context(), self.raw_id(), var_id, zero)
    }

    // ── Finite differences ─────────────────────────────────────────

    /// Finite-difference approximation of the `order`-th derivative of this
    /// expression with respect to `var`, on the stencil `points`, evaluated
    /// at `var` itself:
    ///
    /// ```text
    /// d^order self / d var^order  ≈  Σᵢ wᵢ · self(var → points[i])
    /// ```
    ///
    /// The weights `wᵢ` are exact rationals (or exact expressions in `h`)
    /// from Fornberg's algorithm.  Formal `Derivative(f, var)` nodes inside
    /// `self` are replaced first, each by the finite difference of its own
    /// order on the same stencil; `order = 0` then simply performs that
    /// replacement.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let h = ctx.symbol("h");
    /// let stencil = [&x - &h, x.clone(), &x + &h];
    /// // central difference of x³ is exact up to the h² term: 3x² + h²
    /// let d = x.powi(3).differentiate_finite(&x, &stencil, 1).expand();
    /// assert_eq!(d.to_string(), "h^2 + 3*x^2");
    ///
    /// // replace a formal derivative node
    /// let d = x.sin().formal_diff(&x).differentiate_finite(&x, &stencil, 0);
    /// assert!(!d.to_string().contains("Derivative"));
    /// ```
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn differentiate_finite(&self, var: &Ex, points: &[Ex], order: usize) -> Ex {
        crate::calculus::finite_diff::differentiate_finite(self, var, points, order)
    }

    /// Factorize this integer expression into prime factors.
    ///
    /// Evaluates the expression and, if it is an exact integer, returns its
    /// prime factorization as `(BigInt_prime, exponent)` pairs. Handles
    /// arbitrary-precision integers without any `f64` or `i64` truncation.
    /// Returns `None` if the expression is not an integer.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use num_bigint::BigInt;
    ///
    /// let ctx = Context::new();
    /// let n = ctx.int(60);
    /// let factors = n.factorize().unwrap();
    /// assert_eq!(
    ///     factors,
    ///     vec![(BigInt::from(2), 2), (BigInt::from(3), 1), (BigInt::from(5), 1)]
    /// );
    /// ```
    pub fn factorize(&self) -> Option<Vec<(BigInt, u32)>> {
        let evaled = self.eval();
        let inner = evaled.inner.read();
        let arena = &inner.arena;
        match arena.node(evaled.raw_id()) {
            crate::base::node::ExprNode::Num(nid) => {
                let r = arena.num(*nid);
                if !r.is_integer() {
                    return None;
                }
                let n = r.numer();
                if n.is_zero() {
                    return None;
                }
                Some(crate::domains::ntheory::factorint(n.clone()))
            }
            _ => None,
        }
    }

    /// Check if this expression evaluates to a prime number.
    ///
    /// Accesses the exact `Ratio<BigInt>` value in the arena and uses the
    /// BigInt primality test — no lossy `f64` conversion. Returns `None`
    /// if the expression cannot be evaluated to an integer.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let n = ctx.int(104729);
    /// assert_eq!(n.is_prime_value(), Some(true));
    ///
    /// let n = ctx.int(60);
    /// assert_eq!(n.is_prime_value(), Some(false));
    ///
    /// let half = ctx.rational(1, 2);
    /// assert_eq!(half.is_prime_value(), None);
    /// ```
    pub fn is_prime_value(&self) -> Option<bool> {
        let evaled = self.eval();
        let inner = evaled.inner.read();
        let arena = &inner.arena;
        match arena.node(evaled.raw_id()) {
            crate::base::node::ExprNode::Num(nid) => {
                let r = arena.num(*nid);
                if !r.is_integer() {
                    return None;
                }
                let n = r.numer();
                Some(crate::domains::ntheory::isprime(n.clone()))
            }
            _ => None,
        }
    }

    // ── Plotting API (Wave P6) ─────────────────────────────────────

    /// Validate the common plotting arguments and return the variable name.
    ///
    /// Checks: `var` is a symbol, the plot range `[a, b]` is finite with
    /// `a < b`, and no free symbol other than `var` occurs in `self`.
    fn plot_check_args(
        &self,
        var: &Ex,
        bounds: Option<Interval<f64>>,
        operation: &'static str,
    ) -> Result<String, SymplexError> {
        use crate::base::node::ExprNode;

        let var_id = self.checked_id(var);
        let is_symbol = matches!(self.inner.read().arena.node(var_id), ExprNode::Symbol(_));
        if !is_symbol {
            return Err(SymplexError::InvalidArgument {
                operation,
                reason: format!("plot variable must be a symbol, got `{var}`"),
            });
        }
        if let Some(range) = bounds {
            let (a, b) = (range.lower, range.upper);
            if !a.is_finite() || !b.is_finite() {
                return Err(SymplexError::InvalidArgument {
                    operation,
                    reason: format!("plot range must be finite, got [{a}, {b}]"),
                });
            }
            if a >= b {
                return Err(SymplexError::InvalidArgument {
                    operation,
                    reason: format!("plot range must satisfy a < b, got [{a}, {b}]"),
                });
            }
        }
        if let Some(other) = self.free_symbols().into_iter().find(|s| s != var) {
            return Err(SymplexError::FreeSymbol {
                name: format!("{other}"),
            });
        }
        Ok(format!("{var}"))
    }

    /// Build a fast `f64 → f64` evaluator for `self` in `var`.
    ///
    /// Uses [`compile`](Self::compile) when the expression is compilable.
    /// Nodes without a compiled form (user `Apply`, unevaluated integrals,
    /// …) fall back to exact substitution of the sample point followed by
    /// [`eval_f64`](Self::eval_f64); points where that fails yield `NaN`.
    fn plot_evaluator(
        &self,
        var: &Ex,
        var_name: &str,
    ) -> Result<Box<dyn Fn(f64) -> f64 + Send + Sync>, SymplexError> {
        match self.compile(&[var_name]) {
            Ok(f) => Ok(Box::new(move |x: f64| f(&[x]))),
            Err(SymplexError::NotImplemented(_)) => {
                let expr = self.clone();
                let var = var.clone();
                let ctx = self.context();
                Ok(Box::new(move |x: f64| match ctx.from_f64(x) {
                    Ok(v) => expr.subs(&var, &v).eval().eval_f64().unwrap_or(f64::NAN),
                    Err(_) => f64::NAN,
                }))
            }
            Err(e) => Err(e),
        }
    }

    /// Internal: domain-aware adaptive sampling for plotting.
    ///
    /// Uses [`calculus_util::singularities`](crate::calculus::calculus_util::singularities) to find excluded points,
    /// [`calculus_util::estimate_frequency`](crate::calculus::calculus_util::estimate_frequency) to determine sampling density,
    /// and [`sampling::sample_compiled`](crate::plotting::sampling::sample_compiled) for adaptive refinement.
    fn sample_expression(
        &self,
        var: &Ex,
        a: f64,
        b: f64,
        operation: &'static str,
    ) -> Result<crate::plotting::sampling::PlotData, SymplexError> {
        use crate::base::node::ExprNode;

        let range = Interval::closed(a, b);
        let var_name = self.plot_check_args(var, Some(range), operation)?;
        let var_id = self.checked_id(var);

        // Step 1: Domain analysis (needs write lock for singularities)
        let (excluded_points, min_points) = {
            let mut inner = self.inner.write();
            match inner.arena.node(var_id) {
                ExprNode::Symbol(sid) => {
                    let sid = *sid;
                    let excluded = crate::calculus::calculus_util::singularities(
                        &mut inner.arena,
                        self.raw_id(),
                        var_id,
                        sid,
                        range,
                    );
                    let freq = crate::calculus::calculus_util::estimate_frequency(
                        &inner.arena,
                        self.raw_id(),
                        var_id,
                        sid,
                    );
                    let min_pts = match freq {
                        Some(f) => crate::plotting::sampling::min_points_for_frequency(f, range),
                        None => 200,
                    };
                    (excluded, min_pts)
                }
                _ => (Vec::new(), 200),
            }
        }; // write lock dropped

        // Step 2: Build the evaluator and sample.
        let f = self.plot_evaluator(var, &var_name)?;
        let opts = crate::plotting::sampling::SampleOptions {
            min_points,
            ..Default::default()
        };
        let data = crate::plotting::sampling::sample_compiled(&*f, range, &excluded_points, &opts)?;
        if !data.points.iter().any(|(_, y)| y.is_finite()) {
            return Err(SymplexError::ComputationFailed {
                operation,
                reason: format!("`{self}` has no finite real values on [{a}, {b}]"),
            });
        }
        Ok(data)
    }

    /// Generate an ASCII-art plot of this expression over `[a, b]`.
    ///
    /// Compiles the expression for fast numerical evaluation, performs
    /// domain-aware adaptive sampling (singularities are detected and
    /// skipped), and renders the result as a 60×21 character grid.
    ///
    /// # Errors
    ///
    /// * [`SymplexError::InvalidArgument`] if `var` is not a symbol, or the
    ///   range is not finite with `a < b`.
    /// * [`SymplexError::FreeSymbol`] if the expression contains a symbol
    ///   other than `var`.
    /// * [`SymplexError::ComputationFailed`] if the expression has no finite
    ///   real value anywhere on the range.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let plot = x.sin().textplot(&x, 0.0, 6.28).unwrap();
    /// assert!(plot.lines().count() >= 21);
    ///
    /// let y = ctx.symbol("y");
    /// assert!(matches!((&x + &y).textplot(&x, 0.0, 1.0), Err(SymplexError::FreeSymbol { .. })));
    /// assert!(x.textplot(&x, 1.0, 0.0).is_err());
    /// ```
    pub fn textplot(&self, var: &Ex, a: f64, b: f64) -> Result<String, SymplexError> {
        let plot_data = self.sample_expression(var, a, b, "textplot")?;
        Ok(crate::plotting::textplot::textplot(
            &plot_data.points,
            60,
            21,
            None,
        ))
    }

    /// Generate an SVG plot of this expression over `[a, b]`.
    ///
    /// Returns a self-contained SVG document with axes, grid, and the
    /// function curve rendered as `<polyline>` segments (one per
    /// continuous branch).
    ///
    /// # Errors
    ///
    /// Same conditions as [`textplot`](Self::textplot).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let svg = x.sin().to_svg(&x, 0.0, 6.28).unwrap();
    /// assert!(svg.starts_with("<svg") || svg.contains("<svg"));
    /// assert!(svg.contains("</svg>"));
    /// ```
    pub fn to_svg(&self, var: &Ex, a: f64, b: f64) -> Result<String, SymplexError> {
        let plot_data = self.sample_expression(var, a, b, "to_svg")?;
        let series = vec![(plot_data.points.as_slice(), "f(x)")];
        let opts = crate::plotting::svg_plot::SvgPlotOptions::default();
        Ok(crate::plotting::svg_plot::svg_plot(&series, &opts))
    }

    /// Generate TikZ/PGFplots code for this expression over `[a, b]`.
    ///
    /// Returns a complete `tikzpicture` environment with axis options and
    /// coordinate data, ready to `\input` into a LaTeX document that loads
    /// `pgfplots`.
    ///
    /// # Errors
    ///
    /// Same conditions as [`textplot`](Self::textplot).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let tikz = x.sin().to_tikz(&x, 0.0, 6.28).unwrap();
    /// assert!(tikz.contains("\\begin{axis}"));
    /// assert!(tikz.contains("\\end{tikzpicture}"));
    /// ```
    pub fn to_tikz(&self, var: &Ex, a: f64, b: f64) -> Result<String, SymplexError> {
        let plot_data = self.sample_expression(var, a, b, "to_tikz")?;
        Ok(crate::plotting::tikz_plot::tikz_plot(
            &[(&plot_data.points, "f(x)")],
            None,
            Some("x"),
            Some("y"),
            false,
            false,
        ))
    }

    /// Generate `(x, y)` sample data for this expression at `n`
    /// uniformly-spaced points over `[a, b]` (both endpoints included).
    ///
    /// Points where the function is not a finite real number produce `NaN`
    /// y-values, so the returned vector always has exactly `n` entries.
    ///
    /// # Errors
    ///
    /// * [`SymplexError::InvalidArgument`] if `var` is not a symbol, `n < 2`,
    ///   or the range is not finite with `a < b`.
    /// * [`SymplexError::FreeSymbol`] if the expression contains a symbol
    ///   other than `var`.
    /// * [`SymplexError::ComputationFailed`] if *every* sample is
    ///   non-finite (e.g. `ln(x)` on `[-2, -1]`).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let data = x.powi(2).plot_data(&x, 0.0, 1.0, 11).unwrap();
    /// assert_eq!(data.len(), 11);
    /// assert!((data[5].1 - 0.25).abs() < 1e-12); // x = 0.5
    ///
    /// assert!(x.plot_data(&x, 0.0, 1.0, 1).is_err());
    /// assert!(x.ln().plot_data(&x, -2.0, -1.0, 5).is_err());
    /// ```
    pub fn plot_data(
        &self,
        var: &Ex,
        a: f64,
        b: f64,
        n: usize,
    ) -> Result<Vec<(f64, f64)>, SymplexError> {
        let var_name = self.plot_check_args(var, Some(Interval::closed(a, b)), "plot_data")?;
        if n < 2 {
            return Err(SymplexError::InvalidArgument {
                operation: "plot_data",
                reason: format!("need at least 2 sample points, got {n}"),
            });
        }
        let f = self.plot_evaluator(var, &var_name)?;
        let step = (b - a) / (n as f64 - 1.0);
        let data: Vec<(f64, f64)> = (0..n)
            .map(|i| {
                // Hit the right endpoint exactly.
                let x = if i + 1 == n { b } else { a + i as f64 * step };
                (x, f(x))
            })
            .collect();
        if !data.iter().any(|(_, y)| y.is_finite()) {
            return Err(SymplexError::ComputationFailed {
                operation: "plot_data",
                reason: format!("`{self}` has no finite real values on [{a}, {b}]"),
            });
        }
        Ok(data)
    }

    /// Evaluate this expression at each of `points` and return a two-column
    /// [`DataTable`](crate::plotting::data_export::DataTable) (`x`, `f(x)`)
    /// for export to CSV, JSON, Markdown, LaTeX, ….
    ///
    /// Non-finite results are recorded as `NaN` / `Inf` cells; they are not
    /// an error.
    ///
    /// # Errors
    ///
    /// * [`SymplexError::InvalidArgument`] if `var` is not a symbol or any
    ///   input point is not finite.
    /// * [`SymplexError::FreeSymbol`] if the expression contains a symbol
    ///   other than `var`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let table = x.powi(2).eval_table(&x, &[0.0, 1.0, 2.0]).unwrap();
    /// assert_eq!(table.nrows(), 3);
    /// assert_eq!(table.rows[2], vec!["2", "4"]);
    /// assert!(table.to_csv().starts_with("x,f(x)\n"));
    /// ```
    pub fn eval_table(
        &self,
        var: &Ex,
        points: &[f64],
    ) -> Result<crate::plotting::data_export::DataTable, SymplexError> {
        let var_name = self.plot_check_args(var, None, "eval_table")?;
        if let Some(bad) = points.iter().find(|p| !p.is_finite()) {
            return Err(SymplexError::InvalidArgument {
                operation: "eval_table",
                reason: format!("input points must be finite, got {bad}"),
            });
        }
        let f = self.plot_evaluator(var, &var_name)?;
        Ok(crate::plotting::data_export::DataTable::from_evaluation(
            "x", "f(x)", points, f,
        ))
    }
}

#[cfg(test)]
mod evalf_complex_tests {
    #[test]
    fn eval_complex64_pure_real() {
        let ctx = crate::api::context::Context::new();
        let x = ctx.symbol("x");
        let expr = &x.powi(2) + 1;
        let at_2 = expr.subs(&x, &ctx.int(2));
        let z = at_2.eval_complex64().unwrap();
        assert!((z.re - 5.0).abs() < 1e-10);
        assert!(z.im.abs() < 1e-10);
    }

    #[test]
    fn eval_complex64_pure_imaginary() {
        let ctx = crate::api::context::Context::new();
        let i = ctx.i_unit();
        let z = i.eval_complex64().unwrap();
        assert!(z.re.abs() < 1e-10);
        assert!((z.im - 1.0).abs() < 1e-10);
    }

    #[test]
    fn eval_complex64_mixed() {
        let ctx = crate::api::context::Context::new();
        let expr = &ctx.int(3) + &(&ctx.int(4) * &ctx.i_unit());
        let z = expr.eval_complex64().unwrap();
        assert!((z.re - 3.0).abs() < 1e-10);
        assert!((z.im - 4.0).abs() < 1e-10);
    }

    #[test]
    fn eval_f64_rejects_complex() {
        let ctx = crate::api::context::Context::new();
        let i = ctx.i_unit();
        assert!(
            i.eval_f64().is_err(),
            "eval_f64 should reject pure imaginary"
        );
    }
}
