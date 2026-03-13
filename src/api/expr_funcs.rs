//! Numeric expression methods — math functions, calculus, algebra, solving.
//!
//! This module contains all `impl Expr<Numeric>` method blocks.
//! The type definitions live in `expr.rs`.

use num_bigint::BigInt;
use num_traits::Zero;
use tracing::debug_span;

use crate::api::expr::{BoolEx, Ex, Expr, Numeric, SetEx, SetValued};
use crate::base::assumptions::{Assumption, Props};
use crate::base::errors::SymplexError;

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

    /// Cube root: `∛self`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn cbrt(&self) -> Ex {
        let id = self.inner.write().arena.cbrt(self.raw_id());
        self.wrap(id)
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

    /// Symbolic summation: `Sum(body, var=lower..upper)`.
    ///
    /// When evaluated (`.eval()`), if `lower` and `upper` are concrete integers,
    /// the sum is computed by substituting each integer value for `var` in `body`.
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

    /// Symbolic product: `Product(body, var=lower..upper)`.
    ///
    /// When evaluated (`.eval()`), if `lower` and `upper` are concrete integers,
    /// the product is computed by substituting each integer value for `var` in `body`.
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

    /// Test whether the infinite series `Σ_{k=1}^{∞} self(var)` converges.
    ///
    /// Returns `Some(true)` if the series converges, `Some(false)` if it
    /// diverges, or `None` if the test is inconclusive.
    ///
    /// Applies several tests in sequence: divergence test, p-series test,
    /// and geometric series test.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let k = ctx.symbol("k");
    /// // 1/k² converges (p-series with p=2)
    /// let body = k.powi(-2);
    /// assert_eq!(body.is_convergent(&k), Some(true));
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

    /// Decompose this expression into its real part.
    ///
    /// Assumes unadorned symbols are real. Returns the real component
    /// of the expression when written as `re + im·i`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let i = ctx.i_unit();
    /// let z = &ctx.int(3) + &(&ctx.int(4) * &i);
    /// assert_eq!(format!("{}", z.re()), "3");
    /// ```
    #[must_use]
    pub fn re(&self) -> Ex {
        let (re, _im) = self.inner.write().arena.as_real_imag_expr(self.raw_id());
        self.wrap(re)
    }

    /// Decompose this expression into its imaginary part.
    ///
    /// Assumes unadorned symbols are real. Returns the imaginary
    /// coefficient (without the `i` factor) when the expression is
    /// written as `re + im·i`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let i = ctx.i_unit();
    /// let z = &ctx.int(3) + &(&ctx.int(4) * &i);
    /// assert_eq!(format!("{}", z.im()), "4");
    /// ```
    #[must_use]
    pub fn im(&self) -> Ex {
        let (_re, im) = self.inner.write().arena.as_real_imag_expr(self.raw_id());
        self.wrap(im)
    }

    /// Complex argument (phase angle): `arg(z) = atan2(im(z), re(z))`.
    ///
    /// Returns the angle in (-π, π] between the positive real axis and z.
    /// Handles all four quadrants correctly.
    #[must_use]
    pub fn arg(&self) -> Ex {
        let im = self.im();
        let re = self.re();
        im.atan2(&re)
    }

    /// Complex conjugate: `conjugate(a + bi) = a - bi`.
    #[must_use]
    pub fn conjugate(&self) -> Ex {
        let re = self.re();
        let im = self.im();
        let i_id = self.inner.read().arena.i_unit();
        let i_ex: Ex = self.wrap(i_id);
        &re - &(&im * &i_ex)
    }

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
    /// symbol, the assumption is silently ignored.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let t = ctx.symbol("t")
    ///     .assume(Assumption::Positive)
    ///     .assume(Assumption::Real);
    /// assert_eq!(t.is_positive(), Some(true));
    /// assert_eq!(t.is_real(), Some(true));
    /// ```
    pub fn assume(self, assumption: Assumption) -> Ex {
        use crate::base::node::ExprNode;
        let mut inner = self.inner.write();
        if let ExprNode::Symbol(sid) = inner.arena.node(self.raw_id()) {
            let sid = *sid;
            let (prop, value) = assumption.to_prop_value();
            let mut a = inner.arena.symbol_assumptions(sid);
            if value {
                a.assert_true(prop);
            } else {
                a.assert_false(prop);
            }
            inner.arena.set_symbol_assumptions(sid, a);
            inner
                .assumptions
                .lock()
                .set_symbol_assumptions(self.raw_id(), a);
        }
        drop(inner);
        self
    }

    /// Mathematical equality: attempts to determine if `self - other == 0`.
    ///
    /// Uses layered detection:
    /// 1. Structural identity (same `ExprId` — O(1))
    /// 2. Compute `self - other` and check if canonically zero
    /// 3. Expand `self - other` and check again
    ///
    /// Returns `Some(true)` if provably equal, `Some(false)` if provably
    /// not equal, or `None` if unknown.
    #[must_use]
    pub fn equals(&self, other: &Ex) -> Option<bool> {
        // Layer 1: structural identity (same arena node).
        let other_id = self.checked_id(other);
        if self.raw_id() == other_id {
            return Some(true);
        }

        // Layer 2: compute self - other and check if zero.
        let diff = self - other;
        if diff.is_zero_structural() {
            return Some(true);
        }

        // Layer 3: expand the difference and check again.
        let expanded = diff.expand();
        if expanded.is_zero_structural() {
            return Some(true);
        }

        // Layer 4: simplify the difference (catches trig identities, etc.)
        let simplified = diff.smart_simplify();
        if simplified.is_zero_structural() {
            return Some(true);
        }

        // Could not determine equality.
        None
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

    /// Compute a definite integral: `∫_lower^upper self dx`.
    ///
    /// Computes the antiderivative via [`integrate`](Ex::integrate),
    /// then evaluates `F(upper) - F(lower)`. If the antiderivative
    /// is unevaluated (returned an `Integral` node), the result will
    /// contain unevaluated terms.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// // ∫₀¹ x² dx = 1/3
    /// let result = x.powi(2).definite_integral(&x, &ctx.int(0), &ctx.int(1));
    /// assert_eq!(format!("{result}"), "1/3");
    /// ```
    #[must_use = "returns the definite integral value"]
    pub fn definite_integral(&self, var: &Ex, lower: &Ex, upper: &Ex) -> Ex {
        let anti = self.integrate(var);
        let f_upper = anti.subs(var, upper);
        let f_lower = anti.subs(var, lower);
        &f_upper - &f_lower
    }

    /// Compute the Taylor series around `point` to the given `order`.
    ///
    /// Returns the truncated polynomial with `order` terms:
    /// `f(a) + f'(a)(x-a) + f''(a)(x-a)²/2! + ...`
    ///
    /// If `point` is zero, this is a Maclaurin series.
    /// If the series cannot be computed, returns a formal `Series` node.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let zero = ctx.int(0);
    /// let expr = x.exp();
    /// let s = expr.series(&x, &zero, 4);
    /// let expanded = s.expand().eval();
    /// let result = format!("{expanded}");
    /// assert!(result.contains("x"), "should have x term: {result}");
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

    /// Compute the Maclaurin series (Taylor series around 0) to the
    /// given `order`.
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
    /// let s = x.sin().maclaurin(&x, 4);
    /// let result = s.expand().eval();
    /// let text = format!("{result}");
    /// assert!(text.contains("x"), "should have x term: {text}");
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
    /// expansion of the function around `a`. For a simple pole at `a`,
    /// this equals `lim_{x→a} (x-a) * f(x)`.
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
        let var_id = self.checked_id(var);
        let point_id = self.checked_id(point);
        let _span = debug_span!("limit", expr = ?self.raw_id(), var = ?var_id).entered();
        let mut inner = self.inner.write();
        match inner.arena.limit_expr(self.raw_id(), var_id, point_id) {
            Ok(id) => {
                drop(inner);
                self.wrap(id)
            }
            Err(_) => {
                let id = inner.arena.intern(crate::base::node::ExprNode::Limit(
                    self.raw_id(),
                    var_id,
                    point_id,
                ));
                drop(inner);
                self.wrap(id)
            }
        }
    }

    /// Like [`limit`](Self::limit), but returns `Err` if the result
    /// contains unevaluated forms (e.g. a formal `Limit` node).
    pub fn try_limit(&self, var: &Ex, point: &Ex) -> Result<Ex, SymplexError> {
        let result = self.limit(var, point);
        if result.has_unevaluated() {
            Err(SymplexError::ComputationFailed {
                operation: "limit",
                reason: "could not compute limit".into(),
            })
        } else {
            Ok(result)
        }
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

    /// Expand logarithmic expressions.
    ///
    /// Applies logarithm properties:
    /// - `ln(a * b)` → `ln(a) + ln(b)`
    /// - `ln(a^n)` → `n * ln(a)`
    /// - `ln(a / b)` → `ln(a) - ln(b)`
    ///
    /// These rules are valid for positive real arguments.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// let expr = (&x * &y).ln();
    /// let expanded = expr.expand_log();
    /// let s = format!("{expanded}");
    /// assert!(s.contains("ln(x)") && s.contains("ln(y)"), "should expand: {s}");
    /// ```
    #[must_use = "returns the expanded form; does not modify in place"]
    pub fn expand_log(&self) -> Ex {
        let id = self.inner.write().arena.expand_log_expr(self.raw_id());
        self.wrap(id)
    }

    /// Combine logarithmic terms (inverse of [`expand_log`](Self::expand_log)).
    ///
    /// Applies: `ln(a) + ln(b) → ln(a·b)` and `n·ln(a) → ln(aⁿ)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// let expr = &x.ln() + &y.ln();
    /// let combined = expr.log_combine();
    /// let s = format!("{combined}");
    /// assert!(s.contains("ln"), "should combine logs: {s}");
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
    /// let x = ctx.symbol("x").assume(Assumption::Positive);
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
    /// let refined = expr.refine_with(&[(&x, Assumption::Positive)]);
    /// assert_eq!(format!("{refined}"), format!("{x}"));
    /// // Original x is unchanged — no permanent assumption was set:
    /// assert!(x.is_positive().is_none());
    /// ```
    #[must_use = "returns the refined form; does not modify in place"]
    pub fn refine_with(
        &self,
        temp_assumptions: &[(&Ex, crate::base::assumptions::Assumption)],
    ) -> Ex {
        use crate::base::assumptions::{AssumptionCache, Assumptions};
        use crate::base::node::ExprNode;

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

        // Save original assumptions for symbols we're temporarily overriding.
        let mut saved: Vec<(crate::base::node::SymbolId, Assumptions)> = Vec::new();
        for (i, (_var, assumption)) in temp_assumptions.iter().enumerate() {
            if let ExprNode::Symbol(sid) = arena.node(var_ids[i]) {
                let sid = *sid;
                saved.push((sid, arena.symbol_assumptions(sid)));
                let mut a = arena.symbol_assumptions(sid);
                let (prop, value) = assumption.to_prop_value();
                if value {
                    a.assert_true(prop);
                } else {
                    a.assert_false(prop);
                }
                a.forward_chain();
                arena.set_symbol_assumptions(sid, a);
            }
        }

        // Run refine with a fresh assumption cache (picks up the temp assumptions).
        let mut temp_cache = AssumptionCache::new();
        let id = crate::simplify::refine::refine_full(arena, &mut temp_cache, self.raw_id());

        // Restore original assumptions.
        for (sid, original) in saved {
            arena.set_symbol_assumptions(sid, original);
        }

        // Invalidate the main assumption cache since we temporarily mutated symbols.
        {
            let mut main_cache = assumptions.lock();
            *main_cache = AssumptionCache::new();
        }

        drop(inner);
        self.wrap(id)
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
    /// Returns the expression unchanged if it is not polynomial in `var`.
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
    /// ```
    #[must_use = "returns the collected form; does not modify in place"]
    pub fn collect(&self, var: &Ex) -> Ex {
        let var_id = self.checked_id(var);
        let id = self.inner.write().arena.collect_expr(self.raw_id(), var_id);
        self.wrap(id)
    }

    /// Combine fractions over a common denominator.
    ///
    /// For a sum of terms, decomposes each into numerator/denominator,
    /// computes a common denominator, scales each numerator, and
    /// rebuilds as a single fraction.
    ///
    /// Returns the expression unchanged if it is not a sum or if all
    /// terms already have denominator 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// let expr = &x.powi(-1) + &y.powi(-1);
    /// let combined = expr.together();
    /// // 1/x + 1/y → (x + y) / (x*y)
    /// let s = format!("{combined}");
    /// assert!(s.contains("x*y"), "should have common denom x*y: {s}");
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
    /// Equivalent to calling [`together`](Self::together) to combine
    /// fractions over a common denominator, then [`cancel`](Self::cancel)
    /// with each free symbol to remove common factors.
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
    /// let s = format!("{simplified}");
    /// assert!(s.contains("2"), "simplify_rational should combine: {s}");
    /// ```
    #[must_use = "returns the simplified form; does not modify in place"]
    pub fn simplify_rational(&self) -> Ex {
        let together = self.together();
        // Try to cancel with each free symbol
        let syms = together.free_symbols();
        let mut result = together;
        for sym in &syms {
            result = result.cancel(sym);
        }
        result
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
    /// or is the zero polynomial.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// assert_eq!((&x.powi(3) + &x + 1).degree(&x), Some(3));
    /// assert_eq!(x.sin().degree(&x), None);
    /// ```
    #[must_use]
    pub fn degree(&self, var: &Ex) -> Option<usize> {
        let var_id = self.checked_id(var);
        let inner = self.inner.read();
        inner.arena.degree_of(self.raw_id(), var_id)
    }

    /// Return the coefficients of this expression as a polynomial in `var`,
    /// in ascending degree order: `[a_0, a_1, a_2, ...]`.
    ///
    /// Returns `None` if the expression is not polynomial in `var`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// // x^2 + 3*x + 5 → coefficients [5, 3, 1]
    /// let expr = &x.powi(2) + &x * 3 + 5;
    /// let cs = expr.coeffs(&x).unwrap();
    /// let strs: Vec<String> = cs.iter().map(|c| format!("{c}")).collect();
    /// assert_eq!(strs, vec!["5", "3", "1"]);
    /// ```
    #[must_use]
    pub fn coeffs(&self, var: &Ex) -> Option<Vec<Ex>> {
        let var_id = self.checked_id(var);
        let mut inner = self.inner.write();
        let ids = inner.arena.coefficients_of(self.raw_id(), var_id)?;
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

    /// Decompose this expression into (numerator, denominator).
    ///
    /// For `a / b` (expressed as `a * b^(-1)`), returns `(a, b)`.
    /// For expressions without a denominator, returns `(self, 1)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// let expr = &x / &y;
    /// let (n, d) = expr.as_numer_denom();
    /// assert_eq!(format!("{n}"), "x");
    /// assert_eq!(format!("{d}"), "y");
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
    /// Returns a vector of values of `var` that make this expression
    /// zero.  Supports linear, quadratic, and higher-degree polynomial
    /// equations (via rational root finding).
    ///
    /// Returns an empty vector if:
    /// - The expression is not polynomial in `var`.
    /// - No closed-form solutions can be found.
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
    /// ```
    pub fn solve(&self, var: &Ex) -> Result<Vec<Ex>, SymplexError> {
        let var_id = self.checked_id(var);
        let _span = debug_span!("solve", expr = ?self.raw_id(), var = ?var_id).entered();
        let mut inner = self.inner.write();
        // Try the internal solver first — it handles polynomials, transcendental
        // equations (exp, ln, sin, sqrt via inversion peeling), change-of-variable,
        // Lambert W, and symbolic linear equations.
        let solutions = inner.arena.solve_for(self.raw_id(), var_id);
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
        let poly = crate::poly::polybridge::expr_to_poly(&inner.arena, self.raw_id(), var_id);
        drop(inner);
        if poly.is_none() {
            return Err(SymplexError::ComputationFailed {
                operation: "solve",
                reason: "expression is not polynomial in the given variable and transcendental solver could not find solutions".into(),
            });
        }
        Ok(vec![])
    }

    /// Solve `self = 0` for `var`, returning an empty vector on failure.
    ///
    /// This is a convenience wrapper around [`solve`](Ex::solve) that
    /// returns `vec![]` if the solver fails (e.g., expression is not
    /// polynomial). Use [`solve`](Ex::solve) for diagnostic information.
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

    /// Solve `self = 0`, returning solutions as a `FiniteSet`.
    ///
    /// This is a set-valued variant of [`solve`](Ex::solve) — instead of
    /// returning a `Vec<Ex>`, it returns a `SetEx` (a `FiniteSet` node).
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

    /// Numerical root finding via Newton's method.
    ///
    /// Finds a numerical root of `self = 0` near `initial_guess` by
    /// iterating `x_{n+1} = x_n - f(x_n)/f'(x_n)`.
    ///
    /// # Arguments
    /// - `var` — the variable to solve for
    /// - `initial_guess` — starting point for iteration
    /// - `max_iterations` — maximum number of Newton steps
    /// - `tolerance` — convergence threshold (stop when `|f(x)| < tolerance`)
    ///
    /// # Errors
    /// Returns `Err` if the method doesn't converge within `max_iterations`,
    /// if the derivative is zero, or if evaluation fails.
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
    /// ```
    pub fn solve_numeric(
        &self,
        var: &Ex,
        initial_guess: f64,
        max_iterations: usize,
        tolerance: f64,
    ) -> Result<f64, SymplexError> {
        let deriv = self.diff(var);
        let mut x = initial_guess;

        for _ in 0..max_iterations {
            // Build a rational approximation of x and substitute.
            let r = match num_rational::Ratio::<num_bigint::BigInt>::from_float(x) {
                Some(r) => r,
                None => {
                    return Err(SymplexError::ComputationFailed {
                        operation: "solve_numeric",
                        reason: format!("could not approximate x = {x} as rational"),
                    });
                }
            };
            let x_rational = {
                let mut inner = self.inner.write();
                let nid = inner.arena.intern_num(r);
                let id = inner.arena.intern(crate::base::node::ExprNode::Num(nid));
                drop(inner);
                self.wrap(id)
            };

            let f_val = self.subs(var, &x_rational).eval_f64()?;
            let fp_val = deriv.subs(var, &x_rational).eval_f64()?;

            if fp_val.abs() < 1e-30 {
                return Err(SymplexError::ComputationFailed {
                    operation: "solve_numeric",
                    reason: "derivative is effectively zero".into(),
                });
            }

            x -= f_val / fp_val;

            if f_val.abs() < tolerance {
                return Ok(x);
            }
        }

        Err(SymplexError::ComputationFailed {
            operation: "solve_numeric",
            reason: format!(
                "did not converge within {} iterations (last x = {x})",
                max_iterations
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
    /// precision exceeds `EvalConfig::max_evalf_precision`, or if
    /// intermediate computation produces NaN.
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
    /// Calls [`eval_decimal`](Ex::eval_decimal) with 16 digits of precision and
    /// parses the result to `f64`. This avoids the common pattern of
    /// `.eval_decimal(15).unwrap().parse::<f64>().unwrap()`.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`eval_decimal`](Ex::eval_decimal), plus a
    /// [`SymplexError::NotImplemented`] if the decimal string cannot
    /// be parsed to `f64`.
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
        // eval() first to reduce exact values (sin(0)→0, Gamma(5)→24, etc.)
        // before numerical computation. The eval_complex64 call below
        // will work on the simplified expression.
        let (re, im) = self.eval().eval_complex64()?;
        if im.abs() > 1e-15 {
            return Err(SymplexError::ComputationFailed {
                operation: "eval_f64",
                reason: format!(
                    "expression has nonzero imaginary part (im={im}); use eval_complex64() for complex results"
                ),
            });
        }
        Ok(re)
    }

    /// Evaluates the expression to a complex f64 pair `(real, imaginary)`.
    ///
    /// Uses 16 decimal digits of precision internally. Returns both the
    /// real and imaginary parts, correctly handling complex expressions
    /// like `sqrt(-1)` → `(0.0, 1.0)`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the expression contains free symbols or if
    /// the arbitrary-precision engine fails.
    pub fn eval_complex64(&self) -> Result<(f64, f64), SymplexError> {
        let s = self.eval_decimal(16)?;
        parse_complex_evalf_string(&s)
    }

    // ── Code generation ────────────────────────────────────────────

    /// Compile this expression into a callable closure for fast numerical evaluation.
    ///
    /// `var_names` specifies the variable-to-index mapping: the returned
    /// closure takes `&[f64]` where index 0 corresponds to `var_names[0]`, etc.
    ///
    /// Returns `None` if the expression contains nodes that cannot be
    /// numerically evaluated (e.g., `ImaginaryUnit`, unevaluated integrals).
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
    /// ```
    #[allow(clippy::type_complexity)]
    pub fn compile(&self, var_names: &[&str]) -> Option<Box<dyn Fn(&[f64]) -> f64 + Send + Sync>> {
        // Pre-pass: run eval() to catch exact-zero terms (sin(0)→0, exp(0)→1,
        // cos(π)→-1, perfect-square roots, etc.) before code emission.
        // This is cheap (single bottom-up walk) and can eliminate entire
        // subexpressions, improving both precision and performance.
        let evaled_id = {
            let mut inner = self.inner.write();
            crate::transforms::eval::eval(&mut inner.arena, self.raw_id())
        };
        let inner = self.inner.read();
        crate::output::lambdify::lambdify(&inner.arena, evaled_id, var_names)
    }

    /// Perform common subexpression elimination (CSE).
    ///
    /// Identifies repeated subexpressions and extracts them into named
    /// temporaries (`__cse_0`, `__cse_1`, …), reducing redundant
    /// computation when generating code.
    ///
    /// Returns a list of `(name, value)` bindings and the rewritten
    /// expression where common subexpressions are replaced by their names.
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
        // Try numerical evaluation
        if let Ok(v) = substituted.eval_f64() {
            if v.abs() < 1e-10 {
                return Some(true);
            }
            if v.abs() > 1e-6 {
                return Some(false);
            }
        }
        // Try expand + eval
        let expanded = substituted.expand().eval();
        if expanded.is_zero_structural() {
            return Some(true);
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

    /// Substitute multiple integer values and evaluate to f64.
    ///
    /// Combines `subs_i64` for each variable, then `eval()`, then `evalf_f64()`.
    pub fn eval_f64_with(&self, subs: &[(&Ex, i64)]) -> Result<f64, SymplexError> {
        let mut result = self.clone();
        for (var, val) in subs {
            result = result.subs_i64(var, *val);
        }
        result.eval().eval_f64()
    }

    /// Substitute multiple rational values and evaluate to f64.
    pub fn eval_f64_with_rational(&self, subs: &[(&Ex, i64, i64)]) -> Result<f64, SymplexError> {
        let ctx = self.context();
        let mut result = self.clone();
        for (var, p, q) in subs {
            let val = ctx.rational(*p, *q);
            result = result.subs(var, &val);
        }
        result.eval().eval_f64()
    }

    /// Substitute multiple integer values simultaneously.
    pub fn subs_map_i64(&self, subs: &[(&Ex, i64)]) -> Ex {
        let mut result = self.clone();
        for (var, val) in subs {
            result = result.subs_i64(var, *val);
        }
        result
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

    // ── Formal power series ────────────────────────────────────────

    /// Compute the formal power series of this expression about `point`.
    ///
    /// Returns a [`FormalPowerSeries`](crate::calculus::formal_series::FormalPowerSeries)
    /// that provides access to individual coefficients and truncation.
    ///
    /// For known elementary functions (exp, sin, cos, sinh, cosh, ln(1+x),
    /// atan, (1+x)^α, 1/(1-x)), returns a closed-form coefficient formula.
    /// For other functions, falls back to computing Taylor coefficients.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let zero = ctx.int(0);
    /// let series = x.exp().fps(&x, &zero);
    /// assert!(series.has_closed_form());
    /// ```
    #[must_use]
    pub fn fps(&self, var: &Ex, point: &Ex) -> crate::calculus::formal_series::FormalPowerSeries {
        let var_id = self.checked_id(var);
        let point_id = self.checked_id(point);
        let mut inner = self.inner.write();
        crate::calculus::formal_series::fps(&mut inner.arena, self.raw_id(), var_id, point_id)
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
    /// ```
    #[must_use]
    pub fn fps_maclaurin(&self, var: &Ex) -> crate::calculus::formal_series::FormalPowerSeries {
        let var_id = self.checked_id(var);
        let mut inner = self.inner.write();
        let zero = inner.arena.zero;
        crate::calculus::formal_series::fps(&mut inner.arena, self.raw_id(), var_id, zero)
    }

    // ── Finite differences ─────────────────────────────────────────

    /// Replace derivatives in this expression with finite difference
    /// approximations.
    ///
    /// When encountering `Derivative(f, x)`, replaces it with the central
    /// difference formula `(f(x + h/2) - f(x - h/2)) / h` where `h` is
    /// the symbol `_h`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let expr = x.powi(2).formal_diff(&x);
    /// let finite = expr.differentiate_finite(&x);
    /// let s = format!("{finite}");
    /// assert!(s.contains("_h"), "should contain step size: {s}");
    /// ```
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn differentiate_finite(&self, var: &Ex) -> Ex {
        let var_id = self.checked_id(var);
        let id = {
            let mut guard = self.inner.write();
            crate::calculus::finite_diff::differentiate_finite(
                &mut guard.arena,
                self.raw_id(),
                var_id,
            )
        };
        self.wrap(id)
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

    /// Internal: domain-aware adaptive sampling for plotting.
    ///
    /// Uses [`calculus_util::singularities`] to find excluded points,
    /// [`calculus_util::estimate_frequency`] to determine sampling density,
    /// and [`sampling::sample_compiled`] for adaptive refinement.
    fn sample_expression(&self, var: &Ex, a: f64, b: f64) -> crate::plotting::sampling::PlotData {
        use crate::base::node::ExprNode;

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
                        (a, b),
                    );
                    let freq = crate::calculus::calculus_util::estimate_frequency(
                        &inner.arena,
                        self.raw_id(),
                        var_id,
                        sid,
                    );
                    let min_pts = match freq {
                        Some(f) => crate::plotting::sampling::min_points_for_frequency(f, (a, b)),
                        None => 200,
                    };
                    (excluded, min_pts)
                }
                _ => (Vec::new(), 200),
            }
        }; // write lock dropped

        // Step 2: Compile and sample (compile acquires a read lock)
        let var_name = format!("{var}");
        let compiled = self.compile(&[&var_name]);
        match compiled {
            Some(f) => {
                let opts = crate::plotting::sampling::SampleOptions {
                    min_points,
                    ..Default::default()
                };
                let f_single = move |x: f64| -> f64 { f(&[x]) };
                crate::plotting::sampling::sample_compiled(
                    &f_single,
                    (a, b),
                    &excluded_points,
                    &opts,
                )
            }
            None => {
                // Fallback: symbolic substitution
                let n = min_points.max(2);
                let step = (b - a) / (n as f64 - 1.0);
                let points: Vec<(f64, f64)> = (0..n)
                    .map(|i| {
                        let x = a + i as f64 * step;
                        let (p, q) = f64_to_rational_approx(x);
                        let val = self.context().rational(p, q);
                        let y = self.subs(var, &val).eval().eval_f64().unwrap_or(f64::NAN);
                        (x, y)
                    })
                    .collect();
                crate::plotting::sampling::PlotData {
                    points,
                    asymptotes: Vec::new(),
                    excluded: excluded_points,
                }
            }
        }
    }

    /// Generate ASCII art plot of this expression over `[a, b]`.
    ///
    /// Compiles the expression for fast numerical evaluation, performs
    /// domain-aware adaptive sampling, and renders the result as a
    /// character grid.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let plot = x.sin().textplot(&x, 0.0, 6.28);
    /// assert!(!plot.is_empty());
    /// ```
    #[must_use]
    pub fn textplot(&self, var: &Ex, a: f64, b: f64) -> String {
        let plot_data = self.sample_expression(var, a, b);
        crate::plotting::textplot::textplot(&plot_data.points, 60, 21, None)
    }

    /// Generate SVG plot of this expression over `[a, b]`.
    ///
    /// Returns a self-contained SVG string with axes, grid, and the
    /// function curve rendered as a `<polyline>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let svg = x.sin().to_svg(&x, 0.0, 6.28);
    /// assert!(svg.contains("<svg"));
    /// ```
    #[must_use]
    pub fn to_svg(&self, var: &Ex, a: f64, b: f64) -> String {
        let plot_data = self.sample_expression(var, a, b);
        let series = vec![(plot_data.points.as_slice(), "f(x)")];
        let opts = crate::plotting::svg_plot::SvgPlotOptions::default();
        crate::plotting::svg_plot::svg_plot(&series, &opts)
    }

    /// Generate TikZ/PGFplots code for this expression over `[a, b]`.
    ///
    /// Returns a string containing a complete `tikzpicture` environment
    /// with axis options and coordinate data.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let tikz = x.sin().to_tikz(&x, 0.0, 6.28);
    /// assert!(tikz.contains("\\begin{axis}"));
    /// ```
    #[must_use]
    pub fn to_tikz(&self, var: &Ex, a: f64, b: f64) -> String {
        let plot_data = self.sample_expression(var, a, b);
        crate::plotting::tikz_plot::tikz_plot(
            &[(&plot_data.points, "f(x)")],
            None,
            Some("x"),
            Some("y"),
            false,
            false,
        )
    }

    /// Generate `(x, y)` sample data for this expression over `[a, b]`.
    ///
    /// Compiles the expression to a closure and evaluates it at `n`
    /// uniformly-spaced points. Points where the function is not finite
    /// produce `NaN` y-values.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let data = x.powi(2).plot_data(&x, 0.0, 1.0, 10);
    /// assert_eq!(data.len(), 10);
    /// ```
    #[must_use]
    pub fn plot_data(&self, var: &Ex, a: f64, b: f64, n: usize) -> Vec<(f64, f64)> {
        // Extract variable name from the expression.
        let var_name = format!("{var}");
        let compiled = self.compile(&[&var_name]);
        let n = n.max(2);
        let step = (b - a) / (n as f64 - 1.0);

        match compiled {
            Some(f) => (0..n)
                .map(|i| {
                    let x = a + i as f64 * step;
                    let y = f(&[x]);
                    (x, y)
                })
                .collect(),
            None => {
                // Fallback: use symbolic substitution + eval_f64
                (0..n)
                    .map(|i| {
                        let x = a + i as f64 * step;
                        let (p, q) = f64_to_rational_approx(x);
                        let val = self.context().rational(p, q);
                        let y = self.subs(var, &val).eval().eval_f64().unwrap_or(f64::NAN);
                        (x, y)
                    })
                    .collect()
            }
        }
    }

    /// Export evaluation table as a [`DataTable`](crate::plotting::data_export::DataTable).
    ///
    /// Evaluates this expression at each point in `points` and returns
    /// a two-column table (`x`, `f(x)`) suitable for export to CSV,
    /// JSON, LaTeX, and other formats.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let table = x.powi(2).eval_table(&x, &[0.0, 1.0, 2.0]);
    /// assert_eq!(table.nrows(), 3);
    /// ```
    #[must_use]
    pub fn eval_table(&self, var: &Ex, points: &[f64]) -> crate::plotting::data_export::DataTable {
        let var_name = format!("{var}");
        let compiled = self.compile(&[&var_name]);
        match compiled {
            Some(f) => {
                crate::plotting::data_export::DataTable::from_evaluation("x", "f(x)", points, |x| {
                    f(&[x])
                })
            }
            None => {
                let values: Vec<(f64, f64)> = points
                    .iter()
                    .map(|&x| {
                        let (p, q) = f64_to_rational_approx(x);
                        let val = self.context().rational(p, q);
                        let y = self.subs(var, &val).eval().eval_f64().unwrap_or(f64::NAN);
                        (x, y)
                    })
                    .collect();
                crate::plotting::data_export::DataTable::from_points("x", "f(x)", &values)
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper: parse complex evalf strings
// ═══════════════════════════════════════════════════════════════════════════

/// Convert an `f64` to a `(numerator, denominator)` rational approximation.
///
/// Uses a denominator of 10^9 for up to ~9 digits of decimal precision,
/// then reduces by the GCD. This is used as a fallback when `compile()`
/// fails and we need to substitute numeric values symbolically.
fn f64_to_rational_approx(x: f64) -> (i64, i64) {
    if x == 0.0 {
        return (0, 1);
    }
    if !x.is_finite() {
        return (if x > 0.0 { i64::MAX } else { i64::MIN }, 1);
    }
    // If the value is very close to an integer, just return it.
    let rounded = x.round();
    if (x - rounded).abs() < 1e-12 && rounded.abs() < i64::MAX as f64 {
        return (rounded as i64, 1);
    }
    let denom: i64 = 1_000_000_000; // 10^9
    let numer = (x * denom as f64).round() as i64;
    let g = gcd_i64(numer.unsigned_abs(), denom as u64) as i64;
    (numer / g, denom / g)
}

/// Simple GCD for unsigned 64-bit integers (Euclidean algorithm).
fn gcd_i64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a.max(1)
}

/// Parse the string output of `evalf()` into a complex (f64, f64) pair.
///
/// Handles formats:
/// - `"3.14"` → `(3.14, 0.0)`
/// - `"2.5*I"` → `(0.0, 2.5)`
/// - `"-0.866*I"` → `(0.0, -0.866)`
/// - `"1.5 + 2.3*I"` → `(1.5, 2.3)`
/// - `"1.5 - 2.3*I"` → `(1.5, -2.3)`
/// - `"-1 + 2*I"` → `(-1.0, 2.0)`
/// - `"I"` → `(0.0, 1.0)`
/// - `"-I"` → `(0.0, -1.0)`
fn parse_complex_evalf_string(s: &str) -> Result<(f64, f64), SymplexError> {
    let s = s.trim();

    // Try pure real first
    if let Ok(re) = s.parse::<f64>() {
        return Ok((re, 0.0));
    }

    // Pure imaginary: "I", "-I", "2.5*I", "-0.866*I"
    if s == "I" || s == "i" {
        return Ok((0.0, 1.0));
    }
    if s == "-I" || s == "-i" {
        return Ok((0.0, -1.0));
    }
    if let Some(coeff) = s.strip_suffix("*I").or_else(|| s.strip_suffix("*i"))
        && let Ok(im) = coeff.parse::<f64>()
    {
        return Ok((0.0, im));
    }

    // Complex: "a + b*I" or "a - b*I"
    // Find the last '+' or '-' that separates real and imaginary parts
    // (not at position 0, which would be a negative sign on the real part)
    let bytes = s.as_bytes();
    let mut split_pos = None;
    let mut split_is_minus = false;
    for i in (1..bytes.len()).rev() {
        if (bytes[i] == b'+' || bytes[i] == b'-')
            && (bytes[i - 1] == b' ' || bytes[i - 1].is_ascii_digit())
        {
            split_pos = Some(i);
            split_is_minus = bytes[i] == b'-';
            break;
        }
    }

    if let Some(pos) = split_pos {
        let re_str = s[..pos].trim();
        let im_part = s[pos + 1..].trim();
        let im_str = im_part
            .trim_end_matches("*I")
            .trim_end_matches("*i")
            .trim_end_matches('I')
            .trim_end_matches('i')
            .trim();

        let re = re_str.parse::<f64>().map_err(|e| {
            SymplexError::NotImplemented(format!("could not parse real part '{}': {}", re_str, e))
        })?;

        let im_val = if im_str.is_empty() {
            1.0 // just "I" after the +/-
        } else {
            im_str.parse::<f64>().map_err(|e| {
                SymplexError::NotImplemented(format!(
                    "could not parse imaginary part '{}': {}",
                    im_str, e
                ))
            })?
        };

        let im = if split_is_minus { -im_val } else { im_val };
        return Ok((re, im));
    }

    Err(SymplexError::NotImplemented(format!(
        "could not parse '{}' as complex number",
        s
    )))
}

#[cfg(test)]
mod evalf_complex_tests {
    #[test]
    fn eval_complex64_pure_real() {
        let ctx = crate::api::context::Context::new();
        let x = ctx.symbol("x");
        let expr = &x.powi(2) + 1;
        let at_2 = expr.subs(&x, &ctx.int(2));
        let (re, im) = at_2.eval_complex64().unwrap();
        assert!((re - 5.0).abs() < 1e-10);
        assert!(im.abs() < 1e-10);
    }

    #[test]
    fn eval_complex64_pure_imaginary() {
        let ctx = crate::api::context::Context::new();
        let i = ctx.i_unit();
        let (re, im) = i.eval_complex64().unwrap();
        assert!(re.abs() < 1e-10);
        assert!((im - 1.0).abs() < 1e-10);
    }

    #[test]
    fn eval_complex64_mixed() {
        let ctx = crate::api::context::Context::new();
        let expr = &ctx.int(3) + &(&ctx.int(4) * &ctx.i_unit());
        let (re, im) = expr.eval_complex64().unwrap();
        assert!((re - 3.0).abs() < 1e-10);
        assert!((im - 4.0).abs() < 1e-10);
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
