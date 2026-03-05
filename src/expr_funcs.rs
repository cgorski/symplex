//! Numeric expression methods — math functions, calculus, algebra, solving.
//!
//! This module contains all `impl Expr<Numeric>` method blocks.
//! The type definitions live in `expr.rs`.

use tracing::debug_span;

use crate::assumptions::{Assumption, Props};
use crate::errors::SymplexError;
use crate::expr::{BoolEx, Ex, Expr, Numeric, SetEx, SetValued};

// ═══════════════════════════════════════════════════════════════════════════
// impl Expr<Numeric> — Numeric-specific methods
// ═══════════════════════════════════════════════════════════════════════════

impl Expr<Numeric> {
    /// The additive identity (0) in the global default context.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let z = Ex::zero();
    /// assert_eq!(format!("{z}"), "0");
    /// assert!(z.is_zero_structural());
    /// ```
    #[must_use]
    pub fn zero() -> Ex {
        crate::int(0)
    }

    /// The multiplicative identity (1) in the global default context.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let o = Ex::one();
    /// assert_eq!(format!("{o}"), "1");
    /// assert!(o.is_one_structural());
    /// ```
    #[must_use]
    pub fn one() -> Ex {
        crate::int(1)
    }

    // ── Math functions ─────────────────────────────────────────────

    /// Raise to a symbolic power: `self ^ exp`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn pow(&self, exp: &Ex) -> Ex {
        let id = self.inner.write().arena.pow(self.id, exp.id);
        self.wrap(id)
    }

    /// Raise to an integer power: `self ^ n`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn powi(&self, n: i64) -> Ex {
        let mut inner = self.inner.write();
        let exp = inner.arena.int(n);
        let id = inner.arena.pow(self.id, exp);
        drop(inner);
        self.wrap(id)
    }

    /// Sine: `sin(self)`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn sin(&self) -> Ex {
        let id = self.inner.write().arena.sin(self.id);
        self.wrap(id)
    }

    /// Cosine: `cos(self)`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn cos(&self) -> Ex {
        let id = self.inner.write().arena.cos(self.id);
        self.wrap(id)
    }

    /// Tangent: `tan(self)`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn tan(&self) -> Ex {
        let id = self.inner.write().arena.tan(self.id);
        self.wrap(id)
    }

    /// Natural exponential: `e^self`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn exp(&self) -> Ex {
        let id = self.inner.write().arena.exp(self.id);
        self.wrap(id)
    }

    /// Natural logarithm: `ln(self)`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn ln(&self) -> Ex {
        let id = self.inner.write().arena.ln(self.id);
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
        let id = self.inner.write().arena.sqrt(self.id);
        self.wrap(id)
    }

    /// Cube root: `∛self`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn cbrt(&self) -> Ex {
        let id = self.inner.write().arena.cbrt(self.id);
        self.wrap(id)
    }

    /// Nth root: `self^(1/n)`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn nthroot(&self, n: i64) -> Ex {
        let mut inner = self.inner.write();
        let frac = inner.arena.rational(1, n);
        let id = inner.arena.pow(self.id, frac);
        drop(inner);
        self.wrap(id)
    }

    /// Absolute value (or complex modulus): `|self|`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn abs(&self) -> Ex {
        let id = self.inner.write().arena.abs(self.id);
        self.wrap(id)
    }

    /// Inverse sine (arcsin): `asin(self)`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn asin(&self) -> Ex {
        let id = self.inner.write().arena.asin(self.id);
        self.wrap(id)
    }

    /// Inverse cosine (arccos): `acos(self)`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn acos(&self) -> Ex {
        let id = self.inner.write().arena.acos(self.id);
        self.wrap(id)
    }

    /// Inverse tangent (arctan): `atan(self)`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn atan(&self) -> Ex {
        let id = self.inner.write().arena.atan(self.id);
        self.wrap(id)
    }

    /// Two-argument arctangent: `atan2(y, x)`.
    ///
    /// Returns the angle in (-π, π] between the positive x-axis and the
    /// point (x, y). Correctly handles all four quadrants.
    #[must_use]
    pub fn atan2(&self, x: &Ex) -> Ex {
        let id = self.inner.write().arena.atan2(self.id, x.id);
        self.wrap(id)
    }

    /// Hyperbolic sine: `sinh(self)`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn sinh(&self) -> Ex {
        let id = self.inner.write().arena.sinh(self.id);
        self.wrap(id)
    }

    /// Hyperbolic cosine: `cosh(self)`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn cosh(&self) -> Ex {
        let id = self.inner.write().arena.cosh(self.id);
        self.wrap(id)
    }

    /// Hyperbolic tangent: `tanh(self)`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn tanh(&self) -> Ex {
        let id = self.inner.write().arena.tanh(self.id);
        self.wrap(id)
    }

    /// Inverse hyperbolic sine: `asinh(self)`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn asinh(&self) -> Ex {
        let id = self.inner.write().arena.asinh(self.id);
        self.wrap(id)
    }

    /// Inverse hyperbolic cosine: `acosh(self)`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn acosh(&self) -> Ex {
        let id = self.inner.write().arena.acosh(self.id);
        self.wrap(id)
    }

    /// Inverse hyperbolic tangent: `atanh(self)`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn atanh(&self) -> Ex {
        let id = self.inner.write().arena.atanh(self.id);
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
        let id = self.inner.write().arena.sign(self.id);
        self.wrap(id)
    }

    /// Floor function: `⌊self⌋` (greatest integer ≤ self).
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn floor(&self) -> Ex {
        let id = self.inner.write().arena.floor(self.id);
        self.wrap(id)
    }

    /// Ceiling function: `⌈self⌉` (least integer ≥ self).
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn ceiling(&self) -> Ex {
        let id = self.inner.write().arena.ceiling(self.id);
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
        let mut inner = self.inner.write();
        let ids: smallvec::SmallVec<[crate::node::ExprId; 4]> =
            smallvec::smallvec![self.id, other.id];
        let id = inner.arena.intern(crate::node::ExprNode::Min(ids));
        drop(inner);
        self.wrap(id)
    }

    /// Binary maximum: `max(self, other)`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn max_with(&self, other: &Ex) -> Ex {
        let mut inner = self.inner.write();
        let ids: smallvec::SmallVec<[crate::node::ExprId; 4]> =
            smallvec::smallvec![self.id, other.id];
        let id = inner.arena.intern(crate::node::ExprNode::Max(ids));
        drop(inner);
        self.wrap(id)
    }

    /// N-ary minimum of a collection of expressions.
    pub fn min_of(ctx: &crate::context::Context, exprs: impl IntoIterator<Item = Ex>) -> Ex {
        let items: Vec<Ex> = exprs.into_iter().collect();
        if items.is_empty() {
            return ctx.infinity();
        }
        let mut inner = items[0].inner.write();
        let ids: smallvec::SmallVec<[crate::node::ExprId; 4]> =
            items.iter().map(|e| e.id).collect();
        let id = inner.arena.intern(crate::node::ExprNode::Min(ids));
        drop(inner);
        items[0].wrap(id)
    }

    /// N-ary maximum of a collection of expressions.
    pub fn max_of(ctx: &crate::context::Context, exprs: impl IntoIterator<Item = Ex>) -> Ex {
        let items: Vec<Ex> = exprs.into_iter().collect();
        if items.is_empty() {
            return ctx.neg_infinity();
        }
        let mut inner = items[0].inner.write();
        let ids: smallvec::SmallVec<[crate::node::ExprId; 4]> =
            items.iter().map(|e| e.id).collect();
        let id = inner.arena.intern(crate::node::ExprNode::Max(ids));
        drop(inner);
        items[0].wrap(id)
    }

    /// Symbolic summation: `Sum(body, var=lower..upper)`.
    ///
    /// When evaluated (`.eval()`), if `lower` and `upper` are concrete integers,
    /// the sum is computed by substituting each integer value for `var` in `body`.
    pub fn symbolic_sum(body: &Ex, var: &Ex, lower: &Ex, upper: &Ex) -> Ex {
        let mut inner = body.inner.write();
        let id = inner.arena.intern(crate::node::ExprNode::Sum(
            body.id, var.id, lower.id, upper.id,
        ));
        drop(inner);
        body.wrap(id)
    }

    /// Symbolic product: `Product(body, var=lower..upper)`.
    ///
    /// When evaluated (`.eval()`), if `lower` and `upper` are concrete integers,
    /// the product is computed by substituting each integer value for `var` in `body`.
    pub fn symbolic_product(body: &Ex, var: &Ex, lower: &Ex, upper: &Ex) -> Ex {
        let mut inner = body.inner.write();
        let id = inner.arena.intern(crate::node::ExprNode::Product_(
            body.id, var.id, lower.id, upper.id,
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
        let _span = debug_span!("is_convergent").entered();
        let mut inner = self.inner.write();
        inner.arena.is_convergent_expr(self.id, var.id)
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
        let node = inner.arena.node(self.id).clone();
        if let crate::node::ExprNode::Sum(body, var, lower, upper) = node
            && let Some(closed) = inner.arena.eval_sum_symbolic_expr(body, var, lower, upper)
        {
            let result = crate::eval::eval(&mut inner.arena, closed);
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
        let (re, _im) = self.inner.write().arena.as_real_imag_expr(self.id);
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
        let (_re, im) = self.inner.write().arena.as_real_imag_expr(self.id);
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
    /// let result = symplex::int(5).gamma().eval();
    /// assert_eq!(format!("{result}"), "24");
    /// ```
    #[must_use]
    pub fn gamma(&self) -> Ex {
        let id = self.inner.write().arena.gamma(self.id);
        self.wrap(id)
    }

    /// Log-gamma function: ln(Γ(self)).
    ///
    /// For positive integer arguments, `.eval()` computes `ln((n-1)!)`.
    #[must_use]
    pub fn log_gamma(&self) -> Ex {
        let id = self.inner.write().arena.log_gamma(self.id);
        self.wrap(id)
    }

    /// Digamma function: ψ(self) = Γ'(self)/Γ(self).
    #[must_use]
    pub fn digamma(&self) -> Ex {
        let id = self.inner.write().arena.digamma(self.id);
        self.wrap(id)
    }

    /// Error function: erf(self) = 2/√π ∫₀ˢᵉˡᶠ e^(-t²) dt.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let result = symplex::int(0).erf().eval();
    /// assert_eq!(format!("{result}"), "0");
    /// ```
    #[must_use]
    pub fn erf(&self) -> Ex {
        let id = self.inner.write().arena.erf(self.id);
        self.wrap(id)
    }

    /// Complementary error function: erfc(self) = 1 - erf(self).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let result = symplex::int(0).erfc().eval();
    /// assert_eq!(format!("{result}"), "1");
    /// ```
    #[must_use]
    pub fn erfc(&self) -> Ex {
        let id = self.inner.write().arena.erfc(self.id);
        self.wrap(id)
    }

    /// Beta function: B(self, other) = Γ(self)Γ(other)/Γ(self+other).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let result = symplex::int(2).beta(&symplex::int(3)).eval();
    /// assert_eq!(format!("{result}"), "1/12");
    /// ```
    #[must_use]
    pub fn beta(&self, other: &Ex) -> Ex {
        let id = self.inner.write().arena.beta(self.id, other.id);
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
    /// let result = symplex::int(5).factorial().eval();
    /// assert_eq!(format!("{result}"), "120");
    /// ```
    #[must_use]
    pub fn factorial(&self) -> Ex {
        let id = self.inner.write().arena.factorial(self.id);
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
    /// let result = symplex::int(10).binomial(&symplex::int(3)).eval();
    /// assert_eq!(format!("{result}"), "120");
    /// ```
    #[must_use]
    pub fn binomial(&self, k: &Ex) -> Ex {
        let id = self.inner.write().arena.binomial(self.id, k.id);
        self.wrap(id)
    }

    // ── Combinatorial functions (Apply-based) ──────────────────────

    /// Double factorial: `self!!`.
    ///
    /// For non-negative integer arguments, `.eval()` computes the exact value.
    /// `0!! = 1`, `1!! = 1`, `(-1)!! = 1`.
    #[must_use]
    pub fn factorial2(&self) -> Ex {
        let id = self.inner.write().arena.factorial2(self.id);
        self.wrap(id)
    }

    /// Subfactorial (derangement count): `!self`.
    ///
    /// For non-negative integer arguments, `.eval()` computes the exact value.
    #[must_use]
    pub fn subfactorial(&self) -> Ex {
        let id = self.inner.write().arena.subfactorial(self.id);
        self.wrap(id)
    }

    /// Rising factorial (Pochhammer symbol): `(self)_n`.
    ///
    /// `rising_factorial(x, n) = x * (x+1) * ... * (x+n-1)`.
    #[must_use]
    pub fn rising_factorial(&self, n: &Ex) -> Ex {
        let id = self.inner.write().arena.rising_factorial(self.id, n.id);
        self.wrap(id)
    }

    /// Falling factorial: `self^(n) = self * (self-1) * ... * (self-n+1)`.
    #[must_use]
    pub fn falling_factorial(&self, n: &Ex) -> Ex {
        let id = self.inner.write().arena.falling_factorial(self.id, n.id);
        self.wrap(id)
    }

    /// Fibonacci number: `F(self)`.
    ///
    /// For non-negative integer arguments, `.eval()` computes the exact value.
    /// `F(0) = 0`, `F(1) = 1`, `F(n) = F(n-1) + F(n-2)`.
    #[must_use]
    pub fn fibonacci(&self) -> Ex {
        let id = self.inner.write().arena.fibonacci(self.id);
        self.wrap(id)
    }

    /// Lucas number: `L(self)`.
    ///
    /// For non-negative integer arguments, `.eval()` computes the exact value.
    /// `L(0) = 2`, `L(1) = 1`, `L(n) = L(n-1) + L(n-2)`.
    #[must_use]
    pub fn lucas(&self) -> Ex {
        let id = self.inner.write().arena.lucas(self.id);
        self.wrap(id)
    }

    /// Bernoulli number: `B(self)`.
    ///
    /// For non-negative integer arguments, `.eval()` computes the exact value.
    /// `B(0) = 1`, `B(1) = -1/2`, `B(2) = 1/6`.
    #[must_use]
    pub fn bernoulli_number(&self) -> Ex {
        let id = self.inner.write().arena.bernoulli_number(self.id);
        self.wrap(id)
    }

    /// Harmonic number: `H(self) = 1 + 1/2 + ... + 1/self`.
    ///
    /// For non-negative integer arguments, `.eval()` computes the exact value.
    /// `H(0) = 0`.
    #[must_use]
    pub fn harmonic(&self) -> Ex {
        let id = self.inner.write().arena.harmonic(self.id);
        self.wrap(id)
    }

    /// Catalan number: `C(self) = (2n)! / ((n+1)! * n!)`.
    ///
    /// For non-negative integer arguments, `.eval()` computes the exact value.
    #[must_use]
    pub fn catalan_number(&self) -> Ex {
        let id = self.inner.write().arena.catalan_number(self.id);
        self.wrap(id)
    }

    /// Bell number: `B(self)`.
    ///
    /// For non-negative integer arguments, `.eval()` computes the exact value.
    /// `B(0) = 1`, `B(1) = 1`, `B(2) = 2`, `B(3) = 5`.
    #[must_use]
    pub fn bell(&self) -> Ex {
        let id = self.inner.write().arena.bell(self.id);
        self.wrap(id)
    }

    /// Euler number: `E(self)`.
    ///
    /// For non-negative integer arguments, `.eval()` computes the exact value.
    /// Odd indices are 0. `E(0) = 1`, `E(2) = -1`, `E(4) = 5`.
    #[must_use]
    pub fn euler_number(&self) -> Ex {
        let id = self.inner.write().arena.euler_number(self.id);
        self.wrap(id)
    }

    // ── Special functions (Apply-based) ────────────────────────────

    /// Heaviside step function: 0 for x<0, 1/2 for x=0, 1 for x>0.
    #[must_use]
    pub fn heaviside(&self) -> Ex {
        let id = self.inner.write().arena.heaviside(self.id);
        self.wrap(id)
    }

    /// Dirac delta distribution: 0 for x≠0, symbolic at x=0.
    #[must_use]
    pub fn dirac_delta(&self) -> Ex {
        let id = self.inner.write().arena.dirac_delta(self.id);
        self.wrap(id)
    }

    /// Lambert W function (principal branch): W(x)·exp(W(x)) = x.
    #[must_use]
    pub fn lambertw(&self) -> Ex {
        let id = self.inner.write().arena.lambertw(self.id);
        self.wrap(id)
    }

    // ── Relational operators (return BoolEx) ───────────────────────

    /// Greater than: `self > other`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn gt(&self, other: &Ex) -> BoolEx {
        let id = self.inner.write().arena.gt(self.id, other.id);
        self.wrap_as(id)
    }

    /// Greater than or equal: `self >= other`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn ge(&self, other: &Ex) -> BoolEx {
        let id = self.inner.write().arena.ge(self.id, other.id);
        self.wrap_as(id)
    }

    /// Less than: `self < other` (implemented as `other > self`).
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn lt(&self, other: &Ex) -> BoolEx {
        let id = self.inner.write().arena.gt(other.id, self.id);
        self.wrap_as(id)
    }

    /// Less than or equal: `self <= other` (implemented as `other >= self`).
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn le(&self, other: &Ex) -> BoolEx {
        let id = self.inner.write().arena.ge(other.id, self.id);
        self.wrap_as(id)
    }

    /// Mathematical equality test (boolean-valued): `self == other`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn eq_expr(&self, other: &Ex) -> BoolEx {
        let id = self.inner.write().arena.eq_(self.id, other.id);
        self.wrap_as(id)
    }

    /// Not-equal test (boolean-valued): `self != other`.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn ne_expr(&self, other: &Ex) -> BoolEx {
        let id = self.inner.write().arena.ne_(self.id, other.id);
        self.wrap_as(id)
    }

    /// Piecewise function from `(value, condition)` pairs.
    ///
    /// Returns the value of the first pair whose condition is true.
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn piecewise(pairs: &[(&Ex, &BoolEx)]) -> Ex {
        if pairs.is_empty() {
            return Ex::zero();
        }
        let first = pairs[0].0;
        let arena_pairs: Vec<(crate::node::ExprId, crate::node::ExprId)> =
            pairs.iter().map(|(v, c)| (v.id, c.id)).collect();
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
            .query(&inner.arena, self.id, Props::POSITIVE)
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
        if inner.arena.is_zero_structural(self.id) {
            return Some(true);
        }
        // Layer 2: assumption system.
        inner
            .assumptions
            .lock()
            .query(&inner.arena, self.id, Props::ZERO)
    }

    /// Query any mathematical property via [`Props`].
    ///
    /// Returns `Some(true)` if provably true, `Some(false)` if provably
    /// false, or `None` if unknown.
    #[must_use]
    pub fn query(&self, prop: Props) -> Option<bool> {
        let inner = self.inner.read();
        inner.assumptions.lock().query(&inner.arena, self.id, prop)
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
    /// let t = symplex::var("t")
    ///     .assume(Assumption::Positive)
    ///     .assume(Assumption::Real);
    /// assert_eq!(t.is_positive(), Some(true));
    /// assert_eq!(t.is_real(), Some(true));
    /// ```
    pub fn assume(self, assumption: Assumption) -> Ex {
        use crate::node::ExprNode;
        let mut inner = self.inner.write();
        if let ExprNode::Symbol(sid) = inner.arena.node(self.id) {
            let sid = *sid;
            let (prop, value) = assumption.to_prop_value();
            let mut a = inner.arena.symbol_assumptions(sid);
            if value {
                a.assert_true(prop);
            } else {
                a.assert_false(prop);
            }
            inner.arena.set_symbol_assumptions(sid, a);
            inner.assumptions.lock().set_symbol_assumptions(self.id, a);
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
        if self.id == other.id && self.ctx_id == other.ctx_id {
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
        let _span = debug_span!("diff", expr = ?self.id, var = ?var.id).entered();
        let id = self.inner.write().arena.diff_wrt(self.id, var.id);
        self.wrap(id)
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
    /// let x = symplex::var("x");
    /// let y = symplex::var("y");
    /// let dy_dx = y.formal_diff(&x);
    /// let s = format!("{dy_dx}");
    /// assert!(s.contains("Derivative") || s.contains("d/d"), "got: {s}");
    /// ```
    #[must_use]
    pub fn formal_diff(&self, var: &Ex) -> Ex {
        let id = self
            .inner
            .write()
            .arena
            .intern(crate::node::ExprNode::Derivative(self.id, var.id));
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
    /// let x = symplex::var("x");
    /// let y = symplex::var("y");
    /// let expr = &x.powi(2) + &y.powi(2);
    /// let result = expr.diff_with_dependent(&x, &[&y]);
    /// // d/dx(x² + y²) with y depending on x = 2x + 2y·dy/dx
    /// ```
    pub fn diff_with_dependent(&self, var: &Ex, dependent_vars: &[&Ex]) -> Ex {
        let mut deps = rustc_hash::FxHashSet::default();
        for dep in dependent_vars {
            deps.insert(dep.id);
        }
        let id = {
            let mut guard = self.inner.write();
            crate::diff::diff_with_deps(&mut guard.arena, self.id, var.id, &deps)
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
            crate::subs::eval_derivatives(&mut guard.arena, self.id)
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
        let _span = debug_span!("integrate", expr = ?self.id, var = ?var.id).entered();
        let id = self.inner.write().arena.integrate_expr(self.id, var.id);
        self.wrap(id)
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
    /// Returns the original expression unchanged if expansion around
    /// the point is not possible (e.g., pole at the expansion point).
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
    /// let s = expr.series(&x, &zero, 4).unwrap();
    /// let expanded = s.expand().eval();
    /// let result = format!("{expanded}");
    /// assert!(result.contains("x"), "should have x term: {result}");
    /// ```
    #[must_use = "returns the series expansion; does not modify in place"]
    pub fn series(&self, var: &Ex, point: &Ex, order: u32) -> Result<Ex, SymplexError> {
        let _span = debug_span!("series", expr = ?self.id, order = order).entered();
        let id = self
            .inner
            .write()
            .arena
            .series_expr(self.id, var.id, point.id, order)?;
        Ok(self.wrap(id))
    }

    /// Compute a Taylor series, returning the expression unchanged on failure.
    ///
    /// Convenience wrapper around [`series`](Ex::series).
    #[must_use = "returns the series expansion; does not modify in place"]
    pub fn series_or_self(&self, var: &Ex, point: &Ex, order: u32) -> Ex {
        self.series(var, point, order)
            .unwrap_or_else(|_| self.clone())
    }

    /// Compute the Maclaurin series (Taylor series around 0) to the
    /// given `order`.
    ///
    /// This is a convenience shorthand for `self.series(var, &zero, order)`
    /// that avoids needing to construct a zero expression manually.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let x = symplex::var("x");
    /// let s = x.sin().maclaurin(&x, 4).unwrap();
    /// let result = s.expand().eval();
    /// let text = format!("{result}");
    /// assert!(text.contains("x"), "should have x term: {text}");
    /// ```
    #[must_use = "returns the series expansion; does not modify in place"]
    pub fn maclaurin(&self, var: &Ex, order: u32) -> Result<Ex, SymplexError> {
        let mut inner = self.inner.write();
        let zero = inner.arena.zero;
        let id = inner.arena.series_expr(self.id, var.id, zero, order)?;
        drop(inner);
        Ok(self.wrap(id))
    }

    /// Compute a Maclaurin series, returning the expression unchanged on failure.
    ///
    /// Convenience wrapper around [`maclaurin`](Ex::maclaurin).
    #[must_use = "returns the series expansion; does not modify in place"]
    pub fn maclaurin_or_self(&self, var: &Ex, order: u32) -> Ex {
        self.maclaurin(var, order).unwrap_or_else(|_| self.clone())
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
    /// let result = f.residue(&x, &ctx.int(0)).unwrap();
    /// assert_eq!(format!("{result}"), "1");
    /// ```
    #[must_use = "returns the residue value; does not modify in place"]
    pub fn residue(&self, var: &Ex, point: &Ex) -> Result<Ex, SymplexError> {
        let _span = debug_span!("residue", expr = ?self.id, var = ?var.id).entered();
        let id = self
            .inner
            .write()
            .arena
            .residue_expr(self.id, var.id, point.id)?;
        Ok(self.wrap(id))
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
        let _span = debug_span!("fourier_series", expr = ?self.id, var = ?var.id).entered();
        let id = self
            .inner
            .write()
            .arena
            .fourier_series_expr(self.id, var.id, n_terms);
        self.wrap(id)
    }

    /// Compute the Laplace transform L{self}(s) with respect to time variable `t`.
    ///
    /// Uses a table-based approach supporting constants, polynomials in `t`,
    /// exponentials, trigonometric and hyperbolic functions, plus linearity
    /// and the frequency-shift property.
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
    /// let result = (&t * 2).exp().laplace(&t, &s).unwrap();
    /// ```
    #[must_use = "returns the Laplace transform; does not modify in place"]
    pub fn laplace(&self, t: &Ex, s: &Ex) -> Result<Ex, SymplexError> {
        let _span = debug_span!("laplace", expr = ?self.id, t = ?t.id, s = ?s.id).entered();
        let id = self
            .inner
            .write()
            .arena
            .laplace_transform_expr(self.id, t.id, s.id)?;
        Ok(self.wrap(id))
    }

    /// Compute the inverse Laplace transform L⁻¹{self}(t) with respect to
    /// frequency variable `s`.
    ///
    /// Uses partial fraction decomposition followed by table lookup for
    /// each term.
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
    /// let result = (&ctx.int(1) / &s).inverse_laplace(&s, &t).unwrap();
    /// ```
    #[must_use = "returns the inverse Laplace transform; does not modify in place"]
    pub fn inverse_laplace(&self, s: &Ex, t: &Ex) -> Result<Ex, SymplexError> {
        let _span = debug_span!("inverse_laplace", expr = ?self.id, s = ?s.id, t = ?t.id).entered();
        let id = self
            .inner
            .write()
            .arena
            .inverse_laplace_transform_expr(self.id, s.id, t.id)?;
        Ok(self.wrap(id))
    }

    /// Compute the limit of this expression as `var` approaches `point`.
    ///
    /// Uses direct substitution, L'Hôpital's rule (for 0/0 and ∞/∞),
    /// and series expansion as fallbacks. Returns the expression
    /// unchanged if the limit cannot be determined.
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
    /// let result = expr.limit(&x, &ctx.int(0)).unwrap();
    /// assert_eq!(format!("{result}"), "1");
    /// ```
    #[must_use = "returns the limit value; does not modify in place"]
    pub fn limit(&self, var: &Ex, point: &Ex) -> Result<Ex, SymplexError> {
        let _span = debug_span!("limit", expr = ?self.id, var = ?var.id).entered();
        let id = self
            .inner
            .write()
            .arena
            .limit_expr(self.id, var.id, point.id)?;
        Ok(self.wrap(id))
    }

    /// Compute a limit, returning the expression unchanged on failure.
    ///
    /// Convenience wrapper around [`limit`](Ex::limit).
    #[must_use = "returns the limit value; does not modify in place"]
    pub fn limit_or_self(&self, var: &Ex, point: &Ex) -> Ex {
        self.limit(var, point).unwrap_or_else(|_| self.clone())
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
        let id = self.inner.write().arena.expand_trig_expr(self.id);
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
        let id = self.inner.write().arena.expand_log_expr(self.id);
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
    /// let combined = expr.logcombine();
    /// let s = format!("{combined}");
    /// assert!(s.contains("ln"), "should combine logs: {s}");
    /// ```
    #[must_use = "returns the combined form; does not modify in place"]
    pub fn logcombine(&self) -> Ex {
        let id = self.inner.write().arena.log_combine_expr(self.id);
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
        let id = self.inner.write().arena.trig_combine_expr(self.id);
        self.wrap(id)
    }

    /// Dedicated trigonometric simplification.
    ///
    /// Goes beyond the pattern-based rules in [`simplify`](Self::simplify)
    /// by trying exhaustive Pythagorean replacements (sin²→1−cos² and
    /// cos²→1−sin²) and picking the result with the fewest operations.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let x = symplex::var("x");
    /// let expr = &x.sin().powi(2) + &x.cos().powi(2);
    /// assert_eq!(format!("{}", expr.trigsimp()), "1");
    /// ```
    #[must_use = "returns the simplified form; does not modify in place"]
    pub fn trigsimp(&self) -> Ex {
        let id = self.inner.write().arena.trigsimp_expr(self.id);
        self.wrap(id)
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
    /// let result = symplex::int(5).factorial().eval();
    /// let four_fact = symplex::int(4).factorial().eval();
    /// let ratio = (&result / &four_fact).combsimp();
    /// assert_eq!(format!("{ratio}"), "5");
    /// ```
    #[must_use = "returns the simplified form; does not modify in place"]
    pub fn combsimp(&self) -> Ex {
        let id = self.inner.write().arena.combsimp_expr(self.id);
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
    /// let expr = symplex::rational(333333, 1000000);
    /// let result = expr.nsimplify(1e-5);
    /// assert_eq!(format!("{result}"), "1/3");
    /// ```
    #[must_use = "returns the simplified form; does not modify in place"]
    pub fn nsimplify(&self, tolerance: f64) -> Ex {
        let id = self.inner.write().arena.nsimplify_expr(self.id, tolerance);
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
    /// let result = expr.powsimp();
    /// let s = format!("{result}");
    /// assert!(s.contains("a + b") || s.contains("b + a"), "should combine: {s}");
    /// ```
    #[must_use = "returns the simplified form; does not modify in place"]
    pub fn powsimp(&self) -> Ex {
        let id = self.inner.write().arena.powsimp_expr(self.id);
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
    /// let x = symplex::var("x");
    /// let result = x.sin().rewrite_as_exp();
    /// let s = format!("{result}");
    /// assert!(s.contains("exp") || s.contains("E"), "should contain exponentials: {s}");
    /// ```
    #[must_use = "returns the rewritten form; does not modify in place"]
    pub fn rewrite_as_exp(&self) -> Ex {
        let id = self.inner.write().arena.rewrite_as_exp_expr(self.id);
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
    /// let x = symplex::var("x");
    /// let i = symplex::i_unit();
    /// let expr = (&i * &x).exp();
    /// let result = expr.rewrite_as_trig();
    /// let s = format!("{result}");
    /// assert!(s.contains("cos") && s.contains("sin"), "should contain trig: {s}");
    /// ```
    #[must_use = "returns the rewritten form; does not modify in place"]
    pub fn rewrite_as_trig(&self) -> Ex {
        let id = self.inner.write().arena.rewrite_as_trig_expr(self.id);
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
        let id = self.inner.write().arena.collect_expr(self.id, var.id);
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
        let id = self.inner.write().arena.together_expr(self.id);
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
        let _span = debug_span!("cancel", expr = ?self.id, var = ?var.id).entered();
        let id = self.inner.write().arena.cancel_expr(self.id, var.id);
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
    /// // 1/x + 1/x → 2/x after ratsimp
    /// let expr = &x.powi(-1) + &x.powi(-1);
    /// let simplified = expr.ratsimp();
    /// let s = format!("{simplified}");
    /// assert!(s.contains("2"), "ratsimp should combine: {s}");
    /// ```
    #[must_use = "returns the simplified form; does not modify in place"]
    pub fn ratsimp(&self) -> Ex {
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
    /// let decomposed = expr.apart(&x);
    /// let s = format!("{decomposed}");
    /// // Should be decomposed into simpler fractions
    /// assert!(s != format!("{expr}") || s.contains("1/"), "should decompose: {s}");
    /// ```
    #[must_use = "returns the decomposed form; does not modify in place"]
    pub fn apart(&self, var: &Ex) -> Ex {
        let id = self.inner.write().arena.apart_expr(self.id, var.id);
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
        let _span = debug_span!("factor", expr = ?self.id, var = ?var.id).entered();
        let id = self.inner.write().arena.factor_expr(self.id, var.id);
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
        let (gcd_id, inner_id) = self.inner.write().arena.factor_terms_pair_expr(self.id);
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
        let id = self.inner.write().arena.rationalize_denom_expr(self.id);
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
    /// let groups = expr.separatevars(&[&x, &y]);
    /// assert!(groups.len() >= 2, "should separate into multiple groups");
    /// ```
    #[must_use]
    pub fn separatevars(&self, vars: &[&Ex]) -> Vec<(Vec<Ex>, Ex)> {
        let var_ids: Vec<crate::node::ExprId> = vars.iter().map(|v| v.id).collect();
        let raw = self
            .inner
            .write()
            .arena
            .separatevars_expr(self.id, &var_ids);
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
    /// let x = symplex::var("x");
    /// assert_eq!((&x.powi(3) + &x + 1).degree(&x), Some(3));
    /// assert_eq!(x.sin().degree(&x), None);
    /// ```
    #[must_use]
    pub fn degree(&self, var: &Ex) -> Option<usize> {
        let inner = self.inner.read();
        inner.arena.degree_of(self.id, var.id)
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
        let mut inner = self.inner.write();
        let ids = inner.arena.coefficients_of(self.id, var.id)?;
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
        let (n, d) = inner.arena.as_numer_denom_expr(self.id);
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
    /// let x = symplex::var("x");
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
        let mut inner = self.inner.write();
        let id = inner.arena.poly_gcd_expr(self.id, other.id, var.id)?;
        drop(inner);
        Some(self.wrap(id))
    }

    /// Compute the polynomial LCM of `self` and `other` with respect to `var`.
    ///
    /// Returns `None` if either expression is not polynomial in `var`.
    #[must_use]
    pub fn poly_lcm(&self, other: &Ex, var: &Ex) -> Option<Ex> {
        let mut inner = self.inner.write();
        let id = inner.arena.poly_lcm_expr(self.id, other.id, var.id)?;
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
        let _span = debug_span!("solve", expr = ?self.id, var = ?var.id).entered();
        let mut inner = self.inner.write();
        // Check if the expression is polynomial in var.
        let poly = crate::polybridge::expr_to_poly(&inner.arena, self.id, var.id);
        if poly.is_none() {
            return Err(SymplexError::ComputationFailed {
                operation: "solve",
                reason: "expression is not polynomial in the given variable".into(),
            });
        }
        let solutions = inner.arena.solve_for(self.id, var.id);
        drop(inner);
        Ok(solutions
            .into_iter()
            .map(|sol| self.wrap(sol.value))
            .collect())
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
    /// let result = x.solve_gt(&x).unwrap();
    /// let s = format!("{result}");
    /// assert!(!s.contains("EmptySet"), "x > 0 should not be empty: {s}");
    /// ```
    pub fn solve_gt(&self, var: &Ex) -> Result<SetEx, SymplexError> {
        let id = self.inner.write().arena.solve_inequality_expr(
            self.id,
            var.id,
            crate::inequalities::Relation::Gt,
        )?;
        Ok(self.wrap_as::<SetValued>(id))
    }

    /// Solve `self >= 0` for `var`, returning the solution as a set.
    ///
    /// Like [`solve_gt`](Ex::solve_gt), but includes the roots themselves
    /// (where `self = 0`).
    pub fn solve_ge(&self, var: &Ex) -> Result<SetEx, SymplexError> {
        let id = self.inner.write().arena.solve_inequality_expr(
            self.id,
            var.id,
            crate::inequalities::Relation::Ge,
        )?;
        Ok(self.wrap_as::<SetValued>(id))
    }

    /// Solve `self < 0` for `var`, returning the solution as a set.
    ///
    /// Uses the sign-chart method with a strict less-than relation.
    pub fn solve_lt(&self, var: &Ex) -> Result<SetEx, SymplexError> {
        let id = self.inner.write().arena.solve_inequality_expr(
            self.id,
            var.id,
            crate::inequalities::Relation::Lt,
        )?;
        Ok(self.wrap_as::<SetValued>(id))
    }

    /// Solve `self <= 0` for `var`, returning the solution as a set.
    ///
    /// Like [`solve_lt`](Ex::solve_lt), but includes the roots themselves.
    pub fn solve_le(&self, var: &Ex) -> Result<SetEx, SymplexError> {
        let id = self.inner.write().arena.solve_inequality_expr(
            self.id,
            var.id,
            crate::inequalities::Relation::Le,
        )?;
        Ok(self.wrap_as::<SetValued>(id))
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
    /// let result = poly.solveset(&x);
    /// let s = format!("{result}");
    /// // Should contain {2, 3} or similar
    /// assert!(!s.contains("EmptySet"), "solveset: {s}");
    /// ```
    pub fn solveset(&self, var: &Ex) -> SetEx {
        let id = self.inner.write().arena.solveset_expr(self.id, var.id);
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
    /// let root = expr.nsolve(&x, 1.0, 50, 1e-12).unwrap();
    /// assert!((root - 0.7390851332).abs() < 1e-8);
    /// ```
    pub fn nsolve(
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
                        operation: "nsolve",
                        reason: format!("could not approximate x = {x} as rational"),
                    });
                }
            };
            let x_rational = {
                let mut inner = self.inner.write();
                let nid = inner.arena.intern_num(r);
                let id = inner.arena.intern(crate::node::ExprNode::Num(nid));
                drop(inner);
                self.wrap(id)
            };

            let f_val = self.subs(var, &x_rational).evalf_f64()?;
            let fp_val = deriv.subs(var, &x_rational).evalf_f64()?;

            if fp_val.abs() < 1e-30 {
                return Err(SymplexError::ComputationFailed {
                    operation: "nsolve",
                    reason: "derivative is effectively zero".into(),
                });
            }

            x -= f_val / fp_val;

            if f_val.abs() < tolerance {
                return Ok(x);
            }
        }

        Err(SymplexError::ComputationFailed {
            operation: "nsolve",
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
    /// Returns `Some((solution, constants))` where `solution` is the general
    /// solution and `constants` are the arbitrary constants (C1, C2, etc.).
    /// Returns `None` if the ODE type is not recognized.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let x = symplex::var("x");
    /// let y = symplex::var("y");
    /// let dy = y.formal_diff(&x);  // y'
    /// let ode = &dy + &(&y * 2);   // y' + 2y = 0
    /// if let Some((sol, constants)) = ode.dsolve(&y, &x) {
    ///     let s = format!("{sol}");
    ///     assert!(s.contains("exp"), "solution should contain exp: {s}");
    /// }
    /// ```
    pub fn dsolve(&self, func: &Ex, var: &Ex) -> Option<(Ex, Vec<Ex>)> {
        let result = {
            let mut guard = self.inner.write();
            crate::ode::dsolve(&mut guard.arena, self.id, func.id, var.id)
        };
        result.map(|r| {
            let solution = self.wrap(r.solution);
            let constants = r.constants.into_iter().map(|c| self.wrap(c)).collect();
            (solution, constants)
        })
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
    /// unbound symbols (e.g., `x.evalf(10)` without substituting a value).
    ///
    /// Returns [`SymplexError::Unevaluable`] if the expression contains
    /// nodes that cannot be evaluated to a finite number (infinity, NaN,
    /// imaginary unit, unevaluated derivatives/integrals, user functions).
    ///
    /// Returns [`SymplexError::PrecisionExhausted`] if the requested
    /// precision exceeds `EvalConfig::max_evalf_precision`, or if
    /// intermediate computation produces NaN.
    #[must_use = "returns the numerical value as a string"]
    pub fn evalf(&self, digits: u32) -> Result<String, SymplexError> {
        let _span = debug_span!("evalf", expr = ?self.id, digits = digits).entered();
        // Reduce exact values before numerical evaluation.
        // This ensures e.g. Gamma(5) → 24 (exact) rather than
        // computing 23.9999... via Stirling series.
        let evaled = self.eval();
        let guard = evaled.inner.read();
        crate::evalf::evalf(&guard.arena, evaled.id, digits)
    }

    /// Convenience: evaluate to an `f64`.
    ///
    /// Calls [`evalf`](Ex::evalf) with 16 digits of precision and
    /// parses the result to `f64`. This avoids the common pattern of
    /// `.evalf(15).unwrap().parse::<f64>().unwrap()`.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`evalf`](Ex::evalf), plus a
    /// [`SymplexError::NotImplemented`] if the decimal string cannot
    /// be parsed to `f64`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let x = symplex::var("x");
    /// let val = x.powi(2).subs_i64(&x, 3).evalf_f64().unwrap();
    /// assert!((val - 9.0).abs() < 1e-10);
    /// ```
    pub fn evalf_f64(&self) -> Result<f64, SymplexError> {
        // eval() first to reduce exact values (sin(0)→0, Gamma(5)→24, etc.)
        // before numerical computation. The evalf_complex64 call below
        // will work on the simplified expression.
        let (re, im) = self.eval().evalf_complex64()?;
        if im.abs() > 1e-15 {
            return Err(SymplexError::ComputationFailed {
                operation: "evalf_f64",
                reason: format!(
                    "expression has nonzero imaginary part (im={im}); use evalf_complex64() for complex results"
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
    pub fn evalf_complex64(&self) -> Result<(f64, f64), SymplexError> {
        let s = self.evalf(16)?;
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
    /// let x = symplex::var("x");
    /// let f = &x.powi(2) + 1;
    /// let func = f.lambdify(&["x"]).expect("should compile");
    /// assert!((func(&[3.0]) - 10.0).abs() < 1e-10);
    /// ```
    #[allow(clippy::type_complexity)]
    pub fn lambdify(&self, var_names: &[&str]) -> Option<Box<dyn Fn(&[f64]) -> f64 + Send + Sync>> {
        let inner = self.inner.read();
        crate::lambdify::lambdify(&inner.arena, self.id, var_names)
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
    /// let x = symplex::var("x");
    /// let sin_x = x.sin();
    /// let expr = &sin_x.powi(2) + &sin_x;
    /// let (bindings, result) = expr.cse();
    /// // sin(x) may be extracted as a common subexpression
    /// let _ = format!("{result}");
    /// ```
    pub fn cse(&self) -> (Vec<(Ex, Ex)>, Ex) {
        let result = {
            let mut guard = self.inner.write();
            crate::cse::cse(&mut guard.arena, self.id)
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
    /// let x = symplex::var("x");
    /// let f = x.powi(2) + 1;
    /// let code = f.to_rust_fn("my_func", &["x"]).unwrap();
    /// assert!(code.contains("pub fn my_func"));
    /// ```
    pub fn to_rust_fn(&self, name: &str, args: &[&str]) -> Result<String, SymplexError> {
        let mut guard = self.inner.write();
        guard.arena.to_rust_fn(self.id, name, args)
    }

    /// Generate a Rust function body as a string with custom code generation options.
    ///
    /// See [`CodegenOptions`](crate::codegen::CodegenOptions) for available
    /// settings (precision, math backend, annotations, CSE toggle).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use symplex::prelude::*;
    /// use symplex::codegen::{CodegenOptions, Precision};
    ///
    /// let x = symplex::var("x");
    /// let f = x.powi(2) + 1;
    /// let opts = CodegenOptions { precision: Precision::F32, ..Default::default() };
    /// let code = f.to_rust_fn_with_options("my_func", &["x"], &opts).unwrap();
    /// assert!(code.contains("f32"));
    /// ```
    pub fn to_rust_fn_with_options(
        &self,
        name: &str,
        args: &[&str],
        options: &crate::codegen::CodegenOptions,
    ) -> Result<String, SymplexError> {
        let mut guard = self.inner.write();
        crate::codegen::to_rust_fn_with_options(&mut guard.arena, self.id, name, args, options)
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
    pub fn sum_of(ctx: &crate::context::Context, exprs: impl IntoIterator<Item = Ex>) -> Ex {
        let items: Vec<Ex> = exprs.into_iter().collect();
        if items.is_empty() {
            return ctx.int(0);
        }
        let mut inner = items[0].inner.write();
        let ids: smallvec::SmallVec<[crate::node::ExprId; 8]> =
            items.iter().map(|e| e.id).collect();
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
    pub fn product_of(ctx: &crate::context::Context, exprs: impl IntoIterator<Item = Ex>) -> Ex {
        let items: Vec<Ex> = exprs.into_iter().collect();
        if items.is_empty() {
            return ctx.int(1);
        }
        let mut inner = items[0].inner.write();
        let ids: smallvec::SmallVec<[crate::node::ExprId; 8]> =
            items.iter().map(|e| e.id).collect();
        let id = inner.arena.mul(&ids);
        drop(inner);
        items[0].wrap(id)
    }

    // ── Expression walking ─────────────────────────────────────────

    /// Walk the expression bottom-up, applying a user-provided transformation
    /// at each node.
    ///
    /// The closure receives an [`ExprView`](crate::expr_view::ExprView) for
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
        F: Fn(crate::expr_view::ExprView<'_>) -> Option<Ex>,
    {
        let result_id = {
            let mut guard = self.inner.write();
            crate::walk::walk_and_rebuild(&mut guard.arena, self.id, &|arena, id| {
                let view = crate::expr_view::ExprView { id, arena };
                f(view).map(|ex| ex.id)
            })
        };
        self.wrap(result_id)
    }

    // ── Solver utilities ───────────────────────────────────────────

    /// Check whether `val` is a solution of `self = 0` for variable `var`.
    ///
    /// Substitutes `val` for `var`, evaluates, and checks if the result is zero.
    /// Returns `true` if the residual is zero (structurally or numerically),
    /// `false` if definitely non-zero, or falls back to structural check.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::vars;
    /// vars!(x);
    /// let poly = expr!(x^2 - 4);
    /// assert!(poly.check_solution(&x, &symplex::int(2)));
    /// assert!(poly.check_solution(&x, &symplex::int(-2)));
    /// assert!(!poly.check_solution(&x, &symplex::int(3)));
    /// ```
    #[must_use]
    pub fn check_solution(&self, var: &Ex, val: &Ex) -> bool {
        let substituted = self.subs(var, val).eval().simplify();
        if substituted.is_zero_structural() {
            return true;
        }
        // Try numerical evaluation
        if let Ok(v) = substituted.evalf_f64() {
            return v.abs() < 1e-10;
        }
        // Try expand + eval
        let expanded = substituted.expand().eval();
        expanded.is_zero_structural()
    }

    /// Classify an ODE represented as `self = 0`.
    ///
    /// `self` should contain formal derivative nodes (created via
    /// [`formal_diff`](Self::formal_diff)). `func` is the dependent
    /// variable (e.g., `y`) and `var` is the independent variable (e.g., `x`).
    ///
    /// Returns an [`OdeType`](crate::ode::OdeType) describing the
    /// recognized ODE class, or [`OdeType::Unknown`](crate::ode::OdeType::Unknown)
    /// if the form is not recognized.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::ode::OdeType;
    ///
    /// let x = symplex::var("x");
    /// let y = symplex::var("y");
    /// let dy = y.formal_diff(&x);
    /// let ode = &dy - &x; // y' = x
    /// assert_eq!(ode.classify_ode(&y, &x), OdeType::SimpleSeparable);
    /// ```
    #[must_use]
    pub fn classify_ode(&self, func: &Ex, var: &Ex) -> crate::ode::OdeType {
        let mut guard = self.inner.write();
        crate::ode::classify_ode(&mut guard.arena, self.id, func.id, var.id)
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
    /// let x = symplex::var("x");
    /// let y = symplex::var("y");
    /// let dy = y.formal_diff(&x);
    /// let ode = &dy - &x; // y' - x = 0
    /// // Solution: y = x²/2
    /// let sol = &x.powi(2) / 2;
    /// assert!(ode.checkodesol(&sol, &y, &x));
    /// ```
    #[must_use]
    pub fn checkodesol(&self, solution: &Ex, func: &Ex, var: &Ex) -> bool {
        {
            let mut guard = self.inner.write();
            crate::ode::checkodesol(&mut guard.arena, self.id, solution.id, func.id, var.id)
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
        let id =
            self.inner
                .write()
                .arena
                .interval(self.id, end.id, crate::node::INTERVAL_BOTH_CLOSED);
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
        let id =
            self.inner
                .write()
                .arena
                .interval(self.id, end.id, crate::node::INTERVAL_BOTH_OPEN);
        self.wrap_as(id)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper: parse complex evalf strings
// ═══════════════════════════════════════════════════════════════════════════

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
    fn evalf_complex64_pure_real() {
        let x = crate::var("x");
        let expr = &x.powi(2) + 1;
        let at_2 = expr.subs(&x, &crate::int(2));
        let (re, im) = at_2.evalf_complex64().unwrap();
        assert!((re - 5.0).abs() < 1e-10);
        assert!(im.abs() < 1e-10);
    }

    #[test]
    fn evalf_complex64_pure_imaginary() {
        let i = crate::i_unit();
        let (re, im) = i.evalf_complex64().unwrap();
        assert!(re.abs() < 1e-10);
        assert!((im - 1.0).abs() < 1e-10);
    }

    #[test]
    fn evalf_complex64_mixed() {
        let _ctx = crate::default_context();
        let expr = &crate::int(3) + &(&crate::int(4) * &crate::i_unit());
        let (re, im) = expr.evalf_complex64().unwrap();
        assert!((re - 3.0).abs() < 1e-10);
        assert!((im - 4.0).abs() < 1e-10);
    }

    #[test]
    fn evalf_f64_rejects_complex() {
        let i = crate::i_unit();
        assert!(
            i.evalf_f64().is_err(),
            "evalf_f64 should reject pure imaginary"
        );
    }
}
