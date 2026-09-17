//! Complex-analysis methods on [`Ex`](crate::api::expr::Ex): `re`, `im`,
//! `conjugate`, `arg`, polar form — plus the 0.2 special-function
//! constructors (`si`, `ci`, `ei`, `li`, `zeta`, `polygamma`,
//! `kronecker_delta`).
//!
//! # Realness is never assumed
//!
//! A bare symbol `x` is *not* treated as real.  `x.re()` returns the
//! unevaluated node `re(x)` unless `x` carries a `Real` (or stronger)
//! assumption, in which case `x.re() == x` and `x.im() == 0`.  Use
//! [`Context::symbol_with`](crate::api::context::Context::symbol_with) or
//! [`Ex::assume`](crate::api::expr::Ex::assume) to declare assumptions.
//!
//! Principal-branch semantics apply throughout: `arg(z) ∈ (−π, π]`.

use crate::api::expr::{Ex, Expr, Numeric};
use crate::base::assumptions::Props;

// ═══════════════════════════════════════════════════════════════════════════
// impl Expr<Numeric> — complex analysis
// ═══════════════════════════════════════════════════════════════════════════

impl Expr<Numeric> {
    /// Real part `re(self)`.
    ///
    /// Evaluates at construction whenever the real part is determinable
    /// (numbers, constants, symbols assumed real, sums, real scalings,
    /// products/powers of fully decomposable factors, `exp`, `sin`, `cos`,
    /// `sinh`, `cosh`, `ln`, …).  Otherwise an unevaluated `re(…)` node is
    /// returned — never a silently-wrong answer.
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
    ///
    /// // Unknown symbols stay symbolic …
    /// let w = ctx.symbol("w");
    /// assert_eq!(format!("{}", w.re()), "re(w)");
    /// assert_eq!(format!("{}", (&i * &w).re()), "-im(w)");
    ///
    /// // … unless assumed real.
    /// let x = ctx.symbol_with("x", &[Assumption::Real]);
    /// assert_eq!(x.re(), x);
    /// ```
    #[must_use]
    pub fn re(&self) -> Ex {
        let id = self.inner.write().arena.re(self.raw_id());
        self.wrap(id)
    }

    /// Imaginary part `im(self)` — a *real* quantity such that
    /// `self = re(self) + i·im(self)`.
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
    ///
    /// let x = ctx.symbol_with("x", &[Assumption::Real]);
    /// assert!(x.im().is_zero_structural());
    /// assert_eq!(format!("{}", x.exp().mul(&i).im()), "exp(x)");
    /// ```
    #[must_use]
    pub fn im(&self) -> Ex {
        let id = self.inner.write().arena.im(self.raw_id());
        self.wrap(id)
    }

    /// Complex conjugate `conjugate(self)`.
    ///
    /// Distributes over sums, products and integer powers, and commutes
    /// with real-analytic functions that have no branch cut off the real
    /// axis (`exp`, `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`, `Γ`,
    /// `erf`, `erfc`, `ψ`, `ζ`, `Si`).  Functions with branch cuts (`ln`,
    /// non-integer powers, inverse trig/hyperbolic, `W`, …) stay as
    /// unevaluated `conjugate(…)` nodes unless the argument is provably
    /// real.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let i = ctx.i_unit();
    /// let z = &ctx.int(3) + &(&ctx.int(4) * &i);
    /// assert_eq!(format!("{}", z.conjugate()), "-4*I + 3");
    ///
    /// let w = ctx.symbol("w");
    /// assert_eq!(format!("{}", w.sin().conjugate()), "sin(conjugate(w))");
    /// assert_eq!(w.conjugate().conjugate(), w);
    /// ```
    #[must_use]
    pub fn conjugate(&self) -> Ex {
        let id = self.inner.write().arena.conjugate(self.raw_id());
        self.wrap(id)
    }

    /// Principal complex argument `arg(self) ∈ (−π, π]`.
    ///
    /// Positive reals give `0`, negative reals give `π`, `i` gives `π/2`,
    /// and `a + bi` with real `a`, `b` gives `atan2(b, a)` (folded for
    /// numeric arguments).  Positive real factors are discarded:
    /// `arg(3·z) = arg(z)`.  When the sign or complex structure cannot be
    /// determined the result is an unevaluated `arg(…)` node.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let z = &ctx.int(1) + &ctx.i_unit();
    /// assert_eq!(format!("{}", z.arg()), "1/4*pi");
    /// assert_eq!(format!("{}", ctx.int(-2).arg()), "pi");
    ///
    /// let x = ctx.symbol_with("x", &[Assumption::Real]);
    /// assert_eq!(format!("{}", x.arg()), "arg(x)"); // sign unknown
    /// ```
    #[must_use]
    pub fn arg(&self) -> Ex {
        let id = self.inner.write().arena.arg(self.raw_id());
        self.wrap(id)
    }

    /// Decompose into `(re, im)` with `self = re + i·im`.
    ///
    /// Both parts are real-valued expressions; unknown quantities appear as
    /// `re(…)`/`im(…)` nodes.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol_with("x", &[Assumption::Real]);
    /// let z = (&ctx.i_unit() * &x).exp();          // e^{ix}
    /// let (re, im) = z.as_real_imag();
    /// assert_eq!(format!("{re}"), "cos(x)");
    /// assert_eq!(format!("{im}"), "sin(x)");
    /// ```
    #[must_use]
    pub fn as_real_imag(&self) -> (Ex, Ex) {
        let (re, im) = self.inner.write().arena.as_real_imag_expr(self.raw_id());
        (self.wrap(re), self.wrap(im))
    }

    /// Rewrite as `re + i·im`, expanding every unknown symbol `z` into
    /// `re(z) + i·im(z)` (symbols assumed real are left alone).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let z = ctx.symbol("z");
    /// assert_eq!(format!("{}", z.expand_complex()), "im(z)*I + re(z)");
    ///
    /// let x = ctx.symbol_with("x", &[Assumption::Real]);
    /// let e = (&ctx.i_unit() * &x).exp().expand_complex();
    /// assert_eq!(format!("{e}"), "sin(x)*I + cos(x)");
    /// ```
    #[must_use]
    pub fn expand_complex(&self) -> Ex {
        let id = {
            let mut guard = self.inner.write();
            crate::base::complex::expand_complex(&mut guard.arena, self.raw_id())
        };
        self.wrap(id)
    }

    /// Polar form `(|self|, arg(self))`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let z = &ctx.int(3) + &(&ctx.int(4) * &ctx.i_unit());
    /// let (r, theta) = z.polar();
    /// assert!((r.eval_f64().unwrap() - 5.0).abs() < 1e-12);
    /// assert!((theta.eval_f64().unwrap() - (4.0f64).atan2(3.0)).abs() < 1e-12);
    /// ```
    #[must_use]
    pub fn polar(&self) -> (Ex, Ex) {
        (self.abs(), self.arg())
    }

    /// `|self|² = re² + im² = self·conjugate(self)`.
    ///
    /// When the real/imaginary decomposition is fully determined the result
    /// is `re² + im²`; otherwise it is `self·conjugate(self)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let z = &ctx.int(3) + &(&ctx.int(4) * &ctx.i_unit());
    /// assert_eq!(format!("{}", z.abs_squared().eval()), "25");
    ///
    /// let w = ctx.symbol("w");
    /// assert_eq!(format!("{}", w.abs_squared()), "w*conjugate(w)");
    /// ```
    #[must_use]
    pub fn abs_squared(&self) -> Ex {
        let id = {
            let mut guard = self.inner.write();
            let arena = &mut guard.arena;
            let parts = crate::base::complex::decompose(arena, self.raw_id());
            if parts.exact {
                let two = arena.int(2);
                let re2 = arena.pow(parts.re, two);
                let im2 = arena.pow(parts.im, two);
                arena.add(&[re2, im2])
            } else {
                let conj = arena.conjugate(self.raw_id());
                arena.mul(&[self.raw_id(), conj])
            }
        };
        self.wrap(id)
    }

    /// Three-valued test: is this expression real-valued?
    ///
    /// `Some(true)` if provably real (assumptions, or a decomposition with a
    /// structurally zero imaginary part), `Some(false)` if provably not real
    /// (e.g. a non-zero numeric imaginary part), `None` if unknown.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let i = ctx.i_unit();
    /// assert_eq!(ctx.int(3).is_real_valued(), Some(true));
    /// assert_eq!((&ctx.int(3) + &i).is_real_valued(), Some(false));
    /// assert_eq!(ctx.symbol("z").is_real_valued(), None);
    /// assert_eq!(ctx.symbol("z").abs().is_real_valued(), Some(true));
    /// let x = ctx.symbol_with("x", &[Assumption::Real]);
    /// assert_eq!((&x + &i).is_real_valued(), Some(false));
    /// ```
    #[must_use]
    pub fn is_real_valued(&self) -> Option<bool> {
        if let Some(v) = self.query(Props::REAL) {
            return Some(v);
        }
        let mut guard = self.inner.write();
        let arena = &mut guard.arena;
        let parts = crate::base::complex::decompose(arena, self.raw_id());
        if parts.im == arena.zero() {
            return Some(true);
        }
        if parts.exact {
            // A structurally non-zero *number* as imaginary part → not real.
            if let Some(r) = arena.as_num(parts.im)
                && !num_traits::Zero::is_zero(r)
            {
                return Some(false);
            }
            // Symbolic imaginary part: decide via assumptions on im.
            let im_id = parts.im;
            drop(guard);
            let im_ex = self.wrap(im_id);
            return match im_ex.query(Props::ZERO) {
                Some(true) => Some(true),
                _ => match im_ex.query(Props::NONZERO) {
                    Some(true) => Some(false),
                    _ => None,
                },
            };
        }
        None
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// impl Expr<Numeric> — special functions (0.2 nodes)
// ═══════════════════════════════════════════════════════════════════════════

impl Expr<Numeric> {
    /// Sine integral `Si(self) = ∫₀ˣ sin(t)/t dt`.
    ///
    /// Exact values: `Si(0) = 0`, `Si(∞) = π/2`, `Si(−x) = −Si(x)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// assert_eq!(format!("{}", x.si()), "Si(x)");
    /// assert_eq!(format!("{}", x.si().diff(&x)), "sin(x)/x");
    /// let v = ctx.int(1).si().eval_f64().unwrap();
    /// assert!((v - 0.946083070367183).abs() < 1e-14);
    /// ```
    #[must_use]
    pub fn si(&self) -> Ex {
        let id = self.inner.write().arena.si(self.raw_id());
        self.wrap(id)
    }

    /// Cosine integral `Ci(self) = γ + ln(x) + ∫₀ˣ (cos(t) − 1)/t dt`.
    ///
    /// Exact values: `Ci(∞) = 0`.  For `x < 0` the numerical value is
    /// complex: `Ci(−x) = Ci(x) + iπ`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let v = ctx.int(1).ci().eval_f64().unwrap();
    /// assert!((v - 0.337403922900968).abs() < 1e-14);
    /// ```
    #[must_use]
    pub fn ci(&self) -> Ex {
        let id = self.inner.write().arena.ci(self.raw_id());
        self.wrap(id)
    }

    /// Exponential integral `Ei(self) = −∫_{−x}^{∞} e^{−t}/t dt` (Cauchy
    /// principal value for `x > 0`).
    ///
    /// Exact values: `Ei(−∞) = 0`, `Ei(∞) = ∞`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let v = ctx.int(1).ei().eval_f64().unwrap();
    /// assert!((v - 1.895117816355937).abs() < 1e-14);
    /// let x = ctx.symbol("x");
    /// assert_eq!(format!("{}", x.ei().diff(&x)), "exp(x)/x");
    /// ```
    #[must_use]
    pub fn ei(&self) -> Ex {
        let id = self.inner.write().arena.ei(self.raw_id());
        self.wrap(id)
    }

    /// Logarithmic integral `li(self) = ∫₀ˣ dt/ln(t) = Ei(ln x)`.
    ///
    /// Exact values: `li(0) = 0`, `li(1) = −∞`, `li(e^y) = Ei(y)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let v = ctx.int(2).li().eval_f64().unwrap();
    /// assert!((v - 1.045163780117493).abs() < 1e-14);
    /// let x = ctx.symbol("x");
    /// assert_eq!(format!("{}", x.li().diff(&x)), "1/ln(x)");
    /// ```
    #[must_use]
    pub fn li(&self) -> Ex {
        let id = self.inner.write().arena.li(self.raw_id());
        self.wrap(id)
    }

    /// Riemann zeta function `ζ(self)`.
    ///
    /// Exact values: `ζ(1) = zoo`, `ζ(0) = −1/2`, `ζ(−n) = −Bₙ₊₁/(n+1)`,
    /// `ζ(2k)` as a rational multiple of `π^{2k}`; odd positive arguments
    /// stay symbolic.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// assert_eq!(format!("{}", ctx.int(2).zeta()), "1/6*pi^2");
    /// assert_eq!(format!("{}", ctx.int(4).zeta()), "1/90*pi^4");
    /// assert_eq!(format!("{}", ctx.int(-1).zeta()), "-1/12");
    /// assert_eq!(format!("{}", ctx.int(3).zeta()), "zeta(3)");
    /// let v = ctx.int(3).zeta().eval_f64().unwrap();
    /// assert!((v - 1.202056903159594).abs() < 1e-14);
    /// ```
    #[must_use]
    pub fn zeta(&self) -> Ex {
        let id = self.inner.write().arena.zeta(self.raw_id());
        self.wrap(id)
    }

    /// Polygamma function `ψ⁽ⁿ⁾(self)` — the `n`-th derivative of the
    /// digamma function.
    ///
    /// `ψ⁽⁰⁾` is canonicalised to [`digamma`](Self::digamma); `ψ⁽ⁿ⁾(1)`,
    /// `ψ⁽ⁿ⁾(1/2)` and small integer / half-integer shifts of them fold to
    /// multiples of `ζ(n+1)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let one = ctx.int(1);
    /// // trigamma(1) = ζ(2) = π²/6
    /// assert_eq!(format!("{}", one.polygamma(&one)), "1/6*pi^2");
    /// let x = ctx.symbol("x");
    /// assert_eq!(format!("{}", x.digamma().diff(&x)), "polygamma(1, x)");
    /// let v = ctx.int(1).polygamma(&one).eval_f64().unwrap();
    /// assert!((v - std::f64::consts::PI.powi(2) / 6.0).abs() < 1e-14);
    /// ```
    #[must_use]
    pub fn polygamma(&self, n: &Ex) -> Ex {
        let n_id = self.checked_id(n);
        let id = self.inner.write().arena.polygamma(n_id, self.raw_id());
        self.wrap(id)
    }

    /// Kronecker delta `δ(self, other)`: `1` if the arguments are equal,
    /// `0` if they are provably different numbers, otherwise symbolic.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let i = ctx.symbol("i");
    /// let j = ctx.symbol("j");
    /// assert_eq!(format!("{}", i.kronecker_delta(&i)), "1");
    /// assert_eq!(format!("{}", ctx.int(2).kronecker_delta(&ctx.int(3))), "0");
    /// let d = i.kronecker_delta(&j);
    /// assert_eq!(format!("{d}"), "KroneckerDelta(i, j)");
    /// assert_eq!(d, j.kronecker_delta(&i)); // symmetric
    /// assert_eq!(format!("{}", (&i + 1).kronecker_delta(&i)), "0");
    /// ```
    #[must_use]
    pub fn kronecker_delta(&self, other: &Ex) -> Ex {
        let other_id = self.checked_id(other);
        let id = self
            .inner
            .write()
            .arena
            .kronecker_delta(self.raw_id(), other_id);
        self.wrap(id)
    }
}
