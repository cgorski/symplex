//! Summation, products, and series extension methods on [`Ex`](crate::api::expr::Ex).
//!
//! * [`Ex::summation`] / [`Ex::try_summation`] — symbolic `Σ`
//! * [`Ex::product_over`] / [`Ex::try_product_over`] — symbolic `Π`
//! * [`Ex::hypergeometric_ratio`] — the term ratio `t(k+1)/t(k)`
//! * [`Ex::is_absolutely_convergent`] — absolute convergence of `Σ t(k)`
//! * [`Ex::series_at_infinity`] / [`Ex::series_at_neg_infinity`] — asymptotic expansions

use tracing::debug_span;

use crate::api::expr::{Ex, Expr, Numeric};
use crate::base::errors::SymplexError;
use crate::base::node::ExprNode;
use crate::calculus::summation::{self, SumOutcome};

impl Expr<Numeric> {
    // ── Summation ──────────────────────────────────────────────────

    /// Evaluate `Σ_{var=lower}^{upper} self` symbolically.
    ///
    /// Bounds may be concrete integers, symbolic expressions, or
    /// `ctx.infinity()` / `ctx.neg_infinity()`.  Strategies include exact
    /// enumeration for small concrete ranges, Faulhaber's formula (any
    /// degree, Bernoulli numbers), partial-fraction telescoping, harmonic
    /// numbers, geometric and arithmetico-geometric series, binomial
    /// identities, Gosper's algorithm, and — for infinite sums — p-series
    /// (`ζ(2m)` in closed form), alternating series, and a table of
    /// classical power series (`Σ xᵏ/k! = eˣ`, `sin`, `cos`, `atan`, …).
    ///
    /// When no closed form is known the formal `Sum` node is returned.
    /// Sums that provably diverge to `±∞` evaluate to `oo` / `-oo`; for
    /// oscillating divergence the formal `Sum` is kept.  Use
    /// [`try_summation`](Self::try_summation) to get an error instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let k = ctx.symbol("k");
    /// let n = ctx.symbol("n");
    ///
    /// // Σ_{k=1}^{n} k² = n³/3 + n²/2 + n/6
    /// let s = k.powi(2).summation(&k, &ctx.int(1), &n);
    /// assert_eq!(s.subs_i64(&n, 10).eval().to_string(), "385");
    ///
    /// // Σ_{k=1}^{∞} 1/k² = π²/6
    /// let basel = k.powi(-2).summation(&k, &ctx.int(1), &ctx.infinity());
    /// assert_eq!(basel.to_string(), "1/6*pi^2");
    ///
    /// // Σ_{k=0}^{∞} x^k/k! = e^x
    /// let x = ctx.symbol("x");
    /// let e = (x.pow(&k) / k.factorial()).summation(&k, &ctx.int(0), &ctx.infinity());
    /// assert_eq!(e.to_string(), "exp(x)");
    /// ```
    #[must_use = "returns the evaluated sum; does not modify in place"]
    pub fn summation(&self, var: &Ex, lower: &Ex, upper: &Ex) -> Ex {
        let var_id = self.checked_id(var);
        let lo_id = self.checked_id(lower);
        let hi_id = self.checked_id(upper);
        let _span = debug_span!("summation", body = ?self.raw_id()).entered();
        let mut inner = self.inner.write();
        let outcome = summation::summation(&mut inner.arena, self.raw_id(), var_id, lo_id, hi_id);
        let id = match outcome {
            SumOutcome::Closed(id) => id,
            SumOutcome::Divergent(Some(inf)) => inf,
            SumOutcome::Divergent(None) | SumOutcome::Unevaluated => inner
                .arena
                .intern(ExprNode::Sum(self.raw_id(), var_id, lo_id, hi_id)),
        };
        drop(inner);
        self.wrap(id)
    }

    /// Like [`summation`](Self::summation), but returns `Err` when the sum
    /// diverges ([`SymplexError::Divergent`]) or no closed form was found
    /// ([`SymplexError::ComputationFailed`]).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let k = ctx.symbol("k");
    /// let harmonic = k.powi(-1).try_summation(&k, &ctx.int(1), &ctx.infinity());
    /// assert!(matches!(harmonic, Err(SymplexError::Divergent { .. })));
    ///
    /// let geometric = ctx.rational(1, 2).pow(&k).try_summation(&k, &ctx.int(0), &ctx.infinity());
    /// assert_eq!(geometric.unwrap().to_string(), "2");
    /// ```
    pub fn try_summation(&self, var: &Ex, lower: &Ex, upper: &Ex) -> Result<Ex, SymplexError> {
        let var_id = self.checked_id(var);
        let lo_id = self.checked_id(lower);
        let hi_id = self.checked_id(upper);
        let _span = debug_span!("try_summation", body = ?self.raw_id()).entered();
        let mut inner = self.inner.write();
        let outcome = summation::summation(&mut inner.arena, self.raw_id(), var_id, lo_id, hi_id);
        drop(inner);
        match outcome {
            SumOutcome::Closed(id) => {
                let result = self.wrap(id);
                if result.has_unevaluated() {
                    Err(SymplexError::ComputationFailed {
                        operation: "summation",
                        reason: "no closed form for part of the sum".into(),
                    })
                } else {
                    Ok(result)
                }
            }
            SumOutcome::Divergent(Some(inf)) => Err(SymplexError::Divergent {
                operation: "summation",
                reason: format!("the series diverges to {}", self.wrap(inf)),
            }),
            SumOutcome::Divergent(None) => Err(SymplexError::Divergent {
                operation: "summation",
                reason: "the series diverges (terms do not tend to zero)".into(),
            }),
            SumOutcome::Unevaluated => Err(SymplexError::ComputationFailed {
                operation: "summation",
                reason: "no closed form found".into(),
            }),
        }
    }

    // ── Products ───────────────────────────────────────────────────

    /// Evaluate `Π_{var=lower}^{upper} self` symbolically.
    ///
    /// Handles constant factors (`cⁿ`), `Π k = n!`, `Π (k+a)` as Gamma /
    /// factorial ratios, `Π aᶠ⁽ᵏ⁾ = a^{Σ f(k)}`, products of factors, and
    /// rational functions whose numerator and denominator factor into
    /// linear factors over ℚ (`Π (1 − 1/k²) = (n+1)/(2n)`).  Infinite
    /// products are evaluated through the limit of the finite closed form
    /// when that limit is exactly computable.
    ///
    /// Returns the formal `Product` node when no closed form is known.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let k = ctx.symbol("k");
    /// let n = ctx.symbol("n");
    ///
    /// assert_eq!(k.product_over(&k, &ctx.int(1), &n).to_string(), "n!");
    ///
    /// // Π_{k=1}^{n} (1 + 1/k) = n + 1
    /// let p = (&ctx.int(1) + &k.powi(-1)).product_over(&k, &ctx.int(1), &n);
    /// assert_eq!(p.to_string(), "n + 1");
    ///
    /// // Π_{k=2}^{∞} (1 − 1/k²) = 1/2
    /// let p = (&ctx.int(1) - &k.powi(-2)).product_over(&k, &ctx.int(2), &ctx.infinity());
    /// assert_eq!(p.to_string(), "1/2");
    /// ```
    #[must_use = "returns the evaluated product; does not modify in place"]
    pub fn product_over(&self, var: &Ex, lower: &Ex, upper: &Ex) -> Ex {
        let var_id = self.checked_id(var);
        let lo_id = self.checked_id(lower);
        let hi_id = self.checked_id(upper);
        let _span = debug_span!("product_over", body = ?self.raw_id()).entered();
        let mut inner = self.inner.write();
        let outcome = summation::product(&mut inner.arena, self.raw_id(), var_id, lo_id, hi_id);
        let id = match outcome {
            SumOutcome::Closed(id) => id,
            SumOutcome::Divergent(Some(inf)) => inf,
            SumOutcome::Divergent(None) | SumOutcome::Unevaluated => inner
                .arena
                .intern(ExprNode::Product_(self.raw_id(), var_id, lo_id, hi_id)),
        };
        drop(inner);
        self.wrap(id)
    }

    /// Like [`product_over`](Self::product_over), but returns `Err` when the
    /// product diverges or no closed form was found.
    pub fn try_product_over(&self, var: &Ex, lower: &Ex, upper: &Ex) -> Result<Ex, SymplexError> {
        let var_id = self.checked_id(var);
        let lo_id = self.checked_id(lower);
        let hi_id = self.checked_id(upper);
        let mut inner = self.inner.write();
        let outcome = summation::product(&mut inner.arena, self.raw_id(), var_id, lo_id, hi_id);
        drop(inner);
        match outcome {
            SumOutcome::Closed(id) => {
                let result = self.wrap(id);
                if result.has_unevaluated() {
                    Err(SymplexError::ComputationFailed {
                        operation: "product",
                        reason: "no closed form for part of the product".into(),
                    })
                } else {
                    Ok(result)
                }
            }
            SumOutcome::Divergent(_) => Err(SymplexError::Divergent {
                operation: "product",
                reason: "the infinite product diverges".into(),
            }),
            SumOutcome::Unevaluated => Err(SymplexError::ComputationFailed {
                operation: "product",
                reason: "no closed form found".into(),
            }),
        }
    }

    // ── Hypergeometric terms ───────────────────────────────────────

    /// If `self` is a hypergeometric term in `var`, return the ratio
    /// `self(var+1) / self(var)` as a rational function of `var`.
    ///
    /// Returns `None` when the ratio is not a rational function of `var`
    /// (e.g. for `sin(k)` or `k^k`).  Symbolic parameters other than `var`
    /// are allowed.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let k = ctx.symbol("k");
    /// let n = ctx.symbol("n");
    ///
    /// let r = k.factorial().hypergeometric_ratio(&k).unwrap();
    /// assert_eq!(r.to_string(), "k + 1");
    ///
    /// let r = n.binomial(&k).hypergeometric_ratio(&k).unwrap();
    /// assert_eq!(r.subs_i64(&n, 5).subs_i64(&k, 2).eval().to_string(), "1");
    ///
    /// assert!(k.sin().hypergeometric_ratio(&k).is_none());
    /// ```
    #[must_use]
    pub fn hypergeometric_ratio(&self, var: &Ex) -> Option<Ex> {
        let var_id = self.checked_id(var);
        let mut inner = self.inner.write();
        let r = crate::calculus::gosper::is_hypergeometric(&mut inner.arena, self.raw_id(), var_id);
        drop(inner);
        r.map(|id| self.wrap(id))
    }

    // ── Convergence ────────────────────────────────────────────────

    /// Test whether `Σ |self(var)|` converges (absolute convergence).
    ///
    /// Sign-alternating factors such as `(−1)^k` are stripped before the
    /// convergence tests run, so conditionally convergent series like
    /// `Σ (−1)^k/k` return `Some(false)` here but `Some(true)` from
    /// [`is_convergent`](Self::is_convergent).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let k = ctx.symbol("k");
    /// let alt = ctx.int(-1).pow(&k) / &k;
    /// assert_eq!(alt.is_convergent(&k), Some(true));
    /// assert_eq!(alt.is_absolutely_convergent(&k), Some(false));
    /// ```
    #[must_use]
    pub fn is_absolutely_convergent(&self, var: &Ex) -> Option<bool> {
        let var_id = self.checked_id(var);
        let _span = debug_span!("is_absolutely_convergent").entered();
        let mut inner = self.inner.write();
        crate::calculus::convergence::is_absolutely_convergent(
            &mut inner.arena,
            self.raw_id(),
            var_id,
        )
    }

    // ── Asymptotic expansions ──────────────────────────────────────

    /// Asymptotic expansion of `self` as `var → +∞`, with `n_terms` terms
    /// in powers of `1/var`.
    ///
    /// Internally substitutes `var = 1/t`, expands (Laurent-)series at
    /// `t = 0`, and substitutes back.  Returns a formal `Series` node when
    /// the expansion cannot be computed.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// // x/(x+1) = 1 − 1/x + 1/x² − …
    /// let s = (&x / &(&x + 1)).series_at_infinity(&x, 3);
    /// let at_10 = s.subs_i64(&x, 10).eval_f64().unwrap();
    /// assert!((at_10 - 0.91).abs() < 1e-12);
    /// ```
    #[must_use = "returns the expansion; does not modify in place"]
    pub fn series_at_infinity(&self, var: &Ex, n_terms: u32) -> Ex {
        self.series_at_infinity_impl(var, n_terms, false)
    }

    /// Asymptotic expansion of `self` as `var → −∞` (see
    /// [`series_at_infinity`](Self::series_at_infinity)).
    #[must_use = "returns the expansion; does not modify in place"]
    pub fn series_at_neg_infinity(&self, var: &Ex, n_terms: u32) -> Ex {
        self.series_at_infinity_impl(var, n_terms, true)
    }

    /// Like [`series_at_infinity`](Self::series_at_infinity), but returns
    /// `Err` if the expansion could not be computed.
    pub fn try_series_at_infinity(&self, var: &Ex, n_terms: u32) -> Result<Ex, SymplexError> {
        let r = self.series_at_infinity(var, n_terms);
        if r.has_unevaluated() {
            Err(SymplexError::ComputationFailed {
                operation: "series_at_infinity",
                reason: "could not compute asymptotic expansion".into(),
            })
        } else {
            Ok(r)
        }
    }

    fn series_at_infinity_impl(&self, var: &Ex, n_terms: u32, negative: bool) -> Ex {
        let var_id = self.checked_id(var);
        let _span = debug_span!("series_at_infinity", expr = ?self.raw_id()).entered();
        let mut inner = self.inner.write();
        let id = crate::calculus::series::series_at_infinity(
            &mut inner.arena,
            self.raw_id(),
            var_id,
            n_terms,
            negative,
        );
        let id = match id {
            Ok(id) => id,
            Err(_) => {
                let point = if negative {
                    inner.arena.neg_infinity
                } else {
                    inner.arena.infinity
                };
                let order = inner.arena.int(n_terms as i64);
                inner
                    .arena
                    .intern(ExprNode::Series(self.raw_id(), var_id, point, order))
            }
        };
        drop(inner);
        self.wrap(id)
    }
}
