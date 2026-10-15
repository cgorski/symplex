//! Summation, products, and series extension methods on [`Ex`].
//!
//! * [`Ex::summation`] / [`Ex::try_summation`] — symbolic `Σ`
//! * [`Ex::product_over`] / [`Ex::try_product_over`] — symbolic `Π`
//! * [`Ex::hypergeometric_ratio`] — the term ratio `t(k+1)/t(k)`
//! * [`Ex::is_absolutely_convergent`] — absolute convergence of `Σ t(k)`
//! * [`Ex::series_at_infinity`] / [`Ex::series_at_neg_infinity`] — asymptotic expansions
//! * [`Ex::series_dir`] / [`Ex::try_series_dir`] — one-sided expansions
//!
//! # What a series is
//!
//! [`Ex::series`] and the methods here compute the **asymptotic expansion
//! as `var → point` through real values** (two-sided for `series`, one-sided
//! for [`Ex::series_dir`] and at `±∞`), in the same sense as
//! [`Ex::limit`]: every term of exponent `< order` is present and exact, and
//! the remainder is `O((var − point)^order)` along the reals.  It is not a
//! Laurent series over the complex plane: `exp(−1/x²)` expands to `0` at
//! `0` (SymPy: `O(x**4)`), `exp(−1/x)` has no two-sided expansion there
//! (`0` from the right, none from the left).  One-sided expansions from the
//! right and those at `±∞` may contain logarithms (`ln Γ(x)` at `∞` is
//! Stirling's series, `Ei(x)` at `0⁺` is `γ + ln x + x + …`).  Where no
//! such expansion exists or it cannot be established, a formal `Series`
//! node is returned (`try_*` variants: `Err`).  Use
//! [`Ex::residue`] for residues: a series is not a Laurent expansion at an
//! essential singularity.

use tracing::debug_span;

use crate::api::expr::{Ex, Expr, Numeric};
use crate::base::errors::SymplexError;
use crate::base::node::ExprNode;
use crate::calculus::limit::Direction;
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
    /// identities (`Σ P(k)·C(n,k)·xᵏ` for any polynomial `P`), Gosper's
    /// algorithm, and — for infinite sums — p-series (`ζ(2m)` in closed
    /// form, `ζ(2m+1)` as a `zeta` node, `Σ (−1)ᵏ/(2k+1)² = G`), alternating
    /// series, and a table of classical power series (`Σ xᵏ/k! = eˣ`, `sin`,
    /// `cos`, `atan`, …).
    ///
    /// An infinite sum with a *symbolic* ratio (`Σ xᵏ`, `Σ k·xᵏ`, `Σ
    /// xᵏ/k²`, `Σ xᵏ/k`, the table series) whose convergence the declared
    /// assumptions decide gives the closed form (`Σ (1/(a + 1))ᵏ = 1/a` for
    /// a positive `a`) or the divergence; otherwise it is
    /// `Piecewise((closed form, region), (Sum, True))` with the exact
    /// convergence region: `|x| < 1` for `Σ P(k)·xᵏ`, `|x| ≤ 1` for
    /// `Σ xᵏ/k² = Li₂(x)`, `|x| ≤ 1 ∧ x ≠ 1` for `Σ xᵏ/k = −ln(1 − x)`
    /// (`|x| < 1`, sufficient, for `Σ C(k + c, k)·xᵏ`, where the circle
    /// depends on `c`).  Up to 0.53 these stayed unevaluated.  Numeric
    /// ratios are decided exactly.  Case splits respect the declared
    /// assumptions too: `Σ_{k=0}^{n} (a + 1)ᵏ` for a positive `a` has no
    /// `a + 1 = 1` branch.  `eval` of a formal `Sum` node keeps such a sum
    /// as it is (the `Piecewise` holds the `Sum` itself).
    ///
    /// Reversed limits follow Karr's convention (as SymPy): `upper = lower −
    /// 1` is the empty sum `0`, and `Σ_{k=a}^{b} = −Σ_{k=b+1}^{a−1}` for `b <
    /// a − 1` (`Σ_{k=5}^{2} k = −7`; up to 0.54 it was `0`).  A closed form
    /// for a symbolic upper limit holds for every integer value of it the
    /// assumptions allow (where no term is at a pole), or it is returned as
    /// `Piecewise((closed form, upper ≥ h), (Sum, True))`:
    /// `Σ_{k=0}^{n} C(n, k) = 2ⁿ` holds for `n ≥ 0` only (at `n = −1` the
    /// sum is empty; SymPy returns `2**n`), and is returned as such for an
    /// `n` declared non-negative.
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
    ///
    /// // Σ_{k=1}^{∞} 1/k³ = ζ(3)
    /// let apery = k.powi(-3).summation(&k, &ctx.int(1), &ctx.infinity());
    /// assert_eq!(apery.to_string(), "zeta(3)");
    ///
    /// // Σ_{k=1}^{∞} x^k/k² = Li₂(x) where |x| ≤ 1
    /// let li2 = (x.pow(&k) / k.powi(2)).summation(&k, &ctx.int(1), &ctx.infinity());
    /// assert!(li2.to_string().starts_with("Piecewise(polylog(2, x) if 1 >= abs(x), "));
    /// assert_eq!(li2.subs(&x, &ctx.rational(1, 2)).eval().to_string(),
    ///            "-1/2*ln(2)^2 + 1/12*pi^2");
    ///
    /// // Reversed limits: Σ_{k=5}^{2} k = −(3 + 4) (Karr's convention)
    /// assert_eq!(k.summation(&k, &ctx.int(5), &ctx.int(2)).to_string(), "-7");
    ///
    /// // Σ_{k=0}^{n} C(n, k) = 2ⁿ for n ≥ 0; at n = −1 the sum is empty
    /// let row = ctx.parse("binomial(n, k)").unwrap().summation(&k, &ctx.int(0), &n);
    /// assert!(row.to_string().starts_with("Piecewise(2^n if n >= 0, "));
    /// assert_eq!(row.subs_i64(&n, -1).eval().to_string(), "0");
    /// assert_eq!(row.subs_i64(&n, 5).eval().to_string(), "32");
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
            SumOutcome::Conditional { value, region } => {
                let sum = inner
                    .arena
                    .intern(ExprNode::Sum(self.raw_id(), var_id, lo_id, hi_id));
                let otherwise = inner.arena.bool_true;
                inner.arena.piecewise(&[(value, region), (sum, otherwise)])
            }
            SumOutcome::Divergent(None) | SumOutcome::Unevaluated => inner
                .arena
                .intern(ExprNode::Sum(self.raw_id(), var_id, lo_id, hi_id)),
        };
        drop(inner);
        self.wrap(id)
    }

    /// Like [`summation`](Self::summation), but returns `Err` when the sum
    /// diverges ([`SymplexError::Divergent`]), no closed form was found, or
    /// the closed form holds only under a condition the assumptions do not
    /// decide — convergence, or the upper limits where a finite sum's closed
    /// form holds ([`SymplexError::ComputationFailed`], the message names
    /// the condition).
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
            SumOutcome::Conditional { value, region } => {
                let (value, region) = (self.wrap(value), self.wrap(region));
                Err(SymplexError::ComputationFailed {
                    operation: "summation",
                    reason: format!(
                        "the sum is {value} where {region}, a condition the assumptions do \
                         not decide"
                    ),
                })
            }
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
    /// when that limit is exactly computable, and `Π b^{f(k)}` as
    /// `b^{Σ f(k)}`.
    ///
    /// A non-integer power `Π f(k)^e` is `(Π f(k))^e` only where the
    /// principal powers multiply: for factors positive over the range (and
    /// negative ones, with the phase `((−1)^e)^count`: `Π_{k=1}^{n} √(−k) =
    /// iⁿ·√(n!)`); with factors of both signs (`Π √(k − 3)`) the product
    /// stays unevaluated.  An empty range (`upper < lower`) gives `1`.
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
            // (Products have no conditional outcome.)
            SumOutcome::Divergent(None)
            | SumOutcome::Conditional { .. }
            | SumOutcome::Unevaluated => {
                inner
                    .arena
                    .intern(ExprNode::Product_(self.raw_id(), var_id, lo_id, hi_id))
            }
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
            SumOutcome::Conditional { .. } | SumOutcome::Unevaluated => {
                Err(SymplexError::ComputationFailed {
                    operation: "product",
                    reason: "no closed form found".into(),
                })
            }
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
        .to_option()
    }

    // ── Asymptotic expansions ──────────────────────────────────────

    /// Asymptotic expansion of `self` as `var → +∞`, with every term of
    /// exponent `< n_terms` in `1/var` (what a series is: see
    /// [`series_dir`](Self::series_dir)).
    ///
    /// Internally substitutes `var = 1/t`, expands at `t → 0⁺`, and
    /// substitutes back; coefficients may contain `ln(var)` (Stirling's
    /// series `ln Γ(x) = x ln x − x − ½ ln x + ½ ln 2π + 1/(12x) + …`,
    /// `asinh x = ln 2 + ln x + 1/(4x²) + …`), and a fractional power gives
    /// a series in powers of `var^(−1/2)` or `var^(−1/3)` (`√(x + 1) = √x +
    /// 1/(2√x) − …`).  Returns a formal `Series` node when the expansion
    /// cannot be computed.
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

    /// The expansion of `self` as `var → point` from the side `dir`, with
    /// every term of exponent `< order` in `var − point`.
    ///
    /// Like [`series`](Self::series) (which is `series_dir` with
    /// [`Direction::Both`]) this is the **asymptotic expansion as
    /// `var → point` through real values**, in the same sense as
    /// [`limit_dir`](Self::limit_dir): every term of exponent `< order` is
    /// present and exact, and the remainder is `O((var − point)^order)` as
    /// `var` approaches `point` from the side(s) in question.  It is not a
    /// Laurent series over the complex plane: `exp(−1/x²)` expands to `0` at
    /// `0` (SymPy: `O(x**4)`), `exp(−1/x)` has no two-sided expansion there
    /// (`0` from the right, none from the left).  From either side, and at
    /// `±∞` (where the side is implied and `dir` is ignored), the expansion
    /// may contain logarithms — `ln(var − point)` from the right,
    /// `ln(point − var)` from the left: `Ei(x) = γ + ln x + x + …` for
    /// `x → 0⁺`, `x^x = 1 + x·(ln(−x) + iπ) + …` for `x → 0⁻`, Stirling's
    /// series for `ln Γ(x)` at `∞`.  At a branch point a one-sided
    /// expansion may be a Puiseux series, in powers of `√(var − point)` or
    /// `∛(var − point)` (of `√(point − var)` from the left), as SymPy gives:
    /// `asin(x²)` at `x → 1⁺` is `π/2 − 2i·√(x − 1) − …`.  Where no such
    /// expansion exists or it cannot be established, a formal `Series` node
    /// is returned.  For residues use [`residue`](Self::residue): at an
    /// essential singularity a series is not the Laurent expansion.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let f = (-(1 / &x)).exp(); // exp(−1/x)
    /// let zero = ctx.int(0);
    /// // O(x⁴) from the right, no expansion from the left or two-sided
    /// assert_eq!(f.series_dir(&x, &zero, 4, Direction::Right).to_string(), "0");
    /// assert!(f.series_dir(&x, &zero, 4, Direction::Left).has_unevaluated());
    /// assert!(f.series(&x, &zero, 4).has_unevaluated());
    /// // Ei(x) = γ + ln x + x + x²/4 + … for x → 0⁺
    /// let ei = x.ei().series_dir(&x, &zero, 3, Direction::Right);
    /// assert!(ei.to_string().contains("ln(x)"), "{ei}");
    /// // A branch point: asin(x²) for x → 1⁻ is π/2 − 2√(1 − x) + …
    /// let one = ctx.int(1);
    /// let a = x.powi(2).asin();
    /// assert!(a.series(&x, &one, 2).has_unevaluated()); // no two-sided one
    /// let left = a.series_dir(&x, &one, 1, Direction::Left);
    /// assert_eq!(left.to_string(), "-2*sqrt(-x + 1) + 1/2*pi");
    /// ```
    #[must_use = "returns the expansion; does not modify in place"]
    pub fn series_dir(&self, var: &Ex, point: &Ex, order: u32, dir: Direction) -> Ex {
        let var_id = self.checked_id(var);
        let point_id = self.checked_id(point);
        let _span = debug_span!("series_dir", expr = ?self.raw_id(), order = order).entered();
        let mut inner = self.inner.write();
        let id = match crate::calculus::series::series_dir(
            &mut inner.arena,
            self.raw_id(),
            var_id,
            point_id,
            order,
            dir,
        ) {
            Ok(id) => id,
            Err(_) => {
                let order_id = inner.arena.int(order as i64);
                inner
                    .arena
                    .intern(ExprNode::Series(self.raw_id(), var_id, point_id, order_id))
            }
        };
        drop(inner);
        self.wrap(id)
    }

    /// Like [`series_dir`](Self::series_dir), but returns `Err` if there
    /// is no expansion.
    pub fn try_series_dir(
        &self,
        var: &Ex,
        point: &Ex,
        order: u32,
        dir: Direction,
    ) -> Result<Ex, SymplexError> {
        let r = self.series_dir(var, point, order, dir);
        if r.has_unevaluated() {
            Err(SymplexError::ComputationFailed {
                operation: "series_dir",
                reason: "no expansion from this side".into(),
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
