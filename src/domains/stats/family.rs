//! The [`Family`] trait every distribution implements, and
//! [`Distribution`], the handle that owns the generic exact machinery
//! (integrate or sum the density over a region, clamp a distribution
//! function to its support, fall back from a missing closed form).
//!
//! A family supplies what is *specific* to it — support, density, and the
//! closed forms it happens to have — and gets every query for free; a
//! family that wraps another ([`Truncated`](super::Truncated),
//! [`Affine`](super::Affine), [`Mixture`](super::Mixture)) composes by
//! calling the inner [`Distribution`]'s machinery.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use crate::api::context::Context;
use crate::api::expr::Ex;
use crate::api::poly_ex::Poly;
use crate::base::errors::SymplexError;

use super::sample::Rng;
use super::support::{Kind, Piece, Support, is_neg_inf, is_pos_inf};

/// A closure producing one `f64` sample per call.
pub type Sampler = Box<dyn FnMut(&mut Rng) -> f64 + Send>;

/// A probability distribution family: its support and density, plus
/// whatever closed forms it has (each defaults to "none", which makes the
/// generic machinery in [`Distribution`] integrate or sum instead).
///
/// Implement this to add a family; wrap the value in
/// [`Distribution::from_family`].  Every closed form here is *on the
/// support*: `cdf(x)` is `P(X ≤ x)` for `x` inside the support only, and
/// [`Distribution::cdf`] clamps it to `0`/`1` outside.
///
/// The `Any` supertrait lets [`Distribution::downcast_ref`] recover the
/// concrete family (the closure results in `sum_distribution` need to know
/// they have two normals).
pub trait Family: Any + Send + Sync + fmt::Debug {
    /// The family's name (`"Normal"`, `"Binomial"`, …).
    fn name(&self) -> &str;

    /// The context the parameters live in (a cheap handle clone).
    fn context(&self) -> Context;

    /// Where the mass is.
    fn support(&self) -> Support;

    /// The density (continuous) or probability mass function (discrete) as
    /// an expression in `x`, valid on the support.
    fn density(&self, x: &Ex) -> Ex;

    /// The parameters, named, in the family's documented order.
    fn parameters(&self) -> Vec<(&'static str, Ex)>;

    /// Structural equality with another family (see [`same_family`]).
    fn eq_family(&self, other: &dyn Family) -> bool;

    /// Closed-form mean.
    fn mean(&self) -> Option<Ex> {
        None
    }

    /// Closed-form variance.
    fn variance(&self) -> Option<Ex> {
        None
    }

    /// Closed-form raw moment `E[Xⁿ]`, `n ≥ 1`.
    fn raw_moment(&self, _n: u32) -> Option<Ex> {
        None
    }

    /// Closed-form `P(X ≤ x)` for `x` in the support.
    fn cdf(&self, _x: &Ex) -> Option<Ex> {
        None
    }

    /// Closed-form moment generating function `E[e^{tX}]`.
    fn mgf(&self, _t: &Ex) -> Option<Ex> {
        None
    }

    /// Closed-form quantile function (inverse CDF) at `p ∈ (0, 1)`.
    fn quantile(&self, _p: &Ex) -> Option<Ex> {
        None
    }

    /// Closed-form entropy (differential for a continuous family, Shannon
    /// for a discrete one), in nats.
    fn entropy(&self) -> Option<Ex> {
        None
    }

    /// `P(X ∈ region)` when the family can answer more directly than by
    /// integrating its density over `support ∩ region` (a mixture sums its
    /// components).  `None` leaves it to the generic route.
    fn probability_of(&self, _region: &Support) -> Option<Result<Ex, SymplexError>> {
        None
    }

    /// `E[g(X)·1_{X ∈ region}]` for `g` in the symbol `x`, when the family
    /// can answer directly.  `None` leaves it to the generic route.
    fn expectation_over(
        &self,
        _g: &Ex,
        _x: &Ex,
        _region: &Support,
    ) -> Option<Result<Ex, SymplexError>> {
        None
    }

    /// A sampler, when the family has a route of its own (a wrapper
    /// transforms its inner family's samples).  `None` leaves it to the
    /// generic route: inverse transform through the quantile, or cumulative
    /// sums of the mass function.
    fn sampler(&self) -> Option<Result<Sampler, SymplexError>> {
        None
    }

    /// `Name(p₁, p₂, …)`.
    fn fmt_display(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(", self.name())?;
        for (i, (_, p)) in self.parameters().iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{p}")?;
        }
        f.write_str(")")
    }
}

/// The equality every family's [`Family::eq_family`] delegates to: the
/// other family is the same concrete type and compares equal.
pub fn same_family<T: Family + PartialEq>(a: &T, other: &dyn Family) -> bool {
    (other as &dyn Any)
        .downcast_ref::<T>()
        .is_some_and(|b| a == b)
}

/// A probability distribution: a shared handle to a [`Family`] with the
/// generic exact machinery on top.  Construct with the associated
/// functions (`Distribution::normal(…)`, `binomial(…)`, …, each with a
/// `try_` twin validating numeric parameters), with the wrapper
/// constructors ([`truncated`](Self::truncated), [`affine`](Self::affine),
/// [`transformed`](Self::transformed), [`mixture`](Self::mixture)), or from
/// your own family with [`from_family`](Self::from_family).
#[derive(Clone)]
pub struct Distribution(Arc<dyn Family>);

impl PartialEq for Distribution {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq_family(other.0.as_ref())
    }
}

impl fmt::Debug for Distribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&*self.0, f)
    }
}

impl fmt::Display for Distribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt_display(f)
    }
}

/// A symbol named `_{prefix}` (or `_{prefix}1`, `_{prefix}2`, …) that
/// occurs in none of `avoid`, for use as a bound variable.
pub(crate) fn fresh_symbol(ctx: &Context, prefix: &str, avoid: &[&Ex]) -> Ex {
    let mut n = 0u32;
    loop {
        let name = if n == 0 {
            format!("_{prefix}")
        } else {
            format!("_{prefix}{n}")
        };
        let s = ctx.symbol(&name);
        if !avoid.iter().any(|e| e.contains(&s)) {
            return s;
        }
        n += 1;
    }
}

fn not_implemented(msg: impl Into<String>) -> SymplexError {
    SymplexError::NotImplemented(msg.into())
}

impl Distribution {
    /// A distribution from any [`Family`].
    pub fn from_family(family: impl Family) -> Self {
        Distribution(Arc::new(family))
    }

    /// The family behind the distribution.
    pub fn family(&self) -> &dyn Family {
        self.0.as_ref()
    }

    /// The concrete family, when it is a `T`.
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::stats::{Distribution, Normal};
    ///
    /// let ctx = Context::new();
    /// let d = Distribution::normal(ctx.int(0), ctx.int(2));
    /// let n = d.downcast_ref::<Normal>().unwrap();
    /// assert_eq!(n.std, ctx.int(2));
    /// ```
    pub fn downcast_ref<T: Family>(&self) -> Option<&T> {
        (self.0.as_ref() as &dyn Any).downcast_ref::<T>()
    }

    /// The context the parameters live in.
    pub fn context(&self) -> Context {
        self.0.context()
    }

    /// The family's name (`"Normal"`, `"Binomial"`, `"Truncated"`, …).
    pub fn name(&self) -> &str {
        self.0.name()
    }

    /// The parameters, named.
    pub fn parameters(&self) -> Vec<(&'static str, Ex)> {
        self.0.parameters()
    }

    /// The support.
    pub fn support(&self) -> Support {
        self.0.support()
    }

    /// Continuous or discrete.
    pub fn kind(&self) -> Kind {
        self.0.support().kind()
    }

    /// `true` for a continuous family.
    pub fn is_continuous(&self) -> bool {
        self.kind() == Kind::Continuous
    }

    /// The density (continuous) or probability mass function (discrete) as
    /// an expression in `x`.  Outside the support the value is `0`
    /// mathematically; the returned expression is the formula on the
    /// support, so integrate it over [`support`](Self::support).  SymPy:
    /// `density(X)(x)`.
    pub fn density(&self, x: &Ex) -> Ex {
        self.0.density(x)
    }

    /// The parameters' expressions (for symbol-avoidance).
    fn param_exprs(&self) -> Vec<Ex> {
        self.0.parameters().into_iter().map(|(_, e)| e).collect()
    }

    /// A bound variable that occurs in no parameter and in none of `also`.
    pub(crate) fn fresh_var(&self, prefix: &str, also: &[&Ex]) -> Ex {
        let params = self.param_exprs();
        let mut avoid: Vec<&Ex> = params.iter().collect();
        avoid.extend_from_slice(also);
        fresh_symbol(&self.context(), prefix, &avoid)
    }

    // ── Moments ────────────────────────────────────────────────────────

    /// `E[X]`: the closed form when the family has one, otherwise the
    /// integral / sum of `x·f(x)` over the support (which may stay
    /// unevaluated, or be a divergent integral, for a family without a
    /// mean).  SymPy: `E(X)`.
    pub fn mean(&self) -> Ex {
        match self.0.mean() {
            Some(m) => m,
            None => {
                let x = self.fresh_var("x", &[]);
                self.expectation(&x, &x)
            }
        }
    }

    /// `Var[X]`: the closed form, else `E[X²] − E[X]²`.  SymPy:
    /// `variance(X)`.
    pub fn variance(&self) -> Ex {
        match self.0.variance() {
            Some(v) => v,
            None => {
                let m = self.mean();
                (self.moment(2) - m.powi(2)).simplify()
            }
        }
    }

    /// Standard deviation `√Var[X]`.  SymPy: `std(X)`.
    pub fn std(&self) -> Ex {
        self.variance().sqrt()
    }

    /// The `n`-th raw moment `E[Xⁿ]`: the closed form, else the
    /// expectation of `xⁿ`.  SymPy: `moment(X, n)`.
    pub fn moment(&self, n: u32) -> Ex {
        if n == 0 {
            return self.context().one();
        }
        match self.0.raw_moment(n) {
            Some(m) => m,
            None => {
                let x = self.fresh_var("x", &[]);
                let integrand = x.powi(i64::from(n)) * self.0.density(&x);
                self.integrate_over(&integrand, &x, &self.support())
            }
        }
    }

    /// The `n`-th central moment `E[(X − μ)ⁿ]`.  SymPy: `cmoment(X, n)`.
    pub fn central_moment(&self, n: u32) -> Ex {
        let mu = self.mean();
        let x = self.fresh_var("x", &[&mu]);
        self.expectation(&(&x - mu).powi(i64::from(n)), &x)
    }

    /// Skewness `E[(X − μ)³] / σ³`.  SymPy: `skewness(X)`.
    pub fn skewness(&self) -> Ex {
        (self.central_moment(3) / self.std().powi(3)).simplify()
    }

    /// Kurtosis `E[(X − μ)⁴] / σ⁴` (not excess).  SymPy: `kurtosis(X)`.
    pub fn kurtosis(&self) -> Ex {
        (self.central_moment(4) / self.variance().powi(2)).simplify()
    }

    // ── Distribution functions ─────────────────────────────────────────

    /// The closed-form CDF on the support, if the family has one, else
    /// the integral / sum of the density from the support's lower end.
    fn cdf_on_support(&self, x: &Ex) -> Ex {
        if let Some(c) = self.0.cdf(x) {
            return c;
        }
        let support = self.support();
        if let Some(values) = support.as_points() {
            // Σ pᵢ · [vᵢ ≤ x]: decided where possible, else a Heaviside.
            let mut acc = self.context().zero();
            for v in &values {
                let p = self.0.density(v);
                match (x - v).is_nonnegative() {
                    Some(true) => acc += p,
                    Some(false) => {}
                    None => acc += p * (x - v).heaviside(),
                }
            }
            return acc.simplify();
        }
        let t = self.fresh_var("t", &[x]);
        let dens = self.0.density(&t);
        let ctx = self.context();
        let lo = match support.as_interval() {
            Some((lo, _, _, _)) => lo.clone(),
            None => ctx.neg_infinity(),
        };
        match support.kind() {
            Kind::Continuous => dens.integrate_definite(&t, &lo, x),
            Kind::Discrete => dens.summation(&t, &lo, &x.floor()),
        }
    }

    /// Cumulative distribution function `P(X ≤ x)` as an expression in
    /// `x`, on the whole line: the family's closed form (else integration
    /// / summation from the support's lower end), clamped to `0` below the
    /// support and `1` above it.  A numeric `x` gets the branch decided;
    /// a symbolic one a `Piecewise`.  SymPy: `cdf(X)(x)`.
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::stats::Distribution;
    ///
    /// let ctx = Context::new();
    /// let u = Distribution::uniform(ctx.int(0), ctx.int(1));
    /// assert_eq!(u.cdf(&ctx.rational(1, 3)), ctx.rational(1, 3));
    /// assert_eq!(u.cdf(&ctx.int(3)), ctx.int(1));
    /// assert_eq!(u.cdf(&ctx.int(-2)), ctx.int(0));
    /// ```
    pub fn cdf(&self, x: &Ex) -> Ex {
        let ctx = self.context();
        let support = self.support();
        let Some((lo, hi, _, _)) = support.as_interval() else {
            // Points: the indicator sum is already total.
            return self.cdf_on_support(x);
        };
        let lo_finite = !is_neg_inf(lo);
        let hi_finite = !is_pos_inf(hi);
        // Decide the branch for a numeric argument.
        if lo_finite && (x - lo).is_negative() == Some(true) {
            return ctx.zero();
        }
        if hi_finite && (x - hi).is_nonnegative() == Some(true) {
            return ctx.one();
        }
        let on = self.cdf_on_support(x);
        if (!lo_finite || (x - lo).is_nonnegative() == Some(true))
            && (!hi_finite || (hi - x).is_positive() == Some(true))
        {
            // A numeric argument: fold `floor(2)`, `(3/4)^2`, … so the
            // value reads as a number.
            return if x.free_symbols().is_empty() {
                on.eval()
            } else {
                on
            };
        }
        // Symbolic argument: a piecewise on the whole line.
        let mut pairs: Vec<(Ex, crate::api::expr::BoolEx)> = Vec::new();
        if lo_finite {
            pairs.push((ctx.zero(), x.lt(lo)));
        }
        if hi_finite {
            pairs.push((on, x.lt(hi)));
            pairs.push((ctx.one(), ctx.bool_true()));
        } else {
            pairs.push((on, ctx.bool_true()));
        }
        let refs: Vec<(&Ex, &crate::api::expr::BoolEx)> =
            pairs.iter().map(|(a, b)| (a, b)).collect();
        Ex::piecewise(&refs)
    }

    /// Moment generating function `E[e^{tX}]` in `t`: the closed form,
    /// else the expectation.  SymPy: `moment_generating_function(X)(t)`.
    pub fn mgf(&self, t: &Ex) -> Ex {
        match self.0.mgf(t) {
            Some(m) => m,
            None => {
                let x = self.fresh_var("x", &[t]);
                self.expectation(&(t * &x).exp(), &x)
            }
        }
    }

    /// Characteristic function `E[e^{itX}]` in `t`.  SymPy:
    /// `characteristic_function(X)(t)`.
    pub fn characteristic_function(&self, t: &Ex) -> Ex {
        let it = self.context().i_unit() * t;
        match self.0.mgf(&it) {
            Some(m) => m,
            None => {
                let x = self.fresh_var("x", &[t]);
                self.expectation(&(&it * &x).exp(), &x)
            }
        }
    }

    /// Quantile function (inverse CDF) at `p`, when the family has a
    /// closed form.  SymPy: `quantile(X)(p)`.
    pub fn quantile(&self, p: &Ex) -> Option<Ex> {
        self.0.quantile(p)
    }

    /// The median, when the quantile function has a closed form.  SymPy:
    /// `median(X)`.
    pub fn median(&self) -> Option<Ex> {
        self.quantile(&self.context().rational(1, 2))
    }

    /// Entropy in nats — differential (`−E[ln f(X)]`) for a continuous
    /// family, Shannon (`−Σ p ln p`) for a discrete one — the family's
    /// closed form when it has one, else the expectation of `−ln f` with
    /// the logarithm expanded (`ln(ab) → ln a + ln b`, `ln eᵘ → u`; the
    /// density is positive on the support).  SymPy: `entropy(X)`.
    pub fn entropy(&self) -> Ex {
        if let Some(h) = self.0.entropy() {
            return h;
        }
        let x = self.fresh_var("x", &[]);
        let neg_log_density = (-self.0.density(&x).ln().expand_log()).simplify();
        self.expectation(&neg_log_density, &x)
    }

    // ── Expectations and probabilities ─────────────────────────────────

    /// `E[g(X)]` for an expression `g` in the symbol `x`: a polynomial `g`
    /// with `x`-free coefficients goes through the family's closed-form raw
    /// moments (exact and cheap); anything else is `g·density` integrated
    /// (continuous) or summed (discrete) over the support.  The result may
    /// contain an unevaluated `Integral` or `Sum` when the crate's
    /// integrator cannot close the form — check with
    /// [`has_unevaluated`](Ex::has_unevaluated).  SymPy: `E(expr)`.
    pub fn expectation(&self, g: &Ex, x: &Ex) -> Ex {
        if let Some(by_moments) = self.expectation_by_moments(g, x) {
            return by_moments;
        }
        let integrand = g * self.0.density(x);
        self.integrate_over(&integrand, x, &self.support())
    }

    /// `E[g(X)]` through the raw moments, when `g` is a polynomial in `x`
    /// and every needed moment has a closed form.
    fn expectation_by_moments(&self, g: &Ex, x: &Ex) -> Option<Ex> {
        let poly = Poly::new(&g.expand(), &[x])?;
        let mut acc = self.context().zero();
        for (exps, coeff) in poly.terms() {
            let n = *exps.first()?;
            let m = if n == 0 {
                self.context().one()
            } else {
                self.0.raw_moment(n)?
            };
            acc += coeff * m;
        }
        Some(acc.simplify())
    }

    /// `P(X ∈ region)` for a region of the line (an event's set, from
    /// [`RandomVariable::probability`](super::RandomVariable::probability)
    /// or built directly with [`Support`]'s constructors): the region is
    /// clipped to the support piece by piece, and each piece is measured
    /// through the closed-form CDF when the family has one, else by exact
    /// integration / summation.
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::stats::{Distribution, Support};
    ///
    /// let ctx = Context::new();
    /// let y = Distribution::binomial(ctx.int(5), ctx.rational(1, 3));
    /// // P(Y > 2) = P(Y ∈ (2, ∞)) = 17/81
    /// let region = Support::from_pieces(
    ///     symplex::stats::Kind::Continuous,
    ///     vec![symplex::stats::Piece::Interval {
    ///         lo: ctx.int(2), hi: ctx.infinity(), lo_open: true, hi_open: true,
    ///     }],
    /// );
    /// assert_eq!(y.probability_of(&region)?, ctx.rational(17, 81));
    /// # Ok::<(), SymplexError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`SymplexError::NotImplemented`] when a point's membership in the
    /// support cannot be decided (symbolic table values).
    pub fn probability_of(&self, region: &Support) -> Result<Ex, SymplexError> {
        if let Some(direct) = self.0.probability_of(region) {
            return direct;
        }
        let ctx = self.context();
        let support = self.support();
        let clipped = support.intersect(region).ok_or_else(|| {
            not_implemented(format!(
                "probability on {support}: cannot decide which listed values lie in {region}"
            ))
        })?;
        let mut total = ctx.zero();
        for piece in clipped.pieces() {
            total += match piece {
                Piece::Interval {
                    lo,
                    hi,
                    lo_open,
                    hi_open,
                } => self.interval_mass(lo, hi, *lo_open, *hi_open, &support),
                Piece::Point(v) => match support.kind() {
                    Kind::Continuous => ctx.zero(),
                    Kind::Discrete => self.0.density(v).eval(),
                },
            };
        }
        Ok(total.simplify())
    }

    /// `P(lo ≤ X ≤ hi)` for an interval already clipped to the support.
    fn interval_mass(
        &self,
        lo: &Ex,
        hi: &Ex,
        lo_open: bool,
        hi_open: bool,
        support: &Support,
    ) -> Ex {
        let ctx = self.context();
        let (slo, shi) = match support.as_interval() {
            Some((a, b, _, _)) => (Some(a), Some(b)),
            None => (None, None),
        };
        // F at the support's own lower end is 0 and at its upper end 1 by
        // definition (the closed forms need not fold there: Φ(ln 0)…).
        let at_lower_end = is_neg_inf(lo) || slo.is_some_and(|s| (lo - s).is_zero() == Some(true));
        let at_upper_end = is_pos_inf(hi) || shi.is_some_and(|s| (hi - s).is_zero() == Some(true));
        let probe = self.fresh_var("t", &[lo, hi]);
        if self.0.cdf(&probe).is_some() {
            let upper = if at_upper_end {
                Some(ctx.one())
            } else {
                match support.kind() {
                    Kind::Continuous => self.0.cdf(hi),
                    // hi is an integer after lattice normalisation: P(X ≤ hi).
                    Kind::Discrete => self.0.cdf(hi),
                }
            };
            let lower = if at_lower_end {
                Some(ctx.zero())
            } else {
                match support.kind() {
                    Kind::Continuous => self.0.cdf(lo),
                    // P(X < lo) = F(lo − 1) on the integer lattice.
                    Kind::Discrete => self.0.cdf(&(lo - ctx.one())),
                }
            };
            if let (Some(u), Some(l)) = (upper, lower) {
                return (u - l).simplify();
            }
        }
        let _ = (lo_open, hi_open); // strictness is already folded into integer ends / irrelevant
        let t = probe;
        let dens = self.0.density(&t);
        match support.kind() {
            Kind::Continuous => dens.integrate_definite(&t, lo, hi).simplify(),
            Kind::Discrete => dens.summation(&t, lo, hi).simplify(),
        }
    }

    /// `E[g(X)·1_{X ∈ region}]` for `g` in the symbol `x`: `g·density`
    /// over the support clipped to the region — the numerator of a
    /// conditional expectation.
    ///
    /// # Errors
    ///
    /// As [`probability_of`](Self::probability_of).
    pub fn expectation_over(&self, g: &Ex, x: &Ex, region: &Support) -> Result<Ex, SymplexError> {
        if let Some(direct) = self.0.expectation_over(g, x, region) {
            return direct;
        }
        let support = self.support();
        let clipped = support.intersect(region).ok_or_else(|| {
            not_implemented(format!(
                "expectation on {support}: cannot decide which listed values lie in {region}"
            ))
        })?;
        let integrand = g * self.0.density(x);
        Ok(self.integrate_over(&integrand, x, &clipped))
    }

    /// Integrate (continuous) or sum (discrete) an expression in `x` over
    /// every piece of `region`.  The integrand is simplified first so that
    /// products such as `eˣ · e^{−x²/2}` reach the integrator as one
    /// exponential.
    pub(crate) fn integrate_over(&self, integrand: &Ex, x: &Ex, region: &Support) -> Ex {
        let ctx = self.context();
        let integrand = integrand.simplify();
        // Integer ends for a lattice region (open ends moved inwards).
        let region = region.normalize_lattice();
        let mut acc = ctx.zero();
        for piece in region.pieces() {
            acc += match piece {
                Piece::Interval { lo, hi, .. } => match region.kind() {
                    Kind::Continuous => integrand.integrate_definite(x, lo, hi),
                    Kind::Discrete => integrand.summation(x, lo, hi),
                },
                // The integrand at a listed value: for a table the mass
                // function is a `Piecewise`, which `eval` resolves.
                Piece::Point(v) => integrand.subs(x, v).eval(),
            };
        }
        acc.simplify()
    }

    // ── Sampling ───────────────────────────────────────────────────────

    /// `n` samples as `f64`: the family's own route when it has one, else
    /// inverse transform sampling through the closed-form quantile
    /// (continuous) or the cumulative sums of the mass function over a
    /// finite support (discrete).  Parameters must evaluate numerically.
    /// SymPy: `sample(X, size=n)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::NotImplemented`] if the family has no route;
    /// [`SymplexError::Unevaluable`] if a parameter is symbolic.
    pub fn sample(&self, n: usize, rng: &mut Rng) -> Result<Vec<f64>, SymplexError> {
        let mut sampler = self.sampler()?;
        Ok((0..n).map(|_| sampler(rng)).collect())
    }

    /// One sample; see [`sample`](Self::sample).
    pub fn sample_one(&self, rng: &mut Rng) -> Result<f64, SymplexError> {
        Ok(self.sampler()?(rng))
    }

    /// A sampler for repeated draws (the quantile is compiled once).
    pub fn sampler(&self) -> Result<Sampler, SymplexError> {
        if let Some(own) = self.0.sampler() {
            return own;
        }
        let ctx = self.context();
        let support = self.support();
        match support.kind() {
            Kind::Continuous => {
                let p = self.fresh_var("p", &[]);
                let q = self.0.quantile(&p).ok_or_else(|| {
                    not_implemented(format!(
                        "sampling {}: no closed-form quantile function",
                        self.name()
                    ))
                })?;
                let name = p.to_string();
                let q = q.compile(&[name.as_str()])?;
                Ok(Box::new(move |rng: &mut Rng| q.call(&[rng.next_f64()])))
            }
            Kind::Discrete => {
                if let Some(values) = support.as_points() {
                    let mut cumulative = Vec::with_capacity(values.len());
                    let mut points = Vec::with_capacity(values.len());
                    let mut acc = 0.0;
                    for v in &values {
                        acc += self.0.density(v).eval_f64()?;
                        cumulative.push(acc);
                        points.push(v.eval_f64()?);
                    }
                    return Ok(Box::new(move |rng: &mut Rng| {
                        let u = rng.next_f64() * acc;
                        let idx = cumulative.partition_point(|c| *c < u);
                        points[idx.min(points.len().saturating_sub(1))]
                    }));
                }
                let Some((lo, hi, _, _)) = support.as_interval() else {
                    return Err(not_implemented(format!(
                        "sampling {}: the support is not a single integer range",
                        self.name()
                    )));
                };
                if is_neg_inf(lo) || is_pos_inf(hi) {
                    return Err(not_implemented(format!(
                        "sampling {}: no sampling route for an infinite discrete support",
                        self.name()
                    )));
                }
                let lo_v = lo.eval_f64()?;
                let hi_v = hi.eval_f64()?;
                if !(lo_v.is_finite() && hi_v.is_finite() && hi_v >= lo_v) {
                    return Err(not_implemented(format!(
                        "sampling {}: the support is not a finite integer range",
                        self.name()
                    )));
                }
                let x = self.fresh_var("x", &[]);
                let name = x.to_string();
                let pmf = self.0.density(&x).compile(&[name.as_str()])?;
                let mut values = Vec::new();
                let mut k = lo_v;
                while k <= hi_v && values.len() < 1_000_000 {
                    values.push(k);
                    k += 1.0;
                }
                let mut cumulative = Vec::with_capacity(values.len());
                let mut acc = 0.0;
                for &k in &values {
                    acc += pmf.call(&[k]);
                    cumulative.push(acc);
                }
                let _ = ctx;
                Ok(Box::new(move |rng: &mut Rng| {
                    let u = rng.next_f64() * acc;
                    let idx = cumulative.partition_point(|c| *c < u);
                    values[idx.min(values.len().saturating_sub(1))]
                }))
            }
        }
    }
}
