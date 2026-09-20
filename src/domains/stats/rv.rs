//! The random-variable type and the generic exact machinery every
//! distribution family plugs into.

use std::fmt;

use crate::api::context::Context;
use crate::api::expr::{BoolEx, Ex};
use crate::base::errors::SymplexError;

use super::continuous::ContinuousFamily;
use super::discrete::DiscreteFamily;

/// Where a distribution lives: an interval of ℝ (continuous) or a set of
/// integers (discrete).  Bounds are expressions so that they may depend on
/// parameters (`Uniform(a, b)` has support `[a, b]`); `None` means
/// unbounded on that side.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Support {
    /// A continuous variable on `[lo, hi]` (either end may be infinite).
    Continuous {
        /// Lower end (`None` = −∞).
        lo: Option<Ex>,
        /// Upper end (`None` = +∞).
        hi: Option<Ex>,
    },
    /// A discrete variable on the integers `lo ..= hi` (either end may be
    /// infinite).
    Discrete {
        /// Lowest value (`None` = −∞).
        lo: Option<Ex>,
        /// Highest value (`None` = +∞).
        hi: Option<Ex>,
    },
    /// A discrete variable on an explicit finite list of values (not
    /// necessarily integers): a [`Finite`](super::DiscreteFamily::Finite)
    /// table.
    Finite(Vec<Ex>),
}

/// A probability distribution: one of the continuous or discrete
/// families.  Construct with the associated functions (`normal`,
/// `binomial`, …), which validate numeric parameters.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Distribution {
    /// A continuous family (density on an interval).
    Continuous(ContinuousFamily),
    /// A discrete family (probability mass on integers).
    Discrete(DiscreteFamily),
}

impl Distribution {
    /// The support.
    pub fn support(&self) -> Support {
        match self {
            Distribution::Continuous(f) => f.support(),
            Distribution::Discrete(f) => f.support(),
        }
    }

    /// The density (continuous) or probability mass function (discrete) as
    /// an expression in `var`.  Outside the support the value is `0`
    /// mathematically; the returned expression is the formula on the
    /// support only, so integrate it over [`support`](Self::support).
    pub fn density(&self, var: &Ex) -> Ex {
        match self {
            Distribution::Continuous(f) => f.density(var),
            Distribution::Discrete(f) => f.pmf(var),
        }
    }

    /// `true` for a continuous family.
    pub fn is_continuous(&self) -> bool {
        matches!(self, Distribution::Continuous(_))
    }

    /// Closed-form mean, if the family has one.
    pub fn mean(&self, ctx: &Context) -> Option<Ex> {
        match self {
            Distribution::Continuous(f) => f.mean(ctx),
            Distribution::Discrete(f) => f.mean(ctx),
        }
    }

    /// Closed-form variance, if the family has one.
    pub fn variance(&self, ctx: &Context) -> Option<Ex> {
        match self {
            Distribution::Continuous(f) => f.variance(ctx),
            Distribution::Discrete(f) => f.variance(ctx),
        }
    }

    /// Closed-form raw moment `E[Xⁿ]`, if the family has one.
    pub fn raw_moment(&self, n: u32, ctx: &Context) -> Option<Ex> {
        match self {
            Distribution::Continuous(f) => f.raw_moment(n, ctx),
            Distribution::Discrete(f) => f.raw_moment(n, ctx),
        }
    }

    /// Closed-form cumulative distribution function `P(X ≤ var)`, if the
    /// family has one (as an expression valid on the support).
    pub fn cdf(&self, var: &Ex) -> Option<Ex> {
        match self {
            Distribution::Continuous(f) => f.cdf(var),
            Distribution::Discrete(f) => f.cdf(var),
        }
    }

    /// Closed-form moment generating function `E[e^{tX}]` in `t`, if the
    /// family has one.
    pub fn mgf(&self, t: &Ex) -> Option<Ex> {
        match self {
            Distribution::Continuous(f) => f.mgf(t),
            Distribution::Discrete(f) => f.mgf(t),
        }
    }

    /// Closed-form quantile function (inverse CDF) at `p`, if the family
    /// has one.
    pub fn quantile(&self, p: &Ex) -> Option<Ex> {
        match self {
            Distribution::Continuous(f) => f.quantile(p),
            Distribution::Discrete(f) => f.quantile(p),
        }
    }

    /// The family's name (`"Normal"`, `"Binomial"`, …).
    pub fn name(&self) -> &'static str {
        match self {
            Distribution::Continuous(f) => f.name(),
            Distribution::Discrete(f) => f.name(),
        }
    }
}

impl fmt::Display for Distribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Distribution::Continuous(d) => write!(f, "{d}"),
            Distribution::Discrete(d) => write!(f, "{d}"),
        }
    }
}

/// A random variable: a symbol with a distribution.  Expressions in the
/// symbol are random quantities; the queries below evaluate them exactly.
///
/// See the [module docs](super) for an overview and examples.
#[derive(Clone, Debug)]
pub struct RandomVariable {
    ctx: Context,
    symbol: Ex,
    dist: Distribution,
}

impl RandomVariable {
    /// A random variable named `name` with distribution `dist`.  The name
    /// becomes a symbol in `ctx`; use [`symbol`](Self::symbol) to build
    /// expressions in it.
    pub fn new(ctx: &Context, name: &str, dist: Distribution) -> Self {
        RandomVariable {
            ctx: ctx.clone(),
            symbol: ctx.symbol(name),
            dist,
        }
    }

    /// The variable's symbol.
    pub fn symbol(&self) -> &Ex {
        &self.symbol
    }

    /// The distribution.
    pub fn distribution(&self) -> &Distribution {
        &self.dist
    }

    /// The context the variable lives in.
    pub fn context(&self) -> &Context {
        &self.ctx
    }

    /// The support of the variable.
    pub fn support(&self) -> Support {
        self.dist.support()
    }

    /// The density (continuous) or probability mass function (discrete) as
    /// an expression in `var`.  SymPy: `density(X)(x)`.
    pub fn density(&self, var: &Ex) -> Ex {
        self.dist.density(var)
    }

    /// `E[X]`: the closed form when the family has one, otherwise
    /// [`expectation`](Self::expectation) of the symbol.  SymPy: `E(X)`.
    pub fn mean(&self) -> Ex {
        match self.dist.mean(&self.ctx) {
            Some(m) => m,
            None => self.expectation(&self.symbol),
        }
    }

    /// `Var[X]`: the closed form when the family has one, otherwise
    /// `E[X²] − E[X]²`.  SymPy: `variance(X)`.
    pub fn variance(&self) -> Ex {
        match self.dist.variance(&self.ctx) {
            Some(v) => v,
            None => {
                let m = self.mean();
                (self.expectation(&self.symbol.powi(2)) - m.powi(2)).simplify()
            }
        }
    }

    /// Standard deviation `√Var[X]`.  SymPy: `std(X)`.
    pub fn std(&self) -> Ex {
        self.variance().sqrt()
    }

    /// The `n`-th raw moment `E[Xⁿ]`: the family's closed form, else the
    /// expectation of `Xⁿ`.  SymPy: `moment(X, n)`.
    pub fn moment(&self, n: u32) -> Ex {
        match self.dist.raw_moment(n, &self.ctx) {
            Some(m) => m,
            None => self.integrate_over_support(
                &(self.symbol.powi(i64::from(n)) * self.dist.density(&self.symbol)),
            ),
        }
    }

    /// The `n`-th central moment `E[(X − μ)ⁿ]`.  SymPy: `cmoment(X, n)`.
    pub fn central_moment(&self, n: u32) -> Ex {
        let mu = self.mean();
        self.expectation(&(&self.symbol - mu).powi(i64::from(n)))
    }

    /// Skewness `E[(X − μ)³] / σ³`.  SymPy: `skewness(X)`.
    pub fn skewness(&self) -> Ex {
        (self.central_moment(3) / self.std().powi(3)).simplify()
    }

    /// Kurtosis `E[(X − μ)⁴] / σ⁴` (not excess).  SymPy: `kurtosis(X)`.
    pub fn kurtosis(&self) -> Ex {
        (self.central_moment(4) / self.variance().powi(2)).simplify()
    }

    /// Cumulative distribution function `P(X ≤ var)` as an expression in
    /// `var`: the family's closed form, else the integral / sum of the
    /// density over the support up to `var`.  SymPy: `cdf(X)(x)`.
    pub fn cdf(&self, var: &Ex) -> Ex {
        if let Some(c) = self.dist.cdf(var) {
            return c;
        }
        let t = self.ctx.symbol("_t_cdf");
        let dens = self.dist.density(&t);
        match self.dist.support() {
            Support::Continuous { lo, .. } => {
                let lo = lo.unwrap_or_else(|| self.ctx.neg_infinity());
                dens.integrate_definite(&t, &lo, var)
            }
            Support::Discrete { lo, .. } => {
                let lo = lo.unwrap_or_else(|| self.ctx.neg_infinity());
                dens.summation(&t, &lo, var)
            }
            Support::Finite(values) => {
                // Σ pᵢ · [vᵢ ≤ var], as a piecewise-free sum of indicator
                // terms when `var` is symbolic: use Heaviside(var − vᵢ).
                let mut acc = self.ctx.zero();
                for v in &values {
                    let p = self.dist.density(v);
                    match (var - v).is_nonnegative() {
                        Some(true) => acc += p,
                        Some(false) => {}
                        None => acc += p * (var - v).heaviside(),
                    }
                }
                acc.simplify()
            }
        }
    }

    /// Moment generating function `E[e^{tX}]` in `t`: the closed form, else
    /// the expectation.  SymPy: `moment_generating_function(X)(t)`.
    pub fn mgf(&self, t: &Ex) -> Ex {
        match self.dist.mgf(t) {
            Some(m) => m,
            None => self.expectation(&(t * &self.symbol).exp()),
        }
    }

    /// Characteristic function `E[e^{itX}]` in `t`.  SymPy:
    /// `characteristic_function(X)(t)`.
    pub fn characteristic_function(&self, t: &Ex) -> Ex {
        let it = self.ctx.i_unit() * t;
        match self.dist.mgf(&it) {
            Some(m) => m,
            None => self.expectation(&(&it * &self.symbol).exp()),
        }
    }

    /// Quantile function (inverse CDF) at `p`, when the family has a
    /// closed form.  SymPy: `quantile(X)(p)`.
    pub fn quantile(&self, p: &Ex) -> Option<Ex> {
        self.dist.quantile(p)
    }

    /// Entropy: the differential entropy `−E[ln f(X)]` of a continuous
    /// variable or the Shannon entropy `−Σ p ln p` of a discrete one — the
    /// family's closed form when it has one (`Normal`: `½ ln(2πeσ²)`),
    /// otherwise the generic expectation (see [`entropy`](super::entropy)).
    /// SymPy: `entropy(X)`.
    pub fn entropy(&self) -> Ex {
        match &self.dist {
            Distribution::Continuous(f) => {
                f.entropy(&self.ctx).unwrap_or_else(|| super::entropy(self))
            }
            Distribution::Discrete(_) => super::entropy(self),
        }
    }

    /// The median, when the quantile function has a closed form.  SymPy:
    /// `median(X)`.
    pub fn median(&self) -> Option<Ex> {
        self.quantile(&self.ctx.rational(1, 2))
    }

    /// `E[g(X)]` for an expression `g` in the variable's symbol: exact
    /// integration (continuous) or summation (discrete) of `g · density`
    /// over the support.  The result may contain an unevaluated `Integral`
    /// or `Sum` when the crate's integrator cannot close the form — check
    /// with [`has_unevaluated`](Ex::has_unevaluated).  SymPy: `E(expr)`.
    pub fn expectation(&self, g: &Ex) -> Ex {
        // A polynomial in X with X-free coefficients: Σ cₙ E[Xⁿ] through the
        // family's closed-form moments (exact and cheap).
        if let Some(by_moments) = self.expectation_by_moments(g) {
            return by_moments;
        }
        let integrand = g * self.dist.density(&self.symbol);
        self.integrate_over_support(&integrand)
    }

    /// `E[g(X)]` for a polynomial `g` in the symbol when every needed raw
    /// moment has a closed form; `None` otherwise (fall back to
    /// integration).
    fn expectation_by_moments(&self, g: &Ex) -> Option<Ex> {
        use crate::api::poly_ex::Poly;
        let poly = Poly::new(&g.expand(), &[&self.symbol])?;
        let mut acc = self.ctx.zero();
        for (exps, coeff) in poly.terms() {
            let n = *exps.first()?;
            let m = if n == 0 {
                self.ctx.one()
            } else {
                self.dist.raw_moment(n, &self.ctx)?
            };
            acc += coeff * m;
        }
        Some(acc.simplify())
    }

    /// `P(event)` for a relation or conjunction of relations in the
    /// variable's symbol, exactly: the density is integrated (summed) over
    /// the part of the support where the condition holds.  Supported
    /// conditions: `X < a`, `X ≤ a`, `X > a`, `X ≥ a`, `X = a` (zero for a
    /// continuous variable, the mass for a discrete one), and conjunctions
    /// of these (`a < X ∧ X < b`).  SymPy: `P(cond)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::NotImplemented`] for conditions of another shape
    /// (disjunctions, non-linear conditions in `X`).
    pub fn probability(&self, event: &BoolEx) -> Result<Ex, SymplexError> {
        let (lo, hi, lo_strict, hi_strict, point) = self.event_bounds(event)?;
        let support = self.dist.support();
        if let Support::Finite(values) = &support {
            return self.probability_on_finite(values, lo, hi, lo_strict, hi_strict, point);
        }
        if let Some(a) = point {
            return Ok(match self.dist {
                Distribution::Continuous(_) => self.ctx.zero(),
                Distribution::Discrete(_) => self.dist.density(&a).simplify(),
            });
        }
        let one = self.ctx.one();
        // Clip to the support.
        let (slo, shi) = match &support {
            Support::Continuous { lo, hi } | Support::Discrete { lo, hi } => {
                (lo.clone(), hi.clone())
            }
            Support::Finite(_) => (None, None),
        };
        let lo = match (lo, &slo) {
            (Some(a), Some(s)) => Some(a.max_with(s)),
            (Some(a), None) => Some(a),
            (None, s) => s.clone(),
        };
        let hi = match (hi, &shi) {
            (Some(b), Some(s)) => Some(b.min_with(s)),
            (Some(b), None) => Some(b),
            (None, s) => s.clone(),
        };
        let t = &self.symbol;
        let dens = self.dist.density(t);
        Ok(match self.dist {
            Distribution::Continuous(_) => {
                // Strictness does not matter for a continuous variable.
                // The family's closed-form CDF first (`F(hi) − F(lo)`, with
                // `F(±∞)` = 1 / 0 by definition), else exact integration.
                if let Some(p) = self.probability_from_cdf(lo.as_ref(), hi.as_ref()) {
                    return Ok(p);
                }
                let lo = lo.unwrap_or_else(|| self.ctx.neg_infinity());
                let hi = hi.unwrap_or_else(|| self.ctx.infinity());
                dens.integrate_definite(t, &lo, &hi).simplify()
            }
            Distribution::Discrete(_) => {
                // Integer bounds: a strict bound moves inwards by one; a
                // non-integer bound rounds inwards.
                let lo = lo.map(|a| {
                    let a = if lo_strict {
                        a.floor() + &one
                    } else {
                        a.ceiling()
                    };
                    a.simplify()
                });
                let hi = hi.map(|b| {
                    let b = if hi_strict {
                        b.ceiling() - &one
                    } else {
                        b.floor()
                    };
                    b.simplify()
                });
                let lo = lo.unwrap_or_else(|| self.ctx.neg_infinity());
                let hi = hi.unwrap_or_else(|| self.ctx.infinity());
                dens.summation(t, &lo, &hi).simplify()
            }
        })
    }

    /// `P(lo ≤ X ≤ hi)` through the family's closed-form CDF, when it has
    /// one and the bounds do not already exceed the support (the caller has
    /// clipped them).  `None` leaves the question to integration.
    fn probability_from_cdf(&self, lo: Option<&Ex>, hi: Option<&Ex>) -> Option<Ex> {
        let probe = self.ctx.symbol("_t_cdf");
        // Only families with a closed form (a `None` here means "integrate").
        self.dist.cdf(&probe)?;
        let upper = match hi {
            Some(h) => self.dist.cdf(h)?,
            None => self.ctx.one(),
        };
        let lower = match lo {
            Some(l) => self.dist.cdf(l)?,
            None => self.ctx.zero(),
        };
        Some((upper - lower).simplify())
    }

    /// Integrate or sum an expression in the symbol over the support.  The
    /// integrand is simplified first so that products such as
    /// `eˣ · e^{−x²/2}` reach the integrator as one exponential.
    fn integrate_over_support(&self, integrand: &Ex) -> Ex {
        let t = &self.symbol;
        let integrand = integrand.simplify();
        match self.dist.support() {
            Support::Continuous { lo, hi } => {
                let lo = lo.unwrap_or_else(|| self.ctx.neg_infinity());
                let hi = hi.unwrap_or_else(|| self.ctx.infinity());
                integrand.integrate_definite(t, &lo, &hi).simplify()
            }
            Support::Discrete { lo, hi } => {
                let lo = lo.unwrap_or_else(|| self.ctx.neg_infinity());
                let hi = hi.unwrap_or_else(|| self.ctx.infinity());
                integrand.summation(t, &lo, &hi).simplify()
            }
            Support::Finite(values) => {
                // The integrand already contains the pmf as a factor; on a
                // finite table `Σᵢ integrand(vᵢ)` is exact and needs no
                // summation engine — but `density(vᵢ)` inside the integrand
                // is a `Piecewise`, so evaluate `g(vᵢ)·pᵢ` directly instead.
                let mut acc = self.ctx.zero();
                for v in &values {
                    acc += integrand.subs(t, v).eval();
                }
                acc.simplify()
            }
        }
    }

    /// `P(event)` on an explicit finite table: sum the probabilities of the
    /// listed values that satisfy the bounds.  Bounds are compared exactly;
    /// an undecidable comparison (symbolic values) is `NotImplemented`.
    fn probability_on_finite(
        &self,
        values: &[Ex],
        lo: Option<Ex>,
        hi: Option<Ex>,
        lo_strict: bool,
        hi_strict: bool,
        point: Option<Ex>,
    ) -> Result<Ex, SymplexError> {
        let undecided = |v: &Ex| {
            SymplexError::NotImplemented(format!(
                "probability on a finite table: cannot decide whether `{v}` satisfies the event"
            ))
        };
        let mut acc = self.ctx.zero();
        for v in values {
            let keep = if let Some(a) = &point {
                v.equals(a).ok_or_else(|| undecided(v))?
            } else {
                let above = match &lo {
                    None => true,
                    Some(a) => {
                        let d = v - a;
                        let ok = if lo_strict {
                            d.is_positive()
                        } else {
                            d.is_nonnegative()
                        };
                        ok.ok_or_else(|| undecided(v))?
                    }
                };
                let below = match &hi {
                    None => true,
                    Some(b) => {
                        let d = b - v;
                        let ok = if hi_strict {
                            d.is_positive()
                        } else {
                            d.is_nonnegative()
                        };
                        ok.ok_or_else(|| undecided(v))?
                    }
                };
                above && below
            };
            if keep {
                acc += self.dist.density(v).eval();
            }
        }
        Ok(acc.simplify())
    }

    /// Decompose an event into `(lo, hi, lo_strict, hi_strict, point)` on
    /// the variable: `X > a` → `lo = a` strict, `X ≤ b` → `hi = b`,
    /// `X = a` → `point`.  Relations are stored canonically as `Gt`/`Ge`
    /// (`X < a` is `a > X`) and `Eq_`; `And` nests.
    #[allow(clippy::type_complexity)]
    fn event_bounds(
        &self,
        event: &BoolEx,
    ) -> Result<(Option<Ex>, Option<Ex>, bool, bool, Option<Ex>), SymplexError> {
        use crate::base::node::ExprNode;
        let unsupported = || {
            SymplexError::NotImplemented(format!(
                "probability of the event `{event}`: only relations `X < a`, `X ≤ a`, `X > a`, \
                 `X ≥ a`, `X = a` in the random variable and their conjunctions are supported"
            ))
        };
        // Flatten the event into relation nodes `(kind, l, r)` with
        // kind ∈ {Gt, Ge, Eq}: `l > r`, `l ≥ r`, `l = r`.
        #[derive(Clone, Copy)]
        enum Rel {
            Gt,
            Ge,
            Eq,
        }
        let root = self.symbol.checked_id(event);
        let mut rels: Vec<(Rel, Ex, Ex)> = Vec::new();
        {
            let inner = self.symbol.inner.read();
            let arena = &inner.arena;
            let mut stack = vec![root];
            while let Some(id) = stack.pop() {
                match arena.node(id) {
                    ExprNode::And(parts) => stack.extend(parts.iter().copied()),
                    ExprNode::Gt(l, r) => {
                        rels.push((Rel::Gt, self.symbol.wrap(*l), self.symbol.wrap(*r)))
                    }
                    ExprNode::Ge(l, r) => {
                        rels.push((Rel::Ge, self.symbol.wrap(*l), self.symbol.wrap(*r)))
                    }
                    ExprNode::Eq_(l, r) => {
                        rels.push((Rel::Eq, self.symbol.wrap(*l), self.symbol.wrap(*r)))
                    }
                    _ => return Err(unsupported()),
                }
            }
        }
        let x = &self.symbol;
        let mut lo: Option<Ex> = None;
        let mut hi: Option<Ex> = None;
        let mut lo_strict = false;
        let mut hi_strict = false;
        let mut point: Option<Ex> = None;
        let mut raise_lo = |a: Ex, strict: bool| {
            lo = Some(match lo.take() {
                Some(l0) => l0.max_with(&a),
                None => a,
            });
            lo_strict |= strict;
        };
        let mut lower_hi = |b: Ex, strict: bool| {
            hi = Some(match hi.take() {
                Some(h) => h.min_with(&b),
                None => b,
            });
            hi_strict |= strict;
        };
        for (kind, l, r) in rels {
            let (bound, x_on_left) = if l == *x && !r.contains(x) {
                (r, true)
            } else if r == *x && !l.contains(x) {
                (l, false)
            } else {
                return Err(unsupported());
            };
            match (kind, x_on_left) {
                // X > a / X ≥ a
                (Rel::Gt, true) => raise_lo(bound, true),
                (Rel::Ge, true) => raise_lo(bound, false),
                // a > X / a ≥ X
                (Rel::Gt, false) => lower_hi(bound, true),
                (Rel::Ge, false) => lower_hi(bound, false),
                (Rel::Eq, _) => point = Some(bound),
            }
        }
        Ok((lo, hi, lo_strict, hi_strict, point))
    }
}

impl fmt::Display for RandomVariable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ~ {}", self.symbol, self.dist)
    }
}
