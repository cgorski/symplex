//! Families built from other families: conditioning on a region
//! ([`Truncated`]), affine maps ([`Affine`]), general transformations by
//! the change-of-variables formula ([`Transformed`]) and finite mixtures
//! ([`Mixture`]).  Each is a [`Family`] whose closed forms are transported
//! from the inner distribution where the transport is exact, and whose
//! remaining queries go through the inner [`Distribution`]'s machinery.

use std::fmt;

use crate::api::context::Context;
use crate::api::expr::Ex;
use crate::base::errors::SymplexError;
use crate::base::interval::{Interval, IntervalKind};

use super::family::{Distribution, Family, Sampler, same_family};
use super::sample::Rng;
use super::support::{Kind, Piece, Support, is_neg_inf, is_pos_inf};

fn invalid(reason: impl Into<String>) -> SymplexError {
    SymplexError::invalid_argument("stats", reason)
}

fn not_implemented(reason: impl Into<String>) -> SymplexError {
    SymplexError::NotImplemented(reason.into())
}

// ═══════════════════════════════════════════════════════════════════════════
// Truncated
// ═══════════════════════════════════════════════════════════════════════════

/// The inner distribution conditioned on a region: density `f(x)/P(R)`
/// on `support ∩ R`.  Built by [`Distribution::truncated`] /
/// [`RandomVariable::given`](super::RandomVariable::given).  SymPy:
/// `given(X, cond)`.
#[derive(Clone, Debug, PartialEq)]
pub struct Truncated {
    /// The unconditioned distribution.
    pub inner: Distribution,
    /// The conditioning region (as a subset of ℝ).
    pub region: Support,
    /// `P_inner(region)`, the normalising mass.
    pub mass: Ex,
}

impl Truncated {
    /// The support after clipping.
    fn clipped(&self) -> Support {
        // The constructor verified this intersection is decidable.
        self.inner
            .support()
            .intersect(&self.region)
            .unwrap_or_else(|| self.inner.support())
    }
}

impl Family for Truncated {
    fn name(&self) -> &str {
        "Truncated"
    }

    fn context(&self) -> Context {
        self.inner.context()
    }

    fn parameters(&self) -> Vec<(&'static str, Ex)> {
        let mut p = self.inner.parameters();
        p.push(("mass", self.mass.clone()));
        p
    }

    fn eq_family(&self, other: &dyn Family) -> bool {
        same_family(self, other)
    }

    fn support(&self) -> Support {
        self.clipped()
    }

    fn density(&self, x: &Ex) -> Ex {
        self.inner.density(x) / &self.mass
    }

    // (F(x) − F(lo)) / mass on a single interval [lo, hi].
    fn cdf(&self, x: &Ex) -> Option<Ex> {
        let clipped = self.clipped();
        let lo = &clipped.as_interval()?.lower;
        let probe = self.inner.fresh_var("t", &[x]);
        self.inner.family().cdf(&probe)?;
        let ctx = self.context();
        let f_lo = if is_neg_inf(lo) {
            ctx.zero()
        } else {
            self.inner.family().cdf(lo)?
        };
        let f_x = self.inner.family().cdf(x)?;
        Some(((f_x - f_lo) / &self.mass).simplify())
    }

    // Q(F(lo) + p·mass) when the inner family has both closed forms.
    fn quantile(&self, p: &Ex) -> Option<Ex> {
        let clipped = self.clipped();
        let lo = &clipped.as_interval()?.lower;
        let ctx = self.context();
        let f_lo = if is_neg_inf(lo) {
            ctx.zero()
        } else {
            self.inner.family().cdf(lo)?
        };
        self.inner.family().quantile(&(f_lo + p * &self.mass))
    }

    fn probability_of(&self, region: &Support) -> Option<Result<Ex, SymplexError>> {
        // P(X ∈ A | X ∈ R) = P(X ∈ A ∩ R) / P(R): clip to R first, then
        // let the inner machinery measure.
        let both = self
            .clipped()
            .intersect(region)?
            .with_kind(Kind::Continuous);
        Some(
            self.inner
                .probability_of(&both)
                .map(|p| (p / &self.mass).simplify()),
        )
    }

    fn expectation_over(
        &self,
        g: &Ex,
        x: &Ex,
        region: &Support,
    ) -> Option<Result<Ex, SymplexError>> {
        let both = self
            .clipped()
            .intersect(region)?
            .with_kind(Kind::Continuous);
        Some(
            self.inner
                .expectation_over(g, x, &both)
                .map(|e| (e / &self.mass).simplify()),
        )
    }

    fn sampler(&self) -> Option<Result<Sampler, SymplexError>> {
        // Inverse transform through the truncated quantile when it exists;
        // otherwise rejection from the inner sampler against numeric bounds.
        let ctx = self.context();
        let p = self.inner.fresh_var("p", &[]);
        if let Some(q) = Family::quantile(self, &p) {
            let name = p.to_string();
            return Some(
                q.compile(&[name.as_str()]).map(|q| -> Sampler {
                    Box::new(move |rng: &mut Rng| q.call(&[rng.next_f64()]))
                }),
            );
        }
        let clipped = self.clipped();
        let Some(iv) = clipped.as_interval() else {
            return Some(Err(not_implemented(
                "sampling a truncated distribution needs a single-interval region",
            )));
        };
        let lo_v = if is_neg_inf(&iv.lower) {
            Ok(f64::NEG_INFINITY)
        } else {
            iv.lower.eval_f64()
        };
        let hi_v = if is_pos_inf(&iv.upper) {
            Ok(f64::INFINITY)
        } else {
            iv.upper.eval_f64()
        };
        // The same interval in `f64`, keeping which ends are excluded.
        let window = match (lo_v, hi_v) {
            (Ok(lower), Ok(upper)) => Interval {
                lower,
                upper,
                kind: iv.kind,
            },
            (Err(e), _) | (_, Err(e)) => return Some(Err(e)),
        };
        let mut inner = match self.inner.sampler() {
            Ok(s) => s,
            Err(e) => return Some(Err(e)),
        };
        let _ = ctx;
        Some(Ok(Box::new(move |rng: &mut Rng| {
            // The region has positive mass (validated at construction), so
            // the loop terminates; cap it anyway so a numerically empty
            // region cannot spin forever.
            for _ in 0..1_000_000 {
                let v = inner(rng);
                if window.contains(&v) {
                    return v;
                }
            }
            f64::NAN
        })))
    }

    fn fmt_display(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} | {}", self.inner, self.region)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Affine
// ═══════════════════════════════════════════════════════════════════════════

/// `Y = aX + b` for `a ≠ 0` of known sign.  Every closed form of `X` is
/// transported exactly: `E[Y] = aμ + b`, `Var[Y] = a²σ²`, raw moments by
/// the binomial expansion, `F_Y(y) = F_X((y−b)/a)` (mirrored for `a < 0`),
/// `M_Y(t) = e^{bt} M_X(at)`, `Q_Y(p) = aQ_X(p) + b` (or `aQ_X(1−p) + b`),
/// `H[Y] = H[X] + ln|a|` (continuous).  SymPy: `density(a*X + b)`.
#[derive(Clone, Debug, PartialEq)]
pub struct Affine {
    /// The inner distribution of `X`.
    pub inner: Distribution,
    /// Slope `a ≠ 0`.
    pub a: Ex,
    /// Intercept `b`.
    pub b: Ex,
    /// `true` when `a > 0`.
    pub increasing: bool,
}

impl Affine {
    /// `(y − b) / a`.
    fn inverse(&self, y: &Ex) -> Ex {
        (y - &self.b) / &self.a
    }

    /// The image of the inner support.
    fn image(&self) -> Support {
        let inner = self.inner.support();
        let map = |v: &Ex| (&self.a * v + &self.b).simplify();
        let map_end = |v: &Ex| {
            if is_pos_inf(v) || is_neg_inf(v) {
                let ctx = self.context();
                let positive = is_pos_inf(v) == self.increasing;
                if positive {
                    ctx.infinity()
                } else {
                    ctx.neg_infinity()
                }
            } else {
                map(v)
            }
        };
        let pieces = inner
            .pieces()
            .iter()
            .map(|p| match p {
                Piece::Point(v) => Piece::Point(map(v)),
                Piece::Interval(iv) => {
                    let image = iv.as_ref().map(map_end);
                    // A decreasing map swaps the ends and their openness.
                    Piece::Interval(if self.increasing {
                        image
                    } else {
                        image.reversed()
                    })
                }
            })
            .collect();
        Support::from_pieces(inner.kind(), pieces)
    }
}

impl Family for Affine {
    fn name(&self) -> &str {
        "Affine"
    }

    fn context(&self) -> Context {
        self.inner.context()
    }

    fn parameters(&self) -> Vec<(&'static str, Ex)> {
        let mut p = vec![("a", self.a.clone()), ("b", self.b.clone())];
        p.extend(self.inner.parameters());
        p
    }

    fn eq_family(&self, other: &dyn Family) -> bool {
        same_family(self, other)
    }

    fn support(&self) -> Support {
        self.image()
    }

    // f_X((y−b)/a) / |a| for a density; the mass function is carried over
    // point by point (the constructor only allows a = ±1 on a lattice).
    fn density(&self, y: &Ex) -> Ex {
        let inner = self.inner.density(&self.inverse(y));
        match self.inner.kind() {
            Kind::Continuous => inner / self.a.abs(),
            Kind::Discrete => inner,
        }
    }

    fn mean(&self) -> Option<Ex> {
        Some((&self.a * self.inner.family().mean()? + &self.b).simplify())
    }

    fn variance(&self) -> Option<Ex> {
        Some((self.a.powi(2) * self.inner.family().variance()?).simplify())
    }

    // E[(aX + b)ⁿ] = Σ_k C(n, k) aᵏ bⁿ⁻ᵏ E[Xᵏ]
    fn raw_moment(&self, n: u32) -> Option<Ex> {
        let ctx = self.context();
        let mut acc = ctx.zero();
        for k in 0..=n {
            let m = if k == 0 {
                ctx.one()
            } else {
                self.inner.family().raw_moment(k)?
            };
            acc += ctx.int(i64::from(n)).binomial(&ctx.int(i64::from(k)))
                * self.a.powi(i64::from(k))
                * self.b.powi(i64::from(n - k))
                * m;
        }
        Some(acc.simplify())
    }

    fn cdf(&self, y: &Ex) -> Option<Ex> {
        let x = self.inverse(y);
        if self.increasing {
            self.inner.family().cdf(&x)
        } else {
            let ctx = self.context();
            match self.inner.kind() {
                // P(aX + b ≤ y) = P(X ≥ x) = 1 − F(x⁻)
                Kind::Continuous => Some(ctx.one() - self.inner.family().cdf(&x)?),
                // On the lattice: 1 − F(x − 1).
                Kind::Discrete => Some(ctx.one() - self.inner.family().cdf(&(x - ctx.one()))?),
            }
        }
    }

    // e^{bt} M_X(at)
    fn mgf(&self, t: &Ex) -> Option<Ex> {
        Some((&self.b * t).exp() * self.inner.family().mgf(&(&self.a * t))?)
    }

    fn quantile(&self, p: &Ex) -> Option<Ex> {
        let q = if self.increasing {
            self.inner.family().quantile(p)?
        } else {
            self.inner.family().quantile(&(self.context().one() - p))?
        };
        Some((&self.a * q + &self.b).simplify())
    }

    fn entropy(&self) -> Option<Ex> {
        let h = self.inner.family().entropy()?;
        Some(match self.inner.kind() {
            Kind::Continuous => h + self.a.abs().ln(),
            Kind::Discrete => h,
        })
    }

    fn sampler(&self) -> Option<Result<Sampler, SymplexError>> {
        let a = match self.a.eval_f64() {
            Ok(v) => v,
            Err(e) => return Some(Err(e)),
        };
        let b = match self.b.eval_f64() {
            Ok(v) => v,
            Err(e) => return Some(Err(e)),
        };
        let mut inner = match self.inner.sampler() {
            Ok(s) => s,
            Err(e) => return Some(Err(e)),
        };
        Some(Ok(Box::new(move |rng: &mut Rng| a * inner(rng) + b)))
    }

    fn fmt_display(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}·[{}] + {}", self.a, self.inner, self.b)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Transformed
// ═══════════════════════════════════════════════════════════════════════════

/// `Y = g(X)` for a continuous `X` and a `g` with explicit inverse
/// branches: `f_Y(y) = Σ_i f_X(h_i(y)) |h_i′(y)|` (the change-of-variables
/// formula), on the image of the support.  Built by
/// [`Distribution::transformed`], which recognises a strictly monotone `g`
/// on the support (one branch), and `X²`, `|X|`, and even powers on a
/// support symmetric enough for the two-branch formula.  SymPy:
/// `density(g(X))`.
#[derive(Clone, Debug, PartialEq)]
pub struct Transformed {
    /// The inner distribution of `X`.
    pub inner: Distribution,
    /// The map, as an expression in `var`.
    pub map: Ex,
    /// The symbol `g` is written in.
    pub var: Ex,
    /// The inverse branches `h_i(y)` as expressions in `image_var`, each
    /// with the part of the inner support it covers.
    pub branches: Vec<(Ex, Support)>,
    /// The symbol the branches are written in.
    pub image_var: Ex,
    /// The support of `Y`.
    pub image: Support,
}

impl Transformed {
    /// `Some(true)` when `dh > 0` on the image, `Some(false)` when `dh < 0`,
    /// `None` when its sign cannot be decided at a sample interior point.
    fn branch_derivative_sign(&self, dh: &Ex) -> Option<bool> {
        let ctx = self.context();
        let iv = self.image.as_interval()?;
        let (lo, hi) = (&iv.lower, &iv.upper);
        // A point strictly inside the image.
        let probe = if is_neg_inf(lo) && is_pos_inf(hi) {
            ctx.one()
        } else if is_neg_inf(lo) {
            hi - ctx.one()
        } else if is_pos_inf(hi) {
            lo + ctx.one()
        } else {
            (lo + hi) / ctx.int(2)
        };
        let at = dh.subs(&self.image_var, &probe).simplify();
        match at.is_positive() {
            Some(true) => Some(true),
            Some(false) => match at.is_negative() {
                Some(true) => Some(false),
                _ => None,
            },
            None => None,
        }
    }
}

impl Family for Transformed {
    fn name(&self) -> &str {
        "Transformed"
    }

    fn context(&self) -> Context {
        self.inner.context()
    }

    fn parameters(&self) -> Vec<(&'static str, Ex)> {
        let mut p = vec![("map", self.map.clone())];
        p.extend(self.inner.parameters());
        p
    }

    fn eq_family(&self, other: &dyn Family) -> bool {
        same_family(self, other)
    }

    fn support(&self) -> Support {
        self.image.clone()
    }

    // LOTUS: E[Yⁿ] = E_X[g(X)ⁿ], which the inner distribution's moments or
    // integrator usually close more readily than the image density does
    // (`E[e^N] = e^{1/2}` through the Gaussian integral).
    fn raw_moment(&self, n: u32) -> Option<Ex> {
        let e = self
            .inner
            .expectation(&self.map.powi(i64::from(n)), &self.var);
        (!e.has_unevaluated()).then_some(e)
    }

    fn density(&self, y: &Ex) -> Ex {
        let mut acc = self.context().zero();
        for (h, _) in &self.branches {
            let h_y = h.subs(&self.image_var, y);
            let dh = h.diff(&self.image_var);
            // Each branch is monotone on the image, so |h′| is h′ or −h′
            // throughout: decide the sign at an interior point of the image
            // instead of carrying `abs`.
            let signed = match self.branch_derivative_sign(&dh) {
                Some(true) => dh,
                Some(false) => -dh,
                None => dh.abs(),
            };
            acc += self.inner.density(&h_y) * signed.subs(&self.image_var, y);
        }
        acc.simplify()
    }

    fn sampler(&self) -> Option<Result<Sampler, SymplexError>> {
        let name = self.var.to_string();
        let g = match self.map.compile(&[name.as_str()]) {
            Ok(g) => g,
            Err(e) => return Some(Err(e)),
        };
        let mut inner = match self.inner.sampler() {
            Ok(s) => s,
            Err(e) => return Some(Err(e)),
        };
        Some(Ok(Box::new(move |rng: &mut Rng| g.call(&[inner(rng)]))))
    }

    fn fmt_display(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} with {} ~ {}", self.map, self.var, self.inner)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Mixture
// ═══════════════════════════════════════════════════════════════════════════

/// A finite mixture `Σ wᵢ Fᵢ` of distributions of one kind.  Every query is
/// the weighted sum of the components' answers, so nothing is lost when
/// their supports overlap.
#[derive(Clone, Debug, PartialEq)]
pub struct Mixture {
    /// `(weight, component)` pairs; the weights sum to `1`.
    pub components: Vec<(Ex, Distribution)>,
}

impl Mixture {
    fn ctx(&self) -> Context {
        // The constructor guarantees at least one component.
        self.components
            .first()
            .map(|(_, d)| d.context())
            .unwrap_or_default()
    }
}

impl Family for Mixture {
    fn name(&self) -> &str {
        "Mixture"
    }

    fn context(&self) -> Context {
        self.ctx()
    }

    fn parameters(&self) -> Vec<(&'static str, Ex)> {
        self.components
            .iter()
            .flat_map(|(w, d)| {
                let mut p = vec![("weight", w.clone())];
                p.extend(d.parameters());
                p
            })
            .collect()
    }

    fn eq_family(&self, other: &dyn Family) -> bool {
        same_family(self, other)
    }

    // The union of the components' supports, as given (they may overlap;
    // every query below sums per component, never integrates the union).
    fn support(&self) -> Support {
        let kind = self
            .components
            .first()
            .map_or(Kind::Continuous, |(_, d)| d.kind());
        let pieces = self
            .components
            .iter()
            .flat_map(|(_, d)| d.support().pieces().to_vec())
            .collect();
        Support::from_pieces(kind, pieces)
    }

    // Σ wᵢ fᵢ(x)·[x ∈ Sᵢ]; the indicator is dropped when every component
    // has the same support.
    fn density(&self, x: &Ex) -> Ex {
        let ctx = self.ctx();
        let supports: Vec<Support> = self.components.iter().map(|(_, d)| d.support()).collect();
        let same = supports.windows(2).all(|w| w[0] == w[1]);
        let mut acc = ctx.zero();
        for ((w, d), s) in self.components.iter().zip(&supports) {
            let f = d.density(x);
            if same {
                acc += w * f;
            } else {
                let cond = s
                    .to_set(&ctx)
                    .to_condition(x)
                    .unwrap_or_else(|_| ctx.bool_true());
                let zero = ctx.zero();
                acc += w * Ex::piecewise(&[(&f, &cond), (&zero, &ctx.bool_true())]);
            }
        }
        acc.simplify()
    }

    fn mean(&self) -> Option<Ex> {
        let mut acc = self.ctx().zero();
        for (w, d) in &self.components {
            acc += w * d.family().mean()?;
        }
        Some(acc.simplify())
    }

    fn raw_moment(&self, n: u32) -> Option<Ex> {
        let mut acc = self.ctx().zero();
        for (w, d) in &self.components {
            acc += w * d.family().raw_moment(n)?;
        }
        Some(acc.simplify())
    }

    fn cdf(&self, x: &Ex) -> Option<Ex> {
        // Each component's whole-line CDF (clamped), weighted.
        let mut acc = self.ctx().zero();
        for (w, d) in &self.components {
            acc += w * d.cdf(x);
        }
        Some(acc.simplify())
    }

    fn mgf(&self, t: &Ex) -> Option<Ex> {
        let mut acc = self.ctx().zero();
        for (w, d) in &self.components {
            acc += w * d.family().mgf(t)?;
        }
        Some(acc.simplify())
    }

    fn probability_of(&self, region: &Support) -> Option<Result<Ex, SymplexError>> {
        let mut acc = self.ctx().zero();
        for (w, d) in &self.components {
            match d.probability_of(region) {
                Ok(p) => acc += w * p,
                Err(e) => return Some(Err(e)),
            }
        }
        Some(Ok(acc.simplify()))
    }

    fn expectation_over(
        &self,
        g: &Ex,
        x: &Ex,
        region: &Support,
    ) -> Option<Result<Ex, SymplexError>> {
        let mut acc = self.ctx().zero();
        for (w, d) in &self.components {
            match d.expectation_over(g, x, region) {
                Ok(e) => acc += w * e,
                Err(e) => return Some(Err(e)),
            }
        }
        Some(Ok(acc.simplify()))
    }

    fn sampler(&self) -> Option<Result<Sampler, SymplexError>> {
        let mut weights = Vec::with_capacity(self.components.len());
        let mut samplers = Vec::with_capacity(self.components.len());
        let mut acc = 0.0;
        for (w, d) in &self.components {
            match w.eval_f64() {
                Ok(v) => {
                    acc += v;
                    weights.push(acc);
                }
                Err(e) => return Some(Err(e)),
            }
            match d.sampler() {
                Ok(s) => samplers.push(s),
                Err(e) => return Some(Err(e)),
            }
        }
        Some(Ok(Box::new(move |rng: &mut Rng| {
            let u = rng.next_f64() * acc;
            let idx = weights.partition_point(|c| *c < u);
            let idx = idx.min(samplers.len().saturating_sub(1));
            samplers[idx](rng)
        })))
    }

    fn fmt_display(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Mixture(")?;
        for (i, (w, d)) in self.components.iter().enumerate() {
            if i > 0 {
                f.write_str(" + ")?;
            }
            write!(f, "{w}·[{d}]")?;
        }
        f.write_str(")")
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Constructors
// ═══════════════════════════════════════════════════════════════════════════

impl Distribution {
    /// The distribution conditioned on `X ∈ region`: density `f/P(R)` on
    /// `support ∩ R`.  SymPy: `given(X, cond)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if the region has probability zero
    /// (e.g. a point of a continuous distribution, or a region outside the
    /// support); [`SymplexError::NotImplemented`] if the clipping cannot
    /// be decided.
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::stats::{Distribution, Support};
    ///
    /// let ctx = Context::new();
    /// let n = Distribution::normal(ctx.int(0), ctx.int(1));
    /// // The half-normal: N | N > 0.
    /// let half = n.truncated(&Support::half_line(ctx.int(0)))?;
    /// assert_eq!(half.mean().equals(&(ctx.int(2) / ctx.pi()).sqrt()), Some(true));
    /// # Ok::<(), SymplexError>(())
    /// ```
    pub fn truncated(&self, region: &Support) -> Result<Distribution, SymplexError> {
        let mass = self.probability_of(region)?;
        if mass.is_zero() == Some(true) {
            return Err(invalid(format!(
                "cannot condition on a region of probability zero ({region})"
            )));
        }
        let clipped = self.support().intersect(region).ok_or_else(|| {
            not_implemented(format!(
                "conditioning {}: cannot decide which listed values lie in {region}",
                self
            ))
        })?;
        if clipped.is_empty() {
            return Err(invalid(format!(
                "cannot condition on a region outside the support ({region})"
            )));
        }
        Ok(Distribution::from_family(Truncated {
            inner: self.clone(),
            region: region.clone(),
            mass,
        }))
    }

    /// The distribution of `aX + b`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `a` is zero or of unknown sign
    /// (a symbolic `a` needs a `Positive`/`Negative` assumption), or if
    /// `X` is discrete on an infinite lattice and `a ≠ ±1` (the image is
    /// no longer the integer lattice).
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::stats::Distribution;
    ///
    /// let ctx = Context::new();
    /// let z = Distribution::normal(ctx.int(0), ctx.int(1));
    /// let y = z.affine(ctx.int(2), ctx.int(1))?;      // 2Z + 1 ~ Normal(1, 2)
    /// assert_eq!(y.mean(), ctx.int(1));
    /// assert_eq!(y.variance(), ctx.int(4));
    /// # Ok::<(), SymplexError>(())
    /// ```
    pub fn affine(&self, a: Ex, b: Ex) -> Result<Distribution, SymplexError> {
        let increasing = match a.is_positive() {
            Some(true) => true,
            Some(false) => {
                if a.is_negative() == Some(true) {
                    false
                } else {
                    return Err(invalid("the slope of an affine map must be non-zero"));
                }
            }
            None => {
                return Err(invalid(format!(
                    "the sign of the slope `{a}` must be known (give the symbol an assumption)"
                )));
            }
        };
        if self.kind() == Kind::Discrete && self.support().as_points().is_none() {
            let unit = a.abs().equals(&a.context().one());
            if unit != Some(true) {
                return Err(invalid(
                    "an affine map of a lattice distribution must have slope ±1 (use `transformed` on a finite table)",
                ));
            }
        }
        Ok(Distribution::from_family(Affine {
            inner: self.clone(),
            a,
            b,
            increasing,
        }))
    }

    /// The distribution of `g(X)` for an expression `g` in `x`, by the
    /// change-of-variables formula.  Recognised shapes: an affine `g`
    /// (delegates to [`affine`](Self::affine)); a continuous `X` with a
    /// strictly monotone `g` on the support whose inverse `solve` finds
    /// (`eˣ`, `ln x`, `1/x` on a positive support, `x³`, …); `X²`, `|X|`
    /// and even powers of a continuous `X` (two branches); and a finite
    /// table, whose values are mapped and merged.  SymPy: `density(g(X))`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::NotImplemented`] for other shapes (a non-monotone
    /// `g` that is not an even power / absolute value, a lattice `X`).
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::stats::Distribution;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let z = Distribution::normal(ctx.int(0), ctx.int(1));
    /// // Z² is χ²(1): its density is e^{−y/2} / √(2πy) (SymPy: density(Z**2)).
    /// let sq = z.transformed(&x, &x.powi(2))?;
    /// let y = ctx.symbol("y");
    /// let expected = (-&y / 2).exp() / (2 * ctx.pi() * &y).sqrt();
    /// assert_eq!((sq.density(&y) - expected).simplify(), ctx.int(0));
    /// assert_eq!(sq.mean(), ctx.int(1));
    /// # Ok::<(), SymplexError>(())
    /// ```
    pub fn transformed(&self, x: &Ex, g: &Ex) -> Result<Distribution, SymplexError> {
        let ctx = self.context();
        // Affine: a·x + b.
        if let Some(poly) = crate::api::poly_ex::Poly::new(&g.expand(), &[x])
            && poly.total_degree().unwrap_or(0) <= 1
        {
            let a = poly.coeff_monomial(&[1]).unwrap_or_else(|_| ctx.zero());
            let b = poly.coeff_monomial(&[0]).unwrap_or_else(|_| ctx.zero());
            return self.affine(a, b);
        }
        // A finite table (or a finite integer range, enumerated): map the
        // values and merge equal images.
        if let Some(values) = self
            .support()
            .as_points()
            .or_else(|| self.enumerate_lattice())
        {
            let mut table: Vec<(Ex, Ex)> = Vec::new();
            for v in &values {
                let image = g.subs(x, v).eval();
                let mass = self.density(v).eval();
                match table
                    .iter_mut()
                    .find(|(w, _)| w.equals(&image) == Some(true))
                {
                    Some((_, p)) => *p = (&*p + mass).simplify(),
                    None => table.push((image, mass)),
                }
            }
            return Ok(Distribution::finite(&ctx, table));
        }
        if self.kind() == Kind::Discrete {
            return Err(not_implemented(
                "transforming a lattice distribution: only affine maps with slope ±1 are supported",
            ));
        }
        let support = self.support();
        let iv = support.as_interval().ok_or_else(|| {
            not_implemented("transforming a distribution whose support is not one interval")
        })?;
        let (lo, hi) = (&iv.lower, &iv.upper);
        let y = super::family::fresh_symbol(&ctx, "y", &[g, x, lo, hi]);
        // Two-branch even shapes: x², |x|, x^{2k} on any support (the
        // branches are clipped to the support piece by piece).
        if let Some(branches) = even_branches(x, g, &y, &ctx) {
            let image = even_image(g, x, &support, &ctx);
            return Ok(Distribution::from_family(Transformed {
                inner: self.clone(),
                map: g.clone(),
                var: x.clone(),
                branches: branches.into_iter().map(|h| (h, support.clone())).collect(),
                image_var: y,
                image,
            }));
        }
        // Strictly monotone g on the (open) interior of the support: one
        // branch from `solve`.  A pole or kink at an endpoint (`1/x` at 0)
        // does not affect the distribution of g(X).
        let domain = ctx.interval(lo, hi, IntervalKind::Open);
        let inc = g.is_strictly_increasing(x, &domain);
        let dec = g.is_strictly_decreasing(x, &domain);
        let increasing = match (inc, dec) {
            (Some(true), _) => true,
            (_, Some(true)) => false,
            _ => {
                return Err(not_implemented(format!(
                    "transforming by `{g}`: the map is not known to be strictly monotone on {support} \
                     and is not an even power or absolute value"
                )));
            }
        };
        let solutions = (g - &y).solve(x).map_err(|e| {
            not_implemented(format!("transforming by `{g}`: no inverse found ({e})"))
        })?;
        let [h] = solutions.as_slice() else {
            return Err(not_implemented(format!(
                "transforming by `{g}`: the inverse is not a single branch"
            )));
        };
        // The image's ends are one-sided limits of g at the support's ends
        // (finite ends included: `1/x` at 0⁺ is ∞).
        let glo = if is_neg_inf(lo) {
            g.limit(x, lo)
        } else {
            g.limit_right(x, lo)
        };
        let ghi = if is_pos_inf(hi) {
            g.limit(x, hi)
        } else {
            g.limit_left(x, hi)
        };
        // The support's openness carries over to the image; a decreasing
        // map swaps the ends with it.  (`from_pieces` opens infinite ends.)
        let mapped = Interval {
            lower: glo,
            upper: ghi,
            kind: iv.kind,
        };
        let image = Support::from_pieces(
            Kind::Continuous,
            vec![Piece::Interval(if increasing {
                mapped
            } else {
                mapped.reversed()
            })],
        );
        Ok(Distribution::from_family(Transformed {
            inner: self.clone(),
            map: g.clone(),
            var: x.clone(),
            branches: vec![(h.clone(), support)],
            image_var: y,
            image,
        }))
    }

    /// The values of a discrete distribution on a finite numeric integer
    /// range (at most 100 000 of them); `None` otherwise.
    fn enumerate_lattice(&self) -> Option<Vec<Ex>> {
        if self.kind() != Kind::Discrete {
            return None;
        }
        let support = self.support();
        let iv = support.as_interval()?;
        let lo = iv.lower.eval().as_i64()?;
        let hi = iv.upper.eval().as_i64()?;
        if hi < lo || hi - lo > 100_000 {
            return None;
        }
        let ctx = self.context();
        Some((lo..=hi).map(|k| ctx.int(k)).collect())
    }

    /// A finite mixture `Σ wᵢ · Fᵢ`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if there are no components, the
    /// components' kinds differ, a numeric weight is negative, or all
    /// weights are numeric and do not sum to exactly `1`.
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::stats::Distribution;
    ///
    /// let ctx = Context::new();
    /// let a = Distribution::normal(ctx.int(-1), ctx.int(1));
    /// let b = Distribution::normal(ctx.int(3), ctx.int(2));
    /// let m = Distribution::mixture(&[(ctx.rational(1, 4), a), (ctx.rational(3, 4), b)])?;
    /// assert_eq!(m.mean(), ctx.int(2));                       // ¼·(−1) + ¾·3
    /// assert_eq!(m.moment(2), ctx.rational(41, 4));            // ¼·2 + ¾·13
    /// # Ok::<(), SymplexError>(())
    /// ```
    pub fn mixture(components: &[(Ex, Distribution)]) -> Result<Distribution, SymplexError> {
        let Some((_, first)) = components.first() else {
            return Err(invalid("a mixture needs at least one component"));
        };
        let kind = first.kind();
        let mut total = num_rational::Ratio::<num_bigint::BigInt>::from_integer(0.into());
        let mut all_numeric = true;
        for (w, d) in components {
            if d.kind() != kind {
                return Err(invalid(
                    "the components of a mixture must all be continuous or all be discrete",
                ));
            }
            if d.context().id != first.context().id {
                return Err(invalid(
                    "the components of a mixture must live in one context",
                ));
            }
            match w.eval().as_rational() {
                Some(q) => {
                    if q < num_rational::Ratio::from_integer(0.into()) {
                        return Err(invalid(format!("the weight `{w}` is negative")));
                    }
                    total += q;
                }
                None => all_numeric = false,
            }
        }
        if all_numeric && total != num_rational::Ratio::from_integer(1.into()) {
            return Err(invalid(format!("the weights sum to {total}, not 1")));
        }
        Ok(Distribution::from_family(Mixture {
            components: components.to_vec(),
        }))
    }
}

/// The inverse branches of an even shape `x²`, `|x|`, `x^{2k}` (as
/// expressions in `y`): `±√y`, `±y`, `±y^{1/(2k)}`.
fn even_branches(x: &Ex, g: &Ex, y: &Ex, ctx: &Context) -> Option<Vec<Ex>> {
    if *g == x.abs() {
        return Some(vec![y.clone(), -y]);
    }
    let poly = crate::api::poly_ex::Poly::new(g, &[x])?;
    let terms = poly.terms();
    let [(exps, coeff)] = terms.as_slice() else {
        return None;
    };
    let n = *exps.first()?;
    if n < 2 || n % 2 != 0 || coeff.equals(&ctx.one()) != Some(true) {
        return None;
    }
    let root = y.pow(&ctx.rational(1, i64::from(n)));
    Some(vec![root.clone(), -root])
}

/// The image of `support` under an even shape: `[0, max(g(lo), g(hi)))`,
/// or `[0, ∞)` when an end is infinite.
fn even_image(g: &Ex, x: &Ex, support: &Support, ctx: &Context) -> Support {
    let Some(iv) = support.as_interval() else {
        return Support::half_line(ctx.zero());
    };
    let (lo, hi) = (&iv.lower, &iv.upper);
    if is_neg_inf(lo) || is_pos_inf(hi) {
        return Support::half_line(ctx.zero());
    }
    let a = g.subs(x, lo).simplify();
    let b = g.subs(x, hi).simplify();
    let top = a.max_with(&b).simplify();
    // If the support does not contain 0 the minimum is min(g(lo), g(hi)).
    let contains_zero = support.contains(&ctx.zero());
    let bottom = if contains_zero == Some(false) {
        a.min_with(&b).simplify()
    } else {
        ctx.zero()
    };
    Support::interval(bottom, top)
}
