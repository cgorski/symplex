//! Families built from other families: conditioning on a region
//! ([`Truncated`]), affine maps ([`Affine`]), general transformations by
//! the change-of-variables formula ([`Transformed`]) and finite mixtures
//! ([`Mixture`]).  Each is a [`Family`] whose closed forms are transported
//! from the inner distribution where the transport is exact, and whose
//! remaining queries go through the inner [`Distribution`]'s machinery.

use std::fmt;

use num_traits::Signed;

use crate::api::context::Context;
use crate::api::expr::{BoolEx, Ex, SetEx};
use crate::base::errors::SymplexError;
use crate::base::interval::{Interval, IntervalKind};
use crate::base::node::{ExprId, ExprNode};
use crate::base::truth::{SignSet, Truth};

use super::family::{Distribution, Family, Sampler, TailOrders, family_boilerplate};
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
    family_boilerplate!(Truncated, "Truncated");

    fn context(&self) -> Context {
        self.inner.context()
    }

    fn parameters(&self) -> Vec<(&'static str, Ex)> {
        let mut p = self.inner.parameters();
        p.push(("mass", self.mass.clone()));
        p
    }

    fn support(&self) -> Support {
        self.clipped()
    }

    fn density(&self, x: &Ex) -> Ex {
        self.inner.density(x) / &self.mass
    }

    // (F(x) − F(lo⁻)) / mass on a single interval [lo, hi]; F(lo⁻) is the
    // inner `mass_below` (0 at the inner support's own end, F(lo − 1) on
    // the lattice: the atom at `lo` belongs to the truncated variable).  A
    // truncation in the inner far upper tail (`N | N > 10`) takes
    // (S(lo⁻) − S(x)) / mass instead: the two F are within 10⁻²³ of 1.
    fn cdf(&self, x: &Ex) -> Option<Ex> {
        let clipped = self.clipped();
        let lo = &clipped.as_interval()?.lower;
        let probe = self.inner.fresh_var("t", &[x]);
        self.inner.family().cdf(&probe)?;
        if self.inner.family().sf(&probe).is_some()
            && self.inner.in_far_upper_tail(lo)
            && let Some(s_lo) = self.inner.mass_above(lo)
        {
            return Some(((s_lo - self.inner.sf_on_support(x)) / &self.mass).simplify());
        }
        let f_lo = self.inner.mass_below(lo)?;
        let f_x = self.inner.cdf_on_support(x);
        Some(((f_x - f_lo) / &self.mass).simplify())
    }

    // (S(x) − S(hi)) / mass on [lo, hi]; S(hi) = 0 at the inner support's
    // upper end.
    fn sf(&self, x: &Ex) -> Option<Ex> {
        let clipped = self.clipped();
        let hi = &clipped.as_interval()?.upper;
        let probe = self.inner.fresh_var("t", &[x]);
        if self.inner.family().cdf(&probe).is_none() && self.inner.family().sf(&probe).is_none() {
            return None;
        }
        let inner_hi = self.inner.support().as_interval().map(|s| s.upper.clone());
        let s_hi = if is_pos_inf(hi)
            || inner_hi.is_some_and(|u| Truth::from((hi - &u).is_zero()).is_true())
        {
            self.context().zero()
        } else {
            self.inner.sf_on_support(hi)
        };
        Some(((self.inner.sf_on_support(x) - s_hi) / &self.mass).simplify())
    }

    // The inner domain, with an end freed where the truncation bounds the
    // support on that side (a bounded tail has every exponential moment);
    // an unbounded side keeps the inner tail, since the region is a finite
    // union of intervals and contains a whole neighbourhood of that infinity.
    fn mgf_domain(&self) -> Option<Interval<Ex>> {
        let d = self.inner.family().mgf_domain()?;
        let clipped = self.clipped();
        let iv = clipped.as_interval()?;
        let ctx = self.context();
        let (lo_free, hi_free) = (!is_neg_inf(&iv.lower), !is_pos_inf(&iv.upper));
        if lo_free && hi_free {
            return None;
        }
        let lower = if lo_free { ctx.neg_infinity() } else { d.lower };
        let upper = if hi_free { ctx.infinity() } else { d.upper };
        let kind = d
            .kind
            .with_lower_open(lo_free || d.kind.lower_open())
            .with_upper_open(hi_free || d.kind.upper_open());
        Some(Interval { lower, upper, kind })
    }

    // The inner tails that the region leaves unbounded.
    fn tail_orders(&self) -> Option<TailOrders> {
        let inner = self.inner.family().tail_orders()?;
        let clipped = self.clipped();
        let reaches = |inf: fn(&Ex) -> bool, lower: bool| {
            clipped.pieces().iter().any(|p| match p {
                Piece::Interval(iv) => inf(if lower { &iv.lower } else { &iv.upper }),
                Piece::Point(_) => false,
            })
        };
        Some(TailOrders {
            left: inner.left.filter(|_| reaches(is_neg_inf, true)),
            right: inner.right.filter(|_| reaches(is_pos_inf, false)),
        })
    }

    // Q(F(lo⁻) + p·mass) when the inner family has both closed forms.
    fn quantile(&self, p: &Ex) -> Option<Ex> {
        let clipped = self.clipped();
        let lo = &clipped.as_interval()?.lower;
        let f_lo = self.inner.mass_below(lo)?;
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

    /// A discrete inner law with listed values (a table, or a mixture with
    /// one), not only the integer lattice: `P(X ≥ x)` is not `P(X > ⌈x⌉ − 1)`
    /// there (up to 0.56 `−X` of a mixture of a binomial and the table
    /// `{9/8, 5/4}` counted the atom `9/8` in `P(X ≥ 5/4)`).
    fn listed_values(&self) -> bool {
        self.inner.kind() == Kind::Discrete
            && self
                .inner
                .support()
                .pieces()
                .iter()
                .any(|p| matches!(p, Piece::Point(_)))
    }

    /// `P(X = x)` for the discrete inner law, membership decided piece by
    /// piece; `None` when it cannot be (a symbolic `x`).
    fn atom_mass(&self, x: &Ex) -> Option<Ex> {
        self.inner
            .probability_of(&Support::points(vec![x.clone()]))
            .ok()
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
    family_boilerplate!(Affine, "Affine");

    fn context(&self) -> Context {
        self.inner.context()
    }

    fn parameters(&self) -> Vec<(&'static str, Ex)> {
        let mut p = vec![("a", self.a.clone()), ("b", self.b.clone())];
        p.extend(self.inner.parameters());
        p
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

    // Each side through the inner distribution's non-cancelling forms
    // (`cdf_on_support` / `sf_on_support`): a decreasing map turns the
    // inner upper tail into the lower one.
    fn cdf(&self, y: &Ex) -> Option<Ex> {
        let x = self.inverse(y);
        let family = self.inner.family();
        if self.increasing {
            family.cdf(&x)?;
            Some(self.inner.cdf_on_support(&x))
        } else {
            // With listed values, P(X ≥ x) = P(X > x) + P(X = x).
            if self.listed_values() {
                let atom = self.atom_mass(&x)?;
                return Some(self.inner.sf(&x) + atom);
            }
            let ctx = self.context();
            // P(aX + b ≤ y) = P(X ≥ x): P(X > x) for a density; on the
            // lattice P(X > ⌈x⌉ − 1) (for an integer x that is P(X > x − 1),
            // for a non-integer one P(X > ⌊x⌋)).
            let at = match self.inner.kind() {
                Kind::Continuous => x,
                Kind::Discrete => x.ceiling() - ctx.one(),
            };
            if family.cdf(&at).is_none() && family.sf(&at).is_none() {
                return None;
            }
            Some(self.inner.sf_on_support(&at))
        }
    }

    fn sf(&self, y: &Ex) -> Option<Ex> {
        let x = self.inverse(y);
        let family = self.inner.family();
        if self.increasing {
            if family.cdf(&x).is_none() && family.sf(&x).is_none() {
                return None;
            }
            Some(self.inner.sf_on_support(&x))
        } else {
            // With listed values, P(X < x) = P(X ≤ x) − P(X = x).
            if self.listed_values() {
                let atom = self.atom_mass(&x)?;
                return Some(self.inner.cdf(&x) - atom);
            }
            let ctx = self.context();
            // P(aX + b > y) = P(X < x): F(x) for a density, F(⌈x⌉ − 1) on
            // the lattice.
            let at = match self.inner.kind() {
                Kind::Continuous => x,
                Kind::Discrete => x.ceiling() - ctx.one(),
            };
            family.cdf(&at)?;
            Some(self.inner.cdf_on_support(&at))
        }
    }

    // e^{bt} M_X(at)
    fn mgf(&self, t: &Ex) -> Option<Ex> {
        Some((&self.b * t).exp() * self.inner.family().mgf(&(&self.a * t))?)
    }

    // The inner tails, swapped for a < 0.
    fn tail_orders(&self) -> Option<TailOrders> {
        let inner = self.inner.family().tail_orders()?;
        Some(if self.increasing {
            inner
        } else {
            TailOrders {
                left: inner.right,
                right: inner.left,
            }
        })
    }

    // {t : at ∈ D_X} = D_X / a, its ends swapped for a < 0.
    fn mgf_domain(&self) -> Option<Interval<Ex>> {
        let d = self.inner.family().mgf_domain()?;
        let ctx = self.context();
        let map_end = |v: &Ex| {
            if is_pos_inf(v) || is_neg_inf(v) {
                if is_pos_inf(v) == self.increasing {
                    ctx.infinity()
                } else {
                    ctx.neg_infinity()
                }
            } else {
                (v / &self.a).simplify()
            }
        };
        let image = d.as_ref().map(map_end);
        Some(if self.increasing {
            image
        } else {
            image.reversed()
        })
    }

    // e^{ibt} φ_X(at), through the inner distribution's own route.
    fn characteristic_function(&self, t: &Ex) -> Option<Ex> {
        let ctx = self.context();
        Some(
            (ctx.i_unit() * &self.b * t).exp() * self.inner.characteristic_function(&(&self.a * t)),
        )
    }

    fn quantile(&self, p: &Ex) -> Option<Ex> {
        let q = if self.increasing {
            self.inner.family().quantile(p)?
        } else {
            // Reflecting a lattice quantile is off by one lattice step
            // wherever F hits 1 − p exactly; leave that to the numeric
            // route (`quantile_f64` searches the transported cdf).
            if self.inner.kind() == Kind::Discrete {
                return None;
            }
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
    /// `h′(y)` for a branch `h`.  A real odd root `sign(u)·|u|^(1/n)`
    /// differentiates to `sign(u)²/(n|u|^(1−1/n))`, 0/0 at `u = 0`;
    /// `1/g′(h(y))` is the same slope with the pole showing (for the real
    /// root of an odd power, and for any branch holding a real root).
    fn branch_slope(&self, h: &Ex, odd_power: bool) -> Ex {
        if odd_power || holds_sign(h) {
            (self.context().one() / self.map.diff(&self.var).subs(&self.var, h)).simplify()
        } else {
            h.diff(&self.image_var)
        }
    }

    /// The inner values bounding `{g(X) ≤ y}`: `(None, Some(h(y)))` for
    /// an increasing branch (`X ≤ h(y)`), `(Some(h(y)), None)` for a
    /// decreasing one, both for the two branches of an even map; `None`
    /// when a slope's sign is not decided.
    fn event_ends(&self, y: &Ex) -> Option<(Option<Ex>, Option<Ex>)> {
        let odd_power = odd_power_inverse(&self.map, &self.var, &self.image_var).is_some();
        let (mut lower, mut upper) = (None, None);
        for (h, _) in &self.branches {
            let at = h.subs(&self.image_var, y);
            let slot = match self.branch_derivative_sign(&self.branch_slope(h, odd_power)) {
                SignSet::POSITIVE => &mut upper,
                SignSet::NEGATIVE => &mut lower,
                _ => return None,
            };
            if slot.replace(at).is_some() {
                return None;
            }
        }
        Some((lower, upper))
    }

    /// `POSITIVE` when `dh > 0` on the image, `NEGATIVE` when `dh < 0`,
    /// `UNKNOWN` when its sign cannot be decided at a sample interior point.
    fn branch_derivative_sign(&self, dh: &Ex) -> SignSet {
        let ctx = self.context();
        let Some(iv) = self.image.as_interval() else {
            return SignSet::UNKNOWN;
        };
        let (lo, hi) = (&iv.lower, &iv.upper);
        // Points strictly inside the image; the next is tried where `dh`
        // has no sign (`2x³ + 1` has its flat point at the first probe 1).
        let probes = if is_neg_inf(lo) && is_pos_inf(hi) {
            vec![ctx.one(), ctx.int(2), ctx.int(-1)]
        } else if is_neg_inf(lo) {
            vec![hi - ctx.one(), hi - ctx.int(2), hi - ctx.rational(1, 2)]
        } else if is_pos_inf(hi) {
            vec![lo + ctx.one(), lo + ctx.int(2), lo + ctx.rational(1, 2)]
        } else {
            let w = hi - lo;
            vec![
                lo + &w / ctx.int(2),
                lo + &w / ctx.int(3),
                lo + &w * ctx.rational(2, 3),
            ]
        };
        probes
            .iter()
            .find_map(|probe| {
                let at = dh.subs(&self.image_var, probe).simplify();
                match Truth::from(at.is_positive()) {
                    Truth::True => Some(SignSet::POSITIVE),
                    Truth::False if Truth::from(at.is_negative()).is_true() => {
                        Some(SignSet::NEGATIVE)
                    }
                    _ => None,
                }
            })
            .unwrap_or(SignSet::UNKNOWN)
    }
}

impl Family for Transformed {
    family_boilerplate!(Transformed, "Transformed");

    fn context(&self) -> Context {
        self.inner.context()
    }

    fn parameters(&self) -> Vec<(&'static str, Ex)> {
        let mut p = vec![("map", self.map.clone())];
        p.extend(self.inner.parameters());
        p
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
        let odd_power = odd_power_inverse(&self.map, &self.var, &self.image_var).is_some();
        for (h, _) in &self.branches {
            let h_y = h.subs(&self.image_var, y);
            let dh = self.branch_slope(h, odd_power);
            // Each branch is monotone on the image, so |h′| is h′ or −h′
            // throughout: decide the sign at an interior point of the image
            // instead of carrying `abs`.
            let signed = match self.branch_derivative_sign(&dh) {
                SignSet::POSITIVE => dh,
                SignSet::NEGATIVE => -dh,
                _ => dh.abs(),
            };
            acc += self.inner.density(&h_y) * signed.subs(&self.image_var, y);
        }
        acc.simplify()
    }

    // P(g(X) ≤ y) through the inner distribution function: F_X(h(y)) for
    // an increasing branch, S_X(h(y)) for a decreasing one, and
    // F_X(h₊(y)) − F_X(h₋(y)) for the two branches of an even map (the
    // inner law is continuous, so the ends carry no mass).  Up to 0.45 the
    // density was integrated: slow, refused by quadrature at a kink, and
    // wrong with a complex inverse.
    fn cdf(&self, y: &Ex) -> Option<Ex> {
        match self.event_ends(y)? {
            (None, Some(u)) => Some(self.inner.cdf(&u)),
            (Some(l), None) => Some(self.inner.sf(&l)),
            (Some(l), Some(u)) => Some(self.inner.cdf(&u) - self.inner.cdf(&l)),
            (None, None) => None,
        }
    }

    fn sf(&self, y: &Ex) -> Option<Ex> {
        match self.event_ends(y)? {
            (None, Some(u)) => Some(self.inner.sf(&u)),
            (Some(l), None) => Some(self.inner.cdf(&l)),
            (Some(l), Some(u)) => Some(self.inner.cdf(&l) + self.inner.sf(&u)),
            (None, None) => None,
        }
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
    family_boilerplate!(Mixture, "Mixture");

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

    // The union of the components' supports, overlapping pieces merged
    // (`[0, 2] ∪ [1, 3]` is `[0, 3]`) wherever their ends compare: the
    // generic routes integrate the mixture's own density — which already
    // sums the components — piece by piece, so an overlap was counted
    // twice (up to 0.45 `E[cos X]`, the entropy and `E[X | X > ½]` of
    // ½ U(0, 2) + ½ U(1, 3) were wrong).  Ends that do not compare
    // (symbolic parameters) keep the pieces as given.
    fn support(&self) -> Support {
        let kind = self
            .components
            .first()
            .map_or(Kind::Continuous, |(_, d)| d.kind());
        let mut pieces: Vec<Piece> = Vec::new();
        for (_, d) in &self.components {
            let s = if kind == Kind::Discrete {
                d.support().normalize_lattice()
            } else {
                d.support()
            };
            for piece in s.pieces() {
                if !pieces.contains(piece) {
                    pieces.push(piece.clone());
                }
            }
        }
        let pieces = merge_pieces(kind, &pieces).unwrap_or(pieces);
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

    fn sf(&self, x: &Ex) -> Option<Ex> {
        // Each component's whole-line survival function, weighted.
        let mut acc = self.ctx().zero();
        for (w, d) in &self.components {
            acc += w * d.sf(x);
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

    // Each side the heaviest component's: the smaller order (a symbolic
    // `Min` when two cannot be compared).  Each weight is taken to be
    // positive.
    fn tail_orders(&self) -> Option<TailOrders> {
        let ctx = self.ctx();
        let heavier = |a: Option<Ex>, b: Option<Ex>| match (a, b) {
            (None, o) | (o, None) => o,
            (Some(a), Some(b)) => Some(match super::family::sign_of(&(&a - &b)).to_ordering() {
                Some(std::cmp::Ordering::Greater) => b,
                Some(_) => a,
                None => Ex::min_of(&ctx, [a, b]),
            }),
        };
        let mut acc = TailOrders {
            left: None,
            right: None,
        };
        for (_, d) in &self.components {
            if let Some(o) = d.family().tail_orders() {
                acc.left = heavier(acc.left, o.left);
                acc.right = heavier(acc.right, o.right);
            }
        }
        (acc.left.is_some() || acc.right.is_some()).then_some(acc)
    }

    // The intersection of the components' domains (each weight is taken to
    // be positive; [`intersect_domains`]); `None` when no component
    // restricts `t`.
    fn mgf_domain(&self) -> Option<Interval<Ex>> {
        let mut acc: Option<Interval<Ex>> = None;
        for (_, d) in &self.components {
            let Some(dom) = d.family().mgf_domain() else {
                continue;
            };
            acc = Some(match acc {
                None => dom,
                Some(cur) => intersect_domains(cur, dom)?,
            });
        }
        acc
    }

    // Σ wᵢ φᵢ(t): each component through its own route.
    fn characteristic_function(&self, t: &Ex) -> Option<Ex> {
        let mut acc = self.ctx().zero();
        for (w, d) in &self.components {
            acc += w * d.characteristic_function(t);
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

/// The real inverse of `g = a·(x + s)ⁿ + b` for an odd `n ≥ 3`:
/// `real_root((y − b)/a, n) − s`.  `solve` gives the principal root
/// `∛((y − b)/a) − s`, complex for negative `(y − b)/a`: up to 0.45
/// the CDF of `X³` for a Student t, Cauchy or logistic `X` was off by `−1`
/// (negative) and its density had the wrong sign below 0.
fn odd_power_inverse(g: &Ex, x: &Ex, y: &Ex) -> Option<Ex> {
    let poly = crate::api::poly_ex::Poly::try_new(&g.expand(), &[x]).ok()?;
    let terms = poly.terms();
    let n = terms.iter().filter_map(|(e, _)| e.first().copied()).max()?;
    if n < 3 || n.is_multiple_of(2) {
        return None;
    }
    let coeff = |k: u32| {
        terms
            .iter()
            .find(|(e, _)| e.first() == Some(&k))
            .map(|(_, c)| c.clone())
    };
    let ctx = g.context();
    let a = coeff(n)?;
    let s = coeff(n - 1).map_or_else(
        || ctx.zero(),
        |c| (c / (&a * ctx.int(i64::from(n)))).simplify(),
    );
    let b = g.subs(x, &(-&s)).simplify();
    let shifted = &a * (x + &s).powi(i64::from(n)) + &b;
    if !Truth::from((shifted - g).expand().simplify().is_zero()).is_true() {
        return None;
    }
    let root = ((y - &b) / &a).real_root(n).ok()?;
    Some((root - s).simplify())
}

/// Does `h` hold a `sign` node (a real root `sign(u)·|u|^(1/n)`)?
fn holds_sign(h: &Ex) -> bool {
    let inner = h.inner.read();
    let arena = &inner.arena;
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![h.raw_id()];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        if matches!(arena.node(id), ExprNode::Sign(_)) {
            return true;
        }
        stack.extend(arena.children(id));
    }
    false
}

/// Why [`monotone_inverse`] accepted no inverse.
enum NoInverse {
    /// `solve` gave no roots (its error).
    Solve(SymplexError),
    /// No real root, or several of which none is proved the inverse.
    NotSingle,
    /// The one real root, not proved the inverse on the image.
    Unproved(Ex),
}

impl NoInverse {
    fn into_error(self, g: &Ex, support: &Support) -> SymplexError {
        not_implemented(match self {
            NoInverse::Solve(e) => format!("transforming by `{g}`: no inverse found ({e})"),
            NoInverse::NotSingle => {
                format!("transforming by `{g}`: the inverse is not a single branch")
            }
            NoInverse::Unproved(h) => format!(
                "transforming by `{g}`: the inverse `{h}` found is not proved real and continuous on the image of {support}, or not to invert the map there"
            ),
        })
    }
}

/// The inverse of a strictly monotone `g` on the open support, as an
/// expression in `y`, given the open interior of the image.  `g(x) = y` is
/// solved for a symbol with the image's sign ([`image_symbol`]), and each
/// real root is a candidate that [`inverse_on_support`] must accept; then
/// the real root of `a(x + s)ⁿ + b` for an odd `n` ([`odd_power_inverse`],
/// injective on ℝ); then a root whose principal odd roots are replaced by
/// real ones ([`real_root_form`]: `sinh(x + 5)³` on ℝ is inverted by
/// `asinh(∛ȳ) − 5`, the real cube root, where `solve`'s `asinh(∛y) − 5` is
/// complex for `y < 0`).  Up to 0.55 `y` was a plain symbol, so the roots of
/// `ln x = y`, `√x = y`, `x^(1/3) = y` held only under conditions on `y`
/// (`im y ∈ (−π, π]`, `y ≥ 0`) and `ln x`, `√x` were refused.
fn monotone_inverse(
    g: &Ex,
    x: &Ex,
    y: &Ex,
    support: &Interval<Ex>,
    image: &Interval<Ex>,
) -> Result<Ex, NoInverse> {
    let ctx = g.context();
    let (lo, hi) = (&support.lower, &support.upper);
    let accepted = |h: &Ex| inverse_on_support(g, x, h, y, lo, hi).is_true();
    let solved = match image_symbol(&ctx, image, &[g, x, lo, hi]) {
        Some(yv) => match (g - &yv).solve(x) {
            Ok(roots) => Ok((roots, yv)),
            Err(e) => match roots_on_image(g, x, &yv, image) {
                roots if roots.is_empty() => Err(NoInverse::Solve(e)),
                roots => Ok((roots, yv)),
            },
        },
        None => Err(NoInverse::NotSingle),
    };
    // A strictly monotone real map has one real inverse; `solve` also
    // lists the complex roots of `x³ = y` (`ω·∛y`), which drop out.
    let i = ctx.i_unit();
    let candidates: Vec<Ex> = match &solved {
        Ok((roots, yv)) => roots
            .iter()
            .filter(|h| !h.contains(&i))
            .map(|h| h.subs(yv, y))
            .collect(),
        Err(_) => Vec::new(),
    };
    if let Some(h) = candidates.iter().find(|h| accepted(h)) {
        return Ok(h.clone());
    }
    // A principal odd root is complex where its argument is negative, and
    // the real root takes over there.
    if let Some(h) = odd_power_inverse(g, x, y) {
        return Ok(h);
    }
    if let Ok((roots, yv)) = &solved {
        let real_forms = roots
            .iter()
            .filter(|h| !h.contains(&i))
            .filter_map(|h| real_root_form(h, roots, yv))
            .map(|h| h.subs(yv, y));
        for h in real_forms {
            if accepted(&h) {
                return Ok(h);
            }
        }
    }
    match (solved, candidates.as_slice()) {
        (Err(e), _) => Err(e),
        (Ok(_), [h]) => Err(NoInverse::Unproved(h.clone())),
        (Ok(_), _) => Err(NoInverse::NotSingle),
    }
}

/// The roots of `g(x) = y` that hold under a condition on `y` which holds
/// on the whole open image (`atan x = y` has the root `tan y` when `|y| <
/// π/2`, and the image of a real line under `atan` is `(−π/2, π/2)`):
/// the image must lie in the set `reduce_inequalities` makes of the
/// condition.  Empty when the solve is not conditional or no condition is
/// proved.
fn roots_on_image(g: &Ex, x: &Ex, yv: &Ex, image: &Interval<Ex>) -> Vec<Ex> {
    use crate::transforms::solve::{Reason, RootSet, Solutions};
    let eq = g - yv;
    let outcome = {
        let mut inner = eq.inner.write();
        crate::transforms::solve::solve_classified(&mut inner.arena, eq.raw_id(), x.raw_id())
    };
    let Solutions::Refused(Reason::Conditional(groups)) = outcome else {
        return Vec::new();
    };
    let image_set = g
        .context()
        .interval(&image.lower, &image.upper, IntervalKind::Open);
    let mut out = Vec::new();
    for group in groups {
        let RootSet::Points(points) = group.roots else {
            continue;
        };
        let cond: BoolEx = eq.wrap_as(group.cond);
        // (a condition on other parameters too is not decided here)
        let only_y = cond.free_symbols().iter().all(|s| s == yv);
        let holds = only_y
            && SetEx::reduce_inequalities(std::slice::from_ref(&cond), yv)
                .is_ok_and(|region| Truth::from(image_set.is_subset(&region)).is_true());
        if holds {
            out.extend(points.into_iter().map(|p| eq.wrap(p)));
        }
    }
    out
}

/// The symbol `g(x) = y` is solved for: declared with the sign of the
/// image's open interior `(a, b)` — positive when `a ≥ 0`, negative when
/// `b ≤ 0`, real otherwise — so that `solve` decides the principal-branch
/// conditions of its roots where the sign does (`ln x = y`: `eʸ` for a
/// real `y`; `√x = y`, `x^(1/3) = y`: `y²`, `y³` for a positive one;
/// for a plain `y` the roots are conditional and `solve` refuses).  Its
/// name occurs in none of `avoid` and is new to the context or already
/// carries exactly these assumptions, so no other symbol's assumptions
/// change.
fn image_symbol(ctx: &Context, image: &Interval<Ex>, avoid: &[&Ex]) -> Option<Ex> {
    use crate::base::assumptions::{Assumption, Assumptions};
    let (tag, assumption) = if Truth::from(image.lower.is_nonnegative()).is_true() {
        ("ypos", Assumption::Positive)
    } else if Truth::from(image.upper.is_nonpositive()).is_true() {
        ("yneg", Assumption::Negative)
    } else {
        ("yreal", Assumption::Real)
    };
    let wanted = Assumptions::declare(tag, Assumptions::default(), &[assumption]).ok()?;
    // (a bound on the names tried; every one taken is not expected)
    for n in 0..64u32 {
        let name = if n == 0 {
            format!("_{tag}")
        } else {
            format!("_{tag}{n}")
        };
        let declared = {
            let inner = ctx.inner.read();
            inner
                .arena
                .symbols
                .get(&name)
                .map(|sid| inner.arena.symbol_assumptions(sid) == wanted)
        };
        let candidate = match declared {
            None => ctx.symbol_with(&name, &[assumption]).ok()?,
            Some(true) => ctx.symbol(&name),
            Some(false) => continue,
        };
        if !avoid.iter().any(|e| e.contains(&candidate)) {
            return Some(candidate);
        }
    }
    None
}

/// `h` with its principal odd roots `u^(p/q)` (`q ≥ 3` odd, `u` depending
/// on `y`, one `u` and one `q` throughout) replaced by real roots
/// `(sign(u)·|u|^(1/q))ᵖ`, when that is again a root of the equation
/// `solve` gave `roots` for (with `h` among them).  It is: for `u ≥ 0` the
/// real root is the principal one, and for `u < 0` it is `ωᵐ` times it
/// (`ω = e^(2πi/q)`, `m = (q − 1)/2`), so the new expression takes at
/// each `y` the value of `h` or of `h` with `u^(p/q) ↦ ω^(mp)·u^(p/q)`,
/// which must be one of `roots` (an exact identity in `y`).  `None` when
/// it is not, or `h` has no such roots or several bases.
fn real_root_form(h: &Ex, roots: &[Ex], y: &Ex) -> Option<Ex> {
    let OddRadicals { u, q, radicals } = odd_radicals(h, y)?;
    let ctx = h.context();
    let m = (q - 1) / 2;
    let i = ctx.i_unit();
    // ω^j = cos(2πj/q) + i·sin(2πj/q), its standard values folded
    let omega_pow = |j: i64| {
        let angle = ctx.pi() * ctx.rational(2 * j.rem_euclid(q), q);
        (angle.cos() + &i * angle.sin()).eval()
    };
    let rotated: Vec<(Ex, Ex)> = radicals
        .iter()
        .map(|(r, p)| (r.clone(), omega_pow(m * p) * r))
        .collect();
    let pairs: Vec<(&Ex, &Ex)> = rotated.iter().map(|(a, b)| (a, b)).collect();
    let twin = h.subs_map(&pairs).expand();
    let is_root = roots.iter().any(|r| {
        let r = r.expand();
        r == twin || Truth::from((&twin - &r).simplify().is_zero()).is_true()
    });
    if !is_root {
        return None;
    }
    let real = u.real_root(u32::try_from(q).ok()?).ok()?;
    let replaced: Vec<(Ex, Ex)> = radicals
        .iter()
        .map(|(r, p)| (r.clone(), real.powi(*p)))
        .collect();
    let pairs: Vec<(&Ex, &Ex)> = replaced.iter().map(|(a, b)| (a, b)).collect();
    Some(h.subs_map(&pairs))
}

/// The principal odd roots of one base in an expression ([`odd_radicals`]).
struct OddRadicals {
    /// The base `u`.
    u: Ex,
    /// The odd root index `q ≥ 3`.
    q: i64,
    /// Each power `u^(p/q)` with its `p`.
    radicals: Vec<(Ex, i64)>,
}

/// The powers `u^(p/q)` in `h` with `u` depending on `y` and `q ≥ 3` odd,
/// when they share one `u` and one `q`.
fn odd_radicals(h: &Ex, y: &Ex) -> Option<OddRadicals> {
    let found: Vec<(ExprId, ExprId, i64, i64)> = {
        let inner = h.inner.read();
        let arena = &inner.arena;
        let y_id = y.raw_id();
        let mut found = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let mut stack = vec![h.raw_id()];
        while let Some(id) = stack.pop() {
            if !seen.insert(id) || !crate::base::walk::contains(arena, id, y_id) {
                continue;
            }
            if let ExprNode::Pow(b, e) = arena.node(id)
                && crate::base::walk::contains(arena, *b, y_id)
                && let Some(q) = arena.as_num(*e)
            {
                let (p, d) = (q.numer(), q.denom());
                if let (Ok(p), Ok(d)) = (i64::try_from(p), i64::try_from(d))
                    && d >= 3
                    && d % 2 == 1
                {
                    found.push((id, *b, p, d));
                }
            }
            stack.extend(arena.children(id));
        }
        found
    };
    let &(_, base, _, q) = found.first()?;
    if found.iter().any(|&(_, b, _, d)| b != base || d != q) {
        return None;
    }
    let radicals = found.iter().map(|&(id, _, p, _)| (h.wrap(id), p)).collect();
    Some(OddRadicals {
        u: h.wrap(base),
        q,
        radicals,
    })
}

/// Is `h` (an expression in `y`, a root of `g(x) = y` from `solve`) the
/// inverse of `g` on the whole open support `(lo, hi)`, where `g` is
/// strictly monotone?  It is when `h` is real and continuous on the image
/// ([`real_and_continuous_on_image`]) and `h(g(x₀)) = x₀` at one point
/// ([`inverts_at_a_point`]): the points of the image where `h` agrees with
/// the inverse are then closed and open in it (a real root of `g = y` near
/// one in the open support lies in it, and `g` is injective there), and the
/// image is connected.  Up to 0.54 this was sampled at two to four points
/// with an `f64` tolerance: the principal `∛(y) + 1/10` passed at `x =
/// 1/4, 3/4` for `(x − 1/10)³` on `(0, 1)`, and the density came out
/// complex for `x < 1/10`.
fn inverse_on_support(g: &Ex, x: &Ex, h: &Ex, y: &Ex, lo: &Ex, hi: &Ex) -> Truth {
    real_and_continuous_on_image(g, x, h, y, lo, hi).and(inverts_at_a_point(g, x, h, y, lo, hi))
}

/// What one node of a candidate inverse needs of its argument (in the
/// image variable) to be real and continuous, or of a subexpression free of
/// it.
#[derive(Clone, Copy)]
enum Need {
    /// `a ≥ 0` (a power with a positive non-integer exponent).
    NonNegative,
    /// `a > 0` (`ln a`; a power with a non-integer or variable exponent).
    Positive,
    /// `a ≠ 0` (a negative integer power).
    NonZero,
    /// `−1 ≤ a ≤ 1` (`asin`, `acos`).
    UnitClosed,
    /// `−1 < a < 1` (`atanh`).
    UnitOpen,
    /// `a ≥ 1` (`acosh`).
    AtLeastOne,
    /// `a ≥ −1/e` (the principal `W`).
    LambertW,
    /// `cos a ≠ 0` (`tan`).
    TanPole,
    /// `a^e` with an exponent `e` free of the variable (this id), read by value.
    ConstantExponent(ExprId),
    /// A subexpression free of the variable must be real …
    Real,
    /// … or positive (the base of a power whose exponent varies).
    PositiveConstant,
}

/// The requirements of every node of `h` on the path to `y`, or `None`
/// when a node is not one whose real line is known here (`Piecewise`,
/// `floor`, a library function, …: undecided).  Read under the arena's
/// lock; decided after it is released.
fn inverse_requirements(h: &Ex, y: &Ex) -> Option<Vec<(ExprId, Need)>> {
    let inner = h.inner.read();
    let arena = &inner.arena;
    let y_id = y.raw_id();
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![h.raw_id()];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        if !crate::base::walk::contains(arena, id, y_id) {
            out.push((id, Need::Real));
            continue;
        }
        match arena.node(id) {
            ExprNode::Symbol(_) => {}
            ExprNode::Add(cs) => stack.extend(cs.iter().copied()),
            // A real root `sign(a)·|a|^r` (`r > 0`) is continuous where `a`
            // is, though `sign(a)` alone jumps at `a = 0`: each `sign(a)`
            // (or a positive power of it) needs such a partner.
            ExprNode::Mul(cs) => {
                for &c in cs {
                    let sign_arg = match arena.node(c) {
                        ExprNode::Sign(a) => Some(*a),
                        ExprNode::Pow(b, e) => match arena.node(*b) {
                            ExprNode::Sign(a)
                                if arena
                                    .as_num(*e)
                                    .is_some_and(|k| k.is_integer() && k.is_positive()) =>
                            {
                                Some(*a)
                            }
                            _ => None,
                        },
                        _ => None,
                    };
                    match sign_arg {
                        Some(a) => {
                            let partnered = cs.iter().any(|&d| match arena.node(d) {
                                ExprNode::Pow(b, e) => {
                                    matches!(arena.node(*b), ExprNode::Abs(inner) if *inner == a)
                                        && arena.as_num(*e).is_some_and(|r| r.is_positive())
                                }
                                _ => false,
                            });
                            if !partnered {
                                return None;
                            }
                            stack.push(a);
                        }
                        None => stack.push(c),
                    }
                }
            }
            // `|a|^r` with `r > 0` is real and continuous where `a` is.
            ExprNode::Pow(b, e)
                if matches!(arena.node(*b), ExprNode::Abs(_))
                    && arena.as_num(*e).is_some_and(|r| r.is_positive()) =>
            {
                stack.push(*b);
            }
            ExprNode::Neg(a)
            | ExprNode::Exp(a)
            | ExprNode::Sin(a)
            | ExprNode::Cos(a)
            | ExprNode::Sinh(a)
            | ExprNode::Cosh(a)
            | ExprNode::Tanh(a)
            | ExprNode::Atan(a)
            | ExprNode::Asinh(a)
            | ExprNode::Abs(a)
            | ExprNode::Erf(a)
            | ExprNode::Erfc(a) => stack.push(*a),
            ExprNode::Ln(a) => {
                out.push((*a, Need::Positive));
                stack.push(*a);
            }
            ExprNode::Asin(a) | ExprNode::Acos(a) => {
                out.push((*a, Need::UnitClosed));
                stack.push(*a);
            }
            ExprNode::Atanh(a) => {
                out.push((*a, Need::UnitOpen));
                stack.push(*a);
            }
            ExprNode::Acosh(a) => {
                out.push((*a, Need::AtLeastOne));
                stack.push(*a);
            }
            ExprNode::LambertW(a) => {
                out.push((*a, Need::LambertW));
                stack.push(*a);
            }
            ExprNode::Tan(a) => {
                out.push((*a, Need::TanPole));
                stack.push(*a);
            }
            ExprNode::Pow(b, e) => {
                if crate::base::walk::contains(arena, *e, y_id) {
                    if crate::base::walk::contains(arena, *b, y_id) {
                        out.push((*b, Need::Positive));
                        stack.push(*b);
                    } else {
                        out.push((*b, Need::PositiveConstant));
                    }
                    stack.push(*e);
                } else {
                    out.push((*b, Need::ConstantExponent(*e)));
                    stack.push(*b);
                }
            }
            node => match node.func(arena) {
                // `erf⁻¹` is real and continuous on `(−1, 1)`.
                Some((crate::base::func::Func::ErfInv, args)) if args.len() == 1 => {
                    out.push((args[0], Need::UnitOpen));
                    stack.push(args[0]);
                }
                _ => return None,
            },
        }
    }
    Some(out)
}

/// Is `h` real and continuous on the image of the open support `(lo, hi)`
/// under `g`?  Each requirement of [`inverse_requirements`] on an argument
/// `a(y)` is decided at `y = g(x)` over the support by
/// `reduce_inequalities` (exactly: the set where it fails must be empty);
/// a subexpression free of `y` must be proved real.  `Unknown` when one is
/// not decided.
fn real_and_continuous_on_image(g: &Ex, x: &Ex, h: &Ex, y: &Ex, lo: &Ex, hi: &Ex) -> Truth {
    let Some(reqs) = inverse_requirements(h, y) else {
        return Truth::Unknown;
    };
    let ctx = g.context();
    let plain = super::family::fresh_symbol(&ctx, "xr", &[g, h, lo, hi]);
    let Ok(xr) = ctx.symbol_with(
        &plain.to_string(),
        &[crate::base::assumptions::Assumption::Real],
    ) else {
        return Truth::Unknown;
    };
    let (zero, one) = (ctx.zero(), ctx.one());
    // `a need 0` for every real x in the support: the set where it fails is empty.
    let holds = |a: &Ex, need: Need| -> Truth {
        let ax = a.subs(y, g).subs(x, &xr);
        // A sign the assumptions give for every real `x` (`e^{x + eˣ} + e⁻¹ > 0`,
        // where the sign chart cannot evaluate `e^{1000 + e¹⁰⁰⁰}`).  Not for
        // `≠ 0`: `is_zero` of an expression in `x` reads "not identically 0".
        let signed = match need {
            Need::NonNegative => ax.is_nonnegative(),
            Need::Positive => ax.is_positive(),
            _ => None,
        };
        if Truth::from(signed).is_true() {
            return Truth::True;
        }
        let fails = match need {
            Need::NonNegative => ax.lt(&zero),
            Need::Positive => ax.le(&zero),
            Need::NonZero => ax.eq_expr(&zero),
            _ => return Truth::Unknown,
        };
        let empty = |conds: &[BoolEx]| match SetEx::reduce_inequalities(conds, &xr) {
            Ok(s) => Truth::from(s.is_empty()),
            Err(_) => Truth::Unknown,
        };
        // On the whole line first (the support's ends may be symbolic).
        if empty(std::slice::from_ref(&fails)).is_true() {
            return Truth::True;
        }
        let mut conds = vec![fails];
        if !is_neg_inf(lo) {
            conds.push(xr.gt(lo));
        }
        if !is_pos_inf(hi) {
            conds.push(xr.lt(hi));
        }
        empty(&conds)
    };
    Truth::all(reqs.into_iter().map(|(id, need)| {
        let a = h.wrap(id);
        match need {
            Need::Real => Truth::from(a.is_real()),
            Need::PositiveConstant => Truth::from(a.is_positive()),
            Need::ConstantExponent(e) => match h.wrap(e).as_rational() {
                Some(q) if q.is_integer() && !q.is_negative() => Truth::True,
                Some(q) if q.is_integer() => holds(&a, Need::NonZero),
                Some(q) if q.is_positive() => holds(&a, Need::NonNegative),
                Some(_) => holds(&a, Need::Positive),
                None => Truth::from(h.wrap(e).is_real()).and(holds(&a, Need::Positive)),
            },
            Need::UnitClosed => {
                holds(&(&one - &a), Need::NonNegative).and(holds(&(&one + &a), Need::NonNegative))
            }
            Need::UnitOpen => {
                holds(&(&one - &a), Need::Positive).and(holds(&(&one + &a), Need::Positive))
            }
            Need::AtLeastOne => holds(&(&a - &one), Need::NonNegative),
            Need::LambertW => holds(&(&a + (-&one).exp()), Need::NonNegative),
            Need::TanPole => holds(&a.cos(), Need::NonZero),
            need => holds(&a, need),
        }
    }))
}

/// Does `h(g(x₀)) = x₀` at one interior point `x₀` of the support?  Proved
/// exactly when the difference simplifies to 0; otherwise from the
/// certified value of `h(g(x₀))` (real, `h` being real on the image): a real
/// root of `g = g(x₀)` other than `x₀` lies outside the support (`g` is
/// strictly monotone on it), at least as far as the nearer end, so a value
/// within half that distance of `x₀` is `x₀` and one beyond it is not.  On
/// the whole line every real root is `x₀`.
fn inverts_at_a_point(g: &Ex, x: &Ex, h: &Ex, y: &Ex, lo: &Ex, hi: &Ex) -> Truth {
    let ctx = g.context();
    let half = ctx.rational(1, 2);
    let (x0, reach) = match (is_neg_inf(lo), is_pos_inf(hi)) {
        (true, true) => return Truth::True,
        (false, true) => (lo + &half, half.clone()),
        (true, false) => (hi - &half, half.clone()),
        (false, false) => (lo + (hi - lo) * &half, (hi - lo) * &half),
    };
    let diff = (h.subs(y, &g.subs(x, &x0)) - &x0).eval();
    if Truth::from(diff.simplify().is_zero()).is_true() {
        return Truth::True;
    }
    match (diff.eval_f64(), reach.eval_f64()) {
        (Ok(d), Ok(r)) if d.is_finite() && r.is_finite() && r > 0.0 => {
            Truth::from_bool(2.0 * d.abs() < r)
        }
        _ => Truth::Unknown,
    }
}

/// `a < b`, `a = b` or `a > b` for interval ends (`±∞` included), when
/// decidable.
fn compare_ends(a: &Ex, b: &Ex) -> Option<std::cmp::Ordering> {
    use std::cmp::Ordering::{Equal, Greater, Less};
    let rank = |e: &Ex| {
        if is_neg_inf(e) {
            Some(-1)
        } else if is_pos_inf(e) {
            Some(1)
        } else {
            None
        }
    };
    match (rank(a), rank(b)) {
        (Some(x), Some(y)) => Some(x.cmp(&y)),
        (Some(x), None) => Some(if x < 0 { Less } else { Greater }),
        (None, Some(y)) => Some(if y < 0 { Greater } else { Less }),
        (None, None) => match (a - b).is_zero() {
            Some(true) => Some(Equal),
            _ => super::family::sign_of(&(a - b)).to_ordering(),
        },
    }
}

/// The pieces of a union made disjoint: intervals sorted and merged where
/// they overlap (or touch at a closed end; on the integer lattice, where
/// they are adjacent), listed values dropped when an interval covers them
/// and kept once otherwise.  `None` when two ends do not compare.
fn merge_pieces(kind: Kind, pieces: &[Piece]) -> Option<Vec<Piece>> {
    use std::cmp::Ordering::{Greater, Less};
    let mut ivs: Vec<Interval<Ex>> = Vec::new();
    let mut points: Vec<Ex> = Vec::new();
    for p in pieces {
        match p {
            Piece::Interval(iv) => ivs.push(iv.clone()),
            Piece::Point(v) => points.push(v.clone()),
        }
    }
    // Insertion sort by lower end (a handful of pieces), refusing an
    // undecidable comparison.
    let mut sorted: Vec<Interval<Ex>> = Vec::with_capacity(ivs.len());
    for iv in ivs {
        let mut at = sorted.len();
        for (i, s) in sorted.iter().enumerate() {
            if compare_ends(&iv.lower, &s.lower)? == Less {
                at = i;
                break;
            }
        }
        sorted.insert(at, iv);
    }
    let ctx = pieces.first().map(|p| match p {
        Piece::Interval(iv) => iv.lower.context(),
        Piece::Point(v) => v.context(),
    })?;
    let mut merged: Vec<Interval<Ex>> = Vec::new();
    for iv in sorted {
        let Some(cur) = merged.last_mut() else {
            merged.push(iv);
            continue;
        };
        // Does `iv` start inside `cur` (or right after it on the lattice)?
        let reach = if kind == Kind::Discrete && !is_pos_inf(&cur.upper) {
            &cur.upper + ctx.one()
        } else {
            cur.upper.clone()
        };
        let joins = match compare_ends(&iv.lower, &reach)? {
            Less => true,
            Greater => false,
            std::cmp::Ordering::Equal => {
                kind == Kind::Discrete || !(iv.kind.lower_open() && cur.kind.upper_open())
            }
        };
        if !joins {
            merged.push(iv);
            continue;
        }
        let (upper, upper_open) = match compare_ends(&iv.upper, &cur.upper)? {
            Greater => (iv.upper.clone(), iv.kind.upper_open()),
            Less => (cur.upper.clone(), cur.kind.upper_open()),
            std::cmp::Ordering::Equal => (
                cur.upper.clone(),
                cur.kind.upper_open() && iv.kind.upper_open(),
            ),
        };
        let lower_open = cur.kind.lower_open()
            && (compare_ends(&iv.lower, &cur.lower)? != std::cmp::Ordering::Equal
                || iv.kind.lower_open());
        *cur = Interval::closed(cur.lower.clone(), upper)
            .with_kind(IntervalKind::from_open_ends(lower_open, upper_open));
    }
    let mut out: Vec<Piece> = merged.iter().cloned().map(Piece::Interval).collect();
    for v in points {
        let covered = merged.iter().any(|iv| {
            Truth::from(Support::from_pieces(kind, vec![Piece::Interval(iv.clone())]).contains(&v))
                .is_true()
        });
        let listed = out.iter().any(|p| matches!(p, Piece::Point(w) if *w == v));
        if !covered && !listed {
            out.push(Piece::Point(v));
        }
    }
    Some(out)
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
        if Truth::from(mass.is_zero()).is_true() {
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
    /// `X` is discrete on an integer lattice and `a ≠ ±1` or `b` is a
    /// non-integer number (the image is no longer the integer lattice; a
    /// finite range can be enumerated with [`transformed`](Self::transformed)).
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
                if Truth::from(a.is_negative()).is_true() {
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
            match Truth::from(a.abs().equals(&a.context().one())) {
                Truth::True => {}
                Truth::False => {
                    return Err(invalid(
                        "an affine map of a lattice distribution must have slope ±1 (use `transformed` on a finite table)",
                    ));
                }
                // Not proved ±1 is not proved otherwise: say so.
                Truth::Unknown => {
                    return Err(invalid(format!(
                        "cannot decide whether the slope `{a}` of an affine map of a lattice distribution is ±1"
                    )));
                }
            }
            if Truth::from(b.is_integer()).is_false() {
                return Err(invalid(format!(
                    "an affine map of a lattice distribution must have an integer intercept, got `{b}` (use `transformed` on a finite range)"
                )));
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
    /// for an image variable of the image's sign (`eˣ`, `ln x`, `√x`,
    /// `x^(1/3)`, `1/x` on a positive support, `x³`, `x² + x` on a positive
    /// support, …) and which is proved real and continuous on the whole
    /// image and to invert `g` at one point (otherwise refused); the real
    /// root replaces a principal odd root where that is complex (`x³`,
    /// `a(x + s)ⁿ + b` with an odd `n`, `sinh(x + 5)³`, `e^((x − 1/10)³)`);
    /// `X²`, `|X|` and even powers of a continuous `X` (two branches on a
    /// support symmetric about `0`, one on a support to one side of `0`);
    /// and a finite table or finite integer range, whose values are mapped
    /// and merged.  SymPy: `density(g(X))`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::NotImplemented`] for other shapes (a non-monotone
    /// `g` that is not an even power / absolute value, an even shape on a
    /// support straddling `0` asymmetrically, an infinite lattice `X`
    /// under a non-affine `g`).
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
    /// // ln X for X ~ Exponential(1) has the density e^{y − e^y} on ℝ.
    /// let e = Distribution::exponential(ctx.int(1)).transformed(&x, &x.ln())?;
    /// let gumbel = (&y - y.exp()).exp();
    /// assert_eq!((e.density(&y) - gumbel).simplify(), ctx.int(0));
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
            match self.affine(a, b) {
                Ok(d) => return Ok(d),
                // A lattice image that leaves the integers (slope ≠ ±1 or
                // a fractional shift) is enumerated below when the range
                // is finite; anything else is the affine route's error.
                Err(e) => {
                    if self.kind() != Kind::Discrete || self.enumerate_lattice().is_none() {
                        return Err(e);
                    }
                }
            }
        }
        // A finite table (or a finite integer range, enumerated): map the
        // values and merge equal images.
        if let Some(values) = self
            .support()
            .as_points()
            .or_else(|| self.enumerate_lattice())
        {
            // A rational image is merged through a map keyed by its value;
            // only an image that is not a rational is compared with every
            // entry.  Up to 0.56 every pair was compared by `equals`, each
            // comparison interning a difference: `X⁵` of a hypergeometric on
            // 19,532 atoms exhausted 2 GB.
            //
            // An image that is not a rational is compared by `equals` only
            // with the entries its value does not already separate it from
            // ([`certified_apart`]): those in its own and the neighbouring
            // buckets of [`value_bucket`], and those without a value.  Up to
            // 0.57 each was compared with every entry: `√X` of a 2000-sided
            // die made two million `equals` calls (fuzz_stats timed out
            // after 30 s).
            let mut table: Vec<(Ex, Ex)> = Vec::new();
            let mut rational_at: std::collections::HashMap<crate::base::numeric::Q, usize> =
                std::collections::HashMap::new();
            let mut values_at: Vec<Option<ValueKey>> = Vec::new();
            let mut buckets: std::collections::HashMap<(bool, i64), Vec<usize>> =
                std::collections::HashMap::new();
            // Entries that are not rationals and have no bucket.
            let mut unvalued: Vec<usize> = Vec::new();
            for v in &values {
                let image = g.subs(x, v).eval();
                let mass = self.density(v).eval();
                let rational = image.as_rational();
                let value = value_key(&image);
                let bucket = value.and_then(|k| Some((k.tiny, value_bucket(k.f)?)));
                let same = |i: &usize| {
                    !keys_apart(values_at[*i], value)
                        && Truth::from(table[*i].0.equals(&image)).is_true()
                };
                let found = match (&rational, bucket) {
                    (Some(q), _) => rational_at.get(q).copied(),
                    (None, _) => None,
                }
                .or_else(|| match bucket {
                    Some((tiny, k)) => (k - 1..=k + 1)
                        .filter_map(|k| buckets.get(&(tiny, k)))
                        .flatten()
                        .copied()
                        .filter(|&i| rational.is_none() || table[i].0.as_rational().is_none())
                        .chain(unvalued.iter().copied())
                        .find(same),
                    None if rational.is_some() => unvalued
                        .iter()
                        .copied()
                        .chain(buckets.values().flatten().copied())
                        .filter(|&i| table[i].0.as_rational().is_none())
                        .find(same),
                    None => (0..table.len()).find(same),
                });
                match found {
                    Some(i) => table[i].1 = (&table[i].1 + mass).simplify(),
                    None => {
                        let i = table.len();
                        if let Some(q) = rational {
                            rational_at.insert(q, i);
                        }
                        match bucket {
                            Some(k) => buckets.entry(k).or_default().push(i),
                            None => unvalued.push(i),
                        }
                        table.push((image, mass));
                        values_at.push(value);
                    }
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
        // Even shapes: x², |x|, x^{2k}.  Both branches on a support
        // symmetric about 0, the live one on a support to one side of 0.
        if let Some(branches) = even_branches(x, g, &y, &ctx) {
            let branches = live_even_branches(branches, g, &support)?;
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
        let interior = |lower: &Ex, upper: &Ex| Interval {
            lower: lower.clone(),
            upper: upper.clone(),
            kind: IntervalKind::Open,
        };
        let image_interior = if increasing {
            interior(&glo, &ghi)
        } else {
            interior(&ghi, &glo)
        };
        let h = monotone_inverse(g, x, &y, &interior(lo, hi), &image_interior)
            .map_err(|e| e.into_error(g, &support))?;
        let h = &h;
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

/// The intersection of two moment-generating-function domains (both
/// contain `0`): the larger lower end and the smaller upper end, each with
/// the openness of the end it came from (open if both ends agree and either
/// is open).  Two ends that cannot be compared (symbolic rates) and are
/// equally open give the symbolic `Min`/`Max`; `None` when their openness
/// differs.
fn intersect_domains(a: Interval<Ex>, b: Interval<Ex>) -> Option<Interval<Ex>> {
    use std::cmp::Ordering;
    // `pick(x, y, x_open, y_open, upper)`: the tighter end and its openness.
    let pick = |x: Ex, y: Ex, x_open: bool, y_open: bool, upper: bool| -> Option<(Ex, bool)> {
        let infinite = |v: &Ex| if upper { is_pos_inf(v) } else { is_neg_inf(v) };
        if infinite(&x) {
            return Some((y, y_open));
        }
        if infinite(&y) {
            return Some((x, x_open));
        }
        let Some(ord) = super::family::sign_of(&(&x - &y)).to_ordering() else {
            if x_open != y_open {
                return None;
            }
            let ctx = x.context();
            let end = if upper {
                Ex::min_of(&ctx, [x, y])
            } else {
                Ex::max_of(&ctx, [x, y])
            };
            return Some((end, x_open));
        };
        Some(match (ord, upper) {
            (Ordering::Equal, _) => (x, x_open || y_open),
            (Ordering::Less, true) | (Ordering::Greater, false) => (x, x_open),
            _ => (y, y_open),
        })
    };
    let (lower, lo_open) = pick(
        a.lower,
        b.lower,
        a.kind.lower_open(),
        b.kind.lower_open(),
        false,
    )?;
    let (upper, hi_open) = pick(
        a.upper,
        b.upper,
        a.kind.upper_open(),
        b.kind.upper_open(),
        true,
    )?;
    Some(Interval {
        lower,
        upper,
        kind: IntervalKind::from_open_ends(lo_open, hi_open),
    })
}

/// The branches of an even shape that land inside `support` everywhere
/// on the image — the density formula of the inner family is valid on its
/// support only, so a branch that leaves it must not be summed.  Both on
/// a support symmetric about `0`, the non-negative one when the support
/// lies in `[0, ∞)`, the non-positive one when it lies in `(−∞, 0]`.
///
/// # Errors
///
/// [`SymplexError::NotImplemented`] when the support straddles `0`
/// asymmetrically (`Uniform(−1, 3)²`: two branches on `[0, 1]`, one on
/// `(1, 9]`).
fn live_even_branches(
    branches: Vec<Ex>,
    g: &Ex,
    support: &Support,
) -> Result<Vec<Ex>, SymplexError> {
    let refuse = || {
        not_implemented(format!(
            "transforming by `{g}`: the support {support} straddles 0 asymmetrically, so the \
             two-branch change of variables holds on part of the image only"
        ))
    };
    let Some(iv) = support.as_interval() else {
        return Err(refuse());
    };
    let (lo, hi) = (&iv.lower, &iv.upper);
    let (lo_inf, hi_inf) = (is_neg_inf(lo), is_pos_inf(hi));
    // `even_branches` lists the non-negative branch first.
    let [non_negative, non_positive] = branches.as_slice() else {
        return Ok(branches);
    };
    if !lo_inf && Truth::from(lo.is_nonnegative()).is_true() {
        return Ok(vec![non_negative.clone()]);
    }
    if !hi_inf && Truth::from((-hi).is_nonnegative()).is_true() {
        return Ok(vec![non_positive.clone()]);
    }
    let symmetric =
        (lo_inf && hi_inf) || (!lo_inf && !hi_inf && Truth::from((lo + hi).is_zero()).is_true());
    if symmetric {
        return Ok(branches);
    }
    Err(refuse())
}

/// The inverse branches of an even shape `x²`, `|x|`, `x^{2k}` (as
/// expressions in `y`): `±√y`, `±y`, `±y^{1/(2k)}`, the non-negative
/// branch first.
/// Do two `eval_f64` values prove their constants different?  Each is the
/// certified value correctly rounded, so it lies within an ulp (`≤ ε·|v|`
/// for a normal double) of the constant: values farther apart than four
/// times `ε` of the larger magnitude cannot belong to one number.  `false`
/// (undecided) for a missing value or one outside the normal range.
pub(super) fn certified_apart(a: Option<f64>, b: Option<f64>) -> bool {
    let (Some(a), Some(b)) = (a, b) else {
        return false;
    };
    if !(a.is_normal() && b.is_normal()) {
        return false;
    }
    (a - b).abs() > 4.0 * f64::EPSILON * a.abs().max(b.abs())
}

/// What separates an image of [`Distribution::transformed`] from others: its
/// `eval_f64` (`f`, `tiny = false`) when that is a normal double, else for
/// a real value below the normal range (`eval_f64` is `0` or subnormal) the
/// `eval_f64` of `ln |image|` (`tiny = true`).  Both are certified values
/// correctly rounded, so equal images get equal keys; `exp(−k)` of a
/// 7000-sided die underflowed, and its 6255 images were compared pairwise.
#[derive(Clone, Copy, Debug)]
struct ValueKey {
    tiny: bool,
    f: f64,
}

fn value_key(image: &Ex) -> Option<ValueKey> {
    let v = image.eval_f64().ok()?;
    if v.is_normal() {
        return Some(ValueKey { tiny: false, f: v });
    }
    if !v.is_finite() {
        return None;
    }
    let l = image.abs().ln().eval_f64().ok()?;
    l.is_normal().then_some(ValueKey { tiny: true, f: l })
}

/// Are the images of two keys proved different?  A value below the normal
/// range and one at least four times its bottom are; two values (or two
/// logarithms) by [`certified_apart`].
fn keys_apart(a: Option<ValueKey>, b: Option<ValueKey>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) if a.tiny != b.tiny => {
            let normal = if a.tiny { b.f } else { a.f };
            normal.abs() >= 4.0 * f64::MIN_POSITIVE
        }
        (Some(a), Some(b)) => certified_apart(Some(a.f), Some(b.f)),
        _ => false,
    }
}

/// The bucket of a normal `f64` among runs of 32 consecutive doubles (in
/// the order of their values), `None` otherwise: two values that
/// [`certified_apart`] cannot separate (at most about 8 doubles apart) lie
/// in the same or neighbouring buckets.
fn value_bucket(v: f64) -> Option<i64> {
    if !v.is_normal() {
        return None;
    }
    let bits = i64::from_ne_bytes(v.to_bits().to_ne_bytes());
    // Monotone in `v`: the sign-magnitude bits of a negative value reversed.
    let ordered = if bits < 0 { i64::MIN - bits } else { bits };
    Some(ordered >> 5)
}

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
    if n < 2 || n % 2 != 0 || !Truth::from(coeff.equals(&ctx.one())).is_true() {
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
    let bottom = if Truth::from(contains_zero).is_false() {
        a.min_with(&b).simplify()
    } else {
        ctx.zero()
    };
    Support::interval(bottom, top)
}
