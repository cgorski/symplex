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
use std::cmp::Ordering;
use std::fmt;
use std::sync::Arc;

use crate::api::context::Context;
use crate::api::expr::Ex;
use crate::api::poly_ex::Poly;
use crate::base::errors::SymplexError;
use crate::base::interval::Interval;

use super::sample::Rng;
use super::support::{Kind, Piece, Support, is_neg_inf, is_pos_inf};

/// A closure producing one `f64` sample per call.
pub type Sampler = Box<dyn FnMut(&mut Rng) -> f64 + Send>;

/// A tail probability below this (`2⁻³²`) is far: see
/// [`Distribution::tail_of`].
const FAR_TAIL: f64 = 1.0 / 4_294_967_296.0;

/// Where a numeric point lies relative to a distribution's far tails.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tail {
    Lower,
    Upper,
    Neither,
}

/// Relative half-width (`2⁻⁴⁴`) of the band around a quantile level
/// inside which an `f64` sum of masses does not decide on which side of
/// the level it lies: each mass carries a few ulps from its evaluation and
/// the compensated sum adds two more, so a sum outside the band is on the
/// side it shows with a margin of hundreds; inside, the exact sum decides.
/// (A band, not a tolerance: it never moves the answer.)
const TIE_BAND: f64 = 1.0 / 17_592_186_044_416.0;

/// The condition a discrete quantile at `p` looks for, on the smaller
/// tail (as [`numdist`](super::numdist)'s lattice inversions do): `F(k) ≥
/// p` for `p ≤ ½`, and `S(k) ≤ q` with `q = 1 − p` otherwise — the same
/// condition, but only the small tail has digits left near its level.
/// `1 − p` is exact for `p ≥ ½` (Sterbenz), so `q` is the exact
/// complement of the double `p`.
#[derive(Clone, Copy, Debug)]
enum Level {
    /// `F(k) ≥ p`.
    Lower(f64),
    /// `S(k) ≤ q`.
    Upper(f64),
}

impl Level {
    fn of(p: f64) -> Self {
        if p <= 0.5 {
            Level::Lower(p)
        } else {
            Level::Upper(1.0 - p)
        }
    }

    /// Has the exact tail crossed the level?  The sign of `tail − level`,
    /// the level read as the exact binary number it is, evaluated in
    /// arbitrary precision; a tie (`F(k) = p`, `S(k) = q`) has crossed.
    fn crossed(self, tail: &Ex) -> Result<bool, SymplexError> {
        let (level, lower) = match self {
            Level::Lower(p) => (p, true),
            Level::Upper(q) => (q, false),
        };
        let d = (tail - tail.context().from_f64(level)?).eval_f64()?;
        if d.is_nan() {
            return Err(SymplexError::computation_failed(
                "quantile_f64",
                "the distribution function is not a number",
            ));
        }
        Ok(if lower { d >= 0.0 } else { d <= 0.0 })
    }

    /// The same from an `f64` sum `t` of masses, when it lies outside
    /// [`TIE_BAND`] around the level; `None` inside it, for a `NaN`, or for
    /// a subnormal level (whose neighbourhood `f64` cannot resolve).
    fn crossed_f64(self, t: f64) -> Option<bool> {
        let (level, lower) = match self {
            Level::Lower(p) => (p, true),
            Level::Upper(q) => (q, false),
        };
        if level.is_nan() || level < f64::MIN_POSITIVE || t.is_nan() {
            return None;
        }
        if t > level * (1.0 + TIE_BAND) {
            Some(lower)
        } else if t < level * (1.0 - TIE_BAND) {
            Some(!lower)
        } else {
            None
        }
    }
}

/// A bracket `[a, b]` with `g(a) ≤ 0 ≤ g(b)` for an increasing `g` on the
/// hull `[lo, hi]` of a support (`g` is negative at `lo` and positive at
/// `hi`), grown geometrically outward from the estimate `x0` — a first
/// step of `2⁻²⁶·|x0|` (`2⁻²⁶` from 0), quadrupling — or, without one,
/// from the middle of a finite hull, from a unit (or `|end|`) inside a
/// half-finite one, or from 0.
fn grow_bracket(
    g: &impl Fn(f64) -> f64,
    x0: Option<f64>,
    lo: f64,
    hi: f64,
) -> Result<(f64, f64), SymplexError> {
    let fail = |why: String| SymplexError::computation_failed("quantile_f64", why);
    let inside = |v: f64| v.is_finite() && v > lo && v < hi;
    let s = match x0.filter(|&v| inside(v)) {
        Some(v) => v,
        None => match (lo.is_finite(), hi.is_finite()) {
            (true, true) => lo + 0.5 * (hi - lo),
            (true, false) => lo + 1.0_f64.max(lo.abs()),
            (false, true) => hi - 1.0_f64.max(hi.abs()),
            (false, false) => 0.0,
        },
    };
    let gs = g(s);
    if gs.is_nan() {
        return Err(fail(format!(
            "the distribution function is not a number at {s}"
        )));
    }
    if gs == 0.0 {
        return Ok((s, s));
    }
    let up = gs < 0.0;
    let mut near = s;
    let mut d = if s == 0.0 { 1.0 } else { s.abs() } * 2.0_f64.powi(-26);
    for _ in 0..600 {
        let far = if up { (s + d).min(hi) } else { (s - d).max(lo) };
        let gf = g(far);
        if gf.is_nan() {
            return Err(fail(format!(
                "the distribution function is not a number at {far}"
            )));
        }
        if (gf >= 0.0) == up {
            return Ok(if up { (near, far) } else { (far, near) });
        }
        if !far.is_finite() {
            break;
        }
        near = far;
        d *= 4.0;
    }
    Err(fail(format!(
        "no bracket found for the level starting from {s}"
    )))
}

/// A running sum with Neumaier's compensation (the improved Kahan–Babuška
/// algorithm, Neumaier 1974): its rounding error stays within a couple of
/// ulps however many terms are added.
#[derive(Default)]
struct CompensatedSum {
    sum: f64,
    carry: f64,
}

impl CompensatedSum {
    fn add(&mut self, x: f64) {
        let t = self.sum + x;
        self.carry += if self.sum.abs() >= x.abs() {
            (self.sum - t) + x
        } else {
            (x - t) + self.sum
        };
        self.sum = t;
    }

    fn value(&self) -> f64 {
        self.sum + self.carry
    }
}

/// The sign of `e`: the symbolic tests first, then, for a constant they
/// cannot decide (`3 − √2 − 10⁻¹⁰⁰`, `10¹⁰⁰ − √2`), the sign of its value —
/// which `evalf` gets right whenever the value is not `0` (read from its
/// decimal expansion, which does not underflow as an `f64` would).
/// `None` for a symbolic `e` of unknown sign, or a constant that evaluates
/// to `0` without being known to be `0`.
pub(crate) fn sign_of(e: &Ex) -> Option<Ordering> {
    if e.is_zero() == Some(true) {
        return Some(Ordering::Equal);
    }
    if e.is_positive() == Some(true) {
        return Some(Ordering::Greater);
    }
    if e.is_negative() == Some(true) {
        return Some(Ordering::Less);
    }
    if !e.free_symbols().is_empty() {
        return None;
    }
    let digits = e.eval_decimal(5).ok()?;
    let magnitude = digits.trim_start_matches('-');
    if digits.contains('I')
        || !magnitude.starts_with(|c: char| c.is_ascii_digit())
        || magnitude.trim_start_matches(['0', '.']).is_empty()
    {
        return None;
    }
    Some(if digits.starts_with('-') {
        Ordering::Less
    } else {
        Ordering::Greater
    })
}

/// An end of a support piece as an `f64` (`±∞` for the infinities).
fn lattice_end(e: &Ex) -> Result<f64, SymplexError> {
    if is_neg_inf(e) {
        Ok(f64::NEG_INFINITY)
    } else if is_pos_inf(e) {
        Ok(f64::INFINITY)
    } else {
        e.eval_f64()
    }
}

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

    /// Closed-form `P(X ≤ x)` for `x` in the support, in its classic form
    /// (`½ + ½ erf(z/√2)` for a normal): the form of a symbolic CDF and of
    /// the mass of an interval around the median.
    fn cdf(&self, _x: &Ex) -> Option<Ex> {
        None
    }

    /// `P(X ≤ x)` again, written so that it keeps its relative accuracy
    /// where it is small: `½ erfc(−z/√2)` for a normal, whose classic
    /// `½ + ½ erf(z/√2)` is the difference of two numbers within `10⁻³⁰⁰`
    /// of each other in a far tail and evaluates to `0`.  The expression
    /// must equal [`cdf`](Family::cdf) everywhere; the generic machinery
    /// uses it for a numeric `x` below the median.  `None` (the default):
    /// the classic form is already accurate in the lower tail.
    fn cdf_lower(&self, _x: &Ex) -> Option<Ex> {
        None
    }

    /// Closed-form survival function `P(X > x)` for `x` in the support,
    /// written so that it keeps its relative accuracy where it is small —
    /// `½ erfc(z/√2)`, `Γ(k, x)/Γ(k)`, `I_{1−x}(β, α)` — rather than as
    /// `1 − cdf`, which evaluates to `0` in a far tail.  On the lattice
    /// `P(X > x) = P(X > ⌊x⌋)`.  The generic machinery uses it for a
    /// numeric `x` above the median (and whenever there is no
    /// [`cdf`](Family::cdf)); `None` (the default) makes it use `1 − cdf`.
    fn sf(&self, _x: &Ex) -> Option<Ex> {
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
    /// transforms its inner family's samples; `Gamma` and the families
    /// built on it, `Poisson`, `Geometric` and `NegativeBinomial` run exact
    /// algorithms of their own).  `None` leaves it to the generic route:
    /// inverse transform through the quantile, or cumulative sums of the
    /// mass function.  Convert the parameters once here (`eval_f64`) and
    /// return `Some(Err(_))` when one is symbolic, so the closure itself
    /// cannot fail.
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
///
/// It cannot be the trait's default: a provided `eq_family` would need
/// `Self: PartialEq`, which is only expressible with `where Self: Sized`,
/// and that takes the method out of the vtable that `Distribution`'s
/// `PartialEq` dispatches through.  The crate-private `family_boilerplate!`
/// macro writes the delegation instead.
pub fn same_family<T: Family + PartialEq>(a: &T, other: &dyn Family) -> bool {
    (other as &dyn Any)
        .downcast_ref::<T>()
        .is_some_and(|b| a == b)
}

/// The items of an [`impl Family`](Family) that every family writes the
/// same way.  The three-argument form is for a struct whose parameters are
/// `Ex` fields: `name`, `context` (the first field's), `parameters` (the
/// fields, in order, named as written) and `eq_family` (through
/// [`same_family`]).  The two-argument form emits `name` and `eq_family`
/// only, for a family whose context and parameters are not plain fields
/// (a table, a wrapper around another distribution).  The type is named
/// first in both forms, as the call site reads.
///
/// ```ignore
/// impl Family for Normal {
///     family_boilerplate!(Normal, "Normal", [mean, std]);
///     // support, density, closed forms…
/// }
/// impl Family for Truncated {
///     family_boilerplate!(Truncated, "Truncated");
///     fn context(&self) -> Context { self.inner.context() }
///     // …
/// }
/// ```
macro_rules! family_boilerplate {
    ($ty:ident, $name:literal, [$first:ident $(, $field:ident)*]) => {
        fn name(&self) -> &str {
            $name
        }
        fn context(&self) -> $crate::api::context::Context {
            self.$first.context()
        }
        fn parameters(&self) -> Vec<(&'static str, $crate::api::expr::Ex)> {
            vec![(stringify!($first), self.$first.clone()) $(, (stringify!($field), self.$field.clone()))*]
        }
        fn eq_family(&self, other: &dyn $crate::stats::Family) -> bool {
            $crate::stats::same_family(self, other)
        }
    };
    ($ty:ident, $name:literal) => {
        fn name(&self) -> &str {
            $name
        }
        fn eq_family(&self, other: &dyn $crate::stats::Family) -> bool {
            $crate::stats::same_family(self, other)
        }
    };
}
pub(crate) use family_boilerplate;

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

    /// Where `x` lies, when it and the parameters are numeric: in the far
    /// lower tail, the far upper tail, or neither — read off the classic
    /// closed-form `F(x)` in `f64`.  A far tail is beyond [`FAR_TAIL`]
    /// (`2⁻³²`): short of it the classic form loses at most 32 bits to
    /// cancellation, which `evalf` recovers at any precision, and it is
    /// kept (`P(−1 < N < 1)` stays `erf(√2/2)`); beyond it the classic
    /// form of that tail is a difference of two numbers that may agree to
    /// hundreds of digits, and the family's non-cancelling form is used.
    /// (A classic far lower tail may itself evaluate to `0.0`: still in the
    /// far lower tail.  One that cancels beyond `evalf`'s budget fails to
    /// evaluate — `1 − (3 − x)²/(3 − √2)²` at `x = √2 + 10⁻¹⁰⁰` — and the
    /// tail is then read off the non-cancelling forms themselves; 0.28 took
    /// such a point for a central one and kept the classic form.)
    fn tail_of(&self, x: &Ex) -> Tail {
        if !x.free_symbols().is_empty() {
            return Tail::Neither;
        }
        let Some(classic) = self.0.cdf(x) else {
            return Tail::Neither;
        };
        match classic.eval_f64() {
            Ok(p) if p.is_finite() && p < FAR_TAIL => Tail::Lower,
            Ok(p) if p.is_finite() && p > 1.0 - FAR_TAIL => Tail::Upper,
            Ok(_) => Tail::Neither,
            Err(_) => {
                let far = |form: Option<Ex>| {
                    form.and_then(|f| f.eval_f64().ok())
                        .is_some_and(|v| v.is_finite() && v < FAR_TAIL)
                };
                if far(self.0.cdf_lower(x)) {
                    Tail::Lower
                } else if far(self.0.sf(x)) {
                    Tail::Upper
                } else {
                    Tail::Neither
                }
            }
        }
    }

    /// Is the numeric `x` in the far upper tail (`P(X > x) < 2⁻³²` by the
    /// classic CDF), where a probability must be formed from survival
    /// functions?  See [`tail_of`](Self::tail_of).
    pub(crate) fn in_far_upper_tail(&self, x: &Ex) -> bool {
        self.tail_of(x) == Tail::Upper
    }

    /// The closed-form CDF on the support, if the family has one, else
    /// the integral / sum of the density from the support's lower end.
    /// Nothing is clamped: `x` is taken to lie in the support (see
    /// [`cdf`](Self::cdf) for the whole line).  A numeric `x` in the far
    /// lower tail gets the family's [`cdf_lower`](Family::cdf_lower) when
    /// it has one (`½ erfc(−z/√2)` for a normal, whose classic `½ + ½ erf`
    /// evaluates to `0` there).
    pub(crate) fn cdf_on_support(&self, x: &Ex) -> Ex {
        if let Some(c) = self.0.cdf(x) {
            if self.tail_of(x) == Tail::Lower
                && let Some(lower) = self.0.cdf_lower(x)
            {
                return lower;
            }
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
            Some(iv) => iv.lower.clone(),
            None => ctx.neg_infinity(),
        };
        // On the lattice `P(X ≤ x)` sums up to `⌊x⌋`, folded for a numeric
        // `x`: the summation takes an unevaluated `floor(4)` for a symbolic
        // bound and looks for a closed form (4.5 s for a hypergeometric
        // `P(X ≤ 4)`, against 0.6 ms for the five terms).
        let hi = match support.kind() {
            Kind::Continuous => x.clone(),
            Kind::Discrete => x.floor().eval(),
        };
        support.accumulate(&dens, &t, &lo, &hi)
    }

    /// `P(X < lo)` through the closed-form CDF, for the lower end `lo` of
    /// a region already clipped to the support: `0` at `−∞` and at the
    /// support's own lower end (where the closed form need not fold —
    /// `Φ(ln 0)` for a log-normal), else `F(lo)` for a density and
    /// `F(lo − 1)` on the integer lattice, whose atom at `lo` belongs to
    /// the region.  `None` when the family has no closed-form CDF.
    pub(crate) fn mass_below(&self, lo: &Ex) -> Option<Ex> {
        let ctx = self.context();
        if is_neg_inf(lo) {
            return Some(ctx.zero());
        }
        let support = self.support();
        if let Some(s) = support.as_interval()
            && !is_neg_inf(&s.lower)
            && (lo - &s.lower).is_zero() == Some(true)
        {
            return Some(ctx.zero());
        }
        let at = match support.kind() {
            Kind::Continuous => lo.clone(),
            Kind::Discrete => lo - ctx.one(),
        };
        self.0.cdf(&at)?;
        Some(self.cdf_on_support(&at))
    }

    /// `P(X > x)` on the support: for a numeric `x` in the far upper tail
    /// the family's [`sf`](Family::sf) (the non-cancelling form),
    /// elsewhere `1 − F(x)` with the classic `F`; a family with a survival
    /// function but no CDF uses the survival function throughout.
    pub(crate) fn sf_on_support(&self, x: &Ex) -> Ex {
        let ctx = self.context();
        match self.0.cdf(x) {
            Some(c) => {
                if self.tail_of(x) == Tail::Upper
                    && let Some(s) = self.0.sf(x)
                {
                    return s;
                }
                ctx.one() - c
            }
            None => self
                .0
                .sf(x)
                .unwrap_or_else(|| ctx.one() - self.cdf_on_support(x)),
        }
    }

    /// `P(X ≥ lo)` for the lower end `lo` of a region already clipped to
    /// the support, through the closed forms: `1` at `−∞` and at the
    /// support's own lower end, else `P(X > lo)` for a density and
    /// `P(X > lo − 1)` on the integer lattice ([`sf_on_support`]).  `None`
    /// when the family has neither a closed-form CDF nor a survival
    /// function.
    ///
    /// [`sf_on_support`]: Self::sf_on_support
    pub(crate) fn mass_above(&self, lo: &Ex) -> Option<Ex> {
        let ctx = self.context();
        if is_neg_inf(lo) {
            return Some(ctx.one());
        }
        let support = self.support();
        if let Some(s) = support.as_interval()
            && !is_neg_inf(&s.lower)
            && (lo - &s.lower).is_zero() == Some(true)
        {
            return Some(ctx.one());
        }
        let at = match support.kind() {
            Kind::Continuous => lo.clone(),
            Kind::Discrete => lo - ctx.one(),
        };
        if self.0.cdf(&at).is_none() && self.0.sf(&at).is_none() {
            return None;
        }
        Some(self.sf_on_support(&at))
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
        let Some(iv) = support.as_interval() else {
            // Points: the indicator sum is already total.
            return self.cdf_on_support(x);
        };
        let (lo, hi) = (&iv.lower, &iv.upper);
        let lo_finite = !is_neg_inf(lo);
        let hi_finite = !is_pos_inf(hi);
        // Decide the branch for a numeric argument (`sign_of`: on its value
        // where the symbolic test cannot, `3 − (√2 + 10⁻¹⁰⁰) > 0`, rather
        // than leave a `Piecewise` whose conditions `evalf` cannot decide).
        let from_lo = if lo_finite { sign_of(&(x - lo)) } else { None };
        let to_hi = if hi_finite { sign_of(&(hi - x)) } else { None };
        if from_lo == Some(Ordering::Less) {
            return ctx.zero();
        }
        // A density puts no mass on the lower end itself, and the closed
        // form need not fold there (`Φ(ln 0)` for a log-normal).
        if support.kind() == Kind::Continuous && from_lo == Some(Ordering::Equal) {
            return ctx.zero();
        }
        if matches!(to_hi, Some(Ordering::Less | Ordering::Equal)) {
            return ctx.one();
        }
        let on = self.cdf_on_support(x);
        if (!lo_finite || matches!(from_lo, Some(Ordering::Greater | Ordering::Equal)))
            && (!hi_finite || to_hi == Some(Ordering::Greater))
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

    /// Survival function `P(X > x)` as an expression in `x`, on the whole
    /// line: `1` below the support, `0` at or above its upper end, and on
    /// the support the complement of the CDF — written, for a numeric `x`
    /// in the far upper tail, in the family's non-cancelling form
    /// ([`Family::sf`]: `½ erfc(z/√2)` for a normal, `Γ(k, x)/Γ(k)` for a
    /// gamma), so that it keeps its digits where `1 − F(x)` would evaluate
    /// to `0`.  A symbolic `x` gets a `Piecewise`.  `scipy.stats.<dist>.sf`.
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::stats::Distribution;
    ///
    /// let ctx = Context::new();
    /// let n = Distribution::normal(ctx.int(0), ctx.int(1));
    /// let tail = n.sf(&ctx.int(20));
    /// assert_eq!(tail, ctx.rational(1, 2) * (ctx.int(10) * ctx.int(2).sqrt()).erfc());
    /// // mpmath: ncdf(-20) = 2.7536241186062336951e-89
    /// assert!((tail.eval_f64()? / 2.753_624_118_606_233_7e-89 - 1.0).abs() < 1e-14);
    /// # Ok::<(), SymplexError>(())
    /// ```
    pub fn sf(&self, x: &Ex) -> Ex {
        let ctx = self.context();
        let support = self.support();
        let Some(iv) = support.as_interval() else {
            return self.sf_on_support(x);
        };
        let (lo, hi) = (&iv.lower, &iv.upper);
        let lo_finite = !is_neg_inf(lo);
        let hi_finite = !is_pos_inf(hi);
        let from_lo = if lo_finite { sign_of(&(x - lo)) } else { None };
        let to_hi = if hi_finite { sign_of(&(hi - x)) } else { None };
        if from_lo == Some(Ordering::Less) {
            return ctx.one();
        }
        if support.kind() == Kind::Continuous && from_lo == Some(Ordering::Equal) {
            return ctx.one();
        }
        if matches!(to_hi, Some(Ordering::Less | Ordering::Equal)) {
            return ctx.zero();
        }
        let on = self.sf_on_support(x);
        if (!lo_finite || matches!(from_lo, Some(Ordering::Greater | Ordering::Equal)))
            && (!hi_finite || to_hi == Some(Ordering::Greater))
        {
            return if x.free_symbols().is_empty() {
                on.eval()
            } else {
                on
            };
        }
        let mut pairs: Vec<(Ex, crate::api::expr::BoolEx)> = Vec::new();
        if lo_finite {
            pairs.push((ctx.one(), x.lt(lo)));
        }
        if hi_finite {
            pairs.push((on, x.lt(hi)));
            pairs.push((ctx.zero(), ctx.bool_true()));
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

    /// The quantile at `p ∈ (0, 1)` as an `f64`.  `Normal`, `StudentT`,
    /// `ChiSquared`, `FDistribution`, `Beta`, `Gamma`, `Binomial` and
    /// `Poisson` with numeric parameters go through the `f64` kernel
    /// [`numdist`](super::numdist) (`scipy.stats.<dist>.ppf`, about
    /// `1e-15`, microseconds).  Otherwise a continuous distribution
    /// evaluates its closed form when it has one, else finds the root of
    /// `F(x) = p` by Brent's method on the compiled distribution function
    /// over a bracket grown from the support's ends.
    ///
    /// A discrete distribution returns the smallest atom `k` with
    /// `F(k) ≥ p`, `p` read as the exact binary number it is (a level that
    /// equals a jump of `F` only up to rounding is decided by that number:
    /// the double nearest `0.9` exceeds `9/10`).  For `p > ½` the same
    /// condition is decided on the smaller tail, `S(k) ≤ q` with `q = 1 − p`
    /// (exact in floating point), since `F(k)` next to `1` has no digits
    /// left to compare.  Each candidate is decided exactly — the sign of
    /// `F(k) − p` (or `S(k) − q`) through the family's non-cancelling
    /// closed forms ([`cdf`](Self::cdf), [`sf`](Self::sf)), in arbitrary
    /// precision — by a doubling search and bisection from a guess (the
    /// closed-form quantile, a walk of the compiled mass function, the
    /// mean).  A family without a closed form for that tail
    /// (`NegativeBinomial` has a survival function but no CDF,
    /// `Hypergeometric` neither) is walked instead: the mass function
    /// accumulated from the end where the tail starts (upward from the
    /// lower end for `F(k) ≥ p`, downward from a finite upper end for
    /// `S(k) ≤ q`), summed exactly wherever the `f64` sum lies within
    /// `2⁻⁴⁴` of the level.  An upper level on a lattice unbounded above
    /// with neither closed form is walked upward and resolved only to the
    /// rounding of the sum (`n·2⁻⁵³`).  A table of listed values is summed
    /// like a walk.  Parameters must evaluate numerically.
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::stats::Distribution;
    ///
    /// let ctx = Context::new();
    /// // scipy: stats.t.ppf(0.975, 5) = 2.5705818356363146
    /// let t5 = Distribution::student_t(ctx.int(5));
    /// assert!((t5.quantile_f64(0.975)? - 2.570_581_835_636_314_6).abs() < 1e-14);
    /// // S(k) = (2/3)^k: S(90) = 1.42e-16 > 2⁻⁵³ ≥ S(91) = 9.46e-17
    /// // (mpmath; scipy's geom.ppf compares F(k) next to 1 and says 90).
    /// let g = Distribution::geometric(ctx.rational(1, 3));
    /// assert_eq!(g.quantile_f64(1.0 - f64::EPSILON / 2.0)?, 91.0);
    /// # Ok::<(), SymplexError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] for `p` outside `(0, 1)` or an
    /// invalid numeric parameter; [`SymplexError::Unevaluable`] for
    /// symbolic parameters; [`SymplexError::ComputationFailed`] if no
    /// bracket is found.
    pub fn quantile_f64(&self, p: f64) -> Result<f64, SymplexError> {
        if !(p > 0.0 && p < 1.0) {
            return Err(SymplexError::invalid_argument(
                "quantile_f64",
                format!("p must lie strictly between 0 and 1, got {p}"),
            ));
        }
        if let Some(q) = self.kernel_quantile_f64(p) {
            return q;
        }
        let support = self.support();
        if support.kind() == Kind::Discrete {
            return self.discrete_quantile_f64(p, &support);
        }
        let ctx = self.context();
        if let Some(q) = self.0.quantile(&ctx.from_f64(p)?) {
            return q.eval_f64();
        }
        self.continuous_quantile_f64(p, &support)
    }

    /// [`quantile_f64`](Self::quantile_f64) of a density without a closed
    /// quantile or an `f64` kernel: the root of the smaller tail against
    /// its level ([`Level`]), `F(x) = p` for `p ≤ ½` and `S(x) = q` with
    /// `q = 1 − p` otherwise, compared relative to the level — `(F − p)/p`,
    /// `(q − S)/q` — and located to a relative `4ε` of `x` (no absolute
    /// floor above the subnormals).  A probe of the tail is
    /// [`cdf_on_support`](Self::cdf_on_support) /
    /// [`sf_on_support`](Self::sf_on_support) at the numeric point, which
    /// take the family's non-cancelling form in a far tail (not the
    /// whole-line forms, whose `eval` would fold the exact value).  The
    /// classic CDF, compiled to `f64` where every node has a kernel, only
    /// supplies a starting estimate; it is `1 − S` rounded next to 1 and
    /// `F` with an absolute error of order `ε` next to 0.
    ///
    /// 0.28 bracketed and solved `F_classic(x) − p` to an absolute `2·10⁻¹²`:
    /// near `p = 1` the rounded CDF put the root anywhere it reads 1
    /// (`2·T₃ + 1` at `1 − 2⁻⁵³`: `524287`, whose tail is `6.1·10⁻¹⁷`), and
    /// small quantiles were decided to that absolute width (the minimum of
    /// five `Exp(1)` at `10⁻¹²`: `0`, truly `2·10⁻¹³`).
    fn continuous_quantile_f64(&self, p: f64, support: &Support) -> Result<f64, SymplexError> {
        // A tail may still fold to a number at one point (`F(0) = ½` for a
        // Student t of any ν), but not along the search.
        if let Some((name, _)) = self
            .parameters()
            .into_iter()
            .find(|(_, e)| !e.free_symbols().is_empty())
        {
            return Err(SymplexError::Unevaluable {
                reason: format!("quantile_f64: the parameter {name} is symbolic"),
            });
        }
        let ctx = self.context();
        let level = Level::of(p);
        // The hull of the pieces (one interval, or a mixture's several).
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for piece in support.pieces() {
            let (a, b) = match piece {
                Piece::Interval(iv) => (lattice_end(&iv.lower)?, lattice_end(&iv.upper)?),
                Piece::Point(v) => {
                    let v = v.eval_f64()?;
                    (v, v)
                }
            };
            lo = lo.min(a);
            hi = hi.max(b);
        }
        if support.is_empty() || lo.is_nan() || hi.is_nan() {
            return Err(SymplexError::computation_failed(
                "quantile_f64",
                "unsupported support shape",
            ));
        }
        // The level's tail at a numeric point, as a relative miss that
        // increases with `v`: `(F − p)/p` or `(q − S)/q`.  At or beyond an
        // end of the hull `F` is 0 or 1.
        let g = |v: f64| -> f64 {
            let (lvl, sign) = match level {
                Level::Lower(p) => (p, 1.0),
                Level::Upper(q) => (q, -1.0),
            };
            let t = if v.is_nan() {
                f64::NAN
            } else if v <= lo || v >= hi {
                let below = v <= lo;
                match level {
                    Level::Lower(_) => f64::from(u8::from(!below)),
                    Level::Upper(_) => f64::from(u8::from(below)),
                }
            } else {
                let Ok(at) = ctx.from_f64(v) else {
                    return f64::NAN;
                };
                let tail = match level {
                    Level::Lower(_) => self.cdf_on_support(&at),
                    Level::Upper(_) => self.sf_on_support(&at),
                };
                tail.eval_f64().unwrap_or(f64::NAN)
            };
            sign * (t - lvl) / lvl
        };
        let x0 = self.classic_quantile_estimate(p, lo, hi);
        let (a, b) = grow_bracket(&g, x0, lo, hi)?;
        if a == b {
            return Ok(a);
        }
        let opts = crate::domains::optimize::RootOpts {
            xtol: f64::MIN_POSITIVE,
            rtol: 4.0 * f64::EPSILON,
            max_iter: 1200,
        };
        crate::domains::optimize::brent_root(g, a, b, &opts)
            .map_err(|e| SymplexError::computation_failed("quantile_f64", e.to_string()))
    }

    /// A starting point for [`continuous_quantile_f64`]: the root of the
    /// classic CDF compiled to `f64`, when it compiles and brackets `p` on
    /// `[lo, hi]` (grown from a finite end, or from `±1`).  Only an
    /// estimate: see there.
    ///
    /// [`continuous_quantile_f64`]: Self::continuous_quantile_f64
    fn classic_quantile_estimate(&self, p: f64, lo: f64, hi: f64) -> Option<f64> {
        let x = self.fresh_var("x", &[]);
        let name = x.to_string();
        let compiled = self.cdf(&x).compile(&[name.as_str()]).ok()?;
        let g = |v: f64| compiled.call(&[v]) - p;
        let (mut a, mut b) = match (lo.is_finite(), hi.is_finite()) {
            (true, true) => (lo, hi),
            (true, false) => (lo, lo + 1.0),
            (false, true) => (hi - 1.0, hi),
            (false, false) => (-1.0, 1.0),
        };
        let mut step = 1.0;
        for _ in 0..1100 {
            let (ga, gb) = (g(a), g(b));
            if ga <= 0.0 && gb >= 0.0 {
                let opts = crate::domains::optimize::RootOpts {
                    xtol: f64::MIN_POSITIVE,
                    ..Default::default()
                };
                return crate::domains::optimize::brent_root(g, a, b, &opts)
                    .ok()
                    .filter(|v| v.is_finite());
            }
            if (lo.is_finite() && ga > 0.0) || (hi.is_finite() && gb < 0.0) || ga.is_nan() {
                return None;
            }
            step *= 2.0;
            if ga > 0.0 {
                a -= step;
            }
            if gb < 0.0 {
                b += step;
            }
            if !(a.is_finite() && b.is_finite()) {
                return None;
            }
        }
        None
    }

    /// [`quantile_f64`](Self::quantile_f64) through the `f64` kernel
    /// [`numdist`](super::numdist) for the eight families it covers, when
    /// every parameter is numeric; `None` leaves it to the generic route.
    /// `p` goes to the lattice kernels as given: their inversion already
    /// decides `F(k) ≥ p` on the smaller tail.  (0.27 subtracted an
    /// absolute slack of `10⁻¹²` from `p`, which turned every level below
    /// `10⁻¹²` into the smallest normal double and every level within
    /// `10⁻¹²` of `1` into a smaller one: `poisson(10⁶)` at `10⁻¹³` gave
    /// 962716 and `poisson(7/3)` at `1 − 2⁻⁵³` gave 20, for 992660 and 24.)
    fn kernel_quantile_f64(&self, p: f64) -> Option<Result<f64, SymplexError>> {
        use super::continuous::{Beta, ChiSquared, FDistribution, Gamma, Normal, StudentT};
        use super::discrete::{Binomial, Poisson};
        use super::numdist;
        let num = |e: &Ex| -> Option<f64> {
            if e.free_symbols().is_empty() {
                e.eval_f64().ok().filter(|v| v.is_finite())
            } else {
                None
            }
        };
        if let Some(d) = self.downcast_ref::<Normal>() {
            let (mean, std) = (num(&d.mean)?, num(&d.std)?);
            if std <= 0.0 {
                return None;
            }
            return Some(numdist::norm::ppf(p).map(|z| mean + std * z));
        }
        if let Some(d) = self.downcast_ref::<StudentT>() {
            return Some(numdist::t::ppf(p, num(&d.dof)?));
        }
        if let Some(d) = self.downcast_ref::<ChiSquared>() {
            return Some(numdist::chi2::ppf(p, num(&d.dof)?));
        }
        if let Some(d) = self.downcast_ref::<FDistribution>() {
            return Some(numdist::f::ppf(p, num(&d.d1)?, num(&d.d2)?));
        }
        if let Some(d) = self.downcast_ref::<Beta>() {
            return Some(numdist::beta::ppf(p, num(&d.alpha)?, num(&d.beta)?));
        }
        if let Some(d) = self.downcast_ref::<Gamma>() {
            return Some(numdist::gamma::ppf(p, num(&d.shape)?, num(&d.scale)?));
        }
        if let Some(d) = self.downcast_ref::<Binomial>() {
            let (n, prob) = (num(&d.n)?, num(&d.p)?);
            if n.fract() != 0.0 {
                return None;
            }
            return Some(numdist::binom::ppf(p, n, prob));
        }
        if let Some(d) = self.downcast_ref::<Poisson>() {
            return Some(numdist::poisson::ppf(p, num(&d.rate)?));
        }
        // `aX + b` of a density: the inner quantile at `p` (`a > 0`), or at
        // `1 − p` (`a < 0`) where that is exact, `p ≥ ½` (Sterbenz); a
        // lattice reflection is left to the lattice route (see
        // `Affine::quantile`).
        if let Some(d) = self.downcast_ref::<super::wrappers::Affine>()
            && d.inner.kind() == Kind::Continuous
            && (d.increasing || p >= 0.5)
        {
            let (a, b) = (num(&d.a)?, num(&d.b)?);
            let at = if d.increasing { p } else { 1.0 - p };
            return Some(d.inner.quantile_f64(at).map(|x| a * x + b));
        }
        None
    }

    /// [`quantile_f64`](Self::quantile_f64) on a discrete support: the
    /// smallest atom whose tail has crossed the level (see there).  Nothing
    /// symbolic is built for the whole line: a CDF with a symbolic argument
    /// is a `Sum` for a family without a closed form, and closing it
    /// (`Hypergeometric`) took seconds.
    fn discrete_quantile_f64(&self, p: f64, support: &Support) -> Result<f64, SymplexError> {
        let level = Level::of(p);
        if let Some(values) = support.as_points() {
            return self.table_quantile_f64(level, &values);
        }
        let probe = self.fresh_var("x", &[]);
        let closed = self.0.cdf(&probe).is_some()
            || (matches!(level, Level::Upper(_)) && self.0.sf(&probe).is_some());
        let lattice = support.normalize_lattice();
        let Some(iv) = lattice.as_interval() else {
            return self.pieces_quantile_f64(level, &lattice);
        };
        let (lo, hi) = (lattice_end(&iv.lower)?, lattice_end(&iv.upper)?);
        if closed {
            let k0 = self.lattice_guess(p, level, lo, hi)?;
            return self.lattice_search(level, k0, lo, hi);
        }
        if let Some(k) = self.lattice_walk(level, lo, hi, false)? {
            return Ok(k);
        }
        // Neither a closed form nor a walk: decide each probe on the summed
        // CDF (exact, and slow).
        self.lattice_search(level, self.mean_f64().unwrap_or(f64::NAN), lo, hi)
    }

    /// The closed-form mean as a finite `f64`.
    fn mean_f64(&self) -> Option<f64> {
        self.0
            .mean()
            .and_then(|m| m.eval_f64().ok())
            .filter(|v| v.is_finite())
    }

    /// Has the tail that `level` names, at the atom `x` of the support,
    /// crossed the level?  `F(x) ≥ p` or `S(x) ≤ q`, from
    /// [`cdf_on_support`](Self::cdf_on_support) /
    /// [`sf_on_support`](Self::sf_on_support) (the non-cancelling form in a
    /// far tail) and [`Level::crossed`].  Not the whole-line `cdf`/`sf`:
    /// their `eval` of the numeric value would fold the exact form once
    /// more (`1 − (1 − 10⁻¹⁵)¹⁰⁰⁰` is a 15 000-digit rational).
    fn tail_crossed(&self, level: Level, x: &Ex) -> Result<bool, SymplexError> {
        let tail = match level {
            Level::Lower(_) => self.cdf_on_support(x),
            Level::Upper(_) => self.sf_on_support(x),
        };
        level.crossed(&tail)
    }

    /// The smallest lattice point of `lo..=hi` whose tail has crossed
    /// `level`, each probe decided exactly ([`tail_crossed`]), by
    /// [`numdist`](super::numdist)'s doubling search and bisection from the
    /// guess `k0` (the first finite one of `k0`, `lo`, `hi`, `0`).
    ///
    /// [`tail_crossed`]: Self::tail_crossed
    fn lattice_search(&self, level: Level, k0: f64, lo: f64, hi: f64) -> Result<f64, SymplexError> {
        if lo.is_nan() || hi.is_nan() || lo > hi {
            return Err(SymplexError::computation_failed(
                "quantile_f64",
                format!("empty lattice {lo}..={hi}"),
            ));
        }
        let ctx = self.context();
        let failure = std::cell::RefCell::new(None);
        let crossed = |k: f64| match ctx.from_f64(k).and_then(|kx| self.tail_crossed(level, &kx)) {
            Ok(c) => Some(c),
            Err(e) => {
                failure.borrow_mut().get_or_insert(e);
                None
            }
        };
        let k0 = [k0, lo, hi]
            .into_iter()
            .find(|v| v.is_finite())
            .unwrap_or(0.0);
        super::numdist::discrete_search("quantile_f64", crossed, k0, lo, hi)
            .map_err(|e| failure.into_inner().unwrap_or(e))
    }

    /// A starting point for [`lattice_search`](Self::lattice_search): the
    /// closed-form quantile (`DiscreteUniform`), else a walk of the
    /// compiled mass function, else the mean; `NaN` when there is none.
    fn lattice_guess(&self, p: f64, level: Level, lo: f64, hi: f64) -> Result<f64, SymplexError> {
        let ctx = self.context();
        if let Some(q) = self.0.quantile(&ctx.from_f64(p)?)
            && let Ok(v) = q.eval_f64()
            && v.is_finite()
        {
            return Ok(v);
        }
        if let Ok(Some(k)) = self.lattice_walk(level, lo, hi, true) {
            return Ok(k);
        }
        Ok(self.mean_f64().unwrap_or(f64::NAN))
    }

    /// Walk the lattice `lo..=hi` accumulating the mass function from the
    /// end where the tail named by `level` starts, to the first point
    /// where it crosses: upward from a finite `lo` for `F(k) ≥ p`,
    /// downward from a finite `hi` for `S(k) ≤ q`.  Both are sums of
    /// positive terms, kept to a couple of ulps by compensated summation
    /// beyond the few ulps of each compiled term; where the `f64` sum lies
    /// within [`TIE_BAND`] of the level the decision is taken on the exact
    /// sum of the masses instead.  An upper level on a lattice unbounded
    /// above is walked upward comparing `F(k)` with `1 − q` — resolved only
    /// to the sum's rounding, and stopped at `1 − 4·2⁻⁵²`.  The mass
    /// function is compiled to `f64` when every node has a kernel, else (and
    /// wherever a compiled value is not finite) evaluated exactly at each
    /// atom.  A `guess` (for [`lattice_search`](Self::lattice_search), which
    /// decides exactly) needs the compiled mass function and decides in
    /// `f64` throughout.  `None` when no direction applies or the walk
    /// exceeds `LATTICE_WALK_LIMIT` atoms.
    fn lattice_walk(
        &self,
        level: Level,
        lo: f64,
        hi: f64,
        guess: bool,
    ) -> Result<Option<f64>, SymplexError> {
        const LATTICE_WALK_LIMIT: usize = 100_000;
        let ctx = self.context();
        let k_var = self.fresh_var("k", &[]);
        let compiled = self
            .0
            .density(&k_var)
            .compile(&[k_var.to_string().as_str()])
            .ok();
        if compiled.is_none() && guess {
            return Ok(None);
        }
        let exact_pmf =
            |k: f64| -> Result<Ex, SymplexError> { Ok(self.0.density(&ctx.from_f64(k)?).eval()) };
        let pmf = |k: f64| -> Result<f64, SymplexError> {
            match compiled.as_ref().map(|f| f.call(&[k])) {
                Some(v) if v.is_finite() => Ok(v),
                _ => exact_pmf(k)?.eval_f64(),
            }
        };
        // Has the tail crossed, from its `f64` sum `t` — or, within the
        // band of a walk that answers, from the exact sum of f over a..=b?
        let decide = |t: f64, a: f64, b: f64| -> Result<bool, SymplexError> {
            if let Some(c) = level.crossed_f64(t) {
                return Ok(c);
            }
            if guess {
                return Ok(match level {
                    Level::Lower(p) => t >= p,
                    Level::Upper(q) => t <= q,
                });
            }
            let mut acc = ctx.zero();
            let mut j = a;
            while j <= b {
                acc += exact_pmf(j)?;
                j += 1.0;
            }
            level.crossed(&acc)
        };
        let mut acc = CompensatedSum::default();
        match level {
            Level::Lower(_) if lo.is_finite() => {
                let mut k = lo;
                for _ in 0..LATTICE_WALK_LIMIT {
                    acc.add(pmf(k)?);
                    if decide(acc.value(), lo, k)? || k >= hi {
                        return Ok(Some(k));
                    }
                    k += 1.0;
                }
                Ok(None)
            }
            Level::Upper(_) if hi.is_finite() => {
                // S(hi) = 0 ≤ q; going down, S(k − 1) = S(k) + f(k).
                let mut k = hi;
                for _ in 0..LATTICE_WALK_LIMIT {
                    if k <= lo {
                        return Ok(Some(k));
                    }
                    acc.add(pmf(k)?);
                    if !decide(acc.value(), k, hi)? {
                        return Ok(Some(k));
                    }
                    k -= 1.0;
                }
                Ok(None)
            }
            Level::Upper(q) if lo.is_finite() => {
                let target = (1.0 - q).min(1.0 - 4.0 * f64::EPSILON);
                let mut k = lo;
                for _ in 0..LATTICE_WALK_LIMIT {
                    acc.add(pmf(k)?);
                    if acc.value() >= target {
                        return Ok(Some(k));
                    }
                    k += 1.0;
                }
                Ok(None)
            }
            _ => Ok(None),
        }
    }

    /// A distribution on listed values (a table, a mixture of tables): the
    /// smallest value whose tail has crossed `level`, the masses summed in
    /// `f64` from the end where the tail starts (from the bottom for
    /// `F ≥ p`, from the top for `S ≤ q`) and exactly wherever the sum lies
    /// within [`TIE_BAND`] of the level.  A value listed twice (a mixture's
    /// shared atom) carries its whole mass once.  (0.27 compared the sum
    /// from the bottom with `p − 10⁻¹²`.)
    fn table_quantile_f64(&self, level: Level, values: &[Ex]) -> Result<f64, SymplexError> {
        let mut seen: Vec<&Ex> = Vec::with_capacity(values.len());
        let mut pts: Vec<(f64, Ex, f64)> = Vec::with_capacity(values.len());
        for v in values {
            if seen.contains(&v) {
                continue;
            }
            seen.push(v);
            let mass = self.0.density(v).eval();
            let m = mass.eval_f64()?;
            pts.push((v.eval_f64()?, mass, m));
        }
        pts.sort_by(|a, b| a.0.total_cmp(&b.0));
        let ctx = self.context();
        let exact_sum = |range: &[(f64, Ex, f64)]| -> Ex {
            range
                .iter()
                .fold(ctx.zero(), |acc, (_, mass, _)| acc + mass)
        };
        let mut acc = CompensatedSum::default();
        match level {
            Level::Lower(_) => {
                for (i, (v, _, m)) in pts.iter().enumerate() {
                    acc.add(*m);
                    let crossed = match level.crossed_f64(acc.value()) {
                        Some(c) => c,
                        None => level.crossed(&exact_sum(&pts[..=i]))?,
                    };
                    if crossed {
                        return Ok(*v);
                    }
                }
                Err(SymplexError::computation_failed(
                    "quantile_f64",
                    "the masses do not reach p",
                ))
            }
            Level::Upper(_) => {
                // S(v_last) = 0 ≤ q; going down, S(vᵢ) = S(vᵢ₊₁) + mᵢ₊₁.
                let Some(&(mut answer, _, _)) = pts.last() else {
                    return Err(SymplexError::computation_failed(
                        "quantile_f64",
                        "no values listed",
                    ));
                };
                for i in (1..pts.len()).rev() {
                    acc.add(pts[i].2);
                    let crossed = match level.crossed_f64(acc.value()) {
                        Some(c) => c,
                        None => level.crossed(&exact_sum(&pts[i..]))?,
                    };
                    if !crossed {
                        break;
                    }
                    answer = pts[i - 1].0;
                }
                Ok(answer)
            }
        }
    }

    /// A discrete support of several pieces (a mixture of a lattice family
    /// and a table): the smallest crossing atom of each piece — a lattice
    /// interval searched as in [`lattice_search`](Self::lattice_search), a
    /// listed value decided directly — and the least of those.
    fn pieces_quantile_f64(&self, level: Level, lattice: &Support) -> Result<f64, SymplexError> {
        let mut best: Option<f64> = None;
        for piece in lattice.pieces() {
            let found = match piece {
                Piece::Point(v) => {
                    if self.tail_crossed(level, v)? {
                        Some(v.eval_f64()?)
                    } else {
                        None
                    }
                }
                Piece::Interval(iv) => {
                    let (a, b) = (lattice_end(&iv.lower)?, lattice_end(&iv.upper)?);
                    if b.is_finite() && !self.tail_crossed(level, &iv.upper)? {
                        None
                    } else {
                        Some(self.lattice_search(level, f64::NAN, a, b)?)
                    }
                }
            };
            if let Some(k) = found {
                best = Some(best.map_or(k, |b| b.min(k)));
            }
        }
        best.ok_or_else(|| {
            SymplexError::computation_failed("quantile_f64", "the masses do not reach p")
        })
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
    ///     vec![symplex::stats::Piece::Interval(Interval::open(ctx.int(2), ctx.infinity()))],
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
                Piece::Interval(iv) => self.interval_mass(iv, &support),
                Piece::Point(v) => match support.kind() {
                    Kind::Continuous => ctx.zero(),
                    Kind::Discrete => self.0.density(v).eval(),
                },
            };
        }
        Ok(total.simplify())
    }

    /// `P(X ∈ iv)` for an interval already clipped to the support.  The
    /// interval's openness plays no part here: on the lattice it has been
    /// folded into integer ends by `normalize_lattice`, and a continuous
    /// distribution puts no mass on an endpoint.
    fn interval_mass(&self, iv: &Interval<Ex>, support: &Support) -> Ex {
        let ctx = self.context();
        let (lo, hi) = (&iv.lower, &iv.upper);
        // A density puts no mass on a single point (`X ≤ 0` clipped to
        // `[0, ∞)` is `[0, 0]`, where a closed form need not fold: `Φ(ln 0)`).
        if support.kind() == Kind::Continuous && (hi - lo).is_zero() == Some(true) {
            return ctx.zero();
        }
        // F at the support's own upper end is 1 by definition (the closed
        // form need not fold there); `mass_below` does the same at the
        // lower end.
        let at_upper_end = is_pos_inf(hi)
            || support
                .as_interval()
                .is_some_and(|s| (hi - &s.upper).is_zero() == Some(true));
        let probe = self.fresh_var("t", &[lo, hi]);
        let has_cdf = self.0.cdf(&probe).is_some();
        let has_sf = self.0.sf(&probe).is_some();
        // Up to the support's upper end: `P(X ≥ lo)` directly, the survival
        // function in the far upper tail (`1 − F` evaluates to `0` there).
        if at_upper_end
            && (has_cdf || has_sf)
            && let Some(above) = self.mass_above(lo)
        {
            return above.simplify();
        }
        if has_cdf {
            // An interval in the far upper tail: `P(X ≥ lo) − P(X > hi)`,
            // two survival functions, instead of `F(hi) − F(lo⁻)`, two
            // numbers next to `1`.  `hi` is an integer after lattice
            // normalisation, so on either kind `P(X > hi)` is `S(hi)`.
            if has_sf
                && self.tail_of(lo) == Tail::Upper
                && let Some(above) = self.mass_above(lo)
            {
                return (above - self.sf_on_support(hi)).simplify();
            }
            // Otherwise `F(hi) − F(lo⁻)`, each in its non-cancelling form.
            if let Some(below) = self.mass_below(lo) {
                return (self.cdf_on_support(hi) - below).simplify();
            }
        }
        let t = probe;
        let dens = self.0.density(&t);
        support.accumulate(&dens, &t, lo, hi).simplify()
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
    /// exponential; when that leaves an unevaluated `Integral`/`Sum` the
    /// expanded integrand is tried (`x·(1 − e^{−λx})·e^{−λx}`, the mean of
    /// a maximum of exponentials, closes term by term).
    pub(crate) fn integrate_over(&self, integrand: &Ex, x: &Ex, region: &Support) -> Ex {
        let ctx = self.context();
        let integrand = integrand.simplify();
        // Integer ends for a lattice region (open ends moved inwards).
        let region = region.normalize_lattice();
        let over = |iv: &Interval<Ex>, f: &Ex| region.accumulate(f, x, &iv.lower, &iv.upper);
        let mut acc = ctx.zero();
        for piece in region.pieces() {
            acc += match piece {
                Piece::Interval(iv) => {
                    let first = over(iv, &integrand);
                    if first.has_unevaluated() {
                        let expanded = integrand.expand();
                        if expanded != integrand {
                            let second = over(iv, &expanded);
                            if !second.has_unevaluated() {
                                acc += second;
                                continue;
                            }
                        }
                    }
                    first
                }
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
    /// finite support (discrete).  Every built-in family has a route, each
    /// exact in distribution (see the [module docs](super)); a custom
    /// [`Family`] on an infinite lattice or without a quantile needs its
    /// own [`Family::sampler`].  Parameters must evaluate numerically.
    /// SymPy: `sample(X, size=n)`.
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::stats::{Distribution, Rng};
    ///
    /// let ctx = Context::new();
    /// let g = Distribution::gamma(ctx.int(3), ctx.int(2));   // mean 6
    /// let s = g.sample(4000, &mut Rng::new(1))?;
    /// let mean = s.iter().sum::<f64>() / 4000.0;
    /// assert!((mean - 6.0).abs() < 0.3);
    /// # Ok::<(), SymplexError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`SymplexError::NotImplemented`] if the family has no route;
    /// [`SymplexError::Unevaluable`] if a parameter is symbolic;
    /// [`SymplexError::InvalidArgument`] if a numeric parameter is outside
    /// the family's domain (an unchecked constructor accepted it), or if `n`
    /// samples cannot be allocated.
    pub fn sample(&self, n: usize, rng: &mut Rng) -> Result<Vec<f64>, SymplexError> {
        let mut sampler = self.sampler()?;
        let mut out = Vec::new();
        out.try_reserve_exact(n).map_err(|_| {
            SymplexError::invalid_argument("sample", format!("cannot allocate {n} samples"))
        })?;
        out.extend((0..n).map(|_| sampler(rng)));
        Ok(out)
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
                let Some(iv) = support.as_interval() else {
                    return Err(not_implemented(format!(
                        "sampling {}: the support is not a single integer range",
                        self.name()
                    )));
                };
                let (lo, hi) = (&iv.lower, &iv.upper);
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
