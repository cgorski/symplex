//! Order statistics: the distribution of the `k`-th smallest of `n`
//! independent draws from a distribution, as a [`Family`] of its own.
//!
//! For a continuous parent with distribution function `F` and density `f`
//! (David & Nagaraja, *Order Statistics*, §2.1),
//!
//! * density: `f_(k)(x) = k·C(n, k)·F(x)^{k−1}·(1 − F(x))^{n−k}·f(x)`,
//! * distribution function: `F_(k)(x) = I_{F(x)}(k, n − k + 1)`, the
//!   regularised incomplete beta function (`Σ_{j≥k} C(n, j) F^j (1−F)^{n−j}`),
//!
//! and the support is the parent's.  Moments go through the generic
//! machinery of [`Distribution`] (exact integration of the density), so
//! for `Uniform(0, 1)` the `k`-th of `n` has mean `k/(n + 1)` exactly.
//!
//! For a discrete parent on a finite set of values the order statistic is
//! enumerated exactly into a [`Finite`](super::Finite) table:
//! `P(X_(k) = v) = P(X_(k) ≤ v) − P(X_(k) ≤ v⁻)` with
//! `P(X_(k) ≤ v) = Σ_{j≥k} C(n, j) F(v)^j (1 − F(v))^{n−j}`.  On an
//! infinite integer lattice the same difference `I_{F(x)} − I_{F(x−1)}` is
//! the mass function of the [`OrderStatistic`] family.
//!
//! The masses of a table are numbers: where the parent's tails `F(v)` are
//! rationals they are computed in integers over the common denominator
//! (never by expanding the polynomial above in a symbolic `F`), and a
//! finite lattice whose exact masses would be too long (`EXACT_TABLE_BUDGET`:
//! `Binomial(843, p)` with a 14-digit `p`) keeps the lattice formula of
//! the family instead of a table.  The first of one draw is the parent
//! itself, and the minimum and the maximum use the parent's tails directly:
//! `P(max ≤ x) = F(x)ⁿ`, `P(min > x) = S(x)ⁿ`.
//!
//! ```
//! use symplex::prelude::*;
//! use symplex::stats::Distribution;
//! use symplex::stats::order::{maximum_of, minimum_of};
//!
//! let ctx = Context::new();
//! let u = Distribution::uniform(ctx.int(0), ctx.int(1));
//! let m = maximum_of(&u, 3)?;                       // density 3x² on [0, 1]
//! assert_eq!(m.mean(), ctx.rational(3, 4));          // n/(n+1)
//! let d = minimum_of(&Distribution::die(ctx.int(6)), 2)?;
//! assert_eq!(d.density(&ctx.int(1)).eval(), ctx.rational(11, 36));
//! # Ok::<(), SymplexError>(())
//! ```

use std::fmt;

use crate::api::context::Context;
use crate::api::expr::Ex;
use num_bigint::BigInt;
use num_traits::{One, Zero};

use crate::base::errors::SymplexError;
use crate::base::interval::{Interval, IntervalKind};
use crate::base::numeric::{Q, gcd_big};
use crate::base::truth::Truth;

use super::data::binomial_q;
use super::family::{Distribution, Family, Sampler, TailOrders, family_boilerplate};
use super::sample::Rng;
use super::support::{Kind, Support, is_neg_inf, is_pos_inf};

const OP: &str = "stats::order";

fn invalid(reason: impl Into<String>) -> SymplexError {
    SymplexError::invalid_argument(OP, reason)
}

/// The largest finite discrete support enumerated into an exact table;
/// beyond it the lattice formula is used instead.
const MAX_ENUMERATED: i64 = 10_000;

/// The `k`-th smallest of `n` independent copies of `inner` (`1 ≤ k ≤ n`;
/// `k = 1` is the minimum, `k = n` the maximum).
#[derive(Clone, Debug, PartialEq)]
pub struct OrderStatistic {
    /// The parent distribution.
    pub inner: Distribution,
    /// Sample size `n ≥ 1`.
    pub n: usize,
    /// Rank `k`, `1 ≤ k ≤ n`.
    pub k: usize,
}

impl OrderStatistic {
    /// The family after validating `1 ≤ k ≤ n`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `n = 0` or `k ∉ [1, n]`.
    pub fn try_new(inner: Distribution, n: usize, k: usize) -> Result<Self, SymplexError> {
        if n == 0 {
            return Err(invalid("an order statistic needs a sample of size n ≥ 1"));
        }
        if k == 0 || k > n {
            return Err(invalid(format!(
                "the rank k must satisfy 1 ≤ k ≤ n, got k = {k}, n = {n}"
            )));
        }
        Ok(OrderStatistic { inner, n, k })
    }

    fn ctx(&self) -> Context {
        self.inner.context()
    }

    /// `n` and `k` as expressions.
    fn nk(&self) -> (Ex, Ex) {
        let ctx = self.ctx();
        (ctx.int(usize_to_i64(self.n)), ctx.int(usize_to_i64(self.k)))
    }

    /// The parent's `F(x)` as a single expression: `0` / `1` when `x` is
    /// decidably below / above the support, else the parent's
    /// [`cdf_on_support`](Distribution::cdf_on_support) — the closed form
    /// when the family has one, else the integral / sum of the density
    /// from the support's lower end.  (The whole-line [`Distribution::cdf`]
    /// would wrap a symbolic `x` in a `Piecewise`.)
    fn parent_cdf(&self, x: &Ex) -> Ex {
        let ctx = self.ctx();
        if let Some(iv) = self.inner.support().as_interval() {
            let (lo, hi) = (&iv.lower, &iv.upper);
            if !is_neg_inf(lo) && Truth::from((x - lo).is_negative()).is_true() {
                return ctx.zero();
            }
            if !is_pos_inf(hi) && Truth::from((x - hi).is_nonnegative()).is_true() {
                return ctx.one();
            }
        }
        self.inner.cdf_on_support(x)
    }

    /// The parent's `P(X > x)` as a single expression, as
    /// [`parent_cdf`](Self::parent_cdf): `1` / `0` when `x` is decidably
    /// below / above the support, else the parent's `sf_on_support` (its
    /// non-cancelling survival function in the far upper tail).
    fn parent_sf(&self, x: &Ex) -> Ex {
        let ctx = self.ctx();
        if let Some(iv) = self.inner.support().as_interval() {
            let (lo, hi) = (&iv.lower, &iv.upper);
            if !is_neg_inf(lo) && Truth::from((x - lo).is_negative()).is_true() {
                return ctx.one();
            }
            if !is_pos_inf(hi) && Truth::from((x - hi).is_nonnegative()).is_true() {
                return ctx.zero();
            }
        }
        self.inner.sf_on_support(x)
    }

    /// `I_u(k, n − k + 1)`: `P(X_(k) ≤ x)` as a function of `u = F(x)`.
    fn beta_cdf(&self, u: &Ex) -> Ex {
        let ctx = self.ctx();
        let (n, k) = self.nk();
        u.betainc_regularized(&k, &(n - &k + 1), &ctx.zero())
    }
}

fn usize_to_i64(n: usize) -> i64 {
    i64::try_from(n).unwrap_or(i64::MAX)
}

impl Family for OrderStatistic {
    family_boilerplate!(OrderStatistic, "OrderStatistic");

    fn context(&self) -> Context {
        self.ctx()
    }

    fn parameters(&self) -> Vec<(&'static str, Ex)> {
        let (n, k) = self.nk();
        let mut p = vec![("n", n), ("k", k)];
        p.extend(self.inner.parameters());
        p
    }

    fn support(&self) -> Support {
        self.inner.support()
    }

    // Continuous: k·C(n,k)·F^{k−1}·(1−F)^{n−k}·f.
    // Lattice: I_{F(x)}(k, n−k+1) − I_{F(x−1)}(k, n−k+1); for the minimum
    // that is S(x−1)ⁿ − S(x)ⁿ in the parent's survival function (its
    // closed, non-cancelling upper tail: I_F(1, n) = 1 − (1 − F)ⁿ written
    // in powers of F loses every digit where F is near 1).  The maximum
    // keeps the general form (F(x)ⁿ − F(x−1)ⁿ is the same number, but as
    // powers of the `Piecewise` tail of a negative binomial law it takes
    // the summation of a moment six times as long to refuse).
    fn density(&self, x: &Ex) -> Ex {
        let ctx = self.ctx();
        if self.inner.kind() == Kind::Discrete && self.k == 1 && self.n > 1 {
            let n = usize_to_i64(self.n);
            let below = x - 1;
            return self.parent_sf(&below).powi(n) - self.parent_sf(x).powi(n);
        }
        let f_x = self.parent_cdf(x);
        match self.inner.kind() {
            Kind::Continuous => {
                let (_, k) = self.nk();
                let choose = ctx.from_ratio(binomial_q(self.n, self.k));
                let mut acc = k * choose * self.inner.density(x);
                if self.k > 1 {
                    acc *= f_x.powi(usize_to_i64(self.k - 1));
                }
                if self.n > self.k {
                    acc *= (ctx.one() - &f_x).powi(usize_to_i64(self.n - self.k));
                }
                acc
            }
            Kind::Discrete => {
                let below = self.parent_cdf(&(x - 1));
                self.beta_cdf(&f_x) - self.beta_cdf(&below)
            }
        }
    }

    // The maximum is below x when every draw is: F(x)ⁿ.
    fn cdf(&self, x: &Ex) -> Option<Ex> {
        let f_x = self.parent_cdf(x);
        if self.k == self.n {
            return Some(f_x.powi(usize_to_i64(self.n)));
        }
        Some(self.beta_cdf(&f_x))
    }

    // P(X_(k) > x) = 1 − I_F(k, n − k + 1) = I_{1−F}(n − k + 1, k), with
    // 1 − F the parent's survival function: the maximum of n normals
    // beyond 20 is n·½ erfc(10√2) to first order, not 1 − (1 − …)ⁿ.
    // The minimum is above x when every draw is: S(x)ⁿ.
    fn sf(&self, x: &Ex) -> Option<Ex> {
        let ctx = self.ctx();
        let (n, k) = self.nk();
        let s_x = self.parent_sf(x);
        if self.k == 1 {
            return Some(s_x.powi(usize_to_i64(self.n)));
        }
        Some(s_x.betainc_regularized(&(n - &k + 1), &k, &ctx.zero()))
    }

    // The tails of X_(k): P(X_(k) > x) ~ C(n, k−1) S(x)^{n−k+1} and
    // P(X_(k) ≤ x) ~ C(n, k) F(x)^k, so a parent's exponential rate scales
    // by n − k + 1 on the right and by k on the left (a heavy tail, end 0,
    // stays heavy).  A scaled finite end is taken as closed: whether the
    // order statistic's mgf is finite exactly there depends on the
    // parent's polynomial factor (the minimum of two Gamma(1/10) has a
    // finite mgf at t = 2/θ), so that one point keeps the integral.
    fn mgf_domain(&self) -> Option<Interval<Ex>> {
        let d = self.inner.family().mgf_domain()?;
        let (n, k) = self.nk();
        let m = &n - &k + 1;
        let scale = |v: Ex, factor: &Ex, open: bool| -> (Ex, bool) {
            if is_pos_inf(&v) || is_neg_inf(&v) || Truth::from(v.is_zero()).is_true() {
                (v, open)
            } else {
                ((factor * v).simplify(), false)
            }
        };
        let (lower, lo_open) = scale(d.lower, &k, d.kind.lower_open());
        let (upper, hi_open) = scale(d.upper, &m, d.kind.upper_open());
        Some(Interval {
            lower,
            upper,
            kind: IntervalKind::from_open_ends(lo_open, hi_open),
        })
    }

    // The same tails scale the parent's orders: S(x)^{n−k+1} with
    // S(x) ~ x^{−r} has order (n−k+1)·r on the right, F(x)^k order k·r on
    // the left (the maximum of two Cauchy variables has a left tail of
    // order 2 and no mean only from the right).
    fn tail_orders(&self) -> Option<TailOrders> {
        let parent = self.inner.family().tail_orders()?;
        let (n, k) = self.nk();
        let m = &n - &k + 1;
        Some(TailOrders {
            left: parent.left.map(|r| (&k * r).simplify()),
            right: parent.right.map(|r| (m * r).simplify()),
        })
    }

    // n draws from the parent, sorted; the k-th smallest.
    fn sampler(&self) -> Option<Result<Sampler, SymplexError>> {
        let mut inner = match self.inner.sampler() {
            Ok(s) => s,
            Err(e) => return Some(Err(e)),
        };
        let (n, k) = (self.n, self.k);
        let mut buf = vec![0.0_f64; n];
        Some(Ok(Box::new(move |rng: &mut Rng| {
            for slot in buf.iter_mut() {
                *slot = inner(rng);
            }
            buf.sort_by(f64::total_cmp);
            buf.get(k.saturating_sub(1)).copied().unwrap_or(f64::NAN)
        })))
    }

    fn fmt_display(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "OrderStatistic(k={}, n={}, {})",
            self.k, self.n, self.inner
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Constructors
// ═══════════════════════════════════════════════════════════════════════════

/// The distribution of the `k`-th smallest of `n` independent draws from
/// `dist`: an [`OrderStatistic`] family for a continuous parent (or a
/// parent on an infinite integer lattice), and an exactly enumerated
/// [`Finite`](super::Finite) table for a parent with finitely many values.
///
/// The first of one draw (`n = 1`) is the parent, which is returned as it
/// is (up to 0.60.0 a family whose `variance()` for `Poisson(4)` was a
/// `Sum` that `eval_f64` refused, where the parent says `4`).  A parent on
/// a finite integer lattice whose exact table would be too long (the
/// masses of the 2nd of 2 draws of `Binomial(843, 99999999999991/10¹⁴)`
/// are rationals of 79,000 bits each; up to 0.60.0 the call did not return
/// in 70 s) is given as the [`OrderStatistic`] family: the same law by the
/// lattice formula, its masses evaluated when they are asked for.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::Distribution;
/// use symplex::stats::order::order_statistic;
///
/// let ctx = Context::new();
/// let u = Distribution::uniform(ctx.int(0), ctx.int(1));
/// // The second smallest of five: mean 2/6, P(X_(2) ≤ 1/2) = I_{1/2}(2, 4) = 13/16
/// let x2 = order_statistic(&u, 5, 2)?;
/// assert_eq!(x2.mean(), ctx.rational(1, 3));
/// assert_eq!(x2.cdf(&ctx.rational(1, 2)), ctx.rational(13, 16));
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] if `n = 0` or `k ∉ [1, n]`, or a
/// finite parent has values that cannot be ordered numerically.
pub fn order_statistic(
    dist: &Distribution,
    n: usize,
    k: usize,
) -> Result<Distribution, SymplexError> {
    let family = OrderStatistic::try_new(dist.clone(), n, k)?;
    if n == 1 {
        return Ok(dist.clone());
    }
    if dist.kind() == Kind::Discrete
        && let Some(table) = finite_table(dist)?
        && let Some(exact) = enumerate(dist, &table, n, k)
    {
        return Ok(exact);
    }
    Ok(Distribution::from_family(family))
}

/// The minimum of `n` independent draws (`k = 1`).
///
/// # Errors
///
/// As [`order_statistic`].
pub fn minimum_of(dist: &Distribution, n: usize) -> Result<Distribution, SymplexError> {
    order_statistic(dist, n, 1)
}

/// The maximum of `n` independent draws (`k = n`).
///
/// # Errors
///
/// As [`order_statistic`].
pub fn maximum_of(dist: &Distribution, n: usize) -> Result<Distribution, SymplexError> {
    order_statistic(dist, n, n)
}

/// The `(value, mass)` table of a discrete distribution with finitely many
/// values, sorted by value; `None` for an infinite (or huge) lattice.
fn finite_table(dist: &Distribution) -> Result<Option<Vec<(Ex, Ex)>>, SymplexError> {
    let support = dist.support();
    let values: Vec<Ex> = if let Some(points) = support.as_points() {
        points
    } else if let Some(iv) = support.as_interval() {
        let (Some(lo), Some(hi)) = (iv.lower.eval().as_i64(), iv.upper.eval().as_i64()) else {
            return Ok(None);
        };
        if hi < lo || hi - lo >= MAX_ENUMERATED {
            return Ok(None);
        }
        let ctx = dist.context();
        (lo..=hi).map(|v| ctx.int(v)).collect()
    } else {
        return Ok(None);
    };
    let mut keyed: Vec<(f64, Ex)> = Vec::with_capacity(values.len());
    for v in values {
        let key = v.eval_f64().map_err(|_| {
            invalid(format!(
                "the values of a finite parent must be numeric to order them, got `{v}`"
            ))
        })?;
        keyed.push((key, v));
    }
    // The `f64` keys only pre-sort: distinct values can share a key (`1` and
    // `1 + 10⁻²⁰`, integers beyond 2⁵³, `√2` and its double), which up to
    // 0.54 kept them in the order given and swapped their order-statistic
    // masses.  Each adjacent pair is confirmed exactly and a pair out of
    // order is moved back (an insertion sort, linear on sorted input).
    keyed.sort_by(|a, b| a.0.total_cmp(&b.0));
    for i in 1..keyed.len() {
        let mut j = i;
        while j > 0 {
            match exact_order(&keyed[j - 1].1, &keyed[j].1) {
                Some(std::cmp::Ordering::Greater) => {
                    keyed.swap(j - 1, j);
                    j -= 1;
                }
                Some(_) => break,
                None => {
                    return Err(invalid(format!(
                        "cannot order the values `{}` and `{}` of a finite parent",
                        keyed[j - 1].1,
                        keyed[j].1
                    )));
                }
            }
        }
    }
    let table = keyed
        .into_iter()
        .map(|(_, v)| {
            let p = dist.density(&v).eval();
            (v, p)
        })
        .collect();
    Ok(Some(table))
}

/// The order of two numeric values, exactly: rationals compared, others by
/// the exact or certified sign of their difference; `None` when undecided.
fn exact_order(a: &Ex, b: &Ex) -> Option<std::cmp::Ordering> {
    if let (Some(x), Some(y)) = (a.as_rational(), b.as_rational()) {
        return Some(x.cmp(&y));
    }
    super::family::sign_of(&(a - b)).to_ordering()
}

/// Work allowed to the exact table of an order statistic of a finite
/// lattice, in limb operations: the number of atoms times the square of
/// the length, in 64-bit words, of a mass (`n` times the bits of the common
/// denominator of the parent's tails) — what reducing every mass to lowest
/// terms costs.  The bound is the largest count the benchmark of the 0.60.1
/// block needs, with a margin of 5 %: `Binomial(200, 123456789/10⁹)`, 6
/// draws, 201 atoms of 35,900 bits, 6.4·10⁷ (1.9 s in a release build;
/// `Binomial(843, 1/3)`, 6 draws, 1.3·10⁷: 0.5 s).  A table is worth that
/// much: the moments of a table are sums of its rationals, those of the
/// family a symbolic sum over the lattice (20 s for 844 values).  Beyond the
/// bound the family's lattice formula is the law: the 2nd of 2 draws of
/// `Binomial(843, p)` with a 14-digit `p` is 1.3·10⁹.
const EXACT_TABLE_BUDGET: u64 = 1 << 26;

/// The parent's tails `F(vᵢ)` at the sorted values of its table, as
/// integers over one denominator.
enum Tails {
    /// `F(vᵢ) = numerators[i] / denominator`.
    Exact {
        numerators: Vec<BigInt>,
        denominator: BigInt,
    },
    /// Rationals, too long for [`EXACT_TABLE_BUDGET`].
    OverBudget,
    /// Some tail is not a rational number.
    NotRational,
}

/// The work of the exact table of `atoms` masses whose tails have a common
/// denominator of `bits` bits (see [`EXACT_TABLE_BUDGET`]).
fn exact_table_work(atoms: usize, n: usize, bits: u64) -> u64 {
    let n = u64::try_from(n).unwrap_or(u64::MAX);
    let atoms = u64::try_from(atoms).unwrap_or(u64::MAX);
    let limbs = n.saturating_mul(bits) / 64 + 1;
    atoms.saturating_mul(limbs.saturating_mul(limbs))
}

/// The parent's tails as exact rationals over their common denominator:
/// the partial sums of the masses when every mass is a rational literal;
/// else, on an integer lattice, the parent's own `cdf` at each value (the
/// masses of `Binomial(843, 1/3)` are kept as products of powers, its
/// tails are rationals).  With a `budget` (a lattice, which has the
/// family's formula to fall back on) the work stops as soon as a
/// denominator shows the table to be beyond it.
fn exact_tails(dist: &Distribution, table: &[(Ex, Ex)], n: usize, budget: Option<u64>) -> Tails {
    let atoms = table.len();
    let over = |bits: u64| budget.is_some_and(|b| exact_table_work(atoms, n, bits) > b);
    let masses: Option<Vec<Q>> = table.iter().map(|(_, p)| p.as_rational()).collect();
    let (rationals, cumulative) = match masses {
        Some(masses) => (masses, false),
        None => {
            if dist.support().as_interval().is_none() {
                return Tails::NotRational;
            }
            let mut tails = Vec::with_capacity(atoms);
            for (v, _) in table {
                let Some(f) = dist.cdf(v).eval().as_rational() else {
                    return Tails::NotRational;
                };
                if over(f.denom().bits()) {
                    return Tails::OverBudget;
                }
                tails.push(f);
            }
            (tails, true)
        }
    };
    // The common denominator: each step one gcd of the size reached, which
    // the budget bounds before it is taken.
    let mut denominator = BigInt::one();
    for r in &rationals {
        let d = r.denom();
        if over(d.bits().max(denominator.bits())) {
            return Tails::OverBudget;
        }
        if d.is_one() || *d == denominator {
            continue;
        }
        let g = gcd_big(&denominator, d);
        denominator = &denominator / &g * d;
    }
    if over(denominator.bits()) {
        return Tails::OverBudget;
    }
    let mut numerators = Vec::with_capacity(atoms);
    let mut sum = BigInt::zero();
    for r in &rationals {
        let a = r.numer() * (&denominator / r.denom());
        if cumulative {
            numerators.push(a);
        } else {
            sum += a;
            numerators.push(sum.clone());
        }
    }
    Tails::Exact {
        numerators,
        denominator,
    }
}

/// `num / dⁿ` in lowest terms, for `d > 0` and `d_n = dⁿ`.  Every prime of
/// the denominator divides `d`, so a common factor of the numerator and
/// the denominator shows in `gcd(num mod d, d)`: gcds of the length of
/// `d`, not of `dⁿ` (and none of the binary gcds of `Q::new`).  A round
/// divides both by `h = gcd(num, d, den)`; when `h = 1` no prime of `den`
/// (they all divide `d`) divides `num`, and the fraction is in lowest
/// terms.  A prime `p` is done after at most `n` rounds (a round removes
/// all of it from `num`, or `v_p(d)` of the at most `n·v_p(d)` in `den`):
/// at most `n + 1` rounds.
fn reduced(mut num: BigInt, d: &BigInt, d_n: &BigInt) -> Q {
    if num.is_zero() {
        return Q::zero();
    }
    let mut den = d_n.clone();
    loop {
        let in_d = gcd_big(&(&num % d), d);
        if in_d.is_one() {
            break;
        }
        let h = gcd_big(&in_d, &den);
        if h.is_one() {
            break;
        }
        num /= &h;
        den /= &h;
    }
    Q::new_raw(num, den)
}

/// The masses `G(Fᵢ) − G(Fᵢ₋₁)` of the `k`-th of `n` draws, with
/// `G(u) = Σ_{j≥k} C(n, j) uʲ (1 − u)ⁿ⁻ʲ` and `Fᵢ = aᵢ/d`: in integers,
/// `dⁿ·G(a/d) = Σ_{j≥k} C(n, j) aʲ (d − a)ⁿ⁻ʲ`, and one reduction to lowest
/// terms per mass.  `None` only for an `n` beyond `u32` (no such table).
fn exact_masses(numerators: &[BigInt], d: &BigInt, n: usize, k: usize) -> Option<Vec<Q>> {
    let exp = |e: usize| u32::try_from(e).ok();
    let d_n = d.pow(exp(n)?);
    let choose: Vec<BigInt> = (k..=n).map(|j| binomial_q(n, j).to_integer()).collect();
    let mut previous = BigInt::zero();
    let mut out = Vec::with_capacity(numerators.len());
    for a in numerators {
        let b = d - a;
        // b⁰, …, b^{n−k}; the term of j = k + i takes b^{n−k−i}.
        let mut b_pows = Vec::with_capacity(choose.len());
        let mut b_pow = BigInt::one();
        for _ in &choose {
            b_pows.push(b_pow.clone());
            b_pow *= &b;
        }
        let mut a_pow = a.pow(exp(k)?);
        let mut current = BigInt::zero();
        for (c, b_pow) in choose.iter().zip(b_pows.iter().rev()) {
            current += c * &a_pow * b_pow;
            a_pow *= a;
        }
        out.push(reduced(&current - &previous, d, &d_n));
        previous = current;
    }
    Some(out)
}

/// The exact table of the `k`-th of `n` draws from a finite parent; `None`
/// where the parent is a lattice with rational tails too long for a table,
/// and the family's formula is to be the law instead
/// ([`EXACT_TABLE_BUDGET`]).  A parent whose tails are not rational numbers
/// (a symbol in its parameters, `Binomial(30, 1/π)`) keeps the symbolic
/// table of 0.60.0 ([`symbolic_table`]).
///
/// Up to 0.60.0 every table was built by expanding `G` in the symbolic
/// partial sums of the masses: with the masses of `Binomial(843,
/// 99999999999991/10¹⁴)`, which are products of powers, the multinomial
/// expansion did not end in 70 s.
fn enumerate(dist: &Distribution, table: &[(Ex, Ex)], n: usize, k: usize) -> Option<Distribution> {
    let ctx = dist.context();
    let lattice = dist.support().as_interval().is_some();
    match exact_tails(dist, table, n, lattice.then_some(EXACT_TABLE_BUDGET)) {
        Tails::Exact {
            numerators,
            denominator,
        } => {
            let Some(masses) = exact_masses(&numerators, &denominator, n, k) else {
                // (a table without the lattice formula is always a table)
                return (!lattice).then(|| symbolic_table(dist, table, n, k));
            };
            let out = table
                .iter()
                .zip(masses)
                .map(|((v, _), mass)| (v.clone(), ctx.from_ratio(mass)))
                .collect();
            Some(Distribution::finite(&ctx, out))
        }
        Tails::OverBudget => None,
        Tails::NotRational => Some(symbolic_table(dist, table, n, k)),
    }
}

/// The table of the `k`-th of `n` draws from a finite parent whose tails
/// are not rational numbers (symbols, irrational constants): the
/// polynomial `G` expanded in the partial sums of the masses, as up to
/// 0.60.0.  The expansion of a sum of `a` terms to the power `n` grows
/// with both: `Binomial(30, 1/π)`, 5 draws, takes 13 s.
fn symbolic_table(dist: &Distribution, table: &[(Ex, Ex)], n: usize, k: usize) -> Distribution {
    let ctx = dist.context();
    // G(u) = Σ_{j≥k} C(n, j) uʲ (1−u)ⁿ⁻ʲ
    let g = |u: &Ex| -> Ex {
        let mut acc = ctx.zero();
        for j in k..=n {
            let mut term = ctx.from_ratio(binomial_q(n, j));
            if j > 0 {
                term *= u.powi(usize_to_i64(j));
            }
            if n > j {
                term *= (ctx.one() - u).powi(usize_to_i64(n - j));
            }
            acc += term;
        }
        acc.eval()
    };
    let mut cumulative = ctx.zero();
    let mut previous = ctx.zero();
    let mut out = Vec::with_capacity(table.len());
    for (v, p) in table {
        cumulative = (&cumulative + p).eval();
        let current = g(&cumulative);
        let mass = (&current - &previous).simplify();
        previous = current;
        out.push((v.clone(), mass));
    }
    Distribution::finite(&ctx, out)
}
