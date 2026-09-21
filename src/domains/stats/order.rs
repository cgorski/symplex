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
use crate::base::errors::SymplexError;

use super::data::binomial_q;
use super::family::{Distribution, Family, Sampler, same_family};
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

    /// The parent's `F(x)` as a single expression: the closed form on the
    /// support when the family has one, else the integral / sum of the
    /// density from the support's lower end; `0` / `1` when `x` is
    /// decidably below / above the support.
    fn parent_cdf(&self, x: &Ex) -> Ex {
        let ctx = self.ctx();
        let support = self.inner.support();
        if let Some((lo, hi, _, _)) = support.as_interval() {
            if !is_neg_inf(lo) && (x - lo).is_negative() == Some(true) {
                return ctx.zero();
            }
            if !is_pos_inf(hi) && (x - hi).is_nonnegative() == Some(true) {
                return ctx.one();
            }
        }
        if let Some(c) = self.inner.family().cdf(x) {
            return c;
        }
        if support.as_points().is_some() {
            return self.inner.cdf(x);
        }
        let t = self.inner.fresh_var("t", &[x]);
        let dens = self.inner.density(&t);
        let lo = support
            .as_interval()
            .map_or_else(|| ctx.neg_infinity(), |(lo, _, _, _)| lo.clone());
        match support.kind() {
            Kind::Continuous => dens.integrate_definite(&t, &lo, x),
            Kind::Discrete => dens.summation(&t, &lo, &x.floor()),
        }
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
    fn name(&self) -> &str {
        "OrderStatistic"
    }

    fn context(&self) -> Context {
        self.ctx()
    }

    fn parameters(&self) -> Vec<(&'static str, Ex)> {
        let (n, k) = self.nk();
        let mut p = vec![("n", n), ("k", k)];
        p.extend(self.inner.parameters());
        p
    }

    fn eq_family(&self, other: &dyn Family) -> bool {
        same_family(self, other)
    }

    fn support(&self) -> Support {
        self.inner.support()
    }

    // Continuous: k·C(n,k)·F^{k−1}·(1−F)^{n−k}·f.
    // Lattice: I_{F(x)}(k, n−k+1) − I_{F(x−1)}(k, n−k+1).
    fn density(&self, x: &Ex) -> Ex {
        let ctx = self.ctx();
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

    fn cdf(&self, x: &Ex) -> Option<Ex> {
        Some(self.beta_cdf(&self.parent_cdf(x)))
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
    if dist.kind() == Kind::Discrete
        && let Some(table) = finite_table(dist)?
    {
        return enumerate(dist, &table, n, k);
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
    } else if let Some((lo, hi, _, _)) = support.as_interval() {
        let (Some(lo), Some(hi)) = (lo.eval().as_i64(), hi.eval().as_i64()) else {
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
    keyed.sort_by(|a, b| a.0.total_cmp(&b.0));
    let table = keyed
        .into_iter()
        .map(|(_, v)| {
            let p = dist.density(&v).eval();
            (v, p)
        })
        .collect();
    Ok(Some(table))
}

/// The exact table of the `k`-th of `n` draws from a finite parent.
fn enumerate(
    dist: &Distribution,
    table: &[(Ex, Ex)],
    n: usize,
    k: usize,
) -> Result<Distribution, SymplexError> {
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
    Ok(Distribution::finite(&ctx, out))
}
