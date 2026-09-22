//! Descriptive statistics on observed data, exactly — location, spread,
//! shape, ranks and the measures of association (Pearson, Spearman,
//! Kendall's τ-b and τ-c, Goodman–Kruskal's γ, Somers' D).
//!
//! **Rule:** a function lives here iff it *describes* a sample or the
//! association of two samples without inference; tests are in
//! [`super::hypothesis`], interval estimates in [`super::estimation`].
//!
//! Observations are exact rationals ([`Q`]); every quantity that is a
//! rational function of the data (mean, variance, covariance, median,
//! quantiles, ranks, Spearman's ρ, Kendall's τ, …) comes back as a `Q`,
//! and the few that involve a root (standard deviation, Pearson's r,
//! z-scores) as an [`Ex`] in a [`Context`], so nothing is rounded until
//! the caller asks for a float.  Python's `statistics` module on
//! `Fraction`s and `scipy.stats` are the reference implementations named
//! in the tests.
//!
//! # Conventions
//!
//! * [`Ddof`] chooses the divisor of a variance: `Population` (`n`) or
//!   `Sample` (`n − 1`, the unbiased estimator; `statistics.variance`,
//!   `numpy.var(ddof=1)`).
//! * [`QuantileMethod`] chooses the interpolation of a quantile:
//!   `Exclusive` (`statistics.quantiles` default, `numpy` `weibull`) and
//!   `Inclusive` (`statistics.quantiles(method='inclusive')`, `numpy`
//!   `linear`, R type 7).
//! * Ties in ranks take the average rank (`scipy.stats.rankdata`).

use num_bigint::BigInt;
use num_traits::{One, Signed, ToPrimitive, Zero};

use super::common::{invalid, qi, qu};
use crate::api::context::Context;
use crate::api::expr::Ex;
use crate::base::errors::SymplexError;
use crate::base::interval::Interval;
pub use crate::base::numeric::Q;

/// The divisor of a variance / covariance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Ddof {
    /// Divide by `n`: the variance of the data as a distribution
    /// (`statistics.pvariance`).
    Population,
    /// Divide by `n − 1`: the unbiased estimate of the source's variance
    /// (`statistics.variance`).  Needs `n ≥ 2`.
    Sample,
}

impl Ddof {
    fn divisor(self, op: &'static str, n: usize) -> Result<Q, SymplexError> {
        match self {
            Ddof::Population => {
                if n == 0 {
                    return Err(invalid(op, "variance of an empty sample"));
                }
                Ok(qu(n))
            }
            Ddof::Sample => {
                if n < 2 {
                    return Err(invalid(
                        op,
                        "sample variance needs at least two observations",
                    ));
                }
                Ok(qu(n - 1))
            }
        }
    }
}

/// How a quantile between two order statistics is interpolated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QuantileMethod {
    /// Position `p·(n + 1)` (R type 6; `statistics.quantiles` default;
    /// numpy `weibull`).  Extrapolates by clamping to the extremes.
    Exclusive,
    /// Position `1 + p·(n − 1)` (R type 7; `statistics.quantiles(method=
    /// 'inclusive')`; numpy default `linear`).
    Inclusive,
}

/// The sum of the observations.
pub fn sum(data: &[Q]) -> Q {
    data.iter().fold(Q::zero(), |acc, x| acc + x)
}

/// The arithmetic mean.  `statistics.mean`.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] on an empty sample.
///
/// ```
/// use symplex::stats::data::{mean, Q};
/// use symplex::linprog::qi;
/// let d: Vec<Q> = [2, 4, 4, 4, 5, 5, 7, 9].map(qi).to_vec();
/// assert_eq!(mean(&d).unwrap(), qi(5));
/// ```
pub fn mean(data: &[Q]) -> Result<Q, SymplexError> {
    if data.is_empty() {
        return Err(invalid("mean", "mean of an empty sample"));
    }
    Ok(sum(data) / qu(data.len()))
}

/// `Σ (xᵢ − x̄)²`, the total sum of squares about the mean.
pub fn sum_of_squares(data: &[Q]) -> Result<Q, SymplexError> {
    let m = mean(data)?;
    Ok(data
        .iter()
        .map(|x| {
            let d = x - &m;
            &d * &d
        })
        .fold(Q::zero(), |acc, x| acc + x))
}

/// The variance with the given divisor.  `statistics.pvariance` /
/// `statistics.variance`.
///
/// ```
/// use symplex::stats::data::{variance, Ddof, Q};
/// use symplex::linprog::{q, qi};
/// let d: Vec<Q> = [2, 4, 4, 4, 5, 5, 7, 9].map(qi).to_vec();
/// assert_eq!(variance(&d, Ddof::Population).unwrap(), qi(4));
/// assert_eq!(variance(&d, Ddof::Sample).unwrap(), q(32, 7));
/// ```
pub fn variance(data: &[Q], ddof: Ddof) -> Result<Q, SymplexError> {
    let div = ddof.divisor("variance", data.len())?;
    Ok(sum_of_squares(data)? / div)
}

/// The standard deviation `√variance`, as an exact expression.
pub fn std(ctx: &Context, data: &[Q], ddof: Ddof) -> Result<Ex, SymplexError> {
    Ok(ctx.from_ratio(variance(data, ddof)?).sqrt())
}

/// The covariance of two equally long samples.  `statistics.covariance`
/// (which uses the `Sample` divisor).
pub fn covariance(x: &[Q], y: &[Q], ddof: Ddof) -> Result<Q, SymplexError> {
    const OP: &str = "covariance";
    if x.len() != y.len() {
        return Err(invalid(
            OP,
            format!(
                "covariance of samples of different sizes ({} and {})",
                x.len(),
                y.len()
            ),
        ));
    }
    let div = ddof.divisor(OP, x.len())?;
    let (mx, my) = (mean(x)?, mean(y)?);
    let s = x
        .iter()
        .zip(y)
        .map(|(a, b)| (a - &mx) * (b - &my))
        .fold(Q::zero(), |acc, v| acc + v);
    Ok(s / div)
}

/// Pearson's correlation coefficient `cov(x, y) / (σₓ σᵧ)` as an exact
/// expression (the divisor cancels).  `statistics.correlation`.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] if either sample is constant.
pub fn pearson(ctx: &Context, x: &[Q], y: &[Q]) -> Result<Ex, SymplexError> {
    let sxy = covariance(x, y, Ddof::Population)?;
    let sxx = variance(x, Ddof::Population)?;
    let syy = variance(y, Ddof::Population)?;
    if sxx.is_zero() || syy.is_zero() {
        return Err(invalid(
            "pearson",
            "correlation with a constant sample is undefined",
        ));
    }
    Ok((ctx.from_ratio(sxy) / (ctx.from_ratio(sxx) * ctx.from_ratio(syy)).sqrt()).simplify())
}

/// The observations in ascending order.
pub fn sorted(data: &[Q]) -> Vec<Q> {
    let mut v = data.to_vec();
    v.sort();
    v
}

/// The median: the middle order statistic, or the mean of the two middle
/// ones.  `statistics.median`.
pub fn median(data: &[Q]) -> Result<Q, SymplexError> {
    if data.is_empty() {
        return Err(invalid("median", "median of an empty sample"));
    }
    let s = sorted(data);
    let n = s.len();
    Ok(if n % 2 == 1 {
        s[n / 2].clone()
    } else {
        (&s[n / 2 - 1] + &s[n / 2]) / qi(2)
    })
}

/// The `p`-quantile (`0 ≤ p ≤ 1`) by linear interpolation between order
/// statistics, exactly.  `statistics.quantiles` / `numpy.quantile`.
///
/// ```
/// use symplex::stats::data::{quantile, QuantileMethod, Q};
/// use symplex::linprog::{q, qi};
/// let d: Vec<Q> = [2, 4, 4, 4, 5, 5, 7, 9].map(qi).to_vec();
/// // statistics.quantiles(d, n=4): [4, 9/2, 13/2]; method='inclusive': [4, 9/2, 11/2]
/// assert_eq!(quantile(&d, &q(3, 4), QuantileMethod::Exclusive).unwrap(), q(13, 2));
/// assert_eq!(quantile(&d, &q(3, 4), QuantileMethod::Inclusive).unwrap(), q(11, 2));
/// ```
pub fn quantile(data: &[Q], p: &Q, method: QuantileMethod) -> Result<Q, SymplexError> {
    const OP: &str = "quantile";
    if data.is_empty() {
        return Err(invalid(OP, "quantile of an empty sample"));
    }
    if p.is_negative() || *p > Q::one() {
        return Err(invalid(
            OP,
            format!("the quantile level must lie in [0, 1], got {p}"),
        ));
    }
    let s = sorted(data);
    let n = s.len();
    // 1-based fractional position.
    let pos = match method {
        QuantileMethod::Exclusive => p * qu(n + 1),
        QuantileMethod::Inclusive => Q::one() + p * qu(n - 1),
    };
    let floor = pos.floor();
    let frac = &pos - &floor;
    let k = floor.to_integer().to_i64().unwrap_or(0);
    // Clamp to the extremes (the exclusive method extrapolates otherwise).
    if k < 1 {
        return Ok(s[0].clone());
    }
    let k = k as usize;
    if k >= n {
        return Ok(s[n - 1].clone());
    }
    let lo = &s[k - 1];
    let hi = &s[k];
    Ok(lo + (hi - lo) * frac)
}

/// The `n − 1` cut points dividing the data into `n` equal-probability
/// groups (`statistics.quantiles(data, n=n, method=…)`).
pub fn quantiles(data: &[Q], n: usize, method: QuantileMethod) -> Result<Vec<Q>, SymplexError> {
    if n < 2 {
        return Err(invalid("quantiles", "quantiles need n ≥ 2 groups"));
    }
    (1..n)
        .map(|i| quantile(data, &(qu(i) / qu(n)), method))
        .collect()
}

/// The interquartile range `Q₃ − Q₁`.
pub fn iqr(data: &[Q], method: QuantileMethod) -> Result<Q, SymplexError> {
    let q1 = quantile(data, &Q::new(BigInt::from(1), BigInt::from(4)), method)?;
    let q3 = quantile(data, &Q::new(BigInt::from(3), BigInt::from(4)), method)?;
    Ok(q3 - q1)
}

/// The sample range `[min, max]`: the smallest observation as `lower`,
/// the largest as `upper`.
///
/// ```
/// use symplex::stats::data::min_max;
/// use symplex::linprog::qi;
///
/// let range = min_max(&[3, 1, 4, 1, 5].map(qi))?;
/// assert_eq!((range.lower, range.upper), (qi(1), qi(5)));
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] on an empty sample.
pub fn min_max(data: &[Q]) -> Result<Interval<Q>, SymplexError> {
    let first = data
        .first()
        .ok_or_else(|| invalid("min_max", "min/max of an empty sample"))?;
    let mut lo = first.clone();
    let mut hi = first.clone();
    for x in data {
        if *x < lo {
            lo = x.clone();
        }
        if *x > hi {
            hi = x.clone();
        }
    }
    Ok(Interval::closed(lo, hi))
}

/// The most frequent values, ascending (all of them when tied).
/// `statistics.multimode`.
pub fn modes(data: &[Q]) -> Vec<Q> {
    let s = sorted(data);
    let mut best = 0usize;
    let mut out: Vec<Q> = Vec::new();
    let mut i = 0;
    while i < s.len() {
        let mut j = i;
        while j < s.len() && s[j] == s[i] {
            j += 1;
        }
        let count = j - i;
        if count > best {
            best = count;
            out.clear();
            out.push(s[i].clone());
        } else if count == best {
            out.push(s[i].clone());
        }
        i = j;
    }
    out
}

/// Frequency table: distinct values ascending with their counts.
pub fn frequencies(data: &[Q]) -> Vec<(Q, usize)> {
    let s = sorted(data);
    let mut out: Vec<(Q, usize)> = Vec::new();
    for x in s {
        match out.last_mut() {
            Some((v, c)) if *v == x => *c += 1,
            _ => out.push((x, 1)),
        }
    }
    out
}

/// Ranks `1..=n` with ties given their average rank (`scipy.stats.rankdata`,
/// `method='average'`), in the data's order.
///
/// ```
/// use symplex::stats::data::{ranks, Q};
/// use symplex::linprog::{q, qi};
/// let d: Vec<Q> = [3, 1, 4, 1, 5].map(qi).to_vec();
/// // scipy: rankdata([3, 1, 4, 1, 5]) = [3, 1.5, 4, 1.5, 5]
/// assert_eq!(ranks(&d), vec![qi(3), q(3, 2), qi(4), q(3, 2), qi(5)]);
/// ```
pub fn ranks(data: &[Q]) -> Vec<Q> {
    let mut idx: Vec<usize> = (0..data.len()).collect();
    idx.sort_by(|&a, &b| data[a].cmp(&data[b]));
    let mut out = vec![Q::zero(); data.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j < idx.len() && data[idx[j]] == data[idx[i]] {
            j += 1;
        }
        // Positions i+1 ..= j (1-based) share the average rank.
        let avg = (qu(i + 1) + qu(j)) / qi(2);
        for &k in &idx[i..j] {
            out[k] = avg.clone();
        }
        i = j;
    }
    out
}

/// The tie groups of a sample: the sizes `t` of every group of equal
/// values with `t ≥ 2` (for tie corrections).
pub fn tie_sizes(data: &[Q]) -> Vec<usize> {
    frequencies(data)
        .into_iter()
        .filter_map(|(_, c)| (c >= 2).then_some(c))
        .collect()
}

/// Spearman's rank correlation: Pearson's `r` of the ranks, as an exact
/// expression (rational when there are no ties in either sample, where
/// it equals `1 − 6Σd²/(n(n²−1))`).  `scipy.stats.spearmanr`.
pub fn spearman(ctx: &Context, x: &[Q], y: &[Q]) -> Result<Ex, SymplexError> {
    pearson(ctx, &ranks(x), &ranks(y))
}

/// Kendall's τ-b: `(concordant − discordant) / √((n₀ − n₁)(n₀ − n₂))` with
/// the tie corrections `n₁`, `n₂`, as an exact expression.
/// `scipy.stats.kendalltau` (default `variant='b'`).
pub fn kendall_tau(ctx: &Context, x: &[Q], y: &[Q]) -> Result<Ex, SymplexError> {
    const OP: &str = "kendall_tau";
    if x.len() != y.len() {
        return Err(invalid(OP, "Kendall's τ of samples of different sizes"));
    }
    let n = x.len();
    if n < 2 {
        return Err(invalid(OP, "Kendall's τ needs at least two pairs"));
    }
    let (mut conc, mut disc) = (0i64, 0i64);
    for i in 0..n {
        for j in i + 1..n {
            let sx = x[i].cmp(&x[j]);
            let sy = y[i].cmp(&y[j]);
            if sx == std::cmp::Ordering::Equal || sy == std::cmp::Ordering::Equal {
                continue;
            }
            if sx == sy {
                conc += 1;
            } else {
                disc += 1;
            }
        }
    }
    let pairs = |t: &[usize]| -> i64 { t.iter().map(|&k| (k * (k - 1) / 2) as i64).sum() };
    let n0 = (n * (n - 1) / 2) as i64;
    let n1 = pairs(&tie_sizes(x));
    let n2 = pairs(&tie_sizes(y));
    let denom = (n0 - n1) * (n0 - n2);
    if denom <= 0 {
        return Err(invalid(
            OP,
            "Kendall's τ with a constant sample is undefined",
        ));
    }
    Ok((ctx.int(conc - disc) / ctx.int(denom).sqrt()).simplify())
}

/// The sample skewness `m₃ / m₂^{3/2}` with population moments (biased;
/// `scipy.stats.skew(bias=True)`), as an exact expression.
pub fn skewness(ctx: &Context, data: &[Q]) -> Result<Ex, SymplexError> {
    let (m2, m3) = (central_moment(data, 2)?, central_moment(data, 3)?);
    if m2.is_zero() {
        return Err(invalid(
            "skewness",
            "skewness of a constant sample is undefined",
        ));
    }
    Ok((ctx.from_ratio(m3) / ctx.from_ratio(m2).pow(&ctx.rational(3, 2))).simplify())
}

/// The excess kurtosis `m₄ / m₂² − 3` with population moments (biased;
/// `scipy.stats.kurtosis(fisher=True, bias=True)`), exactly.
pub fn kurtosis(data: &[Q]) -> Result<Q, SymplexError> {
    let (m2, m4) = (central_moment(data, 2)?, central_moment(data, 4)?);
    if m2.is_zero() {
        return Err(invalid(
            "kurtosis",
            "kurtosis of a constant sample is undefined",
        ));
    }
    Ok(m4 / (&m2 * &m2) - qi(3))
}

/// The `k`-th central moment `(1/n) Σ (xᵢ − x̄)ᵏ`.
pub fn central_moment(data: &[Q], k: u32) -> Result<Q, SymplexError> {
    let m = mean(data)?;
    let s = data
        .iter()
        .map(|x| pow_q(&(x - &m), k))
        .fold(Q::zero(), |acc, v| acc + v);
    Ok(s / qu(data.len()))
}

fn pow_q(x: &Q, k: u32) -> Q {
    let mut acc = Q::one();
    for _ in 0..k {
        acc *= x;
    }
    acc
}

/// The median absolute deviation `median(|xᵢ − median|)` (unscaled;
/// `scipy.stats.median_abs_deviation`).
pub fn median_abs_deviation(data: &[Q]) -> Result<Q, SymplexError> {
    let m = median(data)?;
    let devs: Vec<Q> = data.iter().map(|x| (x - &m).abs()).collect();
    median(&devs)
}

/// The z-scores `(xᵢ − x̄) / σ` as exact expressions.
pub fn zscores(ctx: &Context, data: &[Q], ddof: Ddof) -> Result<Vec<Ex>, SymplexError> {
    let m = mean(data)?;
    let s = std(ctx, data, ddof)?;
    if variance(data, ddof)?.is_zero() {
        return Err(invalid(
            "zscores",
            "z-scores of a constant sample are undefined",
        ));
    }
    Ok(data
        .iter()
        .map(|x| (ctx.from_ratio(x - &m) / &s).simplify())
        .collect())
}

/// The geometric mean `(Π xᵢ)^{1/n}` of positive data, exactly.
pub fn geometric_mean(ctx: &Context, data: &[Q]) -> Result<Ex, SymplexError> {
    const OP: &str = "geometric_mean";
    if data.is_empty() {
        return Err(invalid(OP, "geometric mean of an empty sample"));
    }
    if data.iter().any(|x| !x.is_positive()) {
        return Err(invalid(OP, "geometric mean needs positive observations"));
    }
    let prod = data.iter().fold(Q::one(), |acc, x| acc * x);
    Ok(ctx
        .from_ratio(prod)
        .pow(&ctx.rational(1, data.len() as i64))
        .simplify())
}

/// The harmonic mean `n / Σ (1/xᵢ)` of positive data.  `statistics.harmonic_mean`.
pub fn harmonic_mean(data: &[Q]) -> Result<Q, SymplexError> {
    const OP: &str = "harmonic_mean";
    if data.is_empty() {
        return Err(invalid(OP, "harmonic mean of an empty sample"));
    }
    if data.iter().any(|x| !x.is_positive()) {
        return Err(invalid(OP, "harmonic mean needs positive observations"));
    }
    let s = data.iter().fold(Q::zero(), |acc, x| acc + x.recip());
    Ok(qu(data.len()) / s)
}

/// The mean after removing the `⌊n·p⌋` smallest and largest observations
/// (`scipy.stats.trim_mean(data, p)`).
pub fn trimmed_mean(data: &[Q], p: &Q) -> Result<Q, SymplexError> {
    const OP: &str = "trimmed_mean";
    if p.is_negative() || *p >= Q::new(BigInt::from(1), BigInt::from(2)) {
        return Err(invalid(OP, "the trimming proportion must lie in [0, 1/2)"));
    }
    let s = sorted(data);
    let cut = (p * qu(s.len()))
        .floor()
        .to_integer()
        .to_usize()
        .unwrap_or(0);
    if 2 * cut >= s.len() {
        return Err(invalid(OP, "trimming removes every observation"));
    }
    mean(&s[cut..s.len() - cut])
}

/// Observations flagged by Tukey's fences: outside `[Q₁ − k·IQR, Q₃ + k·IQR]`
/// (`k = 3/2` is the usual fence).  Returns the indices.
pub fn iqr_outliers(data: &[Q], k: &Q, method: QuantileMethod) -> Result<Vec<usize>, SymplexError> {
    let q1 = quantile(data, &Q::new(BigInt::from(1), BigInt::from(4)), method)?;
    let q3 = quantile(data, &Q::new(BigInt::from(3), BigInt::from(4)), method)?;
    let spread = &q3 - &q1;
    let lo = &q1 - k * &spread;
    let hi = &q3 + k * &spread;
    Ok(data
        .iter()
        .enumerate()
        .filter_map(|(i, x)| (*x < lo || *x > hi).then_some(i))
        .collect())
}

/// Observations whose modified z-score `0.6745·(xᵢ − median)/MAD` exceeds
/// `threshold` in absolute value (Iglewicz–Hoaglin; `3.5` is the usual
/// threshold).  The constant `0.6745` is `Φ⁻¹(3/4)` rounded; the exact
/// comparison here uses the rational `6745/10000`.  Returns the indices.
pub fn mad_outliers(data: &[Q], threshold: &Q) -> Result<Vec<usize>, SymplexError> {
    let m = median(data)?;
    let mad = median_abs_deviation(data)?;
    if mad.is_zero() {
        return Err(invalid(
            "mad_outliers",
            "the median absolute deviation is zero",
        ));
    }
    let c = Q::new(BigInt::from(6745), BigInt::from(10_000));
    Ok(data
        .iter()
        .enumerate()
        .filter_map(|(i, x)| {
            let z = (&c * (x - &m) / &mad).abs();
            (z > *threshold).then_some(i)
        })
        .collect())
}

/// Parse a slice of integers into observations.
pub fn from_i64(data: &[i64]) -> Vec<Q> {
    data.iter().map(|&x| qi(x)).collect()
}

/// Exact observations from floats (each `f64` is a dyadic rational, so
/// this is lossless; `0.1` becomes `3602879701896397/36028797018963968`).
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] on a NaN or infinite value.
pub fn from_f64(data: &[f64]) -> Result<Vec<Q>, SymplexError> {
    data.iter()
        .map(|&x| {
            crate::base::numeric::f64_to_ratio_exact(x)
                .ok_or_else(|| invalid("from_f64", format!("{x} is not a finite number")))
        })
        .collect()
}

/// The observations as `f64`s (correctly rounded even when the numerator or
/// denominator alone exceeds `f64::MAX`).
pub fn to_f64(data: &[Q]) -> Vec<f64> {
    data.iter()
        .map(|x| x.to_f64().unwrap_or(f64::NAN))
        .collect()
}

/// `n choose k` as an exact rational (`0` for `k > n`): the integer
/// [`combinatorics::binomial`](crate::base::combinatorics::binomial) as a
/// `Q`.
pub fn binomial_q(n: usize, k: usize) -> Q {
    Q::from_integer(crate::base::combinatorics::binomial(n as u64, k as u64))
}

// ── Ordinal association ───────────────────────────────────────────────

/// The classification of the `n(n−1)/2` pairs of observations of two
/// variables: concordant, discordant, tied on `x` only, tied on `y` only,
/// tied on both.  The five counts sum to [`pairs`](Self::pairs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ConcordanceCounts {
    /// Pairs ordered the same way on both variables (`C`, SAS `P`).
    pub concordant: usize,
    /// Pairs ordered opposite ways (`D`, SAS `Q`).
    pub discordant: usize,
    /// Pairs tied on `x` but not on `y` (`T_x`).
    pub ties_x: usize,
    /// Pairs tied on `y` but not on `x` (`T_y`).
    pub ties_y: usize,
    /// Pairs tied on both (`T_xy`).
    pub ties_both: usize,
}

impl ConcordanceCounts {
    /// The total number of pairs `n(n−1)/2`.
    #[must_use]
    pub fn pairs(&self) -> usize {
        self.concordant + self.discordant + self.ties_x + self.ties_y + self.ties_both
    }
}

/// Count the concordant, discordant and tied pairs of two variables.
///
/// ```
/// use symplex::stats::data::{ConcordanceCounts, concordance_counts, from_i64};
///
/// let x = from_i64(&[1, 2, 2, 3]);
/// let y = from_i64(&[1, 1, 2, 3]);
/// let c = concordance_counts(&x, &y)?;
/// assert_eq!(c, ConcordanceCounts { concordant: 4, discordant: 0, ties_x: 1, ties_y: 1, ties_both: 0 });
/// assert_eq!(c.pairs(), 6);
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for unequal lengths or fewer than two
/// observations.
pub fn concordance_counts(x: &[Q], y: &[Q]) -> Result<ConcordanceCounts, SymplexError> {
    const OP: &str = "concordance_counts";
    if x.len() != y.len() {
        return Err(invalid(
            OP,
            format!(
                "the two variables must have the same length ({} and {})",
                x.len(),
                y.len()
            ),
        ));
    }
    let n = x.len();
    if n < 2 {
        return Err(invalid(OP, "needs at least two observations"));
    }
    let mut c = ConcordanceCounts {
        concordant: 0,
        discordant: 0,
        ties_x: 0,
        ties_y: 0,
        ties_both: 0,
    };
    for i in 0..n {
        for j in i + 1..n {
            let sx = x[i].cmp(&x[j]);
            let sy = y[i].cmp(&y[j]);
            match (sx, sy) {
                (std::cmp::Ordering::Equal, std::cmp::Ordering::Equal) => c.ties_both += 1,
                (std::cmp::Ordering::Equal, _) => c.ties_x += 1,
                (_, std::cmp::Ordering::Equal) => c.ties_y += 1,
                _ if sx == sy => c.concordant += 1,
                _ => c.discordant += 1,
            }
        }
    }
    Ok(c)
}

/// Goodman and Kruskal's γ (1954): `(C − D) / (C + D)`, the ordinal
/// association ignoring every tied pair.  Exact.
///
/// ```
/// use symplex::linprog::q;
/// use symplex::stats::data::{from_i64, goodman_kruskal_gamma};
///
/// let x = from_i64(&[1, 2, 2, 3, 3, 3, 4, 4, 5, 1, 2, 4]);
/// let y = from_i64(&[1, 1, 2, 2, 3, 2, 4, 3, 5, 2, 3, 4]);
/// // C = 44, D = 3 → γ = 41/47
/// assert_eq!(goodman_kruskal_gamma(&x, &y)?, q(41, 47));
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for unequal lengths, fewer than two
/// observations, or no untied pair (`C + D = 0`).
pub fn goodman_kruskal_gamma(x: &[Q], y: &[Q]) -> Result<Q, SymplexError> {
    let c = concordance_counts(x, y)?;
    let untied = c.concordant + c.discordant;
    if untied == 0 {
        return Err(invalid(
            "goodman_kruskal_gamma",
            "every pair is tied on one of the variables, γ is undefined",
        ));
    }
    Ok((qu(c.concordant) - qu(c.discordant)) / qu(untied))
}

/// Which variable Somers' D treats as dependent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dependent {
    /// `D_{Y|X}`: `y` depends on `x`; pairs tied on `x` are excluded
    /// (`scipy.stats.somersd(x, y)`).
    Y,
    /// `D_{X|Y}`: `x` depends on `y`; pairs tied on `y` are excluded
    /// (`scipy.stats.somersd(y, x)`).
    X,
    /// The symmetric version: `(C − D)` over the mean of the two
    /// denominators.
    Symmetric,
}

/// Somers' D (1962): the ordinal association of a dependent on an
/// independent variable,
///
/// `D_{Y|X} = (C − D) / (C + D + T_y)`, `D_{X|Y} = (C − D) / (C + D + T_x)`,
/// `D_sym = 2 (C − D) / (2 (C + D) + T_x + T_y)`,
///
/// where `T_y` counts the pairs tied on `y` only (so `D_{Y|X}` drops the
/// pairs tied on the independent `x`).  Exact.  `scipy.stats.somersd(x,
/// y).statistic` is `D_{Y|X}` (`x` the row / independent variable).
///
/// ```
/// use symplex::linprog::q;
/// use symplex::stats::data::{Dependent, from_i64, somers_d};
///
/// let x = from_i64(&[1, 2, 2, 3, 3, 3, 4, 4, 5, 1, 2, 4]);
/// let y = from_i64(&[1, 1, 2, 2, 3, 2, 4, 3, 5, 2, 3, 4]);
/// // scipy: somersd(x, y).statistic = 0.732142857142857 (= 41/56); somersd(y, x) = 0.745454545454545 (= 41/55)
/// assert_eq!(somers_d(&x, &y, Dependent::Y)?, q(41, 56));
/// assert_eq!(somers_d(&x, &y, Dependent::X)?, q(41, 55));
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for unequal lengths, fewer than two
/// observations, or a zero denominator (the independent variable
/// constant).
pub fn somers_d(x: &[Q], y: &[Q], dependent: Dependent) -> Result<Q, SymplexError> {
    let c = concordance_counts(x, y)?;
    let diff = qu(c.concordant) - qu(c.discordant);
    let untied = c.concordant + c.discordant;
    let denom = match dependent {
        Dependent::Y => qu(untied + c.ties_y),
        Dependent::X => qu(untied + c.ties_x),
        Dependent::Symmetric => qu(2 * untied + c.ties_x + c.ties_y) / qi(2),
    };
    if denom.is_zero() {
        return Err(invalid(
            "somers_d",
            "the independent variable is constant, Somers' D is undefined",
        ));
    }
    Ok(diff / denom)
}

/// Stuart's τ-c (Kendall's τ-c, 1953): `2m (C − D) / (n² (m − 1))` with
/// `m = min(#distinct x, #distinct y)`, the tie-adjusted τ for
/// rectangular tables.  Exact.  `scipy.stats.kendalltau(x, y, variant='c')`.
///
/// ```
/// use symplex::linprog::q;
/// use symplex::stats::data::{from_i64, kendall_tau_c};
///
/// let x = from_i64(&[1, 2, 2, 3, 3, 3, 4, 4, 5, 1, 2, 4]);
/// let y = from_i64(&[1, 1, 2, 2, 3, 2, 4, 3, 5, 2, 3, 4]);
/// // scipy: kendalltau(x, y, variant='c').statistic = 0.711805555555556 (= 205/288)
/// assert_eq!(kendall_tau_c(&x, &y)?, q(205, 288));
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for unequal lengths, fewer than two
/// observations, or a constant variable (`m = 1`).
pub fn kendall_tau_c(x: &[Q], y: &[Q]) -> Result<Q, SymplexError> {
    let c = concordance_counts(x, y)?;
    let m = frequencies(x).len().min(frequencies(y).len());
    if m < 2 {
        return Err(invalid(
            "kendall_tau_c",
            "a constant variable has no rank correlation",
        ));
    }
    let n = x.len();
    let diff = qu(c.concordant) - qu(c.discordant);
    Ok(qu(2 * m) * diff / (qu(n * n) * qu(m - 1)))
}
