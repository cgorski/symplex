//! Descriptive statistics on observed data, exactly.
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
use num_integer::Integer;
use num_traits::{One, Signed, ToPrimitive, Zero};

use crate::api::context::Context;
use crate::api::expr::Ex;
use crate::base::errors::SymplexError;
pub use crate::base::numeric::Q;

fn invalid(reason: impl Into<String>) -> SymplexError {
    SymplexError::invalid_argument("stats::data", reason)
}

fn qi(n: i64) -> Q {
    Q::from_integer(BigInt::from(n))
}

fn qu(n: usize) -> Q {
    Q::from_integer(BigInt::from(n))
}

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
    fn divisor(self, n: usize) -> Result<Q, SymplexError> {
        match self {
            Ddof::Population => {
                if n == 0 {
                    return Err(invalid("variance of an empty sample"));
                }
                Ok(qu(n))
            }
            Ddof::Sample => {
                if n < 2 {
                    return Err(invalid("sample variance needs at least two observations"));
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
        return Err(invalid("mean of an empty sample"));
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
    let div = ddof.divisor(data.len())?;
    Ok(sum_of_squares(data)? / div)
}

/// The standard deviation `√variance`, as an exact expression.
pub fn std(ctx: &Context, data: &[Q], ddof: Ddof) -> Result<Ex, SymplexError> {
    Ok(ctx.from_ratio(variance(data, ddof)?).sqrt())
}

/// The covariance of two equally long samples.  `statistics.covariance`
/// (which uses the `Sample` divisor).
pub fn covariance(x: &[Q], y: &[Q], ddof: Ddof) -> Result<Q, SymplexError> {
    if x.len() != y.len() {
        return Err(invalid(format!(
            "covariance of samples of different sizes ({} and {})",
            x.len(),
            y.len()
        )));
    }
    let div = ddof.divisor(x.len())?;
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
        return Err(invalid("correlation with a constant sample is undefined"));
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
        return Err(invalid("median of an empty sample"));
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
    if data.is_empty() {
        return Err(invalid("quantile of an empty sample"));
    }
    if p.is_negative() || *p > Q::one() {
        return Err(invalid(format!(
            "the quantile level must lie in [0, 1], got {p}"
        )));
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
        return Err(invalid("quantiles need n ≥ 2 groups"));
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

/// The smallest and largest observations.
pub fn min_max(data: &[Q]) -> Result<(Q, Q), SymplexError> {
    let first = data
        .first()
        .ok_or_else(|| invalid("min/max of an empty sample"))?;
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
    Ok((lo, hi))
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
    if x.len() != y.len() {
        return Err(invalid("Kendall's τ of samples of different sizes"));
    }
    let n = x.len();
    if n < 2 {
        return Err(invalid("Kendall's τ needs at least two pairs"));
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
        return Err(invalid("Kendall's τ with a constant sample is undefined"));
    }
    Ok((ctx.int(conc - disc) / ctx.int(denom).sqrt()).simplify())
}

/// The sample skewness `m₃ / m₂^{3/2}` with population moments (biased;
/// `scipy.stats.skew(bias=True)`), as an exact expression.
pub fn skewness(ctx: &Context, data: &[Q]) -> Result<Ex, SymplexError> {
    let (m2, m3) = (central_moment(data, 2)?, central_moment(data, 3)?);
    if m2.is_zero() {
        return Err(invalid("skewness of a constant sample is undefined"));
    }
    Ok((ctx.from_ratio(m3) / ctx.from_ratio(m2).pow(&ctx.rational(3, 2))).simplify())
}

/// The excess kurtosis `m₄ / m₂² − 3` with population moments (biased;
/// `scipy.stats.kurtosis(fisher=True, bias=True)`), exactly.
pub fn kurtosis(data: &[Q]) -> Result<Q, SymplexError> {
    let (m2, m4) = (central_moment(data, 2)?, central_moment(data, 4)?);
    if m2.is_zero() {
        return Err(invalid("kurtosis of a constant sample is undefined"));
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
        return Err(invalid("z-scores of a constant sample are undefined"));
    }
    Ok(data
        .iter()
        .map(|x| (ctx.from_ratio(x - &m) / &s).simplify())
        .collect())
}

/// The geometric mean `(Π xᵢ)^{1/n}` of positive data, exactly.
pub fn geometric_mean(ctx: &Context, data: &[Q]) -> Result<Ex, SymplexError> {
    if data.is_empty() {
        return Err(invalid("geometric mean of an empty sample"));
    }
    if data.iter().any(|x| !x.is_positive()) {
        return Err(invalid("geometric mean needs positive observations"));
    }
    let prod = data.iter().fold(Q::one(), |acc, x| acc * x);
    Ok(ctx
        .from_ratio(prod)
        .pow(&ctx.rational(1, data.len() as i64))
        .simplify())
}

/// The harmonic mean `n / Σ (1/xᵢ)` of positive data.  `statistics.harmonic_mean`.
pub fn harmonic_mean(data: &[Q]) -> Result<Q, SymplexError> {
    if data.is_empty() {
        return Err(invalid("harmonic mean of an empty sample"));
    }
    if data.iter().any(|x| !x.is_positive()) {
        return Err(invalid("harmonic mean needs positive observations"));
    }
    let s = data.iter().fold(Q::zero(), |acc, x| acc + x.recip());
    Ok(qu(data.len()) / s)
}

/// The mean after removing the `⌊n·p⌋` smallest and largest observations
/// (`scipy.stats.trim_mean(data, p)`).
pub fn trimmed_mean(data: &[Q], p: &Q) -> Result<Q, SymplexError> {
    if p.is_negative() || *p >= Q::new(BigInt::from(1), BigInt::from(2)) {
        return Err(invalid("the trimming proportion must lie in [0, 1/2)"));
    }
    let s = sorted(data);
    let cut = (p * qu(s.len()))
        .floor()
        .to_integer()
        .to_usize()
        .unwrap_or(0);
    if 2 * cut >= s.len() {
        return Err(invalid("trimming removes every observation"));
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
        return Err(invalid("the median absolute deviation is zero"));
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
                .ok_or_else(|| invalid(format!("{x} is not a finite number")))
        })
        .collect()
}

/// The observations as `f64`s.
pub fn to_f64(data: &[Q]) -> Vec<f64> {
    data.iter()
        .map(|x| x.numer().to_f64().unwrap_or(f64::NAN) / x.denom().to_f64().unwrap_or(f64::NAN))
        .collect()
}

/// `n choose k` as an exact rational (`0` for `k > n`).
pub fn binomial_q(n: usize, k: usize) -> Q {
    if k > n {
        return Q::zero();
    }
    let mut num = BigInt::one();
    let mut den = BigInt::one();
    for i in 0..k {
        num *= BigInt::from(n - i);
        den *= BigInt::from(i + 1);
    }
    let g = num.gcd(&den);
    Q::new(num / &g, den / g)
}
