//! Hypothesis tests, effect sizes, multiple-comparison corrections,
//! resampling, power / sample-size utilities and contingency-table tools.
//!
//! **Rule:** a function lives here iff it *tests* something (a statistic
//! with a p-value), measures the size of an effect, adjusts for multiple
//! testing, resamples, computes power or a sample size, or diagnoses a
//! contingency table.  Confidence intervals for a parameter are in
//! [`super::estimation`]; descriptive measures of association in
//! [`super::data`]; analysis of variance in [`super::anova`].
//!
//! The tests take exact observations ([`Q`], `Ratio<BigInt>`, see
//! [`super::data`]) and follow one principle: **exact where possible,
//! honest where not**.
//!
//! * A test **statistic** is exact: a rational (`U`, `H`, `χ²`, `F`, an odds
//!   ratio) or a rational times a square root (`t`, `z`, Cohen's `d`), as an
//!   [`Ex`].
//! * A **p-value** is an exact *expression* of the statistic — the Student-t
//!   tail through `betainc_regularized`, a χ² tail through `uppergamma`,
//!   a normal tail through `erfc`, or an exact rational for the discrete
//!   exact tests (binomial, Fisher, McNemar, sign, exact Mann–Whitney /
//!   Wilcoxon / Kendall) — that the caller evaluates with
//!   [`Ex::eval_f64`] (or [`TestResult::p_value_f64`]).  Nothing is rounded
//!   before the caller asks.
//! * The few quantities that are inherently numerical (confidence limits
//!   through a quantile, the Kolmogorov–Smirnov `D` of a transcendental
//!   CDF, bootstrap / permutation p-values, power) are `f64` and say so.
//!
//! **Tiny p-values.**  Because a p-value is an exact expression, nothing is
//! lost until it is converted: `p_value_f64` (that is, [`Ex::eval_f64`])
//! underflows to `0.0` below about `1e-308`, but the expression still holds
//! the value (`χ² = 12800` on one degree of freedom is
//! `uppergamma(1/2, 6400)/Γ(1/2) ≈ 2.31e-2782`).  Every result type with an
//! exact p-value implements [`PValue`] and mirrors its methods inherently:
//! `p_value_log10` / `p_value_ln` evaluate the logarithm *as an expression*,
//! in arbitrary precision, so they are finite for any positive `p`
//! (`−2781.64` for the example), and `p_value_decimal(digits)` gives the
//! value itself as a decimal string with exponent
//! (`"2.3100265595063985852e-2782"`).
//!
//! Every function names the `scipy.stats` / `statsmodels` routine whose
//! conventions it follows (alternative hypotheses, tie corrections,
//! continuity corrections, two-sided definitions of the discrete tests);
//! the tests in `tests/v13/v13_hypothesis.rs` pin the agreement to ≥ 1e-9.
//!
//! ```
//! use symplex::prelude::*;
//! use symplex::stats::data::from_i64;
//! use symplex::stats::hypothesis::{t_test_one_sample, Alternative};
//!
//! let ctx = Context::new();
//! let x = from_i64(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
//! let r = t_test_one_sample(&ctx, &x, &symplex::linprog::qi(5), Alternative::TwoSided)?;
//! // scipy: ttest_1samp(range(1, 11), 5) → statistic 0.5222329678670935, pvalue 0.6141172548083939
//! assert!((r.statistic_f64()? - 0.522_232_967_867_093_5).abs() < 1e-12);
//! assert!((r.p_value_f64()? - 0.614_117_254_808_393_9).abs() < 1e-12);
//! assert_eq!(r.df, Some(ctx.int(9)));
//! # Ok::<(), SymplexError>(())
//! ```

use std::cmp::Ordering;
use std::f64::consts::LN_10;

use num_bigint::BigInt;
use num_traits::{One, Pow, Signed, ToPrimitive, Zero};

use super::common::{
    check_alpha, check_confidence, check_finite, check_sample, check_unit_open, chi_squared_sf,
    chi_squared_sf_q, ex, invalid, norm_cdf, norm_isf, norm_sf, q_to_f64, qi, qu,
    student_t_quantile_f64, usize_to_i64,
};
use super::data::{self, Ddof, Q};
use super::family::Distribution;
use super::sample::Rng;
use crate::api::context::Context;
use crate::api::expr::Ex;
use crate::base::errors::SymplexError;
use crate::base::interval::Interval;
use crate::calculus::definite::{QuadOpts, quadrature};
use crate::domains::optimize::partition_point_by;
use crate::output::codegen::numeric_rt::lgamma;

// ═══════════════════════════════════════════════════════════════════════════
// Small helpers
// ═══════════════════════════════════════════════════════════════════════════

fn factorial_big(n: usize) -> BigInt {
    (1..=n).fold(BigInt::one(), |acc, k| acc * BigInt::from(k))
}

fn sum_big(v: &[BigInt]) -> BigInt {
    v.iter().fold(BigInt::zero(), |acc, x| acc + x)
}

fn square(x: &Q) -> Q {
    x * x
}

fn check_same_len(op: &'static str, x: &[Q], y: &[Q]) -> Result<(), SymplexError> {
    if x.len() != y.len() {
        return Err(invalid(
            op,
            format!(
                "paired samples must have the same size ({} and {})",
                x.len(),
                y.len()
            ),
        ));
    }
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// Result types
// ═══════════════════════════════════════════════════════════════════════════

/// The alternative hypothesis of a test (`scipy`'s `alternative=`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Alternative {
    /// `'two-sided'`: the parameter differs from its null value.
    TwoSided,
    /// `'less'`: the parameter is smaller than its null value.
    Less,
    /// `'greater'`: the parameter is larger than its null value.
    Greater,
}

/// A result carrying an exact p-value expression.
///
/// The p-value of every test in `stats` is an exact expression ([`Ex`]);
/// this trait is the common way to read it.  [`p_value_f64`] is the usual
/// conversion and underflows to `0.0` when `p < f64::MIN_POSITIVE`
/// (about `1e-308`); [`p_value_log10`] and [`p_value_ln`] evaluate the
/// logarithm *as an expression*, in arbitrary precision, and so stay finite
/// for any positive `p`; [`p_value_decimal`] prints `p` itself to any
/// number of digits.  Every implementing type also offers the same methods
/// inherently, so the trait need not be imported to use them.
///
/// [`p_value_f64`]: PValue::p_value_f64
/// [`p_value_log10`]: PValue::p_value_log10
/// [`p_value_ln`]: PValue::p_value_ln
/// [`p_value_decimal`]: PValue::p_value_decimal
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::PValue;
/// use symplex::stats::data::from_i64;
/// use symplex::stats::hypothesis::chi_square_independence;
///
/// let ctx = Context::new();
/// // A 2 × 2 table with χ² = 12800 on one degree of freedom.
/// let table = [from_i64(&[9000, 1000]), from_i64(&[1000, 9000])];
/// let r = chi_square_independence(&ctx, &table, false)?;
/// assert_eq!(r.p_value_f64()?, 0.0); // underflows: the true p is 2.31e-2782
/// // mpmath: log10(gammainc(0.5, 6400, regularized=True)) = -2781.6363830267828509
/// assert!((r.p_value_log10()? + 2781.636_383_026_783).abs() < 1e-9);
/// assert!((r.p_value_ln()? + 6404.954_469_687_346).abs() < 1e-9);
/// assert_eq!(r.p_value_decimal(20)?, "2.3100265595063985852e-2782");
/// # Ok::<(), SymplexError>(())
/// ```
///
/// An exact `0` (a discrete or degenerate test) gives `log10 p = ln p = −∞`;
/// an exact `1` gives `0`:
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::PValue;
/// use symplex::stats::data::from_i64;
/// use symplex::stats::hypothesis::{Alternative, pearson_test};
///
/// let ctx = Context::new();
/// let x = from_i64(&[1, 2, 3, 4]);
/// let y = from_i64(&[2, 4, 6, 8]); // r = 1 exactly
/// let two_sided = pearson_test(&ctx, &x, &y, Alternative::TwoSided)?;
/// assert_eq!(two_sided.p_value_ex(), &ctx.zero());
/// assert_eq!(two_sided.p_value_log10()?, f64::NEG_INFINITY);
/// let less = pearson_test(&ctx, &x, &y, Alternative::Less)?;
/// assert_eq!(less.p_value_ex(), &ctx.one());
/// assert_eq!(less.p_value_log10()?, 0.0);
/// # Ok::<(), SymplexError>(())
/// ```
pub trait PValue {
    /// The p-value as an exact expression.
    fn p_value_ex(&self) -> &Ex;

    /// The p-value as an `f64`.  Underflows to `0.0` below about `1e-308`;
    /// use [`p_value_log10`](PValue::p_value_log10) or
    /// [`p_value_decimal`](PValue::p_value_decimal) for tiny values.  An
    /// exact rational is converted directly (correctly rounded), anything
    /// else by [`Ex::eval_f64`].
    ///
    /// # Errors
    ///
    /// Propagates the evaluation error of the expression (not expected for
    /// the expressions `stats` builds).
    fn p_value_f64(&self) -> Result<f64, SymplexError> {
        p_value_f64_of(self.p_value_ex())
    }

    /// `log10 p`, finite even when `p` underflows `f64` (`−2781.64` for
    /// the χ² example above): `ln p` is evaluated as an expression and
    /// divided by `ln 10`.  An exact `0` gives `−∞`; an exact `1` gives `0`.
    ///
    /// # Errors
    ///
    /// As [`p_value_f64`](PValue::p_value_f64).
    fn p_value_log10(&self) -> Result<f64, SymplexError> {
        p_value_log10_of(self.p_value_ex())
    }

    /// `ln p`, evaluated as an expression so it is finite for any positive
    /// `p`.  An exact `0` gives `−∞`; an exact `1` gives `0`.
    ///
    /// # Errors
    ///
    /// As [`p_value_f64`](PValue::p_value_f64).
    fn p_value_ln(&self) -> Result<f64, SymplexError> {
        p_value_ln_of(self.p_value_ex())
    }

    /// `p` to `digits` significant digits as a decimal string, with an
    /// exponent when one is needed (`"2.3100265595063985852e-2782"`);
    /// [`Ex::eval_decimal`], or for an exact rational `p` the same format
    /// by exact integer arithmetic (rounded to nearest, ties to even) —
    /// the exact tests' rationals can have `10⁵` bits.
    ///
    /// # Errors
    ///
    /// As [`Ex::eval_decimal`].
    fn p_value_decimal(&self, digits: u32) -> Result<String, SymplexError> {
        let p = self.p_value_ex();
        match p.as_rational().and_then(|q| rational_decimal(&q, digits)) {
            Some(s) => Ok(s),
            None => p.eval_decimal(digits),
        }
    }
}

/// A rational `q ∈ (0, 1]` to `digits` significant digits in the format of
/// [`Ex::eval_decimal`] (`"0.34375"`, `"0.0012"`, `"3.7266589428590489823e-16"`),
/// by exact integer arithmetic: `q·10^k ∈ [1, 10)` fixes the exponent, the
/// digits are `round(q·10^{k+digits−1})` with ties to even.  The exact
/// discrete tests produce rationals of `10⁵` bits, whose `eval_decimal`
/// takes seconds.  `None` outside `(0, 1]` or for `digits = 0`.
fn rational_decimal(q: &Q, digits: u32) -> Option<String> {
    if !q.is_positive() || *q > Q::one() || digits == 0 {
        return None;
    }
    let (num, den) = (q.numer(), q.denom());
    let ten = BigInt::from(10);
    // A lower bound on k = ⌈−log₁₀ q⌉ from the bit lengths (q > 2^(bn − bd − 1)),
    // then up to the first k with q·10^k ≥ 1.
    let gap = den.bits().saturating_sub(num.bits() + 1);
    let mut k = ((gap as f64) * std::f64::consts::LOG10_2).floor() as u64;
    k = k.saturating_sub(1);
    let mut scaled = num * Pow::pow(&ten, k);
    while scaled < *den {
        scaled *= &ten;
        k += 1;
    }
    let top = Pow::pow(&ten, digits - 1);
    let (mut kept, rest) = num_integer::Integer::div_rem(&(scaled * &top), den);
    match (rest * 2u32).cmp(den) {
        Ordering::Greater => kept += 1u32,
        Ordering::Equal if num_integer::Integer::is_odd(&kept) => kept += 1u32,
        _ => {}
    }
    let mut exponent = -i64::try_from(k).ok()?;
    if kept == &top * &ten {
        kept = top;
        exponent += 1;
    }
    let text = kept.to_string();
    let (lead, tail) = text.split_at(1);
    let tail = tail.trim_end_matches('0');
    let dot_tail = if tail.is_empty() {
        String::new()
    } else {
        format!(".{tail}")
    };
    Some(match exponent {
        0 => format!("{lead}{dot_tail}"),
        -4..=-1 => {
            let zeros = "0".repeat(usize::try_from(-exponent - 1).ok()?);
            format!("0.{zeros}{lead}{tail}")
        }
        _ => format!("{lead}{dot_tail}e{exponent}"),
    })
}

/// `ln p` of a p-value expression: `−∞` for an exact `0`, `0` for an exact
/// `1`, otherwise `ln(p)` evaluated as an expression — in arbitrary
/// precision, so the result is finite even when `p` is below
/// `f64::MIN_POSITIVE`.  (`ln` of a structural zero would evaluate to `-oo`
/// and fail to parse, hence the exact check first.)
///
/// The logarithm is expanded first (`ln(c·e^{−x}) = ln c − x`, the even-df
/// χ² tails), and when the evaluator still cannot certify `ln(p)` — its
/// error bound for `exp(−x)` is absolute, so `ln(exp(−2601/10))` was
/// `PrecisionExhausted` although `p = 1.1e-113` evaluates fine — `ln p` is
/// read off the certified decimal expansion `m·10^e` of `p` as `ln m + e
/// ln 10`.
pub(crate) fn p_value_ln_of(p: &Ex) -> Result<f64, SymplexError> {
    let reduced = p.eval();
    match reduced.as_rational() {
        Some(q) if q.is_zero() => return Ok(f64::NEG_INFINITY),
        Some(q) if q.is_one() => return Ok(0.0),
        Some(q) if q.is_positive() => return Ok(ln_of_rational(&q)),
        _ => {}
    }
    match reduced.ln().expand_log().eval_f64() {
        Err(err @ SymplexError::PrecisionExhausted { .. }) => ln_from_decimal(&reduced).ok_or(err),
        other => other,
    }
}

/// `p` as an `f64`: an exact rational converted directly ([`q_to_f64`],
/// correctly rounded), anything else by [`Ex::eval_f64`].  The exact
/// discrete tests produce rationals of `10⁵` bits (`binomial_test` at `n =
/// 20 000`, `p₀ = 3/20`), whose `eval_f64` takes seconds.
fn p_value_f64_of(p: &Ex) -> Result<f64, SymplexError> {
    match p.as_rational() {
        Some(q) => Ok(q_to_f64(&q)),
        None => p.eval_f64(),
    }
}

/// `ln q` of a positive rational without evaluating an expression:
/// `ln_1p(q − 1)` above `½` (accurate next to `1`), else `ln(q·2ˢ) − s ln 2`
/// with `q·2ˢ ∈ (½, 2)`, so neither underflows.
fn ln_of_rational(q: &Q) -> f64 {
    let (num, den) = (q.numer(), q.denom());
    if *q > Q::new_raw(BigInt::one(), BigInt::from(2)) {
        return q_to_f64(&Q::new_raw(num - den, den.clone())).ln_1p();
    }
    let shift = den.bits().saturating_sub(num.bits());
    let scaled = Q::new_raw(num << shift, den.clone());
    q_to_f64(&scaled).ln() - shift as f64 * std::f64::consts::LN_2
}

/// `ln p = ln m + e·ln 10` from the decimal expansion `m·10^e` of a
/// positive `p` ([`Ex::eval_decimal`], which certifies its digits);
/// `None` if `p` does not evaluate or is not positive.
fn ln_from_decimal(p: &Ex) -> Option<f64> {
    let s = p.eval_decimal(20).ok()?;
    let (mantissa, exponent) = match s.split_once('e') {
        Some((m, e)) => (m, e.parse::<i64>().ok()?),
        None => (s.as_str(), 0),
    };
    let m: f64 = mantissa.parse().ok()?;
    (m > 0.0 && m.is_finite()).then(|| m.ln() + exponent as f64 * LN_10)
}

/// `log10 p = ln p / ln 10`, with `ln p` from [`p_value_ln_of`].
pub(crate) fn p_value_log10_of(p: &Ex) -> Result<f64, SymplexError> {
    p_value_ln_of(p).map(|ln_p| ln_p / LN_10)
}

/// Inherent `p_value_log10` / `p_value_ln` / `p_value_decimal` on a type
/// implementing [`PValue`], so the methods are discoverable without
/// importing the trait.
macro_rules! p_value_accessors {
    ($ty:ty) => {
        impl $ty {
            /// `log10` of the p-value, finite even when
            /// [`p_value_f64`](Self::p_value_f64) underflows to `0.0`: the
            /// logarithm is evaluated as an expression.  An exact `0` gives
            /// `−∞`, an exact `1` gives `0`.  See
            /// [`PValue`](crate::stats::PValue).
            ///
            /// # Errors
            ///
            /// As [`p_value_f64`](Self::p_value_f64).
            pub fn p_value_log10(&self) -> Result<f64, $crate::base::errors::SymplexError> {
                <Self as $crate::stats::PValue>::p_value_log10(self)
            }

            /// `ln` of the p-value, evaluated as an expression so it is
            /// finite for any positive `p`.  An exact `0` gives `−∞`, an
            /// exact `1` gives `0`.  See [`PValue`](crate::stats::PValue).
            ///
            /// # Errors
            ///
            /// As [`p_value_f64`](Self::p_value_f64).
            pub fn p_value_ln(&self) -> Result<f64, $crate::base::errors::SymplexError> {
                <Self as $crate::stats::PValue>::p_value_ln(self)
            }

            /// The p-value to `digits` significant digits as a decimal
            /// string with exponent (`"2.3100265595063985852e-2782"`).
            /// See [`PValue`](crate::stats::PValue).
            ///
            /// # Errors
            ///
            /// As [`Ex::eval_decimal`](crate::api::expr::Ex::eval_decimal).
            pub fn p_value_decimal(
                &self,
                digits: u32,
            ) -> Result<String, $crate::base::errors::SymplexError> {
                <Self as $crate::stats::PValue>::p_value_decimal(self, digits)
            }
        }
    };
}
pub(crate) use p_value_accessors;

/// The outcome of a test: an exact statistic, an exact p-value expression
/// and (when the reference distribution has one) the degrees of freedom.
#[derive(Clone, Debug, PartialEq)]
pub struct TestResult {
    /// The test statistic, exact (a rational, or a rational times a root).
    pub statistic: Ex,
    /// The p-value as an exact expression; evaluate with
    /// [`p_value_f64`](Self::p_value_f64), or with
    /// [`p_value_log10`](Self::p_value_log10) /
    /// [`p_value_decimal`](Self::p_value_decimal) when it may be below
    /// `1e-308`.
    pub p_value: Ex,
    /// Degrees of freedom of the reference distribution, if any (exact; the
    /// Welch–Satterthwaite `ν` is a rational).
    pub df: Option<Ex>,
    /// The alternative the p-value refers to.
    pub alternative: Alternative,
}

impl TestResult {
    /// The p-value as an `f64`.
    ///
    /// # Errors
    ///
    /// Propagates the evaluation error of the expression (not expected for
    /// the expressions this module builds).
    pub fn p_value_f64(&self) -> Result<f64, SymplexError> {
        p_value_f64_of(&self.p_value)
    }

    /// The statistic as an `f64`.
    ///
    /// # Errors
    ///
    /// As [`p_value_f64`](Self::p_value_f64).
    pub fn statistic_f64(&self) -> Result<f64, SymplexError> {
        self.statistic.eval_f64()
    }

    /// The p-value as an exact rational when it is one (the discrete exact
    /// tests: binomial, Fisher, McNemar, sign, exact rank tests).
    #[must_use]
    pub fn p_value_exact(&self) -> Option<Q> {
        self.p_value.as_rational()
    }

    /// The statistic as an exact rational when it is one (`U`, `H`, `χ²`,
    /// `F`, counts and proportions).
    #[must_use]
    pub fn statistic_exact(&self) -> Option<Q> {
        self.statistic.as_rational()
    }
}

impl PValue for TestResult {
    fn p_value_ex(&self) -> &Ex {
        &self.p_value
    }
}
p_value_accessors!(TestResult);

/// How the p-value of a rank test is computed.
///
/// There is no `auto`: the caller chooses.  scipy's `method='auto'` (1.18)
/// is `Exact` for `mannwhitneyu` when there are no ties and one sample has
/// at most 8 observations; for `wilcoxon` when there are at most 50
/// non-zero differences without ties or zeros (with ties or zeros and at
/// most 13 of them it runs a permutation test instead); for `kendalltau`
/// (the `exact` flag of [`kendall_test`]) when there are no ties and `n ≤
/// 33` or at most one pair is discordant (or concordant) — and
/// `Asymptotic` otherwise.  scipy's default continuity correction is on
/// for `mannwhitneyu` and off for `wilcoxon`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RankMethod {
    /// The exact permutation distribution of the statistic (an exact
    /// rational p-value).  Only valid without ties.
    Exact,
    /// The normal approximation with the tie correction, optionally with the
    /// continuity correction (`scipy`'s `use_continuity` / `correction`).
    Asymptotic {
        /// Move the statistic half a unit towards its null mean.
        continuity: bool,
    },
}

/// The outcome of a Pearson χ² test.
#[derive(Clone, Debug, PartialEq)]
pub struct ChiSquareResult {
    /// The χ² statistic, exact.
    pub statistic: Q,
    /// Degrees of freedom.
    pub df: usize,
    /// `P(χ²_df ≥ statistic)` as an exact expression (`uppergamma(df/2,
    /// statistic/2) / Γ(df/2)`).
    pub p_value: Ex,
    /// The expected counts under the null, exact (one row for a
    /// goodness-of-fit test).
    pub expected: Vec<Vec<Q>>,
}

impl ChiSquareResult {
    /// The p-value as an `f64`.
    ///
    /// # Errors
    ///
    /// Propagates the evaluation error of the expression.
    pub fn p_value_f64(&self) -> Result<f64, SymplexError> {
        p_value_f64_of(&self.p_value)
    }
}

impl PValue for ChiSquareResult {
    fn p_value_ex(&self) -> &Ex {
        &self.p_value
    }
}
p_value_accessors!(ChiSquareResult);

/// A ratio estimate (odds ratio, relative risk) with a Wald confidence
/// interval on the log scale.
#[derive(Clone, Debug, PartialEq)]
pub struct RatioEstimate {
    /// The point estimate, exact.
    pub estimate: Q,
    /// The `confidence` Wald interval, computed on the log scale and
    /// exponentiated.
    pub ci: Interval<f64>,
    /// The confidence level of `ci`.
    pub confidence: f64,
}

/// The outcome of a Kolmogorov–Smirnov test (numerical: the statistic
/// compares an empirical CDF against a transcendental one).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KsResult {
    /// `D` (two-sided), `D⁺` (`Greater`) or `D⁻` (`Less`).
    pub statistic: f64,
    /// The p-value: Kolmogorov's asymptotic distribution two-sided,
    /// Smirnov's exact one-sided distribution otherwise.
    pub p_value: f64,
    /// The alternative the p-value refers to.
    pub alternative: Alternative,
}

/// Adjusted p-values and rejection flags of a multiple-comparison
/// procedure, in the order of the input.
#[derive(Clone, Debug, PartialEq)]
pub struct Adjusted {
    /// The adjusted p-values (clipped to `1`).
    pub p_adjusted: Vec<f64>,
    /// Whether each hypothesis is rejected at the family-wise (or false
    /// discovery) level `alpha`.
    pub reject: Vec<bool>,
}

/// How a bootstrap confidence interval is read off the resampled
/// statistics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BootstrapMethod {
    /// The `α/2` and `1 − α/2` quantiles of the bootstrap distribution.
    Percentile,
    /// The reflected ("basic", "reverse percentile") interval
    /// `(2θ̂ − q_{1−α/2}, 2θ̂ − q_{α/2})`.
    Basic,
}

/// The outcome of a randomised permutation test.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PermutationResult {
    /// The observed statistic.
    pub statistic: f64,
    /// `(#{permuted statistics at least as extreme} + 1) / (n_permutations + 1)`.
    pub p_value: f64,
}

// ═══════════════════════════════════════════════════════════════════════════
// Exact tails of the reference distributions
// ═══════════════════════════════════════════════════════════════════════════

/// A statistic `num / √var` with `num`, `var` rational and `var > 0`: a
/// `t` or `z` whose square is rational and whose sign is known exactly.
struct RootRatio {
    num: Q,
    var: Q,
}

impl RootRatio {
    fn to_ex(&self, ctx: &Context) -> Ex {
        (ex(ctx, &self.num) / ex(ctx, &self.var).sqrt()).simplify()
    }

    fn square(&self) -> Q {
        &self.num * &self.num / &self.var
    }

    fn in_tail(&self, alt: Alternative) -> bool {
        in_tail(self.num.is_negative(), self.num.is_positive(), alt)
    }
}

/// Whether a statistic with the sign of `num` lies in the alternative's tail.
fn in_tail(num_is_negative: bool, num_is_positive: bool, alt: Alternative) -> bool {
    match alt {
        Alternative::Greater | Alternative::TwoSided => !num_is_negative,
        Alternative::Less => !num_is_positive,
    }
}

/// The p-value of a Student-t statistic with `df` degrees of freedom.
/// With `z = ν/(t² + ν)` (rational), `P(|T| ≥ |t|) = I_z(ν/2, ½)` and
/// `P(T ≥ t) = ½ I_z(ν/2, ½)` for `t ≥ 0`.
fn student_p_value(ctx: &Context, df: &Q, stat: &RootRatio, alt: Alternative) -> Ex {
    let t2 = stat.square();
    let z = df / (&t2 + df);
    let two_sided =
        ex(ctx, &z).betainc_regularized(&ex(ctx, &(df / qi(2))), &ctx.rational(1, 2), &ctx.zero());
    one_sided_from_symmetric(ctx, two_sided, stat.in_tail(alt), alt)
}

/// The p-value of a standard-normal statistic: `P(|Z| ≥ |z|) = erfc(|z|/√2)`,
/// `P(Z ≥ z) = ½ erfc(z/√2)`.
fn normal_p_value(ctx: &Context, stat: &RootRatio, alt: Alternative) -> Ex {
    let two_sided = ex(ctx, &(stat.square() / qi(2))).sqrt().erfc();
    one_sided_from_symmetric(ctx, two_sided, stat.in_tail(alt), alt)
}

/// Turn the two-sided tail `P(|X| ≥ |x|)` of a symmetric distribution into
/// the requested one: half of it when `x` lies in the alternative's tail,
/// its complement otherwise.
fn one_sided_from_symmetric(ctx: &Context, two_sided: Ex, in_tail: bool, alt: Alternative) -> Ex {
    match alt {
        Alternative::TwoSided => two_sided,
        Alternative::Greater | Alternative::Less => {
            let half = ctx.rational(1, 2) * two_sided;
            if in_tail { half } else { ctx.one() - half }
        }
    }
}

fn result(
    ctx: &Context,
    statistic: Ex,
    p_value: Q,
    df: Option<Ex>,
    alt: Alternative,
) -> TestResult {
    TestResult {
        statistic,
        p_value: ex(ctx, &p_value),
        df,
        alternative: alt,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. Exact discrete tests
// ═══════════════════════════════════════════════════════════════════════════

/// The exact `Binomial(n, p₀)`, `p₀ = a/c` in lowest terms, as integer
/// weights `w(i) = C(n, i)·aⁱ·(c − a)ⁿ⁻ⁱ` over the one denominator `cⁿ`
/// (`P(X = i) = w(i)/cⁿ`).  The weights are streamed by the exact integer
/// recurrence `w(i+1) = w(i)·(n − i)·a / ((i + 1)(c − a))`, so a tail is
/// one pass of integer additions and nothing but the running weight is
/// stored.  (Before 0.27 the pmf was a table of reduced rationals, added
/// one by one: `binomial_test(0, 5000, ½)` took minutes.)
struct BinomialExact {
    n: usize,
    a: BigInt,
    b: BigInt,
    c: BigInt,
    total: BigInt,
}

impl BinomialExact {
    /// `p₀ ∈ [0, 1]`.
    fn new(n: usize, p0: &Q) -> Self {
        let a = p0.numer().clone();
        let c = p0.denom().clone();
        let b = &c - &a;
        let total = Pow::pow(&c, n);
        Self { n, a, b, c, total }
    }

    /// `sum / cⁿ` in lowest terms.  Only primes of `c` can be common, so
    /// `gcd(sum, c)` is divided out until none is left — a remainder by the
    /// small `c` per step, where [`Q::new`]'s full `gcd` of two
    /// `n log₂ c`-bit integers took seconds at `n = 20 000`.
    fn ratio(&self, sum: BigInt) -> Q {
        if sum.is_zero() {
            return Q::zero();
        }
        let (mut sum, mut total) = (sum, self.total.clone());
        loop {
            let g = num_integer::Integer::gcd(&(&sum % &self.c), &self.c);
            let g = num_integer::Integer::gcd(&(&total % &g), &g);
            if g.is_one() {
                break;
            }
            sum /= &g;
            total /= &g;
        }
        Q::new_raw(sum, total)
    }

    /// `Σ_{i ≤ upto} w(i)` and, with `stop_at`, `w(stop_at)`: the weights
    /// up to `max(upto, stop_at)`.
    fn walk(&self, upto: Option<usize>, stop_at: Option<usize>) -> (BigInt, BigInt) {
        let last = upto.max(stop_at).unwrap_or(0).min(self.n);
        let mut sum = BigInt::zero();
        let mut at = BigInt::zero();
        if self.b.is_zero() {
            // p₀ = 1: all the mass (cⁿ = 1) sits at n.
            if upto == Some(self.n) {
                sum = self.total.clone();
            }
            if stop_at == Some(self.n) {
                at = self.total.clone();
            }
            return (sum, at);
        }
        let mut w = Pow::pow(&self.b, self.n);
        for i in 0..=last {
            if upto.is_some_and(|u| i <= u) {
                sum += &w;
            }
            if stop_at == Some(i) {
                at = w.clone();
            }
            if i < last {
                w = w * BigInt::from(self.n - i) * &self.a / (BigInt::from(i + 1) * &self.b);
            }
        }
        (sum, at)
    }

    /// `Σ_{i : w(i) ≤ bound} w(i)`, one full pass.
    fn sum_at_most(&self, bound: &BigInt) -> BigInt {
        if self.b.is_zero() {
            return if self.total <= *bound {
                self.total.clone()
            } else {
                BigInt::zero()
            };
        }
        let mut sum = BigInt::zero();
        let mut w = Pow::pow(&self.b, self.n);
        for i in 0..=self.n {
            if w <= *bound {
                sum += &w;
            }
            if i < self.n {
                w = w * BigInt::from(self.n - i) * &self.a / (BigInt::from(i + 1) * &self.b);
            }
        }
        sum
    }

    /// `P(X ≤ k)`, exact.
    fn cdf(&self, k: usize) -> Q {
        self.ratio(self.walk(Some(k), None).0)
    }

    /// `P(X ≥ k) = 1 − P(X ≤ k − 1)`, exact.
    fn sf(&self, k: usize) -> Q {
        match k.checked_sub(1) {
            None => Q::one(),
            Some(below) => self.ratio(&self.total - self.walk(Some(below), None).0),
        }
    }

    /// scipy's two-sided p-value of a discrete exact test: the total mass
    /// of the outcomes no more likely than the observed one, `Σ_{i : P(i) ≤
    /// P(k)} P(i)`.  scipy compares the floating pmf with a relative slack
    /// of `1e-7`; here the comparison is exact.
    fn two_sided(&self, k: usize) -> Q {
        let at = self.walk(None, Some(k)).1;
        self.ratio(self.sum_at_most(&at)).min(Q::one())
    }
}

/// Exact binomial test of `P(success) = p₀` from `k` successes in `n`
/// trials: the statistic is the proportion `k/n`; the p-value is
/// `P(X ≤ k)` (`Less`), `P(X ≥ k)` (`Greater`) or, two-sided, the total
/// probability of the outcomes no more likely than `k` (`Σ_{i: P(i) ≤ P(k)}
/// P(i)`), an exact rational.  `scipy.stats.binomtest(k, n, p, alternative)`.
///
/// **Two-sided: exact comparison.**  scipy decides "no more likely" on
/// floating pmf values with a relative slack of `10⁻⁷`, so it also adds
/// outcomes that are slightly *more* likely than `k`; here the comparison
/// is exact.  They rarely differ, but then visibly: for `k = 198, n = 950,
/// p₀ = ¼`, `P(X = 278)` exceeds `P(X = 198)` by a relative `2.97·10⁻⁸`, so
/// symplex gives `p = 0.0027180105491057661`, scipy `0.0030516140249515723`
/// (which includes `x = 278`).  Cost: one pass of `BigInt` additions over
/// the `n + 1` outcomes, whose weights have about `n·log₂ c` bits for `p₀ =
/// a/c`, so it grows as `n²`: milliseconds at `n = 5000`, about two
/// seconds (debug build) at `n = 20 000` with `p₀ = 3/20`.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::linprog::q;
/// use symplex::stats::hypothesis::{binomial_test, Alternative};
///
/// let ctx = Context::new();
/// // scipy: binomtest(7, 10, 0.5).pvalue = 0.34375 = 11/32
/// let r = binomial_test(&ctx, 7, 10, &q(1, 2), Alternative::TwoSided)?;
/// assert_eq!(r.p_value_exact(), Some(q(11, 32)));
/// assert_eq!(r.statistic_exact(), Some(q(7, 10)));
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for `n = 0`, `k > n` or `p₀ ∉ [0, 1]`.
pub fn binomial_test(
    ctx: &Context,
    k: usize,
    n: usize,
    p0: &Q,
    alt: Alternative,
) -> Result<TestResult, SymplexError> {
    const OP: &str = "binomial_test";
    if n == 0 {
        return Err(invalid(OP, "the number of trials must be positive"));
    }
    if k > n {
        return Err(invalid(OP, format!("{k} successes exceed {n} trials")));
    }
    if p0.is_negative() || *p0 > Q::one() {
        return Err(invalid(OP, "the null proportion must lie in [0, 1]"));
    }
    let dist = BinomialExact::new(n, p0);
    let p = match alt {
        Alternative::Less => dist.cdf(k),
        Alternative::Greater => dist.sf(k),
        Alternative::TwoSided => dist.two_sided(k),
    };
    Ok(result(ctx, ex(ctx, &(qu(k) / qu(n))), p, None, alt))
}

/// The exact `Hypergeometric(N = n₁ + n₂, n₁, n)` as integer weights
/// `w(x) = C(n₁, x)·C(n₂, n − x)` over the support `lo ..= hi` (`lo = max(0,
/// n − n₂)`, `hi = min(n, n₁)`) and their total `C(N, n)`: `P(X = x) =
/// w(x)/total`.  Tail sums are integer additions over one denominator — a
/// thousand times faster than adding reduced `Q`s at 2 000 points.
struct HypergeomExact {
    lo: usize,
    weights: Vec<BigInt>,
    total: BigInt,
}

impl HypergeomExact {
    /// `w(lo)` from binomials, the rest by the exact integer ratio
    /// `w(x+1) = w(x)·(n₁ − x)(n − x) / ((x + 1)(n₂ + x + 1 − n))`.
    fn new(n1: usize, n2: usize, n: usize) -> Self {
        let lo = n.saturating_sub(n2);
        let hi = n.min(n1);
        let total = data::binomial_q(n1 + n2, n).to_integer();
        let mut w = (data::binomial_q(n1, lo) * data::binomial_q(n2, n - lo)).to_integer();
        let mut weights = Vec::with_capacity(hi - lo + 1);
        for x in lo..hi {
            weights.push(w.clone());
            w = w * BigInt::from(n1 - x) * BigInt::from(n - x)
                / (BigInt::from(x + 1) * BigInt::from(n2 + x + 1 - n));
        }
        weights.push(w);
        Self { lo, weights, total }
    }

    /// `P(X ∈ {x : keep(x − lo, w(x))})`, an exact rational.
    fn mass_where(&self, keep: impl Fn(usize, &BigInt) -> bool) -> Q {
        let sum = self
            .weights
            .iter()
            .enumerate()
            .filter(|(i, w)| keep(*i, w))
            .fold(BigInt::zero(), |acc, (_, w)| acc + w);
        Q::new(sum, self.total.clone())
    }

    /// `P(X ≤ a)`, `P(X ≥ a)`, or the two-sided total mass of the outcomes
    /// no more likely than `a` (`Σ_{x: w(x) ≤ w(a)} w(x) / total`).
    fn p_value(&self, a: usize, alt: Alternative) -> Q {
        let idx = a - self.lo;
        match alt {
            Alternative::Less => self.mass_where(|i, _| i <= idx),
            Alternative::Greater => self.mass_where(|i, _| i >= idx),
            Alternative::TwoSided => {
                let at = &self.weights[idx];
                self.mass_where(|_, w| w <= at).min(Q::one())
            }
        }
    }
}

/// Support size (number of possible values of the top-left cell given the
/// margins) above which [`fisher_exact`] leaves the exact `BigInt`
/// enumeration for its numeric route.
pub const FISHER_EXACT_NUMERIC_THRESHOLD: usize = 2_000;

/// Longest walk of the pmf recurrence before [`HypergeomNumeric`] falls
/// back to Stirling (`lgamma`) differences, whose absolute error in the
/// logarithm (`~1e-5` at arguments near `10⁹`) is then invisible: a point
/// that far from the mode carries a mass below `e^{-10⁴}`.
const HYPERGEOM_MAX_WALK: usize = 2_000_000;

/// Relative tolerance under which two floating pmf values count as equal
/// for the two-sided cutoff (the walk from the mode loses about
/// `steps · ε`); exact ties — a symmetric distribution — are recognised
/// without it.
const HYPERGEOM_TIE_TOL: f64 = 1e-11;

/// A term below this fraction of the running sum ends an outer-tail walk.
const HYPERGEOM_TAIL_CUTOFF: f64 = 1e-22;

/// The direction of a walk along the lattice.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Walk {
    Down,
    Up,
}

/// `Hypergeometric(N = n₁ + n₂, n₁, n)` in floating point, for the large
/// tables of [`fisher_exact`].  The pmf is never formed from `lgamma`
/// differences — nine terms of size `10¹⁰` cancelling to `−10` lose five
/// digits — but walked from the mode with the exact ratio
/// `f(x+1)/f(x) = (n₁ − x)(n − x) / ((x + 1)(n₂ + x + 1 − n))`, so a tail
/// is accurate to about `steps · ε` (`10⁻¹²` for the `10⁴`-point window of
/// a `4·10⁶` table).  Every tail is summed *outwards* from its inner end
/// (terms decreasing) in units of `f(mode)`, and the normalisation is the
/// walked total rather than `C(N, n)`.
struct HypergeomNumeric {
    n1: usize,
    n2: usize,
    n: usize,
    lo: usize,
    hi: usize,
    mode: usize,
    /// `ln Σ_x f(x)/f(mode)`.
    ln_total: f64,
}

impl HypergeomNumeric {
    fn new(n1: usize, n2: usize, n: usize) -> Self {
        let lo = n.saturating_sub(n2);
        let hi = n.min(n1);
        let wide = |v: usize| v as u128;
        // ⌊(n + 1)(n₁ + 1) / (N + 2)⌋ is a mode; clamped into the support.
        let mode = (wide(n) + 1) * (wide(n1) + 1) / (wide(n1) + wide(n2) + 2);
        let mode = usize::try_from(mode).unwrap_or(hi).clamp(lo, hi);
        let mut me = Self {
            n1,
            n2,
            n,
            lo,
            hi,
            mode,
            ln_total: 0.0,
        };
        let below = if mode > lo {
            me.ln_outer_tail(mode - 1, Walk::Down).exp()
        } else {
            0.0
        };
        let above = if mode < hi {
            me.ln_outer_tail(mode + 1, Walk::Up).exp()
        } else {
            0.0
        };
        me.ln_total = (1.0 + below + above).ln();
        me
    }

    /// `f(x) = f(lo + hi − x)`: `n₁ = n₂` or `2n = N`.
    fn symmetric(&self) -> bool {
        self.n1 == self.n2 || 2 * self.n == self.n1 + self.n2
    }

    /// `f(x + 1) / f(x)` for `lo ≤ x < hi` (every factor is an exact
    /// integer below `2⁵³`).
    fn ratio_up(&self, x: usize) -> f64 {
        let num = (self.n1 - x) as f64 * (self.n - x) as f64;
        let den = (x + 1) as f64 * (self.n2 + x + 1 - self.n) as f64;
        num / den
    }

    /// `ln C(n, k)` by Stirling.
    fn ln_binom(n: usize, k: usize) -> f64 {
        lgamma(n as f64 + 1.0) - lgamma(k as f64 + 1.0) - lgamma((n - k) as f64 + 1.0)
    }

    /// `ln(f(x)/f(mode))` from `lgamma` differences.
    fn ln_shape_stirling(&self, x: usize) -> f64 {
        let (n1, n2, n, m) = (self.n1, self.n2, self.n, self.mode);
        Self::ln_binom(n1, x) - Self::ln_binom(n1, m) + Self::ln_binom(n2, n - x)
            - Self::ln_binom(n2, n - m)
    }

    /// `ln(f(x)/f(mode))`: the recurrence walked from the mode, or
    /// Stirling beyond [`HYPERGEOM_MAX_WALK`] steps.
    fn ln_shape(&self, x: usize) -> f64 {
        let m = self.mode;
        if x.abs_diff(m) > HYPERGEOM_MAX_WALK {
            return self.ln_shape_stirling(x);
        }
        let mut acc = LogProduct::default();
        if x > m {
            for y in m..x {
                acc.mul(self.ratio_up(y));
            }
        } else {
            for y in (x..m).rev() {
                acc.div(self.ratio_up(y));
            }
        }
        acc.ln()
    }

    /// `ln Σ f(y)/f(mode)` over `y` from `start` to the support's end in
    /// direction `dir`.  Meant for an *outer* tail (`start` on or beyond
    /// the mode in that direction), whose terms decrease: the walk stops
    /// once a term falls below [`HYPERGEOM_TAIL_CUTOFF`] of the sum.
    fn ln_outer_tail(&self, start: usize, dir: Walk) -> f64 {
        let ln_start = self.ln_shape(start);
        let mut sum = 1.0_f64;
        let mut term = 1.0_f64;
        let mut y = start;
        loop {
            match dir {
                Walk::Up => {
                    if y >= self.hi {
                        break;
                    }
                    term *= self.ratio_up(y);
                    y += 1;
                }
                Walk::Down => {
                    if y <= self.lo {
                        break;
                    }
                    term /= self.ratio_up(y - 1);
                    y -= 1;
                }
            }
            sum += term;
            if term < HYPERGEOM_TAIL_CUTOFF * sum {
                break;
            }
        }
        ln_start + sum.ln()
    }

    /// `ln P(X ≤ a)` (`Down`) or `ln P(X ≥ a)` (`Up`): the outer tail when
    /// `a` lies beyond the mode in that direction, else one minus the
    /// opposite outer tail.
    fn ln_tail(&self, a: usize, dir: Walk) -> f64 {
        let m = self.mode;
        match dir {
            Walk::Down if a >= self.hi => 0.0,
            Walk::Up if a <= self.lo => 0.0,
            Walk::Down if a < m => self.ln_outer_tail(a, Walk::Down) - self.ln_total,
            Walk::Up if a > m => self.ln_outer_tail(a, Walk::Up) - self.ln_total,
            Walk::Down => (-(self.ln_outer_tail(a + 1, Walk::Up) - self.ln_total).exp()).ln_1p(),
            Walk::Up => (-(self.ln_outer_tail(a - 1, Walk::Down) - self.ln_total).exp()).ln_1p(),
        }
    }

    /// The two-sided cutoff on the far side of the mode from `a < mode`:
    /// the smallest `g ≥ mode` with `f(g) ≤ f(a)` (up to
    /// [`HYPERGEOM_TIE_TOL`]), or `None` when even `f(hi)` exceeds `f(a)`.
    /// A symmetric distribution mirrors `a` exactly.
    fn cutoff_above(&self, a: usize) -> Option<usize> {
        if self.symmetric() {
            return Some(self.lo + self.hi - a);
        }
        let target = self.ln_shape(a) + HYPERGEOM_TIE_TOL;
        let m = self.mode;
        let mut acc = LogProduct::default();
        let mut y = m;
        loop {
            if acc.ln() <= target {
                return Some(y);
            }
            if y >= self.hi {
                return None;
            }
            if y - m >= HYPERGEOM_MAX_WALK {
                // Far tail: the first point past `y` where the (decreasing)
                // Stirling shape drops to the target; `hi` qualifies.
                if self.ln_shape_stirling(self.hi) > target {
                    return None;
                }
                return Some(partition_point_by(y + 1, self.hi, |g| {
                    self.ln_shape_stirling(g) <= target
                }));
            }
            acc.mul(self.ratio_up(y));
            y += 1;
        }
    }

    /// Mirror image of [`cutoff_above`](Self::cutoff_above) for `a > mode`:
    /// the largest `g ≤ mode` with `f(g) ≤ f(a)`.
    fn cutoff_below(&self, a: usize) -> Option<usize> {
        if self.symmetric() {
            return Some(self.lo + self.hi - a);
        }
        let target = self.ln_shape(a) + HYPERGEOM_TIE_TOL;
        let m = self.mode;
        let mut acc = LogProduct::default();
        let mut y = m;
        loop {
            if acc.ln() <= target {
                return Some(y);
            }
            if y <= self.lo {
                return None;
            }
            if m - y >= HYPERGEOM_MAX_WALK {
                // Mirror image: the shape rises towards `y`, so search the
                // distance below `y` for the first point where it still
                // exceeds the target; the cutoff is one step further down.
                if self.ln_shape_stirling(self.lo) > target {
                    return None;
                }
                let first_above =
                    partition_point_by(self.lo, y, |g| self.ln_shape_stirling(g) > target);
                return Some(first_above - 1);
            }
            acc.div(self.ratio_up(y - 1));
            y -= 1;
        }
    }

    /// `ln p` of Fisher's test at `a` (in the support) for `alt`.
    fn ln_p_value(&self, a: usize, alt: Alternative) -> f64 {
        match alt {
            Alternative::Less => self.ln_tail(a, Walk::Down),
            Alternative::Greater => self.ln_tail(a, Walk::Up),
            Alternative::TwoSided => {
                let m = self.mode;
                if a == m {
                    return 0.0;
                }
                let (near, far) = if a < m {
                    (
                        self.ln_tail(a, Walk::Down),
                        self.cutoff_above(a).map(|g| self.ln_tail(g, Walk::Up)),
                    )
                } else {
                    (
                        self.ln_tail(a, Walk::Up),
                        self.cutoff_below(a).map(|g| self.ln_tail(g, Walk::Down)),
                    )
                };
                match far {
                    Some(far) => log_add_exp(near, far).min(0.0),
                    None => near,
                }
            }
        }
    }
}

/// A running product kept as `acc · e^{shift}` with `acc ∈ [1e-250, 1]`
/// (or above `1` briefly on a flat top), so a walk of a million factors
/// below one never underflows.
#[derive(Clone, Copy)]
struct LogProduct {
    acc: f64,
    shift: f64,
}

impl Default for LogProduct {
    fn default() -> Self {
        Self {
            acc: 1.0,
            shift: 0.0,
        }
    }
}

impl LogProduct {
    fn renormalise(&mut self) {
        if self.acc < 1e-250 || self.acc > 1e250 {
            self.shift += self.acc.ln();
            self.acc = 1.0;
        }
    }

    fn mul(&mut self, r: f64) {
        self.acc *= r;
        self.renormalise();
    }

    fn div(&mut self, r: f64) {
        self.acc /= r;
        self.renormalise();
    }

    fn ln(&self) -> f64 {
        self.shift + self.acc.ln()
    }
}

/// `ln(eᵃ + eᵇ)` without overflow.
fn log_add_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
    hi + (lo - hi).exp().ln_1p()
}

/// A numeric p-value known through its logarithm, as an expression: the
/// dyadic `ctx.from_f64(p)` when `p` is a normal `f64`, else `exp(ln p)`
/// (with `ln p` dyadic) so that [`PValue::p_value_log10`] and
/// [`PValue::p_value_decimal`] stay informative where `p_value_f64`
/// underflows to `0.0`.
fn numeric_p_value(ctx: &Context, ln_p: f64) -> Result<Ex, SymplexError> {
    if ln_p == f64::NEG_INFINITY {
        return Ok(ctx.zero());
    }
    let p = ln_p.exp().min(1.0);
    if p >= f64::MIN_POSITIVE {
        ctx.from_f64(p)
    } else {
        Ok(ctx.from_f64(ln_p)?.exp())
    }
}

/// Fisher's exact test on a 2×2 table `[[a, b], [c, d]]`: the statistic is
/// the sample odds ratio `ad/(bc)` (`+∞` when `bc = 0`); the p-value is
/// from the hypergeometric distribution of `a` given the margins —
/// `P(X ≤ a)` (`Less`), `P(X ≥ a)` (`Greater`), or two-sided the total mass
/// of the tables no more likely than the observed one.
/// `scipy.stats.fisher_exact(table, alternative)`.
///
/// **Exact below the threshold, numeric above.**  While the support of
/// `a` has at most [`FISHER_EXACT_NUMERIC_THRESHOLD`] (2 000) points the
/// p-value is an exact rational from `BigInt` binomials.  Above it — cells
/// in the `10⁴`–`10¹¹` range — the p-value is numeric: the pmf is walked
/// from its mode with the exact ratio `f(x+1)/f(x)`, each tail summed
/// outwards from its inner end, and the result is `ctx.from_f64(p)`
/// (dyadic, so [`TestResult::p_value_exact`] is `Some` but not the exact
/// rational), agreeing with scipy to `1e-9` relative or better.  Below
/// `1e-308` the expression is `exp(ln p)` so that
/// [`p_value_log10`](TestResult::p_value_log10) stays finite where
/// [`p_value_f64`](TestResult::p_value_f64) is `0.0`.  The two-sided
/// cutoff on the far side of the mode compares floating pmf values with a
/// `1e-11` relative tolerance except for a symmetric distribution
/// (`n₁ = n₂` or `2n = N`), whose mirror point is exact.  `[[10⁶, 10⁶ +
/// 7], [10⁶ − 3, 10⁶]]` takes milliseconds (it took minutes before 0.18.1).
///
/// ```
/// use symplex::prelude::*;
/// use symplex::linprog::{q, qi};
/// use symplex::stats::hypothesis::{fisher_exact, Alternative};
///
/// let ctx = Context::new();
/// // scipy: fisher_exact([[8, 2], [1, 5]]) → statistic 20.0, pvalue 0.034965034965034975 = 5/143
/// let r = fisher_exact(&ctx, [[8, 2], [1, 5]], Alternative::TwoSided)?;
/// assert_eq!(r.statistic_exact(), Some(qi(20)));
/// assert_eq!(r.p_value_exact(), Some(q(5, 143)));
/// // A large table takes the numeric route: scipy 0.9992021157304142 (mpmath 0.99920211598773724586)
/// let big = fisher_exact(&ctx, [[1_000_000, 1_000_007], [999_997, 1_000_000]], Alternative::TwoSided)?;
/// assert!((big.p_value_f64()? - 0.999_202_115_987_737_2).abs() < 1e-12);
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] if a row or a column of the table is
/// empty (the test is undefined; scipy reports `p = 1`, odds ratio NaN),
/// or if a margin overflows `usize`.
pub fn fisher_exact(
    ctx: &Context,
    table: [[usize; 2]; 2],
    alt: Alternative,
) -> Result<TestResult, SymplexError> {
    const OP: &str = "fisher_exact";
    let [[a, b], [c, d]] = table;
    let margin = |x: usize, y: usize| {
        x.checked_add(y)
            .ok_or_else(|| invalid(OP, "a margin of the table overflows usize"))
    };
    let (n1, n2, n, n_col2) = (margin(a, b)?, margin(c, d)?, margin(a, c)?, margin(b, d)?);
    margin(n1, n2)?;
    if n1 == 0 || n2 == 0 || n == 0 || n_col2 == 0 {
        return Err(invalid(OP, "a row or a column of the table is empty"));
    }
    let odds = if b == 0 || c == 0 {
        ctx.infinity()
    } else {
        ex(ctx, &(qu(a) * qu(d) / (qu(b) * qu(c))))
    };
    let lo = n.saturating_sub(n2);
    let hi = n.min(n1);
    if hi - lo >= FISHER_EXACT_NUMERIC_THRESHOLD {
        let ln_p = HypergeomNumeric::new(n1, n2, n).ln_p_value(a, alt);
        return Ok(TestResult {
            statistic: odds,
            p_value: numeric_p_value(ctx, ln_p)?,
            df: None,
            alternative: alt,
        });
    }
    let p = HypergeomExact::new(n1, n2, n).p_value(a, alt);
    Ok(result(ctx, odds, p, None, alt))
}

/// McNemar's test on the discordant counts `b` (row 1 / column 2) and `c`
/// (row 2 / column 1) of a paired 2×2 table.  `exact`: statistic
/// `min(b, c)`, p-value `min(1, 2·P(Binomial(b + c, ½) ≤ min(b, c)))`, an
/// exact rational.  Otherwise the χ² statistic `max(|b − c| − 1, 0)²/(b +
/// c)` (`correction`, Edwards 1948) or `(b − c)²/(b + c)` with `P(χ²₁ ≥
/// ·)`.  `statsmodels.stats.contingency_tables.mcnemar(table, exact,
/// correction)`, except at `b = c` with the correction: the continuity
/// correction moves `|b − c|` towards `0` and stops there (as R's
/// `mcnemar.test` and scipy's Yates correction in `chi2_contingency` do),
/// so the statistic is `0` and `p = 1` — statsmodels squares `−1` into
/// `1/(b + c)` (`b = c = 5`: `χ² = 1/10`, `p = 0.7518`, which symplex
/// returned before 0.27), smaller than the exact test's `p = 1`.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::linprog::q;
/// use symplex::stats::hypothesis::mcnemar_test;
///
/// let ctx = Context::new();
/// // statsmodels: mcnemar([[100, 5], [15, 100]], exact=True) → statistic 5.0, pvalue 0.04138946533203125 = 5425/131072
/// let r = mcnemar_test(&ctx, 5, 15, true, true)?;
/// assert_eq!(r.p_value_exact(), Some(q(5425, 131072)));
/// // statsmodels: mcnemar(..., exact=False, correction=True) → statistic 4.05, pvalue 0.0441713449084427
/// let r = mcnemar_test(&ctx, 5, 15, false, true)?;
/// assert_eq!(r.statistic_exact(), Some(q(81, 20)));
/// assert!((r.p_value_f64()? - 0.044_171_344_908_442_7).abs() < 1e-12);
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] when `b + c = 0` (no discordant pairs)
/// or `b + c` overflows `usize`.
pub fn mcnemar_test(
    ctx: &Context,
    b: usize,
    c: usize,
    exact: bool,
    correction: bool,
) -> Result<TestResult, SymplexError> {
    const OP: &str = "mcnemar_test";
    let n = b
        .checked_add(c)
        .ok_or_else(|| invalid(OP, "b + c overflows usize"))?;
    if n == 0 {
        return Err(invalid(OP, "there are no discordant pairs"));
    }
    if exact {
        let k = b.min(c);
        let half = Q::new(BigInt::one(), BigInt::from(2));
        let p = (BinomialExact::new(n, &half).cdf(k) * qi(2)).min(Q::one());
        return Ok(result(
            ctx,
            ctx.int(k as i64),
            p,
            None,
            Alternative::TwoSided,
        ));
    }
    let gap = b.abs_diff(c);
    let diff = qu(if correction {
        gap.saturating_sub(1)
    } else {
        gap
    });
    let stat = &diff * &diff / qu(n);
    Ok(TestResult {
        statistic: ex(ctx, &stat),
        p_value: chi_squared_sf_q(ctx, 1, &stat),
        df: Some(ctx.one()),
        alternative: Alternative::TwoSided,
    })
}

/// The sign test of `median = μ₀`: with `n₊` observations above and `n₋`
/// below `μ₀` (ties dropped), the statistic is `M = (n₊ − n₋)/2` and the
/// p-value is the exact binomial test of `n₊` out of `n₊ + n₋` with
/// `p = ½` (`statsmodels.stats.descriptivestats.sign_test(x, mu0)`, which
/// is two-sided; `Greater` is `P(X ≥ n₊)`, `Less` is `P(X ≤ n₊)`).
///
/// ```
/// use symplex::prelude::*;
/// use symplex::linprog::{q, qi};
/// use symplex::stats::data::from_i64;
/// use symplex::stats::hypothesis::{sign_test, Alternative};
///
/// let ctx = Context::new();
/// let x = from_i64(&[3, 5, 7, 8, 9, 11, 12, 15, 2, 6]);
/// // statsmodels: sign_test(x, mu0=5) = (2.5, 0.1796875)   (7 above, 2 below; 0.1796875 = 23/128)
/// let r = sign_test(&ctx, &x, &qi(5), Alternative::TwoSided)?;
/// assert_eq!(r.statistic_exact(), Some(q(5, 2)));
/// assert_eq!(r.p_value_exact(), Some(q(23, 128)));
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] when every observation equals `μ₀`.
pub fn sign_test(
    ctx: &Context,
    x: &[Q],
    mu0: &Q,
    alt: Alternative,
) -> Result<TestResult, SymplexError> {
    const OP: &str = "sign_test";
    let pos = x.iter().filter(|v| *v > mu0).count();
    let neg = x.iter().filter(|v| *v < mu0).count();
    if pos + neg == 0 {
        return Err(invalid(OP, "every observation equals the null median"));
    }
    let half = Q::new(BigInt::one(), BigInt::from(2));
    let inner = binomial_test(ctx, pos, pos + neg, &half, alt)?;
    let m = (qu(pos) - qu(neg)) / qi(2);
    Ok(TestResult {
        statistic: ex(ctx, &m),
        ..inner
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Categorical
// ═══════════════════════════════════════════════════════════════════════════

/// A table of counts from integer rows (a convenience for the table-valued
/// functions of this module).
///
/// ```
/// use symplex::stats::hypothesis::counts;
/// use symplex::linprog::qi;
/// let t = counts(&[&[10, 20], &[30, 40]]);
/// assert_eq!(t[1][0], qi(30));
/// ```
#[must_use]
pub fn counts(rows: &[&[i64]]) -> Vec<Vec<Q>> {
    rows.iter().map(|r| data::from_i64(r)).collect()
}

/// A table of counts from `usize` rows — the shape
/// [`confusion_matrix`](super::agreement::confusion_matrix) and
/// [`RatingTable::count_table`](super::agreement::RatingTable::count_table)
/// produce, so they compose with [`chi_square_independence`], [`g_test`],
/// [`expected_counts`] and the residuals.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::linprog::q;
/// use symplex::stats::agreement::confusion_matrix;
/// use symplex::stats::data::from_i64;
/// use symplex::stats::hypothesis::{chi_square_independence, counts_usize};
///
/// let ctx = Context::new();
/// let a = from_i64(&[1, 2, 3, 1, 2, 3, 1, 1, 2, 3]);
/// let b = from_i64(&[1, 2, 3, 1, 3, 3, 1, 2, 2, 3]);
/// let m = confusion_matrix(&a, &b, &from_i64(&[1, 2, 3]))?;
/// let r = chi_square_independence(&ctx, &counts_usize(&m), false)?;
/// assert_eq!(r.df, 4);
/// // Row sums 4, 3, 3 and column sums 3, 3, 4 over N = 10: E₁₁ = 4 · 3 / 10.
/// assert_eq!(r.expected[0][0], q(6, 5));
/// # Ok::<(), SymplexError>(())
/// ```
#[must_use]
pub fn counts_usize(rows: &[Vec<usize>]) -> Vec<Vec<Q>> {
    rows.iter()
        .map(|r| r.iter().map(|&c| qu(c)).collect())
        .collect()
}

/// Check a rectangular table of non-negative entries; returns `(rows, cols)`.
fn check_table(op: &'static str, table: &[Vec<Q>]) -> Result<(usize, usize), SymplexError> {
    let r = table.len();
    let c = table.first().map_or(0, Vec::len);
    if r == 0 || c == 0 {
        return Err(invalid(op, "the table is empty"));
    }
    if table.iter().any(|row| row.len() != c) {
        return Err(invalid(op, "the table is not rectangular"));
    }
    if table.iter().flatten().any(Signed::is_negative) {
        return Err(invalid(op, "counts must be non-negative"));
    }
    Ok((r, c))
}

/// Row sums, column sums and the grand total.
fn margins(table: &[Vec<Q>]) -> (Vec<Q>, Vec<Q>, Q) {
    let cols = table[0].len();
    let rows: Vec<Q> = table.iter().map(|r| data::sum(r)).collect();
    let col_sums: Vec<Q> = (0..cols)
        .map(|j| table.iter().fold(Q::zero(), |acc, r| acc + &r[j]))
        .collect();
    let total = data::sum(&rows);
    (rows, col_sums, total)
}

/// The expected counts under independence together with the margins they
/// were computed from.
struct Expected {
    /// `Eᵢⱼ = rowᵢ · colⱼ / N`.
    cells: Vec<Vec<Q>>,
    rows: Vec<Q>,
    cols: Vec<Q>,
    total: Q,
}

/// The expected counts of a checked table, erroring on an empty row or
/// column.
fn expected_of(op: &'static str, table: &[Vec<Q>]) -> Result<Expected, SymplexError> {
    check_table(op, table)?;
    let (rows, cols, total) = margins(table);
    if rows.iter().any(Zero::is_zero) || cols.iter().any(Zero::is_zero) {
        return Err(invalid(
            op,
            "a row or a column of the table is empty, so an expected count is zero",
        ));
    }
    let cells = rows
        .iter()
        .map(|r| cols.iter().map(|c| r * c / &total).collect())
        .collect();
    Ok(Expected {
        cells,
        rows,
        cols,
        total,
    })
}

/// The expected counts under independence, `Eᵢⱼ = rowᵢ · colⱼ / N`, exact.
/// `statsmodels.stats.contingency_tables.Table(t).fittedvalues`;
/// `scipy.stats.contingency.expected_freq`.
///
/// ```
/// use symplex::linprog::q;
/// use symplex::stats::hypothesis::{counts, expected_counts};
///
/// let t = counts(&[&[10, 20, 30], &[6, 9, 17]]);
/// // statsmodels: Table(t).fittedvalues[0][0] = 10.434782608695652 (= 240/23)
/// assert_eq!(expected_counts(&t)?[0][0], q(240, 23));
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for an empty, ragged or negative
/// table, or an empty row / column.
pub fn expected_counts(table: &[Vec<Q>]) -> Result<Vec<Vec<Q>>, SymplexError> {
    Ok(expected_of("expected_counts", table)?.cells)
}

/// Each cell's contribution `(Oᵢⱼ − Eᵢⱼ)² / Eᵢⱼ` to Pearson's χ², exact
/// (they sum to the statistic of [`chi_square_independence`] without
/// Yates' correction).  `Table(t).chi2_contribs`.
///
/// ```
/// use symplex::linprog::q;
/// use symplex::stats::hypothesis::{chi2_contributions, counts};
///
/// let t = counts(&[&[10, 20, 30], &[6, 9, 17]]);
/// // statsmodels: Table(t).chi2_contribs[1][1] = 0.11712893553223394 (= 625/5336)
/// assert_eq!(chi2_contributions(&t)?[1][1], q(625, 5336));
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// As [`expected_counts`].
pub fn chi2_contributions(table: &[Vec<Q>]) -> Result<Vec<Vec<Q>>, SymplexError> {
    let expected = expected_of("chi2_contributions", table)?.cells;
    Ok(table
        .iter()
        .zip(&expected)
        .map(|(o, e)| o.iter().zip(e).map(|(o, e)| square(&(o - e)) / e).collect())
        .collect())
}

/// The standardized (Pearson) residuals `(Oᵢⱼ − Eᵢⱼ) / √Eᵢⱼ`, exact
/// expressions; their squares are the [`chi2_contributions`].
/// `statsmodels` `Table(t).resid_pearson` (statsmodels reserves the name
/// `standardized_resids` for the [`adjusted_residuals`]).
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::hypothesis::{counts, standardized_residuals};
///
/// let ctx = Context::new();
/// let t = counts(&[&[10, 20, 30], &[6, 9, 17]]);
/// // statsmodels: Table(t).resid_pearson[0][1] = 0.24993752342773828
/// let r = standardized_residuals(&ctx, &t)?;
/// assert!((r[0][1].eval_f64()? - 0.249_937_523_427_738_28).abs() < 1e-12);
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// As [`expected_counts`].
pub fn standardized_residuals(
    ctx: &Context,
    table: &[Vec<Q>],
) -> Result<Vec<Vec<Ex>>, SymplexError> {
    let expected = expected_of("standardized_residuals", table)?.cells;
    Ok(table
        .iter()
        .zip(&expected)
        .map(|(o, e)| {
            o.iter()
                .zip(e)
                .map(|(o, e)| (ex(ctx, &(o - e)) / ex(ctx, e).sqrt()).simplify())
                .collect()
        })
        .collect())
}

/// Haberman's adjusted residuals (1973): the Pearson residual divided by
/// its standard error under independence,
///
/// `rᵢⱼ = (Oᵢⱼ − Eᵢⱼ) / √(Eᵢⱼ (1 − rowᵢ/N)(1 − colⱼ/N))`,
///
/// approximately standard normal, so `|r| > 2` flags a cell.  Exact
/// expressions.  `statsmodels` `Table(t).standardized_resids`.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::hypothesis::{adjusted_residuals, counts};
///
/// let ctx = Context::new();
/// let t = counts(&[&[10, 20, 30], &[6, 9, 17]]);
/// // statsmodels: Table(t).standardized_resids[0][1] = 0.5121226989905664
/// let r = adjusted_residuals(&ctx, &t)?;
/// assert!((r[0][1].eval_f64()? - 0.512_122_698_990_566_4).abs() < 1e-12);
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// As [`expected_counts`], and for a table with a single row or column
/// (the adjustment factor vanishes).
pub fn adjusted_residuals(ctx: &Context, table: &[Vec<Q>]) -> Result<Vec<Vec<Ex>>, SymplexError> {
    const OP: &str = "adjusted_residuals";
    let Expected {
        cells: expected,
        rows,
        cols,
        total,
    } = expected_of(OP, table)?;
    if rows.len() < 2 || cols.len() < 2 {
        return Err(invalid(
            OP,
            "adjusted residuals need at least two rows and two columns",
        ));
    }
    let row_f: Vec<Q> = rows.iter().map(|r| Q::one() - r / &total).collect();
    let col_f: Vec<Q> = cols.iter().map(|c| Q::one() - c / &total).collect();
    Ok(table
        .iter()
        .zip(&expected)
        .zip(&row_f)
        .map(|((o, e), rf)| {
            o.iter()
                .zip(e)
                .zip(&col_f)
                .map(|((o, e), cf)| {
                    let var = e * rf * cf;
                    (ex(ctx, &(o - e)) / ex(ctx, &var).sqrt()).simplify()
                })
                .collect()
        })
        .collect())
}

/// `Σ (max(|O − E| − shift, 0))² / E` over a table and its expected counts
/// (the correction never moves an observed count past its expected one,
/// as in scipy ≥ 1.7 and R's `chisq.test`).
fn pearson_statistic(observed: &[Vec<Q>], expected: &[Vec<Q>], shift: &Q) -> Q {
    observed
        .iter()
        .zip(expected)
        .flat_map(|(o, e)| o.iter().zip(e))
        .fold(Q::zero(), |acc, (o, e)| {
            let d = ((o - e).abs() - shift).max(Q::zero());
            acc + &d * &d / e
        })
}

/// Pearson's χ² test of independence on an `r × c` table of counts:
/// `χ² = Σ (Oᵢⱼ − Eᵢⱼ)²/Eᵢⱼ` with `Eᵢⱼ = rowᵢ · colⱼ / N`, `df = (r−1)(c−1)`;
/// with `correction` and `df = 1` Yates' `(max(|O − E| − ½, 0))²` (the
/// correction is clamped to `|O − E|`, exactly as scipy ≥ 1.7 and R).  The
/// statistic and the expected counts are exact rationals; the p-value is
/// `P(χ²_df ≥ χ²)`.
/// `scipy.stats.chi2_contingency(table, correction)`.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::linprog::q;
/// use symplex::stats::hypothesis::{chi_square_independence, counts};
///
/// let ctx = Context::new();
/// let t = counts(&[&[10, 20, 30], &[6, 9, 17]]);
/// // scipy: chi2_contingency(t) → statistic 0.27157465150403504, dof 2, pvalue 0.873028283380073
/// let r = chi_square_independence(&ctx, &t, true)?;
/// assert_eq!(r.df, 2);
/// assert_eq!(r.expected[0][0], q(240, 23));
/// assert!((r.p_value_f64()? - 0.873_028_283_380_073).abs() < 1e-12);
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for a table with fewer than two rows
/// or columns, a ragged or negative table, or an empty row / column.
pub fn chi_square_independence(
    ctx: &Context,
    table: &[Vec<Q>],
    correction: bool,
) -> Result<ChiSquareResult, SymplexError> {
    const OP: &str = "chi_square_independence";
    let (r, c) = check_table(OP, table)?;
    if r < 2 || c < 2 {
        return Err(invalid(
            OP,
            "the table needs at least two rows and two columns",
        ));
    }
    let expected = expected_of(OP, table)?.cells;
    let df = (r - 1) * (c - 1);
    let shift = if correction && df == 1 {
        Q::new(BigInt::one(), BigInt::from(2))
    } else {
        Q::zero()
    };
    let statistic = pearson_statistic(table, &expected, &shift);
    Ok(ChiSquareResult {
        p_value: chi_squared_sf_q(ctx, df, &statistic),
        statistic,
        df,
        expected,
    })
}

/// Pearson's χ² goodness-of-fit test of observed counts against expected
/// ones (`None`: all equal to the mean count): `χ² = Σ (O − E)²/E`, `df = k −
/// 1 − ddof`.  The expected counts must sum to the observed total (exactly;
/// scipy tolerates `1e-8`).  `scipy.stats.chisquare(f_obs, f_exp, ddof)`.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::linprog::qi;
/// use symplex::stats::data::from_i64;
/// use symplex::stats::hypothesis::chi_square_goodness_of_fit;
///
/// let ctx = Context::new();
/// // scipy: chisquare([16, 18, 16, 14, 12, 12]) → statistic 2.0, pvalue 0.8491450360846096
/// let r = chi_square_goodness_of_fit(&ctx, &from_i64(&[16, 18, 16, 14, 12, 12]), None, 0)?;
/// assert_eq!(r.statistic, qi(2));
/// assert_eq!(r.df, 5);
/// assert!((r.p_value_f64()? - 0.849_145_036_084_609_6).abs() < 1e-12);
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for fewer than two categories, a
/// non-positive expected count, mismatched lengths or totals, or `ddof ≥ k − 1`.
pub fn chi_square_goodness_of_fit(
    ctx: &Context,
    observed: &[Q],
    expected: Option<&[Q]>,
    ddof: usize,
) -> Result<ChiSquareResult, SymplexError> {
    const OP: &str = "chi_square_goodness_of_fit";
    let k = observed.len();
    if k < 2 {
        return Err(invalid(OP, "at least two categories are needed"));
    }
    if observed.iter().any(Signed::is_negative) {
        return Err(invalid(OP, "observed counts must be non-negative"));
    }
    let expected: Vec<Q> = match expected {
        Some(e) => {
            if e.len() != k {
                return Err(invalid(OP, "observed and expected have different lengths"));
            }
            if e.iter().any(|v| !v.is_positive()) {
                return Err(invalid(OP, "expected counts must be positive"));
            }
            if data::sum(e) != data::sum(observed) {
                return Err(invalid(
                    OP,
                    "observed and expected counts have different totals",
                ));
            }
            e.to_vec()
        }
        None => {
            let m = data::mean(observed)?;
            if !m.is_positive() {
                return Err(invalid(OP, "the observed counts are all zero"));
            }
            vec![m; k]
        }
    };
    if ddof + 1 >= k {
        return Err(invalid(
            OP,
            format!("ddof = {ddof} leaves no degrees of freedom"),
        ));
    }
    let df = k - 1 - ddof;
    let statistic = observed
        .iter()
        .zip(&expected)
        .fold(Q::zero(), |acc, (o, e)| {
            let d = o - e;
            acc + &d * &d / e
        });
    Ok(ChiSquareResult {
        p_value: chi_squared_sf_q(ctx, df, &statistic),
        statistic,
        df,
        expected: vec![expected],
    })
}

/// The likelihood-ratio (G) test of independence on an `r × c` table:
/// `G = 2 Σ Oᵢⱼ ln(Oᵢⱼ/Eᵢⱼ)` (cells with `O = 0` contribute `0`), an exact
/// expression, with `P(χ²_{(r−1)(c−1)} ≥ G)`.  No continuity correction:
/// `scipy.stats.chi2_contingency(table, correction=False, lambda_='log-likelihood')`.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::hypothesis::{counts, g_test};
///
/// let ctx = Context::new();
/// let t = counts(&[&[10, 20, 30], &[6, 9, 17]]);
/// // scipy: chi2_contingency(t, correction=False, lambda_='log-likelihood')
/// //        → statistic 0.2740265420246606, pvalue 0.8719586542812721
/// let r = g_test(&ctx, &t)?;
/// assert!((r.statistic_f64()? - 0.274_026_542_024_660_6).abs() < 1e-12);
/// assert!((r.p_value_f64()? - 0.871_958_654_281_272_1).abs() < 1e-12);
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// As [`chi_square_independence`].
pub fn g_test(ctx: &Context, table: &[Vec<Q>]) -> Result<TestResult, SymplexError> {
    const OP: &str = "g_test";
    let (r, c) = check_table(OP, table)?;
    if r < 2 || c < 2 {
        return Err(invalid(
            OP,
            "the table needs at least two rows and two columns",
        ));
    }
    let expected = expected_of(OP, table)?.cells;
    let df = (r - 1) * (c - 1);
    let mut terms: Vec<Ex> = Vec::new();
    for (o_row, e_row) in table.iter().zip(&expected) {
        for (o, e) in o_row.iter().zip(e_row) {
            if o.is_zero() || o == e {
                continue;
            }
            terms.push(ex(ctx, o) * (ex(ctx, o) / ex(ctx, e)).ln());
        }
    }
    if terms.is_empty() {
        return Ok(TestResult {
            statistic: ctx.zero(),
            p_value: ctx.one(),
            df: Some(ctx.int(df as i64)),
            alternative: Alternative::TwoSided,
        });
    }
    let sum = terms.into_iter().fold(ctx.zero(), |acc, t| acc + t);
    let statistic = ctx.int(2) * sum;
    Ok(TestResult {
        p_value: chi_squared_sf(ctx, df, &statistic),
        statistic,
        df: Some(ctx.int(df as i64)),
        alternative: Alternative::TwoSided,
    })
}

/// Cramér's `V = √(χ² / (N · min(r − 1, c − 1)))` of an `r × c` table
/// (without Yates' correction), as an exact expression.
/// `scipy.stats.contingency.association(table, method='cramer', correction=False)`.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::hypothesis::{counts, cramers_v};
///
/// let ctx = Context::new();
/// // scipy: association([[10, 20, 30], [6, 9, 17]], method='cramer', correction=False) = 0.05433137570422292
/// let v = cramers_v(&ctx, &counts(&[&[10, 20, 30], &[6, 9, 17]]))?;
/// assert!((v.eval_f64()? - 0.054_331_375_704_222_92).abs() < 1e-12);
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// As [`chi_square_independence`].
pub fn cramers_v(ctx: &Context, table: &[Vec<Q>]) -> Result<Ex, SymplexError> {
    let r = chi_square_independence(ctx, table, false)?;
    let (rows, cols) = (table.len(), table[0].len());
    let (_, _, total) = margins(table);
    let k = qu((rows - 1).min(cols - 1));
    Ok(ex(ctx, &(r.statistic / (total * k))).sqrt().simplify())
}

fn check_2x2_margins(op: &'static str, table: [[usize; 2]; 2]) -> Result<(), SymplexError> {
    let [[a, b], [c, d]] = table;
    let empty = |x: usize, y: usize| x == 0 && y == 0;
    if empty(a, b) || empty(c, d) || empty(a, c) || empty(b, d) {
        return Err(invalid(op, "a row or a column of the table is empty"));
    }
    Ok(())
}

/// The φ coefficient of a 2×2 table `[[a, b], [c, d]]`:
/// `(ad − bc) / √((a+b)(c+d)(a+c)(b+d))`, as an exact expression (Pearson's
/// `r` of the two indicator variables; `φ² = χ²/N` without correction).
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::hypothesis::phi_coefficient;
///
/// let ctx = Context::new();
/// // numpy: corrcoef of the indicator vectors of [[20, 10], [5, 15]] = 0.40824829046386296
/// let phi = phi_coefficient(&ctx, [[20, 10], [5, 15]])?;
/// assert!((phi.eval_f64()? - 0.408_248_290_463_862_96).abs() < 1e-12);
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for an empty row or column.
pub fn phi_coefficient(ctx: &Context, table: [[usize; 2]; 2]) -> Result<Ex, SymplexError> {
    const OP: &str = "phi_coefficient";
    check_2x2_margins(OP, table)?;
    let [[a, b], [c, d]] = table.map(|row| row.map(qu));
    let num = &a * &d - &b * &c;
    let den = (&a + &b) * (&c + &d) * (&a + &c) * (&b + &d);
    Ok((ex(ctx, &num) / ex(ctx, &den).sqrt()).simplify())
}

/// Wald interval `exp(ln θ̂ ± z_{1−α/2} · se)`.
fn log_wald_ci(estimate: &Q, se: f64, confidence: f64) -> Interval<f64> {
    let z = norm_isf((1.0 - confidence) / 2.0);
    let log = q_to_f64(estimate).ln();
    Interval::closed((log - z * se).exp(), (log + z * se).exp())
}

/// The sample odds ratio `ad/(bc)` of a 2×2 table `[[a, b], [c, d]]` with
/// the log-scale Wald interval `exp(ln OR ± z √(1/a + 1/b + 1/c + 1/d))`.
/// `statsmodels.stats.contingency_tables.Table2x2(t).oddsratio` /
/// `.oddsratio_confint(alpha)`.
///
/// ```
/// use symplex::linprog::q;
/// use symplex::stats::hypothesis::odds_ratio;
///
/// // statsmodels: Table2x2([[20, 10], [5, 15]]).oddsratio = 6.0,
/// //              .oddsratio_confint(0.05) = (1.6931795592741443, 21.261773332199304)
/// let r = odds_ratio([[20, 10], [5, 15]], 0.95)?;
/// assert_eq!(r.estimate, q(6, 1));
/// assert!((r.ci.lower - 1.693_179_559_274_144_3).abs() < 1e-9);
/// assert!((r.ci.upper - 21.261_773_332_199_304).abs() < 1e-9);
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] if any cell is zero (the estimate or
/// its standard error is infinite) or `confidence ∉ (0, 1)`.
pub fn odds_ratio(table: [[usize; 2]; 2], confidence: f64) -> Result<RatioEstimate, SymplexError> {
    const OP: &str = "odds_ratio";
    check_confidence(OP, confidence)?;
    let [[a, b], [c, d]] = table;
    if a == 0 || b == 0 || c == 0 || d == 0 {
        return Err(invalid(
            OP,
            "every cell must be positive for a finite odds ratio",
        ));
    }
    let estimate = qu(a) * qu(d) / (qu(b) * qu(c));
    let se = (1.0 / a as f64 + 1.0 / b as f64 + 1.0 / c as f64 + 1.0 / d as f64).sqrt();
    Ok(RatioEstimate {
        ci: log_wald_ci(&estimate, se, confidence),
        estimate,
        confidence,
    })
}

/// The relative risk of a 2×2 table `[[exposed cases, exposed non-cases],
/// [control cases, control non-cases]]`: `(a/(a+b)) / (c/(c+d))` with the
/// log-scale Wald interval `exp(ln RR ± z √(1/a − 1/(a+b) + 1/c − 1/(c+d)))`.
/// `scipy.stats.contingency.relative_risk(a, a+b, c, c+d)` and its
/// `.confidence_interval(confidence)`.  The variance is formed exactly as
/// `b/(a(a+b)) + d/(c(c+d))` and rounded once (scipy subtracts the
/// reciprocals in floating point, which loses digits when `b ≪ a`).
///
/// ```
/// use symplex::linprog::q;
/// use symplex::stats::hypothesis::relative_risk;
///
/// // scipy: relative_risk(20, 30, 5, 20).relative_risk = 2.6666666666666665,
/// //        .confidence_interval(0.95) = (1.198028521436089, 5.935677643623166)
/// let r = relative_risk([[20, 10], [5, 15]], 0.95)?;
/// assert_eq!(r.estimate, q(8, 3));
/// assert!((r.ci.lower - 1.198_028_521_436_089).abs() < 1e-9);
/// assert!((r.ci.upper - 5.935_677_643_623_166).abs() < 1e-9);
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] if either case count is zero, a row
/// total overflows `usize`, or `confidence ∉ (0, 1)`.
pub fn relative_risk(
    table: [[usize; 2]; 2],
    confidence: f64,
) -> Result<RatioEstimate, SymplexError> {
    const OP: &str = "relative_risk";
    check_confidence(OP, confidence)?;
    let [[a, b], [c, d]] = table;
    let (Some(n1), Some(n2)) = (a.checked_add(b), c.checked_add(d)) else {
        return Err(invalid(OP, "a row total overflows usize"));
    };
    if a == 0 || c == 0 {
        return Err(invalid(
            OP,
            "both case counts must be positive for a finite relative risk",
        ));
    }
    let estimate = (qu(a) / qu(n1)) / (qu(c) / qu(n2));
    let var = qu(b) / (qu(a) * qu(n1)) + qu(d) / (qu(c) * qu(n2));
    let se = q_to_f64(&var).sqrt();
    Ok(RatioEstimate {
        ci: log_wald_ci(&estimate, se, confidence),
        estimate,
        confidence,
    })
}

/// Cohen's `h = 2 asin √p₁ − 2 asin √p₂`, the effect size of a difference
/// of proportions, as an exact expression.
/// `statsmodels.stats.proportion.proportion_effectsize(p1, p2)`.
///
/// The expression is the difference folded into one arctangent,
///
/// `h = 2 atan((p₁ − p₂) / (√(p₁(1 − p₁)) + √(p₂(1 − p₂))))`
///
/// (with `A = asin √p₁`, `B = asin √p₂`: `tan(A − B) = (√(p₁q₂) −
/// √(p₂q₁))/(√(q₁q₂) + √(p₁p₂))`, `q = 1 − p`, and multiplying both by
/// `√(p₁q₂) + √(p₂q₁)` leaves the form above; `A − B ∈ [−π/2, π/2]`), so
/// nothing cancels — `p₁ − p₂` is an exact rational, the denominator a
/// sum of non-negative roots — and the arctangent is well conditioned even
/// where `h` is near `±π`.  (Before 0.27 the two arcsines were
/// subtracted: `cohens_h(½, ½ + 10⁻¹⁰⁰)` evaluated to `0`, and
/// `cohens_h(1 − 10⁻³⁰, 1)` failed with `PrecisionExhausted`, an arcsine
/// of an argument within `10⁻³⁰` of `1`.)  Equal proportions give an
/// exact `0`, `(0, 1)` and `(1, 0)` give `∓π`.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::linprog::q;
/// use symplex::stats::hypothesis::cohens_h;
///
/// let ctx = Context::new();
/// // statsmodels: proportion_effectsize(0.5, 0.4) = 0.20135792079033088
/// let h = cohens_h(&ctx, &q(1, 2), &q(2, 5))?;
/// assert!((h.eval_f64()? - 0.201_357_920_790_330_88).abs() < 1e-12);
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for a proportion outside `[0, 1]`.
pub fn cohens_h(ctx: &Context, p1: &Q, p2: &Q) -> Result<Ex, SymplexError> {
    const OP: &str = "cohens_h";
    for p in [p1, p2] {
        if p.is_negative() || *p > Q::one() {
            return Err(invalid(OP, "proportions must lie in [0, 1]"));
        }
    }
    if p1 == p2 {
        return Ok(ctx.zero());
    }
    let spread = |p: &Q| p * (Q::one() - p);
    let (s1, s2) = (spread(p1), spread(p2));
    if s1.is_zero() && s2.is_zero() {
        // {p₁, p₂} = {0, 1}: A − B = ±π/2.
        let pi = ctx.pi();
        return Ok(if p1 > p2 { pi } else { -pi });
    }
    let tangent = ex(ctx, &(p1 - p2)) / (ex(ctx, &s1).sqrt() + ex(ctx, &s2).sqrt());
    Ok(ctx.int(2) * tangent.atan())
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Means and proportions
// ═══════════════════════════════════════════════════════════════════════════

fn student_result(ctx: &Context, df: &Q, stat: &RootRatio, alt: Alternative) -> TestResult {
    TestResult {
        statistic: stat.to_ex(ctx),
        p_value: student_p_value(ctx, df, stat, alt),
        df: Some(ex(ctx, df)),
        alternative: alt,
    }
}

fn normal_result(ctx: &Context, stat: &RootRatio, alt: Alternative) -> TestResult {
    TestResult {
        statistic: stat.to_ex(ctx),
        p_value: normal_p_value(ctx, stat, alt),
        df: None,
        alternative: alt,
    }
}

/// One-sample Student t-test of `mean = μ₀`: `t = (x̄ − μ₀) / (s/√n)` with
/// the sample standard deviation `s`, `df = n − 1`.  The statistic is exact
/// (`(x̄ − μ₀)√n / √s²`); the p-value is the exact Student-t tail
/// `I_{ν/(t²+ν)}(ν/2, ½)` (two-sided) or half of it / its complement.
/// `scipy.stats.ttest_1samp(x, popmean, alternative)`.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::linprog::qi;
/// use symplex::stats::data::from_i64;
/// use symplex::stats::hypothesis::{t_test_one_sample, Alternative};
///
/// let ctx = Context::new();
/// let x = from_i64(&[5, 7, 8, 9, 10, 12]);
/// // scipy: ttest_1samp(x, 6, alternative='greater') → statistic 2.521097420448054, pvalue 0.026551745875898917
/// let r = t_test_one_sample(&ctx, &x, &qi(6), Alternative::Greater)?;
/// assert!((r.statistic_f64()? - 2.521_097_420_448_054).abs() < 1e-12);
/// assert!((r.p_value_f64()? - 0.026_551_745_875_898_917).abs() < 1e-12);
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for fewer than two observations or a
/// constant sample (zero variance).
pub fn t_test_one_sample(
    ctx: &Context,
    x: &[Q],
    mu0: &Q,
    alt: Alternative,
) -> Result<TestResult, SymplexError> {
    const OP: &str = "t_test_one_sample";
    check_sample(OP, "the sample", x, 2)?;
    let n = x.len();
    let var = data::variance(x, Ddof::Sample)?;
    if var.is_zero() {
        return Err(invalid(OP, "the sample is constant (zero variance)"));
    }
    let stat = RootRatio {
        num: data::mean(x)? - mu0,
        var: var / qu(n),
    };
    Ok(student_result(ctx, &qu(n - 1), &stat, alt))
}

/// Two-sample t-test of `mean(x) = mean(y)`.  `equal_var`: Student's test
/// with the pooled variance `s²ₚ = ((n₁−1)s₁² + (n₂−1)s₂²)/(n₁+n₂−2)`,
/// `t = (x̄ − ȳ)/√(s²ₚ(1/n₁ + 1/n₂))`, `df = n₁ + n₂ − 2`.  Otherwise Welch's
/// test `t = (x̄ − ȳ)/√(s₁²/n₁ + s₂²/n₂)` with the Welch–Satterthwaite
/// `ν = (s₁²/n₁ + s₂²/n₂)² / ((s₁²/n₁)²/(n₁−1) + (s₂²/n₂)²/(n₂−1))`, an exact
/// rational.  `scipy.stats.ttest_ind(x, y, equal_var, alternative)`.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::linprog::q;
/// use symplex::stats::data::from_i64;
/// use symplex::stats::hypothesis::{t_test_two_sample, Alternative};
///
/// let ctx = Context::new();
/// let x = from_i64(&[20, 22, 19, 20, 22, 20, 21]);
/// let y = from_i64(&[28, 32, 36, 24, 29, 32]);
/// // scipy: ttest_ind(x, y, equal_var=False) → statistic -5.529270507777645,
/// //        pvalue 0.0017938799124542811, df 5.650760744239527 (exactly 3527433605/624240481)
/// let r = t_test_two_sample(&ctx, &x, &y, false, Alternative::TwoSided)?;
/// assert!((r.statistic_f64()? - -5.529_270_507_777_645).abs() < 1e-12);
/// assert!((r.p_value_f64()? - 0.001_793_879_912_454_281_1).abs() < 1e-12);
/// assert_eq!(r.df, Some(ctx.from_ratio(q(3_527_433_605, 624_240_481))));
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] if either sample has fewer than two
/// observations or both are constant.
pub fn t_test_two_sample(
    ctx: &Context,
    x: &[Q],
    y: &[Q],
    equal_var: bool,
    alt: Alternative,
) -> Result<TestResult, SymplexError> {
    const OP: &str = "t_test_two_sample";
    check_sample(OP, "the first sample", x, 2)?;
    check_sample(OP, "the second sample", y, 2)?;
    let (n1, n2) = (x.len(), y.len());
    let (v1, v2) = (
        data::variance(x, Ddof::Sample)?,
        data::variance(y, Ddof::Sample)?,
    );
    let num = data::mean(x)? - data::mean(y)?;
    if v1.is_zero() && v2.is_zero() {
        return Err(invalid(OP, "both samples are constant (zero variance)"));
    }
    if equal_var {
        let df = qu(n1 + n2 - 2);
        let pooled = (qu(n1 - 1) * &v1 + qu(n2 - 1) * &v2) / &df;
        let var = pooled * (qu(n1).recip() + qu(n2).recip());
        Ok(student_result(ctx, &df, &RootRatio { num, var }, alt))
    } else {
        let (a, b) = (&v1 / qu(n1), &v2 / qu(n2));
        let var = &a + &b;
        let df = &var * &var / (&a * &a / qu(n1 - 1) + &b * &b / qu(n2 - 1));
        Ok(student_result(ctx, &df, &RootRatio { num, var }, alt))
    }
}

/// Paired t-test: the one-sample test of the differences `xᵢ − yᵢ` against
/// `0`.  `scipy.stats.ttest_rel(x, y, alternative)`.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::data::from_i64;
/// use symplex::stats::hypothesis::{t_test_paired, Alternative};
///
/// let ctx = Context::new();
/// let before = from_i64(&[200, 190, 210, 180, 195, 205]);
/// let after = from_i64(&[190, 185, 200, 182, 190, 195]);
/// // scipy: ttest_rel(before, after) → statistic 3.258473117707668, pvalue 0.022483670687634263
/// let r = t_test_paired(&ctx, &before, &after, Alternative::TwoSided)?;
/// assert!((r.statistic_f64()? - 3.258_473_117_707_668).abs() < 1e-12);
/// assert!((r.p_value_f64()? - 0.022_483_670_687_634_263).abs() < 1e-12);
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for unequal lengths, fewer than two
/// pairs, or constant differences.
pub fn t_test_paired(
    ctx: &Context,
    x: &[Q],
    y: &[Q],
    alt: Alternative,
) -> Result<TestResult, SymplexError> {
    const OP: &str = "t_test_paired";
    check_same_len(OP, x, y)?;
    let d: Vec<Q> = x.iter().zip(y).map(|(a, b)| a - b).collect();
    check_sample(OP, "the paired sample", &d, 2)?;
    if data::variance(&d, Ddof::Sample)?.is_zero() {
        return Err(invalid(OP, "the differences are constant (zero variance)"));
    }
    t_test_one_sample(ctx, &d, &Q::zero(), alt)
}

/// One-proportion z-test of `p = p₀` from `k` successes in `n` trials:
/// `z = (k/n − p₀) / √(p₀(1 − p₀)/n)` (the null variance), with the normal
/// tail `½ erfc(z/√2)`.
/// `statsmodels.stats.proportion.proportions_ztest(k, n, value=p0, prop_var=p0)`.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::linprog::q;
/// use symplex::stats::hypothesis::{z_test_proportion, Alternative};
///
/// let ctx = Context::new();
/// // statsmodels: proportions_ztest(60, 100, value=0.5, prop_var=0.5) = (2.0, 0.04550026389635844)
/// let r = z_test_proportion(&ctx, 60, 100, &q(1, 2), Alternative::TwoSided)?;
/// assert_eq!(r.statistic, ctx.int(2));
/// assert!((r.p_value_f64()? - 0.045_500_263_896_358_44).abs() < 1e-12);
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for `n = 0`, `k > n` or `p₀ ∉ (0, 1)`.
pub fn z_test_proportion(
    ctx: &Context,
    k: usize,
    n: usize,
    p0: &Q,
    alt: Alternative,
) -> Result<TestResult, SymplexError> {
    const OP: &str = "z_test_proportion";
    if n == 0 {
        return Err(invalid(OP, "the number of trials must be positive"));
    }
    if k > n {
        return Err(invalid(OP, format!("{k} successes exceed {n} trials")));
    }
    if !p0.is_positive() || *p0 >= Q::one() {
        return Err(invalid(
            OP,
            "the null proportion must lie strictly between 0 and 1",
        ));
    }
    let stat = RootRatio {
        num: qu(k) / qu(n) - p0,
        var: p0 * (Q::one() - p0) / qu(n),
    };
    Ok(normal_result(ctx, &stat, alt))
}

/// Two-proportion z-test of `p₁ = p₂` with the pooled estimate
/// `p̂ = (k₁ + k₂)/(n₁ + n₂)`: `z = (k₁/n₁ − k₂/n₂) / √(p̂(1 − p̂)(1/n₁ + 1/n₂))`.
/// `statsmodels.stats.proportion.proportions_ztest([k1, k2], [n1, n2])`.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::hypothesis::{z_test_two_proportions, Alternative};
///
/// let ctx = Context::new();
/// // statsmodels: proportions_ztest([45, 30], [100, 100]) = (2.1908902300206647, 0.028459736916310555)
/// let r = z_test_two_proportions(&ctx, 45, 100, 30, 100, Alternative::TwoSided)?;
/// assert!((r.statistic_f64()? - 2.190_890_230_020_664_7).abs() < 1e-12);
/// assert!((r.p_value_f64()? - 0.028_459_736_916_310_555).abs() < 1e-12);
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for an empty sample, `k > n`, or a
/// pooled proportion of `0` or `1` (zero variance).
pub fn z_test_two_proportions(
    ctx: &Context,
    k1: usize,
    n1: usize,
    k2: usize,
    n2: usize,
    alt: Alternative,
) -> Result<TestResult, SymplexError> {
    const OP: &str = "z_test_two_proportions";
    if n1 == 0 || n2 == 0 {
        return Err(invalid(OP, "both samples must be non-empty"));
    }
    if k1 > n1 || k2 > n2 {
        return Err(invalid(OP, "successes exceed trials"));
    }
    let pooled = (qu(k1) + qu(k2)) / (qu(n1) + qu(n2));
    if pooled.is_zero() || pooled.is_one() {
        return Err(invalid(
            OP,
            "the pooled proportion is 0 or 1 (zero variance)",
        ));
    }
    let stat = RootRatio {
        num: qu(k1) / qu(n1) - qu(k2) / qu(n2),
        var: &pooled * (Q::one() - &pooled) * (qu(n1).recip() + qu(n2).recip()),
    };
    Ok(normal_result(ctx, &stat, alt))
}

// ═══════════════════════════════════════════════════════════════════════════
// 3b. Inference on Pearson's r
// ═══════════════════════════════════════════════════════════════════════════

/// The population sums of squares and cross-products of a pair.
struct CrossMoments {
    sxy: Q,
    sxx: Q,
    syy: Q,
}

/// The cross-moments of a pair, erroring on a constant sample.
fn cross_moments(
    op: &'static str,
    x: &[Q],
    y: &[Q],
    min_n: usize,
) -> Result<CrossMoments, SymplexError> {
    check_same_len(op, x, y)?;
    if x.len() < min_n {
        return Err(invalid(
            op,
            format!(
                "needs at least {min_n} paired observations, got {}",
                x.len()
            ),
        ));
    }
    let sxy = data::covariance(x, y, Ddof::Population)?;
    let sxx = data::variance(x, Ddof::Population)?;
    let syy = data::variance(y, Ddof::Population)?;
    if sxx.is_zero() || syy.is_zero() {
        return Err(invalid(op, "a constant sample has no correlation"));
    }
    Ok(CrossMoments { sxy, sxx, syy })
}

/// The t statistic of Pearson's `r`: `t = r √(n − 2) / √(1 − r²)`, as an
/// exact expression (`t² = r²(n−2)/(1−r²)` is rational).
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::data::from_i64;
/// use symplex::stats::hypothesis::pearson_t_statistic;
///
/// let ctx = Context::new();
/// let x = from_i64(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
/// let y = from_i64(&[2, 1, 4, 3, 7, 8, 5, 6, 10, 9]);
/// // r = 13/15, t² = 169/7: t = 4.913538149119947
/// assert!((pearson_t_statistic(&ctx, &x, &y)?.eval_f64()? - 4.913_538_149_119_947).abs() < 1e-12);
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for unequal lengths, fewer than three
/// pairs, a constant sample, or `|r| = 1` (infinite `t`).
pub fn pearson_t_statistic(ctx: &Context, x: &[Q], y: &[Q]) -> Result<Ex, SymplexError> {
    const OP: &str = "pearson_t_statistic";
    let CrossMoments { sxy, sxx, syy } = cross_moments(OP, x, y, 3)?;
    let resid = &sxx * &syy - square(&sxy);
    if resid.is_zero() {
        return Err(invalid(OP, "|r| = 1, the t statistic is infinite"));
    }
    let var = resid / qu(x.len() - 2);
    Ok((ex(ctx, &sxy) / ex(ctx, &var).sqrt()).simplify())
}

/// The test of `H₀: ρ = 0` for Pearson's `r`: the statistic is `r`
/// ([`data::pearson`], exact); the p-value uses `t = r √((n−2)/(1−r²))`
/// with `n − 2` degrees of freedom (exact Student-t tail
/// `I_{ν/(t²+ν)}(ν/2, ½)`; `t²` is rational even when `r` is not).
/// `|r| = 1` gives `p = 0` in the alternative's direction.
/// `scipy.stats.pearsonr(x, y, alternative)`.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::linprog::q;
/// use symplex::stats::data::from_i64;
/// use symplex::stats::hypothesis::{Alternative, pearson_test};
///
/// let ctx = Context::new();
/// let x = from_i64(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
/// let y = from_i64(&[2, 1, 4, 3, 7, 8, 5, 6, 10, 9]);
/// // scipy: pearsonr(x, y) → statistic 0.866666666666666 (= 13/15), pvalue 0.001173538180155
/// let r = pearson_test(&ctx, &x, &y, Alternative::TwoSided)?;
/// assert_eq!(r.statistic, ctx.from_ratio(q(13, 15)));
/// assert_eq!(r.df, Some(ctx.int(8)));
/// assert!((r.p_value_f64()? - 0.001_173_538_180_155).abs() < 1e-12);
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for unequal lengths, fewer than three
/// pairs, or a constant sample.
pub fn pearson_test(
    ctx: &Context,
    x: &[Q],
    y: &[Q],
    alt: Alternative,
) -> Result<TestResult, SymplexError> {
    const OP: &str = "pearson_test";
    let CrossMoments { sxy, sxx, syy } = cross_moments(OP, x, y, 3)?;
    let r = (ex(ctx, &sxy) / (ex(ctx, &sxx) * ex(ctx, &syy)).sqrt()).simplify();
    let df = qu(x.len() - 2);
    let r2 = square(&sxy) / (&sxx * &syy);
    let one_minus = Q::one() - &r2;
    let tail = in_tail(sxy.is_negative(), sxy.is_positive(), alt);
    let p_value = if one_minus.is_zero() {
        // |r| = 1: t is infinite in the direction of sign(r).
        let extreme = match alt {
            Alternative::TwoSided => true,
            Alternative::Greater => sxy.is_positive(),
            Alternative::Less => sxy.is_negative(),
        };
        if extreme { ctx.zero() } else { ctx.one() }
    } else {
        // t² = r²(n−2)/(1−r²); z = ν/(t² + ν) = (1 − r²)/((1 − r²) + r²) … kept as a rational.
        let t2 = &r2 * &df / &one_minus;
        let z = &df / (&t2 + &df);
        let two_sided = ex(ctx, &z).betainc_regularized(
            &(ex(ctx, &df) / ctx.int(2)),
            &ctx.rational(1, 2),
            &ctx.zero(),
        );
        one_sided_from_symmetric(ctx, two_sided, tail, alt)
    };
    Ok(TestResult {
        statistic: r,
        p_value,
        df: Some(ex(ctx, &df)),
        alternative: alt,
    })
}

/// The test that two independent samples' correlations are equal
/// (Fisher's z):
///
/// `z = (atanh r₁ − atanh r₂) / √(1/(n₁ − 3) + 1/(n₂ − 3))`,
///
/// referred to the standard normal; `Greater` tests `ρ₁ > ρ₂`.  The
/// statistic and the `erfc` p-value are expressions in the (dyadic-exact)
/// inputs.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::hypothesis::{Alternative, compare_two_correlations};
///
/// let ctx = Context::new();
/// // scipy.stats.norm: z = 2.251706268343729, two-sided p = 0.024340840282246236
/// let r = compare_two_correlations(&ctx, 0.7, 50, 0.4, 60, Alternative::TwoSided)?;
/// assert!((r.statistic_f64()? - 2.251_706_268_343_729).abs() < 1e-12);
/// assert!((r.p_value_f64()? - 0.024_340_840_282_246_236).abs() < 1e-12);
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for `|r| ≥ 1`, a non-finite `r`, or
/// a sample of fewer than four observations.
pub fn compare_two_correlations(
    ctx: &Context,
    r1: f64,
    n1: usize,
    r2: f64,
    n2: usize,
    alt: Alternative,
) -> Result<TestResult, SymplexError> {
    const OP: &str = "compare_two_correlations";
    for (name, r, n) in [("first", r1, n1), ("second", r2, n2)] {
        if !r.is_finite() || r.abs() >= 1.0 {
            return Err(invalid(
                OP,
                format!("the {name} correlation must lie strictly between −1 and 1, got {r}"),
            ));
        }
        if n < 4 {
            return Err(invalid(
                OP,
                format!("the {name} sample needs at least four observations, got {n}"),
            ));
        }
    }
    let diff = ctx.from_f64(r1)?.atanh() - ctx.from_f64(r2)?.atanh();
    let var = Q::one() / qu(n1 - 3) + Q::one() / qu(n2 - 3);
    let statistic = diff / ex(ctx, &var).sqrt();
    let two_sided = (statistic.abs() / ctx.int(2).sqrt()).erfc();
    let sign = r1.atanh() - r2.atanh();
    let tail = in_tail(sign < 0.0, sign > 0.0, alt);
    Ok(TestResult {
        statistic,
        p_value: one_sided_from_symmetric(ctx, two_sided, tail, alt),
        df: None,
        alternative: alt,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Rank tests
// ═══════════════════════════════════════════════════════════════════════════

/// `Σ (t³ − t)` over tie-group sizes `t` — the tie term of the rank-test
/// variances and corrections (Mann–Whitney, Wilcoxon, Kruskal–Wallis,
/// Friedman) — exact in `Q`: `t³` overflows `usize` from `t ≈ 2.6·10⁶`.
/// Sizes below `2` contribute `0`.  Pair with [`data::tie_sizes`].
///
/// ```
/// use symplex::linprog::qi;
/// use symplex::stats::hypothesis::tie_term;
///
/// // two groups of 2 and one of 3: (8 − 2) + (8 − 2) + (27 − 3) = 36
/// assert_eq!(tie_term(&[2, 2, 3]), qi(36));
/// ```
pub fn tie_term(tie_sizes: &[usize]) -> Q {
    tie_sizes
        .iter()
        .fold(Q::zero(), |acc, &t| acc + cubic_minus_linear(t))
}

/// [`tie_term`] of a sample's tie groups.
fn tie_term_of(x: &[Q]) -> Q {
    tie_term(&data::tie_sizes(x))
}

/// `n³ − n` exactly.
fn cubic_minus_linear(n: usize) -> Q {
    let n = qu(n);
    &n * &n * &n - &n
}

/// `a · b` of two counts, exactly.
fn qmul(a: usize, b: usize) -> Q {
    qu(a) * qu(b)
}

/// `n(n + 1)`, exactly.
fn pronic(n: usize) -> Q {
    qmul(n, n + 1)
}

/// Frequencies of `U = u`, `u = 0..=mn`, over the `C(m+n, m)` equally likely
/// arrangements of `m` and `n` distinct values: the coefficients of the
/// Gaussian binomial `[m+n choose m]_q = Π_{i=1}^{m} (1 − q^{n+i})/(1 − q^i)`.
/// `None` when `mn + 1` does not fit in `usize`.
fn mann_whitney_frequencies(m: usize, n: usize) -> Option<Vec<BigInt>> {
    let size = m.checked_mul(n)?.checked_add(1)?;
    let mut c = vec![BigInt::zero(); size];
    c[0] = BigInt::one();
    for i in 1..=m {
        let shift = n + i;
        for t in (shift..size).rev() {
            let (lo, hi) = c.split_at_mut(t);
            hi[0] -= &lo[t - shift];
        }
        for t in i..size {
            let (lo, hi) = c.split_at_mut(t);
            hi[0] += &lo[t - i];
        }
    }
    Some(c)
}

/// Frequencies of `T⁺ = s`, `s = 0..=n(n+1)/2`, over the `2ⁿ` equally likely
/// sign patterns of the ranks `1..=n` (the number of subsets with sum `s`).
/// `None` when `n(n+1)/2 + 1` does not fit in `usize`.
fn signed_rank_frequencies(n: usize) -> Option<Vec<BigInt>> {
    let size = n.checked_mul(n + 1)?.checked_div(2)?.checked_add(1)?;
    let mut c = vec![BigInt::zero(); size];
    c[0] = BigInt::one();
    for k in 1..=n {
        for s in (k..size).rev() {
            let (lo, hi) = c.split_at_mut(s);
            hi[0] += &lo[s - k];
        }
    }
    Some(c)
}

/// The error for an exact rank distribution whose table would not fit.
fn exact_table_too_large(op: &'static str) -> SymplexError {
    invalid(op, "the exact distribution's table is too large for usize")
}

/// Mahonian numbers `M(n, k)`, `k = 0..=cmax`: permutations of `n` with `k`
/// inversions (Kendall, "Rank Correlation Methods", ch. 5).
fn inversion_counts(n: usize, cmax: usize) -> Vec<BigInt> {
    let mut c = vec![BigInt::zero(); cmax + 1];
    c[0] = BigInt::one();
    for j in 2..=n {
        // In place: prefix sums S(k) = Σ_{t ≤ k} M(j − 1, t), then
        // M(j, k) = S(k) − S(k − j), descending so that S(k − j) is intact.
        for t in 1..=cmax {
            let (lo, hi) = c.split_at_mut(t);
            hi[0] += &lo[t - 1];
        }
        for k in (j..=cmax).rev() {
            let (lo, hi) = c.split_at_mut(k);
            hi[0] -= &lo[k - j];
        }
    }
    c
}

/// `Σ_{k ≤ upto} freq[k] / total`.
fn cdf_of(freq: &[BigInt], total: &BigInt, upto: usize) -> Q {
    let upto = upto.min(freq.len().saturating_sub(1));
    Q::new(sum_big(&freq[..=upto]), total.clone())
}

/// `Σ_{k ≥ from} freq[k] / total`.
fn sf_of(freq: &[BigInt], total: &BigInt, from: usize) -> Q {
    if from >= freq.len() {
        return Q::zero();
    }
    Q::new(sum_big(&freq[from..]), total.clone())
}

fn rank_statistic_to_index(op: &'static str, v: &Q) -> Result<usize, SymplexError> {
    if !v.is_integer() || v.is_negative() {
        return Err(invalid(
            op,
            "the exact distribution needs an integer statistic",
        ));
    }
    v.to_integer()
        .to_usize()
        .ok_or_else(|| invalid(op, "the statistic is too large"))
}

/// Continuity correction of a discrete statistic compared with a normal:
/// move `num` half a unit towards zero in the direction of the
/// alternative (`sign(num)` two-sided, `+1` greater, `−1` less).
fn continuity_shift(num: &Q, alt: Alternative) -> Q {
    let half = Q::new(BigInt::one(), BigInt::from(2));
    match alt {
        Alternative::Greater => num - half,
        Alternative::Less => num + half,
        Alternative::TwoSided => match num.cmp(&Q::zero()) {
            Ordering::Greater => num - half,
            Ordering::Less => num + half,
            Ordering::Equal => num.clone(),
        },
    }
}

/// Mann–Whitney U test of two independent samples.  With the ranks of the
/// pooled sample, `U₁ = R₁ − n₁(n₁+1)/2` is the statistic (for `x`; `U₂ =
/// n₁n₂ − U₁`).  `Exact`: the p-value is `P(U ≥ U₁)` (`Greater`), `P(U ≥
/// U₂)` (`Less`) or `min(1, 2 P(U ≥ max(U₁, U₂)))` from the exact
/// distribution of `U` (no ties allowed), an exact rational.  `Asymptotic`:
/// `z = (U₁ − n₁n₂/2 ∓ ½) / σ` with `σ² = n₁n₂/12 · ((N+1) − Σ(t³−t)/(N(N−1)))`
/// (tie correction) and the normal tail.
/// `scipy.stats.mannwhitneyu(x, y, use_continuity, alternative, method)`.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::linprog::{q, qi};
/// use symplex::stats::data::from_i64;
/// use symplex::stats::hypothesis::{mann_whitney_u, Alternative, RankMethod};
///
/// let ctx = Context::new();
/// let males = from_i64(&[19, 22, 16, 29, 24]);
/// let females = from_i64(&[20, 11, 17, 12]);
/// // scipy: mannwhitneyu(males, females, method='exact') = (17.0, 0.1111111111111111)
/// let r = mann_whitney_u(&ctx, &males, &females, Alternative::TwoSided, RankMethod::Exact)?;
/// assert_eq!(r.statistic_exact(), Some(qi(17)));
/// assert_eq!(r.p_value_exact(), Some(q(1, 9)));
/// // scipy: mannwhitneyu(males, females, method='asymptotic') → pvalue 0.11134688653314039
/// let r = mann_whitney_u(&ctx, &males, &females, Alternative::TwoSided, RankMethod::Asymptotic { continuity: true })?;
/// assert!((r.p_value_f64()? - 0.111_346_886_533_140_39).abs() < 1e-12);
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for an empty sample, ties with
/// `Exact`, or a pooled sample that is constant (`Asymptotic`).
pub fn mann_whitney_u(
    ctx: &Context,
    x: &[Q],
    y: &[Q],
    alt: Alternative,
    method: RankMethod,
) -> Result<TestResult, SymplexError> {
    const OP: &str = "mann_whitney_u";
    check_sample(OP, "the first sample", x, 1)?;
    check_sample(OP, "the second sample", y, 1)?;
    let (n1, n2) = (x.len(), y.len());
    let pooled: Vec<Q> = x.iter().chain(y).cloned().collect();
    let ranks = data::ranks(&pooled);
    let r1 = data::sum(&ranks[..n1]);
    let u1 = r1 - pronic(n1) / qi(2);
    let n1n2 = qmul(n1, n2);
    let u2 = &n1n2 - &u1;
    let statistic = ex(ctx, &u1);
    match method {
        RankMethod::Exact => {
            if !data::tie_sizes(&pooled).is_empty() {
                return Err(invalid(
                    OP,
                    "the exact method needs a pooled sample without ties",
                ));
            }
            let u = match alt {
                Alternative::Greater => &u1,
                Alternative::Less => &u2,
                Alternative::TwoSided => (&u1).max(&u2),
            };
            let k = rank_statistic_to_index(OP, u)?;
            let freq = mann_whitney_frequencies(n1, n2).ok_or_else(|| exact_table_too_large(OP))?;
            let total = sum_big(&freq);
            let sf = sf_of(&freq, &total, k);
            let p = match alt {
                Alternative::TwoSided => (sf * qi(2)).min(Q::one()),
                _ => sf,
            };
            Ok(result(ctx, statistic, p, None, alt))
        }
        RankMethod::Asymptotic { continuity } => {
            let n = n1 + n2;
            let var = &n1n2 / qi(12) * (qu(n + 1) - tie_term_of(&pooled) / qmul(n, n - 1));
            if !var.is_positive() {
                return Err(invalid(OP, "every observation is identical"));
            }
            let mut num = &u1 - &n1n2 / qi(2);
            if continuity {
                num = continuity_shift(&num, alt);
            }
            Ok(TestResult {
                statistic,
                p_value: normal_p_value(ctx, &RootRatio { num, var }, alt),
                df: None,
                alternative: alt,
            })
        }
    }
}

/// Wilcoxon signed-rank test of the differences `dᵢ = xᵢ − yᵢ` (or of `x`
/// alone) against a symmetric distribution about `0`.  Zero differences
/// are dropped (`zero_method='wilcox'`); `|d|` is ranked with average
/// ranks; `T⁺`, `T⁻` are the rank sums of the positive and negative
/// differences.  The statistic is `min(T⁺, T⁻)` two-sided and `T⁺`
/// one-sided.  `Exact` (no ties in `|d|`): `P(T⁺ ≤ T⁺)` (`Less`), `P(T⁺ ≥
/// T⁺)` (`Greater`), `min(1, 2 min(cdf, sf))` two-sided, from the `2ⁿ` sign
/// patterns, an exact rational.  `Asymptotic`: `z = (T⁺ − n(n+1)/4 ∓ ½)/σ`
/// with `σ² = (n(n+1)(2n+1) − Σ(t³−t)/2)/24` and the normal tail.
/// `scipy.stats.wilcoxon(x, y, correction, alternative, method)`.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::linprog::{q, qi};
/// use symplex::stats::data::from_i64;
/// use symplex::stats::hypothesis::{wilcoxon_signed_rank, Alternative, RankMethod};
///
/// let ctx = Context::new();
/// let x = from_i64(&[125, 115, 130, 140, 140, 115, 140, 125, 140, 135]);
/// let y = from_i64(&[110, 122, 125, 120, 140, 124, 123, 137, 134, 145]);
/// // scipy: wilcoxon(x, y, method='exact') = (18.0, 0.65234375)   (one zero difference dropped; 0.65234375 = 167/256)
/// let r = wilcoxon_signed_rank(&ctx, &x, Some(&y), Alternative::TwoSided, RankMethod::Exact)?;
/// assert_eq!(r.statistic_exact(), Some(qi(18)));
/// assert_eq!(r.p_value_exact(), Some(q(167, 256)));
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for unequal lengths, no non-zero
/// differences, or ties in `|d|` with `Exact`.
pub fn wilcoxon_signed_rank(
    ctx: &Context,
    x: &[Q],
    y: Option<&[Q]>,
    alt: Alternative,
    method: RankMethod,
) -> Result<TestResult, SymplexError> {
    const OP: &str = "wilcoxon_signed_rank";
    let d: Vec<Q> = match y {
        Some(y) => {
            check_same_len(OP, x, y)?;
            x.iter().zip(y).map(|(a, b)| a - b).collect()
        }
        None => x.to_vec(),
    };
    let d: Vec<Q> = d.into_iter().filter(|v| !v.is_zero()).collect();
    let n = d.len();
    if n == 0 {
        return Err(invalid(OP, "every difference is zero"));
    }
    let abs: Vec<Q> = d.iter().map(Signed::abs).collect();
    let ranks = data::ranks(&abs);
    let (mut r_plus, mut r_minus) = (Q::zero(), Q::zero());
    for (v, r) in d.iter().zip(&ranks) {
        if v.is_positive() {
            r_plus += r;
        } else {
            r_minus += r;
        }
    }
    let statistic = match alt {
        Alternative::TwoSided => ex(ctx, (&r_plus).min(&r_minus)),
        _ => ex(ctx, &r_plus),
    };
    match method {
        RankMethod::Exact => {
            if !data::tie_sizes(&abs).is_empty() {
                return Err(invalid(
                    OP,
                    "the exact method needs |differences| without ties",
                ));
            }
            let k = rank_statistic_to_index(OP, &r_plus)?;
            let freq = signed_rank_frequencies(n).ok_or_else(|| exact_table_too_large(OP))?;
            let total = BigInt::one() << n;
            let (cdf, sf) = (cdf_of(&freq, &total, k), sf_of(&freq, &total, k));
            let p = match alt {
                Alternative::Less => cdf,
                Alternative::Greater => sf,
                Alternative::TwoSided => (cdf.min(sf) * qi(2)).min(Q::one()),
            };
            Ok(result(ctx, statistic, p, None, alt))
        }
        RankMethod::Asymptotic { continuity } => {
            let var = (pronic(n) * (qi(2) * qu(n) + Q::one()) - tie_term_of(&abs) / qi(2)) / qi(24);
            if !var.is_positive() {
                return Err(invalid(OP, "the variance of the rank sum is zero"));
            }
            let mut num = &r_plus - pronic(n) / qi(4);
            if continuity {
                num = continuity_shift(&num, alt);
            }
            Ok(TestResult {
                statistic,
                p_value: normal_p_value(ctx, &RootRatio { num, var }, alt),
                df: None,
                alternative: alt,
            })
        }
    }
}

/// Kruskal–Wallis H test of `k` independent groups: with the pooled ranks,
/// `H = 12/(N(N+1)) Σ Rᵢ²/nᵢ − 3(N+1)`, divided by the tie correction
/// `1 − Σ(t³−t)/(N³−N)`; `H` is exact, `df = k − 1`, and the p-value is
/// `P(χ²_{k−1} ≥ H)`.  `scipy.stats.kruskal(*groups)`.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::linprog::q;
/// use symplex::stats::data::from_i64;
/// use symplex::stats::hypothesis::kruskal_wallis;
///
/// let ctx = Context::new();
/// let g = [from_i64(&[1, 3, 5, 7, 9]), from_i64(&[2, 4, 6, 8, 10]), from_i64(&[11, 12, 13, 14, 15])];
/// // scipy: kruskal(*g) → statistic 9.5, pvalue 0.008651695203120634
/// let r = kruskal_wallis(&ctx, &g)?;
/// assert_eq!(r.statistic_exact(), Some(q(19, 2)));
/// assert!((r.p_value_f64()? - 0.008_651_695_203_120_634).abs() < 1e-12);
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for fewer than two groups, an empty
/// group, or identical observations throughout.
pub fn kruskal_wallis(ctx: &Context, groups: &[Vec<Q>]) -> Result<TestResult, SymplexError> {
    const OP: &str = "kruskal_wallis";
    let k = groups.len();
    if k < 2 {
        return Err(invalid(OP, "at least two groups are needed"));
    }
    if groups.iter().any(Vec::is_empty) {
        return Err(invalid(OP, "every group must be non-empty"));
    }
    let pooled: Vec<Q> = groups.iter().flatten().cloned().collect();
    let n = pooled.len();
    let ranks = data::ranks(&pooled);
    let mut ssbn = Q::zero();
    let mut start = 0;
    for g in groups {
        let r = data::sum(&ranks[start..start + g.len()]);
        ssbn += &r * &r / qu(g.len());
        start += g.len();
    }
    let h = qi(12) / pronic(n) * ssbn - qi(3) * qu(n + 1);
    let correction = Q::one() - tie_term_of(&pooled) / cubic_minus_linear(n);
    if correction.is_zero() {
        return Err(invalid(OP, "every observation is identical"));
    }
    let h = h / correction;
    Ok(TestResult {
        p_value: chi_squared_sf_q(ctx, k - 1, &h),
        statistic: ex(ctx, &h),
        df: Some(ctx.int(usize_to_i64(OP, k - 1)?)),
        alternative: Alternative::TwoSided,
    })
}

/// Friedman's test of `k ≥ 3` treatments measured on `n` blocks (`blocks[i]`
/// holds the `k` measurements of block `i`).  Ranking within each block,
/// `χ²_F = (12/(nk(k+1)) Σ Rⱼ² − 3n(k+1)) / (1 − Σ_blocks Σ(t³−t)/(n(k³−k)))`
/// is exact, `df = k − 1`, and the p-value is `P(χ²_{k−1} ≥ χ²_F)`.
/// `scipy.stats.friedmanchisquare(*columns)` (which takes the treatments as
/// separate arrays — the transpose of `blocks`).
///
/// ```
/// use symplex::prelude::*;
/// use symplex::linprog::q;
/// use symplex::stats::data::from_i64;
/// use symplex::stats::hypothesis::friedman;
///
/// let ctx = Context::new();
/// let blocks = [from_i64(&[10, 9, 8]), from_i64(&[9, 8, 6]), from_i64(&[7, 5, 4]), from_i64(&[8, 6, 3])];
/// // scipy: friedmanchisquare([10, 9, 7, 8], [9, 8, 5, 6], [8, 6, 4, 3]) → statistic 8.0, pvalue 0.018315638888734182
/// let r = friedman(&ctx, &blocks)?;
/// assert_eq!(r.statistic_exact(), Some(q(8, 1)));
/// assert!((r.p_value_f64()? - 0.018_315_638_888_734_182).abs() < 1e-12);
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for fewer than three treatments, no
/// blocks, ragged blocks, or ties within every block throughout.
pub fn friedman(ctx: &Context, blocks: &[Vec<Q>]) -> Result<TestResult, SymplexError> {
    const OP: &str = "friedman";
    let n = blocks.len();
    if n == 0 {
        return Err(invalid(OP, "at least one block is needed"));
    }
    let k = blocks[0].len();
    if k < 3 {
        return Err(invalid(OP, "at least three treatments are needed"));
    }
    if blocks.iter().any(|b| b.len() != k) {
        return Err(invalid(
            OP,
            "every block must hold the same number of treatments",
        ));
    }
    let mut column_sums = vec![Q::zero(); k];
    let mut ties = Q::zero();
    for b in blocks {
        let r = data::ranks(b);
        for (s, v) in column_sums.iter_mut().zip(&r) {
            *s += v;
        }
        ties += tie_term_of(b);
    }
    let correction = Q::one() - ties / (cubic_minus_linear(k) * qu(n));
    if correction.is_zero() {
        return Err(invalid(OP, "every block is constant"));
    }
    let ssbn = column_sums.iter().fold(Q::zero(), |acc, r| acc + r * r);
    let nk1 = qmul(n, k + 1);
    let stat = (qi(12) / (&nk1 * qu(k)) * ssbn - qi(3) * &nk1) / correction;
    Ok(TestResult {
        p_value: chi_squared_sf_q(ctx, k - 1, &stat),
        statistic: ex(ctx, &stat),
        df: Some(ctx.int(usize_to_i64(OP, k - 1)?)),
        alternative: Alternative::TwoSided,
    })
}

/// Spearman's rank correlation test: the statistic is `ρ`
/// ([`data::spearman`], exact); the p-value uses `t = ρ √((n−2)/(1−ρ²))`
/// with `n − 2` degrees of freedom (exact Student-t tail; `t²` is rational
/// even when `ρ` is not).  `|ρ| = 1` gives `p = 0` in the alternative's
/// direction.  `scipy.stats.spearmanr(x, y, alternative)`.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::linprog::q;
/// use symplex::stats::data::from_i64;
/// use symplex::stats::hypothesis::{spearman_test, Alternative};
///
/// let ctx = Context::new();
/// let x = from_i64(&[1, 2, 3, 4, 5, 6, 7, 8]);
/// let y = from_i64(&[2, 1, 4, 3, 7, 8, 5, 6]);
/// // scipy: spearmanr(x, y) → statistic 0.7619047619047621 (= 16/21), pvalue 0.028004939153071815
/// let r = spearman_test(&ctx, &x, &y, Alternative::TwoSided)?;
/// assert_eq!(r.statistic, ctx.from_ratio(q(16, 21)));
/// assert!((r.p_value_f64()? - 0.028_004_939_153_071_815).abs() < 1e-12);
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for unequal lengths, fewer than three
/// pairs, or a constant sample.
pub fn spearman_test(
    ctx: &Context,
    x: &[Q],
    y: &[Q],
    alt: Alternative,
) -> Result<TestResult, SymplexError> {
    const OP: &str = "spearman_test";
    check_same_len(OP, x, y)?;
    check_sample(OP, "the paired sample", x, 3)?;
    let n = x.len();
    let (rx, ry) = (data::ranks(x), data::ranks(y));
    let sxy = data::covariance(&rx, &ry, Ddof::Population)?;
    let sxx = data::variance(&rx, Ddof::Population)?;
    let syy = data::variance(&ry, Ddof::Population)?;
    if sxx.is_zero() || syy.is_zero() {
        return Err(invalid(OP, "a constant sample has no rank correlation"));
    }
    let rho = data::spearman(ctx, x, y)?;
    let df = qu(n - 2);
    let r2 = &sxy * &sxy / (&sxx * &syy);
    let one_minus = Q::one() - &r2;
    let p_value = if one_minus.is_zero() {
        // |ρ| = 1: t is infinite in the direction of sign(ρ).
        let extreme = match alt {
            Alternative::TwoSided => true,
            Alternative::Greater => sxy.is_positive(),
            Alternative::Less => sxy.is_negative(),
        };
        if extreme { ctx.zero() } else { ctx.one() }
    } else {
        // t = sxy / √(sxx·syy·(1 − ρ²)/(n − 2)) has t² = ρ²(n−2)/(1−ρ²).
        let stat = RootRatio {
            num: sxy,
            var: &sxx * &syy * &one_minus / &df,
        };
        student_p_value(ctx, &df, &stat, alt)
    };
    Ok(TestResult {
        statistic: rho,
        p_value,
        df: Some(ex(ctx, &df)),
        alternative: alt,
    })
}

/// Kendall's τ-b test.  The statistic is `τ_b = (C − D)/√((n₀−n₁)(n₀−n₂))`
/// ([`data::kendall_tau`], exact).  `exact` (no ties): the p-value from the
/// exact distribution of the number of inversions (Kendall, "Rank
/// Correlation Methods"), an exact rational — scipy's `method='exact'`.
/// Otherwise `z = (C − D)/√var` with the tie-corrected variance
/// `var = (m(2n+5) − Σt(t−1)(2t+5) − Σu(u−1)(2u+5))/18 + 2n₁n₂/m + x₀y₀/(9m(n−2))`,
/// `m = n(n−1)`, and the normal tail — scipy's `method='asymptotic'`.
/// `scipy.stats.kendalltau(x, y, method, alternative)`.  The last term
/// needs a tie group of three or more in both samples, so it is added only
/// then; with `n = 2` the asymptotic `var = 1`, `z = ±1` and the two-sided
/// `p = erfc(1/√2) ≈ 0.3173`, where scipy (1.18) evaluates that term
/// anyway and raises `ZeroDivisionError` (`n − 2 = 0`).  (The exact
/// `p = 1` of `n = 2` shows how poor the normal approximation is there.)
///
/// ```
/// use symplex::prelude::*;
/// use symplex::linprog::q;
/// use symplex::stats::data::from_i64;
/// use symplex::stats::hypothesis::{kendall_test, Alternative};
///
/// let ctx = Context::new();
/// let x = from_i64(&[1, 2, 3, 4, 5, 6, 7, 8]);
/// let y = from_i64(&[2, 1, 4, 3, 7, 8, 5, 6]);
/// // scipy: kendalltau(x, y, method='exact') → statistic 0.5714285714285714 (= 4/7), pvalue 0.06101190476190476 (= 41/672)
/// let r = kendall_test(&ctx, &x, &y, Alternative::TwoSided, true)?;
/// assert_eq!(r.statistic, ctx.from_ratio(q(4, 7)));
/// assert_eq!(r.p_value_exact(), Some(q(41, 672)));
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for unequal lengths, fewer than two
/// pairs, a constant sample, or ties with `exact`.
pub fn kendall_test(
    ctx: &Context,
    x: &[Q],
    y: &[Q],
    alt: Alternative,
    exact: bool,
) -> Result<TestResult, SymplexError> {
    const OP: &str = "kendall_test";
    check_same_len(OP, x, y)?;
    check_sample(OP, "the paired sample", x, 2)?;
    let n = x.len();
    let (mut con, mut dis) = (0usize, 0usize);
    for i in 0..n {
        for j in i + 1..n {
            let (sx, sy) = (x[i].cmp(&x[j]), y[i].cmp(&y[j]));
            if sx == Ordering::Equal || sy == Ordering::Equal {
                continue;
            }
            if sx == sy {
                con += 1;
            } else {
                dis += 1;
            }
        }
    }
    let tot = n * (n - 1) / 2;
    let tie_stats = |t: &[usize]| -> (usize, Q, Q) {
        // (pairs tied, Σ t(t−1)(t−2), Σ t(t−1)(2t+5))
        t.iter().fold((0, Q::zero(), Q::zero()), |(p, a, b), &t| {
            (
                p + t * (t - 1) / 2,
                a + qu(t * (t - 1) * (t - 2)),
                b + qu(t * (t - 1) * (2 * t + 5)),
            )
        })
    };
    let (xtie, x0, x1) = tie_stats(&data::tie_sizes(x));
    let (ytie, y0, y1) = tie_stats(&data::tie_sizes(y));
    if xtie == tot || ytie == tot {
        return Err(invalid(OP, "a constant sample has no rank correlation"));
    }
    let tau = data::kendall_tau(ctx, x, y)?;
    if !exact {
        let m = qu(n * (n - 1));
        let mut var = (&m * qu(2 * n + 5) - &x1 - &y1) / qi(18) + qi(2) * qu(xtie * ytie) / &m;
        if !x0.is_zero() && !y0.is_zero() {
            var += &x0 * &y0 / (qi(9) * &m * qu(n - 2));
        }
        if !var.is_positive() {
            return Err(invalid(OP, "the variance of the statistic is zero"));
        }
        let stat = RootRatio {
            num: qu(con) - qu(dis),
            var,
        };
        return Ok(TestResult {
            statistic: tau,
            p_value: normal_p_value(ctx, &stat, alt),
            df: None,
            alternative: alt,
        });
    }
    if xtie > 0 || ytie > 0 {
        return Err(invalid(OP, "the exact method needs samples without ties"));
    }
    // scipy: c = concordant count; work in the left tail of the symmetric
    // distribution of the inversion count.
    let c = tot - dis;
    let in_right_tail = c >= tot - c;
    let cmin = c.min(tot - c);
    let freq = inversion_counts(n, cmin);
    let total = factorial_big(n);
    let left = Q::new(sum_big(&freq), total.clone());
    let at = Q::new(freq[cmin].clone(), total);
    let p = match alt {
        Alternative::TwoSided => (left * qi(2)).min(Q::one()),
        Alternative::Greater | Alternative::Less => {
            if in_right_tail == (alt == Alternative::Greater) {
                left
            } else {
                Q::one() - left + at
            }
        }
    };
    Ok(result(ctx, tau, p, None, alt))
}

/// Kolmogorov's distribution: `P(K > x) = 2 Σ_{k≥1} (−1)^{k−1} e^{−2k²x²}`.
fn kolmogorov_sf(x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    if x < 1.0 {
        // Jacobi form: K(x) = √(2π)/x · Σ_{k≥1} exp(−(2k−1)²π²/(8x²)).
        let mut s = 0.0;
        for k in 1..=20 {
            let m = f64::from(2 * k - 1);
            s += (-m * m * std::f64::consts::PI * std::f64::consts::PI / (8.0 * x * x)).exp();
        }
        1.0 - (2.0 * std::f64::consts::PI).sqrt() / x * s
    } else {
        let mut s = 0.0;
        for k in 1..=200 {
            let kf = f64::from(k);
            let term = (-2.0 * kf * kf * x * x).exp();
            s += if k % 2 == 1 { term } else { -term };
            if term < 1e-18 {
                break;
            }
        }
        (2.0 * s).clamp(0.0, 1.0)
    }
}

/// Smirnov's exact one-sided distribution (Birnbaum–Tingey):
/// `P(D⁺ₙ ≥ d) = d Σ_{j=0}^{⌊n(1−d)⌋} C(n, j) (1 − d − j/n)^{n−j} (d + j/n)^{j−1}`,
/// summed in log space.  `scipy.stats.ksone.sf(d, n)`.
fn smirnov_sf(n: usize, d: f64) -> f64 {
    if d <= 0.0 {
        return 1.0;
    }
    if d >= 1.0 {
        return 0.0;
    }
    let nf = n as f64;
    let log_n_fact = lgamma(nf + 1.0);
    let mut sum = 0.0;
    for j in 0..=n {
        let jf = j as f64;
        let a = 1.0 - d - jf / nf;
        if a <= 0.0 {
            break;
        }
        let log_term = if j == 0 {
            nf * a.ln() - d.ln()
        } else {
            log_n_fact - lgamma(jf + 1.0) - lgamma(nf - jf + 1.0)
                + (nf - jf) * a.ln()
                + (jf - 1.0) * (d + jf / nf).ln()
        };
        sum += log_term.exp();
    }
    (d * sum).clamp(0.0, 1.0)
}

/// One-sample Kolmogorov–Smirnov test of `x` against a continuous
/// distribution.  `D⁺ = max(i/n − F(x₍ᵢ₎))`, `D⁻ = max(F(x₍ᵢ₎) − (i−1)/n)`,
/// `D = max(D⁺, D⁻)`, with `F` evaluated numerically (the exact CDF
/// expression of `dist` at each observation, then `eval_f64`).  The
/// two-sided p-value is asymptotic — Kolmogorov's distribution at `√n D`;
/// the one-sided p-values are Smirnov's exact `P(D⁺ₙ ≥ d)` (as scipy does
/// for every `method`).  Numerical throughout, because `F` is
/// transcendental for the distributions this test is used with.
/// `scipy.stats.ks_1samp(x, cdf, alternative, method='asymp')`.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::Distribution;
/// use symplex::stats::data::from_f64;
/// use symplex::stats::hypothesis::{ks_one_sample, Alternative};
///
/// let ctx = Context::new();
/// let x = from_f64(&[-1.2, -0.3, 0.1, 0.4, 0.9, 1.5, 2.2, -0.7])?;
/// let normal = Distribution::normal(ctx.int(0), ctx.int(1));
/// // scipy: ks_1samp(x, norm.cdf, method='asymp') → statistic 0.19093987465324047, pvalue 0.9324475218943081
/// let r = ks_one_sample(&x, &normal, Alternative::TwoSided)?;
/// assert!((r.statistic - 0.190_939_874_653_240_47).abs() < 1e-12);
/// assert!((r.p_value - 0.932_447_521_894_308_1).abs() < 1e-9);
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for an empty sample or a discrete
/// distribution; the CDF's evaluation error if a parameter is symbolic.
pub fn ks_one_sample(
    x: &[Q],
    dist: &Distribution,
    alt: Alternative,
) -> Result<KsResult, SymplexError> {
    const OP: &str = "ks_one_sample";
    check_sample(OP, "the sample", x, 1)?;
    if !dist.is_continuous() {
        return Err(invalid(OP, "the reference distribution must be continuous"));
    }
    let ctx = dist.context();
    let sorted = data::sorted(x);
    let n = sorted.len() as f64;
    let (mut d_plus, mut d_minus) = (0.0f64, 0.0f64);
    for (i, v) in sorted.iter().enumerate() {
        let f = dist.cdf(&ex(&ctx, v)).eval_f64()?;
        d_plus = d_plus.max((i + 1) as f64 / n - f);
        d_minus = d_minus.max(f - i as f64 / n);
    }
    let (statistic, p_value) = match alt {
        Alternative::TwoSided => {
            let d = d_plus.max(d_minus);
            (d, kolmogorov_sf(n.sqrt() * d))
        }
        Alternative::Greater => (d_plus, smirnov_sf(sorted.len(), d_plus)),
        Alternative::Less => (d_minus, smirnov_sf(sorted.len(), d_minus)),
    };
    Ok(KsResult {
        statistic,
        p_value,
        alternative: alt,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. Effect sizes
// ═══════════════════════════════════════════════════════════════════════════

/// Cohen's `d = (x̄ − ȳ) / s` as an exact expression, with `s²` the pooled
/// variance `((n₁−1)s₁² + (n₂−1)s₂²)/(n₁+n₂−2)` (`pooled`) or the plain
/// average `(s₁² + s₂²)/2` (Cohen's original definition for equal sizes).
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::data::from_i64;
/// use symplex::stats::hypothesis::cohens_d;
///
/// let ctx = Context::new();
/// let x = from_i64(&[20, 22, 19, 20, 22, 20, 21]);
/// let y = from_i64(&[28, 32, 36, 24, 29, 32]);
/// // numpy: (mean(x) − mean(y)) / sqrt(((n1−1)var(x, ddof=1) + (n2−1)var(y, ddof=1))/(n1+n2−2)) = -3.3080302571461795
/// let d = cohens_d(&ctx, &x, &y, true)?;
/// assert!((d.eval_f64()? - -3.308_030_257_146_179_5).abs() < 1e-12);
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for a sample with fewer than two
/// observations or two constant samples.
pub fn cohens_d(ctx: &Context, x: &[Q], y: &[Q], pooled: bool) -> Result<Ex, SymplexError> {
    const OP: &str = "cohens_d";
    check_sample(OP, "the first sample", x, 2)?;
    check_sample(OP, "the second sample", y, 2)?;
    let (n1, n2) = (x.len(), y.len());
    let (v1, v2) = (
        data::variance(x, Ddof::Sample)?,
        data::variance(y, Ddof::Sample)?,
    );
    let var = if pooled {
        (qu(n1 - 1) * &v1 + qu(n2 - 1) * &v2) / qu(n1 + n2 - 2)
    } else {
        (&v1 + &v2) / qi(2)
    };
    if var.is_zero() {
        return Err(invalid(OP, "both samples are constant (zero variance)"));
    }
    let num = data::mean(x)? - data::mean(y)?;
    Ok(RootRatio { num, var }.to_ex(ctx))
}

/// Hedges' `g = d · J` with the pooled Cohen's `d` and the small-sample
/// correction `J = 1 − 3/(4(n₁ + n₂) − 9)` (Hedges & Olkin's approximation
/// of `Γ(ν/2)/(√(ν/2) Γ((ν−1)/2))`), as an exact expression.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::data::from_i64;
/// use symplex::stats::hypothesis::hedges_g;
///
/// let ctx = Context::new();
/// let x = from_i64(&[20, 22, 19, 20, 22, 20, 21]);
/// let y = from_i64(&[28, 32, 36, 24, 29, 32]);
/// // numpy: d · (1 − 3/(4·13 − 9)) = -3.3080302571461795 · (40/43) = -3.077237448508074
/// let g = hedges_g(&ctx, &x, &y)?;
/// assert!((g.eval_f64()? - -3.077_237_448_508_074).abs() < 1e-12);
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// As [`cohens_d`].
pub fn hedges_g(ctx: &Context, x: &[Q], y: &[Q]) -> Result<Ex, SymplexError> {
    let d = cohens_d(ctx, x, y, true)?;
    let n = usize_to_i64("hedges_g", x.len() + y.len())?;
    let j = ctx.one() - ctx.rational(3, 4 * n - 9);
    Ok((d * j).simplify())
}

/// Glass's `Δ = (x̄ − ȳ) / s_y`, standardised by the second (control)
/// sample's standard deviation alone, as an exact expression.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::data::from_i64;
/// use symplex::stats::hypothesis::glass_delta;
///
/// let ctx = Context::new();
/// let x = from_i64(&[20, 22, 19, 20, 22, 20, 21]);
/// let y = from_i64(&[28, 32, 36, 24, 29, 32]);
/// // numpy: (mean(x) − mean(y)) / std(y, ddof=1) = -2.329471985471971
/// let g = glass_delta(&ctx, &x, &y)?;
/// assert!((g.eval_f64()? - -2.329_471_985_471_971).abs() < 1e-12);
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for an empty `x`, fewer than two
/// observations in `y`, or a constant `y`.
pub fn glass_delta(ctx: &Context, x: &[Q], y: &[Q]) -> Result<Ex, SymplexError> {
    const OP: &str = "glass_delta";
    check_sample(OP, "the first sample", x, 1)?;
    check_sample(OP, "the control sample", y, 2)?;
    let var = data::variance(y, Ddof::Sample)?;
    if var.is_zero() {
        return Err(invalid(
            OP,
            "the control sample is constant (zero variance)",
        ));
    }
    let num = data::mean(x)? - data::mean(y)?;
    Ok(RootRatio { num, var }.to_ex(ctx))
}

/// The rank-biserial correlation of a Mann–Whitney `U₁` (the statistic for
/// the first sample of sizes `n₁`, `n₂`): `r = 2U₁/(n₁n₂) − 1 = P(X > Y) −
/// P(X < Y)`, which equals Cliff's δ.  (Some sources report `1 − 2U/(n₁n₂)`
/// with the smaller `U`, i.e. `|r|`.)
///
/// ```
/// use symplex::linprog::{q, qi};
/// use symplex::stats::hypothesis::rank_biserial;
///
/// // U₁ = 17 for sizes 5 and 4: r = 34/20 − 1 = 7/10
/// assert_eq!(rank_biserial(&qi(17), 5, 4)?, q(7, 10));
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for an empty sample or `U ∉ [0, n₁n₂]`.
pub fn rank_biserial(u1: &Q, n1: usize, n2: usize) -> Result<Q, SymplexError> {
    const OP: &str = "rank_biserial";
    if n1 == 0 || n2 == 0 {
        return Err(invalid(OP, "both samples must be non-empty"));
    }
    let n1n2 = qmul(n1, n2);
    if u1.is_negative() || *u1 > n1n2 {
        return Err(invalid(OP, "U must lie in [0, n₁n₂]"));
    }
    Ok(qi(2) * u1 / n1n2 - Q::one())
}

/// `η² = SS_between / SS_total` of `k` groups, exact (the same quantity as
/// [`AnovaResult::eta_squared`](super::anova::AnovaResult::eta_squared)).
///
/// ```
/// use symplex::linprog::q;
/// use symplex::stats::data::from_i64;
/// use symplex::stats::hypothesis::eta_squared;
///
/// let g = [from_i64(&[6, 8, 4, 5, 3, 4]), from_i64(&[8, 12, 9, 11, 6, 8]), from_i64(&[13, 9, 11, 8, 7, 12])];
/// // SS_between = 84, SS_within = 68
/// assert_eq!(eta_squared(&g)?, q(84, 152));
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for fewer than two groups, an empty
/// group, or identical observations throughout.
pub fn eta_squared(groups: &[Vec<Q>]) -> Result<Q, SymplexError> {
    const OP: &str = "eta_squared";
    let super::anova::GroupSums {
        ss_between,
        ss_within,
        ..
    } = super::anova::sums_of_squares(OP, groups)?;
    let total = &ss_between + &ss_within;
    if total.is_zero() {
        return Err(invalid(OP, "every observation is identical"));
    }
    Ok(ss_between / total)
}

/// Cliff's `δ = (#{xᵢ > yⱼ} − #{xᵢ < yⱼ}) / (n₁n₂)`, exact.
///
/// ```
/// use symplex::linprog::q;
/// use symplex::stats::data::from_i64;
/// use symplex::stats::hypothesis::cliffs_delta;
///
/// let x = from_i64(&[19, 22, 16, 29, 24]);
/// let y = from_i64(&[20, 11, 17, 12]);
/// assert_eq!(cliffs_delta(&x, &y)?, q(7, 10));
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for an empty sample.
pub fn cliffs_delta(x: &[Q], y: &[Q]) -> Result<Q, SymplexError> {
    const OP: &str = "cliffs_delta";
    check_sample(OP, "the first sample", x, 1)?;
    check_sample(OP, "the second sample", y, 1)?;
    let mut diff = 0i64;
    for a in x {
        for b in y {
            diff += match a.cmp(b) {
                Ordering::Greater => 1,
                Ordering::Less => -1,
                Ordering::Equal => 0,
            };
        }
    }
    Ok(qi(diff) / qmul(x.len(), y.len()))
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. Multiple comparisons
// ═══════════════════════════════════════════════════════════════════════════

fn check_pvalues(op: &'static str, p: &[f64], alpha: f64) -> Result<(), SymplexError> {
    if p.is_empty() {
        return Err(invalid(op, "no p-values"));
    }
    if let Some(bad) = p.iter().find(|&&v| !(0.0..=1.0).contains(&v)) {
        return Err(invalid(
            op,
            format!("p-values must lie in [0, 1], got {bad}"),
        ));
    }
    check_alpha(op, alpha)
}

/// Indices that sort `p` ascending (stable).
fn ascending_order(p: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..p.len()).collect();
    idx.sort_by(|&a, &b| p[a].total_cmp(&p[b]));
    idx
}

/// Scatter sorted results back into the input order.
fn unsort(order: &[usize], sorted_p: Vec<f64>, sorted_reject: Vec<bool>) -> Adjusted {
    let mut p_adjusted = vec![0.0; order.len()];
    let mut reject = vec![false; order.len()];
    for (rank, &i) in order.iter().enumerate() {
        p_adjusted[i] = sorted_p[rank].min(1.0);
        reject[i] = sorted_reject[rank];
    }
    Adjusted { p_adjusted, reject }
}

/// Bonferroni: `p̃ᵢ = min(1, m·pᵢ)`, reject where `pᵢ ≤ α/m`.
/// `statsmodels.stats.multitest.multipletests(p, alpha, method='bonferroni')`,
/// comparison included: in floating point `p ≤ α/m` and `m·p ≤ α` can
/// disagree in the last bit (`p = 0.05/11` at `α = 0.05` is rejected,
/// although `p̃ = 11·p` rounds to `0.05000000000000001`); before 0.27
/// symplex compared `p̃ ≤ α` and differed from statsmodels there.
///
/// ```
/// use symplex::stats::hypothesis::bonferroni;
///
/// // statsmodels: multipletests([0.01, 0.04, 0.03, 0.2], method='bonferroni')
/// //   → pvals_corrected [0.04, 0.16, 0.12, 0.8], reject [True, False, False, False]
/// let a = bonferroni(&[0.01, 0.04, 0.03, 0.2], 0.05)?;
/// assert!((a.p_adjusted[1] - 0.16).abs() < 1e-15);
/// assert_eq!(a.reject, vec![true, false, false, false]);
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for no p-values, a p-value outside
/// `[0, 1]`, or `alpha ∉ (0, 1)`.
pub fn bonferroni(p: &[f64], alpha: f64) -> Result<Adjusted, SymplexError> {
    check_pvalues("bonferroni", p, alpha)?;
    let m = p.len() as f64;
    let p_adjusted: Vec<f64> = p.iter().map(|&v| (v * m).min(1.0)).collect();
    let reject = p.iter().map(|&v| v <= alpha / m).collect();
    Ok(Adjusted { p_adjusted, reject })
}

/// Holm's step-down procedure: with `p₍₁₎ ≤ … ≤ p₍ₘ₎`, `p̃₍ᵢ₎ = min(1,
/// max_{j ≤ i} (m − j + 1) p₍ⱼ₎)`; reject `p₍ᵢ₎` while every `p₍ⱼ₎ ≤ α/(m −
/// j + 1)`, `j ≤ i` (equivalently `p̃₍ᵢ₎ ≤ α`, up to the last bit — the
/// comparison is statsmodels', as in [`bonferroni`]).
/// `multipletests(p, alpha, method='holm')`.
///
/// ```
/// use symplex::stats::hypothesis::holm;
///
/// // statsmodels: multipletests([0.01, 0.04, 0.03, 0.2], method='holm')
/// //   → pvals_corrected [0.04, 0.09, 0.09, 0.2], reject [True, False, False, False]
/// let a = holm(&[0.01, 0.04, 0.03, 0.2], 0.05)?;
/// assert!((a.p_adjusted[2] - 0.09).abs() < 1e-15);
/// assert_eq!(a.reject, vec![true, false, false, false]);
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// As [`bonferroni`].
pub fn holm(p: &[f64], alpha: f64) -> Result<Adjusted, SymplexError> {
    check_pvalues("holm", p, alpha)?;
    let m = p.len();
    let order = ascending_order(p);
    let mut sorted_p = Vec::with_capacity(m);
    let mut sorted_reject = Vec::with_capacity(m);
    let mut running = 0.0f64;
    let mut rejecting = true;
    for (rank, &i) in order.iter().enumerate() {
        let k = (m - rank) as f64;
        running = running.max(p[i] * k);
        sorted_p.push(running);
        rejecting = rejecting && p[i] <= alpha / k;
        sorted_reject.push(rejecting);
    }
    Ok(unsort(&order, sorted_p, sorted_reject))
}

/// Step-up false-discovery-rate control with the critical constants
/// `cᵢ = i/(m·scale)`: `p̃₍ᵢ₎ = min(1, min_{j ≥ i} p₍ⱼ₎/cⱼ)`; reject `p₍ᵢ₎` up to
/// the largest `i` with `p₍ᵢ₎ ≤ α cᵢ` (statsmodels' `fdr_bh` / `fdr_by`).
fn fdr_step_up(
    op: &'static str,
    p: &[f64],
    alpha: f64,
    scale: f64,
) -> Result<Adjusted, SymplexError> {
    check_pvalues(op, p, alpha)?;
    let m = p.len();
    let order = ascending_order(p);
    let factor: Vec<f64> = (1..=m).map(|i| i as f64 / m as f64 / scale).collect();
    let mut sorted_p = vec![0.0; m];
    let mut running = f64::INFINITY;
    for rank in (0..m).rev() {
        running = running.min(p[order[rank]] / factor[rank]);
        sorted_p[rank] = running;
    }
    let last = (0..m)
        .rev()
        .find(|&rank| p[order[rank]] <= alpha * factor[rank]);
    let sorted_reject = (0..m).map(|rank| last.is_some_and(|l| rank <= l)).collect();
    Ok(unsort(&order, sorted_p, sorted_reject))
}

/// Benjamini–Hochberg: `p̃₍ᵢ₎ = min(1, min_{j ≥ i} m·p₍ⱼ₎/j)`; reject the
/// hypotheses up to the largest `i` with `p₍ᵢ₎ ≤ α i/m`.
/// `multipletests(p, alpha, method='fdr_bh')`.
///
/// ```
/// use symplex::stats::hypothesis::benjamini_hochberg;
///
/// // statsmodels: multipletests([0.01, 0.04, 0.03, 0.2], method='fdr_bh')
/// //   → pvals_corrected [0.04, 0.05333333333333334, 0.05333333333333334, 0.2], reject [True, False, False, False]
/// let a = benjamini_hochberg(&[0.01, 0.04, 0.03, 0.2], 0.05)?;
/// assert!((a.p_adjusted[1] - 0.053_333_333_333_333_34).abs() < 1e-15);
/// assert_eq!(a.reject, vec![true, false, false, false]);
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// As [`bonferroni`].
pub fn benjamini_hochberg(p: &[f64], alpha: f64) -> Result<Adjusted, SymplexError> {
    fdr_step_up("benjamini_hochberg", p, alpha, 1.0)
}

/// Benjamini–Yekutieli (FDR under arbitrary dependence): Benjamini–Hochberg
/// with the constants divided by `Σ_{j=1}^{m} 1/j`.
/// `multipletests(p, alpha, method='fdr_by')`.
///
/// ```
/// use symplex::stats::hypothesis::benjamini_yekutieli;
///
/// // statsmodels: multipletests([0.01, 0.04, 0.03, 0.2], method='fdr_by')
/// //   → pvals_corrected [0.08333333333333331, 0.1111111111111111, 0.1111111111111111, 0.41666666666666663]
/// let a = benjamini_yekutieli(&[0.01, 0.04, 0.03, 0.2], 0.05)?;
/// assert!((a.p_adjusted[0] - 0.083_333_333_333_333_31).abs() < 1e-15);
/// assert_eq!(a.reject, vec![false; 4]);
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// As [`bonferroni`].
pub fn benjamini_yekutieli(p: &[f64], alpha: f64) -> Result<Adjusted, SymplexError> {
    let harmonic: f64 = (1..=p.len()).map(|j| 1.0 / j as f64).sum();
    fdr_step_up("benjamini_yekutieli", p, alpha, harmonic)
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. Resampling
// ═══════════════════════════════════════════════════════════════════════════

/// The `p`-quantile of an ascending sample by linear interpolation
/// (`numpy.quantile`'s default).
fn quantile_sorted(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let h = (n - 1) as f64 * p;
    let lo = (h.floor() as usize).min(n - 1);
    let hi = (lo + 1).min(n - 1);
    let frac = h - lo as f64;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

fn check_f64_data(
    op: &'static str,
    name: &str,
    data: &[f64],
    min: usize,
) -> Result<(), SymplexError> {
    if data.len() < min {
        return Err(invalid(
            op,
            format!("{name} needs at least {min} observations"),
        ));
    }
    if let Some(bad) = data.iter().find(|v| !v.is_finite()) {
        return Err(invalid(
            op,
            format!("{name} contains a non-finite value {bad}"),
        ));
    }
    Ok(())
}

/// A bootstrap confidence interval for `statistic(data)`: `n_resamples`
/// resamples with replacement drawn with the deterministic `rng`, then the
/// `Percentile` interval `[q_{α/2}, q_{1−α/2}]` of the resampled statistics
/// or the `Basic` interval `[2θ̂ − q_{1−α/2}, 2θ̂ − q_{α/2}]`.  Quantiles are
/// linearly interpolated.  (`scipy.stats.bootstrap(method='percentile' |
/// 'basic')` up to the random stream.)
///
/// ```
/// use symplex::stats::Rng;
/// use symplex::stats::hypothesis::{bootstrap_ci, BootstrapMethod};
///
/// let data: Vec<f64> = (1..=20).map(f64::from).collect();
/// let mean = |x: &[f64]| x.iter().sum::<f64>() / x.len() as f64;
/// let ci = bootstrap_ci(&data, mean, 2000, 0.95, &mut Rng::new(7), BootstrapMethod::Percentile)?;
/// assert!(ci.lower < 10.5 && 10.5 < ci.upper);
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for empty or non-finite data,
/// `n_resamples = 0`, or `confidence ∉ (0, 1)`;
/// [`SymplexError::ComputationFailed`] if the statistic is non-finite.
pub fn bootstrap_ci(
    data: &[f64],
    statistic: impl Fn(&[f64]) -> f64,
    n_resamples: usize,
    confidence: f64,
    rng: &mut Rng,
    method: BootstrapMethod,
) -> Result<Interval<f64>, SymplexError> {
    const OP: &str = "bootstrap_ci";
    check_f64_data(OP, "the data", data, 1)?;
    check_confidence(OP, confidence)?;
    if n_resamples == 0 {
        return Err(invalid(OP, "at least one resample is needed"));
    }
    let n = data.len();
    let observed = statistic(data);
    let mut resample = vec![0.0; n];
    let mut stats = Vec::with_capacity(n_resamples);
    for _ in 0..n_resamples {
        for slot in &mut resample {
            *slot = data[rng.below(n)];
        }
        let s = statistic(&resample);
        if !s.is_finite() {
            return Err(SymplexError::computation_failed(
                OP,
                "the statistic of a resample is not finite",
            ));
        }
        stats.push(s);
    }
    stats.sort_by(f64::total_cmp);
    let alpha = 1.0 - confidence;
    let (lo, hi) = (
        quantile_sorted(&stats, alpha / 2.0),
        quantile_sorted(&stats, 1.0 - alpha / 2.0),
    );
    Ok(match method {
        BootstrapMethod::Percentile => Interval::closed(lo, hi),
        BootstrapMethod::Basic => Interval::closed(2.0 * observed - hi, 2.0 * observed - lo),
    })
}

/// A randomised two-sample permutation test of `statistic(x, y)`: the
/// pooled sample is shuffled `n_permutations` times with the deterministic
/// `rng` and split into the original sizes; the p-value is `(#{permuted
/// statistics at least as extreme} + 1) / (n_permutations + 1)` — `T* ≥ T`
/// (`Greater`), `T* ≤ T` (`Less`), `2 min(·, ·)` clipped to `1` two-sided
/// (`scipy.stats.permutation_test(permutation_type='independent')`'s
/// definitions, up to the random stream).
///
/// ```
/// use symplex::stats::Rng;
/// use symplex::stats::hypothesis::{permutation_test, Alternative};
///
/// let x = [1.1, 2.3, 1.9, 2.8, 2.2, 1.7];
/// let y = [4.9, 5.2, 4.4, 5.8, 5.1, 4.7];
/// let diff = |a: &[f64], b: &[f64]| {
///     a.iter().sum::<f64>() / a.len() as f64 - b.iter().sum::<f64>() / b.len() as f64
/// };
/// let r = permutation_test(&x, &y, diff, 4000, &mut Rng::new(1), Alternative::TwoSided)?;
/// assert!(r.p_value < 0.01);
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for an empty or non-finite sample or
/// `n_permutations = 0`; [`SymplexError::ComputationFailed`] if the
/// statistic is non-finite.
pub fn permutation_test(
    x: &[f64],
    y: &[f64],
    statistic: impl Fn(&[f64], &[f64]) -> f64,
    n_permutations: usize,
    rng: &mut Rng,
    alt: Alternative,
) -> Result<PermutationResult, SymplexError> {
    const OP: &str = "permutation_test";
    check_f64_data(OP, "the first sample", x, 1)?;
    check_f64_data(OP, "the second sample", y, 1)?;
    if n_permutations == 0 {
        return Err(invalid(OP, "at least one permutation is needed"));
    }
    let observed = statistic(x, y);
    if !observed.is_finite() {
        return Err(SymplexError::computation_failed(
            OP,
            "the observed statistic is not finite",
        ));
    }
    let n1 = x.len();
    let mut pooled: Vec<f64> = x.iter().chain(y).copied().collect();
    let slack = 1e-14 * observed.abs();
    let (mut count_ge, mut count_le) = (0usize, 0usize);
    for _ in 0..n_permutations {
        // Fisher–Yates shuffle.
        for i in (1..pooled.len()).rev() {
            let j = rng.below(i + 1);
            pooled.swap(i, j);
        }
        let s = statistic(&pooled[..n1], &pooled[n1..]);
        if !s.is_finite() {
            return Err(SymplexError::computation_failed(
                OP,
                "the statistic of a permutation is not finite",
            ));
        }
        if s >= observed - slack {
            count_ge += 1;
        }
        if s <= observed + slack {
            count_le += 1;
        }
    }
    let denom = (n_permutations + 1) as f64;
    let greater = (count_ge + 1) as f64 / denom;
    let less = (count_le + 1) as f64 / denom;
    let p_value = match alt {
        Alternative::Greater => greater,
        Alternative::Less => less,
        Alternative::TwoSided => (2.0 * greater.min(less)).min(1.0),
    };
    Ok(PermutationResult {
        statistic: observed,
        p_value,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. Power and sample size
// ═══════════════════════════════════════════════════════════════════════════

/// The sample size for estimating a proportion `p` to within `±margin` at
/// the given confidence: `n = ⌈z²_{1−α/2} p(1−p) / margin²⌉`.
/// `statsmodels.stats.proportion.samplesize_confint_proportion(p, margin, alpha)`
/// (which returns the un-rounded `n`).
///
/// ```
/// use symplex::stats::hypothesis::sample_size_for_proportion;
///
/// // statsmodels: samplesize_confint_proportion(0.5, 0.03) = 1067.0718946372576  → 1068
/// assert_eq!(sample_size_for_proportion(0.03, 0.95, 0.5)?, 1068);
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for a non-finite or non-positive
/// `margin`, `confidence ∉ (0, 1)` or `p ∉ (0, 1)`;
/// [`SymplexError::ComputationFailed`] when the sample size does not fit
/// in `usize` (`margin = 10⁻¹⁰` needs `9.6·10¹⁹`; before 0.27 the
/// conversion saturated to `usize::MAX`).
pub fn sample_size_for_proportion(
    margin: f64,
    confidence: f64,
    p: f64,
) -> Result<usize, SymplexError> {
    const OP: &str = "sample_size_for_proportion";
    check_finite(OP, "margin", margin)?;
    if margin <= 0.0 {
        return Err(invalid(OP, "the margin must be positive"));
    }
    check_confidence(OP, confidence)?;
    check_unit_open(OP, "p", p)?;
    let z = norm_isf((1.0 - confidence) / 2.0);
    let n = (z * z * p * (1.0 - p) / (margin * margin)).ceil();
    // `n` is positive, possibly +∞ (`margin²` underflowing); `usize::MAX as
    // f64` is 2^64 (or 2^32), itself out of range.
    if n >= usize::MAX as f64 {
        return Err(SymplexError::computation_failed(
            OP,
            format!("the required sample size {n:e} does not fit in usize"),
        ));
    }
    Ok(n as usize)
}

/// statsmodels' two-sided normal power with `nobs = n/2` (equal groups):
/// `Φ̄(z_{α/2} − h√(n/2)) + Φ(−z_{α/2} − h√(n/2))`.
fn normal_power_two_sided(effect: f64, n_per_group: f64, alpha: f64) -> f64 {
    let crit = norm_isf(alpha / 2.0);
    let shift = effect * (n_per_group / 2.0).sqrt();
    norm_sf(crit - shift) + norm_cdf(-crit - shift)
}

fn check_proportions_and_alpha(
    op: &'static str,
    p1: f64,
    p2: f64,
    alpha: f64,
) -> Result<(), SymplexError> {
    check_unit_open(op, "p1", p1)?;
    check_unit_open(op, "p2", p2)?;
    check_alpha(op, alpha)
}

/// The power of the two-sided two-proportion z-test with `n_per_group` per
/// group at level `alpha`, by the normal approximation on Cohen's `h`:
/// `Φ̄(z_{α/2} − h√(n/2)) + Φ(−z_{α/2} − h√(n/2))`.
/// `statsmodels.stats.power.NormalIndPower().power(proportion_effectsize(p1, p2), n, alpha, ratio=1)`.
///
/// ```
/// use symplex::stats::hypothesis::power_two_proportions;
///
/// // statsmodels: NormalIndPower().power(proportion_effectsize(0.5, 0.4), nobs1=200, alpha=0.05, ratio=1) = 0.5214145419211713
/// let p = power_two_proportions(0.5, 0.4, 200, 0.05)?;
/// assert!((p - 0.521_414_541_921_171_3).abs() < 1e-9);
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for proportions or `alpha` outside
/// `(0, 1)` or `n_per_group = 0`.
pub fn power_two_proportions(
    p1: f64,
    p2: f64,
    n_per_group: usize,
    alpha: f64,
) -> Result<f64, SymplexError> {
    const OP: &str = "power_two_proportions";
    check_proportions_and_alpha(OP, p1, p2, alpha)?;
    if n_per_group == 0 {
        return Err(invalid(OP, "the group size must be positive"));
    }
    Ok(normal_power_two_sided(
        cohens_h_f64(p1, p2),
        n_per_group as f64,
        alpha,
    ))
}

/// Cohen's `h` in floating point, in the non-cancelling form of
/// [`cohens_h`]: `2 atan((p₁ − p₂)/(√(p₁(1 − p₁)) + √(p₂(1 − p₂))))` (the
/// callers pass `p ∈ (0, 1)`, so the denominator is positive).
fn cohens_h_f64(p1: f64, p2: f64) -> f64 {
    if p1 == p2 {
        return 0.0;
    }
    let den = (p1 * (1.0 - p1)).sqrt() + (p2 * (1.0 - p2)).sqrt();
    2.0 * ((p1 - p2) / den).atan()
}

/// The smallest integer `n ≥ start` with `power(n) ≥ target` for an
/// increasing `power`, by galloping then bisection
/// ([`partition_point_by`]).
fn smallest_n_with_power(
    op: &'static str,
    start: usize,
    target: f64,
    power: impl Fn(usize) -> Result<f64, SymplexError>,
) -> Result<usize, SymplexError> {
    // 2^40 on 64-bit targets; on 32-bit ones (wasm32) the search stops at
    // `usize::MAX / 2`, which no power calculation reaches either.
    const CAP: usize = if usize::BITS >= 41 {
        1 << 40
    } else {
        usize::MAX / 2
    };
    // The first error stops the search (`true` ends the gallop at once)
    // and is reported instead of a sample size.
    let mut error = None;
    let n = partition_point_by(start, CAP, |n| match power(n) {
        Ok(pw) => pw >= target,
        Err(e) => {
            error.get_or_insert(e);
            true
        }
    });
    if let Some(e) = error {
        return Err(e);
    }
    if n >= CAP {
        return Err(SymplexError::computation_failed(
            op,
            "the required sample size exceeds 2^40",
        ));
    }
    Ok(n)
}

/// The per-group sample size for the two-sided two-proportion z-test to
/// reach `power` at level `alpha`: the smallest `n` with
/// [`power_two_proportions`]`(p1, p2, n, alpha) ≥ power`, i.e. the ceiling of
/// `statsmodels.stats.power.NormalIndPower().solve_power(proportion_effectsize(p1, p2), alpha=alpha, power=power, ratio=1)`.
///
/// ```
/// use symplex::stats::hypothesis::sample_size_two_proportions;
///
/// // statsmodels: NormalIndPower().solve_power(proportion_effectsize(0.5, 0.4), alpha=0.05, power=0.8, ratio=1) = 387.1677468578098
/// assert_eq!(sample_size_two_proportions(0.5, 0.4, 0.05, 0.8)?, 388);
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for proportions, `alpha` or `power`
/// outside `(0, 1)`, `power ≤ alpha`, or `p1 = p2`.
pub fn sample_size_two_proportions(
    p1: f64,
    p2: f64,
    alpha: f64,
    power: f64,
) -> Result<usize, SymplexError> {
    const OP: &str = "sample_size_two_proportions";
    check_proportions_and_alpha(OP, p1, p2, alpha)?;
    check_unit_open(OP, "power", power)?;
    if power <= alpha {
        return Err(invalid(OP, "the target power must exceed alpha"));
    }
    if p1 == p2 {
        return Err(invalid(OP, "equal proportions have no finite sample size"));
    }
    smallest_n_with_power(OP, 1, power, |n| power_two_proportions(p1, p2, n, alpha))
}

/// `ln(1 + t) − t` without cancellation near `t = 0`: with `w = t/(2 + t)`,
/// `ln(1 + t) = 2 atanh w`, so `ln(1 + t) − t = −t²/(2 + t) + 2 Σ_{j≥1}
/// w^{2j+1}/(2j + 1)` (`|w| ≤ ⅓` for `|t| < ½`; beyond, the direct
/// difference loses at most a factor 3).
fn ln_1p_minus(t: f64) -> f64 {
    if t.abs() >= 0.5 {
        return t.ln_1p() - t;
    }
    let w = t / (2.0 + t);
    let w2 = w * w;
    let mut power = w * w2;
    let mut sum = 0.0;
    for j in 1..=40 {
        let term = power / f64::from(2 * j + 1);
        sum += term;
        if term.abs() <= f64::EPSILON * sum.abs() {
            break;
        }
        power *= w2;
    }
    -t * t / (2.0 + t) + 2.0 * sum
}

/// The Stirling remainder `ln Γ(k) − ((k − ½) ln k − k + ½ ln 2π)`: its
/// asymptotic series from `k = 15` (the first omitted term is below
/// `3·10⁻¹⁶` there), `lgamma` below.
fn stirling_remainder(k: f64) -> f64 {
    if k < 15.0 {
        let tau = 2.0 * std::f64::consts::PI;
        return lgamma(k) - ((k - 0.5) * k.ln() - k + 0.5 * tau.ln());
    }
    let r = k.recip();
    let r2 = r * r;
    r * (1.0 / 12.0 - r2 * (1.0 / 360.0 - r2 * (1.0 / 1260.0 - r2 * (1.0 / 1680.0 - r2 / 1188.0))))
}

/// The `χ²_ν` density.  With `k = ν/2` and `u = v/ν`, `ln f(v) = −ln 2 −
/// ½ ln(2πk) − s(k) − ln u + k·(ln u − u + 1)` (`s` the Stirling
/// remainder), which keeps its digits at any `ν`: the textbook `(k − 1)
/// ln v − v/2 − k ln 2 − ln Γ(k)` cancels terms of size `ν ln ν`, and at
/// `ν = 2·10⁹` lost `10⁻⁶` of the power (before 0.27).
fn chi_squared_density(df: f64) -> impl Fn(f64) -> f64 {
    let k = df / 2.0;
    let log_norm = -std::f64::consts::LN_2
        - 0.5 * (2.0 * std::f64::consts::PI * k).ln()
        - stirling_remainder(k);
    move |v: f64| -> f64 {
        if v <= 0.0 {
            return 0.0;
        }
        let t = (v - df) / df;
        (log_norm - t.ln_1p() + k * ln_1p_minus(t)).exp()
    }
}

/// The power of the two-sided two-sample Student t-test (equal group
/// sizes) for a standardised effect `d` at level `alpha`, through the
/// noncentral t distribution: with `ν = 2n − 2`, `δ = d√(n/2)` and the
/// critical `t_c = t_{1−α/2, ν}`,
/// `power = 1 − ∫₀^∞ [Φ(t_c√(v/ν) − δ) − Φ(−t_c√(v/ν) − δ)] f_{χ²_ν}(v) dv`,
/// integrated by adaptive Gauss–Kronrod quadrature (about `1e-10`).
/// `statsmodels.stats.power.TTestIndPower().power(d, nobs1=n, alpha=alpha, ratio=1)`.
///
/// ```
/// use symplex::stats::hypothesis::power_t_test_two_sample;
///
/// // statsmodels: TTestIndPower().power(0.5, nobs1=30, alpha=0.05, ratio=1) = 0.47789652076016464
/// let p = power_t_test_two_sample(0.5, 30, 0.05)?;
/// assert!((p - 0.477_896_520_760_164_64).abs() < 1e-8);
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for a non-finite effect, `n < 2` or
/// `alpha ∉ (0, 1)`; the quantile's or quadrature's error if they fail.
pub fn power_t_test_two_sample(
    effect_size: f64,
    n_per_group: usize,
    alpha: f64,
) -> Result<f64, SymplexError> {
    const OP: &str = "power_t_test_two_sample";
    check_finite(OP, "effect_size", effect_size)?;
    check_alpha(OP, alpha)?;
    if n_per_group < 2 {
        return Err(invalid(OP, "each group needs at least two observations"));
    }
    let n = n_per_group as f64;
    let df = 2.0 * n - 2.0;
    let delta = effect_size * (n / 2.0).sqrt();
    let t_crit = student_t_quantile_f64(OP, df, 1.0 - alpha / 2.0)?;
    let density = chi_squared_density(df);
    let integrand = move |v: f64| -> f64 {
        let scale = (v / df).sqrt();
        (norm_cdf(t_crit * scale - delta) - norm_cdf(-t_crit * scale - delta)) * density(v)
    };
    // The χ²_ν mass outside [ν − 40σ, ν + 40σ + 50] (σ = √(2ν)) is far below
    // double precision; a finite window keeps the adaptive rule on the peak.
    let sd = (2.0 * df).sqrt();
    let lo = (df - 40.0 * sd).max(0.0);
    let hi = df + 40.0 * sd + 50.0;
    let opts = QuadOpts::default();
    let accept = quadrature(&integrand, lo, hi, &opts)?.value;
    Ok((1.0 - accept).clamp(0.0, 1.0))
}

/// The per-group sample size for the two-sided two-sample t-test to reach
/// `power` at level `alpha`: the smallest `n ≥ 2` with
/// [`power_t_test_two_sample`]`(d, n, alpha) ≥ power`, i.e. the ceiling of
/// `statsmodels.stats.power.TTestIndPower().solve_power(d, alpha=alpha, power=power, ratio=1)`.
///
/// ```
/// use symplex::stats::hypothesis::sample_size_t_test_two_sample;
///
/// // statsmodels: TTestIndPower().solve_power(0.5, alpha=0.05, power=0.8, ratio=1) = 63.765610588911635  → 64
/// assert_eq!(sample_size_t_test_two_sample(0.5, 0.05, 0.8)?, 64);
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for `d = 0`, or `alpha` / `power`
/// outside `(0, 1)` with `power ≤ alpha`.
pub fn sample_size_t_test_two_sample(
    effect_size: f64,
    alpha: f64,
    power: f64,
) -> Result<usize, SymplexError> {
    const OP: &str = "sample_size_t_test_two_sample";
    check_finite(OP, "effect_size", effect_size)?;
    check_alpha(OP, alpha)?;
    check_unit_open(OP, "power", power)?;
    if power <= alpha {
        return Err(invalid(OP, "the target power must exceed alpha"));
    }
    if effect_size == 0.0 {
        return Err(invalid(OP, "a zero effect has no finite sample size"));
    }
    smallest_n_with_power(OP, 2, power, |n| {
        power_t_test_two_sample(effect_size, n, alpha)
    })
}
