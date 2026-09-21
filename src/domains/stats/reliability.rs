//! Scale reliability, item analysis and further agreement / association
//! statistics: Cronbach's α and its relatives (standardized α, KR-20,
//! split-half with the Spearman–Brown prophecy, α-if-deleted, Guttman's
//! λ₂), classical item analysis (difficulty, the D discrimination index,
//! point-biserial and corrected item–total correlations), a confidence
//! interval and a test for Cohen's κ, κ_max, Cochran's Q, the ordinal
//! association measures (Goodman–Kruskal γ, Somers' D, Stuart's τ-c),
//! contingency-table diagnostics (expected counts, χ² contributions,
//! standardized and adjusted residuals) and inference on Pearson's r
//! (Fisher's z, a confidence interval, the t-test, comparing two r's).
//!
//! **Exact where rational.**  Everything that is a rational function of
//! the data (α, KR-20, α-if-deleted, κ and its large-sample variance,
//! Cochran's Q, γ, D, τ-c, expected counts, χ² contributions, item
//! difficulties and D indices) comes back as a [`Q`]; anything with a
//! square root (a correlation, a standard error, a standardized residual,
//! λ₂, a t or z statistic) as an exact [`Ex`]; and only confidence limits
//! that need a normal quantile are `f64`.  `scipy.stats`, `statsmodels`
//! and `fractions.Fraction` re-implementations of the cited formulas are
//! the references named in the tests.
//!
//! ```
//! use symplex::linprog::q;
//! use symplex::stats::agreement::RatingTable;
//! use symplex::stats::reliability::{cronbach_alpha, item_difficulty};
//!
//! // Five respondents (rows) answering three items (columns).
//! let t = RatingTable::from_i64(&[&[1, 1, 1], &[1, 1, 0], &[1, 0, 0], &[0, 1, 0], &[0, 0, 0]])?;
//! // Fraction: Σ item variances = 3/10 + 3/10 + 1/5 = 4/5, total-score variance = 13/10,
//! // α = 3/2 · (1 − 8/13) = 15/26.
//! assert_eq!(cronbach_alpha(&t)?, q(15, 26));
//! assert_eq!(item_difficulty(&t)?, vec![q(3, 5), q(3, 5), q(1, 5)]);
//! # Ok::<(), symplex::prelude::SymplexError>(())
//! ```
//!
//! # Conventions
//!
//! * A **response matrix** is a [`RatingTable`] read as *respondents ×
//!   items*: rows are respondents (`n`), columns are items (`k`) — the
//!   same type [`super::agreement`] reads as items × raters.  The
//!   functions here need every cell present; [`cronbach_alpha_complete`]
//!   drops incomplete respondents first.
//! * A **contingency table** is a `&[Vec<Q>]` of counts, as in
//!   [`super::hypothesis`] (build one with
//!   [`counts`](super::hypothesis::counts)).
//! * Two **ordinal variables** are two equally long slices of `Q`.
//! * Tests return a [`TestResult`]: the χ² tail is `Γ(df/2, x/2)/Γ(df/2)`,
//!   the normal tail `erfc(|z|/√2)`, the Student tail a regularized
//!   incomplete beta — exact expressions evaluated by
//!   [`TestResult::p_value_f64`], as in [`super::hypothesis`].

use num_bigint::BigInt;
use num_traits::{One, Signed, Zero};

use super::agreement::{RatingTable, confusion_matrix};
use super::data::{self, Ddof, Q};
use super::hypothesis::{Alternative, TestResult};
use crate::api::context::Context;
use crate::api::expr::Ex;
use crate::base::errors::SymplexError;
use crate::base::interval::Interval;
use crate::output::codegen::numeric_rt::erfcinv;

// ═══════════════════════════════════════════════════════════════════════════
// Small helpers
// ═══════════════════════════════════════════════════════════════════════════

fn invalid(op: &'static str, reason: impl Into<String>) -> SymplexError {
    SymplexError::invalid_argument(op, reason)
}

fn qi(n: i64) -> Q {
    Q::from_integer(BigInt::from(n))
}

fn qu(n: usize) -> Q {
    Q::from_integer(BigInt::from(n))
}

fn ex(ctx: &Context, q: &Q) -> Ex {
    ctx.from_ratio(q.clone())
}

fn exu(ctx: &Context, n: usize) -> Ex {
    ex(ctx, &qu(n))
}

fn sum(values: impl IntoIterator<Item = Q>) -> Q {
    values.into_iter().fold(Q::zero(), |acc, x| acc + x)
}

fn square(x: &Q) -> Q {
    x * x
}

fn q_to_f64(q: &Q) -> f64 {
    data::to_f64(std::slice::from_ref(q))
        .first()
        .copied()
        .unwrap_or(f64::NAN)
}

/// `Φ⁻¹(1 − α)`: the upper `α` quantile of the standard normal.
fn norm_isf(alpha: f64) -> f64 {
    std::f64::consts::SQRT_2 * erfcinv(2.0 * alpha)
}

fn check_confidence(op: &'static str, confidence: f64) -> Result<(), SymplexError> {
    if confidence > 0.0 && confidence < 1.0 {
        Ok(())
    } else {
        Err(invalid(
            op,
            format!("the confidence level must lie strictly between 0 and 1, got {confidence}"),
        ))
    }
}

fn check_same_len(op: &'static str, x: &[Q], y: &[Q]) -> Result<(), SymplexError> {
    if x.len() != y.len() {
        return Err(invalid(
            op,
            format!(
                "the two variables must have the same length ({} and {})",
                x.len(),
                y.len()
            ),
        ));
    }
    Ok(())
}

fn check_dichotomous<'a>(
    op: &'static str,
    values: impl IntoIterator<Item = &'a Q>,
    what: &str,
) -> Result<(), SymplexError> {
    for v in values {
        if !(v.is_zero() || v.is_one()) {
            return Err(invalid(
                op,
                format!("{what} must be scored 0 or 1, found {v}"),
            ));
        }
    }
    Ok(())
}

/// Pearson's `r` of two columns, with this module's error message when one
/// of them is constant.
fn pearson_of(
    ctx: &Context,
    op: &'static str,
    x: &[Q],
    y: &[Q],
    what: &str,
) -> Result<Ex, SymplexError> {
    data::pearson(ctx, x, y).map_err(|_| {
        invalid(
            op,
            format!("{what} is constant, so its correlation is undefined"),
        )
    })
}

/// `P(χ²_df ≥ x) = Γ(df/2, x/2) / Γ(df/2)` as an expression (`1` for `x ≤ 0`).
fn chi_squared_sf(ctx: &Context, df: usize, x: &Q) -> Ex {
    if !x.is_positive() {
        return ctx.one();
    }
    let half_df = exu(ctx, df) / ctx.int(2);
    (ex(ctx, x) / ctx.int(2)).uppergamma(&half_df) / half_df.gamma()
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

/// Whether a statistic with the sign of `num` lies in the alternative's tail.
fn in_tail(num_is_negative: bool, num_is_positive: bool, alt: Alternative) -> bool {
    match alt {
        Alternative::Greater | Alternative::TwoSided => !num_is_negative,
        Alternative::Less => !num_is_positive,
    }
}

/// The test of a statistic `z = num / √var` (`num`, `var > 0` rational)
/// against the standard normal: `P(|Z| ≥ |z|) = erfc(|z|/√2)`.
fn normal_test(ctx: &Context, num: &Q, var: &Q, alt: Alternative) -> TestResult {
    let statistic = (ex(ctx, num) / ex(ctx, var).sqrt()).simplify();
    let half_z2 = square(num) / var / qi(2);
    let two_sided = ex(ctx, &half_z2).sqrt().erfc();
    let tail = in_tail(num.is_negative(), num.is_positive(), alt);
    TestResult {
        statistic,
        p_value: one_sided_from_symmetric(ctx, two_sided, tail, alt),
        df: None,
        alternative: alt,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Response matrices
// ═══════════════════════════════════════════════════════════════════════════

/// The respondents × items scores of a complete table.
fn score_matrix(op: &'static str, table: &RatingTable) -> Result<Vec<Vec<Q>>, SymplexError> {
    table
        .rows()
        .iter()
        .enumerate()
        .map(|(i, r)| {
            r.iter()
                .cloned()
                .map(|c| {
                    c.ok_or_else(|| {
                        invalid(
                            op,
                            format!(
                                "respondent {i} has a missing item score (drop incomplete rows first)"
                            ),
                        )
                    })
                })
                .collect()
        })
        .collect()
}

/// The rows of a table with every cell present.
fn complete_score_rows(table: &RatingTable) -> Vec<Vec<Q>> {
    table
        .rows()
        .iter()
        .filter_map(|r| r.iter().cloned().collect::<Option<Vec<Q>>>())
        .collect()
}

/// The columns (items) of a respondents × items matrix.
fn columns(rows: &[Vec<Q>]) -> Vec<Vec<Q>> {
    let k = rows.first().map_or(0, Vec::len);
    (0..k)
        .map(|j| rows.iter().map(|r| r[j].clone()).collect())
        .collect()
}

/// The total score of every respondent.
fn row_sums(rows: &[Vec<Q>]) -> Vec<Q> {
    rows.iter().map(|r| data::sum(r)).collect()
}

/// `(n, k)` after checking there are at least two respondents and
/// `min_items` items.
fn check_scale(
    op: &'static str,
    rows: &[Vec<Q>],
    min_items: usize,
) -> Result<(usize, usize), SymplexError> {
    let n = rows.len();
    let k = rows.first().map_or(0, Vec::len);
    if n < 2 {
        return Err(invalid(
            op,
            format!("needs at least two complete respondents (rows), got {n}"),
        ));
    }
    if k < min_items {
        return Err(invalid(
            op,
            format!("needs at least {min_items} items (columns), got {k}"),
        ));
    }
    Ok((n, k))
}

/// Cronbach's α of a complete respondents × items matrix.
fn alpha_of_rows(op: &'static str, rows: &[Vec<Q>]) -> Result<Q, SymplexError> {
    let (_, k) = check_scale(op, rows, 2)?;
    let total_var = data::variance(&row_sums(rows), Ddof::Sample)?;
    if total_var.is_zero() {
        return Err(invalid(
            op,
            "every respondent has the same total score (zero total variance), α is undefined",
        ));
    }
    let item_var = columns(rows)
        .iter()
        .map(|c| data::variance(c, Ddof::Sample))
        .collect::<Result<Vec<Q>, SymplexError>>()?;
    Ok(qu(k) / qu(k - 1) * (Q::one() - sum(item_var) / total_var))
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. Internal consistency
// ═══════════════════════════════════════════════════════════════════════════

/// Cronbach's α (Cronbach 1951) of a respondents × items matrix:
///
/// `α = k/(k−1) · (1 − Σᵢ s²ᵢ / s²_X)`
///
/// with `k` items, the sample variance `s²ᵢ` of item `i` and the sample
/// variance `s²_X` of the total scores (the divisor cancels, so the
/// population variances give the same α).  Every cell must be present;
/// use [`cronbach_alpha_complete`] to drop incomplete respondents.
/// `pingouin.cronbach_alpha`; equals `k/(k−1)·(1 − trace(S)/1ᵀS1)` for
/// the item covariance matrix `S`.
///
/// ```
/// use symplex::linprog::q;
/// use symplex::stats::agreement::RatingTable;
/// use symplex::stats::reliability::cronbach_alpha;
///
/// let t = RatingTable::from_i64(&[
///     &[4, 3, 5, 4], &[2, 2, 3, 1], &[5, 4, 4, 5], &[3, 3, 2, 3],
///     &[1, 2, 1, 2], &[4, 4, 5, 3], &[2, 1, 2, 2], &[5, 5, 4, 4],
/// ])?;
/// // Fraction: 4/3 · (1 − (55/7) / (180/7)) = 25/27; numpy 0.9259259259259258
/// assert_eq!(cronbach_alpha(&t)?, q(25, 27));
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for a missing cell, fewer than two
/// respondents or items, or a constant total score.
pub fn cronbach_alpha(table: &RatingTable) -> Result<Q, SymplexError> {
    const OP: &str = "cronbach_alpha";
    alpha_of_rows(OP, &score_matrix(OP, table)?)
}

/// Cronbach's α over the respondents with every item answered
/// (list-wise deletion): rows with a missing cell are dropped, then
/// [`cronbach_alpha`] is computed on the rest.
///
/// ```
/// use symplex::linprog::q;
/// use symplex::stats::agreement::RatingTable;
/// use symplex::stats::reliability::cronbach_alpha_complete;
///
/// let t = RatingTable::from_i64_missing(&[
///     &[Some(1), Some(1), Some(1)],
///     &[Some(1), Some(1), Some(0)],
///     &[Some(1), None, Some(0)],
///     &[Some(1), Some(0), Some(0)],
///     &[Some(0), Some(1), Some(0)],
///     &[Some(0), Some(0), Some(0)],
/// ])?;
/// // The third respondent is dropped; the remaining five give 15/26 (Fraction).
/// assert_eq!(cronbach_alpha_complete(&t)?, q(15, 26));
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// As [`cronbach_alpha`], on the complete rows.
pub fn cronbach_alpha_complete(table: &RatingTable) -> Result<Q, SymplexError> {
    alpha_of_rows("cronbach_alpha_complete", &complete_score_rows(table))
}

/// The `k(k−1)/2` inter-item Pearson correlations, `(i, j)` with `i < j`
/// in lexicographic order.
fn inter_item_correlations(
    ctx: &Context,
    op: &'static str,
    rows: &[Vec<Q>],
) -> Result<Vec<Ex>, SymplexError> {
    let (_, k) = check_scale(op, rows, 2)?;
    let items = columns(rows);
    let mut out = Vec::with_capacity(k * (k - 1) / 2);
    for i in 0..k {
        for j in i + 1..k {
            out.push(pearson_of(
                ctx,
                op,
                &items[i],
                &items[j],
                &format!("item {i} or item {j}"),
            )?);
        }
    }
    Ok(out)
}

fn mean_ex(ctx: &Context, values: Vec<Ex>) -> Ex {
    let m = values.len();
    let total = values.into_iter().fold(ctx.zero(), |acc, r| acc + r);
    total / exu(ctx, m)
}

/// The mean of the `k(k−1)/2` inter-item Pearson correlations, as an
/// exact expression (each `r` is a rational over a square root).
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for a missing cell, fewer than two
/// respondents or items, or a constant item.
pub fn average_inter_item_correlation(
    ctx: &Context,
    table: &RatingTable,
) -> Result<Ex, SymplexError> {
    const OP: &str = "average_inter_item_correlation";
    let rows = score_matrix(OP, table)?;
    Ok(mean_ex(ctx, inter_item_correlations(ctx, OP, &rows)?))
}

/// The standardized α: Cronbach's α of the standardized items, a function
/// of the mean inter-item correlation `r̄` alone,
///
/// `α_std = k r̄ / (1 + (k − 1) r̄)`
///
/// (the Spearman–Brown formula applied to `r̄`).  Returns an exact
/// expression since each `r` involves a square root.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::agreement::RatingTable;
/// use symplex::stats::reliability::standardized_alpha;
///
/// let ctx = Context::new();
/// let t = RatingTable::from_i64(&[
///     &[4, 3, 5, 4], &[2, 2, 3, 1], &[5, 4, 4, 5], &[3, 3, 2, 3],
///     &[1, 2, 1, 2], &[4, 4, 5, 3], &[2, 1, 2, 2], &[5, 5, 4, 4],
/// ])?;
/// // numpy: r̄ = 0.760452911806828, 4r̄/(1 + 3r̄) = 0.926997592306080
/// assert!((standardized_alpha(&ctx, &t)?.eval_f64()? - 0.926_997_592_306_080).abs() < 1e-12);
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// As [`average_inter_item_correlation`].
pub fn standardized_alpha(ctx: &Context, table: &RatingTable) -> Result<Ex, SymplexError> {
    const OP: &str = "standardized_alpha";
    let rows = score_matrix(OP, table)?;
    let k = rows[0].len();
    let r = mean_ex(ctx, inter_item_correlations(ctx, OP, &rows)?);
    Ok(spearman_brown(ctx, &r, k))
}

/// The Kuder–Richardson formula 20 (Kuder & Richardson 1937) for
/// dichotomous (0/1) items:
///
/// `KR-20 = k/(k−1) · (1 − Σᵢ pᵢ qᵢ / σ²_X)`
///
/// with the proportion correct `pᵢ`, `qᵢ = 1 − pᵢ`, and the *population*
/// variance `σ²_X` of the total scores — `pᵢqᵢ` is the population variance
/// of item `i`, so with this pairing KR-20 equals [`cronbach_alpha`]
/// exactly on 0/1 data (a sample `s²_X` would make it smaller by the
/// factor `(n−1)/n` inside the ratio).
///
/// ```
/// use symplex::linprog::q;
/// use symplex::stats::agreement::RatingTable;
/// use symplex::stats::reliability::{cronbach_alpha, kr20};
///
/// let t = RatingTable::from_i64(&[
///     &[1, 1, 1, 1, 1], &[1, 1, 1, 0, 1], &[1, 1, 0, 1, 0], &[1, 0, 1, 0, 0], &[0, 1, 0, 1, 1],
///     &[1, 0, 0, 0, 0], &[0, 0, 1, 0, 0], &[1, 1, 1, 1, 0], &[0, 0, 0, 0, 1], &[1, 1, 0, 1, 1],
/// ])?;
/// // Fraction: 5/4 · (1 − (21/100 + 6/25 + 3/4) / (9/5)) = 95/196
/// assert_eq!(kr20(&t)?, q(95, 196));
/// assert_eq!(kr20(&t)?, cronbach_alpha(&t)?);
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for a missing cell, a score other
/// than 0 or 1, fewer than two respondents or items, or a constant total.
pub fn kr20(table: &RatingTable) -> Result<Q, SymplexError> {
    const OP: &str = "kr20";
    let rows = score_matrix(OP, table)?;
    check_dichotomous(OP, rows.iter().flatten(), "every item")?;
    let (_, k) = check_scale(OP, &rows, 2)?;
    let total_var = data::variance(&row_sums(&rows), Ddof::Population)?;
    if total_var.is_zero() {
        return Err(invalid(
            OP,
            "every respondent has the same total score (zero total variance), KR-20 is undefined",
        ));
    }
    let pq = columns(&rows)
        .iter()
        .map(|c| {
            let p = data::mean(c)?;
            Ok(&p * (Q::one() - &p))
        })
        .collect::<Result<Vec<Q>, SymplexError>>()?;
    Ok(qu(k) / qu(k - 1) * (Q::one() - sum(pq) / total_var))
}

/// How the items are split into two halves for [`split_half`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SplitHalf {
    /// Items `0, 2, 4, …` (the 1st, 3rd, 5th, …) against items `1, 3, 5, …`.
    OddEven,
    /// The first `⌊k/2⌋` items against the remaining `⌈k/2⌉`.
    FirstLast,
    /// One flag per item: `true` puts the item in the first half.
    Custom(Vec<bool>),
}

impl SplitHalf {
    /// The membership mask (`true` = first half) for `k` items.
    fn mask(&self, op: &'static str, k: usize) -> Result<Vec<bool>, SymplexError> {
        let mask: Vec<bool> = match self {
            SplitHalf::OddEven => (0..k).map(|i| i % 2 == 0).collect(),
            SplitHalf::FirstLast => (0..k).map(|i| i < k / 2).collect(),
            SplitHalf::Custom(m) => {
                if m.len() != k {
                    return Err(invalid(
                        op,
                        format!(
                            "the split has {} flags but the table has {k} items",
                            m.len()
                        ),
                    ));
                }
                m.clone()
            }
        };
        let first = mask.iter().filter(|&&b| b).count();
        if first == 0 || first == k {
            return Err(invalid(op, "both halves must contain at least one item"));
        }
        Ok(mask)
    }
}

/// The two half-scale totals of every respondent (`mask[j]` = item `j`
/// belongs to the first half).
fn split_totals(rows: &[Vec<Q>], mask: &[bool]) -> (Vec<Q>, Vec<Q>) {
    rows.iter()
        .map(|r| {
            let mut first = Q::zero();
            let mut second = Q::zero();
            for (x, &in_first) in r.iter().zip(mask) {
                if in_first {
                    first += x;
                } else {
                    second += x;
                }
            }
            (first, second)
        })
        .unzip()
}

/// Pearson's `r` between the two half-scale totals (no prophecy
/// correction), as an exact expression.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for a missing cell, fewer than two
/// respondents or items, an invalid split, or a constant half.
pub fn split_half_correlation(
    ctx: &Context,
    table: &RatingTable,
    split: &SplitHalf,
) -> Result<Ex, SymplexError> {
    const OP: &str = "split_half_correlation";
    let rows = score_matrix(OP, table)?;
    let (_, k) = check_scale(OP, &rows, 2)?;
    let mask = split.mask(OP, k)?;
    let (h1, h2) = split_totals(&rows, &mask);
    pearson_of(ctx, OP, &h1, &h2, "one of the half-scale totals")
}

/// Split-half reliability: Pearson's `r` between the two half-scale
/// totals, stepped up to the full length by the Spearman–Brown prophecy
/// `2r / (1 + r)`.  Returns an exact expression.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::linprog::q;
/// use symplex::stats::agreement::RatingTable;
/// use symplex::stats::reliability::{split_half, SplitHalf};
///
/// let ctx = Context::new();
/// let t = RatingTable::from_i64(&[
///     &[1, 1, 1, 1, 1], &[1, 1, 1, 0, 1], &[1, 1, 0, 1, 0], &[1, 0, 1, 0, 0], &[0, 1, 0, 1, 1],
///     &[1, 0, 0, 0, 0], &[0, 0, 1, 0, 0], &[1, 1, 1, 1, 0], &[0, 0, 0, 0, 1], &[1, 1, 0, 1, 1],
/// ])?;
/// // numpy: r(items {0, 3, 4}, items {1, 2}) = 0.523809523809524 = 11/21, 2r/(1+r) = 11/16
/// let rel = split_half(&ctx, &t, &SplitHalf::Custom(vec![true, false, false, true, true]))?;
/// assert_eq!(rel.simplify(), ctx.from_ratio(q(11, 16)));
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// As [`split_half_correlation`].
pub fn split_half(
    ctx: &Context,
    table: &RatingTable,
    split: &SplitHalf,
) -> Result<Ex, SymplexError> {
    let r = split_half_correlation(ctx, table, split)?;
    Ok(spearman_brown(ctx, &r, 2))
}

/// The Spearman–Brown prophecy formula: the reliability of a test
/// lengthened by the factor `k`,
///
/// `ρ_k = k ρ / (1 + (k − 1) ρ)`
///
/// (`k = 2` steps a half-test correlation up to the full test).
///
/// ```
/// use symplex::prelude::*;
/// let ctx = Context::new();
/// // A half-test correlation of 0.6 predicts 2·0.6/1.6 = 0.75 for the full test.
/// let r = symplex::stats::reliability::spearman_brown(&ctx, &ctx.rational(3, 5), 2);
/// assert_eq!(r.simplify(), ctx.rational(3, 4));
/// ```
pub fn spearman_brown(ctx: &Context, r: &Ex, k: usize) -> Ex {
    let kq = exu(ctx, k);
    (&kq * r / (ctx.one() + (kq - ctx.one()) * r)).simplify()
}

/// Cronbach's α of the scale with each item removed in turn: entry `j`
/// is the α of the `k − 1` items other than `j`.  An entry larger than
/// the full-scale α flags an item that lowers internal consistency.
///
/// ```
/// use symplex::linprog::q;
/// use symplex::stats::agreement::RatingTable;
/// use symplex::stats::reliability::alpha_if_deleted;
///
/// let t = RatingTable::from_i64(&[
///     &[4, 3, 5, 4], &[2, 2, 3, 1], &[5, 4, 4, 5], &[3, 3, 2, 3],
///     &[1, 2, 1, 2], &[4, 4, 5, 3], &[2, 1, 2, 2], &[5, 5, 4, 4],
/// ])?;
/// // Fraction re-implementation: α without item 0 = 52/61, …
/// assert_eq!(alpha_if_deleted(&t)?, vec![q(52, 61), q(65, 72), q(198, 211), q(201, 220)]);
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for a missing cell, fewer than two
/// respondents or three items, or a constant reduced total.
pub fn alpha_if_deleted(table: &RatingTable) -> Result<Vec<Q>, SymplexError> {
    const OP: &str = "alpha_if_deleted";
    let rows = score_matrix(OP, table)?;
    let (_, k) = check_scale(OP, &rows, 3)?;
    (0..k)
        .map(|j| {
            let reduced: Vec<Vec<Q>> = rows
                .iter()
                .map(|r| {
                    r.iter()
                        .enumerate()
                        .filter(|&(i, _)| i != j)
                        .map(|(_, x)| x.clone())
                        .collect()
                })
                .collect();
            alpha_of_rows(OP, &reduced)
        })
        .collect()
}

/// Guttman's λ₂ (Guttman 1945), a lower bound to reliability that is
/// never below α:
///
/// `λ₂ = (s²_X − Σᵢ s²ᵢ + √(k/(k−1) · Σᵢ≠ⱼ s²ᵢⱼ)) / s²_X`
///
/// with the item covariances `sᵢⱼ` (sample divisor; it cancels).  Exact
/// expression.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::agreement::RatingTable;
/// use symplex::stats::reliability::guttman_lambda2;
///
/// let ctx = Context::new();
/// let t = RatingTable::from_i64(&[
///     &[4, 3, 5, 4], &[2, 2, 3, 1], &[5, 4, 4, 5], &[3, 3, 2, 3],
///     &[1, 2, 1, 2], &[4, 4, 5, 3], &[2, 1, 2, 2], &[5, 5, 4, 4],
/// ])?;
/// // numpy: (180/7 − 55/7 + √(766/21)) / (180/7) = 0.929315917921957
/// assert!((guttman_lambda2(&ctx, &t)?.eval_f64()? - 0.929_315_917_921_957).abs() < 1e-12);
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for a missing cell, fewer than two
/// respondents or items, or a constant total score.
pub fn guttman_lambda2(ctx: &Context, table: &RatingTable) -> Result<Ex, SymplexError> {
    const OP: &str = "guttman_lambda2";
    let rows = score_matrix(OP, table)?;
    let (_, k) = check_scale(OP, &rows, 2)?;
    let total_var = data::variance(&row_sums(&rows), Ddof::Sample)?;
    if total_var.is_zero() {
        return Err(invalid(
            OP,
            "every respondent has the same total score (zero total variance), λ₂ is undefined",
        ));
    }
    let items = columns(&rows);
    let mut trace = Q::zero();
    let mut off = Q::zero();
    for i in 0..k {
        trace += data::variance(&items[i], Ddof::Sample)?;
        for j in 0..k {
            if i != j {
                off += square(&data::covariance(&items[i], &items[j], Ddof::Sample)?);
            }
        }
    }
    let root = ex(ctx, &(qu(k) / qu(k - 1) * off)).sqrt();
    Ok(((ex(ctx, &(&total_var - trace)) + root) / ex(ctx, &total_var)).simplify())
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Item analysis
// ═══════════════════════════════════════════════════════════════════════════

/// The total score `Σⱼ xᵢⱼ` of every respondent.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for a missing cell.
pub fn total_scores(table: &RatingTable) -> Result<Vec<Q>, SymplexError> {
    Ok(row_sums(&score_matrix("total_scores", table)?))
}

/// Item difficulty: the mean score of every item — for 0/1 items the
/// proportion of respondents answering correctly (`p`-value in classical
/// test theory; higher means *easier*).
///
/// ```
/// use symplex::linprog::q;
/// use symplex::stats::agreement::RatingTable;
/// use symplex::stats::reliability::item_difficulty;
///
/// let t = RatingTable::from_i64(&[
///     &[1, 1, 1, 1, 1], &[1, 1, 1, 0, 1], &[1, 1, 0, 1, 0], &[1, 0, 1, 0, 0], &[0, 1, 0, 1, 1],
///     &[1, 0, 0, 0, 0], &[0, 0, 1, 0, 0], &[1, 1, 1, 1, 0], &[0, 0, 0, 0, 1], &[1, 1, 0, 1, 1],
/// ])?;
/// assert_eq!(item_difficulty(&t)?, vec![q(7, 10), q(3, 5), q(1, 2), q(1, 2), q(1, 2)]);
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for a missing cell.
pub fn item_difficulty(table: &RatingTable) -> Result<Vec<Q>, SymplexError> {
    let rows = score_matrix("item_difficulty", table)?;
    columns(&rows).iter().map(|c| data::mean(c)).collect()
}

/// The upper and lower groups of the thirds rule: `g = ⌊n/3⌋`
/// respondents each.  Respondents are ordered by total score with a
/// stable sort, so ties at a boundary are resolved by row order (the
/// earlier row is taken first into either group).
fn extreme_groups(
    op: &'static str,
    rows: &[Vec<Q>],
) -> Result<(Vec<usize>, Vec<usize>), SymplexError> {
    let n = rows.len();
    let g = n / 3;
    if g == 0 {
        return Err(invalid(
            op,
            format!("the thirds rule needs at least three respondents, got {n}"),
        ));
    }
    let totals = row_sums(rows);
    let mut asc: Vec<usize> = (0..n).collect();
    asc.sort_by(|&a, &b| totals[a].cmp(&totals[b]));
    let mut desc: Vec<usize> = (0..n).collect();
    desc.sort_by(|&a, &b| totals[b].cmp(&totals[a]));
    Ok((desc[..g].to_vec(), asc[..g].to_vec()))
}

fn group_mean(rows: &[Vec<Q>], group: &[usize], j: usize) -> Result<Q, SymplexError> {
    let v: Vec<Q> = group.iter().map(|&i| rows[i][j].clone()).collect();
    data::mean(&v)
}

/// The item discrimination index `D` (Kelley 1939; Ebel 1965): the mean
/// score of the item in the upper group minus that in the lower group,
/// for 0/1 items the difference of the proportions correct, in `[−1, 1]`.
///
/// **Thirds rule.**  Respondents are ranked by total score; the upper
/// group is the `⌊n/3⌋` with the highest totals and the lower group the
/// `⌊n/3⌋` with the lowest.  Ties at a boundary are broken by row order
/// (a stable sort in each direction: among equal totals the earlier row
/// enters the group first).
///
/// ```
/// use symplex::linprog::q;
/// use symplex::stats::agreement::RatingTable;
/// use symplex::stats::reliability::item_discrimination_index;
///
/// let t = RatingTable::from_i64(&[
///     &[1, 1, 1, 1, 1], &[1, 1, 1, 0, 1], &[1, 1, 0, 1, 0], &[1, 0, 1, 0, 0], &[0, 1, 0, 1, 1],
///     &[1, 0, 0, 0, 0], &[0, 0, 1, 0, 0], &[1, 1, 1, 1, 0], &[0, 0, 0, 0, 1], &[1, 1, 0, 1, 1],
/// ])?;
/// // Fraction re-implementation: upper = rows {0, 1, 7}, lower = rows {5, 6, 8}.
/// assert_eq!(item_discrimination_index(&t)?, vec![q(2, 3), q(1, 1), q(2, 3), q(2, 3), q(1, 3)]);
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for a missing cell or fewer than three
/// respondents.
pub fn item_discrimination_index(table: &RatingTable) -> Result<Vec<Q>, SymplexError> {
    const OP: &str = "item_discrimination_index";
    let rows = score_matrix(OP, table)?;
    let (upper, lower) = extreme_groups(OP, &rows)?;
    let k = rows[0].len();
    (0..k)
        .map(|j| Ok(group_mean(&rows, &upper, j)? - group_mean(&rows, &lower, j)?))
        .collect()
}

/// The point-biserial correlation of a 0/1 item with a continuous score
/// (usually the total): Pearson's `r` of the two, equivalently
/// `(M₁ − M₀)/σ_total · √(p q)` with the group means `M₁`, `M₀`.  Exact
/// expression.  `scipy.stats.pointbiserialr(item, total)`.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::linprog::q;
/// use symplex::stats::data::from_i64;
/// use symplex::stats::reliability::point_biserial;
///
/// let ctx = Context::new();
/// let item = from_i64(&[1, 1, 0, 1, 0, 0, 1, 1, 0, 0]);
/// let total = from_i64(&[5, 4, 3, 2, 3, 1, 1, 4, 1, 4]);
/// // scipy: pointbiserialr(item, total).statistic = 0.285714285714286 (= 2/7)
/// assert_eq!(point_biserial(&ctx, &item, &total)?, ctx.from_ratio(q(2, 7)));
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for unequal lengths, fewer than two
/// respondents, a score other than 0 or 1, or a constant item or score.
pub fn point_biserial(ctx: &Context, item: &[Q], total: &[Q]) -> Result<Ex, SymplexError> {
    const OP: &str = "point_biserial";
    check_same_len(OP, item, total)?;
    if item.len() < 2 {
        return Err(invalid(OP, "needs at least two respondents"));
    }
    check_dichotomous(OP, item, "the item")?;
    pearson_of(ctx, OP, item, total, "the item or the score")
}

/// Pearson's `r` of every item with the total score (uncorrected: the
/// item is part of the total).  Exact expressions.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for a missing cell, fewer than two
/// respondents or items, or a constant item or total.
pub fn item_total_correlation(ctx: &Context, table: &RatingTable) -> Result<Vec<Ex>, SymplexError> {
    const OP: &str = "item_total_correlation";
    let rows = score_matrix(OP, table)?;
    check_scale(OP, &rows, 2)?;
    let totals = row_sums(&rows);
    columns(&rows)
        .iter()
        .enumerate()
        .map(|(j, c)| pearson_of(ctx, OP, c, &totals, &format!("item {j} or the total")))
        .collect()
}

/// The corrected item–total correlation: Pearson's `r` of every item
/// with the total of the *other* items (`total − item`), as used in item
/// analysis to avoid the item's correlation with itself.  Exact
/// expressions.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::linprog::q;
/// use symplex::stats::agreement::RatingTable;
/// use symplex::stats::reliability::corrected_item_total_correlation;
///
/// let ctx = Context::new();
/// let t = RatingTable::from_i64(&[
///     &[4, 3, 5, 4], &[2, 2, 3, 1], &[5, 4, 4, 5], &[3, 3, 2, 3],
///     &[1, 2, 1, 2], &[4, 4, 5, 3], &[2, 1, 2, 2], &[5, 5, 4, 4],
/// ])?;
/// // numpy: corrcoef(item 1, total − item 1) = 0.833333333333333 (= 5/6)
/// let r = corrected_item_total_correlation(&ctx, &t)?;
/// assert_eq!(r[1], ctx.from_ratio(q(5, 6)));
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for a missing cell, fewer than two
/// respondents or items, or a constant item or remainder.
pub fn corrected_item_total_correlation(
    ctx: &Context,
    table: &RatingTable,
) -> Result<Vec<Ex>, SymplexError> {
    const OP: &str = "corrected_item_total_correlation";
    let rows = score_matrix(OP, table)?;
    check_scale(OP, &rows, 2)?;
    let totals = row_sums(&rows);
    columns(&rows)
        .iter()
        .enumerate()
        .map(|(j, c)| {
            let rest: Vec<Q> = totals.iter().zip(c).map(|(t, x)| t - x).collect();
            pearson_of(
                ctx,
                OP,
                c,
                &rest,
                &format!("item {j} or the rest of the scale"),
            )
        })
        .collect()
}

/// The classical item-analysis summary of one item.
#[derive(Clone, Debug, PartialEq)]
pub struct ItemSummary {
    /// Mean score ([`item_difficulty`]).
    pub difficulty: Q,
    /// Upper-third minus lower-third mean ([`item_discrimination_index`]).
    pub discrimination: Q,
    /// Pearson's `r` with the total score; `None` if the item or the total
    /// is constant.
    pub item_total: Option<Ex>,
    /// Pearson's `r` with the total of the other items; `None` if either
    /// is constant.
    pub corrected_item_total: Option<Ex>,
    /// Cronbach's α without this item; `None` with fewer than three items
    /// or when that α is undefined.
    pub alpha_if_deleted: Option<Q>,
}

/// One [`ItemSummary`] per item: difficulty, discrimination index,
/// item–total and corrected item–total correlations, α-if-deleted.  The
/// correlations and α-if-deleted are `None` where undefined instead of
/// failing the whole summary.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for a missing cell, fewer than three
/// respondents or two items.
pub fn item_response_summary(
    ctx: &Context,
    table: &RatingTable,
) -> Result<Vec<ItemSummary>, SymplexError> {
    const OP: &str = "item_response_summary";
    let rows = score_matrix(OP, table)?;
    check_scale(OP, &rows, 2)?;
    let (upper, lower) = extreme_groups(OP, &rows)?;
    let totals = row_sums(&rows);
    let deleted: Option<Vec<Q>> = alpha_if_deleted(table).ok();
    columns(&rows)
        .iter()
        .enumerate()
        .map(|(j, c)| {
            let rest: Vec<Q> = totals.iter().zip(c).map(|(t, x)| t - x).collect();
            Ok(ItemSummary {
                difficulty: data::mean(c)?,
                discrimination: group_mean(&rows, &upper, j)? - group_mean(&rows, &lower, j)?,
                item_total: data::pearson(ctx, c, &totals).ok(),
                corrected_item_total: data::pearson(ctx, c, &rest).ok(),
                alpha_if_deleted: deleted.as_ref().and_then(|d| d.get(j).cloned()),
            })
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Agreement extras: κ inference, κ_max, Cochran's Q
// ═══════════════════════════════════════════════════════════════════════════

/// Cohen's κ with the ingredients of its large-sample inference, from a
/// `k × k` confusion matrix.
struct KappaMoments {
    kappa: Q,
    /// Large-sample variance of κ̂ (Fleiss, Cohen & Everitt 1969).
    var: Q,
    /// Variance of κ̂ under `H₀: κ = 0`.
    var0: Q,
    /// κ_max given the marginals.
    max: Q,
}

fn kappa_moments(op: &'static str, table: &[Vec<usize>]) -> Result<KappaMoments, SymplexError> {
    let k = table.len();
    if k == 0 || table.iter().any(|r| r.len() != k) {
        return Err(invalid(
            op,
            "the confusion matrix must be square and non-empty",
        ));
    }
    let n: usize = table.iter().flatten().sum();
    if n == 0 {
        return Err(invalid(op, "the confusion matrix is empty"));
    }
    let nq = qu(n);
    let p: Vec<Vec<Q>> = table
        .iter()
        .map(|r| r.iter().map(|&c| qu(c) / &nq).collect())
        .collect();
    let pr: Vec<Q> = p.iter().map(|r| sum(r.iter().cloned())).collect();
    let pc: Vec<Q> = (0..k)
        .map(|j| sum(p.iter().map(|r| r[j].clone())))
        .collect();
    let po = sum((0..k).map(|i| p[i][i].clone()));
    let pe = sum(pr.iter().zip(&pc).map(|(r, c)| r * c));
    let one_minus_pe = Q::one() - &pe;
    if one_minus_pe.is_zero() {
        return Err(invalid(
            op,
            "the expected agreement is 1 (a single category), κ is undefined",
        ));
    }
    let kappa = (&po - &pe) / &one_minus_pe;
    let one_minus_k = Q::one() - &kappa;
    let term_a = sum((0..k).map(|i| {
        let d = Q::one() - (&pr[i] + &pc[i]) * &one_minus_k;
        &p[i][i] * square(&d)
    }));
    let mut term_b = Q::zero();
    for i in 0..k {
        for j in 0..k {
            if i != j {
                term_b += &p[i][j] * square(&(&pc[i] + &pr[j]));
            }
        }
    }
    term_b *= square(&one_minus_k);
    let term_c = square(&(&kappa - &pe * &one_minus_k));
    let scale = square(&one_minus_pe) * &nq;
    let var = (term_a + term_b - term_c) / &scale;
    let marg = sum(pr.iter().zip(&pc).map(|(r, c)| r * c * (r + c)));
    let var0 = (&pe + square(&pe) - marg) / scale;
    let p_max = sum(pr.iter().zip(&pc).map(|(r, c)| r.min(c).clone()));
    let max = (p_max - &pe) / one_minus_pe;
    Ok(KappaMoments {
        kappa,
        var,
        var0,
        max,
    })
}

/// The sorted distinct values of two raters' ratings.
fn observed_categories(a: &[Q], b: &[Q]) -> Vec<Q> {
    let mut v: Vec<Q> = a.iter().chain(b).cloned().collect();
    v.sort();
    v.dedup();
    v
}

fn confusion_of(op: &'static str, a: &[Q], b: &[Q]) -> Result<Vec<Vec<usize>>, SymplexError> {
    check_same_len(op, a, b)?;
    if a.is_empty() {
        return Err(invalid(op, "needs at least one item"));
    }
    confusion_matrix(a, b, &observed_categories(a, b))
}

/// Cohen's κ with its large-sample standard error and a normal-theory
/// confidence interval.
#[derive(Clone, Debug, PartialEq)]
pub struct KappaCi {
    /// The coefficient, exact.
    pub kappa: Q,
    /// The large-sample variance of κ̂ (Fleiss, Cohen & Everitt 1969),
    /// exact.
    pub variance: Q,
    /// The standard error `√variance`, exact.
    pub se: Ex,
    /// The normal-theory interval `κ ∓ z_{α/2} · se`.
    pub ci: Interval<f64>,
    /// The confidence level `ci` refers to.
    pub confidence: f64,
}

fn kappa_ci_of(
    ctx: &Context,
    op: &'static str,
    table: &[Vec<usize>],
    confidence: f64,
) -> Result<KappaCi, SymplexError> {
    check_confidence(op, confidence)?;
    let m = kappa_moments(op, table)?;
    if m.var.is_negative() {
        return Err(SymplexError::computation_failed(
            op,
            "the large-sample variance of κ came out negative",
        ));
    }
    let z = norm_isf((1.0 - confidence) / 2.0);
    let delta = z * q_to_f64(&m.var).sqrt();
    let kappa_f = q_to_f64(&m.kappa);
    Ok(KappaCi {
        se: ex(ctx, &m.var).sqrt(),
        ci: Interval::closed(kappa_f - delta, kappa_f + delta),
        confidence,
        kappa: m.kappa,
        variance: m.var,
    })
}

/// Cohen's κ of two raters with the large-sample variance of Fleiss,
/// Cohen & Everitt (1969),
///
/// `Var(κ̂) = [Σᵢ pᵢᵢ(1 − (pᵢ· + p·ᵢ)(1 − κ))² + (1 − κ)² Σᵢ≠ⱼ pᵢⱼ(p·ᵢ + pⱼ·)² − (κ − p_e(1 − κ))²] / (n (1 − p_e)²)`
///
/// and the interval `κ ± z_{α/2} √Var(κ̂)`.  κ and the variance are exact;
/// the limits are `f64` (they need a normal quantile).
/// `statsmodels.stats.inter_rater.cohens_kappa(table, return_results=True)`
/// → `var_kappa`, `kappa_low`, `kappa_upp` (95 %).
///
/// ```
/// use symplex::prelude::*;
/// use symplex::linprog::q;
/// use symplex::stats::reliability::kappa_ci_from_confusion;
///
/// let ctx = Context::new();
/// // statsmodels: cohens_kappa([[20, 5], [10, 15]], return_results=True)
/// //   kappa 0.4, var_kappa 0.016128, kappa_low 0.151092290476661, kappa_upp 0.648907709523339
/// let ci = kappa_ci_from_confusion(&ctx, &[vec![20, 5], vec![10, 15]], 0.95)?;
/// assert_eq!((ci.kappa, ci.variance), (q(2, 5), q(252, 15625)));
/// assert!((ci.ci.lower - 0.151_092_290_476_661).abs() < 1e-12);
/// assert!((ci.ci.upper - 0.648_907_709_523_339).abs() < 1e-12);
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for unequal or empty ratings, a
/// confidence level outside `(0, 1)`, or an expected agreement of 1.
pub fn cohen_kappa_ci(
    ctx: &Context,
    a: &[Q],
    b: &[Q],
    confidence: f64,
) -> Result<KappaCi, SymplexError> {
    const OP: &str = "cohen_kappa_ci";
    kappa_ci_of(ctx, OP, &confusion_of(OP, a, b)?, confidence)
}

/// [`cohen_kappa_ci`] from a `k × k` confusion matrix (rows: rater A,
/// columns: rater B).
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for a non-square or empty table, a
/// confidence level outside `(0, 1)`, or an expected agreement of 1.
pub fn kappa_ci_from_confusion(
    ctx: &Context,
    table: &[Vec<usize>],
    confidence: f64,
) -> Result<KappaCi, SymplexError> {
    kappa_ci_of(ctx, "kappa_ci_from_confusion", table, confidence)
}

fn kappa_test_of(
    ctx: &Context,
    op: &'static str,
    table: &[Vec<usize>],
    alt: Alternative,
) -> Result<TestResult, SymplexError> {
    let m = kappa_moments(op, table)?;
    if !m.var0.is_positive() {
        return Err(invalid(
            op,
            "the null variance of κ is zero, the test is undefined",
        ));
    }
    Ok(normal_test(ctx, &m.kappa, &m.var0, alt))
}

/// The test of `H₀: κ = 0` for two raters: `z = κ̂ / √Var₀(κ̂)` with the
/// variance under independence (Fleiss, Cohen & Everitt 1969)
///
/// `Var₀(κ̂) = [p_e + p_e² − Σᵢ pᵢ· p·ᵢ (pᵢ· + p·ᵢ)] / (n (1 − p_e)²)`,
///
/// referred to the standard normal.  The statistic is exact
/// (`κ/√Var₀`); `Greater` is the usual one-sided `κ > 0`.
/// `statsmodels` `cohens_kappa(...).z_value`, `pvalue_one_sided`
/// (`Greater`), `pvalue_two_sided`.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::hypothesis::Alternative;
/// use symplex::stats::reliability::kappa_test_from_confusion;
///
/// let ctx = Context::new();
/// // statsmodels: z_value 2.886751345948128, pvalue_two_sided 0.003892417122779
/// let r = kappa_test_from_confusion(&ctx, &[vec![20, 5], vec![10, 15]], Alternative::TwoSided)?;
/// assert!((r.statistic_f64()? - 2.886_751_345_948_128).abs() < 1e-12);
/// assert!((r.p_value_f64()? - 0.003_892_417_122_779).abs() < 1e-12);
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for unequal or empty ratings, an
/// expected agreement of 1, or a zero null variance.
pub fn kappa_test(
    ctx: &Context,
    a: &[Q],
    b: &[Q],
    alt: Alternative,
) -> Result<TestResult, SymplexError> {
    const OP: &str = "kappa_test";
    kappa_test_of(ctx, OP, &confusion_of(OP, a, b)?, alt)
}

/// [`kappa_test`] from a `k × k` confusion matrix.
///
/// # Errors
///
/// As [`kappa_test`], for a non-square or empty table.
pub fn kappa_test_from_confusion(
    ctx: &Context,
    table: &[Vec<usize>],
    alt: Alternative,
) -> Result<TestResult, SymplexError> {
    kappa_test_of(ctx, "kappa_test_from_confusion", table, alt)
}

/// The largest κ the two raters' marginal distributions allow (Umesh,
/// Peterson & Sauber 1989; Cohen 1960 §"κ_max"):
///
/// `κ_max = (Σᵢ min(pᵢ·, p·ᵢ) − p_e) / (1 − p_e)`,
///
/// attained when each rater's counts overlap as much as the marginals
/// permit; `κ / κ_max` is the agreement relative to what was achievable.
/// `statsmodels` `cohens_kappa(...).kappa_max`.
///
/// ```
/// use symplex::linprog::q;
/// use symplex::stats::data::from_i64;
/// use symplex::stats::reliability::cohen_kappa_maximum;
///
/// let a = from_i64(&[0, 0, 1, 1, 1, 0]);
/// let b = from_i64(&[0, 1, 1, 1, 0, 0]);
/// // Equal marginals (3/6 each): κ_max = 1.
/// assert_eq!(cohen_kappa_maximum(&a, &b)?, q(1, 1));
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for unequal or empty ratings or an
/// expected agreement of 1.
pub fn cohen_kappa_maximum(a: &[Q], b: &[Q]) -> Result<Q, SymplexError> {
    const OP: &str = "cohen_kappa_maximum";
    Ok(kappa_moments(OP, &confusion_of(OP, a, b)?)?.max)
}

/// [`cohen_kappa_maximum`] from a `k × k` confusion matrix.
///
/// ```
/// use symplex::linprog::q;
/// use symplex::stats::reliability::kappa_maximum_from_confusion;
///
/// // statsmodels: cohens_kappa([[20, 5], [10, 15]], return_results=True).kappa_max = 0.8
/// assert_eq!(kappa_maximum_from_confusion(&[vec![20, 5], vec![10, 15]])?, q(4, 5));
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for a non-square or empty table or an
/// expected agreement of 1.
pub fn kappa_maximum_from_confusion(table: &[Vec<usize>]) -> Result<Q, SymplexError> {
    Ok(kappa_moments("kappa_maximum_from_confusion", table)?.max)
}

/// Cochran's Q test (Cochran 1950) that `k` matched binary treatments /
/// raters have the same success rate, on a subjects × treatments table
/// of 0/1 responses:
///
/// `Q = (k − 1) (k Σⱼ Cⱼ² − (Σⱼ Cⱼ)²) / (k Σᵢ Rᵢ − Σᵢ Rᵢ²)`
///
/// with the column totals `Cⱼ` and row totals `Rᵢ`, referred to
/// `χ²_{k−1}`.  `Q` is exact; the p-value is the exact χ² tail.  Subjects
/// with a constant row contribute nothing (they are *not* dropped, as in
/// statsmodels).  `statsmodels.stats.contingency_tables.cochrans_q(x)`.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::linprog::q;
/// use symplex::stats::agreement::RatingTable;
/// use symplex::stats::reliability::cochrans_q;
///
/// let ctx = Context::new();
/// let t = RatingTable::from_i64(&[
///     &[1, 1, 0], &[1, 1, 0], &[1, 0, 0], &[1, 1, 1], &[0, 1, 0], &[1, 0, 0],
///     &[1, 1, 0], &[1, 1, 0], &[0, 0, 0], &[1, 1, 1], &[1, 0, 0], &[1, 1, 0],
/// ])?;
/// // statsmodels: cochrans_q(x) → statistic 11.555555555555555 (= 104/9), pvalue 0.003095586852365, df 2
/// let r = cochrans_q(&ctx, &t)?;
/// assert_eq!(r.statistic_exact(), Some(q(104, 9)));
/// assert_eq!(r.df, Some(ctx.int(2)));
/// assert!((r.p_value_f64()? - 0.003_095_586_852_365).abs() < 1e-12);
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for a missing cell, a response other
/// than 0 or 1, fewer than two subjects or treatments, or every subject
/// constant (zero denominator).
pub fn cochrans_q(ctx: &Context, table: &RatingTable) -> Result<TestResult, SymplexError> {
    const OP: &str = "cochrans_q";
    let rows = score_matrix(OP, table)?;
    check_dichotomous(OP, rows.iter().flatten(), "every response")?;
    let (_, k) = check_scale(OP, &rows, 2)?;
    let row_tot = row_sums(&rows);
    let col_tot: Vec<Q> = columns(&rows).iter().map(|c| data::sum(c)).collect();
    let total = data::sum(&row_tot);
    let denom = qu(k) * &total - sum(row_tot.iter().map(square));
    if denom.is_zero() {
        return Err(invalid(
            OP,
            "every subject responds identically under every treatment, Q is undefined",
        ));
    }
    let num = qu(k) * sum(col_tot.iter().map(square)) - square(&total);
    let statistic = qu(k - 1) * num / denom;
    Ok(TestResult {
        p_value: chi_squared_sf(ctx, k - 1, &statistic),
        statistic: ex(ctx, &statistic),
        df: Some(exu(ctx, k - 1)),
        alternative: Alternative::TwoSided,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Ordinal association
// ═══════════════════════════════════════════════════════════════════════════

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
/// use symplex::stats::data::from_i64;
/// use symplex::stats::reliability::{concordance_counts, ConcordanceCounts};
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
    check_same_len(OP, x, y)?;
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
/// use symplex::stats::data::from_i64;
/// use symplex::stats::reliability::goodman_kruskal_gamma;
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
/// use symplex::stats::data::from_i64;
/// use symplex::stats::reliability::{somers_d, Dependent};
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
/// use symplex::stats::data::from_i64;
/// use symplex::stats::reliability::kendall_tau_c;
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
    let m = data::frequencies(x).len().min(data::frequencies(y).len());
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

// ═══════════════════════════════════════════════════════════════════════════
// 5. Contingency-table diagnostics
// ═══════════════════════════════════════════════════════════════════════════

/// Check a rectangular table of non-negative counts; returns `(rows, cols)`.
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

/// Row sums, column sums and the grand total of a checked table.
fn margins(table: &[Vec<Q>]) -> (Vec<Q>, Vec<Q>, Q) {
    let cols = table[0].len();
    let rows: Vec<Q> = table.iter().map(|r| data::sum(r)).collect();
    let col_sums: Vec<Q> = (0..cols)
        .map(|j| table.iter().fold(Q::zero(), |acc, r| acc + &r[j]))
        .collect();
    let total = data::sum(&rows);
    (rows, col_sums, total)
}

/// Expected counts and the margins, erroring on an empty row or column.
type Expected = (Vec<Vec<Q>>, Vec<Q>, Vec<Q>, Q);

fn expected_of(op: &'static str, table: &[Vec<Q>]) -> Result<Expected, SymplexError> {
    check_table(op, table)?;
    let (rows, cols, total) = margins(table);
    if rows.iter().any(Zero::is_zero) || cols.iter().any(Zero::is_zero) {
        return Err(invalid(
            op,
            "a row or a column of the table is empty, so an expected count is zero",
        ));
    }
    let expected = rows
        .iter()
        .map(|r| cols.iter().map(|c| r * c / &total).collect())
        .collect();
    Ok((expected, rows, cols, total))
}

/// The expected counts under independence, `Eᵢⱼ = rowᵢ · colⱼ / N`, exact.
/// `statsmodels.stats.contingency_tables.Table(t).fittedvalues`;
/// `scipy.stats.contingency.expected_freq`.
///
/// ```
/// use symplex::linprog::q;
/// use symplex::stats::hypothesis::counts;
/// use symplex::stats::reliability::expected_counts;
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
    Ok(expected_of("expected_counts", table)?.0)
}

/// Each cell's contribution `(Oᵢⱼ − Eᵢⱼ)² / Eᵢⱼ` to Pearson's χ², exact
/// (they sum to the statistic of
/// [`chi_square_independence`](super::hypothesis::chi_square_independence)
/// without Yates' correction).  `Table(t).chi2_contribs`.
///
/// ```
/// use symplex::linprog::q;
/// use symplex::stats::hypothesis::counts;
/// use symplex::stats::reliability::chi2_contributions;
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
    let (expected, ..) = expected_of("chi2_contributions", table)?;
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
/// use symplex::stats::hypothesis::counts;
/// use symplex::stats::reliability::standardized_residuals;
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
    let (expected, ..) = expected_of("standardized_residuals", table)?;
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
/// use symplex::stats::hypothesis::counts;
/// use symplex::stats::reliability::adjusted_residuals;
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
    let (expected, rows, cols, total) = expected_of(OP, table)?;
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

// ═══════════════════════════════════════════════════════════════════════════
// 6. Inference on Pearson's r
// ═══════════════════════════════════════════════════════════════════════════

/// Fisher's z-transform `z = atanh(r) = ½ ln((1 + r)/(1 − r))`, whose
/// sampling distribution is approximately normal with variance
/// `1/(n − 3)`.
///
/// ```
/// use symplex::prelude::*;
/// let ctx = Context::new();
/// // atanh(0.8) = 1.0986122886681098 (= ln 3)
/// let z = symplex::stats::reliability::fisher_z(&ctx.rational(4, 5));
/// assert!((z.eval_f64()? - 1.098_612_288_668_109_8).abs() < 1e-12);
/// # Ok::<(), SymplexError>(())
/// ```
#[must_use]
pub fn fisher_z(r: &Ex) -> Ex {
    r.atanh()
}

/// A confidence interval for a population correlation by Fisher's z:
/// `tanh(atanh(r) ∓ z_{α/2} / √(n − 3))`.  `f64` throughout (a normal
/// quantile is involved).  `scipy.stats.pearsonr(x, y).confidence_interval
/// (confidence_level)`.
///
/// ```
/// use symplex::stats::reliability::pearson_ci;
///
/// // scipy: pearsonr(range(1, 6), [1, 3, 2, 5, 4]).confidence_interval(0.95)
/// //   → (-0.279640041969355, 0.9861961933012714); r = 0.8, n = 5
/// let ci = pearson_ci(0.8, 5, 0.95)?;
/// assert!((ci.lower + 0.279_640_041_969_355).abs() < 1e-12);
/// assert!((ci.upper - 0.986_196_193_301_271_4).abs() < 1e-12);
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for `|r| ≥ 1` or non-finite `r`,
/// `n < 4`, or a confidence level outside `(0, 1)`.
pub fn pearson_ci(r: f64, n: usize, confidence: f64) -> Result<Interval<f64>, SymplexError> {
    const OP: &str = "pearson_ci";
    if !r.is_finite() || r.abs() >= 1.0 {
        return Err(invalid(
            OP,
            format!("the correlation must lie strictly between −1 and 1, got {r}"),
        ));
    }
    if n < 4 {
        return Err(invalid(
            OP,
            format!("Fisher's z interval needs at least four observations, got {n}"),
        ));
    }
    check_confidence(OP, confidence)?;
    let z = r.atanh();
    let se = 1.0 / ((n - 3) as f64).sqrt();
    let zc = norm_isf((1.0 - confidence) / 2.0);
    Ok(Interval::closed((z - zc * se).tanh(), (z + zc * se).tanh()))
}

/// The population sums of squares and cross-products of a pair,
/// erroring on a constant sample.
fn cross_moments(
    op: &'static str,
    x: &[Q],
    y: &[Q],
    min_n: usize,
) -> Result<(Q, Q, Q), SymplexError> {
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
    Ok((sxy, sxx, syy))
}

/// The t statistic of Pearson's `r`: `t = r √(n − 2) / √(1 − r²)`, as an
/// exact expression (`t² = r²(n−2)/(1−r²)` is rational).
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::data::from_i64;
/// use symplex::stats::reliability::pearson_t_statistic;
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
    let (sxy, sxx, syy) = cross_moments(OP, x, y, 3)?;
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
/// use symplex::stats::hypothesis::Alternative;
/// use symplex::stats::reliability::pearson_test;
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
    let (sxy, sxx, syy) = cross_moments(OP, x, y, 3)?;
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
/// use symplex::stats::hypothesis::Alternative;
/// use symplex::stats::reliability::compare_two_correlations;
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
