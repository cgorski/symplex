//! Scale reliability and item analysis: Cronbach's α and its relatives
//! (standardized α, KR-20, split-half with the Spearman–Brown prophecy,
//! α-if-deleted, Guttman's λ₂) and classical item analysis (difficulty,
//! the D discrimination index, point-biserial and corrected item–total
//! correlations).
//!
//! **Rule:** a function lives here iff it measures the reliability of a
//! *scale* (a respondents × items response matrix) or analyses its items.
//! In 0.18 the κ inference, κ_max and Cochran's Q moved to
//! [`super::agreement`]; the ordinal association measures to
//! [`super::data`]; the contingency-table diagnostics and the tests on
//! Pearson's r to [`super::hypothesis`]; Fisher's z and the r interval to
//! [`super::estimation`].  The transitional re-exports of those names from
//! this module were removed in 0.22.
//!
//! **Exact where rational.**  Everything that is a rational function of
//! the data (α, KR-20, α-if-deleted, item difficulties and D indices)
//! comes back as a [`Q`]; anything with a square root (a correlation,
//! λ₂) as an exact [`Ex`].  `pingouin`, `numpy` and `fractions.Fraction`
//! re-implementations of the cited formulas are the references named in
//! the tests.
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
//! * A `ctx: &Context` is a parameter exactly when the result is an
//!   [`Ex`] built from rational inputs; [`spearman_brown`] takes its
//!   context from the correlation it is given.

use num_traits::{One, Zero};

use super::agreement::RatingTable;
use super::common::{ex, ex_usize, invalid, qu};
use super::data::{self, Ddof, Q};
use crate::api::context::Context;
use crate::api::expr::Ex;
use crate::base::errors::SymplexError;

// ═══════════════════════════════════════════════════════════════════════════
// Small helpers
// ═══════════════════════════════════════════════════════════════════════════

fn sum(values: impl IntoIterator<Item = Q>) -> Q {
    values.into_iter().fold(Q::zero(), |acc, x| acc + x)
}

fn square(x: &Q) -> Q {
    x * x
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
    total / ex_usize(ctx, m)
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
    Ok(spearman_brown(&r, k))
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
    Ok(spearman_brown(&r, 2))
}

/// The Spearman–Brown prophecy formula: the reliability of a test
/// lengthened by the factor `k`,
///
/// `ρ_k = k ρ / (1 + (k − 1) ρ)`
///
/// (`k = 2` steps a half-test correlation up to the full test).  The
/// result lives in the context of `r`.
///
/// ```
/// use symplex::prelude::*;
/// let ctx = Context::new();
/// // A half-test correlation of 0.6 predicts 2·0.6/1.6 = 0.75 for the full test.
/// let r = symplex::stats::reliability::spearman_brown(&ctx.rational(3, 5), 2);
/// assert_eq!(r.simplify(), ctx.rational(3, 4));
/// ```
#[must_use]
pub fn spearman_brown(r: &Ex, k: usize) -> Ex {
    let ctx = r.context();
    let kq = ex_usize(&ctx, k);
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
    (0..k).map(|j| alpha_without(OP, &rows, j)).collect()
}

/// Cronbach's α of the complete matrix `rows` without item `j`.
fn alpha_without(op: &'static str, rows: &[Vec<Q>], j: usize) -> Result<Q, SymplexError> {
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
    alpha_of_rows(op, &reduced)
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
    if item.len() != total.len() {
        return Err(invalid(
            OP,
            format!(
                "the two variables must have the same length ({} and {})",
                item.len(),
                total.len()
            ),
        ));
    }
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
    let (_, k) = check_scale(OP, &rows, 2)?;
    let (upper, lower) = extreme_groups(OP, &rows)?;
    let totals = row_sums(&rows);
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
                // Item by item: one item whose removal leaves a constant
                // total must not blank the others (it did).
                alpha_if_deleted: if k >= 3 {
                    alpha_without(OP, &rows, j).ok()
                } else {
                    None
                },
            })
        })
        .collect()
}
