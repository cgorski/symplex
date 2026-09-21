//! Inter-rater agreement, exactly: percent agreement, Cohen's κ (plain
//! and weighted), Scott's π, Fleiss' κ, Krippendorff's α, Gwet's AC₁,
//! the intraclass correlations of Shrout & Fleiss and Kendall's W.
//!
//! Ratings live in a [`RatingTable`] — items × raters, each cell an
//! exact rational ([`Q`]) or missing.  Every coefficient in this module
//! is a rational function of the counts, so every result is a `Q`
//! (nothing is rounded); the references named in the tests are
//! `statsmodels.stats.inter_rater`, the `krippendorff` package, and the
//! formulas of the cited papers.
//!
//! ```
//! use symplex::stats::agreement::{cohen_kappa, RatingTable, krippendorff_alpha, Level};
//! use symplex::stats::data::from_i64;
//! use symplex::linprog::q;
//!
//! // Two raters, three categories.
//! let a = from_i64(&[1, 2, 3, 1, 2, 3, 1, 1, 2, 3]);
//! let b = from_i64(&[1, 2, 3, 1, 3, 3, 1, 2, 2, 3]);
//! let k = cohen_kappa(&a, &b)?;
//! assert_eq!(k.observed, q(4, 5));
//! // Krippendorff's α from an items × raters table (a missing cell is `None`).
//! let t = RatingTable::from_i64_missing(&[
//!     &[Some(1), Some(1), None],
//!     &[Some(2), Some(2), Some(2)],
//!     &[Some(1), Some(2), Some(1)],
//!     &[Some(2), Some(2), Some(1)],
//! ])?;
//! // krippendorff.alpha(reliability_data=[[1,2,1,2],[1,2,2,2],[nan,2,1,1]], level_of_measurement='nominal') = 1/3
//! assert_eq!(krippendorff_alpha(&t, Level::Nominal)?, q(1, 3));
//! # Ok::<(), symplex::prelude::SymplexError>(())
//! ```
//!
//! # Conventions
//!
//! * A pair of raters is given as two slices `a`, `b` of equal length
//!   (item `i` is `a[i]`, `b[i]`); several raters as a [`RatingTable`].
//! * Categories are the *values* of the ratings.  Where a coefficient
//!   depends on the set of categories (weighted κ by position, Gwet's
//!   AC₁ through the number of categories) the sorted distinct values
//!   observed are used unless the caller passes the categories.
//! * A coefficient whose denominator vanishes (every rating identical,
//!   expected agreement 1, …) is undefined and reported as an
//!   [`SymplexError::InvalidArgument`].

use num_bigint::BigInt;
use num_traits::{One, Zero};

use crate::base::errors::SymplexError;
use crate::domains::stats::data::{self, Q};

fn invalid(op: &'static str, reason: impl Into<String>) -> SymplexError {
    SymplexError::invalid_argument(op, reason)
}

fn qu(n: usize) -> Q {
    Q::from_integer(BigInt::from(n))
}

fn qi(n: i64) -> Q {
    Q::from_integer(BigInt::from(n))
}

fn sum(values: impl IntoIterator<Item = Q>) -> Q {
    values.into_iter().fold(Q::zero(), |acc, x| acc + x)
}

fn square(x: &Q) -> Q {
    x * x
}

fn check_pair(op: &'static str, a: &[Q], b: &[Q]) -> Result<(), SymplexError> {
    if a.len() != b.len() {
        return Err(invalid(
            op,
            format!("the two raters rated {} and {} items", a.len(), b.len()),
        ));
    }
    if a.is_empty() {
        return Err(invalid(op, "needs at least one item"));
    }
    Ok(())
}

// ── Rating tables ────────────────────────────────────────────────────

/// Ratings of `n_items` items by `n_raters` raters (items are rows,
/// raters columns); a cell is `None` when that rater did not rate that
/// item.  Rectangular by construction.
///
/// ```
/// use symplex::stats::agreement::RatingTable;
/// use symplex::linprog::qi;
///
/// let t = RatingTable::from_i64(&[&[1, 1, 2], &[3, 3, 3], &[2, 1, 2]])?;
/// assert_eq!((t.n_items(), t.n_raters()), (3, 3));
/// assert_eq!(t.categories(), vec![qi(1), qi(2), qi(3)]);
/// assert_eq!(t.count_table(&t.categories())?, vec![vec![2, 1, 0], vec![0, 0, 3], vec![1, 2, 0]]);
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RatingTable {
    rows: Vec<Vec<Option<Q>>>,
    n_raters: usize,
}

impl RatingTable {
    /// A table from its rows (one per item, one cell per rater).
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if there is no item, no rater,
    /// or the rows have different lengths.
    pub fn new(rows: Vec<Vec<Option<Q>>>) -> Result<Self, SymplexError> {
        let op = "RatingTable::new";
        let n_raters = match rows.first() {
            Some(r) => r.len(),
            None => return Err(invalid(op, "a rating table needs at least one item")),
        };
        if n_raters == 0 {
            return Err(invalid(op, "a rating table needs at least one rater"));
        }
        if let Some((i, r)) = rows.iter().enumerate().find(|(_, r)| r.len() != n_raters) {
            return Err(invalid(
                op,
                format!(
                    "item {i} has {} cells but the table has {n_raters} raters",
                    r.len()
                ),
            ));
        }
        Ok(Self { rows, n_raters })
    }

    /// A table from borrowed rows.
    pub fn from_rows(rows: &[&[Option<Q>]]) -> Result<Self, SymplexError> {
        Self::new(rows.iter().map(|r| r.to_vec()).collect())
    }

    /// A complete table of integer ratings (rows are items).
    pub fn from_i64(rows: &[&[i64]]) -> Result<Self, SymplexError> {
        Self::new(
            rows.iter()
                .map(|r| r.iter().map(|&x| Some(qi(x))).collect())
                .collect(),
        )
    }

    /// A table of integer ratings with missing cells (rows are items).
    pub fn from_i64_missing(rows: &[&[Option<i64>]]) -> Result<Self, SymplexError> {
        Self::new(
            rows.iter()
                .map(|r| r.iter().map(|x| x.map(qi)).collect())
                .collect(),
        )
    }

    /// A table from its *columns*: one slice per rater, one cell per item
    /// (the layout of Krippendorff's "reliability data" matrices).
    pub fn from_raters_i64(raters: &[&[Option<i64>]]) -> Result<Self, SymplexError> {
        let op = "RatingTable::from_raters_i64";
        let n_items = match raters.first() {
            Some(r) => r.len(),
            None => return Err(invalid(op, "a rating table needs at least one rater")),
        };
        if raters.iter().any(|r| r.len() != n_items) {
            return Err(invalid(op, "every rater must have one cell per item"));
        }
        if n_items == 0 {
            return Err(invalid(op, "a rating table needs at least one item"));
        }
        Self::new(
            (0..n_items)
                .map(|i| raters.iter().map(|r| r[i].map(qi)).collect())
                .collect(),
        )
    }

    /// Number of items (rows).
    pub fn n_items(&self) -> usize {
        self.rows.len()
    }

    /// Number of raters (columns).
    pub fn n_raters(&self) -> usize {
        self.n_raters
    }

    /// The rows (items).
    pub fn rows(&self) -> &[Vec<Option<Q>>] {
        &self.rows
    }

    /// The ratings of item `i` (`None` if out of range).
    pub fn item(&self, i: usize) -> Option<&[Option<Q>]> {
        self.rows.get(i).map(|r| r.as_slice())
    }

    /// The ratings given by rater `j` (`None` if out of range).
    pub fn rater(&self, j: usize) -> Option<Vec<Option<Q>>> {
        (j < self.n_raters).then(|| self.rows.iter().map(|r| r[j].clone()).collect())
    }

    /// The rating of item `i` by rater `j` (`None` if missing or out of
    /// range).
    pub fn get(&self, i: usize, j: usize) -> Option<&Q> {
        self.rows
            .get(i)
            .and_then(|r| r.get(j))
            .and_then(|c| c.as_ref())
    }

    /// `true` when no cell is missing.
    pub fn is_complete(&self) -> bool {
        self.rows.iter().all(|r| r.iter().all(|c| c.is_some()))
    }

    /// The distinct rating values present, ascending.
    pub fn categories(&self) -> Vec<Q> {
        let mut v: Vec<Q> = self.rows.iter().flatten().flatten().cloned().collect();
        v.sort();
        v.dedup();
        v
    }

    /// Items × categories counts: how many raters gave item `i` the value
    /// `categories[c]` (`statsmodels.stats.inter_rater.aggregate_raters`).
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `categories` repeats a value
    /// or a rating is not among them.
    pub fn count_table(&self, categories: &[Q]) -> Result<Vec<Vec<usize>>, SymplexError> {
        let op = "RatingTable::count_table";
        check_categories(op, categories)?;
        self.rows
            .iter()
            .map(|row| {
                let mut counts = vec![0usize; categories.len()];
                for x in row.iter().flatten() {
                    let c = categories
                        .iter()
                        .position(|k| k == x)
                        .ok_or_else(|| invalid(op, format!("rating {x} is not a category")))?;
                    counts[c] += 1;
                }
                Ok(counts)
            })
            .collect()
    }

    /// The ratings of raters `j1` and `j2` on the items both rated, as two
    /// aligned vectors.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] for a rater index out of range.
    pub fn paired_ratings(&self, j1: usize, j2: usize) -> Result<(Vec<Q>, Vec<Q>), SymplexError> {
        if j1 >= self.n_raters || j2 >= self.n_raters {
            return Err(invalid(
                "RatingTable::paired_ratings",
                format!(
                    "rater index out of range (the table has {} raters)",
                    self.n_raters
                ),
            ));
        }
        Ok(self
            .rows
            .iter()
            .filter_map(|r| match (&r[j1], &r[j2]) {
                (Some(x), Some(y)) => Some((x.clone(), y.clone())),
                _ => None,
            })
            .unzip())
    }

    /// The rows with every cell present.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if any cell is missing.
    pub fn complete_rows(&self) -> Result<Vec<Vec<Q>>, SymplexError> {
        self.rows
            .iter()
            .enumerate()
            .map(|(i, r)| {
                r.iter()
                    .cloned()
                    .map(|c| {
                        c.ok_or_else(|| {
                            invalid(
                                "RatingTable::complete_rows",
                                format!("item {i} has a missing rating"),
                            )
                        })
                    })
                    .collect()
            })
            .collect()
    }
}

fn check_categories(op: &'static str, categories: &[Q]) -> Result<(), SymplexError> {
    if categories.is_empty() {
        return Err(invalid(op, "needs at least one category"));
    }
    for (i, c) in categories.iter().enumerate() {
        if categories[..i].contains(c) {
            return Err(invalid(op, format!("category {c} is listed twice")));
        }
    }
    Ok(())
}

/// Frequency of every rating value over all present cells, ascending.
///
/// ```
/// use symplex::stats::agreement::{category_frequencies, RatingTable};
/// use symplex::linprog::qi;
///
/// let t = RatingTable::from_i64_missing(&[&[Some(1), Some(2)], &[Some(2), None]])?;
/// assert_eq!(category_frequencies(&t), vec![(qi(1), 1), (qi(2), 2)]);
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
pub fn category_frequencies(table: &RatingTable) -> Vec<(Q, usize)> {
    let all: Vec<Q> = table.rows.iter().flatten().flatten().cloned().collect();
    data::frequencies(&all)
}

/// The confusion matrix of two raters: `m[i][j]` counts the items rated
/// `categories[i]` by `a` and `categories[j]` by `b`.
///
/// ```
/// use symplex::stats::agreement::confusion_matrix;
/// use symplex::stats::data::from_i64;
///
/// let (a, b) = (from_i64(&[0, 0, 1, 1]), from_i64(&[0, 1, 1, 1]));
/// assert_eq!(confusion_matrix(&a, &b, &from_i64(&[0, 1]))?, vec![vec![1, 1], vec![0, 2]]);
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for unequal or empty ratings, a
/// repeated category, or a rating outside the categories.
pub fn confusion_matrix(
    a: &[Q],
    b: &[Q],
    categories: &[Q],
) -> Result<Vec<Vec<usize>>, SymplexError> {
    let op = "confusion_matrix";
    check_pair(op, a, b)?;
    check_categories(op, categories)?;
    let index = |x: &Q| {
        categories
            .iter()
            .position(|k| k == x)
            .ok_or_else(|| invalid(op, format!("rating {x} is not a category")))
    };
    let k = categories.len();
    let mut m = vec![vec![0usize; k]; k];
    for (x, y) in a.iter().zip(b) {
        m[index(x)?][index(y)?] += 1;
    }
    Ok(m)
}

fn observed_categories(a: &[Q], b: &[Q]) -> Vec<Q> {
    let mut v: Vec<Q> = a.iter().chain(b).cloned().collect();
    v.sort();
    v.dedup();
    v
}

// ── Percent agreement ────────────────────────────────────────────────

/// The fraction of items on which two raters agree.
///
/// ```
/// use symplex::stats::agreement::percent_agreement;
/// use symplex::stats::data::from_i64;
/// use symplex::linprog::q;
///
/// assert_eq!(percent_agreement(&from_i64(&[1, 2, 2, 3]), &from_i64(&[1, 2, 3, 3]))?, q(3, 4));
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for unequal or empty ratings.
pub fn percent_agreement(a: &[Q], b: &[Q]) -> Result<Q, SymplexError> {
    check_pair("percent_agreement", a, b)?;
    let agree = a.iter().zip(b).filter(|(x, y)| x == y).count();
    Ok(qu(agree) / qu(a.len()))
}

/// The mean, over all pairs of raters, of their percent agreement on the
/// items both rated.  Pairs with no item in common are left out.
///
/// ```
/// use symplex::stats::agreement::{pairwise_percent_agreement, RatingTable};
/// use symplex::linprog::q;
///
/// let t = RatingTable::from_i64(&[&[1, 1, 2], &[2, 2, 2], &[1, 2, 1]])?;
/// // pairs (0,1): 2/3, (0,2): 2/3, (1,2): 1/3 → mean 5/9
/// assert_eq!(pairwise_percent_agreement(&t)?, q(5, 9));
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] with fewer than two raters or when
/// no two raters rated a common item.
pub fn pairwise_percent_agreement(table: &RatingTable) -> Result<Q, SymplexError> {
    let op = "pairwise_percent_agreement";
    let m = table.n_raters();
    if m < 2 {
        return Err(invalid(op, "needs at least two raters"));
    }
    let mut total = Q::zero();
    let mut pairs = 0usize;
    for j1 in 0..m {
        for j2 in j1 + 1..m {
            let (a, b) = table.paired_ratings(j1, j2)?;
            if a.is_empty() {
                continue;
            }
            total += percent_agreement(&a, &b)?;
            pairs += 1;
        }
    }
    if pairs == 0 {
        return Err(invalid(op, "no two raters rated a common item"));
    }
    Ok(total / qu(pairs))
}

// ── Cohen's κ and relatives ──────────────────────────────────────────

/// A chance-corrected agreement coefficient with its ingredients:
/// `kappa = (observed − expected) / (1 − expected)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KappaResult {
    /// The coefficient.
    pub kappa: Q,
    /// Observed (weighted) agreement `p_o`.
    pub observed: Q,
    /// Agreement expected by chance `p_e`.
    pub expected: Q,
}

/// Disagreement weights for a weighted κ over `k` ordered categories
/// (positions `0..k` in the confusion matrix; `statsmodels`'
/// `cohens_kappa(table, wt=…)` / Wikipedia convention: `0` on the
/// diagonal, larger for farther-apart categories).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Weights {
    /// `w_ij = [i ≠ j]`: plain Cohen's κ.
    Unweighted,
    /// `w_ij = |i − j| / (k − 1)` (`wt='linear'`).
    Linear,
    /// `w_ij = (i − j)² / (k − 1)²` (`wt='quadratic'`).
    Quadratic,
    /// An explicit `k × k` matrix of disagreement weights.
    Custom(Vec<Vec<Q>>),
}

impl Weights {
    fn matrix(&self, op: &'static str, k: usize) -> Result<Vec<Vec<Q>>, SymplexError> {
        let dist = |i: usize, j: usize| qu(i.abs_diff(j));
        match self {
            Weights::Unweighted => Ok((0..k)
                .map(|i| {
                    (0..k)
                        .map(|j| if i == j { Q::zero() } else { Q::one() })
                        .collect()
                })
                .collect()),
            Weights::Linear | Weights::Quadratic => {
                if k < 2 {
                    return Err(invalid(op, "weighted κ needs at least two categories"));
                }
                let span = qu(k - 1);
                Ok((0..k)
                    .map(|i| {
                        (0..k)
                            .map(|j| {
                                let d = dist(i, j) / &span;
                                if *self == Weights::Linear {
                                    d
                                } else {
                                    square(&d)
                                }
                            })
                            .collect()
                    })
                    .collect())
            }
            Weights::Custom(w) => {
                if w.len() != k || w.iter().any(|r| r.len() != k) {
                    return Err(invalid(
                        op,
                        format!("the weight matrix must be {k} × {k} like the confusion matrix"),
                    ));
                }
                Ok(w.clone())
            }
        }
    }
}

/// κ from a `k × k` confusion matrix (Cohen 1960; Cohen 1968 for the
/// weighted form):
///
/// `κ = 1 − Σ w_ij p_ij / Σ w_ij p_i· p_·j`
///
/// with `p_ij = n_ij / n` and the marginals `p_i·`, `p_·j`; the plain κ
/// is the `Unweighted` case, `(p_o − p_e) / (1 − p_e)`.  The `observed`
/// and `expected` fields are `1 − Σ w p_ij` and `1 − Σ w p_i· p_·j`.
/// `statsmodels.stats.inter_rater.cohens_kappa(table, weights, wt)`.
///
/// ```
/// use symplex::stats::agreement::{kappa_from_confusion, Weights};
/// use symplex::linprog::q;
///
/// let table = vec![vec![20, 5], vec![10, 15]];
/// // statsmodels: cohens_kappa([[20, 5], [10, 15]]).kappa = 0.4
/// assert_eq!(kappa_from_confusion(&table, &Weights::Unweighted)?.kappa, q(2, 5));
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for a non-square or empty table,
/// mismatched custom weights, or an expected disagreement of zero (κ
/// undefined).
pub fn kappa_from_confusion(
    table: &[Vec<usize>],
    weights: &Weights,
) -> Result<KappaResult, SymplexError> {
    let op = "cohen_kappa";
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
    let w = weights.matrix(op, k)?;
    let row: Vec<usize> = table.iter().map(|r| r.iter().sum()).collect();
    let col: Vec<usize> = (0..k).map(|j| table.iter().map(|r| r[j]).sum()).collect();
    let nq = qu(n);
    let mut d_obs = Q::zero();
    let mut d_exp = Q::zero();
    for (i, r) in table.iter().enumerate() {
        for (j, &cell) in r.iter().enumerate() {
            d_obs += &w[i][j] * qu(cell);
            d_exp += &w[i][j] * qu(row[i] * col[j]);
        }
    }
    d_obs /= &nq;
    d_exp /= square(&nq);
    if d_exp.is_zero() {
        return Err(invalid(
            op,
            "the expected disagreement is zero (a single category), κ is undefined",
        ));
    }
    Ok(KappaResult {
        kappa: Q::one() - &d_obs / &d_exp,
        observed: Q::one() - d_obs,
        expected: Q::one() - d_exp,
    })
}

/// Cohen's κ of two raters on nominal categories (Cohen 1960):
/// `κ = (p_o − p_e) / (1 − p_e)` with `p_e = Σ_c p_a(c) p_b(c)`.
/// `statsmodels.stats.inter_rater.cohens_kappa`.
///
/// ```
/// use symplex::stats::agreement::cohen_kappa;
/// use symplex::stats::data::from_i64;
/// use symplex::linprog::q;
///
/// let a = from_i64(&[0, 0, 1, 1, 1, 0]);
/// let b = from_i64(&[0, 1, 1, 1, 0, 0]);
/// let k = cohen_kappa(&a, &b)?;
/// // p_o = 4/6, p_e = (3/6)(3/6) + (3/6)(3/6) = 1/2 → κ = 1/3
/// assert_eq!((k.observed, k.expected, k.kappa), (q(2, 3), q(1, 2), q(1, 3)));
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// As [`kappa_from_confusion`].
pub fn cohen_kappa(a: &[Q], b: &[Q]) -> Result<KappaResult, SymplexError> {
    weighted_kappa(a, b, &Weights::Unweighted)
}

/// Weighted κ of two raters on ordered categories (Cohen 1968): the
/// categories are the sorted distinct values observed and the weights
/// are by position (see [`Weights`]).  Use [`confusion_matrix`] with an
/// explicit category list and [`kappa_from_confusion`] to include
/// unobserved categories.
///
/// ```
/// use symplex::stats::agreement::{weighted_kappa, Weights};
/// use symplex::stats::data::from_i64;
/// use symplex::linprog::q;
///
/// let a = from_i64(&[1, 2, 3, 2, 1]);
/// let b = from_i64(&[1, 3, 3, 2, 2]);
/// // statsmodels: cohens_kappa(table, wt='quadratic').kappa = 0.6875
/// assert_eq!(weighted_kappa(&a, &b, &Weights::Quadratic)?.kappa, q(11, 16));
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// As [`kappa_from_confusion`].
pub fn weighted_kappa(a: &[Q], b: &[Q], weights: &Weights) -> Result<KappaResult, SymplexError> {
    check_pair("cohen_kappa", a, b)?;
    let table = confusion_matrix(a, b, &observed_categories(a, b))?;
    kappa_from_confusion(&table, weights)
}

/// Scott's π of two raters (Scott 1955): like Cohen's κ but with the
/// chance agreement from the pooled marginal,
/// `p_e = Σ_c ((n_a(c) + n_b(c)) / 2n)²`.  Equals Fleiss' κ for two
/// raters.
///
/// ```
/// use symplex::stats::agreement::scott_pi;
/// use symplex::stats::data::from_i64;
/// use symplex::linprog::q;
///
/// let a = from_i64(&[0, 0, 1, 1, 1, 0]);
/// let b = from_i64(&[0, 1, 1, 1, 0, 0]);
/// // pooled marginal (1/2, 1/2): p_e = 1/2, p_o = 2/3 → π = 1/3
/// assert_eq!(scott_pi(&a, &b)?, q(1, 3));
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for unequal or empty ratings, or a
/// single category (π undefined).
pub fn scott_pi(a: &[Q], b: &[Q]) -> Result<Q, SymplexError> {
    let op = "scott_pi";
    check_pair(op, a, b)?;
    let n = a.len();
    let p_o = percent_agreement(a, b)?;
    let two_n = qu(2 * n);
    let p_e = sum(observed_categories(a, b).iter().map(|c| {
        let pooled = a.iter().filter(|x| *x == c).count() + b.iter().filter(|x| *x == c).count();
        square(&(qu(pooled) / &two_n))
    }));
    let denom = Q::one() - &p_e;
    if denom.is_zero() {
        return Err(invalid(op, "a single category, π is undefined"));
    }
    Ok((p_o - p_e) / denom)
}

// ── Fleiss' κ ────────────────────────────────────────────────────────

/// Fleiss' κ from an `N × k` table of counts `n_ij` (item `i`, category
/// `j`) with the same number `n ≥ 2` of raters per item (Fleiss 1971):
///
/// `P_i = (Σ_j n_ij² − n) / (n(n − 1))`, `P̄ = mean_i P_i`,
/// `p_j = Σ_i n_ij / (N n)`, `P̄_e = Σ_j p_j²`, `κ = (P̄ − P̄_e) / (1 − P̄_e)`.
///
/// `statsmodels.stats.inter_rater.fleiss_kappa(table)`.
///
/// ```
/// use symplex::stats::agreement::fleiss_kappa;
/// use symplex::linprog::q;
///
/// // 3 raters, 4 items, 2 categories.
/// let counts = vec![vec![3, 0], vec![0, 3], vec![2, 1], vec![3, 0]];
/// // statsmodels: fleiss_kappa(counts) = 0.625 = 5/8
/// assert_eq!(fleiss_kappa(&counts)?, q(5, 8));
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for an empty or ragged table, rows
/// not all summing to the same `n ≥ 2`, or `P̄_e = 1` (κ undefined).
pub fn fleiss_kappa(counts: &[Vec<usize>]) -> Result<Q, SymplexError> {
    let op = "fleiss_kappa";
    let n_items = counts.len();
    let k = counts.first().map_or(0, |r| r.len());
    if n_items == 0 || k == 0 || counts.iter().any(|r| r.len() != k) {
        return Err(invalid(
            op,
            "the count table must be rectangular and non-empty",
        ));
    }
    let n: usize = counts[0].iter().sum();
    if n < 2 {
        return Err(invalid(op, "needs at least two raters per item"));
    }
    if counts.iter().any(|r| r.iter().sum::<usize>() != n) {
        return Err(invalid(
            op,
            "every item must be rated by the same number of raters",
        ));
    }
    let per_item = qu(n * (n - 1));
    let p_bar = sum(counts.iter().map(|r| {
        let sq: usize = r.iter().map(|&c| c * c).sum();
        qu(sq - n) / &per_item
    })) / qu(n_items);
    let total = qu(n_items * n);
    let p_e = sum((0..k).map(|j| {
        let col: usize = counts.iter().map(|r| r[j]).sum();
        square(&(qu(col) / &total))
    }));
    let denom = Q::one() - &p_e;
    if denom.is_zero() {
        return Err(invalid(op, "a single category, κ is undefined"));
    }
    Ok((p_bar - p_e) / denom)
}

/// Fleiss' κ of a complete [`RatingTable`] (each item rated by every
/// rater): the counts are aggregated over the observed categories, then
/// [`fleiss_kappa`].  `fleiss_kappa(aggregate_raters(data)[0])`.
///
/// ```
/// use symplex::stats::agreement::{fleiss_kappa_ratings, RatingTable};
/// use symplex::linprog::q;
///
/// let t = RatingTable::from_i64(&[&[0, 0, 0], &[1, 1, 1], &[0, 0, 1], &[0, 0, 0]])?;
/// assert_eq!(fleiss_kappa_ratings(&t)?, q(5, 8));
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for a table with missing cells, and
/// as [`fleiss_kappa`].
pub fn fleiss_kappa_ratings(table: &RatingTable) -> Result<Q, SymplexError> {
    if !table.is_complete() {
        return Err(invalid("fleiss_kappa", "Fleiss' κ needs a complete table"));
    }
    fleiss_kappa(&table.count_table(&table.categories())?)
}

// ── Krippendorff's α ─────────────────────────────────────────────────

/// The level of measurement of the ratings, which fixes Krippendorff's
/// difference function `δ²(c, k)` between two values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Level {
    /// `δ² = [c ≠ k]`.
    Nominal,
    /// `δ² = (Σ_{g=c}^{k} n_g − (n_c + n_k)/2)²` over the categories in
    /// order, `n_g` the number of pairable values `g`.
    Ordinal,
    /// `δ² = (c − k)²`.
    Interval,
    /// `δ² = ((c − k) / (c + k))²` (`0` when `c + k = 0`).
    Ratio,
}

/// Krippendorff's α (Krippendorff 2011, "Computing Krippendorff's
/// alpha-reliability") of a [`RatingTable`] with any pattern of missing
/// cells:
///
/// `α = 1 − D_o / D_e`,
/// `D_o = Σ_{c,k} o_ck δ²(c,k)`, `D_e = Σ_{c,k} (n_c n_k − [c = k] n_c) / (n − 1) · δ²(c,k)`,
///
/// where the coincidence matrix `o_ck = Σ_u n_uc (n_uk − [c = k]) / (m_u − 1)`
/// sums over the items `u` with `m_u ≥ 2` ratings, `n_c = Σ_k o_ck` and
/// `n = Σ_c n_c`.  The `krippendorff` package
/// (`krippendorff.alpha(reliability_data=…, level_of_measurement=…)`).
///
/// ```
/// use symplex::stats::agreement::{krippendorff_alpha, Level, RatingTable};
/// use symplex::linprog::q;
///
/// // Columns are raters; the third rater skipped the first item.
/// let t = RatingTable::from_raters_i64(&[
///     &[Some(1), Some(2), Some(3), Some(3), Some(2)],
///     &[Some(1), Some(2), Some(3), Some(3), Some(2)],
///     &[None, Some(3), Some(3), Some(3), Some(2)],
/// ])?;
/// // krippendorff.alpha(reliability_data=..., level_of_measurement='nominal') = 0.7796610169491526
/// assert_eq!(krippendorff_alpha(&t, Level::Nominal)?, q(46, 59));
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] with fewer than two distinct
/// values, no item with two or more ratings, or every pairable value
/// identical (α undefined).
pub fn krippendorff_alpha(table: &RatingTable, level: Level) -> Result<Q, SymplexError> {
    let op = "krippendorff_alpha";
    let cats = table.categories();
    let v = cats.len();
    if v < 2 {
        return Err(invalid(op, "needs at least two distinct values"));
    }
    let counts = table.count_table(&cats)?;
    let mut o = vec![vec![Q::zero(); v]; v];
    let mut pairable_items = 0usize;
    for row in &counts {
        let m: usize = row.iter().sum();
        if m < 2 {
            continue;
        }
        pairable_items += 1;
        let denom = qu(m - 1);
        for (c, &n_uc) in row.iter().enumerate() {
            if n_uc == 0 {
                continue;
            }
            for (k, &n_uk) in row.iter().enumerate() {
                let pairs = n_uc * (n_uk - usize::from(c == k));
                if pairs > 0 {
                    o[c][k] += qu(pairs) / &denom;
                }
            }
        }
    }
    if pairable_items == 0 {
        return Err(invalid(
            op,
            "needs at least one item rated by two or more raters",
        ));
    }
    let n_c: Vec<Q> = o.iter().map(|r| sum(r.iter().cloned())).collect();
    let n = sum(n_c.iter().cloned());
    // Prefix sums of the pairable counts for the ordinal metric.
    let mut prefix = vec![Q::zero()];
    for x in &n_c {
        let last = prefix[prefix.len() - 1].clone();
        prefix.push(last + x);
    }
    let delta = |c: usize, k: usize| -> Q {
        match level {
            Level::Nominal => {
                if c == k {
                    Q::zero()
                } else {
                    Q::one()
                }
            }
            Level::Interval => square(&(&cats[c] - &cats[k])),
            Level::Ratio => {
                let s = &cats[c] + &cats[k];
                if s.is_zero() {
                    Q::zero()
                } else {
                    square(&((&cats[c] - &cats[k]) / s))
                }
            }
            Level::Ordinal => {
                let (lo, hi) = (c.min(k), c.max(k));
                let between = &prefix[hi + 1] - &prefix[lo];
                square(&(between - (&n_c[lo] + &n_c[hi]) / qi(2)))
            }
        }
    };
    let mut d_o = Q::zero();
    let mut d_e = Q::zero();
    for c in 0..v {
        for k in 0..v {
            let d = delta(c, k);
            if d.is_zero() {
                continue;
            }
            d_o += &o[c][k] * &d;
            let expected = &n_c[c] * &n_c[k] - if c == k { n_c[c].clone() } else { Q::zero() };
            d_e += expected * d;
        }
    }
    d_e /= n - Q::one();
    if d_e.is_zero() {
        return Err(invalid(
            op,
            "every pairable value is identical, α is undefined",
        ));
    }
    Ok(Q::one() - d_o / d_e)
}

// ── Gwet's AC₁ ───────────────────────────────────────────────────────

/// Gwet's AC₁ for nominal ratings by any number of raters, with missing
/// cells (Gwet 2008, *Br. J. Math. Stat. Psychol.* 61, eqs. for AC₁):
///
/// `AC₁ = (p_a − p_e) / (1 − p_e)`,
/// `p_a = mean over items with r_i ≥ 2 of Σ_k r_ik (r_ik − 1) / (r_i (r_i − 1))`,
/// `p_e = Σ_k π_k (1 − π_k) / (K − 1)`, `π_k = mean over rated items of r_ik / r_i`,
///
/// where `r_ik` raters put item `i` in category `k`, `r_i = Σ_k r_ik`,
/// and `K` is the number of categories (the sorted distinct values when
/// `categories` is `None`; pass them to count unobserved categories).
///
/// ```
/// use symplex::stats::agreement::{gwet_ac1, RatingTable};
/// use symplex::linprog::q;
///
/// let t = RatingTable::from_i64(&[&[0, 0, 0], &[1, 1, 1], &[0, 0, 1], &[0, 0, 0]])?;
/// // Fractions: p_a = 5/6, π = (2/3, 1/3), p_e = 4/9 → AC1 = 7/10
/// assert_eq!(gwet_ac1(&t, None)?, q(7, 10));
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] with fewer than two categories, a
/// rating outside `categories`, or no item with two or more ratings.
pub fn gwet_ac1(table: &RatingTable, categories: Option<&[Q]>) -> Result<Q, SymplexError> {
    let op = "gwet_ac1";
    let cats: Vec<Q> = match categories {
        Some(c) => c.to_vec(),
        None => table.categories(),
    };
    let kcat = cats.len();
    if kcat < 2 {
        return Err(invalid(op, "needs at least two categories"));
    }
    let counts = table.count_table(&cats)?;
    let mut pa_sum = Q::zero();
    let mut pairable = 0usize;
    let mut rated = 0usize;
    let mut pi = vec![Q::zero(); kcat];
    for row in &counts {
        let r: usize = row.iter().sum();
        if r == 0 {
            continue;
        }
        rated += 1;
        for (k, &c) in row.iter().enumerate() {
            pi[k] += qu(c) / qu(r);
        }
        if r >= 2 {
            pairable += 1;
            let agree: usize = row.iter().map(|&c| c * (c.saturating_sub(1))).sum();
            pa_sum += qu(agree) / qu(r * (r - 1));
        }
    }
    if pairable == 0 {
        return Err(invalid(
            op,
            "needs at least one item rated by two or more raters",
        ));
    }
    let p_a = pa_sum / qu(pairable);
    let p_e = sum(pi.iter().map(|p| {
        let p = p / qu(rated);
        &p * (Q::one() - &p)
    })) / qu(kcat - 1);
    let denom = Q::one() - &p_e;
    if denom.is_zero() {
        return Err(invalid(op, "the chance agreement is 1, AC₁ is undefined"));
    }
    Ok((p_a - p_e) / denom)
}

// ── Intraclass correlation ───────────────────────────────────────────

/// The six intraclass correlations of Shrout & Fleiss (1979) for `n`
/// items rated by `k` raters.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IccForm {
    /// ICC(1): one-way random, single rating —
    /// `(MSR − MSW) / (MSR + (k − 1) MSW)`.
    Icc1,
    /// ICC(1,k): one-way random, mean of `k` ratings — `(MSR − MSW) / MSR`.
    Icc1Average,
    /// ICC(2,1): two-way random, single rating —
    /// `(MSR − MSE) / (MSR + (k − 1) MSE + k (MSC − MSE) / n)`.
    Icc2Single,
    /// ICC(2,k): two-way random, mean of `k` ratings —
    /// `(MSR − MSE) / (MSR + (MSC − MSE) / n)`.
    Icc2Average,
    /// ICC(3,1): two-way mixed, single rating —
    /// `(MSR − MSE) / (MSR + (k − 1) MSE)`.
    Icc3Single,
    /// ICC(3,k): two-way mixed, mean of `k` ratings — `(MSR − MSE) / MSR`.
    Icc3Average,
}

/// The mean squares of the two-way ANOVA behind the intraclass
/// correlations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IccAnova {
    /// Between-items (rows) mean square, `k Σ_i (x̄_i· − x̄)² / (n − 1)`.
    pub msr: Q,
    /// Between-raters (columns) mean square, `n Σ_j (x̄_·j − x̄)² / (k − 1)`.
    pub msc: Q,
    /// Residual mean square, `(SST − SSR − SSC) / ((n − 1)(k − 1))`.
    pub mse: Q,
    /// Within-items mean square, `(SSC + SSE) / (n (k − 1))`.
    pub msw: Q,
}

/// The ANOVA mean squares of a complete `n × k` [`RatingTable`].
///
/// ```
/// use symplex::stats::agreement::{icc_anova, RatingTable};
/// use symplex::linprog::q;
///
/// let t = RatingTable::from_i64(&[&[1, 2], &[3, 5], &[2, 2]])?;
/// let a = icc_anova(&t)?;
/// assert_eq!((a.msr, a.msc, a.mse), (q(7, 2), q(3, 2), q(1, 2)));
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for a table with missing cells or
/// fewer than two items or two raters.
pub fn icc_anova(table: &RatingTable) -> Result<IccAnova, SymplexError> {
    let op = "icc";
    let x = table
        .complete_rows()
        .map_err(|_| invalid(op, "the intraclass correlation needs a complete table"))?;
    let (n, k) = (x.len(), table.n_raters());
    if n < 2 {
        return Err(invalid(op, "needs at least two items"));
    }
    if k < 2 {
        return Err(invalid(op, "needs at least two raters"));
    }
    let grand = sum(x.iter().flatten().cloned()) / qu(n * k);
    let row_means: Vec<Q> = x.iter().map(|r| sum(r.iter().cloned()) / qu(k)).collect();
    let col_means: Vec<Q> = (0..k)
        .map(|j| sum(x.iter().map(|r| r[j].clone())) / qu(n))
        .collect();
    let ssr = qu(k) * sum(row_means.iter().map(|m| square(&(m - &grand))));
    let ssc = qu(n) * sum(col_means.iter().map(|m| square(&(m - &grand))));
    let sst = sum(x.iter().flatten().map(|v| square(&(v - &grand))));
    let sse = &sst - &ssr - &ssc;
    let ssw = &ssc + &sse;
    Ok(IccAnova {
        msr: ssr / qu(n - 1),
        msc: ssc / qu(k - 1),
        mse: sse / qu((n - 1) * (k - 1)),
        msw: ssw / qu(n * (k - 1)),
    })
}

/// The intraclass correlation of a complete `n × k` [`RatingTable`] in
/// the given form (Shrout & Fleiss 1979, *Psychological Bulletin* 86,
/// Table 4; see [`IccForm`] for the formulas).
///
/// ```
/// use symplex::stats::agreement::{icc, IccForm, RatingTable};
/// use symplex::linprog::q;
///
/// let t = RatingTable::from_i64(&[&[1, 2], &[3, 5], &[2, 2]])?;
/// // MSR = 7/2, MSE = 1/2 → ICC(3,1) = (7/2 − 1/2) / (7/2 + 1/2) = 3/4
/// assert_eq!(icc(&t, IccForm::Icc3Single)?, q(3, 4));
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// As [`icc_anova`], and [`SymplexError::InvalidArgument`] when the
/// form's denominator is zero.
pub fn icc(table: &RatingTable, form: IccForm) -> Result<Q, SymplexError> {
    let a = icc_anova(table)?;
    let (n, k) = (qu(table.n_items()), qu(table.n_raters()));
    let k1 = &k - Q::one();
    let (num, den) = match form {
        IccForm::Icc1 => (&a.msr - &a.msw, &a.msr + &k1 * &a.msw),
        IccForm::Icc1Average => (&a.msr - &a.msw, a.msr.clone()),
        IccForm::Icc2Single => (
            &a.msr - &a.mse,
            &a.msr + &k1 * &a.mse + &k * (&a.msc - &a.mse) / &n,
        ),
        IccForm::Icc2Average => (&a.msr - &a.mse, &a.msr + (&a.msc - &a.mse) / &n),
        IccForm::Icc3Single => (&a.msr - &a.mse, &a.msr + &k1 * &a.mse),
        IccForm::Icc3Average => (&a.msr - &a.mse, a.msr.clone()),
    };
    if den.is_zero() {
        return Err(invalid(
            "icc",
            "the denominator is zero (no variance between items), the ICC is undefined",
        ));
    }
    Ok(num / den)
}

// ── Kendall's W ──────────────────────────────────────────────────────

/// Kendall's coefficient of concordance `W` for `m` raters (columns)
/// scoring `n` items (rows), with the tie correction (Kendall & Babington
/// Smith 1939; Siegel & Castellan 1988):
///
/// `W = 12 Σ_i (R_i − R̄)² / (m² (n³ − n) − m Σ_j T_j)`,
///
/// where `R_i` is the sum over raters of the (average) rank of item `i`
/// within that rater's column and `T_j = Σ (t³ − t)` over the tie groups
/// of rater `j`.  `W = χ²_Friedman / (m (n − 1))`.
///
/// ```
/// use symplex::stats::agreement::{kendall_w, RatingTable};
/// use symplex::linprog::q;
///
/// // Three raters rank four items; rows are items.
/// let t = RatingTable::from_i64(&[&[1, 1, 2], &[2, 3, 1], &[3, 2, 3], &[4, 4, 4]])?;
/// // scipy: friedmanchisquare(*rows).statistic / (m(n−1)) = 0.7777… = 7/9
/// assert_eq!(kendall_w(&t)?, q(7, 9));
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for a table with missing cells,
/// fewer than two items or raters, or every rater tying all items.
pub fn kendall_w(table: &RatingTable) -> Result<Q, SymplexError> {
    let op = "kendall_w";
    let x = table
        .complete_rows()
        .map_err(|_| invalid(op, "Kendall's W needs a complete table"))?;
    let (n, m) = (x.len(), table.n_raters());
    if n < 2 {
        return Err(invalid(op, "needs at least two items"));
    }
    if m < 2 {
        return Err(invalid(op, "needs at least two raters"));
    }
    let mut rank_sums = vec![Q::zero(); n];
    let mut ties = BigInt::zero();
    for j in 0..m {
        let col: Vec<Q> = x.iter().map(|r| r[j].clone()).collect();
        for (i, r) in data::ranks(&col).into_iter().enumerate() {
            rank_sums[i] += r;
        }
        for t in data::tie_sizes(&col) {
            let t = BigInt::from(t);
            ties += &t * &t * &t - &t;
        }
    }
    let mean = sum(rank_sums.iter().cloned()) / qu(n);
    let s = sum(rank_sums.iter().map(|r| square(&(r - &mean))));
    let (nb, mb) = (BigInt::from(n), BigInt::from(m));
    let denom = &mb * &mb * (&nb * &nb * &nb - &nb) - &mb * ties;
    if denom.is_zero() {
        return Err(invalid(op, "every rater ties all items, W is undefined"));
    }
    Ok(qi(12) * s / Q::from_integer(denom))
}
