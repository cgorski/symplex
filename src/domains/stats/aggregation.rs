//! Turning many raters' labels into one answer: majority, plurality and
//! weighted votes (exact), the Dawid–Skene EM model of rater confusion,
//! Bradley–Terry strengths from pairwise comparisons, and the worker
//! quality helpers that go with them (accuracy against gold labels,
//! per-category precision / recall / F₁, gold-question screening).
//!
//! **Rule:** a function lives here iff it aggregates several raters'
//! labels into an answer or scores a rater against one.  The confidence
//! intervals for a proportion ([`proportion_interval`] and its exact
//! forms) are interval estimates and moved to [`super::estimation`] in
//! 0.18; they are re-exported here for one release.
//!
//! Labels are category indices `0..n_categories`; a [`LabelTable`] holds
//! them items × raters with `None` for a missing label.  Votes and
//! accuracies are exact ([`Q`]); the two iterative estimators
//! ([`dawid_skene`], [`bradley_terry`]) are `f64`.
//!
//! ```
//! use symplex::stats::aggregation::{majority_vote, worker_accuracy};
//! use symplex::linprog::q;
//!
//! let v = majority_vote(&[Some(2), Some(0), Some(2), None, Some(1)]);
//! assert_eq!((v.winner, v.counts), (Some(2), vec![1, 1, 2]));
//! let acc = worker_accuracy(&[Some(2), Some(0), Some(2), None, Some(1)], &[2, 0, 1, 1, 1])?;
//! assert_eq!((acc.correct, acc.answered, acc.accuracy), (3, 4, Some(q(3, 4))));
//! # Ok::<(), symplex::prelude::SymplexError>(())
//! ```

use num_traits::{One, Zero};

use super::common::{invalid, qu};
use crate::base::errors::SymplexError;
use crate::domains::stats::data::Q;

/// Moved to [`stats::estimation`](super::estimation) in 0.18; this
/// re-export is kept for one release.
pub use super::estimation::{
    IntervalMethod, proportion_interval, proportion_interval_exact, proportion_interval_symbolic,
    z_for_confidence,
};

// ── Label tables ─────────────────────────────────────────────────────

/// Labels of `n_items` items by `n_raters` raters (rows are items), each
/// a category index below `n_categories` or `None` when missing.
///
/// ```
/// use symplex::stats::aggregation::LabelTable;
///
/// let t = LabelTable::from_rows(&[&[Some(0), Some(0)], &[Some(1), None]], 2)?;
/// assert_eq!((t.n_items(), t.n_raters(), t.n_categories()), (2, 2, 2));
/// assert!(LabelTable::from_rows(&[&[Some(5)]], 2).is_err());
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LabelTable {
    rows: Vec<Vec<Option<usize>>>,
    n_raters: usize,
    n_categories: usize,
}

impl LabelTable {
    /// A table from its rows.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if there is no item, no rater or
    /// no category, the rows differ in length, or a label is
    /// `≥ n_categories`.
    pub fn new(rows: Vec<Vec<Option<usize>>>, n_categories: usize) -> Result<Self, SymplexError> {
        let op = "LabelTable::new";
        if n_categories == 0 {
            return Err(invalid(op, "needs at least one category"));
        }
        let n_raters = match rows.first() {
            Some(r) => r.len(),
            None => return Err(invalid(op, "a label table needs at least one item")),
        };
        if n_raters == 0 {
            return Err(invalid(op, "a label table needs at least one rater"));
        }
        for (i, r) in rows.iter().enumerate() {
            if r.len() != n_raters {
                return Err(invalid(
                    op,
                    format!(
                        "item {i} has {} cells but the table has {n_raters} raters",
                        r.len()
                    ),
                ));
            }
            if let Some(l) = r.iter().flatten().find(|&&l| l >= n_categories) {
                return Err(invalid(
                    op,
                    format!("item {i} has label {l} but there are {n_categories} categories"),
                ));
            }
        }
        Ok(Self {
            rows,
            n_raters,
            n_categories,
        })
    }

    /// A table from borrowed rows.
    pub fn from_rows(rows: &[&[Option<usize>]], n_categories: usize) -> Result<Self, SymplexError> {
        Self::new(rows.iter().map(|r| r.to_vec()).collect(), n_categories)
    }

    /// A complete table (no missing labels).
    pub fn complete(rows: &[&[usize]], n_categories: usize) -> Result<Self, SymplexError> {
        Self::new(
            rows.iter()
                .map(|r| r.iter().map(|&l| Some(l)).collect())
                .collect(),
            n_categories,
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

    /// Number of categories.
    pub fn n_categories(&self) -> usize {
        self.n_categories
    }

    /// The rows (items).
    pub fn rows(&self) -> &[Vec<Option<usize>>] {
        &self.rows
    }

    /// The labels of item `i` (`None` if out of range).
    pub fn item(&self, i: usize) -> Option<&[Option<usize>]> {
        self.rows.get(i).map(|r| r.as_slice())
    }

    /// The labels given by rater `j` (`None` if out of range).
    pub fn rater(&self, j: usize) -> Option<Vec<Option<usize>>> {
        (j < self.n_raters).then(|| self.rows.iter().map(|r| r[j]).collect())
    }

    /// Items × raters × categories indicator counts `n_ikl` (`1` when
    /// rater `k` gave item `i` the label `l`), the input of
    /// [`dawid_skene_counts`].
    pub fn to_counts(&self) -> Vec<Vec<Vec<usize>>> {
        self.rows
            .iter()
            .map(|r| {
                r.iter()
                    .map(|l| {
                        let mut c = vec![0usize; self.n_categories];
                        if let Some(l) = l {
                            c[*l] = 1;
                        }
                        c
                    })
                    .collect()
            })
            .collect()
    }
}

// ── Votes ────────────────────────────────────────────────────────────

/// The outcome of a vote over category indices.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Vote {
    /// The unique category with the most votes (`None` on a tie or when
    /// no vote was cast).
    pub winner: Option<usize>,
    /// The categories sharing the top count, ascending (empty when no
    /// vote was cast).
    pub tied: Vec<usize>,
    /// Votes per category.
    pub counts: Vec<usize>,
}

fn vote_from_counts(counts: Vec<usize>) -> Vote {
    let top = counts.iter().copied().max().unwrap_or(0);
    let tied: Vec<usize> = if top == 0 {
        Vec::new()
    } else {
        counts
            .iter()
            .enumerate()
            .filter_map(|(c, &n)| (n == top).then_some(c))
            .collect()
    };
    let winner = (tied.len() == 1).then(|| tied[0]);
    Vote {
        winner,
        tied,
        counts,
    }
}

/// The category chosen by most raters (missing labels are ignored; the
/// count vector runs to the largest label seen).
///
/// ```
/// use symplex::stats::aggregation::majority_vote;
///
/// let v = majority_vote(&[Some(1), Some(1), Some(0), None]);
/// assert_eq!((v.winner, v.tied, v.counts), (Some(1), vec![1], vec![1, 2]));
/// let tie = majority_vote(&[Some(0), Some(2)]);
/// assert_eq!((tie.winner, tie.tied), (None, vec![0, 2]));
/// ```
pub fn majority_vote(labels: &[Option<usize>]) -> Vote {
    let n = labels.iter().flatten().max().map_or(0, |m| m + 1);
    let mut counts = vec![0usize; n];
    for &l in labels.iter().flatten() {
        counts[l] += 1;
    }
    vote_from_counts(counts)
}

/// The majority vote of every item of a [`LabelTable`] (counts have
/// length `n_categories`).
///
/// ```
/// use symplex::stats::aggregation::{majority_votes, LabelTable};
///
/// let t = LabelTable::from_rows(&[&[Some(0), Some(0), Some(1)], &[Some(1), None, Some(2)]], 3)?;
/// let v = majority_votes(&t);
/// assert_eq!(v[0].winner, Some(0));
/// assert_eq!((v[1].winner, v[1].counts.clone()), (None, vec![0, 1, 1]));
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
pub fn majority_votes(table: &LabelTable) -> Vec<Vote> {
    table
        .rows
        .iter()
        .map(|r| {
            let mut counts = vec![0usize; table.n_categories];
            for &l in r.iter().flatten() {
                counts[l] += 1;
            }
            vote_from_counts(counts)
        })
        .collect()
}

/// A plurality vote with a quorum: the top category wins only if it has
/// at least `threshold` of the votes cast (`threshold` in `[0, 1]`).
///
/// ```
/// use symplex::stats::aggregation::plurality;
/// use symplex::linprog::q;
///
/// let labels = [Some(0), Some(0), Some(1), Some(2)];
/// assert_eq!(plurality(&labels, &q(1, 2))?.winner, Some(0));
/// assert_eq!(plurality(&labels, &q(2, 3))?.winner, None);
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for a threshold outside `[0, 1]`.
pub fn plurality(labels: &[Option<usize>], threshold: &Q) -> Result<Vote, SymplexError> {
    if threshold < &Q::zero() || threshold > &Q::one() {
        return Err(invalid("plurality", "the threshold must lie in [0, 1]"));
    }
    let mut vote = majority_vote(labels);
    if let Some(w) = vote.winner {
        let cast: usize = vote.counts.iter().sum();
        if qu(vote.counts[w]) < threshold * qu(cast) {
            vote.winner = None;
        }
    }
    Ok(vote)
}

/// The outcome of a weighted vote.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightedVote {
    /// The unique category with the largest total weight.
    pub winner: Option<usize>,
    /// The categories sharing the top weight, ascending.
    pub tied: Vec<usize>,
    /// Total weight per category.
    pub scores: Vec<Q>,
}

/// A vote where rater `k`'s label counts `weights[k]` (non-negative,
/// exact).
///
/// A weight of `0` is a vote not cast: it adds nothing to any score, so a
/// label backed only by zero-weight raters cannot win — when every weight
/// is zero the result has `winner: None` and `tied: []` even if a single
/// label was cast, exactly as [`majority_vote`] treats a rater whose
/// label is `None`.  (Compare `majority_vote(&[Some(3)])`, one *counted*
/// vote, which does elect `3`.)  `scores` still runs to the largest label
/// seen.
///
/// ```
/// use symplex::stats::aggregation::weighted_vote;
/// use symplex::linprog::{q, qi};
///
/// let labels = [Some(0), Some(1), Some(1)];
/// let v = weighted_vote(&labels, &[q(3, 1), q(1, 1), q(1, 1)])?;
/// assert_eq!((v.winner, v.scores), (Some(0), vec![q(3, 1), q(2, 1)]));
/// let none = weighted_vote(&[Some(2)], &[qi(0)])?;
/// assert_eq!((none.winner, none.tied, none.scores), (None, vec![], vec![qi(0), qi(0), qi(0)]));
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] if the lengths differ or a weight
/// is negative.
pub fn weighted_vote(
    labels: &[Option<usize>],
    weights: &[Q],
) -> Result<WeightedVote, SymplexError> {
    let op = "weighted_vote";
    if labels.len() != weights.len() {
        return Err(invalid(op, "one weight per rater is needed"));
    }
    if weights.iter().any(|w| w < &Q::zero()) {
        return Err(invalid(op, "weights must be non-negative"));
    }
    let n = labels.iter().flatten().max().map_or(0, |m| m + 1);
    let mut scores = vec![Q::zero(); n];
    for (l, w) in labels.iter().zip(weights) {
        if let Some(l) = l {
            scores[*l] += w;
        }
    }
    let top = scores.iter().max().cloned();
    let tied: Vec<usize> = match top {
        Some(t) if !t.is_zero() => scores
            .iter()
            .enumerate()
            .filter_map(|(c, s)| (*s == t).then_some(c))
            .collect(),
        _ => Vec::new(),
    };
    let winner = (tied.len() == 1).then(|| tied[0]);
    Ok(WeightedVote {
        winner,
        tied,
        scores,
    })
}

// ── Dawid–Skene ──────────────────────────────────────────────────────

/// How the Dawid–Skene EM iteration is started.
#[derive(Clone, Debug, PartialEq)]
pub enum DawidSkeneInit {
    /// Each item's posterior is spread evenly over its majority-vote
    /// winners (uniform when it has no labels).
    MajorityVote,
    /// Explicit items × categories posteriors (rows are normalised).
    Posteriors(Vec<Vec<f64>>),
}

/// Options of [`dawid_skene`].
#[derive(Clone, Debug, PartialEq)]
pub struct DawidSkeneOpts {
    /// Maximum number of EM iterations (default 1000).
    pub max_iter: usize,
    /// Convergence: the largest change of any posterior falls below this
    /// (default `1e-10`).
    pub tol: f64,
    /// Pseudo-count added to every cell of every confusion matrix in the
    /// M-step (default `0`, the original algorithm; a small positive
    /// value keeps rare cells away from exact zeros).
    pub smoothing: f64,
    /// Starting posteriors (default majority vote).
    pub init: DawidSkeneInit,
}

impl Default for DawidSkeneOpts {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1e-10,
            smoothing: 0.0,
            init: DawidSkeneInit::MajorityVote,
        }
    }
}

/// The Dawid–Skene estimates.
#[derive(Clone, Debug, PartialEq)]
pub struct DawidSkene {
    /// Items × categories: `T_ij = P(item i is in class j | labels)`.
    pub posteriors: Vec<Vec<f64>>,
    /// Raters × true class × given label: `π^(k)_jl`.
    pub confusion: Vec<Vec<Vec<f64>>>,
    /// Class prevalences `p_j`.
    pub priors: Vec<f64>,
    /// EM iterations performed.
    pub iterations: usize,
    /// Whether the posteriors changed by less than `tol` in the last
    /// iteration.
    pub converged: bool,
    /// The marginal log-likelihood `Σ_i ln Σ_j p_j Π_k Π_l (π^(k)_jl)^{n_ikl}`
    /// at the returned parameters.
    pub log_likelihood: f64,
}

impl DawidSkene {
    /// The most probable class of every item (the smallest index on a
    /// tie).
    pub fn labels(&self) -> Vec<usize> {
        self.posteriors
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .fold((0usize, f64::NEG_INFINITY), |(bi, bv), (i, &v)| {
                        if v > bv { (i, v) } else { (bi, bv) }
                    })
                    .0
            })
            .collect()
    }
}

fn ln_or_neg_inf(p: f64) -> f64 {
    if p > 0.0 { p.ln() } else { f64::NEG_INFINITY }
}

fn ds_m_step(
    counts: &[Vec<Vec<usize>>],
    t: &[Vec<f64>],
    j: usize,
    smoothing: f64,
) -> (Vec<f64>, Vec<Vec<Vec<f64>>>) {
    let n_items = counts.len();
    let n_raters = counts[0].len();
    let priors: Vec<f64> = (0..j)
        .map(|c| t.iter().map(|row| row[c]).sum::<f64>() / n_items as f64)
        .collect();
    let confusion = (0..n_raters)
        .map(|k| {
            (0..j)
                .map(|c| {
                    let mut num: Vec<f64> = (0..j)
                        .map(|l| {
                            counts
                                .iter()
                                .zip(t)
                                .map(|(item, row)| row[c] * item[k][l] as f64)
                                .sum::<f64>()
                                + smoothing
                        })
                        .collect();
                    let den: f64 = num.iter().sum();
                    if den > 0.0 {
                        for v in &mut num {
                            *v /= den;
                        }
                        num
                    } else {
                        vec![1.0 / j as f64; j]
                    }
                })
                .collect()
        })
        .collect();
    (priors, confusion)
}

fn ds_log_posteriors(item: &[Vec<usize>], priors: &[f64], confusion: &[Vec<Vec<f64>>]) -> Vec<f64> {
    let mut lp: Vec<f64> = priors.iter().map(|&p| ln_or_neg_inf(p)).collect();
    for (k, labels) in item.iter().enumerate() {
        for (l, &c) in labels.iter().enumerate() {
            if c > 0 {
                for (j, v) in lp.iter_mut().enumerate() {
                    *v += c as f64 * ln_or_neg_inf(confusion[k][j][l]);
                }
            }
        }
    }
    lp
}

fn ds_e_step(
    counts: &[Vec<Vec<usize>>],
    priors: &[f64],
    confusion: &[Vec<Vec<f64>>],
    prev: &[Vec<f64>],
) -> Vec<Vec<f64>> {
    counts
        .iter()
        .zip(prev)
        .map(|(item, old)| {
            let lp = ds_log_posteriors(item, priors, confusion);
            let m = lp.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if !m.is_finite() {
                return old.to_vec();
            }
            let w: Vec<f64> = lp.iter().map(|&v| (v - m).exp()).collect();
            let s: f64 = w.iter().sum();
            w.into_iter().map(|v| v / s).collect()
        })
        .collect()
}

fn ds_log_likelihood(
    counts: &[Vec<Vec<usize>>],
    priors: &[f64],
    confusion: &[Vec<Vec<f64>>],
) -> f64 {
    counts
        .iter()
        .map(|item| {
            let lp = ds_log_posteriors(item, priors, confusion);
            let m = lp.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if !m.is_finite() {
                return f64::NEG_INFINITY;
            }
            m + lp.iter().map(|&v| (v - m).exp()).sum::<f64>().ln()
        })
        .sum()
}

fn ds_initial(
    counts: &[Vec<Vec<usize>>],
    j: usize,
    init: &DawidSkeneInit,
) -> Result<Vec<Vec<f64>>, SymplexError> {
    let op = "dawid_skene";
    match init {
        DawidSkeneInit::MajorityVote => Ok(counts
            .iter()
            .map(|item| {
                let totals: Vec<usize> = (0..j).map(|l| item.iter().map(|r| r[l]).sum()).collect();
                let v = vote_from_counts(totals);
                if v.tied.is_empty() {
                    vec![1.0 / j as f64; j]
                } else {
                    let share = 1.0 / v.tied.len() as f64;
                    (0..j)
                        .map(|c| if v.tied.contains(&c) { share } else { 0.0 })
                        .collect()
                }
            })
            .collect()),
        DawidSkeneInit::Posteriors(t) => {
            if t.len() != counts.len() || t.iter().any(|r| r.len() != j) {
                return Err(invalid(
                    op,
                    "the initial posteriors must be items × categories",
                ));
            }
            t.iter()
                .map(|row| {
                    if row.iter().any(|v| !v.is_finite() || *v < 0.0) {
                        return Err(invalid(
                            op,
                            "initial posteriors must be finite and non-negative",
                        ));
                    }
                    let s: f64 = row.iter().sum();
                    if s <= 0.0 {
                        return Err(invalid(op, "an initial posterior row sums to zero"));
                    }
                    Ok(row.iter().map(|v| v / s).collect())
                })
                .collect()
        }
    }
}

/// The Dawid–Skene model from items × raters × categories counts
/// `n_ikl` (how many times rater `k` gave item `i` the label `l`; the
/// general form of Dawid & Skene 1979 where a rater may label an item
/// several times).  See [`dawid_skene`].
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for an empty or ragged table, fewer
/// than two categories, or invalid options.
pub fn dawid_skene_counts(
    counts: &[Vec<Vec<usize>>],
    n_categories: usize,
    opts: &DawidSkeneOpts,
) -> Result<DawidSkene, SymplexError> {
    let op = "dawid_skene";
    let j = n_categories;
    if j < 2 {
        return Err(invalid(op, "needs at least two categories"));
    }
    let n_raters = counts.first().map_or(0, |i| i.len());
    if counts.is_empty() || n_raters == 0 {
        return Err(invalid(op, "needs at least one item and one rater"));
    }
    if counts
        .iter()
        .any(|i| i.len() != n_raters || i.iter().any(|r| r.len() != j))
    {
        return Err(invalid(
            op,
            "the counts must be items × raters × categories",
        ));
    }
    if opts.max_iter == 0 {
        return Err(invalid(op, "max_iter must be positive"));
    }
    if !(opts.tol > 0.0 && opts.tol.is_finite()) {
        return Err(invalid(op, "tol must be a positive finite number"));
    }
    if !(opts.smoothing >= 0.0 && opts.smoothing.is_finite()) {
        return Err(invalid(
            op,
            "smoothing must be a non-negative finite number",
        ));
    }
    let mut t = ds_initial(counts, j, &opts.init)?;
    let mut iterations = 0;
    let mut converged = false;
    for it in 1..=opts.max_iter {
        let (priors, confusion) = ds_m_step(counts, &t, j, opts.smoothing);
        let next = ds_e_step(counts, &priors, &confusion, &t);
        let delta = next
            .iter()
            .zip(&t)
            .flat_map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x - y).abs()))
            .fold(0.0, f64::max);
        t = next;
        iterations = it;
        if delta < opts.tol {
            converged = true;
            break;
        }
    }
    let (priors, confusion) = ds_m_step(counts, &t, j, opts.smoothing);
    let log_likelihood = ds_log_likelihood(counts, &priors, &confusion);
    Ok(DawidSkene {
        posteriors: t,
        confusion,
        priors,
        iterations,
        converged,
        log_likelihood,
    })
}

/// The Dawid–Skene EM estimate of true classes, rater confusion matrices
/// and class prevalences (Dawid & Skene 1979, *Applied Statistics* 28,
/// 20–28).  Iterates, from the [`DawidSkeneInit`] posteriors `T_ij`:
///
/// * M-step: `p_j = Σ_i T_ij / I`,
///   `π^(k)_jl = Σ_i T_ij n_ikl / Σ_l Σ_i T_ij n_ikl` (a class a rater
///   never saw gets a uniform row);
/// * E-step: `T_ij ∝ p_j Π_k Π_l (π^(k)_jl)^{n_ikl}`,
///
/// until the posteriors move by less than `tol` (in log space, so exact
/// zeros are harmless).  Deterministic.
///
/// ```
/// use symplex::stats::aggregation::{dawid_skene, DawidSkeneOpts, LabelTable};
///
/// let t = LabelTable::complete(&[&[0, 0, 1], &[1, 1, 1], &[0, 0, 0], &[1, 0, 1]], 2)?;
/// let ds = dawid_skene(&t, &DawidSkeneOpts::default())?;
/// assert!(ds.converged);
/// assert_eq!(ds.labels(), vec![0, 1, 0, 1]);
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// As [`dawid_skene_counts`].
pub fn dawid_skene(table: &LabelTable, opts: &DawidSkeneOpts) -> Result<DawidSkene, SymplexError> {
    dawid_skene_counts(&table.to_counts(), table.n_categories, opts)
}

// ── Bradley–Terry ────────────────────────────────────────────────────

/// Options of [`bradley_terry`].
#[derive(Clone, Debug, PartialEq)]
pub struct BradleyTerryOpts {
    /// Maximum number of MM iterations (default 10 000).
    pub max_iter: usize,
    /// Convergence: the largest change of any normalised strength falls
    /// below this (default `1e-12`).
    pub tol: f64,
}

impl Default for BradleyTerryOpts {
    fn default() -> Self {
        Self {
            max_iter: 10_000,
            tol: 1e-12,
        }
    }
}

/// Bradley–Terry strengths.
#[derive(Clone, Debug, PartialEq)]
pub struct BradleyTerry {
    /// Strengths `p_i > 0`, normalised to sum to 1: `P(i beats j) = p_i / (p_i + p_j)`.
    pub strengths: Vec<f64>,
    /// MM iterations performed.
    pub iterations: usize,
    /// Whether the strengths moved by less than `tol` in the last
    /// iteration.
    pub converged: bool,
}

/// One pairwise comparison: player `winner` beat player `loser` (both
/// indices below the number of players).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PairwiseOutcome {
    /// The index of the player who won.
    pub winner: usize,
    /// The index of the player who lost.
    pub loser: usize,
}

/// A wins matrix (`wins[i][j]` = times `i` beat `j`) from a list of
/// [`PairwiseOutcome`]s among `n` players — the input of
/// [`bradley_terry`].
///
/// ```
/// use symplex::stats::aggregation::{wins_matrix, PairwiseOutcome};
///
/// let beat = |winner, loser| PairwiseOutcome { winner, loser };
/// assert_eq!(
///     wins_matrix(&[beat(0, 1), beat(0, 1), beat(1, 2)], 3)?,
///     vec![vec![0, 2, 0], vec![0, 0, 1], vec![0, 0, 0]],
/// );
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for a player index `≥ n` or a
/// player beating itself.
pub fn wins_matrix(
    outcomes: &[PairwiseOutcome],
    n: usize,
) -> Result<Vec<Vec<usize>>, SymplexError> {
    let op = "wins_matrix";
    let mut w = vec![vec![0usize; n]; n];
    for outcome in outcomes {
        let (a, b) = (outcome.winner, outcome.loser);
        if a >= n || b >= n {
            return Err(invalid(
                op,
                format!("player index out of range in ({a}, {b}), n = {n}"),
            ));
        }
        if a == b {
            return Err(invalid(op, format!("player {a} cannot play itself")));
        }
        w[a][b] += 1;
    }
    Ok(w)
}

/// Every player reachable from every other along "beat" edges.
fn strongly_connected(wins: &[Vec<usize>]) -> bool {
    let n = wins.len();
    let reach = |forward: bool| -> bool {
        let mut seen = vec![false; n];
        let mut stack = vec![0usize];
        seen[0] = true;
        while let Some(i) = stack.pop() {
            for j in 0..n {
                let edge = if forward { wins[i][j] } else { wins[j][i] };
                if edge > 0 && !seen[j] {
                    seen[j] = true;
                    stack.push(j);
                }
            }
        }
        seen.iter().all(|&s| s)
    };
    reach(true) && reach(false)
}

/// Maximum-likelihood Bradley–Terry strengths from a wins matrix
/// (`wins[i][j]` = times `i` beat `j`) by the MM algorithm of Hunter
/// (2004, *Ann. Statist.* 32, 384–406):
///
/// `p_i ← W_i / Σ_{j ≠ i} N_ij / (p_i + p_j)`, `W_i = Σ_j wins[i][j]`,
/// `N_ij = wins[i][j] + wins[j][i]`,
///
/// renormalised to `Σ p_i = 1` after every step.  The estimate exists
/// iff the beat graph is strongly connected (Ford 1957); otherwise an
/// error.
///
/// ```
/// use symplex::stats::aggregation::{bradley_terry, BradleyTerryOpts};
///
/// // Every pair split 1–1: equal strengths.
/// let w = vec![vec![0, 1, 1], vec![1, 0, 1], vec![1, 1, 0]];
/// let bt = bradley_terry(&w, &BradleyTerryOpts::default())?;
/// assert!(bt.strengths.iter().all(|p| (p - 1.0 / 3.0).abs() < 1e-12));
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for a non-square matrix with fewer
/// than two players, a non-zero diagonal, a beat graph that is not
/// strongly connected, or invalid options.
pub fn bradley_terry(
    wins: &[Vec<usize>],
    opts: &BradleyTerryOpts,
) -> Result<BradleyTerry, SymplexError> {
    let op = "bradley_terry";
    let n = wins.len();
    if n < 2 || wins.iter().any(|r| r.len() != n) {
        return Err(invalid(
            op,
            "the wins matrix must be square with at least two players",
        ));
    }
    if (0..n).any(|i| wins[i][i] != 0) {
        return Err(invalid(op, "the diagonal of the wins matrix must be zero"));
    }
    if opts.max_iter == 0 {
        return Err(invalid(op, "max_iter must be positive"));
    }
    if !(opts.tol > 0.0 && opts.tol.is_finite()) {
        return Err(invalid(op, "tol must be a positive finite number"));
    }
    if !strongly_connected(wins) {
        return Err(invalid(
            op,
            "the beat graph is not strongly connected (Ford 1957): the maximum-likelihood strengths do not exist",
        ));
    }
    let total_wins: Vec<f64> = wins
        .iter()
        .map(|r| r.iter().sum::<usize>() as f64)
        .collect();
    let mut p = vec![1.0 / n as f64; n];
    let mut iterations = 0;
    let mut converged = false;
    for it in 1..=opts.max_iter {
        let mut next: Vec<f64> = (0..n)
            .map(|i| {
                let denom: f64 = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| (wins[i][j] + wins[j][i]) as f64 / (p[i] + p[j]))
                    .sum();
                total_wins[i] / denom
            })
            .collect();
        let s: f64 = next.iter().sum();
        for v in &mut next {
            *v /= s;
        }
        let delta = next
            .iter()
            .zip(&p)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        p = next;
        iterations = it;
        if delta < opts.tol {
            converged = true;
            break;
        }
    }
    Ok(BradleyTerry {
        strengths: p,
        iterations,
        converged,
    })
}

// ── Worker quality ───────────────────────────────────────────────────

/// A worker's accuracy against gold labels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Accuracy {
    /// Items answered correctly.
    pub correct: usize,
    /// Items answered (label present).
    pub answered: usize,
    /// `correct / answered`, `None` when nothing was answered.
    pub accuracy: Option<Q>,
}

/// A worker's accuracy on items with gold labels (missing labels are
/// not counted as answered).
///
/// ```
/// use symplex::stats::aggregation::worker_accuracy;
/// use symplex::linprog::q;
///
/// let acc = worker_accuracy(&[Some(0), Some(1), None, Some(1)], &[0, 0, 1, 1])?;
/// assert_eq!((acc.correct, acc.answered, acc.accuracy), (2, 3, Some(q(2, 3))));
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] if the lengths differ.
pub fn worker_accuracy(labels: &[Option<usize>], gold: &[usize]) -> Result<Accuracy, SymplexError> {
    if labels.len() != gold.len() {
        return Err(invalid(
            "worker_accuracy",
            "one gold label per item is needed",
        ));
    }
    let answered = labels.iter().flatten().count();
    let correct = labels
        .iter()
        .zip(gold)
        .filter(|(l, g)| **l == Some(**g))
        .count();
    let accuracy = (answered > 0).then(|| qu(correct) / qu(answered));
    Ok(Accuracy {
        correct,
        answered,
        accuracy,
    })
}

/// Precision, recall and F₁ of one category, over the answered items.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CategoryMetrics {
    /// The category.
    pub category: usize,
    /// Answered `category`, gold `category`.
    pub true_positives: usize,
    /// Answered `category`, gold something else.
    pub false_positives: usize,
    /// Answered something else, gold `category`.
    pub false_negatives: usize,
    /// `tp / (tp + fp)`, `None` when the category was never answered.
    pub precision: Option<Q>,
    /// `tp / (tp + fn)`, `None` when the category never occurs in the
    /// answered gold items.
    pub recall: Option<Q>,
    /// `2 tp / (2 tp + fp + fn)`, `None` when all three counts are zero.
    pub f1: Option<Q>,
}

/// Per-category precision, recall and F₁ of a worker against gold
/// labels, over the items the worker answered.
///
/// ```
/// use symplex::stats::aggregation::category_metrics;
/// use symplex::linprog::q;
///
/// let m = category_metrics(&[Some(0), Some(0), Some(1), None], &[0, 1, 1, 0], 2)?;
/// assert_eq!((m[0].precision.clone(), m[0].recall.clone(), m[0].f1.clone()), (Some(q(1, 2)), Some(q(1, 1)), Some(q(2, 3))));
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] if the lengths differ or a label or
/// gold value is `≥ n_categories`.
pub fn category_metrics(
    labels: &[Option<usize>],
    gold: &[usize],
    n_categories: usize,
) -> Result<Vec<CategoryMetrics>, SymplexError> {
    let op = "category_metrics";
    if labels.len() != gold.len() {
        return Err(invalid(op, "one gold label per item is needed"));
    }
    if labels
        .iter()
        .flatten()
        .chain(gold)
        .any(|&l| l >= n_categories)
    {
        return Err(invalid(op, format!("a label is not below {n_categories}")));
    }
    let ratio = |num: usize, den: usize| (den > 0).then(|| qu(num) / qu(den));
    Ok((0..n_categories)
        .map(|c| {
            let (mut tp, mut fp, mut fneg) = (0usize, 0usize, 0usize);
            for (l, &g) in labels.iter().zip(gold) {
                if let Some(l) = l {
                    match (*l == c, g == c) {
                        (true, true) => tp += 1,
                        (true, false) => fp += 1,
                        (false, true) => fneg += 1,
                        (false, false) => {}
                    }
                }
            }
            CategoryMetrics {
                category: c,
                true_positives: tp,
                false_positives: fp,
                false_negatives: fneg,
                precision: ratio(tp, tp + fp),
                recall: ratio(tp, tp + fneg),
                f1: ratio(2 * tp, 2 * tp + fp + fneg),
            }
        })
        .collect())
}

/// Pass / fail of every rater on the gold items: `Some(true)` when the
/// rater's accuracy on the gold items answered is at least `threshold`,
/// `None` for a rater who answered no gold item.  `gold[i]` is the gold
/// label of item `i` or `None` for a non-gold item.
///
/// ```
/// use symplex::stats::aggregation::{gold_screening, LabelTable};
/// use symplex::linprog::q;
///
/// let t = LabelTable::from_rows(&[&[Some(0), Some(1), None], &[Some(1), Some(1), None], &[Some(0), Some(0), Some(0)]], 2)?;
/// let gold = [Some(0), Some(1), None];
/// assert_eq!(gold_screening(&t, &gold, &q(3, 4))?, vec![Some(true), Some(false), None]);
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] if `gold` does not have one entry
/// per item, a gold label is out of range, or the threshold is outside
/// `[0, 1]`.
pub fn gold_screening(
    table: &LabelTable,
    gold: &[Option<usize>],
    threshold: &Q,
) -> Result<Vec<Option<bool>>, SymplexError> {
    let op = "gold_screening";
    if gold.len() != table.n_items() {
        return Err(invalid(op, "one gold entry per item is needed"));
    }
    if gold.iter().flatten().any(|&g| g >= table.n_categories) {
        return Err(invalid(op, "a gold label is not a category"));
    }
    if threshold < &Q::zero() || threshold > &Q::one() {
        return Err(invalid(op, "the threshold must lie in [0, 1]"));
    }
    Ok((0..table.n_raters)
        .map(|j| {
            let (labels, golds): (Vec<Option<usize>>, Vec<usize>) = table
                .rows
                .iter()
                .zip(gold)
                .filter_map(|(r, g)| g.map(|g| (r[j], g)))
                .unzip();
            let (mut answered, mut correct) = (0usize, 0usize);
            for (l, g) in labels.iter().zip(&golds) {
                if let Some(l) = l {
                    answered += 1;
                    if l == g {
                        correct += 1;
                    }
                }
            }
            (answered > 0).then(|| qu(correct) >= threshold * qu(answered))
        })
        .collect())
}

// ── Shared initialisation of the rater models ────────────────────────

/// Every item's posterior spread evenly over its majority-vote winners
/// (uniform when it has no labels) — the `MajorityVote` start of both
/// EM models.  Bit-identical to the `DawidSkeneInit::MajorityVote` branch
/// of [`ds_initial`].
fn majority_vote_posteriors(table: &LabelTable) -> Vec<Vec<f64>> {
    let k = table.n_categories;
    majority_votes(table)
        .into_iter()
        .map(|v| {
            if v.tied.is_empty() {
                vec![1.0 / k as f64; k]
            } else {
                let share = 1.0 / v.tied.len() as f64;
                (0..k)
                    .map(|c| if v.tied.contains(&c) { share } else { 0.0 })
                    .collect()
            }
        })
        .collect()
}

/// Caller-supplied starting posteriors checked (items × categories,
/// finite, non-negative, no zero row) and normalised row by row, with
/// the errors named after `op`.  The same rules as the
/// `DawidSkeneInit::Posteriors` branch of [`ds_initial`].
fn normalised_posteriors(
    op: &'static str,
    t: &[Vec<f64>],
    n_items: usize,
    k: usize,
) -> Result<Vec<Vec<f64>>, SymplexError> {
    if t.len() != n_items || t.iter().any(|r| r.len() != k) {
        return Err(invalid(
            op,
            "the initial posteriors must be items × categories",
        ));
    }
    t.iter()
        .map(|row| {
            if row.iter().any(|v| !v.is_finite() || *v < 0.0) {
                return Err(invalid(
                    op,
                    "initial posteriors must be finite and non-negative",
                ));
            }
            let s: f64 = row.iter().sum();
            if s <= 0.0 {
                return Err(invalid(op, "an initial posterior row sums to zero"));
            }
            Ok(row.iter().map(|v| v / s).collect())
        })
        .collect()
}

/// The `max_iter` / `tol` / `smoothing` checks shared by the EM models.
fn check_em_opts(
    op: &'static str,
    max_iter: usize,
    tol: f64,
    smoothing: f64,
) -> Result<(), SymplexError> {
    if max_iter == 0 {
        return Err(invalid(op, "max_iter must be positive"));
    }
    if !(tol > 0.0 && tol.is_finite()) {
        return Err(invalid(op, "tol must be a positive finite number"));
    }
    if !(smoothing >= 0.0 && smoothing.is_finite()) {
        return Err(invalid(
            op,
            "smoothing must be a non-negative finite number",
        ));
    }
    Ok(())
}

/// The largest absolute change between two items × categories tables.
fn max_abs_change(next: &[Vec<f64>], prev: &[Vec<f64>]) -> f64 {
    next.iter()
        .zip(prev)
        .flat_map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x - y).abs()))
        .fold(0.0, f64::max)
}

/// The most probable category of every row (the smallest index on a
/// tie).
fn argmax_rows(posteriors: &[Vec<f64>]) -> Vec<usize> {
    posteriors
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .fold((0usize, f64::NEG_INFINITY), |(bi, bv), (i, &v)| {
                    if v > bv { (i, v) } else { (bi, bv) }
                })
                .0
        })
        .collect()
}

// ── MAP Dawid–Skene ──────────────────────────────────────────────────

/// Dirichlet priors of [`dawid_skene_map`]: one on the class prevalences
/// and one, shared by every rater, on each row of the confusion matrix.
///
/// The M-step is the posterior *mode*, `(count + α − 1) / Σ (count + α − 1)`,
/// so every `α` must be `≥ 1` (an `α < 1` puts the mode on the boundary,
/// where the multinomial log-likelihood is unbounded); `α = 1` everywhere
/// is the flat prior and reproduces [`dawid_skene`] exactly.  In this
/// parametrisation the pseudo-count of a cell is `α − 1`, so
/// `symmetric(k, 1, 1 + s, 1 + s)` is [`dawid_skene`] with
/// `smoothing = s`.
///
/// ```
/// use symplex::stats::aggregation::DawidSkenePriors;
///
/// let p = DawidSkenePriors::symmetric(3, 2.0, 5.0, 1.5);
/// assert_eq!(p.class_prior_alpha, vec![2.0, 2.0, 2.0]);
/// assert_eq!(p.confusion_alpha[1], vec![1.5, 5.0, 1.5]);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct DawidSkenePriors {
    /// `α_j` of the `Dirichlet(α)` prior on the class prevalences `p_j`
    /// (length `n_categories`).
    pub class_prior_alpha: Vec<f64>,
    /// `β_jl` of the `Dirichlet(β_j·)` prior on row `j` (true class) of
    /// every rater's confusion matrix, indexed `[true][observed]`
    /// (`n_categories × n_categories`, shared across raters).
    pub confusion_alpha: Vec<Vec<f64>>,
}

impl DawidSkenePriors {
    /// The same `class_alpha` for every class, `diag_alpha` on the
    /// diagonal of the confusion prior (a rater tends to give the true
    /// label) and `off_diag_alpha` elsewhere.
    pub fn symmetric(
        n_categories: usize,
        class_alpha: f64,
        diag_alpha: f64,
        off_diag_alpha: f64,
    ) -> Self {
        let k = n_categories;
        Self {
            class_prior_alpha: vec![class_alpha; k],
            confusion_alpha: (0..k)
                .map(|j| {
                    (0..k)
                        .map(|l| if j == l { diag_alpha } else { off_diag_alpha })
                        .collect()
                })
                .collect(),
        }
    }
}

/// The MAP M-step: `p_j = (Σ_i T_ij + a_j) / (I + Σ_j a_j)` and
/// `π^(k)_jl = (Σ_i T_ij n_ikl + b_jl + s) / Σ_l (…)` with the prior
/// offsets `a = α − 1`, `b = β − 1` and the extra pseudo-count `s`.  A
/// row whose denominator is zero (flat prior, no smoothing, a class the
/// rater never saw) is uniform, as in [`ds_m_step`]; with all offsets
/// zero the arithmetic is bit-identical to [`ds_m_step`].
fn ds_map_m_step(
    counts: &[Vec<Vec<usize>>],
    t: &[Vec<f64>],
    j: usize,
    class_offset: &[f64],
    conf_offset: &[Vec<f64>],
    smoothing: f64,
) -> (Vec<f64>, Vec<Vec<Vec<f64>>>) {
    let n_items = counts.len();
    let n_raters = counts[0].len();
    let class_den = n_items as f64 + class_offset.iter().sum::<f64>();
    let priors: Vec<f64> = (0..j)
        .map(|c| (t.iter().map(|row| row[c]).sum::<f64>() + class_offset[c]) / class_den)
        .collect();
    let confusion = (0..n_raters)
        .map(|k| {
            (0..j)
                .map(|c| {
                    let mut num: Vec<f64> = (0..j)
                        .map(|l| {
                            counts
                                .iter()
                                .zip(t)
                                .map(|(item, row)| row[c] * item[k][l] as f64)
                                .sum::<f64>()
                                + conf_offset[c][l]
                                + smoothing
                        })
                        .collect();
                    let den: f64 = num.iter().sum();
                    if den > 0.0 {
                        for v in &mut num {
                            *v /= den;
                        }
                        num
                    } else {
                        vec![1.0 / j as f64; j]
                    }
                })
                .collect()
        })
        .collect();
    (priors, confusion)
}

/// Maximum-a-posteriori Dawid–Skene: the EM of [`dawid_skene`] with
/// Dirichlet priors on the class prevalences and on every confusion
/// row, whose M-step is the posterior mode (the E-step is unchanged):
///
/// * M-step: `p_j = (Σ_i T_ij + α_j − 1) / (I + Σ_j (α_j − 1))`,
///   `π^(k)_jl = (Σ_i T_ij n_ikl + β_jl − 1) / Σ_l (Σ_i T_ij n_ikl + β_jl − 1)`;
/// * E-step: `T_ij ∝ p_j Π_k Π_l (π^(k)_jl)^{n_ikl}`.
///
/// The priors regularise the maximum-likelihood estimate: a rater who
/// gave a single label has a degenerate MLE row `(1, 0, …)`, the MAP row
/// `(n + β_jj − 1, β_jl − 1, …) / (n + Σ_l (β_jl − 1))` stays in the
/// interior.  With every `α = β = 1` the result is bit-identical to
/// [`dawid_skene`] (same start, same iterations); `opts.smoothing` is
/// added to every confusion cell on top of `β − 1`, so it may be left at
/// `0`.  `log_likelihood` is the marginal log-likelihood at the returned
/// parameters (not the log-posterior).  Deterministic.
///
/// ```
/// use symplex::stats::aggregation::{dawid_skene_map, DawidSkeneOpts, DawidSkenePriors, LabelTable};
///
/// // Rater 2 always answers 0: its MLE confusion rows are (1, 0); the MAP rows are not.
/// let t = LabelTable::complete(&[&[0, 0, 0], &[1, 1, 0], &[0, 0, 0], &[1, 1, 0]], 2)?;
/// let priors = DawidSkenePriors::symmetric(2, 1.0, 3.0, 2.0);
/// let ds = dawid_skene_map(&t, &priors, &DawidSkeneOpts::default())?;
/// assert!(ds.converged);
/// assert_eq!(ds.labels(), vec![0, 1, 0, 1]);
/// assert!(ds.confusion[2][1][0] < 1.0 && ds.confusion[2][1][1] > 0.0);
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for fewer than two categories, a
/// prior of the wrong shape or with an entry that is not finite or is
/// `< 1`, or invalid options.
pub fn dawid_skene_map(
    table: &LabelTable,
    priors: &DawidSkenePriors,
    opts: &DawidSkeneOpts,
) -> Result<DawidSkene, SymplexError> {
    let op = "dawid_skene_map";
    let j = table.n_categories;
    if j < 2 {
        return Err(invalid(op, "needs at least two categories"));
    }
    check_em_opts(op, opts.max_iter, opts.tol, opts.smoothing)?;
    if priors.class_prior_alpha.len() != j {
        return Err(invalid(
            op,
            format!("the class prior needs one α per category ({j})"),
        ));
    }
    if priors.confusion_alpha.len() != j || priors.confusion_alpha.iter().any(|r| r.len() != j) {
        return Err(invalid(
            op,
            format!("the confusion prior must be {j} × {j} (true × observed)"),
        ));
    }
    let valid_alpha = |a: &f64| a.is_finite() && *a >= 1.0;
    if !priors.class_prior_alpha.iter().all(valid_alpha)
        || !priors.confusion_alpha.iter().flatten().all(valid_alpha)
    {
        return Err(invalid(
            op,
            "every prior α must be a finite number ≥ 1 (the M-step is the posterior mode)",
        ));
    }
    let class_offset: Vec<f64> = priors.class_prior_alpha.iter().map(|a| a - 1.0).collect();
    let conf_offset: Vec<Vec<f64>> = priors
        .confusion_alpha
        .iter()
        .map(|r| r.iter().map(|b| b - 1.0).collect())
        .collect();
    let counts = table.to_counts();
    let mut t = match &opts.init {
        DawidSkeneInit::MajorityVote => majority_vote_posteriors(table),
        DawidSkeneInit::Posteriors(t0) => normalised_posteriors(op, t0, table.n_items(), j)?,
    };
    let mut iterations = 0;
    let mut converged = false;
    for it in 1..=opts.max_iter {
        let (p, pi) = ds_map_m_step(&counts, &t, j, &class_offset, &conf_offset, opts.smoothing);
        let next = ds_e_step(&counts, &p, &pi, &t);
        let delta = max_abs_change(&next, &t);
        t = next;
        iterations = it;
        if delta < opts.tol {
            converged = true;
            break;
        }
    }
    let (p, pi) = ds_map_m_step(&counts, &t, j, &class_offset, &conf_offset, opts.smoothing);
    let log_likelihood = ds_log_likelihood(&counts, &p, &pi);
    Ok(DawidSkene {
        posteriors: t,
        confusion: pi,
        priors: p,
        iterations,
        converged,
        log_likelihood,
    })
}

// ── MACE ─────────────────────────────────────────────────────────────

/// How the [`mace`] EM iteration is started (posteriors over the true
/// labels; the first M-step takes `P(rater r spammed on item i) =
/// 1 − T_{i, a_ir}`, the posterior mass off the label the rater gave).
#[derive(Clone, Debug, PartialEq)]
pub enum MaceInit {
    /// Each item's posterior is spread evenly over its majority-vote
    /// winners (uniform when it has no labels).
    MajorityVote,
    /// Every item uniform over the labels: the first M-step then puts
    /// `θ_r ≈ 1/K` and `ξ_r` at the rater's empirical label frequencies.
    Uniform,
    /// Explicit items × categories posteriors (rows are normalised).
    Posteriors(Vec<Vec<f64>>),
}

/// Options of [`mace`].
#[derive(Clone, Debug, PartialEq)]
pub struct MaceOpts {
    /// Maximum number of EM iterations (default 1000).
    pub max_iter: usize,
    /// Convergence: the largest change of any posterior falls below this
    /// (default `1e-10`).
    pub tol: f64,
    /// Dirichlet add-`δ` smoothing of the M-step: `δ` is added to both
    /// counts behind `θ_r` and to every cell of `ξ_r` before normalising
    /// (default `0.1`; `0` is plain maximum likelihood, where a rater who
    /// never spammed gets `θ_r = 1` exactly).  MACE's reference
    /// implementation exposes the same constant as `--smoothing`, with
    /// default `0.01 / K`.
    pub smoothing: f64,
    /// Starting posteriors (default majority vote).
    pub init: MaceInit,
}

impl Default for MaceOpts {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1e-10,
            smoothing: 0.1,
            init: MaceInit::MajorityVote,
        }
    }
}

/// The MACE estimates.
#[derive(Clone, Debug, PartialEq)]
pub struct Mace {
    /// Items × categories: `T_it = P(item i has true label t | labels)`.
    pub posteriors: Vec<Vec<f64>>,
    /// `θ_r`: the probability that rater `r` copies the true label
    /// rather than spamming.
    pub competence: Vec<f64>,
    /// `ξ_r`: the distribution rater `r` draws a label from when
    /// spamming (raters × categories).
    pub spam_distribution: Vec<Vec<f64>>,
    /// EM iterations performed.
    pub iterations: usize,
    /// Whether the posteriors changed by less than `tol` in the last
    /// iteration.
    pub converged: bool,
    /// The marginal log-likelihood
    /// `Σ_i ln (1/K) Σ_t Π_r (θ_r [a_ir = t] + (1 − θ_r) ξ_r(a_ir))` at the
    /// returned parameters (the true-label prior is uniform, so an item
    /// without labels contributes `0`).
    pub log_likelihood: f64,
}

impl Mace {
    /// The most probable label of every item (the smallest index on a
    /// tie).
    pub fn labels(&self) -> Vec<usize> {
        argmax_rows(&self.posteriors)
    }
}

/// The rater parameters of the MACE model.
struct MaceParams {
    /// `θ_r`.
    competence: Vec<f64>,
    /// `ξ_r`, raters × categories.
    spam: Vec<Vec<f64>>,
}

/// The responsibilities of one MACE E-step.
struct MaceResponsibilities {
    /// `T_it`, items × categories.
    posteriors: Vec<Vec<f64>>,
    /// `ρ_ir = P(rater r spammed on item i | labels)`, items × raters
    /// (`0` where the label is missing).
    spamming: Vec<Vec<f64>>,
}

/// The log of every item's per-label likelihoods
/// `ln Π_r (θ_r [a_ir = t] + (1 − θ_r) ξ_r(a_ir))` for `t = 0..k`.
fn mace_log_terms(row: &[Option<usize>], k: usize, p: &MaceParams) -> Vec<f64> {
    let mut lp = vec![0.0f64; k];
    for (r, a) in row.iter().enumerate() {
        let Some(a) = a else { continue };
        let theta = p.competence[r];
        let spam = (1.0 - theta) * p.spam[r][*a];
        for (t, v) in lp.iter_mut().enumerate() {
            *v += ln_or_neg_inf(if t == *a { theta + spam } else { spam });
        }
    }
    lp
}

/// E-step: `T_it ∝ Π_r (θ_r [a_ir = t] + (1 − θ_r) ξ_r(a_ir))` and
/// `ρ_ir = 1 − T_{i,a_ir} · θ_r / (θ_r + (1 − θ_r) ξ_r(a_ir))` (a rater
/// whose label differs from the true one certainly spammed; one whose
/// label agrees spammed with the posterior odds of the spam branch).  An
/// item whose labels have probability zero under the parameters gets a
/// uniform posterior.
fn mace_e_step(rows: &[Vec<Option<usize>>], k: usize, p: &MaceParams) -> MaceResponsibilities {
    let mut posteriors = Vec::with_capacity(rows.len());
    let mut spamming = Vec::with_capacity(rows.len());
    for row in rows {
        let lp = mace_log_terms(row, k, p);
        let m = lp.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let gamma: Vec<f64> = if m.is_finite() {
            let w: Vec<f64> = lp.iter().map(|&v| (v - m).exp()).collect();
            let s: f64 = w.iter().sum();
            w.into_iter().map(|v| v / s).collect()
        } else {
            vec![1.0 / k as f64; k]
        };
        let rho: Vec<f64> = row
            .iter()
            .enumerate()
            .map(|(r, a)| {
                let Some(a) = a else { return 0.0 };
                let theta = p.competence[r];
                let den = theta + (1.0 - theta) * p.spam[r][*a];
                let copied = if den > 0.0 {
                    gamma[*a] * theta / den
                } else {
                    0.0
                };
                1.0 - copied
            })
            .collect();
        posteriors.push(gamma);
        spamming.push(rho);
    }
    MaceResponsibilities {
        posteriors,
        spamming,
    }
}

/// M-step with add-`δ` smoothing:
/// `θ_r = (Σ_i (1 − ρ_ir) + δ) / (n_r + 2δ)` and
/// `ξ_r(l) = (Σ_i ρ_ir [a_ir = l] + δ) / (Σ_i ρ_ir + Kδ)`, the sums over
/// the items rater `r` labelled (`n_r` of them).  A zero denominator
/// (`δ = 0`) gives `θ_r = 1/2`, respectively a uniform `ξ_r`.
fn mace_m_step(
    rows: &[Vec<Option<usize>>],
    k: usize,
    resp: &MaceResponsibilities,
    smoothing: f64,
) -> MaceParams {
    let n_raters = rows.first().map_or(0, |r| r.len());
    let mut copied = vec![0.0f64; n_raters];
    let mut spammed = vec![0.0f64; n_raters];
    let mut spam_counts = vec![vec![0.0f64; k]; n_raters];
    for (row, rho) in rows.iter().zip(&resp.spamming) {
        for (r, a) in row.iter().enumerate() {
            if let Some(a) = a {
                copied[r] += 1.0 - rho[r];
                spammed[r] += rho[r];
                spam_counts[r][*a] += rho[r];
            }
        }
    }
    let competence = (0..n_raters)
        .map(|r| {
            let den = copied[r] + spammed[r] + 2.0 * smoothing;
            if den > 0.0 {
                (copied[r] + smoothing) / den
            } else {
                0.5
            }
        })
        .collect();
    let spam = (0..n_raters)
        .map(|r| {
            let den = spammed[r] + k as f64 * smoothing;
            if den > 0.0 {
                spam_counts[r]
                    .iter()
                    .map(|c| (c + smoothing) / den)
                    .collect()
            } else {
                vec![1.0 / k as f64; k]
            }
        })
        .collect();
    MaceParams { competence, spam }
}

/// `Σ_i ln (1/K) Σ_t Π_r P(a_ir | t)`.
fn mace_log_likelihood(rows: &[Vec<Option<usize>>], k: usize, p: &MaceParams) -> f64 {
    let ln_k = (k as f64).ln();
    rows.iter()
        .map(|row| {
            let lp = mace_log_terms(row, k, p);
            let m = lp.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if !m.is_finite() {
                return f64::NEG_INFINITY;
            }
            m + lp.iter().map(|&v| (v - m).exp()).sum::<f64>().ln() - ln_k
        })
        .sum()
}

/// The MACE model of rater competence (Hovy, Berg-Kirkpatrick, Vaswani &
/// Hovy 2013, *Learning Whom to Trust with MACE*, NAACL-HLT, 1120–1130):
/// item `i` has a true label `T_i` drawn uniformly from the `K`
/// categories; rater `r` either copies it (`S_ir = 0`, probability
/// `θ_r`) or *spams* (`S_ir = 1`, probability `1 − θ_r`) a label from a
/// rater-specific distribution `ξ_r` — the generative story of their §2,
/// whose marginal likelihood (their eq. 1) is
/// `Π_i Σ_t (1/K) Π_r (θ_r [a_ir = t] + (1 − θ_r) ξ_r(a_ir))`.  Fitted by
/// the EM of their §3 over the latent `T_i` and `S_ir`:
///
/// * E-step: `T_it ∝ Π_r (θ_r [a_ir = t] + (1 − θ_r) ξ_r(a_ir))` and
///   `ρ_ir = P(S_ir = 1 | labels) = 1 − T_{i,a_ir} θ_r / (θ_r + (1 − θ_r) ξ_r(a_ir))`;
/// * M-step: `θ_r = (Σ_i (1 − ρ_ir) + δ) / (n_r + 2δ)`,
///   `ξ_r(l) = (Σ_i ρ_ir [a_ir = l] + δ) / (Σ_i ρ_ir + Kδ)`
///   (`δ = opts.smoothing`, the add-`δ` of the reference implementation's
///   `--em` mode),
///
/// started from the [`MaceInit`] posteriors with
/// `ρ_ir = 1 − T_{i,a_ir}` and iterated until the posteriors move by less
/// than `tol`.  Deterministic — no random restarts; use
/// `MaceInit::Posteriors` to try several starts.  Missing labels are
/// skipped; an item nobody labelled has a uniform posterior.
///
/// ```
/// use symplex::stats::aggregation::{mace, LabelTable, MaceOpts};
///
/// // Rater 2 answers 0 whatever the item.
/// let t = LabelTable::complete(
///     &[&[0, 0, 0], &[1, 1, 0], &[2, 2, 0], &[0, 0, 0], &[1, 1, 0], &[2, 2, 0]],
///     3,
/// )?;
/// let m = mace(&t, &MaceOpts::default())?;
/// assert!(m.converged);
/// assert_eq!(m.labels(), vec![0, 1, 2, 0, 1, 2]);
/// assert!(m.competence[0] > 0.9 && m.competence[2] < 0.5);
/// assert!(m.spam_distribution[2][0] > 0.8);
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for fewer than two categories or
/// invalid options (including initial posteriors of the wrong shape).
pub fn mace(table: &LabelTable, opts: &MaceOpts) -> Result<Mace, SymplexError> {
    let op = "mace";
    let k = table.n_categories;
    if k < 2 {
        return Err(invalid(op, "needs at least two categories"));
    }
    check_em_opts(op, opts.max_iter, opts.tol, opts.smoothing)?;
    let rows = table.rows();
    let posteriors = match &opts.init {
        MaceInit::MajorityVote => majority_vote_posteriors(table),
        MaceInit::Uniform => vec![vec![1.0 / k as f64; k]; table.n_items()],
        MaceInit::Posteriors(t0) => normalised_posteriors(op, t0, table.n_items(), k)?,
    };
    let spamming = rows
        .iter()
        .zip(&posteriors)
        .map(|(row, g)| row.iter().map(|a| a.map_or(0.0, |a| 1.0 - g[a])).collect())
        .collect();
    let mut resp = MaceResponsibilities {
        posteriors,
        spamming,
    };
    let mut iterations = 0;
    let mut converged = false;
    for it in 1..=opts.max_iter {
        let params = mace_m_step(rows, k, &resp, opts.smoothing);
        let next = mace_e_step(rows, k, &params);
        let delta = max_abs_change(&next.posteriors, &resp.posteriors);
        resp = next;
        iterations = it;
        if delta < opts.tol {
            converged = true;
            break;
        }
    }
    let params = mace_m_step(rows, k, &resp, opts.smoothing);
    let log_likelihood = mace_log_likelihood(rows, k, &params);
    Ok(Mace {
        posteriors: resp.posteriors,
        competence: params.competence,
        spam_distribution: params.spam,
        iterations,
        converged,
        log_likelihood,
    })
}

// ── Confusion against gold, posterior entropy ────────────────────────

/// Every rater's exact confusion matrix against gold labels: entry
/// `[gold][given]` is the fraction of the rater's answered gold items of
/// class `gold` that were labelled `given` (rows sum to `1`; a gold class
/// the rater never answered gets a uniform row, the convention of
/// [`dawid_skene`]).  `gold[i]` is the gold label of item `i` or `None`
/// for a non-gold item.  Convert with `to_f64` to initialise or check
/// the confusion matrices of [`dawid_skene`] / [`dawid_skene_map`].
///
/// ```
/// use symplex::stats::aggregation::{rater_confusion_from_gold, LabelTable};
/// use symplex::linprog::{q, qi};
///
/// let t = LabelTable::from_rows(
///     &[&[Some(0), Some(1)], &[Some(0), Some(0)], &[Some(1), None], &[Some(1), Some(1)]],
///     2,
/// )?;
/// let c = rater_confusion_from_gold(&t, &[Some(0), Some(0), Some(1), None])?;
/// assert_eq!(c[0], vec![vec![qi(1), qi(0)], vec![qi(0), qi(1)]]);
/// assert_eq!(c[1], vec![vec![q(1, 2), q(1, 2)], vec![q(1, 2), q(1, 2)]]);
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] if `gold` does not have one entry
/// per item or a gold label is out of range.
pub fn rater_confusion_from_gold(
    table: &LabelTable,
    gold: &[Option<usize>],
) -> Result<Vec<Vec<Vec<Q>>>, SymplexError> {
    let op = "rater_confusion_from_gold";
    if gold.len() != table.n_items() {
        return Err(invalid(op, "one gold entry per item is needed"));
    }
    if gold.iter().flatten().any(|&g| g >= table.n_categories) {
        return Err(invalid(op, "a gold label is not a category"));
    }
    let k = table.n_categories;
    Ok((0..table.n_raters)
        .map(|r| {
            let mut counts = vec![vec![0usize; k]; k];
            for (row, g) in table.rows.iter().zip(gold) {
                if let (Some(a), Some(g)) = (row[r], g) {
                    counts[*g][a] += 1;
                }
            }
            counts
                .into_iter()
                .map(|row| {
                    let total: usize = row.iter().sum();
                    if total == 0 {
                        vec![qu(1) / qu(k); k]
                    } else {
                        row.into_iter().map(|c| qu(c) / qu(total)).collect()
                    }
                })
                .collect()
        })
        .collect())
}

/// The Shannon entropy in bits, `−Σ_t p_t log₂ p_t`, of every row of a
/// posterior table — `0` for a certain item, `log₂ K` for a uniform one —
/// to rank items by how much the raters left undecided.  Each row is
/// normalised by its positive mass first (entries `≤ 0` contribute
/// nothing); a row without positive mass has entropy `0`.
///
/// ```
/// use symplex::stats::aggregation::posterior_entropy;
///
/// let h = posterior_entropy(&[vec![1.0, 0.0], vec![0.5, 0.5], vec![0.5, 0.25, 0.25]]);
/// assert_eq!(h, vec![0.0, 1.0, 1.5]);
/// ```
pub fn posterior_entropy(posteriors: &[Vec<f64>]) -> Vec<f64> {
    posteriors
        .iter()
        .map(|row| {
            let mass: f64 = row.iter().filter(|v| **v > 0.0).sum();
            if !(mass > 0.0 && mass.is_finite()) {
                return 0.0;
            }
            row.iter().filter(|v| **v > 0.0).fold(0.0, |h, v| {
                let p = v / mass;
                h - p * p.log2()
            })
        })
        .collect()
}
