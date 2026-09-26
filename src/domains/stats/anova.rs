//! Analysis of variance — one-way, factorial (two-way) and
//! repeated-measures — and the post-hoc pairwise comparisons that follow a
//! significant `F`.
//!
//! **Rule:** every ANOVA and every post-hoc procedure lives here; the
//! one-way `F` test moved in from `hypothesis` in 0.18.
//!
//! As in [`super::hypothesis`], everything that is a rational function of the
//! observations is **exact** ([`Q`]): sums of squares, mean squares, `F`,
//! `η²` and partial `η²`, the Greenhouse–Geisser and Huynh–Feldt `ε`,
//! Mauchly's `W`.  p-values are exact **expressions** ([`Ex`]) — the `F`
//! tail `I_{d₂/(d₂ + d₁F)}(d₂/2, d₁/2)` through `betainc_regularized`, a χ²
//! tail through `uppergamma` — that the caller evaluates with
//! [`Ex::eval_f64`] (or the `p_value_f64` helpers).  The studentized range
//! distribution behind Tukey's HSD has no closed form and is integrated
//! numerically in `f64` — each tail directly, to a relative `1e-14` down to
//! underflow — so those p-values and limits are `f64` and say so.
//!
//! | Function | Design | Reference |
//! |---|---|---|
//! | [`anova_one_way`] | `k` independent groups | `scipy.stats.f_oneway` |
//! | [`anova_two_way`] | `A × B` factorial with interaction, balanced or unbalanced; Type I, II or III sums of squares ([`SsType`]) | `statsmodels.stats.anova.anova_lm(ols('y ~ C(A) * C(B)').fit(), typ)` |
//! | [`anova_repeated_measures`] | one within-subject factor, `n` subjects × `k` conditions | `statsmodels.stats.anova.AnovaRM`; `pingouin.rm_anova(correction=True)`, `pingouin.epsilon`, `pingouin.sphericity` |
//! | [`tukey_hsd`] | all pairwise differences of `k` independent groups | `scipy.stats.tukey_hsd` |
//! | [`pairwise_t_tests`] | pairwise Welch tests with Bonferroni / Holm adjustment | `scipy.stats.ttest_ind` + `statsmodels.stats.multitest.multipletests` |
//!
//! ```
//! use symplex::prelude::*;
//! use symplex::linprog::q;
//! use symplex::stats::anova::{anova_two_way, TwoWayData};
//!
//! let ctx = Context::new();
//! // A 2 × 3 design, three replicates per cell: cells[a][b] = observations.
//! let data = TwoWayData::from_i64(&[
//!     &[&[4, 5, 6], &[6, 7, 8], &[9, 10, 12]],
//!     &[&[5, 5, 7], &[8, 9, 11], &[13, 14, 16]],
//! ])?;
//! // statsmodels: anova_lm(ols('y ~ C(A) * C(B)', df).fit(), typ=2)
//! //   C(A)       sum_sq 24.5      df 1   F 14.225806451612754  PR(>F) 0.00266347768868363
//! //   C(B)       sum_sq 148.7777… df 2   F 43.19354838709679   PR(>F) 3.29199072704e-06
//! //   C(A):C(B)  sum_sq 8.333333… df 2   F 2.4193548387096655  PR(>F) 0.13098893805732365
//! //   Residual   sum_sq 20.66666… df 12
//! let r = anova_two_way(&ctx, &data)?;
//! assert_eq!(r.factor_a.ss, q(49, 2));
//! assert_eq!(r.factor_a.f, Some(q(441, 31)));
//! assert_eq!(r.factor_b.ss, q(1339, 9));
//! assert_eq!(r.interaction.ss, q(25, 3));
//! assert_eq!((r.residual.ss, r.residual.df), (q(62, 3), 12));
//! assert!((r.factor_a.p_value_f64()? - 0.002_663_477_688_683_533_4).abs() < 1e-12);
//! # Ok::<(), SymplexError>(())
//! ```

use std::f64::consts::{LN_2, PI, SQRT_2};

use num_traits::{One, Signed, Zero};

use super::common::{
    check_confidence, check_unit_open, chi_squared_sf, ex, f_sf, f_sf_rational, invalid, qi, qu,
};
use super::data::{self, Q};
use super::hypothesis::{self, Alternative, PValue, TestResult, p_value_accessors};
use super::numdist::norm::{pdf as norm_pdf, sf as norm_sf};
use super::regression::ols;
use crate::api::context::Context;
use crate::api::expr::Ex;
use crate::base::errors::SymplexError;
use crate::base::interval::{Bounds, Interval};
use crate::base::numeric::ratio_to_f64;
use crate::domains::exact_matrix::QMatrix;
use crate::domains::optimize::{RootOpts, brent_root, grow_bracket};
use crate::output::codegen::numeric_rt::{erf, erfc};

// ═══════════════════════════════════════════════════════════════════════════
// Small helpers
// ═══════════════════════════════════════════════════════════════════════════

fn failed(op: &'static str, reason: impl Into<String>) -> SymplexError {
    SymplexError::computation_failed(op, reason)
}

fn to_f64(op: &'static str, q: &Q) -> Result<f64, SymplexError> {
    ratio_to_f64(q).ok_or_else(|| failed(op, format!("{q} does not fit in an f64")))
}

/// `Σ (xᵢ − x̄)²` of a non-empty slice (the caller guarantees non-emptiness;
/// an empty slice contributes `0`, which is the right value for a sum).
fn centred_ss(x: &[Q]) -> Q {
    data::sum_of_squares(x).unwrap_or_else(|_| Q::zero())
}

// ═══════════════════════════════════════════════════════════════════════════
// One-way ANOVA
// ═══════════════════════════════════════════════════════════════════════════

/// The between- and within-group sums of squares of a collection of
/// groups, with the total count and the number of groups.
pub(crate) struct GroupSums {
    /// `Σ nᵢ (x̄ᵢ − x̄)²`.
    pub(crate) ss_between: Q,
    /// `Σᵢ Σⱼ (xᵢⱼ − x̄ᵢ)²`.
    pub(crate) ss_within: Q,
    /// `N`, the total number of observations.
    pub(crate) n: usize,
    /// `k`, the number of groups.
    pub(crate) k: usize,
}

/// The sums of squares of `k ≥ 2` non-empty groups (shared with
/// [`eta_squared`](super::hypothesis::eta_squared)).
pub(crate) fn sums_of_squares(
    op: &'static str,
    groups: &[Vec<Q>],
) -> Result<GroupSums, SymplexError> {
    let k = groups.len();
    if k < 2 {
        return Err(invalid(op, "at least two groups are needed"));
    }
    if groups.iter().any(Vec::is_empty) {
        return Err(invalid(op, "every group must be non-empty"));
    }
    let all: Vec<Q> = groups.iter().flatten().cloned().collect();
    let grand = data::mean(&all)?;
    let mut ss_between = Q::zero();
    let mut ss_within = Q::zero();
    for g in groups {
        let m = data::mean(g)?;
        let d = &m - &grand;
        ss_between += qu(g.len()) * &d * &d;
        ss_within += data::sum_of_squares(g)?;
    }
    Ok(GroupSums {
        ss_between,
        ss_within,
        n: all.len(),
        k,
    })
}

/// The outcome of a one-way analysis of variance.
#[derive(Clone, Debug, PartialEq)]
pub struct AnovaResult {
    /// The `F` statistic `(SS_between/df_between) / (SS_within/df_within)`,
    /// exact.
    pub f: Q,
    /// `k − 1`.
    pub df_between: usize,
    /// `N − k`.
    pub df_within: usize,
    /// `P(F_{df_between, df_within} ≥ f)` as an exact expression.
    pub p_value: Ex,
    /// `Σ nᵢ (x̄ᵢ − x̄)²`.
    pub ss_between: Q,
    /// `Σᵢ Σⱼ (xᵢⱼ − x̄ᵢ)²`.
    pub ss_within: Q,
    /// `SS_between / SS_total`, the proportion of variance explained.
    pub eta_squared: Q,
}

impl AnovaResult {
    /// The p-value as an `f64`.
    ///
    /// # Errors
    ///
    /// Propagates the evaluation error of the expression.
    pub fn p_value_f64(&self) -> Result<f64, SymplexError> {
        self.p_value.eval_f64()
    }
}

impl PValue for AnovaResult {
    fn p_value_ex(&self) -> &Ex {
        &self.p_value
    }
}
p_value_accessors!(AnovaResult);

/// One-way analysis of variance of `k` independent groups:
/// `F = (SS_between/(k−1)) / (SS_within/(N−k))` with the sums of squares and
/// `F` exact, `η² = SS_between/SS_total`, and `P(F_{k−1, N−k} ≥ F)` as an
/// exact expression.  `scipy.stats.f_oneway(*groups)`.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::linprog::q;
/// use symplex::stats::anova::anova_one_way;
/// use symplex::stats::data::from_i64;
///
/// let ctx = Context::new();
/// let g = [from_i64(&[6, 8, 4, 5, 3, 4]), from_i64(&[8, 12, 9, 11, 6, 8]), from_i64(&[13, 9, 11, 8, 7, 12])];
/// // scipy: f_oneway(*g) → statistic 9.264705882352942 (= 315/34), pvalue 0.0023987773293929083
/// let r = anova_one_way(&ctx, &g)?;
/// assert_eq!(r.f, q(315, 34));
/// assert_eq!((r.df_between, r.df_within), (2, 15));
/// assert!((r.p_value_f64()? - 0.002_398_777_329_392_908_3).abs() < 1e-12);
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for fewer than two groups, an empty
/// group, `N ≤ k`, or zero within-group variance.
pub fn anova_one_way(ctx: &Context, groups: &[Vec<Q>]) -> Result<AnovaResult, SymplexError> {
    const OP: &str = "anova_one_way";
    let GroupSums {
        ss_between,
        ss_within,
        n,
        k,
    } = sums_of_squares(OP, groups)?;
    if n <= k {
        return Err(invalid(
            OP,
            "at least one group needs more than one observation",
        ));
    }
    if ss_within.is_zero() {
        return Err(invalid(OP, "the within-group variance is zero"));
    }
    let (df_between, df_within) = (k - 1, n - k);
    let f = (&ss_between / qu(df_between)) / (&ss_within / qu(df_within));
    let total = &ss_between + &ss_within;
    let eta_squared = &ss_between / &total;
    Ok(AnovaResult {
        p_value: f_sf(ctx, df_between, df_within, &f),
        f,
        df_between,
        df_within,
        ss_between,
        ss_within,
        eta_squared,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Rows of an ANOVA table
// ═══════════════════════════════════════════════════════════════════════════

/// Which line of an ANOVA table a row is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Source {
    /// The main effect of factor `A` (two-way).
    FactorA,
    /// The main effect of factor `B` (two-way).
    FactorB,
    /// The `A × B` interaction (two-way).
    Interaction,
    /// The within-subject factor (repeated measures).
    Conditions,
    /// Between-subject variation (repeated measures; not tested).
    Subjects,
    /// The error term the effects are tested against.
    Residual,
    /// The corrected total, `Σ (y − ȳ)²` with `N − 1` degrees of freedom.
    Total,
}

/// One row of an ANOVA table.  Effects carry an `F` test and effect sizes;
/// the residual row carries only its mean square; the total row only `ss`
/// and `df`.
#[derive(Clone, Debug, PartialEq)]
pub struct AnovaRow {
    /// Which line this is.
    pub source: Source,
    /// The sum of squares, exact.
    pub ss: Q,
    /// Degrees of freedom.
    pub df: usize,
    /// `ss / df` (`None` for the total row).
    pub ms: Option<Q>,
    /// `ms / ms_residual`, exact (effects only).
    pub f: Option<Q>,
    /// `P(F_{df, df_residual} ≥ f)` as an exact expression (effects only).
    pub p_value: Option<Ex>,
    /// `ss / ss_total` (effects only).
    pub eta_squared: Option<Q>,
    /// `ss / (ss + ss_residual)` (effects only).
    pub partial_eta_squared: Option<Q>,
}

impl AnovaRow {
    /// An effect row tested against the residual `(ss_resid, df_resid)`.
    fn effect(
        ctx: &Context,
        source: Source,
        ss: Q,
        df: usize,
        ss_resid: &Q,
        df_resid: usize,
        ss_total: &Q,
    ) -> Self {
        let test = FTest::of(ctx, &ss, df, ss_resid, df_resid);
        Self::with_test(source, ss, df, test, ss_resid, ss_total)
    }

    /// An effect row from an already computed `F` test.
    fn with_test(
        source: Source,
        ss: Q,
        df: usize,
        test: FTest,
        ss_resid: &Q,
        ss_total: &Q,
    ) -> Self {
        Self {
            source,
            eta_squared: Some(&ss / ss_total),
            partial_eta_squared: Some(&ss / (&ss + ss_resid)),
            ms: Some(&ss / qu(df)),
            ss,
            df,
            f: Some(test.f),
            p_value: Some(test.p_value),
        }
    }

    fn untested(source: Source, ss: Q, df: usize) -> Self {
        Self {
            source,
            ms: Some(&ss / qu(df)),
            ss,
            df,
            f: None,
            p_value: None,
            eta_squared: None,
            partial_eta_squared: None,
        }
    }

    fn total(ss: Q, df: usize) -> Self {
        Self {
            source: Source::Total,
            ss,
            df,
            ms: None,
            f: None,
            p_value: None,
            eta_squared: None,
            partial_eta_squared: None,
        }
    }

    /// The p-value as an `f64`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if this row has no `F` test (the
    /// residual, subjects and total rows); otherwise the evaluation error of
    /// the expression (not expected).
    pub fn p_value_f64(&self) -> Result<f64, SymplexError> {
        self.tested_p_value("AnovaRow::p_value_f64")?.eval_f64()
    }

    /// `log10` of the p-value, finite even when
    /// [`p_value_f64`](Self::p_value_f64) underflows to `0.0`; see
    /// [`PValue`].  (`AnovaRow` cannot implement the trait itself: the
    /// residual and total rows have no p-value.)
    ///
    /// # Errors
    ///
    /// As [`p_value_f64`](Self::p_value_f64).
    pub fn p_value_log10(&self) -> Result<f64, SymplexError> {
        hypothesis::p_value_log10_of(self.tested_p_value("AnovaRow::p_value_log10")?)
    }

    /// `ln` of the p-value, evaluated as an expression; see [`PValue`].
    ///
    /// # Errors
    ///
    /// As [`p_value_f64`](Self::p_value_f64).
    pub fn p_value_ln(&self) -> Result<f64, SymplexError> {
        hypothesis::p_value_ln_of(self.tested_p_value("AnovaRow::p_value_ln")?)
    }

    /// The p-value to `digits` significant digits as a decimal string with
    /// exponent; see [`PValue`].
    ///
    /// # Errors
    ///
    /// As [`p_value_f64`](Self::p_value_f64).
    pub fn p_value_decimal(&self, digits: u32) -> Result<String, SymplexError> {
        self.tested_p_value("AnovaRow::p_value_decimal")?
            .eval_decimal(digits)
    }

    /// The p-value expression of an effect row, or the `InvalidArgument`
    /// error every accessor reports for an untested row.
    fn tested_p_value(&self, op: &'static str) -> Result<&Ex, SymplexError> {
        self.p_value
            .as_ref()
            .ok_or_else(|| invalid(op, format!("the {:?} row has no F test", self.source)))
    }
}

/// An `F` statistic with its tail probability (a private carrier so the two
/// travel together).
struct FTest {
    f: Q,
    p_value: Ex,
}

impl FTest {
    /// `F = (ss/df) / (ss_resid/df_resid)` and `P(F_{df, df_resid} ≥ F)`.
    fn of(ctx: &Context, ss: &Q, df: usize, ss_resid: &Q, df_resid: usize) -> Self {
        let f = (ss / qu(df)) / (ss_resid / qu(df_resid));
        let p_value = f_sf(ctx, df, df_resid, &f);
        Self { f, p_value }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Two-way ANOVA
// ═══════════════════════════════════════════════════════════════════════════

/// One observation of a two-way layout in long form: the (0-based) levels of
/// the two factors and the response.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Observation {
    /// Level of factor `A`, `0 ≤ a < a_levels`.
    pub a: usize,
    /// Level of factor `B`, `0 ≤ b < b_levels`.
    pub b: usize,
    /// The response.
    pub y: Q,
}

/// The observations of an `A × B` factorial design, stored by cell:
/// `cells[a][b]` is the (non-empty) vector of replicates at level `a` of
/// `A` and level `b` of `B`.  Cell sizes may differ (an unbalanced design);
/// every cell must contain at least one observation.
///
/// Build it from nested vectors ([`from_cells`](Self::from_cells),
/// [`from_i64`](Self::from_i64)) or from long-form rows
/// ([`from_long`](Self::from_long)).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TwoWayData {
    a_levels: usize,
    b_levels: usize,
    cells: Vec<Vec<Vec<Q>>>,
}

impl TwoWayData {
    /// From `cells[a][b] = replicates`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if either factor has fewer than two
    /// levels, the rows have different lengths, or a cell is empty.
    pub fn from_cells(cells: Vec<Vec<Vec<Q>>>) -> Result<Self, SymplexError> {
        const OP: &str = "TwoWayData::from_cells";
        let a_levels = cells.len();
        if a_levels < 2 {
            return Err(invalid(OP, "factor A needs at least two levels"));
        }
        let b_levels = cells.first().map_or(0, Vec::len);
        if b_levels < 2 {
            return Err(invalid(OP, "factor B needs at least two levels"));
        }
        for (a, row) in cells.iter().enumerate() {
            if row.len() != b_levels {
                return Err(invalid(
                    OP,
                    format!(
                        "level {a} of A has {} cells, expected {b_levels}",
                        row.len()
                    ),
                ));
            }
            if let Some((b, _)) = row.iter().enumerate().find(|(_, c)| c.is_empty()) {
                return Err(invalid(
                    OP,
                    format!("cell (A = {a}, B = {b}) has no observations"),
                ));
            }
        }
        Ok(Self {
            a_levels,
            b_levels,
            cells,
        })
    }

    /// From integer observations, `cells[a][b] = replicates`.
    ///
    /// # Errors
    ///
    /// As [`from_cells`](Self::from_cells).
    pub fn from_i64(cells: &[&[&[i64]]]) -> Result<Self, SymplexError> {
        Self::from_cells(
            cells
                .iter()
                .map(|row| row.iter().map(|c| data::from_i64(c)).collect())
                .collect(),
        )
    }

    /// From long-form rows (one [`Observation`] per response).  The numbers
    /// of levels are `max(a) + 1` and `max(b) + 1`; every combination of
    /// levels must occur.
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::linprog::qi;
    /// use symplex::stats::anova::{Observation, TwoWayData};
    ///
    /// let rows: Vec<Observation> = [(0, 0, 4), (0, 0, 5), (0, 1, 6), (1, 0, 5), (1, 1, 8), (1, 1, 9)]
    ///     .iter()
    ///     .map(|&(a, b, y)| Observation { a, b, y: qi(y) })
    ///     .collect();
    /// let data = TwoWayData::from_long(&rows)?;
    /// assert_eq!((data.a_levels(), data.b_levels(), data.n_obs()), (2, 2, 6));
    /// assert_eq!(data.cell(1, 1), Some(&[qi(8), qi(9)][..]));
    /// assert!(!data.is_balanced());
    /// # Ok::<(), SymplexError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] for no rows, fewer than two levels
    /// of either factor, or a combination of levels with no observation.
    pub fn from_long(rows: &[Observation]) -> Result<Self, SymplexError> {
        const OP: &str = "TwoWayData::from_long";
        if rows.is_empty() {
            return Err(invalid(OP, "no observations"));
        }
        // Every combination of levels needs an observation, so there are at
        // most `rows.len()` of them: checked before `max + 1` (which
        // overflowed at `usize::MAX`) and before the table is allocated.
        let a_levels = rows.iter().map(|o| o.a).max().unwrap_or(0).checked_add(1);
        let b_levels = rows.iter().map(|o| o.b).max().unwrap_or(0).checked_add(1);
        let (a_levels, b_levels) = match (a_levels, b_levels) {
            (Some(a), Some(b)) if a.checked_mul(b).is_some_and(|ab| ab <= rows.len()) => (a, b),
            _ => {
                return Err(invalid(
                    OP,
                    "more combinations of levels than observations: some combination has none",
                ));
            }
        };
        let mut cells = vec![vec![Vec::new(); b_levels]; a_levels];
        for o in rows {
            // `o.a < a_levels` and `o.b < b_levels` by construction of the maxima.
            if let Some(cell) = cells.get_mut(o.a).and_then(|row| row.get_mut(o.b)) {
                cell.push(o.y.clone());
            }
        }
        Self::from_cells(cells).map_err(|e| match e {
            SymplexError::InvalidArgument { reason, .. } => invalid(OP, reason),
            other => other,
        })
    }

    /// Number of levels of factor `A`.
    #[must_use]
    pub fn a_levels(&self) -> usize {
        self.a_levels
    }

    /// Number of levels of factor `B`.
    #[must_use]
    pub fn b_levels(&self) -> usize {
        self.b_levels
    }

    /// Total number of observations `N`.
    #[must_use]
    pub fn n_obs(&self) -> usize {
        self.cells.iter().flatten().map(Vec::len).sum()
    }

    /// The replicates of cell `(a, b)`, or `None` when either index is out
    /// of range.
    #[must_use]
    pub fn cell(&self, a: usize, b: usize) -> Option<&[Q]> {
        self.cells.get(a)?.get(b).map(Vec::as_slice)
    }

    /// All cells, `cells[a][b] = replicates`.
    #[must_use]
    pub fn cells(&self) -> &[Vec<Vec<Q>>] {
        &self.cells
    }

    /// Whether every cell has the same number of replicates.
    #[must_use]
    pub fn is_balanced(&self) -> bool {
        let mut sizes = self.cells.iter().flatten().map(Vec::len);
        match sizes.next() {
            Some(first) => sizes.all(|s| s == first),
            None => true,
        }
    }

    /// The observations in `cells[a][b][r]` order.
    fn flatten(&self) -> Vec<Q> {
        self.cells.iter().flatten().flatten().cloned().collect()
    }

    /// The observations at level `a` of `A`, all `B` levels pooled.
    fn a_slice(&self, a: usize) -> Vec<Q> {
        self.cells
            .get(a)
            .map(|row| row.iter().flatten().cloned().collect())
            .unwrap_or_default()
    }

    /// The observations at level `b` of `B`, all `A` levels pooled.
    fn b_slice(&self, b: usize) -> Vec<Q> {
        self.cells
            .iter()
            .filter_map(|row| row.get(b))
            .flatten()
            .cloned()
            .collect()
    }
}

/// How the sums of squares of a factorial design are attributed to its
/// terms.  For a **balanced** design the three types coincide; they differ
/// only when cell sizes are unequal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SsType {
    /// Sequential: `SS(A)`, `SS(B | A)`, `SS(AB | A, B)` — each term adjusted
    /// for the ones before it, in the order `A`, `B`, `A×B`.  R's `anova()`,
    /// `anova_lm(typ=1)`.
    TypeI,
    /// Hierarchical: `SS(A | B)`, `SS(B | A)`, `SS(AB | A, B)` — each main
    /// effect adjusted for the other, the interaction for both.
    /// `car::Anova(type=2)`, `anova_lm(typ=2)`.
    TypeII,
    /// Marginal: every term adjusted for all others, including the
    /// interaction, under **sum-to-zero (effects) contrasts** — the
    /// `contr.sum` / `C(A, Sum)` coding of SPSS and `car::Anova(type=3)`.
    /// `anova_lm(ols('y ~ C(A, Sum) * C(B, Sum)').fit(), typ=3)`.  (With the
    /// default treatment contrasts Type III main effects are not the
    /// classical ones; this crate always uses sum-to-zero coding here.)
    TypeIII,
}

/// The table of a two-way analysis of variance; see [`anova_two_way`].
#[derive(Clone, Debug, PartialEq)]
pub struct TwoWayAnova {
    /// The attribution used for the main effects.
    pub ss_type: SsType,
    /// Main effect of `A`, `df = a − 1`.
    pub factor_a: AnovaRow,
    /// Main effect of `B`, `df = b − 1`.
    pub factor_b: AnovaRow,
    /// The `A × B` interaction, `df = (a − 1)(b − 1)`.
    pub interaction: AnovaRow,
    /// Within-cell variation, `df = N − ab`.
    pub residual: AnovaRow,
    /// `Σ (y − ȳ)²`, `df = N − 1`.
    pub total: AnovaRow,
    /// `ȳ`, exact.
    pub grand_mean: Q,
    /// `cell_means[a][b]`, exact.
    pub cell_means: Vec<Vec<Q>>,
}

impl TwoWayAnova {
    /// The rows in table order: `A`, `B`, `A × B`, residual, total.
    #[must_use]
    pub fn rows(&self) -> Vec<&AnovaRow> {
        vec![
            &self.factor_a,
            &self.factor_b,
            &self.interaction,
            &self.residual,
            &self.total,
        ]
    }
}

/// Which terms a reduced model contains (the intercept is always present).
#[derive(Clone, Copy)]
struct Terms {
    a: bool,
    b: bool,
    ab: bool,
}

/// The contrast coding of a factor's dummy columns.  Residual sums of
/// squares do not depend on it; Type III sums of squares do.
#[derive(Clone, Copy)]
enum Coding {
    /// Indicators of levels `1..n` (level `0` is the reference).
    Treatment,
    /// Sum-to-zero: indicators of levels `1..n`, with level `0` coded `−1`
    /// in every column.  Which level is omitted does not change the span.
    Sum,
}

/// The `n_levels − 1` codes of `level`.
fn codes(level: usize, n_levels: usize, coding: Coding) -> Vec<Q> {
    (1..n_levels)
        .map(|i| match coding {
            Coding::Treatment => {
                if level == i {
                    Q::one()
                } else {
                    Q::zero()
                }
            }
            Coding::Sum => {
                if level == i {
                    Q::one()
                } else if level == 0 {
                    -Q::one()
                } else {
                    Q::zero()
                }
            }
        })
        .collect()
}

/// The residual sum of squares of the model with the given terms, by exact
/// least squares on the dummy-coded design.
fn reduced_rss(
    op: &'static str,
    data: &TwoWayData,
    y: &[Q],
    terms: Terms,
    coding: Coding,
) -> Result<Q, SymplexError> {
    let mut rows: Vec<Vec<Q>> = Vec::with_capacity(y.len());
    for (a, row) in data.cells.iter().enumerate() {
        let ca = codes(a, data.a_levels, coding);
        for (b, cell) in row.iter().enumerate() {
            let cb = codes(b, data.b_levels, coding);
            let mut r = Vec::new();
            if terms.a {
                r.extend(ca.iter().cloned());
            }
            if terms.b {
                r.extend(cb.iter().cloned());
            }
            if terms.ab {
                for x in &ca {
                    for z in &cb {
                        r.push(x * z);
                    }
                }
            }
            rows.extend(std::iter::repeat_n(r, cell.len()));
        }
    }
    ols(y, &rows, true).map(|fit| fit.ssr).map_err(|e| {
        failed(
            op,
            format!("least-squares fit of a reduced model failed: {e}"),
        )
    })
}

/// Two-way analysis of variance with **Type II** sums of squares — the
/// attribution of `statsmodels`' `anova_lm(model, typ=2)` and
/// `car::Anova(type=2)`.  Equivalent to [`anova_two_way_with`] with
/// [`SsType::TypeII`]; for a balanced design every type gives the same
/// table.
///
/// See the [module documentation](self) for an example.
///
/// # Errors
///
/// As [`anova_two_way_with`].
pub fn anova_two_way(ctx: &Context, data: &TwoWayData) -> Result<TwoWayAnova, SymplexError> {
    anova_two_way_with(ctx, data, SsType::TypeII)
}

/// Two-way analysis of variance of the `A × B` design in `data`, with the
/// interaction, using the sums of squares of `ss_type`.
///
/// The model is `y = μ + αₐ + βᵦ + (αβ)ₐᵦ + e`; every sum of squares is the
/// exact difference of the residual sums of squares of two nested
/// least-squares fits on the dummy-coded design, the residual is the pooled
/// within-cell `Σ (y − ȳₐᵦ)²` with `N − ab` degrees of freedom, and each
/// effect is tested by `F = MS_effect / MS_residual` with
/// `P(F_{df, N−ab} ≥ F)` as an exact expression.  Effect sizes are
/// `η² = SS/SS_total` and partial `η² = SS/(SS + SS_residual)`.
///
/// **Balanced designs**: with equal cell sizes the factors are orthogonal
/// and the Type I, II and III sums of squares coincide.  **Unbalanced
/// designs**: they differ for the main effects (never for the interaction);
/// see [`SsType`].
///
/// ```
/// use symplex::prelude::*;
/// use symplex::linprog::{q, qi};
/// use symplex::stats::anova::{anova_two_way_with, SsType, TwoWayData};
///
/// let ctx = Context::new();
/// // Unbalanced 2 × 3 design (cell sizes 4, 2, 3 / 2, 4, 3).
/// let data = TwoWayData::from_i64(&[
///     &[&[4, 5, 6, 7], &[6, 8], &[9, 10, 12]],
///     &[&[5, 7], &[8, 9, 11, 10], &[13, 14, 16]],
/// ])?;
/// // statsmodels: anova_lm(ols('y ~ C(A) * C(B)', df).fit(), typ=1): C(A) sum_sq 37.555555555555636
/// //              anova_lm(..., typ=2):                             C(A) sum_sq 24.000000000000096
/// //              anova_lm(ols('y ~ C(A, Sum) * C(B, Sum)', df).fit(), typ=3): C(A, Sum) sum_sq 22.61538461538463
/// let t1 = anova_two_way_with(&ctx, &data, SsType::TypeI)?;
/// let t2 = anova_two_way_with(&ctx, &data, SsType::TypeII)?;
/// let t3 = anova_two_way_with(&ctx, &data, SsType::TypeIII)?;
/// assert_eq!(t1.factor_a.ss, q(338, 9));
/// assert_eq!(t2.factor_a.ss, qi(24));
/// assert_eq!(t3.factor_a.ss, q(294, 13));
/// // The interaction is the same under every type: sum_sq 8.666666666666684, F 2.2285714285714326.
/// assert_eq!(t1.interaction.ss, q(26, 3));
/// assert_eq!(t3.interaction.f, Some(q(78, 35)));
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] if every cell has a single observation
/// (no residual degrees of freedom), the response is constant, or the
/// within-cell variance is zero.
pub fn anova_two_way_with(
    ctx: &Context,
    data: &TwoWayData,
    ss_type: SsType,
) -> Result<TwoWayAnova, SymplexError> {
    const OP: &str = "anova_two_way";
    let (a, b) = (data.a_levels, data.b_levels);
    let n = data.n_obs();
    if n <= a * b {
        return Err(invalid(
            OP,
            "every cell has a single observation: no residual degrees of freedom",
        ));
    }
    let df_resid = n - a * b;
    let y = data.flatten();
    let grand_mean = data::mean(&y)?;
    let ss_total = centred_ss(&y);
    if ss_total.is_zero() {
        return Err(invalid(OP, "the response is constant"));
    }
    let cell_means = data
        .cells
        .iter()
        .map(|row| {
            row.iter()
                .map(|c| data::mean(c))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    // Residual sums of squares of the nested models that do not need a
    // least-squares solve: `∅` (the total), `A`, `B` and the full model.
    let rss_full = data
        .cells
        .iter()
        .flatten()
        .fold(Q::zero(), |acc, c| acc + centred_ss(c));
    if rss_full.is_zero() {
        return Err(invalid(
            OP,
            "the within-cell variance is zero: F is undefined",
        ));
    }
    let rss_a = (0..a).fold(Q::zero(), |acc, i| acc + centred_ss(&data.a_slice(i)));
    let rss_b = (0..b).fold(Q::zero(), |acc, j| acc + centred_ss(&data.b_slice(j)));
    let additive = Terms {
        a: true,
        b: true,
        ab: false,
    };
    let rss_ab = reduced_rss(OP, data, &y, additive, Coding::Treatment)?;
    let (ss_a, ss_b) = match ss_type {
        SsType::TypeI => (&ss_total - &rss_a, &rss_a - &rss_ab),
        SsType::TypeII => (&rss_b - &rss_ab, &rss_a - &rss_ab),
        SsType::TypeIII => {
            let without_a = Terms {
                a: false,
                b: true,
                ab: true,
            };
            let without_b = Terms {
                a: true,
                b: false,
                ab: true,
            };
            let rss_no_a = reduced_rss(OP, data, &y, without_a, Coding::Sum)?;
            let rss_no_b = reduced_rss(OP, data, &y, without_b, Coding::Sum)?;
            (rss_no_a - &rss_full, rss_no_b - &rss_full)
        }
    };
    let ss_ab = &rss_ab - &rss_full;
    Ok(TwoWayAnova {
        ss_type,
        factor_a: AnovaRow::effect(
            ctx,
            Source::FactorA,
            ss_a,
            a - 1,
            &rss_full,
            df_resid,
            &ss_total,
        ),
        factor_b: AnovaRow::effect(
            ctx,
            Source::FactorB,
            ss_b,
            b - 1,
            &rss_full,
            df_resid,
            &ss_total,
        ),
        interaction: AnovaRow::effect(
            ctx,
            Source::Interaction,
            ss_ab,
            (a - 1) * (b - 1),
            &rss_full,
            df_resid,
            &ss_total,
        ),
        residual: AnovaRow::untested(Source::Residual, rss_full, df_resid),
        total: AnovaRow::total(ss_total, n - 1),
        grand_mean,
        cell_means,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Repeated measures
// ═══════════════════════════════════════════════════════════════════════════

/// Mauchly's test of sphericity; see [`RepeatedMeasuresAnova::mauchly`].
#[derive(Clone, Debug, PartialEq)]
pub struct Mauchly {
    /// `W = Π λᵢ / (Σ λᵢ / (k−1))^{k−1}` over the `k − 1` non-zero eigenvalues
    /// of the double-centred covariance matrix, exact (`1` under sphericity).
    pub w: Q,
    /// `χ² = −(n − 1) ρ ln W` with Box's `ρ = 1 − (2d² + d + 2)/(6d(n−1))`,
    /// `d = k − 1`, as an exact expression.
    pub chi_squared: Ex,
    /// `k(k − 1)/2 − 1`.
    pub df: usize,
    /// The p-value with Box's (1949) second-order correction
    /// `min(1, P₁ + ω₂ (P₂ − P₁))`, `Pᵢ` the χ² tails at `df` and `df + 4`,
    /// `ω₂ = (d+2)(d−1)(d−2)(2d³ + 6d² + 3d + 2) / (288 (n−1)² d² ρ²)`, as
    /// an exact expression (for `k = 3` the correction term vanishes).
    /// This `ω₂` is the second coefficient of Box's expansion of the
    /// moments `E[Wʰ]` (the `B₃` Bernoulli-polynomial term; checked
    /// against that expansion in mpmath); `pingouin.sphericity` writes
    /// `3k` for its `3d`, so its p-value differs for `k ≥ 4`.  The
    /// truncated series can exceed `1` when `ω₂ > 1` (few subjects for
    /// many conditions); it is capped there.
    pub p_value: Ex,
}

impl Mauchly {
    /// The statistic as an `f64`.
    ///
    /// # Errors
    ///
    /// Propagates the evaluation error of the expression.
    pub fn chi_squared_f64(&self) -> Result<f64, SymplexError> {
        self.chi_squared.eval_f64()
    }

    /// The p-value as an `f64`.
    ///
    /// # Errors
    ///
    /// Propagates the evaluation error of the expression.
    pub fn p_value_f64(&self) -> Result<f64, SymplexError> {
        self.p_value.eval_f64()
    }
}

impl PValue for Mauchly {
    fn p_value_ex(&self) -> &Ex {
        &self.p_value
    }
}
p_value_accessors!(Mauchly);

/// The outcome of a one-way repeated-measures analysis of variance; see
/// [`anova_repeated_measures`].
#[derive(Clone, Debug, PartialEq)]
pub struct RepeatedMeasuresAnova {
    /// `n`, the number of subjects.
    pub n_subjects: usize,
    /// `k`, the number of conditions.
    pub n_conditions: usize,
    /// The within-subject factor, `df = k − 1`, tested against the error.
    pub conditions: AnovaRow,
    /// Between-subject variation, `df = n − 1` (removed, not tested).
    pub subjects: AnovaRow,
    /// The subject × condition residual, `df = (k − 1)(n − 1)`.
    pub error: AnovaRow,
    /// `Σ (y − ȳ)²`, `df = nk − 1`.
    pub total: AnovaRow,
    /// `F = MS_conditions / MS_error`, exact (also `conditions.f`).
    pub f: Q,
    /// `P(F_{k−1, (k−1)(n−1)} ≥ F)` under sphericity, exact expression.
    pub p_value: Ex,
    /// Greenhouse–Geisser `ε̂ = (tr S̃)² / ((k − 1) tr S̃²)` of the
    /// double-centred sample covariance `S̃`, exact; `1/(k−1) ≤ ε̂ ≤ 1`.
    pub epsilon_gg: Q,
    /// Huynh–Feldt `ε̃ = (n(k−1)ε̂ − 2) / ((k−1)(n − 1 − (k−1)ε̂))`, exact and
    /// **uncapped** (it can exceed `1`); `None` when its denominator is not
    /// positive (too few subjects for the correction to be defined).
    pub epsilon_hf: Option<Q>,
    /// The Greenhouse–Geisser corrected p-value: the `F` tail with both
    /// degrees of freedom multiplied by `ε̂`.
    pub p_value_gg: Ex,
    /// The Huynh–Feldt corrected p-value, with `min(ε̃, 1)`; `None` when
    /// `epsilon_hf` is.
    pub p_value_hf: Option<Ex>,
    /// Mauchly's sphericity test; `None` for `k = 2` (sphericity holds
    /// trivially) or a singular covariance matrix (`n − 1 < k − 1`, or a
    /// degenerate sample), where `W` is `0` and the test is undefined.
    pub mauchly: Option<Mauchly>,
    /// `ȳ`, exact.
    pub grand_mean: Q,
    /// The mean of each condition over subjects, exact.
    pub condition_means: Vec<Q>,
    /// The mean of each subject over conditions, exact.
    pub subject_means: Vec<Q>,
}

impl RepeatedMeasuresAnova {
    /// The uncorrected p-value as an `f64`.
    ///
    /// # Errors
    ///
    /// Propagates the evaluation error of the expression.
    pub fn p_value_f64(&self) -> Result<f64, SymplexError> {
        self.p_value.eval_f64()
    }

    /// The Greenhouse–Geisser corrected p-value as an `f64`.
    ///
    /// # Errors
    ///
    /// Propagates the evaluation error of the expression.
    pub fn p_value_gg_f64(&self) -> Result<f64, SymplexError> {
        self.p_value_gg.eval_f64()
    }

    /// The Huynh–Feldt corrected p-value as an `f64`, if defined.
    ///
    /// # Errors
    ///
    /// Propagates the evaluation error of the expression.
    pub fn p_value_hf_f64(&self) -> Result<Option<f64>, SymplexError> {
        self.p_value_hf.as_ref().map(Ex::eval_f64).transpose()
    }

    /// `log10` of the Greenhouse–Geisser corrected p-value, finite even
    /// when [`p_value_gg_f64`](Self::p_value_gg_f64) underflows to `0.0`;
    /// see [`PValue`].
    ///
    /// # Errors
    ///
    /// Propagates the evaluation error of the expression.
    pub fn p_value_gg_log10(&self) -> Result<f64, SymplexError> {
        hypothesis::p_value_log10_of(&self.p_value_gg)
    }

    /// `log10` of the Huynh–Feldt corrected p-value, if defined; see
    /// [`p_value_gg_log10`](Self::p_value_gg_log10).
    ///
    /// # Errors
    ///
    /// Propagates the evaluation error of the expression.
    pub fn p_value_hf_log10(&self) -> Result<Option<f64>, SymplexError> {
        self.p_value_hf
            .as_ref()
            .map(hypothesis::p_value_log10_of)
            .transpose()
    }

    /// The rows in table order: conditions, subjects, error, total.
    #[must_use]
    pub fn rows(&self) -> Vec<&AnovaRow> {
        vec![&self.conditions, &self.subjects, &self.error, &self.total]
    }
}

/// The uncorrected p-value (the `F` tail under sphericity).
impl PValue for RepeatedMeasuresAnova {
    fn p_value_ex(&self) -> &Ex {
        &self.p_value
    }
}
p_value_accessors!(RepeatedMeasuresAnova);

/// Mauchly's `W` and its χ² approximation from the `k × k` sample covariance
/// `s` and the trace of its double-centred form `S̃ = C S C`.
fn mauchly(
    ctx: &Context,
    s: &[Vec<Q>],
    trace: &Q,
    n: usize,
    k: usize,
) -> Result<Option<Mauchly>, SymplexError> {
    const OP: &str = "anova_repeated_measures";
    let d = k - 1;
    // The product of the d non-zero eigenvalues of S̃ = C S C equals
    // det(MᵀSM)/det(MᵀM) for any k × d matrix M whose columns span the
    // contrast space; successive differences do.
    let m = QMatrix::new(
        (0..k)
            .map(|row| {
                (0..d)
                    .map(|j| {
                        if row == j {
                            Q::one()
                        } else if row == j + 1 {
                            -Q::one()
                        } else {
                            Q::zero()
                        }
                    })
                    .collect()
            })
            .collect(),
    )
    .map_err(|e| failed(OP, e.to_string()))?;
    let s_mat = QMatrix::new(s.to_vec()).map_err(|e| failed(OP, e.to_string()))?;
    let mt = m.transpose();
    let mtsm = mt.matmul(&s_mat)?.matmul(&m)?;
    let mtm = mt.matmul(&m)?;
    let product = mtsm.det()? / mtm.det()?;
    if !product.is_positive() {
        return Ok(None);
    }
    let mean_eigenvalue = trace / qu(d);
    let denominator = (0..d).fold(Q::one(), |acc, _| acc * &mean_eigenvalue);
    let w = product / denominator;
    let (dq, n1) = (qu(d), qu(n - 1));
    let rho = Q::one() - (qi(2) * &dq * &dq + &dq + qi(2)) / (qi(6) * &dq * &n1);
    if !rho.is_positive() {
        return Ok(None);
    }
    let scale = &n1 * &dq * &rho;
    let omega2 = (&dq + qi(2))
        * (&dq - qi(1))
        * (&dq - qi(2))
        * (qi(2) * &dq * &dq * &dq + qi(6) * &dq * &dq + qi(3) * &dq + qi(2))
        / (qi(288) * &scale * &scale);
    let df = d * (d + 1) / 2 - 1;
    let chi_squared = ex(ctx, &(-(&n1 * &rho))) * ex(ctx, &w).ln();
    let p1 = chi_squared_sf(ctx, df, &chi_squared);
    let p2 = chi_squared_sf(ctx, df + 4, &chi_squared);
    // `p₂ > p₁` and `ω₂ ≥ 0`, so the expansion is at least `p₁ > 0`; but
    // `ω₂` grows like `d²/n²` and past `ω₂ ≈ 1` the truncated series
    // exceeds `1` near the mode of `χ²` (`d = 10`, `n = 11`: `1.0063`).
    let p_value = (&p1 + ex(ctx, &omega2) * (p2 - &p1)).min_with(&ctx.one());
    Ok(Some(Mauchly {
        w,
        chi_squared,
        df,
        p_value,
    }))
}

/// One-way repeated-measures analysis of variance: `subjects_by_condition[i]`
/// is the row of subject `i` (`n` rows) over the `k` conditions (columns).
///
/// The additive model `y_ij = μ + πᵢ + τⱼ + e_ij` gives
/// `SS_total = SS_subjects + SS_conditions + SS_error` with
/// `SS_conditions = n Σⱼ (ȳ_·ⱼ − ȳ)²`, `SS_subjects = k Σᵢ (ȳ_ᵢ· − ȳ)²`, and
/// `F = (SS_conditions/(k−1)) / (SS_error/((k−1)(n−1)))`, all exact
/// (`statsmodels.stats.anova.AnovaRM`).  Partial `η² = SS_conditions /
/// (SS_conditions + SS_error)` (pingouin's `ng2` is `SS_conditions /
/// SS_total` here, the row's `eta_squared`).
///
/// Sphericity: the Greenhouse–Geisser `ε̂` is computed exactly from the
/// double-centred sample covariance without eigenvalues, `(tr S̃)² / ((k−1)
/// tr S̃²)`, the Huynh–Feldt `ε̃` from it, and the corrected p-values are the
/// `F` tails with degrees of freedom `(k−1)ε` and `(k−1)(n−1)ε`
/// (`pingouin.rm_anova(correction=True)`, `pingouin.epsilon`).  Mauchly's
/// `W` is exact and its χ² approximation is Box's (1949) second-order
/// expansion (`pingouin.sphericity` for `k = 3`; see [`Mauchly`]).
///
/// ```
/// use symplex::prelude::*;
/// use symplex::linprog::q;
/// use symplex::stats::anova::anova_repeated_measures;
/// use symplex::stats::data::from_i64;
///
/// let ctx = Context::new();
/// // Five subjects measured under three conditions.
/// let y = [
///     from_i64(&[5, 7, 9]),
///     from_i64(&[4, 5, 8]),
///     from_i64(&[6, 8, 10]),
///     from_i64(&[3, 6, 4]),
///     from_i64(&[7, 9, 13]),
/// ];
/// // statsmodels: AnovaRM(long, 'y', 'subject', within=['cond']).fit()
/// //   → F Value 12.179775280898882, Num DF 2, Den DF 8, Pr > F 0.0037355110334743136
/// // pingouin: epsilon(wide, correction='gg') = 0.5426457491265326, ('hf') = 0.5877873360597936
/// //           rm_anova(..., correction=True).p_GG_corr = 0.02126436585826144
/// let r = anova_repeated_measures(&ctx, &y)?;
/// assert_eq!(r.f, q(1084, 89));
/// assert_eq!((r.conditions.df, r.error.df), (2, 8));
/// assert!((r.p_value_f64()? - 0.003_735_511_033_474_317).abs() < 1e-12);
/// assert_eq!(r.epsilon_gg, q(7921, 14597));
/// assert_eq!(r.epsilon_hf, Some(q(4168, 7091)));
/// assert!((r.p_value_gg_f64()? - 0.021_264_365_858_261_566).abs() < 1e-12);
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for fewer than two subjects or
/// conditions, ragged rows, or zero error variance (every subject's profile
/// is a shift of the condition means).
pub fn anova_repeated_measures(
    ctx: &Context,
    subjects_by_condition: &[Vec<Q>],
) -> Result<RepeatedMeasuresAnova, SymplexError> {
    const OP: &str = "anova_repeated_measures";
    let n = subjects_by_condition.len();
    if n < 2 {
        return Err(invalid(OP, "at least two subjects are needed"));
    }
    let k = subjects_by_condition.first().map_or(0, Vec::len);
    if k < 2 {
        return Err(invalid(OP, "at least two conditions are needed"));
    }
    if let Some((i, row)) = subjects_by_condition
        .iter()
        .enumerate()
        .find(|(_, r)| r.len() != k)
    {
        return Err(invalid(
            OP,
            format!(
                "subject {i} has {} observations, expected {k} (one per condition)",
                row.len()
            ),
        ));
    }
    let all: Vec<Q> = subjects_by_condition.iter().flatten().cloned().collect();
    let grand_mean = data::mean(&all)?;
    let ss_total = centred_ss(&all);
    let condition_means: Vec<Q> = (0..k)
        .map(|j| {
            subjects_by_condition
                .iter()
                .fold(Q::zero(), |acc, r| acc + &r[j])
                / qu(n)
        })
        .collect();
    let subject_means: Vec<Q> = subjects_by_condition
        .iter()
        .map(|r| data::sum(r) / qu(k))
        .collect();
    let sq = |x: &Q| x * x;
    let ss_conditions = qu(n)
        * condition_means
            .iter()
            .fold(Q::zero(), |acc, m| acc + sq(&(m - &grand_mean)));
    let ss_subjects = qu(k)
        * subject_means
            .iter()
            .fold(Q::zero(), |acc, m| acc + sq(&(m - &grand_mean)));
    let ss_error = &ss_total - &ss_conditions - &ss_subjects;
    if !ss_error.is_positive() {
        return Err(invalid(
            OP,
            "the error variance is zero (every subject's profile is a shift of the condition means): F is undefined",
        ));
    }
    let (df_c, df_s, df_e) = (k - 1, n - 1, (k - 1) * (n - 1));
    let test = FTest::of(ctx, &ss_conditions, df_c, &ss_error, df_e);
    let (f, p_value) = (test.f.clone(), test.p_value.clone());
    let conditions = AnovaRow::with_test(
        Source::Conditions,
        ss_conditions,
        df_c,
        test,
        &ss_error,
        &ss_total,
    );

    // Sample covariance of the conditions across subjects and its
    // double-centred form S̃ = C S C, C = I − 11ᵀ/k.
    let s: Vec<Vec<Q>> = (0..k)
        .map(|a| {
            (0..k)
                .map(|b| {
                    subjects_by_condition.iter().fold(Q::zero(), |acc, r| {
                        acc + (&r[a] - &condition_means[a]) * (&r[b] - &condition_means[b])
                    }) / qu(n - 1)
                })
                .collect()
        })
        .collect();
    let row_means: Vec<Q> = s.iter().map(|r| data::sum(r) / qu(k)).collect();
    let all_mean = data::sum(&row_means) / qu(k);
    let s_tilde: Vec<Vec<Q>> = (0..k)
        .map(|a| {
            (0..k)
                .map(|b| &s[a][b] - &row_means[a] - &row_means[b] + &all_mean)
                .collect()
        })
        .collect();
    let trace = (0..k).fold(Q::zero(), |acc, a| acc + &s_tilde[a][a]);
    let trace_sq = s_tilde
        .iter()
        .flatten()
        .fold(Q::zero(), |acc, v| acc + sq(v));
    // SS_error > 0 ⇒ S̃ ≠ 0 ⇒ tr S̃² > 0.
    if !trace_sq.is_positive() {
        return Err(failed(
            OP,
            "the double-centred covariance vanished although SS_error > 0",
        ));
    }
    let epsilon_gg = sq(&trace) / (qu(df_c) * &trace_sq);
    let hf_denominator = qu(df_c) * (qu(df_s) - qu(df_c) * &epsilon_gg);
    let epsilon_hf = hf_denominator
        .is_positive()
        .then(|| (qu(n) * qu(df_c) * &epsilon_gg - qi(2)) / hf_denominator);
    let corrected = |eps: &Q| f_sf_rational(ctx, &(qu(df_c) * eps), &(qu(df_e) * eps), &f);
    let p_value_gg = corrected(&epsilon_gg);
    let one = Q::one();
    let p_value_hf = epsilon_hf
        .as_ref()
        .map(|e| corrected(if *e > one { &one } else { e }));
    let mauchly = if k >= 3 {
        mauchly(ctx, &s, &trace, n, k)?
    } else {
        None
    };
    Ok(RepeatedMeasuresAnova {
        n_subjects: n,
        n_conditions: k,
        conditions,
        subjects: AnovaRow::untested(Source::Subjects, ss_subjects, df_s),
        error: AnovaRow::untested(Source::Residual, ss_error, df_e),
        total: AnovaRow::total(ss_total, n * k - 1),
        f,
        p_value,
        epsilon_gg,
        epsilon_hf,
        p_value_gg,
        p_value_hf,
        mauchly,
        grand_mean,
        condition_means,
        subject_means,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// The studentized range distribution
// ═══════════════════════════════════════════════════════════════════════════

/// An `m`-point Gauss–Legendre rule on `[−1, 1]`, applied panel-wise.
struct GaussLegendre {
    nodes: Vec<f64>,
    weights: Vec<f64>,
}

impl GaussLegendre {
    /// Nodes by Newton's method on `P_m` from the Chebyshev-like initial
    /// guesses `cos(π(i + ¾)/(m + ½))`; weights `2 / ((1 − x²) P_m′(x)²)`.
    fn new(m: usize) -> Self {
        let mut nodes = Vec::with_capacity(m);
        let mut weights = Vec::with_capacity(m);
        for i in 0..m {
            let mut x = (PI * (i as f64 + 0.75) / (m as f64 + 0.5)).cos();
            for _ in 0..100 {
                let (p, d) = legendre(m, x);
                let step = p / d;
                x -= step;
                if step.abs() < 1e-15 {
                    break;
                }
            }
            // `P_m′` at the final node: the derivative of the previous iterate
            // is off by `P_m″ · step`, a relative `10⁻¹⁴` in the weight.
            let (_, dp) = legendre(m, x);
            nodes.push(x);
            weights.push(2.0 / ((1.0 - x * x) * dp * dp));
        }
        Self { nodes, weights }
    }

    /// `∫ₐᵇ f` with `[a, b]` split into `panels` equal panels.
    fn integrate(&self, f: &dyn Fn(f64) -> f64, a: f64, b: f64, panels: usize) -> f64 {
        let panels = panels.max(1);
        let h = (b - a) / panels as f64;
        let half = 0.5 * h;
        let mut total = 0.0;
        for p in 0..panels {
            let mid = a + (p as f64 + 0.5) * h;
            let mut acc = 0.0;
            for (x, w) in self.nodes.iter().zip(&self.weights) {
                acc += w * f(mid + half * x);
            }
            total += half * acc;
        }
        total
    }
}

/// `(P_m(x), P_m′(x))` by the three-term recurrence.
fn legendre(m: usize, x: f64) -> (f64, f64) {
    let mut p0 = 1.0;
    let mut p1 = x;
    for j in 2..=m {
        let jf = j as f64;
        let p2 = ((2.0 * jf - 1.0) * x * p1 - (jf - 1.0) * p0) / jf;
        p0 = p1;
        p1 = p2;
    }
    let dp = m as f64 * (x * p1 - p0) / (x * x - 1.0);
    (p1, dp)
}

/// Which tail of a distribution on `[0, ∞)` a quadrature computes.  Each
/// tail is integrated directly, never as `1 −` the other: past `10⁻¹⁶` the
/// complement of a CDF is rounding noise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tail {
    /// `P(X ≤ x)`.
    Lower,
    /// `P(X > x)`.
    Upper,
}

impl Tail {
    /// The tail at `x = 0` (`Lower`: `0`, `Upper`: `1`); swapped at `x = ∞`.
    fn at_zero(self) -> f64 {
        match self {
            Tail::Lower => 0.0,
            Tail::Upper => 1.0,
        }
    }

    fn at_infinity(self) -> f64 {
        1.0 - self.at_zero()
    }
}

/// `P(a < Z ≤ a + w)` for a standard normal `Z` and `w ≥ 0`, without
/// cancellation.  The interval is passed by its width: `a + w` rounds `w`
/// away when `w ≪ |a|`.  A short interval (`h(|m| + 1) ≤ 0.1` with midpoint
/// `m = a + h` and half-width `h = w/2`) by the series of `∫_{−h}^{h} e^{−my − y²/2} dy`
/// through the Hermite generating function `e^{xt − t²/2} = Σ Heₙ(x) tⁿ/n!`:
///
/// `P = φ(m) · 2h · Σ_{j≥0} He_{2j}(m) h^{2j} / ((2j + 1)(2j)!)`,
///
/// whose terms fall by at least `(h(|m|+1))²/2` each; otherwise a
/// difference of two tails on the same side of `0` (their ratio is then at
/// most about `e^{−0.1}`, so only a few ulps are lost) or a sum of two
/// `erf` values across `0`.
fn norm_interval(a: f64, w: f64) -> f64 {
    let h = 0.5 * w;
    let m = a + h;
    if h * (m.abs() + 1.0) <= 0.1 {
        // He_{n+1} = m He_n − n He_{n−1}, two steps per term.
        let (mut even, mut odd) = (1.0, m); // He_{2j−2}, He_{2j−1}
        let (mut sum, mut hpow, mut fact) = (1.0, 1.0, 1.0);
        for j in 1..=12 {
            let n = (2 * j) as f64;
            even = m * odd - (n - 1.0) * even; // He_{2j}
            odd = m * even - n * odd; // He_{2j+1}
            hpow *= h * h;
            fact *= (n - 1.0) * n; // (2j)!
            let term = even * hpow / ((n + 1.0) * fact);
            sum += term;
            if term.abs() <= 1e-17 * sum.abs() {
                break;
            }
        }
        return norm_pdf(m) * w * sum;
    }
    let b = a + w;
    if a >= 0.0 {
        0.5 * (erfc(a / SQRT_2) - erfc(b / SQRT_2))
    } else if b <= 0.0 {
        0.5 * (erfc(-b / SQRT_2) - erfc(-a / SQRT_2))
    } else {
        0.5 * (erf(b / SQRT_2) + erf(-a / SQRT_2))
    }
}

/// `c = √(2 (ln k + 39.2))`: the minimum of `k` standard normals falls
/// below `−c` with probability `k Φ(−c) ≤ k φ(c)/c < 10⁻¹⁸`.
fn min_normal_cut(k: usize) -> f64 {
    (2.0 * ((k as f64).ln() + 39.2)).sqrt()
}

/// One tail of the range `W` of `k ≥ 2` independent standard normals at
/// `w ≥ 0`, by composite 16-point Gauss–Legendre quadrature over the
/// position `z` of the minimum:
///
/// * `P(W ≤ w) = k ∫ φ(z) P(z < Z ≤ z + w)^{k−1} dz` ([`norm_interval`]);
/// * `P(W > w) = k ∫ φ(z) Φ̄(z)^{k−1} [1 − (1 − r)^{k−1}] dz` with
///   `r = Φ̄(z + w)/Φ̄(z)`: the density of the minimum times the chance that
///   one of the other `k − 1` exceeds `z + w`, and `1 − (1 − r)^{k−1}` is
///   `−expm1((k − 1) ln1p(−r))` — no difference of nearly equal numbers
///   anywhere, so the upper tail keeps its relative accuracy down to
///   underflow.
///
/// Windows: the upper-tail integrand is at most `k(k−1) φ(z) Φ̄(z + w)`,
/// which peaks at `z = −w/2` and falls by `e^{−d²}` at distance `d`, and at
/// most the density of the minimum, below `−c` ([`min_normal_cut`]) at
/// total mass `10⁻¹⁸`: `[min(−w/2 − 7.5, −c), −w/2 + 7.5]`.  The lower-tail
/// integrand is at most the density of the minimum and, for small `w`,
/// `k w^{k−1} φ(z)^k`: `[−c, 8.5]`.  Panels are at most `2.8` widths of the
/// integrand's features wide (`1/√k` for the lower tail at small `w`,
/// `1/√(2 ln k)` — the spread of the minimum — for the upper tail).
fn normal_range_tail(w: f64, k: usize, tail: Tail, rule: &GaussLegendre) -> f64 {
    if w.is_nan() || w <= 0.0 {
        return tail.at_zero();
    }
    if w == f64::INFINITY {
        return tail.at_infinity();
    }
    let kf = k as f64;
    let km1 = kf - 1.0;
    let cut = min_normal_cut(k);
    let integral = match tail {
        Tail::Lower => {
            let f = |z: f64| norm_pdf(z) * norm_interval(z, w).powf(km1);
            let (a, b, width) = (-cut, 8.5, (2.8 / kf.sqrt()).min(2.0));
            rule.integrate(&f, a, b, ((b - a) / width).ceil() as usize)
        }
        Tail::Upper => {
            let f = |z: f64| {
                let above = norm_sf(z);
                if above <= 0.0 {
                    return 0.0;
                }
                let r = norm_sf(z + w) / above;
                let one_exceeds = -(km1 * (-r).ln_1p()).exp_m1();
                norm_pdf(z) * above.powf(km1) * one_exceeds
            };
            let centre = -0.5 * w;
            let (a, b) = ((centre - 7.5).min(-cut), centre + 7.5);
            let width = (2.8 / (2.0 * kf.ln()).max(1.0).sqrt()).min(2.0);
            rule.integrate(&f, a, b, ((b - a) / width).ceil() as usize)
        }
    };
    (kf * integral).clamp(0.0, 1.0)
}

/// `R(x) = ln Γ(x) − [(x − ½) ln x − x + ½ ln 2π]`, the remainder of
/// Stirling's formula, for `x ≥ ½`, to a few ulps.  From `x = 10` the
/// asymptotic series `Σ B₂ₖ/(2k(2k−1) x^{2k−1})` through `k = 7` (DLMF
/// 5.11.1; the next term is below `3·10⁻¹⁷` there); below, the shift
/// `R(x) = R(x + 1) + (x + ½) ln(1 + 1/x) − 1` with the difference summed
/// as the positive series `Σ_{j≥1} v^{2j}/(2j + 1)`, `v = 1/(2x + 1)` (from
/// `ln(1 + 1/x) = ln((1 + v)/(1 − v))`), so nothing cancels.  (`ln Γ` in
/// `f64` and a subtraction lose about `10⁻¹⁵` absolutely, which the
/// density prefactor would carry as a relative error.)
fn stirling_remainder(x: f64) -> f64 {
    const B: [f64; 7] = [
        1.0 / 12.0,
        -1.0 / 360.0,
        1.0 / 1260.0,
        -1.0 / 1680.0,
        1.0 / 1188.0,
        -691.0 / 360_360.0,
        1.0 / 156.0,
    ];
    let mut x = x;
    let mut shift = 0.0;
    while x < 10.0 {
        let v2 = (1.0 / (2.0 * x + 1.0)).powi(2);
        let (mut pow, mut sum) = (v2, 0.0);
        for j in 1..=40 {
            let term = pow / (2 * j + 1) as f64;
            sum += term;
            if term <= 1e-17 * sum {
                break;
            }
            pow *= v2;
        }
        shift += sum;
        x += 1.0;
    }
    let inv2 = 1.0 / (x * x);
    shift + B.iter().rev().fold(0.0, |acc, &b| acc * inv2 + b) / x
}

/// `ln(1 + u) − u`, without the cancellation of the two terms for small `u`
/// (the alternating series `−u²/2 + u³/3 − …`).
fn ln1p_minus_u(u: f64) -> f64 {
    if u.abs() >= 0.25 {
        return u.ln_1p() - u;
    }
    let mut term = u * u;
    let mut sum = 0.0;
    for n in 2..=40 {
        let contribution = term / n as f64;
        sum += if n % 2 == 0 {
            -contribution
        } else {
            contribution
        };
        term *= u;
        if contribution.abs() <= 1e-18 * sum.abs() {
            break;
        }
    }
    sum
}

/// Above this many degrees of freedom `S = √(χ²_ν/ν)` is `1` for every
/// purpose of a double: the studentized range is the range of `k` normals
/// up to a relative `O(q⁴/ν)` (the upper tail is `0` in `f64` past
/// `q ≈ 55`).
const NU_NORMAL_LIMIT: f64 = 1e30;

/// The outer window ends where the log-integrand has fallen this far below
/// its maximum; by log-concavity (see [`studentized_range_tail`]) the mass
/// beyond is below `e^{−45} ≈ 3·10⁻²⁰` of the total.
const OUTER_LOG_DROP: f64 = 45.0;

/// The largest change of the log-integrand across one outer panel: a
/// 16-point Gauss–Legendre panel integrates `e^{−12x}` on `[0, 1]` to a
/// relative `10⁻²⁰`.
const OUTER_PANEL_LOG_CHANGE: f64 = 12.0;

/// A log-integrand maximum below which the outer integral underflows (see
/// [`studentized_range_tail`]): `e^{−760} · 3000 < 4.9·10⁻³²⁴`.
const UNDERFLOW_LOG_MODE: f64 = -760.0;

/// Cap on the evaluations of each search loop of [`studentized_range_tail`]
/// (each loop at least doubles a step, so a few hundred reach any double).
const OUTER_SEARCH_CAP: usize = 4096;

/// One tail of the studentized range `Q = W/S` for `k` groups and `ν`
/// degrees of freedom (`W` the range of `k` standard normals, `S =
/// √(χ²_ν/ν)` independent), computed directly — the upper tail is never
/// `1 − P(Q ≤ q)`:
///
/// `P(Q ≤ q) = ∫₀^∞ f_ν(s) P(W ≤ qs) ds`, `P(Q > q) = ∫₀^∞ f_ν(s) P(W > qs) ds`
///
/// with the range tails of [`normal_range_tail`].  Arguments are validated
/// by the caller (`k ≥ 2`, `ν ≥ 1`, `q` not NaN).
///
/// **Variable.**  The outer integral runs over `t = ln s`, with integrand
/// `s f_ν(s) P(W ≶ qs)`: near `s = 1` the abscissae stay resolved when
/// `1/√(2ν)` is below the spacing of doubles, and near `s = 0` (a far upper
/// tail at small `ν`, where the mass sits at `s ≈ 1/q`) they keep their
/// relative precision.  `ln(s f_ν(s))` is evaluated with the `O(ν)` terms
/// cancelled analytically (`x = ν/2`, `u = e^t − 1`):
///
/// `ln(s f_ν(s)) = ln 2 + ½ ln(x/2π) − R(x) + 2x·(t − u) − x u²`
///
/// with `R` the Stirling remainder of `ln Γ(x)` and `t − u = ln(1+u) − u`
/// by its series for small `u` ([`ln1p_minus_u`]).
///
/// **Window.**  `f_ν` is log-concave for `ν ≥ 1`, and so are both tails of
/// `W` (the joint density of the minimum and maximum of normals is
/// log-concave, and so are its linear images and their integrals —
/// Prékopa), hence the integrand is log-concave in `s` and unimodal in
/// `t`.  Its mode is bracketed from the estimate `s² ≈ ν/(ν + q²/2)`
/// (upper tail; `P(W > w) ≈ e^{−w²/4}`) or `(ν + k − 1)/ν` (lower tail;
/// `P(W ≤ w) ∝ w^{k−1}`) by doubling steps and refined by golden-section
/// search; from there 16-point panels march outwards on each side, each
/// twice the previous but halved until the log-integrand changes by at
/// most [`OUTER_PANEL_LOG_CHANGE`] across it, until it is
/// [`OUTER_LOG_DROP`] below the mode.
///
/// **Accuracy.**  Against mpmath (the same double integral at 25–45
/// digits, with the range tail both in this non-cancelling form and as
/// `1 − P(W ≤ w)` at high precision) and the exact `k = 2` case
/// `P(Q > q) = 2 P(T_ν > q/√2)`: a relative `10⁻¹⁵` in the body, a few
/// `10⁻¹⁴` in a far upper tail, where the complementary error function
/// itself is that accurate.
///
/// Past [`NU_NORMAL_LIMIT`] the range tail itself is returned.
fn studentized_range_tail(q: f64, k: usize, nu: f64, tail: Tail) -> Result<f64, SymplexError> {
    const OP: &str = "studentized_range";
    if q.is_nan() || q <= 0.0 {
        return Ok(tail.at_zero());
    }
    if q == f64::INFINITY {
        return Ok(tail.at_infinity());
    }
    let rule = GaussLegendre::new(16);
    if nu >= NU_NORMAL_LIMIT {
        return Ok(normal_range_tail(q, k, tail, &rule));
    }
    let x = 0.5 * nu;
    let log_prefactor = LN_2 + 0.5 * (x / (2.0 * PI)).ln() - stirling_remainder(x);
    let log_weight = |t: f64| {
        let u = t.exp_m1();
        let t_minus_u = if u.abs() < 0.25 {
            ln1p_minus_u(u)
        } else {
            t - u
        };
        log_prefactor + 2.0 * x * t_minus_u - x * u * u
    };
    let range_tail = |t: f64| normal_range_tail(q * t.exp(), k, tail, &rule);
    let log_h = |t: f64| log_weight(t) + range_tail(t).ln();
    let not_found = || {
        failed(
            OP,
            format!(
                "the studentized range quadrature found no window (q = {q}, k = {k}, df = {nu})"
            ),
        )
    };

    // 1. Bracket the mode, walking uphill with doubling steps from the
    //    estimate.  Where the integrand is 0 (the range tail underflowed)
    //    the mode lies towards smaller s for the upper tail, larger s for
    //    the lower.
    let scale = 1.0 / (2.0 * nu + 1.0).sqrt();
    let (t0, default_dir) = match tail {
        Tail::Upper => {
            let r = q / (2.0 * nu).sqrt();
            let t0 = if r > 1e100 {
                -r.ln()
            } else {
                -0.5 * (r * r).ln_1p()
            };
            (t0, -1.0)
        }
        Tail::Lower => (0.5 * ((k - 1) as f64 / nu).ln_1p(), 1.0),
    };
    let l0 = log_h(t0);
    let (l_right, l_left) = (log_h(t0 + scale), log_h(t0 - scale));
    let dir = if l_right > l0 && l_right >= l_left {
        1.0
    } else if l_left > l0 {
        -1.0
    } else if l0 == f64::NEG_INFINITY {
        default_dir
    } else {
        0.0
    };
    let (mut lo, mut mode, mut hi, mut l_mode) = (t0 - scale, t0, t0 + scale, l0);
    if dir != 0.0 {
        let l_first = if dir > 0.0 { l_right } else { l_left };
        let (mut prev, mut cur, mut l_cur) = (t0, t0 + dir * scale, l_first);
        let mut step = scale;
        let mut bracketed = false;
        for _ in 0..OUTER_SEARCH_CAP {
            step *= 2.0;
            let next = cur + dir * step;
            if next.is_nan() || next.abs() >= 1500.0 {
                // `e^t` is 0 or ∞ in `f64` from here on.  An integrand that
                // underflowed everywhere on the way is a tail below the
                // smallest double (the lower tail at a tiny `q`).
                if l_cur == f64::NEG_INFINITY {
                    return Ok(0.0);
                }
                break;
            }
            let l_next = log_h(next);
            if l_next > l_cur || (l_next == f64::NEG_INFINITY && l_cur == f64::NEG_INFINITY) {
                (prev, cur, l_cur) = (cur, next, l_next);
            } else {
                (lo, hi) = if dir > 0.0 {
                    (prev, next)
                } else {
                    (next, prev)
                };
                (mode, l_mode) = (cur, l_cur);
                bracketed = true;
                break;
            }
        }
        if !bracketed || !l_mode.is_finite() {
            return Err(not_found());
        }
    }
    // The integrand is unimodal and the window lies inside |t| < 1500, so
    // the integral is below `3000 e^{l_mode}`: under `e^{−760}` that is
    // below the smallest subnormal.  (Where the range tail underflows at
    // the estimate — the lower tail at a tiny `q` — the walk can stop at a
    // point where the density is `e^{−10¹²⁰}` and the march below could
    // not resolve the integrand.)
    if l_mode < UNDERFLOW_LOG_MODE {
        return Ok(0.0);
    }

    // 2. Golden-section refinement (the mode need only be known to a small
    //    fraction of the integrand's width, which is at least about
    //    `scale/√k`).
    let tol = 0.05 * scale / (k as f64).sqrt();
    for _ in 0..OUTER_SEARCH_CAP {
        if hi - lo <= tol {
            break;
        }
        const GOLDEN: f64 = 0.381_966_011_250_105_1;
        let left = mode - lo > hi - mode;
        let probe = if left {
            mode - GOLDEN * (mode - lo)
        } else {
            mode + GOLDEN * (hi - mode)
        };
        if probe == mode {
            break;
        }
        let l_probe = log_h(probe);
        match (l_probe > l_mode, left) {
            (true, true) => (hi, mode, l_mode) = (mode, probe, l_probe),
            (true, false) => (lo, mode, l_mode) = (mode, probe, l_probe),
            (false, true) => lo = probe,
            (false, false) => hi = probe,
        }
    }

    // 3. March outwards from the mode, one 16-point panel at a time, until
    //    the log-integrand is `OUTER_LOG_DROP` below the mode.  A panel
    //    doubles the previous one but is halved until the log-integrand
    //    changes by at most `OUTER_PANEL_LOG_CHANGE` across it: fine at the
    //    peak, wide in an exponential tail (small ν), narrow where the
    //    decay is super-exponential (large s).
    let h = |t: f64| log_weight(t).exp() * range_tail(t);
    let floor = l_mode - OUTER_LOG_DROP;
    let first = 0.25 * scale / (k as f64).sqrt();
    let mut total = 0.0;
    for dir in [-1.0, 1.0] {
        let (mut pos, mut l_pos, mut step) = (mode, l_mode, first);
        let mut reached = false;
        'march: for _ in 0..OUTER_SEARCH_CAP {
            let (mut next, mut l_next) = (pos + dir * step, f64::NAN);
            for _ in 0..OUTER_SEARCH_CAP {
                next = pos + dir * step;
                if !next.is_finite() || next == pos {
                    break 'march;
                }
                l_next = log_h(next);
                // An endpoint where the integrand underflowed ends the march
                // (it is below any floor above the underflow threshold).
                if (l_next - l_pos).abs() <= OUTER_PANEL_LOG_CHANGE || l_next == f64::NEG_INFINITY {
                    break;
                }
                step *= 0.5;
            }
            total += rule.integrate(&h, pos.min(next), pos.max(next), 1);
            if l_next < floor {
                reached = true;
                break;
            }
            (pos, l_pos, step) = (next, l_next, 2.0 * step);
        }
        if !reached {
            return Err(not_found());
        }
    }
    Ok(total.clamp(0.0, 1.0))
}

fn check_studentized_range_args(op: &'static str, k: usize, df: f64) -> Result<(), SymplexError> {
    if k < 2 {
        return Err(invalid(op, format!("k must be at least 2, got {k}")));
    }
    if !(df.is_finite() && df >= 1.0) {
        return Err(invalid(
            op,
            format!("the degrees of freedom must be finite and at least 1, got {df}"),
        ));
    }
    Ok(())
}

/// `P(Q ≤ q)` for the studentized range distribution of `k` groups with `df`
/// degrees of freedom, by composite Gauss–Legendre quadrature of the double
/// integral (relative accuracy about `1e-14`, also in the lower tail as
/// `q → 0`).  `scipy.stats.studentized_range.cdf(q, k, df)`.
///
/// `df` must be finite; from `df = 1e30` on the result is the `df = ∞`
/// limit (the range of `k` standard normals) to double precision.
///
/// ```
/// use symplex::stats::anova::studentized_range_cdf;
///
/// // scipy: studentized_range.cdf(3.0, 3, 12) = 0.8729674086442558
/// assert!((studentized_range_cdf(3.0, 3, 12.0)? - 0.872_967_408_644_255_8).abs() < 1e-8);
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for `k < 2`, `df < 1`, a non-finite
/// `df` or a NaN `q`; [`SymplexError::ComputationFailed`] if the quadrature
/// finds no window (not expected).
pub fn studentized_range_cdf(q: f64, k: usize, df: f64) -> Result<f64, SymplexError> {
    const OP: &str = "studentized_range_cdf";
    check_studentized_range_args(OP, k, df)?;
    if q.is_nan() {
        return Err(invalid(OP, "q must not be NaN"));
    }
    studentized_range_tail(q, k, df, Tail::Lower)
}

/// `P(Q > q)` of the studentized range distribution, integrated directly
/// from the upper tail of the range of `k` normals (never as `1 − cdf`), so
/// it keeps its relative accuracy (about `1e-14`) down to underflow.
/// `scipy.stats.studentized_range.sf(q, k, df)` — which *is* `1 − cdf`
/// and is rounding noise below `10⁻¹⁵`.
///
/// ```
/// use symplex::stats::anova::studentized_range_sf;
///
/// // mpmath (dps 25, the double integral with the non-cancelling range tail):
/// //   P(Q > 40 | k = 4, df = 100) = 9.8157947218104339959e-49
/// // scipy: studentized_range.sf(40, 4, 100) = 7.771561172376096e-16 (1 − cdf)
/// let p = studentized_range_sf(40.0, 4, 100.0)?;
/// assert!((p / 9.815_794_721_810_433_995_9e-49 - 1.0).abs() < 1e-13);
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// As [`studentized_range_cdf`].
pub fn studentized_range_sf(q: f64, k: usize, df: f64) -> Result<f64, SymplexError> {
    const OP: &str = "studentized_range_sf";
    check_studentized_range_args(OP, k, df)?;
    if q.is_nan() {
        return Err(invalid(OP, "q must not be NaN"));
    }
    studentized_range_tail(q, k, df, Tail::Upper)
}

/// The quantile `q_p` with `P(Q ≤ q_p) = p` of the studentized range
/// distribution (the Tukey critical value for `p = 1 − α`), by Brent's
/// method on the logarithm of the *smaller* tail in `ln q`: `ln P(Q ≤ q) =
/// ln p` for `p ≤ ½`, `ln P(Q > q) = ln(1 − p)` above (`1 − p` is exact
/// there), so levels near `0` and near `1` keep their relative accuracy.
/// `scipy.stats.studentized_range.ppf(p, k, df)`.
///
/// ```
/// use symplex::stats::anova::studentized_range_quantile;
///
/// // scipy: studentized_range.ppf(0.95, 3, 15) = 3.6733776588970977
/// assert!((studentized_range_quantile(0.95, 3, 15.0)? - 3.673_377_658_897_097_7).abs() < 1e-6);
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for `k < 2`, `df < 1`, a non-finite
/// `df` or `p ∉ (0, 1)`; [`SymplexError::ComputationFailed`] if the root
/// search fails.
pub fn studentized_range_quantile(p: f64, k: usize, df: f64) -> Result<f64, SymplexError> {
    const OP: &str = "studentized_range_quantile";
    check_studentized_range_args(OP, k, df)?;
    check_unit_open(OP, "p", p)?;
    let (tail, level) = if p <= 0.5 {
        (Tail::Lower, p)
    } else {
        (Tail::Upper, 1.0 - p)
    };
    let ln_level = level.ln();
    // In `x = ln q`; an underflowed tail is floored so `g` stays finite and
    // monotone (`ln level ≥ ln 2⁻¹⁰⁷⁴ > −1e4`).
    let g = |x: f64| match studentized_range_tail(x.exp(), k, df, tail) {
        Ok(v) => v.ln().max(-1e4) - ln_level,
        Err(_) => f64::NAN,
    };
    let bracket = grow_bracket(g, 0.0, 1.0, Bounds::free(), 64).map_err(|e| {
        failed(
            OP,
            format!("no bracket for the studentized range quantile: {e}"),
        )
    })?;
    let opts = RootOpts {
        xtol: 1e-14,
        ..RootOpts::default()
    };
    let root = brent_root(g, bracket.lower, bracket.upper, &opts)
        .map_err(|e| failed(OP, e.to_string()))?
        .exp();
    if root.is_finite() && root > 0.0 {
        Ok(root)
    } else {
        Err(failed(
            OP,
            "the studentized range quantile did not converge",
        ))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Post-hoc comparisons
// ═══════════════════════════════════════════════════════════════════════════

/// One pairwise comparison of Tukey's HSD; see [`tukey_hsd`].
#[derive(Clone, Debug, PartialEq)]
pub struct PairwiseComparison {
    /// Index of the first group.
    pub i: usize,
    /// Index of the second group (`i < j`).
    pub j: usize,
    /// `x̄ᵢ − x̄ⱼ`, exact.
    pub diff: Q,
    /// `√(MSE/2 · (1/nᵢ + 1/nⱼ))`, exact (a square root of a rational).
    pub se: Ex,
    /// `|diff| / se`, exact: the studentized range statistic.
    pub statistic: Ex,
    /// `P(Q_{k, N−k} ≥ statistic)`, the family-wise adjusted p-value
    /// ([`studentized_range_sf`]; `f64`: the studentized range distribution
    /// is integrated numerically, to a relative `1e-14` also in the far
    /// tail).
    pub p_adj: f64,
    /// `diff ± q_{confidence, k, N−k} · se`.
    pub ci: Interval<f64>,
}

/// Tukey's honestly-significant-difference test of all `k(k−1)/2` pairwise
/// differences of means of `k` independent groups, with the Tukey–Kramer
/// standard error for unequal sizes: `se = √(MSE/2 · (1/nᵢ + 1/nⱼ))`,
/// `MSE = SS_within/(N − k)`, statistic `|x̄ᵢ − x̄ⱼ|/se` referred to the
/// studentized range distribution with `k` groups and `N − k` degrees of
/// freedom, and simultaneous `confidence` intervals `x̄ᵢ − x̄ⱼ ± q · se`.
/// `scipy.stats.tukey_hsd(*groups)` and `.confidence_interval(confidence)`.
///
/// Differences, standard errors and statistics are exact; the p-values and
/// interval limits are `f64`.  The p-values are the studentized range
/// upper tail integrated directly ([`studentized_range_sf`]), so a wide
/// separation gets its true tiny p-value (`scipy.stats.tukey_hsd` reports
/// `1 − cdf`, rounding noise below `10⁻¹⁵`).
///
/// ```
/// use symplex::prelude::*;
/// use symplex::linprog::{q, qi};
/// use symplex::stats::anova::tukey_hsd;
/// use symplex::stats::data::from_i64;
///
/// let ctx = Context::new();
/// let g = [from_i64(&[6, 8, 4, 5, 3, 4]), from_i64(&[8, 12, 9, 11, 6, 8]), from_i64(&[13, 9, 11, 8, 7, 12])];
/// // scipy: r = tukey_hsd(*g); r.statistic[0, 1] = -4.0, r.pvalue[0, 1] = 0.013913287267276697,
/// //        r.confidence_interval(0.95).low[0, 1] = -7.192999, .high[0, 1] = -0.807001
/// let pairs = tukey_hsd(&ctx, &g, 0.95)?;
/// assert_eq!((pairs[0].i, pairs[0].j), (0, 1));
/// assert_eq!(pairs[0].diff, qi(-4));
/// assert_eq!(pairs[0].se, ctx.from_ratio(q(34, 45)).sqrt());
/// assert!((pairs[0].p_adj - 0.013_913_287_267_276_697).abs() < 1e-7);
/// assert!((pairs[0].ci.lower - -7.192_999).abs() < 1e-5);
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for fewer than two groups, an empty
/// group, `N ≤ k`, zero within-group variance, or `confidence ∉ (0, 1)`;
/// [`SymplexError::ComputationFailed`] if a difference, standard error or
/// statistic does not fit in an `f64` (the limits and p-values are `f64`).
pub fn tukey_hsd(
    ctx: &Context,
    groups: &[Vec<Q>],
    confidence: f64,
) -> Result<Vec<PairwiseComparison>, SymplexError> {
    const OP: &str = "tukey_hsd";
    check_confidence(OP, confidence)?;
    let k = groups.len();
    if k < 2 {
        return Err(invalid(OP, "at least two groups are needed"));
    }
    if groups.iter().any(Vec::is_empty) {
        return Err(invalid(OP, "every group must be non-empty"));
    }
    let n: usize = groups.iter().map(Vec::len).sum();
    if n <= k {
        return Err(invalid(
            OP,
            "at least one group needs more than one observation",
        ));
    }
    let df = n - k;
    let means = groups
        .iter()
        .map(|g| data::mean(g))
        .collect::<Result<Vec<_>, _>>()?;
    let ss_within = groups.iter().fold(Q::zero(), |acc, g| acc + centred_ss(g));
    if ss_within.is_zero() {
        return Err(invalid(OP, "the within-group variance is zero"));
    }
    let mse = ss_within / qu(df);
    let df_f = df as f64;
    let q_crit = studentized_range_quantile(confidence, k, df_f)?;
    let mut out = Vec::with_capacity(k * (k - 1) / 2);
    for i in 0..k {
        for j in i + 1..k {
            let diff = &means[i] - &means[j];
            let var = &mse / qi(2) * (qu(groups[i].len()).recip() + qu(groups[j].len()).recip());
            let se = ex(ctx, &var).sqrt();
            let statistic = ex(ctx, &diff.abs()) / &se;
            let (diff_f, se_f) = (to_f64(OP, &diff)?, to_f64(OP, &var)?.sqrt());
            let stat_f = to_f64(OP, &(&diff * &diff / &var))?.sqrt();
            let p_adj = studentized_range_tail(stat_f, k, df_f, Tail::Upper)?;
            out.push(PairwiseComparison {
                i,
                j,
                diff,
                se,
                statistic,
                p_adj,
                ci: Interval::closed(diff_f - q_crit * se_f, diff_f + q_crit * se_f),
            });
        }
    }
    Ok(out)
}

/// A family-wise adjustment of pairwise p-values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Adjustment {
    /// `p̃ = min(1, m p)` ([`hypothesis::bonferroni`]).
    Bonferroni,
    /// Holm's step-down procedure ([`hypothesis::holm`]).
    Holm,
}

/// One pairwise Welch t-test with its adjusted p-value; see
/// [`pairwise_t_tests`].
#[derive(Clone, Debug, PartialEq)]
pub struct PairwiseTTest {
    /// Index of the first group.
    pub i: usize,
    /// Index of the second group (`i < j`).
    pub j: usize,
    /// `x̄ᵢ − x̄ⱼ`, exact.
    pub diff: Q,
    /// The two-sided Welch test of the pair (exact statistic, exact p-value
    /// expression, rational Welch–Satterthwaite `df`).
    pub test: TestResult,
    /// The adjusted p-value.
    pub p_adj: f64,
    /// Whether the pair differs at family-wise level `alpha`.
    pub reject: bool,
}

/// All `k(k−1)/2` pairwise two-sided Welch t-tests
/// ([`hypothesis::t_test_two_sample`] with `equal_var = false`) with the
/// p-values adjusted by `adjustment` at family-wise level `alpha` — the
/// post-hoc procedure to use when variances differ between groups.
/// `scipy.stats.ttest_ind(gᵢ, gⱼ, equal_var=False)` per pair, then
/// `statsmodels.stats.multitest.multipletests(p, alpha, method)`.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::linprog::qi;
/// use symplex::stats::anova::{pairwise_t_tests, Adjustment};
/// use symplex::stats::data::from_i64;
///
/// let ctx = Context::new();
/// let g = [from_i64(&[6, 8, 4, 5, 3, 4]), from_i64(&[8, 12, 9, 11, 6, 8]), from_i64(&[13, 9, 11, 8, 7, 12])];
/// // scipy: ttest_ind(g[0], g[1], equal_var=False).pvalue = 0.00644386616395533, (g[0], g[2]) 0.002386749612426776,
/// //        (g[1], g[2]) 0.46515103975349575
/// // statsmodels: multipletests(p, method='holm')[1] = [0.01288773232791066, 0.007160248837280328, 0.46515103975349575]
/// let t = pairwise_t_tests(&ctx, &g, Adjustment::Holm, 0.05)?;
/// assert_eq!(t.len(), 3);
/// assert_eq!(t[1].diff, qi(-5));
/// assert!((t[0].p_adj - 0.012_887_732_327_910_66).abs() < 1e-12);
/// assert_eq!(t.iter().map(|p| p.reject).collect::<Vec<_>>(), vec![true, true, false]);
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for fewer than two groups, a group
/// with fewer than two observations, a pair of constant groups, or
/// `alpha ∉ (0, 1)`.
pub fn pairwise_t_tests(
    ctx: &Context,
    groups: &[Vec<Q>],
    adjustment: Adjustment,
    alpha: f64,
) -> Result<Vec<PairwiseTTest>, SymplexError> {
    const OP: &str = "pairwise_t_tests";
    let k = groups.len();
    if k < 2 {
        return Err(invalid(OP, "at least two groups are needed"));
    }
    let mut pairs = Vec::with_capacity(k * (k - 1) / 2);
    let mut p_values = Vec::with_capacity(k * (k - 1) / 2);
    for i in 0..k {
        for j in i + 1..k {
            let test = hypothesis::t_test_two_sample(
                ctx,
                &groups[i],
                &groups[j],
                false,
                Alternative::TwoSided,
            )?;
            p_values.push(test.p_value_f64()?);
            let diff = data::mean(&groups[i])? - data::mean(&groups[j])?;
            pairs.push((i, j, diff, test));
        }
    }
    let adjusted = match adjustment {
        Adjustment::Bonferroni => hypothesis::bonferroni(&p_values, alpha)?,
        Adjustment::Holm => hypothesis::holm(&p_values, alpha)?,
    };
    Ok(pairs
        .into_iter()
        .zip(adjusted.p_adjusted)
        .zip(adjusted.reject)
        .map(|(((i, j, diff, test), p_adj), reject)| PairwiseTTest {
            i,
            j,
            diff,
            test,
            p_adj,
            reject,
        })
        .collect())
}
