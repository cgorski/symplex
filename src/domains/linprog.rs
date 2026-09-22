//! Exact linear programming over ℚ: two-phase simplex with Bland's rule,
//! dual values and Farkas infeasibility certificates.
//!
//! Everything is exact — no tolerances, no "numerically infeasible"
//! verdicts.  Problem data, optima, duals and certificates are
//! `Ratio<BigInt>`; internally the dense tableau pivots on integers with a
//! common denominator (see [Algorithm and cost](#algorithm-and-cost)).
//! Use [`linprog_matrix`] to feed [`Matrix`] data whose entries are
//! numeric literals.
//!
//! # Problem form
//!
//! [`LpProblem`] is a builder for
//!
//! ```text
//! minimise / maximise   cᵀx
//! subject to            aᵢ·x  ≤ / = / ≥  bᵢ        (one row per constraint)
//!                       lⱼ ≤ xⱼ ≤ uⱼ                 (default 0 ≤ xⱼ < ∞)
//! ```
//!
//! [`linprog`] is a SciPy-shaped convenience wrapper; [`feasible_nonneg`]
//! answers "is there an `x ≥ 0` with `A x = b`?" exactly,
//! [`feasible_nonneg_certified`] additionally returns the Farkas certificate
//! when the answer is no, and [`nonneg_combination`] asks the same question
//! about a target vector and a list of generating vectors (cone membership).
//!
//! # Algorithm and cost
//!
//! Two-phase dense simplex.  Pivots follow Dantzig's rule; after twelve
//! consecutive degenerate (zero-length) steps Bland's rule takes over until
//! the next improving step, so cycling is impossible while the highly
//! degenerate certificate LPs are not condemned to Bland's slow walk from
//! their first zero pivot.  Every `≤`/`≥` row gets a
//! slack, every row gets an artificial (their columns double as `B⁻¹`, from
//! which the duals are read), free variables are split `x = x⁺ − x⁻`, finite
//! lower bounds are shifted away and a finite *upper* bound on a variable
//! that also has a lower bound costs one extra row.
//!
//! The tableau is kept **integral** (Bareiss/Edmonds integer pivoting):
//! each constraint row is scaled once by the lcm of its denominators, and
//! every pivot applies the fraction-free rule `row ← (row·p − row[s]·pivot_row)
//! / d`, so all entries are integers sharing the common denominator `d`
//! (the current pivot, `±det B`).  Entering and leaving variables are
//! chosen from the same rational values a `Ratio` tableau would hold — the
//! pivot sequence, optimum, duals and certificates are identical — but no
//! gcd is computed in the pivot loop.  Each pivot is `O(m·n)` big-integer
//! operations; a 60-row × 160-variable problem with default bounds solves
//! in under 100 ms in a release build.
//!
//! # Duals and certificates
//!
//! When the status is [`LpStatus::Optimal`], [`LpSolution::duals`] holds one
//! **shadow price** `yᵢ` per constraint, in insertion order: the rate of
//! change of the optimal objective value *of the problem as posed* with
//! respect to `bᵢ`.  Consequently, for a minimisation `yᵢ ≤ 0` on `≤` rows
//! and `yᵢ ≥ 0` on `≥` rows (the signs flip for a maximisation), `yᵢ` is
//! free on `=` rows, and with the reduced costs `r = c − Aᵀy`:
//!
//! * complementary slackness: `yᵢ·(aᵢ·x* − bᵢ) = 0` for every row;
//! * `rⱼ = 0` unless `x*ⱼ` sits at a finite bound (for a minimisation
//!   `rⱼ ≥ 0` at a lower bound and `rⱼ ≤ 0` at an upper bound; reversed for
//!   a maximisation);
//! * strong duality: `cᵀx* = yᵀb + Σⱼ rⱼ x*ⱼ`, which reduces to
//!   `cᵀx* = yᵀb` under the default bounds `x ≥ 0`.
//!
//! When the status is [`LpStatus::Infeasible`], [`LpSolution::farkas`] is a
//! **Farkas certificate** `y` (one entry per constraint) with `yᵢ ≥ 0` on
//! `≤` rows, `yᵢ ≤ 0` on `≥` rows, free on `=` rows, such that, with
//! `g = Aᵀy`,
//!
//! ```text
//! Σⱼ  inf { gⱼ·xⱼ : lⱼ ≤ xⱼ ≤ uⱼ }   >   yᵀb
//! ```
//!
//! where every infimum is finite (`gⱼ > 0 ⇒ lⱼ` finite, `gⱼ < 0 ⇒ uⱼ`
//! finite, `gⱼ = 0` contributes `0`).  Any feasible `x` would satisfy
//! `(Aᵀy)·x ≤ yᵀb`, so the inequality proves that none exists.  With no
//! finite bounds this is the textbook form `Aᵀy = 0, yᵀb < 0`.  The
//! certificate is `None` only when infeasibility is caused by the bounds
//! alone (`lⱼ > uⱼ`).
//!
//! # Budgets
//!
//! A solve can be bounded by a [`Budget`] — an absolute deadline, a time
//! limit counted from the start of the solve, and/or a cap on the number
//! of pivots ([`LpProblem::with_budget`]).  The budget is checked at every
//! pivot; when it runs out the solve stops and reports
//! [`LpStatus::BudgetExhausted`] (an answer, not an error), with no point,
//! objective or certificate.  Without a budget the solver behaves exactly
//! as before.  `Budget` is the crate-wide type
//! ([`symplex::Budget`](crate::base::budget::Budget)), re-exported here.
//!
//! # Examples
//!
//! ```
//! use symplex::linprog::{LpProblem, LpStatus, qi};
//!
//! // max 3x + 2y  s.t.  x + y ≤ 4,  x + 3y ≤ 6,  x, y ≥ 0
//! let sol = LpProblem::maximize(vec![qi(3), qi(2)])
//!     .le(vec![qi(1), qi(1)], qi(4))
//!     .le(vec![qi(1), qi(3)], qi(6))
//!     .solve()
//!     .unwrap();
//! assert_eq!(sol.status, LpStatus::Optimal);
//! assert_eq!(sol.x, vec![qi(4), qi(0)]);
//! assert_eq!(sol.objective, Some(qi(12)));
//! // Shadow prices: the first constraint is binding with price 3.
//! assert_eq!(sol.duals, vec![qi(3), qi(0)]);
//! ```

use std::fmt;
use std::time::Instant;

use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::Ratio;
use num_traits::{One, Signed, Zero};

use crate::api::context::Context;
use crate::api::expr::Ex;
use crate::base::errors::SymplexError;
use crate::base::interval::Bounds;
use crate::domains::exact_kernel::{Cell, W256, pivot_row_update};
use crate::domains::matrix::Matrix;

// The exact rational type and its literal constructors live in
// `base::numeric` (shared with the exact matrices, polytopes and
// certificates); they keep their `linprog::` paths.
pub use crate::base::numeric::{Q, q, qi};

fn invalid(operation: &'static str, reason: impl Into<String>) -> SymplexError {
    SymplexError::invalid_argument(operation, reason)
}

fn failed(operation: &'static str, reason: impl Into<String>) -> SymplexError {
    SymplexError::computation_failed(operation, reason)
}

// ═══════════════════════════════════════════════════════════════════════════
// Problem description
// ═══════════════════════════════════════════════════════════════════════════

/// Optimisation direction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Objective {
    /// Minimise `cᵀx`.
    Minimize,
    /// Maximise `cᵀx`.
    Maximize,
}

/// Relation of a constraint row to its right-hand side.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Relation {
    Eq,
    Le,
    Ge,
}

#[derive(Clone, Debug)]
struct Constraint {
    row: Vec<Q>,
    rhs: Q,
    relation: Relation,
}

// The budget types live in `base::budget` (shared with the certificate
// provers); this is their historical public path.
pub use crate::base::budget::{Budget, BudgetHit};

/// Why a budgeted search stopped without an answer: its budget ran out,
/// or an LP failed.
pub(crate) enum Stop {
    Budget(BudgetHit),
    Error(SymplexError),
}

impl From<SymplexError> for Stop {
    fn from(e: SymplexError) -> Self {
        Stop::Error(e)
    }
}

/// The budget of one prover call, shared by every LP it runs: the
/// deadline and the pivots spent so far against the pivot cap.  Solving
/// through the meter charges each LP's pivots to the running total, so a
/// cap spans all the LPs of the call and the `i64 → i128 → W256 → BigInt`
/// attempts inside each.
pub(crate) struct LpMeter {
    /// The call's budget with its time limit already resolved
    /// ([`Budget::start`]).
    budget: Budget,
    spent: usize,
}

impl LpMeter {
    /// Start the clock for one call: the budget's relative `time_limit`
    /// becomes an absolute deadline now (the earlier of the two if both
    /// are set; see [`Budget::start`]).
    pub(crate) fn start(budget: &Budget) -> Self {
        LpMeter {
            budget: budget.start(),
            spent: 0,
        }
    }

    /// Pivots charged so far.
    pub(crate) fn spent(&self) -> usize {
        self.spent
    }

    /// What is left for the next LP.
    pub(crate) fn remaining(&self) -> Budget {
        Budget {
            deadline: self.budget.deadline,
            time_limit: None,
            max_pivots: self.budget.max_pivots.map(|m| m.saturating_sub(self.spent)),
        }
    }

    /// Solve `lp` under the remaining budget and charge its pivots.
    pub(crate) fn solve(&mut self, lp: LpProblem) -> Result<LpSolution, Stop> {
        let report = lp.with_budget(self.remaining()).solve_report()?;
        self.spent += report.pivots;
        match report.budget_hit {
            Some(hit) => Err(Stop::Budget(hit)),
            None => Ok(report.solution),
        }
    }
}

/// A linear program in builder form.  See the [module docs](self) for
/// the problem shape and the meaning of the results.
///
/// Every variable defaults to `0 ≤ xⱼ < ∞`; change that with
/// [`bounds`](Self::bounds) or [`free`](Self::free).  Constraint rows are
/// kept in insertion order, which is also the order of
/// [`LpSolution::duals`] and [`LpSolution::farkas`].
///
/// # Examples
///
/// ```
/// use symplex::linprog::{LpProblem, LpStatus, q, qi};
///
/// // min x + y  s.t.  x + 2y ≥ 1,  3x + y ≥ 1,  x, y ≥ 0   →  (1/5, 2/5)
/// let sol = LpProblem::minimize(vec![qi(1), qi(1)])
///     .ge(vec![qi(1), qi(2)], qi(1))
///     .ge(vec![qi(3), qi(1)], qi(1))
///     .solve()
///     .unwrap();
/// assert_eq!(sol.status, LpStatus::Optimal);
/// assert_eq!(sol.x, vec![q(1, 5), q(2, 5)]);
/// assert_eq!(sol.objective, Some(q(3, 5)));
/// ```
#[derive(Clone, Debug)]
pub struct LpProblem {
    objective: Objective,
    c: Vec<Q>,
    constraints: Vec<Constraint>,
    bounds: Vec<Bounds<Q>>,
    /// First out-of-range variable index passed to `bounds`, reported by
    /// `solve`.
    bad_var: Option<usize>,
    budget: Budget,
}

impl LpProblem {
    fn new(objective: Objective, c: Vec<Q>) -> Self {
        let n = c.len();
        LpProblem {
            objective,
            c,
            constraints: Vec::new(),
            bounds: vec![Bounds::at_least(Q::zero()); n],
            bad_var: None,
            budget: Budget::default(),
        }
    }

    /// Start a minimisation of `cᵀx` over `x ∈ ℚ^{c.len()}`.
    pub fn minimize(c: Vec<Q>) -> Self {
        Self::new(Objective::Minimize, c)
    }

    /// Start a maximisation of `cᵀx` over `x ∈ ℚ^{c.len()}`.
    pub fn maximize(c: Vec<Q>) -> Self {
        Self::new(Objective::Maximize, c)
    }

    fn add(&mut self, row: Vec<Q>, rhs: Q, relation: Relation) {
        self.constraints.push(Constraint { row, rhs, relation });
    }

    /// Add the equality constraint `row·x = rhs`.
    pub fn eq(mut self, row: Vec<Q>, rhs: Q) -> Self {
        self.add(row, rhs, Relation::Eq);
        self
    }

    /// Add the inequality constraint `row·x ≤ rhs`.
    pub fn le(mut self, row: Vec<Q>, rhs: Q) -> Self {
        self.add(row, rhs, Relation::Le);
        self
    }

    /// Add the inequality constraint `row·x ≥ rhs`.
    pub fn ge(mut self, row: Vec<Q>, rhs: Q) -> Self {
        self.add(row, rhs, Relation::Ge);
        self
    }

    /// Set the bounds `lower ≤ x[var] ≤ upper` of one variable: `.bounds(j,
    /// Bounds::closed(q(1, 2), qi(3)))`, `.bounds(j, Bounds::at_most(qi(3)))`,
    /// ….  An absent side is unbounded; [`Bounds::free`] makes the variable
    /// free (see also [`free`](Self::free)).
    ///
    /// An out-of-range `var` is reported by [`solve`](Self::solve).
    pub fn bounds(mut self, var: usize, bounds: Bounds<Q>) -> Self {
        match self.bounds.get_mut(var) {
            Some(slot) => *slot = bounds,
            None => self.bad_var = self.bad_var.or(Some(var)),
        }
        self
    }

    /// Make `x[var]` free (`−∞ < x[var] < ∞`).
    pub fn free(self, var: usize) -> Self {
        self.bounds(var, Bounds::free())
    }

    /// Number of decision variables (the length of `c`).
    pub fn num_vars(&self) -> usize {
        self.c.len()
    }

    /// Number of constraint rows added so far.
    pub fn num_constraints(&self) -> usize {
        self.constraints.len()
    }

    /// Bound the solve by a [`Budget`] (deadline and/or pivot cap).  When
    /// it runs out, [`solve`](Self::solve) returns
    /// [`LpStatus::BudgetExhausted`].  The default budget is unlimited.
    #[must_use]
    pub fn with_budget(mut self, budget: Budget) -> Self {
        self.budget = budget;
        self
    }

    /// The budget the solve runs under (unlimited unless
    /// [`with_budget`](Self::with_budget) was called).
    pub fn budget(&self) -> &Budget {
        &self.budget
    }

    /// Solve the program exactly.
    ///
    /// The returned [`LpSolution`] reports [`LpStatus::Optimal`],
    /// [`LpStatus::Infeasible`] (with a Farkas certificate),
    /// [`LpStatus::Unbounded`] or — only under a [`Budget`] —
    /// [`LpStatus::BudgetExhausted`]; none of these is an error.
    ///
    /// # Errors
    ///
    /// * [`SymplexError::InvalidArgument`] for malformed input: no
    ///   variables, a constraint row whose length differs from `c.len()`,
    ///   or a bound set for a variable index `≥ c.len()`.
    /// * [`SymplexError::ComputationFailed`] if the pivot cap
    ///   (`10 000 + 50·(m + n)`) is exceeded — not expected in practice,
    ///   since Bland's rule rules out cycling.
    pub fn solve(&self) -> Result<LpSolution, SymplexError> {
        self.solve_report().map(|r| r.solution)
    }

    /// [`solve`](Self::solve) with the bookkeeping the certificate provers
    /// need to share one budget across many LPs: the pivots spent and, on
    /// [`LpStatus::BudgetExhausted`], which limit ran out.
    pub(crate) fn solve_report(&self) -> Result<SolveReport, SymplexError> {
        self.validate()?;
        solve_lp(self)
    }

    fn validate(&self) -> Result<(), SymplexError> {
        let n = self.c.len();
        if n == 0 {
            return Err(invalid(
                "linprog",
                "the objective must have at least one variable",
            ));
        }
        if let Some(bad) = self.bad_var {
            return Err(invalid(
                "linprog",
                format!("bounds were set for variable {bad} but there are only {n} variables"),
            ));
        }
        for (i, con) in self.constraints.iter().enumerate() {
            if con.row.len() != n {
                return Err(invalid(
                    "linprog",
                    format!(
                        "constraint {i} has {} coefficients but there are {n} variables",
                        con.row.len()
                    ),
                ));
            }
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Solution
// ═══════════════════════════════════════════════════════════════════════════

/// Outcome of a linear program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LpStatus {
    /// A finite optimum was found; `x`, `objective` and `duals` are set.
    Optimal,
    /// No point satisfies all constraints and bounds; `farkas` holds a
    /// certificate (see the [module docs](self)).
    Infeasible,
    /// The objective can be improved without limit over the feasible set.
    Unbounded,
    /// The [`Budget`] set with [`LpProblem::with_budget`] ran out before
    /// the solve finished: nothing is known about the problem, and `x`,
    /// `objective`, `duals` and `farkas` are empty.
    BudgetExhausted,
}

/// [`LpSolution`] plus the bookkeeping shared-budget callers need.
pub(crate) struct SolveReport {
    pub(crate) solution: LpSolution,
    /// Pivots begun, across both phases and every cell-type attempt.
    pub(crate) pivots: usize,
    /// Which limit ran out, when `solution.status` is `BudgetExhausted`.
    pub(crate) budget_hit: Option<BudgetHit>,
}

/// Result of [`LpProblem::solve`] / [`linprog`].
///
/// See the [module docs](self) for the exact meaning and sign conventions
/// of `duals` and `farkas`.
#[derive(Clone, Debug)]
pub struct LpSolution {
    /// Optimal / infeasible / unbounded.
    pub status: LpStatus,
    /// The optimal point (empty unless `status == Optimal`).
    pub x: Vec<Q>,
    /// The optimal objective value `cᵀx` (`None` unless `Optimal`).
    pub objective: Option<Q>,
    /// Shadow prices, one per constraint in insertion order (empty unless
    /// `Optimal`): `yᵢ = ∂(optimal objective)/∂bᵢ`.
    pub duals: Vec<Q>,
    /// Farkas infeasibility certificate, one entry per constraint (`Some`
    /// only when `Infeasible` and the constraints — not the bounds alone —
    /// are contradictory).
    pub farkas: Option<Vec<Q>>,
}

impl LpSolution {
    /// `true` when `status == LpStatus::Optimal`.
    pub fn is_optimal(&self) -> bool {
        self.status == LpStatus::Optimal
    }

    /// The optimal point as exact rational expressions in `ctx`.
    pub fn x_ex(&self, ctx: &Context) -> Vec<Ex> {
        self.x.iter().map(|v| ctx.from_ratio(v.clone())).collect()
    }

    /// The shadow prices as exact rational expressions in `ctx`.
    pub fn duals_ex(&self, ctx: &Context) -> Vec<Ex> {
        self.duals
            .iter()
            .map(|v| ctx.from_ratio(v.clone()))
            .collect()
    }

    fn infeasible(farkas: Option<Vec<Q>>) -> Self {
        LpSolution {
            status: LpStatus::Infeasible,
            x: Vec::new(),
            objective: None,
            duals: Vec::new(),
            farkas,
        }
    }

    fn unbounded() -> Self {
        LpSolution {
            status: LpStatus::Unbounded,
            x: Vec::new(),
            objective: None,
            duals: Vec::new(),
            farkas: None,
        }
    }

    fn budget_exhausted() -> Self {
        LpSolution {
            status: LpStatus::BudgetExhausted,
            x: Vec::new(),
            objective: None,
            duals: Vec::new(),
            farkas: None,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Standard-form conversion
// ═══════════════════════════════════════════════════════════════════════════

/// How an original variable is represented in the standard form.
#[derive(Clone, Debug)]
enum VarMap {
    /// `x = lo + z[col]`, `z ≥ 0`.
    Shifted { col: usize, lo: Q },
    /// `x = hi − z[col]`, `z ≥ 0` (only an upper bound).
    Mirrored { col: usize, hi: Q },
    /// `x = z[pos] − z[neg]` (free variable).
    Split { pos: usize, neg: usize },
}

/// `min c·z  s.t.  A z = b, z ≥ 0, b ≥ 0`, plus the bookkeeping needed to
/// map results back to the caller's variables and constraints.
struct Standard {
    a: Vec<Vec<Q>>,
    b: Vec<Q>,
    c: Vec<Q>,
    /// Objective constant (min-form) introduced by variable shifts.
    constant: Q,
    var_map: Vec<VarMap>,
    /// `+1` / `−1`: whether original constraint `i` was negated to make
    /// its right-hand side non-negative.
    row_sign: Vec<Q>,
    /// Number of original constraints (leading rows of `a`).
    m_orig: usize,
}

/// Outcome of standardisation: either a standard-form problem or an
/// immediate verdict from the bounds.
enum Standardized {
    Ready(Standard),
    BoundsInfeasible,
}

fn standardize(p: &LpProblem) -> Standardized {
    let n = p.c.len();
    let m = p.constraints.len();
    let zero = Q::zero();
    let one = Q::one();

    // Variables.
    let mut var_map = Vec::with_capacity(n);
    let mut ncols = 0usize;
    // Rows for two-sided bounds `z ≤ hi − lo`, added after the constraints.
    let mut bound_rows: Vec<(usize, Q)> = Vec::new();
    for bounds in &p.bounds {
        match (&bounds.lower, &bounds.upper) {
            (Some(lo), Some(hi)) => {
                if hi < lo {
                    return Standardized::BoundsInfeasible;
                }
                var_map.push(VarMap::Shifted {
                    col: ncols,
                    lo: lo.clone(),
                });
                bound_rows.push((ncols, hi - lo));
                ncols += 1;
            }
            (Some(lo), None) => {
                var_map.push(VarMap::Shifted {
                    col: ncols,
                    lo: lo.clone(),
                });
                ncols += 1;
            }
            (None, Some(hi)) => {
                var_map.push(VarMap::Mirrored {
                    col: ncols,
                    hi: hi.clone(),
                });
                ncols += 1;
            }
            (None, None) => {
                var_map.push(VarMap::Split {
                    pos: ncols,
                    neg: ncols + 1,
                });
                ncols += 2;
            }
        }
    }
    let n_slack: usize = p
        .constraints
        .iter()
        .filter(|c| c.relation != Relation::Eq)
        .count();
    let total_cols = ncols + n_slack + bound_rows.len();
    let total_rows = m + bound_rows.len();

    // Objective in min form: (sign) · c, expressed in z.
    let sign = match p.objective {
        Objective::Minimize => one.clone(),
        Objective::Maximize => -one.clone(),
    };
    let mut c = vec![zero.clone(); total_cols];
    let mut constant = zero.clone();
    for (j, vm) in var_map.iter().enumerate() {
        let cj = &sign * &p.c[j];
        match vm {
            VarMap::Shifted { col, lo } => {
                c[*col] = cj.clone();
                if !lo.is_zero() {
                    constant += &cj * lo;
                }
            }
            VarMap::Mirrored { col, hi } => {
                c[*col] = -cj.clone();
                constant += &cj * hi;
            }
            VarMap::Split { pos, neg } => {
                c[*pos] = cj.clone();
                c[*neg] = -cj;
            }
        }
    }

    // Constraint rows.
    let mut a = vec![vec![zero.clone(); total_cols]; total_rows];
    let mut b = vec![zero.clone(); total_rows];
    let mut row_sign = vec![one.clone(); m];
    let mut next_slack = ncols;
    for (i, con) in p.constraints.iter().enumerate() {
        let mut rhs = con.rhs.clone();
        for (j, vm) in var_map.iter().enumerate() {
            let aij = &con.row[j];
            if aij.is_zero() {
                continue;
            }
            match vm {
                VarMap::Shifted { col, lo } => {
                    a[i][*col] = aij.clone();
                    // The default bound `lo = 0` is by far the most common;
                    // a rational product and difference per entry would
                    // otherwise dominate the conversion.
                    if !lo.is_zero() {
                        rhs -= aij * lo;
                    }
                }
                VarMap::Mirrored { col, hi } => {
                    a[i][*col] = -aij.clone();
                    rhs -= aij * hi;
                }
                VarMap::Split { pos, neg } => {
                    a[i][*pos] = aij.clone();
                    a[i][*neg] = -aij.clone();
                }
            }
        }
        match con.relation {
            Relation::Le => {
                a[i][next_slack] = one.clone();
                next_slack += 1;
            }
            Relation::Ge => {
                a[i][next_slack] = -one.clone();
                next_slack += 1;
            }
            Relation::Eq => {}
        }
        if rhs.is_negative() {
            row_sign[i] = -one.clone();
            for v in a[i].iter_mut() {
                if !v.is_zero() {
                    *v = -std::mem::take(v);
                }
            }
            rhs = -rhs;
        }
        b[i] = rhs;
    }
    for (k, (col, width)) in bound_rows.into_iter().enumerate() {
        let r = m + k;
        a[r][col] = one.clone();
        a[r][next_slack] = one.clone();
        next_slack += 1;
        b[r] = width;
    }
    debug_assert_eq!(next_slack, total_cols);

    Standardized::Ready(Standard {
        a,
        b,
        c,
        constant,
        var_map,
        row_sign,
        m_orig: m,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Simplex tableau
// ═══════════════════════════════════════════════════════════════════════════

/// Dense tableau for `min c·z s.t. A z = b, z ≥ 0` with one artificial
/// column per row.  Column layout: `[z columns | artificials | rhs]`.
///
/// **Integer pivoting.**  Every constraint row `i` is first multiplied by
/// `row_scale[i]`, the least common multiple of its denominators, so the
/// initial tableau is integral.  Pivots then follow Bareiss's fraction-free
/// Gauss–Jordan rule: all entries stay integers and share one common
/// denominator `d` (the current pivot, `±det B` of the scaled system).
/// The rational tableau entry is `rows[i][j] / d`; the reduced cost is
/// `obj[j] / (d · obj_scale)` where `obj_scale` clears the denominators of
/// the current cost vector.  Sign tests and the ratio test are integer
/// comparisons (cross-multiplied), so no gcd runs inside the pivot loop.
///
/// Scaling row `i` by `sᵢ` turns its artificial `aᵢ` into `a'ᵢ = sᵢ·aᵢ`.
/// Phase 1 therefore gives `a'ᵢ` the cost `1/sᵢ`, so that its objective is
/// still `Σ aᵢ`, every reduced cost is the same rational a `Ratio` tableau
/// of the unscaled system would hold, and the pivot sequence (Dantzig /
/// Bland choices, ratio tests, tie-breaks) is identical to it.
///
/// `obj` holds `d·obj_scale` times the reduced costs
/// `cⱼ − c_B B⁻¹ Aⱼ` and, in its last entry, `−(current objective value)`
/// at the same scale.  Because the artificial columns start as the
/// identity, at any time they hold `d·B⁻¹`, and the duals of the scaled
/// system are `y'ᵢ = c_{a'ᵢ} − obj[a'ᵢ]/(d·obj_scale)`; the duals of the
/// caller's rows are `yᵢ = sᵢ · y'ᵢ`.
///
/// **Hybrid arithmetic.**  The tableau is generic over its cell type
/// ([`Cell`]): it first runs on `i64` cells with `i128` intermediates, and
/// the first value that does not fit aborts that attempt, after which the
/// whole problem is solved again on the next wider cells (`i128`, then
/// 256-bit limbs, then `BigInt`).  Every decision (entering column, ratio
/// test, tie-break) is a sign test or an exact comparison of products, so
/// all runs take the same pivot path and produce the same answer; the
/// fixed-width runs merely never touch the heap.  The exact divisions of
/// the fraction-free update are done by multiplication with the inverse of
/// the pivot's odd part modulo the word size (Jebelean) — verified by one
/// multiplication for `i64` and the 256-bit cells, and by the quotient's
/// high-half bound for `i128` — so there is no long division anywhere in
/// the fixed-width pivot loops; the `BigInt` cells divide and check the
/// remainder.
/// Certificate LPs (small polynomial coefficients, 16–20 rows) peak
/// between 60 and 200 bits, so they finish on `i128` or the 256-bit cells.
struct Tableau<'a, T: Cell> {
    /// Row-major `m × width` integer tableau.
    rows: Vec<T>,
    width: usize,
    obj: Vec<T>,
    /// Common denominator of `rows` (sign may be negative after an
    /// artificial is driven out on a negative pivot).
    d: T,
    /// Positive integer clearing the denominators of the installed costs.
    obj_scale: T,
    /// Positive integer each constraint row was multiplied by.
    row_scale: Vec<T>,
    basis: Vec<usize>,
    m: usize,
    /// Number of genuine (non-artificial) columns.
    n: usize,
    pivots_done: usize,
    max_pivots: usize,
    /// The caller's [`Budget`], checked at every pivot.
    budget: &'a Budget,
    /// Pivots *begun* by this and every earlier cell-type attempt of the
    /// same solve — what `budget.max_pivots` is measured against.  A pivot
    /// that overflows its cell type half-way is charged too: the work was
    /// done, and the wider retry will do it again.
    spent: &'a mut usize,
    /// Number of consecutive degenerate pivots (zero-length steps) so far.
    /// Dantzig's rule can only cycle through degenerate pivots, so after
    /// [`STALL_LIMIT`] of them in a row the entering rule switches to
    /// Bland's (which cannot cycle) until the next improving pivot resets
    /// the count.  Permanent Bland after the *first* degenerate pivot —
    /// the previous policy — made the highly degenerate certificate LPs
    /// (many zero right-hand sides) take tens of thousands of pivots.
    stall: usize,
}

/// Consecutive degenerate pivots tolerated under Dantzig's rule before
/// Bland's rule takes over for the rest of the stall.
const STALL_LIMIT: usize = 12;

enum Step {
    Optimal,
    Unbounded,
}

/// Why a tableau run stopped early: a value did not fit the cell type
/// (retry with a wider one), the caller's budget ran out, or the pivot
/// cap was hit.
enum Halt {
    Overflow,
    Budget(BudgetHit),
    Error(SymplexError),
}

impl From<SymplexError> for Halt {
    fn from(e: SymplexError) -> Self {
        Halt::Error(e)
    }
}

/// Least common multiple of the denominators of `values`.  Unit
/// denominators (the common case: integer data) are skipped, so a row of
/// integers costs no big-integer arithmetic at all.
fn denominator_lcm<'a>(values: impl Iterator<Item = &'a Q>) -> BigInt {
    values
        .map(Ratio::denom)
        .filter(|d| !d.is_one())
        .fold(<BigInt as One>::one(), |l, d| l.lcm(d))
}

impl<'a, T: Cell> Tableau<'a, T> {
    /// Build the initial tableau; `None` if an entry does not fit `T`.
    fn new(std: &Standard, budget: &'a Budget, spent: &'a mut usize) -> Option<Self> {
        let m = std.a.len();
        let n = std.c.len();
        let width = n + m + 1;
        let mut rows = Vec::with_capacity(m * width);
        let mut row_scale = Vec::with_capacity(m);
        for i in 0..m {
            let s = denominator_lcm(std.a[i].iter().chain(std::iter::once(&std.b[i])));
            let scaled = |q: &Q| {
                if Zero::is_zero(q) {
                    Some(T::cell_zero())
                } else {
                    T::from_ratio_scaled(q, &s)
                }
            };
            for q in &std.a[i] {
                rows.push(scaled(q)?);
            }
            for k in 0..m {
                rows.push(if k == i {
                    T::cell_one()
                } else {
                    T::cell_zero()
                });
            }
            rows.push(scaled(&std.b[i])?);
            row_scale.push(T::from_big(&s)?);
        }
        let basis: Vec<usize> = (0..m).map(|i| n + i).collect();
        Some(Tableau {
            rows,
            width,
            obj: vec![T::cell_zero(); width],
            d: T::cell_one(),
            obj_scale: T::cell_one(),
            row_scale,
            basis,
            m,
            n,
            pivots_done: 0,
            max_pivots: 10_000 + 50 * (m + n),
            budget,
            spent,
            stall: 0,
        })
    }

    /// `Err(Halt::Budget(Deadline))` once the budget's deadline has passed.
    #[inline]
    fn check_deadline(&self) -> Result<(), Halt> {
        if self.budget.deadline_passed() {
            return Err(Halt::Budget(BudgetHit::Deadline));
        }
        Ok(())
    }

    /// `Err(Halt::Budget(MaxPivots))` if another pivot would exceed the
    /// budget's pivot cap.
    #[inline]
    fn check_pivot_budget(&self) -> Result<(), Halt> {
        if let Some(cap) = self.budget.max_pivots
            && *self.spent >= cap
        {
            return Err(Halt::Budget(BudgetHit::MaxPivots));
        }
        Ok(())
    }

    #[inline]
    fn rhs_col(&self) -> usize {
        self.n + self.m
    }

    #[inline]
    fn row(&self, i: usize) -> &[T] {
        &self.rows[i * self.width..(i + 1) * self.width]
    }

    /// Sign of the rational value `z / d`.
    #[inline]
    fn sign_of(&self, z: &T) -> std::cmp::Ordering {
        use std::cmp::Ordering::*;
        match (z.signum(), self.d.signum()) {
            (Equal, _) => Equal,
            (a, b) if a == b => Greater,
            _ => Less,
        }
    }

    /// Rational value of a tableau entry.
    #[inline]
    fn value(&self, z: &T) -> Q {
        Ratio::new(z.to_big(), self.d.to_big())
    }

    /// Install a new objective (`costs` over all `n + m` columns) and price
    /// out the current basis so that `obj` holds true reduced costs (times
    /// `d · obj_scale`).  `None` on overflow.
    fn set_objective(&mut self, costs: &[Q]) -> Option<()> {
        let width = self.width;
        let s = denominator_lcm(costs.iter());
        let int_cost = |q: &Q| T::from_ratio_scaled(q, &s);
        let mut obj = vec![T::cell_zero(); width];
        for (o, c) in obj.iter_mut().zip(costs) {
            if !Zero::is_zero(c) {
                *o = int_cost(c)?.mul(&self.d)?;
            }
        }
        for (i, &k) in self.basis.iter().enumerate() {
            let f = int_cost(&costs[k])?;
            if f.is_zero() {
                continue;
            }
            let row = self.row(i);
            for (o, r) in obj.iter_mut().zip(row) {
                if !r.is_zero() {
                    *o = o.sub_mul(&f, r)?;
                }
            }
        }
        self.obj = obj;
        self.obj_scale = T::from_big(&s)?;
        Some(())
    }

    /// Fraction-free pivot on `(r, s)`: every other row `i` (and the
    /// objective row) becomes `(row_i·p − row_i[s]·row_r) / d`, the pivot
    /// row is unchanged, and `d ← p`.  `None` on overflow (the tableau is
    /// then unusable; the caller restarts with wider cells).  The row
    /// update is the kernel's [`pivot_row_update`], shared with the exact
    /// matrices' Gauss–Jordan and Bareiss eliminations.
    fn pivot(&mut self, r: usize, s: usize) -> Option<()> {
        *self.spent += 1;
        let w = self.width;
        let prow: Vec<T> = self.rows[r * w..(r + 1) * w]
            .iter_mut()
            .map(|v| std::mem::replace(v, T::cell_zero()))
            .collect();
        let p = prow[s].clone();
        let d = T::divisor(&std::mem::replace(&mut self.d, p.clone()));
        for i in 0..self.m {
            if i == r {
                continue;
            }
            pivot_row_update(&mut self.rows[i * w..(i + 1) * w], &prow, s, &p, &d)?;
        }
        pivot_row_update(&mut self.obj, &prow, s, &p, &d)?;
        self.rows[r * w..(r + 1) * w]
            .iter_mut()
            .zip(prow)
            .for_each(|(slot, v)| *slot = v);
        self.basis[r] = s;
        self.pivots_done += 1;
        Some(())
    }

    /// Run the simplex method on the current objective, considering only
    /// columns `< limit` as candidates to enter the basis.
    fn run(&mut self, limit: usize) -> Result<Step, Halt> {
        use std::cmp::Ordering;
        loop {
            self.check_deadline()?;
            if self.pivots_done >= self.max_pivots {
                return Err(failed(
                    "linprog",
                    format!(
                        "pivot cap of {} exceeded; the problem is degenerate beyond \
                         what the solver handles",
                        self.max_pivots
                    ),
                )
                .into());
            }
            // Entering column: Dantzig (most negative reduced cost, lowest
            // index on ties) or Bland (lowest index with negative cost).
            // Reduced costs share the positive factor |d|·obj_scale, so the
            // integer entries compare like the rationals once oriented by
            // the sign of d.  The scaled artificial a'ᵢ = sᵢ·aᵢ has reduced
            // cost rc(aᵢ)/sᵢ; multiplying by sᵢ compares the reduced cost of
            // the *unscaled* artificial, so Dantzig's choice is exactly the
            // one a `Ratio` tableau of the caller's system would make.
            let neg = self.d.is_negative();
            let oriented = |z: &T, j: usize| -> Option<T> {
                let v = if neg { z.neg()? } else { z.clone() };
                if j >= self.n {
                    v.mul(&self.row_scale[j - self.n])
                } else {
                    Some(v)
                }
            };
            let mut entering: Option<(usize, T)> = None;
            for j in 0..limit {
                if self.sign_of(&self.obj[j]) != Ordering::Less {
                    continue;
                }
                let v = oriented(&self.obj[j], j).ok_or(Halt::Overflow)?;
                match &entering {
                    Some((_, best)) if v >= *best => {}
                    _ => entering = Some((j, v)),
                }
                if self.stall >= STALL_LIMIT {
                    break;
                }
            }
            let Some((s, _)) = entering else {
                return Ok(Step::Optimal);
            };
            // Leaving row: minimum ratio rhsᵢ/aᵢₛ (d cancels), ties broken by
            // smallest basic index.  Every eligible aᵢₛ has the sign of d, so
            // rhsᵢ/aᵢₛ < rhsₗ/aₗₛ  ⇔  rhsᵢ·aₗₛ < rhsₗ·aᵢₛ  (cross-multiplication
            // by a positive product; no gcd needed).
            let rhs = self.rhs_col();
            let mut leaving: Option<usize> = None;
            for i in 0..self.m {
                let a = &self.row(i)[s];
                if self.sign_of(a) != Ordering::Greater {
                    continue;
                }
                let better = match leaving {
                    None => true,
                    Some(l) => {
                        let (ri, rl) = (&self.row(i)[rhs], &self.row(l)[rhs]);
                        let al = &self.row(l)[s];
                        match T::cmp_products(ri, al, rl, a) {
                            Ordering::Less => true,
                            Ordering::Equal => self.basis[i] < self.basis[l],
                            Ordering::Greater => false,
                        }
                    }
                };
                if better {
                    leaving = Some(i);
                }
            }
            let Some(r) = leaving else {
                return Ok(Step::Unbounded);
            };
            self.check_pivot_budget()?;
            if self.row(r)[rhs].is_zero() {
                self.stall += 1;
            } else {
                self.stall = 0;
            }
            self.pivot(r, s).ok_or(Halt::Overflow)?;
        }
    }

    /// Value of the current basic solution over the genuine columns.
    fn solution(&self) -> Vec<Q> {
        let rhs = self.rhs_col();
        let mut z = vec![Q::zero(); self.n];
        for (i, &k) in self.basis.iter().enumerate() {
            if k < self.n {
                z[k] = self.value(&self.row(i)[rhs]);
            }
        }
        z
    }

    /// `−(current objective value)` — the last entry of the objective row.
    fn neg_objective(&self) -> Q {
        Ratio::new(
            self.obj[self.rhs_col()].to_big(),
            self.d.to_big() * self.obj_scale.to_big(),
        )
    }

    /// Pivot artificial variables out of the basis wherever possible.
    /// Rows whose genuine part is entirely zero are redundant; their
    /// artificial stays basic at level zero and never moves again.  These
    /// pivots are charged to the budget like any other.
    fn drive_out_artificials(&mut self) -> Result<(), Halt> {
        for r in 0..self.m {
            if self.basis[r] < self.n {
                continue;
            }
            if let Some(s) = (0..self.n).find(|&j| !self.row(r)[j].is_zero()) {
                self.check_deadline()?;
                self.check_pivot_budget()?;
                self.pivot(r, s).ok_or(Halt::Overflow)?;
            }
        }
        Ok(())
    }

    /// Phase-1 cost vector over all `n + m` columns: `0` for genuine
    /// columns and `1/sᵢ` for the (scaled) artificial of row `i`, so that
    /// the phase-1 objective is `Σ aᵢ` of the unscaled system.
    fn phase1_costs(&self) -> Vec<Q> {
        let mut costs = vec![Q::zero(); self.n + self.m];
        for (c, s) in costs[self.n..].iter_mut().zip(&self.row_scale) {
            *c = Ratio::new(<BigInt as One>::one(), s.to_big());
        }
        costs
    }

    /// Current duals of the caller's (unscaled) rows: `yᵢ = sᵢ · (c_{a'ᵢ} −
    /// reduced cost of a'ᵢ)`, where the artificial's cost is `1/sᵢ` in
    /// phase 1 and `0` in phase 2.
    fn duals(&self, phase1: bool) -> Vec<Q> {
        let denom = self.d.to_big() * self.obj_scale.to_big();
        (0..self.m)
            .map(|i| {
                let s = self.row_scale[i].to_big();
                let reduced = Ratio::new(self.obj[self.n + i].to_big(), denom.clone());
                let scaled_reduced = reduced * Ratio::from_integer(s);
                if phase1 {
                    Q::one() - scaled_reduced
                } else {
                    -scaled_reduced
                }
            })
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Driver
// ═══════════════════════════════════════════════════════════════════════════

fn solve_lp(p: &LpProblem) -> Result<SolveReport, SymplexError> {
    let prof = tracing::enabled!(target: "symplex::linprog::prof", tracing::Level::DEBUG);
    let started = Instant::now();
    let sf = match standardize(p) {
        Standardized::Ready(s) => s,
        Standardized::BoundsInfeasible => {
            return Ok(SolveReport {
                solution: LpSolution::infeasible(None),
                pivots: 0,
                budget_hit: None,
            });
        }
    };
    // The clock of a relative `time_limit` starts here, once for the whole
    // solve.
    let budget = p.budget.start();
    // One pivot counter for every attempt, so that the budget's pivot cap
    // is a cap on the whole solve.
    let mut spent = 0usize;
    // `Some` when the attempt settled the problem (or failed for good);
    // `None` when it overflowed and the next cell type should try.
    let settle =
        |r: Result<LpSolution, Halt>, spent: usize| -> Option<Result<SolveReport, SymplexError>> {
            match r {
                Ok(solution) => Some(Ok(SolveReport {
                    solution,
                    pivots: spent,
                    budget_hit: None,
                })),
                Err(Halt::Budget(hit)) => Some(Ok(SolveReport {
                    solution: LpSolution::budget_exhausted(),
                    pivots: spent,
                    budget_hit: Some(hit),
                })),
                Err(Halt::Error(e)) => Some(Err(e)),
                Err(Halt::Overflow) => None,
            }
        };
    // Per-attempt timing, for profiling the certificate provers
    // (`RUST_LOG=symplex::linprog::prof=debug`).
    let attempt = |cell: &'static str, r: &Result<LpSolution, Halt>, spent: usize| {
        if prof {
            tracing::debug!(
                target: "symplex::linprog::prof",
                cell,
                rows = sf.a.len(),
                cols = sf.c.len(),
                overflowed = matches!(r, Err(Halt::Overflow)),
                spent,
                micros = started.elapsed().as_micros() as u64,
                "linprog attempt"
            );
        }
    };
    // Fixed-width cells first (i64, then i128 and 256-bit limbs, each with
    // exact double-width intermediates); the same algorithm on BigInt
    // cells if a value outgrows them all.  Fraction-free entries are minors
    // of the scaled system: a 16-row certificate LP typically peaks around
    // 70–120 bits, a 20-row one around 130–190.
    let r = solve_standard::<i64>(p, &sf, &budget, &mut spent);
    attempt("i64", &r, spent);
    if let Some(done) = settle(r, spent) {
        return done;
    }
    let r = solve_standard::<i128>(p, &sf, &budget, &mut spent);
    attempt("i128", &r, spent);
    if let Some(done) = settle(r, spent) {
        return done;
    }
    let r = solve_standard::<W256>(p, &sf, &budget, &mut spent);
    attempt("W256", &r, spent);
    if let Some(done) = settle(r, spent) {
        return done;
    }
    tracing::debug!(target: "symplex::linprog", rows = sf.a.len(), cols = sf.c.len(), "256-bit tableau overflowed; solving on BigInt");
    let r = solve_standard::<BigInt>(p, &sf, &budget, &mut spent);
    attempt("BigInt", &r, spent);
    settle(r, spent).unwrap_or_else(|| {
        Err(failed(
            "linprog",
            "internal: fraction-free update was not exact on BigInt cells",
        ))
    })
}

/// One cell-type attempt under `budget` (already [`started`](Budget::start)).
fn solve_standard<T: Cell>(
    p: &LpProblem,
    sf: &Standard,
    budget: &Budget,
    spent: &mut usize,
) -> Result<LpSolution, Halt> {
    let mut t = Tableau::<T>::new(sf, budget, spent).ok_or(Halt::Overflow)?;
    let m = t.m;
    let n = t.n;

    // Phase 1: minimise the sum of the artificials.
    let phase1 = t.phase1_costs();
    t.set_objective(&phase1).ok_or(Halt::Overflow)?;
    match t.run(n + m)? {
        Step::Optimal => {}
        Step::Unbounded => {
            // Cannot happen: the phase-1 objective is bounded below by 0.
            return Err(failed("linprog", "phase 1 reported unbounded").into());
        }
    }
    let infeasibility = -t.neg_objective();
    if infeasibility.is_positive() {
        // Farkas certificate from the phase-1 duals, mapped back to the
        // caller's rows (bound rows are absorbed into the box infimum).
        let y_std = t.duals(true);
        let farkas: Vec<Q> = (0..sf.m_orig)
            .map(|i| -(&sf.row_sign[i] * &y_std[i]))
            .collect();
        return Ok(LpSolution::infeasible(Some(farkas)));
    }

    // Phase 2.
    t.drive_out_artificials()?;
    let mut phase2 = vec![Q::zero(); n + m];
    phase2[..n].clone_from_slice(&sf.c);
    t.set_objective(&phase2).ok_or(Halt::Overflow)?;
    match t.run(n)? {
        Step::Optimal => {}
        Step::Unbounded => return Ok(LpSolution::unbounded()),
    }

    if tracing::enabled!(target: "symplex::linprog::growth", tracing::Level::TRACE) {
        let bits = t
            .rows
            .iter()
            .chain(t.obj.iter())
            .chain(std::iter::once(&t.d))
            .map(|c| c.to_big().bits())
            .max()
            .unwrap_or(0);
        tracing::trace!(target: "symplex::linprog::growth", rows = t.m, cols = t.n, pivots = t.pivots_done, max_bits = bits, "final tableau");
    }
    let z = t.solution();
    let x: Vec<Q> = sf
        .var_map
        .iter()
        .map(|vm| match vm {
            VarMap::Shifted { col, lo } => lo + &z[*col],
            VarMap::Mirrored { col, hi } => hi - &z[*col],
            VarMap::Split { pos, neg } => &z[*pos] - &z[*neg],
        })
        .collect();
    let objective: Q = p.c.iter().zip(x.iter()).map(|(c, v)| c * v).sum();
    // The tableau's objective (min form, without the shift constant) must
    // agree with the value recomputed from x.
    debug_assert_eq!(
        {
            let min_form: Q =
                sf.c.iter().zip(z.iter()).map(|(c, v)| c * v).sum::<Q>() + &sf.constant;
            match p.objective {
                Objective::Minimize => min_form,
                Objective::Maximize => -min_form,
            }
        },
        objective
    );

    let y_std = t.duals(false);
    let dir = match p.objective {
        Objective::Minimize => Q::one(),
        Objective::Maximize => -Q::one(),
    };
    let duals: Vec<Q> = (0..sf.m_orig)
        .map(|i| &dir * &(&sf.row_sign[i] * &y_std[i]))
        .collect();

    Ok(LpSolution {
        status: LpStatus::Optimal,
        x,
        objective: Some(objective),
        duals,
        farkas: None,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Convenience wrappers
// ═══════════════════════════════════════════════════════════════════════════

/// SciPy-shaped entry point: minimise `cᵀx` subject to `A_ub x ≤ b_ub`,
/// `A_eq x = b_eq` and `bounds[j].lower ≤ xⱼ ≤ bounds[j].upper`.
///
/// `bounds` may be empty (every variable then defaults to `0 ≤ xⱼ < ∞`) or
/// have one [`Bounds`] per variable, an absent side meaning unbounded on
/// that side.  Constraints are numbered with the `≤` rows first, then the
/// `=` rows — that is the order of [`LpSolution::duals`] / [`farkas`](LpSolution::farkas).
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for malformed input (empty `c`, a row
/// of the wrong length, `b` of the wrong length, `bounds` of the wrong
/// length).
///
/// # Examples
///
/// ```
/// use symplex::linprog::{linprog, LpStatus, q, qi};
///
/// // min −x − y   s.t.  x + 2y ≤ 4,  3x + y ≤ 6,  x, y ≥ 0
/// let sol = linprog(
///     &[qi(-1), qi(-1)],
///     &[vec![qi(1), qi(2)], vec![qi(3), qi(1)]],
///     &[qi(4), qi(6)],
///     &[],
///     &[],
///     &[],
/// )
/// .unwrap();
/// assert_eq!(sol.status, LpStatus::Optimal);
/// assert_eq!(sol.x, vec![q(8, 5), q(6, 5)]);
/// assert_eq!(sol.objective, Some(q(-14, 5)));
/// ```
pub fn linprog(
    c: &[Q],
    a_ub: &[Vec<Q>],
    b_ub: &[Q],
    a_eq: &[Vec<Q>],
    b_eq: &[Q],
    bounds: &[Bounds<Q>],
) -> Result<LpSolution, SymplexError> {
    if a_ub.len() != b_ub.len() {
        return Err(invalid(
            "linprog",
            format!(
                "A_ub has {} rows but b_ub has {} entries",
                a_ub.len(),
                b_ub.len()
            ),
        ));
    }
    if a_eq.len() != b_eq.len() {
        return Err(invalid(
            "linprog",
            format!(
                "A_eq has {} rows but b_eq has {} entries",
                a_eq.len(),
                b_eq.len()
            ),
        ));
    }
    if !bounds.is_empty() && bounds.len() != c.len() {
        return Err(invalid(
            "linprog",
            format!(
                "bounds has {} entries but there are {} variables",
                bounds.len(),
                c.len()
            ),
        ));
    }
    let mut p = LpProblem::minimize(c.to_vec());
    for (row, rhs) in a_ub.iter().zip(b_ub) {
        p = p.le(row.clone(), rhs.clone());
    }
    for (row, rhs) in a_eq.iter().zip(b_eq) {
        p = p.eq(row.clone(), rhs.clone());
    }
    for (j, b) in bounds.iter().enumerate() {
        p = p.bounds(j, b.clone());
    }
    p.solve()
}

/// One-line human-readable summary: status, point, objective and duals.
///
/// ```
/// use symplex::linprog::{LpProblem, qi};
///
/// let sol = LpProblem::maximize(vec![qi(3), qi(2)])
///     .le(vec![qi(1), qi(1)], qi(4))
///     .le(vec![qi(1), qi(3)], qi(6))
///     .solve()
///     .unwrap();
/// assert_eq!(sol.to_string(), "Optimal: x = (4, 0), objective = 12, duals = (3, 0)");
/// ```
impl fmt::Display for LpSolution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let list = |v: &[Q]| {
            let parts: Vec<String> = v.iter().map(ToString::to_string).collect();
            format!("({})", parts.join(", "))
        };
        match self.status {
            LpStatus::Optimal => {
                write!(f, "Optimal: x = {}", list(&self.x))?;
                if let Some(obj) = &self.objective {
                    write!(f, ", objective = {obj}")?;
                }
                write!(f, ", duals = {}", list(&self.duals))
            }
            LpStatus::Infeasible => match &self.farkas {
                Some(y) => write!(f, "Infeasible: Farkas certificate y = {}", list(y)),
                None => write!(f, "Infeasible: contradictory bounds"),
            },
            LpStatus::Unbounded => write!(f, "Unbounded"),
            LpStatus::BudgetExhausted => write!(f, "Budget exhausted"),
        }
    }
}

/// Outcome of an exact feasibility question (see [`feasible_nonneg_certified`]
/// and [`nonneg_combination`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Feasibility {
    /// A witness `x ≥ 0` with `A x = b`.
    Feasible(Vec<Q>),
    /// No such `x`; `farkas` is a vector `y` with `Aᵀy ≥ 0` and `yᵀb < 0`
    /// (one entry per equation), which proves it.
    Infeasible {
        /// The Farkas certificate, `None` only if the solver could not
        /// produce one (never the case for pure equality systems).
        farkas: Option<Vec<Q>>,
    },
}

impl Feasibility {
    /// The witness, if feasible.
    pub fn witness(&self) -> Option<&[Q]> {
        match self {
            Feasibility::Feasible(x) => Some(x),
            Feasibility::Infeasible { .. } => None,
        }
    }

    /// `true` for [`Feasibility::Feasible`].
    pub fn is_feasible(&self) -> bool {
        matches!(self, Feasibility::Feasible(_))
    }
}

/// Exact feasibility of `A x = b, x ≥ 0`: `Some(x)` with a solution, or
/// `None` if none exists.  Use [`feasible_nonneg_certified`] when you also
/// want the Farkas certificate for the infeasible case.
///
/// This is the question behind many certificate searches (Farkas,
/// Positivstellensatz-style combinations, Carathéodory decompositions);
/// floating-point solvers answer it only up to a tolerance.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] if `a_eq` is empty, jagged, or its
/// length differs from `b_eq`.
///
/// # Examples
///
/// ```
/// use symplex::linprog::{feasible_nonneg, q, qi};
///
/// // x/3 + y/7 = 1,  x − y = 0  →  x = y = 21/10
/// let x = feasible_nonneg(
///     &[vec![q(1, 3), q(1, 7)], vec![qi(1), qi(-1)]],
///     &[qi(1), qi(0)],
/// )
/// .unwrap()
/// .unwrap();
/// assert_eq!(x, vec![q(21, 10), q(21, 10)]);
///
/// // x + y = −1 has no non-negative solution.
/// assert!(feasible_nonneg(&[vec![qi(1), qi(1)]], &[qi(-1)]).unwrap().is_none());
/// ```
pub fn feasible_nonneg(a_eq: &[Vec<Q>], b_eq: &[Q]) -> Result<Option<Vec<Q>>, SymplexError> {
    let Some(first) = a_eq.first() else {
        return Err(invalid(
            "feasible_nonneg",
            "need at least one equation to determine the number of variables",
        ));
    };
    let n = first.len();
    if n == 0 {
        return Err(invalid(
            "feasible_nonneg",
            "equations must have at least one variable",
        ));
    }
    let sol = linprog(&vec![Q::zero(); n], &[], &[], a_eq, b_eq, &[])?;
    Ok(match sol.status {
        LpStatus::Optimal => Some(sol.x),
        LpStatus::Infeasible => None,
        // A zero objective cannot be unbounded.
        LpStatus::Unbounded => None,
        // No budget is set here; an unanswered question must not read as "no".
        LpStatus::BudgetExhausted => {
            return Err(failed("feasible_nonneg", "budget exhausted"));
        }
    })
}

/// Like [`feasible_nonneg`], but an infeasible system comes back with its
/// Farkas certificate: a `y` (one entry per equation) with `Aᵀy ≥ 0`
/// component-wise and `yᵀb < 0`, so `x ≥ 0 ⇒ yᵀ(Ax) ≥ 0 > yᵀb`.
///
/// # Errors
///
/// As [`feasible_nonneg`].
///
/// # Examples
///
/// ```
/// use symplex::linprog::{feasible_nonneg_certified, Feasibility, qi};
///
/// // x + y = 1  and  x + y = 2 cannot both hold.
/// let a = [vec![qi(1), qi(1)], vec![qi(1), qi(1)]];
/// match feasible_nonneg_certified(&a, &[qi(1), qi(2)]).unwrap() {
///     Feasibility::Infeasible { farkas: Some(y) } => {
///         // Aᵀy = (y₀ + y₁, y₀ + y₁) ≥ 0  and  y₀ + 2·y₁ < 0
///         let g = &y[0] + &y[1];
///         assert!(g >= qi(0));
///         assert!(&y[0] + &y[1] * qi(2) < qi(0));
///     }
///     other => panic!("expected a certificate, got {other:?}"),
/// }
/// ```
pub fn feasible_nonneg_certified(a_eq: &[Vec<Q>], b_eq: &[Q]) -> Result<Feasibility, SymplexError> {
    let Some(first) = a_eq.first() else {
        return Err(invalid(
            "feasible_nonneg_certified",
            "need at least one equation to determine the number of variables",
        ));
    };
    let n = first.len();
    if n == 0 {
        return Err(invalid(
            "feasible_nonneg_certified",
            "equations must have at least one variable",
        ));
    }
    let sol = linprog(&vec![Q::zero(); n], &[], &[], a_eq, b_eq, &[])?;
    Ok(match sol.status {
        LpStatus::Optimal => Feasibility::Feasible(sol.x),
        LpStatus::Infeasible => Feasibility::Infeasible { farkas: sol.farkas },
        // A zero objective cannot be unbounded.
        LpStatus::Unbounded => Feasibility::Infeasible { farkas: None },
        // No budget is set here; an unanswered question must not read as "no".
        LpStatus::BudgetExhausted => {
            return Err(failed("feasible_nonneg_certified", "budget exhausted"));
        }
    })
}

/// Is `target` a non-negative combination `Σ λⱼ vⱼ` of the `vectors`?
///
/// The cone-membership form of [`feasible_nonneg_certified`]: the vectors
/// are the *columns* of `A` (each `vectors[j]` has one entry per coordinate
/// of `target`), and the answer is the coefficient vector `λ ≥ 0` or a
/// Farkas certificate `y` with `y·vⱼ ≥ 0` for every `j` and `y·target < 0`
/// (a hyperplane separating `target` from the cone).
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] if `vectors` is empty or some vector
/// has a different length than `target`.
///
/// # Examples
///
/// ```
/// use symplex::linprog::{nonneg_combination, Feasibility, q, qi};
///
/// // (1, 1) = ½·(2, 0) + 1·(0, 1)
/// let cone = [vec![qi(2), qi(0)], vec![qi(0), qi(1)]];
/// assert_eq!(
///     nonneg_combination(&cone, &[qi(1), qi(1)]).unwrap(),
///     Feasibility::Feasible(vec![q(1, 2), qi(1)])
/// );
/// // (−1, 1) is outside the cone spanned by (2, 0) and (0, 1).
/// assert!(!nonneg_combination(&cone, &[qi(-1), qi(1)]).unwrap().is_feasible());
/// ```
pub fn nonneg_combination(vectors: &[Vec<Q>], target: &[Q]) -> Result<Feasibility, SymplexError> {
    if vectors.is_empty() {
        return Err(invalid("nonneg_combination", "need at least one vector"));
    }
    let m = target.len();
    if m == 0 {
        return Err(invalid(
            "nonneg_combination",
            "target must have at least one coordinate",
        ));
    }
    if let Some((j, v)) = vectors.iter().enumerate().find(|(_, v)| v.len() != m) {
        return Err(invalid(
            "nonneg_combination",
            format!(
                "vector {j} has {} coordinates but the target has {m}",
                v.len()
            ),
        ));
    }
    // Rows of A are coordinates, columns are the vectors.
    let a_eq: Vec<Vec<Q>> = (0..m)
        .map(|i| vectors.iter().map(|v| v[i].clone()).collect())
        .collect();
    feasible_nonneg_certified(&a_eq, target).map_err(|e| match e {
        SymplexError::InvalidArgument { reason, .. } => invalid("nonneg_combination", reason),
        other => other,
    })
}

/// Rational value of every entry of `m` (after constant folding), or an
/// `InvalidArgument` naming the offending entry.
fn matrix_to_q(m: &Matrix, what: &str) -> Result<Vec<Vec<Q>>, SymplexError> {
    if let Some(rows) = m.to_rational_rows() {
        return Ok(rows);
    }
    let evaluated = m.eval();
    evaluated.to_rational_rows().ok_or_else(|| {
        let bad = evaluated
            .iter()
            .find(|e| e.as_rational().is_none())
            .map(|e| e.to_string())
            .unwrap_or_default();
        invalid(
            "linprog_matrix",
            format!("{what} must contain only numeric literals; found `{bad}`"),
        )
    })
}

/// A column or row vector as a flat list.
fn matrix_to_vec(m: &Matrix, what: &str) -> Result<Vec<Q>, SymplexError> {
    if m.ncols() != 1 && m.nrows() != 1 {
        return Err(invalid(
            "linprog_matrix",
            format!(
                "{what} must be a row or column vector, got {}×{}",
                m.nrows(),
                m.ncols()
            ),
        ));
    }
    // Either one row or one entry per row: flattening gives the entries in
    // order in both cases.
    Ok(matrix_to_q(m, what)?.into_iter().flatten().collect())
}

/// Append the rows of `A x (relation) b` to `p`, checking shapes.
fn add_matrix_block(
    p: &mut LpProblem,
    a: Option<&Matrix>,
    b: Option<&Matrix>,
    relation: Relation,
    name: &str,
) -> Result<(), SymplexError> {
    let n = p.num_vars();
    match (a, b) {
        (None, None) => Ok(()),
        (Some(a), Some(b)) => {
            if a.ncols() != n {
                return Err(invalid(
                    "linprog_matrix",
                    format!("A_{name} has {} columns but c has {n} entries", a.ncols()),
                ));
            }
            let bv = matrix_to_vec(b, &format!("b_{name}"))?;
            if bv.len() != a.nrows() {
                return Err(invalid(
                    "linprog_matrix",
                    format!(
                        "A_{name} has {} rows but b_{name} has {} entries",
                        a.nrows(),
                        bv.len()
                    ),
                ));
            }
            let rows = matrix_to_q(a, &format!("A_{name}"))?;
            for (row, rhs) in rows.into_iter().zip(bv) {
                p.add(row, rhs, relation);
            }
            Ok(())
        }
        _ => Err(invalid(
            "linprog_matrix",
            format!("A_{name} and b_{name} must be given together"),
        )),
    }
}

/// Solve an LP given as [`Matrix`] data with numeric-literal entries.
///
/// * `c`: the objective as an `n×1` column (or `1×n` row) vector;
/// * `a_ub`, `b_ub`: `A_ub x ≤ b_ub` (`m₁×n` and `m₁×1`), both or neither;
/// * `a_eq`, `b_eq`: `A_eq x = b_eq`, both or neither;
/// * bounds are the default `x ≥ 0` (use [`LpProblem`] for anything else).
///
/// Entries are constant-folded with `eval()` first, so `1/2 + 1/3` is
/// accepted; a symbol or transcendental constant is rejected.  Constraints
/// are numbered `≤` rows first, then `=` rows.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] if an entry is not a numeric literal,
/// a matrix is missing its partner, or the shapes are inconsistent.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::linprog::{linprog_matrix, Objective, LpStatus, qi};
///
/// let ctx = Context::new();
/// let c = matrix![ctx, [3], [2]];
/// let a = matrix![ctx, [1, 1], [1, 3]];
/// let b = matrix![ctx, [4], [6]];
/// let sol = linprog_matrix(Objective::Maximize, &c, Some(&a), Some(&b), None, None).unwrap();
/// assert_eq!(sol.status, LpStatus::Optimal);
/// assert_eq!(sol.x_ex(&ctx), vec![ctx.int(4), ctx.int(0)]);
/// assert_eq!(sol.objective, Some(qi(12)));
///
/// let x = ctx.symbol("x");
/// let bad = Matrix::new(vec![vec![x, ctx.int(1)]]).unwrap();
/// assert!(linprog_matrix(Objective::Minimize, &c, Some(&bad), Some(&matrix![ctx, [1]]), None, None).is_err());
/// ```
pub fn linprog_matrix(
    objective: Objective,
    c: &Matrix,
    a_ub: Option<&Matrix>,
    b_ub: Option<&Matrix>,
    a_eq: Option<&Matrix>,
    b_eq: Option<&Matrix>,
) -> Result<LpSolution, SymplexError> {
    let cv = matrix_to_vec(c, "c")?;
    let mut p = match objective {
        Objective::Minimize => LpProblem::minimize(cv),
        Objective::Maximize => LpProblem::maximize(cv),
    };
    add_matrix_block(&mut p, a_ub, b_ub, Relation::Le, "ub")?;
    add_matrix_block(&mut p, a_eq, b_eq, Relation::Eq, "eq")?;
    p.solve()
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn textbook_max() {
        let sol = LpProblem::maximize(vec![qi(3), qi(2)])
            .le(vec![qi(1), qi(1)], qi(4))
            .le(vec![qi(1), qi(3)], qi(6))
            .solve()
            .unwrap();
        assert_eq!(sol.status, LpStatus::Optimal);
        assert_eq!(sol.x, vec![qi(4), qi(0)]);
        assert_eq!(sol.objective, Some(qi(12)));
    }

    #[test]
    fn infeasible_has_certificate() {
        // x + y ≤ 1 and x + y ≥ 2 with x, y ≥ 0.
        let sol = LpProblem::minimize(vec![qi(1), qi(1)])
            .le(vec![qi(1), qi(1)], qi(1))
            .ge(vec![qi(1), qi(1)], qi(2))
            .solve()
            .unwrap();
        assert_eq!(sol.status, LpStatus::Infeasible);
        let y = sol.farkas.unwrap();
        assert!(!y[0].is_negative());
        assert!(!y[1].is_positive());
        // g = Aᵀy = (y0 + y1, y0 + y1) must be ≥ 0 (lower bounds 0) with
        // inf = 0, and yᵀb = y0 + 2 y1 must be < 0.
        let g = &y[0] + &y[1];
        assert!(!g.is_negative());
        assert!((&y[0] + &(&y[1] * &qi(2))).is_negative());
    }

    #[test]
    fn unbounded_detected() {
        let sol = LpProblem::maximize(vec![qi(1), qi(0)])
            .le(vec![qi(0), qi(1)], qi(1))
            .solve()
            .unwrap();
        assert_eq!(sol.status, LpStatus::Unbounded);
    }

    #[test]
    fn free_variable_negative_optimum() {
        // min x  s.t. x ≥ −3 (as a constraint), x free.
        let sol = LpProblem::minimize(vec![qi(1)])
            .ge(vec![qi(1)], qi(-3))
            .free(0)
            .solve()
            .unwrap();
        assert_eq!(sol.status, LpStatus::Optimal);
        assert_eq!(sol.x, vec![qi(-3)]);
    }

    /// White-box: an artificial that stays basic at level zero after
    /// phase 1 is driven out on a *negative* genuine entry, which flips the
    /// sign of the integer tableau's common denominator `d`.  Phase 2 must
    /// then pivot and read values correctly with `d < 0`.
    #[test]
    fn negative_common_denominator_after_driving_out_artificials() {
        // Row 0: −x − y = 0 (rhs 0, so it is not negated by standardise;
        // both genuine entries are negative, so phase 1 never pivots on
        // them).  Row 1: x + z ≤ 5.  Objective: minimise z, which forces a
        // phase-2 pivot (the slack of row 1 enters).
        let p = LpProblem::minimize(vec![qi(0), qi(0), qi(1)])
            .eq(vec![qi(-1), qi(-1), qi(0)], qi(0))
            .le(vec![qi(1), qi(0), qi(1)], qi(5));
        let Standardized::Ready(sf) = standardize(&p) else {
            panic!("bounds are fine");
        };
        // The same white-box walk on both cell types: the hybrid design's
        // promise is that they take the same path.
        fn walk<T: Cell>(sf: &Standard) -> (Vec<Q>, usize) {
            let budget = Budget::default();
            let mut spent = 0;
            let mut t = Tableau::<T>::new(sf, &budget, &mut spent).expect("fits");
            let (m, n) = (t.m, t.n);
            let phase1 = t.phase1_costs();
            t.set_objective(&phase1).expect("fits");
            assert!(matches!(t.run(n + m).ok().unwrap(), Step::Optimal));
            assert!(t.neg_objective().is_zero(), "feasible");
            assert!(t.basis[0] >= n, "artificial of row 0 still basic");
            assert_eq!(t.d.signum(), std::cmp::Ordering::Greater);
            assert!(t.drive_out_artificials().is_ok(), "fits");
            assert!(t.basis[0] < n, "artificial driven out");
            assert!(
                t.d.is_negative(),
                "pivot on a negative entry flips the common denominator (d = {:?})",
                t.d
            );
            let pivots_before = t.pivots_done;
            let mut phase2 = vec![Q::zero(); n + m];
            phase2[..n].clone_from_slice(&sf.c);
            t.set_objective(&phase2).expect("fits");
            assert!(matches!(t.run(n).ok().unwrap(), Step::Optimal));
            assert!(t.pivots_done > pivots_before, "phase 2 pivoted with d < 0");
            let z = t.solution();
            assert!(z.iter().all(|v| !v.is_negative()), "z = {z:?}");
            (z, t.pivots_done)
        }
        assert_eq!(walk::<i64>(&sf), walk::<BigInt>(&sf));
        // Cross-check the end-to-end driver on the same problem.
        let sol = p.solve().unwrap();
        assert_eq!(sol.status, LpStatus::Optimal);
        assert_eq!(sol.x, vec![qi(0), qi(0), qi(0)]);
        assert_eq!(sol.objective, Some(qi(0)));
        assert_eq!(sol.duals[1], qi(0));
    }

    /// White-box: an LP whose tableau entries outgrow `i64` must fall
    /// back to `BigInt` cells and give the same answer the `BigInt` run
    /// gives on its own; small problems must stay in `i64`.
    #[test]
    fn hybrid_arithmetic_falls_back_to_bigint_on_overflow() {
        // Coefficients around 2^40: a single fraction-free pivot multiplies
        // two of them, so the i64 attempt overflows immediately.
        let big = |k: i64| qi(k) * qi(1 << 40);
        let p = LpProblem::minimize(vec![qi(1), qi(1), qi(1)])
            .ge(vec![big(3), big(1), big(2)], big(7))
            .ge(vec![big(1), big(5), big(1)], big(11))
            .le(vec![big(2), big(1), big(3)], big(40));
        let Standardized::Ready(sf) = standardize(&p) else {
            panic!("bounds are fine");
        };
        assert!(
            matches!(
                solve_standard::<i64>(&p, &sf, &Budget::default(), &mut 0),
                Err(Halt::Overflow)
            ),
            "the i64 attempt must report overflow, not a wrong answer"
        );
        let sol = p.solve().unwrap();
        let via_big = solve_standard::<BigInt>(&p, &sf, &Budget::default(), &mut 0)
            .ok()
            .unwrap();
        assert_eq!(sol.status, LpStatus::Optimal);
        assert_eq!(sol.x, via_big.x);
        assert_eq!(sol.objective, via_big.objective);
        assert_eq!(sol.duals, via_big.duals);
        // The optimum is exact: 3x + y + 2z ≥ 7, x + 5y + z ≥ 11 (scaled),
        // minimise x + y + z.
        let obj = sol.objective.unwrap();
        assert!(obj.is_positive());
        // A small LP never leaves the i64 path.
        let small = LpProblem::minimize(vec![qi(1), qi(2)])
            .ge(vec![qi(1), qi(1)], qi(1))
            .le(vec![qi(3), qi(1)], qi(6));
        let Standardized::Ready(sf2) = standardize(&small) else {
            panic!("bounds are fine");
        };
        assert!(solve_standard::<i64>(&small, &sf2, &Budget::default(), &mut 0).is_ok());
    }

    /// White-box: entries beyond `i128` but within 256 bits are solved on
    /// the 256-bit cells, with the answer of the `BigInt` run.
    #[test]
    fn hybrid_arithmetic_uses_256_bit_cells_before_bigint() {
        // Coefficients around 2^70: one pivot multiplies two of them
        // (2^140), so i64 and i128 overflow; three rows keep the minors
        // under 2^230.
        let big = |k: i64| qi(k) * Ratio::from_integer(BigInt::from(1u128 << 70));
        let p = LpProblem::minimize(vec![qi(1), qi(1), qi(1)])
            .ge(vec![big(3), big(1), big(2)], big(7))
            .ge(vec![big(1), big(5), big(1)], big(11))
            .le(vec![big(2), big(1), big(3)], big(40));
        let Standardized::Ready(sf) = standardize(&p) else {
            panic!("bounds are fine");
        };
        assert!(matches!(
            solve_standard::<i64>(&p, &sf, &Budget::default(), &mut 0),
            Err(Halt::Overflow)
        ));
        assert!(matches!(
            solve_standard::<i128>(&p, &sf, &Budget::default(), &mut 0),
            Err(Halt::Overflow)
        ));
        let via_w256 = solve_standard::<W256>(&p, &sf, &Budget::default(), &mut 0)
            .ok()
            .unwrap();
        let via_big = solve_standard::<BigInt>(&p, &sf, &Budget::default(), &mut 0)
            .ok()
            .unwrap();
        assert_eq!(via_w256.status, LpStatus::Optimal);
        assert_eq!(via_w256.x, via_big.x);
        assert_eq!(via_w256.objective, via_big.objective);
        assert_eq!(via_w256.duals, via_big.duals);
        let sol = p.solve().unwrap();
        assert_eq!(sol.x, via_big.x);
        assert_eq!(sol.duals, via_big.duals);
    }

    /// The pivot cap is a cap on the whole solve: an `i64` attempt that
    /// overflows has spent its pivots, and the `BigInt` retry inherits the
    /// count instead of starting afresh.
    #[test]
    fn budget_pivots_are_shared_across_cell_type_attempts() {
        let big = |k: i64| qi(k) * qi(1 << 40);
        let p = LpProblem::minimize(vec![qi(1), qi(1), qi(1)])
            .ge(vec![big(3), big(1), big(2)], big(7))
            .ge(vec![big(1), big(5), big(1)], big(11))
            .le(vec![big(2), big(1), big(3)], big(40));
        let full = p.solve_report().unwrap();
        assert_eq!(full.solution.status, LpStatus::Optimal);
        assert!(full.pivots >= 2, "pivots = {}", full.pivots);
        let Standardized::Ready(sf) = standardize(&p) else {
            panic!("bounds are fine");
        };
        // How many pivots the i64 attempt manages before it overflows.
        let mut wasted = 0usize;
        assert!(matches!(
            solve_standard::<i64>(&p, &sf, &Budget::default(), &mut wasted),
            Err(Halt::Overflow)
        ));
        // Pivots spent in the overflowed attempt count: a budget that
        // covers the BigInt run on its own but not both is exhausted.
        let capped = p
            .clone()
            .with_budget(Budget::max_pivots(full.pivots - wasted))
            .solve_report()
            .unwrap();
        assert_eq!(capped.solution.status, LpStatus::BudgetExhausted);
        assert_eq!(capped.budget_hit, Some(BudgetHit::MaxPivots));
        assert!(capped.solution.x.is_empty() && capped.solution.objective.is_none());
        // Exactly the total is enough.
        let exact = p
            .with_budget(Budget::max_pivots(full.pivots))
            .solve_report()
            .unwrap();
        assert_eq!(exact.solution.status, LpStatus::Optimal);
        assert_eq!(exact.solution.x, full.solution.x);
        assert_eq!(exact.pivots, full.pivots);
    }

    #[test]
    fn budget_deadline_in_the_past_stops_before_the_first_pivot() {
        let p = LpProblem::maximize(vec![qi(3), qi(2)])
            .le(vec![qi(1), qi(1)], qi(4))
            .le(vec![qi(1), qi(3)], qi(6));
        let past = Instant::now() - Duration::from_secs(1);
        let r = p
            .clone()
            .with_budget(Budget::deadline(past))
            .solve_report()
            .unwrap();
        assert_eq!(r.solution.status, LpStatus::BudgetExhausted);
        assert_eq!(r.budget_hit, Some(BudgetHit::Deadline));
        assert_eq!(r.pivots, 0);
        assert_eq!(r.solution.to_string(), "Budget exhausted");
        let r = p
            .with_budget(Budget::within(Duration::ZERO))
            .solve_report()
            .unwrap();
        assert_eq!(r.budget_hit, Some(BudgetHit::Deadline));
    }

    #[test]
    fn malformed_inputs_are_errors() {
        assert!(LpProblem::minimize(vec![]).solve().is_err());
        assert!(
            LpProblem::minimize(vec![qi(1)])
                .le(vec![qi(1), qi(2)], qi(1))
                .solve()
                .is_err()
        );
        assert!(
            LpProblem::minimize(vec![qi(1)])
                .bounds(3, Bounds::free())
                .solve()
                .is_err()
        );
    }
}
