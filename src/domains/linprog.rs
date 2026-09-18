//! Exact linear programming over ℚ: two-phase simplex with Bland's rule,
//! dual values and Farkas infeasibility certificates.
//!
//! Everything is computed with `Ratio<BigInt>` arithmetic, so optima,
//! dual values and certificates are *exact* — no tolerances, no
//! "numerically infeasible" verdicts.  The numeric core works on dense
//! tableaux; use [`linprog_matrix`] to feed [`Matrix`] data whose entries
//! are numeric literals.
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
//! [`linprog`] is a SciPy-shaped convenience wrapper and
//! [`feasible_nonneg`] answers "is there an `x ≥ 0` with `A x = b`?" exactly.
//!
//! # Algorithm and cost
//!
//! Two-phase dense simplex.  Pivots follow Dantzig's rule until the first
//! degenerate (zero-length) step, after which Bland's rule is used for the
//! rest of the solve, so cycling is impossible.  Every `≤`/`≥` row gets a
//! slack, every row gets an artificial (their columns double as `B⁻¹`, from
//! which the duals are read), free variables are split `x = x⁺ − x⁻`, finite
//! lower bounds are shifted away and a finite *upper* bound on a variable
//! that also has a lower bound costs one extra row.  Exact rationals make
//! each pivot `O(m·n)` big-number operations; a 100-row × 200-variable
//! problem with default bounds solves in seconds in release builds.
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

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Signed, Zero};

use crate::api::context::Context;
use crate::api::expr::Ex;
use crate::base::errors::SymplexError;
use crate::domains::matrix::Matrix;

/// Exact rational number used throughout this module.
pub type Q = Ratio<BigInt>;

/// The rational `n / d`.
///
/// # Panics
///
/// Panics if `d == 0` (a programming error, like a zero literal
/// denominator).
///
/// # Examples
///
/// ```
/// use symplex::linprog::q;
/// assert_eq!(q(2, 4), q(1, 2));
/// ```
pub fn q(n: i64, d: i64) -> Q {
    Ratio::new(BigInt::from(n), BigInt::from(d))
}

/// The integer `n` as a rational.
///
/// # Examples
///
/// ```
/// use symplex::linprog::{q, qi};
/// assert_eq!(qi(3), q(6, 2));
/// ```
pub fn qi(n: i64) -> Q {
    Ratio::from_integer(BigInt::from(n))
}

fn invalid(operation: &'static str, reason: impl Into<String>) -> SymplexError {
    SymplexError::InvalidArgument {
        operation,
        reason: reason.into(),
    }
}

fn failed(operation: &'static str, reason: impl Into<String>) -> SymplexError {
    SymplexError::ComputationFailed {
        operation,
        reason: reason.into(),
    }
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
    bounds: Vec<(Option<Q>, Option<Q>)>,
    /// First out-of-range variable index passed to `bounds`, reported by
    /// `solve`.
    bad_var: Option<usize>,
}

impl LpProblem {
    fn new(objective: Objective, c: Vec<Q>) -> Self {
        let n = c.len();
        LpProblem {
            objective,
            c,
            constraints: Vec::new(),
            bounds: vec![(Some(Q::zero()), None); n],
            bad_var: None,
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

    /// Set the bounds `lo ≤ x[var] ≤ hi` of one variable.  `None` means
    /// unbounded on that side; `(None, None)` makes the variable free.
    ///
    /// An out-of-range `var` is reported by [`solve`](Self::solve).
    pub fn bounds(mut self, var: usize, lo: Option<Q>, hi: Option<Q>) -> Self {
        match self.bounds.get_mut(var) {
            Some(slot) => *slot = (lo, hi),
            None => self.bad_var = self.bad_var.or(Some(var)),
        }
        self
    }

    /// Make `x[var]` free (`−∞ < x[var] < ∞`).
    pub fn free(self, var: usize) -> Self {
        self.bounds(var, None, None)
    }

    /// Number of decision variables (the length of `c`).
    pub fn num_vars(&self) -> usize {
        self.c.len()
    }

    /// Number of constraint rows added so far.
    pub fn num_constraints(&self) -> usize {
        self.constraints.len()
    }

    /// Solve the program exactly.
    ///
    /// The returned [`LpSolution`] reports [`LpStatus::Optimal`],
    /// [`LpStatus::Infeasible`] (with a Farkas certificate) or
    /// [`LpStatus::Unbounded`]; none of these is an error.
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
    for (lo, hi) in &p.bounds {
        match (lo, hi) {
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
                constant += &cj * lo;
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
                    rhs -= aij * lo;
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
/// `obj` holds the reduced costs `dⱼ = cⱼ − c_B B⁻¹ Aⱼ` and, in its last
/// entry, `−(current objective value)`.  Because the artificial columns
/// start as the identity, at any time they hold `B⁻¹`, and
/// `d[art_i] = c_{art_i} − yᵢ` where `y = c_B B⁻¹` are the current duals.
struct Tableau {
    rows: Vec<Vec<Q>>,
    obj: Vec<Q>,
    basis: Vec<usize>,
    m: usize,
    /// Number of genuine (non-artificial) columns.
    n: usize,
    pivots_done: usize,
    max_pivots: usize,
    /// Switches to Bland's rule permanently after the first degenerate
    /// pivot (a zero-length step), which is the only situation in which
    /// Dantzig's rule could cycle.
    bland: bool,
}

enum Step {
    Optimal,
    Unbounded,
}

impl Tableau {
    fn new(std: &Standard) -> Self {
        let m = std.a.len();
        let n = std.c.len();
        let width = n + m + 1;
        let zero = Q::zero();
        let one = Q::one();
        let mut rows = Vec::with_capacity(m);
        for i in 0..m {
            let mut row = Vec::with_capacity(width);
            row.extend(std.a[i].iter().cloned());
            row.extend((0..m).map(|k| if k == i { one.clone() } else { zero.clone() }));
            row.push(std.b[i].clone());
            rows.push(row);
        }
        let basis: Vec<usize> = (0..m).map(|i| n + i).collect();
        Tableau {
            rows,
            obj: vec![zero; width],
            basis,
            m,
            n,
            pivots_done: 0,
            max_pivots: 10_000 + 50 * (m + n),
            bland: false,
        }
    }

    #[inline]
    fn rhs_col(&self) -> usize {
        self.n + self.m
    }

    /// Install a new objective (`costs` over all `n + m` columns) and price
    /// out the current basis so that `obj` holds true reduced costs.
    fn set_objective(&mut self, costs: &[Q]) {
        let width = self.rhs_col() + 1;
        let mut obj = vec![Q::zero(); width];
        obj[..costs.len()].clone_from_slice(costs);
        for (i, &k) in self.basis.iter().enumerate() {
            let f = obj[k].clone();
            if f.is_zero() {
                continue;
            }
            let row = &self.rows[i];
            for (o, r) in obj.iter_mut().zip(row.iter()) {
                if !r.is_zero() {
                    *o -= &f * r;
                }
            }
        }
        self.obj = obj;
    }

    fn pivot(&mut self, r: usize, s: usize) {
        let p = self.rows[r][s].clone();
        if !p.is_one() {
            for v in self.rows[r].iter_mut() {
                if !v.is_zero() {
                    *v = std::mem::take(v) / &p;
                }
            }
        }
        let pivot_row = std::mem::take(&mut self.rows[r]);
        for (i, row) in self.rows.iter_mut().enumerate() {
            if i == r {
                continue;
            }
            let f = row[s].clone();
            if f.is_zero() {
                continue;
            }
            for (v, pr) in row.iter_mut().zip(pivot_row.iter()) {
                if !pr.is_zero() {
                    *v -= &f * pr;
                }
            }
        }
        let f = self.obj[s].clone();
        if !f.is_zero() {
            for (v, pr) in self.obj.iter_mut().zip(pivot_row.iter()) {
                if !pr.is_zero() {
                    *v -= &f * pr;
                }
            }
        }
        self.rows[r] = pivot_row;
        self.basis[r] = s;
        self.pivots_done += 1;
    }

    /// Run the simplex method on the current objective, considering only
    /// columns `< limit` as candidates to enter the basis.
    fn run(&mut self, limit: usize) -> Result<Step, SymplexError> {
        loop {
            if self.pivots_done >= self.max_pivots {
                return Err(failed(
                    "linprog",
                    format!(
                        "pivot cap of {} exceeded; the problem is degenerate beyond \
                         what the solver handles",
                        self.max_pivots
                    ),
                ));
            }
            // Entering column: Dantzig (most negative reduced cost, lowest
            // index on ties) or Bland (lowest index with negative cost).
            let mut entering: Option<usize> = None;
            for j in 0..limit {
                if !self.obj[j].is_negative() {
                    continue;
                }
                match entering {
                    None => entering = Some(j),
                    Some(e) if self.obj[j] < self.obj[e] => entering = Some(j),
                    Some(_) => {}
                }
                if self.bland {
                    break;
                }
            }
            let Some(s) = entering else {
                return Ok(Step::Optimal);
            };
            // Leaving row: minimum ratio, ties broken by smallest basic index.
            let rhs = self.rhs_col();
            let mut leaving: Option<(usize, Q)> = None;
            for i in 0..self.m {
                let a = &self.rows[i][s];
                if !a.is_positive() {
                    continue;
                }
                let ratio = &self.rows[i][rhs] / a;
                let better = match &leaving {
                    None => true,
                    Some((lr, lratio)) => {
                        ratio < *lratio || (ratio == *lratio && self.basis[i] < self.basis[*lr])
                    }
                };
                if better {
                    leaving = Some((i, ratio));
                }
            }
            let Some((r, ratio)) = leaving else {
                return Ok(Step::Unbounded);
            };
            if ratio.is_zero() {
                self.bland = true;
            }
            self.pivot(r, s);
        }
    }

    /// Value of the current basic solution over the genuine columns.
    fn solution(&self) -> Vec<Q> {
        let rhs = self.rhs_col();
        let mut z = vec![Q::zero(); self.n];
        for (i, &k) in self.basis.iter().enumerate() {
            if k < self.n {
                z[k] = self.rows[i][rhs].clone();
            }
        }
        z
    }

    /// Pivot artificial variables out of the basis wherever possible.
    /// Rows whose genuine part is entirely zero are redundant; their
    /// artificial stays basic at level zero and never moves again.
    fn drive_out_artificials(&mut self) {
        for r in 0..self.m {
            if self.basis[r] < self.n {
                continue;
            }
            if let Some(s) = (0..self.n).find(|&j| !self.rows[r][j].is_zero()) {
                self.pivot(r, s);
            }
        }
    }

    /// Current duals `y = c_B B⁻¹` read off the artificial columns:
    /// `yᵢ = c_{art_i} − d[art_i]`.
    fn duals(&self, art_cost: &Q) -> Vec<Q> {
        (0..self.m)
            .map(|i| art_cost - &self.obj[self.n + i])
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Driver
// ═══════════════════════════════════════════════════════════════════════════

fn solve_lp(p: &LpProblem) -> Result<LpSolution, SymplexError> {
    let sf = match standardize(p) {
        Standardized::Ready(s) => s,
        Standardized::BoundsInfeasible => return Ok(LpSolution::infeasible(None)),
    };
    let mut t = Tableau::new(&sf);
    let m = t.m;
    let n = t.n;

    // Phase 1: minimise the sum of the artificials.
    let mut phase1 = vec![Q::zero(); n + m];
    for v in phase1[n..].iter_mut() {
        *v = Q::one();
    }
    t.set_objective(&phase1);
    match t.run(n + m)? {
        Step::Optimal => {}
        Step::Unbounded => {
            // Cannot happen: the phase-1 objective is bounded below by 0.
            return Err(failed("linprog", "phase 1 reported unbounded"));
        }
    }
    let infeasibility = -t.obj[t.rhs_col()].clone();
    if infeasibility.is_positive() {
        // Farkas certificate from the phase-1 duals, mapped back to the
        // caller's rows (bound rows are absorbed into the box infimum).
        let y_std = t.duals(&Q::one());
        let farkas: Vec<Q> = (0..sf.m_orig)
            .map(|i| -(&sf.row_sign[i] * &y_std[i]))
            .collect();
        return Ok(LpSolution::infeasible(Some(farkas)));
    }

    // Phase 2.
    t.drive_out_artificials();
    let mut phase2 = vec![Q::zero(); n + m];
    phase2[..n].clone_from_slice(&sf.c);
    t.set_objective(&phase2);
    match t.run(n)? {
        Step::Optimal => {}
        Step::Unbounded => return Ok(LpSolution::unbounded()),
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

    let y_std = t.duals(&Q::zero());
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
/// `A_eq x = b_eq` and `bounds[j].0 ≤ xⱼ ≤ bounds[j].1`.
///
/// `bounds` may be empty (every variable then defaults to `0 ≤ xⱼ < ∞`) or
/// have one `(lo, hi)` pair per variable, `None` meaning unbounded on that
/// side.  Constraints are numbered with the `≤` rows first, then the `=`
/// rows — that is the order of [`LpSolution::duals`] / [`farkas`](LpSolution::farkas).
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
    bounds: &[(Option<Q>, Option<Q>)],
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
    for (j, (lo, hi)) in bounds.iter().enumerate() {
        p = p.bounds(j, lo.clone(), hi.clone());
    }
    p.solve()
}

/// Exact feasibility of `A x = b, x ≥ 0`: `Some(x)` with a solution, or
/// `None` if none exists.
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
                .bounds(3, None, None)
                .solve()
                .is_err()
        );
    }
}
