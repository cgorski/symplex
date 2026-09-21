//! Discrete-time Markov chains on a finite state space with an exact
//! rational transition matrix (SymPy's `stats.DiscreteMarkovChain`).
//!
//! Every structural question (communication classes, periods,
//! irreducibility, absorbing states) is answered on the graph of positive
//! entries, and every quantitative one (`n`-step transitions, stationary
//! distributions, the fundamental matrix, absorption and hitting
//! probabilities, expected hitting times) by exact linear algebra on
//! [`QMatrix`], so the answers are rationals, not floats.
//!
//! ```
//! use symplex::prelude::*;
//! use symplex::stats::markov::MarkovChain;
//! use symplex::linprog::{q, qi};
//!
//! // SymPy: DiscreteMarkovChain('Y', [0, 1, 2], T).stationary_distribution() = [2/7, 3/7, 2/7]
//! let p = QMatrix::new(vec![
//!     vec![q(1, 2), q(1, 2), qi(0)],
//!     vec![q(1, 3), qi(0), q(2, 3)],
//!     vec![qi(0), qi(1), qi(0)],
//! ])?;
//! let chain = MarkovChain::new(p)?;
//! assert!(chain.is_irreducible() && chain.is_regular());
//! assert_eq!(chain.stationary_distribution()?, vec![q(2, 7), q(3, 7), q(2, 7)]);
//! # Ok::<(), SymplexError>(())
//! ```
//!
//! # Conventions
//!
//! * `P[i][j]` is the probability of moving from state `i` to state `j`;
//!   rows sum to `1`.  Distributions are row vectors: `μ_k = μ_0 Pᵏ`.
//! * *Ergodic* means irreducible (Kemeny–Snell, SymPy's `is_ergodic`);
//!   *regular* means irreducible and aperiodic, equivalently some `Pᵏ` has
//!   every entry positive (SymPy's `is_regular`).
//! * A chain is *absorbing* when it has an absorbing state and every state
//!   can reach one; its transient states are then listed in ascending order
//!   in the fundamental matrix and the absorption probabilities.

use num_integer::Integer;
use num_traits::{One, Signed, ToPrimitive, Zero};

use crate::base::errors::SymplexError;
use crate::domains::exact_matrix::QMatrix;

use super::data::Q;
use super::sample::Rng;

const OP: &str = "stats::markov";

fn invalid(reason: impl Into<String>) -> SymplexError {
    SymplexError::invalid_argument(OP, reason)
}

fn failed(reason: impl Into<String>) -> SymplexError {
    SymplexError::computation_failed(OP, reason)
}

/// Square matrix product without the (impossible here) shape error.
fn mul_square(a: &QMatrix, b: &QMatrix) -> QMatrix {
    let n = a.nrows();
    QMatrix::from_fn(n, n, |i, j| {
        (0..n).fold(Q::zero(), |acc, k| acc + a.get(i, k) * b.get(k, j))
    })
}

/// The `rows × cols` submatrix picked out by two index lists (both
/// non-empty).
fn pick(m: &QMatrix, rows: &[usize], cols: &[usize]) -> QMatrix {
    QMatrix::from_fn(rows.len(), cols.len(), |i, j| {
        m.get(rows[i], cols[j]).clone()
    })
}

/// A finite Markov chain with an exact transition matrix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MarkovChain {
    p: QMatrix,
    labels: Option<Vec<String>>,
}

impl MarkovChain {
    /// A chain from its transition matrix.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] unless `p` is square, every entry
    /// is non-negative and every row sums to exactly `1`.
    pub fn new(p: QMatrix) -> Result<Self, SymplexError> {
        if !p.is_square() {
            return Err(invalid(format!(
                "a transition matrix must be square, got {}×{}",
                p.nrows(),
                p.ncols()
            )));
        }
        for (i, row) in p.rows().enumerate() {
            if let Some(v) = row.iter().find(|v| v.is_negative()) {
                return Err(invalid(format!("negative entry {v} in row {i}")));
            }
            let sum = row.iter().fold(Q::zero(), |acc, v| acc + v);
            if !sum.is_one() {
                return Err(invalid(format!("row {i} sums to {sum}, not 1")));
            }
        }
        Ok(MarkovChain { p, labels: None })
    }

    /// A chain with named states (`labels[i]` is the name of state `i`).
    ///
    /// # Errors
    ///
    /// As [`new`](Self::new), plus [`SymplexError::InvalidArgument`] if
    /// there is not exactly one label per state.
    pub fn with_labels(p: QMatrix, labels: Vec<String>) -> Result<Self, SymplexError> {
        if labels.len() != p.nrows() {
            return Err(invalid(format!(
                "{} labels for {} states",
                labels.len(),
                p.nrows()
            )));
        }
        let mut chain = Self::new(p)?;
        chain.labels = Some(labels);
        Ok(chain)
    }

    /// The transition matrix.
    pub fn transition_matrix(&self) -> &QMatrix {
        &self.p
    }

    /// The number of states.
    pub fn n_states(&self) -> usize {
        self.p.nrows()
    }

    /// The state labels, if any were given.
    pub fn labels(&self) -> Option<&[String]> {
        self.labels.as_deref()
    }

    /// The index of the state named `label`.
    pub fn state_index(&self, label: &str) -> Option<usize> {
        self.labels
            .as_ref()
            .and_then(|l| l.iter().position(|s| s == label))
    }

    fn check_state(&self, state: usize) -> Result<(), SymplexError> {
        if state >= self.n_states() {
            return Err(invalid(format!(
                "state {state} out of range for {} states",
                self.n_states()
            )));
        }
        Ok(())
    }

    // ── Transitions ──────────────────────────────────────────────────────

    /// The `k`-step transition matrix `Pᵏ` (`P⁰ = I`), by repeated
    /// squaring.
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::stats::markov::MarkovChain;
    /// use symplex::linprog::{q, qi};
    ///
    /// let p = QMatrix::new(vec![vec![q(1, 2), q(1, 2)], vec![qi(1), qi(0)]])?;
    /// let chain = MarkovChain::new(p)?;
    /// // SymPy: T**2 = [[3/4, 1/4], [1/2, 1/2]]
    /// assert_eq!(chain.n_step(2).row(0), &[q(3, 4), q(1, 4)]);
    /// # Ok::<(), SymplexError>(())
    /// ```
    pub fn n_step(&self, k: usize) -> QMatrix {
        let mut result = QMatrix::identity(self.n_states());
        let mut base = self.p.clone();
        let mut e = k;
        while e > 0 {
            if e & 1 == 1 {
                result = mul_square(&result, &base);
            }
            e >>= 1;
            if e > 0 {
                base = mul_square(&base, &base);
            }
        }
        result
    }

    /// The state distribution after `k` steps from the row vector
    /// `initial`: `initial · Pᵏ`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] unless `initial` has one
    /// non-negative entry per state and sums to `1`.
    pub fn distribution_after(&self, initial: &[Q], k: usize) -> Result<Vec<Q>, SymplexError> {
        let n = self.n_states();
        if initial.len() != n {
            return Err(invalid(format!(
                "the initial distribution has {} entries for {n} states",
                initial.len()
            )));
        }
        if initial.iter().any(|v| v.is_negative()) {
            return Err(invalid("the initial distribution has a negative entry"));
        }
        let total = initial.iter().fold(Q::zero(), |acc, v| acc + v);
        if !total.is_one() {
            return Err(invalid(format!(
                "the initial distribution sums to {total}, not 1"
            )));
        }
        let pk = self.n_step(k);
        Ok((0..n)
            .map(|j| (0..n).fold(Q::zero(), |acc, i| acc + &initial[i] * pk.get(i, j)))
            .collect())
    }

    // ── Graph structure ──────────────────────────────────────────────────

    /// `reach[i][j]`: is `j` reachable from `i` in one or more steps?
    /// (Warshall's transitive closure of the positive-entry graph.)
    fn reachability(&self) -> Vec<Vec<bool>> {
        let n = self.n_states();
        let mut reach: Vec<Vec<bool>> = (0..n)
            .map(|i| (0..n).map(|j| self.p.get(i, j).is_positive()).collect())
            .collect();
        for k in 0..n {
            let via_k = reach[k].clone();
            for row in reach.iter_mut() {
                if !row[k] {
                    continue;
                }
                for (cell, &through) in row.iter_mut().zip(&via_k) {
                    if through {
                        *cell = true;
                    }
                }
            }
        }
        reach
    }

    /// The communication classes (states that reach each other, or a
    /// single state that reaches nothing that returns), each ascending,
    /// ordered by smallest member.  SymPy: `communication_classes()`
    /// (which also reports recurrence and period; see
    /// [`closed_classes`](Self::closed_classes) and
    /// [`period_of`](Self::period_of)).
    pub fn communication_classes(&self) -> Vec<Vec<usize>> {
        let n = self.n_states();
        let reach = self.reachability();
        let mut assigned = vec![false; n];
        let mut classes = Vec::new();
        for i in 0..n {
            if assigned[i] {
                continue;
            }
            let class: Vec<usize> = (i..n)
                .filter(|&j| j == i || (reach[i][j] && reach[j][i]))
                .collect();
            for &j in &class {
                assigned[j] = true;
            }
            classes.push(class);
        }
        classes
    }

    /// The closed (recurrent) communication classes: those no transition
    /// leaves.  Every finite chain has at least one.
    pub fn closed_classes(&self) -> Vec<Vec<usize>> {
        let n = self.n_states();
        self.communication_classes()
            .into_iter()
            .filter(|class| {
                class
                    .iter()
                    .all(|&i| (0..n).all(|j| class.contains(&j) || !self.p.get(i, j).is_positive()))
            })
            .collect()
    }

    /// The transient states: those whose class is not closed (the chain
    /// leaves them for good with probability one).
    pub fn transient_states(&self) -> Vec<usize> {
        let closed = self.closed_classes();
        (0..self.n_states())
            .filter(|i| !closed.iter().any(|c| c.contains(i)))
            .collect()
    }

    /// `true` if every state reaches every other (one communication
    /// class).
    pub fn is_irreducible(&self) -> bool {
        self.communication_classes().len() == 1
    }

    /// The period of `state`: the gcd of the lengths of all cycles through
    /// it, or `None` for a state the chain never returns to.  Computed on
    /// the state's class as `gcd(d(u) + 1 − d(v))` over its edges `u → v`,
    /// `d` being breadth-first distances from `state`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `state` is out of range.
    pub fn period_of(&self, state: usize) -> Result<Option<usize>, SymplexError> {
        self.check_state(state)?;
        let class = self
            .communication_classes()
            .into_iter()
            .find(|c| c.contains(&state))
            .unwrap_or_else(|| vec![state]);
        Ok(self.class_period(&class))
    }

    fn class_period(&self, class: &[usize]) -> Option<usize> {
        let n = self.n_states();
        let &start = class.first()?;
        // Breadth-first distances inside the class.
        let mut dist: Vec<Option<usize>> = vec![None; n];
        dist[start] = Some(0);
        let mut queue = std::collections::VecDeque::from([start]);
        while let Some(u) = queue.pop_front() {
            let du = dist[u].unwrap_or(0);
            for &v in class {
                if self.p.get(u, v).is_positive() && dist[v].is_none() {
                    dist[v] = Some(du + 1);
                    queue.push_back(v);
                }
            }
        }
        let mut g = 0usize;
        for &u in class {
            for &v in class {
                if !self.p.get(u, v).is_positive() {
                    continue;
                }
                if let (Some(du), Some(dv)) = (dist[u], dist[v]) {
                    g = g.gcd(&(du + 1).abs_diff(dv));
                }
            }
        }
        (g > 0).then_some(g)
    }

    /// `true` if every class that has a cycle has period `1`.
    pub fn is_aperiodic(&self) -> bool {
        self.communication_classes()
            .iter()
            .all(|c| self.class_period(c).is_none_or(|d| d == 1))
    }

    /// Irreducible (Kemeny–Snell's *ergodic*; SymPy `is_ergodic`).
    pub fn is_ergodic(&self) -> bool {
        self.is_irreducible()
    }

    /// Irreducible and aperiodic — some `Pᵏ` is entrywise positive, and
    /// `Pᵏ → 1π` (SymPy `is_regular`).
    pub fn is_regular(&self) -> bool {
        self.is_irreducible() && self.is_aperiodic()
    }

    // ── Stationary distributions ─────────────────────────────────────────

    /// One stationary distribution `π` (`πP = π`, `Σπ = 1`) per closed
    /// class, each supported on its class — a basis of the extreme points
    /// of the stationary simplex, ordered as
    /// [`closed_classes`](Self::closed_classes).  Exact: the nullspace of
    /// `P_Cᵀ − I` on each class, normalised.  For an irreducible chain the
    /// single entry is the unique stationary distribution.
    /// SymPy: `stationary_distribution()` (which returns a parametric
    /// mixture for reducible chains).
    pub fn stationary_distributions(&self) -> Vec<Vec<Q>> {
        let n = self.n_states();
        self.closed_classes()
            .iter()
            .filter_map(|class| {
                let sub = pick(&self.p, class, class);
                let m = class.len();
                let a = sub.transpose().sub(&QMatrix::identity(m)).ok()?;
                let basis = a.nullspace();
                let v = basis.first()?.col(0);
                let total = v.iter().fold(Q::zero(), |acc, x| acc + x);
                if total.is_zero() {
                    return None;
                }
                let mut pi = vec![Q::zero(); n];
                for (k, &state) in class.iter().enumerate() {
                    pi[state] = &v[k] / &total;
                }
                Some(pi)
            })
            .collect()
    }

    /// The unique stationary distribution of a chain with a single closed
    /// class (every irreducible chain).
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if the chain has several closed
    /// classes (then every mixture of the
    /// [`stationary_distributions`](Self::stationary_distributions) is
    /// stationary).
    pub fn stationary_distribution(&self) -> Result<Vec<Q>, SymplexError> {
        let mut all = self.stationary_distributions();
        match all.len() {
            1 => all
                .pop()
                .ok_or_else(|| failed("no stationary distribution")),
            k => Err(invalid(format!(
                "the chain has {k} closed classes, so its stationary distribution is not unique"
            ))),
        }
    }

    // ── Absorbing chains ─────────────────────────────────────────────────

    /// The absorbing states (`P[i][i] = 1`), ascending.
    pub fn absorbing_states(&self) -> Vec<usize> {
        (0..self.n_states())
            .filter(|&i| self.p.get(i, i).is_one())
            .collect()
    }

    /// `true` if the chain has an absorbing state and every state can
    /// reach one (SymPy `is_absorbing_chain`).
    pub fn is_absorbing_chain(&self) -> bool {
        let absorbing = self.absorbing_states();
        if absorbing.is_empty() {
            return false;
        }
        let reach = self.reachability();
        (0..self.n_states()).all(|i| absorbing.iter().any(|&a| a == i || reach[i][a]))
    }

    /// The transient and absorbing state lists of an absorbing chain.
    fn absorbing_split(&self) -> Result<(Vec<usize>, Vec<usize>), SymplexError> {
        if !self.is_absorbing_chain() {
            return Err(invalid(
                "not an absorbing chain (some state cannot reach an absorbing state)",
            ));
        }
        let absorbing = self.absorbing_states();
        let transient: Vec<usize> = (0..self.n_states())
            .filter(|i| !absorbing.contains(i))
            .collect();
        if transient.is_empty() {
            return Err(invalid(
                "every state is absorbing; there are no transient states",
            ));
        }
        Ok((transient, absorbing))
    }

    /// The fundamental matrix `N = (I − Q)⁻¹` of an absorbing chain, `Q`
    /// being the transitions among the transient states (ascending):
    /// `N[i][j]` is the expected number of visits to transient `j` starting
    /// from transient `i`.  SymPy: `fundamental_matrix()` on an absorbing
    /// chain.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] unless the chain is absorbing with
    /// at least one transient state.
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::stats::markov::MarkovChain;
    /// use symplex::linprog::{q, qi};
    ///
    /// // Gambler's ruin on 0..=4 with a fair coin; SymPy: N = [[3/2, 1, 1/2], [1, 2, 1], [1/2, 1, 3/2]]
    /// let h = q(1, 2);
    /// let p = QMatrix::new(vec![
    ///     vec![qi(1), qi(0), qi(0), qi(0), qi(0)],
    ///     vec![h.clone(), qi(0), h.clone(), qi(0), qi(0)],
    ///     vec![qi(0), h.clone(), qi(0), h.clone(), qi(0)],
    ///     vec![qi(0), qi(0), h.clone(), qi(0), h.clone()],
    ///     vec![qi(0), qi(0), qi(0), qi(0), qi(1)],
    /// ])?;
    /// let chain = MarkovChain::new(p)?;
    /// assert_eq!(chain.fundamental_matrix()?.row(1), &[qi(1), qi(2), qi(1)]);
    /// assert_eq!(chain.expected_steps_to_absorption()?, vec![qi(3), qi(4), qi(3)]);
    /// # Ok::<(), SymplexError>(())
    /// ```
    pub fn fundamental_matrix(&self) -> Result<QMatrix, SymplexError> {
        let (transient, _) = self.absorbing_split()?;
        let q = pick(&self.p, &transient, &transient);
        QMatrix::identity(transient.len())
            .sub(&q)?
            .inv()
            .map_err(|e| failed(format!("I − Q is singular: {e}")))
    }

    /// The absorption probabilities `B = N R` of an absorbing chain:
    /// `B[i][j]` is the probability that transient state `i` (ascending) is
    /// eventually absorbed in absorbing state `j` (ascending).  SymPy:
    /// `absorbing_probabilities()`.
    ///
    /// # Errors
    ///
    /// As [`fundamental_matrix`](Self::fundamental_matrix).
    pub fn absorption_probabilities(&self) -> Result<QMatrix, SymplexError> {
        let (transient, absorbing) = self.absorbing_split()?;
        let n = self.fundamental_matrix()?;
        let r = pick(&self.p, &transient, &absorbing);
        n.matmul(&r)
    }

    /// The expected number of steps until absorption from each transient
    /// state (ascending): `t = N·1`.
    ///
    /// # Errors
    ///
    /// As [`fundamental_matrix`](Self::fundamental_matrix).
    pub fn expected_steps_to_absorption(&self) -> Result<Vec<Q>, SymplexError> {
        let n = self.fundamental_matrix()?;
        Ok(n.rows()
            .map(|row| row.iter().fold(Q::zero(), |acc, v| acc + v))
            .collect())
    }

    // ── Hitting probabilities and times ──────────────────────────────────

    fn check_target(&self, target: &[usize]) -> Result<Vec<usize>, SymplexError> {
        if target.is_empty() {
            return Err(invalid("the target set is empty"));
        }
        for &t in target {
            self.check_state(t)?;
        }
        let mut sorted = target.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        Ok(sorted)
    }

    /// The probability `hᵢ` of ever entering `target` from each state `i`
    /// (`hᵢ = 1` on the target): the minimal non-negative solution of
    /// `hᵢ = Σⱼ P[i][j] hⱼ` off the target, found exactly by setting
    /// `h = 0` on the states that cannot reach the target and solving the
    /// remaining linear system.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `target` is empty or names a
    /// state out of range.
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::stats::markov::MarkovChain;
    /// use symplex::linprog::{q, qi};
    ///
    /// // Fair gambler's ruin on 0..=4: P(ruin | start k) = 1 − k/4 (SymPy linear solve).
    /// let h = q(1, 2);
    /// let p = QMatrix::new(vec![
    ///     vec![qi(1), qi(0), qi(0), qi(0), qi(0)],
    ///     vec![h.clone(), qi(0), h.clone(), qi(0), qi(0)],
    ///     vec![qi(0), h.clone(), qi(0), h.clone(), qi(0)],
    ///     vec![qi(0), qi(0), h.clone(), qi(0), h.clone()],
    ///     vec![qi(0), qi(0), qi(0), qi(0), qi(1)],
    /// ])?;
    /// let chain = MarkovChain::new(p)?;
    /// assert_eq!(chain.hitting_probability(&[0])?, vec![qi(1), q(3, 4), q(1, 2), q(1, 4), qi(0)]);
    /// # Ok::<(), SymplexError>(())
    /// ```
    pub fn hitting_probability(&self, target: &[usize]) -> Result<Vec<Q>, SymplexError> {
        let target = self.check_target(target)?;
        let n = self.n_states();
        let reach = self.reachability();
        let mut h = vec![Q::zero(); n];
        for &t in &target {
            h[t] = Q::one();
        }
        // States off the target that can reach it: the unknowns.
        let unknown: Vec<usize> = (0..n)
            .filter(|&i| !target.contains(&i) && target.iter().any(|&t| reach[i][t]))
            .collect();
        if unknown.is_empty() {
            return Ok(h);
        }
        // (I − P_UU) h_U = P_UT · 1
        let a = QMatrix::identity(unknown.len()).sub(&pick(&self.p, &unknown, &unknown))?;
        let b = QMatrix::col_vector(
            unknown
                .iter()
                .map(|&i| {
                    target
                        .iter()
                        .fold(Q::zero(), |acc, &t| acc + self.p.get(i, t))
                })
                .collect(),
        );
        let sol = a
            .solve(&b)
            .map_err(|e| failed(format!("the hitting-probability system is singular: {e}")))?;
        for (k, &i) in unknown.iter().enumerate() {
            h[i] = sol.get(k, 0).clone();
        }
        Ok(h)
    }

    /// The expected number of steps `kᵢ` to first enter `target` from each
    /// state `i` (`kᵢ = 0` on the target): the solution of
    /// `kᵢ = 1 + Σⱼ P[i][j] kⱼ` off the target, exactly.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `target` is empty or out of
    /// range; [`SymplexError::ComputationFailed`] if some state reaches the
    /// target with probability `< 1` (its expected time is infinite).
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::stats::markov::MarkovChain;
    /// use symplex::linprog::{q, qi};
    ///
    /// let p = QMatrix::new(vec![
    ///     vec![q(1, 2), q(1, 2), qi(0)],
    ///     vec![q(1, 3), qi(0), q(2, 3)],
    ///     vec![qi(0), qi(1), qi(0)],
    /// ])?;
    /// let chain = MarkovChain::new(p)?;
    /// // SymPy solve of k0 = 1 + k0/2 + k1/2, k1 = 1 + k0/3: k0 = 9/2, k1 = 5/2.
    /// assert_eq!(chain.expected_hitting_time(&[2])?, vec![q(9, 2), q(5, 2), qi(0)]);
    /// # Ok::<(), SymplexError>(())
    /// ```
    pub fn expected_hitting_time(&self, target: &[usize]) -> Result<Vec<Q>, SymplexError> {
        let target = self.check_target(target)?;
        let h = self.hitting_probability(&target)?;
        let n = self.n_states();
        let infinite: Vec<usize> = (0..n).filter(|&i| !h[i].is_one()).collect();
        if !infinite.is_empty() {
            return Err(failed(format!(
                "states {infinite:?} reach the target with probability < 1, so their expected hitting time is infinite"
            )));
        }
        let unknown: Vec<usize> = (0..n).filter(|i| !target.contains(i)).collect();
        let mut k = vec![Q::zero(); n];
        if unknown.is_empty() {
            return Ok(k);
        }
        // (I − P_UU) k_U = 1
        let a = QMatrix::identity(unknown.len()).sub(&pick(&self.p, &unknown, &unknown))?;
        let b = QMatrix::col_vector(vec![Q::one(); unknown.len()]);
        let sol = a
            .solve(&b)
            .map_err(|e| failed(format!("the hitting-time system is singular: {e}")))?;
        for (idx, &i) in unknown.iter().enumerate() {
            k[i] = sol.get(idx, 0).clone();
        }
        Ok(k)
    }

    // ── Ergodic chains ───────────────────────────────────────────────────

    /// Kemeny–Snell's fundamental matrix of an irreducible chain,
    /// `Z = (I − P + W)⁻¹` with every row of `W` the stationary
    /// distribution.  SymPy: `fundamental_matrix()` on a non-absorbing
    /// chain.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] unless the chain is irreducible.
    pub fn fundamental_matrix_ergodic(&self) -> Result<QMatrix, SymplexError> {
        if !self.is_irreducible() {
            return Err(invalid(
                "the ergodic fundamental matrix needs an irreducible chain",
            ));
        }
        let pi = self.stationary_distribution()?;
        let n = self.n_states();
        let w = QMatrix::from_fn(n, n, |_, j| pi[j].clone());
        QMatrix::identity(n)
            .sub(&self.p)?
            .add(&w)?
            .inv()
            .map_err(|e| failed(format!("I − P + W is singular: {e}")))
    }

    /// The mean first-passage times of an irreducible chain:
    /// `M[i][j] = (Z[j][j] − Z[i][j]) / πⱼ` for `i ≠ j`, the expected number
    /// of steps to first reach `j` from `i`; the diagonal is `0` (see
    /// [`mean_recurrence_times`](Self::mean_recurrence_times) for `1/πᵢ`).
    /// Agrees with [`expected_hitting_time`](Self::expected_hitting_time)
    /// of each singleton.
    ///
    /// # Errors
    ///
    /// As [`fundamental_matrix_ergodic`](Self::fundamental_matrix_ergodic).
    pub fn mean_first_passage_times(&self) -> Result<QMatrix, SymplexError> {
        let z = self.fundamental_matrix_ergodic()?;
        let pi = self.stationary_distribution()?;
        let n = self.n_states();
        Ok(QMatrix::from_fn(n, n, |i, j| {
            if i == j || pi[j].is_zero() {
                Q::zero()
            } else {
                (z.get(j, j) - z.get(i, j)) / &pi[j]
            }
        }))
    }

    /// The mean recurrence times `1/πᵢ` of an irreducible chain.
    ///
    /// # Errors
    ///
    /// As [`stationary_distribution`](Self::stationary_distribution).
    pub fn mean_recurrence_times(&self) -> Result<Vec<Q>, SymplexError> {
        if !self.is_irreducible() {
            return Err(invalid("mean recurrence times need an irreducible chain"));
        }
        Ok(self
            .stationary_distribution()?
            .iter()
            .map(|p| if p.is_zero() { Q::zero() } else { p.recip() })
            .collect())
    }

    // ── Simulation ───────────────────────────────────────────────────────

    /// A sample path of `steps` transitions from `initial`, inverse-transform
    /// sampled row by row with the deterministic generator `rng`; the
    /// result has `steps + 1` states and begins with `initial`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `initial` is out of range.
    pub fn sample_path(
        &self,
        initial: usize,
        steps: usize,
        rng: &mut Rng,
    ) -> Result<Vec<usize>, SymplexError> {
        self.check_state(initial)?;
        let n = self.n_states();
        let rows: Vec<Vec<f64>> = self
            .p
            .rows()
            .map(|row| {
                row.iter()
                    .map(|q| q.numer().to_f64().unwrap_or(0.0) / q.denom().to_f64().unwrap_or(1.0))
                    .collect()
            })
            .collect();
        let mut path = Vec::with_capacity(steps + 1);
        let mut state = initial;
        path.push(state);
        for _ in 0..steps {
            let u = rng.next_f64();
            let row = &rows[state];
            let mut acc = 0.0;
            let mut next = None;
            for (j, &pj) in row.iter().enumerate() {
                if pj <= 0.0 {
                    continue;
                }
                acc += pj;
                if u < acc {
                    next = Some(j);
                    break;
                }
            }
            // Rounding left u above the last cumulative sum: take the last
            // state with positive probability.
            state = next
                .or_else(|| (0..n).rev().find(|&j| row[j] > 0.0))
                .unwrap_or(state);
            path.push(state);
        }
        Ok(path)
    }
}
