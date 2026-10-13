//! Limits on one call: an absolute deadline, a relative time limit, and a
//! cap on the number of steps.
//!
//! [`Budget`] is the one spelling of "how long may this run" for every
//! budgeted algorithm — the exact simplex ([`linprog`](crate::linprog)),
//! the certificate provers (`PolyhedronOpts`, `SosOpts`) and whatever
//! follows.  It is *declarative*: a relative `time_limit` is turned into
//! an absolute deadline when the call **starts** ([`Budget::start`]), so
//! an option set built once gives every call a fresh allowance, while an
//! absolute `deadline` is shared by every call under it.
//!
//! The two helpers [`deadline_from`] and [`deadline_passed`] are the rule
//! behind that — the earlier of the absolute deadline and now + limit, and
//! "has it passed" with `None` never passing.
//!
//! # Deterministic work budgets
//!
//! A deadline is the caller's choice; the limits the engines set
//! themselves must not depend on the machine or on what the context
//! already holds, or the same input gives different answers.  They count
//! **work**, in units named by a [`BudgetKind`]:
//!
//! * `Work { spent, cap, kind }` is a budget charged explicitly (a
//!   quadrature charges one unit per integrand evaluation);
//! * `WorkScope` (RAII, crate-internal) bounds the nodes an engine builds
//!   while it lives, counted as **interns attempted** in the arena — every
//!   construction, whether hash-consing finds the node already present or
//!   not.  Scopes of one kind nest (the inner one is the tighter of its own
//!   cap and what the enclosing one has left); kinds do not interact.
//!
//! Up to 0.48 the series and integration limits measured the *growth* of
//! the arena, which depends on what hash-consing finds already present: in
//! a context that had done the computation before, nothing was new, the
//! limit never bit, and `series(sign(asin(asin(x))), x, 1, 3)` repeated
//! seven times in one context took 0.70, 1.88, 4.01 … 15.47 s.  The
//! quadrature stopped after 20 s of wall-clock time, so its answer depended
//! on the speed of the machine.  When a budget surfaces as an error it is
//! [`SymplexError::BudgetExhausted`] with its kind.

use std::fmt;
use std::time::{Duration, Instant};

use crate::base::arena::Arena;
use crate::base::errors::SymplexError;

/// A limit on one call: an absolute deadline, a relative time limit and/or
/// a cap on the number of steps (pivots, iterations).  Each defaults to
/// "none".
///
/// The deadline is checked at every step (for the simplex: before the
/// entering column is chosen), the step cap before the step is performed,
/// and both are shared by the `i64 → i128 → 256-bit → BigInt` attempts of
/// one solve: steps begun by an attempt that overflowed its cell type
/// still count, and the deadline is absolute.  When it runs out,
/// [`LpProblem::solve`](crate::linprog::LpProblem::solve) returns
/// [`LpStatus::BudgetExhausted`](crate::linprog::LpStatus::BudgetExhausted).
///
/// **Absolute vs relative.**  [`deadline`](Self::deadline) is an instant;
/// [`time_limit`](Self::time_limit) is measured from the moment the
/// budgeted call starts ([`start`](Self::start) resolves it; when both are
/// set the earlier applies).  [`within`](Self::within) is the eager form:
/// it fixes the deadline *now*.
///
/// `#[non_exhaustive]`: build it with the constructors and `with_*`
/// builders.
///
/// ```
/// use std::time::Duration;
/// use symplex::linprog::{Budget, LpProblem, LpStatus, qi};
///
/// // min x + y  s.t.  x + 2y ≥ 1,  3x + y ≥ 1  needs two pivots: one is not enough.
/// let p = LpProblem::minimize(vec![qi(1), qi(1)])
///     .ge(vec![qi(1), qi(2)], qi(1))
///     .ge(vec![qi(3), qi(1)], qi(1));
/// let sol = p.clone().with_budget(Budget::max_pivots(1)).solve().unwrap();
/// assert_eq!(sol.status, LpStatus::BudgetExhausted);
/// assert!(sol.x.is_empty() && sol.objective.is_none());
/// // A generous budget changes nothing.
/// let sol = p.with_budget(Budget::within(Duration::from_secs(60)).with_max_pivots(1000)).solve().unwrap();
/// assert_eq!(sol.status, LpStatus::Optimal);
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct Budget {
    /// Stop once this instant has passed.
    pub deadline: Option<Instant>,
    /// Time allowed, measured from the start of the budgeted call.  When
    /// both this and `deadline` are set the earlier one applies.
    pub time_limit: Option<Duration>,
    /// Stop before performing this many steps — simplex pivots, across
    /// both phases and all cell-type attempts (and, for a prover, across
    /// every LP of the call).
    pub max_pivots: Option<usize>,
}

impl Budget {
    /// No limit at all.
    pub const fn unlimited() -> Self {
        Budget {
            deadline: None,
            time_limit: None,
            max_pivots: None,
        }
    }

    /// A budget with only an absolute deadline.
    pub const fn deadline(at: Instant) -> Self {
        Budget {
            deadline: Some(at),
            time_limit: None,
            max_pivots: None,
        }
    }

    /// A budget with only a pivot (step) cap.
    pub const fn max_pivots(n: usize) -> Self {
        Budget {
            deadline: None,
            time_limit: None,
            max_pivots: Some(n),
        }
    }

    /// A budget whose deadline is `duration` from **now** (eager: the clock
    /// starts here, not when the call starts).  A duration too large to
    /// represent as an instant means no deadline.  For a per-call
    /// allowance use [`time_limit`](Self::time_limit).
    pub fn within(duration: Duration) -> Self {
        Budget {
            deadline: Instant::now().checked_add(duration),
            time_limit: None,
            max_pivots: None,
        }
    }

    /// A budget with only a relative time limit, resolved when the call
    /// starts.
    pub const fn time_limit(duration: Duration) -> Self {
        Budget {
            deadline: None,
            time_limit: Some(duration),
            max_pivots: None,
        }
    }

    /// Set (or replace) the absolute deadline.
    #[must_use]
    pub fn with_deadline(mut self, at: Instant) -> Self {
        self.deadline = Some(at);
        self
    }

    /// Set (or replace) the relative time limit.
    #[must_use]
    pub fn with_time_limit(mut self, limit: Duration) -> Self {
        self.time_limit = Some(limit);
        self
    }

    /// Set (or replace) the pivot cap.
    #[must_use]
    pub fn with_max_pivots(mut self, n: usize) -> Self {
        self.max_pivots = Some(n);
        self
    }

    /// Set (or replace) the step cap — the same field as
    /// [`with_max_pivots`](Self::with_max_pivots), named for algorithms
    /// whose step is not a pivot.
    #[must_use]
    pub fn with_max_steps(self, n: usize) -> Self {
        self.with_max_pivots(n)
    }

    /// Is every limit absent?
    pub const fn is_unlimited(&self) -> bool {
        self.deadline.is_none() && self.time_limit.is_none() && self.max_pivots.is_none()
    }

    /// The deadline of a call starting now: the earlier of `deadline` and
    /// now + `time_limit` (see [`deadline_from`]).
    pub fn deadline_from_now(&self) -> Option<Instant> {
        deadline_from(self.deadline, self.time_limit)
    }

    /// The budget of a call starting now: the relative time limit is
    /// folded into the absolute deadline (the earlier of the two) and
    /// cleared, so the result can be checked with
    /// [`deadline_passed`](Self::deadline_passed) alone.
    #[must_use]
    pub fn start(&self) -> Self {
        Budget {
            deadline: self.deadline_from_now(),
            time_limit: None,
            max_pivots: self.max_pivots,
        }
    }

    /// Has the absolute deadline passed?  Only `deadline` is consulted —
    /// call [`start`](Self::start) first to account for a `time_limit`.
    pub fn deadline_passed(&self) -> bool {
        deadline_passed(self.deadline)
    }

    /// Which limit has run out after `spent` steps, if any: the deadline
    /// is reported before the step cap.
    pub fn exhausted(&self, spent: usize) -> Option<BudgetHit> {
        if self.deadline_passed() {
            return Some(BudgetHit::Deadline);
        }
        if self.max_pivots.is_some_and(|cap| spent >= cap) {
            return Some(BudgetHit::MaxPivots);
        }
        None
    }
}

/// Which limit of a [`Budget`] ran out.
///
/// `#[non_exhaustive]`: match with a `_` arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BudgetHit {
    /// The deadline (or time limit) passed.
    Deadline,
    /// The pivot / step cap was reached.
    MaxPivots,
}

impl fmt::Display for BudgetHit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            BudgetHit::Deadline => "deadline",
            BudgetHit::MaxPivots => "max_pivots",
        })
    }
}

/// What a deterministic work budget counts, and so which one ran out
/// ([`SymplexError::BudgetExhausted`]).
/// Every kind is measured in work done — nodes built, evaluations made —
/// never in time, so the outcome does not depend on the machine or on the
/// history of the context.
///
/// `#[non_exhaustive]`: match with a `_` arm.
///
/// ```
/// use symplex::BudgetKind;
///
/// assert_eq!(BudgetKind::QuadratureEvaluations.to_string(), "quadrature evaluations");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BudgetKind {
    /// Expression nodes built (interns attempted) by one expansion of the
    /// series engine's last resort for limits.
    SeriesExpansion,
    /// Expression nodes built (interns attempted) by one top-level
    /// `integrate`.
    Integration,
    /// Integrand evaluations of one numerical quadrature.
    QuadratureEvaluations,
}

impl BudgetKind {
    /// Why the computation stopped, as the error message says it (the
    /// quadrature's text is the one it had before the variant existed).
    pub(crate) const fn reason(self) -> &'static str {
        match self {
            BudgetKind::SeriesExpansion => "the series expansion exceeded its work budget",
            BudgetKind::Integration => "the integration exceeded its work budget",
            BudgetKind::QuadratureEvaluations => {
                "quadrature of the definite integral did not converge: the evaluation budget was \
                 exhausted"
            }
        }
    }
}

impl fmt::Display for BudgetKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            BudgetKind::SeriesExpansion => "series expansion",
            BudgetKind::Integration => "integration",
            BudgetKind::QuadratureEvaluations => "quadrature evaluations",
        })
    }
}

/// A deterministic budget charged explicitly: `spent` of `cap` units of
/// `kind`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Work {
    spent: u64,
    cap: u64,
    kind: BudgetKind,
}

impl Work {
    /// Nothing spent of `cap` units of `kind`.
    pub(crate) const fn new(kind: BudgetKind, cap: u64) -> Self {
        Work {
            spent: 0,
            cap,
            kind,
        }
    }

    /// The units spent so far.
    pub(crate) const fn spent(&self) -> u64 {
        self.spent
    }

    /// The units the budget started with.
    pub(crate) const fn cap(&self) -> u64 {
        self.cap
    }

    /// The units left.
    pub(crate) const fn left(&self) -> u64 {
        self.cap.saturating_sub(self.spent)
    }

    /// Charge `units` for `operation`: the [`BudgetExhausted`]
    /// error, with nothing charged, when fewer are left.
    ///
    /// [`BudgetExhausted`]: crate::errors::SymplexError::BudgetExhausted
    pub(crate) fn charge(
        &mut self,
        units: u64,
        operation: &'static str,
    ) -> Result<(), SymplexError> {
        if units > self.left() {
            return Err(SymplexError::BudgetExhausted {
                operation,
                kind: self.kind,
            });
        }
        self.spent += units;
        Ok(())
    }
}

/// A bound on the nodes built in `arena` while the scope lives, counted
/// as interns attempted (see the [module docs](self)): the tighter of
/// `cap` and what an enclosing scope of the same kind has left.  The
/// scope holds the arena and dereferences to it; dropping it — also when
/// a panic unwinds — closes it.
pub(crate) struct WorkScope<'a> {
    arena: &'a mut Arena,
}

impl<'a> WorkScope<'a> {
    /// Open a scope of `kind` allowing `cap` more interns in `arena`.
    pub(crate) fn enter(arena: &'a mut Arena, kind: BudgetKind, cap: u64) -> Self {
        arena.open_work_scope(kind, cap);
        WorkScope { arena }
    }
}

impl std::ops::Deref for WorkScope<'_> {
    type Target = Arena;
    fn deref(&self) -> &Arena {
        self.arena
    }
}

impl std::ops::DerefMut for WorkScope<'_> {
    fn deref_mut(&mut self) -> &mut Arena {
        self.arena
    }
}

impl Drop for WorkScope<'_> {
    fn drop(&mut self) {
        self.arena.close_work_scope();
    }
}

/// The deadline of a call starting now under an absolute `deadline`
/// and/or a relative `time_limit`: the earlier of the two (a limit too
/// large to represent as an instant is no limit).  The one rule every
/// budgeted option set follows.
pub fn deadline_from(deadline: Option<Instant>, time_limit: Option<Duration>) -> Option<Instant> {
    let from_limit = time_limit.and_then(|t| Instant::now().checked_add(t));
    match (deadline, from_limit) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

/// Has `deadline` passed?  (`None` never passes.)
pub fn deadline_passed(deadline: Option<Instant>) -> bool {
    deadline.is_some_and(|d| Instant::now() >= d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builders_and_defaults() {
        assert!(Budget::default().is_unlimited());
        assert_eq!(Budget::default(), Budget::unlimited());
        let b = Budget::max_pivots(3).with_time_limit(Duration::from_secs(1));
        assert_eq!(b.max_pivots, Some(3));
        assert_eq!(b.time_limit, Some(Duration::from_secs(1)));
        assert!(b.deadline.is_none() && !b.is_unlimited());
        assert_eq!(Budget::max_pivots(3), Budget::default().with_max_steps(3));
    }

    #[test]
    fn deadlines() {
        let past = Instant::now() - Duration::from_secs(1);
        assert!(Budget::deadline(past).deadline_passed());
        assert_eq!(
            Budget::deadline(past).exhausted(0),
            Some(BudgetHit::Deadline)
        );
        assert!(!Budget::default().deadline_passed());
        assert!(Budget::within(Duration::ZERO).deadline_passed());
        // A relative limit is not a deadline until the call starts.
        let b = Budget::time_limit(Duration::ZERO);
        assert!(!b.deadline_passed());
        let started = b.start();
        assert!(started.time_limit.is_none() && started.deadline_passed());
        // The earlier of the two wins.
        let far = Instant::now() + Duration::from_secs(3600);
        let d = deadline_from(Some(far), Some(Duration::ZERO)).expect("some deadline");
        assert!(d < far);
        assert_eq!(deadline_from(Some(far), None), Some(far));
        assert!(deadline_from(None, None).is_none());
        assert!(!deadline_passed(None));
        // An unrepresentable limit is no limit.
        assert!(deadline_from(None, Some(Duration::MAX)).is_none());
    }

    #[test]
    fn work_charges_until_the_cap() {
        let mut w = Work::new(BudgetKind::QuadratureEvaluations, 3);
        assert_eq!((w.spent(), w.left(), w.cap()), (0, 3, 3));
        assert!(w.charge(2, "evalf").is_ok());
        // Not enough left: nothing is charged.
        let err = w.charge(2, "evalf").unwrap_err();
        assert!(matches!(
            err,
            SymplexError::BudgetExhausted {
                operation: "evalf",
                kind: BudgetKind::QuadratureEvaluations
            }
        ));
        assert_eq!(w.left(), 1);
        assert!(w.charge(1, "evalf").is_ok());
        assert!(w.charge(1, "evalf").is_err());
    }

    /// A scope counts interns attempted — found nodes too — so a warm
    /// arena spends exactly what a fresh one does; scopes of one kind nest
    /// with the tighter end, other kinds are independent.
    #[test]
    fn work_scopes_count_interns_and_nest() {
        let mut arena = Arena::new();
        let x = arena.symbol("x");
        let two = arena.int(2);
        let warm = arena.pow(x, two);
        assert!(!arena.work_exhausted(BudgetKind::SeriesExpansion));
        {
            let mut outer = WorkScope::enter(&mut arena, BudgetKind::SeriesExpansion, 3);
            let mut steps = 0;
            while !outer.work_exhausted(BudgetKind::SeriesExpansion) {
                // Already present: hash-consing finds it, the scope still pays.
                assert_eq!(outer.pow(x, two), warm);
                steps += 1;
            }
            assert!((1..=4).contains(&steps), "{steps}");
            {
                let mut inner = WorkScope::enter(&mut outer, BudgetKind::Integration, 100);
                assert!(inner.work_exhausted(BudgetKind::SeriesExpansion));
                assert!(!inner.work_exhausted(BudgetKind::Integration));
                let mut same = WorkScope::enter(&mut inner, BudgetKind::SeriesExpansion, 100);
                // The enclosing scope of the kind has nothing left.
                assert!(same.work_exhausted(BudgetKind::SeriesExpansion));
                same.pow(x, two);
            }
            assert!(!outer.work_exhausted(BudgetKind::Integration));
        }
        assert!(!arena.work_exhausted(BudgetKind::SeriesExpansion));
    }

    #[test]
    fn step_cap() {
        let b = Budget::max_pivots(2);
        assert_eq!(b.exhausted(0), None);
        assert_eq!(b.exhausted(1), None);
        assert_eq!(b.exhausted(2), Some(BudgetHit::MaxPivots));
        assert_eq!(BudgetHit::MaxPivots.to_string(), "max_pivots");
        assert_eq!(BudgetHit::Deadline.to_string(), "deadline");
    }
}
