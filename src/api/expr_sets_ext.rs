//! Set-algebra, boolean-logic, and piecewise methods on `SetEx` / `BoolEx` / [`Ex`](crate::api::expr::Ex).
//!
//! * [`SetEx`] — normal-form evaluation ([`simplify`](SetEx::simplify)),
//!   eager set operations (`difference`, `symmetric_difference`,
//!   `absolute_complement`), three-valued queries (`contains`, `is_subset`,
//!   `is_disjoint`, `is_empty`, …), bounds and measure, topology, normal-form
//!   accessors, and conversion to a membership condition.
//! * [`BoolEx`] — [`simplify`](BoolEx::simplify), assumption-aware
//!   [`eval`](BoolEx::eval), normal forms, tautology / satisfiability,
//!   truth tables, and [`solve_for`](BoolEx::solve_for).
//! * [`Ex`] — [`is_in`](Ex::is_in) and [`piecewise_simplify`](Ex::piecewise_simplify).
//! * [`reduce_inequalities`] — free-function form of
//!   [`SetEx::reduce_inequalities`].
//!
//! All backend work is done by [`crate::transforms::sets`] and
//! [`crate::transforms::logic`].

use tracing::debug_span;

use crate::api::expr::{BoolEx, Boolean, Ex, Expr, Numeric, SetEx, SetValued};
use crate::base::errors::SymplexError;
use crate::base::node::ExprId;

// ═══════════════════════════════════════════════════════════════════════════
// impl Expr<SetValued>
// ═══════════════════════════════════════════════════════════════════════════

impl Expr<SetValued> {
    /// Evaluate a set expression to **normal form**: a union of
    /// pairwise-disjoint, ascending intervals followed by one finite set of
    /// isolated points, whenever all endpoints are comparable real numbers.
    ///
    /// Sets with symbolic endpoints stay structural, but the safe
    /// identities (`A ∪ ∅ = A`, `A ∩ U = A`, `A ∪ A = A`, `A \ A = ∅`,
    /// `(Aᶜ)ᶜ = A`, flattening, deduplication) are still applied.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let a = ctx.interval(&ctx.int(0), &ctx.int(2), false, false);
    /// let b = ctx.interval(&ctx.int(1), &ctx.int(3), false, false);
    /// assert_eq!(format!("{}", a.intersection(&b).simplify()), "[1, 2]");
    /// assert_eq!(format!("{}", a.union(&b).simplify()), "[0, 3]");
    ///
    /// // Solver output: (−∞,−2) ∪ (2,∞) intersected with [0, 5] → (2, 5]
    /// let x = ctx.symbol("x");
    /// let sol = (&x.powi(2) - 4).solve_gt(&x);
    /// let window = ctx.interval(&ctx.int(0), &ctx.int(5), false, false);
    /// assert_eq!(format!("{}", sol.intersection(&window).simplify()), "(2, 5]");
    /// ```
    #[must_use = "returns the simplified form; does not modify in place"]
    pub fn simplify(&self) -> SetEx {
        let _span = debug_span!("set_simplify", expr = ?self.raw_id()).entered();
        let id = {
            let mut inner = self.inner.write();
            crate::transforms::sets::simplify_set(&mut inner.arena, self.raw_id())
        };
        self.wrap(id)
    }

    /// Exact evaluation of the numeric endpoints / elements of the set
    /// (`[sqrt(4), 3]` → `[2, 3]`).  Does not perform set algebra; see
    /// [`simplify`](SetEx::simplify) for that.
    #[must_use = "returns the evaluated form; does not modify in place"]
    pub fn eval(&self) -> SetEx {
        let id = self.inner.write().arena.eval_expr(self.raw_id());
        self.wrap(id)
    }

    /// Set difference `self \ other`, evaluated to normal form.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let a = ctx.interval(&ctx.int(0), &ctx.int(3), false, false);
    /// let b = ctx.interval(&ctx.int(1), &ctx.int(2), false, false);
    /// assert_eq!(format!("{}", a.difference(&b)), "[0, 1) ∪ (2, 3]");
    /// ```
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn difference(&self, other: &SetEx) -> SetEx {
        let other_id = self.checked_id(other);
        let id = {
            let mut inner = self.inner.write();
            crate::transforms::sets::set_difference(&mut inner.arena, self.raw_id(), other_id)
        };
        self.wrap(id)
    }

    /// Symmetric difference `(self \ other) ∪ (other \ self)`, evaluated to
    /// normal form.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let a = ctx.interval(&ctx.int(0), &ctx.int(2), false, false);
    /// let b = ctx.interval(&ctx.int(1), &ctx.int(3), false, false);
    /// assert_eq!(format!("{}", a.symmetric_difference(&b)), "[0, 1) ∪ (2, 3]");
    /// ```
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn symmetric_difference(&self, other: &SetEx) -> SetEx {
        let other_id = self.checked_id(other);
        let id = {
            let mut inner = self.inner.write();
            crate::transforms::sets::symmetric_difference(&mut inner.arena, self.raw_id(), other_id)
        };
        self.wrap(id)
    }

    /// Absolute complement `ℝ \ self`, evaluated to normal form.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let a = ctx.interval(&ctx.int(0), &ctx.int(1), false, true);
    /// assert_eq!(format!("{}", a.absolute_complement()), "(-oo, 0) ∪ [1, oo)");
    /// assert_eq!(format!("{}", ctx.empty_set().absolute_complement()), "(-oo, oo)");
    /// ```
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn absolute_complement(&self) -> SetEx {
        let id = {
            let mut inner = self.inner.write();
            crate::transforms::sets::absolute_complement(&mut inner.arena, self.raw_id())
        };
        self.wrap(id)
    }

    /// Set membership: is `elem ∈ self`?
    ///
    /// Three-valued: `Some(true)` / `Some(false)` when membership can be
    /// decided (numeric elements in evaluable sets, exact differences such
    /// as `x ∈ [x, x + 1]`, substitution into `ConditionSet`s), `None`
    /// otherwise — never a guess.
    ///
    /// For the structural "appears as a sub-expression" check, use
    /// `set.as_ex().contains(&needle)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let a = ctx.interval(&ctx.int(0), &ctx.int(1), true, false); // (0, 1]
    /// assert_eq!(a.contains(&ctx.int(0)), Some(false));
    /// assert_eq!(a.contains(&ctx.int(1)), Some(true));
    /// assert_eq!(a.contains(&ctx.rational(1, 2)), Some(true));
    /// assert_eq!(a.contains(&ctx.symbol("x")), None);
    /// ```
    #[must_use]
    pub fn contains(&self, elem: &Ex) -> Option<bool> {
        let elem_id = self.checked_id(elem);
        let mut inner = self.inner.write();
        crate::transforms::sets::set_contains(&mut inner.arena, self.raw_id(), elem_id)
    }

    /// Is `self ⊆ other`?  Three-valued.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let a = ctx.interval(&ctx.int(0), &ctx.int(1), false, false);
    /// let b = ctx.interval(&ctx.int(-1), &ctx.int(2), true, true);
    /// assert_eq!(a.is_subset(&b), Some(true));
    /// assert_eq!(b.is_subset(&a), Some(false));
    /// ```
    #[must_use]
    pub fn is_subset(&self, other: &SetEx) -> Option<bool> {
        let other_id = self.checked_id(other);
        let mut inner = self.inner.write();
        crate::transforms::sets::is_subset(&mut inner.arena, self.raw_id(), other_id)
    }

    /// Is `self ⊇ other`?  Three-valued.
    #[must_use]
    pub fn is_superset(&self, other: &SetEx) -> Option<bool> {
        other.is_subset(self)
    }

    /// Is `self ∩ other = ∅`?  Three-valued.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let a = ctx.interval(&ctx.int(0), &ctx.int(1), false, true); // [0, 1)
    /// let b = ctx.interval(&ctx.int(1), &ctx.int(2), false, false); // [1, 2]
    /// assert_eq!(a.is_disjoint(&b), Some(true));
    /// ```
    #[must_use]
    pub fn is_disjoint(&self, other: &SetEx) -> Option<bool> {
        let other_id = self.checked_id(other);
        let mut inner = self.inner.write();
        crate::transforms::sets::is_disjoint(&mut inner.arena, self.raw_id(), other_id)
    }

    /// Is the set empty?  Three-valued.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let a = ctx.interval(&ctx.int(0), &ctx.int(1), false, false);
    /// let b = ctx.interval(&ctx.int(2), &ctx.int(3), false, false);
    /// assert_eq!(a.intersection(&b).is_empty(), Some(true));
    /// assert_eq!(a.is_empty(), Some(false));
    /// ```
    #[must_use]
    pub fn is_empty(&self) -> Option<bool> {
        let mut inner = self.inner.write();
        crate::transforms::sets::is_empty(&mut inner.arena, self.raw_id())
    }

    /// Greatest lower bound (`-oo` when unbounded below).
    ///
    /// Returns `None` for the empty set or when the set cannot be
    /// evaluated to normal form.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let a = ctx.interval(&ctx.int(0), &ctx.int(1), true, true);
    /// let b = ctx.finite_set(&[ctx.int(5)]);
    /// let u = a.union(&b);
    /// assert_eq!(format!("{}", u.inf().unwrap()), "0");
    /// assert_eq!(format!("{}", u.sup().unwrap()), "5");
    /// assert!(ctx.empty_set().inf().is_none());
    /// ```
    #[must_use]
    pub fn inf(&self) -> Option<Ex> {
        let id = {
            let mut inner = self.inner.write();
            crate::transforms::sets::inf(&mut inner.arena, self.raw_id())?
        };
        Some(self.wrap_as::<Numeric>(id))
    }

    /// Least upper bound (`oo` when unbounded above).  See [`inf`](SetEx::inf).
    #[must_use]
    pub fn sup(&self) -> Option<Ex> {
        let id = {
            let mut inner = self.inner.write();
            crate::transforms::sets::sup(&mut inner.arena, self.raw_id())?
        };
        Some(self.wrap_as::<Numeric>(id))
    }

    /// Lebesgue measure (total length); `oo` for unbounded sets, `0` for
    /// finite sets.  `None` when the set cannot be evaluated.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let a = ctx.interval(&ctx.int(0), &ctx.int(1), false, false);
    /// let b = ctx.interval(&ctx.int(2), &ctx.rational(5, 2), true, true);
    /// assert_eq!(format!("{}", a.union(&b).measure().unwrap()), "3/2");
    /// assert_eq!(format!("{}", ctx.reals().measure().unwrap()), "oo");
    /// ```
    #[must_use]
    pub fn measure(&self) -> Option<Ex> {
        let id = {
            let mut inner = self.inner.write();
            crate::transforms::sets::measure(&mut inner.arena, self.raw_id())?
        };
        Some(self.wrap_as::<Numeric>(id))
    }

    /// Topological boundary (the finite endpoints and isolated points).
    /// `None` when the set cannot be evaluated to normal form.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let a = ctx.interval(&ctx.int(0), &ctx.int(1), true, true);
    /// assert_eq!(format!("{}", a.boundary().unwrap()), "{0, 1}");
    /// assert_eq!(format!("{}", a.closure().unwrap()), "[0, 1]");
    /// assert_eq!(format!("{}", a.closure().unwrap().interior().unwrap()), "(0, 1)");
    /// ```
    #[must_use]
    pub fn boundary(&self) -> Option<SetEx> {
        let id = {
            let mut inner = self.inner.write();
            crate::transforms::sets::boundary(&mut inner.arena, self.raw_id())?
        };
        Some(self.wrap(id))
    }

    /// Topological closure.  `None` when the set cannot be evaluated.
    #[must_use]
    pub fn closure(&self) -> Option<SetEx> {
        let id = {
            let mut inner = self.inner.write();
            crate::transforms::sets::closure(&mut inner.arena, self.raw_id())?
        };
        Some(self.wrap(id))
    }

    /// Topological interior.  `None` when the set cannot be evaluated.
    #[must_use]
    pub fn interior(&self) -> Option<SetEx> {
        let id = {
            let mut inner = self.inner.write();
            crate::transforms::sets::interior(&mut inner.arena, self.raw_id())?
        };
        Some(self.wrap(id))
    }

    /// Is the set open in ℝ?  Three-valued.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let a = ctx.interval(&ctx.int(0), &ctx.int(1), true, true);
    /// assert_eq!(a.is_open(), Some(true));
    /// assert_eq!(a.is_closed(), Some(false));
    /// assert_eq!(ctx.reals().is_open(), Some(true));
    /// assert_eq!(ctx.reals().is_closed(), Some(true));
    /// ```
    #[must_use]
    pub fn is_open(&self) -> Option<bool> {
        let mut inner = self.inner.write();
        crate::transforms::sets::is_open(&mut inner.arena, self.raw_id())
    }

    /// Is the set closed in ℝ?  Three-valued.
    #[must_use]
    pub fn is_closed(&self) -> Option<bool> {
        let mut inner = self.inner.write();
        crate::transforms::sets::is_closed(&mut inner.arena, self.raw_id())
    }

    /// Normal-form accessor: the pieces of the set as
    /// `(lo, hi, lo_open, hi_open)` in ascending order.  Isolated points
    /// appear as `(p, p, false, false)`.  `None` when the set cannot be
    /// evaluated to normal form.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let sol = (&x.powi(2) - 4).solve_gt(&x); // (−∞, −2) ∪ (2, ∞)
    /// let parts = sol.as_intervals().unwrap();
    /// assert_eq!(parts.len(), 2);
    /// assert_eq!(format!("{}", parts[0].0), "-oo");
    /// assert_eq!(format!("{}", parts[0].1), "-2");
    /// assert!(parts[0].2 && parts[0].3);
    /// ```
    #[must_use]
    pub fn as_intervals(&self) -> Option<Vec<(Ex, Ex, bool, bool)>> {
        let parts = {
            let mut inner = self.inner.write();
            crate::transforms::sets::as_intervals(&mut inner.arena, self.raw_id())?
        };
        Some(
            parts
                .into_iter()
                .map(|(lo, hi, lo_open, hi_open)| {
                    (
                        self.wrap_as::<Numeric>(lo),
                        self.wrap_as::<Numeric>(hi),
                        lo_open,
                        hi_open,
                    )
                })
                .collect(),
        )
    }

    /// The elements of a finite set (numeric elements in ascending order,
    /// possibly followed by symbolic ones).  `None` if the set is not
    /// known to be finite.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let roots = (&x.powi(2) - 4).solve_le(&x).boundary().unwrap();
    /// let elems = roots.as_finite_set().unwrap();
    /// assert_eq!(elems.len(), 2);
    /// assert!(ctx.reals().as_finite_set().is_none());
    /// assert_eq!(ctx.empty_set().as_finite_set(), Some(vec![]));
    /// ```
    #[must_use]
    pub fn as_finite_set(&self) -> Option<Vec<Ex>> {
        let elems = {
            let mut inner = self.inner.write();
            crate::transforms::sets::as_finite_set(&mut inner.arena, self.raw_id())?
        };
        Some(
            elems
                .into_iter()
                .map(|e| self.wrap_as::<Numeric>(e))
                .collect(),
        )
    }

    /// Membership of `var` as a boolean expression:
    /// `(a, b] ∪ {c}` becomes `(x > a) ∧ (b ≥ x) ∨ (x = c)`.
    ///
    /// Works for symbolic sets too; `ConditionSet`s substitute `var` into
    /// their condition.  Returns `Err(InvalidArgument)` if `self` contains
    /// a node that is not a set constructor.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let a = ctx.interval(&ctx.int(0), &ctx.int(1), true, false);
    /// let cond = a.to_condition(&x).unwrap();
    /// assert_eq!(format!("{cond}"), "x > 0 & 1 >= x");
    /// // and back again:
    /// assert_eq!(format!("{}", cond.solve_for(&x).unwrap()), "(0, 1]");
    /// ```
    pub fn to_condition(&self, var: &Ex) -> Result<BoolEx, SymplexError> {
        let var_id = self.checked_id(var);
        let id = {
            let mut inner = self.inner.write();
            crate::transforms::sets::to_condition(&mut inner.arena, self.raw_id(), var_id)?
        };
        Ok(self.wrap_as::<Boolean>(id))
    }

    /// Reduce a system of univariate conditions in `var` to its solution
    /// set in normal form.
    ///
    /// Each condition may combine `>`, `>=`, `<`, `<=`, `=`, `!=` atoms in
    /// `var` with `and` / `or` / `not`.  Atoms are solved with the
    /// inequality solver (sign charts) and equation solver, then combined
    /// with set algebra.  The conditions are conjoined.
    ///
    /// # Errors
    ///
    /// * `InvalidArgument` — `var` is not a symbol, `conds` is empty, or a
    ///   condition contains a non-relational atom / does not involve `var`.
    /// * `ComputationFailed` — an atom could not be solved.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let c1 = x.powi(2).gt(&ctx.int(4));
    /// let c2 = x.lt(&ctx.int(5));
    /// let sol = SetEx::reduce_inequalities(&[c1, c2], &x).unwrap();
    /// assert_eq!(format!("{sol}"), "(-oo, -2) ∪ (2, 5)");
    /// ```
    pub fn reduce_inequalities(conds: &[BoolEx], var: &Ex) -> Result<SetEx, SymplexError> {
        let Some(first) = conds.first() else {
            return Err(SymplexError::InvalidArgument {
                operation: "reduce_inequalities",
                reason: "at least one condition is required".into(),
            });
        };
        let var_id = first.checked_id(var);
        let cond_ids: Vec<ExprId> = conds.iter().map(|c| first.checked_id(c)).collect();
        let id = {
            let mut inner = first.inner.write();
            crate::transforms::sets::reduce_inequalities(&mut inner.arena, &cond_ids, var_id)?
        };
        Ok(first.wrap_as::<SetValued>(id))
    }
}

/// Reduce a system of univariate conditions in `var` to its solution set.
///
/// Free-function form of [`SetEx::reduce_inequalities`]; see there for
/// details, errors and examples.
pub fn reduce_inequalities(conds: &[BoolEx], var: &Ex) -> Result<SetEx, SymplexError> {
    SetEx::reduce_inequalities(conds, var)
}

// ═══════════════════════════════════════════════════════════════════════════
// impl Expr<Boolean>
// ═══════════════════════════════════════════════════════════════════════════

impl Expr<Boolean> {
    /// Simplify a boolean expression.
    ///
    /// The operands of every relational atom are simplified with the
    /// numeric engine (each operand individually, memoised), then boolean
    /// algebra is applied: flattening of nested `and`/`or`,
    /// duplicate removal, constant folding, negation pushed to the atoms
    /// (De Morgan, double negation, `¬(a < b) = a ≥ b`), absorption
    /// (`A ∧ (A ∨ B) = A`), complements (`A ∧ ¬A = false`), merging of
    /// relationals on the same operands (`x > 0 ∧ x ≥ 0 = x > 0`,
    /// `x > 0 ∨ x = 0 = x ≥ 0`), and folding of relationals whose operands
    /// can be ordered (`2 > 1`, `π > 3`, `x + 1 > x`).
    ///
    /// The result is in negation normal form with children sorted
    /// deterministically.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let p = x.gt(&ctx.int(0));
    /// let q = x.ge(&ctx.int(0));
    /// assert_eq!(p.and(&q).simplify(), p);
    /// assert_eq!(format!("{}", p.not().simplify()), "0 >= x");
    /// assert_eq!(format!("{}", p.or(&p.not()).simplify()), "True");
    /// ```
    #[must_use = "returns the simplified form; does not modify in place"]
    pub fn simplify(&self) -> BoolEx {
        let _span = debug_span!("bool_simplify", expr = ?self.raw_id()).entered();
        let id = {
            let mut inner = self.inner.write();
            crate::transforms::logic::simplify_bool_full(&mut inner.arena, self.raw_id())
        };
        self.wrap(id)
    }

    /// Evaluate a boolean expression.
    ///
    /// Numeric sub-expressions are evaluated exactly and relationals with
    /// numeric operands are folded (`5 > 3` → `True`).  In addition,
    /// relationals whose truth follows from the assumption system are
    /// folded: `x > 0` is `True` when `x` was declared `Positive`.
    /// Connectives are constant-folded.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let t = ctx.symbol_with("t", &[Assumption::Positive]);
    /// assert_eq!(format!("{}", t.gt(&ctx.int(0)).eval()), "True");
    /// assert_eq!(format!("{}", t.le(&ctx.int(0)).eval()), "False");
    /// let x = ctx.symbol("x");
    /// assert_eq!(format!("{}", x.gt(&ctx.int(0)).eval()), "x > 0");
    /// ```
    #[must_use = "returns the evaluated form; does not modify in place"]
    pub fn eval(&self) -> BoolEx {
        let id = {
            let mut inner = self.inner.write();
            let crate::api::context::ContextInner {
                ref mut arena,
                ref assumptions,
                ..
            } = *inner;
            let mut guard = assumptions.lock();
            crate::transforms::logic::eval_bool(arena, &mut guard, self.raw_id())
        };
        self.wrap(id)
    }

    /// Negation normal form: negations pushed onto the atoms
    /// (`¬(A ∧ B)` → `¬A ∨ ¬B`, `¬(a > b)` → `b ≥ a`), decidable
    /// relationals folded.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let y = ctx.symbol("y");
    /// let e = x.gt(&ctx.int(0)).and(&y.gt(&ctx.int(0))).not();
    /// assert_eq!(format!("{}", e.to_nnf()), "0 >= x | 0 >= y");
    /// ```
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn to_nnf(&self) -> BoolEx {
        let id = {
            let mut inner = self.inner.write();
            crate::transforms::logic::to_nnf(&mut inner.arena, self.raw_id())
        };
        self.wrap(id)
    }

    /// Conjunctive normal form (a conjunction of disjunctions of
    /// literals), simplified.  If distribution would produce more than a
    /// few thousand clauses the simplified NNF is returned instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let p = ctx.symbol("p").gt(&ctx.int(0));
    /// let q = ctx.symbol("q").gt(&ctx.int(0));
    /// let r = ctx.symbol("r").gt(&ctx.int(0));
    /// let e = p.or(&q.and(&r));
    /// assert_eq!(format!("{}", e.to_cnf()), "(p > 0 | q > 0) & (p > 0 | r > 0)");
    /// ```
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn to_cnf(&self) -> BoolEx {
        let id = {
            let mut inner = self.inner.write();
            crate::transforms::logic::to_cnf(&mut inner.arena, self.raw_id())
        };
        self.wrap(id)
    }

    /// Disjunctive normal form (a disjunction of conjunctions of
    /// literals), simplified.  Same size guard as [`to_cnf`](BoolEx::to_cnf).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let p = ctx.symbol("p").gt(&ctx.int(0));
    /// let q = ctx.symbol("q").gt(&ctx.int(0));
    /// let r = ctx.symbol("r").gt(&ctx.int(0));
    /// let e = p.and(&q.or(&r));
    /// assert_eq!(format!("{}", e.to_dnf()), "p > 0 & q > 0 | p > 0 & r > 0");
    /// ```
    #[must_use = "returns a new expression; does not modify in place"]
    pub fn to_dnf(&self) -> BoolEx {
        let id = {
            let mut inner = self.inner.write();
            crate::transforms::logic::to_dnf(&mut inner.arena, self.raw_id())
        };
        self.wrap(id)
    }

    /// Is this formula true under every assignment?  Three-valued.
    ///
    /// A propositional proof (each relational pair is a three-valued
    /// variable, opaque atoms are two-valued; DPLL with unit propagation,
    /// up to 24 variables) gives `Some(true)`.  A propositional
    /// counter-example is trusted only when the atoms are independent
    /// (each relational is linear in its own symbol).  If neither applies
    /// and every atom is a relational in one common free symbol, the
    /// question is decided exactly through the inequality solver.
    /// Otherwise `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let p = x.gt(&ctx.int(0));
    /// assert_eq!(p.or(&p.not()).is_tautology(), Some(true));
    /// // x > 1 → x > 0 is not propositional, but exact on the real line:
    /// assert_eq!(x.gt(&ctx.int(1)).implies(&p).is_tautology(), Some(true));
    /// assert_eq!(p.is_tautology(), Some(false));
    /// ```
    #[must_use]
    pub fn is_tautology(&self) -> Option<bool> {
        let mut inner = self.inner.write();
        crate::transforms::logic::is_tautology(&mut inner.arena, self.raw_id())
    }

    /// Is this formula false under every assignment?  Three-valued; see
    /// [`is_tautology`](BoolEx::is_tautology) for the decision procedure.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let e = x.gt(&ctx.int(1)).and(&x.lt(&ctx.int(0)));
    /// assert_eq!(e.is_contradiction(), Some(true));
    /// assert_eq!(e.satisfiable(), Some(false));
    /// ```
    #[must_use]
    pub fn is_contradiction(&self) -> Option<bool> {
        let mut inner = self.inner.write();
        crate::transforms::logic::is_contradiction(&mut inner.arena, self.raw_id())
    }

    /// Does some assignment make this formula true?  Three-valued
    /// (`None` when the question cannot be decided — e.g. relationals
    /// that share variables non-trivially and are propositionally
    /// consistent).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let y = ctx.symbol("y");
    /// assert_eq!(x.gt(&ctx.int(0)).satisfiable(), Some(true));
    /// // independent linear atoms: exact
    /// assert_eq!(x.gt(&ctx.int(0)).and(&y.gt(&ctx.int(0))).satisfiable(), Some(true));
    /// // x > y and x > 0 share x: undecided rather than guessed
    /// assert_eq!(x.gt(&y).and(&x.gt(&ctx.int(0))).satisfiable(), None);
    /// ```
    #[must_use]
    pub fn satisfiable(&self) -> Option<bool> {
        let mut inner = self.inner.write();
        crate::transforms::logic::satisfiable(&mut inner.arena, self.raw_id())
    }

    /// The distinct atomic sub-formulas (relationals and opaque
    /// propositions), in post-order.  Constants are excluded.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let p = x.gt(&ctx.int(0));
    /// let q = x.lt(&ctx.int(5));
    /// let e = p.and(&q).or(&p.not());
    /// assert_eq!(e.atoms().len(), 2);
    /// ```
    #[must_use]
    pub fn atoms(&self) -> Vec<BoolEx> {
        let inner = self.inner.read();
        let ids = crate::transforms::logic::atoms(&inner.arena, self.raw_id());
        drop(inner);
        ids.into_iter().map(|id| self.wrap(id)).collect()
    }

    /// Truth table over the given variables (at most 8).
    ///
    /// Each row is `(values, result)` with `values` in the order of
    /// `vars`, most significant variable first (`false` before `true`).
    /// Relational variables and their negations are recognised as the same
    /// variable; rows whose variable values are mutually inconsistent
    /// (`x > 0` and `x < 0` both true) are omitted.
    ///
    /// # Errors
    ///
    /// `InvalidArgument` for more than 8 variables or constant variables;
    /// `ComputationFailed` if the formula is not determined by `vars`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let p = ctx.symbol("p").gt(&ctx.int(0));
    /// let q = ctx.symbol("q").gt(&ctx.int(0));
    /// let rows = p.implies(&q).truth_table(&[p.clone(), q.clone()]).unwrap();
    /// assert_eq!(rows.len(), 4);
    /// assert_eq!(rows[2], (vec![true, false], false)); // p ∧ ¬q falsifies p → q
    /// ```
    pub fn truth_table(&self, vars: &[BoolEx]) -> Result<Vec<(Vec<bool>, bool)>, SymplexError> {
        let var_ids: Vec<ExprId> = vars.iter().map(|v| self.checked_id(v)).collect();
        let mut inner = self.inner.write();
        crate::transforms::logic::truth_table(&mut inner.arena, self.raw_id(), &var_ids)
    }

    /// Solve this condition for `var` as a set (method form of
    /// [`SetEx::reduce_inequalities`]).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let cond = x.ge(&ctx.int(0)).or(&x.lt(&ctx.int(-3)));
    /// assert_eq!(format!("{}", cond.solve_for(&x).unwrap()), "(-oo, -3) ∪ [0, oo)");
    /// let cond = x.powi(2).le(&ctx.int(1)).not();
    /// assert_eq!(format!("{}", cond.solve_for(&x).unwrap()), "(-oo, -1) ∪ (1, oo)");
    /// ```
    pub fn solve_for(&self, var: &Ex) -> Result<SetEx, SymplexError> {
        reduce_inequalities(std::slice::from_ref(self), var)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// impl Expr<Numeric>
// ═══════════════════════════════════════════════════════════════════════════

impl Expr<Numeric> {
    /// Set membership `self ∈ set`; alias of [`SetEx::contains`].
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let sol = (&x.powi(2) - 4).solve_gt(&x); // (−∞, −2) ∪ (2, ∞)
    /// assert_eq!(ctx.int(3).is_in(&sol), Some(true));
    /// assert_eq!(ctx.int(0).is_in(&sol), Some(false));
    /// assert_eq!(x.is_in(&sol), None);
    /// ```
    #[must_use]
    pub fn is_in(&self, set: &SetEx) -> Option<bool> {
        set.contains(self)
    }

    /// Simplify every `Piecewise` node in this expression.
    ///
    /// Conditions are simplified with boolean algebra; `false` branches
    /// are dropped; evaluation stops at the first `true` branch; a branch
    /// repeating an earlier condition is unreachable and dropped; adjacent
    /// branches with identical values are merged; a single remaining branch
    /// with condition `true` collapses to its value.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let pos = x.gt(&ctx.int(0));
    /// let nonpos = x.le(&ctx.int(0));
    /// let never = ctx.int(1).gt(&ctx.int(2));
    /// let pw = Ex::piecewise(&[(&x, &never), (&x, &pos), (&x, &nonpos)]);
    /// assert_eq!(pw.piecewise_simplify(), x);
    /// ```
    #[must_use = "returns the simplified form; does not modify in place"]
    pub fn piecewise_simplify(&self) -> Ex {
        let id = {
            let mut inner = self.inner.write();
            crate::transforms::logic::piecewise_simplify(&mut inner.arena, self.raw_id())
        };
        self.wrap(id)
    }
}
