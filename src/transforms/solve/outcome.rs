//! The outcome of a solve: every root ([`Solutions::Complete`]), some of
//! them ([`Solutions::Partial`]), or nothing claimed
//! ([`Solutions::Refused`]), over a [`RootSet`]: no root (a proof, with its
//! [`Reason`]), every value, finitely many points, or the points and
//! periodic families of a general solve.
//!
//! Up to 0.48 the solver answered with `Option<Vec<Solution>>` in 25
//! functions and with `SolveOutcome { Solutions(Vec), Identity,
//! NoSolution, Unresolved }` elsewhere, and an empty list meant opposite
//! things: "no root was found" for `SolveOutcome::Solutions(vec![])` (and
//! for the 26 `!s.is_empty()` guards of the dispatch, which went on to the
//! next strategy), "a range restriction proves there is no root" for the
//! `Some(vec![])` of the inversion peeling.  The branches of an inversion
//! were merged by hand, and the merge turned one refused branch next to an
//! empty one into a proof (`|x + i|² = 4` had "no solution" for a real `x`
//! up to 0.48) and roots next to a refused branch into all the roots.  A
//! general solve that met a periodic function of a periodic family said so
//! in a thread-local flag, `PERIOD_CONFLICT`.  Now:
//!
//! - "no root" is [`RootSet::Empty`] with its reason and "nothing found"
//!   is [`Solutions::Refused`]; neither is an empty list
//!   ([`RootSet::Points`] and [`RootSet::Families`] are never empty), and
//!   the type has no `Default`, no `is_empty` and no conversion that
//!   collapses the three cases into `vec![]`.
//! - Alternatives — the branches of an inverse, the factors of a product —
//!   are merged by one rule, [`Solutions::union`]: a refused alternative
//!   makes found roots [`Partial`](Solutions::Partial) and a proof of no
//!   root a refusal; a factor that vanishes identically makes every value
//!   a root.
//! - Nested periodic families are the refusal [`Reason::NestedPeriods`],
//!   which [`union`](Solutions::union) keeps over every other alternative,
//!   so that the general solve that met them refuses as a whole.
//!
//! # Principal and general solves
//!
//! A solve in the principal form ([`Form::Principal`](super::context::Form))
//! claims the roots on the principal branches of the inverse functions:
//! `sin x = 1/2` is `Complete(Points([π/6, 5π/6]))` there, the documented
//! meaning of `Ex::solve`.  A general solve claims every root;
//! [`RootSet::Families`] holds its points and its families (the roots that
//! contain the integer parameter `n`) in the order found.
//!
//! # Roots under a condition on the parameters
//!
//! A root that holds only when the parameters satisfy a condition
//! (`√x = y` has the root `y²` only for `−π/2 < arg y ≤ π/2`, and none
//! otherwise) is a claim no list can carry: [`Reason::Conditional`] holds
//! such roots with their exact condition ([`Guarded`]).  For every
//! consumer of roots it is a refusal (`Ex::solve` refuses, the crate's
//! root searches see no root); [`Solutions::into_set`] writes it as
//! `{y²} ∩ ConditionSet(x, cond)`, the roots when `cond` holds and `∅`
//! otherwise (a `ConditionSet` whose condition does not involve its
//! variable is all of the variable's domain or nothing).  Up to 0.52 the
//! solve was refused and the set was the `ConditionSet` of the equation;
//! up to 0.50 the root was returned for every value (SymPy 1.14:
//! `solve(sqrt(x) - y, x)` → `[y**2]`, wrong for `y = −1`).
//!
//! # Sets
//!
//! [`Solutions::into_set`] is SymPy's `solveset` convention: a complete
//! outcome is its set, a partial one the set united with the condition set
//! `{x : f(x) = 0}`, a refusal the condition set alone (a conditional one
//! its guarded roots).

use std::borrow::Cow;

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};

/// What a solve claims about the roots of `f = 0` in the domain of the
/// unknown (see the [module documentation](self)).
#[must_use]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Solutions {
    /// Exactly these roots (in a principal solve: exactly the roots on the
    /// principal branches); [`RootSet::Empty`] is a proof that there is
    /// none.
    Complete(RootSet),
    /// These solve the equation; others may exist.  The set is
    /// [`Points`](RootSet::Points) or [`Families`](RootSet::Families).
    Partial(RootSet, Missing),
    /// Nothing is claimed.
    Refused(Reason),
}

/// A set of roots (see the [module documentation](self)).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum RootSet {
    /// No root.
    Empty(Reason),
    /// Every value of the unknown (the equation reduces to `0 = 0`).
    All,
    /// Finitely many roots, each once, in the order found; never empty.
    /// Inside the solver the roots of a general solve travel here too,
    /// families included, until the solve sorts them into
    /// [`Families`](Self::Families).
    Points(Vec<ExprId>),
    /// The roots of a general solve, in the order found: points, and
    /// periodic families linear in the integer parameter `n` (the roots
    /// that contain `n`, if any); never empty.
    Families {
        /// Points and families.
        roots: Vec<ExprId>,
        /// The integer parameter of the families.
        n: ExprId,
    },
}

/// What a [`Solutions::Partial`] may lack.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Missing {
    /// An alternative — a branch of an inverse function, a factor of a
    /// product — whose roots were not found (or, for
    /// [`Reason::Conditional`], hold only under a condition).
    Alternative(Reason),
}

impl Missing {
    /// What two alternatives lack together: conditional roots of both are
    /// kept together, and an alternative whose roots are not known at all
    /// wins over conditional ones (the set is then not known).  Up to 0.52
    /// the first of the two was kept, which was enough while every missing
    /// part meant "not known".
    pub(crate) fn merged(a: Option<Missing>, b: Option<Missing>) -> Option<Missing> {
        match (a, b) {
            (None, b) => b,
            (a, None) => a,
            (
                Some(Missing::Alternative(Reason::Conditional(mut g))),
                Some(Missing::Alternative(Reason::Conditional(h))),
            ) => {
                g.extend(h);
                Some(Missing::Alternative(Reason::Conditional(g)))
            }
            (Some(Missing::Alternative(Reason::Conditional(_))), b) => b,
            (a, _) => a,
        }
    }
}

/// Roots that solve the equation exactly when `cond`, a condition on the
/// parameters only, holds (see [`Reason::Conditional`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Guarded {
    /// The condition on the parameters (free of the unknown).
    pub(crate) cond: ExprId,
    /// The roots when it holds: [`RootSet::Points`] or
    /// [`RootSet::Families`].
    pub(crate) roots: RootSet,
}

/// Why a set of roots is empty or a solve claims nothing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Reason {
    /// No strategy applies, or none found a root: nothing is known (the
    /// empty list of 0.48).
    NotFound,
    /// The inverted function does not take the value: a range restriction
    /// (`eˣ = 0`; `eˣ = −1`, `sin f = 2`, `cosh f = 1/2`, `erf f = 2` for a
    /// real `f`; `|f| = −1`) or `f^(−k) = 0`.
    NoPreimage,
    /// A general solve met a periodic function of a periodic family
    /// (`sin(cos x) = 1/2`: `cos x = π/6 + 2nπ`, then `x = ±acos(π/6 +
    /// 2nπ) + 2mπ`), whose solutions need two independent integer
    /// parameters where the solve has one.  Up to 0.48 the thread-local
    /// flag `PERIOD_CONFLICT`.
    NestedPeriods,
    /// In words, for the user.
    Said(Cow<'static, str>),
    /// The roots hold only when a condition on the parameters holds, and
    /// there is no root otherwise: an exact set ([`Solutions::into_set`])
    /// but no list (see the [module documentation](self)).  Never empty.
    Conditional(Vec<Guarded>),
}

impl Reason {
    /// A reason in words.
    pub(crate) fn said(text: impl Into<Cow<'static, str>>) -> Self {
        Reason::Said(text.into())
    }

    /// The reason as a sentence for an error message.
    pub(crate) fn text(&self) -> Cow<'static, str> {
        match self {
            Reason::NotFound => Cow::Borrowed("no strategy found a root"),
            Reason::NoPreimage => {
                Cow::Borrowed("the inverted function does not take the value (range restriction)")
            }
            Reason::NestedPeriods => Cow::Borrowed(
                "the general solution nests periodic families, which need independent integer \
                 parameters",
            ),
            Reason::Said(s) => s.clone(),
            Reason::Conditional(_) => Cow::Borrowed(
                "the roots hold only for some values of the parameters (solve_as_set gives them \
                 with their condition)",
            ),
        }
    }

    /// [`text`](Self::text), naming the roots and conditions of a
    /// [`Reason::Conditional`].
    pub(crate) fn text_in(&self, arena: &Arena) -> Cow<'static, str> {
        let Reason::Conditional(groups) = self else {
            return self.text();
        };
        let shown: Vec<String> = groups
            .iter()
            .map(|g| {
                let roots: Vec<String> = g
                    .roots
                    .roots()
                    .unwrap_or(&[])
                    .iter()
                    .map(|&r| arena.display(r).to_string())
                    .collect();
                format!("{} only when {}", roots.join(", "), arena.display(g.cond))
            })
            .collect();
        Cow::Owned(format!(
            "the roots hold only for some values of the parameters: {} (no root otherwise; \
             solve_as_set gives the conditional set)",
            shown.join("; ")
        ))
    }
}

impl Solutions {
    /// The roots a strategy found: `Complete(Points)`, or
    /// `Refused(NotFound)` when there are none — an empty list is "nothing
    /// found", never a proof (for that see [`exactly`](Self::exactly)).
    pub(crate) fn found(roots: Vec<ExprId>) -> Self {
        if roots.is_empty() {
            Solutions::Refused(Reason::NotFound)
        } else {
            Solutions::Complete(RootSet::Points(roots))
        }
    }

    /// Exactly the roots `roots`: `Complete(Points)`, or
    /// `Complete(Empty(empty))` — a proof — when there are none.
    pub(crate) fn exactly(roots: Vec<ExprId>, empty: Reason) -> Self {
        if roots.is_empty() {
            Solutions::Complete(RootSet::Empty(empty))
        } else {
            Solutions::Complete(RootSet::Points(roots))
        }
    }

    /// No root, for `reason`.
    pub(crate) fn none(reason: Reason) -> Self {
        Solutions::Complete(RootSet::Empty(reason))
    }

    /// Nothing claimed, for `reason`.
    pub(crate) fn refused(reason: Reason) -> Self {
        Solutions::Refused(reason)
    }

    /// Is at least one root claimed (complete or partial)?
    pub(crate) fn has_roots(&self) -> bool {
        self.roots().is_some()
    }

    /// The roots claimed — of a `Complete` or a `Partial` outcome with
    /// points or families — or `None` (no root, every value, refused).
    pub(crate) fn roots(&self) -> Option<&[ExprId]> {
        match self {
            Solutions::Complete(set) | Solutions::Partial(set, _) => set.roots(),
            Solutions::Refused(_) => None,
        }
    }

    /// Every root, when the outcome is complete: `Some` of the roots of
    /// `Complete(Points | Families)`, `Some` of none for a proof that there
    /// is none; `None` for every value, a partial outcome or a refusal.
    /// For the callers that need all the roots (a sign chart, a count
    /// against a degree, the poles of a partial fraction).
    pub(crate) fn points_if_complete(self) -> Option<Vec<ExprId>> {
        match self {
            Solutions::Complete(RootSet::Empty(_)) => Some(Vec::new()),
            Solutions::Complete(set) if set.roots().is_some() => Some(set.into_roots()),
            _ => None,
        }
    }

    /// The roots claimed — all of a complete outcome, the known ones of a
    /// partial one — and none otherwise (no root, every value, refused).
    /// The policy of the crate's root searches up to 0.48, which read
    /// "nothing found" as "no root here": a caller for which a refusal must
    /// not pass for "no root" matches on the outcome instead.
    pub(crate) fn claimed_points(self) -> Vec<ExprId> {
        match self {
            Solutions::Complete(set) | Solutions::Partial(set, _) => set.into_roots(),
            Solutions::Refused(_) => Vec::new(),
        }
    }

    /// The roots claimed and what they may lack (`None`: complete), or the
    /// outcome itself when it claims no root (no root, every value,
    /// refused).
    pub(crate) fn into_found(self) -> Result<(Vec<ExprId>, Option<Missing>), Self> {
        match self {
            Solutions::Complete(set) if set.roots().is_some() => Ok((set.into_roots(), None)),
            Solutions::Partial(set, m) if set.roots().is_some() => Ok((set.into_roots(), Some(m))),
            other => Err(other),
        }
    }

    /// `roots` with what they may lack: `Complete(Points)` when nothing
    /// (`None`), else `Partial`; with no roots, `Refused(NotFound)` or the
    /// refusal of the missing part.
    pub(crate) fn with_missing(roots: Vec<ExprId>, missing: Option<Missing>) -> Self {
        match missing {
            None => Solutions::found(roots),
            Some(Missing::Alternative(reason)) if roots.is_empty() => Solutions::Refused(reason),
            Some(m) => Solutions::Partial(RootSet::Points(roots), m),
        }
    }

    /// The same outcome, lacking `missing` too: an outcome derived from a
    /// partial one (the roots in `x` of a substitution whose roots in `t`
    /// are partial) is partial, or refused when it has no root.
    pub(crate) fn lacking(self, missing: Option<Missing>) -> Self {
        match missing {
            None => self,
            Some(Missing::Alternative(reason)) => self.union(Solutions::Refused(reason)),
        }
    }

    /// Is this a proof that there is no root?
    pub(crate) fn is_none_proved(&self) -> bool {
        matches!(self, Solutions::Complete(RootSet::Empty(_)))
    }

    /// Is this the refusal of a general solve that met nested periodic
    /// families ([`Reason::NestedPeriods`]), which must reach the solve
    /// it belongs to?
    pub(crate) fn is_nested_periods(&self) -> bool {
        matches!(self, Solutions::Refused(Reason::NestedPeriods))
    }

    /// The same claim about the roots `f(roots)` (each once, in order):
    /// for a change of variable `x = g(t)`, whose roots in `x` are the
    /// images of those in `t`.  No root, every value and refusals are
    /// unchanged.
    pub(crate) fn map_roots(self, mut f: impl FnMut(ExprId) -> ExprId) -> Self {
        let mut map = |roots: Vec<ExprId>| -> Vec<ExprId> {
            let mut out: Vec<ExprId> = Vec::with_capacity(roots.len());
            for r in roots {
                let v = f(r);
                if !out.contains(&v) {
                    out.push(v);
                }
            }
            out
        };
        // conditional roots are roots too (their conditions are on the
        // parameters only, unchanged by a change of the unknown)
        match self {
            Solutions::Complete(set) => Solutions::Complete(set.map_roots(&mut map)),
            Solutions::Partial(set, Missing::Alternative(reason)) => {
                let set = set.map_roots(&mut map);
                Solutions::Partial(set, Missing::Alternative(map_guarded(reason, &mut map)))
            }
            Solutions::Refused(reason) => Solutions::Refused(map_guarded(reason, &mut map)),
        }
    }

    /// The same outcome holding only when `cond` (a condition on the
    /// parameters) does, and with no root otherwise: its roots become
    /// [`Reason::Conditional`] (conditions already there are joined with
    /// `cond`), a proof of no root stays one, and an unknown part stays
    /// unknown.  For an inversion that takes a principal branch whose range
    /// the right-hand side meets only for some parameter values (`√f = y`:
    /// `f = y²` when `−π/2 < arg y ≤ π/2`).  `cond` is simplified first
    /// (`logic::simplify_bool`): one that holds for every value leaves the
    /// outcome as it is, one that never holds leaves no root (up to 0.54
    /// `√x = y` for an imaginary `y` was refused naming the condition `0 > 0 ∨
    /// (0 = 0 ∧ −i·y ≥ 0)`).
    pub(crate) fn under(self, arena: &mut Arena, cond: ExprId) -> Self {
        let cond = crate::transforms::logic::simplify_bool(arena, cond);
        if cond == arena.bool_true {
            return self;
        }
        if cond == arena.bool_false {
            return Solutions::none(Reason::said(
                "the principal branch never reaches the right-hand side",
            ));
        }
        let guard = |arena: &mut Arena, reason: Reason| match reason {
            Reason::Conditional(groups) => Reason::Conditional(
                groups
                    .into_iter()
                    .map(|g| Guarded {
                        cond: if g.cond == cond {
                            cond
                        } else {
                            arena.and(&[cond, g.cond])
                        },
                        roots: g.roots,
                    })
                    .collect(),
            ),
            other => other,
        };
        match self {
            Solutions::Complete(RootSet::Empty(reason)) => {
                Solutions::Complete(RootSet::Empty(reason))
            }
            Solutions::Complete(RootSet::All) => Solutions::Refused(Reason::NotFound),
            Solutions::Complete(roots) => {
                Solutions::Refused(Reason::Conditional(vec![Guarded { cond, roots }]))
            }
            Solutions::Partial(roots, Missing::Alternative(reason)) => {
                let known = Solutions::Refused(Reason::Conditional(vec![Guarded { cond, roots }]));
                known.union(Solutions::Refused(guard(arena, reason)))
            }
            Solutions::Refused(reason) => Solutions::Refused(guard(arena, reason)),
        }
    }

    /// Is this a [`Reason::Conditional`] refusal (roots under a condition,
    /// none otherwise)?
    pub(crate) fn is_conditional(&self) -> bool {
        matches!(self, Solutions::Refused(Reason::Conditional(_)))
    }

    /// The same outcome with conditional roots ([`Reason::Conditional`])
    /// taken as not found: for the outcome of a substituted equation (`t =
    /// eˣ`, `t = tan(x/2)`, `t = x^(1/L)`) whose roots are mapped back by
    /// solving `g(x) = t₀` rather than by a map of the roots, so that a
    /// conditional root in `t` is never read as one in `x`.
    pub(crate) fn without_conditions(self) -> Self {
        let plain = |reason: Reason| match reason {
            Reason::Conditional(_) => Reason::NotFound,
            other => other,
        };
        match self {
            Solutions::Partial(set, Missing::Alternative(reason)) => {
                Solutions::Partial(set, Missing::Alternative(plain(reason)))
            }
            Solutions::Refused(reason) => Solutions::Refused(plain(reason)),
            complete => complete,
        }
    }

    /// Union over alternatives (branches of an inverse function, factors of
    /// a product, members of a family of right-hand sides): the one place
    /// that decides how claims combine.
    ///
    /// - [`Reason::NestedPeriods`] on either side wins: the general solve
    ///   that met it refuses as a whole.
    /// - Every value on either side ([`RootSet::All`], a factor that
    ///   vanishes identically) wins next.
    /// - Two complete outcomes give the union of their roots (`self`'s
    ///   first, each root once); no root on both sides keeps `self`'s
    ///   reason.
    /// - A refused or partial side makes found roots partial, and no root
    ///   a refusal: one unsolved alternative may hold roots.  Up to 0.48
    ///   the hand-written merge of `peel_two_branches` made "no root"
    ///   next to a refused branch a proof (`|x + i|² = 4` for a real `x`:
    ///   the branch `|x + i| = −2` is empty, `|x + i| = 2` refused, and
    ///   `±√3` solve it).
    pub(crate) fn union(self, other: Self) -> Self {
        if self.is_nested_periods() {
            return self;
        }
        if other.is_nested_periods() {
            return other;
        }
        let all = |s: &Solutions| matches!(s, Solutions::Complete(RootSet::All));
        if all(&self) {
            return self;
        }
        if all(&other) {
            return other;
        }
        let (a, a_missing) = self.into_parts();
        let (b, b_missing) = other.into_parts();
        let set = match (a, b) {
            (Some(a), Some(b)) => Some(a.union(b)),
            (a, b) => a.or(b),
        };
        let missing = Missing::merged(a_missing, b_missing);
        match (set, missing) {
            (Some(set), None) => Solutions::Complete(set),
            (Some(set), Some(m)) if set.roots().is_some() => Solutions::Partial(set, m),
            (_, Some(Missing::Alternative(reason))) => Solutions::Refused(reason),
            // (unreachable: a side without a set carries a missing part)
            (None, None) => Solutions::Refused(Reason::NotFound),
        }
    }

    /// The set this outcome claims (or `None` for a refusal), and what it
    /// may lack.
    fn into_parts(self) -> (Option<RootSet>, Option<Missing>) {
        match self {
            Solutions::Complete(set) => (Some(set), None),
            Solutions::Partial(set, m) => (Some(set), Some(m)),
            Solutions::Refused(reason) => (None, Some(Missing::Alternative(reason))),
        }
    }

    /// The roots of a general solve in the integer parameter `n` as
    /// [`RootSet::Families`] (the end of a general solve; see
    /// [`RootSet::Points`]).
    pub(crate) fn with_families(self, n: ExprId) -> Self {
        let sort = |set: RootSet| match set {
            RootSet::Points(roots) => RootSet::Families { roots, n },
            other => other,
        };
        let sort_reason = |reason: Reason| match reason {
            Reason::Conditional(groups) => Reason::Conditional(
                groups
                    .into_iter()
                    .map(|g| Guarded {
                        cond: g.cond,
                        roots: sort(g.roots),
                    })
                    .collect(),
            ),
            other => other,
        };
        match self {
            Solutions::Complete(set) => Solutions::Complete(sort(set)),
            Solutions::Partial(set, Missing::Alternative(reason)) => {
                Solutions::Partial(sort(set), Missing::Alternative(sort_reason(reason)))
            }
            Solutions::Refused(reason) => Solutions::Refused(sort_reason(reason)),
        }
    }

    /// The solution set of `expr = 0` in `var`, SymPy's `solveset`
    /// convention: a complete outcome is its set (`EmptySet`, the universal
    /// set, a finite set, the finite set of the points united with an image
    /// set `{f(n) : n ∈ ℤ}` per family, simplified); a partial one that set
    /// united with `ConditionSet(var, Eq(expr, 0))`; a refusal the condition
    /// set alone.  Roots under a condition ([`Reason::Conditional`]) are
    /// `roots ∩ ConditionSet(var, cond)`, in place of that condition set.
    pub(crate) fn into_set(self, arena: &mut Arena, var: ExprId, expr: ExprId) -> ExprId {
        let condition = |arena: &mut Arena| {
            let zero = arena.zero;
            let cond = arena.eq_(expr, zero);
            arena.intern(ExprNode::ConditionSet(var, cond))
        };
        let unknown = |arena: &mut Arena, reason: Reason| match reason {
            Reason::Conditional(groups) => guarded_set(arena, var, groups),
            _ => condition(arena),
        };
        match self {
            Solutions::Complete(set) => set.into_set(arena),
            Solutions::Partial(set, Missing::Alternative(reason)) => {
                let known = set.into_set(arena);
                let unknown = unknown(arena, reason);
                arena.set_union(&[known, unknown])
            }
            Solutions::Refused(reason) => unknown(arena, reason),
        }
    }
}

/// `reason` with the roots of a [`Reason::Conditional`] mapped by `map`.
fn map_guarded(reason: Reason, map: &mut dyn FnMut(Vec<ExprId>) -> Vec<ExprId>) -> Reason {
    match reason {
        Reason::Conditional(groups) => Reason::Conditional(
            groups
                .into_iter()
                .map(|g| Guarded {
                    cond: g.cond,
                    roots: g.roots.map_roots(&mut |r| map(r)),
                })
                .collect(),
        ),
        other => other,
    }
}

/// The set of conditional roots: `⋃ roots ∩ ConditionSet(var, cond)`.
fn guarded_set(arena: &mut Arena, var: ExprId, groups: Vec<Guarded>) -> ExprId {
    let mut parts: Vec<ExprId> = Vec::with_capacity(groups.len());
    for g in groups {
        let cond = crate::transforms::logic::simplify_bool(arena, g.cond);
        if cond == arena.bool_false {
            continue;
        }
        let roots = g.roots.into_set(arena);
        if cond == arena.bool_true {
            parts.push(roots);
            continue;
        }
        let when = arena.intern(ExprNode::ConditionSet(var, cond));
        parts.push(arena.set_intersection(&[roots, when]));
    }
    arena.set_union(&parts)
}

impl RootSet {
    /// The roots of [`Points`](Self::Points) or
    /// [`Families`](Self::Families).
    pub(crate) fn roots(&self) -> Option<&[ExprId]> {
        match self {
            RootSet::Points(roots) | RootSet::Families { roots, .. } if !roots.is_empty() => {
                Some(roots)
            }
            _ => None,
        }
    }

    /// The roots of [`Points`](Self::Points) or
    /// [`Families`](Self::Families), owned (empty for the others: callers
    /// test [`roots`](Self::roots) first).
    fn into_roots(self) -> Vec<ExprId> {
        match self {
            RootSet::Points(roots) | RootSet::Families { roots, .. } => roots,
            RootSet::Empty(_) | RootSet::All => Vec::new(),
        }
    }

    /// The union of two sets of roots (`self`'s first, each root once).
    fn union(self, other: Self) -> Self {
        match (self, other) {
            (RootSet::All, _) | (_, RootSet::All) => RootSet::All,
            (empty @ RootSet::Empty(_), RootSet::Empty(_)) => empty,
            (RootSet::Empty(_), set) | (set, RootSet::Empty(_)) => set,
            (RootSet::Points(a), RootSet::Points(b)) => RootSet::Points(merged(a, b)),
            (RootSet::Families { roots, n }, RootSet::Points(b))
            | (RootSet::Families { roots, n }, RootSet::Families { roots: b, .. }) => {
                RootSet::Families {
                    roots: merged(roots, b),
                    n,
                }
            }
            (RootSet::Points(a), RootSet::Families { roots, n }) => RootSet::Families {
                roots: merged(a, roots),
                n,
            },
        }
    }

    /// The set with its roots mapped by `map` (no root and every value
    /// unchanged).
    fn map_roots(self, map: &mut impl FnMut(Vec<ExprId>) -> Vec<ExprId>) -> Self {
        match self {
            RootSet::Points(roots) => RootSet::Points(map(roots)),
            RootSet::Families { roots, n } => RootSet::Families {
                roots: map(roots),
                n,
            },
            other => other,
        }
    }

    /// The set of these roots (see [`Solutions::into_set`]).
    fn into_set(self, arena: &mut Arena) -> ExprId {
        match self {
            RootSet::Empty(_) => arena.empty_set,
            RootSet::All => arena.universal_set,
            RootSet::Points(roots) => arena.finite_set(&roots),
            RootSet::Families { roots, n } => {
                let mut plain: Vec<ExprId> = Vec::new();
                let mut parts: Vec<ExprId> = Vec::new();
                for r in roots {
                    let v = crate::transforms::eval::eval(arena, r);
                    if crate::base::walk::has_free_var(arena, v, n) {
                        parts.push(crate::transforms::sets::image_set(arena, n, v));
                    } else {
                        plain.push(v);
                    }
                }
                if !plain.is_empty() {
                    parts.push(arena.finite_set(&plain));
                }
                let union = arena.set_union(&parts);
                crate::transforms::sets::simplify_set(arena, union)
            }
        }
    }
}

/// `a` followed by the roots of `b` not in it, each root once.
fn merged(a: Vec<ExprId>, b: Vec<ExprId>) -> Vec<ExprId> {
    let mut out: Vec<ExprId> = Vec::with_capacity(a.len() + b.len());
    for r in a.into_iter().chain(b) {
        if !out.contains(&r) {
            out.push(r);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn points(arena: &mut Arena, ks: &[i64]) -> Vec<ExprId> {
        ks.iter().map(|&k| arena.int(k)).collect()
    }

    #[test]
    fn union_of_alternatives() {
        let mut arena = Arena::new();
        let p12 = points(&mut arena, &[1, 2]);
        let p23 = points(&mut arena, &[2, 3]);
        let p123 = points(&mut arena, &[1, 2, 3]);
        let empty = || Solutions::none(Reason::NoPreimage);
        let refused = || Solutions::refused(Reason::NotFound);
        // complete ∪ complete: the union, `self` first, each root once
        assert_eq!(
            Solutions::found(p12.clone()).union(Solutions::found(p23.clone())),
            Solutions::Complete(RootSet::Points(p123.clone()))
        );
        // no root ∪ no root: no root; no root ∪ roots: the roots
        assert_eq!(empty().union(empty()), empty());
        assert_eq!(
            empty().union(Solutions::found(p12.clone())),
            Solutions::found(p12.clone())
        );
        // no root ∪ refused: refused (the proof needs every branch)
        assert_eq!(empty().union(refused()), refused());
        assert_eq!(refused().union(empty()), refused());
        // roots ∪ refused: partial
        assert_eq!(
            Solutions::found(p12.clone()).union(refused()),
            Solutions::Partial(
                RootSet::Points(p12.clone()),
                Missing::Alternative(Reason::NotFound)
            )
        );
        // partial ∪ complete stays partial
        let partial = Solutions::found(p12.clone()).union(refused());
        assert!(matches!(
            partial.union(Solutions::found(p23)),
            Solutions::Partial(RootSet::Points(ref r), _) if *r == p123
        ));
        // every value wins over a refusal, nested periods over everything
        let all = Solutions::Complete(RootSet::All);
        assert_eq!(refused().union(all.clone()), all);
        let nested = Solutions::refused(Reason::NestedPeriods);
        assert_eq!(all.clone().union(nested.clone()), nested);
        assert_eq!(Solutions::found(p12).union(nested.clone()), nested);
    }

    #[test]
    fn found_and_exactly_differ_on_no_roots() {
        assert_eq!(
            Solutions::found(Vec::new()),
            Solutions::Refused(Reason::NotFound)
        );
        assert!(Solutions::exactly(Vec::new(), Reason::NoPreimage).is_none_proved());
        assert!(!Solutions::found(Vec::new()).has_roots());
    }
}
