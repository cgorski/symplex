//! The context of one solve: where the unknown ranges ([`Domain`]), how the
//! other symbols are taken ([`Params`]), and whether principal roots or
//! periodic families are wanted ([`Form`]), carried as one value,
//! [`SolveCx`], through the solver.
//!
//! Up to 0.48 this was three thread-local flags of `solve.rs`
//! (`DECLARED_UNKNOWN`, `COMPLEX_VARIABLE`, `GENERAL_SOLVE`) and the
//! `period: Option<ExprId>` parameter, which besides carrying the integer
//! parameter of the families also switched the domain: the range
//! restrictions honoured the declared domain of the unknown only when it was
//! `None`.  Each peeling arm picked one of four realness predicates by hand,
//! and `inequalities::real_roots`, a question about the real line, called
//! the declared-domain solver and got real roots only because the arms it
//! reached took the unknown real anyway.  Now the entry point states the
//! domain, the two ways into a sub-solve are named ([`SolveCx::families`],
//! [`SolveCx::substituted`]), and an arm asks
//! [`SolveCx::range_restricted`] whether a range restriction may be used.
//!
//! # Which context
//!
//! | caller | context | entry point in `solve.rs` |
//! |---|---|---|
//! | `Ex::solve` (principal roots in the declared domain) | `SolveCx::principal(Domain::Complex)`, [`Params::AsDeclared`] | `solve_classified` |
//! | the crate's root searches (integrate, limit, residues, ODEs, …), the equations of `reduce_inequalities` | `SolveCx::principal(Domain::Real)` (`.as_candidates()` for `known_real_roots`: [`SolveCx::conditional_roots_as_candidates`]) | `known_real_roots`, `solve_classified_real`, `solutions` |
//! | the real zeros of a periodic function (sign charts, extrema), `periodic_candidates` | `SolveCx::general(Domain::Real, n)` | `solve_general` |
//! | `Ex::solve_general` | `SolveCx::general(Domain::declared(var), n)`, [`Params::AsDeclared`] | `solve_general_declared` |
//! | `Ex::solve_as_set` | `SolveCx::general(Domain::Complex, n)`, [`Params::AsDeclared`] | `solve_general_over_c` |
//!
//! # Domains
//!
//! A [`Domain`] is intersected with the declared assumptions of the
//! unknown: `Domain::Complex` is the declared domain itself (ℂ for a plain
//! symbol, ℝ for one declared real), `Domain::Real` adds realness to the
//! declaration.  A substituted unknown (`t = eˣ`, `u = r·x`) takes the same
//! domain relative to its own declaration.  Narrower declared sets
//! (positive, integer) stay a post-filter (`restrict_to_domain`).
//!
//! # Conventions kept from 0.48 (to be removed)
//!
//! The context also carries the conventions of 0.48 that the domain alone
//! does not explain; each is a named method, so a call site says which one
//! it relies on:
//!
//! - In a principal solve over the declared domain (`Ex::solve`) the range
//!   restrictions ([`SolveCx::range_restricted`]) and |·| and the |·| cases
//!   ([`SolveCx::real`], [`SolveCx::unknown_may_be_nonreal`], since 0.49)
//!   honour the declared domain; Lambert W and the periodic realness tests
//!   take the unknown real there ([`SolveCx::real_for_real_var`],
//!   [`SolveCx::variable_may_be_nonreal`]), and the families searched when
//!   every principal candidate fails take it real too
//!   ([`SolveCx::families`]).

//! - Parameters (other symbols without assumptions) are taken real
//!   ([`Params::Real`]) by the crate's root searches; the public solvers
//!   take them as declared ([`Params::AsDeclared`], since 0.50: a plain
//!   `a` may be complex, so `e^(a·x) = −1` for a real `x` is not a
//!   contradiction — `a = iπ`, `x = 1` solves it).
//!
//! (A general solve that nests periodic families refuses with the outcome
//! `Reason::NestedPeriods`, up to 0.48 the thread-local flag
//! `PERIOD_CONFLICT`.)

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};

/// Where the unknown of a solve ranges, intersected with its declared
/// assumptions (see the [module documentation](self)).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Domain {
    /// The declared domain of the unknown: ℂ for a plain symbol.
    Complex,
    /// The real line.
    Real,
    /// The integers.
    #[expect(
        dead_code,
        reason = "no entry point asks for it yet (planned: the summation index, a public \
                  `Ex::solutions(var, Domain)`)"
    )]
    Integer,
}

impl Domain {
    /// The declared domain of the unknown `var`: [`Domain::Real`] for a
    /// symbol declared real (positive, integer, …), [`Domain::Complex`]
    /// otherwise.
    pub(crate) fn declared(arena: &Arena, var: ExprId) -> Domain {
        use crate::base::assumptions::Props;
        let ExprNode::Symbol(sid) = *arena.node(var) else {
            return Domain::Complex;
        };
        let mut declared = arena.symbol_assumptions(sid);
        declared.forward_chain();
        if declared.query(Props::REAL) == Some(true) {
            Domain::Real
        } else {
            Domain::Complex
        }
    }

    /// Is the unknown taken real (realness added to its declaration)?
    fn is_real(self) -> bool {
        !matches!(self, Domain::Complex)
    }
}

/// How the symbols other than the unknown are taken by the realness tests
/// of a solve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Params {
    /// As declared: a symbol without assumptions is complex (integration's
    /// convention; the public solvers since 0.50).
    AsDeclared,
    /// A symbol without assumptions is taken real (every solve up to 0.49;
    /// the crate's root searches).
    Real,
}

/// Principal roots or periodic families.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Form {
    /// The principal roots (principal branches of the inverse functions).
    Principal,
    /// Every root, periodic families written in the fresh integer
    /// parameter `n` (`asin c + 2nπ`, `π − asin c + 2nπ`).
    General {
        /// The integer parameter of the families.
        n: ExprId,
    },
}

/// The entry point a solve started from, on which the conventions listed in
/// the [module documentation](self) are keyed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Request {
    /// The unknown real throughout (the crate's root searches,
    /// `solve_general`).
    Real,
    /// Principal roots over the declared domain (`Ex::solve`), honoured by
    /// the range restrictions only.
    PrincipalDeclared,
    /// A solution set over the declared domain (`Ex::solve_as_set`),
    /// honoured by every arm.
    SetDeclared,
}

/// The context of a solve: the domain of the unknown, the convention for
/// the parameters, principal or general form (see the [module
/// documentation](self)).
///
/// Built by [`principal`](Self::principal) or [`general`](Self::general) at
/// an entry point and passed by reference; a sub-solve gets its own context
/// from [`families`](Self::families) or [`substituted`](Self::substituted).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SolveCx {
    unknown: Domain,
    params: Params,
    form: Form,
    request: Request,
    /// A solution set is wanted (a general solve, or a substituted equation
    /// solved inside one): the Lambert W strategies then also take the other
    /// real branch.
    all_branches: bool,
    /// Roots that hold only for some values of the parameters may be
    /// returned ([`as_candidates`](Self::as_candidates)).
    candidates: bool,
}

impl SolveCx {
    /// A principal solve with the unknown in `unknown`.
    pub(crate) fn principal(unknown: Domain) -> Self {
        SolveCx {
            unknown,
            params: Params::Real,
            form: Form::Principal,
            request: if unknown == Domain::Complex {
                Request::PrincipalDeclared
            } else {
                Request::Real
            },
            all_branches: false,
            candidates: false,
        }
    }

    /// A general solve with the unknown in `unknown`, the families written
    /// in the integer parameter `n`.
    pub(crate) fn general(unknown: Domain, n: ExprId) -> Self {
        SolveCx {
            unknown,
            params: Params::Real,
            form: Form::General { n },
            request: if unknown == Domain::Complex {
                Request::SetDeclared
            } else {
                Request::Real
            },
            all_branches: true,
            candidates: false,
        }
    }

    /// The context of the periodic families of the equation of this solve,
    /// in the integer parameter `n` (the candidates a principal solve
    /// searches when every principal one fails, or that a caller restricts
    /// to a window).  The unknown is taken real unless the request is a
    /// solution set over the declared domain.
    pub(crate) fn families(&self, n: ExprId) -> Self {
        SolveCx {
            unknown: self.legacy_unknown(),
            form: Form::General { n },
            all_branches: true,
            ..*self
        }
    }

    /// The context of a substituted equation solved for its own unknown
    /// (`t = eˣ`, `t = tan(x/2)`, `t = x^(1/L)`) by a principal solve: the
    /// domain of the request's principal roots, also inside a general solve.
    pub(crate) fn substituted(&self) -> Self {
        SolveCx {
            unknown: if self.request == Request::Real {
                Domain::Real
            } else {
                Domain::Complex
            },
            form: Form::Principal,
            ..*self
        }
    }

    /// This context with the other symbols taken per `params` (the
    /// constructors take them real, [`Params::Real`]).
    pub(crate) fn with_params(self, params: Params) -> Self {
        SolveCx { params, ..self }
    }

    /// This context with the unknown taken real: for an equation whose
    /// every solution is real (`|x − π| + x = 0`, i.e. `x = −|x − π|`).
    pub(crate) fn with_real_unknown(self) -> Self {
        SolveCx {
            unknown: Domain::Real,
            ..self
        }
    }

    /// The integer parameter of a general solve.
    pub(crate) fn period(&self) -> Option<ExprId> {
        match self.form {
            Form::General { n } => Some(n),
            Form::Principal => None,
        }
    }

    /// Is a solution set wanted (Lambert W's other real branch)?
    pub(crate) fn all_branches(&self) -> bool {
        self.all_branches
    }

    /// This context for a caller that takes the roots as candidate points
    /// ([`conditional_roots_as_candidates`](Self::conditional_roots_as_candidates)).
    pub(crate) fn as_candidates(self) -> Self {
        SolveCx {
            candidates: true,
            ..self
        }
    }

    /// May a root that holds only for some values of the parameters be
    /// returned (`√x = y` gives `x = y²`, a root only for `−π/2 < arg y ≤
    /// π/2`)?  Only for the crate's root searches through `known_real_roots`
    /// ([`as_candidates`](Self::as_candidates)): they take the roots as
    /// candidate points, which they use or check (the ODE solver verifies
    /// each family against the equation: `2∛(3(C1 + y)/8) = x + C2` gives
    /// `y = (x + C2)³/3 − C1`, a solution of `y″² = 4y′` for every `x`).
    /// The public solvers (`Ex::solve`, `reduce_inequalities`) and the
    /// callers that need every root and only roots (residues, partial
    /// fractions) refuse instead, since 0.51: their roots are claims.
    /// Since 0.53 the solver carries such a root with its exact condition
    /// on the parameters where it has one (`Reason::Conditional`):
    /// `Ex::solve` still refuses, `Ex::solve_as_set` gives `{y²} ∩
    /// ConditionSet(x, y ≥ 0)` for `√x = y` with a real `y`.
    pub(crate) fn conditional_roots_as_candidates(&self) -> bool {
        self.candidates
    }

    /// May the range restriction of an inversion `g(f) = c` be used —
    /// `exp f = c ≤ 0`, `sin f = c` or `cos f = c` with `|c| > 1`,
    /// `cosh f = c < 1` have no solution for a real `f`: is `f` certainly
    /// real for the unknown `var` in [`unknown`](Self::unknown), the other
    /// symbols taken per the context's [`Params`]?  Under `Ex::solve` a
    /// plain `x` is complex, and `eˣ = −1` has the principal root `iπ`
    /// rather than no solution.  (Before 0.31 the restrictions were applied
    /// to any argument, and `exp(i·x) = −1` (solved by `x = π`),
    /// `cos(i·x) = 2` (`x = ±acosh 2`), `exp(√x) = −1` (`x = −π²`) were
    /// reported to have no solution.)
    pub(crate) fn range_restricted(&self, arena: &mut Arena, f: ExprId, var: ExprId) -> bool {
        self.real_over(arena, f, var, self.unknown)
    }

    /// Is `f` certainly real for the unknown `var` in
    /// [`unknown`](Self::unknown) (the other symbols per the context's
    /// [`Params`])?  The test of the arms whose inversion holds for a real
    /// argument only: `|f| = c` has the two solutions `f = ±c` for a real
    /// `f`, while a complex `f` has modulus `c` on a whole curve (`|x| = 1`
    /// for a plain `x`: the unit circle).  Up to 0.48 these arms took the
    /// unknown real under `Ex::solve` ([`real_for_real_var`]), and `|x| = 1`
    /// gave `[1, −1]`, `|eˣ| = 2` gave `[ln 2, iπ + ln 2]` (the solutions:
    /// the line `Re x = ln 2`) as all the roots.
    ///
    /// [`real_for_real_var`]: Self::real_for_real_var
    pub(crate) fn real(&self, arena: &mut Arena, f: ExprId, var: ExprId) -> bool {
        self.real_over(arena, f, var, self.unknown)
    }

    /// Is `var` possibly non-real in [`unknown`](Self::unknown) (a plain
    /// symbol under `Ex::solve`)?  The gate of the strategies that assume a
    /// real unknown and honour the declared domain (the |·| cases).
    pub(crate) fn unknown_may_be_nonreal(&self, arena: &Arena, var: ExprId) -> bool {
        use crate::base::assumptions::Props;
        if self.unknown.is_real() {
            return false;
        }
        match *arena.node(var) {
            ExprNode::Symbol(sid) => arena.symbol_assumptions(sid).query(Props::REAL) != Some(true),
            _ => true,
        }
    }

    /// Is `e` certainly real for the unknown `var` taken real, unless the
    /// request is a solution set over the declared domain — the realness
    /// test of the arms that do not honour [`unknown`](Self::unknown) in a
    /// principal solve over the declared domain (see the [module
    /// documentation](self)): `|f| = c` (only a real `f` has the two
    /// solutions `±c`), the period `2πi` of `exp`, `sinh`, `cosh`, `tanh`
    /// of a non-real argument, a constant base `a^f` that is not `exp` of a
    /// real exponent.  In a general solve it is
    /// [`range_restricted`](Self::range_restricted).
    pub(crate) fn real_for_real_var(&self, arena: &mut Arena, e: ExprId, var: ExprId) -> bool {
        self.real_over(arena, e, var, self.legacy_unknown())
    }

    /// Is `var` possibly non-real for the strategies that assume a real
    /// unknown (Lambert W's real branches, the periodic atoms of a change of
    /// variable), in the domain [`real_for_real_var`] uses?
    ///
    /// [`real_for_real_var`]: Self::real_for_real_var
    pub(crate) fn variable_may_be_nonreal(&self, arena: &Arena, var: ExprId) -> bool {
        use crate::base::assumptions::Props;
        if self.legacy_unknown().is_real() {
            return false;
        }
        match *arena.node(var) {
            ExprNode::Symbol(sid) => arena.symbol_assumptions(sid).query(Props::REAL) != Some(true),
            _ => true,
        }
    }

    /// The domain of the unknown in [`real_for_real_var`] and
    /// [`families`]: real unless the request is a solution set over the
    /// declared domain.
    ///
    /// [`real_for_real_var`]: Self::real_for_real_var
    /// [`families`]: Self::families
    fn legacy_unknown(&self) -> Domain {
        if self.request == Request::SetDeclared {
            Domain::Complex
        } else {
            Domain::Real
        }
    }

    /// An assumption cache for the facts of a value free of the unknown
    /// (the right-hand side of an inversion): its symbols as declared, and
    /// those without declared assumptions taken real under
    /// [`Params::Real`] (the root searches and `reduce_inequalities`, where
    /// `ln x = a` is `x = e^a` for every real `a`).
    pub(crate) fn parameter_cache(
        &self,
        arena: &Arena,
        e: ExprId,
    ) -> crate::base::assumptions::AssumptionCache {
        use crate::base::assumptions::{AssumptionCache, Assumptions, Props};
        let mut cache = AssumptionCache::new();
        if self.params == Params::Real {
            for s in crate::base::walk::free_symbols(arena, e) {
                if let ExprNode::Symbol(sid) = *arena.node(s)
                    && arena.symbol_assumptions(sid) == Assumptions::default()
                {
                    let mut real = Assumptions::default();
                    real.known_true |= Props::REAL;
                    cache.set_symbol_assumptions(s, real);
                }
            }
        }
        cache
    }

    /// Is `e` certainly real for the unknown `var` in `unknown` (intersected
    /// with its declaration), the other symbols without declared
    /// assumptions taken real under [`Params::Real`]?  Decided by the
    /// assumption system: `exp(x)`, `cos(2x + 1)`, `|x|` are real for a
    /// real `x`, `I·x`, `√x` and `ln x` are not known to be.
    fn real_over(&self, arena: &mut Arena, e: ExprId, var: ExprId, unknown: Domain) -> bool {
        use crate::base::assumptions::{AssumptionCache, Assumptions, Props};
        let mut cache = AssumptionCache::new();
        for s in crate::base::walk::free_symbols(arena, e) {
            let ExprNode::Symbol(sid) = *arena.node(s) else {
                continue;
            };
            let declared = arena.symbol_assumptions(sid);
            let take_real = if s == var {
                unknown.is_real()
            } else {
                self.params == Params::Real && declared == Assumptions::default()
            };
            if take_real {
                let mut real = declared;
                real.known_true |= Props::REAL;
                if !real.is_contradictory() {
                    cache.set_symbol_assumptions(s, real);
                }
            }
        }
        cache.query(arena, e, Props::REAL) == Some(true)
    }
}
