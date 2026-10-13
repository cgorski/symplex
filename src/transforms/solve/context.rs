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
//! | `Ex::solve` (principal roots in the declared domain) | `SolveCx::principal(Domain::Complex)` | `solve_classified` |
//! | the crate's root searches (integrate, limit, residues, ODEs, …), the equations of `reduce_inequalities` | `SolveCx::principal(Domain::Real)` | `solve`, `solve_classified_real`, `solve_with` |
//! | `Ex::solve_general`, `periodic_candidates` | `SolveCx::general(Domain::Real, n)` | `solve_general` |
//! | `Ex::solve_as_set` | `SolveCx::general(Domain::Complex, n)` | `solve_general_over_c` |
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
//! - In a principal solve over the declared domain (`Ex::solve`) only the
//!   range restrictions ([`SolveCx::range_restricted`]) honour the
//!   declared domain; |·|, the |·| cases, Lambert W and the periodic
//!   realness tests take the unknown real there
//!   ([`SolveCx::real_for_real_var`], [`SolveCx::variable_may_be_nonreal`]),
//!   and the families searched when every principal candidate fails take
//!   it real too ([`SolveCx::families`]).
//! - The split of two exponentials tests the realness of their exponent
//!   difference only in such a solve ([`SolveCx::declared_principal`]).
//! - Parameters (other symbols without assumptions) are taken real
//!   ([`Params::Real`]).
//! - The outcome side channel `PERIOD_CONFLICT` (a general solve that
//!   nests periodic families) stays a thread-local of `solve.rs`.

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
    /// convention).
    #[expect(
        dead_code,
        reason = "every solve still takes its parameters real (declared parameters are \
                  planned)"
    )]
    AsDeclared,
    /// A symbol without assumptions is taken real (every solve up to 0.48).
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
    /// unknown (Lambert W's real branches, the |·| cases, the split of two
    /// exponentials), in the domain [`real_for_real_var`] uses?
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

    /// Is this a principal solve over the declared domain whose other arms
    /// take the unknown real (`Ex::solve`, see the [module
    /// documentation](self))?  The split of two exponentials tests the
    /// realness of their exponent difference only here.
    pub(crate) fn declared_principal(&self) -> bool {
        self.unknown == Domain::Complex && self.legacy_unknown().is_real()
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
