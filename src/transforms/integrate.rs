//! Symbolic integration (antiderivatives).
//!
//! This module implements [`integrate`], which computes the indefinite
//! integral of an expression with respect to a symbol.
//!
//! # Supported integrands
//!
//! - **Power rule:** `∫ x^n dx = x^(n+1)/(n+1)` for n ≠ -1
//! - **Logarithmic:** `∫ x^(-1) dx = ln(|x|)`
//! - **Trigonometric:** `∫ sin(x) dx = -cos(x)`, `∫ cos(x) dx = sin(x)`
//! - **Tangent:** `∫ tan(x) dx = -ln|cos(x)|`
//! - **Natural log:** `∫ ln(x) dx = x·ln(x) - x`
//! - **Exponential:** `∫ exp(x) dx = exp(x)`
//! - **Linearity:** `∫ (f + g) dx = ∫f dx + ∫g dx`
//! - **Constant factor:** `∫ c·f dx = c · ∫f dx` (when c is independent of x)
//! - **Constants:** `∫ c dx = c·x`
//! - **Standard forms:** `∫ 1/(x²+a²) dx = (1/a)·atan(x/a)`, etc.
//! - **Partial fractions:** apart→integrate pipeline for rational integrands
//! - **Special functions:** `∫ f(a·x + b)·R(x)` with `f` = `exp`, `sin`, `cos`,
//!   `sinh`, `cosh` and linear denominators in `R` (`Ei`, `Si`/`Ci`,
//!   `Shi`/`Chi`), `∫ ln(a·x + b)⁻ⁿ` (`li`), polynomials times `e^q`, `sin q`,
//!   `cos q` for a quadratic `q` (`erf`/`erfi`, Fresnel `S`/`C`), `∫ ln(1 +
//!   a·x)/x` (`Li₂`), and the substitutions `x = eᵗ`, `u = eˣ` that lead to them
//!
//! For integrands that don't match any rule, an unevaluated
//! `Integral(body, var)` node is returned.
//!
//! # Design
//!
//! Integration is computed bottom-up using an iterative post-order
//! traversal, mirroring the design of `diff.rs`. Results are
//! constructed through canonical arena constructors to preserve
//! invariants.

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use num_traits::One;
use num_traits::Signed;
use num_traits::Zero;

use crate::base::arena::Arena;
use crate::base::budget::{BudgetKind, Work, WorkScope};
use crate::base::node::{ExprId, ExprNode, SymbolId};
use crate::base::numeric::Q;
use crate::base::stage::stage;
use crate::base::truth::Truth;
use crate::poly::Poly;

/// Integrate `expr` with respect to `var`.
///
/// Returns the antiderivative. If integration cannot be performed,
/// returns an unevaluated `Integral(expr, var)` node.
pub(crate) fn integrate(arena: &mut Arena, expr: ExprId, var: ExprId) -> ExprId {
    let call = IntegrateCall::enter();
    if call.depth > MAX_INTEGRATE_NESTING {
        tracing::debug!("integrate: nesting limit reached");
        return arena.intern(ExprNode::Integral(expr, var));
    }
    // The outermost call remembers the jumps at the poles of `tan w` for
    // all the substitutions and degenerate cases below it.
    let _jumps = crate::transforms::trig_integ::JumpMemo::enter();
    // ... and owns the work budget the nested calls draw from.
    let mut scope;
    let arena: &mut Arena = if call.outermost {
        scope = WorkScope::enter(arena, BudgetKind::Integration, INTEGRATE_WORK);
        &mut scope
    } else {
        arena
    };
    stage!(arena, "integrate", expr, integrate_impl(arena, expr, var))
}

/// Largest nesting of `integrate` calls (a substitution or a Risch tower
/// level re-enters it); a deeper call stays unevaluated.  A safety net
/// against a strategy that keeps nesting larger integrands: before 0.31's
/// guard in [`crate::calculus::risch::try_risch_tower`], the tower's
/// coefficient integrals of `∫ ln(c + d·x)/(a + b·x³) dx` nested 1,100
/// deep and overflowed the stack.
const MAX_INTEGRATE_NESTING: usize = 48;

/// One `integrate` call on this thread.  The outermost call owns the step
/// budget and the memo (and, in [`integrate`], the work budget); nested
/// calls (substitutions re-entering the pipeline, `integrate_definite`)
/// draw from them.  Dropping the guard
/// — also when a panic unwinds through the call, as under the Rubi
/// harness's `catch_unwind` — restores the nesting depth and, for the
/// outermost call, clears the memo, so no later call on another arena can
/// see its entries.
struct IntegrateCall {
    outermost: bool,
    /// Enclosing `integrate` calls on this thread.
    depth: usize,
}

impl IntegrateCall {
    fn enter() -> Self {
        let depth = INTEGRATE_DEPTH.with(|d| {
            let depth = d.get();
            d.set(depth + 1);
            depth
        });
        let outermost = depth == 0;
        if outermost {
            STEPS.with(|s| {
                s.set(Work::new(
                    BudgetKind::IntegrationSteps,
                    INTEGRATE_NODE_BUDGET,
                ))
            });
            STEP_END.with(|c| c.set(u64::MAX));
            INTEGRATE_MEMO.with(|m| m.borrow_mut().clear());
            VERDICT_MEMO.with(|m| m.borrow_mut().clear());
            JUMP_MEMO.with(|m| m.borrow_mut().clear());
            CLEARED_MEMO.with(|m| m.borrow_mut().clear());
            VARIES_MEMO.with(|m| m.borrow_mut().clear());
        }
        IntegrateCall { outermost, depth }
    }
}

impl Drop for IntegrateCall {
    fn drop(&mut self) {
        let _ = INTEGRATE_DEPTH.try_with(|d| d.set(d.get().saturating_sub(1)));
        if self.outermost {
            let _ = INTEGRATE_MEMO.try_with(|m| {
                if let Ok(mut m) = m.try_borrow_mut() {
                    m.clear();
                }
            });
            let _ = VERDICT_MEMO.try_with(|m| {
                if let Ok(mut m) = m.try_borrow_mut() {
                    m.clear();
                }
            });
            let _ = JUMP_MEMO.try_with(|m| {
                if let Ok(mut m) = m.try_borrow_mut() {
                    m.clear();
                }
            });
            let _ = CLEARED_MEMO.try_with(|m| {
                if let Ok(mut m) = m.try_borrow_mut() {
                    m.clear();
                }
            });
            let _ = VARIES_MEMO.try_with(|m| {
                if let Ok(mut m) = m.try_borrow_mut() {
                    m.clear();
                }
            });
        }
    }
}

/// Upper bound on `integrate_node` calls for one top-level `integrate`.
/// Every successful integration in the test suite (≈ 4 000 of them) takes
/// at most ~600; a failing search can branch much further before it gives
/// up — `∫ x·ln(x + sin 2) dx` made 524 540 calls (12 s) to return the
/// unevaluated form.  At the cap the remaining sub-integrals stay
/// unevaluated, which the callers already treat as "no closed form".  A
/// [`Work`] of kind [`BudgetKind::IntegrationSteps`].
const INTEGRATE_NODE_BUDGET: u64 = 20_000;

/// Upper bound on the interns one top-level `integrate` may attempt (a
/// [`WorkScope`] of kind [`BudgetKind::Integration`]), which bounds the
/// nodes it can add to the arena.  Successful integrations of the Rubi
/// suite and the test suite stay far below it; a search that keeps building
/// larger candidates (as the complex partial fractions of `∫ x³/(x⁸ + 1)
/// dx` did in 0.24, past 8 GB) stops here and leaves the remaining
/// sub-integrals unevaluated, so no integrand can exhaust memory through
/// the arena.  Up to 0.48 it was 2,000,000 nodes of arena growth, which
/// depends on what the context already holds.  The largest top-level
/// integration of the Rubi suite attempts 11.3 million interns (and adds
/// 1.0 million nodes), of the golden corpus 2.5 million.
const INTEGRATE_WORK: u64 = 20_000_000;

// The step budget lives beside the call nesting in thread-locals: the
// strategies reach `integrate_node` through many paths that carry only the
// arena.  The outermost call sets it afresh, so it does not depend on what
// the thread did before.
thread_local! {
    /// Enclosing `integrate` calls on this thread.
    static INTEGRATE_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// The steps of the outermost call ([`INTEGRATE_NODE_BUDGET`]).
    static STEPS: std::cell::Cell<Work> = const {
        std::cell::Cell::new(Work::new(BudgetKind::IntegrationSteps, INTEGRATE_NODE_BUDGET))
    };
    /// The spent step count at which the current [`StepCap`] ends
    /// (`u64::MAX`: none).
    static STEP_END: std::cell::Cell<u64> = const { std::cell::Cell::new(u64::MAX) };
}

/// A bound of `steps` further `integrate_node` steps on a nested
/// computation, within the top-level budget; the previous bound is
/// restored on drop (also when a panic unwinds).  While one is active, a
/// failure is not remembered in `INTEGRATE_MEMO`: it may be the bound's.
struct StepCap(u64);

impl StepCap {
    fn enter(steps: u64) -> Self {
        let used = STEPS.with(std::cell::Cell::get).spent();
        let old = STEP_END.with(|c| {
            let old = c.get();
            c.set(old.min(used.saturating_add(steps)));
            old
        });
        StepCap(old)
    }

    fn active() -> bool {
        STEP_END.with(std::cell::Cell::get) != u64::MAX
    }
}

impl Drop for StepCap {
    fn drop(&mut self) {
        let _ = STEP_END.try_with(|c| c.set(self.0));
    }
}

/// Has the outermost `integrate` attempted more than [`INTEGRATE_WORK`]
/// interns?
fn arena_budget_exhausted(arena: &Arena) -> bool {
    let exhausted = arena.work_exhausted(BudgetKind::Integration);
    if exhausted {
        tracing::debug!("integrate: arena budget exhausted");
    }
    exhausted
}

/// Charge one `integrate_node` step; `true` once the budget is spent (the
/// step does not fit; then no later one does either) or the current
/// [`StepCap`]'s steps are.
fn node_budget_exhausted() -> bool {
    STEPS.with(|s| {
        let mut steps = s.get();
        let spent = steps.charge(1, "integrate").is_err();
        s.set(steps);
        spent || steps.spent() > STEP_END.with(std::cell::Cell::get)
    })
}

fn integrate_impl(arena: &mut Arena, expr: ExprId, var: ExprId) -> ExprId {
    let var_sym = match arena.node(var) {
        ExprNode::Symbol(sid) => *sid,
        _ => {
            // var is not a symbol — can't integrate w.r.t. a non-symbol.
            return arena.intern(ExprNode::Integral(expr, var));
        }
    };
    // Exact constants first: `ln 1`, `atan 0`, `tan 0` are zero but not
    // structurally so, and a route that divides by a constant it cannot
    // see is zero is wrong — `∫ ln(x)/(x² + ln 1) dx` came out as
    // `atan(x/√(ln 1))`-terms (fuzz_integrate, 0.25).
    let expr = crate::transforms::eval::eval(arena, expr);

    // A `RootSum` that depends on the variable (an earlier answer, met by
    // by parts or a Risch coefficient) has no route here: every strategy
    // would take its body's `ln(x − t)` for a function of `x` alone.
    if crate::base::walk::post_order_ids(arena, expr)
        .iter()
        .any(|&id| {
            matches!(arena.node(id), ExprNode::RootSum(..)) && contains_var(arena, id, var_sym)
        })
    {
        return arena.intern(ExprNode::Integral(expr, var));
    }

    // `b^g` with `g` depending on the variable is `exp(g·ln b)` (the
    // principal power), which every exponential route takes.  Up to 0.46
    // only a lone `c^g` was rewritten: `∫ 2^x/(1 + 2^x) dx`, `∫ dt/(1 + 2^t)`,
    // `∫ a^(t − s)/(1 + a^t) dt`, `∫ x^x·(1 + ln x) dx` stayed unevaluated
    // (SymPy's test_integrals and test_risch).  The answer is written back
    // in the powers.
    if let Some((rewritten, bases)) = variable_exponents_as_exp(arena, expr, var_sym) {
        let r = integrate_impl(arena, rewritten, var);
        if !crate::base::walk::has_unevaluated(arena, r) {
            return exponentials_back_to_powers(arena, r, &bases);
        }
    }

    // Linearity first (see `integrate_node_uncached`): every stage below —
    // the substitutions, the Risch tower, heurisch — then sees the
    // integrand without its constant factors.  Before 0.30 they saw `c·f`:
    // `∫ a/cosh x`, `∫ π·e^x/(e^{2x} + 1)`, `∫ √2/(e^x + 1)` stayed
    // unevaluated although `∫ f` had a closed form.
    if let Some((constants, f)) = split_constant_factors(arena, expr, var_sym) {
        let big_f = integrate_stages(arena, f, var, var_sym);
        if matches!(arena.node(big_f), ExprNode::Integral(..)) {
            return arena.intern(ExprNode::Integral(expr, var));
        }
        return constants_times_antiderivative(arena, expr, f, big_f, &constants, var, var_sym)
            .unwrap_or_else(|| arena.intern(ExprNode::Integral(expr, var)));
    }
    integrate_stages(arena, expr, var, var_sym)
}

/// `expr` with every power `b^g` whose exponent `g` depends on the variable
/// (and `b ≠ e`) written `exp(g·ln b)`, and the bases `b`; `None` when there
/// is none.
fn variable_exponents_as_exp(
    arena: &mut Arena,
    expr: ExprId,
    var_sym: SymbolId,
) -> Option<(ExprId, Vec<ExprId>)> {
    let mut bases: Vec<ExprId> = Vec::new();
    let mut out = expr;
    for id in crate::base::walk::post_order_ids(arena, expr) {
        let ExprNode::Pow(b, g) = *arena.node(id) else {
            continue;
        };
        if b == arena.e_const() || !contains_var(arena, g, var_sym) {
            continue;
        }
        let ln_b = arena.ln(b);
        let ln_b = crate::transforms::eval::eval(arena, ln_b);
        let arg = arena.mul(&[g, ln_b]);
        let e = arena.exp(arg);
        out = arena.subs_structural(out, id, e);
        if !bases.contains(&b) {
            bases.push(b);
        }
    }
    if bases.is_empty() {
        return None;
    }
    let out = crate::transforms::eval::eval(arena, out);
    // The rewrite must be complete, or the call would come back here.
    let left = crate::base::walk::post_order_ids(arena, out)
        .into_iter()
        .any(|id| matches!(*arena.node(id), ExprNode::Pow(b, g) if b != arena.e_const() && contains_var(arena, g, var_sym)));
    (!left && out != expr).then_some((out, bases))
}

/// `e` with every `exp(c·ln b)` for a base `b` of [`variable_exponents_as_exp`]
/// written back as `b^c`.
fn exponentials_back_to_powers(arena: &mut Arena, e: ExprId, bases: &[ExprId]) -> ExprId {
    let mut out = e;
    for id in crate::base::walk::post_order_ids(arena, e) {
        let ExprNode::Exp(arg) = *arena.node(id) else {
            continue;
        };
        let ExprNode::Mul(factors) = arena.node(arg).clone() else {
            continue;
        };
        for &b in bases {
            let ln_b = arena.ln(b);
            let ln_b = crate::transforms::eval::eval(arena, ln_b);
            if let Some(pos) = factors.iter().position(|&f| f == ln_b) {
                let rest: SmallVec<[ExprId; 4]> = factors
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i != pos)
                    .map(|(_, &f)| f)
                    .collect();
                let c = arena.mul(&rest);
                let p = arena.pow(b, c);
                out = arena.subs_structural(out, id, p);
                break;
            }
        }
    }
    if out == e {
        e
    } else {
        crate::transforms::eval::eval(arena, out)
    }
}

/// The stages of [`integrate_impl`] on an evaluated integrand, then — when
/// they leave it unevaluated — once more on its normal form as a fraction:
/// a polynomial in its transcendental subexpressions with reduced
/// rational coefficients ([`crate::calculus::risch::reduced_fraction_terms`];
/// `(x²·cos x + cos x)/(x² + 1)` is `cos x`, `(eˣ·x·(x + 1) + 1)/(x + 1)`
/// is `x·eˣ + 1/(x + 1)`), and the degenerate parameter values in
/// `Piecewise`.  The normal form is its own normal form, so this retries
/// at most once.
fn integrate_stages(arena: &mut Arena, expr: ExprId, var: ExprId, var_sym: SymbolId) -> ExprId {
    let result = integrate_stages_once(arena, expr, var, var_sym);
    let result = if let ExprNode::Integral(_, _) = arena.node(result)
        && let Some(reduced) = crate::calculus::risch::reduced_fraction_terms(arena, expr, var)
    {
        let r = stage!(
            arena,
            "reduced_fraction_retry",
            reduced,
            integrate_stages_once(arena, reduced, var, var_sym)
        );
        if crate::base::walk::has_unevaluated(arena, r) {
            result
        } else {
            accept_or_unevaluated(arena, expr, r, var, var_sym)
        }
    } else {
        result
    };
    let result = if crate::base::walk::has_unevaluated(arena, result) {
        integrate_terms_separately(arena, expr, var, var_sym).unwrap_or(result)
    } else {
        result
    };

    // Piecewise wrapping for parametric degenerate cases
    stage!(
        arena,
        "piecewise_wrap",
        result,
        try_piecewise_wrap(arena, result, expr, var, var_sym)
    )
}

/// `∫ (f₁ + … + fₖ) = Σ ∫ fᵢ` with every term through the whole pipeline,
/// for a sum the stages left (partly) unevaluated.  Inside the stages the
/// terms of a sum are integrated one by one too, but only by the rules of
/// `integrate_node`: the substitutions, the Risch tower and heurisch run on
/// the whole integrand, and fit one term but not the sum.  `∫ (x^(5/2) +
/// x²·|x|)/(x³ + x² + 1) dx` kept `∫ x^(5/2)/(x³ + x² + 1)` unevaluated,
/// which `x = s²` solves on its own (and the `|x|` makes that substitution
/// refuse the sum).  A product with a sum factor is distributed first
/// (`(√x + |x|)/(x + x⁻² + 1)`).  `None` unless every term has a closed
/// form.
fn integrate_terms_separately(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    let sum = match arena.node(expr) {
        ExprNode::Add(_) => expr,
        ExprNode::Mul(children)
            if children
                .iter()
                .any(|&c| matches!(arena.node(c), ExprNode::Add(_))) =>
        {
            crate::transforms::expand::expand(arena, expr)
        }
        _ => return None,
    };
    let ExprNode::Add(terms) = arena.node(sum).clone() else {
        return None;
    };
    if terms.len() < 2 || terms.len() > MAX_SEPARATE_TERMS {
        return None;
    }
    let mut parts = Vec::with_capacity(terms.len());
    for &t in &terms {
        if !contains_var(arena, t, var_sym) {
            parts.push(arena.mul(&[t, var]));
            continue;
        }
        let (constants, f) = match split_constant_factors(arena, t, var_sym) {
            Some((c, f)) => (c, f),
            None => (SmallVec::new(), t),
        };
        let big_f = integrate_stages(arena, f, var, var_sym);
        if crate::base::walk::has_unevaluated(arena, big_f) {
            return None;
        }
        parts.push(wrap_with_constants(arena, big_f, &constants));
    }
    tracing::debug!("integrate: the terms of a sum, each through every stage");
    let sum = arena.add(&parts);
    let r = accept_or_unevaluated(arena, expr, sum, var, var_sym);
    (!crate::base::walk::has_unevaluated(arena, r)).then_some(r)
}

/// Sums with more terms are not retried term by term
/// ([`integrate_terms_separately`]): every term re-runs every stage.
const MAX_SEPARATE_TERMS: usize = 8;

/// One pass of the stages of [`integrate_stages`].
fn integrate_stages_once(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> ExprId {
    // Every stage's closed form faces the same evidence rule before it is
    // returned ([`accept_or_unevaluated`]); a rejected one counts as no answer,
    // so the next stage still gets its chance.  Up to 0.28.0 only some
    // routes inside the stages checked their candidates: by parts returned
    // `x·e^{kx}/k − e^{kx}/k²` for `∫ x·e^{kx} dx` with a `k` that is 0
    // without being so structurally — undefined everywhere.
    let result = integrate_node(arena, expr, var, var_sym, 20);
    let result = accept_or_unevaluated(arena, expr, result, var, var_sym);

    // Substitution-based strategies (u = e^{ax}, x = s^q, hyperbolic → exp,
    // piecewise-defined integrands).  Each one re-enters the full pipeline
    // on the transformed integrand, bounded by `SUBST_DEPTH`.
    let result = if let ExprNode::Integral(_, _) = arena.node(result)
        && let Some(r) = stage!(
            arena,
            "substitutions",
            expr,
            try_substitution_strategies(arena, expr, var, var_sym)
        ) {
        accept_or_unevaluated(arena, expr, r, var, var_sym)
    } else {
        result
    };

    // If the rule-based integrator returned an unevaluated Integral node,
    // try the Risch tower (exact method for exp/ln integrands) before
    // falling back to the heuristic integrator.
    if let ExprNode::Integral(_, _) = arena.node(result) {
        match stage!(
            arena,
            "risch_tower",
            expr,
            crate::calculus::risch::try_risch_tower(arena, expr, var)
        ) {
            // A dropped coefficient in the tower's θ-polynomial extraction
            // once returned a wrong closed form (0.22.3).
            crate::calculus::risch::TowerResult::Elementary(id) => {
                accept_or_unevaluated(arena, expr, id, var, var_sym)
            }
            crate::calculus::risch::TowerResult::NonElementary => {
                // Proved non-elementary — keep the unevaluated Integral node.
                // Skip heurisch: it can't succeed and would waste cycles.
                result
            }
            crate::calculus::risch::TowerResult::NotApplicable => {
                // Tower couldn't handle this — fall through to heurisch.
                if let Some(heurisch_result) = stage!(
                    arena,
                    "heurisch",
                    expr,
                    crate::transforms::heurisch::heurisch_integrate(arena, expr, var, var_sym)
                ) {
                    accept_or_unevaluated(arena, expr, heurisch_result, var, var_sym)
                } else {
                    result
                }
            }
        }
    } else {
        result
    }
}

/// A stage's answer `candidate` for `∫ f d(var)` in the form in which
/// `integrate` returns it, or the unevaluated `Integral(f, var)` when that
/// form is [`candidate_rejected`].  [`analytic_log_of_complex_arguments`]
/// comes first, so the evidence rule judges the answer that would be
/// returned.
fn accept_or_unevaluated(
    arena: &mut Arena,
    f: ExprId,
    candidate: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> ExprId {
    let unevaluated = arena.intern(ExprNode::Integral(f, var));
    if candidate == unevaluated {
        return candidate;
    }
    let candidate = analytic_log_of_complex_arguments(arena, candidate, var_sym);
    if candidate_rejected(arena, f, candidate, var, var_sym) {
        tracing::debug!("integrate: rejecting an unverified closed form");
        return unevaluated;
    }
    candidate
}

/// `true` when the candidate `big_f` for `∫ f d(var)` must not be used:
/// [`antiderivative_rejected`], or it introduces a `RootOf` (one its
/// integrand does not contain).  The routes that check their own
/// candidates call it on the candidate as they built it; every stage of
/// `integrate_impl` calls it through [`accept_or_unevaluated`].
///
/// Such a `RootOf` comes from `solve` in the root-based fallback of
/// `apart`, and a sum of logarithms over those roots is the rational
/// integrator's `RootSum` written out term by term.  It also names its
/// polynomial in the integration (or substitution) variable, which the
/// structural `subs` of a sample point or of a back-substitution rewrites
/// along with the free occurrences (`RootOf(u⁸ − u⁶ + …, 0)` became
/// `RootOf(e^{8x} − …, 0)`, which is no polynomial).  Such pieces had never
/// been integrated before [`contains_var`] learned that the polynomial's
/// variable is bound.
///
/// The evidence rule judges `big_f` after
/// [`analytic_log_of_complex_arguments`], the form in which the stage will
/// return it: a route may build `ln|b·x + a|` for an unassumed `a`, `b`,
/// which the stage exit turns into `ln(b·x + a)`; the complex round of
/// [`check_antiderivative`] would reject the first (decision D4, 0.31).
fn candidate_rejected(
    arena: &mut Arena,
    f: ExprId,
    big_f: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> bool {
    if introduces_root_of(arena, f, big_f) {
        tracing::debug!("integrate: rejecting a closed form over implicit roots");
        return true;
    }
    let big_f = analytic_log_of_complex_arguments(arena, big_f, var_sym);
    antiderivative_rejected(arena, f, big_f, var, var_sym)
}

/// Does `big_f` contain a `RootOf` node that `f` does not?
fn introduces_root_of(arena: &Arena, f: ExprId, big_f: ExprId) -> bool {
    let root_ofs = |e: ExprId| -> Vec<ExprId> {
        crate::base::walk::post_order_ids(arena, e)
            .into_iter()
            .filter(|&id| matches!(arena.node(id), ExprNode::RootOf(..)))
            .collect()
    };
    let new = root_ofs(big_f);
    !new.is_empty() && {
        let old: FxHashSet<ExprId> = root_ofs(f).into_iter().collect();
        new.iter().any(|id| !old.contains(id))
    }
}

/// Every `ln|u|` in `e` whose `u` depends on the variable and is not known
/// to be real ([`log_argument_is_real`]) becomes `ln u`.  `|u|` of a
/// non-real `u` is not the real-variable device the integrator means by it
/// (`d/dx ln|u| = Re(u′/u) ≠ u′/u`), while `ln u` is an antiderivative of
/// `u′/u` for real and complex `u` alike (analytic along the real line as
/// long as `u` avoids the negative real axis, which a `u` with a non-zero
/// imaginary part does; a real `u` only adds the constant `iπ` where it is
/// negative).  So `ln|u|` is kept only where realness is established.
/// Up to 0.28.0 only an explicit `i` counted: `∫ dx/(x − √(1/2 − √5/2))`
/// came out as `ln|x − √(1/2 − √5/2)|`, whose derivative is the conjugate
/// of the integrand.
fn analytic_log_of_complex_arguments(arena: &mut Arena, e: ExprId, var_sym: SymbolId) -> ExprId {
    let mut out = e;
    let mut reals = crate::base::assumptions::AssumptionCache::new();
    for id in crate::base::walk::post_order_ids(arena, e) {
        if let ExprNode::Ln(inner) = arena.node(id).clone()
            && let ExprNode::Abs(u) = arena.node(inner).clone()
            && contains_var(arena, u, var_sym)
            && !log_argument_is_real(arena, u, var_sym, &mut reals)
        {
            let plain = arena.ln(u);
            out = arena.subs_structural(out, id, plain);
        }
    }
    out
}

/// Is the argument `u` of an `ln|u|` real, as far as its constants go?
///
/// The variable-dependent structure of `u` is read with the real-variable
/// convention (`√(x − 5)` is real where the integrand is); what is decided
/// here is each maximal variable-free subexpression, by
/// [`constant_is_real`].  A constant that is not known to be real makes
/// `u` not known to be real — in particular every constant with a symbol
/// that has no declared assumptions (decision D4, 0.31: the variable is
/// real, a parameter is what its assumptions say, and an unassumed one
/// may be complex).  So `ln|a·x + 1|` stays only for a declared-real `a`;
/// up to 0.30 an unassumed `a` counted as real and `∫ dx/(a·x + 1)` was
/// `ln|a·x + 1|/a`, whose derivative `re(conj(u)·a)/|u|²` is not
/// `1/(a·x + 1)` for a non-real `a` (SymPy, Mathematica and Maple all
/// return `log(a·x + 1)/a`).
fn log_argument_is_real(
    arena: &mut Arena,
    u: ExprId,
    var_sym: SymbolId,
    reals: &mut crate::base::assumptions::AssumptionCache,
) -> bool {
    let mut stack = vec![u];
    let mut seen: FxHashSet<ExprId> = FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        if contains_var(arena, id, var_sym) {
            arena.node(id).for_each_child(|c| stack.push(c));
        } else if !constant_is_real(arena, id, reals) {
            return false;
        }
    }
    true
}

/// Is the variable-free `c` known to be real under the declared
/// assumptions of its parameters?
/// [`crate::transforms::realness::constant_realness_as_declared`] to
/// [`FTC_CHECK_DIGITS`] digits (`√(1/2 − √5/2)` is not, `√(3 − √5)` is,
/// `a + 1` is for a declared-real `a` and undecided for an unassumed one,
/// `√a` is undecided either way).  An undecided `c` is not known to be
/// real: the answer then keeps `ln u`, which is still an antiderivative.
fn constant_is_real(
    arena: &mut Arena,
    c: ExprId,
    reals: &mut crate::base::assumptions::AssumptionCache,
) -> bool {
    Truth::from(crate::transforms::realness::constant_realness_as_declared(
        arena,
        c,
        FTC_CHECK_DIGITS,
        reals,
    ))
    .is_true()
}

/// Check whether `expr` is a suitable candidate for the `u` factor in
/// integration by parts.  Returns `true` when `expr` is a polynomial
/// in `var` **or** when it is `ln(inner)` with `inner` depending on `var`.
fn is_by_parts_candidate(arena: &Arena, expr: ExprId, var: ExprId, var_sym: SymbolId) -> bool {
    if is_polynomial_in(arena, expr, var, var_sym) {
        return true;
    }
    match arena.node(expr) {
        ExprNode::Ln(inner)
        | ExprNode::Asin(inner)
        | ExprNode::Acos(inner)
        | ExprNode::Atan(inner)
        | ExprNode::Erf(inner)
        | ExprNode::Erfc(inner) => contains_var(arena, *inner, var_sym),
        _ => is_erfi_of_var(arena, expr, var_sym) || is_special_by_parts_fn(arena, expr, var_sym),
    }
}

/// Is `expr` one of the functions of the variable whose derivative is an
/// elementary function of the argument, which by parts takes for `u`
/// like the inverse trigonometric functions: `Ei`, `Si`, `Ci`, `li`,
/// `Shi`, `Chi`, the Fresnel integrals, `asinh`, `acosh`, `atanh`?  Up to
/// 0.46 none was: `∫ Ei(x) dx`, `∫ Si(x) dx`, `∫ x·asinh(x/2) dx` stayed
/// unevaluated, and with `∫ Ei` the three-factor rule's `∫ x·eˣ·ln x dx`
/// (SymPy's test_integrals and test_meijerint; SymPy integrates all of
/// them).
fn is_special_by_parts_fn(arena: &Arena, expr: ExprId, var_sym: SymbolId) -> bool {
    use crate::base::libfn::LibFn;
    // Of a linear argument only: `atanh` of the quotients the rational
    // integrator leaves, taken for a `u`, sent failing searches down long
    // chains of by parts (`∫ atan(x³)/(√x − ½) dx`: 77 ms → 1.1 s).
    let arg = match arena.node(expr) {
        ExprNode::Ei(g)
        | ExprNode::Si(g)
        | ExprNode::Ci(g)
        | ExprNode::Li(g)
        | ExprNode::Asinh(g)
        | ExprNode::Acosh(g)
        | ExprNode::Atanh(g) => *g,
        ExprNode::Lib(LibFn::Shi | LibFn::Chi | LibFn::FresnelS | LibFn::FresnelC, args)
            if args.len() == 1 =>
        {
            args[0]
        }
        _ => return false,
    };
    contains_var(arena, arg, var_sym) && linear_in_structurally(arena, arg, var_sym)
}

/// Is `g` (as written) `c₁·x + c₀` for constants `c₁`, `c₀`: `x`, a product
/// of `x` with constants, a sum of such terms and constants?
fn linear_in_structurally(arena: &Arena, g: ExprId, var_sym: SymbolId) -> bool {
    let term_ok = |t: ExprId| -> bool {
        if !contains_var(arena, t, var_sym) {
            return true;
        }
        match arena.node(t) {
            ExprNode::Symbol(s) => *s == var_sym,
            ExprNode::Mul(fs) => {
                fs.iter()
                    .filter(|&&f| contains_var(arena, f, var_sym))
                    .count()
                    == 1
                    && fs.iter().all(|&f| {
                        !contains_var(arena, f, var_sym)
                            || matches!(arena.node(f), ExprNode::Symbol(s) if *s == var_sym)
                    })
            }
            _ => false,
        }
    };
    match arena.node(g) {
        ExprNode::Add(ts) => ts.iter().all(|&t| term_ok(t)),
        _ => term_ok(g),
    }
}

/// `∫ F(g) dx` by parts with `dv = dx` for a by-parts candidate `F`
/// ([`is_by_parts_candidate`]) on its own: `v = g/a` for a linear `g = a·x
/// + b` (then `∫ v·du = ∫ g·F′(g) dg/a`: `e^g` for `Ei`, `sin g` for
/// `Si`), `v = x` otherwise.  `None` when `∫ v·du` has no closed form
/// (or still holds `F`).
fn by_parts_dv_one(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    if depth < 2 {
        return None;
    }
    let v = single_argument(arena, expr)
        .and_then(|g| {
            let (a, _) = symbolic_linear_coeff_of(arena, g, var, var_sym)?;
            Some(arena.div(g, a))
        })
        .unwrap_or(var);
    let du = crate::transforms::diff::diff(arena, expr, var);
    if crate::base::walk::contains(arena, du, expr) {
        return None;
    }
    let v_du = arena.mul(&[v, du]);
    let v_du = crate::transforms::eval::eval(arena, v_du);
    let mut rest = integrate_node(arena, v_du, var, var_sym, depth - 1);
    // For the integrand itself (not a factor met deep in a search) also
    // the substitutions: `∫ asin(1/x) dx` leaves `∫ dx/(x·√(1 − x⁻²))`,
    // `∫ li(x) dx` leaves `∫ x/ln x dx` (`x = eᵗ`).
    if crate::base::walk::has_unevaluated(arena, rest) && depth >= DV_ONE_NESTED_DEPTH {
        rest = integrate_nested(arena, v_du, var)?;
    }
    if crate::base::walk::has_unevaluated(arena, rest) {
        return None;
    }
    tracing::debug!("integrate: by parts with dv = dx");
    let uv = arena.mul(&[v, expr]);
    Some(arena.sub(uv, rest))
}

/// The least `integrate_node` depth (the stages start at 20) at which
/// [`by_parts_dv_one`] hands `∫ v·du` to the whole pipeline: the integrand
/// itself only (at 19, inside the u-substitutions of `∫ atan(x³)/(√x − ½) dx`,
/// a failing search took 1.1 s for 77 ms).
const DV_ONE_NESTED_DEPTH: usize = 20;

/// The argument of a one-argument function node, if any.  A library function
/// of several arguments has no single one (`besselj(ν, x)` takes it last,
/// `lambertw(x, k)` and `elliptic_f(φ, m)` first; up to 0.48 the last was
/// taken for all of them), nor has a user function: `None`.
fn single_argument(arena: &Arena, expr: ExprId) -> Option<ExprId> {
    use crate::base::func::{ArgCount, Family};
    let node = arena.node(expr);
    let f = node.func_head()?;
    let library = f.family() == Family::Lib && f.spec().arity == ArgCount::Exactly(1);
    node.unary_arg_if(|f| library || by_parts_unary(f))
}

/// The node-backed one-argument functions [`single_argument`] reads: the
/// inverse trigonometric and hyperbolic functions, `ln`, `Ei`, `Si`, `Ci`,
/// `li`, `erf`, `erfc` (the by-parts `u`s, whose derivatives are simpler).
fn by_parts_unary(f: crate::base::func::Func) -> bool {
    use crate::base::func::{Elementary, Func};
    matches!(
        f.elementary(),
        Some(Elementary::InverseTrig | Elementary::InverseHyperbolic | Elementary::Ln)
    ) || matches!(
        f,
        Func::Ei | Func::Si | Func::Ci | Func::Li | Func::Erf | Func::Erfc
    )
}

/// Is `expr` `erfi(g)` with `g` depending on the variable?  The error
/// functions are by-parts `u`s like the inverse trigonometric functions:
/// their derivatives are Gaussians, so `∫ xⁿ·erf(a·x + b)` is a polynomial
/// times a Gaussian (unevaluated up to 0.43; SymPy integrates them).
fn is_erfi_of_var(arena: &Arena, expr: ExprId, var_sym: SymbolId) -> bool {
    matches!(arena.node(expr), ExprNode::Lib(crate::base::libfn::LibFn::Erfi, args)
        if args.len() == 1 && contains_var(arena, args[0], var_sym))
}

/// LIATE priority for integration by parts: lower = better choice for u.
/// L(og) = 1, I(nverse trig) = 2, A(lgebraic/polynomial) = 3,
/// T(rig) = 4, E(xponential) = 5, other = 6.
fn liate_rank(arena: &Arena, expr: ExprId, var: ExprId, var_sym: SymbolId) -> u8 {
    match arena.node(expr) {
        ExprNode::Ln(_) => 1,
        ExprNode::Asin(_)
        | ExprNode::Acos(_)
        | ExprNode::Atan(_)
        | ExprNode::Erf(_)
        | ExprNode::Erfc(_) => 2,
        _ if is_erfi_of_var(arena, expr, var_sym) => 2,
        _ if is_special_by_parts_fn(arena, expr, var_sym) => 2,
        _ if is_polynomial_in(arena, expr, var, var_sym) => 3,
        ExprNode::Sin(_) | ExprNode::Cos(_) | ExprNode::Tan(_) => 4,
        ExprNode::Exp(_) | ExprNode::Sinh(_) | ExprNode::Cosh(_) | ExprNode::Tanh(_) => 5,
        _ => 6,
    }
}

/// Check whether `expr` is `Pow(var, 2)`, i.e. `x²`.
fn is_var_squared(arena: &Arena, expr: ExprId, var: ExprId) -> bool {
    if let ExprNode::Pow(b, e) = arena.node(expr)
        && *b == var
        && let Some(n) = arena.as_num(*e)
    {
        return *n == num_rational::Ratio::from_integer(2.into());
    }
    false
}

/// Check whether `expr` represents `-x²` in canonical form.
///
/// Handles both `Neg(Pow(var, 2))` and `Mul([-1, Pow(var, 2)])`.
fn is_neg_var_squared(arena: &Arena, expr: ExprId, var: ExprId) -> bool {
    match arena.node(expr).clone() {
        ExprNode::Neg(inner) => is_var_squared(arena, inner, var),
        ExprNode::Mul(ref children) if children.len() == 2 => {
            let neg_one_val = num_rational::Ratio::<num_bigint::BigInt>::from_integer((-1).into());
            let has_neg_one = children
                .iter()
                .any(|&c| arena.as_num(c).is_some_and(|n| *n == neg_one_val));
            let has_var_sq = children.iter().any(|&c| is_var_squared(arena, c, var));
            has_neg_one && has_var_sq
        }
        _ => false,
    }
}

/// Try to recognise standard‑form integrals of the shape
/// `(x² ± a²)^n` where `n` is −1 or −1/2.
///
/// Returns `Some(antiderivative)` on success.
fn try_standard_form_integral(
    arena: &mut Arena,
    _expr: ExprId,
    base: ExprId,
    exp: ExprId,
    var: ExprId,
    _var_sym: SymbolId,
) -> Option<ExprId> {
    // ── Check exponent ─────────────────────────────────────────────
    let exp_val = arena.as_num(exp)?.clone();
    let neg_one = num_rational::Ratio::<num_bigint::BigInt>::from_integer((-1).into());
    let neg_half = num_rational::Ratio::<num_bigint::BigInt>::new((-1).into(), 2.into());

    let is_neg_one = exp_val == neg_one;
    let is_neg_half = exp_val == neg_half;
    if !is_neg_one && !is_neg_half {
        return None;
    }

    // ── Check base is Add with exactly 2 children ──────────────────
    let children = match arena.node(base).clone() {
        ExprNode::Add(c) if c.len() == 2 => c,
        _ => return None,
    };

    // ── Classify each child ────────────────────────────────────────
    let mut const_val: Option<Q> = None;
    let mut has_pos_x2 = false;
    let mut has_neg_x2 = false;

    for &child in children.iter() {
        if let Some(n) = arena.as_num(child) {
            const_val = Some(n.clone());
        } else if is_var_squared(arena, child, var) {
            has_pos_x2 = true;
        } else if is_neg_var_squared(arena, child, var) {
            has_neg_x2 = true;
        } else {
            return None;
        }
    }

    let c_val = const_val?;
    if !has_pos_x2 && !has_neg_x2 {
        return None;
    }

    let a_squared = c_val.abs();
    if a_squared.is_zero() {
        return None;
    }

    // ── Build x/a and 1/a (simplify when a²=1) ────────────────────
    let a_sq_is_one = a_squared == num_rational::Ratio::<num_bigint::BigInt>::one();

    let x_over_a = if a_sq_is_one {
        var
    } else {
        let a_sq_id = arena.num_ratio(a_squared.clone());
        let nh = arena.rational(-1, 2);
        let a_inv = arena.pow(a_sq_id, nh); // (a²)^{-1/2} = 1/a
        arena.mul(&[var, a_inv])
    };

    // 1/a   (only needed for exp == -1 forms)
    let one_over_a = if a_sq_is_one {
        None
    } else {
        let a_sq_id = arena.num_ratio(a_squared.clone());
        let nh = arena.rational(-1, 2);
        Some(arena.pow(a_sq_id, nh))
    };

    // ── Match patterns ─────────────────────────────────────────────

    // Pattern: x² + a²  (c_val > 0, positive x²)
    if has_pos_x2 && c_val.is_positive() {
        if is_neg_one {
            // A3: ∫ (x²+a²)^{-1} dx = (1/a)·atan(x/a)
            let atan_val = arena.atan(x_over_a);
            return Some(match one_over_a {
                Some(inv_a) => arena.mul(&[inv_a, atan_val]),
                None => atan_val,
            });
        }
        if is_neg_half {
            // A5: ∫ (x²+a²)^{-1/2} dx = asinh(x/a)
            let asinh_val = arena.asinh(x_over_a);
            return Some(asinh_val);
        }
    }

    // Pattern: a² − x²  (c_val > 0, negative x²)
    if has_neg_x2 && c_val.is_positive() {
        if is_neg_half {
            // A4: ∫ (a²−x²)^{-1/2} dx = asin(x/a)
            let asin_val = arena.asin(x_over_a);
            return Some(asin_val);
        }
        if is_neg_one {
            // A7: ∫ (a²−x²)^{-1} dx = (1/a)·atanh(x/a)
            let atanh_val = arena.atanh(x_over_a);
            return Some(match one_over_a {
                Some(inv_a) => arena.mul(&[inv_a, atanh_val]),
                None => atanh_val,
            });
        }
    }

    // Pattern: x² − a²  (c_val < 0, positive x², a² = |c_val|)
    if has_pos_x2 && c_val.is_negative() && is_neg_half {
        // A6: ∫ (x²−a²)^{-1/2} dx = ln|x + √(x²−a²)|, the real-variable
        // antiderivative on both x > a and x < −a.  `acosh(x/a)` (SymPy's
        // answer, and ours before 0.24) differs from it by a constant for
        // x > a but has the wrong derivative for x < −a.
        return Some(ln_abs_x_plus_sqrt(arena, var, base));
    }

    None
}

/// `ln|v + √q|`, the real-variable antiderivative of `v′/√q` when
/// `q = v² − c` for a constant `c > 0` (`q` is `x² − a²`, or `a·u² + d`
/// with `v = √a·u`): `d/dx ln|v + √(v² − c)| = v′/√(v² − c)` on both
/// `v > √c` and `v < −√c`.
fn ln_abs_x_plus_sqrt(arena: &mut Arena, v: ExprId, q: ExprId) -> ExprId {
    let half = arena.rational(1, 2);
    let sqrt_q = arena.pow(q, half);
    let sum = arena.add(&[v, sqrt_q]);
    let abs_sum = arena.abs(sum);
    arena.ln(abs_sum)
}

/// Try integrating `(ax²+bx+c)^exp` via completing the square.
///
/// Handles two exponent values:
/// - `exp = -1`:   `∫ 1/(ax²+bx+c) dx` → atan form
/// - `exp = -1/2`: `∫ 1/√(ax²+bx+c) dx` → asinh / acosh / asin form
fn try_complete_square_integral(
    arena: &mut Arena,
    base: ExprId,
    exp: ExprId,
    var: ExprId,
    _var_sym: SymbolId,
) -> Option<ExprId> {
    let exp_r = arena.as_num(exp)?.clone();
    let neg_one = num_rational::Ratio::<num_bigint::BigInt>::from_integer((-1).into());
    let neg_half = num_rational::Ratio::<num_bigint::BigInt>::new((-1).into(), 2.into());

    let is_neg_one = exp_r == neg_one;
    let is_neg_half = exp_r == neg_half;
    if !is_neg_one && !is_neg_half {
        return None;
    }

    // base must be a quadratic in var: ax² + bx + c
    let poly_opt = crate::poly::polybridge::expr_to_poly(arena, base, var);

    // ── Symbolic fallback for irrational coefficients ──────────────
    // When expr_to_poly fails (e.g., coefficients contain √5 or ∛2),
    // try symbolic_quadratic_coeffs and arena-level completing the square.
    // This enables integration of terms produced by apart for cyclotomic
    // denominators like x⁵−1 and x⁸−1.
    if poly_opt.is_none() && is_neg_one {
        tracing::debug!(
            "try_complete_square: expr_to_poly failed, trying symbolic quadratic coefficients"
        );
        if let Some((c_id, d_id, e_id)) = symbolic_quadratic_coeffs(arena, base, var, _var_sym) {
            tracing::debug!(
                "try_complete_square: symbolic quadratic coefficients extracted, completing the square"
            );

            // Normalize to monic: b_sym = d/c, c_norm = e/c
            let b_sym = arena.div(d_id, c_id);
            let c_norm = arena.div(e_id, c_id);

            // Complete the square: x² + bx + c_norm = (x + b/2)² + (c_norm − b²/4)
            let two = arena.int(2);
            let half_b = arena.div(b_sym, two);
            let half_b_sq = arena.mul(&[half_b, half_b]);
            let disc = arena.sub(c_norm, half_b_sq);
            let disc = crate::transforms::eval::eval(arena, disc);

            // Sign test: discriminant must be positive for the atan form.
            let disc_sign = crate::poly::algebraic::sign_checked(arena, disc);
            tracing::debug!(?disc_sign, "try_complete_square: discriminant sign");
            if disc_sign != Some(1) {
                tracing::debug!(
                    "try_complete_square: discriminant non-positive, atan form not applicable"
                );
                return None;
            }

            // Result: (1/(a·√d)) · atan((x + b/2) / √d)
            let half = arena.rational(1, 2);
            let sqrt_disc = arena.pow(disc, half);
            let shifted = arena.add(&[var, half_b]);
            let ratio = arena.div(shifted, sqrt_disc);
            let atan_val = arena.atan(ratio);

            let sqrt_disc2 = arena.pow(disc, half);
            let a_sqrt_d = arena.mul(&[c_id, sqrt_disc2]);
            tracing::debug!(
                "try_complete_square: symbolic completing-the-square succeeded → atan form"
            );
            return Some(arena.div(atan_val, a_sqrt_d));
        }
    }

    // ── Rational-coefficient path ──────────────────────────────────
    let poly = poly_opt?;
    if poly.degree()? != 2 {
        return None;
    }

    let a_coeff = poly.coeff(2);
    let b_coeff = poly.coeff(1);
    let c_coeff = poly.coeff(0);

    if a_coeff.is_zero() {
        return None;
    }

    // ── exp = -1: ∫ 1/(ax²+bx+c) dx ───────────────────────────────
    if is_neg_one {
        // Normalize to monic: divide by a
        let b = &b_coeff / &a_coeff;
        let c = &c_coeff / &a_coeff;

        // If there's no linear term, this is a standard form — let the other helper handle it.
        if b.is_zero() {
            return None;
        }

        // Complete the square: x² + bx + c = (x + b/2)² + (c - b²/4)
        let half_b = &b / &num_rational::Ratio::from_integer(2.into());
        let d = &c - &(&half_b * &half_b); // d = c - b²/4

        if d.is_zero() || d.is_negative() {
            return None; // Can't use atan form if d ≤ 0
        }

        // Build (x + b/2)
        let half_b_id = {
            let nid = arena.intern_num(half_b.clone());
            arena.intern(crate::base::node::ExprNode::Num(nid))
        };
        let shifted = arena.add(&[var, half_b_id]);

        // Build √d
        let d_id = {
            let nid = arena.intern_num(d.clone());
            arena.intern(crate::base::node::ExprNode::Num(nid))
        };
        let half = arena.rational(1, 2);
        let sqrt_d = arena.pow(d_id, half);

        // Result: (1/(a·√d)) · atan((x+b/2)/√d)
        let ratio = arena.div(shifted, sqrt_d);
        let atan_result = arena.atan(ratio);

        // Divide by a·√d
        let a_id = {
            let nid = arena.intern_num(a_coeff);
            arena.intern(crate::base::node::ExprNode::Num(nid))
        };
        // Rebuild √d for the denominator (arena IDs are Copy, but let's be explicit)
        let sqrt_d2 = arena.pow(d_id, half);
        let a_sqrt_d = arena.mul(&[a_id, sqrt_d2]);

        return Some(arena.div(atan_result, a_sqrt_d));
    }

    // ── exp = -1/2: ∫ 1/√(ax²+bx+c) dx ───────────────────────────
    // Complete the square: ax²+bx+c = a·(x + b/(2a))² + (c − b²/(4a))
    // Let u = x + b/(2a),  d = c − b²/(4a).
    // Then ∫ 1/√(a·u² + d) du.
    let two_r = num_rational::Ratio::<num_bigint::BigInt>::from_integer(2.into());
    let four_r = num_rational::Ratio::<num_bigint::BigInt>::from_integer(4.into());

    let shift = &b_coeff / &(&a_coeff * &two_r); // b/(2a)
    let d = &c_coeff - &(&b_coeff * &b_coeff / &(&a_coeff * &four_r)); // c − b²/(4a)

    // Build u = x + b/(2a)
    let u_expr = if shift.is_zero() {
        var
    } else {
        let shift_id = arena.num_ratio(shift.clone());
        arena.add(&[var, shift_id])
    };

    let half = arena.rational(1, 2);

    if a_coeff.is_positive() {
        // a > 0
        let a_id = arena.num_ratio(a_coeff.clone());
        let sqrt_a = arena.pow(a_id, half); // √a
        let inv_sqrt_a = {
            let neg_half_e = arena.rational(-1, 2);
            arena.pow(a_id, neg_half_e)
        }; // 1/√a

        if d.is_positive() {
            // a > 0, d > 0: (1/√a) · asinh(u·√a / √d)
            let d_id = arena.num_ratio(d.clone());
            let sqrt_d = arena.pow(d_id, half);
            let u_sqrt_a = arena.mul(&[u_expr, sqrt_a]);
            let arg = arena.div(u_sqrt_a, sqrt_d);
            let asinh_val = arena.asinh(arg);
            return Some(arena.mul(&[inv_sqrt_a, asinh_val]));
        } else if d.is_negative() {
            // a > 0, d < 0: (1/√a) · ln|u·√a + √(a u² + d)|, valid on both
            // sides of the gap (acosh(u·√a/√|d|) is wrong for u < 0).
            let u_sqrt_a = arena.mul(&[u_expr, sqrt_a]);
            let ln_val = ln_abs_x_plus_sqrt(arena, u_sqrt_a, base);
            return Some(arena.mul(&[inv_sqrt_a, ln_val]));
        } else {
            // a > 0, d = 0: √(a·u²) = √a·|u| for the real variable, so the
            // antiderivative is sign(u)·ln|u|/√a (0.23 dropped the sign:
            // ∫ 1/√((3x − 2)²) dx was ln|x − 2/3|/3, wrong for x < 2/3;
            // found by the Rubi suite).
            let abs_u = arena.abs(u_expr);
            let ln_u = arena.ln(abs_u);
            let sign_u = arena.sign(u_expr);
            return Some(arena.mul(&[inv_sqrt_a, sign_u, ln_u]));
        }
    } else if a_coeff.is_negative() && d.is_positive() {
        // a < 0, d > 0: (1/√|a|) · asin(u·√|a| / √d)
        let abs_a = a_coeff.abs();
        let abs_a_id = arena.num_ratio(abs_a.clone());
        let sqrt_abs_a = arena.pow(abs_a_id, half);
        let inv_sqrt_abs_a = {
            let neg_half_e = arena.rational(-1, 2);
            arena.pow(abs_a_id, neg_half_e)
        };
        let d_id = arena.num_ratio(d.clone());
        let sqrt_d = arena.pow(d_id, half);
        let u_sqrt_abs_a = arena.mul(&[u_expr, sqrt_abs_a]);
        let arg = arena.div(u_sqrt_abs_a, sqrt_d);
        let asin_val = arena.asin(arg);
        return Some(arena.mul(&[inv_sqrt_abs_a, asin_val]));
    }

    None
}

/// Detect `sec(x)·tan(x)` and `csc(x)·cot(x)` patterns in a product.
///
/// - `sin(x) · cos(x)^{-2}` → `cos(x)^{-1}`   (∫ sec·tan dx = sec)
/// - `cos(x) · sin(x)^{-2}` → `-sin(x)^{-1}`   (∫ csc·cot dx = −csc)
fn try_trig_recip_product(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    if dependent.len() != 2 {
        return None;
    }

    let neg_two = num_rational::Ratio::<num_bigint::BigInt>::from_integer((-2).into());

    for (i, j) in [(0usize, 1usize), (1, 0)] {
        let node_i = arena.node(dependent[i]).clone();
        let node_j = arena.node(dependent[j]).clone();

        // Pattern: sin(g) · cos(g)^{-2} → cos(g)^{-1} [/ chain coeff]
        if let ExprNode::Sin(inner_sin) = node_i
            && let ExprNode::Pow(base_j, exp_j) = node_j
            && let ExprNode::Cos(inner_cos) = arena.node(base_j).clone()
            && inner_sin == inner_cos
            && let Some(e) = arena.as_num(exp_j)
            && *e == neg_two
        {
            let neg_one_e = arena.int(-1);
            if inner_sin == var {
                return Some(arena.pow(base_j, neg_one_e));
            } else if let Some((a_expr, _)) =
                symbolic_linear_coeff_of(arena, inner_sin, var, var_sym)
            {
                let recip = arena.pow(base_j, neg_one_e);
                return Some(arena.div(recip, a_expr));
            }
        }

        // Pattern: cos(g) · sin(g)^{-2} → −sin(g)^{-1} [/ chain coeff]
        if let ExprNode::Cos(inner_cos) = node_i
            && let ExprNode::Pow(base_j, exp_j) = node_j
            && let ExprNode::Sin(inner_sin) = arena.node(base_j).clone()
            && inner_cos == inner_sin
            && let Some(e) = arena.as_num(exp_j)
            && *e == neg_two
        {
            let neg_one_e = arena.int(-1);
            if inner_cos == var {
                let recip = arena.pow(base_j, neg_one_e);
                return Some(arena.neg(recip));
            } else if let Some((a_expr, _)) =
                symbolic_linear_coeff_of(arena, inner_cos, var, var_sym)
            {
                let recip = arena.pow(base_j, neg_one_e);
                let neg_recip = arena.neg(recip);
                return Some(arena.div(neg_recip, a_expr));
            }
        }
    }

    None
}

/// Detect `x / √(ax²+bx+c)` and integrate using the decomposition:
///
///   `∫ x/√R dx = √R/a − (b/(2a))·∫ 1/√R dx`
///
/// where `R = ax²+bx+c`.
fn try_x_over_sqrt_quadratic(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    if dependent.len() != 2 {
        return None;
    }

    // Find which factor is var and which is Pow(quadratic, -1/2)
    let pow_idx = if dependent[0] == var {
        1
    } else if dependent[1] == var {
        0
    } else {
        return None;
    };

    // Check the other factor is Pow(base, -1/2)
    let (base, exp_id) = match arena.node(dependent[pow_idx]).clone() {
        ExprNode::Pow(b, e) => (b, e),
        _ => return None,
    };

    let exp_r = arena.as_num(exp_id)?.clone();
    let neg_half = num_rational::Ratio::<num_bigint::BigInt>::new((-1).into(), 2.into());
    if exp_r != neg_half {
        return None;
    }

    // base must be quadratic in var
    let poly = crate::poly::polybridge::expr_to_poly(arena, base, var)?;
    if poly.degree()? != 2 {
        return None;
    }

    let a_coeff = poly.coeff(2);
    let b_coeff = poly.coeff(1);

    if a_coeff.is_zero() {
        return None;
    }

    // √R = base^{1/2}
    let half = arena.rational(1, 2);
    let sqrt_r = arena.pow(base, half);

    // First term: √R / a
    let a_id = arena.num_ratio(a_coeff.clone());
    let first_term = arena.div(sqrt_r, a_id);

    if b_coeff.is_zero() {
        // No linear term: ∫ x/√(ax²+c) dx = √(ax²+c)/a
        return Some(first_term);
    }

    // Need I_0 = ∫ 1/√R dx
    let i_0 = integrate_node(
        arena,
        dependent[pow_idx],
        var,
        var_sym,
        depth.saturating_sub(1),
    );
    if matches!(arena.node(i_0), ExprNode::Integral(_, _)) {
        return None;
    }

    // b/(2a)
    let two_r = num_rational::Ratio::<num_bigint::BigInt>::from_integer(2.into());
    let b_over_2a = &b_coeff / &(&a_coeff * &two_r);
    let b_over_2a_id = arena.num_ratio(b_over_2a.clone());

    // Result: √R/a − (b/(2a))·I_0
    let second_term = arena.mul(&[b_over_2a_id, i_0]);
    Some(arena.sub(first_term, second_term))
}

/// Try to integrate `(a² ± x²)^{1/2}` forms using trig substitution results.
///
/// Handles the three standard trig substitution patterns with positive
/// half-exponent:
/// - `∫ √(a²−x²) dx = ½(x·√(a²−x²) + a²·asin(x/a))`
/// - `∫ √(x²+a²) dx = ½(x·√(x²+a²) + a²·asinh(x/a))`
/// - `∫ √(x²−a²) dx = ½(x·√(x²−a²) − a²·acosh(x/a))`
fn try_trig_sub_sqrt_integral(
    arena: &mut Arena,
    base: ExprId,
    exp: ExprId,
    var: ExprId,
    _var_sym: SymbolId,
) -> Option<ExprId> {
    // ── Check exponent is 1/2 ──────────────────────────────────────
    let exp_val = arena.as_num(exp)?.clone();
    let pos_half = num_rational::Ratio::<num_bigint::BigInt>::new(1.into(), 2.into());
    if exp_val != pos_half {
        return None;
    }

    // ── Check base is Add with exactly 2 children ──────────────────
    let children = match arena.node(base).clone() {
        ExprNode::Add(c) if c.len() == 2 => c,
        _ => return None,
    };

    // ── Classify each child ────────────────────────────────────────
    let mut const_val: Option<Q> = None;
    let mut has_pos_x2 = false;
    let mut has_neg_x2 = false;

    for &child in children.iter() {
        if let Some(n) = arena.as_num(child) {
            const_val = Some(n.clone());
        } else if is_var_squared(arena, child, var) {
            has_pos_x2 = true;
        } else if is_neg_var_squared(arena, child, var) {
            has_neg_x2 = true;
        } else {
            return None;
        }
    }

    let c_val = const_val?;
    if !has_pos_x2 && !has_neg_x2 {
        return None;
    }

    let a_squared = c_val.abs();
    if a_squared.is_zero() {
        return None;
    }

    let half = arena.rational(1, 2);
    let a_sq_is_one = a_squared == num_rational::Ratio::<num_bigint::BigInt>::one();

    // √(base) for reuse in the result
    let sqrt_base = arena.pow(base, half);

    // a² as an expression
    let a_sq_expr = if a_sq_is_one {
        arena.one
    } else {
        arena.num_ratio(a_squared.clone())
    };

    // x/a = x · (a²)^{-1/2}
    let x_over_a = if a_sq_is_one {
        var
    } else {
        let neg_half = arena.rational(-1, 2);
        let a_inv = arena.pow(a_sq_expr, neg_half);
        arena.mul(&[var, a_inv])
    };

    // ── Pattern: a² − x²  (c_val > 0, negative x²) ───────────────
    // ∫ √(a²−x²) dx = ½(x·√(a²−x²) + a²·asin(x/a))
    if has_neg_x2 && c_val.is_positive() {
        let x_sqrt = arena.mul(&[var, sqrt_base]);
        let asin_term = arena.asin(x_over_a);
        let a_sq_asin = if a_sq_is_one {
            asin_term
        } else {
            arena.mul(&[a_sq_expr, asin_term])
        };
        let sum = arena.add(&[x_sqrt, a_sq_asin]);
        return Some(arena.mul(&[half, sum]));
    }

    // ── Pattern: x² + a²  (c_val > 0, positive x²) ───────────────
    // ∫ √(x²+a²) dx = ½(x·√(x²+a²) + a²·asinh(x/a))
    if has_pos_x2 && c_val.is_positive() {
        let x_sqrt = arena.mul(&[var, sqrt_base]);
        let asinh_term = arena.asinh(x_over_a);
        let a_sq_asinh = if a_sq_is_one {
            asinh_term
        } else {
            arena.mul(&[a_sq_expr, asinh_term])
        };
        let sum = arena.add(&[x_sqrt, a_sq_asinh]);
        return Some(arena.mul(&[half, sum]));
    }

    // ── Pattern: x² − a²  (c_val < 0, positive x²) ───────────────
    // ∫ √(x²−a²) dx = ½(x·√(x²−a²) − a²·ln|x + √(x²−a²)|), valid for
    // x > a and x < −a (the acosh(x/a) form is wrong for x < −a).
    if has_pos_x2 && c_val.is_negative() {
        let x_sqrt = arena.mul(&[var, sqrt_base]);
        let ln_term = ln_abs_x_plus_sqrt(arena, var, base);
        let a_sq_ln = if a_sq_is_one {
            ln_term
        } else {
            arena.mul(&[a_sq_expr, ln_term])
        };
        let diff = arena.sub(x_sqrt, a_sq_ln);
        return Some(arena.mul(&[half, diff]));
    }

    None
}

/// Integrate a product of a linear polynomial times `(quadratic)^{-1}`:
///
///   `∫ (ax+b) / (cx²+dx+e) dx`
///
/// Decomposes the linear numerator as a multiple of the derivative of the
/// quadratic denominator plus a constant remainder:
///
///   `ax+b = (a/(2c))·(2cx+d) + (b − ad/(2c))`
///
/// Then:
///   - First part:  `(a/(2c)) · ln|cx²+dx+e|`
///   - Second part: `(b − ad/(2c)) · ∫ 1/(cx²+dx+e) dx`  (completing the square)
fn try_linear_over_quadratic(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    if dependent.len() != 2 {
        return None;
    }

    // Identify linear factor and Pow(quadratic, -1) factor.
    let (linear_idx, pow_idx) = {
        let mut li = None;
        let mut pi = None;
        for (i, &d) in dependent.iter().enumerate() {
            if let ExprNode::Pow(_, _) = arena.node(d) {
                if pi.is_none() {
                    pi = Some(i);
                }
            } else if li.is_none() {
                li = Some(i);
            }
        }
        (li?, pi?)
    };

    let linear = dependent[linear_idx];
    let (pow_base, pow_exp) = match arena.node(dependent[pow_idx]).clone() {
        ExprNode::Pow(b, e) => (b, e),
        _ => return None,
    };

    // Exponent must be exactly −1.
    let exp_val = arena.as_num(pow_exp)?.clone();
    let neg_one = num_rational::Ratio::<num_bigint::BigInt>::from_integer((-1).into());
    if exp_val != neg_one {
        return None;
    }

    // ── Try rational-coefficient path first ────────────────────────
    let lin_poly = crate::poly::polybridge::expr_to_poly(arena, linear, var);
    let quad_poly = crate::poly::polybridge::expr_to_poly(arena, pow_base, var);

    if let (Some(lp), Some(qp)) = (&lin_poly, &quad_poly) {
        tracing::trace!("try_linear_over_quadratic: rational coefficient path");
        if lp.degree() == Some(1) && qp.degree() == Some(2) {
            let a_coeff = lp.coeff(1);
            let b_coeff = lp.coeff(0);
            let c_coeff = qp.coeff(2);
            let d_coeff = qp.coeff(1);

            use num_traits::Zero;
            if !c_coeff.is_zero() && !a_coeff.is_zero() {
                // Decompose: ax+b = (a/(2c))·(2cx+d) + (b − ad/(2c))
                let two_r = num_rational::Ratio::<num_bigint::BigInt>::from_integer(2.into());
                let a_over_2c = &a_coeff / &(&c_coeff * &two_r);
                let remainder = &b_coeff - &(&a_coeff * &d_coeff / &(&c_coeff * &two_r));

                let mut terms: Vec<ExprId> = Vec::new();

                // First term: (a/(2c)) · ln|cx²+dx+e|
                if !a_over_2c.is_zero() {
                    let coeff_id = arena.num_ratio(a_over_2c.clone());
                    let abs_quad = arena.abs(pow_base);
                    let ln_quad = arena.ln(abs_quad);
                    terms.push(arena.mul(&[coeff_id, ln_quad]));
                }

                // Second term: remainder · ∫ 1/(cx²+dx+e) dx
                if !remainder.is_zero() {
                    let inv_quad = arena.pow(pow_base, pow_exp); // (quad)^{-1}
                    let inv_integral =
                        integrate_node(arena, inv_quad, var, var_sym, depth.saturating_sub(1));
                    if matches!(arena.node(inv_integral), ExprNode::Integral(_, _)) {
                        return None;
                    }
                    let rem_id = arena.num_ratio(remainder.clone());
                    terms.push(arena.mul(&[rem_id, inv_integral]));
                }

                return match terms.len() {
                    0 => Some(arena.zero),
                    1 => Some(terms[0]),
                    _ => Some(arena.add(&terms)),
                };
            }
        }

        // Rational polys exist but don't match expected shape.
        return None;
    }

    // ── Symbolic fallback for irrational coefficients ──────────────
    // When expr_to_poly fails (e.g., coefficients contain √5 or √2),
    // extract symbolic coefficients and decompose using arena arithmetic.
    tracing::debug!(
        "try_linear_over_quadratic: expr_to_poly failed, trying symbolic coefficient path"
    );
    let (a_id, b_id) = symbolic_linear_coeff_of(arena, linear, var, var_sym)?;
    let (c_id, d_id, _e_id) = symbolic_quadratic_coeffs(arena, pow_base, var, var_sym)?;
    tracing::debug!(
        "try_linear_over_quadratic: symbolic coefficients extracted for linear/quadratic decomposition"
    );

    // log_coeff = a / (2c)
    let two = arena.int(2);
    let two_c = arena.mul(&[two, c_id]);
    let log_coeff = arena.div(a_id, two_c);
    let log_coeff = crate::transforms::eval::eval(arena, log_coeff);

    // remainder = b − a·d/(2c)
    let a_d = arena.mul(&[a_id, d_id]);
    let a_d_over_2c = arena.div(a_d, two_c);
    let remainder_expr = arena.sub(b_id, a_d_over_2c);
    let remainder_expr = crate::transforms::eval::eval(arena, remainder_expr);

    let mut terms: Vec<ExprId> = Vec::new();

    // A term is dropped only when its coefficient is *proved* to vanish:
    // structurally zero, or a constant shown 0 exactly
    // (`constant_zero_decision`: `√5·√5/5 − 1`).  Up to 0.50 a constant whose
    // `f64` value was below `10⁻¹⁴` counted as 0, so a coefficient `e⁻⁴⁰/2`
    // was dropped; keeping a coefficient that is 0 but not proved is
    // harmless.  A coefficient with a free parameter has no value, and
    // treating that as "zero" returned `∫ (a·x + b)/(x² + 1) dx = 0` — 56
    // wrong answers on the Rubi suite (0.24).
    let known_zero = |arena: &mut Arena, c: ExprId| {
        arena.is_zero_structural(c)
            || Truth::from(crate::transforms::eval::constant_zero_decision(arena, c)).is_true()
    };

    // First term: log_coeff · ln|quadratic|
    if !known_zero(arena, log_coeff) {
        let abs_quad = arena.abs(pow_base);
        let ln_quad = arena.ln(abs_quad);
        terms.push(arena.mul(&[log_coeff, ln_quad]));
    }

    // Second term: remainder · ∫ 1/(cx²+dx+e) dx
    if !known_zero(arena, remainder_expr) {
        let inv_quad = arena.pow(pow_base, pow_exp); // (quad)^{-1}
        let inv_integral = integrate_node(arena, inv_quad, var, var_sym, depth.saturating_sub(1));
        if crate::base::walk::has_unevaluated(arena, inv_integral) {
            return None;
        }
        terms.push(arena.mul(&[remainder_expr, inv_integral]));
    }

    match terms.len() {
        0 => Some(arena.zero),
        1 => Some(terms[0]),
        _ => Some(arena.add(&terms)),
    }
}

/// Check whether `expr` is a rational function of `sin(var)` and `cos(var)`.
///
/// A rational trig function may contain sin(var), cos(var), numeric
/// constants, and arithmetic operations (+, ×, integer powers, negation).
/// The integration variable must appear **only** inside sin/cos.
fn is_rational_trig(arena: &Arena, expr: ExprId, var: ExprId, var_sym: SymbolId) -> bool {
    if !contains_var(arena, expr, var_sym) {
        return true; // constant → trivially rational
    }
    match arena.node(expr).clone() {
        ExprNode::Sin(inner) if inner == var => true,
        ExprNode::Cos(inner) if inner == var => true,
        ExprNode::Add(children) => children
            .iter()
            .all(|&c| is_rational_trig(arena, c, var, var_sym)),
        ExprNode::Mul(children) => children
            .iter()
            .all(|&c| is_rational_trig(arena, c, var, var_sym)),
        ExprNode::Neg(inner) => is_rational_trig(arena, inner, var, var_sym),
        ExprNode::Pow(base, exp) => {
            if !contains_var(arena, exp, var_sym)
                && let Some(e) = arena.as_num(exp)
                && e.is_integer()
            {
                return is_rational_trig(arena, base, var, var_sym);
            }
            false
        }
        _ => false,
    }
}

/// Apply the Weierstrass (half-angle tangent) substitution to integrate a
/// rational function of `sin(var)` and `cos(var)`.
///
/// Substitution: `t = tan(var/2)`, giving
///   - `sin(var) = 2t/(1+t²)`
///   - `cos(var) = (1−t²)/(1+t²)`
///   - `dx        = 2/(1+t²) dt`
///
/// After substitution the integrand becomes a rational function of `t`.
/// With parameters, all declared real, an integrand with `f(x + π) = f(x)`
/// first goes through `t = tan x` ([`try_bioche_tan_substitution`]).
fn try_weierstrass_substitution(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    if depth < 3 {
        return None;
    }
    if !is_rational_trig(arena, expr, var, var_sym) {
        return None;
    }
    // The jump at the poles of `tan(x/2)` with parameters not declared
    // real, as for real ones (up to 0.37 the answer was kept uncorrected:
    // `∫ dx/(a + b·cos x)` dropped by `2π/√(a² − b²)` at every odd multiple
    // of `π` for `a > |b|`; the continuity hunter's 118 jumps).
    let _jumps = crate::transforms::trig_integ::RealParameterJumps::enable(false);
    if has_only_real_parameters(arena, expr, var_sym)
        && let Some(result) = try_bioche_tan_substitution(arena, expr, var, var_sym, depth, false)
    {
        return Some(result);
    }

    tracing::debug!("trying Weierstrass substitution");

    let t = arena.symbol("__wt");
    let ExprNode::Symbol(t_sym) = *arena.node(t) else {
        return None;
    };

    let two = arena.int(2);
    let one = arena.one;
    let t_sq = arena.pow(t, two);
    let one_plus_t_sq = arena.add(&[one, t_sq]);

    let sin_var = arena.sin(var);
    let cos_var = arena.cos(var);

    // sin(var) → 2t/(1+t²)
    let two_t = arena.mul(&[two, t]);
    let sin_sub = arena.div(two_t, one_plus_t_sq);

    // cos(var) → (1−t²)/(1+t²)
    let one_minus_t_sq = arena.sub(one, t_sq);
    let cos_sub = arena.div(one_minus_t_sq, one_plus_t_sq);

    // dx factor: 2/(1+t²)
    let dx_factor = arena.div(two, one_plus_t_sq);

    // Apply substitution
    let mut sub_expr = arena.subs_structural(expr, sin_var, sin_sub);
    sub_expr = arena.subs_structural(sub_expr, cos_var, cos_sub);

    // Multiply by dx factor
    let integrand_t = arena.mul(&[sub_expr, dx_factor]);

    // Aggressively simplify / cancel
    let integrand_t = crate::transforms::eval::eval(arena, integrand_t);
    let integrand_t = crate::transforms::expand::expand(arena, integrand_t);
    let integrand_t = crate::transforms::eval::eval(arena, integrand_t);
    let integrand_t = arena.cancel_expr(integrand_t, t);
    let integrand_t = crate::transforms::eval::eval(arena, integrand_t);
    // Clear nested fractions such as 2/((1+t²)(2 + (1−t²)/(1+t²))) by
    // normalising to a single numerator/denominator pair.
    let integrand_t = clear_nested_fractions(arena, integrand_t, t);
    tracing::debug!(integrand_t = %arena.display(integrand_t), "weierstrass: integrand in t");

    // Integrate w.r.t. t
    let integral_t = integrate_node(arena, integrand_t, t, t_sym, depth.saturating_sub(2));

    if crate::base::walk::has_unevaluated(arena, integral_t) {
        return None;
    }
    // The integrand in `t` is a rational function; check the closed form
    // before it is dressed up in tan(x/2) (the rational integrator's
    // nested-radical failures show up here as e.g. `atan(3·tan(x/2))` for
    // `∫ cos x/(sin²x + 1)`).
    if candidate_rejected(arena, integrand_t, integral_t, t, t_sym) {
        tracing::debug!("weierstrass: closed form in t failed verification");
        return None;
    }

    // Substitute back, t → tan(var/2), continuously across var = π + 2kπ
    // (up to 0.37 `∫ dx/(2 + cos x)` was `2/√3·atan(tan(x/2)/√3)`, which
    // drops by 2π/√3 there; definite integrals taken from it were wrong).
    let half = arena.rational(1, 2);
    let half_var = arena.mul(&[half, var]);
    match crate::transforms::trig_integ::continuous_through_tan(
        arena, integral_t, t, t_sym, half_var,
    ) {
        Some(result) => Some(result),
        None if undecided_jump_kept(arena, integrand_t, integral_t, t, t_sym) => {
            tracing::debug!("weierstrass: the jump at x = π is not decided; uncorrected form");
            let tan_half = arena.tan(half_var);
            let back = arena.subs_structural(integral_t, t, tan_half);
            Some(crate::transforms::trig_integ::atan_tan_terms_to_argument(
                arena, back, half_var,
            ))
        }
        None => {
            tracing::debug!("weierstrass: the jump of G(tan(x/2)) at x = π is not decided");
            None
        }
    }
}

/// Does `expr` have parameters (free symbols other than the variable), all
/// of them declared real?
fn has_only_real_parameters(arena: &Arena, expr: ExprId, var_sym: SymbolId) -> bool {
    let params: Vec<SymbolId> = crate::base::walk::free_symbols(arena, expr)
        .into_iter()
        .filter_map(|s| match *arena.node(s) {
            ExprNode::Symbol(sid) if sid != var_sym => Some(sid),
            _ => None,
        })
        .collect();
    !params.is_empty()
        && params
            .iter()
            .all(|&sid| crate::transforms::realness::symbol_declared_real(arena, sid))
}

/// Bioche's rule for a rational function `f` of `sin x`, `cos x` with
/// `f(x + π) = f(x)`: the substitution `t = tan x` (`sin x = t·c`, `cos x =
/// c`, `c² = 1/(1 + t²)`, `dx = dt/(1 + t²)`) gives a rational function of
/// `t` of half the degree the half-angle substitution gives.  For real
/// parameters that decides the jump of `G(tan x)` at the poles of `tan x`
/// where the half-angle route's cannot be: `∫ dx/(a + b·sin²x)` (`a`, `b > 0`)
/// is `∫ dt/(a + (a + b)·t²)`, an `atan` with a positive coefficient, where
/// `tan(x/2)` gives a quartic whose roots' half-planes depend on the
/// parameters (up to 0.37 the answer was a sum of four logarithms that
/// jumped at every odd multiple of `π`).  Rubi integrates these integrands
/// so (`atan(√(a + b)·tan x/√a)/√(a(a + b))`); the floor term is the
/// continuity correction of [`crate::transforms::trig_integ::continuous_through_tan`].
///
/// `f(tc, c)` is even in `c` exactly when `f(x + π) = f(x)`; its numerator
/// and denominator (cancelled) are then both even or both odd in `c`.
/// `None` when `f` is not so invariant, the integral in `t` is not found or
/// fails the check, or its jump at `t = ±∞` is not decided (the caller
/// then takes the half-angle route) — unless `keep_undecided`, when such a
/// jump is one [`jump_kept_undecided`] lets the half-angle route keep: the
/// answer is then `G(tan x)` uncorrected, as that route would give it (of
/// half the degree in `t`).
fn try_bioche_tan_substitution(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
    keep_undecided: bool,
) -> Option<ExprId> {
    let t = arena.symbol("__bt");
    let c = arena.symbol("__bc");
    let ExprNode::Symbol(t_sym) = *arena.node(t) else {
        return None;
    };
    let sin_var = arena.sin(var);
    let cos_var = arena.cos(var);
    let t_c = arena.mul(&[t, c]);
    let substituted = arena.subs_structural(expr, sin_var, t_c);
    let substituted = arena.subs_structural(substituted, cos_var, c);
    if contains_var(arena, substituted, var_sym) {
        return None;
    }
    let in_c = clear_nested_fractions(arena, substituted, c);
    let (num, den) = crate::poly::polybridge::as_numer_denom(arena, in_c);
    let num_c = crate::transforms::solve::symbolic_poly_coeffs(arena, num, c)?;
    let den_c = crate::transforms::solve::symbolic_poly_coeffs(arena, den, c)?;
    let vanishes = |arena: &mut Arena, cs: &[ExprId], parity: usize| {
        cs.iter()
            .skip(parity)
            .step_by(2)
            .all(|&k| Truth::from(crate::poly::algebraic::is_zero_checked(arena, k)).is_true())
    };
    // `shift` 0: both even; 1: both odd (divided by `c`).
    let shift = if vanishes(arena, &num_c, 1) && vanishes(arena, &den_c, 1) {
        0
    } else if vanishes(arena, &num_c, 0) && vanishes(arena, &den_c, 0) {
        1
    } else {
        return None;
    };
    let one = arena.one;
    let two = arena.int(2);
    let t_sq = arena.pow(t, two);
    let one_plus_t_sq = arena.add(&[one, t_sq]);
    let in_t = |arena: &mut Arena, cs: &[ExprId]| -> ExprId {
        let mut terms: Vec<ExprId> = Vec::new();
        for (j, &k) in cs.iter().enumerate().skip(shift).step_by(2) {
            let half_degree = i64::try_from((j - shift) / 2).unwrap_or(0);
            let e = arena.int(-half_degree);
            let c_pow = arena.pow(one_plus_t_sq, e);
            terms.push(arena.mul(&[k, c_pow]));
        }
        arena.add(&terms)
    };
    let num_t = in_t(arena, &num_c);
    let den_t = in_t(arena, &den_c);
    let dt = arena.mul(&[den_t, one_plus_t_sq]);
    let integrand_t = arena.div(num_t, dt);
    let integrand_t = crate::transforms::eval::eval(arena, integrand_t);
    let integrand_t = clear_nested_fractions(arena, integrand_t, t);
    if contains_var(arena, integrand_t, var_sym)
        || crate::base::walk::contains(arena, integrand_t, c)
    {
        return None;
    }
    tracing::debug!(integrand_t = %arena.display(integrand_t), "bioche: integrand in tan x");
    let integral_t = integrate_node(arena, integrand_t, t, t_sym, depth.saturating_sub(2));
    if crate::base::walk::has_unevaluated(arena, integral_t)
        || candidate_rejected(arena, integrand_t, integral_t, t, t_sym)
    {
        return None;
    }
    let result = match crate::transforms::trig_integ::continuous_through_tan(
        arena, integral_t, t, t_sym, var,
    ) {
        Some(result) => result,
        None if keep_undecided && undecided_jump_kept(arena, integrand_t, integral_t, t, t_sym) => {
            let tan_var = arena.tan(var);
            let back = arena.subs_structural(integral_t, t, tan_var);
            crate::transforms::trig_integ::atan_tan_terms_to_argument(arena, back, var)
        }
        None => return None,
    };
    if candidate_rejected(arena, expr, result, var, var_sym) {
        tracing::debug!("bioche: the antiderivative in x failed verification");
        return None;
    }
    Some(result)
}

/// Largest multiple `k` of the common angle in
/// [`try_trig_rational_substitution`]: `sin k·u` becomes a polynomial of
/// degree `k` in `sin u`, `cos u`.
const MAX_ANGLE_MULTIPLE: i64 = 12;

/// `(sin k·u, cos k·u)` as polynomials in `s = sin u`, `c = cos u` (`k ≥ 1`):
/// `cos k·u = Tₖ(c)`, `sin k·u = s·Uₖ₋₁(c)` (Chebyshev polynomials, DLMF
/// 18.5.1–18.5.2), by the recurrences `Tₙ₊₁ = 2c·Tₙ − Tₙ₋₁`, `Uₙ₊₁ = 2c·Uₙ −
/// Uₙ₋₁`.
fn multiple_angle(arena: &mut Arena, k: i64, s: ExprId, c: ExprId) -> (ExprId, ExprId) {
    let two = arena.int(2);
    let two_c = arena.mul(&[two, c]);
    let minus_one = arena.int(-1);
    let (mut t_prev, mut t_cur) = (arena.one, c);
    let (mut u_prev, mut u_cur) = (arena.zero, arena.one);
    for _ in 1..k {
        let t_lead = arena.mul(&[two_c, t_cur]);
        let t_back = arena.mul(&[minus_one, t_prev]);
        let t_next = arena.add(&[t_lead, t_back]);
        let t_next = crate::transforms::expand::expand(arena, t_next);
        let u_lead = arena.mul(&[two_c, u_cur]);
        let u_back = arena.mul(&[minus_one, u_prev]);
        let u_next = arena.add(&[u_lead, u_back]);
        let u_next = crate::transforms::expand::expand(arena, u_next);
        (t_prev, t_cur) = (t_cur, t_next);
        (u_prev, u_cur) = (u_cur, u_next);
    }
    let sin_k = arena.mul(&[s, u_cur]);
    let sin_k = crate::transforms::expand::expand(arena, sin_k);
    (sin_k, t_cur)
}

/// Is the polynomial `p` (in the symbol `v`, other symbols as coefficients)
/// even (`Some(0)`) or odd (`Some(1)`) in `v`?  `None` when it has both
/// kinds of powers, is zero, or is not a polynomial in `v`.  A coefficient
/// counts as zero when it is so structurally or proved so.
fn parity_in(arena: &mut Arena, p: ExprId, v: ExprId) -> Option<usize> {
    let coeffs = crate::transforms::solve::symbolic_poly_coeffs(arena, p, v)?;
    let mut nonzero = [false, false];
    for (j, &k) in coeffs.iter().enumerate() {
        let zero = arena.is_zero_structural(k)
            || (crate::base::walk::free_symbols(arena, k).is_empty()
                && Truth::from(crate::poly::algebraic::is_zero_checked(arena, k)).is_true());
        if !zero {
            nonzero[j % 2] = true;
        }
    }
    match nonzero {
        [true, false] => Some(0),
        [false, true] => Some(1),
        _ => None,
    }
}

/// `p = κ·p̃` for a polynomial `p` in the symbols `vars` whose coefficients are
/// all constant multiples of one of them, `κ`, which has free parameters:
/// `(κ, p̃)` with `p̃` free of parameters (`a + a·s` = `a·(1 + s)`).  `None`
/// otherwise.  The jump of an antiderivative through `tan w` is decided
/// for `p̃` where it is not for `p` (a parameter not declared real may be
/// complex), and it scales with `κ`.
fn parameter_content(arena: &mut Arena, p: ExprId, vars: &[ExprId]) -> Option<(ExprId, ExprId)> {
    let mut monomials: Vec<(ExprId, ExprId)> = vec![(p, arena.one)];
    for &v in vars {
        let mut next: Vec<(ExprId, ExprId)> = Vec::new();
        for (q, mono) in monomials {
            let coeffs = crate::transforms::solve::symbolic_poly_coeffs(arena, q, v)?;
            for (j, k) in coeffs.into_iter().enumerate() {
                if arena.is_zero_structural(k) {
                    continue;
                }
                let e = arena.int(i64::try_from(j).ok()?);
                let v_j = arena.pow(v, e);
                next.push((k, arena.mul(&[mono, v_j])));
            }
        }
        monomials = next;
    }
    let &(lead, _) = monomials.first()?;
    if crate::base::walk::free_symbols(arena, lead).is_empty() {
        return None;
    }
    let mut terms: Vec<ExprId> = Vec::with_capacity(monomials.len());
    for (k, mono) in monomials {
        let ratio = arena.div(k, lead);
        let ratio = crate::transforms::eval::eval(arena, ratio);
        if !crate::base::walk::free_symbols(arena, ratio).is_empty() {
            return None;
        }
        terms.push(arena.mul(&[ratio, mono]));
    }
    Some((lead, arena.add(&terms)))
}

/// `R/w` for a rational function `R = num/den` of `w` and `o` odd in `w`
/// (`num` and `den` of the parities `pn ≠ pd` in `w`), written in `t` with
/// `w² = 1 − t²`, `o = t`: the integrand in `t` of Bioche's substitutions
/// `t = cos u` (`w = sin u`, `dt = −w·du`) and `t = sin u` (`w = cos u`,
/// `dt = w·du`).  `R/w = w^(pn − pd − 1)·Ñ(w²)/D̃(w²)`, the power `w⁰` or
/// `w⁻²`.
fn odd_part_in_t(
    arena: &mut Arena,
    fraction: (ExprId, ExprId),
    parities: (usize, usize),
    w: ExprId,
    o: ExprId,
    t: ExprId,
) -> Option<ExprId> {
    let one = arena.one;
    let two = arena.int(2);
    let t_sq = arena.pow(t, two);
    let minus_t_sq = arena.neg(t_sq);
    let w_sq = arena.add(&[one, minus_t_sq]);
    let reduce = |arena: &mut Arena, p: ExprId, parity: usize| -> Option<ExprId> {
        let coeffs = crate::transforms::solve::symbolic_poly_coeffs(arena, p, w)?;
        let mut terms: Vec<ExprId> = Vec::new();
        for (j, &k) in coeffs.iter().enumerate().skip(parity).step_by(2) {
            let half = arena.int(i64::try_from((j - parity) / 2).ok()?);
            let k_t = arena.subs_structural(k, o, t);
            let power = arena.pow(w_sq, half);
            terms.push(arena.mul(&[k_t, power]));
        }
        Some(arena.add(&terms))
    };
    let n = reduce(arena, fraction.0, parities.0)?;
    let d = reduce(arena, fraction.1, parities.1)?;
    let d = if parities.0 == 1 {
        d
    } else {
        arena.mul(&[w_sq, d])
    };
    Some(arena.div(n, d))
}

/// Is `e` a rational function of the symbols `vars` (sums, products and
/// integer powers of them and of constants)?
fn is_rational_in_all(arena: &Arena, e: ExprId, vars: &[ExprId]) -> bool {
    vars.iter().all(|&v| match *arena.node(v) {
        ExprNode::Symbol(sid) => is_rational_in(arena, e, sid),
        _ => false,
    })
}

/// `∫ R(sin L₁, cos L₁, tan L₁, …, tan Lₖ) dx` for a rational function `R`
/// (with parameters) of trigonometric functions of linear arguments `Lᵢ =
/// αᵢ·x + βᵢ` whose rates are rational multiples of one another (`sec`,
/// `csc`, `cot` are written with `sin`, `cos`).  `u = g·(α₁·x + β₁)` (`g`
/// the gcd of the ratios `αᵢ/α₁`) makes every `Lᵢ = kᵢ·u + γᵢ` with an
/// integer `kᵢ`, and the multiple-angle and addition formulas give a
/// rational function `R̃(sin u, cos u)`; `du = g·α₁·dx`.  Bioche's rules
/// then pick the substitution that keeps the rational function in `t`
/// small:
///
/// - `R̃` odd in `sin u` (`R̃(u)·du` invariant under `u → −u`): `t = cos u`;
/// - odd in `cos u` (invariant under `u → π − u`): `t = sin u`;
/// - invariant under `u → π + u`: `t = tan u`
///   ([`try_bioche_tan_substitution`]);
/// - otherwise `t = tan(u/2)` ([`try_weierstrass_substitution`]).
///
/// The first two have no poles, so `G(sin u)` and `G(cos u)` are continuous
/// wherever the integrand is, whatever the parameters (`∫ cos u/(a +
/// b·sin u)² du = − 1/(b·(a + b·sin u))`); the last two add the floor
/// correction of [`crate::transforms::trig_integ::continuous_through_tan`].
/// Its jump `J` is decided with every parameter taken for real
/// ([`crate::transforms::trig_integ::RealParameterJumps`]: exact for real
/// values, `J·⌊w/π + 1/2⌋` locally constant for others), and an answer whose
/// jump is still not decided is returned only where the integrand is not
/// continuous across the poles ([`undecided_jump_kept`]).  A parameter
/// that multiplies the whole numerator or denominator (`(a + a·sin u)²`) is
/// split off first ([`parameter_content`]), and parameters only in the
/// numerator are taken term by term ([`trig_rational_in_u`]), which decides
/// `J` without signs.  Radicals: algebraic functions of `tan u`
/// ([`tan_algebraic_in_u`]) and radicals of `sin u` (`cos u`) times an odd
/// power of `cos u` (`sin u`) ([`odd_algebraic_in_u`]); a polynomial factor
/// in `x` goes through [`trig_polynomial_substitution`].  Before 0.38 a
/// trigonometric argument other than `x` itself went to the half-angle
/// substitution only for `sin`, `cos` powers, and `tan` never: `∫ dx/(5 +
/// 3·sin(2x + 1))`, `∫ dx/(2 + 3·tan x)` and 3,395 rational trigonometric
/// integrands of the Rubi suite (4.1–4.7, nearly all over `c + d·x`)
/// stayed unevaluated.  The substitutions follow the mathematics of Rubi's
/// rules for these integrands (4.1–4.7), not their form.  `None` when an
/// argument is not linear, the rates are not commensurable, a multiple
/// exceeds [`MAX_ANGLE_MULTIPLE`], the integrand is of none of these
/// forms, or no substitution gives a verified answer.
fn try_trig_rational_substitution(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    use num_integer::Integer;
    let mut trig: Vec<(ExprId, ExprId, ExprId)> = Vec::new();
    for id in crate::base::walk::post_order_ids(arena, expr) {
        let arg = match *arena.node(id) {
            ExprNode::Sin(a) | ExprNode::Cos(a) | ExprNode::Tan(a) => a,
            _ => continue,
        };
        if !contains_var(arena, arg, var_sym) {
            continue;
        }
        let (alpha, beta) = symbolic_linear_coeff_of(arena, arg, var, var_sym)?;
        trig.push((id, alpha, beta));
    }
    let &(_, alpha0, beta0) = trig.first()?;
    let mut ratios: Vec<Q> = Vec::with_capacity(trig.len());
    for &(_, alpha, _) in &trig {
        let r = arena.div(alpha, alpha0);
        let r = crate::transforms::eval::eval(arena, r);
        ratios.push(arena.as_num(r)?.clone());
    }
    let mut num_gcd = num_bigint::BigInt::zero();
    let mut den_lcm = num_bigint::BigInt::one();
    for r in &ratios {
        num_gcd = num_gcd.gcd(r.numer());
        den_lcm = den_lcm.lcm(r.denom());
    }
    if num_gcd.is_zero() {
        return None;
    }
    let g = Q::new(num_gcd, den_lcm);
    let u = arena.symbol("__tru");
    let s = arena.symbol("__trs");
    let c = arena.symbol("__trc");
    let ExprNode::Symbol(u_sym) = *arena.node(u) else {
        return None;
    };
    let mut sub = expr;
    for (&(node, _, beta), r) in trig.iter().zip(&ratios) {
        let k = r / &g;
        if !k.is_integer() {
            return None;
        }
        let k = i64::try_from(k.to_integer()).ok()?;
        if k == 0 || k.abs() > MAX_ANGLE_MULTIPLE {
            return None;
        }
        let r_id = arena.num_ratio(r.clone());
        let r_beta0 = arena.mul(&[r_id, beta0]);
        let gamma = arena.sub(beta, r_beta0);
        let gamma = crate::transforms::eval::eval(arena, gamma);
        let (sin_k, cos_k) = multiple_angle(arena, k.abs(), s, c);
        let sin_k = if k < 0 { arena.neg(sin_k) } else { sin_k };
        let (sin_l, cos_l) = if arena.is_zero_structural(gamma) {
            (sin_k, cos_k)
        } else {
            let sin_g = arena.sin(gamma);
            let cos_g = arena.cos(gamma);
            let a = arena.mul(&[sin_k, cos_g]);
            let b = arena.mul(&[cos_k, sin_g]);
            let sin_l = arena.add(&[a, b]);
            let a = arena.mul(&[cos_k, cos_g]);
            let b = arena.mul(&[sin_k, sin_g]);
            let cos_l = arena.sub(a, b);
            (sin_l, cos_l)
        };
        let repl = match arena.node(node) {
            ExprNode::Sin(_) => sin_l,
            ExprNode::Cos(_) => cos_l,
            _ => arena.div(sin_l, cos_l),
        };
        sub = arena.subs_structural(sub, node, repl);
    }
    let rate = arena.num_ratio(g.clone());
    let rate = arena.mul(&[rate, alpha0]);
    let shift = arena.num_ratio(g);
    let shift = arena.mul(&[shift, beta0]);
    let shift = crate::transforms::eval::eval(arena, shift);
    if contains_var(arena, sub, var_sym) {
        return trig_polynomial_substitution(arena, expr, sub, (s, c), (rate, shift), var, var_sym);
    }
    // The jumps of answers through `tan w` with parameters, as for real
    // ones (exact for real values, locally constant otherwise), of the
    // answer with the logarithms the stage exit writes `ln u`.
    let undeclared = |arena: &Arena, e: ExprId| {
        crate::base::walk::free_symbols(arena, e)
            .into_iter()
            .any(|v| match *arena.node(v) {
                ExprNode::Symbol(sid) => {
                    !crate::transforms::realness::symbol_declared_real(arena, sid)
                }
                _ => true,
            })
    };
    let plain_logs = undeclared(arena, rate) || undeclared(arena, shift);
    let _jumps = crate::transforms::trig_integ::RealParameterJumps::enable(plain_logs);
    let _strict = crate::transforms::trig_integ::StrictJumps::enable();
    let (kappa, result_u) = if is_rational_in_all(arena, sub, &[s, c]) {
        tracing::debug!(
            "integrate: a rational function of the trigonometric functions of a linear argument"
        );
        let r = clear_nested_fractions(arena, sub, s);
        let (num, den) = crate::poly::polybridge::as_numer_denom(arena, r);
        let num = crate::transforms::expand::expand(arena, num);
        let den = crate::transforms::expand::expand(arena, den);
        // A parameter that multiplies the whole numerator or denominator.
        let (num_k, num) = parameter_content(arena, num, &[s, c]).unwrap_or((arena.one, num));
        let (den_k, den) = parameter_content(arena, den, &[s, c]).unwrap_or((arena.one, den));
        let kappa = arena.div(num_k, den_k);
        (
            kappa,
            trig_rational_in_u(arena, (num, den), (s, c), u, u_sym)?,
        )
    } else {
        let result = match odd_algebraic_in_u(arena, sub, (s, c), u) {
            Some(r) => r,
            None => match tan_algebraic_in_u(arena, sub, (s, c), u) {
                Some(r) => r,
                None => half_angle_radical_in_u(arena, sub, (s, c), u)?,
            },
        };
        (arena.one, result)
    };
    let l = arena.mul(&[rate, var]);
    let l = arena.add(&[l, shift]);
    let back = arena.subs_structural(result_u, u, l);
    let minus_one = arena.int(-1);
    let inv_rate = arena.pow(rate, minus_one);
    let back = arena.mul(&[kappa, inv_rate, back]);
    let back = crate::transforms::eval::eval(arena, back);
    if candidate_rejected(arena, expr, back, var, var_sym) {
        tracing::debug!(
            "integrate: rejecting an unverified trigonometric-substitution closed form"
        );
        return None;
    }
    Some(back)
}

/// The terms of the polynomial `p` in the symbols `vars` grouped by their
/// constant factors with free parameters: `p = Σ πₖ·pₖ` with every `pₖ` free
/// of parameters (`A + B·s + C·s²` = `A·1 + B·s + C·s²`).
fn parameter_terms(arena: &mut Arena, p: ExprId, vars: &[ExprId]) -> Vec<(ExprId, ExprId)> {
    let p = crate::transforms::expand::expand(arena, p);
    let terms: Vec<ExprId> = match arena.node(p) {
        ExprNode::Add(children) => children.to_vec(),
        _ => vec![p],
    };
    let mut groups: Vec<(ExprId, Vec<ExprId>)> = Vec::new();
    for term in terms {
        let factors: Vec<ExprId> = match arena.node(term) {
            ExprNode::Mul(children) => children.to_vec(),
            _ => vec![term],
        };
        let mut key: Vec<ExprId> = Vec::new();
        let mut part: Vec<ExprId> = Vec::new();
        for f in factors {
            let dependent = vars.iter().any(|&v| contains_var_id(arena, f, v));
            if !dependent && !crate::base::walk::free_symbols(arena, f).is_empty() {
                key.push(f);
            } else {
                part.push(f);
            }
        }
        let key = arena.mul(&key);
        let part = arena.mul(&part);
        match groups.iter_mut().find(|(k, _)| *k == key) {
            Some((_, parts)) => parts.push(part),
            None => groups.push((key, vec![part])),
        }
    }
    groups
        .into_iter()
        .map(|(k, parts)| (k, arena.add(&parts)))
        .collect()
}

/// [`try_trig_rational_substitution`] for an integrand with the variable
/// also outside the trigonometric functions, polynomially (`(c + d·x)²·
/// cos²(a + b·x)·sin³(a + b·x)`, Rubi 4.x.10, 4.7.3): `sub` is the
/// integrand with the trigonometric functions written in `s = sin u`, `c =
/// cos u` (`sc`), `u = rate·x + shift` (`linear`).  With `x = (u −
/// shift)/rate` the integrand in `u` has `u` as every argument (multiple
/// angles expanded: `csc(a + b·x)·sin(3a + 3b·x)` is `3 − 4·sin²u`), and
/// goes through the whole pipeline (by parts takes linear arguments only
/// as `x` itself).  `None` when the substitution is `u = x`, the variable
/// is not polynomial in `sub`, or no answer is found.
fn trig_polynomial_substitution(
    arena: &mut Arena,
    expr: ExprId,
    sub: ExprId,
    sc: (ExprId, ExprId),
    linear: (ExprId, ExprId),
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    let (s, c) = sc;
    let (rate, shift) = linear;
    if !is_rational_in_all(arena, sub, &[s, c])
        || crate::transforms::solve::symbolic_poly_coeffs(arena, sub, var).is_none()
    {
        return None;
    }
    let u = arena.symbol("__trp");
    let minus_shift = arena.neg(shift);
    let u_minus = arena.add(&[u, minus_shift]);
    let x_of_u = arena.div(u_minus, rate);
    let in_u = arena.subs_structural(sub, var, x_of_u);
    let sin_u = arena.sin(u);
    let cos_u = arena.cos(u);
    let in_u = arena.subs_structural(in_u, s, sin_u);
    let in_u = arena.subs_structural(in_u, c, cos_u);
    let in_u = arena.div(in_u, rate);
    let in_u = crate::transforms::eval::eval(arena, in_u);
    // `u = x` with nothing rewritten: the pipeline has seen this integrand.
    let same = arena.subs_structural(in_u, u, var);
    if same == expr {
        return None;
    }
    tracing::debug!(integrand_u = %arena.display(in_u), "integrate: polynomial times trigonometric functions of a linear argument");
    let res = integrate_nested_generic(arena, in_u, u)?;
    let l = arena.mul(&[rate, var]);
    let l = arena.add(&[l, shift]);
    let back = arena.subs_structural(res, u, l);
    let back = crate::transforms::eval::eval(arena, back);
    if candidate_rejected(arena, expr, back, var, var_sym) {
        return None;
    }
    Some(back)
}

/// `∫ R(sin u, cos u) du` for `R = num/den`, polynomials in the symbols
/// `s = sin u`, `c = cos u` (`sc`), by Bioche's rules (see
/// [`try_trig_rational_substitution`]).  With a denominator free of
/// parameters, a numerator `Σ πₖ·pₖ` ([`parameter_terms`]) is integrated term
/// by term: each `∫ pₖ/den` is free of parameters, so the jump of an
/// answer through `tan w` is decided for it (`∫ (A + B·cos u)/(1 + cos u)²
/// du`; the jump of the whole depends on whether `A`, `B` are real).
fn trig_rational_in_u(
    arena: &mut Arena,
    fraction: (ExprId, ExprId),
    sc: (ExprId, ExprId),
    u: ExprId,
    u_sym: SymbolId,
) -> Option<ExprId> {
    let (num, den) = fraction;
    let (s, c) = sc;
    let parametric = |arena: &Arena, e: ExprId| {
        crate::base::walk::free_symbols(arena, e)
            .into_iter()
            .any(|v| v != s && v != c)
    };
    let groups = if !parametric(arena, den) && parametric(arena, num) {
        parameter_terms(arena, num, &[s, c])
    } else {
        Vec::new()
    };
    if groups.len() < 2 {
        return trig_rational_single(arena, fraction, sc, u, u_sym);
    }
    let mut sum: Vec<ExprId> = Vec::with_capacity(groups.len());
    for (key, part) in groups {
        let r = trig_rational_single(arena, (part, den), sc, u, u_sym)?;
        sum.push(arena.mul(&[key, r]));
    }
    Some(arena.add(&sum))
}

/// [`trig_rational_in_u`] for one fraction.
fn trig_rational_single(
    arena: &mut Arena,
    fraction: (ExprId, ExprId),
    sc: (ExprId, ExprId),
    u: ExprId,
    u_sym: SymbolId,
) -> Option<ExprId> {
    let (num, den) = fraction;
    let (s, c) = sc;
    let ratio = arena.div(num, den);
    if !contains_var_id(arena, ratio, s) && !contains_var_id(arena, ratio, c) {
        return Some(arena.mul(&[ratio, u]));
    }
    let t = arena.symbol("__trt");
    let ExprNode::Symbol(t_sym) = *arena.node(t) else {
        return None;
    };
    let sin_u = arena.sin(u);
    let cos_u = arena.cos(u);
    // `t = cos u` when `R` is odd in `sin u`, `t = sin u` when odd in `cos u`.
    for (w, o, back, sign) in [(s, c, cos_u, -1), (c, s, sin_u, 1)] {
        let (Some(pn), Some(pd)) = (parity_in(arena, num, w), parity_in(arena, den, w)) else {
            continue;
        };
        if pn == pd {
            continue;
        }
        let Some(in_t) = odd_part_in_t(arena, (num, den), (pn, pd), w, o, t) else {
            continue;
        };
        let sign = arena.int(sign);
        let in_t = arena.mul(&[sign, in_t]);
        let in_t = clear_nested_fractions(arena, in_t, t);
        tracing::debug!(integrand_t = %arena.display(in_t), "bioche: integrand in sin u or cos u");
        let g = integrate_node(arena, in_t, t, t_sym, 18);
        if crate::base::walk::has_unevaluated(arena, g)
            || candidate_rejected(arena, in_t, g, t, t_sym)
        {
            continue;
        }
        return Some(arena.subs_structural(g, t, back));
    }
    let r_s = arena.subs_structural(ratio, s, sin_u);
    let r_u = arena.subs_structural(r_s, c, cos_u);
    let r_u = crate::transforms::eval::eval(arena, r_u);
    let lam = arena.symbol("__trl");
    let joint = |arena: &mut Arena, p: ExprId| -> Option<usize> {
        let ls = arena.mul(&[lam, s]);
        let lc = arena.mul(&[lam, c]);
        let p = arena.subs_structural(p, s, ls);
        let p = arena.subs_structural(p, c, lc);
        parity_in(arena, p, lam)
    };
    if let (Some(pn), Some(pd)) = (joint(arena, num), joint(arena, den))
        && pn == pd
        && let Some(result) = try_bioche_tan_substitution(arena, r_u, u, u_sym, 20, true)
    {
        return Some(result);
    }
    try_weierstrass_substitution(arena, r_u, u, u_sym, 20)
}

/// `∫ R(x) dx` for a rational function `R` with `i` in its coefficients
/// (Gaussian rationals, parameters allowed).  With `i` taken for a symbol
/// `ι` (`ι² = −1`), `R = (N₀ + ι·N₁)/(D₀ + ι·D₁)` with `N₀`, `N₁`, `D₀`, `D₁`
/// free of `i`, and `R = ((N₀D₀ + N₁D₁) + i·(N₁D₀ − N₀D₁))/(D₀² + D₁²)`:
/// two rational functions free of `i`, integrated separately.  This is an
/// identity of rational functions, so no parameter has to be real.  The
/// rational integrators work over ℚ and ℚ(parameters): up to 0.37 `∫ (1 +
/// i·t)⁵/(1 + t²)⁶ dt` stayed unevaluated, and with it the Rubi suite's
/// `(a + i·a·tan(c + d·x))ⁿ` families through `t = tan x`.  For a real
/// variable both parts come out as real antiderivatives (`ln|·|`, `atan`),
/// continuous wherever `R` is.
fn try_gaussian_rational(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    let i = arena.i_unit();
    if depth < 2
        || !crate::base::walk::contains(arena, expr, i)
        || !is_rational_in(arena, expr, var_sym)
    {
        return None;
    }
    let iota = arena.symbol("__gi");
    let e = arena.subs_structural(expr, i, iota);
    if crate::base::walk::contains(arena, e, i) {
        return None;
    }
    let e = clear_nested_fractions(arena, e, var);
    let (n, d) = crate::poly::polybridge::as_numer_denom(arena, e);
    let split = |arena: &mut Arena, p: ExprId| -> Option<(ExprId, ExprId)> {
        let coeffs = crate::transforms::solve::symbolic_poly_coeffs(arena, p, iota)?;
        let mut re: Vec<ExprId> = Vec::new();
        let mut im: Vec<ExprId> = Vec::new();
        for (k, &c) in coeffs.iter().enumerate() {
            let c = if (k / 2) % 2 == 0 { c } else { arena.neg(c) };
            if k % 2 == 0 {
                re.push(c);
            } else {
                im.push(c);
            }
        }
        let re = arena.add(&re);
        let im = arena.add(&im);
        Some((
            crate::transforms::eval::eval(arena, re),
            crate::transforms::eval::eval(arena, im),
        ))
    };
    let (n0, n1) = split(arena, n)?;
    let (d0, d1) = split(arena, d)?;
    let (p, q, den) = if arena.is_zero_structural(d1) {
        (n0, n1, d0)
    } else {
        let a = arena.mul(&[n0, d0]);
        let b = arena.mul(&[n1, d1]);
        let p = arena.add(&[a, b]);
        let a = arena.mul(&[n1, d0]);
        let b = arena.mul(&[n0, d1]);
        let q = arena.sub(a, b);
        let two = arena.int(2);
        let a = arena.pow(d0, two);
        let b = arena.pow(d1, two);
        (p, q, arena.add(&[a, b]))
    };
    let mut parts: Vec<ExprId> = Vec::with_capacity(2);
    for (num, factor) in [(p, arena.one), (q, i)] {
        let num = crate::transforms::expand::expand(arena, num);
        if arena.is_zero_structural(num) {
            continue;
        }
        let f = arena.div(num, den);
        let f = clear_nested_fractions(arena, f, var);
        if crate::base::walk::contains(arena, f, iota) {
            return None;
        }
        let big_f = integrate_node(arena, f, var, var_sym, depth - 1);
        if crate::base::walk::has_unevaluated(arena, big_f) {
            return None;
        }
        // Term by term, so that the terms `c·atan(tan w)` of a substitution
        // through `tan w` stay terms (`add_tan_floor` writes them `c·w`).
        let terms: Vec<ExprId> = match arena.node(big_f) {
            ExprNode::Add(children) => children.to_vec(),
            _ => vec![big_f],
        };
        for term in terms {
            parts.push(arena.mul(&[factor, term]));
        }
    }
    tracing::debug!(
        "integrate: a rational function with Gaussian coefficients, by real and imaginary parts"
    );
    let result = arena.add(&parts);
    if candidate_rejected(arena, expr, result, var, var_sym) {
        return None;
    }
    Some(result)
}

/// At most this many distinct constants in the numerator for
/// [`try_rational_with_constant_coefficients`]: each costs one rational
/// integration over the same denominator.
const MAX_CONSTANT_GROUPS: usize = 8;

/// The largest numerator degree [`try_rational_with_constant_coefficients`]
/// takes (the rational integrator's own bound on exponents): its pieces are
/// built as dense polynomials, and `x^1000000000/(x + 1)` would ask for a
/// billion coefficients.
const MAX_CONSTANT_COEFF_DEGREE: usize = 1000;

/// `∫ N(x)/D(x) dx` for `D ∈ ℚ[x]` and a polynomial numerator whose
/// coefficients are constants, not all rational: `N = Σ κₖ·Pₖ(x)` with
/// distinct constants `κₖ` and `Pₖ ∈ ℚ[x]`, and `∫ N/D = Σ κₖ·∫ Pₖ/D`, each
/// piece the rational integrator's over ℚ (Hermite, Rothstein–Trager),
/// complete there.  Linearity is exact, so the pieces' checked answers need
/// no further check.  Up to 0.46 the integrator over ℚ refused such an
/// integrand (a constant like `1/e` is not in ℚ), the one over ℚ(p₁, …)
/// as well (`e` is no symbol), and the search went down nested by parts and
/// sum distributions that found the same answer piece by piece: `∫ (t⁸ +
/// t⁷/e)/(t¹² + 1/2) dt` took 6.4 s, and through it (`x = t²`, then real and
/// imaginary parts) `∫ (√x + 1/e)/(x³ + √2·i/2) dx` 3.6 s in 0.46 and 5.2 s
/// in the 0.47 development code (the slow `fuzz_integrate` unit of its
/// campaign: 10 s in the instrumented build, against a 30 s CI limit);
/// now 0.012 s and 0.041 s.
fn try_rational_with_constant_coefficients(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    let expr = arena.mul(dependent);
    if depth < 2 || !is_rational_in(arena, expr, var_sym) {
        return None;
    }
    let (num, den) = crate::poly::polybridge::as_numer_denom(arena, expr);
    if !contains_var(arena, den, var_sym)
        || !is_polynomial_in(arena, num, var, var_sym)
        || crate::poly::polybridge::expr_to_poly(arena, den, var).is_none()
        || crate::poly::polybridge::expr_to_poly(arena, num, var).is_some()
    {
        return None;
    }
    let terms = crate::poly::polybridge::symbolic_multipoly_terms(arena, num, &[var])?;
    // The terms `q·κ·xᵈ` of the numerator, `q` rational.
    let mut monomials: Vec<(usize, Q, ExprId)> = Vec::new();
    for (exps, c) in terms {
        let degree = usize::try_from(exps[0]).ok()?;
        if degree > MAX_CONSTANT_COEFF_DEGREE {
            return None;
        }
        let parts: SmallVec<[ExprId; 6]> = match arena.node(c) {
            ExprNode::Add(children) => children.clone(),
            _ => SmallVec::from_slice(&[c]),
        };
        for part in parts {
            let (q, kappa) = arena.as_coeff_term(part);
            if contains_var(arena, kappa, var_sym) {
                return None;
            }
            monomials.push((degree, q, kappa));
        }
    }
    if monomials.iter().all(|(_, _, kappa)| *kappa == arena.one) {
        return None;
    }
    // κₖ ↦ the coefficients of Pₖ, by degree.
    let mut groups: Vec<(ExprId, Vec<Q>)> = Vec::new();
    for (degree, q, kappa) in monomials {
        let k = match groups.iter().position(|(g, _)| *g == kappa) {
            Some(k) => k,
            None => {
                if groups.len() == MAX_CONSTANT_GROUPS {
                    return None;
                }
                groups.push((kappa, Vec::new()));
                groups.len() - 1
            }
        };
        let coeffs = &mut groups[k].1;
        if coeffs.len() <= degree {
            coeffs.resize(degree + 1, Q::zero());
        }
        coeffs[degree] += q;
    }
    let mut parts: Vec<ExprId> = Vec::with_capacity(groups.len());
    for (kappa, coeffs) in groups {
        let p = Poly::from_coeffs(coeffs);
        if p.is_zero() {
            continue;
        }
        let p = crate::poly::polybridge::poly_to_expr(arena, &p, var);
        let piece = arena.div(p, den);
        let big_f = integrate_node(arena, piece, var, var_sym, depth - 1);
        if crate::base::walk::has_unevaluated(arena, big_f) {
            return None;
        }
        // Term by term, as `try_gaussian_rational` writes its parts.
        let terms: Vec<ExprId> = match arena.node(big_f) {
            ExprNode::Add(children) => children.to_vec(),
            _ => vec![big_f],
        };
        for term in terms {
            parts.push(arena.mul(&[kappa, term]));
        }
    }
    tracing::debug!("integrate: a rational function over ℚ times constants, piece by piece");
    Some(arena.add(&parts))
}

/// `∫ A(sin u, cos u) du` for `A = R(sin u, cos u)·√(1 + ε·w)^p` (`w` = `sin u`
/// or `cos u`, `ε = ±1`, every radical `(κ·(1 + ε·w))^(k/2)` of the
/// integrand of that one base: Rubi 4.1.1.1, 4.1.1.2, 4.2.1.1 `(a + a·sin(c +
/// d·x))^(7/2)`, `cos²(c + d·x)·√(a + a·sin(c + d·x))`), `R` rational with
/// poles only where `1 + ε·w = 0`.  `(κ·p)^r = κ^r·p^r` for real `p ≥ 0`, and
/// with `v = u/2 + φ` (`φ = ε·π/4` for `sin u`, `π/2` or `0` for `1 ± cos u`)
/// `1 + ε·w = 2·sin²v`, so `√(1 + ε·w) = √2·σ·sin v` with `σ = sign(sin v)`:
/// the integrand is `σ·g(v)·dv` with `g` rational in `sin v`, `cos v` and
/// `g(v + π) = −g(v)`.
///
/// Where `g` is a polynomial (the integrand is continuous), its parts odd
/// in `sin v` and in `cos v` integrate (`t = cos v`, `t = sin v`) to an
/// antiderivative `G` with `G(v + π) = −G(v)`, and `F = σ·G − 2·G(0)·⌊v/π⌋`
/// is continuous at the zeros `v = kπ` of `sin v`, where `σ` flips (`σ·G`
/// steps by `2·G(0)` at each).  `σ·G` is written without `σ`, with
/// `σ·sin v = √(1 + ε·w)/√2` and `σ·cos v = √(1 + ε·w)·sin 2v/(√2·(1 + ε·w))`:
/// `∫ √(1 + sin x) dx = −2·cos x/√(1 + sin x) + 4√2·⌊(x/2 + π/4)/π⌋`.
/// Rubi's answers are the first term alone and jump by `4√2` at every
/// zero of `1 + sin x`; SymPy 1.14 leaves these integrands unevaluated.
/// Where `g` has poles (at `sin v = 0` only) `F = σ·G`.  `None` for other
/// radicals, a part of `A` with an even power of the radical, poles of `R`
/// elsewhere, or no answer for `G`.
fn half_angle_radical_in_u(
    arena: &mut Arena,
    e: ExprId,
    sc: (ExprId, ExprId),
    u: ExprId,
) -> Option<ExprId> {
    let (s, c) = sc;
    // The radicals `B^(k/2)` and their common base `κ·(1 + ε·w)`.
    let mut base_kind: Option<(ExprId, ExprId)> = None;
    let mut radicals: Vec<(ExprId, ExprId, i64, ExprId)> = Vec::new();
    for id in crate::base::walk::post_order_ids(arena, e) {
        let ExprNode::Pow(base, ex) = *arena.node(id) else {
            continue;
        };
        let Some(r) = arena.as_num(ex).cloned() else {
            continue;
        };
        if r.is_integer() || !(contains_var_id(arena, base, s) || contains_var_id(arena, base, c)) {
            continue;
        }
        if *r.denom() != num_bigint::BigInt::from(2) {
            return None;
        }
        let k = i64::try_from(r.numer()).ok()?;
        let (w, other) = if contains_var_id(arena, base, c) {
            (c, s)
        } else {
            (s, c)
        };
        if contains_var_id(arena, base, other) {
            return None;
        }
        let coeffs = crate::transforms::solve::symbolic_poly_coeffs(arena, base, w)?;
        let [kappa, b1] = coeffs.as_slice() else {
            return None;
        };
        let eps = arena.div(*b1, *kappa);
        let eps = crate::transforms::eval::eval(arena, eps);
        let eps_one = arena.as_num(eps).filter(|q| q.abs().is_one())?.clone();
        let eps = arena.num_ratio(eps_one);
        match base_kind {
            None => base_kind = Some((w, eps)),
            Some(kind) if kind == (w, eps) => {}
            Some(_) => return None,
        }
        radicals.push((id, *kappa, k, w));
    }
    let (w, eps) = base_kind?;
    let eps_positive = arena.as_num(eps).is_some_and(|q| q.is_positive());
    // `√(1 + ε·w)` as the symbol `rad`: `B^(k/2) = κ^(k/2)·rad^k`.
    let rad = arena.symbol("__hrad");
    let mut sub = e;
    for &(id, kappa, k, _) in &radicals {
        let half_k = arena.rational(k, 2);
        let kappa_pow = arena.pow(kappa, half_k);
        let k_id = arena.int(k);
        let rad_pow = arena.pow(rad, k_id);
        let repl = arena.mul(&[kappa_pow, rad_pow]);
        sub = arena.subs_structural(sub, id, repl);
    }
    let ExprNode::Symbol(rad_sym) = *arena.node(rad) else {
        return None;
    };
    if !is_rational_in_all(arena, sub, &[s, c, rad]) || !is_rational_in(arena, sub, rad_sym) {
        return None;
    }
    // `sub = rad·E₁` with `E₁` free of `rad` after `rad² = 1 + ε·w`.
    let (num, den) = crate::poly::polybridge::as_numer_denom(arena, sub);
    let one = arena.one;
    let eps_w = arena.mul(&[eps, w]);
    let rad_sq = arena.add(&[one, eps_w]);
    let reduce_rad = |arena: &mut Arena, p: ExprId| -> Option<(ExprId, ExprId)> {
        let cs = crate::transforms::solve::symbolic_poly_coeffs(arena, p, rad)?;
        let mut even: Vec<ExprId> = Vec::new();
        let mut odd: Vec<ExprId> = Vec::new();
        for (j, &k) in cs.iter().enumerate() {
            let e = arena.int(i64::try_from(j / 2).ok()?);
            let power = arena.pow(rad_sq, e);
            let term = arena.mul(&[k, power]);
            if j % 2 == 0 {
                even.push(term);
            } else {
                odd.push(term);
            }
        }
        let even = arena.add(&even);
        let odd = arena.add(&odd);
        Some((
            crate::transforms::expand::expand(arena, even),
            crate::transforms::expand::expand(arena, odd),
        ))
    };
    let (n_even, n_odd) = reduce_rad(arena, num)?;
    let (d_even, d_odd) = reduce_rad(arena, den)?;
    // `(n₀ + rad·n₁)/(d₀ + rad·d₁)`: only `rad·n₁/d₀` and `n₀/(rad·d₁)` are
    // odd in `rad`.
    let e1 = if arena.is_zero_structural(n_even) && arena.is_zero_structural(d_odd) {
        arena.div(n_odd, d_even)
    } else if arena.is_zero_structural(n_odd) && arena.is_zero_structural(d_even) {
        // `n₀/(rad·d₁) = rad·n₀/((1 + ε·w)·d₁)`.
        let d = arena.mul(&[rad_sq, d_odd]);
        arena.div(n_even, d)
    } else {
        return None;
    };
    // `v = u/2 + φ`; `sin u`, `cos u` in `S = sin v`, `C = cos v`.
    let big_s = arena.symbol("__hs");
    let big_c = arena.symbol("__hc");
    let two = arena.int(2);
    let two_sc = arena.mul(&[two, big_s, big_c]);
    let s_sq = arena.pow(big_s, two);
    let c_sq = arena.pow(big_c, two);
    let c2_minus_s2 = arena.sub(c_sq, s_sq);
    let neg = |arena: &mut Arena, z: ExprId| arena.neg(z);
    let w_is_sin = w == s;
    let (sin_u, cos_u, phi) = match (w_is_sin, eps_positive) {
        (true, true) => (neg(arena, c2_minus_s2), two_sc, arena.rational(1, 4)),
        (true, false) => (c2_minus_s2, neg(arena, two_sc), arena.rational(-1, 4)),
        (false, true) => (
            neg(arena, two_sc),
            neg(arena, c2_minus_s2),
            arena.rational(1, 2),
        ),
        (false, false) => (two_sc, c2_minus_s2, arena.zero),
    };
    let pi = arena.pi();
    let phi = arena.mul(&[phi, pi]);
    // `g(v) = 2·√2·S·E₁(sin u, cos u)` (`du = 2·dv`, `rad = √2·σ·S`).
    let g = arena.subs_structural(e1, s, sin_u);
    let g = arena.subs_structural(g, c, cos_u);
    let sqrt2 = arena.sqrt(two);
    let g = arena.mul(&[two, sqrt2, big_s, g]);
    let g = clear_nested_fractions(arena, g, big_s);
    let (g_num, g_den) = crate::poly::polybridge::as_numer_denom(arena, g);
    // Reduce `C² = 1 − S²` in both: `A(S) + C·B(S)`.
    let reduce_c = |arena: &mut Arena, p: ExprId| -> Option<(ExprId, ExprId)> {
        let cs = crate::transforms::solve::symbolic_poly_coeffs(arena, p, big_c)?;
        let one = arena.one;
        let s_sq = arena.pow(big_s, two);
        let minus = arena.neg(s_sq);
        let one_minus = arena.add(&[one, minus]);
        let mut even: Vec<ExprId> = Vec::new();
        let mut odd: Vec<ExprId> = Vec::new();
        for (j, &k) in cs.iter().enumerate() {
            let e = arena.int(i64::try_from(j / 2).ok()?);
            let power = arena.pow(one_minus, e);
            let term = arena.mul(&[k, power]);
            if j % 2 == 0 {
                even.push(term);
            } else {
                odd.push(term);
            }
        }
        let even = arena.add(&even);
        let odd = arena.add(&odd);
        Some((
            crate::transforms::expand::expand(arena, even),
            crate::transforms::expand::expand(arena, odd),
        ))
    };
    let (a_part, b_part) = reduce_c(arena, g_num)?;
    let (den_s, den_c) = reduce_c(arena, g_den)?;
    if !arena.is_zero_structural(den_c) {
        return None;
    }
    // The denominator: a constant times `Sᵐ`.
    let den_coeffs = crate::transforms::solve::symbolic_poly_coeffs(arena, den_s, big_s)?;
    let nonzero: Vec<usize> = (0..den_coeffs.len())
        .filter(|&j| !arena.is_zero_structural(den_coeffs[j]))
        .collect();
    let [m] = nonzero.as_slice() else {
        return None;
    };
    let mut m = *m;
    // `A + C·B` vanishes to order `k` at `S = 0` (`C = ±1` there) exactly when
    // `Sᵏ` divides `A` and `B`: those powers cancel against `Sᵐ` (`cos⁶u/√(1 +
    // sin u)` has no pole where `σ` flips).
    let lowest = |arena: &mut Arena, p: ExprId| -> Option<usize> {
        if arena.is_zero_structural(p) {
            return Some(usize::MAX);
        }
        let cs = crate::transforms::solve::symbolic_poly_coeffs(arena, p, big_s)?;
        cs.iter().position(|&k| !arena.is_zero_structural(k))
    };
    let k = lowest(arena, a_part)?.min(lowest(arena, b_part)?).min(m);
    let (a_part, b_part, den_s) = if k > 0 {
        let shift = |arena: &mut Arena, p: ExprId| -> Option<ExprId> {
            if arena.is_zero_structural(p) {
                return Some(p);
            }
            let cs = crate::transforms::solve::symbolic_poly_coeffs(arena, p, big_s)?;
            let mut terms: Vec<ExprId> = Vec::new();
            for (j, &c) in cs.iter().enumerate().skip(k) {
                let e = arena.int(i64::try_from(j - k).ok()?);
                let power = arena.pow(big_s, e);
                terms.push(arena.mul(&[c, power]));
            }
            Some(arena.add(&terms))
        };
        let a = shift(arena, a_part)?;
        let b = shift(arena, b_part)?;
        let d = shift(arena, den_s)?;
        m -= k;
        (a, b, d)
    } else {
        (a_part, b_part, den_s)
    };
    let v = arena.symbol("__hv");
    let ExprNode::Symbol(v_sym) = *arena.node(v) else {
        return None;
    };
    let b_times_c = arena.mul(&[big_c, b_part]);
    let half = arena.rational(1, 2);
    let half_u = arena.mul(&[half, u]);
    let v_of_u = arena.add(&[half_u, phi]);
    if m > 0 {
        // Poles where `σ` flips: `F = σ·G`, with `σ = √2·sin v/√(1 + ε·w)`
        // (analytic in the parameters of `u`, as `sign(sin v)` is not).
        let n = arena.add(&[a_part, b_times_c]);
        let big_g = trig_rational_in_u(arena, (n, den_s), (big_s, big_c), v, v_sym)?;
        let back = arena.subs_structural(big_g, v, v_of_u);
        let sin_v = arena.sin(v_of_u);
        let sqrt2 = arena.sqrt(two);
        let w_u = if w_is_sin { arena.sin(u) } else { arena.cos(u) };
        let eps_w_u = arena.mul(&[eps, w_u]);
        let one_eps_w = arena.add(&[one, eps_w_u]);
        let rad_u = arena.sqrt(one_eps_w);
        let minus_one = arena.int(-1);
        let inv_rad = arena.pow(rad_u, minus_one);
        tracing::debug!("integrate: a radical of 1 ± sin u (cos u) through the half angle");
        let big_f = arena.mul(&[sqrt2, sin_v, inv_rad, back]);
        return Some(crate::transforms::eval::eval(arena, big_f));
    }
    // `g(v + π) = −g(v)`: `A` odd and `B` even in `S`.  Each part goes
    // through Bioche's parity routes (`t = cos v`, `t = sin v`) to a
    // polynomial odd in `cos v` (`sin v`), with no constant: `G(v + π) =
    // −G(v)`.
    let odd_or_zero = |arena: &mut Arena, p: ExprId, parity: usize| {
        arena.is_zero_structural(p) || parity_in(arena, p, big_s) == Some(parity)
    };
    if !odd_or_zero(arena, a_part, 1) || !odd_or_zero(arena, b_part, 0) {
        return None;
    }
    // `σ·G` without `σ`: `σ·sin v = rad/√2` and `σ·cos v = rad·sin 2v/(√2·(1 +
    // ε·w))` (`sin²v = (1 + ε·w)/2`), with `rad = √(1 + ε·w)` of `u`: a form
    // analytic in the parameters of `u`, as Rubi writes these answers.
    let sin_u_node = arena.sin(u);
    let cos_u_node = arena.cos(u);
    let w_u = if w_is_sin { sin_u_node } else { cos_u_node };
    let eps_w_u = arena.mul(&[eps, w_u]);
    let one_eps_w = arena.add(&[one, eps_w_u]);
    let s2_u = arena.mul(&[half, one_eps_w]);
    let minus_s2 = arena.neg(s2_u);
    let c2_u = arena.add(&[one, minus_s2]);
    let sin_2v = match (w_is_sin, eps_positive) {
        (true, true) => cos_u_node,
        (true, false) => arena.neg(cos_u_node),
        (false, true) => arena.neg(sin_u_node),
        (false, false) => sin_u_node,
    };
    let mut over_s: Vec<ExprId> = Vec::new();
    let mut over_c: Vec<ExprId> = Vec::new();
    let mut at_zero: Vec<ExprId> = Vec::new();
    let sin_v = arena.sin(v);
    let cos_v = arena.cos(v);
    for (part, fv, sym, sq, out) in [
        (b_times_c, sin_v, big_s, s2_u, &mut over_s),
        (a_part, cos_v, big_c, c2_u, &mut over_c),
    ] {
        if arena.is_zero_structural(part) {
            continue;
        }
        let big_g = trig_rational_in_u(arena, (part, den_s), (big_s, big_c), v, v_sym)?;
        let p = arena.subs_structural(big_g, fv, sym);
        let p = crate::transforms::eval::eval(arena, p);
        if contains_var_id(arena, p, v) || parity_in(arena, p, sym) != Some(1) {
            return None;
        }
        let coeffs = crate::transforms::solve::symbolic_poly_coeffs(arena, p, sym)?;
        for (j, &k) in coeffs.iter().enumerate().skip(1).step_by(2) {
            let e = arena.int(i64::try_from(j / 2).ok()?);
            let power = arena.pow(sq, e);
            out.push(arena.mul(&[k, power]));
            if sym == big_c {
                at_zero.push(k);
            }
        }
    }
    let rad_u = arena.sqrt(one_eps_w);
    let inv_sqrt2 = {
        let r = arena.sqrt(two);
        let minus_one = arena.int(-1);
        arena.pow(r, minus_one)
    };
    let p_s = arena.add(&over_s);
    let p_c = arena.add(&over_c);
    let ratio = arena.div(sin_2v, one_eps_w);
    let c_term = arena.mul(&[ratio, p_c]);
    let bracket = arena.add(&[p_s, c_term]);
    let main = arena.mul(&[rad_u, inv_sqrt2, bracket]);
    // `σ·G` steps by `2·G(0)` at every `v = kπ`; `G(0)` is the sum of the
    // coefficients of the odd polynomial in `cos v` (`sin 0 = 0`).
    let g0 = arena.add(&at_zero);
    let g0 = crate::transforms::eval::eval(arena, g0);
    let minus_two = arena.int(-2);
    let minus_one = arena.int(-1);
    let inv_pi = arena.pow(pi, minus_one);
    let v_over_pi = arena.mul(&[v_of_u, inv_pi]);
    let fl = arena.floor(v_over_pi);
    let correction = arena.mul(&[minus_two, g0, fl]);
    tracing::debug!("integrate: a radical of 1 ± sin u (cos u) through the half angle");
    let big_f = arena.add(&[main, correction]);
    Some(crate::transforms::eval::eval(arena, big_f))
}

/// `∫ A(sin u, cos u) du` for an algebraic `A` odd in `cos u` with every
/// radical a function of `sin u` alone (`t = sin u`), or odd in `sin u`
/// with radicals of `cos u` (`t = cos u`): Bioche's first two rules of
/// [`try_trig_rational_substitution`] for radicals (`cos⁵u·√(a + b·sin u)`,
/// Rubi 4.1.1.2, is `∫ (1 − t²)²·√(a + b·t) dt`).  The integrand in `t`
/// goes through the whole pipeline; `G(sin u)`, `G(cos u)` need no
/// correction (the substitutions have no poles).
fn odd_algebraic_in_u(
    arena: &mut Arena,
    e: ExprId,
    sc: (ExprId, ExprId),
    u: ExprId,
) -> Option<ExprId> {
    let (s, c) = sc;
    let t = arena.symbol("__toa");
    // The bases of the radicals: every one must be free of `w`.
    let radical_bases: Vec<ExprId> = crate::base::walk::post_order_ids(arena, e)
        .into_iter()
        .filter_map(|id| match *arena.node(id) {
            ExprNode::Pow(base, ex) if !arena.as_num(ex).is_some_and(|r| r.is_integer()) => {
                Some(base)
            }
            _ => None,
        })
        .collect();
    let sin_u = arena.sin(u);
    let cos_u = arena.cos(u);
    for (w, o, back, sign) in [(c, s, sin_u, 1), (s, c, cos_u, -1)] {
        let ExprNode::Symbol(w_sym) = *arena.node(w) else {
            continue;
        };
        if radical_bases.iter().any(|&b| contains_var_id(arena, b, w))
            || !is_rational_in(arena, e, w_sym)
        {
            continue;
        }
        let (num, den) = crate::poly::polybridge::as_numer_denom(arena, e);
        let (Some(pn), Some(pd)) = (parity_in(arena, num, w), parity_in(arena, den, w)) else {
            continue;
        };
        if pn == pd {
            continue;
        }
        let Some(in_t) = odd_part_in_t(arena, (num, den), (pn, pd), w, o, t) else {
            continue;
        };
        let sign = arena.int(sign);
        let in_t = arena.mul(&[sign, in_t]);
        let in_t = crate::transforms::eval::eval(arena, in_t);
        if contains_var_id(arena, in_t, s) || contains_var_id(arena, in_t, c) {
            continue;
        }
        tracing::debug!(integrand_t = %arena.display(in_t), "integrate: an algebraic function through t = sin u or cos u");
        let Some(g) = integrate_nested_generic(arena, in_t, t) else {
            continue;
        };
        return Some(arena.subs_structural(g, t, back));
    }
    None
}

/// `∫ A(sin u, cos u) du` for an algebraic function `A` of `tan u` alone
/// (`s = sin u`, `c = cos u` in `sc`): radicals of `k·tan u`, `a + b·tan u`,
/// `a + b·tan²u` (Rubi 4.3, 4.4: `√(d·tan u)`, `(a + i·a·tan u)^(7/2)`) times a
/// rational function of `sin u`, `cos u` invariant under `u → u + π`.
/// With `s = t·c`, `A` is a function of `t` and of `c` in even (or odd)
/// integer powers outside the radicals, `c² = 1/(1 + t²)`, `du = dt/(1 + t²)`;
/// the integrand in `t` goes through the whole pipeline (its radical
/// substitutions).  The answer `G(tan u)` gets the floor correction of
/// [`crate::transforms::trig_integ::continuous_through_tan`] where its
/// jump is decided; otherwise it is kept only when the integrand is not
/// continuous across the poles of `tan u` ([`tan_pole_continuous`]: `√(tan
/// u)` is real on one side only), so that no answer jumps where the
/// integrand is continuous.
fn tan_algebraic_in_u(
    arena: &mut Arena,
    e: ExprId,
    sc: (ExprId, ExprId),
    u: ExprId,
) -> Option<ExprId> {
    let (s, c) = sc;
    let ExprNode::Symbol(c_sym) = *arena.node(c) else {
        return None;
    };
    let t = arena.symbol("__tat");
    let ExprNode::Symbol(t_sym) = *arena.node(t) else {
        return None;
    };
    let tc = arena.mul(&[t, c]);
    let e = arena.subs_structural(e, s, tc);
    let e = crate::transforms::eval::eval(arena, e);
    if contains_var_id(arena, e, s) || !is_rational_in(arena, e, c_sym) {
        return None;
    }
    let (num, den) = crate::poly::polybridge::as_numer_denom(arena, e);
    let num_c = crate::transforms::solve::symbolic_poly_coeffs(arena, num, c)?;
    let den_c = crate::transforms::solve::symbolic_poly_coeffs(arena, den, c)?;
    let parity = |arena: &mut Arena, cs: &[ExprId]| -> Option<usize> {
        let mut nonzero = [false, false];
        for (j, &k) in cs.iter().enumerate() {
            if !arena.is_zero_structural(k) {
                nonzero[j % 2] = true;
            }
        }
        match nonzero {
            [true, false] => Some(0),
            [false, true] => Some(1),
            _ => None,
        }
    };
    let shift = parity(arena, &num_c)?;
    if parity(arena, &den_c)? != shift {
        return None;
    }
    tracing::debug!("integrate: an algebraic function of tan u, through t = tan u");
    let one = arena.one;
    let two = arena.int(2);
    let t_sq = arena.pow(t, two);
    let one_plus_t_sq = arena.add(&[one, t_sq]);
    let in_t = |arena: &mut Arena, cs: &[ExprId]| -> Option<ExprId> {
        let mut terms: Vec<ExprId> = Vec::new();
        for (j, &k) in cs.iter().enumerate().skip(shift).step_by(2) {
            let e = arena.int(-i64::try_from((j - shift) / 2).ok()?);
            let c_pow = arena.pow(one_plus_t_sq, e);
            terms.push(arena.mul(&[k, c_pow]));
        }
        Some(arena.add(&terms))
    };
    let num_t = in_t(arena, &num_c)?;
    let den_t = in_t(arena, &den_c)?;
    let dt = arena.mul(&[den_t, one_plus_t_sq]);
    let integrand_t = arena.div(num_t, dt);
    let integrand_t = crate::transforms::eval::eval(arena, integrand_t);
    if contains_var_id(arena, integrand_t, c) {
        return None;
    }
    tracing::debug!(integrand_t = %arena.display(integrand_t), "tan u: algebraic integrand in t");
    let g = integrate_nested_generic(arena, integrand_t, t)?;
    if let Some(r) = crate::transforms::trig_integ::continuous_through_tan(arena, g, t, t_sym, u) {
        return Some(r);
    }
    if !tan_pole_continuous(arena, integrand_t, t).is_false() {
        tracing::debug!("tan u: the jump at the poles of tan u is not decided");
        return None;
    }
    let tan_u = arena.tan(u);
    let back = arena.subs_structural(g, t, tan_u);
    Some(crate::transforms::trig_integ::atan_tan_terms_to_argument(
        arena, back, u,
    ))
}

/// Is the integrand `f(u) = h(t)·(1 + t²)` (`t = tan u`) continuous across
/// the poles of `tan u`, for its parameters at the generic values of
/// [`FTC_PARAMETER_VALUES`]?  By the limits of `(1 + t²)·h(t)` at `t = ±∞`:
/// `False` when one is infinite or they differ (`√(tan u)` is real on one
/// side only), `True` when they agree (to `10⁻⁹`: only `False` is acted on,
/// by refusing), `Unknown` when a limit is not found.
fn tan_pole_continuous(arena: &mut Arena, h: ExprId, t: ExprId) -> Truth {
    let one = arena.one;
    let two = arena.int(2);
    let t_sq = arena.pow(t, two);
    let one_plus_t_sq = arena.add(&[one, t_sq]);
    let mut k = arena.mul(&[h, one_plus_t_sq]);
    let mut params: Vec<(String, ExprId)> = Vec::new();
    for s in crate::base::walk::free_symbols(arena, k) {
        if s == t {
            continue;
        }
        let ExprNode::Symbol(sid) = *arena.node(s) else {
            return Truth::Unknown;
        };
        params.push((arena.symbol_name(sid).to_owned(), s));
    }
    if params.len() > FTC_PARAMETER_VALUES.len() {
        return Truth::Unknown;
    }
    params.sort_by(|a, b| a.0.cmp(&b.0));
    for (j, &(_, s)) in params.iter().enumerate() {
        let (p, d) = FTC_PARAMETER_VALUES[j];
        let value = arena.rational(p, d);
        k = crate::transforms::subs::subs(arena, k, s, value);
    }
    let k = crate::transforms::eval::eval(arena, k);
    let mut values = Vec::with_capacity(2);
    for point in [arena.infinity(), arena.neg_infinity()] {
        let Ok(v) = arena.limit_expr(k, t, point) else {
            return Truth::Unknown;
        };
        let v = crate::transforms::eval::eval(arena, v);
        if crate::base::walk::has_unevaluated(arena, v) || contains_var_id(arena, v, t) {
            return Truth::Unknown;
        }
        if contains_non_finite(arena, v) {
            return Truth::False;
        }
        let Ok(z) = crate::transforms::evalf::evalf_complex64(arena, v) else {
            return Truth::Unknown;
        };
        if !z.re.is_finite() || !z.im.is_finite() {
            return Truth::False;
        }
        values.push(z);
    }
    let gap = (values[0] - values[1]).norm();
    Truth::from_bool(gap <= 1e-9 * values[0].norm().max(1.0))
}

/// Does `e` contain the symbol `v` (free)?
fn contains_var_id(arena: &Arena, e: ExprId, v: ExprId) -> bool {
    match *arena.node(v) {
        ExprNode::Symbol(sid) => contains_var(arena, e, sid),
        _ => true,
    }
}

/// [`jump_kept_undecided`] for the answer `g` of the integrand `h` in `t =
/// tan w`, and inside the trigonometric substitutions of 0.38
/// ([`crate::transforms::trig_integ::strict_jumps_active`]) only
/// where the integrand is not continuous across the poles of `tan w`
/// ([`tan_pole_continuous`] at generic values of the parameters): those
/// routes return no answer that jumps where the integrand is continuous
/// (`∫ cos²u/(a − b·sin⁴u) du`, a `RootSum` over a polynomial with parameters,
/// is refused).
fn undecided_jump_kept(
    arena: &mut Arena,
    h: ExprId,
    g: ExprId,
    t: ExprId,
    t_sym: SymbolId,
) -> bool {
    jump_kept_undecided(arena, g, t_sym)
        && (!crate::transforms::trig_integ::strict_jumps_active()
            || tan_pole_continuous(arena, h, t).is_false())
}

/// May an antiderivative `G(tan w)` (`g` = `G(t)`, `t` the substitution's
/// symbol) whose jump `G(+∞) − G(−∞)` at the poles of `tan w` is not
/// decided be returned uncorrected, continuous only between those poles?
/// Yes when `G` has parameters — the jump depends on their values and, for
/// parameters not declared real, on the sides of the cuts their logarithms
/// approach (`∫ dx/(a + b·cos x)`; SymPy and Rubi answer such integrands so
/// too, and refusing them lost 435 answers of the Rubi suite) — or a
/// `RootSum` whose end values are not found (a logarithm of degree ≥ 2 in
/// `t`).  Any other undecided jump makes the substitution refuse.
fn jump_kept_undecided(arena: &Arena, g: ExprId, t_sym: SymbolId) -> bool {
    crate::base::walk::free_symbols(arena, g)
        .iter()
        .any(|&s| !matches!(arena.node(s), ExprNode::Symbol(sid) if *sid == t_sym))
        || crate::base::walk::post_order_ids(arena, g)
            .iter()
            .any(|&id| matches!(arena.node(id), ExprNode::RootSum(..)))
}

/// Normalise a rational expression in `var` with nested fractions into a
/// single cancelled `numer/denom`.
///
/// `together` only combines the top-level sum, so sums buried inside
/// powers (e.g. `2/((1+t²)(2 + (1−t²)/(1+t²)))`) are combined bottom-up
/// first, then the whole expression is split into numerator/denominator
/// and cancelled.
///
/// Inside an `integrate` call the result is remembered until the outermost
/// call ends ([`CLEARED_MEMO`]).
pub(crate) fn clear_nested_fractions(arena: &mut Arena, expr: ExprId, var: ExprId) -> ExprId {
    let inside = INTEGRATE_DEPTH.with(std::cell::Cell::get) > 0;
    let key = (arena.serial(), expr, var);
    if inside && let Some(found) = CLEARED_MEMO.with(|m| m.borrow().get(&key).copied()) {
        return found;
    }
    let cleared = clear_nested_fractions_uncached(arena, expr, var);
    if inside {
        CLEARED_MEMO.with(|m| m.borrow_mut().insert(key, cleared));
    }
    cleared
}

thread_local! {
    /// [`clear_nested_fractions`] during the current top-level `integrate`
    /// (cleared with [`INTEGRATE_MEMO`]), by (arena, expression, variable).
    /// The trigonometric substitutions normalise the same pieces again for
    /// every degenerate case of the parameters and every group of a
    /// numerator: `∫ (A + B·sec u + C·sec²u)/(a + b·sec u)⁴ du` made 116
    /// such calls (1.4 s, multiplying out numerators such as `(t² + 1)⁷·…`)
    /// on inputs whose distinct ones took 0.16 s (0.38).
    static CLEARED_MEMO: std::cell::RefCell<FxHashMap<(u64, ExprId, ExprId), ExprId>> =
        std::cell::RefCell::new(FxHashMap::default());
}

/// [`clear_nested_fractions`] without the memo.
fn clear_nested_fractions_uncached(arena: &mut Arena, expr: ExprId, var: ExprId) -> ExprId {
    let mut current = expr;
    // Bottom-up: combine every inner sum over a common denominator.
    for _ in 0..16 {
        let order = crate::base::walk::post_order_ids(arena, current);
        let mut changed = false;
        for id in order {
            if id == current {
                continue;
            }
            if let ExprNode::Add(_) = arena.node(id) {
                let t = crate::poly::polybridge::together(arena, id);
                if t != id {
                    current = arena.subs_structural(current, id, t);
                    changed = true;
                    break;
                }
            }
        }
        if !changed {
            break;
        }
    }
    let together = crate::poly::polybridge::together(arena, current);
    let (n, d) = crate::poly::polybridge::as_numer_denom(arena, together);
    let n = crate::transforms::expand::expand(arena, n);
    let d = crate::transforms::expand::expand(arena, d);
    let n = crate::transforms::eval::eval(arena, n);
    let d = crate::transforms::eval::eval(arena, d);
    let ratio = arena.div(n, d);
    let cancelled = arena.cancel_expr(ratio, var);
    crate::transforms::eval::eval(arena, cancelled)
}

/// Attempt cyclic integration by parts for integrals like `∫ exp(x)·sin(x) dx`.
///
/// After two IBP rounds (with u₂ = du₁, dv₂ = v₁), if the remaining
/// integral is a constant multiple `c` of the original integrand we
/// solve algebraically:
///
/// ```text
///   I = boundary₁ − (boundary₂ − c·I)
///   I(1 − c) = boundary₁ − boundary₂
///   I = (boundary₁ − boundary₂) / (1 − c)
/// ```
fn try_cyclic_ibp(
    arena: &mut Arena,
    factors: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    if factors.len() != 2 || depth < 2 {
        return None;
    }

    // Use LIATE ordering: lower rank = u (trig before exp)
    let (u_idx, dv_idx) = {
        let r0 = liate_rank(arena, factors[0], var, var_sym);
        let r1 = liate_rank(arena, factors[1], var, var_sym);
        if r0 <= r1 { (0, 1) } else { (1, 0) }
    };
    let u1 = factors[u_idx];
    let dv1 = factors[dv_idx];

    // Both must depend on var
    if !contains_var(arena, u1, var_sym) || !contains_var(arena, dv1, var_sym) {
        return None;
    }

    // ── Round 1: ∫ u1·dv1 dx = u1·v1 − ∫ v1·du1 dx ──────────────
    let v1 = integrate_node(arena, dv1, var, var_sym, depth - 1);
    if crate::base::walk::has_unevaluated(arena, v1) {
        tracing::trace!("try_cyclic_ibp: v1 has unevaluated nodes, bailing");
        return None;
    }
    let du1 = crate::transforms::diff::diff(arena, u1, var);
    let boundary1 = arena.mul(&[u1, v1]); // u1·v1

    // ── Round 2: ∫ v1·du1 dx  with u₂ = du1, dv₂ = v1 ───────────
    let v2 = integrate_node(arena, v1, var, var_sym, depth - 1);
    if crate::base::walk::has_unevaluated(arena, v2) {
        tracing::trace!("try_cyclic_ibp: v2 has unevaluated nodes, bailing");
        return None;
    }
    let du2 = crate::transforms::diff::diff(arena, du1, var); // u1''
    let boundary2 = arena.mul(&[du1, v2]); // du1·v2

    // remaining₂ body = v2 · du2
    let remaining2 = arena.mul(&[v2, du2]);
    let original = arena.mul(&[factors[0], factors[1]]);

    // ── Check remaining₂ = c · original for some constant c ──────

    // Fast path: c = −1 (covers exp·sin, exp·cos and similar)
    let sum = arena.add(&[remaining2, original]);
    if sum == arena.zero {
        tracing::debug!("cyclic IBP detected (c = -1)");
        let numerator = arena.sub(boundary1, boundary2);
        let two = arena.int(2);
        return Some(arena.div(numerator, two));
    }

    // Fast path: c = +1 would be degenerate (1−c = 0), skip.
    let diff_check = arena.sub(remaining2, original);
    if diff_check == arena.zero {
        return None;
    }

    // General path: try polynomial cancellation on the ratio.
    let ratio = arena.div(remaining2, original);
    let cancelled = arena.cancel_expr(ratio, var);
    if !contains_var(arena, cancelled, var_sym) && cancelled != arena.one {
        tracing::debug!("cyclic IBP detected (general c)");
        let numerator = arena.sub(boundary1, boundary2);
        let one = arena.one;
        let one_minus_c = arena.sub(one, cancelled);
        return Some(arena.div(numerator, one_minus_c));
    }

    None
}

/// What one top-level `integrate` has learnt about a sub-integrand in a
/// given search context (see [`MemoKey`]).
#[derive(Clone, Copy)]
enum MemoOutcome {
    /// No closed form with this much remaining depth (or less).
    Failed { depth: usize },
    /// A fully evaluated antiderivative.
    Solved(ExprId),
}

/// A sub-integrand in its search context.  The strategies that apply
/// depend on whether `try_risch_rational` may run (its recursion guard) and
/// on the nesting of substitution strategies, so both are part of the key;
/// the arena's serial ([`Arena::serial`]) keeps two arenas on one thread
/// apart (up to 0.48 its address, which a later arena can have again).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct MemoKey {
    arena: u64,
    expr: ExprId,
    var: ExprId,
    in_risch: bool,
    subst_depth: u8,
}

thread_local! {
    /// Outcomes of `integrate_node` during the current top-level
    /// `integrate` (cleared when it starts).  A failing search reaches the
    /// same sub-integrand again and again — by parts, substitution and
    /// partial fractions lead to the same pieces: `∫ x^(3/2)/(sin(−1) −
    /// 2x) dx` made 918 u-substitution attempts on 53 distinct integrands
    /// (0.24).
    static INTEGRATE_MEMO: std::cell::RefCell<FxHashMap<MemoKey, MemoOutcome>> =
        std::cell::RefCell::new(FxHashMap::default());
}

/// Integrate a single node with respect to `var`, remembering the outcome
/// for the rest of the top-level call: a sub-integrand that already failed
/// with at least this much depth fails at once (less depth only prunes
/// more), and a solved one returns its antiderivative.
fn integrate_node(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> ExprId {
    let key = MemoKey {
        arena: arena.serial(),
        expr,
        var,
        in_risch: crate::calculus::risch::rational_guard_active(),
        subst_depth: SUBST_DEPTH.with(std::cell::Cell::get),
    };
    match INTEGRATE_MEMO.with(|m| m.borrow().get(&key).copied()) {
        Some(MemoOutcome::Solved(result)) => return result,
        Some(MemoOutcome::Failed { depth: failed }) if depth <= failed => {
            return arena.intern(ExprNode::Integral(expr, var));
        }
        _ => {}
    }
    // `∫ f(ln x)`-type integrands (`sin(ln x)`, `x²·cos(a + b·ln(c·xⁿ))`)
    // wherever they occur — also as a term of a sum, which the stage-level
    // substitutions never see on its own — and before by parts, which
    // would otherwise take `u = x²` and nest itself ten deep on the
    // substitution's `dv` (`∫ x·cos(ln(x)/3) dx` came out correct but with
    // coefficients such as `1463942969757/3700000000000`).
    let result = if depth > 0
        && !matches!(arena.node(expr), ExprNode::Add(_))
        && let Some(r) = try_log_substitution(arena, expr, var, var_sym, false)
    {
        r
    } else {
        integrate_node_uncached(arena, expr, var, var_sym, depth)
    };
    // Every intermediate result in the form the stage exit gives it: a
    // route that continues from one (by parts differentiates `u`, which
    // may be an earlier `∫`) must not meet `ln|a²x² + 1|` for an
    // unassumed `a`.  That leaves the stage as `ln(a²x² + 1)`, whose
    // derivative is `2a²x/(a²x² + 1)`, while `diff` gives `ln|·|` its
    // complex-safe derivative since 0.31 (`∫ x²(c + a²c·x²)·atan(a·x)`
    // then stayed unevaluated).
    let result = analytic_log_of_complex_arguments(arena, result, var_sym);
    let outcome = if matches!(arena.node(result), ExprNode::Integral(e, v) if *e == expr && *v == var)
    {
        (!StepCap::active()).then_some(MemoOutcome::Failed { depth })
    } else if !crate::base::walk::has_unevaluated(arena, result) {
        Some(MemoOutcome::Solved(result))
    } else {
        None
    };
    if let Some(outcome) = outcome {
        INTEGRATE_MEMO.with(|m| {
            let mut m = m.borrow_mut();
            let keep_old = matches!(
                (m.get(&key), outcome),
                (Some(MemoOutcome::Failed { depth: old }), MemoOutcome::Failed { depth: new }) if *old >= new
            );
            if !keep_old {
                m.insert(key, outcome);
            }
        });
    }
    result
}

/// [`integrate_node`] without the memo.
fn integrate_node_uncached(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> ExprId {
    tracing::trace!(depth = depth, "integrate_node entered");

    if depth == 0 || node_budget_exhausted() || arena_budget_exhausted(arena) {
        return arena.intern(ExprNode::Integral(expr, var));
    }

    // ── Linearity first: ∫ c·f = c·∫ f for factors c free of the variable ──
    // Every rule below then sees the integrand without its constant
    // factors.  Before 0.30 only the `Mul` arm split them off, after the
    // whole-expression rules had run on `c·f`: the trig-power rule handed
    // back `2·∫ sin/cos` (its unevaluated result with the constant
    // re-attached, which the check for an `Integral` node did not see), so
    // `∫ 2·sin x/cos x`, `∫ −sin x/cos x` and `∫ 2·cos x/sin x` stayed
    // unevaluated while `∫ sin x/cos x` worked (and variation of parameters
    // failed on `y″ + y = tan x`).
    if let Some((constants, f)) = split_constant_factors(arena, expr, var_sym) {
        // `f` has fewer factors than `expr`, so the same depth cannot loop,
        // unless rebuilding the product exposed a constant factor again.
        let d = if split_constant_factors(arena, f, var_sym).is_some() {
            depth - 1
        } else {
            depth
        };
        let big_f = integrate_node(arena, f, var, var_sym, d);
        if matches!(arena.node(big_f), ExprNode::Integral(..)) {
            return arena.intern(ExprNode::Integral(expr, var));
        }
        return constants_times_antiderivative(arena, expr, f, big_f, &constants, var, var_sym)
            .unwrap_or_else(|| arena.intern(ExprNode::Integral(expr, var)));
    }

    // Try trig power/product integration first (sin^n, cos^n, sin^m*cos^n)
    if let Some(result) =
        crate::transforms::trig_integ::try_trig_power_integral(arena, expr, var, var_sym)
    {
        // Only use the trig-power result when it is fully evaluated;
        // negative-exponent cases (e.g. sin·cos^{-2}) come back as
        // unevaluated Integral nodes — fall through so the Mul handler
        // can try sec·tan / csc·cot patterns and u-substitution.
        if !matches!(arena.node(result), ExprNode::Integral(_, _)) {
            return result;
        }
    }

    // ── Trig identity rewrites ────────────────────────────────────
    // Rewrite squared trig identities to forms with known antiderivatives.
    if let ExprNode::Pow(trig_base, trig_exp) = arena.node(expr).clone()
        && let Some(n_val) = arena.as_num(trig_exp)
    {
        let two_r = num_rational::Ratio::<num_bigint::BigInt>::from_integer(2.into());
        if *n_val == two_r {
            // tan²(g) → sec²(g) − 1 = cos(g)^{-2} − 1
            if let ExprNode::Tan(inner) = arena.node(trig_base).clone() {
                let cos_inner = arena.cos(inner);
                let neg_two = arena.int(-2);
                let sec_sq = arena.pow(cos_inner, neg_two);
                let rewritten = arena.sub(sec_sq, arena.one);
                return integrate_node(arena, rewritten, var, var_sym, depth - 1);
            }
            // tanh²(g) → 1 − sech²(g) = 1 − cosh(g)^{-2}
            if let ExprNode::Tanh(inner) = arena.node(trig_base).clone() {
                let cosh_inner = arena.cosh(inner);
                let neg_two = arena.int(-2);
                let sech_sq = arena.pow(cosh_inner, neg_two);
                let rewritten = arena.sub(arena.one, sech_sq);
                return integrate_node(arena, rewritten, var, var_sym, depth - 1);
            }
        }
    }

    // ── Type dispatch: rational function detection ────────────────
    // Before dispatching on node type, check if the expression is a
    // rational function P(x)/Q(x).  If so, route to the complete
    // Hermite + Rothstein-Trager algorithm.  This handles ALL structural
    // variants (Mul with negative powers, Pow with negative exponent,
    // etc.) because as_numer_denom normalizes them all to (numer, denom).
    //
    // This is the standard CAS architecture: rational function integration
    // is a solved problem with efficient algorithms, and it should run
    // before any heuristic pattern matching.
    //
    // The result is verified numerically before it is trusted: when the
    // Rothstein–Trager resultant has nested-radical roots the `log_to_real`
    // conversion can emit a closed form with opaque `re(…)`/`im(…)`
    // constants that is simply wrong (0.21 audit: `∫ 1/((x²−4)²+1)`).  A
    // rejected candidate goes to the biquadratic route below; if that does
    // not apply either, the integral is left unevaluated: every remaining
    // strategy for a rational function is the `apart` root-based fallback,
    // which recovers its coefficients from the same nested-radical roots
    // (and recurses deeply on them).  A wrong closed form is never returned
    // silently.
    let risch_rejected = match stage!(
        arena,
        "risch_rational",
        expr,
        crate::calculus::risch::try_risch_rational(arena, expr, var)
    ) {
        Some(result) if !matches!(arena.node(result), ExprNode::Integral(_, _)) => {
            if !candidate_rejected(arena, expr, result, var, var_sym) {
                return result;
            }
            tracing::debug!("integrate: rejecting unverified rational-function closed form");
            true
        }
        _ => false,
    };

    // ── N(x)/(x⁴ + p·x² + q): explicit real factorisation ──────────────
    // Reached only when the general rational integrator could not produce
    // a verified answer, so the output of the cases it handles is unchanged.
    if let Some(result) = stage!(
        arena,
        "biquadratic",
        expr,
        try_biquadratic_rational(arena, expr, var, var_sym)
    ) {
        return result;
    }
    if risch_rejected {
        return arena.intern(ExprNode::Integral(expr, var));
    }
    if let Some(result) = try_poly_over_symbolic_linear(arena, expr, var, var_sym) {
        return result;
    }
    // ── Rational functions over ℚ(p₁, …, p_k): parameters as coefficients ──
    // The rational integrator over the field of the parameters (0.31):
    // `∫ dx/((x + a)(x + b))`, `∫ dx/(x² + a·x + b)`, `∫ dx/(a² − x²)` were
    // unevaluated, `∫ dx/(x² − a²)` was `atan(x/√(−a²))/√(−a²)`.  A
    // rejected candidate falls through to the routes below.
    if let Some(result) = stage!(
        arena,
        "risch_rational_params",
        expr,
        crate::calculus::risch::try_risch_rational_params(arena, expr, var)
    ) && !candidate_rejected(arena, expr, result, var, var_sym)
    {
        return result;
    }
    if let Some(result) = stage!(
        arena,
        "gaussian_rational",
        expr,
        try_gaussian_rational(arena, expr, var, var_sym, depth)
    ) {
        return result;
    }
    if let Some(result) = try_poly_over_symbolic_quadratic(arena, expr, var, var_sym, depth) {
        return result;
    }

    let node = arena.node(expr).clone();

    match node {
        // ── Anything free of var → c * var ─────────────────────────
        // A `RootOf(x⁵ − x + 1, 0)` (its `x` is bound), `sin a`, `Γ(a)`:
        // up to 0.28.0 such a term of a sum fell through to the unevaluated
        // form (`∫ (x + RootOf(a⁵ − a + 1, 0)) dx`).  A sum keeps its
        // term-by-term split below.
        _ if !matches!(node, ExprNode::Add(_)) && !contains_var(arena, expr, var_sym) => {
            arena.mul(&[expr, var])
        }

        // ── Constants (independent of var) → c * var ───────────────
        ExprNode::Num(_)
        | ExprNode::Pi
        | ExprNode::E
        | ExprNode::ImaginaryUnit
        | ExprNode::Infinity
        | ExprNode::NegInfinity
        | ExprNode::ComplexInfinity
        | ExprNode::NaN => {
            // ∫ c dx = c * x
            arena.mul(&[expr, var])
        }

        ExprNode::Symbol(sid) => {
            if sid == var_sym {
                // ∫ x dx = x^2 / 2
                let two = arena.int(2);
                let x_sq = arena.pow(var, two);
                let half = arena.rational(1, 2);
                arena.mul(&[half, x_sq])
            } else {
                // ∫ c dx = c * x (c is independent of var)
                arena.mul(&[expr, var])
            }
        }

        // ── Add: linearity ─────────────────────────────────────────
        ExprNode::Add(ref children) => {
            let integrals: SmallVec<[ExprId; 6]> = children
                .iter()
                .map(|&child| integrate_node(arena, child, var, var_sym, depth - 1))
                .collect();
            arena.add(&integrals)
        }

        // ── Mul: factor out constants ──────────────────────────────
        ExprNode::Mul(ref children) => {
            // Separate constant factors (independent of var) from the rest,
            // with the dependent factors' nested powers flattened.
            let (mut constants, dependent) = partition_factors(arena, children, var_sym);

            if dependent.is_empty() {
                // All constant: ∫ c dx = c * x
                return arena.mul(&[expr, var]);
            }

            // ── A product of polynomials: multiplied out ──
            if let Some(expanded) = expand_polynomial_product(arena, &dependent, var_sym) {
                let expanded = wrap_with_constants(arena, expanded, &constants);
                let result = integrate_node(arena, expanded, var, var_sym, depth - 1);
                if !crate::base::walk::has_unevaluated(arena, result) {
                    return result;
                }
            }

            // ── Collect exponents: exp(u)·exp(v) → exp(u + v) ──
            if let Some(merged) = merge_exponential_factors(arena, &dependent, var, var_sym) {
                let merged = wrap_with_constants(arena, merged, &constants);
                let result = integrate_node(arena, merged, var, var_sym, depth - 1);
                if !crate::base::walk::has_unevaluated(arena, result) {
                    return result;
                }
            }

            if !constants.is_empty() && dependent.len() == 1 {
                // c * f(x) → c * ∫ f(x) dx
                let inner_integral = integrate_node(arena, dependent[0], var, var_sym, depth - 1);
                // Check if the inner integral is unevaluated
                if let ExprNode::Integral(_, _) = arena.node(inner_integral) {
                    // Can't integrate the inner part — return unevaluated for whole
                    return arena.intern(ExprNode::Integral(expr, var));
                }
                constants.push(inner_integral);
                return arena.mul(&constants);
            }

            // ── tanᵐ·sec² and secⁿ·tan forms ───────────────────────────
            if dependent.len() == 2
                && let Some(result) = try_tan_sec_patterns(arena, &dependent, var, var_sym)
            {
                return wrap_with_constants(arena, result, &constants);
            }

            // ── sec(x)·tan(x) and csc(x)·cot(x) forms ──────────────
            if dependent.len() == 2
                && let Some(result) = try_trig_recip_product(arena, &dependent, var, var_sym)
            {
                if constants.is_empty() {
                    return result;
                } else {
                    let mut all = constants.clone();
                    all.push(result);
                    return arena.mul(&all);
                }
            }

            // ── x/√(ax²+bx+c) form ──────────────────────────────────
            if dependent.len() == 2
                && let Some(result) =
                    try_x_over_sqrt_quadratic(arena, &dependent, var, var_sym, depth)
            {
                if constants.is_empty() {
                    return result;
                } else {
                    let mut all = constants.clone();
                    all.push(result);
                    return arena.mul(&all);
                }
            }

            // ── (linear) / (irreducible quadratic) ───────────────────
            if dependent.len() == 2
                && let Some(result) =
                    try_linear_over_quadratic(arena, &dependent, var, var_sym, depth)
            {
                if constants.is_empty() {
                    return result;
                } else {
                    let mut all = constants.clone();
                    all.push(result);
                    return arena.mul(&all);
                }
            }

            // ── DiracDelta sifting property: ∫ f(x)·δ(g(x)) dx = f(root)·H(g(x)) ──
            // Check if any factor in the product is a DiracDelta.
            {
                let all_children: SmallVec<[ExprId; 6]> = children.clone();
                for (i, &child) in all_children.iter().enumerate() {
                    if let ExprNode::DiracDelta(delta_arg) = arena.node(child).clone() {
                        tracing::debug!(
                            "integrate: detected DiracDelta factor in Mul, attempting sifting property"
                        );

                        // Collect the remaining factors as f(x)
                        let other_factors: SmallVec<[ExprId; 4]> = all_children
                            .iter()
                            .enumerate()
                            .filter(|&(j, _)| j != i)
                            .map(|(_, &c)| c)
                            .collect();
                        let f_expr = if other_factors.len() == 1 {
                            other_factors[0]
                        } else if other_factors.is_empty() {
                            arena.one
                        } else {
                            arena.mul(&other_factors)
                        };

                        // Simple case: δ(x) → root = 0
                        if delta_arg == var {
                            let f_at_0 =
                                crate::transforms::subs::subs(arena, f_expr, var, arena.zero);
                            let f_at_0_eval = crate::transforms::eval::eval(arena, f_at_0);
                            let heaviside = arena.intern(ExprNode::Heaviside(var));
                            return arena.mul(&[f_at_0_eval, heaviside]);
                        }

                        // General case: solve δ(g(x)) = 0, i.e. g(x) = 0 for x
                        let solutions =
                            crate::transforms::solve::known_real_roots(arena, delta_arg, var);
                        if solutions.len() == 1 {
                            let root = solutions[0];
                            let f_at_root = crate::transforms::subs::subs(arena, f_expr, var, root);
                            let f_at_root_eval = crate::transforms::eval::eval(arena, f_at_root);
                            let heaviside = arena.intern(ExprNode::Heaviside(delta_arg));
                            return arena.mul(&[f_at_root_eval, heaviside]);
                        }
                        // Several simple real roots: δ(g) = Σ δ(x − rₖ)/|g′(rₖ)|, so
                        // ∫ f·δ(g) = Σ f(rₖ)/|g′(rₖ)|·H(x − rₖ) (up to 0.46 only one
                        // root was taken: `∫ x·δ(x² − 1)·cos x·sin x dx` stayed
                        // unevaluated; SymPy's test_deltafunctions).
                        if let Some(r) = delta_simple_roots(arena, f_expr, delta_arg, var) {
                            return r;
                        }

                        // If we can't solve, fall through to other strategies
                        break;
                    }
                }
            }

            // ── P(x)·|g(x)|, P(x)·sign(g(x)), P(x)·H(g(x)) ─────────────────
            // Before by parts: the factor's sign is constant on each side of
            // the root, so the rest is a smooth integrand.  By parts first took
            // `u = x²`, `dv = |x|/(x³ + x² + 1)` in `∫ x²·|x|/(x³ + x² + 1)`,
            // and `∫ v·du` of the Cardano-root antiderivative (17,000-character
            // integrands) ran into a 74 s refusal of `∫ (√x + |x|)/(x + x⁻² +
            // 1) dx` (`fuzz_integrate`, 2026-10-01).
            if let Some(result) = try_abs_sign_product(arena, &dependent, var, var_sym, depth) {
                return wrap_with_constants(arena, result, &constants);
            }

            // ── P(x)·e^{q(x)}, P(x)·sin q(x), P(x)·cos q(x) for quadratic q ──
            // Before by parts, which takes `u = P` and `dv = e^q` (an `erf`).
            if let Some(result) = stage!(
                arena,
                "poly_times_quadratic",
                expr,
                try_poly_times_quadratic(arena, &dependent, var, var_sym, depth)
            ) {
                return wrap_with_constants(arena, result, &constants);
            }

            // ── ln(a·x + b)·R(x) with a factor of the denominator irreducible
            // of degree ≥ 2: before by parts, whose `v` (Ferrari's roots of a
            // quartic) only feeds nested attempts that cannot succeed (3.4 s
            // of 4.3 s for `ln(−x)·(−x² + 2x − 2)/(x⁴ − x² + 2x + 1)`).
            if let Some(result) = stage!(
                arena,
                "log_times_rational",
                expr,
                try_log_times_rational(arena, &dependent, var, var_sym, depth, true)
            ) {
                return wrap_with_constants(arena, result, &constants);
            }

            // ── N(x)/D(x), D over ℚ, N with constant coefficients: by constants ──
            // Before by parts, whose nested attempts on such a quotient only
            // repeat, piece by piece, what the rational integrator does at once.
            if let Some(result) = stage!(
                arena,
                "rational_constant_coefficients",
                expr,
                try_rational_with_constant_coefficients(arena, &dependent, var, var_sym, depth)
            ) {
                return wrap_with_constants(arena, result, &constants);
            }

            // ── Integration by parts: ∫ u·dv = u·v - ∫ v·du ───────────
            // Try when there are exactly 2 dependent factors:
            // one that's a by-parts candidate (u), and one that's directly
            // integrable (dv).
            if dependent.len() == 2
                && let Some(result) = stage!(
                    arena,
                    "by_parts",
                    expr,
                    try_by_parts_pair(arena, &dependent, var, var_sym, depth)
                )
            {
                return wrap_with_constants(arena, result, &constants);
            }

            // ── Cyclic IBP: ∫ exp·sin, ∫ exp·cos, etc. ────────────
            if dependent.len() == 2
                && let Some(result) = stage!(
                    arena,
                    "cyclic_ibp",
                    expr,
                    try_cyclic_ibp(arena, &dependent, var, var_sym, depth)
                )
            {
                if constants.is_empty() {
                    return result;
                } else {
                    let mut all = constants.clone();
                    all.push(result);
                    return arena.mul(&all);
                }
            }

            // ── P(x)·Q(x)^{k/2}: reduction to R√Q + c∫1/√Q ────────────
            if let Some(result) = try_poly_times_half_power(arena, &dependent, var, var_sym, depth)
            {
                return wrap_with_constants(arena, result, &constants);
            }
            if let Some(result) =
                try_symbolic_quadratic_radical(arena, &dependent, var, var_sym, depth)
            {
                return wrap_with_constants(arena, result, &constants);
            }

            // ── x^{-n}·Q(x)^{-1/2}: substitution x = 1/t ───────────────
            if let Some(result) =
                try_reciprocal_sqrt_substitution(arena, &dependent, var, var_sym, depth)
            {
                return wrap_with_constants(arena, result, &constants);
            }

            // ── Three-factor by parts: u = polynomial, dv = product of the rest ──
            if dependent.len() == 3
                && let Some(result) =
                    try_by_parts_poly_times_pair(arena, &dependent, var, var_sym, depth)
            {
                return wrap_with_constants(arena, result, &constants);
            }

            // ── Several polynomial factors: multiply them out, then retry ──
            if dependent.len() >= 3
                && let Some(result) =
                    try_merge_polynomial_factors(arena, &dependent, var, var_sym, depth)
            {
                return wrap_with_constants(arena, result, &constants);
            }

            // ── Products of trig factors: product-to-sum, then retry ─────
            if dependent.len() >= 2
                && let Some(result) = stage!(
                    arena,
                    "trig_product_to_sum",
                    expr,
                    try_trig_product_to_sum(arena, expr, &dependent, var, var_sym, depth)
                )
            {
                return result;
            }

            // ── Try partial fraction decomposition for rational integrands ──
            // (verified: the root-based fallback of `apart` may recover its
            // coefficients from `f64` roots, which is not an antiderivative.)
            {
                let (_numer, denom) = crate::poly::polybridge::as_numer_denom(arena, expr);
                if denom != arena.one {
                    let decomposed = stage!(
                        arena,
                        "apart",
                        expr,
                        crate::transforms::apart::apart(arena, expr, var)
                    );
                    // Partial fractions over implicit roots could only give
                    // a candidate `candidate_rejected` refuses.
                    if decomposed != expr && !introduces_root_of(arena, expr, decomposed) {
                        let result = integrate_node(arena, decomposed, var, var_sym, depth - 1);
                        if !matches!(arena.node(result), ExprNode::Integral(_, _))
                            && !candidate_rejected(arena, expr, result, var, var_sym)
                        {
                            return result;
                        }
                    }
                }
            }

            // ── Try general u-substitution ──────────────────────────────
            if let Some(result) = stage!(
                arena,
                "u_substitution",
                expr,
                try_u_substitution(arena, &dependent, var, var_sym, depth - 1)
            ) {
                if constants.is_empty() {
                    return result;
                } else {
                    let mut all = constants.clone();
                    all.push(result);
                    return arena.mul(&all);
                }
            }

            // ── sin(u)ᵐ·cos(u)ⁿ times other factors: power reduction ──
            // `e^{x/2}·cos²(3x/2) = e^{x/2}/2 + e^{x/2}·cos(3x)/2`; up to 0.31
            // a trigonometric power times an exponential stayed unevaluated
            // (by parts and the cyclic rule need single factors).
            if dependent.len() >= 2
                && let Some(linear) = crate::transforms::trig_integ::linearize_trig_powers(
                    arena, &dependent, var, var_sym,
                )
            {
                let result = integrate_node(arena, linear, var, var_sym, depth - 1);
                if !crate::base::walk::has_unevaluated(arena, result) {
                    return wrap_with_constants(arena, result, &constants);
                }
            }

            // ── Distribute a sum factor: ∫ (s₁ + … + s_k)·g = Σ ∫ sᵢ·g ──
            if let Some(result) = stage!(
                arena,
                "distribute_sum_factor",
                expr,
                try_distribute_sum_factor(arena, &dependent, var, var_sym, depth)
            ) {
                return wrap_with_constants(arena, result, &constants);
            }

            // ── Weierstrass substitution for rational trig functions ──
            // The constant factors were split off above and are re-applied
            // here, so the substitution must see only the dependent part
            // (passing `expr` applied them twice: `∫ 3·g` came back as `9·∫g`).
            let dependent_product = if dependent.len() == 1 {
                dependent[0]
            } else {
                arena.mul(&dependent)
            };
            if let Some(result) = stage!(
                arena,
                "weierstrass",
                dependent_product,
                try_weierstrass_substitution(arena, dependent_product, var, var_sym, depth)
            ) {
                return wrap_with_constants(arena, result, &constants);
            }

            // ── Special function integration table ──────────────────
            if let Some(result) = stage!(
                arena,
                "special_functions",
                expr,
                try_special_function_integral(arena, &dependent, &constants, var, var_sym, depth)
            ) {
                return result;
            }
            if let Some(result) = try_dilog_integral(arena, &dependent, var, var_sym) {
                return wrap_with_constants(arena, result, &constants);
            }
            if let Some(result) = stage!(
                arena,
                "special_product_as_exponentials",
                expr,
                try_special_product_as_exponentials(arena, &dependent, var, var_sym, depth)
            ) {
                return wrap_with_constants(arena, result, &constants);
            }
            if let Some(result) = try_incomplete_gamma(arena, &dependent, var, var_sym) {
                return wrap_with_constants(arena, result, &constants);
            }
            if let Some(result) = bessel_power_integral(arena, &dependent, var, var_sym) {
                return wrap_with_constants(arena, result, &constants);
            }
            if let Some(result) = try_power_substitution(arena, &dependent, var, var_sym, depth) {
                return wrap_with_constants(arena, result, &constants);
            }
            if let Some(result) = stage!(
                arena,
                "log_times_rational",
                expr,
                try_log_times_rational(arena, &dependent, var, var_sym, depth, false)
            ) {
                return wrap_with_constants(arena, result, &constants);
            }

            // ── A sum factor with a common factor: (g·h)ⁿ = gⁿ·hⁿ ──
            if let Some(split) = split_common_factors(arena, &dependent, var_sym) {
                let split = wrap_with_constants(arena, split, &constants);
                let _cap = StepCap::enter(SPLIT_STEPS);
                let result = integrate_node(arena, split, var, var_sym, depth - 1);
                if !crate::base::walk::has_unevaluated(arena, result) {
                    return result;
                }
            }

            // General product of var-dependent terms — can't integrate without
            // further techniques.
            tracing::debug!("integration: no strategy succeeded, returning unevaluated");
            arena.intern(ExprNode::Integral(expr, var))
        }

        // ── Neg ────────────────────────────────────────────────────
        ExprNode::Neg(inner) => {
            let inner_int = integrate_node(arena, inner, var, var_sym, depth - 1);
            arena.neg(inner_int)
        }

        // ── Pow: power rule ────────────────────────────────────────
        ExprNode::Pow(base, exp) => {
            // Flatten Pow(Pow(a, m), n) → Pow(a, m·n) where that keeps the
            // value (see `flatten_nested_pow`): e.g. 1/√(x²+1) =
            // Pow(Pow(x²+1, 1/2), -1) becomes Pow(x²+1, -1/2) and hits the
            // standard-form / completing-the-square handlers, and √(x²) is
            // |x| for the real integration variable.
            if let Some(flattened) = flatten_nested_pow(arena, expr, var_sym)
                .or_else(|| exp_power_as_exp(arena, expr, var_sym))
                && flattened != expr
            {
                return integrate_node(arena, flattened, var, var_sym, depth - 1);
            }

            // ── |g|ⁿ, integer n ≥ 2: gⁿ for even n, |g|·gⁿ⁻¹ for odd n ──
            // (gⁿ⁻¹ ≥ 0 for odd n, so the product form is exact); the
            // product is then the `P(x)·|g(x)|` shape of `try_abs_sign_product`.
            // Only for g real on the real line: |√x|² = |x|, not x (0.23 gave
            // ∫ |√x|² dx = x²/2; found by fuzz_integrate).
            if let ExprNode::Abs(g) = arena.node(base).clone()
                && contains_var(arena, g, var_sym)
                && real_on_reals(arena, g, var_sym)
                && let Some(n_val) = arena.as_num(exp).cloned()
                && n_val.is_integer()
                && n_val >= num_rational::Ratio::from_integer(2.into())
            {
                let n_int = n_val.to_integer();
                let n_is_even = (&n_int % num_bigint::BigInt::from(2)).is_zero();
                let rewritten = if n_is_even {
                    arena.pow(g, exp)
                } else {
                    let n_minus_1 = arena.num_ratio(&n_val - &Q::one());
                    let g_pow = arena.pow(g, n_minus_1);
                    arena.mul(&[base, g_pow])
                };
                let result = integrate_node(arena, rewritten, var, var_sym, depth - 1);
                if !crate::base::walk::has_unevaluated(arena, result) {
                    return result;
                }
            }

            let base_is_var = base == var;
            let exp_has_var = contains_var(arena, exp, var_sym);
            let base_has_var = contains_var(arena, base, var_sym);

            if base_is_var && !exp_has_var {
                // ∫ x^n dx
                if let Some(n) = arena.as_num(exp) {
                    let n = n.clone();
                    if n == num_rational::Ratio::from_integer((-1).into()) {
                        // ∫ x^(-1) dx = ln(|x|)
                        let abs_x = arena.abs(var);
                        return arena.ln(abs_x);
                    }
                    // ∫ x^n dx = x^(n+1) / (n+1) for n ≠ -1
                    let one = num_rational::Ratio::<num_bigint::BigInt>::one();
                    let n_plus_1 = &n + &one;
                    let n_plus_1_id = {
                        let nid = arena.intern_num(n_plus_1.clone());
                        arena.intern(ExprNode::Num(nid))
                    };
                    let x_pow = arena.pow(var, n_plus_1_id);
                    let recip = {
                        let inv = one / n_plus_1;
                        let nid = arena.intern_num(inv);
                        arena.intern(ExprNode::Num(nid))
                    };
                    return arena.mul(&[recip, x_pow]);
                } else {
                    // Symbolic exponent: ∫ x^n dx = x^(n+1)/(n+1)
                    let one_id = arena.one;
                    let n_plus_1 = arena.add(&[exp, one_id]);
                    let x_pow = arena.pow(var, n_plus_1);
                    return arena.div(x_pow, n_plus_1);
                }
            }

            if !base_has_var && !exp_has_var {
                // Constant: ∫ c dx = c * x
                return arena.mul(&[expr, var]);
            }

            // ── c^{g(x)} with constant c > 0: rewrite as exp(g·ln c) ─────
            if !base_has_var && exp_has_var && base != arena.e_const() {
                let ln_c = arena.ln(base);
                let ln_c = crate::transforms::eval::eval(arena, ln_c);
                let new_exp = arena.mul(&[exp, ln_c]);
                let rewritten = arena.exp(new_exp);
                let result = integrate_node(arena, rewritten, var, var_sym, depth - 1);
                if !crate::base::walk::has_unevaluated(arena, result) {
                    return result;
                }
                return arena.intern(ExprNode::Integral(expr, var));
            }

            // ── tanⁿ(g), n ≥ 3 integer: tanⁿ = tanⁿ⁻²·(sec² − 1) ─────────
            if let ExprNode::Tan(inner) = arena.node(base).clone()
                && let Some(n_val) = arena.as_num(exp).cloned()
                && n_val.is_integer()
                && n_val >= num_rational::Ratio::from_integer(3.into())
                && n_val <= num_rational::Ratio::from_integer(12.into())
            {
                let n_minus_2 =
                    arena.num_ratio(&n_val - &num_rational::Ratio::from_integer(2.into()));
                let tan_pow = arena.pow(base, n_minus_2);
                let cos_inner = arena.cos(inner);
                let neg_two = arena.int(-2);
                let sec_sq = arena.pow(cos_inner, neg_two);
                let term1 = arena.mul(&[tan_pow, sec_sq]);
                let rewritten = arena.sub(term1, tan_pow);
                let result = integrate_node(arena, rewritten, var, var_sym, depth - 1);
                if !crate::base::walk::has_unevaluated(arena, result) {
                    return result;
                }
            }

            // ── sech²(g) = cosh(g)^{-2} → tanh(g) / chain_coeff ──
            if let ExprNode::Cosh(inner) = arena.node(base).clone()
                && let Some(e_val) = arena.as_num(exp)
            {
                let neg_two_r =
                    num_rational::Ratio::<num_bigint::BigInt>::from_integer((-2).into());
                if *e_val == neg_two_r {
                    if inner == var {
                        return arena.tanh(var);
                    }
                    if let Some((a_expr, _)) = symbolic_linear_coeff_of(arena, inner, var, var_sym)
                    {
                        let tanh_inner = arena.tanh(inner);
                        return arena.div(tanh_inner, a_expr);
                    }
                }
            }

            // ── csch²(g) = sinh(g)^{-2} → −coth(g) / chain_coeff ──
            if let ExprNode::Sinh(inner) = arena.node(base).clone()
                && let Some(e_val) = arena.as_num(exp)
            {
                let neg_two_r =
                    num_rational::Ratio::<num_bigint::BigInt>::from_integer((-2).into());
                if *e_val == neg_two_r {
                    if inner == var {
                        let cosh_v = arena.cosh(var);
                        let sinh_v = arena.sinh(var);
                        let neg1 = arena.int(-1);
                        let sinh_inv = arena.pow(sinh_v, neg1);
                        let coth_v = arena.mul(&[cosh_v, sinh_inv]);
                        return arena.neg(coth_v);
                    }
                    if let Some((a_expr, _)) = symbolic_linear_coeff_of(arena, inner, var, var_sym)
                    {
                        let cosh_i = arena.cosh(inner);
                        let sinh_i = arena.sinh(inner);
                        let neg1 = arena.int(-1);
                        let sinh_inv = arena.pow(sinh_i, neg1);
                        let coth_i = arena.mul(&[cosh_i, sinh_inv]);
                        let neg_coth = arena.neg(coth_i);
                        return arena.div(neg_coth, a_expr);
                    }
                }
            }

            // ── ln(x)^n by parts: ∫ ln(x)^n dx = x·ln(x)^n − n·∫ ln(x)^(n−1) dx ──
            if let ExprNode::Ln(inner) = arena.node(base).clone()
                && inner == var
                && !exp_has_var
                && let Some(n_val) = arena.as_num(exp)
            {
                let n_val = n_val.clone();
                if n_val.is_integer() && n_val.is_positive() {
                    let n_i64: i64 = n_val.to_integer().try_into().unwrap_or(0);
                    if n_i64 >= 2 {
                        let x_ln_n = arena.mul(&[var, expr]);
                        let n_id = arena.num_ratio(n_val.clone());
                        let n_minus_1 = {
                            let v = &n_val - &num_rational::Ratio::<num_bigint::BigInt>::one();
                            arena.num_ratio(v.clone())
                        };
                        let ln_x = arena.ln(var);
                        let ln_nm1 = if n_i64 == 2 {
                            ln_x
                        } else {
                            arena.pow(ln_x, n_minus_1)
                        };
                        let sub_int = integrate_node(arena, ln_nm1, var, var_sym, depth - 1);
                        let n_times_sub = arena.mul(&[n_id, sub_int]);
                        return arena.sub(x_ln_n, n_times_sub);
                    }
                }
            }

            // General linear substitution: ∫ (ax+b)^n dx = (ax+b)^(n+1) / (a*(n+1))
            if !exp_has_var
                && base_has_var
                && let Some((a_expr, _b_expr)) = symbolic_linear_coeff_of(arena, base, var, var_sym)
                && let Some(n) = arena.as_num(exp)
            {
                let n = n.clone();
                let neg_one = num_rational::Ratio::from_integer((-1).into());
                if n != neg_one {
                    // ∫ (ax+b)^n dx = (ax+b)^(n+1) / (a*(n+1))
                    let one = num_rational::Ratio::<num_bigint::BigInt>::one();
                    let n_plus_1 = &n + &one;
                    let n_plus_1_id = arena.num_ratio(n_plus_1.clone());
                    let base_pow = arena.pow(base, n_plus_1_id);
                    let denom = arena.mul(&[a_expr, n_plus_1_id]);
                    return arena.div(base_pow, denom);
                } else {
                    // ∫ (ax+b)^(-1) dx = ln|ax+b| / a
                    let abs_base = arena.abs(base);
                    let ln_base = arena.ln(abs_base);
                    return arena.div(ln_base, a_expr);
                }
            }
            // ∫ (αx + β)^n dx for a symbolic exponent: (αx + β)^(n+1)/(α·(n + 1)),
            // the case n = −1 from the degenerate-case wrap (as for xⁿ).  Up to
            // 0.46 only xⁿ had this: `∫ (b + t)^(−a) dt` stayed unevaluated
            // (SymPy's test_meijerint: `(b + t)**(1 − a)/(1 − a)` with the case).
            if !exp_has_var
                && base_has_var
                && arena.as_num(exp).is_none()
                && let Some((a_expr, _)) = symbolic_linear_coeff_of(arena, base, var, var_sym)
            {
                let one = arena.one;
                let n_plus_1 = arena.add(&[exp, one]);
                let n_plus_1 = crate::transforms::eval::eval(arena, n_plus_1);
                if !arena.is_zero_structural(n_plus_1) {
                    let base_pow = arena.pow(base, n_plus_1);
                    let denom = arena.mul(&[a_expr, n_plus_1]);
                    return arena.div(base_pow, denom);
                }
            }

            // ── Standard form integrals (A3–A7) ───────────────────────
            if base_has_var
                && !exp_has_var
                && let Some(result) =
                    try_standard_form_integral(arena, expr, base, exp, var, var_sym)
            {
                return result;
            }

            // ── Symbolic standard form: ∫ (x² + k)^{-1} dx ──────────
            // where k is free of var (handles e.g. ∫ 1/(x²+a²) dx)
            if base_has_var
                && !exp_has_var
                && let Some(exp_val) = arena.as_num(exp)
            {
                let neg_one_r =
                    num_rational::Ratio::<num_bigint::BigInt>::from_integer((-1).into());
                if *exp_val == neg_one_r
                    && let ExprNode::Add(ref ac) = arena.node(base).clone()
                    && ac.len() == 2
                {
                    let (mut x2_found, mut k_id) = (false, None);
                    for &ch in ac.iter() {
                        if is_var_squared(arena, ch, var) {
                            x2_found = true;
                        } else if !contains_var(arena, ch, var_sym) {
                            k_id = Some(ch);
                        }
                    }
                    if x2_found
                        && let Some(k) = k_id
                        && arena.as_num(k).is_none()
                    {
                        // Only use symbolic path when k is NOT pure numeric
                        // (numeric case is handled by try_standard_form_integral)
                        // ∫ 1/(x²+k) dx = (1/√k)·atan(x/√k)
                        let half = arena.rational(1, 2);
                        let sqrt_k = arena.pow(k, half);
                        let x_over_sk = arena.div(var, sqrt_k);
                        let atan_val = arena.atan(x_over_sk);
                        let neg_half = arena.rational(-1, 2);
                        let inv_sk = arena.pow(k, neg_half);
                        return arena.mul(&[inv_sk, atan_val]);
                    }
                }
            }

            // ── Completing the square for 1/(ax²+bx+c) ───────────────
            if base_has_var
                && !exp_has_var
                && let Some(result) = try_complete_square_integral(arena, base, exp, var, var_sym)
            {
                return result;
            }

            // ── Trig substitution: √(a²±x²), √(x²±a²) ──────────────
            if base_has_var
                && !exp_has_var
                && let Some(result) = try_trig_sub_sqrt_integral(arena, base, exp, var, var_sym)
            {
                return result;
            }

            // ── Q^{k/2} for a quadratic Q over ℚ, k = 1, 3: R·√Q + c·∫ dx/√Q ──
            // (up to 0.46 only √(a² ± x²) had a rule: `∫ √(5x² − 6x + 2) dx`
            // stayed unevaluated, `∫ √(5x² + 1) dx` came out of the Möbius
            // substitution; SymPy's test_integrals `√(53225x² − 66732x + 23013)`).
            if base_has_var
                && !exp_has_var
                && arena
                    .as_num(exp)
                    .is_some_and(|r| r.is_positive() && *r.denom() == num_bigint::BigInt::from(2))
                && let Some(result) = try_poly_times_half_power(arena, &[expr], var, var_sym, depth)
            {
                return result;
            }

            // ── P(x)·Q^{±1/2} for a quadratic Q with symbolic coefficients ──
            if base_has_var
                && !exp_has_var
                && let Some(result) =
                    try_symbolic_quadratic_radical(arena, &[expr], var, var_sym, depth)
            {
                return result;
            }

            // ── Try partial fraction decomposition (verified, see the Mul arm) ──
            {
                let (_numer, denom) = crate::poly::polybridge::as_numer_denom(arena, expr);
                if denom != arena.one {
                    let decomposed = stage!(
                        arena,
                        "apart",
                        expr,
                        crate::transforms::apart::apart(arena, expr, var)
                    );
                    if decomposed != expr && !introduces_root_of(arena, expr, decomposed) {
                        let result = integrate_node(arena, decomposed, var, var_sym, depth - 1);
                        if !matches!(arena.node(result), ExprNode::Integral(_, _))
                            && !candidate_rejected(arena, expr, result, var, var_sym)
                        {
                            return result;
                        }
                    }
                }
            }

            // Fallback: if base is an Add and exp is a small positive integer, expand and retry
            if let ExprNode::Add(_) = arena.node(base)
                && let Some(n) = arena.as_num(exp)
                && n.is_integer()
                && n.is_positive()
            {
                let n_i64: i64 = n.to_integer().try_into().unwrap_or(0);
                if (2..=10).contains(&n_i64) {
                    let expanded = crate::transforms::expand::expand(arena, expr);
                    if expanded != expr {
                        let result = integrate_node(arena, expanded, var, var_sym, depth - 1);
                        if !matches!(arena.node(result), ExprNode::Integral(_, _)) {
                            return result;
                        }
                    }
                }
            }

            // ── Weierstrass substitution (Pow arm) ────────────────────
            if let Some(result) = stage!(
                arena,
                "weierstrass",
                expr,
                try_weierstrass_substitution(arena, expr, var, var_sym, depth)
            ) {
                return result;
            }

            // ── 1/ln(x) → li(x) (logarithmic integral) ──────────────
            if let ExprNode::Ln(inner) = arena.node(base).clone()
                && inner == var
                && let Some(n_val) = arena.as_num(exp)
            {
                let neg_one_r =
                    num_rational::Ratio::<num_bigint::BigInt>::from_integer((-1).into());
                if *n_val == neg_one_r {
                    return arena.li(var);
                }
            }
            // ── ln(αx + β)⁻ⁿ → li(αx + β)/α and by parts ──────────────
            if let Some(result) = try_inverse_log_power(arena, base, exp, var, var_sym, depth) {
                return result;
            }

            // ── 1/(ax+b) with symbolic coefficients → ln|ax+b|/a ────
            if let Some(n_val) = arena.as_num(exp) {
                let neg_one_r =
                    num_rational::Ratio::<num_bigint::BigInt>::from_integer((-1).into());
                if *n_val == neg_one_r
                    && base_has_var
                    && let Some((a_expr, _b_expr)) =
                        symbolic_linear_coeff_of(arena, base, var, var_sym)
                {
                    let abs_base = arena.abs(base);
                    let ln_abs = arena.ln(abs_base);
                    return arena.div(ln_abs, a_expr);
                }
            }

            // ── Distribute inverse over Mul: 1/(a·b) → a^(-1)·b^(-1) ──
            // When the base is a Mul and the exponent is a negative integer,
            // distribute the power over each factor.  This transforms
            // Pow(Mul(x, ln(x)), -1) into Mul(x^(-1), ln(x)^(-1)), which
            // lets the Mul arm's u-substitution logic find candidates.
            if let ExprNode::Mul(ref children) = arena.node(base).clone()
                && let Some(e_val) = arena.as_num(exp)
                && e_val.is_negative()
                && e_val.is_integer()
            {
                let factors: SmallVec<[ExprId; 6]> = children
                    .iter()
                    .map(|&child| arena.pow(child, exp))
                    .collect();
                let distributed = arena.mul(&factors);
                if distributed != expr {
                    let result = integrate_node(arena, distributed, var, var_sym, depth - 1);
                    if !matches!(arena.node(result), ExprNode::Integral(_, _)) {
                        return result;
                    }
                }
            }

            // ── A common factor of the sum: (g·h)ⁿ = gⁿ·hⁿ ──
            if let Some(split) = split_common_factors(arena, &[expr], var_sym) {
                let _cap = StepCap::enter(SPLIT_STEPS);
                let result = integrate_node(arena, split, var, var_sym, depth - 1);
                if !crate::base::walk::has_unevaluated(arena, result) {
                    return result;
                }
            }

            // General case: unevaluated.
            tracing::debug!("integration: no strategy succeeded, returning unevaluated");
            arena.intern(ExprNode::Integral(expr, var))
        }

        // ── Elementary functions ────────────────────────────────────
        ExprNode::Sin(inner) => {
            if inner == var {
                // ∫ sin(x) dx = -cos(x)
                let cos_x = arena.cos(var);
                return arena.neg(cos_x);
            }
            // Try u-substitution: if inner = a*x + b, ∫ sin(a*x+b) dx = -cos(a*x+b)/a
            if let Some((a_expr, _)) = symbolic_linear_coeff_of(arena, inner, var, var_sym) {
                let cos_inner = arena.cos(inner);
                let neg_cos = arena.neg(cos_inner);
                return arena.div(neg_cos, a_expr);
            }
            if let Some(result) = fresnel_integral(arena, true, inner, var, var_sym) {
                return result;
            }
            arena.intern(ExprNode::Integral(expr, var))
        }

        ExprNode::Cos(inner) => {
            if inner == var {
                // ∫ cos(x) dx = sin(x)
                return arena.sin(var);
            }
            // Try u-substitution: if inner = a*x + b, ∫ cos(a*x+b) dx = sin(a*x+b)/a
            if let Some((a_expr, _)) = symbolic_linear_coeff_of(arena, inner, var, var_sym) {
                let sin_inner = arena.sin(inner);
                return arena.div(sin_inner, a_expr);
            }
            if let Some(result) = fresnel_integral(arena, false, inner, var, var_sym) {
                return result;
            }
            arena.intern(ExprNode::Integral(expr, var))
        }

        // ── Tan: ∫ tan(x) dx = -ln|cos(x)| ───────────────────────
        ExprNode::Tan(inner) => {
            if inner == var {
                // ∫ tan(x) dx = -ln(|cos(x)|)
                let cos_x = arena.cos(var);
                let abs_cos = arena.abs(cos_x);
                let ln_abs_cos = arena.ln(abs_cos);
                return arena.neg(ln_abs_cos);
            }
            // Try u-substitution: if inner = a*x + b,
            // ∫ tan(a*x+b) dx = -ln|cos(a*x+b)| / a
            if let Some((a_expr, _)) = symbolic_linear_coeff_of(arena, inner, var, var_sym) {
                let cos_inner = arena.cos(inner);
                let abs_cos = arena.abs(cos_inner);
                let ln_abs_cos = arena.ln(abs_cos);
                let neg_ln = arena.neg(ln_abs_cos);
                return arena.div(neg_ln, a_expr);
            }
            arena.intern(ExprNode::Integral(expr, var))
        }

        ExprNode::Exp(inner) => {
            if inner == var {
                // ∫ exp(x) dx = exp(x)
                return arena.exp(var);
            }

            // ── Gaussian form: ∫ exp(a·x²+b·x+c) dx where a < 0 ──
            // Result: √π/(2√(−a)) · exp(c − b²/(4a)) · erf((−2a·x−b)/(2√(−a)))
            if let Some(poly) = crate::poly::polybridge::expr_to_poly(arena, inner, var)
                && poly.degree() == Some(2)
            {
                let coeffs = poly.coeffs(); // [c, b, a]
                let a_coeff = &coeffs[2];
                let b_coeff = &coeffs[1];
                let c_coeff = &coeffs[0];

                if a_coeff.is_negative() {
                    // neg_a = -a (positive)
                    let neg_a = -a_coeff.clone();
                    let neg_a_expr = arena.num_ratio(neg_a.clone());

                    // sqrt(-a)
                    let sqrt_neg_a = arena.sqrt(neg_a_expr);

                    let two = arena.int(2);

                    // front = √π / (2·√(-a))
                    let pi_id = arena.pi;
                    let sqrt_pi = arena.sqrt(pi_id);
                    let two_sqrt_neg_a = arena.mul(&[two, sqrt_neg_a]);
                    let front = arena.div(sqrt_pi, two_sqrt_neg_a);

                    // exp_factor = exp(c - b²/(4a))
                    let b_expr = arena.num_ratio(b_coeff.clone());
                    let a_expr = arena.num_ratio(a_coeff.clone());
                    let c_expr = arena.num_ratio(c_coeff.clone());

                    let b_sq = arena.mul(&[b_expr, b_expr]);
                    let four = arena.int(4);
                    let four_a = arena.mul(&[four, a_expr]);
                    let b_sq_over_4a = arena.div(b_sq, four_a);
                    let exp_arg = arena.sub(c_expr, b_sq_over_4a);
                    let exp_arg = arena.eval_expr(exp_arg);
                    let exp_factor = if arena.is_zero_structural(exp_arg) {
                        arena.int(1)
                    } else {
                        arena.exp(exp_arg)
                    };

                    // erf_arg = (-2a·x - b) / (2·√(-a))
                    // Note: -2a is positive since a < 0
                    let neg_two_a = {
                        let two_r =
                            num_rational::Ratio::<num_bigint::BigInt>::from_integer(2.into());
                        let val = -two_r * a_coeff;
                        arena.num_ratio(val.clone())
                    };
                    let neg_2ax = arena.mul(&[neg_two_a, var]);
                    let erf_numer = arena.sub(neg_2ax, b_expr);
                    // recompute sqrt_neg_a fresh (the prior one may have been consumed)
                    let sqrt_neg_a2 = arena.sqrt(neg_a_expr);
                    let erf_denom = arena.mul(&[two, sqrt_neg_a2]);
                    let erf_arg = arena.div(erf_numer, erf_denom);
                    let erf_term = arena.erf(erf_arg);

                    return arena.mul(&[front, exp_factor, erf_term]);
                }
                if a_coeff.is_positive() {
                    // ∫ exp(a·x²+b·x+c) dx, a > 0:
                    //   √π/(2√a) · exp(c − b²/(4a)) · erfi((2a·x+b)/(2√a))   (0.9)
                    let a_expr = arena.num_ratio(a_coeff.clone());
                    let sqrt_a = arena.sqrt(a_expr);
                    let two = arena.int(2);
                    let pi_id = arena.pi;
                    let sqrt_pi = arena.sqrt(pi_id);
                    let two_sqrt_a = arena.mul(&[two, sqrt_a]);
                    let front = arena.div(sqrt_pi, two_sqrt_a);

                    let b_expr = arena.num_ratio(b_coeff.clone());
                    let c_expr = arena.num_ratio(c_coeff.clone());
                    let b_sq = arena.mul(&[b_expr, b_expr]);
                    let four = arena.int(4);
                    let four_a = arena.mul(&[four, a_expr]);
                    let b_sq_over_4a = arena.div(b_sq, four_a);
                    let exp_arg = arena.sub(c_expr, b_sq_over_4a);
                    let exp_arg = arena.eval_expr(exp_arg);
                    let exp_factor = if arena.is_zero_structural(exp_arg) {
                        arena.int(1)
                    } else {
                        arena.exp(exp_arg)
                    };

                    let two_a = {
                        let two_r =
                            num_rational::Ratio::<num_bigint::BigInt>::from_integer(2.into());
                        let val = two_r * a_coeff;
                        arena.num_ratio(val.clone())
                    };
                    let two_ax = arena.mul(&[two_a, var]);
                    let numer = arena.add(&[two_ax, b_expr]);
                    let sqrt_a2 = arena.sqrt(a_expr);
                    let denom = arena.mul(&[two, sqrt_a2]);
                    let erfi_arg = arena.div(numer, denom);
                    let erfi_term = special_apply(arena, crate::base::libfn::LibFn::Erfi, erfi_arg);

                    return arena.mul(&[front, exp_factor, erfi_term]);
                }
            }

            // Try u-substitution: if inner = a*x + b, ∫ exp(a*x+b) dx = exp(a*x+b)/a
            if let Some((a_expr, _)) = symbolic_linear_coeff_of(arena, inner, var, var_sym) {
                let exp_inner = arena.exp(inner);
                return arena.div(exp_inner, a_expr);
            }
            // ── exp(a·x² + b·x + c) with irrational or symbolic coefficients ──
            if let Some(result) = gaussian_integral(arena, inner, var, var_sym) {
                return result;
            }
            // ── exp(c·xᵖ + d): an incomplete gamma function ──
            if let Some(result) = try_incomplete_gamma(arena, &[expr], var, var_sym) {
                return result;
            }
            arena.intern(ExprNode::Integral(expr, var))
        }

        // ── Ln: ∫ ln(x) dx = x·ln(x) - x ─────────────────────────
        ExprNode::Ln(inner) => {
            if inner == var {
                // ∫ ln(x) dx = x·ln(x) - x
                let ln_var = arena.ln(var);
                let x_ln_x = arena.mul(&[var, ln_var]);
                return arena.sub(x_ln_x, var);
            }
            // ∫ ln(ln(x)) dx = x·ln(ln(x)) − li(x)  (by parts:
            // u = ln(ln(x)), dv = dx  →  du = 1/(x·ln(x)) dx, v = x)
            if let ExprNode::Ln(ln_inner) = arena.node(inner).clone()
                && ln_inner == var
            {
                let ln_x = arena.ln(var);
                let ln_ln_x = arena.ln(ln_x);
                let x_ln_ln_x = arena.mul(&[var, ln_ln_x]);
                let li_x = arena.li(var);
                return arena.sub(x_ln_ln_x, li_x);
            }
            // Try u-substitution: if inner = a*x + b (linear),
            // ∫ ln(a*x+b) dx = ((a*x+b)·ln(a*x+b) - (a*x+b)) / a
            if let Some((a_expr, _)) = symbolic_linear_coeff_of(arena, inner, var, var_sym) {
                let ln_inner = arena.ln(inner);
                let inner_times_ln = arena.mul(&[inner, ln_inner]);
                let diff = arena.sub(inner_times_ln, inner);
                return arena.div(diff, a_expr);
            }
            arena.intern(ExprNode::Integral(expr, var))
        }

        ExprNode::Sinh(inner) => {
            if inner == var {
                // ∫ sinh(x) dx = cosh(x)
                return arena.intern(ExprNode::Cosh(var));
            }
            // u-sub: ∫ sinh(ax+b) dx = cosh(ax+b)/a
            if let Some((a_expr, _)) = symbolic_linear_coeff_of(arena, inner, var, var_sym) {
                let cosh_inner = arena.cosh(inner);
                return arena.div(cosh_inner, a_expr);
            }
            arena.intern(ExprNode::Integral(expr, var))
        }

        ExprNode::Cosh(inner) => {
            if inner == var {
                // ∫ cosh(x) dx = sinh(x)
                return arena.intern(ExprNode::Sinh(var));
            }
            // u-sub: ∫ cosh(ax+b) dx = sinh(ax+b)/a
            if let Some((a_expr, _)) = symbolic_linear_coeff_of(arena, inner, var, var_sym) {
                let sinh_inner = arena.sinh(inner);
                return arena.div(sinh_inner, a_expr);
            }
            arena.intern(ExprNode::Integral(expr, var))
        }

        ExprNode::Tanh(inner) => {
            if inner == var {
                // ∫ tanh(x) dx = ln(cosh(x))
                let cosh_x = arena.cosh(var);
                return arena.ln(cosh_x);
            }
            // u-sub: if inner = a*x + b, ∫ tanh(a*x+b) dx = ln(cosh(a*x+b))/a
            if let Some((a_expr, _)) = symbolic_linear_coeff_of(arena, inner, var, var_sym) {
                let cosh_inner = arena.cosh(inner);
                let ln_cosh = arena.ln(cosh_inner);
                return arena.div(ln_cosh, a_expr);
            }
            arena.intern(ExprNode::Integral(expr, var))
        }

        // ── asin/acos/atan of a linear argument g = ax+b ──────────────
        //   ∫ asin(g) = (g·asin(g) + √(1−g²))/a
        //   ∫ acos(g) = (g·acos(g) − √(1−g²))/a
        //   ∫ atan(g) = (g·atan(g) − ½ ln(1+g²))/a
        ExprNode::Asin(inner) | ExprNode::Acos(inner) | ExprNode::Atan(inner) => {
            let Some((a_expr, _)) = symbolic_linear_coeff_of(arena, inner, var, var_sym) else {
                return by_parts_dv_one(arena, expr, var, var_sym, depth)
                    .unwrap_or_else(|| arena.intern(ExprNode::Integral(expr, var)));
            };
            let g = inner;
            let g_f = arena.mul(&[g, expr]);
            let one = arena.one;
            let two = arena.int(2);
            let g2 = arena.pow(g, two);
            let half = arena.rational(1, 2);
            let numer = match node {
                ExprNode::Asin(_) => {
                    let one_minus_g2 = arena.sub(one, g2);
                    let sqrt_term = arena.pow(one_minus_g2, half);
                    arena.add(&[g_f, sqrt_term])
                }
                ExprNode::Acos(_) => {
                    let one_minus_g2 = arena.sub(one, g2);
                    let sqrt_term = arena.pow(one_minus_g2, half);
                    arena.sub(g_f, sqrt_term)
                }
                _ => {
                    let one_plus_g2 = arena.add(&[one, g2]);
                    let ln_term = arena.ln(one_plus_g2);
                    let half_ln = arena.mul(&[half, ln_term]);
                    arena.sub(g_f, half_ln)
                }
            };
            if a_expr == one {
                return numer;
            }
            arena.div(numer, a_expr)
        }

        // ── erf / erfc of a linear argument g = ax+b ───────────────────
        //   ∫ erf(g)  = (g·erf(g)  + e^{−g²}/√π)/a
        //   ∫ erfc(g) = (g·erfc(g) − e^{−g²}/√π)/a
        ExprNode::Erf(inner) | ExprNode::Erfc(inner) => {
            let Some((a_expr, _)) = symbolic_linear_coeff_of(arena, inner, var, var_sym) else {
                return by_parts_dv_one(arena, expr, var, var_sym, depth)
                    .unwrap_or_else(|| arena.intern(ExprNode::Integral(expr, var)));
            };
            let g = inner;
            let g_f = arena.mul(&[g, expr]);
            let two = arena.int(2);
            let g2 = arena.pow(g, two);
            let neg_g2 = arena.neg(g2);
            let e = arena.exp(neg_g2);
            let pi = arena.pi();
            let sqrt_pi = arena.sqrt(pi);
            let gauss = arena.div(e, sqrt_pi);
            let numer = if matches!(node, ExprNode::Erf(_)) {
                arena.add(&[g_f, gauss])
            } else {
                arena.sub(g_f, gauss)
            };
            if a_expr == arena.one {
                return numer;
            }
            arena.div(numer, a_expr)
        }

        // ── erfi of a linear argument g = ax+b ──────────────────────────
        //   ∫ erfi(g) = (g·erfi(g) − e^{g²}/√π)/a  (unevaluated up to 0.43)
        ExprNode::Lib(crate::base::libfn::LibFn::Erfi, ref args) if args.len() == 1 => {
            let g = args[0];
            let Some((a_expr, _)) = symbolic_linear_coeff_of(arena, g, var, var_sym) else {
                return by_parts_dv_one(arena, expr, var, var_sym, depth)
                    .unwrap_or_else(|| arena.intern(ExprNode::Integral(expr, var)));
            };
            let g_f = arena.mul(&[g, expr]);
            let two = arena.int(2);
            let g2 = arena.pow(g, two);
            let e = arena.exp(g2);
            let pi = arena.pi();
            let sqrt_pi = arena.sqrt(pi);
            let gauss = arena.div(e, sqrt_pi);
            let numer = arena.sub(g_f, gauss);
            if a_expr == arena.one {
                return numer;
            }
            arena.div(numer, a_expr)
        }

        // ── |g|, sign(g), Piecewise ───────────────────────────────────
        ExprNode::Abs(_) | ExprNode::Sign(_) => {
            if let Some(r) = try_abs_sign_product(arena, &[expr], var, var_sym, depth) {
                return r;
            }
            arena.intern(ExprNode::Integral(expr, var))
        }
        ExprNode::Piecewise(ref pairs) => {
            if let Some(r) = integrate_piecewise(arena, pairs, var, var_sym, depth) {
                return r;
            }
            arena.intern(ExprNode::Integral(expr, var))
        }

        // ── DiracDelta: ∫δ(f(x))dx ────────────────────────────────
        ExprNode::DiracDelta(inner) => {
            // ∫δ(x)dx = H(x)
            if inner == var {
                return arena.intern(ExprNode::Heaviside(var));
            }
            // Linear case: ∫δ(ax+b)dx = H(ax+b) / |a|
            if let Some((a_expr, _)) = symbolic_linear_coeff_of(arena, inner, var, var_sym) {
                let h = arena.intern(ExprNode::Heaviside(inner));
                let abs_a = arena.abs(a_expr);
                return arena.div(h, abs_a);
            }
            let one = arena.one;
            delta_simple_roots(arena, one, inner, var)
                .unwrap_or_else(|| arena.intern(ExprNode::Integral(expr, var)))
        }

        // ── Heaviside: ∫ H(g(x)) dx ───────────────────────────────
        ExprNode::Heaviside(inner) => {
            // ∫ H(x) dx = x·H(x)
            if inner == var {
                return arena.mul(&[var, expr]);
            }
            // Linear case: ∫ H(ax+b) dx = (ax+b)·H(ax+b) / a
            if let Some((a_expr, _b_expr)) = symbolic_linear_coeff_of(arena, inner, var, var_sym) {
                let h = arena.intern(ExprNode::Heaviside(inner));
                let product = arena.mul(&[inner, h]);
                return arena.div(product, a_expr);
            }
            // Leave unevaluated
            arena.intern(ExprNode::Integral(expr, var))
        }

        // ── Asinh: ∫ asinh(x) dx = x·asinh(x) - √(x²+1) ─────────
        ExprNode::Asinh(inner) => {
            if inner == var {
                tracing::debug!("integrate: matched asinh(x) direct");
                let asinh_var = arena.asinh(var);
                let x_asinh = arena.mul(&[var, asinh_var]);
                let two = arena.int(2);
                let x2 = arena.pow(var, two);
                let one = arena.one;
                let x2_plus_1 = arena.add(&[x2, one]);
                let half = arena.rational(1, 2);
                let sqrt_term = arena.pow(x2_plus_1, half);
                return arena.sub(x_asinh, sqrt_term);
            }
            // Linear chain rule: ∫ asinh(ax+b) dx = (ax+b)·asinh(ax+b)/a - √((ax+b)²+1)/a
            if let Some((a_expr, _b_expr)) = symbolic_linear_coeff_of(arena, inner, var, var_sym) {
                tracing::debug!("integrate: matched asinh(ax+b) linear");
                let asinh_g = arena.asinh(inner);
                let g_asinh = arena.mul(&[inner, asinh_g]);
                let two = arena.int(2);
                let g2 = arena.pow(inner, two);
                let one = arena.one;
                let g2_plus_1 = arena.add(&[g2, one]);
                let half = arena.rational(1, 2);
                let sqrt_term = arena.pow(g2_plus_1, half);
                let numer = arena.sub(g_asinh, sqrt_term);
                return arena.div(numer, a_expr);
            }
            by_parts_dv_one(arena, expr, var, var_sym, depth)
                .unwrap_or_else(|| arena.intern(ExprNode::Integral(expr, var)))
        }

        // ── Acosh: ∫ acosh(g) dx = (g·acosh(g) − √(g−1)·√(g+1))/a ──
        // for g = a·x + b.  The product of roots is the principal-branch
        // companion of acosh (its derivative is g/(√(g−1)√(g+1))); the
        // single root √(g²−1) of 0.23 had the wrong sign for g < −1.
        ExprNode::Acosh(inner) => {
            let a_expr = if inner == var {
                Some(arena.one)
            } else {
                symbolic_linear_coeff_of(arena, inner, var, var_sym).map(|(a, _)| a)
            };
            if let Some(a_expr) = a_expr {
                tracing::debug!("integrate: matched acosh(ax+b)");
                let acosh_g = arena.acosh(inner);
                let g_acosh = arena.mul(&[inner, acosh_g]);
                let one = arena.one;
                let g_minus_1 = arena.sub(inner, one);
                let g_plus_1 = arena.add(&[inner, one]);
                let half = arena.rational(1, 2);
                let root_minus = arena.pow(g_minus_1, half);
                let root_plus = arena.pow(g_plus_1, half);
                let roots = arena.mul(&[root_minus, root_plus]);
                let numer = arena.sub(g_acosh, roots);
                return if a_expr == arena.one {
                    numer
                } else {
                    arena.div(numer, a_expr)
                };
            }
            by_parts_dv_one(arena, expr, var, var_sym, depth)
                .unwrap_or_else(|| arena.intern(ExprNode::Integral(expr, var)))
        }

        // ── Atanh: ∫ atanh(x) dx = x·atanh(x) + ½·ln(1-x²) ──────
        ExprNode::Atanh(inner) => {
            if inner == var {
                tracing::debug!("integrate: matched atanh(x) direct");
                let atanh_var = arena.atanh(var);
                let x_atanh = arena.mul(&[var, atanh_var]);
                let two = arena.int(2);
                let x2 = arena.pow(var, two);
                let one = arena.one;
                let one_minus_x2 = arena.sub(one, x2);
                let half = arena.rational(1, 2);
                let ln_term = arena.ln(one_minus_x2);
                let half_ln = arena.mul(&[half, ln_term]);
                return arena.add(&[x_atanh, half_ln]);
            }
            // Linear chain rule: ∫ atanh(ax+b) dx
            if let Some((a_expr, _b_expr)) = symbolic_linear_coeff_of(arena, inner, var, var_sym) {
                tracing::debug!("integrate: matched atanh(ax+b) linear");
                let atanh_g = arena.atanh(inner);
                let g_atanh = arena.mul(&[inner, atanh_g]);
                let two = arena.int(2);
                let g2 = arena.pow(inner, two);
                let one = arena.one;
                let one_minus_g2 = arena.sub(one, g2);
                let half = arena.rational(1, 2);
                let ln_term = arena.ln(one_minus_g2);
                let half_ln = arena.mul(&[half, ln_term]);
                let numer = arena.add(&[g_atanh, half_ln]);
                return arena.div(numer, a_expr);
            }
            by_parts_dv_one(arena, expr, var, var_sym, depth)
                .unwrap_or_else(|| arena.intern(ExprNode::Integral(expr, var)))
        }

        // ── expint(ν, a·x + b): ∫ = −expint(ν + 1, a·x + b)/a (DLMF 8.19.13,
        // E′_ν = −E_{ν−1}); unevaluated up to 0.46, as was every ∫ Ei, Si,
        // Ci, li, Shi, Chi, Fresnel S/C below (by parts with dv = dx).
        ExprNode::Lib(crate::base::libfn::LibFn::ExpInt, ref args)
            if args.len() == 2
                && !contains_var(arena, args[0], var_sym)
                && symbolic_linear_coeff_of(arena, args[1], var, var_sym).is_some() =>
        {
            let (nu, g) = (args[0], args[1]);
            let Some((a_expr, _)) = symbolic_linear_coeff_of(arena, g, var, var_sym) else {
                return arena.intern(ExprNode::Integral(expr, var));
            };
            let one = arena.one;
            let nu1 = arena.add(&[nu, one]);
            let e = arena.lib_apply(crate::base::libfn::LibFn::ExpInt, &[nu1, g]);
            let minus_e = arena.neg(e);
            arena.div(minus_e, a_expr)
        }
        _ if is_special_by_parts_fn(arena, expr, var_sym) => {
            by_parts_dv_one(arena, expr, var, var_sym, depth)
                .unwrap_or_else(|| arena.intern(ExprNode::Integral(expr, var)))
        }
        ExprNode::Apply(..) | ExprNode::Lib(..) => {
            orthogonal_polynomial_integral(arena, expr, var, var_sym)
                .or_else(|| bessel_power_integral(arena, &[expr], var, var_sym))
                .unwrap_or_else(|| arena.intern(ExprNode::Integral(expr, var)))
        }
        // ── W(a·x + b): ∫ = (g·W(g) − g + g/W(g))/a (∫ f⁻¹ = x·f⁻¹ − F∘f⁻¹ for
        // f(w) = w·eʷ, F(w) = (w − 1)·eʷ, and e^{W(g)} = g/W(g)); unevaluated up
        // to 0.46 (SymPy's test_heurisch and test_integrals).
        ExprNode::LambertW(g) => {
            let Some((a_expr, _)) = symbolic_linear_coeff_of(arena, g, var, var_sym) else {
                return arena.intern(ExprNode::Integral(expr, var));
            };
            let g_w = arena.mul(&[g, expr]);
            let g_over_w = arena.div(g, expr);
            let num = arena.sub(g_w, g);
            let num = arena.add(&[num, g_over_w]);
            arena.div(num, a_expr)
        }

        // Everything else: unevaluated integral.
        _ => {
            tracing::debug!("integration: no strategy succeeded, returning unevaluated");
            arena.intern(ExprNode::Integral(expr, var))
        }
    }
}

/// Does `expr` depend on the symbol `var` — does `var` occur *free* in it?
///
/// [`crate::base::walk::has_free_symbol`], which reads the binder table
/// shared with `free_symbols` and `subs`: the variable of a `Sum` is bound
/// in its body (not in its limits), a `RootOf` binds its polynomial's
/// variable, and so on.  So `RootOf(x⁵ − x + 1, 0)` — the
/// form in which `solve` returns a root of a polynomial in `x` — is a
/// constant.  Up to 0.28.0 the test was structural, and
/// `1/(x − RootOf(x⁵ − x + 1, 0))` did not look like `1/(x − c)`.
/// An indefinite `Integral` or a `Derivative` does not bind: `∫ f dx` is a
/// function of `x`.
fn contains_var(arena: &Arena, expr: ExprId, var: SymbolId) -> bool {
    crate::base::walk::has_free_symbol(arena, expr, var)
}

/// Check if an expression is a polynomial in the given variable.
/// A polynomial is: the variable itself, a power of the variable with a
/// non-negative integer exponent, a numeric constant, or sums/products of these.
fn is_polynomial_in(arena: &Arena, expr: ExprId, var: ExprId, var_sym: SymbolId) -> bool {
    let mut stack = vec![expr];
    while let Some(e) = stack.pop() {
        if e == var || !contains_var(arena, e, var_sym) {
            continue;
        }
        match arena.node(e) {
            // `xⁿ`, and a polynomial to a small power: `(2x + 1)²` was not a
            // polynomial, so by parts never took it for `u` and `∫ (2x +
            // 1)²·sin x dx` stayed unevaluated (Rubi 4.x.10 after `u = a + b·x`).
            ExprNode::Pow(base, exp) => match arena.as_num(*exp) {
                Some(r)
                    if r.is_integer()
                        && !r.is_negative()
                        && (*base == var || *r <= Q::from_integer(MAX_POLYNOMIAL_POWER.into())) =>
                {
                    stack.push(*base);
                }
                _ => return false,
            },
            ExprNode::Mul(children) | ExprNode::Add(children) => {
                stack.extend(children.iter().copied())
            }
            ExprNode::Neg(inner) => stack.push(*inner),
            ExprNode::Num(_) => {}
            _ => return false,
        }
    }
    true
}

/// Largest power of a polynomial other than `x` itself that
/// [`is_polynomial_in`] counts as a polynomial (by parts differentiates it
/// that many times).
const MAX_POLYNOMIAL_POWER: i64 = 16;

/// Check if `expr` is a linear function of `var` with **numeric** coefficients.
/// Returns `Some(a)` (the leading coefficient as `Ratio<BigInt>`) if linear, `None` otherwise.
#[allow(dead_code)]
fn linear_coeff_of(arena: &Arena, expr: ExprId, _var: ExprId, _var_sym: SymbolId) -> Option<Q> {
    // Try to convert to polynomial in var.
    let poly = crate::poly::polybridge::expr_to_poly(arena, expr, _var)?;
    // Must be degree exactly 1.
    if poly.degree()? != 1 {
        return None;
    }
    let a = poly.coeff(1);
    if a.is_zero() {
        return None;
    }
    Some(a)
}

/// Check if `expr` is a linear function of `var`: `a*var + b` where a ≠ 0,
/// with **symbolic** (possibly non-numeric) coefficients.
///
/// Returns `Some((a_expr, b_expr))` where `a_expr` is the coefficient of `var`
/// and `b_expr` is the constant term — both as `ExprId`s that are free of `var`.
fn symbolic_linear_coeff_of(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<(ExprId, ExprId)> {
    // Fast path: try numeric first (covers the common numeric-coefficient case)
    if let Some(a) = linear_coeff_of(arena, expr, var, var_sym) {
        let a_id = arena.num_ratio(a.clone());
        // Also extract constant term
        if let Some(poly) = crate::poly::polybridge::expr_to_poly(arena, expr, var) {
            let b = poly.coeff(0);
            let b_id = arena.num_ratio(b.clone());
            return Some((a_id, b_id));
        }
        return Some((a_id, arena.zero));
    }

    // Case 1: expr == var → coefficient is 1, constant is 0
    if expr == var {
        return Some((arena.one, arena.zero));
    }

    // Case 2: Neg(inner) → negate coefficient and constant
    if let ExprNode::Neg(inner) = arena.node(expr).clone() {
        if let Some((coeff, constant)) = symbolic_linear_coeff_of(arena, inner, var, var_sym) {
            let neg_coeff = arena.neg(coeff);
            let neg_const = arena.neg(constant);
            return Some((neg_coeff, neg_const));
        }
        return None;
    }

    // Case 3: Mul containing var exactly once, all other factors free of var
    if let ExprNode::Mul(ref children) = arena.node(expr).clone() {
        let mut has_var = false;
        let mut other_factors: SmallVec<[ExprId; 4]> = SmallVec::new();
        let mut var_count = 0u32;
        let mut nontrivial = false;

        for &child in children {
            if child == var {
                var_count += 1;
                if var_count > 1 {
                    return None; // var² or higher
                }
                has_var = true;
            } else if contains_var(arena, child, var_sym) {
                nontrivial = true; // Non-trivial var dependence (see 3b)
            } else {
                other_factors.push(child);
            }
        }

        if has_var && var_count == 1 && !nontrivial {
            let coeff = match other_factors.len() {
                0 => arena.one,
                1 => other_factors[0],
                _ => arena.mul(&other_factors),
            };
            return Some((coeff, arena.zero));
        }

        // Case 3b: c·(a·x + b) — constant factors times one linear sum.
        // `∫ a·a^x = ∫ a^(x+1)` is rewritten `∫ e^{(x+1)·ln a}`, whose
        // exponent is such a product (before 0.30 it stayed unevaluated
        // while `∫ a^x` worked).
        let dependent: SmallVec<[ExprId; 2]> = children
            .iter()
            .copied()
            .filter(|&c| contains_var(arena, c, var_sym))
            .collect();
        if let [sum] = dependent[..]
            && matches!(arena.node(sum), ExprNode::Add(_))
            && let Some((a, b)) = linear_sum_parts(arena, sum, var, var_sym)
        {
            let c: SmallVec<[ExprId; 4]> = children.iter().copied().filter(|&f| f != sum).collect();
            let c = arena.mul(&c);
            let a = arena.mul(&[c, a]);
            let b = arena.mul(&[c, b]);
            return Some((a, b));
        }
    }

    // Case 4: Add → separate var-containing and var-free terms
    if matches!(arena.node(expr), ExprNode::Add(_)) {
        return linear_sum_parts(arena, expr, var, var_sym);
    }

    None
}

/// `(a, b)` for a sum `expr = a·var + b` with `a`, `b` free of `var` (the
/// sum case of [`symbolic_linear_coeff_of`]).
///
/// Every term with the variable must be `cᵢ·var` (or `var`); `a = Σ cᵢ`.
/// The canonical sum keeps `−t·√37/2 − t/2` as two terms (only rational
/// multiples of one term merge), so up to 0.31 `∫ e^{−t·√37/2 − t/2} dt`
/// stayed unevaluated while `∫ e^{−t·(√37 + 1)/2} dt` worked.
fn linear_sum_parts(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<(ExprId, ExprId)> {
    {
        let ExprNode::Add(ref children) = arena.node(expr).clone() else {
            return None;
        };
        let mut var_terms: SmallVec<[ExprId; 4]> = SmallVec::new();
        let mut const_terms: SmallVec<[ExprId; 4]> = SmallVec::new();

        for &child in children {
            if contains_var(arena, child, var_sym) {
                var_terms.push(child);
            } else {
                const_terms.push(child);
            }
        }

        if var_terms.is_empty() {
            return None; // No var dependence — not linear in var
        }

        // Each var-containing term must be c*var; the coefficient is the
        // sum of the c's.
        let mut coeffs: SmallVec<[ExprId; 4]> = SmallVec::new();
        for &term in &var_terms {
            if term == var {
                coeffs.push(arena.one);
                continue;
            }
            let ExprNode::Mul(ref mul_children) = arena.node(term).clone() else {
                return None; // Can't decompose
            };
            let mut vc = 0u32;
            let mut other: SmallVec<[ExprId; 4]> = SmallVec::new();
            for &mc in mul_children {
                if mc == var {
                    vc += 1;
                } else if contains_var(arena, mc, var_sym) {
                    return None;
                } else {
                    other.push(mc);
                }
            }
            if vc != 1 {
                return None;
            }
            coeffs.push(match other.len() {
                0 => arena.one,
                1 => other[0],
                _ => arena.mul(&other),
            });
        }
        let coeff = match coeffs.len() {
            1 => coeffs[0],
            _ => arena.add(&coeffs),
        };
        if coeff == arena.zero {
            return None;
        }

        // Verify coefficient is free of var
        if contains_var(arena, coeff, var_sym) {
            return None;
        }

        let constant = match const_terms.len() {
            0 => arena.zero,
            1 => const_terms[0],
            _ => arena.add(&const_terms),
        };

        // Verify constant is free of var
        if contains_var(arena, constant, var_sym) {
            return None;
        }

        Some((coeff, constant))
    }
}

/// Extract symbolic quadratic coefficients from `cx² + dx + e`.
///
/// Given an expression that is quadratic in `var`, returns
/// `Some((c_expr, d_expr, e_expr))` where `c` is the coefficient of `var²`,
/// `d` is the coefficient of `var`, and `e` is the constant term — all as
/// `ExprId`s that are free of `var`.
///
/// Returns `None` if the expression is not quadratic in `var` (e.g., it
/// contains `var³` or non-polynomial dependence on `var`).
///
/// This is the quadratic analogue of [`symbolic_linear_coeff_of`], used when
/// [`expr_to_poly`](crate::poly::polybridge::expr_to_poly) fails because coefficients are irrational (e.g., `√5`).
fn symbolic_quadratic_coeffs(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<(ExprId, ExprId, ExprId)> {
    symbolic_quadratic_coeffs_of_terms(arena, expr, var, var_sym)
}

/// [`symbolic_quadratic_coeffs`] of the argument `q` of an `exp`, `sin`, `cos`,
/// also after multiplying out a product with a power of a sum
/// (`−(x − μ)²/(2σ²)`, the exponent of a normal density): up to 0.46
/// `∫ x²·e^{−(x−μ)²/(2σ²)} dx` stayed unevaluated for symbolic `μ`, `σ`
/// (SymPy's test_meijerint).  (For denominators and radicands the terms as
/// written: there a multiplied-out quadratic sent failing searches down long
/// chains of completing the square.)
fn quadratic_argument_coeffs(
    arena: &mut Arena,
    q: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<(ExprId, ExprId, ExprId)> {
    if let Some(c) = symbolic_quadratic_coeffs_of_terms(arena, q, var, var_sym) {
        return Some(c);
    }
    if !contains_var(arena, q, var_sym) {
        return None;
    }
    let expanded = crate::transforms::expand::expand(arena, q);
    if expanded == q {
        return None;
    }
    symbolic_quadratic_coeffs_of_terms(arena, expanded, var, var_sym)
}

/// [`symbolic_quadratic_coeffs`] on the terms of `expr` as written.
fn symbolic_quadratic_coeffs_of_terms(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<(ExprId, ExprId, ExprId)> {
    // Fast path: try numeric first.
    tracing::trace!("symbolic_quadratic_coeffs: attempting to extract quadratic coefficients");
    if let Some(poly) = crate::poly::polybridge::expr_to_poly(arena, expr, var) {
        if poly.degree()? == 2 {
            let c = arena.num_ratio(poly.coeff(2).clone());
            let d = arena.num_ratio(poly.coeff(1).clone());
            let e = arena.num_ratio(poly.coeff(0).clone());
            return Some((c, d, e));
        }
        return None;
    }

    // Symbolic path: walk the Add children and classify each term.
    tracing::trace!("symbolic_quadratic_coeffs: rational path failed, trying symbolic extraction");
    let children = match arena.node(expr).clone() {
        ExprNode::Add(c) => c.to_vec(),
        _ => vec![expr],
    };

    let mut x2_terms: SmallVec<[ExprId; 4]> = SmallVec::new(); // coefficients of var²
    let mut x1_terms: SmallVec<[ExprId; 4]> = SmallVec::new(); // coefficients of var
    let mut x0_terms: SmallVec<[ExprId; 4]> = SmallVec::new(); // constant terms

    let two = arena.int(2);

    for &child in &children {
        if !contains_var(arena, child, var_sym) {
            // Constant term.
            x0_terms.push(child);
            continue;
        }

        // Check for var² or scalar * var²
        if child == arena.pow(var, two) {
            x2_terms.push(arena.one);
            continue;
        }

        // Check for Pow(var, 2)
        if let ExprNode::Pow(base, exp) = arena.node(child).clone()
            && base == var
        {
            if let Some(e) = arena.as_num(exp)
                && *e == num_rational::Ratio::from_integer(2.into())
            {
                x2_terms.push(arena.one);
                continue;
            }
            // var^(something else) — not quadratic
            return None;
        }

        // Check for Mul containing var² or var
        if let ExprNode::Mul(ref mul_children) = arena.node(child).clone() {
            let mul_children = mul_children.clone();
            let mut has_var_sq = false;
            let mut var_count = 0u32;
            let mut other_factors: SmallVec<[ExprId; 4]> = SmallVec::new();

            for &mc in &mul_children {
                if mc == var {
                    var_count += 1;
                    if var_count > 2 {
                        return None; // var³ or higher
                    }
                } else if let ExprNode::Pow(base, exp) = arena.node(mc).clone() {
                    if base == var {
                        {
                            let e = arena.as_num(exp)?;
                            if *e == num_rational::Ratio::from_integer(2.into()) {
                                has_var_sq = true;
                            } else if e.is_integer()
                                && *e > num_rational::Ratio::from_integer(2.into())
                            {
                                return None; // var³ or higher
                            } else {
                                // fractional power of var — not polynomial
                                return None;
                            }
                        }
                    } else if contains_var(arena, mc, var_sym) {
                        return None; // non-trivial var dependence
                    } else {
                        other_factors.push(mc);
                    }
                } else if contains_var(arena, mc, var_sym) {
                    return None; // non-trivial var dependence
                } else {
                    other_factors.push(mc);
                }
            }

            let scalar = match other_factors.len() {
                0 => arena.one,
                1 => other_factors[0],
                _ => arena.mul(&other_factors),
            };

            if has_var_sq || var_count == 2 {
                x2_terms.push(scalar);
            } else if var_count == 1 {
                x1_terms.push(scalar);
            } else {
                // No var at all in this Mul child — should have been caught
                // by the contains_var check above, but be safe.
                x0_terms.push(child);
            }
            continue;
        }

        // Check for bare var
        if child == var {
            x1_terms.push(arena.one);
            continue;
        }

        // Check for Neg(something)
        if let ExprNode::Neg(inner) = arena.node(child).clone() {
            if let Some((c, d, e)) = symbolic_quadratic_coeffs(arena, inner, var, var_sym) {
                x2_terms.push(arena.neg(c));
                x1_terms.push(arena.neg(d));
                x0_terms.push(arena.neg(e));
                continue;
            }
            return None;
        }

        // Unrecognized var-dependent term.
        return None;
    }

    // The expression must have a nonzero x² coefficient to be quadratic.
    if x2_terms.is_empty() {
        tracing::trace!("symbolic_quadratic_coeffs: no x² terms found, not quadratic");
        return None;
    }

    tracing::trace!(
        n_x2_terms = x2_terms.len(),
        n_x1_terms = x1_terms.len(),
        n_x0_terms = x0_terms.len(),
        "symbolic_quadratic_coeffs: classified terms"
    );

    let c_expr = match x2_terms.len() {
        1 => x2_terms[0],
        _ => arena.add(&x2_terms),
    };
    let d_expr = match x1_terms.len() {
        0 => arena.zero,
        1 => x1_terms[0],
        _ => arena.add(&x1_terms),
    };
    let e_expr = match x0_terms.len() {
        0 => arena.zero,
        1 => x0_terms[0],
        _ => arena.add(&x0_terms),
    };

    // Verify c is free of var.
    if contains_var(arena, c_expr, var_sym) {
        return None;
    }

    Some((c_expr, d_expr, e_expr))
}

// ═══════════════════════════════════════════════════════════════════════════
// General u-substitution
// ═══════════════════════════════════════════════════════════════════════════

/// Try general u-substitution on a product integrand.
///
/// For each dependent factor `g(u)` where `u = u(x)`, this checks
/// whether the remaining factors equal `du/dx` times a constant `c`.
/// When they do, `∫ c · (du/dx) · g(u) dx = c · G(u)` where `G` is
/// the antiderivative of `g` with respect to `u`.
///
/// The technique works by:
/// 1. Extracting candidate inner arguments from function / power nodes.
/// 2. Computing `du/dx` via [`crate::transforms::diff::diff`].
/// 3. Forming `remaining / du` and checking it is free of `var`.
/// 4. Substituting `u → var` in the factor, integrating, then
///    substituting back.
fn try_u_substitution(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    for (i, &factor) in dependent.iter().enumerate() {
        let candidates = u_sub_candidates(arena, factor, var_sym);

        for u_expr in candidates {
            // Skip the trivial u = var case (already handled elsewhere)
            if u_expr == var {
                continue;
            }

            // Compute du/dx
            let du = crate::transforms::diff::diff(arena, u_expr, var);
            if du == arena.zero {
                continue;
            }

            // Product of the remaining dependent factors
            let remaining_expr = remaining_product(arena, dependent, i);

            // quotient = remaining / du — if free of var, we have our constant
            let quotient = arena.div(remaining_expr, du);

            // Try the raw quotient first; fall back to polynomial cancellation,
            // then to trigonometric simplification (e.g. sec²/(1 + tan²) = 1).
            // A quotient that depends on `var` only through `u` joins the
            // factor: `∫ g(u)·h(u)·u′ dx` (up to 0.31 only a constant
            // quotient counted, and `∫ cos³x·(a + b·cos²x)³·sin x dx` came
            // out of the Weierstrass substitution with `tan(x/2)⁴²`).
            let placeholder = arena.symbol("__usub_u");
            // `u = c·xᵏ + d`: every `xʲ` with `j/k` an integer is
            // `((u − d)/c)^{j/k}` (exact), so `x = 1/u` for `u = 1/x`.  `∫ e^{1/x}/x dx`
            // is `−∫ eᵘ/u du`, `∫ e^{x²}/x dx` is `∫ eᵘ/(2u) du` (unevaluated up to
            // 0.43; SymPy: `−Ei(1/x)`, `Ei(x²)/2`).  Only for a `c` known to be
            // non-zero: `∫ e^{a/x}/x dx` stays unevaluated for an unassumed `a`
            // (`−Ei(a/x)` is undefined at `a = 0`).
            let monomial_u = affine_monomial_parts(arena, u_expr, var, var_sym)
                .filter(|&(c, _, _)| constant_nonzero(arena, c));
            let inverse = monomial_u.is_some();
            let to_u = |arena: &mut Arena, e: ExprId| -> ExprId {
                let mut s = arena.subs_structural(e, u_expr, placeholder);
                let Some((c, k, d)) = monomial_u.as_ref() else {
                    return s;
                };
                if !contains_var(arena, s, var_sym) {
                    return s;
                }
                let u_minus_d = arena.sub(placeholder, *d);
                let u_over_c = arena.div(u_minus_d, *c);
                for id in crate::base::walk::post_order_ids(arena, s) {
                    if let ExprNode::Pow(b, j) = *arena.node(id)
                        && b == var
                        && let Some(r) = arena.as_num(j).map(|j| j / k)
                        && r.is_integer()
                    {
                        let r = arena.num_ratio(r);
                        let u_r = arena.pow(u_over_c, r);
                        s = arena.subs_structural(s, id, u_r);
                    }
                }
                let r = Q::one() / k;
                if r.is_integer() {
                    let r = arena.num_ratio(r);
                    let u_r = arena.pow(u_over_c, r);
                    s = arena.subs_structural(s, var, u_r);
                }
                s
            };
            let mut through_u: Option<ExprId> = None;
            let coeff = if !contains_var(arena, quotient, var_sym) {
                quotient
            } else {
                let cancelled = arena.cancel_expr(quotient, var);
                if !contains_var(arena, cancelled, var_sym) {
                    cancelled
                } else {
                    // No rewriting of `sin`, `cos`, … can make a quotient
                    // that provably depends on `var` constant.
                    let varies = var_outside_trig_arguments(arena, cancelled, var_sym)
                        || varies_numerically(arena, cancelled, var);
                    let trig = if varies {
                        cancelled
                    } else {
                        let trig = arena.trigsimp_expr(cancelled);
                        crate::transforms::eval::eval(arena, trig)
                    };
                    if !contains_var(arena, trig, var_sym) {
                        trig
                    } else {
                        // (For a linear `u` this is a renaming that adds
                        // nothing: every rule takes linear arguments.)
                        if !contains_var(arena, du, var_sym) {
                            continue;
                        }
                        let q_in_u = to_u(arena, quotient);
                        if contains_var(arena, q_in_u, var_sym) {
                            continue;
                        }
                        through_u = Some(quotient);
                        arena.one
                    }
                }
            };
            // `du` may be a constant that is zero without being so
            // structurally, and `trigsimp` shows it: `u = k·x + 1` with
            // `k = sin²1 + cos²1 − 1` gave the coefficient `zoo`, and
            // `∫ x/u dx = zoo` once the self-check stopped passing the
            // earlier candidate with the vanishing denominator `k`.
            if contains_non_finite(arena, coeff) {
                continue;
            }

            // The factor must depend on `var` only through `u`: with `u`
            // replaced by a fresh symbol nothing of `var` may remain
            // (otherwise `g(var)` below would silently absorb it).
            let factor = match through_u {
                Some(q) => arena.mul(&[factor, q]),
                None => factor,
            };
            let factor_in_u = to_u(arena, factor);
            if contains_var(arena, factor_in_u, var_sym) {
                continue;
            }

            // Replace u(x) → var inside the factor to get g(var),
            // integrate g(var) w.r.t. var, then substitute var → u(x) back.
            let g_of_var = if inverse {
                arena.subs_structural(factor_in_u, placeholder, var)
            } else {
                arena.subs_structural(factor, u_expr, var)
            };
            let g_integrated = integrate_node(arena, g_of_var, var, var_sym, depth);

            // If the inner integral is unevaluated, this candidate didn't help
            if matches!(arena.node(g_integrated), ExprNode::Integral(_, _)) {
                continue;
            }

            // G(u) — substitute var back to u(x)
            let antideriv = arena.subs_structural(g_integrated, var, u_expr);
            let antideriv = match continuous_through_tan_of_u(
                arena,
                g_integrated,
                antideriv,
                u_expr,
                var,
                var_sym,
            ) {
                Some(continuous) => continuous,
                None if jump_kept_undecided(arena, g_integrated, var_sym) => {
                    match tan_argument_of(arena, u_expr, var_sym) {
                        Some(w) => crate::transforms::trig_integ::atan_tan_terms_to_argument_in(
                            arena,
                            antideriv,
                            w,
                            Some(var_sym),
                        ),
                        None => antideriv,
                    }
                }
                None => continue,
            };
            tracing::debug!("u-substitution succeeded");
            return Some(arena.mul(&[coeff, antideriv]));
        }
    }
    None
}

/// The back-substituted `antideriv = G(u(x))` of a u-substitution, made
/// continuous across the poles of a `tan w` through which `u` depends on
/// `x` ([`crate::transforms::trig_integ::add_tan_floor`]).  `g` is `G` in
/// `var` (the stand-in for `u`).  Up to 0.37 `∫ tan x/(tan x + 2) dx` was
/// `… + 1/5·atan(tan x)`, which drops by `π/5` at every `x = π/2 + kπ`.
///
/// `antideriv` unchanged when `u` has no `tan` of `var`; `None` (the
/// candidate is not used) when `u` depends on `var` otherwise than through
/// one `tan w`, or the jump `G(U(+∞)) − G(U(−∞))` of `G(U(s))`, `u = U(tan w)`,
/// is not decided.
fn continuous_through_tan_of_u(
    arena: &mut Arena,
    g: ExprId,
    antideriv: ExprId,
    u_expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    let Some(w) = tan_argument_of(arena, u_expr, var_sym) else {
        return Some(antideriv);
    };
    let tan_w = arena.tan(w);
    let s = arena.symbol("__tan_s");
    let ExprNode::Symbol(s_sym) = *arena.node(s) else {
        return None;
    };
    let u_of_s = arena.subs_structural(u_expr, tan_w, s);
    if contains_var(arena, u_of_s, var_sym) {
        return None;
    }
    let h = arena.subs_structural(g, var, u_of_s);
    let (antideriv, jump) = match crate::transforms::trig_integ::infinity_jump(arena, h, s, s_sym) {
        Some(jump) => (antideriv, jump),
        // Parameters not declared real: the jump as for real ones, of the
        // answer with the logarithms the stage exit writes `ln u` (up to
        // 0.37 the answer was kept uncorrected: `∫ tan x/(a + tan x) dx`
        // stepped by `iπ·a/(a² + 1)` at every pole of `tan x`).
        None => {
            let all = crate::base::walk::free_symbols(arena, w)
                .into_iter()
                .any(|v| match *arena.node(v) {
                    ExprNode::Symbol(sid) => {
                        sid != var_sym
                            && !crate::transforms::realness::symbol_declared_real(arena, sid)
                    }
                    _ => true,
                });
            let g = crate::transforms::trig_integ::plain_logs(arena, g, var_sym, all);
            let h = arena.subs_structural(g, var, u_of_s);
            let jump =
                crate::transforms::trig_integ::infinity_jump_real_parameters(arena, h, s, s_sym)?;
            (arena.subs_structural(g, var, u_expr), jump)
        }
    };
    Some(crate::transforms::trig_integ::add_tan_floor_in(
        arena,
        antideriv,
        w,
        jump,
        Some(var_sym),
    ))
}

/// The argument `w` of the first `tan w` (with `w` depending on `var`) in
/// `u` (post-order).
fn tan_argument_of(arena: &Arena, u: ExprId, var_sym: SymbolId) -> Option<ExprId> {
    crate::base::walk::post_order_ids(arena, u)
        .into_iter()
        .find_map(|id| match *arena.node(id) {
            ExprNode::Tan(w) if contains_var(arena, w, var_sym) => Some(w),
            _ => None,
        })
}

/// Does `var` occur in `q` outside the arguments of its trigonometric and
/// hyperbolic functions (`x²·sin x`, `eˣ/cos x`)?  `trigsimp` rewrites
/// those functions only, so it cannot make such a `q` free of `var`: on a
/// sample of the Rubi suite (0.31) none of 4,550 such quotients became
/// constant (the canonical form already cancels equal factors).
fn var_outside_trig_arguments(arena: &Arena, q: ExprId, var_sym: SymbolId) -> bool {
    let mut stack = vec![q];
    let mut seen: FxHashSet<ExprId> = FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Sin(_)
            | ExprNode::Cos(_)
            | ExprNode::Tan(_)
            | ExprNode::Sinh(_)
            | ExprNode::Cosh(_)
            | ExprNode::Tanh(_) => {}
            ExprNode::Symbol(s) if *s == var_sym => return true,
            node => node.for_each_child(|c| stack.push(c)),
        }
    }
    false
}

/// Sample points for [`varies_numerically`].
const VARIES_POINTS: [(i64, i64); 2] = [(3, 7), (5, 11)];

/// Does `q` provably depend on `var`: does it take clearly different
/// values (to 15 digits) at two sample points, with its other free symbols
/// at the generic values of [`FTC_PARAMETER_VALUES`]?  `false` when it
/// does not evaluate at both points or the values agree.  No rewriting
/// can then make `q` free of `var`, and [`try_u_substitution`] skips
/// `trigsimp`, which tries seven strategies on each quotient and found a
/// constant in 1 of 9,500 calls on a sample of the Rubi suite (0.31;
/// `sin 2x/(8 sin x cos x) = 1/4`), while taking 50 % of the time of the
/// integrands that stay unevaluated.
///
/// Inside an `integrate` call the answer is remembered until the outermost
/// call ends ([`VARIES_MEMO`]).
fn varies_numerically(arena: &mut Arena, q: ExprId, var: ExprId) -> bool {
    let inside = INTEGRATE_DEPTH.with(std::cell::Cell::get) > 0;
    let key = (arena.serial(), q, var);
    if inside && let Some(found) = VARIES_MEMO.with(|m| m.borrow().get(&key).copied()) {
        return found;
    }
    let varies = varies_numerically_uncached(arena, q, var);
    if inside {
        VARIES_MEMO.with(|m| m.borrow_mut().insert(key, varies));
    }
    varies
}

thread_local! {
    /// [`varies_numerically`] during the current top-level `integrate`
    /// (cleared with [`INTEGRATE_MEMO`]), by (arena, quotient, variable).
    /// The u-substitution meets the same quotients for every degenerate
    /// case of the parameters and every depth it is retried at: on the Rubi
    /// suite's 4.5.4.2 (0.38) its 45,000 tests took 6.5 s, a fifth of the
    /// integration time, and 24,700 of them were repeats.
    static VARIES_MEMO: std::cell::RefCell<FxHashMap<(u64, ExprId, ExprId), bool>> =
        std::cell::RefCell::new(FxHashMap::default());
}

/// [`varies_numerically`] without the memo.
fn varies_numerically_uncached(arena: &mut Arena, q: ExprId, var: ExprId) -> bool {
    let mut params: Vec<(String, ExprId)> = Vec::new();
    for s in crate::base::walk::free_symbols(arena, q) {
        if s == var {
            continue;
        }
        let ExprNode::Symbol(sid) = *arena.node(s) else {
            return false;
        };
        params.push((arena.symbol_name(sid).to_owned(), s));
    }
    if params.len() > FTC_PARAMETER_VALUES.len() {
        return false;
    }
    params.sort_by(|a, b| a.0.cmp(&b.0));
    let mut bound = q;
    for (k, &(_, s)) in params.iter().enumerate() {
        let (p, d) = FTC_PARAMETER_VALUES[k];
        let value = arena.rational(p, d);
        bound = crate::transforms::subs::subs(arena, bound, s, value);
    }
    let mut values = Vec::with_capacity(VARIES_POINTS.len());
    for &(p, d) in &VARIES_POINTS {
        let point = arena.rational(p, d);
        let at = crate::transforms::subs::subs(arena, bound, var, point);
        let at = crate::transforms::eval::eval(arena, at);
        if crate::base::walk::has_unevaluated(arena, at) {
            return false;
        }
        let Ok(v) = crate::transforms::evalf::evalf_complex(arena, at, 15) else {
            return false;
        };
        values.push(v);
    }
    let (Some(scale), Some(gap)) = (
        crate::transforms::evalf::abs_to_f64(&values[0]),
        crate::transforms::evalf::distance_to_f64(&values[0], &values[1]),
    ) else {
        return false;
    };
    // Only skips `trigsimp`: the coefficient is right either way.
    scale.is_finite() && gap.is_finite() && gap > 1e-8 * scale.max(1.0)
}

/// Collect candidate `u`-expressions from a single factor.
///
/// For function nodes (`sin`, `cos`, `exp`, …) the inner argument and the
/// node itself are returned.  For `Pow(base, exp)` the base is returned
/// (enabling e.g. `u = x² + 1` inside `(x²+1)^{-1}`), and for `Pow` and
/// `Add` factors also the function nodes found inside them (`u = sin x`
/// for `(sin²x + 1)^{-1}` or `sin x + 1`), so that `∫ cos x·R(sin x) dx`
/// is a substitution rather than a Weierstrass problem.
fn u_sub_candidates(arena: &Arena, factor: ExprId, var_sym: SymbolId) -> SmallVec<[ExprId; 4]> {
    let mut out: SmallVec<[ExprId; 4]> = SmallVec::new();
    let node = arena.node(factor).clone();
    // The elementary functions and `|·|` (`Func::elementary`).
    if let Some((f, args)) = node.func(arena)
        && (f.elementary().is_some() || f == crate::base::func::Func::Abs)
        && let [inner] = args[..]
        && contains_var(arena, inner, var_sym)
    {
        out.push(inner);
        // Also try the function node itself as a candidate.
        // E.g., for ln(x), try u = ln(x) (not just u = x).
        // This enables ∫ 1/(x·ln(x)) dx via u = ln(x), du = 1/x dx.
        out.push(factor);
        return out;
    }
    match node {
        ExprNode::Pow(base, _exp) if contains_var(arena, base, var_sym) => {
            out.push(base);
            // The argument of a function under the power: `u = x²` for
            // `x·cos(x²)⁻¹` (up to 0.46 only `u = cos x²` and functions inside the
            // argument were tried, and `∫ x·sec(x²) dx` stayed unevaluated;
            // SymPy's test_manual).
            if let Some(inner) = function_argument(arena, base)
                && contains_var(arena, inner, var_sym)
                && !out.contains(&inner)
            {
                out.push(inner);
            }
            inner_function_candidates(arena, base, var_sym, &mut out);
        }
        ExprNode::Add(_) => {
            inner_function_candidates(arena, factor, var_sym, &mut out);
        }
        // A library function itself: `∫ J₀(x)·J₁(x) dx = −J₀(x)²/2` (`u = J₀`,
        // `du = −J₁ dx`; SymPy's test_meijerint), and its arguments other
        // than the bare variable: `u = x²` for `∫ x·J₁(x²) dx = −J₀(x²)/2`
        // (unevaluated up to 0.46; SymPy 1.14 integrates it).
        ExprNode::Apply(_, args) | ExprNode::Lib(_, args)
            if args.iter().any(|&a| contains_var(arena, a, var_sym)) =>
        {
            out.push(factor);
            for &a in &args {
                if contains_var(arena, a, var_sym)
                    && !matches!(arena.node(a), ExprNode::Symbol(s) if *s == var_sym)
                    && !out.contains(&a)
                {
                    out.push(a);
                }
            }
        }
        _ => {}
    }
    out
}

/// The argument `g` of `sin g`, `cos g`, `tan g`, `exp g`, `ln g`, `sinh g`,
/// `cosh g`, `tanh g`.
fn function_argument(arena: &Arena, e: ExprId) -> Option<ExprId> {
    match *arena.node(e) {
        ExprNode::Sin(g)
        | ExprNode::Cos(g)
        | ExprNode::Tan(g)
        | ExprNode::Exp(g)
        | ExprNode::Ln(g)
        | ExprNode::Sinh(g)
        | ExprNode::Cosh(g)
        | ExprNode::Tanh(g) => Some(g),
        _ => None,
    }
}

/// `∫ xᵐ·G(x) dx` where `x` occurs in `G` only in integer powers `xʲ` with
/// `k | j` and `k | m + 1` for some `k ≥ 2`: `u = xᵏ`, `∫ u^{(m+1)/k − 1}·G/k du`
/// (every `xʲ = u^{j/k}`), and `F(xᵏ)` for its antiderivative `F` (the chain
/// rule, `(xᵏ)^{(m+1)/k − 1} = x^{m+1−k}` exactly).  `∫ x·e^{x⁴} dx` is
/// `∫ e^{u²} du/2 = √π·erfi(x²)/4`, `∫ x·sin(x⁴) dx` a Fresnel integral;
/// `u = x²` is no subexpression, and up to 0.46 they stayed unevaluated
/// (SymPy's test_manual: `√π·erfi(x²)/4`).
fn try_power_substitution(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    use num_integer::Integer;
    if depth < 2 {
        return None;
    }
    let int_exp = |arena: &Arena, e: ExprId| -> Option<i64> {
        if e == var {
            return Some(1);
        }
        let ExprNode::Pow(b, n) = *arena.node(e) else {
            return None;
        };
        if b != var {
            return None;
        }
        arena
            .as_num(n)
            .filter(|q| q.is_integer())
            .and_then(|q| num_traits::ToPrimitive::to_i64(q.numer()))
    };
    let mut m: i64 = 0;
    let mut others: SmallVec<[ExprId; 4]> = SmallVec::new();
    for &d in dependent {
        match int_exp(arena, d) {
            Some(e) => m += e,
            None => others.push(d),
        }
    }
    if others.is_empty() {
        return None;
    }
    let g = arena.mul(&others);
    // The powers `xʲ` of `G`; a bare `x` left over after replacing them
    // makes the substitution fail below.
    let mut k: i64 = m + 1;
    let mut powers: Vec<(ExprId, i64)> = Vec::new();
    for id in crate::base::walk::post_order_ids(arena, g) {
        if id != var
            && let Some(e) = int_exp(arena, id)
        {
            k = k.gcd(&e);
            powers.push((id, e));
        }
    }
    if powers.is_empty() || k < 2 {
        return None;
    }
    // Only for a function that needs it: `e^{x⁴}`, `sin x⁴`, `sec x²`, `√(1 +
    // x⁴)` of a power.  A rational `G` is the rational integrator's, and
    // `ln(xᵏ) = k·ln x` gains nothing: after `x = t²`, `∫ ln(t²)·t/(t⁶ + t² +
    // 2/3) dt` went to `u = t²` and dilogarithms over a cubic's roots, and
    // `∫ (√x + ln x)/(x³ + x + 2/3) dx` ran for minutes (B14's fuzz unit
    // e6cc5985; 72 ms in 0.46).
    let needs = crate::base::walk::post_order_ids(arena, g)
        .into_iter()
        .any(|id| {
            let arg = match *arena.node(id) {
                ExprNode::Exp(a)
                | ExprNode::Sin(a)
                | ExprNode::Cos(a)
                | ExprNode::Tan(a)
                | ExprNode::Sinh(a)
                | ExprNode::Cosh(a)
                | ExprNode::Tanh(a)
                | ExprNode::Erf(a)
                | ExprNode::Erfc(a) => a,
                ExprNode::Pow(b, e) if arena.as_num(e).is_some_and(|q| !q.is_integer()) => b,
                _ => return false,
            };
            powers
                .iter()
                .any(|&(p, _)| crate::base::walk::contains(arena, arg, p))
        });
    if !needs {
        return None;
    }
    let u = arena.symbol("__pwu");
    let ExprNode::Symbol(u_sym) = *arena.node(u) else {
        return None;
    };
    // Larger powers first, so that `x` itself is replaced last.
    powers.sort_by_key(|p| std::cmp::Reverse(p.1));
    powers.dedup();
    let mut g_u = g;
    for &(node, e) in &powers {
        let ue = arena.int(e / k);
        let rep = arena.pow(u, ue);
        g_u = arena.subs_structural(g_u, node, rep);
    }
    if contains_var(arena, g_u, var_sym) {
        return None;
    }
    let lead = arena.int((m + 1) / k - 1);
    let u_lead = arena.pow(u, lead);
    let inv_k = arena.rational(1, k);
    let integrand = arena.mul(&[inv_k, u_lead, g_u]);
    let integrand = crate::transforms::eval::eval(arena, integrand);
    let res = integrate_node(arena, integrand, u, u_sym, depth - 1);
    if crate::base::walk::has_unevaluated(arena, res) {
        return None;
    }
    let k_id = arena.int(k);
    let x_k = arena.pow(var, k_id);
    let back = arena.subs_structural(res, u, x_k);
    tracing::debug!("integrate: power substitution u = x^k");
    Some(crate::transforms::eval::eval(arena, back))
}

/// Elementary function nodes (`sin g`, `exp g`, … with `g` depending on
/// `var`) inside `expr`, in post-order, deduplicated, at most four in total.
fn inner_function_candidates(
    arena: &Arena,
    expr: ExprId,
    var_sym: SymbolId,
    out: &mut SmallVec<[ExprId; 4]>,
) {
    for id in crate::base::walk::post_order_ids(arena, expr) {
        if out.len() >= 4 {
            return;
        }
        if id == expr || out.contains(&id) {
            continue;
        }
        let is_candidate = match arena.node(id) {
            ExprNode::Sin(inner)
            | ExprNode::Cos(inner)
            | ExprNode::Tan(inner)
            | ExprNode::Exp(inner)
            | ExprNode::Ln(inner)
            | ExprNode::Sinh(inner)
            | ExprNode::Cosh(inner)
            | ExprNode::Tanh(inner)
            // `u = erf x` for `∫ erf(x)·e^{−x²}/(1 + erf³x − erf²x − erf x) dx`
            // (SymPy's test_heurisch; unevaluated up to 0.46).
            | ExprNode::Erf(inner)
            | ExprNode::Erfc(inner) => contains_var(arena, *inner, var_sym),
            _ => false,
        };
        if is_candidate {
            out.push(id);
        }
    }
}

/// Build the product of all elements in `children` except index `skip`.
fn remaining_product(arena: &mut Arena, children: &[ExprId], skip: usize) -> ExprId {
    let parts: SmallVec<[ExprId; 4]> = children
        .iter()
        .enumerate()
        .filter(|&(j, _)| j != skip)
        .map(|(_, &c)| c)
        .collect();
    match parts.len() {
        0 => arena.one,
        1 => parts[0],
        _ => arena.mul(&parts),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Substitution strategies (v0.2 gap-fill)
// ═══════════════════════════════════════════════════════════════════════════

thread_local! {
    /// Nesting depth of substitution strategies that re-enter [`integrate`].
    static SUBST_DEPTH: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}

/// Maximum nesting of substitution strategies.
const MAX_SUBST_DEPTH: u8 = 3;

thread_local! {
    /// Set while a rationalising substitution integrates its transformed
    /// integrand ([`integrate_nested_generic`]): no `Piecewise` cases there.
    static GENERIC_ONLY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// [`integrate_nested`] for the integrand in the new variable of a
/// substitution whose answer is checked and wrapped in `x`: its generic
/// answer only.  The cases of the parameters there are cases of the
/// substitution (`b = 0` in `x = (t² − a)/b`), and the wrap of the outer
/// call finds the integrand's own; computing both nested a dozen
/// `Piecewise` levels and took seconds.
fn integrate_nested_generic(arena: &mut Arena, expr: ExprId, var: ExprId) -> Option<ExprId> {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            let _ = GENERIC_ONLY.try_with(|g| g.set(self.0));
        }
    }
    let restore = Restore(GENERIC_ONLY.with(|g| g.replace(true)));
    let result = integrate_nested(arena, expr, var);
    drop(restore);
    result
}

/// Run the full integration pipeline on a transformed integrand from
/// inside a substitution strategy, with a re-entrancy bound.
fn integrate_nested(arena: &mut Arena, expr: ExprId, var: ExprId) -> Option<ExprId> {
    /// Restores the substitution depth on drop, also when a panic unwinds.
    struct Restore(u8);
    impl Drop for Restore {
        fn drop(&mut self) {
            let _ = SUBST_DEPTH.try_with(|d| d.set(self.0));
        }
    }
    let depth = SUBST_DEPTH.with(|d| d.get());
    if depth >= MAX_SUBST_DEPTH {
        return None;
    }
    SUBST_DEPTH.with(|d| d.set(depth + 1));
    let restore = Restore(depth);
    let result = integrate(arena, expr, var);
    drop(restore);
    if crate::base::walk::has_unevaluated(arena, result) {
        None
    } else {
        Some(result)
    }
}

/// Try the substitution-based strategies in order.
fn try_substitution_strategies(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    if let Some(r) = stage!(
        arena,
        "trig_rational",
        expr,
        try_trig_rational_substitution(arena, expr, var, var_sym)
    ) {
        return Some(r);
    }
    if let Some(r) = try_exp_rational_substitution(arena, expr, var, var_sym) {
        return Some(r);
    }
    if let Some(r) = try_exp_linear_substitution(arena, expr, var, var_sym) {
        return Some(r);
    }
    if let Some(r) = try_exp_imaginary_to_trig(arena, expr, var, var_sym) {
        return Some(r);
    }
    if let Some(r) = try_radical_substitution(arena, expr, var, var_sym) {
        return Some(r);
    }
    if let Some(r) = try_even_power_radicals(arena, expr, var, var_sym) {
        return Some(r);
    }
    if let Some(r) = try_mobius_radical_substitution(arena, expr, var, var_sym) {
        return Some(r);
    }
    if let Some(r) = try_exp_atanh_rewrite(arena, expr, var, var_sym) {
        return Some(r);
    }
    if let Some(r) = try_hyperbolic_to_exp(arena, expr, var, var_sym) {
        return Some(r);
    }
    if let Some(r) = stage!(
        arena,
        "log_substitution",
        expr,
        try_log_substitution(arena, expr, var, var_sym, true)
    ) {
        return Some(r);
    }
    if let Some(r) = stage!(
        arena,
        "exp_substitution",
        expr,
        try_exp_general_substitution(arena, expr, var, var_sym)
    ) {
        return Some(r);
    }
    // Last, and for the integrand itself only: a `Piecewise` answer is for
    // real values of the variable, and the routes above (or the caller of
    // a substitution, in its own variable) may have an analytic one
    // (`∫ √x·√(1/x) dx`).
    if SUBST_DEPTH.with(std::cell::Cell::get) == 0
        && let Some(r) = stage!(
            arena,
            "piecewise_split",
            expr,
            try_piecewise_split(arena, expr, var, var_sym)
        )
    {
        return Some(r);
    }
    None
}

/// `(c, n)` with `g = c·xⁿ` for `c`, `n` free of `x`, `n ≠ 0` (`x` itself,
/// `xⁿ`, `c·x`, `c·xⁿ`).
fn monomial_parts(
    arena: &mut Arena,
    g: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<(ExprId, ExprId)> {
    let power = |arena: &Arena, f: ExprId| -> Option<ExprId> {
        if f == var {
            return Some(arena.one);
        }
        match arena.node(f) {
            ExprNode::Pow(b, e)
                if *b == var
                    && !contains_var(arena, *e, var_sym)
                    && !arena.is_zero_structural(*e) =>
            {
                Some(*e)
            }
            _ => None,
        }
    };
    if let Some(n) = power(arena, g) {
        return Some((arena.one, n));
    }
    let ExprNode::Mul(factors) = arena.node(g).clone() else {
        return None;
    };
    let mut n: Option<ExprId> = None;
    let mut rest: SmallVec<[ExprId; 4]> = SmallVec::new();
    for f in factors {
        if let Some(k) = power(arena, f) {
            if n.replace(k).is_some() {
                return None;
            }
        } else if contains_var(arena, f, var_sym) {
            return None;
        } else {
            rest.push(f);
        }
    }
    let c = match rest.len() {
        0 => arena.one,
        1 => rest[0],
        _ => arena.mul(&rest),
    };
    Some((c, n?))
}

/// `(c, k, d)` with `u = c·xᵏ + d` (`c`, `d` free of `x`, `k ≠ 0, 1` a
/// number); `None` otherwise (a linear `u` is a renaming that adds nothing).
fn affine_monomial_parts(
    arena: &mut Arena,
    u: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<(ExprId, Q, ExprId)> {
    if u == var {
        return None;
    }
    let (term, d) = match arena.node(u).clone() {
        ExprNode::Add(terms) => {
            let (dep, free): (SmallVec<[ExprId; 4]>, SmallVec<[ExprId; 4]>) = terms
                .iter()
                .copied()
                .partition(|&t| contains_var(arena, t, var_sym));
            let [t] = dep[..] else {
                return None;
            };
            (t, arena.add(&free))
        }
        _ => (u, arena.zero),
    };
    let (c, k) = monomial_parts(arena, term, var, var_sym)?;
    let k = arena.as_num(k)?.clone();
    (!k.is_one()).then_some((c, k, d))
}

/// `∫ h(x, L) dx` for `L = ln(c·xⁿ)` inside a trigonometric, hyperbolic or
/// exponential function (`sin(ln x)`, `x²·cos(a + b·ln(c·xⁿ))²`,
/// `sinh(ln x)`), `x` occurring elsewhere only in powers: with `w = L`,
/// `x = K·e^{w/n}` (`K = c^{−1/n}`) and `dx = x/n·dw`, the integral is
/// `∫ h(K·e^{w/n}, w)·K·e^{w/n}/n dw`, an exponential times trigonometric
/// functions of `w`.  Every term of the integrand and of its answer
/// carries `Kᵐ` with `e^{m·w/n}`, and `Kᵐ·e^{m·w/n} = xᵐ`: the answer in
/// `x` and `L` is the same for every `K`, so it is computed with `K = 1`
/// and brought back with `e^{q·w} = x^{q·n}` (it satisfies `F′ = h` along
/// the curve of every `K`, whatever `c` and `n` are; the self-check
/// confirms it).  Up to 0.31 `∫ sin(ln x) dx` stayed unevaluated; SymPy:
/// `x·sin(log x)/2 − x·cos(log x)/2`.
///
/// Inside `integrate_node` (`anywhere` false) only for a logarithm inside a
/// trigonometric, hyperbolic or exponential function; as a stage of its own
/// for any integrand that is one in `x` and `L` alone: `∫ xᵐ/ln x` is
/// `Ei((m + 1)·ln x)`, `∫ dx/ln(x)²` is `Ei(ln x) − x/ln x`, `∫ sin(ln x)/(x·ln x)`
/// is `Si(ln x)` (unevaluated up to 0.43; SymPy gives the same).  `L` is
/// real and monotonic on `x > 0`, where the integrand is real, so the
/// answer is continuous there where the one in `L` is.  There the answer may
/// have terms without an `e^{q·w}` (`Ei(2w)` for `∫ x/ln(3x)`), and `K` is
/// kept (`x = K·e^{w/n}`, `e^{q·w} = (x/K)^{q·n}`); powers of the
/// logarithm's argument are `gᵐ = e^{m·w}` (`√(5x)/ln(5x)`).
fn try_log_substitution(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
    anywhere: bool,
) -> Option<ExprId> {
    let order = crate::base::walk::post_order_ids(arena, expr);
    // The logarithm: one `ln(g)` of `x`, no other.
    let mut log: Option<(ExprId, ExprId)> = None;
    for &id in &order {
        if let ExprNode::Ln(g) = *arena.node(id)
            && contains_var(arena, g, var_sym)
        {
            match log {
                Some((l, _)) if l != id => return None,
                _ => log = Some((id, g)),
            }
        }
    }
    let (ln_node, g) = log?;
    let inside_function = order.iter().any(|&id| {
        matches!(arena.node(id),
            ExprNode::Sin(a) | ExprNode::Cos(a) | ExprNode::Tan(a) | ExprNode::Sinh(a)
            | ExprNode::Cosh(a) | ExprNode::Tanh(a) | ExprNode::Exp(a)
            if crate::base::walk::contains(arena, *a, ln_node))
    });
    if !inside_function && !anywhere {
        return None;
    }
    let w = arena.symbol("__lw");
    let ExprNode::Symbol(w_sym) = *arena.node(w) else {
        return None;
    };
    let Some((c, n)) = monomial_parts(arena, g, var, var_sym) else {
        return log_of_linear_substitution(arena, expr, ln_node, g, w, var, var_sym, anywhere);
    };
    // A real exponent only: `ln(c·xⁿ)` is real and monotonic on `x > 0` (up
    // to the constant `arg c`) only then.  For `n = i`, `xⁱ` turns around
    // the unit circle and `ln(c·xⁱ) = ln c + i·ln x` fails across the cut:
    // `∫ x·sec(a + 2·ln(c·xⁱ))³ dx` got an answer wrong for negative `c`
    // (Rubi 4.7.5 #181, `--negative-params`, 0.47 development).  A symbolic
    // `n` is generic (real), as every parameter of an antiderivative.
    let n_real = crate::base::assumptions::AssumptionCache::new().query(
        arena,
        n,
        crate::base::assumptions::Props::REAL,
    );
    if Truth::from(n_real).is_false()
        || (n_real.is_none() && crate::base::walk::free_symbols(arena, n).is_empty())
    {
        return None;
    }
    // K = c^{−1/n} where it is kept (see above).
    let scale = (anywhere && c != arena.one).then(|| {
        let minus_one = arena.int(-1);
        let e = arena.div(minus_one, n);
        let k = arena.pow(c, e);
        crate::transforms::eval::eval(arena, k)
    });
    // x^m → K^m·e^{m·w/n}, x → K·e^{w/n}.
    let x_power = |arena: &mut Arena, m: ExprId| -> ExprId {
        let mw = arena.mul(&[m, w]);
        let mwn = arena.div(mw, n);
        let e = arena.exp(mwn);
        match scale {
            Some(k) => {
                let k_m = arena.pow(k, m);
                arena.mul(&[k_m, e])
            }
            None => e,
        }
    };
    let mut sub = arena.subs_structural(expr, ln_node, w);
    if anywhere {
        if !algebraic_outside(arena, sub, g, var_sym) {
            return None;
        }
        sub = log_argument_powers(arena, sub, g, w, var_sym);
    }
    for id in crate::base::walk::post_order_ids(arena, sub) {
        if let ExprNode::Pow(b, m) = *arena.node(id)
            && b == var
            && !contains_var(arena, m, var_sym)
        {
            let r = x_power(arena, m);
            sub = arena.subs_structural(sub, id, r);
        }
    }
    let one = arena.one;
    let x_w = x_power(arena, one);
    sub = arena.subs_structural(sub, var, x_w);
    if contains_var(arena, sub, var_sym) {
        return None;
    }
    let integrand = arena.mul(&[sub, x_w]);
    let integrand = arena.div(integrand, n);
    let integrand = crate::transforms::eval::eval(arena, integrand);
    let res = integrate_nested(arena, integrand, w)?;
    // e^{q·w + r} → e^r·(x/K)^{q·n}.
    let x_over_k = match scale {
        Some(k) => arena.div(var, k),
        None => var,
    };
    let mut back = res;
    for id in crate::base::walk::post_order_ids(arena, res) {
        if let ExprNode::Exp(arg) = *arena.node(id)
            && crate::base::walk::has_free_symbol(arena, arg, w_sym)
        {
            let (q, r) = symbolic_linear_coeff_of(arena, arg, w, w_sym)?;
            let qn = arena.mul(&[q, n]);
            let x_qn = arena.pow(x_over_k, qn);
            let e_r = arena.exp(r);
            let repl = arena.mul(&[e_r, x_qn]);
            back = arena.subs_structural(back, id, repl);
        }
    }
    let back = crate::transforms::eval::eval(arena, back);
    let back = arena.subs_structural(back, w, ln_node);
    let back = crate::transforms::eval::eval(arena, back);
    if candidate_rejected(arena, expr, back, var, var_sym) {
        tracing::debug!("integrate: rejecting unverified logarithmic-substitution closed form");
        return None;
    }
    Some(back)
}

/// Does the variable occur in `e` (the integrand with its logarithm `ln g`
/// replaced) only in sums, products and powers with exponents free of it,
/// and in `g`?  The stage-level logarithmic substitution takes no other
/// integrand: `x = e^w` inside `sin x` gives `sin(e^w)`, no easier.
fn algebraic_outside(arena: &Arena, e: ExprId, g: ExprId, var_sym: SymbolId) -> bool {
    let mut stack = vec![e];
    let mut seen: FxHashSet<ExprId> = FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) || id == g || !contains_var(arena, id, var_sym) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Symbol(_) => {}
            ExprNode::Add(children) | ExprNode::Mul(children) => stack.extend(children.iter()),
            ExprNode::Neg(inner) => stack.push(*inner),
            ExprNode::Pow(b, k) if !contains_var(arena, *k, var_sym) => stack.push(*b),
            _ => return false,
        }
    }
    true
}

/// `e` with every power `gᵐ` (`m` free of the variable, `g ≠ x`) written
/// `e^{m·w}` for `w = ln g`, and `g` itself `e^w`: `e^{m·ln g} = gᵐ` for the
/// principal branches, whatever `g` is.
fn log_argument_powers(
    arena: &mut Arena,
    e: ExprId,
    g: ExprId,
    w: ExprId,
    var_sym: SymbolId,
) -> ExprId {
    if !matches!(arena.node(g), ExprNode::Add(_) | ExprNode::Mul(_)) {
        return e;
    }
    let mut out = e;
    for id in crate::base::walk::post_order_ids(arena, e) {
        if let ExprNode::Pow(b, m) = *arena.node(id)
            && b == g
            && !contains_var(arena, m, var_sym)
        {
            let mw = arena.mul(&[m, w]);
            let r = arena.exp(mw);
            out = arena.subs_structural(out, id, r);
        }
    }
    let e_w = arena.exp(w);
    arena.subs_structural(out, g, e_w)
}

/// [`try_log_substitution`] for `L = ln(g)` with `g = α·x + β` linear
/// (`β ≠ 0`) and `x` nowhere outside `L` (`sin(ln(a + b·x))`): `w = L`,
/// `dx = e^w/α·dw`, and the answer comes back with `e^{q·w} = g^q`.  As a
/// stage (`anywhere`), `x` may also occur in powers of `g`
/// ([`log_argument_powers`]): `∫ (2x + 1)/ln(2x + 1) dx = Ei(2·ln(2x + 1))/2`.
#[allow(clippy::too_many_arguments)]
fn log_of_linear_substitution(
    arena: &mut Arena,
    expr: ExprId,
    ln_node: ExprId,
    g: ExprId,
    w: ExprId,
    var: ExprId,
    var_sym: SymbolId,
    anywhere: bool,
) -> Option<ExprId> {
    let ExprNode::Symbol(w_sym) = *arena.node(w) else {
        return None;
    };
    let (alpha, beta) = symbolic_linear_coeff_of(arena, g, var, var_sym)?;
    if arena.is_zero_structural(beta) {
        return None;
    }
    let sub = arena.subs_structural(expr, ln_node, w);
    let sub = if anywhere {
        if !algebraic_outside(arena, sub, g, var_sym) {
            return None;
        }
        log_argument_powers(arena, sub, g, w, var_sym)
    } else {
        sub
    };
    if contains_var(arena, sub, var_sym) {
        return None;
    }
    let e_w = arena.exp(w);
    let integrand = arena.mul(&[sub, e_w]);
    let integrand = arena.div(integrand, alpha);
    let integrand = crate::transforms::eval::eval(arena, integrand);
    let res = integrate_nested(arena, integrand, w)?;
    let mut back = res;
    for id in crate::base::walk::post_order_ids(arena, res) {
        if let ExprNode::Exp(arg) = *arena.node(id)
            && crate::base::walk::has_free_symbol(arena, arg, w_sym)
        {
            let (q, r) = symbolic_linear_coeff_of(arena, arg, w, w_sym)?;
            let g_q = arena.pow(g, q);
            let e_r = arena.exp(r);
            let repl = arena.mul(&[e_r, g_q]);
            back = arena.subs_structural(back, id, repl);
        }
    }
    let back = crate::transforms::eval::eval(arena, back);
    let back = arena.subs_structural(back, w, ln_node);
    let back = crate::transforms::eval::eval(arena, back);
    if candidate_rejected(arena, expr, back, var, var_sym) {
        tracing::debug!("integrate: rejecting unverified logarithmic-substitution closed form");
        return None;
    }
    Some(back)
}

/// Collect the `exp(k·x)` nodes of `expr` (with numeric `k`).  Returns
/// `None` if some `exp` argument depends on `var` but is not of that form.
fn collect_exp_multiples(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<Vec<(ExprId, Q)>> {
    let order = crate::base::walk::post_order_ids(arena, expr);
    let mut out = Vec::new();
    for id in order {
        if let ExprNode::Exp(arg) = arena.node(id).clone()
            && contains_var(arena, arg, var_sym)
        {
            let (alpha, beta) = symbolic_linear_coeff_of(arena, arg, var, var_sym)?;
            if !arena.is_zero_structural(beta) {
                return None;
            }
            let k = arena.as_num(alpha)?.clone();
            out.push((id, k));
        }
    }
    Some(out)
}

/// `∫ R(e^{ax}) dx` via `u = e^{ax}`: `∫ R(u)/(a·u) du` for a rational `R`.
///
/// `a` is chosen as the greatest common divisor of all exponent
/// coefficients so that every `e^{kx}` becomes an integer power of `u`.
fn try_exp_rational_substitution(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    use num_integer::Integer;
    let exps = collect_exp_multiples(arena, expr, var, var_sym)?;
    if exps.is_empty() {
        return None;
    }
    // a = gcd(k_i) for rationals: gcd(numerators)/lcm(denominators).
    let mut num_gcd = num_bigint::BigInt::zero();
    let mut den_lcm = num_bigint::BigInt::one();
    for (_, k) in &exps {
        num_gcd = num_gcd.gcd(k.numer());
        den_lcm = den_lcm.lcm(k.denom());
    }
    if num_gcd.is_zero() {
        return None;
    }
    let a = num_rational::Ratio::new(num_gcd, den_lcm);

    let u = arena.symbol("__eu");
    let u_sym = match arena.node(u) {
        ExprNode::Symbol(s) => *s,
        _ => return None,
    };
    let mut sub = expr;
    for (node, k) in &exps {
        let power = k / &a; // integer
        let power_id = arena.num_ratio(power.clone());
        let u_pow = arena.pow(u, power_id);
        sub = arena.subs_structural(sub, *node, u_pow);
    }
    if contains_var(arena, sub, var_sym) {
        return None;
    }
    // Integrand in u: R(u)/(a u)
    let a_id = arena.num_ratio(a.clone());
    let a_u = arena.mul(&[a_id, u]);
    let integrand_u = arena.div(sub, a_u);
    let integrand_u = clear_nested_fractions(arena, integrand_u, u);
    // Must be rational in u.
    let (n, d) = crate::poly::polybridge::as_numer_denom(arena, integrand_u);
    if crate::poly::polybridge::expr_to_poly(arena, n, u).is_none()
        || crate::poly::polybridge::expr_to_poly(arena, d, u).is_none()
    {
        return None;
    }
    let res_u = integrate_node(arena, integrand_u, u, u_sym, 20);
    if crate::base::walk::has_unevaluated(arena, res_u) {
        return None;
    }
    // Back-substitute u → e^{ax}.
    let ax = arena.mul(&[a_id, var]);
    let e_ax = arena.exp(ax);
    let result = arena.subs_structural(res_u, u, e_ax);
    Some(crate::transforms::eval::eval(arena, result))
}

/// `∫ R(e^{L₁}, …, e^{Lₖ}) dx` for linear `Lᵢ = rᵢ·L + γᵢ` with rational
/// `rᵢ`, symbolic `L = α·x + β` (the first exponent) and constants `γᵢ`:
/// `u = e^{g·L}` (`g` the gcd of the `rᵢ`), `du = g·α·u·dx`, every
/// `e^{Lᵢ} = e^{γᵢ}·u^{rᵢ/g}`, and the integrand is rational in `u` over
/// `ℚ(parameters, e^{γᵢ})` (each `e^{γᵢ} ≠ 1` a fresh symbol while the
/// rational integrator runs).  [`try_exp_rational_substitution`] for
/// symbolic rates and shifts: `sinh(a + b·x)`, `sech(c + d·x)²`,
/// `coth(a + b·x)·coth(c + b·x)` (Rubi 6.x) stayed unevaluated.  `u`
/// moves monotonically along a ray for real `x`, so the logarithms of the
/// rational answer jump only at poles of the integrand.
fn try_exp_linear_substitution(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    use num_integer::Integer;
    let mut exps: Vec<(ExprId, ExprId, ExprId)> = Vec::new();
    for id in crate::base::walk::post_order_ids(arena, expr) {
        if let ExprNode::Exp(arg) = *arena.node(id)
            && contains_var(arena, arg, var_sym)
        {
            let (alpha, beta) = symbolic_linear_coeff_of(arena, arg, var, var_sym)?;
            exps.push((id, alpha, beta));
        }
    }
    let &(_, alpha0, beta0) = exps.first()?;
    let mut ratios: Vec<Q> = Vec::with_capacity(exps.len());
    for &(_, alpha, _) in &exps {
        let r = arena.div(alpha, alpha0);
        let r = crate::transforms::eval::eval(arena, r);
        ratios.push(arena.as_num(r)?.clone());
    }
    let mut num_gcd = num_bigint::BigInt::zero();
    let mut den_lcm = num_bigint::BigInt::one();
    for r in &ratios {
        num_gcd = num_gcd.gcd(r.numer());
        den_lcm = den_lcm.lcm(r.denom());
    }
    if num_gcd.is_zero() {
        return None;
    }
    let g = Q::new(num_gcd, den_lcm);
    let u = arena.symbol("__elu");
    let ExprNode::Symbol(u_sym) = *arena.node(u) else {
        return None;
    };
    let mut constants: Vec<(ExprId, ExprId)> = Vec::new();
    let mut sub = expr;
    for (&(node, _, beta), r) in exps.iter().zip(&ratios) {
        let r_id = arena.num_ratio(r.clone());
        let r_beta0 = arena.mul(&[r_id, beta0]);
        let gamma = arena.sub(beta, r_beta0);
        let gamma = crate::transforms::eval::eval(arena, gamma);
        let k = arena.num_ratio(r / &g);
        let u_k = arena.pow(u, k);
        let repl = if arena.is_zero_structural(gamma) {
            u_k
        } else {
            let e_gamma = arena.exp(gamma);
            let kappa = match constants.iter().find(|&&(e, _)| e == e_gamma) {
                Some(&(_, s)) => s,
                None => {
                    let s = arena.symbol(&format!("__elk{}", constants.len()));
                    constants.push((e_gamma, s));
                    s
                }
            };
            arena.mul(&[kappa, u_k])
        };
        sub = arena.subs_structural(sub, node, repl);
    }
    if contains_var(arena, sub, var_sym) {
        return None;
    }
    let g_id = arena.num_ratio(g.clone());
    let du = arena.mul(&[g_id, alpha0, u]);
    let integrand = arena.div(sub, du);
    let integrand = crate::transforms::eval::eval(arena, integrand);
    if !is_rational_in(arena, integrand, u_sym) {
        return None;
    }
    let res = integrate_nested_generic(arena, integrand, u)?;
    let res = analytic_logs(arena, res, u_sym)?;
    let l = arena.mul(&[alpha0, var]);
    let l = arena.add(&[l, beta0]);
    let gl = arena.mul(&[g_id, l]);
    let res = match exp_rate_is_nonreal_constant(arena, g_id, alpha0) {
        Truth::True => circle_continuous_in_u(arena, res, u, u_sym, gl)?,
        Truth::False => res,
        Truth::Unknown => return None,
    };
    let e_gl = arena.exp(gl);
    let mut back = arena.subs_structural(res, u, e_gl);
    for &(e_gamma, kappa) in &constants {
        back = arena.subs_structural(back, kappa, e_gamma);
    }
    let back = crate::transforms::eval::eval(arena, back);
    if candidate_rejected(arena, expr, back, var, var_sym) {
        tracing::debug!("integrate: rejecting unverified exponential-substitution closed form");
        return None;
    }
    Some(back)
}

/// Is the rate `g·α` of `u = e^{g·(α·x + β)}` a constant that is not real?
/// Then `u` does not run along a ray for real `x`: on the unit circle for an
/// imaginary rate, on a spiral otherwise.  `True` when its imaginary part
/// is certified nonzero, `False` when it is proved 0 or the rate has
/// parameters (read as real, as before 0.53), `Unknown` for a constant
/// whose imaginary part does not settle.  Up to 0.52 an imaginary part
/// below `10⁻¹²` of the rate counted as 0.
fn exp_rate_is_nonreal_constant(arena: &mut Arena, g: ExprId, alpha: ExprId) -> Truth {
    let rate = arena.mul(&[g, alpha]);
    let rate = crate::transforms::eval::eval(arena, rate);
    if !crate::base::walk::free_symbols(arena, rate).is_empty() {
        return Truth::False;
    }
    crate::transforms::realness::imaginary_part_sign(arena, rate)
        .is_zero()
        .not()
}

/// The answer `res` in `u = e^{gl}` of [`try_exp_linear_substitution`] for a
/// non-real rate, where `u` leaves the ray along which its logarithms keep
/// off their cuts: `ln u` becomes `gl` (the continuous logarithm of
/// `e^{gl}`), and `None` when any other non-rational function of `u`
/// remains.  `∫ dt/(e^{it} − 2)` came out as `i·(ln e^{it} − ln(e^{it} − 2))/2`,
/// which jumps at `t = π` and at `t = 0` (SymPy's test_integrals; 0.46
/// returned `0` for the integral over `[0, 2π]`, which is `−π`).  Such an
/// integrand goes to [`try_exp_imaginary_to_trig`] instead.
fn circle_continuous_in_u(
    arena: &mut Arena,
    res: ExprId,
    u: ExprId,
    u_sym: SymbolId,
    gl: ExprId,
) -> Option<ExprId> {
    let ln_u = arena.ln(u);
    let res = arena.subs_structural(res, ln_u, gl);
    is_rational_in(arena, res, u_sym).then_some(res)
}

/// `∫ R(e^{iθ₁}, …) dx` for real linear `θⱼ` (an imaginary rate, where
/// [`try_exp_linear_substitution`] cannot keep its logarithms continuous):
/// every `e^{iθ}` written `cos θ + i·sin θ`, and the trigonometric
/// integrand through the whole pipeline, whose routes correct the jumps
/// at the poles of `tan(θ/2)`.
fn try_exp_imaginary_to_trig(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    let mut any = false;
    for id in crate::base::walk::post_order_ids(arena, expr) {
        if let ExprNode::Exp(arg) = *arena.node(id)
            && contains_var(arena, arg, var_sym)
        {
            let (alpha, _) = symbolic_linear_coeff_of(arena, arg, var, var_sym)?;
            let one = arena.one;
            if !exp_rate_is_nonreal_constant(arena, one, alpha).is_true() {
                return None;
            }
            let re = arena.re(alpha);
            let re = crate::transforms::eval::eval(arena, re);
            if !arena.is_zero_structural(re) {
                return None;
            }
            any = true;
        }
    }
    if !any {
        return None;
    }
    let rewritten = arena.rewrite_as_trig_expr(expr);
    let rewritten = crate::transforms::eval::eval(arena, rewritten);
    if rewritten == expr
        || crate::base::walk::post_order_ids(arena, rewritten)
            .into_iter()
            .any(
                |id| matches!(*arena.node(id), ExprNode::Exp(a) if contains_var(arena, a, var_sym)),
            )
    {
        return None;
    }
    let res = integrate_nested(arena, rewritten, var)?;
    if candidate_rejected(arena, expr, res, var, var_sym) {
        tracing::debug!("integrate: rejecting unverified e^(iθ) → trig closed form");
        return None;
    }
    Some(res)
}

/// `∫ f(e^{a₁x+b₁}, …, e^{aₖx+bₖ}) dx` with rational rates `aᵢ`, `x` nowhere
/// else, that is not rational in the exponentials (those are
/// [`try_exp_rational_substitution`]'s and [`try_exp_linear_substitution`]'s):
/// `u = e^{g·x}` (`g` the gcd of the rates), `dx = du/(g·u)`, and the
/// integrand in `u` through the whole pipeline.  `∫ dy/√(2eʸ + 3)` is
/// `∫ du/(u·√(2u + 3))` (SymPy: `(ln(√(2eʸ+3) − √3) − ln(√(2eʸ+3) +
/// √3))/√3`), `∫ exp(eˣ) dx` is `Ei(eˣ)`, `∫ sin(eˣ) dx` is `Si(eˣ)`; up to
/// 0.43 they stayed unevaluated (the ODE solver worked around the first).
/// For real `x`, `u` runs once through `(0, ∞)`, where the answer in `u` is
/// continuous wherever its integrand is.  Rates with parameters are left
/// alone: `e^{a·x}` with a complex `a` does not keep to that ray.
fn try_exp_general_substitution(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    use num_integer::Integer;
    let mut exps: Vec<(ExprId, Q, ExprId)> = Vec::new();
    for id in crate::base::walk::post_order_ids(arena, expr) {
        if let ExprNode::Exp(arg) = *arena.node(id)
            && contains_var(arena, arg, var_sym)
            && let Some((alpha, beta)) = symbolic_linear_coeff_of(arena, arg, var, var_sym)
        {
            let a = arena.as_num(alpha)?.clone();
            exps.push((id, a, beta));
        }
    }
    if exps.is_empty() {
        return None;
    }
    let mut num_gcd = num_bigint::BigInt::zero();
    let mut den_lcm = num_bigint::BigInt::one();
    for (_, a, _) in &exps {
        num_gcd = num_gcd.gcd(a.numer());
        den_lcm = den_lcm.lcm(a.denom());
    }
    if num_gcd.is_zero() {
        return None;
    }
    // `u = e^{−x}` when every rate is negative: `∫ exp(e^{−x}) dx = −Ei(e^{−x})`.
    let g = if exps.iter().all(|(_, a, _)| a.is_negative()) {
        -Q::new(num_gcd, den_lcm)
    } else {
        Q::new(num_gcd, den_lcm)
    };
    let u = arena.symbol("__egu");
    let ExprNode::Symbol(u_sym) = *arena.node(u) else {
        return None;
    };
    let mut sub = expr;
    for (node, a, beta) in &exps {
        let k = arena.num_ratio(a / &g);
        let u_k = arena.pow(u, k);
        let repl = if arena.is_zero_structural(*beta) {
            u_k
        } else {
            let e_beta = arena.exp(*beta);
            arena.mul(&[e_beta, u_k])
        };
        sub = arena.subs_structural(sub, *node, repl);
    }
    if contains_var(arena, sub, var_sym) {
        return None;
    }
    let g_id = arena.num_ratio(g.clone());
    let du = arena.mul(&[g_id, u]);
    let integrand = arena.div(sub, du);
    let integrand = crate::transforms::eval::eval(arena, integrand);
    if is_rational_in(arena, integrand, u_sym) {
        return None;
    }
    // `u` is real and positive: the answer in `u` under the real-variable
    // convention (`ln|√(3 − 2u) − √3|`) is the one wanted (the stage exit
    // turns the logarithms of non-real arguments into principal ones).
    let res = integrate_nested_generic(arena, integrand, u)?;
    let gx = arena.mul(&[g_id, var]);
    let e_gx = arena.exp(gx);
    let back = arena.subs_structural(res, u, e_gx);
    let back = crate::transforms::eval::eval(arena, back);
    if candidate_rejected(arena, expr, back, var, var_sym) {
        tracing::debug!("integrate: rejecting unverified exponential-substitution closed form");
        return None;
    }
    Some(back)
}

/// `∫ f(x, (κ·x²ʲ)^r, …) dx` (`κ` free of `x`, `r` not an integer, `m = 2j·r`
/// an integer): every such radical is `C·x^m` with `C = (κ·x²ʲ)^r·x^(−m)`,
/// whose derivative is 0 (an identity of analytic functions, so for every
/// value of the parameters), so `F = C·G` with `G = ∫ f·x^m/(κ·x²ʲ)^r dx` —
/// Rubi's device for these radicals (`√(b·x⁴)·x⁻²·(…)`).  For real `x`, `C`
/// is constant on each side of `x = 0` (`(κ·p)^r = κ^r·p^r` for `p ≥ 0`): the
/// same constant `κ^r` for an even `m`; for an odd one it changes sign
/// there, and `F = C·(G − G(0))` is continuous.  The Möbius substitution took
/// `s = x/√(b·x⁴)` for `√(b·x⁴)`, which jumps from `−∞` to `+∞` at `x = 0`:
/// `∫ √(b·x⁴)/(1 + x²) dx` stepped by `π·√b` there (`F(1) − F(−1)` = 3.0924
/// for 0.3717 at `b = 3/4`; Rubi 4.3.0 `√(b·tan⁴(c + d·x))` through `t = tan
/// u`).  `None` when no radical is of that form or `G` is not found.
fn try_even_power_radicals(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    let two = num_bigint::BigInt::from(2);
    let mut out = expr;
    let mut constant: Vec<ExprId> = Vec::new();
    let mut odd = false;
    for id in crate::base::walk::post_order_ids(arena, expr) {
        let ExprNode::Pow(base, ex) = *arena.node(id) else {
            continue;
        };
        let Some(r) = arena.as_num(ex).cloned() else {
            continue;
        };
        if r.is_integer() || !contains_var(arena, base, var_sym) {
            continue;
        }
        let factors: Vec<ExprId> = match arena.node(base) {
            ExprNode::Mul(children) => children.to_vec(),
            _ => vec![base],
        };
        let mut power: Option<Q> = None;
        let mut kappa: Vec<ExprId> = Vec::new();
        for f in factors {
            if !contains_var(arena, f, var_sym) {
                kappa.push(f);
                continue;
            }
            match *arena.node(f) {
                ExprNode::Pow(b, e) if b == var && power.is_none() => {
                    power = arena.as_num(e).cloned();
                }
                _ => {
                    power = None;
                    break;
                }
            }
        }
        let Some(n) = power else {
            continue;
        };
        if !n.is_integer() || n.is_negative() || !(n.numer() % &two).is_zero() {
            continue;
        }
        let m = &n * &r;
        if !m.is_integer() {
            continue;
        }
        odd |= !(m.numer() % &two).is_zero();
        let m_id = arena.num_ratio(m.clone());
        let x_m = arena.pow(var, m_id);
        let minus_m = arena.num_ratio(-m);
        let x_minus_m = arena.pow(var, minus_m);
        constant.push(arena.mul(&[id, x_minus_m]));
        out = arena.subs_structural(out, id, x_m);
    }
    if constant.is_empty() {
        return None;
    }
    let out = crate::transforms::eval::eval(arena, out);
    tracing::debug!(rewritten = %arena.display(out), "integrate: radicals of even powers of the variable");
    let mut g = integrate_nested(arena, out, var)?;
    if odd {
        let at_zero = arena.subs_structural(g, var, arena.zero);
        let at_zero = crate::transforms::eval::eval(arena, at_zero);
        if contains_non_finite(arena, at_zero) || contains_var(arena, at_zero, var_sym) {
            return None;
        }
        let minus = arena.neg(at_zero);
        g = arena.add(&[g, minus]);
    }
    constant.push(g);
    let res = arena.mul(&constant);
    if candidate_rejected(arena, expr, res, var, var_sym) {
        return None;
    }
    Some(res)
}

/// `∫ f(x, x^{p/q}) dx` via `x = s^q`: `∫ q·s^{q−1} f(s^q, s^p) ds`.
fn try_radical_substitution(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    use num_integer::Integer;
    let order = crate::base::walk::post_order_ids(arena, expr);
    let mut radicals: Vec<(ExprId, Q)> = Vec::new();
    let mut q_lcm = num_bigint::BigInt::one();
    for id in order {
        if let ExprNode::Pow(base, e) = arena.node(id).clone()
            && base == var
            && let Some(r) = arena.as_num(e).cloned()
            && !r.is_integer()
        {
            q_lcm = q_lcm.lcm(r.denom());
            radicals.push((id, r));
        }
    }
    if radicals.is_empty() {
        return None;
    }
    let q: i64 = q_lcm.to_string().parse().ok()?;
    if !(2..=6).contains(&q) {
        return None;
    }
    let s = arena.symbol("__rs");
    let mut sub = expr;
    for (node, r) in &radicals {
        let k = r * num_rational::Ratio::from_integer(num_bigint::BigInt::from(q));
        let k_id = arena.num_ratio(k.clone());
        let s_pow = arena.pow(s, k_id);
        sub = arena.subs_structural(sub, *node, s_pow);
    }
    let q_id = arena.int(q);
    let s_q = arena.pow(s, q_id);
    sub = arena.subs_structural(sub, var, s_q);
    if contains_var(arena, sub, var_sym) {
        return None;
    }
    // `s = x^{1/q}` is real only where `x ≥ 0` (for even `q`), so the
    // `s`-integrand may not contain anything whose value depends on `s`
    // being real — `|s|`, `sign(s)`, `H(s)`: `∫ |√x| dx` became
    // `∫ 2s·|s| ds = ⅔ s³ sign(s)`, wrong for x < 0 (fuzz_integrate).
    let s_sym = match arena.node(s) {
        ExprNode::Symbol(sid) => *sid,
        _ => return None,
    };
    let real_only = crate::base::walk::post_order_ids(arena, sub)
        .into_iter()
        .any(|id| match arena.node(id) {
            ExprNode::Abs(g) | ExprNode::Sign(g) | ExprNode::Heaviside(g) => {
                contains_var(arena, *g, s_sym)
            }
            _ => false,
        });
    if real_only {
        return None;
    }
    // dx = q s^{q−1} ds
    let qm1 = arena.int(q - 1);
    let s_qm1 = arena.pow(s, qm1);
    let integrand_s = arena.mul(&[q_id, s_qm1, sub]);
    let integrand_s = crate::transforms::eval::eval(arena, integrand_s);
    let res_s = integrate_nested(arena, integrand_s, s)?;
    // The `s`-integrand is analytic, but the rational integrator writes
    // `ln|v(s)|`, the *real* antiderivative — wrong once `s = x^{1/q}` is not
    // real (`∫ x^(−5/2)·atan(√x) dx` gained `−⅔ ln|√x|`, whose derivative
    // at x < 0 is not the integrand; found by `fuzz_integrate`).  Use the
    // analytic `ln v(s)`; any other `|·|`, `sign` or `H` of `s` is refused.
    let res_s = analytic_logs(arena, res_s, s_sym)?;
    // Back-substitute s → x^{1/q}.
    let inv_q = arena.rational(1, q);
    let root = arena.pow(var, inv_q);
    let result = arena.subs_structural(res_s, s, root);
    let result = crate::transforms::eval::eval(arena, result);
    // Verified like the other risky routes: the `s`-integral is computed by
    // the real-variable integrator, and a real-only step can still slip
    // through below the checks above — `(s⁻⁴)^{1/2}` flattened to `s⁻²` gave
    // `∫ cos(√x)/√(x⁻²) dx` a closed form whose derivative is wrong for
    // x < 0 (0.23, fuzz_integrate).
    if candidate_rejected(arena, expr, result, var, var_sym) {
        tracing::debug!("integrate: rejecting unverified radical-substitution closed form");
        return None;
    }
    Some(result)
}

/// `ln|g|` → `ln g` for every `g` depending on `s`; `None` when an `|·|`,
/// `sign` or `H` of `s` remains anywhere else.
fn analytic_logs(arena: &mut Arena, e: ExprId, s_sym: SymbolId) -> Option<ExprId> {
    let order = crate::base::walk::post_order_ids(arena, e);
    let mut out = e;
    for id in order {
        if let ExprNode::Ln(inner) = arena.node(id).clone()
            && let ExprNode::Abs(g) = arena.node(inner).clone()
            && contains_var(arena, g, s_sym)
        {
            let plain = arena.ln(g);
            out = arena.subs_structural(out, id, plain);
        }
    }
    let leftover = crate::base::walk::post_order_ids(arena, out)
        .into_iter()
        .any(|id| match arena.node(id) {
            ExprNode::Abs(g) | ExprNode::Sign(g) | ExprNode::Heaviside(g) => {
                contains_var(arena, *g, s_sym)
            }
            _ => false,
        });
    (!leftover).then_some(out)
}

// ═══════════════════════════════════════════════════════════════════════════
// Radicals of a Möbius function: t = B^{1/q}, t = B₁^{1/q}·B₂^{−1/q}
// ═══════════════════════════════════════════════════════════════════════════

/// Largest root index `q` [`try_mobius_radical_substitution`] takes (that
/// of [`try_radical_substitution`]).
const MAX_MOBIUS_ROOT: i64 = 6;

/// `m = (a·x + b)/(c·x + d)` with `a`, `b`, `c`, `d` free of `x`.
struct Mobius {
    a: ExprId,
    b: ExprId,
    c: ExprId,
    d: ExprId,
}

/// The coefficients of `m` as a Möbius function of `var` (after its
/// fractions are combined and cancelled), or `None`.
fn mobius_parts(arena: &mut Arena, m: ExprId, var: ExprId, var_sym: SymbolId) -> Option<Mobius> {
    let m = clear_nested_fractions(arena, m, var);
    let (numer, denom) = crate::poly::polybridge::as_numer_denom(arena, m);
    let mut linear = |e: ExprId| -> Option<(ExprId, ExprId)> {
        if contains_var(arena, e, var_sym) {
            symbolic_linear_coeff_of(arena, e, var, var_sym)
        } else {
            Some((arena.zero, e))
        }
    };
    let (a, b) = linear(numer)?;
    let (c, d) = linear(denom)?;
    if contains_var(arena, a, var_sym)
        || contains_var(arena, b, var_sym)
        || contains_var(arena, c, var_sym)
        || contains_var(arena, d, var_sym)
    {
        return None;
    }
    Some(Mobius { a, b, c, d })
}

/// `(node, base, e)` for every power `base^e` in `expr` whose base depends
/// on the variable and whose exponent is a rational number that is not an
/// integer; `None` when a power of such a base has a symbolic exponent or
/// one that depends on the variable.
fn radical_powers(
    arena: &Arena,
    expr: ExprId,
    var_sym: SymbolId,
) -> Option<Vec<(ExprId, ExprId, Q)>> {
    let mut out = Vec::new();
    for id in crate::base::walk::post_order_ids(arena, expr) {
        if let ExprNode::Pow(base, e) = *arena.node(id)
            && contains_var(arena, base, var_sym)
        {
            let r = arena.as_num(e)?;
            if !r.is_integer() {
                out.push((id, base, r.clone()));
            }
        } else if let ExprNode::Pow(_, e) = *arena.node(id)
            && contains_var(arena, e, var_sym)
        {
            return None;
        }
    }
    Some(out)
}

/// Denominators of at most this degree go to the rational integrators
/// cancelled and expanded ([`rational_in_factored_form`]); the one over
/// `ℚ(p₁, …)` takes up to this degree in that form.
const FACTORED_FORM_DEGREE: i64 = 10;

/// The largest exponent of `v` in the expanded polynomial `p`.
fn expanded_degree_in(arena: &Arena, p: ExprId, v: ExprId) -> i64 {
    let mut deg = 0;
    for id in crate::base::walk::post_order_ids(arena, p) {
        if id == v {
            deg = deg.max(1);
        } else if let ExprNode::Pow(b, k) = *arena.node(id)
            && b == v
            && let Some(k) = arena
                .as_num(k)
                .and_then(|k| i64::try_from(k.to_integer()).ok())
        {
            deg = deg.max(k);
        }
    }
    deg
}

/// A rational function `e` of `v` as one quotient: with parameters, the
/// numerator multiplied out over the denominator's factors as they come
/// (`(a − c·t²)⁸`), which the rational integrator over `ℚ(p₁, …)` uses
/// as its square-free decomposition; without, cancelled and expanded
/// ([`clear_nested_fractions`]).
fn rational_in_factored_form(arena: &mut Arena, e: ExprId, v: ExprId) -> ExprId {
    let e = crate::transforms::eval::eval(arena, e);
    let cleared = clear_nested_fractions(arena, e, v);
    let has_parameters = crate::base::walk::free_symbols(arena, cleared)
        .into_iter()
        .any(|s| s != v);
    let (_, d) = crate::poly::polybridge::as_numer_denom(arena, cleared);
    if !has_parameters || expanded_degree_in(arena, d, v) <= FACTORED_FORM_DEGREE {
        return cleared;
    }
    let (n, d) = crate::poly::polybridge::fraction_parts(arena, e);
    let n = crate::transforms::expand::expand(arena, n);
    let n = crate::transforms::eval::eval(arena, n);
    // Each factor multiplied out on its own: a Möbius image such as
    // `a·b·t² + a·(1 − b·t²)` is the constant `a`.
    let factors: SmallVec<[ExprId; 6]> = match arena.node(d) {
        ExprNode::Mul(children) => children.clone(),
        _ => SmallVec::from_slice(&[d]),
    };
    let mut expanded: SmallVec<[ExprId; 6]> = SmallVec::new();
    for f in factors {
        let (base, k) = match *arena.node(f) {
            ExprNode::Pow(b, k) => (b, Some(k)),
            _ => (f, None),
        };
        let base = crate::transforms::expand::expand(arena, base);
        let base = crate::transforms::eval::eval(arena, base);
        expanded.push(match k {
            Some(k) => arena.pow(base, k),
            None => base,
        });
    }
    let d = arena.mul(&expanded);
    arena.div(n, d)
}

/// Is `e` a rational function of the symbol `v` (sums, products and
/// integer powers of `v` and of constants)?
fn is_rational_in(arena: &Arena, e: ExprId, v: SymbolId) -> bool {
    let mut depends: FxHashMap<ExprId, bool> = FxHashMap::default();
    for id in crate::base::walk::post_order_ids(arena, e) {
        let node = arena.node(id);
        let mut below = matches!(node, ExprNode::Symbol(s) if *s == v);
        node.for_each_child(|c| below |= depends.get(&c).copied().unwrap_or(false));
        if below {
            let ok = match node {
                ExprNode::Symbol(_) | ExprNode::Add(_) | ExprNode::Mul(_) | ExprNode::Neg(_) => {
                    true
                }
                ExprNode::Pow(_, ex) => arena.as_num(*ex).is_some_and(|r| r.is_integer()),
                _ => false,
            };
            if !ok {
                return false;
            }
        }
        depends.insert(id, below);
    }
    true
}

/// `expr` with every product `B₁^{e₁}·B₂^{e₂}·r` (`e₁ + e₂` an integer)
/// written `t^{q·e₁}·B₂^{e₁+e₂}·r` — exact for `t = B₁^{1/q}·B₂^{−1/q}`,
/// since `B₁^{e₁} = (B₁^{1/q})^{q·e₁}` and `B₂^{e₂} = (B₂^{−1/q})^{−q·e₂}`
/// for principal powers, and `(B₂^{−1/q})^{−q·k} = B₂^k` for an integer
/// `k`.  `None` when a radical of `B₁` or `B₂` is left over (one without
/// its partner, or a pair whose exponents do not sum to an integer:
/// `∛(1 + x)·∛(1 − x)` is elliptic).
fn pair_radicals(
    arena: &mut Arena,
    expr: ExprId,
    b1: ExprId,
    b2: ExprId,
    t: ExprId,
    q: i64,
) -> Option<ExprId> {
    let radical_of = |arena: &Arena, id: ExprId, base: ExprId| -> Option<(ExprId, Q)> {
        match *arena.node(id) {
            ExprNode::Pow(b, e) if b == base => {
                let r = arena.as_num(e)?;
                (!r.is_integer()).then(|| (id, r.clone()))
            }
            _ => None,
        }
    };
    let q_ratio = Q::from_integer(q.into());
    let mut cur = expr;
    for _ in 0..64 {
        let mut hit = None;
        for id in crate::base::walk::post_order_ids(arena, cur) {
            if let ExprNode::Mul(children) = arena.node(id) {
                let p1 = children.iter().find_map(|&c| radical_of(arena, c, b1));
                let p2 = children.iter().find_map(|&c| radical_of(arena, c, b2));
                if let (Some(p1), Some(p2)) = (p1, p2) {
                    hit = Some((id, children.clone(), p1, p2));
                    break;
                }
            }
        }
        let Some((id, children, (n1, e1), (n2, e2))) = hit else {
            break;
        };
        let sum = &e1 + &e2;
        if !sum.is_integer() {
            return None;
        }
        let mut rest: SmallVec<[ExprId; 6]> = children
            .iter()
            .copied()
            .filter(|&c| c != n1 && c != n2)
            .collect();
        let k = arena.num_ratio(&e1 * &q_ratio);
        rest.push(arena.pow(t, k));
        let s = arena.num_ratio(sum);
        rest.push(arena.pow(b2, s));
        let replacement = arena.mul(&rest);
        cur = arena.subs_structural(cur, id, replacement);
    }
    let leftover = crate::base::walk::post_order_ids(arena, cur)
        .into_iter()
        .any(|id| radical_of(arena, id, b1).is_some() || radical_of(arena, id, b2).is_some());
    (!leftover).then_some(cur)
}

/// `∫ R(x, B^{p₁/q₁}, …) dx` for one base `B = (a·x + b)/(c·x + d)` (a
/// linear `a + b·x` among them), and `∫ R(x, B₁^{e₁}·B₂^{e₂}, …) dx` for
/// two bases whose quotient is such a function, with the radicals in
/// products whose exponents sum to integers (`√(1 + x)/√(1 − x)`,
/// `x^{−7/2}/√(a + b·x)`, `(a + b/x)^{3/2}`): the substitution
/// `t = B^{1/q}` (`t = B₁^{1/q}·B₂^{−1/q}`), `q` the least common
/// denominator of the exponents, makes the integrand rational,
/// `x = (d·tᵠ − b)/(a − c·tᵠ)` and `dx = q·t^{q−1}·(a·d − b·c)/(a − c·tᵠ)²·dt`
/// (SymPy's `sqrt_fractional_linear_rule` for one base; the pairing is the
/// classical one of Chebyshev's binomial integrals).
///
/// Exact for principal powers wherever `t` is: `tᵠ = B` (`B₁/B₂`) holds
/// as an identity of complex numbers, and every radical of the integrand is
/// an integer power of `t` times a rational function of `x` (see
/// [`pair_radicals`]), so the answer `G(t(x))` is an antiderivative at
/// complex values of the integrand too, and needs no `|·|` or `sign`
/// (SymPy writes `√(x + 1)/√(1 − x)` as `√((x + 1)/(1 − x))`, which differs
/// in sign for `x > 1`).  Where the integrand is real, `t` is real, and the
/// rational antiderivative `G` (logarithms at its poles, `atan` of
/// polynomials) is continuous between the poles of the `t`-integrand,
/// which are singularities of the integrand or `x = ∞`.
///
/// An integrand `xᵐ·h(xⁿ)` (`n ≥ 2` the gcd of the exponents of `x` in
/// `h`, `m` rational) is first written in `w = xⁿ`, `xᵐ·dx =
/// w^{(m+1−n)/n}·dw/n` ([`power_substituted`]): the binomial
/// `xᵐ·(a + b·xⁿ)ᵖ` becomes a radical of a linear function of `w` when
/// `(m + 1)/n` is an integer, and a pair with the radical of `w` itself when
/// `(m + 1)/n + p` is — Chebyshev's two elementary cases besides an
/// integer `p` (Gradshteyn–Ryzhik 2.202; Rubi 1.1.3.2).  The radicals of
/// `w` stand for powers of `x` (`w^{1/q}` is `x^{n/q}`, not `(xⁿ)^{1/q}`),
/// and the identities used hold for those as they do for principal powers
/// of `w`.  `t` then has the radical of `w` in its numerator, so that
/// `x = 0`, where `x^{−1}`-type powers change sign but the integrand may
/// be continuous (`√(a + b·x²)`), maps to the finite `t = 0`, not across
/// `t = ∞` (where an `atan` or a pair of complex logarithms of `G` jumps).
fn try_mobius_radical_substitution(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    let back = match power_substituted(arena, expr, var, var_sym) {
        Some((f_w, w, n)) => {
            let ExprNode::Symbol(w_sym) = *arena.node(w) else {
                return None;
            };
            let g = mobius_core(arena, f_w, w, w_sym, true)?;
            unmap_power(arena, g, w, var, n)
        }
        None => match mobius_core(arena, expr, var, var_sym, false) {
            Some(g) => g,
            None => mobius_core_split(arena, expr, var, var_sym)?,
        },
    };
    let back = crate::transforms::eval::eval(arena, back);
    if candidate_rejected(arena, expr, back, var, var_sym) {
        tracing::debug!("integrate: rejecting unverified Möbius-radical closed form");
        return None;
    }
    Some(back)
}

/// [`mobius_core`] with every radical base written as monic linear factors
/// times a locally constant coefficient: `(a + b·x)ᵉ = K·(x + a/b)ᵉ` and,
/// for a quadratic `Q = α·x² + β·x + γ` whose discriminant is a square
/// `s²` in `ℚ[parameters]`, `Qᵉ = K·(x − r₁)ᵉ·(x − r₂)ᵉ` with
/// `r = (−β ± s)/(2α)` (`√(b·x + c·x²)`, `√(d² − e²·x²)`,
/// `√(c − a²·c·x²)`; Rubi 1.2.1, 7.3.6).  `K = Qᵉ·∏(x − rᵢ)^{−e}` has
/// derivative 0: it is integrated as a parameter and put back (Rubi's
/// "piecewise constant extraction").  For real parameters it changes
/// value only at a root, where the integrand passes from real to complex
/// — except at a double root (`√(a² + 2a·b·x + b²·x²) = K·(x + a/b)`,
/// real on both sides): there the antiderivative `K·G` is made continuous
/// by `G(r) = 0`, as the `P·|g|` rule does, or refused when `G(r)` is not
/// finite.
fn mobius_core_split(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    let radicals = radical_powers(arena, expr, var_sym)?;
    if radicals.is_empty() {
        return None;
    }
    let mut replaced: Vec<(ExprId, ExprId)> = Vec::new();
    let mut constants: Vec<(ExprId, ExprId)> = Vec::new();
    let mut double_roots: Vec<(ExprId, ExprId)> = Vec::new();
    let two = arena.int(2);
    for (node, base, e) in &radicals {
        if replaced.iter().any(|&(n, _)| n == *node) {
            continue;
        }
        // Monic linear factors of the base, with multiplicities.
        let mut factors: SmallVec<[(ExprId, i64); 2]> = SmallVec::new();
        let mut double: Option<ExprId> = None;
        if let Some((alpha, beta)) = symbolic_linear_coeff_of(arena, *base, var, var_sym) {
            let r = arena.div(beta, alpha);
            let r = arena.neg(r);
            let r = crate::transforms::eval::eval(arena, r);
            factors.push((arena.sub(var, r), 1));
        } else if let Some((a2, a1, a0)) = symbolic_quadratic_coeffs(arena, *base, var, var_sym) {
            let a1_sq = arena.pow(a1, two);
            let four = arena.int(4);
            let prod = arena.mul(&[four, a2, a0]);
            let disc = arena.sub(a1_sq, prod);
            let disc = crate::transforms::expand::expand(arena, disc);
            let disc = crate::transforms::eval::eval(arena, disc);
            let s = crate::calculus::risch::polynomial_sqrt(arena, disc)?;
            let two_a2 = arena.mul(&[two, a2]);
            let neg_a1 = arena.neg(a1);
            if arena.is_zero_structural(s) {
                let r = arena.div(neg_a1, two_a2);
                let r = crate::transforms::eval::eval(arena, r);
                factors.push((arena.sub(var, r), 2));
                double = Some(r);
            } else {
                for sign_s in [arena.neg(s), s] {
                    let num = arena.add(&[neg_a1, sign_s]);
                    let r = arena.div(num, two_a2);
                    let r = crate::transforms::eval::eval(arena, r);
                    factors.push((arena.sub(var, r), 1));
                }
            }
        } else {
            return None;
        }
        let e_id = arena.num_ratio(e.clone());
        let mut k_parts: SmallVec<[ExprId; 3]> = SmallVec::new();
        k_parts.push(arena.pow(*base, e_id));
        let mut new_parts: SmallVec<[ExprId; 3]> = SmallVec::new();
        for &(l, mult) in &factors {
            let em = e * Q::from_integer(mult.into());
            let em_id = arena.num_ratio(em.clone());
            new_parts.push(arena.pow(l, em_id));
            let neg_em = arena.num_ratio(-em);
            k_parts.push(arena.pow(l, neg_em));
        }
        let k = arena.mul(&k_parts);
        let k = crate::transforms::eval::eval(arena, k);
        if k != arena.one {
            let kappa = arena.symbol(&format!("__mk{}", constants.len()));
            constants.push((kappa, k));
            if let Some(r) = double {
                double_roots.push((kappa, r));
            }
            new_parts.push(kappa);
        } else if double.is_some() {
            return None;
        }
        replaced.push((*node, arena.mul(&new_parts)));
    }
    if constants.is_empty() {
        return None;
    }
    let rewritten = arena.subs_map_structural(expr, &replaced);
    let rewritten = crate::transforms::eval::eval(arena, rewritten);
    let mut g = mobius_core(arena, rewritten, var, var_sym, false)?;
    for &(kappa, r) in &double_roots {
        let g1 = crate::transforms::diff::diff(arena, g, kappa);
        let at = arena.subs_structural(g1, var, r);
        let at = crate::transforms::eval::eval(arena, at);
        if contains_var(arena, at, var_sym)
            || contains_non_finite(arena, at)
            || crate::base::walk::has_unevaluated(arena, at)
        {
            return None;
        }
        let shift = arena.mul(&[kappa, at]);
        g = arena.sub(g, shift);
    }
    for &(kappa, k) in &constants {
        g = arena.subs_structural(g, kappa, k);
    }
    Some(g)
}

/// `(f_w, w, n)` with `∫ f dx = ∫ f_w dw` at `w = xⁿ`, `n ≥ 2`, for
/// `f = xᵐ·h` where `x` occurs in `h` only in integer powers `xᵏ` with
/// `n | k` (`m` rational, the exponent of a top-level factor `xᵐ`, or 0):
/// `f_w = w^{(m+1−n)/n}·h(w)/n`.  `None` when `n < 2` or `h` has `x`
/// elsewhere.
fn power_substituted(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<(ExprId, ExprId, i64)> {
    use num_integer::Integer;
    let monomial_exponent = |arena: &Arena, f: ExprId| -> Option<Q> {
        if f == var {
            return Some(Q::one());
        }
        match *arena.node(f) {
            ExprNode::Pow(b, e) if b == var => arena.as_num(e).cloned(),
            _ => None,
        }
    };
    let (m, rest): (Q, SmallVec<[ExprId; 6]>) = match arena.node(expr) {
        ExprNode::Mul(children) => {
            let children = children.clone();
            let mut m = Q::zero();
            let mut rest = SmallVec::new();
            for c in children {
                match monomial_exponent(arena, c) {
                    Some(k) => m += k,
                    None => rest.push(c),
                }
            }
            (m, rest)
        }
        _ => (Q::zero(), SmallVec::from_slice(&[expr])),
    };
    let mut g = num_bigint::BigInt::zero();
    let mut found = false;
    for &r in &rest {
        for id in crate::base::walk::post_order_ids(arena, r) {
            let node = arena.node(id);
            if let ExprNode::Pow(b, e) = *node
                && b == var
            {
                let k = arena.as_num(e)?;
                if !k.is_integer() {
                    return None;
                }
                g = g.gcd(&k.to_integer());
                found = true;
                continue;
            }
            let mut bare = false;
            node.for_each_child(|c| bare |= c == var);
            if bare {
                return None;
            }
        }
    }
    let n = i64::try_from(&g).ok()?;
    if !found || !(2..=12).contains(&n) {
        return None;
    }
    let w = arena.symbol("__mw");
    let n_q = Q::from_integer(n.into());
    let mut pairs: Vec<(ExprId, ExprId)> = Vec::new();
    for &r in &rest {
        for id in crate::base::walk::post_order_ids(arena, r) {
            if let ExprNode::Pow(b, e) = *arena.node(id)
                && b == var
                && let Some(k) = arena.as_num(e).cloned()
                && !pairs.iter().any(|&(old, _)| old == id)
            {
                let k_w = arena.num_ratio(k / &n_q);
                let w_k = arena.pow(w, k_w);
                pairs.push((id, w_k));
            }
        }
    }
    let mut factors: SmallVec<[ExprId; 6]> = SmallVec::new();
    for &r in &rest {
        factors.push(arena.subs_map_structural(r, &pairs));
    }
    let e_w = arena.num_ratio((&m + Q::one() - &n_q) / &n_q);
    factors.push(arena.pow(w, e_w));
    factors.push(arena.rational(1, n));
    let f_w = arena.mul(&factors);
    if contains_var(arena, f_w, var_sym) {
        return None;
    }
    Some((f_w, w, n))
}

/// `e` in `w = xⁿ` written in `x`: `wʳ → x^{n·r}` (the radicals of
/// [`power_substituted`]'s `w` are powers of `x`), then `w → xⁿ`.
fn unmap_power(arena: &mut Arena, e: ExprId, w: ExprId, var: ExprId, n: i64) -> ExprId {
    let n_q = Q::from_integer(n.into());
    let mut pairs: Vec<(ExprId, ExprId)> = Vec::new();
    for id in crate::base::walk::post_order_ids(arena, e) {
        if let ExprNode::Pow(b, r) = *arena.node(id)
            && b == w
            && let Some(r) = arena.as_num(r).cloned()
            && !pairs.iter().any(|&(old, _)| old == id)
        {
            let nr = arena.num_ratio(r * &n_q);
            let x_nr = arena.pow(var, nr);
            pairs.push((id, x_nr));
        }
    }
    let e = arena.subs_map_structural(e, &pairs);
    let n_id = arena.int(n);
    let x_n = arena.pow(var, n_id);
    arena.subs_structural(e, w, x_n)
}

/// The substitution of [`try_mobius_radical_substitution`] in the variable
/// `var` (`x`, or `w = xⁿ`): the antiderivative in `var`, not yet checked.
/// `w_first`: in a pair, the radical of `var` itself goes into the
/// numerator of `t`.
fn mobius_core(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
    w_first: bool,
) -> Option<ExprId> {
    use num_integer::Integer;
    let radicals = radical_powers(arena, expr, var_sym)?;
    let mut bases: SmallVec<[ExprId; 2]> = SmallVec::new();
    let mut q_lcm = num_bigint::BigInt::one();
    for (_, b, e) in &radicals {
        if !bases.contains(b) {
            bases.push(*b);
        }
        q_lcm = q_lcm.lcm(e.denom());
    }
    if bases.is_empty() || bases.len() > 2 {
        return None;
    }
    let q = i64::try_from(&q_lcm).ok()?;
    if !(2..=MAX_MOBIUS_ROOT).contains(&q) {
        return None;
    }
    let t = arena.symbol("__mt");
    let ExprNode::Symbol(t_sym) = *arena.node(t) else {
        return None;
    };
    let q_ratio = Q::from_integer(q.into());
    let inv_q = arena.rational(1, q);
    let (ratio, sub, t_of_x) = if let [b] = bases[..] {
        let mut sub = expr;
        for (node, _, e) in &radicals {
            let k = arena.num_ratio(e * &q_ratio);
            let tk = arena.pow(t, k);
            sub = arena.subs_structural(sub, *node, tk);
        }
        (b, sub, arena.pow(b, inv_q))
    } else {
        let (b1, b2) = if w_first && bases[1] == var {
            (bases[1], bases[0])
        } else {
            (bases[0], bases[1])
        };
        let sub = pair_radicals(arena, expr, b1, b2, t, q)?;
        let neg_inv_q = arena.rational(-1, q);
        let r1 = arena.pow(b1, inv_q);
        let r2 = arena.pow(b2, neg_inv_q);
        (arena.div(b1, b2), sub, arena.mul(&[r1, r2]))
    };
    let Mobius { a, b, c, d } = mobius_parts(arena, ratio, var, var_sym)?;
    let ad = arena.mul(&[a, d]);
    let bc = arena.mul(&[b, c]);
    let det = arena.sub(ad, bc);
    let det = crate::transforms::eval::eval(arena, det);
    if arena.is_zero_structural(det) {
        return None;
    }
    let q_id = arena.int(q);
    let t_q = arena.pow(t, q_id);
    let d_tq = arena.mul(&[d, t_q]);
    let x_num = arena.sub(d_tq, b);
    let c_tq = arena.mul(&[c, t_q]);
    let x_den = arena.sub(a, c_tq);
    let x_of_t = arena.div(x_num, x_den);
    let qm1 = arena.int(q - 1);
    let t_qm1 = arena.pow(t, qm1);
    let two = arena.int(2);
    let den_sq = arena.pow(x_den, two);
    let dx_num = arena.mul(&[q_id, t_qm1, det]);
    let dx_dt = arena.div(dx_num, den_sq);
    let sub = arena.subs_structural(sub, var, x_of_t);
    if contains_var(arena, sub, var_sym) {
        return None;
    }
    let integrand = arena.mul(&[sub, dx_dt]);
    let integrand = rational_in_factored_form(arena, integrand, t);
    if !is_rational_in(arena, integrand, t_sym) {
        return None;
    }
    tracing::debug!(integrand_t = %arena.display(integrand), "integrate: Möbius radical substitution");
    let res = integrate_nested_generic(arena, integrand, t)?;
    let res = analytic_logs(arena, res, t_sym)?;
    let res = logs_off_the_cut_at(arena, res, t, t_of_x, var);
    Some(arena.subs_structural(res, t, t_of_x))
}

/// `g` with every `ln(p(t))` whose argument is a negative number at
/// `t₀ = t(0)` (the image of `x = 0`, when that is a number) written
/// `ln(−p(t))`: the same derivative, and continuous where the path `t(x)`
/// passes `t₀`.  `analytic_logs` writes the `ln|t − 1|` of `∫ dt/(1 − t²)`
/// as `ln(t − 1)`, whose argument at `t = 0` is on the cut: for
/// `∫ dx/√(x² + a)` (`t = x/√(x² + a)`, `t(0) = 0`) the answer jumped by
/// `iπ` at `x = 0` for every non-real `a` and for real `a < 0` (where the
/// integrand is imaginary on `|x| < √−a`); on the real path of a real
/// integrand only a constant changes.
fn logs_off_the_cut_at(
    arena: &mut Arena,
    g: ExprId,
    t: ExprId,
    t_of_x: ExprId,
    var: ExprId,
) -> ExprId {
    let zero = arena.zero;
    let t0 = arena.subs_structural(t_of_x, var, zero);
    let t0 = crate::transforms::eval::eval(arena, t0);
    if arena.as_num(t0).is_none() {
        return g;
    }
    let mut out = g;
    for id in crate::base::walk::post_order_ids(arena, g) {
        let ExprNode::Ln(p) = *arena.node(id) else {
            continue;
        };
        if !crate::base::walk::contains(arena, p, t) {
            continue;
        }
        let at = arena.subs_structural(p, t, t0);
        let at = crate::transforms::eval::eval(arena, at);
        if arena.as_num(at).is_some_and(|v| v.is_negative()) {
            let minus_p = arena.neg(p);
            let flipped = arena.ln(minus_p);
            out = arena.subs_structural(out, id, flipped);
        }
    }
    out
}

/// `∫ f` with every `e^{k·atanh(u)}` (`k` rational) written
/// `(1 + u)^{k/2}·(1 − u)^{−k/2}`: `atanh u = ½·(ln(1 + u) − ln(1 − u))`
/// (DLMF 4.37.24; the principal branches symplex and SymPy evaluate, on
/// the cuts `|u| > 1` too), so the two agree as complex numbers.  The
/// rewritten integrand is algebraic — rational for an even `k` — and goes
/// through the whole pipeline (Rubi 7.3.6, and 7.4.2 with `acoth u =
/// atanh(1/u)`; Rubi's rules rewrite `e^{n·atanh(a·x)}` the same way).
fn try_exp_atanh_rewrite(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    let mut rewritten = expr;
    for id in crate::base::walk::post_order_ids(arena, expr) {
        let ExprNode::Exp(arg) = *arena.node(id) else {
            continue;
        };
        if !contains_var(arena, arg, var_sym) {
            continue;
        }
        let (k, u) = match arena.node(arg).clone() {
            ExprNode::Atanh(u) => (Q::one(), u),
            ExprNode::Mul(fs) if fs.len() == 2 => {
                let (k, u) = match (arena.as_num(fs[0]), arena.node(fs[1])) {
                    (Some(k), ExprNode::Atanh(u)) => (k.clone(), *u),
                    _ => continue,
                };
                (k, u)
            }
            _ => continue,
        };
        let half_k = arena.num_ratio(&k / Q::from_integer(2.into()));
        let neg_half_k = arena.num_ratio(-&k / Q::from_integer(2.into()));
        let one = arena.one;
        let one_plus = arena.add(&[one, u]);
        let one_minus = arena.sub(one, u);
        let p1 = arena.pow(one_plus, half_k);
        let p2 = arena.pow(one_minus, neg_half_k);
        let repl = arena.mul(&[p1, p2]);
        rewritten = arena.subs_structural(rewritten, id, repl);
    }
    if rewritten == expr {
        return None;
    }
    let rewritten = crate::transforms::eval::eval(arena, rewritten);
    let res = integrate_nested(arena, rewritten, var)?;
    if candidate_rejected(arena, expr, res, var, var_sym) {
        tracing::debug!("integrate: rejecting unverified e^(k·atanh u) closed form");
        return None;
    }
    Some(res)
}

/// Rewrite `sinh`/`cosh`/`tanh` in terms of `exp` and retry (e.g.
/// `1/cosh x = 2e^x/(e^{2x}+1)` → `2 atan(e^x)`).
fn try_hyperbolic_to_exp(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    let order = crate::base::walk::post_order_ids(arena, expr);
    let has_hyp = order.iter().any(|&id| {
        matches!(
            arena.node(id),
            ExprNode::Sinh(_) | ExprNode::Cosh(_) | ExprNode::Tanh(_)
        ) && contains_var(arena, id, var_sym)
    });
    if !has_hyp {
        return None;
    }
    let rewritten = arena.rewrite_as_exp_expr(expr);
    if rewritten == expr {
        return None;
    }
    let rewritten = crate::transforms::eval::eval(arena, rewritten);
    try_exp_rational_substitution(arena, rewritten, var, var_sym).or_else(|| {
        let r = try_exp_linear_substitution(arena, rewritten, var, var_sym)?;
        (!candidate_rejected(arena, expr, r, var, var_sym)).then_some(r)
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Algebraic reductions: P(x)·Q(x)^{k/2}
// ═══════════════════════════════════════════════════════════════════════════

/// Split the dependent factors into a rational-coefficient polynomial `P`
/// and a single factor `Q^{k/2}` (`k` odd, `Q` of degree 1 or 2).
fn split_poly_and_half_power(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
) -> Option<(crate::poly::Poly, crate::poly::Poly, ExprId, Q)> {
    let mut half: Option<(ExprId, Q)> = None;
    let mut poly = crate::poly::Poly::from_int(1);
    for &d in dependent {
        if let ExprNode::Pow(base, e) = arena.node(d).clone()
            && let Some(r) = arena.as_num(e).cloned()
            && *r.denom() == num_bigint::BigInt::from(2)
        {
            if half.is_some() {
                return None;
            }
            half = Some((base, r));
            continue;
        }
        let p = crate::poly::polybridge::expr_to_poly(arena, d, var)?;
        poly = poly.mul(&p);
    }
    let (q_expr, k) = half?;
    let q = crate::poly::polybridge::expr_to_poly(arena, q_expr, var)?;
    let qd = q.degree()?;
    if !(1..=2).contains(&qd) {
        return None;
    }
    Some((poly, q, q_expr, k))
}

/// `∫ P(x)·Q(x)^{k/2} dx` for `k ∈ {−1, 1, 3}`: write `P·Q^{(k+1)/2} = P̃`, then
/// find a polynomial `R` and constant `c` with
/// `P̃ = (2R'Q + RQ')/2 + c`, so that the integral is `R√Q + c∫ dx/√Q`.
fn try_poly_times_half_power(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    let (p, q, q_expr, k) = split_poly_and_half_power(arena, dependent, var)?;
    let two = Q::from_integer(2.into());
    let k2 = &k * &two; // odd integer as rational
    let k2: i64 = k2.to_integer().to_string().parse().ok()?;
    if !matches!(k2, -1 | 1 | 3) {
        return None;
    }
    // P̃ = P · Q^{(k+1)/2}
    let mut p_tilde = p;
    for _ in 0..((k2 + 1) / 2) {
        p_tilde = p_tilde.mul(&q);
    }
    let pd = p_tilde.degree()?;
    let qd = q.degree()?;
    if pd == 0 && qd == 2 {
        return None; // plain 1/√Q: handled by the standard forms
    }
    // Unknowns: R = r_0 + … + r_m x^m (m = pd − qd + 1, or pd for linear Q), c.
    let m: i64 = if qd == 2 { pd as i64 - 1 } else { pd as i64 };
    let with_c = qd == 2;
    let n_unknowns = (m + 1).max(0) as usize + usize::from(with_c);
    let n_eq = pd + 1;
    if n_unknowns != n_eq {
        return None;
    }
    // Build the linear system: coefficient of x^j in (2R'Q + RQ')/2 + c.
    let q_prime = q.derivative();
    let mut rows: Vec<Vec<Q>> = vec![vec![Q::zero(); n_unknowns + 1]; n_eq];
    for i in 0..=(m.max(-1)) {
        if i < 0 {
            break;
        }
        let iu = i as usize;
        // basis monomial x^i for R
        let mut mono = vec![Q::zero(); iu + 1];
        mono[iu] = Q::one();
        let r_i = crate::poly::Poly::from_coeffs(mono);
        let term = r_i
            .derivative()
            .mul(&q)
            .scale(&two)
            .add(&r_i.mul(&q_prime))
            .scale(&Q::new(1.into(), 2.into()));
        for (j, row) in rows.iter_mut().enumerate() {
            row[iu] = term.coeff(j);
        }
    }
    if with_c {
        rows[0][n_unknowns - 1] = Q::one();
    }
    for (j, row) in rows.iter_mut().enumerate() {
        row[n_unknowns] = p_tilde.coeff(j);
    }
    let sol = solve_linear_system(rows, n_unknowns)?;
    // Assemble R√Q + c∫1/√Q.
    let r_coeffs: Vec<Q> = sol[..(m.max(-1) + 1) as usize].to_vec();
    let r_poly = crate::poly::Poly::from_coeffs(r_coeffs);
    let r_expr = crate::poly::polybridge::poly_to_expr(arena, &r_poly, var);
    let half = arena.rational(1, 2);
    let sqrt_q = arena.pow(q_expr, half);
    let mut result = arena.mul(&[r_expr, sqrt_q]);
    if with_c {
        let c = sol[n_unknowns - 1].clone();
        if !c.is_zero() {
            let neg_half = arena.rational(-1, 2);
            let inv_sqrt = arena.pow(q_expr, neg_half);
            let base_int = integrate_node(arena, inv_sqrt, var, var_sym, depth - 1);
            if crate::base::walk::has_unevaluated(arena, base_int) {
                return None;
            }
            let c_id = arena.num_ratio(c.clone());
            let c_term = arena.mul(&[c_id, base_int]);
            result = arena.add(&[result, c_term]);
        }
    }
    Some(result)
}

/// Largest degree of `P` in [`try_symbolic_quadratic_radical`].
const MAX_RADICAL_POLY_DEGREE: usize = 6;

/// `∫ P(x)·Q^{±1/2} dx` for `Q = A·x² + B·x + C` whose coefficients are not
/// all rational (parameters, `√2`) and a polynomial `P` over them, by the
/// reduction `Iₙ = ∫ xⁿ/√Q = (xⁿ⁻¹·√Q − (n − ½)·B·Iₙ₋₁ − (n − 1)·C·Iₙ₋₂)/(n·A)`
/// (the derivative of `xⁿ⁻¹·√Q`, as in Gradshteyn–Ryzhik 2.260) down to
/// `I₀ = ln(2√A·√Q + 2A·x + B)/√A`, an antiderivative for every complex
/// `A ≠ 0`, `B`, `C` (its derivative is `√A·(2A·x + B + 2√A·√Q)/(√Q·(…))`
/// whatever the branches), and `P·√Q = P·Q/√Q`.  `A = 0` is a degenerate
/// case of the wrap.  Up to 0.46 `∫ dx/√(a + b·x + c·x²)`, `∫ √(a + b·x +
/// c·x²) dx` and `∫ x·asin(a·x) dx` (by parts) stayed unevaluated (SymPy's
/// test_integrals and test_manual; SymPy gives the same logarithm).
fn try_symbolic_quadratic_radical(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    if depth < 2 {
        return None;
    }
    let mut radical: Option<(ExprId, bool)> = None;
    let mut rest: SmallVec<[ExprId; 4]> = SmallVec::new();
    for &d in dependent {
        if let ExprNode::Pow(b, e) = *arena.node(d)
            && contains_var(arena, b, var_sym)
            && let Some(r) = arena.as_num(e)
            && *r.denom() == num_bigint::BigInt::from(2)
            && (*r.numer() == num_bigint::BigInt::from(1)
                || *r.numer() == num_bigint::BigInt::from(-1))
        {
            if radical.is_some() {
                return None;
            }
            radical = Some((b, r.is_positive()));
        } else {
            rest.push(d);
        }
    }
    let (q, positive) = radical?;
    if crate::poly::polybridge::expr_to_poly(arena, q, var).is_some() {
        return None;
    }
    let (a2, a1, a0) = symbolic_quadratic_coeffs(arena, q, var, var_sym)?;
    if arena.is_zero_structural(a2) {
        return None;
    }
    // A perfect square (`√((a + x)²)`): the logarithm degenerates (its
    // argument vanishes where `a + x` is negative and real).
    let disc = {
        let b2 = arena.mul(&[a1, a1]);
        let four = arena.int(4);
        let ac = arena.mul(&[four, a2, a0]);
        let d = arena.sub(b2, ac);
        let d = crate::transforms::expand::expand(arena, d);
        crate::transforms::eval::eval(arena, d)
    };
    if arena.is_zero_structural(disc) {
        return None;
    }
    let p = if rest.is_empty() {
        arena.one
    } else {
        arena.mul(&rest)
    };
    let mut coeffs = poly_coeffs_in(arena, p, var, var_sym, MAX_RADICAL_POLY_DEGREE)?;
    if positive {
        // P·Q as the numerator over √Q.
        let mut pq = vec![arena.zero; coeffs.len() + 2];
        for (i, &c) in coeffs.iter().enumerate() {
            for (j, &qc) in [a0, a1, a2].iter().enumerate() {
                let t = arena.mul(&[c, qc]);
                pq[i + j] = arena.add(&[pq[i + j], t]);
            }
        }
        coeffs = pq;
    }
    let half = arena.rational(1, 2);
    let sqrt_q = arena.pow(q, half);
    let sqrt_a = arena.pow(a2, half);
    let two = arena.int(2);
    let ln_arg = {
        let t1 = arena.mul(&[two, sqrt_a, sqrt_q]);
        let t2 = arena.mul(&[two, a2, var]);
        arena.add(&[t1, t2, a1])
    };
    let ln = arena.ln(ln_arg);
    let i0 = arena.div(ln, sqrt_a);
    let mut ints: Vec<ExprId> = vec![i0];
    for n in 1..coeffs.len() {
        let n_i = i64::try_from(n).ok()?;
        let n_minus_1 = arena.int(n_i - 1);
        let x_pow = arena.pow(var, n_minus_1);
        let lead = arena.mul(&[x_pow, sqrt_q]);
        let k1 = arena.rational(2 * n_i - 1, 2);
        let t1 = arena.mul(&[k1, a1, ints[n - 1]]);
        let mut num = arena.sub(lead, t1);
        if n >= 2 {
            let k2 = arena.int(n_i - 1);
            let t2 = arena.mul(&[k2, a0, ints[n - 2]]);
            num = arena.sub(num, t2);
        }
        let n_e = arena.int(n_i);
        let den = arena.mul(&[n_e, a2]);
        ints.push(arena.div(num, den));
    }
    let mut terms: Vec<ExprId> = Vec::with_capacity(coeffs.len());
    for (&c, &i) in coeffs.iter().zip(&ints) {
        if !arena.is_zero_structural(c) {
            terms.push(arena.mul(&[c, i]));
        }
    }
    tracing::debug!("integrate: P(x)·Q^(±1/2) for a symbolic quadratic Q");
    let sum = arena.add(&terms);
    Some(crate::transforms::eval::eval(arena, sum))
}

/// Gaussian elimination over ℚ on an augmented matrix (`n` unknowns).
fn solve_linear_system(mut rows: Vec<Vec<Q>>, n: usize) -> Option<Vec<Q>> {
    let m = rows.len();
    let mut pivot_row = 0;
    let mut pivot_cols: Vec<usize> = Vec::new();
    for col in 0..n {
        let Some(p) = (pivot_row..m).find(|&r| !rows[r][col].is_zero()) else {
            continue;
        };
        rows.swap(pivot_row, p);
        let inv = Q::one() / rows[pivot_row][col].clone();
        for v in rows[pivot_row].iter_mut() {
            *v = &*v * &inv;
        }
        let pivot = rows[pivot_row].clone();
        for (r, row) in rows.iter_mut().enumerate() {
            if r != pivot_row && !row[col].is_zero() {
                let f = row[col].clone();
                for (c, cell) in row.iter_mut().enumerate() {
                    let sub = &pivot[c] * &f;
                    *cell = &*cell - &sub;
                }
            }
        }
        pivot_cols.push(col);
        pivot_row += 1;
        if pivot_row == m {
            break;
        }
    }
    // Inconsistent rows?
    for row in rows.iter().skip(pivot_row) {
        if !row[n].is_zero() {
            return None;
        }
    }
    let mut sol = vec![Q::zero(); n];
    for (r, &col) in pivot_cols.iter().enumerate() {
        sol[col] = rows[r][n].clone();
    }
    Some(sol)
}

/// `∫ x^{−n} Q(x)^{−1/2} dx` (`n ≥ 1`, `Q` quadratic) via `x = 1/t`:
/// `= −∫ t^{n−2} (c t² + b t + a)^{−1/2} dt` for `Q = a x² + b x + c`.
fn try_reciprocal_sqrt_substitution(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    if dependent.len() != 2 {
        return None;
    }
    let mut n_neg: Option<i64> = None;
    let mut half: Option<(ExprId, Q)> = None;
    for &d in dependent {
        if let ExprNode::Pow(base, e) = arena.node(d).clone()
            && let Some(r) = arena.as_num(e).cloned()
        {
            if base == var && r.is_integer() && r.is_negative() {
                n_neg = Some(-r.to_integer().to_string().parse::<i64>().ok()?);
                continue;
            }
            if *r.denom() == num_bigint::BigInt::from(2) && contains_var(arena, base, var_sym) {
                half = Some((base, r));
                continue;
            }
        }
        return None;
    }
    let n = n_neg?;
    let (q_expr, k) = half?;
    if k != Q::new((-1).into(), 2.into()) || !(1..=4).contains(&n) {
        return None;
    }
    let q = crate::poly::polybridge::expr_to_poly(arena, q_expr, var)?;
    if q.degree()? != 2 {
        return None;
    }
    // Q(1/t) = (a + b t + c t²)/t²  ⇒  Q^{-1/2} = |t| (c t² + b t + a)^{-1/2};
    // x^{-n} = t^n, dx = −dt/t²  ⇒  integrand −|t|·t^{n−2}·(…)^{-1/2}
    //   = −sign(x)·t^{n−1}·(c t² + b t + a)^{-1/2}.
    // sign(x) is locally constant, so F(x) = sign(x)·G(1/x) with
    // G(t) = ∫ −t^{n−1} (c t² + b t + a)^{-1/2} dt.
    let t = arena.symbol("__rt");
    let t_sym = match arena.node(t) {
        ExprNode::Symbol(s) => *s,
        _ => return None,
    };
    let coeffs = q.coeffs().to_vec(); // [c, b, a]
    let reversed = crate::poly::Poly::from_coeffs(coeffs.iter().rev().cloned().collect());
    let q_rev = crate::poly::polybridge::poly_to_expr(arena, &reversed, t);
    let neg_half = arena.rational(-1, 2);
    let q_rev_pow = arena.pow(q_rev, neg_half);
    let t_pow_id = arena.int(n - 1);
    let t_pow = arena.pow(t, t_pow_id);
    let integrand_t = arena.mul(&[t_pow, q_rev_pow]);
    let neg_integrand = arena.neg(integrand_t);
    let res_t = integrate_node(arena, neg_integrand, t, t_sym, depth - 1);
    if crate::base::walk::has_unevaluated(arena, res_t) {
        return None;
    }
    let inv_x = arena.pow(var, arena.neg_one);
    let g_of_x = arena.subs_structural(res_t, t, inv_x);
    let sgn = arena.sign(var);
    let result = arena.mul(&[sgn, g_of_x]);
    Some(crate::transforms::eval::eval(arena, result))
}

/// `tanᵐ(g)·sec²(g)` → `tanᵐ⁺¹(g)/((m+1)·g')` and `secⁿ(g)·tan(g)` →
/// `secⁿ(g)/(n·g')` for linear `g`.
fn try_tan_sec_patterns(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    if dependent.len() != 2 {
        return None;
    }
    // Identify tan^m(g) and cos^{-n}(g).
    let mut tan_part: Option<(ExprId, Q)> = None;
    let mut sec_part: Option<(ExprId, Q)> = None;
    for &d in dependent {
        let (base, e) = if let ExprNode::Pow(b, e) = arena.node(d).clone() {
            (b, arena.as_num(e)?.clone())
        } else {
            (d, Q::one())
        };
        match arena.node(base).clone() {
            ExprNode::Tan(g) if e.is_integer() && e.is_positive() => tan_part = Some((g, e)),
            ExprNode::Cos(g) if e.is_integer() && e.is_negative() => sec_part = Some((g, -e)),
            _ => return None,
        }
    }
    let (g, m) = tan_part?;
    let (g2, n) = sec_part?;
    if g != g2 {
        return None;
    }
    let (a_expr, _) = symbolic_linear_coeff_of(arena, g, var, var_sym)?;
    let two = Q::from_integer(2.into());
    if n == two {
        // ∫ tan^m sec² = tan^{m+1}/(m+1)
        let m1 = &m + &Q::one();
        let m1_id = arena.num_ratio(m1.clone());
        let tan_g = arena.tan(g);
        let tp = arena.pow(tan_g, m1_id);
        let denom = arena.mul(&[m1_id, a_expr]);
        return Some(arena.div(tp, denom));
    }
    if m == Q::one() {
        // ∫ secⁿ tan = secⁿ/n = cos^{-n}/n
        let cos_g = arena.cos(g);
        let neg_n = arena.num_ratio(-n.clone());
        let sec_n = arena.pow(cos_g, neg_n);
        let n_id = arena.num_ratio(n.clone());
        let denom = arena.mul(&[n_id, a_expr]);
        return Some(arena.div(sec_n, denom));
    }
    None
}

// ═══════════════════════════════════════════════════════════════════════════
// Products: three-factor by parts, trig product-to-sum
// ═══════════════════════════════════════════════════════════════════════════

/// `∫ u(x)·g(x)·h(x) dx` by parts with `dv = g·h`: `u` a logarithm, an
/// inverse trigonometric or another by-parts function of the variable
/// first (LIATE), then a polynomial `P` (e.g. `x·eˣ·sin x`).  Up to 0.46
/// only `u = P` was tried: `∫ x·eˣ·ln x dx` needed `∫ Ei(x) dx` that way
/// (SymPy's test_integrals: `(x − 1)·eˣ·ln x − eˣ + Ei(x)`, which `u = ln x`,
/// `v = (x − 1)·eˣ` gives at once; the ODE `y″ + 2y′ + y = ln x` needs it).
fn try_by_parts_poly_times_pair(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    if rational_product(arena, dependent, var_sym) {
        return None;
    }
    // One `exp`/`sin`/`cos`/`sinh`/`cosh` of a linear argument over a
    // rational function is the special-function table's (`∫ eˣ·(x − 1)/x² dx
    // = eˣ/x` there; by parts with `u = x − 1` left `Ei` terms that cancel
    // only numerically).
    if special_times_rational(arena, dependent, var, var_sym) {
        return None;
    }
    // Without a polynomial factor the rule is not for this integrand
    // (`∫ ln x/(x·(ln²x + 2)) dx` is the substitution `u = ln x`).
    let poly_idx = dependent
        .iter()
        .position(|&d| is_polynomial_in(arena, d, var, var_sym))?;
    // A logarithm or inverse function of a linear argument (its `du` is a
    // simple rational function); for `atan((t + ½)⁶)` the remaining `∫ v·du`
    // only fed long failing searches (`∫ atan(x³)/(√x − ½) dx`: 77 ms → 1.1 s).
    let mut order: Vec<(u8, usize)> = dependent
        .iter()
        .enumerate()
        .filter_map(|(i, &d)| {
            let rank = liate_rank(arena, d, var, var_sym);
            (rank <= 2
                && is_by_parts_candidate(arena, d, var, var_sym)
                && single_argument(arena, d)
                    .is_some_and(|g| linear_in_structurally(arena, g, var_sym)))
            .then_some((rank, i))
        })
        .collect();
    order.sort_unstable();
    order.push((3, poly_idx));
    for (_, idx) in order {
        let u = dependent[idx];
        let dv = remaining_product(arena, dependent, idx);
        if special_v_without_route(arena, u, dv, var, var_sym) {
            continue;
        }
        // A polynomial `dv` multiplied out, so that `v` is one polynomial
        // (`x·(1 + x²)` by parts gave `x·(x³/3 + x) − x⁴/12 − x²/2`).
        let dv = if is_polynomial_in(arena, dv, var, var_sym) {
            crate::transforms::expand::expand(arena, dv)
        } else {
            dv
        };
        let v = integrate_node(arena, dv, var, var_sym, depth - 1);
        if crate::base::walk::has_unevaluated(arena, v) || has_root_sum_of_var(arena, v, var_sym) {
            continue;
        }
        let du = crate::transforms::diff::diff(arena, u, var);
        let v_du = arena.mul(&[v, du]);
        let u_dv = arena.mul(&[u, dv]);
        match by_parts_cycle(arena, u, v, u_dv, v_du, var_sym) {
            Cycle::Solved(result) => return Some(result),
            Cycle::Hopeless => continue,
            Cycle::No => {}
        }
        let v_du = crate::transforms::expand::expand(arena, v_du);
        let rest = integrate_node(arena, v_du, var, var_sym, depth - 1);
        if crate::base::walk::has_unevaluated(arena, rest) {
            continue;
        }
        let uv = arena.mul(&[u, v]);
        return Some(arena.sub(uv, rest));
    }
    None
}

/// What [`by_parts_cycle`] found.
enum Cycle {
    /// `∫ v·du` is not a constant multiple of the integrand.
    No,
    /// `∫ v·du = c·I`: `I = u·v/(1 + c)`.
    Solved(ExprId),
    /// `∫ v·du = −I`: by parts says `I = u·v + I`, nothing.
    Hopeless,
}

/// Is the integral left by parts, `∫ v·du`, a constant multiple `c` of the
/// integrand `u·dv` itself?  Then `I = u·v − c·I` gives `I = u·v/(1 + c)`
/// at once (SymPy's `manualintegrate` solves such cyclic parts the same
/// way), and for `c = −1` by parts proves nothing.  Before, the integrator
/// integrated `v·du` — the same integral again — by parts again, one level
/// deeper each time down to the depth limit, every other strategy running
/// at every level: `∫ x²·sign(x)` (`v = x·sign(x)`, `v·du = 2x²·sign(x)`)
/// is the core of `∫ (√x + |x|)/(x + x⁻² + 1) dx`, which took 74 s to be
/// refused (`fuzz_integrate`, 2026-10-01).  The ratio is the canonical
/// quotient, so only a structural multiple is recognised.
fn by_parts_cycle(
    arena: &mut Arena,
    u: ExprId,
    v: ExprId,
    u_dv: ExprId,
    v_du: ExprId,
    var_sym: SymbolId,
) -> Cycle {
    let ratio = arena.div(v_du, u_dv);
    let c = crate::transforms::eval::eval(arena, ratio);
    if contains_var(arena, c, var_sym)
        || contains_non_finite(arena, c)
        || arena.is_zero_structural(c)
    {
        return Cycle::No;
    }
    let one = arena.one;
    let one_plus_c = arena.add(&[one, c]);
    let one_plus_c = crate::transforms::eval::eval(arena, one_plus_c);
    if arena.is_zero_structural(one_plus_c) {
        return Cycle::Hopeless;
    }
    tracing::debug!("by parts: the remaining integral is a multiple of the integrand");
    let uv = arena.mul(&[u, v]);
    Cycle::Solved(arena.div(uv, one_plus_c))
}

/// Is the product of the factors `dependent` a rational function of the
/// variable over ℚ (sums, products and integer powers of the variable and
/// of rational numbers) with a denominator (a polynomial is not the
/// rational integrator's)?  Then by parts is no strategy: Hermite's reduction
/// and Rothstein–Trager are complete there, and by parts takes `u` = a
/// polynomial factor and `dv` = the reciprocal of the rest, whose `v`
/// (logarithms, arctangents, a `RootSum` or the explicit roots of a
/// quartic) leaves `∫ v·du` harder than the integrand.  Up to 0.44 the
/// algebraic remainder `(−x³/6 + …)/(x⁴ − x³ + x² − x + 2)` of
/// `∫ dx/(x⁵ + x + 2)`, which `try_risch_rational` hands to the heuristic
/// integrator, went down 8 levels of by parts and sum distribution before
/// it gave up (0.5 s of its 0.57 s), for the same final answer.  (With
/// parameters by parts still runs: the rational integrator over `ℚ(a, …)`
/// leaves products such as `(a + b·x)ᵐ·(c + d·x)ⁿ` to it.)
fn rational_product(arena: &Arena, dependent: &[ExprId], var_sym: SymbolId) -> bool {
    let has_denominator = dependent.iter().any(|&d| {
        matches!(arena.node(d), ExprNode::Pow(b, e)
            if arena.as_num(*e).is_some_and(|q| q.is_integer() && q.is_negative())
                && contains_var(arena, *b, var_sym))
    });
    has_denominator
        && dependent.iter().all(|&d| {
            crate::base::walk::post_order_ids(arena, d)
                .into_iter()
                .all(|id| match arena.node(id) {
                    ExprNode::Num(_) | ExprNode::Add(_) | ExprNode::Mul(_) | ExprNode::Neg(_) => {
                        true
                    }
                    ExprNode::Pow(_, e) => arena.as_num(*e).is_some_and(|q| q.is_integer()),
                    ExprNode::Symbol(s) => *s == var_sym,
                    _ => false,
                })
        })
}

/// Does `e` hold a `RootSum` that depends on the variable?  As a by-parts
/// `v` it leaves `∫ v·du` with no route (see [`integrate_impl`]: every
/// strategy would take the `ln(x − t)` of its body for a function of `x`
/// alone).
fn has_root_sum_of_var(arena: &Arena, e: ExprId, var_sym: SymbolId) -> bool {
    crate::base::walk::post_order_ids(arena, e)
        .iter()
        .any(|&id| {
            matches!(arena.node(id), ExprNode::RootSum(..)) && contains_var(arena, id, var_sym)
        })
}

/// Is the product of `dependent` one `exp`, `sin`, `cos`, `sinh`, `cosh` of
/// a linear argument times a rational function with a denominator?
fn special_times_rational(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
) -> bool {
    let mut special = 0usize;
    let mut rest: SmallVec<[ExprId; 4]> = SmallVec::new();
    for &f in dependent {
        match LinearSpecial::of(arena, f) {
            Some((_, arg)) if symbolic_linear_coeff_of(arena, arg, var, var_sym).is_some() => {
                special += 1;
            }
            _ => rest.push(f),
        }
    }
    if special != 1 || rest.is_empty() {
        return false;
    }
    let r = arena.mul(&rest);
    if !is_rational_in(arena, r, var_sym) {
        return false;
    }
    let (_, den) = crate::poly::polybridge::as_numer_denom(arena, r);
    contains_var(arena, den, var_sym)
}

/// Would by parts with this `u` and `dv` build a `v` of `Ei`, `Si`/`Ci`,
/// `Shi`/`Chi` (a function `exp`, `sin`, `cos`, `sinh`, `cosh` of a linear
/// argument times a rational function with a denominator in the variable)
/// only to meet `∫ v·du` with a `du` that is no polynomial, which no route
/// integrates (`Si(x − r)·R(x)` for a proper rational `R`)?  Building that `v`
/// over the complex roots of `x⁸ + 1` and verifying it took 10–30 ms in
/// failing searches (`∫ cos(√x)·atan(x²) dx`: 9 ms in 0.43, 118 ms in 0.45;
/// agent B14's fuzz units).  Checked on the structure, before `v` is built.
fn special_v_without_route(
    arena: &mut Arena,
    u: ExprId,
    dv: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> bool {
    if is_polynomial_in(arena, u, var, var_sym) {
        return false;
    }
    let factors: Vec<ExprId> = match arena.node(dv) {
        ExprNode::Mul(children) => children.to_vec(),
        _ => vec![dv],
    };
    let mut special = 0usize;
    let mut rest: SmallVec<[ExprId; 4]> = SmallVec::new();
    for f in factors {
        if !contains_var(arena, f, var_sym) {
            continue;
        }
        match LinearSpecial::of(arena, f) {
            Some((_, arg)) if symbolic_linear_coeff_of(arena, arg, var, var_sym).is_some() => {
                special += 1;
            }
            _ => rest.push(f),
        }
    }
    if special != 1 || rest.is_empty() {
        return false;
    }
    let r = arena.mul(&rest);
    if !is_rational_in(arena, r, var_sym) {
        return false;
    }
    let (_, den) = crate::poly::polybridge::as_numer_denom(arena, r);
    if !contains_var(arena, den, var_sym) {
        return false;
    }
    let du = crate::transforms::diff::diff(arena, u, var);
    !is_polynomial_in(arena, du, var, var_sym)
}

/// Integration by parts for a product of two dependent factors, one a
/// by-parts candidate `u` (a polynomial, a logarithm or an inverse
/// trigonometric function) and the other `dv` with a closed form:
/// `∫ u·dv = u·v − ∫ v·du`.  The LIATE-preferred ordering is tried first;
/// never for a rational function ([`rational_product`]).
fn try_by_parts_pair(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    if rational_product(arena, dependent, var_sym) {
        return None;
    }
    let orderings = {
        let r0 = liate_rank(arena, dependent[0], var, var_sym);
        let r1 = liate_rank(arena, dependent[1], var, var_sym);
        tracing::debug!(u_rank = r0, dv_rank = r1, "by-parts LIATE ordering");
        if r0 <= r1 {
            [(0usize, 1usize), (1, 0)]
        } else {
            [(1, 0), (0, 1)]
        }
    };
    for (u_idx, dv_idx) in orderings {
        let u = dependent[u_idx];
        let dv = dependent[dv_idx];

        // Check that u is a by-parts candidate (polynomial or ln)
        if !is_by_parts_candidate(arena, u, var, var_sym) {
            continue;
        }
        if special_v_without_route(arena, u, dv, var, var_sym) {
            continue;
        }

        // Check that dv is directly integrable (deep check: reject
        // results that contain nested unevaluated Integral nodes,
        // e.g. Add(Integral(..), Integral(..)) which has a non-
        // Integral top node but is still not fully evaluated).
        let v = integrate_node(arena, dv, var, var_sym, depth - 1);
        if crate::base::walk::has_unevaluated(arena, v) {
            tracing::trace!("by-parts: v = ∫dv has unevaluated nodes, skipping this ordering");
            continue; // dv not integrable
        }
        if has_root_sum_of_var(arena, v, var_sym) {
            continue;
        }

        // Compute du = d(u)/dx
        let du = crate::transforms::diff::diff(arena, u, var);

        // Compute ∫ v·du dx
        let v_du = arena.mul(&[v, du]);
        let u_dv = arena.mul(&[u, dv]);
        match by_parts_cycle(arena, u, v, u_dv, v_du, var_sym) {
            Cycle::Solved(result) => return Some(result),
            Cycle::Hopeless => continue,
            Cycle::No => {}
        }
        let integral_v_du = integrate_node(arena, v_du, var, var_sym, depth - 1);

        // Check if the remaining integral was resolved (deep check:
        // an Add of unevaluated Integrals should not be accepted).
        if crate::base::walk::has_unevaluated(arena, integral_v_du) {
            tracing::trace!("by-parts: ∫v·du has unevaluated nodes, skipping this ordering");
            continue; // Remaining integral not solvable
        }

        // Success: ∫ u·dv = u·v - ∫ v·du
        tracing::debug!("integration by parts succeeded");
        let u_v = arena.mul(&[u, v]);
        return Some(arena.sub(u_v, integral_v_du));
    }
    None
}

/// `∫ P₁(x)·P₂(x)·…·g(x) dx` with two or more polynomial factors: their
/// product multiplied out, and the integrand with one polynomial factor
/// retried.  `∫ x·(1 + a²x²)·atan(a·x) dx` (even `∫ x·(1 + x²)·atan x dx`)
/// stayed unevaluated: the three-factor rule takes `u = x`, while by parts
/// needs `dv = x·(1 + a²x²)` (0.31, the complex-parameter hunt).
fn try_merge_polynomial_factors(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    let (polys, mut rest): (Vec<ExprId>, Vec<ExprId>) = dependent
        .iter()
        .partition(|&&d| is_polynomial_in(arena, d, var, var_sym));
    if polys.len() < 2 || rest.is_empty() {
        return None;
    }
    let p = arena.mul(&polys);
    let p = crate::transforms::expand::expand(arena, p);
    rest.push(p);
    let merged = arena.mul(&rest);
    let r = integrate_node(arena, merged, var, var_sym, depth - 1);
    (!crate::base::walk::has_unevaluated(arena, r)).then_some(r)
}

/// `∫ (s₁ + … + s_k)·g dx = Σ ∫ sᵢ·g dx` for a product with a sum factor
/// (or a square to fourth power of a non-polynomial sum, multiplied out),
/// when every piece has a closed form (the first piece without one ends
/// the attempt).  A sum that is not a polynomial in the variable is taken
/// first: polynomial factors are the `u` of by parts.  Linearity only split
/// sums at the top: `∫ (sin t + cos t)·eᵗ dt` stayed unevaluated (by parts
/// wants a polynomial or a logarithm for `u`, and the cyclic rule does not
/// see `(−cos t − sin t)·eᵗ` as `−1` times the integrand) while
/// `∫ (eᵗ·sin t + eᵗ·cos t) dt` worked (0.31, normal-form hunt).
fn try_distribute_sum_factor(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    if dependent.len() < 2 || depth < 2 {
        return None;
    }
    // A radical of the variable (`√(sec x)`, `(a + b·x)^m`) makes every
    // piece as hard as the product, whose substitutions see it whole: on
    // the Rubi suite such attempts cost 30 s and found almost nothing.
    let radical = dependent.iter().any(|&d| {
        crate::base::walk::post_order_ids(arena, d)
            .iter()
            .any(|&id| {
                matches!(arena.node(id), ExprNode::Pow(b, e)
                if !arena.as_num(*e).is_some_and(|q| q.is_integer())
                    && contains_var(arena, *b, var_sym))
            })
    });
    if radical {
        return None;
    }
    // A rational function over ℚ is the rational integrator's (Hermite and
    // Rothstein–Trager are complete there); its pieces only repeat that
    // work — inside the algebraic-remainder fallback of
    // `try_risch_rational`, five nested splittings made `∫ sinh x·tanh 4x`
    // twenty times slower.
    let product = arena.mul(dependent);
    let (numer, denom) = crate::poly::polybridge::as_numer_denom(arena, product);
    if crate::poly::polybridge::expr_to_poly(arena, numer, var).is_some()
        && crate::poly::polybridge::expr_to_poly(arena, denom, var).is_some()
    {
        return None;
    }
    let is_sum = |arena: &Arena, d: ExprId| matches!(arena.node(d), ExprNode::Add(_));
    // A small power of a sum that is not a polynomial, `(a + b·sin u)³`,
    // multiplied out is a sum as well.
    let is_power_of_sum = |arena: &Arena, d: ExprId| {
        matches!(arena.node(d), ExprNode::Pow(b, n)
        if is_sum(arena, *b)
            && !is_polynomial_in(arena, *b, var, var_sym)
            && arena.as_num(*n).is_some_and(|q| {
                q.is_integer() && *q >= Q::from_integer(2.into()) && *q <= Q::from_integer(4.into())
            }))
    };
    let idx = dependent
        .iter()
        .position(|&d| is_sum(arena, d) && !is_polynomial_in(arena, d, var, var_sym))
        .or_else(|| dependent.iter().position(|&d| is_sum(arena, d)))
        .or_else(|| dependent.iter().position(|&d| is_power_of_sum(arena, d)))?;
    let sum = if is_sum(arena, dependent[idx]) {
        dependent[idx]
    } else {
        crate::transforms::expand::expand(arena, dependent[idx])
    };
    let ExprNode::Add(terms) = arena.node(sum).clone() else {
        return None;
    };
    let rest = remaining_product(arena, dependent, idx);
    let mut parts: SmallVec<[ExprId; 6]> = SmallVec::new();
    for &term in terms.iter() {
        let piece = arena.mul(&[term, rest]);
        let big_f = integrate_node(arena, piece, var, var_sym, depth - 1);
        if crate::base::walk::has_unevaluated(arena, big_f) {
            return None;
        }
        parts.push(big_f);
    }
    Some(arena.add(&parts))
}

/// Rewrite products of `sin`/`cos` factors via product-to-sum identities
/// and retry (e.g. `x·sin x·cos x = x·sin(2x)/2`).
fn try_trig_product_to_sum(
    arena: &mut Arena,
    expr: ExprId,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    let trig_count = dependent
        .iter()
        .filter(|&&d| matches!(arena.node(d), ExprNode::Sin(_) | ExprNode::Cos(_)))
        .count();
    if trig_count < 2 {
        return None;
    }
    let combined = arena.trig_combine_expr(expr);
    let combined = crate::transforms::eval::eval(arena, combined);
    if combined == expr {
        return None;
    }
    let combined = crate::transforms::expand::expand(arena, combined);
    let result = integrate_node(arena, combined, var, var_sym, depth - 1);
    if crate::base::walk::has_unevaluated(arena, result) {
        return None;
    }
    Some(result)
}

// ═══════════════════════════════════════════════════════════════════════════
// |g|, sign(g), Heaviside(g), Piecewise
// ═══════════════════════════════════════════════════════════════════════════

/// The real roots of `g` (as expressions) when all of them are known;
/// `None` otherwise.  [`try_abs_sign_product`] takes the sign of `g`
/// constant between the roots it is given, so a root missing here makes
/// an antiderivative that jumps there:
///
/// * `g` with a `sin` or `cos` of the variable is refused.  Such a `g` has
///   as a rule infinitely many real roots, and the solver returns the
///   principal branches only: `sin(eˣ) = 0` gave `x = ln π` alone, and
///   `∫ sign(sin eˣ) dx` was `(x − ln π)·sign(sin eˣ)` — its derivative is
///   the integrand, but it jumps at `x = ln(kπ)` for every `k ≥ 2`, so
///   `F(b) − F(a)` was wrong across any of them (also `cos(eˣ)`,
///   `eˣ·|sin(eˣ)|`, `sin(e⁻ˣ)`).
/// * For a polynomial over `ℚ` the roots found must be as many as its
///   distinct real roots (Sturm).
/// * A root that does not evaluate is not taken for a complex one: it is
///   refused unless it evaluates to a value off the real axis.
fn real_roots(arena: &mut Arena, g: ExprId, var: ExprId) -> Option<Vec<ExprId>> {
    let var_sym = match arena.node(var) {
        ExprNode::Symbol(s) => *s,
        _ => return None,
    };
    let poly = crate::poly::polybridge::expr_to_poly(arena, g, var);
    if poly.is_none()
        && crate::base::walk::post_order_ids(arena, g)
            .into_iter()
            .any(|id| match arena.node(id) {
                ExprNode::Sin(a) | ExprNode::Cos(a) => contains_var(arena, *a, var_sym),
                _ => false,
            })
    {
        return None;
    }
    let sols = crate::transforms::solve::known_real_roots(arena, g, var);
    if sols.is_empty() && poly.is_none() {
        return None;
    }
    let mut roots = Vec::new();
    for s in sols {
        if !crate::base::walk::free_symbols(arena, s).is_empty() {
            return None; // parametric root: cannot decide
        }
        match crate::transforms::evalf::eval_const_f64(arena, s) {
            Some(v) if v.is_finite() => roots.push(s),
            Some(_) => return None,
            None => {
                // A complex root only when its imaginary part is certified
                // nonzero (an `f64` one may be the residue of a real root).
                let value = crate::transforms::eval::eval(arena, s);
                if !crate::transforms::realness::imaginary_part_sign(arena, value)
                    .is_nonzero()
                    .is_true()
                {
                    return None;
                }
            }
        }
    }
    if let Some(p) = &poly
        && p.degree().unwrap_or(0) > 0
        && crate::poly::sturm::SturmChain::new(p).count_real_roots() != roots.len()
    {
        return None;
    }
    Some(roots)
}

/// `∫ N(x)/(a·x + b) dx` for a polynomial `N` with rational coefficients and
/// a linear denominator whose `a`, `b` are constants that are not both
/// rational (`x + π`, `x + √2`, `x + c`, `x + sin 2`): with `x₀ = −b/a`,
/// synthetic division gives `N(x) = (x − x₀)·Q(x) + N(x₀)`, so the
/// antiderivative is `(1/a)·[∫Q + N(x₀)·ln|a·x + b|]`.  The rational
/// integrator handles all-rational denominators and is left alone for them;
/// 0.22 returned `∫ x/(x + π) dx` unevaluated (and searched for 12 s on
/// `∫ x·ln(x + sin 2) dx`, whose by-parts step needs it).
fn try_poly_over_symbolic_linear(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    let (num, den) = crate::poly::polybridge::as_numer_denom(arena, expr);
    if !contains_var(arena, den, var_sym) {
        return None;
    }
    let (a, b) = symbolic_linear_coeff_of(arena, den, var, var_sym)?;
    if contains_var(arena, a, var_sym)
        || contains_var(arena, b, var_sym)
        || arena.is_zero_structural(a)
        || arena.as_num(a).is_some() && arena.as_num(b).is_some()
    {
        return None;
    }
    let p = crate::poly::polybridge::expr_to_poly(arena, num, var)?;
    let coeffs: Vec<ExprId> = p
        .coeffs()
        .iter()
        .map(|c| arena.num_ratio(c.clone()))
        .collect();
    let n = coeffs.len().checked_sub(1)?;
    // x₀ = −b/a; Horner from the top: q_{n−1} = c_n, q_{k−1} = c_k + x₀·q_k.
    let neg_b = arena.neg(b);
    let x0 = arena.div(neg_b, a);
    let mut q = vec![arena.zero; n];
    let mut carry = coeffs[n];
    for k in (1..=n).rev() {
        q[k - 1] = carry;
        let prod = arena.mul(&[x0, carry]);
        carry = arena.add(&[coeffs[k - 1], prod]);
    }
    let remainder = crate::transforms::eval::eval(arena, carry);
    // ∫Q = Σ q_k x^{k+1}/(k+1).
    let mut terms = Vec::with_capacity(n + 1);
    for (k, &qk) in q.iter().enumerate() {
        let e = arena.int(k as i64 + 1);
        let xp = arena.pow(var, e);
        let inv = arena.rational(1, k as i64 + 1);
        terms.push(arena.mul(&[inv, qk, xp]));
    }
    if !arena.is_zero_structural(remainder) {
        let abs_den = arena.abs(den);
        let log = arena.ln(abs_den);
        terms.push(arena.mul(&[remainder, log]));
    }
    let sum = arena.add(&terms);
    let inv_a = arena.pow(a, arena.neg_one());
    let result = arena.mul(&[inv_a, sum]);
    let result = crate::transforms::eval::eval(arena, result);
    (!candidate_rejected(arena, expr, result, var, var_sym)).then_some(result)
}

/// `∫ N(x)/(c·x² + d·x + e) dx` with `deg N ≥ 2` and constant coefficients
/// that are not all rational (`∫ 2x⁴/(sin(−1) − 2x²) dx`, the radical
/// substitution of `∫ x^(3/2)/(sin(−1) − 2x) dx`; `∫ x²·cos 4/(1 + cos²4·x²)`,
/// by parts from `∫ x·atan(x·cos 4) dx`).  Divides `N` by the quadratic over
/// the field of constant expressions, integrates the quotient termwise and
/// hands the linear remainder over the quadratic back to the integrator
/// (`try_linear_over_quadratic`).  The rational-coefficient case belongs
/// to the rational integrator and is left to it.  Checked by
/// differentiation.
fn try_poly_over_symbolic_quadratic(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    /// Quotients of higher degree are left to the other routes.
    const MAX_NUMERATOR_DEGREE: usize = 16;
    let (num, den) = crate::poly::polybridge::as_numer_denom(arena, expr);
    if !contains_var(arena, den, var_sym) {
        return None;
    }
    if crate::poly::polybridge::expr_to_poly(arena, den, var).is_some()
        && crate::poly::polybridge::expr_to_poly(arena, num, var).is_some()
    {
        return None;
    }
    let dense = |arena: &mut Arena, e: ExprId| -> Option<Vec<ExprId>> {
        let terms = crate::poly::polybridge::symbolic_multipoly_terms(arena, e, &[var])?;
        let degree = terms.iter().map(|(exps, _)| exps[0] as usize).max()?;
        if degree > MAX_NUMERATOR_DEGREE {
            return None;
        }
        let mut coeffs = vec![arena.zero; degree + 1];
        for (exps, c) in terms {
            if contains_var(arena, c, var_sym) {
                return None;
            }
            coeffs[exps[0] as usize] = c;
        }
        Some(coeffs)
    };
    let d = dense(arena, den)?;
    if d.len() != 3 || arena.is_zero_structural(d[2]) {
        return None;
    }
    let mut rem = dense(arena, num)?;
    if rem.len() < 3 {
        return None;
    }
    // rem ← rem − q_k·x^k·(c x² + d x + e), from the top.
    let mut quotient = vec![arena.zero; rem.len() - 2];
    for k in (2..rem.len()).rev() {
        let qk = arena.div(rem[k], d[2]);
        let qk = crate::transforms::eval::eval(arena, qk);
        quotient[k - 2] = qk;
        for (j, &dj) in d.iter().enumerate().take(2) {
            let prod = arena.mul(&[qk, dj]);
            let diff = arena.sub(rem[k - 2 + j], prod);
            rem[k - 2 + j] = crate::transforms::eval::eval(arena, diff);
        }
        rem[k] = arena.zero;
    }
    let mut terms: Vec<ExprId> = Vec::with_capacity(quotient.len() + 1);
    for (j, &qj) in quotient.iter().enumerate() {
        let e = arena.int(i64::try_from(j).ok()? + 1);
        let xp = arena.pow(var, e);
        let inv = arena.rational(1, i64::try_from(j).ok()? + 1);
        terms.push(arena.mul(&[inv, qj, xp]));
    }
    let r1x = arena.mul(&[rem[1], var]);
    let linear = arena.add(&[r1x, rem[0]]);
    let linear = crate::transforms::eval::eval(arena, linear);
    if !arena.is_zero_structural(linear) {
        let inv_den = arena.pow(den, arena.neg_one());
        let piece = arena.mul(&[linear, inv_den]);
        let piece_integral = integrate_node(arena, piece, var, var_sym, depth.saturating_sub(1));
        if crate::base::walk::has_unevaluated(arena, piece_integral) {
            return None;
        }
        terms.push(piece_integral);
    }
    let result = arena.add(&terms);
    let result = crate::transforms::eval::eval(arena, result);
    (!candidate_rejected(arena, expr, result, var, var_sym)).then_some(result)
}

/// Does `e` contain `nan`, `zoo` or `±∞`?
fn contains_non_finite(arena: &Arena, e: ExprId) -> bool {
    crate::base::walk::post_order_ids(arena, e)
        .into_iter()
        .any(|id| {
            matches!(
                arena.node(id),
                ExprNode::NaN
                    | ExprNode::ComplexInfinity
                    | ExprNode::Infinity
                    | ExprNode::NegInfinity
            )
        })
}

/// Does `e` apply a function that is not analytic (`|·|`, `sign`, `H`,
/// `re`, `im`, `conjugate`, `arg`, `min`, `max`, `Piecewise`) to something
/// that depends on the variable?  A `ln|g|` does not count: the stage exit
/// writes it `ln g` where `g` is not real ([`analytic_log_of_complex_arguments`]),
/// and `floor`/`ceiling` do not either (the continuity corrections
/// `J·⌊w/π + 1/2⌋` of trigonometric antiderivatives): they are constant
/// between their jumps, so their derivative is 0 along any path.
fn non_analytic_in(arena: &Arena, e: ExprId, var_sym: SymbolId) -> bool {
    let ids = crate::base::walk::post_order_ids(arena, e);
    let in_log: FxHashSet<ExprId> = ids
        .iter()
        .filter_map(|&id| match arena.node(id) {
            ExprNode::Ln(a) if matches!(arena.node(*a), ExprNode::Abs(_)) => Some(*a),
            _ => None,
        })
        .collect();
    ids.into_iter().any(|id| {
        let non_analytic = match arena.node(id) {
            ExprNode::Abs(_) => !in_log.contains(&id),
            ExprNode::Sign(_)
            | ExprNode::Heaviside(_)
            | ExprNode::Re(_)
            | ExprNode::Im(_)
            | ExprNode::Conjugate(_)
            | ExprNode::Arg(_)
            | ExprNode::Min(_)
            | ExprNode::Max(_)
            | ExprNode::Piecewise(_) => true,
            _ => false,
        };
        non_analytic && contains_var(arena, id, var_sym)
    })
}

/// Is `e` real *and continuous* for every real value of the integration
/// variable, under the declared assumptions of the other symbols?  The
/// `|g|`/`sign(g)` route needs both: `g` may change sign only at its real
/// roots.  Conservative: sums, products, non-negative integer powers (a
/// negative power only of a variable-free base), and `sin`/`cos`/`exp`/
/// `atan`/`sinh`/`cosh`/`tanh`/`abs` of such; roots, logarithms, `tan`,
/// reciprocals of the variable and `sign` are refused.  (`∫ |π/(4 cos x)| dx`
/// came back as `(π/4)·ln|sec x + tan x|`, wrong wherever `cos x < 0` —
/// `c/cos x` has no roots but changes sign at its poles; found by
/// `fuzz_integrate`.)  A symbol other than the variable counts only when
/// it is declared real (decision D4, 0.31): up to 0.30 every symbol did,
/// and [`flatten_nested_pow`] rewrote `√((x + a)²)` to `|x + a|` for an
/// unassumed `a` — `∫ √((x + a)²) dx` came back as the unevaluated
/// `∫ |x + a| dx`, a different integral for a non-real `a`.
fn real_on_reals(arena: &Arena, e: ExprId, var_sym: SymbolId) -> bool {
    crate::base::walk::post_order_ids(arena, e)
        .into_iter()
        .all(|id| match arena.node(id) {
            ExprNode::Symbol(s) => {
                *s == var_sym || crate::transforms::realness::symbol_declared_real(arena, *s)
            }
            ExprNode::Num(_)
            | ExprNode::Pi
            | ExprNode::E
            | ExprNode::EulerGamma
            | ExprNode::Catalan
            | ExprNode::GoldenRatio
            | ExprNode::Add(_)
            | ExprNode::Mul(_)
            | ExprNode::Sin(_)
            | ExprNode::Cos(_)
            | ExprNode::Exp(_)
            | ExprNode::Atan(_)
            | ExprNode::Sinh(_)
            | ExprNode::Cosh(_)
            | ExprNode::Tanh(_)
            | ExprNode::Abs(_) => true,
            ExprNode::Pow(base, exp) => arena.as_num(*exp).is_some_and(|r| {
                r.is_integer() && (!r.is_negative() || !contains_var(arena, *base, var_sym))
            }),
            _ => false,
        })
}

/// `∫ P(x)·|g(x)| dx`, `∫ P(x)·sign(g(x)) dx`, `∫ P(x)·H(g(x)) dx` where
/// `g` has at most one simple real root `r` (or none):
///
/// * one root:  `sign(g)·(G(x) − G(r))` with `G = ∫ P·g` (for `|g|`),
///   `sign(g)·(F(x) − F(r))` with `F = ∫ P` (for `sign`), and
///   `H(g)·(F(x) − F(r))` for `H` — each continuous at `r`;
/// * no root: `sign(g(x₀))` is constant and the factor is replaced by
///   `±g`, `±1` or `0/1` respectively.
fn try_abs_sign_product(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    let idx = dependent.iter().position(|&d| {
        matches!(
            arena.node(d),
            ExprNode::Abs(_) | ExprNode::Sign(_) | ExprNode::Heaviside(_)
        ) && contains_var(arena, d, var_sym)
    })?;
    let node = arena.node(dependent[idx]).clone();
    let g = match node {
        ExprNode::Abs(g) | ExprNode::Sign(g) | ExprNode::Heaviside(g) => g,
        _ => return None,
    };
    // `|g| = sign(g)·g` and the root-shift below assume `g` is real at every
    // real `x`; `|√x|` is `√|x|`, not `sign(√x)·√x` (0.22.2 returned
    // `2/3·x^(3/2)·sign(√x)`, found by `fuzz_integrate`).
    if !real_on_reals(arena, g, var_sym) {
        return None;
    }
    let rest = remaining_product(arena, dependent, idx);
    let roots = real_roots(arena, g, var)?;
    if roots.len() > 1 {
        return None;
    }
    let root = roots.first().copied();
    // The "smooth" integrand whose antiderivative gets the sign factor.
    let smooth = match node {
        ExprNode::Abs(_) => arena.mul(&[rest, g]),
        _ => rest,
    };
    let smooth_int = integrate_node(arena, smooth, var, var_sym, depth - 1);
    if crate::base::walk::has_unevaluated(arena, smooth_int) {
        return None;
    }
    match root {
        Some(r) => {
            let at_r = crate::transforms::subs::subs(arena, smooth_int, var, r);
            let at_r = crate::transforms::eval::eval(arena, at_r);
            // The shift needs a finite G(r); an antiderivative singular at
            // the root (`ln|x|` at 0) made `nan` of the whole answer.
            if contains_non_finite(arena, at_r) {
                return None;
            }
            let shifted = arena.sub(smooth_int, at_r);
            let factor = match node {
                ExprNode::Heaviside(_) => arena.heaviside(g),
                _ => arena.sign(g),
            };
            Some(arena.mul(&[factor, shifted]))
        }
        None => {
            // Constant sign: sample g at a point.
            let sample = crate::transforms::subs::subs(arena, g, var, arena.zero);
            let sample = crate::transforms::eval::eval(arena, sample);
            let sgn = crate::transforms::evalf::eval_const_f64(arena, sample)?;
            // A nonzero printed digit is certified; 0 refuses.
            if sgn == 0.0 || !sgn.is_finite() {
                return None;
            }
            let positive = sgn > 0.0;
            match node {
                ExprNode::Abs(_) | ExprNode::Sign(_) => {
                    if positive {
                        Some(smooth_int)
                    } else {
                        Some(arena.neg(smooth_int))
                    }
                }
                _ => {
                    if positive {
                        Some(smooth_int)
                    } else {
                        Some(arena.zero)
                    }
                }
            }
        }
    }
}

/// Most real kinks [`try_piecewise_split`] splits the line at.
const MAX_SPLIT_POINTS: usize = 6;

/// The order of the real constants `r` and `w`: equal when they are the
/// same node, otherwise by the certified sign of `w − r`
/// ([`sign_checked`](crate::poly::algebraic::sign_checked), exact 0
/// included); `None` when that does not settle.
fn cmp_roots(arena: &mut Arena, r: ExprId, w: ExprId) -> Option<std::cmp::Ordering> {
    if r == w {
        return Some(std::cmp::Ordering::Equal);
    }
    let d = arena.sub(w, r);
    let d = crate::transforms::eval::eval(arena, d);
    crate::poly::algebraic::sign_checked(arena, d).map(|s| 0.cmp(&s))
}

/// `∫ f dx` for an `f` holding `|g|`, `sign(g)`, `H(g)`, `min(g₁, g₂)`,
/// `max(g₁, g₂)` of functions real on the real line whose sign changes
/// only at finitely many numeric real points `r₁ < … < rₖ` (all of them
/// found, or one symbolic root of a linear `g` with a leading coefficient
/// of known sign): on each interval every such factor is `±g`, `±1`, `0/1`,
/// a branch, the plain integrand goes through the whole pipeline, and the
/// pieces are joined continuously, `Piecewise((F₀, x < r₁), (F₁ + k₁, x <
/// r₂), …)` with `kⱼ = kⱼ₋₁ + Fⱼ₋₁(rⱼ) − Fⱼ(rⱼ)`.  The product rule
/// ([`try_abs_sign_product`]) takes at most one root and a polynomial
/// factor: `∫ |x² − 1| dx`, `∫ e^{−|t|} dt`, `∫ e^{|x − 2|} dx`,
/// `∫ min(e^{−x}, eˣ)² dx`, `∫ H(3x − y)·sin(x + y) dx` stayed unevaluated up
/// to 0.46 (SymPy's test_integrals and test_manual answer them as such a
/// `Piecewise`).
fn try_piecewise_split(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    // The kinked nodes and the function whose sign decides each.
    let mut kinks: Vec<(ExprId, ExprId)> = Vec::new();
    for id in crate::base::walk::post_order_ids(arena, expr) {
        let g = match arena.node(id) {
            ExprNode::Abs(g) | ExprNode::Sign(g) | ExprNode::Heaviside(g) => *g,
            ExprNode::Min(args) | ExprNode::Max(args) if args.len() == 2 => {
                let (a, b) = (args[0], args[1]);
                arena.sub(a, b)
            }
            _ => continue,
        };
        if !contains_var(arena, g, var_sym) {
            continue;
        }
        if !real_on_reals(arena, g, var_sym) {
            return None;
        }
        kinks.push((id, g));
    }
    if kinks.is_empty() || kinks.len() > 3 {
        return None;
    }
    // Breakpoints: numeric roots, sorted; or one symbolic root.
    // The numeric roots, distinct and in increasing order.
    let mut points: Vec<ExprId> = Vec::new();
    let mut symbolic: Option<(ExprId, ExprId)> = None;
    for &(_, g) in &kinks {
        match real_roots(arena, g, var) {
            Some(rs) => {
                // Placed by exact or certified comparison ([`cmp_roots`]):
                // up to 0.52 roots within `10⁻¹²` of each other were
                // merged and the rest sorted by `f64` values, and `∫ sign((x
                // − 1)(x − 1 − 10⁻¹⁵)) dx` lost the interval between them.
                for r in rs {
                    crate::transforms::evalf::eval_const_f64(arena, r)?;
                    let mut at = points.len();
                    let mut known = false;
                    for (k, &w) in points.iter().enumerate() {
                        match cmp_roots(arena, r, w)? {
                            std::cmp::Ordering::Equal => {
                                known = true;
                                break;
                            }
                            std::cmp::Ordering::Less => {
                                at = k;
                                break;
                            }
                            std::cmp::Ordering::Greater => {}
                        }
                    }
                    if !known {
                        points.insert(at, r);
                    }
                }
            }
            None => {
                let (a, b) = symbolic_linear_coeff_of(arena, g, var, var_sym)?;
                if symbolic.is_some() {
                    return None;
                }
                let r = arena.div(b, a);
                let r = arena.neg(r);
                symbolic = Some((crate::transforms::eval::eval(arena, r), a));
            }
        }
    }
    if symbolic.is_some() && (!points.is_empty() || kinks.len() > 1) {
        return None;
    }
    if points.len() > MAX_SPLIT_POINTS {
        return None;
    }
    let breaks: Vec<ExprId> = match symbolic {
        Some((r, _)) => vec![r],
        None => points,
    };
    if breaks.is_empty() {
        return None;
    }
    // The sign of each kink's function on each interval.
    let n = breaks.len();
    let mut pieces: Vec<ExprId> = Vec::with_capacity(n + 1);
    for j in 0..=n {
        let mut piece = expr;
        for &(node, g) in &kinks {
            let positive = match symbolic {
                Some((_, a)) => {
                    let s = crate::poly::algebraic::sign_checked(arena, a)?;
                    if s == 0 {
                        return None;
                    }
                    (s > 0) == (j == 1)
                }
                None => {
                    let one = arena.one;
                    let sample = if j == 0 {
                        arena.sub(breaks[0], one)
                    } else if j == n {
                        arena.add(&[breaks[n - 1], one])
                    } else {
                        let half = arena.rational(1, 2);
                        let s = arena.add(&[breaks[j - 1], breaks[j]]);
                        arena.mul(&[half, s])
                    };
                    let at = crate::transforms::subs::subs(arena, g, var, sample);
                    let at = crate::transforms::eval::eval(arena, at);
                    let v = crate::transforms::evalf::eval_const_f64(arena, at)?;
                    // A nonzero printed digit is certified; 0 refuses.
                    if v == 0.0 || !v.is_finite() {
                        return None;
                    }
                    v > 0.0
                }
            };
            let rep = match arena.node(node).clone() {
                ExprNode::Abs(_) => {
                    if positive {
                        g
                    } else {
                        arena.neg(g)
                    }
                }
                ExprNode::Sign(_) => arena.int(if positive { 1 } else { -1 }),
                ExprNode::Heaviside(_) => arena.int(i64::from(positive)),
                ExprNode::Min(args) => args[usize::from(positive)],
                ExprNode::Max(args) => args[usize::from(!positive)],
                _ => return None,
            };
            piece = arena.subs_structural(piece, node, rep);
        }
        pieces.push(crate::transforms::eval::eval(arena, piece));
    }
    let mut antis: Vec<ExprId> = Vec::with_capacity(n + 1);
    for &p in &pieces {
        antis.push(integrate_nested(arena, p, var)?);
    }
    let mut out: Vec<(ExprId, ExprId)> = Vec::with_capacity(n + 1);
    let mut k = arena.zero;
    for j in 0..=n {
        if j > 0 {
            let r = breaks[j - 1];
            let prev = crate::transforms::subs::subs(arena, antis[j - 1], var, r);
            let cur = crate::transforms::subs::subs(arena, antis[j], var, r);
            let diff = arena.sub(prev, cur);
            k = arena.add(&[k, diff]);
            k = crate::transforms::eval::eval(arena, k);
            if contains_non_finite(arena, k) {
                return None;
            }
        }
        let branch = arena.add(&[antis[j], k]);
        let cond = if j < n {
            arena.gt(breaks[j], var)
        } else {
            arena.bool_true
        };
        out.push((branch, cond));
    }
    tracing::debug!("integrate: split at the kinks, joined continuously");
    Some(arena.piecewise(&out))
}

/// `∫ Piecewise((fᵢ, cᵢ)) dx = Piecewise((Fᵢ + kᵢ, cᵢ))`.
///
/// When the conditions form a chain `x < c₁, x < c₂, …, otherwise` (any
/// of `<`, `≤`) with increasing constants, the constants `kᵢ` are chosen so
/// that the antiderivative is continuous at every breakpoint.  For other
/// condition shapes the branch antiderivatives are returned without
/// matching constants (still a valid antiderivative on each branch).
fn integrate_piecewise(
    arena: &mut Arena,
    pairs: &[(ExprId, ExprId)],
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    let mut antis: Vec<ExprId> = Vec::with_capacity(pairs.len());
    for &(val, _) in pairs {
        let f = integrate_node(arena, val, var, var_sym, depth - 1);
        if crate::base::walk::has_unevaluated(arena, f) {
            return None;
        }
        antis.push(f);
    }
    // Chain detection: condition i (< last) is `x < c_i` or `x ≤ c_i`,
    // i.e. Gt(c_i, x) / Ge(c_i, x) with c_i free of var.
    let mut breakpoints: Vec<ExprId> = Vec::new();
    let mut chain = true;
    for &(_, cond) in &pairs[..pairs.len().saturating_sub(1)] {
        match arena.node(cond).clone() {
            ExprNode::Gt(c, v) | ExprNode::Ge(c, v)
                if v == var && !contains_var(arena, c, var_sym) =>
            {
                breakpoints.push(c);
            }
            _ => {
                chain = false;
                break;
            }
        }
    }
    let mut out: Vec<(ExprId, ExprId)> = Vec::with_capacity(pairs.len());
    if chain && pairs.len() >= 2 {
        let mut k = arena.zero;
        out.push((antis[0], pairs[0].1));
        for i in 1..pairs.len() {
            let c = breakpoints[i - 1];
            // k_i = k_{i-1} + F_{i-1}(c) − F_i(c)
            let prev_at_c = crate::transforms::subs::subs(arena, antis[i - 1], var, c);
            let cur_at_c = crate::transforms::subs::subs(arena, antis[i], var, c);
            let diff = arena.sub(prev_at_c, cur_at_c);
            k = arena.add(&[k, diff]);
            k = crate::transforms::eval::eval(arena, k);
            let branch = arena.add(&[antis[i], k]);
            out.push((branch, pairs[i].1));
        }
    } else {
        for (i, &(_, cond)) in pairs.iter().enumerate() {
            out.push((antis[i], cond));
        }
    }
    Some(arena.piecewise(&out))
}

// ═══════════════════════════════════════════════════════════════════════════
// Piecewise parametric wrapping
// ═══════════════════════════════════════════════════════════════════════════

/// Attempt to wrap the integration result in a `Piecewise` for parametric
/// degenerate cases.
///
/// When the result contains denominators involving free symbols (parameters
/// other than the integration variable), we check if setting those parameters
/// to specific values would cause division by zero.  For each such degenerate
/// value, we substitute back into the original integrand, re-integrate the
/// simplified form, and build a `Piecewise` node with explicit conditions.
fn try_piecewise_wrap(
    arena: &mut Arena,
    result: ExprId,
    original_integrand: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> ExprId {
    // If result is an unevaluated Integral, nothing to wrap.
    if matches!(arena.node(result), ExprNode::Integral(_, _)) {
        return result;
    }
    // The integrand of a rationalising substitution: the degenerate cases
    // are those of the integrand in `x`, which the outer call wraps.
    if GENERIC_ONLY.with(std::cell::Cell::get) {
        return result;
    }
    // An integrand free of the variable has the antiderivative `c·x`: where
    // a denominator of `c` vanishes the integrand itself is undefined, and
    // there is no degenerate case to integrate.  In 0.44 the branch `b = 0`
    // of `∫ (cosh⁴(a + bx) − sinh⁴(a + bx))/(cosh⁴(a + bx) + sinh⁴(a + bx))`
    // was wrapped in 8 nested cases `a ≠ −ln(RootOf(t⁸ + 6t⁴ + 1, k))` (Rubi
    // 6.7.1 #1052: 6 ms in 0.43, 0.2 s in 0.44).
    if !contains_var(arena, original_integrand, var_sym) {
        return result;
    }

    // A result that is already such a case analysis — from the wrap of a
    // nested `integrate`, as a substitution re-enters the pipeline — keeps
    // it: its degenerate values count as handled, and only its generic
    // branch is searched for denominators.  `∫ sin(a·ln x) dx` came out with
    // the cases `a = i` and `a = −i` nested twice each (0.31).
    let mut generic = result;
    let mut handled: Vec<(ExprId, ExprId)> = Vec::new();
    while let ExprNode::Piecewise(pairs) = arena.node(generic).clone()
        && pairs.len() == 2
        && pairs[1].1 == arena.bool_true
        && let ExprNode::Ne(p, v) = *arena.node(pairs[0].1)
    {
        handled.push((p, v));
        handled.push((v, p));
        generic = pairs[0].0;
    }

    // Collect denominator expressions from the result.
    let denoms = collect_denominators(arena, generic);
    if denoms.is_empty() {
        return result;
    }

    let mut wrapped = result;
    // The handled pairs with the denominator they came from.
    let mut handled_by_denom: Vec<(ExprId, ExprId, ExprId)> = Vec::new();
    // The `(parameter, value)` cases whose re-integration found no closed
    // form.  The same case met again through another denominator (`a = b`
    // from `a − b` and from `a² − b²`) is the same integrand in the same
    // context and fails again (on the Rubi suite's 4.5.4.2, 3,852 of 8,948
    // failed cases were such repeats; 0.38).
    let mut failed: Vec<(ExprId, ExprId)> = Vec::new();

    for denom in &denoms {
        let denom_syms = crate::base::walk::free_symbols(arena, *denom);
        for sym_expr in &denom_syms {
            // Skip the integration variable.
            if let ExprNode::Symbol(sid) = arena.node(*sym_expr)
                && *sid == var_sym
            {
                continue;
            }
            // Only parameters of the integrand can be degenerate.  A symbol
            // that occurs in the result alone (the bound variable of a
            // `RootSum`, a substitution symbol) would leave the integrand
            // unchanged and re-integrate the same problem forever.
            if !crate::base::walk::contains(arena, original_integrand, *sym_expr) {
                continue;
            }

            // Solve denom = 0 for this parameter symbol.
            let solutions = degenerate_values(arena, *denom, *sym_expr);
            for &degen_val in &solutions {
                // A degenerate value is a constant.  A denominator in both
                // the parameter and the variable (`(a + 1)·x + a`, from a
                // partial answer) has the "solution" `a = −x/(x + 1)`, and
                // substituting it made a zero denominator in disguise that
                // panicked the rational integrator (found by the
                // complex-parameter hunter, 0.31).
                if contains_var(arena, degen_val, var_sym) || contains_non_finite(arena, degen_val)
                {
                    continue;
                }
                // Nor a value in a symbol the integrand does not have: the
                // bound variable of a `RootSum` in a denominator of its body.
                if crate::base::walk::free_symbols(arena, degen_val)
                    .iter()
                    .any(|&s| !crate::base::walk::contains(arena, original_integrand, s))
                {
                    continue;
                }

                // Avoid duplicate wrapping for the same (param, value) pair.
                if handled
                    .iter()
                    .any(|&(p, v)| p == *sym_expr && v == degen_val)
                    || failed.contains(&(*sym_expr, degen_val))
                {
                    continue;
                }
                // Nor for a value whose points are already covered: `b = a`
                // after `a = b` (the denominator `a − b` solved for either
                // symbol), `b = a²/4` after `a = ±2√b`.  Each such wrap nested
                // an unreachable copy of the whole case analysis: the
                // parametric rational integrator's `∫ dx/((x + a)(x + b)(x + c))`
                // came out six levels deep (0.31).
                if degenerate_value_covered(arena, &handled_by_denom, *denom, *sym_expr, degen_val)
                {
                    continue;
                }
                // Nor for a value that an enclosing degenerate case rules
                // out: inside the branch `c = b²/(4a)` the parameter `a` is
                // not 0.  Its integrand at `a = 0` came out as `0` (the
                // `1/(b²x⁴/(4·0) + …)²` of the substituted denominator is
                // `1/zoo²`), and `∫ x²(d + e·x² + f·x⁴ + g·x⁶)/(a + b·x² +
                // c·x⁴)² dx` carried an unreachable `Piecewise(…, (0, True))`
                // (0.31).
                if excluded_by_enclosing_case(arena, *sym_expr, degen_val) {
                    continue;
                }
                // Nor for a non-real constant value of a parameter declared
                // real: `a = ±i√2` in `∫ dx/(a² + 1 + cos x)` (an unreachable
                // branch, and one `piecewise_simplify` cannot drop: `i√2 ≠ −3`
                // after substituting `a = −3` stayed undecided).
                if nonreal_value_of_real_symbol(arena, *sym_expr, degen_val) {
                    continue;
                }
                // Nor for a value the other assumptions of the parameter
                // exclude: `a = 0` and `b = −a` for positive `a`, `b` in
                // `∫ dx/(a + b·sin²x)` (0.42 carried both unreachable
                // branches).  An undecided value keeps its branch.
                if Truth::from(crate::transforms::solve::assumptions_hold(
                    arena, *sym_expr, degen_val,
                ))
                .is_false()
                {
                    continue;
                }

                // Filter: skip if substituting this value makes the original
                // integrand singular (these are poles of the problem, not
                // artifacts of the antiderivative formula).
                let integrand_at_degen =
                    crate::transforms::subs::subs(arena, original_integrand, *sym_expr, degen_val);
                let integrand_at_degen = crate::transforms::eval::eval(arena, integrand_at_degen);
                if has_zero_denominator(arena, integrand_at_degen) {
                    continue;
                }

                // Re-integrate the simplified integrand at the degenerate value.
                tracing::debug!(
                    integrand = %arena.display(integrand_at_degen),
                    "try_piecewise_wrap: integrating a degenerate case"
                );
                let nonzero = NonzeroInCase::enter(arena, collect_denominators(arena, degen_val));
                // A value that real parameters cannot take (`b = 3i/n`, the
                // resonance of `∫ x²·sin(a + b·ln(c·xⁿ)) dx`) gets a small
                // step budget: on the Rubi suite (0.31) such cases were 2,932
                // failed re-integrations taking 34 s (SymPy leaves them
                // unevaluated too) and 223 successes of at most 18 steps.
                let _cap = (!real_at_generic_values(arena, degen_val))
                    .then(|| StepCap::enter(NONREAL_CASE_STEPS));
                let degen_result = stage!(
                    arena,
                    "degenerate_case",
                    integrand_at_degen,
                    integrate(arena, integrand_at_degen, var)
                );
                drop(nonzero);
                let degen_result = crate::transforms::eval::eval(arena, degen_result);

                // Skip if re-integration returned unevaluated, also in part:
                // `x²/2 + ∫ 1/(−x/(x + 1) + x·(−x/(x + 1) + 1)) dx` (an integrand
                // undefined at the degenerate value) made the whole answer
                // unevaluated (0.31, the parametric-rational hunt).
                // Nor a branch that is not finite: the generic formula at
                // the degenerate value (`(Pₙ₊₁ − Pₙ₋₁)/(2n + 1)` at `n = −½`)
                // came back as `zoo`, which is no antiderivative.
                if crate::base::walk::has_unevaluated(arena, degen_result)
                    || contains_non_finite(arena, degen_result)
                {
                    failed.push((*sym_expr, degen_val));
                    continue;
                }

                // Build Piecewise: [(generic, Ne(param, degen)), (degen_result, True)]
                let condition = arena.ne_(*sym_expr, degen_val);
                let true_cond = arena.bool_true;
                wrapped = arena.piecewise(&[(wrapped, condition), (degen_result, true_cond)]);

                handled.push((*sym_expr, degen_val));
                handled_by_denom.push((*denom, *sym_expr, degen_val));
            }
        }
    }

    wrapped
}

/// The values of the parameter `sym` at which `denom` vanishes: the roots
/// of each irreducible factor (over the other parameters) of its numerator
/// as a fraction, or of `denom` itself when that numerator does not split.
/// A factor linear in `sym` gives its root without radicals: `1 − k²/m²`
/// vanishes at `m = k` and `m = −k`, where `solve` on the whole gives
/// `m = ±√(k²)`, which is `±|k|` for a real `k`; the branch `m = |k|` of
/// `∫ sin(kx)·sin(mx) dx` then integrated `sin(kx)·sin(|k|x)` by product to
/// sum into a quotient by `k − |k|`, `0/0` for every `k > 0` (SymPy's
/// test_integrals, `k`, `m` integers; 0.46 returned `0` for the definite
/// integral over `[0, π]` at `m = k`, which is `π/2`).
fn degenerate_values(arena: &mut Arena, denom: ExprId, sym: ExprId) -> Vec<ExprId> {
    let (num, _) = crate::poly::polybridge::fraction_parts(arena, denom);
    let num = crate::transforms::expand::expand(arena, num);
    let factored = arena.factor_expr(num, sym);
    let factors: Vec<ExprId> = match arena.node(factored) {
        ExprNode::Mul(children) => children
            .iter()
            .map(|&c| match *arena.node(c) {
                ExprNode::Pow(b, e) if arena.as_num(e).is_some_and(|r| r.is_integer()) => b,
                _ => c,
            })
            .collect(),
        _ => Vec::new(),
    };
    let split: Vec<ExprId> = factors
        .into_iter()
        .filter(|&f| crate::base::walk::contains(arena, f, sym))
        .collect();
    if split.len() < 2 {
        return crate::transforms::solve::known_real_roots(arena, denom, sym);
    }
    let mut values: Vec<ExprId> = Vec::new();
    for f in split {
        for s in crate::transforms::solve::known_real_roots(arena, f, sym) {
            if !values.contains(&s) {
                values.push(s);
            }
        }
    }
    values
}

/// The step budget ([`StepCap`]) of a degenerate case whose value is not
/// real at real parameter values.
const NONREAL_CASE_STEPS: u64 = 20;

/// Is the parameter value `val` real for some real values of its own
/// parameters: the generic values of [`FTC_PARAMETER_VALUES`] (in name
/// order) with every combination of signs (`sqrt(−a/b)` is real for
/// `a/b < 0`; `3i/n`, `(−1/2 + i√3/2)·∛(−a/b)` are real for none)?
/// `true` when it does not evaluate, or has more than three parameters.
fn real_at_generic_values(arena: &mut Arena, val: ExprId) -> bool {
    let mut params: Vec<(String, ExprId)> = Vec::new();
    for s in crate::base::walk::free_symbols(arena, val) {
        let ExprNode::Symbol(sid) = *arena.node(s) else {
            return true;
        };
        params.push((arena.symbol_name(sid).to_owned(), s));
    }
    if params.len() > 3 {
        return true;
    }
    params.sort_by(|a, b| a.0.cmp(&b.0));
    for signs in 0u32..(1 << params.len()) {
        let mut bound = val;
        for (k, &(_, s)) in params.iter().enumerate() {
            let (p, q) = FTC_PARAMETER_VALUES[k];
            let p = if signs & (1 << k) == 0 { p } else { -p };
            let value = arena.rational(p, q);
            bound = crate::transforms::subs::subs(arena, bound, s, value);
        }
        let bound = crate::transforms::eval::eval(arena, bound);
        match crate::transforms::evalf::evalf_complex(arena, bound, 15) {
            Ok(v) if !crate::transforms::evalf::is_real_to_digits(&v, 12) => {}
            _ => return true,
        }
    }
    false
}

thread_local! {
    /// The denominators of the degenerate values of the enclosing
    /// [`try_piecewise_wrap`] cases on this thread (`a` for `c = b²/(4a)`):
    /// non-zero wherever the case's answer applies.
    static NONZERO_IN_CASE: std::cell::RefCell<Vec<(u64, ExprId)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// The denominators of a degenerate value pushed onto [`NONZERO_IN_CASE`]
/// for the re-integration of its case; popped on drop, also when a panic
/// unwinds.
struct NonzeroInCase(usize);

impl NonzeroInCase {
    fn enter(arena: &Arena, denominators: Vec<ExprId>) -> Self {
        let key = arena.serial();
        NONZERO_IN_CASE.with(|s| {
            let mut s = s.borrow_mut();
            let len = s.len();
            s.extend(denominators.into_iter().map(|d| (key, d)));
            NonzeroInCase(len)
        })
    }
}

impl Drop for NonzeroInCase {
    fn drop(&mut self) {
        let _ = NONZERO_IN_CASE.try_with(|s| {
            if let Ok(mut s) = s.try_borrow_mut() {
                s.truncate(self.0);
            }
        });
    }
}

/// Does `sym = val` make one of the enclosing cases' non-zero
/// denominators ([`NONZERO_IN_CASE`]) vanish?
fn excluded_by_enclosing_case(arena: &mut Arena, sym: ExprId, val: ExprId) -> bool {
    let key = arena.serial();
    let nonzero: Vec<ExprId> = NONZERO_IN_CASE.with(|s| {
        s.borrow()
            .iter()
            .filter(|&&(k, _)| k == key)
            .map(|&(_, d)| d)
            .collect()
    });
    nonzero.into_iter().any(|d| {
        crate::base::walk::contains(arena, d, sym) && {
            let at = crate::transforms::subs::subs(arena, d, sym, val);
            let at = crate::transforms::expand::expand(arena, at);
            let at = crate::transforms::eval::eval(arena, at);
            arena.is_zero_structural(at)
        }
    })
}

/// Is `sym` declared real and `val` a constant that is not real?
fn nonreal_value_of_real_symbol(arena: &mut Arena, sym: ExprId, val: ExprId) -> bool {
    let ExprNode::Symbol(sid) = *arena.node(sym) else {
        return false;
    };
    crate::transforms::realness::symbol_declared_real(arena, sid)
        && crate::base::walk::free_symbols(arena, val).is_empty()
        // Certified beyond `10⁻⁹` of a 16-digit value; a smaller imaginary
        // part keeps the case.
        && crate::transforms::evalf::evalf_complex64(arena, val)
            .is_ok_and(|z| z.im.is_finite() && z.im.abs() > 1e-9 * z.norm().max(1.0))
}

/// Is every point with `sym = val` already a handled degenerate case of
/// [`try_piecewise_wrap`]: does `p − v` vanish there for one handled pair
/// `(p, v)`, or the product of `p − v` over the pairs handled for the same
/// denominator `denom` (its solutions for one symbol cover its zero set)?
fn degenerate_value_covered(
    arena: &mut Arena,
    handled: &[(ExprId, ExprId, ExprId)],
    denom: ExprId,
    sym: ExprId,
    val: ExprId,
) -> bool {
    let vanishes_at = |arena: &mut Arena, e: ExprId| {
        let at = crate::transforms::subs::subs(arena, e, sym, val);
        let at = crate::transforms::expand::expand(arena, at);
        let at = crate::transforms::eval::eval(arena, at);
        arena.is_zero_structural(at)
    };
    let mut same_denom: Vec<ExprId> = Vec::new();
    for &(d, p, v) in handled {
        let diff = arena.sub(p, v);
        if vanishes_at(arena, diff) {
            return true;
        }
        if d == denom {
            same_denom.push(diff);
        }
    }
    if same_denom.len() >= 2 {
        let product = arena.mul(&same_denom);
        return vanishes_at(arena, product);
    }
    false
}

/// Collect all denominator sub-expressions from an expression tree.
///
/// A "denominator" is the base of any `Pow(base, exp)` node where `exp`
/// is a negative rational number.
fn collect_denominators(arena: &Arena, expr: ExprId) -> Vec<ExprId> {
    let mut denoms = Vec::new();
    let mut stack: Vec<ExprId> = vec![expr];
    let mut visited: FxHashSet<ExprId> = FxHashSet::default();

    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        if let ExprNode::Pow(base, exp) = arena.node(id).clone()
            && let Some(r) = arena.as_num(exp)
            && r.is_negative()
        {
            denoms.push(base);
        }
        arena.node(id).for_each_child(|c| stack.push(c));
    }

    denoms
}

/// Check if an expression contains a sub-expression that evaluates to
/// division by zero (a denominator that is structurally zero, or NaN /
/// ComplexInfinity atoms).
fn has_zero_denominator(arena: &Arena, expr: ExprId) -> bool {
    let mut stack: Vec<ExprId> = vec![expr];
    let mut visited: FxHashSet<ExprId> = FxHashSet::default();

    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        if matches!(arena.node(id), ExprNode::NaN | ExprNode::ComplexInfinity) {
            return true;
        }
        if let ExprNode::Pow(base, exp) = arena.node(id).clone()
            && let Some(r) = arena.as_num(exp)
            && r.is_negative()
            && arena.is_zero_structural(base)
        {
            return true;
        }
        arena.node(id).for_each_child(|c| stack.push(c));
    }

    false
}

// ═══════════════════════════════════════════════════════════════════════════
// Special function integration table
// ═══════════════════════════════════════════════════════════════════════════

/// The functions `f` of [`try_special_function_integral`]: `∫ f(α·x + β)/x dx`
/// is an `Ei`, `Si`/`Ci` or `Shi`/`Chi`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LinearSpecial {
    Exp,
    Sin,
    Cos,
    Sinh,
    Cosh,
}

impl LinearSpecial {
    /// The function and its argument when `e` is one of them.
    fn of(arena: &Arena, e: ExprId) -> Option<(Self, ExprId)> {
        match *arena.node(e) {
            ExprNode::Exp(a) => Some((Self::Exp, a)),
            ExprNode::Sin(a) => Some((Self::Sin, a)),
            ExprNode::Cos(a) => Some((Self::Cos, a)),
            ExprNode::Sinh(a) => Some((Self::Sinh, a)),
            ExprNode::Cosh(a) => Some((Self::Cosh, a)),
            _ => None,
        }
    }

    fn apply(self, arena: &mut Arena, a: ExprId) -> ExprId {
        match self {
            Self::Exp => arena.exp(a),
            Self::Sin => arena.sin(a),
            Self::Cos => arena.cos(a),
            Self::Sinh => arena.sinh(a),
            Self::Cosh => arena.cosh(a),
        }
    }

    /// `(g, s)` with `f′ = s·g`.
    fn derivative(self) -> (Self, i64) {
        match self {
            Self::Exp => (Self::Exp, 1),
            Self::Sin => (Self::Cos, 1),
            Self::Cos => (Self::Sin, -1),
            Self::Sinh => (Self::Cosh, 1),
            Self::Cosh => (Self::Sinh, 1),
        }
    }
}

/// Largest power `k` of a linear denominator `ℓᵏ` that
/// [`try_special_function_integral`] reduces by parts.
const MAX_SPECIAL_ORDER: i64 = 12;

/// Is the constant `e` written with a negative sign: a negative number, or a
/// product with a negative numeric factor?
fn has_negative_sign(arena: &Arena, e: ExprId) -> bool {
    match arena.node(e) {
        ExprNode::Num(_) => arena.as_num(e).is_some_and(Signed::is_negative),
        ExprNode::Neg(_) => true,
        ExprNode::Mul(children) => children
            .iter()
            .any(|&c| arena.as_num(c).is_some_and(Signed::is_negative)),
        _ => false,
    }
}

/// `f(c)` for `f` = `sin`, `cos`, `sinh`, `cosh` and a constant `c`, with a
/// negative sign of `c` taken out by parity (`sin(−1) = −sin 1`), which the
/// canonical form does not do.
fn parity_folded(arena: &mut Arena, f: LinearSpecial, c: ExprId) -> ExprId {
    if !has_negative_sign(arena, c) {
        return f.apply(arena, c);
    }
    let neg = arena.neg(c);
    let neg = crate::transforms::eval::eval(arena, neg);
    let v = f.apply(arena, neg);
    match f {
        LinearSpecial::Sin | LinearSpecial::Sinh => arena.neg(v),
        _ => v,
    }
}

/// `∫ f(w + β′)·ℓ⁻¹ dℓ` for `w = κ·ℓ` (`κ`, `β′` free of `ℓ`): `e^{β′}·Ei(w)`,
/// `sin β′·Ci(w) + cos β′·Si(w)`, `cos β′·Ci(w) − sin β′·Si(w)` (SymPy's
/// `EiRule`, `SiRule`, `CiRule`), `Shi(w)`, `Chi(w)` for `β′ = 0` and
/// `(e^{β′}·Ei(w) ∓ e^{−β′}·Ei(−w))/2` for `sinh`, `cosh` otherwise.  (SymPy's
/// `sinh β′·Chi(w) + cosh β′·Shi(w)` has two terms of the size `e^{|β′|+|w|}`
/// that cancel to `e^{|w|−|β′|}` on one side: `∫ cosh(12x)/(x + 2)` lost 21
/// digits for `x > −2`; each `Ei` term is of its own size.)  For a `w` with
/// a negative sign the even `Ci`, `Chi` take `−w` and the odd `Si`, `Shi`
/// change sign with it (`∫ cos(3x)/x` and `∫ cos(−3x)/x` are both `Ci(3x)`);
/// `Ei` has no symmetry.
fn linear_special_base(arena: &mut Arena, fam: LinearSpecial, w: ExprId, beta: ExprId) -> ExprId {
    let zero_shift = arena.is_zero_structural(beta);
    if matches!(fam, LinearSpecial::Sinh | LinearSpecial::Cosh) && !zero_shift {
        let neg_w = arena.neg(w);
        let neg_w = crate::transforms::expand::expand(arena, neg_w);
        let neg_beta = arena.neg(beta);
        let neg_beta = crate::transforms::eval::eval(arena, neg_beta);
        let up = linear_special_base(arena, LinearSpecial::Exp, w, beta);
        let down = linear_special_base(arena, LinearSpecial::Exp, neg_w, neg_beta);
        let half = arena.rational(1, 2);
        let combined = if fam == LinearSpecial::Sinh {
            arena.sub(up, down)
        } else {
            arena.add(&[up, down])
        };
        return arena.mul(&[half, combined]);
    }
    let (w, odd_sign) = if fam != LinearSpecial::Exp && has_negative_sign(arena, w) {
        let neg = arena.neg(w);
        (crate::transforms::expand::expand(arena, neg), -1)
    } else {
        (w, 1)
    };
    let odd_sign = arena.int(odd_sign);
    match fam {
        LinearSpecial::Exp => {
            let ei = arena.ei(w);
            if zero_shift {
                ei
            } else {
                let e = arena.exp(beta);
                arena.mul(&[e, ei])
            }
        }
        LinearSpecial::Sin | LinearSpecial::Cos => {
            let si = arena.si(w);
            let si = arena.mul(&[odd_sign, si]);
            if zero_shift {
                return if fam == LinearSpecial::Sin {
                    si
                } else {
                    arena.ci(w)
                };
            }
            let ci = arena.ci(w);
            let s = parity_folded(arena, LinearSpecial::Sin, beta);
            let c = parity_folded(arena, LinearSpecial::Cos, beta);
            if fam == LinearSpecial::Sin {
                let a = arena.mul(&[s, ci]);
                let b = arena.mul(&[c, si]);
                arena.add(&[a, b])
            } else {
                let a = arena.mul(&[c, ci]);
                let b = arena.mul(&[s, si]);
                arena.sub(a, b)
            }
        }
        LinearSpecial::Sinh => {
            let shi = special_apply(arena, crate::base::libfn::LibFn::Shi, w);
            arena.mul(&[odd_sign, shi])
        }
        LinearSpecial::Cosh => special_apply(arena, crate::base::libfn::LibFn::Chi, w),
    }
}

/// `Jₖ = ∫ f(L)·ℓ⁻ᵏ dℓ` for `L = κ·ℓ + β′` (`k ≥ 1`), by parts:
/// `Jₖ[f] = −f(L)·ℓ^{1−k}/(k − 1) + s·κ/(k − 1)·J_{k−1}[g]` with `f′ = s·g`,
/// down to [`linear_special_base`].  Computed upwards for `f` and `g` together.
#[allow(clippy::too_many_arguments)]
fn linear_special_power(
    arena: &mut Arena,
    fam: LinearSpecial,
    big_l: ExprId,
    ell: ExprId,
    kappa: ExprId,
    w: ExprId,
    beta: ExprId,
    k: i64,
) -> ExprId {
    let partner = fam.derivative().0;
    let fams: SmallVec<[LinearSpecial; 2]> = if partner == fam {
        SmallVec::from_slice(&[fam])
    } else {
        SmallVec::from_slice(&[fam, partner])
    };
    let mut level: SmallVec<[ExprId; 2]> = fams
        .iter()
        .map(|&f| linear_special_base(arena, f, w, beta))
        .collect();
    for j in 2..=k {
        let inv = arena.rational(1, j - 1);
        let power = arena.int(1 - j);
        let ell_pow = arena.pow(ell, power);
        let mut next: SmallVec<[ExprId; 2]> = SmallVec::new();
        for &f in &fams {
            let (g, s) = f.derivative();
            let prev_g = level[fams.iter().position(|&h| h == g).unwrap_or(0)];
            let f_l = f.apply(arena, big_l);
            let neg_inv = arena.neg(inv);
            let boundary = arena.mul(&[neg_inv, f_l, ell_pow]);
            let s = arena.int(s);
            let carried = arena.mul(&[s, inv, kappa, prev_g]);
            next.push(arena.add(&[boundary, carried]));
        }
        level = next;
    }
    level[0]
}

/// `∫ f(α·x + β)·R(x) dx` for `f` = `exp`, `sin`, `cos`, `sinh`, `cosh` and a
/// rational `R` whose partial fractions ([`crate::transforms::apart::apart`])
/// are a polynomial and terms `c·ℓ⁻ᵏ` over linear `ℓ = γ·x + δ`: with
/// `κ = α/γ`, `β′ = β − α·δ/γ` the term is `c/γ·∫ f(κ·ℓ + β′)·ℓ⁻ᵏ dℓ`, by
/// parts down to `Ei`, `Si`/`Ci`, `Shi`/`Chi` ([`linear_special_power`]);
/// the polynomial times `f` goes back to `integrate_node`.  Up to 0.43 only
/// `eˣ/x`, `sin x/x`, `cos x/x`, `sinh x/x`, `cosh x/x` were recognised:
/// `∫ e^{12x}/x dx` stayed unevaluated (SymPy: `Ei(12·x)`) and with it 52
/// Riccati equations of the ODE solver, as did `∫ e^{2x}/(x + 1)`,
/// `∫ eˣ/x²`, `∫ sin(3x)/x`.  For a rate `α` that may vanish, the cases
/// `α = 0` come with the answer ([`vanishing_parameter_values`]).  A
/// denominator with a factor irreducible over ℚ of degree ≥ 2 goes to
/// [`special_over_rational_factors`] (up to 0.44 `∫ e^{2x}/(x² − 2) dx`,
/// `∫ sin x/(x² − 2) dx` stayed unevaluated).
///
/// The arguments `κ·ℓ` vanish only at the root of `ℓ`, a pole of the term,
/// so the branch cuts of `Ei`, `Ci`, `Chi` on the negative axis are met
/// there alone for a real `κ` (`Ei` takes its real value on the axis).
fn try_special_function_integral(
    arena: &mut Arena,
    dependent: &[ExprId],
    constants: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    let mut special: Option<(ExprId, LinearSpecial, ExprId, ExprId, ExprId)> = None;
    let mut rest: SmallVec<[ExprId; 4]> = SmallVec::new();
    for &d in dependent {
        if special.is_none()
            && let Some((fam, arg)) = LinearSpecial::of(arena, d)
            && let Some((alpha, beta)) = symbolic_linear_coeff_of(arena, arg, var, var_sym)
        {
            special = Some((d, fam, arg, alpha, beta));
        } else {
            rest.push(d);
        }
    }
    let (f_node, fam, big_l, alpha, beta) = special?;
    if rest.is_empty() {
        return None;
    }
    let r = arena.mul(&rest);
    if !is_rational_in(arena, r, var_sym) {
        return None;
    }
    let (_, denom) = crate::poly::polybridge::as_numer_denom(arena, r);
    if !contains_var(arena, denom, var_sym) {
        return None;
    }
    let cases = vanishing_parameter_values(arena, alpha, var_sym)?;
    let parts = crate::transforms::apart::apart(arena, r, var);
    let terms: SmallVec<[ExprId; 6]> = match arena.node(parts) {
        ExprNode::Add(children) => children.clone(),
        _ => SmallVec::from_slice(&[parts]),
    };
    let mut poly_terms: SmallVec<[ExprId; 4]> = SmallVec::new();
    let mut pieces: SmallVec<[ExprId; 6]> = SmallVec::new();
    let mut linear = true;
    for t in terms {
        if !contains_var(arena, t, var_sym) || is_polynomial_in(arena, t, var, var_sym) {
            poly_terms.push(t);
            continue;
        }
        let (c, dep) = match split_constant_factors(arena, t, var_sym) {
            Some((c, dep)) => (arena.mul(&c), dep),
            None => (arena.one, t),
        };
        let Some((ell, k, gamma, delta)) = linear_power_parts(arena, dep, var, var_sym) else {
            linear = false;
            break;
        };
        if k > MAX_SPECIAL_ORDER {
            return None;
        }
        let kappa = arena.div(alpha, gamma);
        let kappa = crate::transforms::eval::eval(arena, kappa);
        let shift = arena.mul(&[kappa, delta]);
        let beta_shifted = arena.sub(beta, shift);
        let beta_shifted = crate::transforms::eval::eval(arena, beta_shifted);
        let w = arena.mul(&[kappa, ell]);
        let w = crate::transforms::expand::expand(arena, w);
        let j = linear_special_power(arena, fam, big_l, ell, kappa, w, beta_shifted, k);
        let c_over_gamma = arena.div(c, gamma);
        pieces.push(arena.mul(&[c_over_gamma, j]));
    }
    if !linear {
        // A factor irreducible over ℚ of degree ≥ 2 (`x² − 2`, `x³ + x + 1`):
        // the partial fractions over ℚ, each summed over the roots.
        let special = LinearSpecialArg {
            fam,
            big_l,
            alpha,
            beta,
        };
        let over_q = special_over_rational_factors(arena, &special, r, var)?;
        pieces = SmallVec::from_slice(&[over_q.proper]);
        poly_terms.clear();
        if !arena.is_zero_structural(over_q.polynomial) {
            poly_terms.push(over_q.polynomial);
        }
    }
    if !poly_terms.is_empty() {
        // P·f(L) as a sum of monomials times f(L) (`expand` of the product
        // would also split `e^{αx+β}` into `e^{αx}·e^β`).
        let p = arena.add(&poly_terms);
        let p = crate::transforms::expand::expand(arena, p);
        let monomials: SmallVec<[ExprId; 6]> = match arena.node(p) {
            ExprNode::Add(children) => children.clone(),
            _ => SmallVec::from_slice(&[p]),
        };
        let products: SmallVec<[ExprId; 6]> =
            monomials.iter().map(|&m| arena.mul(&[m, f_node])).collect();
        let p_f = arena.add(&products);
        let v = integrate_node(arena, p_f, var, var_sym, depth - 1);
        if crate::base::walk::has_unevaluated(arena, v) {
            return None;
        }
        pieces.push(v);
    }
    tracing::debug!("integrate: exponential-integral family over linear denominators");
    let sum = arena.add(&pieces);
    // `Ci`/`Chi` next to the real logarithm `ln|x|` of the polynomial part
    // (`∫ (1 − cos 4x)/x dx = ln|x| − Ci(4x)`): `Ci` of the real `|w|`, or the
    // sum steps by `iπ` where the integrand is finite.
    let has_real_log = crate::base::walk::post_order_ids(arena, sum)
        .into_iter()
        .any(|id| matches!(*arena.node(id), ExprNode::Ln(a) if matches!(arena.node(a), ExprNode::Abs(_))));
    let sum = if has_real_log {
        ci_real_on_real_line(arena, sum, var, var_sym)
    } else {
        sum
    };
    let generic = wrap_with_constants(arena, sum, constants);
    let mut all: SmallVec<[ExprId; 6]> = constants.iter().copied().collect();
    all.extend(dependent.iter().copied());
    let integrand = arena.mul(&all);
    // A sum over non-real roots cannot be checked while `Ei`, `Ci`, `Si`,
    // `Chi`, `Shi` of complex arguments do not evaluate: a refused candidate
    // leaves the later routes their chance.
    if !linear && candidate_rejected(arena, integrand, generic, var, var_sym) {
        return None;
    }
    with_parameter_cases(arena, generic, integrand, &cases, var, var_sym, depth)
}

/// `Σ f(rₖ)/|g′(rₖ)|·H(x − rₖ)` over the real roots `rₖ` of `g` (all of them,
/// numeric, simple), an antiderivative of `f·δ(g)`.
fn delta_simple_roots(arena: &mut Arena, f: ExprId, g: ExprId, var: ExprId) -> Option<ExprId> {
    let roots = real_roots(arena, g, var)?;
    if roots.len() < 2 {
        return None;
    }
    let dg = crate::transforms::diff::diff(arena, g, var);
    let mut terms: Vec<ExprId> = Vec::with_capacity(roots.len());
    for r in roots {
        let slope = crate::transforms::subs::subs(arena, dg, var, r);
        let slope = crate::transforms::eval::eval(arena, slope);
        let s = crate::transforms::evalf::eval_const_f64(arena, slope)?;
        // A nonzero printed digit is certified; 0 (a double root?) refuses.
        if s == 0.0 || !s.is_finite() {
            return None;
        }
        let at = crate::transforms::subs::subs(arena, f, var, r);
        let at = crate::transforms::eval::eval(arena, at);
        if contains_non_finite(arena, at) {
            return None;
        }
        let abs_slope = arena.abs(slope);
        let x_minus_r = arena.sub(var, r);
        let h = arena.heaviside(x_minus_r);
        let t = arena.div(at, abs_slope);
        terms.push(arena.mul(&[t, h]));
    }
    let sum = arena.add(&terms);
    Some(crate::transforms::eval::eval(arena, sum))
}

/// `∫ xᵃ·e^{c·xᵖ + d} dx = eᵈ·x^{a+1}·z^{−s}·γ(s, z)/p` with `z = −c·xᵖ`,
/// `s = (a + 1)/p` (constants `a`, `c ≠ 0`, `d`, a rational `p > 0`; DLMF
/// 8.8.13 `dγ(s, z)/dz = z^{s−1}·e^{−z}`).  `z^{−s}·γ(s, z)` is entire in `z`
/// (DLMF 8.7.1), so the answer is continuous wherever `x^{a+1}` is; the
/// upper function `Γ(s, z)` differs by `x^{a+1}·z^{−s}·Γ(s)`, constant on
/// each side of `x = 0` but not the same constant, and jumped there
/// (`∫ e^{−x³} dx`, found by the hunt).  The last resort for the family: the elementary and `erf`
/// cases come from the earlier routes.  Up to 0.46 `∫ x^y·e^{−x} dx`,
/// `∫ xⁿ·e^{−x} dx` (`n` an integer symbol), `∫ x^{1/3}·e^{−x/8} dx` and
/// `∫ e^{−x³} dx` stayed unevaluated (SymPy's test_integrals, test_manual
/// and test_meijerint answer them with `lowergamma`/`uppergamma`).
fn try_incomplete_gamma(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    let mut a_terms: SmallVec<[ExprId; 2]> = SmallVec::new();
    let mut exp_arg: Option<ExprId> = None;
    for &d in dependent {
        match *arena.node(d) {
            ExprNode::Exp(g) if exp_arg.is_none() => exp_arg = Some(g),
            ExprNode::Pow(b, e) if b == var && !contains_var(arena, e, var_sym) => a_terms.push(e),
            _ if d == var => a_terms.push(arena.one),
            _ => return None,
        }
    }
    let g = exp_arg?;
    // g = c·xᵖ + d
    let terms: Vec<ExprId> = match arena.node(g) {
        ExprNode::Add(children) => children.to_vec(),
        _ => vec![g],
    };
    let mut shift: SmallVec<[ExprId; 2]> = SmallVec::new();
    let mut mono: Option<(ExprId, Q)> = None;
    for t in terms {
        if !contains_var(arena, t, var_sym) {
            shift.push(t);
            continue;
        }
        if mono.is_some() {
            return None;
        }
        let (c, n) = monomial_parts(arena, t, var, var_sym)?;
        let n = arena.as_num(n)?.clone();
        mono = Some((c, n));
    }
    let (c, p) = mono?;
    // A negative power (`e^{1/x}`) is the substitution `x = 1/t`'s.
    if !p.is_positive() || arena.is_zero_structural(c) {
        return None;
    }
    let a = if a_terms.is_empty() {
        arena.zero
    } else {
        arena.add(&a_terms)
    };
    let one = arena.one;
    let a1 = arena.add(&[a, one]);
    let a1 = crate::transforms::eval::eval(arena, a1);
    let p_id = arena.num_ratio(p.clone());
    let s = arena.div(a1, p_id);
    let s = crate::transforms::eval::eval(arena, s);
    // A numeric `s` that is a positive integer (an elementary answer) or a
    // half-integer (`erf`) is the other routes'.
    if let Some(sv) = arena.as_num(s)
        && ((sv.is_integer() && sv.is_positive())
            || *(sv * Q::from_integer(2.into())).denom() == num_bigint::BigInt::from(1))
    {
        return None;
    }
    let p_id2 = arena.num_ratio(p);
    let x_p = arena.pow(var, p_id2);
    let neg_c = arena.neg(c);
    let z = arena.mul(&[neg_c, x_p]);
    let z = crate::transforms::eval::eval(arena, z);
    let gamma = arena.lib_apply(crate::base::libfn::LibFn::LowerGamma, &[s, z]);
    let neg_s = arena.neg(s);
    let z_pow = arena.pow(z, neg_s);
    let x_a1 = arena.pow(var, a1);
    let one = arena.one;
    let inv_p = arena.div(one, p_id);
    let mut factors: SmallVec<[ExprId; 5]> = SmallVec::from_slice(&[inv_p, x_a1, z_pow, gamma]);
    if !shift.is_empty() {
        let d = arena.add(&shift);
        factors.push(arena.exp(d));
    }
    tracing::debug!("integrate: x^a·e^(c·x^p) as an incomplete gamma function");
    let r = arena.mul(&factors);
    Some(crate::transforms::eval::eval(arena, r))
}

/// `∫ f(n, …, a·x + b) dx` for a classical orthogonal polynomial `f` of any
/// (also symbolic) degree `n` by its derivative rule read backwards
/// (DLMF 18.9.19–18.9.23, Abramowitz–Stegun 22.8): `∫ Pₙ = (Pₙ₊₁ −
/// Pₙ₋₁)/(2n + 1)`, `∫ Tₙ = (Tₙ₊₁/(n + 1) − Tₙ₋₁/(n − 1))/2`, `∫ Uₙ =
/// Tₙ₊₁/(n + 1)`, `∫ Hₙ = Hₙ₊₁/(2(n + 1))`, `∫ Lₙ = Lₙ − Lₙ₊₁`, `∫ Lₙ⁽ᵅ⁾ =
/// −Lₙ₊₁⁽ᵅ⁻¹⁾`, `∫ Cₙ⁽ᵅ⁾ = Cₙ₊₁⁽ᵅ⁻¹⁾/(2(α − 1))`, `∫ Pₙ⁽ᵅʲᵝ⁾ =
/// 2·Pₙ₊₁⁽ᵅ⁻¹ʲᵝ⁻¹⁾/(n + α + β)`, each over `a`; the vanishing denominators
/// are degenerate cases of the wrap.  Up to 0.46 none was integrated
/// (SymPy's test_manual integrates them all).
fn orthogonal_polynomial_integral(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    use crate::base::libfn::LibFn;
    let ExprNode::Lib(f, args) = arena.node(expr).clone() else {
        return None;
    };
    let g = *args.last()?;
    let params = &args[..args.len() - 1];
    if params.is_empty() || params.iter().any(|&p| contains_var(arena, p, var_sym)) {
        return None;
    }
    let (a, _) = symbolic_linear_coeff_of(arena, g, var, var_sym)?;
    let n = params[0];
    let one = arena.one;
    let two = arena.int(2);
    let n1 = arena.add(&[n, one]);
    let n1 = crate::transforms::eval::eval(arena, n1);
    let minus_one = arena.int(-1);
    let nm1 = arena.add(&[n, minus_one]);
    let nm1 = crate::transforms::eval::eval(arena, nm1);
    let r = match (f, params.len()) {
        (LibFn::Legendre, 1) => {
            let up = arena.lib_apply(f, &[n1, g]);
            let down = arena.lib_apply(f, &[nm1, g]);
            let diff = arena.sub(up, down);
            let two_n = arena.mul(&[two, n]);
            let den = arena.add(&[two_n, one]);
            arena.div(diff, den)
        }
        (LibFn::ChebyshevT, 1) => {
            let up = arena.lib_apply(f, &[n1, g]);
            let down = arena.lib_apply(f, &[nm1, g]);
            let t1 = arena.div(up, n1);
            let t2 = arena.div(down, nm1);
            let diff = arena.sub(t1, t2);
            arena.div(diff, two)
        }
        (LibFn::ChebyshevU, 1) => {
            let t = arena.lib_apply(LibFn::ChebyshevT, &[n1, g]);
            arena.div(t, n1)
        }
        (LibFn::Hermite, 1) => {
            let h = arena.lib_apply(f, &[n1, g]);
            let den = arena.mul(&[two, n1]);
            arena.div(h, den)
        }
        (LibFn::Laguerre, 1) => {
            let up = arena.lib_apply(f, &[n1, g]);
            arena.sub(expr, up)
        }
        (LibFn::AssocLaguerre, 2) => {
            let alpha = arena.add(&[params[1], minus_one]);
            let up = arena.lib_apply(f, &[n1, alpha, g]);
            arena.neg(up)
        }
        (LibFn::Gegenbauer, 2) => {
            let alpha = arena.add(&[params[1], minus_one]);
            let up = arena.lib_apply(f, &[n1, alpha, g]);
            let den = arena.mul(&[two, alpha]);
            arena.div(up, den)
        }
        (LibFn::Jacobi, 3) => {
            let alpha = arena.add(&[params[1], minus_one]);
            let beta = arena.add(&[params[2], minus_one]);
            let up = arena.lib_apply(f, &[n1, alpha, beta, g]);
            let den = arena.add(&[n, params[1], params[2]]);
            let num = arena.mul(&[two, up]);
            arena.div(num, den)
        }
        _ => return None,
    };
    tracing::debug!("integrate: orthogonal polynomial by its derivative rule");
    Some(arena.div(r, a))
}

/// `∫ xᵐ·Z_ν(a·x) dx` for `Z` = `J`, `Y`, `I`, `K` and `m = ν + 1` or `m = 1 − ν`
/// (DLMF 10.6.6, 10.29.4: `(x^{ν+1}Z_{ν+1})′ = x^{ν+1}Z_ν`, `(x^{−ν}Z_ν)′ =
/// −x^{−ν}Z_{ν+1}`, the signs of `I`, `K` as there), and `∫ Z_{±1}(a·x + b)`.
/// `∫ x·J₀(x) dx = x·J₁(x)`, `∫ J₁(x) dx = −J₀(x)` stayed unevaluated up to
/// 0.46 (SymPy's test_meijerint).
fn bessel_power_integral(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    use crate::base::libfn::LibFn;
    let mut bessel: Option<(LibFn, ExprId, ExprId)> = None;
    let mut m_terms: SmallVec<[ExprId; 2]> = SmallVec::new();
    for &d in dependent {
        if let ExprNode::Lib(
            f @ (LibFn::BesselJ | LibFn::BesselY | LibFn::BesselI | LibFn::BesselK),
            args,
        ) = arena.node(d).clone()
            && args.len() == 2
            && !contains_var(arena, args[0], var_sym)
            && bessel.is_none()
        {
            bessel = Some((f, args[0], args[1]));
        } else if d == var {
            m_terms.push(arena.one);
        } else if let ExprNode::Pow(b, e) = *arena.node(d)
            && b == var
            && !contains_var(arena, e, var_sym)
        {
            m_terms.push(e);
        } else {
            return None;
        }
    }
    let (f, nu, g) = bessel?;
    let (a, b) = symbolic_linear_coeff_of(arena, g, var, var_sym)?;
    let m = if m_terms.is_empty() {
        arena.zero
    } else {
        arena.add(&m_terms)
    };
    let m = crate::transforms::eval::eval(arena, m);
    if !arena.is_zero_structural(b) && !arena.is_zero_structural(m) {
        return None;
    }
    let one = arena.one;
    let nu1 = arena.add(&[nu, one]);
    let nu1 = crate::transforms::eval::eval(arena, nu1);
    let minus_one = arena.int(-1);
    let num1 = arena.add(&[nu, minus_one]);
    let num1 = crate::transforms::eval::eval(arena, num1);
    let up_case = arena.sub(m, nu1);
    let up_case = crate::transforms::eval::eval(arena, up_case);
    let one_minus_nu = arena.sub(one, nu);
    let down_case = arena.sub(m, one_minus_nu);
    let down_case = crate::transforms::eval::eval(arena, down_case);
    let (order, sign) = if arena.is_zero_structural(up_case) {
        (nu1, if f == LibFn::BesselK { -1 } else { 1 })
    } else if arena.is_zero_structural(down_case) {
        (num1, if f == LibFn::BesselI { 1 } else { -1 })
    } else {
        return None;
    };
    let z = arena.lib_apply(f, &[order, g]);
    let x_m = arena.pow(var, m);
    let s = arena.int(sign);
    let r = arena.mul(&[s, x_m, z]);
    tracing::debug!("integrate: Bessel function times a power by its derivative rule");
    Some(arena.div(r, a))
}

/// Most factors `exp`, `sin`, `cos`, `sinh`, `cosh` (with multiplicity) that
/// [`try_special_product_as_exponentials`] multiplies out.
const MAX_EXPONENTIAL_PRODUCT: i64 = 4;

/// `∫ f₁(L₁)·…·fₖ(Lₖ)·R(x) dx` for two or more factors (with multiplicity)
/// `f` = `exp`, `sin`, `cos`, `sinh`, `cosh` of linear arguments with numeric
/// coefficients and a rational `R` with a denominator: the factors written
/// as exponentials (`cos x = (e^{ix} + e^{−ix})/2`), multiplied out into
/// terms `c·e^{α·x + β}·R(x)`, each the exponential-integral family of
/// [`try_special_function_integral`] at a complex rate `α` (`Ei` at complex
/// points).  `∫ eˣ·cos x/x dx`, `∫ sin x·e^{−ix}/x dx`, `∫ e^{−x}·sin x/(x²
/// + 1) dx` stayed unevaluated up to 0.46 (SymPy's test_integrals; SymPy
/// answers the first two in this form).  `None` when an `Ei` argument
/// crosses its cut, the negative axis, at a real point where the integrand
/// is finite ([`ei_crosses_cut`]): the answer would jump there.
fn try_special_product_as_exponentials(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    if depth < 3 || dependent.len() < 2 {
        return None;
    }
    let mut specials: SmallVec<[ExprId; 4]> = SmallVec::new();
    let mut rest: SmallVec<[ExprId; 4]> = SmallVec::new();
    let mut count: i64 = 0;
    for &d in dependent {
        let (base, power) = match arena.node(d) {
            ExprNode::Pow(b, e) => match arena.as_num(*e).and_then(|q| {
                (q.is_integer() && q.is_positive())
                    .then(|| num_traits::ToPrimitive::to_i64(q.numer()))
                    .flatten()
            }) {
                Some(k) => (*b, k),
                None => (d, 0),
            },
            _ => (d, 1),
        };
        let special = power > 0
            && LinearSpecial::of(arena, base).is_some_and(|(_, arg)| {
                symbolic_linear_coeff_of(arena, arg, var, var_sym).is_some()
                    && crate::base::walk::free_symbols(arena, arg)
                        .iter()
                        .all(|&s| matches!(arena.node(s), ExprNode::Symbol(t) if *t == var_sym))
            });
        if special {
            count += power;
            specials.push(d);
        } else {
            rest.push(d);
        }
    }
    if !(2..=MAX_EXPONENTIAL_PRODUCT).contains(&count) || rest.is_empty() {
        return None;
    }
    let r = arena.mul(&rest);
    if !is_rational_in(arena, r, var_sym) {
        return None;
    }
    let (_, denom) = crate::poly::polybridge::as_numer_denom(arena, r);
    if !contains_var(arena, denom, var_sym) {
        return None;
    }
    let s = arena.mul(&specials);
    let s = linearized_trig_product(arena, &specials, var, var_sym).unwrap_or_else(|| {
        let s = arena.rewrite_as_exp_expr(s);
        let s = crate::transforms::eval::eval(arena, s);
        crate::transforms::expand::expand(arena, s)
    });
    let terms: Vec<ExprId> = match arena.node(s) {
        ExprNode::Add(children) => children.to_vec(),
        _ => vec![s],
    };
    let mut pieces: SmallVec<[ExprId; 8]> = SmallVec::new();
    for t in terms {
        let factors: Vec<ExprId> = match arena.node(t) {
            ExprNode::Mul(children) => children.to_vec(),
            _ => vec![t],
        };
        let merged = merge_exponential_factors(arena, &factors, var, var_sym).unwrap_or(t);
        let integrand = arena.mul(&[merged, r]);
        let integrand = crate::transforms::eval::eval(arena, integrand);
        let piece = integrate_node(arena, integrand, var, var_sym, depth - 1);
        if crate::base::walk::has_unevaluated(arena, piece) {
            return None;
        }
        pieces.push(piece);
    }
    let sum = arena.add(&pieces);
    let sum = ei_continuous_on_real_line(arena, sum, var, var_sym)?;
    let sum = ci_real_on_real_line(arena, sum, var, var_sym);
    tracing::debug!("integrate: exponential and trigonometric factors as complex exponentials");
    Some(sum)
}

/// The product of the sine and cosine factors `specials` (all of them
/// such, of linear arguments) as a sum of terms with one sine or cosine
/// each (`cos²x = (1 + cos 2x)/2`, `sin x·cos 3x = (sin 4x − sin 2x)/2`), so
/// that their integrals over a rational function come out in `Si`, `Ci`
/// rather than in `Ei` at imaginary points.  `None` for other factors or
/// when a term keeps two of them.
fn linearized_trig_product(
    arena: &mut Arena,
    specials: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    let trig = |arena: &Arena, e: ExprId| {
        let base = match *arena.node(e) {
            ExprNode::Pow(b, _) => b,
            _ => e,
        };
        matches!(
            LinearSpecial::of(arena, base),
            Some((LinearSpecial::Sin | LinearSpecial::Cos, _))
        )
    };
    if !specials.iter().all(|&d| trig(arena, d)) {
        return None;
    }
    let s = arena.mul(specials);
    let s = crate::transforms::expand::expand(arena, s);
    let s = match crate::transforms::trig_integ::linearize_trig_powers(arena, &[s], var, var_sym) {
        Some(l) => l,
        None => s,
    };
    let s = arena.trig_combine_expr(s);
    let s = crate::transforms::eval::eval(arena, s);
    let s = crate::transforms::expand::expand(arena, s);
    let terms: Vec<ExprId> = match arena.node(s) {
        ExprNode::Add(children) => children.to_vec(),
        _ => vec![s],
    };
    for t in terms {
        let factors: Vec<ExprId> = match arena.node(t) {
            ExprNode::Mul(children) => children.to_vec(),
            _ => vec![t],
        };
        let n = factors
            .iter()
            .filter(|&&f| contains_var(arena, f, var_sym))
            .count();
        let single = factors.iter().all(|&f| {
            !contains_var(arena, f, var_sym)
                || matches!(
                    LinearSpecial::of(arena, f),
                    Some((LinearSpecial::Sin | LinearSpecial::Cos, _))
                )
        });
        if n > 1 || !single {
            return None;
        }
    }
    Some(s)
}

/// `e` with every `Ci(w)`, `Chi(w)` of a real linear `w = κ·x + β` written
/// `Ci(w) − ln w + ln|w|`: `Ci(z) − ln z` is entire, so this is `Ci(|w|)` for
/// real `x`, real on both sides of the zero of `w`, where the principal
/// `Ci(w)` has `+iπ`.  A sum of such terms whose integrand is finite there
/// (`∫ sin²(2x)/x dx = (ln|x| − Ci(4x))/2`) is then continuous; with `Ci(4x)`
/// it stepped by `iπ/2` at 0 (found by the hunt).
fn ci_real_on_real_line(arena: &mut Arena, e: ExprId, var: ExprId, var_sym: SymbolId) -> ExprId {
    let mut out = e;
    for id in crate::base::walk::post_order_ids(arena, e) {
        let w = match arena.node(id) {
            ExprNode::Ci(w) => *w,
            ExprNode::Lib(crate::base::libfn::LibFn::Chi, args) if args.len() == 1 => args[0],
            _ => continue,
        };
        if !contains_var(arena, w, var_sym) {
            continue;
        }
        let Some((k, b)) = symbolic_linear_coeff_of(arena, w, var, var_sym) else {
            continue;
        };
        // Proved real: `ln|w| − ln w` is locally constant only for a real
        // `w` (up to 0.52 an imaginary part below `10⁻¹²` of the value was
        // taken for 0).
        let real = |arena: &mut Arena, c: ExprId| {
            crate::transforms::realness::imaginary_part_sign(arena, c)
                .is_zero()
                .is_true()
                && crate::transforms::evalf::evalf_complex64(arena, c)
                    .is_ok_and(|z| z.re.is_finite())
        };
        if !real(arena, k) || !real(arena, b) {
            continue;
        }
        let ln_w = arena.ln(w);
        let abs_w = arena.abs(w);
        let ln_abs = arena.ln(abs_w);
        let fix = arena.sub(ln_abs, ln_w);
        let fixed = arena.add(&[id, fix]);
        out = arena.subs_structural(out, id, fixed);
    }
    if out == e {
        e
    } else {
        crate::transforms::eval::eval(arena, out)
    }
}

/// `e` with every `Ei(p·x + q)` (numeric `p`, `q`) whose argument crosses
/// the negative real axis — the cut, where `Ei` jumps by `2πi` — at a real
/// `x` written `Ei(z) − ln z + ln(x − c)` for its zero `c = −q/p`: `Ei(z) −
/// ln z` is entire, and `ln(x − c)` keeps off its cut for real `x` (`c` is
/// not real when the crossing is away from it), so the antiderivative is
/// continuous where its integrand is, with the same derivative.  (A real
/// argument stays on the axis, where `Ei` takes its real value; one with a
/// constant non-zero imaginary part never meets it.)  `None` when an `Ei`
/// of the variable has an argument of another form.
fn ei_continuous_on_real_line(
    arena: &mut Arena,
    e: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    let mut out = e;
    for id in crate::base::walk::post_order_ids(arena, e) {
        let ExprNode::Ei(z) = *arena.node(id) else {
            continue;
        };
        if !contains_var(arena, z, var_sym) {
            continue;
        }
        let (p, q) = symbolic_linear_coeff_of(arena, z, var, var_sym)?;
        // Where the argument meets the real axis, and on which side of the
        // branch point, by certified signs.  Up to 0.52 they were `f64`
        // values within `10⁻¹²`: a crossing `2·10⁻¹⁴` left of 0 counted as
        // off the cut, and `∫₀² e^{(1+i)x}/(x − 1 + 10⁻¹⁴i) dx` (`4.7265 +
        // 2.6665i`) was `−9.6453 + 11.8945i`, the jump `2πi·e^{1+i}` of `Ei`
        // at `x ≈ 1` left in.
        use crate::base::truth::SignSet;
        let p_side = crate::transforms::realness::imaginary_part_sign(arena, p);
        if p_side == SignSet::REAL {
            return None;
        }
        if p_side == SignSet::ZERO {
            continue;
        }
        let zero_c = arena.div(q, p);
        let zero_c = arena.neg(zero_c);
        let zero_c = crate::transforms::expand::expand(arena, zero_c);
        let zero_c = crate::transforms::eval::eval(arena, zero_c);
        let c_side = crate::transforms::realness::imaginary_part_sign(arena, zero_c);
        if c_side == SignSet::REAL {
            return None;
        }
        if c_side == SignSet::ZERO {
            // The zero `c` of the argument is real: `Ei(z) − ln z` is entire
            // and `ln|x − c|` the real logarithm, whose singularities cancel
            // between the terms where the integrand is finite at `c`
            // (`∫ e^{−x}·sin x/x dx` stepped by `π` at 0 with the principal
            // logarithms of `Ei((−1 ± i)·x)`, found by the hunt).
            let x_minus_c = arena.sub(var, zero_c);
            let abs = arena.abs(x_minus_c);
            let ln_abs = arena.ln(abs);
            let ln_z = arena.ln(z);
            let continuous = arena.sub(ln_abs, ln_z);
            let continuous = arena.add(&[id, continuous]);
            out = arena.subs_structural(out, id, continuous);
            continue;
        }
        // At the real `x₀ = −im q/im p` the argument is `re(p·x₀ + q) =
        // −im(p̄·q)/im p`: on the cut (negative) when `im(p̄·q)` has the sign
        // of `im p`.
        let p_bar = arena.conjugate(p);
        let pq = arena.mul(&[p_bar, q]);
        let pq = crate::transforms::expand::expand(arena, pq);
        let pq = crate::transforms::eval::eval(arena, pq);
        let cross_side = crate::transforms::realness::imaginary_part_sign(arena, pq);
        if cross_side == SignSet::REAL || cross_side == SignSet::ZERO {
            return None;
        }
        if cross_side != p_side {
            continue;
        }
        // c = −q·p̄/(p·p̄), so that a Gaussian rational comes out as `a + b·i`.
        let num = arena.mul(&[q, p_bar]);
        let num = crate::transforms::expand::expand(arena, num);
        let den = arena.mul(&[p, p_bar]);
        let den = crate::transforms::expand::expand(arena, den);
        let c = arena.div(num, den);
        let c = arena.neg(c);
        let c = crate::transforms::expand::expand(arena, c);
        let c = crate::transforms::eval::eval(arena, c);
        let x_minus_c = arena.sub(var, c);
        let x_minus_c = crate::transforms::eval::eval(arena, x_minus_c);
        let ln_z = arena.ln(z);
        let ln_lin = arena.ln(x_minus_c);
        let continuous = arena.sub(ln_lin, ln_z);
        let continuous = arena.add(&[id, continuous]);
        out = arena.subs_structural(out, id, continuous);
    }
    Some(crate::transforms::eval::eval(arena, out))
}

/// `(ℓ, k, γ, δ)` when `dep` is `ℓ⁻ᵏ` for an integer `k ≥ 1` and a linear
/// `ℓ = γ·x + δ`.
fn linear_power_parts(
    arena: &mut Arena,
    dep: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<(ExprId, i64, ExprId, ExprId)> {
    let ExprNode::Pow(ell, e) = *arena.node(dep) else {
        return None;
    };
    let k = arena
        .as_num(e)
        .filter(|q| q.is_integer() && q.is_negative())?;
    let k = i64::try_from(-k.to_integer()).ok()?;
    let (gamma, delta) = symbolic_linear_coeff_of(arena, ell, var, var_sym)?;
    Some((ell, k, gamma, delta))
}

/// The function `f` of a [`try_special_function_integral`] integrand and its
/// linear argument `L = α·x + β`.
struct LinearSpecialArg {
    fam: LinearSpecial,
    /// `L` itself.
    big_l: ExprId,
    alpha: ExprId,
    beta: ExprId,
}

/// What [`special_over_rational_factors`] makes of `∫ f(L)·R(x) dx`.
struct SpecialOverQ {
    /// The antiderivative for the proper part of `R`.
    proper: ExprId,
    /// The polynomial part of `R`, whose product with `f(L)` the caller
    /// integrates.
    polynomial: ExprId,
}

/// Largest degree of the denominator of `R` in
/// [`special_over_rational_factors`] and [`try_log_times_rational`].
const MAX_SPECIAL_DENOM_DEGREE: usize = 24;

/// Largest degree (in the variable, numerator and denominator together, as
/// written) of a rational function that [`rational_over_q`] multiplies out.
const MAX_RATIONAL_DEGREE: u64 = 64;

/// A bound on the degree in `var` of the rational expression `e` multiplied
/// out (the degrees of the factors added, of the terms maximised, of a
/// power multiplied by its exponent, whatever its sign); `None` for a
/// power with a non-integer exponent.
fn rational_degree_bound(arena: &Arena, e: ExprId, var: ExprId) -> Option<u64> {
    let mut deg: FxHashMap<ExprId, u64> = FxHashMap::default();
    for id in crate::base::walk::post_order_ids(arena, e) {
        let d = if id == var {
            1
        } else {
            match arena.node(id) {
                ExprNode::Add(ch) => ch.iter().map(|c| deg.get(c).copied().unwrap_or(0)).max()?,
                ExprNode::Mul(ch) => ch.iter().try_fold(0u64, |acc, c| {
                    acc.checked_add(deg.get(c).copied().unwrap_or(0))
                })?,
                ExprNode::Neg(c) => deg.get(c).copied().unwrap_or(0),
                ExprNode::Pow(b, k) => {
                    let k = arena.as_num(*k).filter(|q| q.is_integer())?;
                    let k = u64::try_from(k.to_integer().magnitude()).ok()?;
                    deg.get(b).copied().unwrap_or(0).checked_mul(k)?
                }
                _ => 0,
            }
        };
        deg.insert(id, d);
    }
    deg.get(&e).copied()
}

/// `r` as `N/D` with `N, D ∈ ℚ[x]` and `D` not constant; `None` when a
/// coefficient is not rational or the degree is larger than
/// [`MAX_RATIONAL_DEGREE`].
fn rational_over_q(arena: &mut Arena, r: ExprId, var: ExprId) -> Option<(Poly, Poly)> {
    if rational_degree_bound(arena, r, var)? > MAX_RATIONAL_DEGREE {
        return None;
    }
    let (n, d) = crate::poly::polybridge::as_numer_denom(arena, r);
    let n = crate::transforms::expand::expand(arena, n);
    let n = crate::transforms::eval::eval(arena, n);
    let d = crate::transforms::expand::expand(arena, d);
    let d = crate::transforms::eval::eval(arena, d);
    let p = crate::poly::polybridge::expr_to_poly(arena, n, var)?;
    let q = crate::poly::polybridge::expr_to_poly(arena, d, var)?;
    (q.degree()? >= 1).then_some((p, q))
}

/// `∫ f(L)·R(x) dx` for `R ∈ ℚ(x)` whose denominator has a factor
/// irreducible over ℚ of degree ≥ 2, from the partial fractions `A/gᵏ` of
/// its proper part over ℚ (`deg A < deg g`):
///
/// - a linear `g` as in [`try_special_function_integral`];
/// - otherwise the poles of order `k ≥ 2` reduced first
///   ([`reduce_special_poles`]: terms `f(L)·S` with `S ∈ ℚ(x)`, and simple
///   poles left), then every simple fraction `M/g` split over the roots `ρ`
///   of `g`: `M/g = Σ_ρ r(ρ)/(x − ρ)` with the residue `r = M·g′⁻¹ mod g`
///   (the first step of Bronstein's full partial fraction decomposition,
///   SymPy's `apart_list_full_decomposition`), and `∫ f(L)/(x − ρ)` an `Ei`,
///   `Si`/`Ci` or `Shi`/`Chi` of `α·(x − ρ)` ([`linear_special_base`]),
///   summed over the roots in radicals or as a `RootSum`
///   ([`factor_roots`]).
///
/// The two terms of a pair of conjugate roots are conjugate on the real
/// line, and each is continuous there: `α·(x − ρ)` never meets the cut of
/// `Ei`, `Ci`, `Chi` (the negative axis) for a non-real `ρ`, and for a real
/// one only at `x = ρ`, a pole of the integrand.  `None` for coefficients
/// outside ℚ, a pole of order `k ≥ 2` with an irrational `α`, or beyond the
/// budgets.
fn special_over_rational_factors(
    arena: &mut Arena,
    special: &LinearSpecialArg,
    r: ExprId,
    var: ExprId,
) -> Option<SpecialOverQ> {
    let (numer, denom) = rational_over_q(arena, r, var)?;
    if denom.degree()? > MAX_SPECIAL_DENOM_DEGREE {
        return None;
    }
    let (quotient, remainder) = numer.div_rem(&denom);
    let polynomial = crate::poly::polybridge::poly_to_expr(arena, &quotient, var);
    let (content, factors) = denom.factor_over_z();
    if content.is_zero() {
        return None;
    }
    let scaled = remainder.scale(&(Q::one() / content));
    let mut pieces: SmallVec<[ExprId; 6]> = SmallVec::new();
    for term in crate::transforms::apart::partial_fractions_over_q(&scaled, &factors) {
        let (a, g, k) = (term.numer, term.factor, term.power);
        if a.is_zero() {
            continue;
        }
        if i64::from(k) > MAX_SPECIAL_ORDER {
            return None;
        }
        let piece = if g.degree() == Some(1) {
            linear_factor_special(arena, special, &a, &g, k, var)
        } else {
            special_over_irreducible(arena, special, &a, &g, k, var)?
        };
        pieces.push(piece);
    }
    let proper = match pieces.len() {
        0 => arena.zero,
        1 => pieces[0],
        _ => arena.add(&pieces),
    };
    Some(SpecialOverQ { proper, polynomial })
}

/// `∫ f(L)·a/gᵏ dx` for a constant `a` and a linear `g = g₁·x + g₀`:
/// `a/g₁ᵏ·∫ f(L)·ℓ⁻ᵏ dℓ` with `ℓ = x + g₀/g₁` ([`linear_special_power`]).
fn linear_factor_special(
    arena: &mut Arena,
    special: &LinearSpecialArg,
    a: &Poly,
    g: &Poly,
    k: u32,
    var: ExprId,
) -> ExprId {
    let g1 = g.coeff(1);
    let mut c = a.coeff(0);
    for _ in 0..k {
        c /= &g1;
    }
    let delta = arena.num_ratio(g.coeff(0) / &g1);
    let ell = arena.add(&[var, delta]);
    let kappa = special.alpha;
    let shift = arena.mul(&[kappa, delta]);
    let beta_shifted = arena.sub(special.beta, shift);
    let beta_shifted = crate::transforms::eval::eval(arena, beta_shifted);
    let w = arena.mul(&[kappa, ell]);
    let w = crate::transforms::expand::expand(arena, w);
    let j = linear_special_power(
        arena,
        special.fam,
        special.big_l,
        ell,
        kappa,
        w,
        beta_shifted,
        i64::from(k),
    );
    let c = arena.num_ratio(c);
    arena.mul(&[c, j])
}

/// The functions met in `∫ f(L)·R(x) dx` by parts: `[(f, 0, s)]` for
/// `f = exp` (`f′ = s·f`), else `[(f, 1, s), (g, 0, s′)]` (`sin`/`cos`,
/// `sinh`/`cosh`): each with the index of the function its derivative is
/// a multiple of, and the sign of that multiple (`f′(L) = s·α·g(L)`).
fn special_closure(fam: LinearSpecial) -> SmallVec<[(LinearSpecial, usize, i64); 2]> {
    let (g, s) = fam.derivative();
    if g == fam {
        SmallVec::from_slice(&[(fam, 0, s)])
    } else {
        let (_, s2) = g.derivative();
        SmallVec::from_slice(&[(fam, 1, s), (g, 0, s2)])
    }
}

/// `∫ Σⱼ Fⱼ(L)·Mⱼ/g dx` with simple poles only, and the part `Σⱼ Fⱼ(L)·Bⱼ/gᵏ⁻¹`
/// of the antiderivative taken off before ([`reduce_special_poles`]).
struct PoleReduction {
    /// The numerators `Bⱼ` over `gᵏ⁻¹`.
    boundary: Vec<Poly>,
    /// The numerators `Mⱼ` over `g` (`deg Mⱼ < deg g`).
    simple: Vec<Poly>,
}

/// `g′⁻¹ mod g` for a square-free `g`.
fn inverse_derivative_mod(g: &Poly) -> Option<Poly> {
    let eg = Poly::extended_gcd(&g.derivative(), g);
    if eg.gcd.degree() != Some(0) {
        return None;
    }
    let c = eg.gcd.coeff(0);
    Some(eg.x.scale(&(Q::one() / c)).rem(g))
}

/// The order of the poles of `∫ Σⱼ Fⱼ(L)·Nⱼ/gᵏ dx` lowered to 1 (`g`
/// square-free, `k ≥ 2`, `deg Nⱼ < k·deg g`, `F` the [`special_closure`]
/// with `Fᵢ′ = sᵢ·α·F_{p(i)}`, `α` rational), as Hermite's reduction does for
/// rational functions: with `Sⱼ = Bⱼ/g^{m−1}`,
/// `(Σⱼ Fⱼ·Sⱼ)′ = Σⱼ Fⱼ·(Sⱼ′ + Σ_{p(i)=j} sᵢ·α·Sᵢ)`, whose part of order `m`
/// is `−(m − 1)·Bⱼ·g′/gᵐ`; `Bⱼ = −Nⱼ·((m − 1)·g′)⁻¹ mod g` cancels the
/// order `m` of the integrand, and the rest is of order `m − 1`.
fn reduce_special_poles(
    g: &Poly,
    k: u32,
    mut numer: Vec<Poly>,
    closure: &[(LinearSpecial, usize, i64)],
    alpha: &Q,
) -> Option<PoleReduction> {
    let inv = inverse_derivative_mod(g)?;
    let gp = g.derivative();
    let n = closure.len();
    let mut boundary = vec![Poly::zero(); n];
    for m in (2..=k).rev() {
        let m1 = Q::from_integer((m - 1).into());
        let b: Vec<Poly> = numer
            .iter()
            .map(|nj| (&nj.rem(g) * &inv).scale(&(-Q::one() / &m1)).rem(g))
            .collect();
        let g_pow = g.pow((k - m) as usize);
        let mut next = Vec::with_capacity(n);
        for j in 0..n {
            boundary[j] = &boundary[j] + &(&b[j] * &g_pow);
            let top = &numer[j] + &(&b[j] * &gp).scale(&m1);
            let (q, rem) = top.div_rem(g);
            if !rem.is_zero() {
                return None;
            }
            let mut cross = b[j].derivative();
            for (i, &(_, p, s)) in closure.iter().enumerate() {
                if p == j {
                    let si = alpha * Q::from_integer(s.into());
                    cross = &cross + &b[i].scale(&si);
                }
            }
            next.push(&q - &cross);
        }
        numer = next;
    }
    Some(PoleReduction {
        boundary,
        simple: numer,
    })
}

/// `∫ f(L)·a/gᵏ dx` for `g` irreducible over ℚ of degree ≥ 2 (see
/// [`special_over_rational_factors`]).
fn special_over_irreducible(
    arena: &mut Arena,
    special: &LinearSpecialArg,
    a: &Poly,
    g: &Poly,
    k: u32,
    var: ExprId,
) -> Option<ExprId> {
    let closure = special_closure(special.fam);
    let mut numer = vec![Poly::zero(); closure.len()];
    numer[0] = a.clone();
    let reduced = if k == 1 {
        PoleReduction {
            boundary: vec![Poly::zero(); closure.len()],
            simple: numer,
        }
    } else {
        let alpha = arena.as_num(special.alpha)?.clone();
        reduce_special_poles(g, k, numer, &closure, &alpha)?
    };
    let mut terms: SmallVec<[ExprId; 4]> = SmallVec::new();
    if k >= 2 {
        let g_expr = crate::poly::polybridge::poly_to_expr(arena, g, var);
        let den = if k == 2 {
            g_expr
        } else {
            let e = arena.int(i64::from(k) - 1);
            arena.pow(g_expr, e)
        };
        for (b, &(f, _, _)) in reduced.boundary.iter().zip(closure.iter()) {
            if b.is_zero() {
                continue;
            }
            let b_expr = crate::poly::polybridge::poly_to_expr(arena, b, var);
            let s = arena.div(b_expr, den);
            let f_l = f.apply(arena, special.big_l);
            terms.push(arena.mul(&[s, f_l]));
        }
    }
    let inv = inverse_derivative_mod(g)?;
    let residues: Vec<Poly> = reduced.simple.iter().map(|m| (m * &inv).rem(g)).collect();
    if residues.iter().any(|r| !r.is_zero()) {
        let roots = factor_roots(arena, g, var)?;
        let sum = sum_over_roots(arena, g, &roots, |arena, rho| {
            let x_minus = arena.sub(var, rho);
            let w = arena.mul(&[special.alpha, x_minus]);
            let w = crate::transforms::expand::expand(arena, w);
            let shift = arena.mul(&[special.alpha, rho]);
            let beta = arena.add(&[special.beta, shift]);
            let beta = crate::transforms::eval::eval(arena, beta);
            let mut parts: SmallVec<[ExprId; 2]> = SmallVec::new();
            for (r, &(f, _, _)) in residues.iter().zip(closure.iter()) {
                if r.is_zero() {
                    continue;
                }
                let c = residue_at(arena, r, rho);
                let base = linear_special_base(arena, f, w, beta);
                parts.push(arena.mul(&[c, base]));
            }
            arena.add(&parts)
        });
        terms.push(sum);
    }
    Some(match terms.len() {
        0 => arena.zero,
        1 => terms[0],
        _ => arena.add(&terms),
    })
}

/// `r(ρ)` for a polynomial `r ∈ ℚ[t]`, multiplied out when that is no
/// larger (`u·ρ + v` with `ρ = p ± √q` collects to `u·p + v ± u·√q`, and
/// `ρ²` with `ρ = (1 + i)/√2` is `i`; Cardano's roots multiplied out are
/// larger).
fn residue_at(arena: &mut Arena, r: &Poly, rho: ExprId) -> ExprId {
    let e = crate::poly::polybridge::poly_to_expr(arena, r, rho);
    let e = crate::transforms::eval::eval(arena, e);
    if matches!(arena.node(rho), ExprNode::Symbol(_)) {
        return e;
    }
    let expanded = crate::transforms::expand::expand(arena, e);
    let expanded = crate::transforms::eval::eval(arena, expanded);
    let size = |arena: &Arena, id: ExprId| crate::base::walk::post_order_ids(arena, id).len();
    if size(arena, expanded) <= size(arena, e) {
        expanded
    } else {
        e
    }
}

/// Largest degree of an irreducible factor whose roots [`factor_roots`]
/// asks `solve` for (the quadratic formula, Cardano, Ferrari).
const MAX_EXPLICIT_ROOT_DEGREE: usize = 4;

/// Largest root (nodes written out) that [`factor_roots`] uses explicitly;
/// Ferrari's roots of a general quartic are far larger and stay implicit
/// (with them, `∫ ln(−x)·(−x² + 2x − 2)/(x⁴ − x² + 2x + 1) dx` was a 16 KB
/// answer).
const MAX_EXPLICIT_ROOT_SIZE: usize = 48;

/// The number of nodes of `e` written out as a tree, counted up to `cap`.
fn written_size(arena: &Arena, e: ExprId, cap: usize) -> usize {
    let mut size: FxHashMap<ExprId, usize> = FxHashMap::default();
    for id in crate::base::walk::post_order_ids(arena, e) {
        let mut s: usize = 1;
        arena
            .node(id)
            .for_each_child(|c| s = s.saturating_add(size.get(&c).copied().unwrap_or(1)));
        size.insert(id, s.min(cap.saturating_add(1)));
    }
    size.get(&e).copied().unwrap_or(1)
}

/// Largest degree of an irreducible factor summed over in a `RootSum`.
const MAX_ROOT_SUM_DEGREE: usize = 16;

/// How the roots of an irreducible factor of a denominator appear in an
/// antiderivative.
enum FactorRoots {
    /// Every root in radicals, from `solve`.
    Explicit(Vec<ExprId>),
    /// A `RootSum` over the factor.
    Implicit,
}

/// The roots of `g` (irreducible over ℚ, degree ≥ 2): explicit when `solve`
/// writes every one of them in radicals of at most
/// [`MAX_EXPLICIT_ROOT_SIZE`] nodes, and the real ones without `i` (Cardano's
/// formula writes the three real roots of `x³ − 3x² + 1` with complex cube
/// roots, whose rounded imaginary parts would decide the side of the cut of
/// `Ei` and `Ci` at every point); otherwise implicit, up to
/// [`MAX_ROOT_SUM_DEGREE`] (a `RootSum` is evaluated at its certified roots,
/// the real ones exactly real).
fn factor_roots(arena: &mut Arena, g: &Poly, var: ExprId) -> Option<FactorRoots> {
    let deg = g.degree()?;
    if deg > MAX_ROOT_SUM_DEGREE {
        return None;
    }
    if deg <= MAX_EXPLICIT_ROOT_DEGREE {
        let g_expr = crate::poly::polybridge::poly_to_expr(arena, g, var);
        let roots: Vec<ExprId> = crate::transforms::solve::known_real_roots(arena, g_expr, var);
        let small = roots.iter().all(|&r| {
            let ids = crate::base::walk::post_order_ids(arena, r);
            written_size(arena, r, MAX_EXPLICIT_ROOT_SIZE) <= MAX_EXPLICIT_ROOT_SIZE
                && !ids.iter().any(|&id| {
                    id == var
                        || matches!(
                            arena.node(id),
                            ExprNode::RootOf(..) | ExprNode::Symbol(_) | ExprNode::Integral(..)
                        )
                })
        });
        let written_real = roots
            .iter()
            .filter(|&&r| {
                !crate::base::walk::post_order_ids(arena, r)
                    .iter()
                    .any(|&id| matches!(arena.node(id), ExprNode::ImaginaryUnit))
            })
            .count();
        if roots.len() == deg
            && small
            && written_real == crate::poly::sturm::SturmChain::new(g).count_real_roots()
        {
            return Some(FactorRoots::Explicit(roots));
        }
    }
    Some(FactorRoots::Implicit)
}

/// `Σ_{g(ρ)=0} body(ρ)`: the explicit sum, or `RootSum(g(t), t ↦ body(t))`.
fn sum_over_roots(
    arena: &mut Arena,
    g: &Poly,
    roots: &FactorRoots,
    mut body: impl FnMut(&mut Arena, ExprId) -> ExprId,
) -> ExprId {
    match roots {
        FactorRoots::Explicit(rs) => {
            let terms: SmallVec<[ExprId; 4]> = rs.iter().map(|&rho| body(arena, rho)).collect();
            arena.add(&terms)
        }
        FactorRoots::Implicit => {
            let t = arena.symbol("__rs_t");
            let b = body(arena, t);
            let g_t = crate::poly::polybridge::poly_to_expr(arena, g, t);
            arena.intern(ExprNode::RootSum(g_t, b, t))
        }
    }
}

/// `∫ ln(ℓ)⁻ⁿ dx` for `ℓ = α·x + β` and an integer `1 ≤ n ≤`
/// [`MAX_SPECIAL_ORDER`]: `t = ln ℓ` makes it `∫ eᵗ·t⁻ⁿ dt/α`, by parts
/// `Jₙ = −ℓ·ln(ℓ)^{1−n}/(n − 1) + J_{n−1}/(n − 1)` down to `J₁ = li(ℓ)`
/// (SymPy's `LiRule`).  Up to 0.43 only `∫ dx/ln x` was recognised: `∫
/// dx/ln(2x + 1)` and `∫ dx/ln(x)²` stayed unevaluated.  `li` has its
/// singularity at `ℓ = 1`, a pole of the integrand, and the integrand is
/// real for `ℓ > 0` only.
fn try_inverse_log_power(
    arena: &mut Arena,
    base: ExprId,
    exp: ExprId,
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    let ExprNode::Ln(ell) = *arena.node(base) else {
        return None;
    };
    let n = arena
        .as_num(exp)
        .filter(|q| q.is_integer() && q.is_negative())?;
    let n = i64::try_from(-n.to_integer()).ok()?;
    if n > MAX_SPECIAL_ORDER {
        return None;
    }
    let (alpha, _) = symbolic_linear_coeff_of(arena, ell, var, var_sym)?;
    let cases = vanishing_parameter_values(arena, alpha, var_sym)?;
    let mut j = arena.li(ell);
    for m in 2..=n {
        let inv = arena.rational(1, m - 1);
        let power = arena.int(1 - m);
        let ln_pow = arena.pow(base, power);
        let neg_inv = arena.neg(inv);
        let boundary = arena.mul(&[neg_inv, ell, ln_pow]);
        let carried = arena.mul(&[inv, j]);
        j = arena.add(&[boundary, carried]);
    }
    let generic = arena.div(j, alpha);
    let integrand = arena.pow(base, exp);
    with_parameter_cases(arena, generic, integrand, &cases, var, var_sym, depth)
}

/// The product of `factors` with every power `sⁿ` (`n` a negative integer)
/// of a sum `s = g·h` whose terms share a factor `g = c·xᵏ`, with a
/// transcendental function of `x` in `h`, or `g = c·e^u`, written `gⁿ·hⁿ`
/// ([`Arena::factor_terms_pair_expr`]; exact for an integer `n`); `None`
/// when there is no such power.  (Algebraic cofactors are the rational and
/// radical integrators' business: splitting the `t·(…)` of the integrands
/// that the trigonometric substitutions build sent the Rubi suite's 4.1.2.1
/// from 9 s to 220 s, 13 timeouts, for answers with `tan(…)¹²²`.)  The
/// canonical form keeps `x·ln x − x` a sum, so the substitution `u = ln x`
/// did not see the `1/x` of `∫ dx/(x·ln x − x)`, which stayed unevaluated
/// up to 0.43 while `∫ dx/(x·(ln x − 1))` worked (SymPy:
/// `log(log(x) − 1)`).
fn split_common_factors(
    arena: &mut Arena,
    factors: &[ExprId],
    var_sym: SymbolId,
) -> Option<ExprId> {
    let mut changed = false;
    let mut out: SmallVec<[ExprId; 6]> = SmallVec::new();
    for &f in factors {
        if let ExprNode::Pow(base, e) = *arena.node(f)
            && matches!(arena.node(base), ExprNode::Add(_))
            && arena
                .as_num(e)
                .is_some_and(|q| q.is_integer() && q.is_negative())
        {
            let (g, h) = arena.factor_terms_pair_expr(base);
            if contains_var(arena, g, var_sym)
                && contains_var(arena, h, var_sym)
                && ((is_monomial_in_var(arena, g, var_sym)
                    && has_transcendental_of_var(arena, h, var_sym))
                    || is_exponential_of_var(arena, g, var_sym))
            {
                let g_e = arena.pow(g, e);
                let h_e = arena.pow(h, e);
                out.push(g_e);
                out.push(h_e);
                changed = true;
                continue;
            }
        }
        out.push(f);
    }
    changed.then(|| arena.mul(&out))
}

/// `∫ e^{a·x² + b·x + c} dx` for coefficients the rational branch of the
/// `Exp` arm does not take (`√2`, a parameter): with `s² = −a`,
/// `√π/(2s)·e^{c − b²/(4a)}·erf(s·x − b/(2s))` when `a` is written with a
/// negative sign, else with `t² = a`, `√π/(2t)·e^{c − b²/(4a)}·erfi(t·x +
/// b/(2t))` (SymPy's `ErfRule`).  Both hold for every complex `a ≠ 0`
/// (only `s² = −a`, `t² = a` is used), and `erf`, `erfi` are entire; the
/// case `a = 0` comes from the stage's wrap (the denominator `√a`).  Up to
/// 0.43 `∫ e^{−a·x²} dx` stayed unevaluated.
fn gaussian_integral(
    arena: &mut Arena,
    q: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    let (a, b, c) = quadratic_argument_coeffs(arena, q, var, var_sym)?;
    if arena.is_zero_structural(a) {
        return None;
    }
    let two = arena.int(2);
    let four = arena.int(4);
    let b2 = arena.pow(b, two);
    let four_a = arena.mul(&[four, a]);
    let b2_4a = arena.div(b2, four_a);
    let shift = arena.sub(c, b2_4a);
    let shift = crate::transforms::eval::eval(arena, shift);
    let pi = arena.pi();
    let sqrt_pi = arena.sqrt(pi);
    let negative = has_negative_sign(arena, a);
    let r = if negative {
        let neg_a = arena.neg(a);
        let neg_a = crate::transforms::eval::eval(arena, neg_a);
        arena.sqrt(neg_a)
    } else {
        arena.sqrt(a)
    };
    let rx = arena.mul(&[r, var]);
    let two_r = arena.mul(&[two, r]);
    let b_2r = arena.div(b, two_r);
    let front = arena.div(sqrt_pi, two_r);
    let special = if negative {
        let arg = arena.sub(rx, b_2r);
        arena.erf(arg)
    } else {
        let arg = arena.add(&[rx, b_2r]);
        special_apply(arena, crate::base::libfn::LibFn::Erfi, arg)
    };
    Some(if arena.is_zero_structural(shift) {
        arena.mul(&[front, special])
    } else {
        let e = arena.exp(shift);
        arena.mul(&[front, e, special])
    })
}

/// `∫ sin(q) dx` (`sine`) or `∫ cos(q) dx` for `q = a·x² + b·x + c`:
/// `√π/√(2a)·(cos θ·S(u) − sin θ·C(u))` and `√π/√(2a)·(cos θ·C(u) + sin θ·S(u))`
/// with `u = (2a·x + b)/(√(2a)·√π)`, `θ = b²/(4a) − c` and the Fresnel
/// integrals `S`, `C` (SymPy's `FresnelSRule`, `FresnelCRule`): `πu²/2 = q +
/// θ`, and only `√(2a)² = 2a` is used, so it holds for every complex `a ≠ 0`;
/// `S`, `C` are entire.  A leading coefficient written with a negative sign
/// is made positive first (`sin q = −sin(−q)`, `cos q = cos(−q)`), which keeps
/// `√(2a)` real for real coefficients.  Up to 0.43 `∫ sin(x²) dx` stayed
/// unevaluated.
fn fresnel_integral(
    arena: &mut Arena,
    sine: bool,
    q: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    use crate::base::libfn::LibFn;
    let (a, b, c) = quadratic_argument_coeffs(arena, q, var, var_sym)?;
    if arena.is_zero_structural(a) {
        return None;
    }
    let (a, b, c, sign) = if has_negative_sign(arena, a) {
        let flip = |arena: &mut Arena, e: ExprId| {
            let n = arena.neg(e);
            crate::transforms::eval::eval(arena, n)
        };
        (
            flip(arena, a),
            flip(arena, b),
            flip(arena, c),
            if sine { -1 } else { 1 },
        )
    } else {
        (a, b, c, 1)
    };
    let two = arena.int(2);
    let four = arena.int(4);
    let b2 = arena.pow(b, two);
    let four_a = arena.mul(&[four, a]);
    let b2_4a = arena.div(b2, four_a);
    let theta = arena.sub(b2_4a, c);
    let theta = crate::transforms::eval::eval(arena, theta);
    let two_a = arena.mul(&[two, a]);
    let root = arena.sqrt(two_a);
    let pi = arena.pi();
    let sqrt_pi = arena.sqrt(pi);
    let two_a_x = arena.mul(&[two_a, var]);
    let lin = arena.add(&[two_a_x, b]);
    let scale = arena.mul(&[root, sqrt_pi]);
    let u = arena.div(lin, scale);
    let s_u = special_apply(arena, LibFn::FresnelS, u);
    let c_u = special_apply(arena, LibFn::FresnelC, u);
    let (main, other) = if sine { (s_u, c_u) } else { (c_u, s_u) };
    let body = if arena.is_zero_structural(theta) {
        main
    } else {
        let ct = parity_folded(arena, LinearSpecial::Cos, theta);
        let st = parity_folded(arena, LinearSpecial::Sin, theta);
        let m = arena.mul(&[ct, main]);
        let o = arena.mul(&[st, other]);
        if sine {
            arena.sub(m, o)
        } else {
            arena.add(&[m, o])
        }
    };
    let front = arena.div(sqrt_pi, root);
    let sign = arena.int(sign);
    let result = arena.mul(&[sign, front, body]);
    Some(crate::transforms::eval::eval(arena, result))
}

/// Largest degree of the polynomial of [`try_poly_times_quadratic`].
const MAX_GAUSSIAN_DEGREE: usize = 12;

/// The coefficients `[p₀, …, pₙ]` (free of the variable) of the polynomial
/// `p` in the variable, multiplied out; `None` when `p` is not one or its
/// degree exceeds `max_degree`.
fn poly_coeffs_in(
    arena: &mut Arena,
    p: ExprId,
    var: ExprId,
    var_sym: SymbolId,
    max_degree: usize,
) -> Option<Vec<ExprId>> {
    let p = crate::transforms::expand::expand(arena, p);
    let terms: SmallVec<[ExprId; 6]> = match arena.node(p) {
        ExprNode::Add(children) => children.clone(),
        _ => SmallVec::from_slice(&[p]),
    };
    let mut by_degree: Vec<Vec<ExprId>> = Vec::new();
    for t in terms {
        let (c, n) = if !contains_var(arena, t, var_sym) {
            (t, 0usize)
        } else {
            let (c, dep) = match split_constant_factors(arena, t, var_sym) {
                Some((c, dep)) => (arena.mul(&c), dep),
                None => (arena.one, t),
            };
            let n = if dep == var {
                1
            } else {
                let ExprNode::Pow(b, e) = *arena.node(dep) else {
                    return None;
                };
                if b != var {
                    return None;
                }
                let e = arena
                    .as_num(e)
                    .filter(|q| q.is_integer() && !q.is_negative())?;
                usize::try_from(e.to_integer()).ok()?
            };
            (c, n)
        };
        if n > max_degree {
            return None;
        }
        if by_degree.len() <= n {
            by_degree.resize(n + 1, Vec::new());
        }
        by_degree[n].push(c);
    }
    Some(
        by_degree
            .into_iter()
            .map(|cs| match cs.len() {
                0 => arena.zero,
                1 => cs[0],
                _ => arena.add(&cs),
            })
            .collect(),
    )
}

/// `∫ P(x)·f(q) dx` for a polynomial `P` and `f(q)` = `e^q`, `sin q`, `cos q`
/// with `q = a·x² + b·x + c` (`a ≠ 0`): `xⁿ·f(q) = xⁿ⁻¹·(q′ − b)·f(q)/(2a)`,
/// and by parts on `xⁿ⁻¹·q′·f(q)`
///
/// - `Iₙ = xⁿ⁻¹·e^q/(2a) − (n − 1)/(2a)·Iₙ₋₂ − b/(2a)·Iₙ₋₁`,
/// - `Sₙ = (−xⁿ⁻¹·cos q + (n − 1)·Cₙ₋₂ − b·Sₙ₋₁)/(2a)`,
/// - `Cₙ = (xⁿ⁻¹·sin q − (n − 1)·Sₙ₋₂ − b·Cₙ₋₁)/(2a)`,
///
/// down to `∫ f(q)` (`erf`/`erfi`, Fresnel), which is integrated only when
/// the answer needs it.  By parts with `u = P` took `dv = e^q`, an `erf`,
/// and `∫ x²·e^{−x²} dx` stayed unevaluated up to 0.43 (SymPy: `−x·e^{−x²}/2 +
/// √π·erf(x)/4`).
fn try_poly_times_quadratic(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    let mut special: Option<(LinearSpecial, ExprId, ExprId, ExprId)> = None;
    let mut polys: SmallVec<[ExprId; 4]> = SmallVec::new();
    for &d in dependent {
        if special.is_none()
            && let Some((fam, q)) = LinearSpecial::of(arena, d)
            && matches!(
                fam,
                LinearSpecial::Exp | LinearSpecial::Sin | LinearSpecial::Cos
            )
            && let Some((a, b, _)) = quadratic_argument_coeffs(arena, q, var, var_sym)
            && !arena.is_zero_structural(a)
        {
            special = Some((fam, q, a, b));
        } else if is_polynomial_in(arena, d, var, var_sym) {
            polys.push(d);
        } else {
            return None;
        }
    }
    let (fam, q, a, b) = special?;
    if polys.is_empty() {
        return None;
    }
    let p = arena.mul(&polys);
    let coeffs = poly_coeffs_in(arena, p, var, var_sym, MAX_GAUSSIAN_DEGREE)?;
    let fams: SmallVec<[LinearSpecial; 2]> = if fam == LinearSpecial::Exp {
        SmallVec::from_slice(&[LinearSpecial::Exp])
    } else {
        SmallVec::from_slice(&[LinearSpecial::Sin, LinearSpecial::Cos])
    };
    // The integrals `∫ f(q)` as placeholders until the answer is known.
    let bases: SmallVec<[ExprId; 2]> = fams
        .iter()
        .enumerate()
        .map(|(k, _)| arena.symbol(&format!("__gauss_base{k}")))
        .collect();
    let two = arena.int(2);
    let two_a = arena.mul(&[two, a]);
    let minus_one = arena.int(-1);
    let inv_2a = arena.pow(two_a, minus_one);
    let at = |f: LinearSpecial| fams.iter().position(|&g| g == f).unwrap_or(0);
    // table[n][k] = ∫ xⁿ·fams[k](q)
    let mut table: Vec<SmallVec<[ExprId; 2]>> = vec![bases.clone()];
    for n in 1..coeffs.len() {
        let n_minus_1 = i64::try_from(n - 1).ok()?;
        let power = arena.int(n_minus_1);
        let x_pow = arena.pow(var, power);
        let mut row: SmallVec<[ExprId; 2]> = SmallVec::new();
        for &f in &fams {
            // xⁿ⁻¹·F(q) with F′ = f (F = e^q, −cos q, sin q), and the
            // integral by parts leaves −(n − 1)·∫ xⁿ⁻²·F(q).
            let (antider, minus_partner) = match f {
                LinearSpecial::Exp => (arena.exp(q), (LinearSpecial::Exp, 1)),
                LinearSpecial::Sin => {
                    let c = arena.cos(q);
                    (arena.neg(c), (LinearSpecial::Cos, -1))
                }
                _ => (arena.sin(q), (LinearSpecial::Sin, 1)),
            };
            let mut terms: SmallVec<[ExprId; 3]> = SmallVec::new();
            terms.push(arena.mul(&[x_pow, antider]));
            if n >= 2 {
                let (g, s) = minus_partner;
                let coeff = arena.int(-s * n_minus_1);
                terms.push(arena.mul(&[coeff, table[n - 2][at(g)]]));
            }
            if !arena.is_zero_structural(b) {
                let neg_b = arena.neg(b);
                terms.push(arena.mul(&[neg_b, table[n - 1][at(f)]]));
            }
            let sum = arena.add(&terms);
            row.push(arena.mul(&[inv_2a, sum]));
        }
        table.push(row);
    }
    let k = at(fam);
    let mut parts: SmallVec<[ExprId; 8]> = SmallVec::new();
    for (n, &c) in coeffs.iter().enumerate() {
        if !arena.is_zero_structural(c) {
            parts.push(arena.mul(&[c, table[n][k]]));
        }
    }
    let sum = arena.add(&parts);
    let mut result = crate::transforms::eval::eval(arena, sum);
    for (&f, &base) in fams.iter().zip(&bases) {
        if !crate::base::walk::contains(arena, result, base) {
            continue;
        }
        let f_q = f.apply(arena, q);
        let integral = integrate_node(arena, f_q, var, var_sym, depth - 1);
        if crate::base::walk::has_unevaluated(arena, integral) {
            return None;
        }
        result = crate::transforms::subs::subs(arena, result, base, integral);
    }
    tracing::debug!("integrate: polynomial times a function of a quadratic, by parts");
    Some(crate::transforms::eval::eval(arena, result))
}

/// The dilogarithm integrals `∫ ln(β + α·xᵏ)/x dx = ln β·ln|x| −
/// Li₂(−α·xᵏ/β)/k` for a positive `β` (`ln(β·(1 + z)) = ln β + ln(1 + z)`
/// needs it) and a number `k ≠ 0`, and
/// `∫ ln(x)/(β + α·x) dx = (ln x·ln(1 + α·x/β) + Li₂(−α·x/β))/α` (an identity of
/// analytic functions; SymPy's `PolylogRule` and `manualintegrate`'s
/// answers).  `Li₂(z)` has its cut on `z ≥ 1`, where `ln(1 − z)` has its own
/// (`evalf` takes both from the same side), so the answers are continuous
/// where the integrands are.  Up to 0.43 both stayed unevaluated (by parts
/// turned one into the other).
fn try_dilog_integral(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    use crate::base::libfn::LibFn;
    let [f, g] = dependent[..] else {
        return None;
    };
    let li2 = |arena: &mut Arena, z: ExprId| -> ExprId {
        let two = arena.int(2);
        arena.lib_apply(LibFn::PolyLog, &[two, z])
    };
    let is_inverse = |arena: &Arena, e: ExprId| -> Option<ExprId> {
        match *arena.node(e) {
            ExprNode::Pow(base, k)
                if arena.as_num(k).is_some_and(|q| *q == -Q::one())
                    && contains_var(arena, base, var_sym) =>
            {
                Some(base)
            }
            _ => None,
        }
    };
    for (ln_factor, other) in [(f, g), (g, f)] {
        let ExprNode::Ln(ell) = *arena.node(ln_factor) else {
            continue;
        };
        let Some(den) = is_inverse(arena, other) else {
            continue;
        };
        if den == var && ell != var {
            // ln(β + α·xᵏ)/x
            let ExprNode::Add(terms) = arena.node(ell).clone() else {
                return None;
            };
            let (dep, free): (SmallVec<[ExprId; 4]>, SmallVec<[ExprId; 4]>) = terms
                .iter()
                .copied()
                .partition(|&t| contains_var(arena, t, var_sym));
            let [monomial] = dep[..] else {
                return None;
            };
            if free.is_empty() {
                return None;
            }
            let (alpha, k) = monomial_parts(arena, monomial, var, var_sym)?;
            arena.as_num(k)?;
            let beta = arena.add(&free);
            let positive = arena.as_num(beta).is_some_and(Signed::is_positive)
                || Truth::from(crate::base::assumptions::AssumptionCache::new().query(
                    arena,
                    beta,
                    crate::base::assumptions::Props::POSITIVE,
                ))
                .is_true();
            if !positive {
                return None;
            }
            let ratio = arena.div(alpha, beta);
            let x_k = arena.pow(var, k);
            let kx = arena.mul(&[ratio, x_k]);
            let z = arena.neg(kx);
            let z = crate::transforms::eval::eval(arena, z);
            let dilog = li2(arena, z);
            let minus_one = arena.int(-1);
            let inv_k = arena.pow(k, minus_one);
            let dilog = arena.mul(&[minus_one, inv_k, dilog]);
            if beta == arena.one {
                return Some(dilog);
            }
            let ln_beta = arena.ln(beta);
            let abs_x = arena.abs(var);
            let ln_x = arena.ln(abs_x);
            let log_part = arena.mul(&[ln_beta, ln_x]);
            return Some(arena.add(&[log_part, dilog]));
        }
        if ell == var && den != var {
            // ln(x)/(β + α·x)
            let (alpha, beta) = symbolic_linear_coeff_of(arena, den, var, var_sym)?;
            if arena.is_zero_structural(beta) {
                return None;
            }
            let k = arena.div(alpha, beta);
            let kx = arena.mul(&[k, var]);
            let kx = crate::transforms::eval::eval(arena, kx);
            let one = arena.one;
            let one_kx = arena.add(&[one, kx]);
            let ln_one_kx = arena.ln(one_kx);
            let product = arena.mul(&[ln_factor, ln_one_kx]);
            let z = arena.neg(kx);
            let dilog = li2(arena, z);
            let sum = arena.add(&[product, dilog]);
            return Some(arena.div(sum, alpha));
        }
    }
    None
}

/// `Li₂(z)`, the `polylog(2, z)` node.
fn dilog(arena: &mut Arena, z: ExprId) -> ExprId {
    let two = arena.int(2);
    arena.lib_apply(crate::base::libfn::LibFn::PolyLog, &[two, z])
}

/// `∫ ln(L)·R(x) dx` for a linear `L = a·x + b` (`a`, `b` rational) and
/// `R ∈ ℚ(x)` with a non-constant denominator.  Hermite's reduction
/// `∫ R = g + ∫ h` (`g` rational, `h = M/D` proper with `D` square-free)
/// gives by parts `ln(L)·g − ∫ a·g/L dx + ∫ ln(L)·h dx`: the middle integral
/// is rational, and `h = Σ_ρ r(ρ)/(x − ρ)` over the roots of `D` (residues
/// `r = M·D′⁻¹ mod D` on each irreducible factor, summed as in
/// [`special_over_rational_factors`]) with
/// `∫ ln(L)/(x − ρ) dx = ln(L)·ln(1 − L/c) + Li₂(L/c)` for `c = a·ρ + b ≠ 0`
/// (`u = L` turns it into `∫ ln(u)/(u − c) du`) and `ln(L)²/2` for `c = 0`.
/// By parts alone (`u = ln L`) leaves `∫ ln(1 − L/c)/L`, and up to 0.44
/// `∫ ln(x)/(x² − 2) dx`, `∫ ln(2x + 1)/(x − 3) dx` stayed unevaluated.
/// With `nonlinear_only` (the call before by parts) only when the
/// denominator of `h` has a factor of degree ≥ 2: over linear factors by
/// parts may meet [`try_dilog_integral`] and keeps its turn.
///
/// The integrand is real for `L > 0`.  There the terms of a non-real `c`
/// never meet the cuts of `ln(1 − z)` and `Li₂(z)` (`z = L/c` is real only for
/// a real `c`), and a pair of conjugate roots gives conjugate terms; for a
/// real `c > 0` both cuts start at `L = c`, a pole of the integrand, beyond
/// which their imaginary parts `π·ln L` and `−π·ln(L/c)` add up to a constant.
fn try_log_times_rational(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
    nonlinear_only: bool,
) -> Option<ExprId> {
    if depth < 2 {
        return None;
    }
    let idx = dependent
        .iter()
        .position(|&d| matches!(arena.node(d), ExprNode::Ln(_)))?;
    let ln_l = dependent[idx];
    let ExprNode::Ln(ell) = *arena.node(ln_l) else {
        return None;
    };
    let lp = crate::poly::polybridge::expr_to_poly(arena, ell, var)?;
    if lp.degree() != Some(1) {
        return None;
    }
    let (a, b) = (lp.coeff(1), lp.coeff(0));
    let rest = remaining_product(arena, dependent, idx);
    if !is_rational_in(arena, rest, var_sym) {
        return None;
    }
    let (numer, denom) = rational_over_q(arena, rest, var)?;
    if denom.degree()? > MAX_SPECIAL_DENOM_DEGREE {
        return None;
    }
    let hr = crate::calculus::risch::hermite::hermite_reduce(&numer, &denom)?;
    if hr.h_numer.is_zero() {
        // Elementary: by parts' business.
        return None;
    }
    let (content, factors) = hr.h_denom.factor_over_z();
    if content.is_zero()
        || (nonlinear_only
            && factors
                .iter()
                .all(|(g, _)| g.degree().is_none_or(|d| d < 2)))
    {
        return None;
    }
    let mut terms: SmallVec<[ExprId; 6]> = SmallVec::new();
    if !hr.g_numer.is_zero() {
        let gn = crate::poly::polybridge::poly_to_expr(arena, &hr.g_numer, var);
        let gd = crate::poly::polybridge::poly_to_expr(arena, &hr.g_denom, var);
        let g = arena.div(gn, gd);
        terms.push(arena.mul(&[ln_l, g]));
        let a_e = arena.num_ratio(a.clone());
        let minus_one = arena.int(-1);
        let inv_l = arena.pow(ell, minus_one);
        let dv = arena.mul(&[a_e, g, inv_l]);
        let dv = crate::transforms::eval::eval(arena, dv);
        let v = integrate_node(arena, dv, var, var_sym, depth - 1);
        if crate::base::walk::has_unevaluated(arena, v) {
            return None;
        }
        terms.push(arena.neg(v));
    }
    let scaled = hr.h_numer.scale(&(Q::one() / content));
    for term in crate::transforms::apart::partial_fractions_over_q(&scaled, &factors) {
        let (m, g, k) = (term.numer, term.factor, term.power);
        if m.is_zero() {
            continue;
        }
        if k != 1 {
            return None;
        }
        let inv = inverse_derivative_mod(&g)?;
        let r = (&m * &inv).rem(&g);
        let roots = if g.degree() == Some(1) {
            let rho = arena.num_ratio(-(g.coeff(0) / g.coeff(1)));
            FactorRoots::Explicit(vec![rho])
        } else {
            factor_roots(arena, &g, var)?
        };
        let sum = sum_over_roots(arena, &g, &roots, |arena, rho| {
            let res = residue_at(arena, &r, rho);
            let a_e = arena.num_ratio(a.clone());
            let b_e = arena.num_ratio(b.clone());
            let a_rho = arena.mul(&[a_e, rho]);
            let c = arena.add(&[a_rho, b_e]);
            let c = crate::transforms::eval::eval(arena, c);
            let term = if arena.is_zero_structural(c) {
                let two = arena.int(2);
                let sq = arena.pow(ln_l, two);
                let half = arena.rational(1, 2);
                arena.mul(&[half, sq])
            } else {
                let z = arena.div(ell, c);
                let z = crate::transforms::eval::eval(arena, z);
                let one = arena.one;
                let one_minus = arena.sub(one, z);
                let one_minus = crate::transforms::eval::eval(arena, one_minus);
                let ln2 = arena.ln(one_minus);
                let product = arena.mul(&[ln_l, ln2]);
                let li = dilog(arena, z);
                arena.add(&[product, li])
            };
            arena.mul(&[res, term])
        });
        terms.push(sum);
    }
    tracing::debug!("integrate: a logarithm times a rational function, by its residues");
    let big_f = arena.add(&terms);
    let product = arena.mul(dependent);
    (!candidate_rejected(arena, product, big_f, var, var_sym)).then_some(big_f)
}

/// Does `h` contain a logarithm, exponential, trigonometric or hyperbolic
/// function of the variable?
fn has_transcendental_of_var(arena: &Arena, h: ExprId, var_sym: SymbolId) -> bool {
    crate::base::walk::post_order_ids(arena, h)
        .into_iter()
        .any(|id| match arena.node(id) {
            ExprNode::Ln(a)
            | ExprNode::Exp(a)
            | ExprNode::Sin(a)
            | ExprNode::Cos(a)
            | ExprNode::Tan(a)
            | ExprNode::Sinh(a)
            | ExprNode::Cosh(a)
            | ExprNode::Tanh(a) => contains_var(arena, *a, var_sym),
            _ => false,
        })
}

/// Is `g` an exponential `e^u` (times constants) with `u` depending on the
/// variable?  (`e^w/(e^{2w}·w + e^{2w}/2)`, what the substitution `x = e^w`
/// makes of `∫ dx/(x²·ln x + x²/2)`, is `e^{−w}/(w + 1/2)`.)
fn is_exponential_of_var(arena: &Arena, g: ExprId, var_sym: SymbolId) -> bool {
    let is_exp = |f: ExprId| matches!(arena.node(f), ExprNode::Exp(_));
    match arena.node(g) {
        ExprNode::Mul(children) => {
            children.iter().filter(|&&c| is_exp(c)).count() == 1
                && children
                    .iter()
                    .all(|&c| is_exp(c) || !contains_var(arena, c, var_sym))
        }
        _ => is_exp(g),
    }
}

/// Steps of `integrate_node` for the integrand of [`split_common_factors`].
const SPLIT_STEPS: u64 = 1_000;

/// Is `g` the variable to a numeric power times constants (`x`, `3·x²`)?
fn is_monomial_in_var(arena: &Arena, g: ExprId, var_sym: SymbolId) -> bool {
    let is_power = |f: ExprId| match arena.node(f) {
        ExprNode::Symbol(s) => *s == var_sym,
        ExprNode::Pow(b, k) => {
            matches!(arena.node(*b), ExprNode::Symbol(s) if *s == var_sym)
                && arena.as_num(*k).is_some()
        }
        _ => false,
    };
    match arena.node(g) {
        ExprNode::Mul(children) => children
            .iter()
            .all(|&c| is_power(c) || !contains_var(arena, c, var_sym)),
        _ => is_power(g),
    }
}

/// Is the constant `c` certainly non-zero: a non-zero number, non-zero by
/// the declared assumptions, or a symbol-free constant whose certified
/// value is not 0?
fn constant_nonzero(arena: &mut Arena, c: ExprId) -> bool {
    use crate::base::assumptions::{AssumptionCache, Props};
    use crate::transforms::evalf::{Settled, ZeroSearch, evalf_settled};
    if let Some(q) = arena.as_num(c) {
        return !q.is_zero();
    }
    if Truth::from(AssumptionCache::new().query(arena, c, Props::NONZERO)).is_true() {
        return true;
    }
    crate::base::walk::free_symbols(arena, c).is_empty()
        && !crate::base::walk::has_unevaluated(arena, c)
        && matches!(
            evalf_settled(arena, c, 20, ZeroSearch::Cap),
            Ok((z, Settled::Certified)) if !z.0.is_zero() || !z.1.is_zero()
        )
}

/// The `(parameter, value)` pairs at which the constant `alpha` vanishes,
/// for an answer that holds for `α ≠ 0` only (`Ei(α·x)` for `∫ e^{αx}/x`):
/// empty when `α` is certainly non-zero, or inside a rationalising
/// substitution ([`GENERIC_ONLY`]); `None` (no answer) when `α` has several
/// parameters or its zeros are not found.  Values the parameter's
/// assumptions exclude are left out.
fn vanishing_parameter_values(
    arena: &mut Arena,
    alpha: ExprId,
    var_sym: SymbolId,
) -> Option<Vec<(ExprId, ExprId)>> {
    if constant_nonzero(arena, alpha) || GENERIC_ONLY.with(std::cell::Cell::get) {
        return Some(Vec::new());
    }
    if contains_var(arena, alpha, var_sym) {
        return None;
    }
    let syms = crate::base::walk::free_symbols(arena, alpha);
    let [s] = syms[..] else {
        return None;
    };
    let solutions = crate::transforms::solve::known_real_roots(arena, alpha, s);
    if solutions.is_empty() {
        return None;
    }
    let mut out = Vec::new();
    for v in solutions {
        if !crate::base::walk::free_symbols(arena, v).is_empty() || contains_non_finite(arena, v) {
            return None;
        }
        if Truth::from(crate::transforms::solve::assumptions_hold(arena, s, v)).is_false()
            || nonreal_value_of_real_symbol(arena, s, v)
        {
            continue;
        }
        out.push((s, v));
    }
    Some(out)
}

/// `generic` with the cases `sᵢ = vᵢ` of [`vanishing_parameter_values`]:
/// `Piecewise((generic, s ≠ v), (∫ f|_{s=v}, True))`, nested; `None` when a
/// case has no closed form.
fn with_parameter_cases(
    arena: &mut Arena,
    generic: ExprId,
    integrand: ExprId,
    cases: &[(ExprId, ExprId)],
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    let mut out = generic;
    for &(s, v) in cases {
        let at = crate::transforms::subs::subs(arena, integrand, s, v);
        let at = crate::transforms::eval::eval(arena, at);
        if has_zero_denominator(arena, at) {
            continue;
        }
        let g = integrate_node(arena, at, var, var_sym, depth - 1);
        if crate::base::walk::has_unevaluated(arena, g) {
            return None;
        }
        let condition = arena.ne_(s, v);
        let true_cond = arena.bool_true;
        out = arena.piecewise(&[(out, condition), (g, true_cond)]);
    }
    Some(out)
}

/// The node of a library function with one argument.
fn special_apply(arena: &mut Arena, f: crate::base::libfn::LibFn, arg: ExprId) -> ExprId {
    arena.lib_apply(f, &[arg])
}

/// Split the factors of a product into those free of `var` and those
/// depending on it.  Dependent factors have their nested powers flattened
/// (`Pow(Pow(a, m), n)` → `Pow(a, m·n)`, or `Pow(|a|, m·n)` when that is
/// the real-valued identity, e.g. `sqrt(x²) = |x|`): the canon layer only
/// merges integer exponents (branch-cut safety), but for integration we
/// need e.g. `Pow(Pow(x²+1, 1/2), -1)` → `Pow(x²+1, -1/2)`.
fn partition_factors(
    arena: &mut Arena,
    children: &[ExprId],
    var_sym: SymbolId,
) -> (SmallVec<[ExprId; 4]>, SmallVec<[ExprId; 4]>) {
    let mut constants: SmallVec<[ExprId; 4]> = SmallVec::new();
    let mut dependent: SmallVec<[ExprId; 4]> = SmallVec::new();
    for &child in children {
        if contains_var(arena, child, var_sym) {
            let flat = flatten_nested_pow(arena, child, var_sym)
                .or_else(|| exp_power_as_exp(arena, child, var_sym))
                .unwrap_or(child);
            dependent.push(flat);
        } else {
            constants.push(child);
        }
    }
    (constants, dependent)
}

/// `exp(u)^c` as `exp(c·u)` (`c` free of the variable), which holds for
/// an integer `c`, and for every `c` when `u` is real at every real value
/// of the variable ([`real_for_real_variable`]): then `ln(eᵘ) = u`.  The
/// canonical form keeps `exp(x)^√37` (for a complex `x` the identity
/// fails), and up to 0.31 `∫ exp(x)^√37·cos 3x dx` stayed unevaluated
/// while `∫ e^{√37·x}·cos 3x dx` worked (normal-form hunt).
fn exp_power_as_exp(arena: &mut Arena, expr: ExprId, var_sym: SymbolId) -> Option<ExprId> {
    let ExprNode::Pow(base, c) = arena.node(expr).clone() else {
        return None;
    };
    let ExprNode::Exp(u) = arena.node(base).clone() else {
        return None;
    };
    if contains_var(arena, c, var_sym) {
        return None;
    }
    let integer = arena.as_num(c).is_some_and(|q| q.is_integer());
    if !integer && !real_for_real_variable(arena, u, var_sym) {
        return None;
    }
    let cu = arena.mul(&[c, u]);
    Some(arena.exp(cu))
}

/// Is `u` real at every real value of the variable?  Its variable-dependent
/// nodes must be real-to-real operations defined on the whole line (those
/// of [`real_on_reals`]); its maximal variable-free subexpressions are
/// decided by [`constant_is_real`] (`√37`, a declared-real `a`).
fn real_for_real_variable(arena: &mut Arena, u: ExprId, var_sym: SymbolId) -> bool {
    let mut reals = crate::base::assumptions::AssumptionCache::new();
    let mut stack = vec![u];
    let mut seen: FxHashSet<ExprId> = FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        if !contains_var(arena, id, var_sym) {
            if !constant_is_real(arena, id, &mut reals) {
                return false;
            }
            continue;
        }
        let whole_line = match arena.node(id) {
            ExprNode::Symbol(s) => *s == var_sym,
            ExprNode::Add(_)
            | ExprNode::Mul(_)
            | ExprNode::Sin(_)
            | ExprNode::Cos(_)
            | ExprNode::Exp(_)
            | ExprNode::Atan(_)
            | ExprNode::Sinh(_)
            | ExprNode::Cosh(_)
            | ExprNode::Tanh(_)
            | ExprNode::Abs(_) => true,
            ExprNode::Pow(_, exp) => arena
                .as_num(*exp)
                .is_some_and(|r| r.is_integer() && !r.is_negative()),
            _ => false,
        };
        if !whole_line {
            return false;
        }
        arena.node(id).for_each_child(|c| stack.push(c));
    }
    true
}

/// Largest degree of a product that [`expand_polynomial_product`] expands.
const MAX_EXPANDED_DEGREE: u32 = 64;

/// The product of `dependent` multiplied out when every factor is a
/// polynomial in the variable (sums, products and non-negative integer
/// powers of it and of constants) and one is a power of a sum, `(a +
/// b·x²)³` — a sum of monomials integrates term by term.  Up to 0.31
/// `∫ x³·(a + b·x²)³ dx` went through by parts three times (a nested
/// answer), and the substitution `u = cos x` of `∫ cos³x·(a + b·cos²x)³·sin x`
/// did not finish within its depth.  `None` otherwise or beyond
/// [`MAX_EXPANDED_DEGREE`].
fn expand_polynomial_product(
    arena: &mut Arena,
    dependent: &[ExprId],
    var_sym: SymbolId,
) -> Option<ExprId> {
    if dependent.len() < 2 {
        return None;
    }
    let mut has_power_of_sum = false;
    let mut degree: u32 = 0;
    for &d in dependent {
        // (the degree of the factor, bounded by exponent products)
        let mut stack: Vec<(ExprId, u32)> = vec![(d, 1)];
        let mut factor_degree: u32 = 0;
        while let Some((id, mult)) = stack.pop() {
            if !contains_var(arena, id, var_sym) {
                continue;
            }
            match arena.node(id) {
                ExprNode::Symbol(_) => factor_degree = factor_degree.max(mult),
                ExprNode::Add(ch) => stack.extend(ch.iter().map(|&c| (c, mult))),
                ExprNode::Mul(ch) => {
                    // A product's degree is at most the sum over its factors:
                    // bound it by the factor count times the largest.
                    let k = u32::try_from(ch.len()).ok()?;
                    stack.extend(ch.iter().map(|&c| (c, mult.saturating_mul(k))));
                }
                ExprNode::Pow(b, n) => {
                    let k = arena
                        .as_num(*n)
                        .filter(|q| q.is_integer() && !q.is_negative())?;
                    let k = u32::try_from(k.to_integer()).ok()?;
                    if matches!(arena.node(*b), ExprNode::Add(_)) && k >= 2 {
                        has_power_of_sum = true;
                    }
                    stack.push((*b, mult.saturating_mul(k)));
                }
                _ => return None,
            }
        }
        degree = degree.saturating_add(factor_degree);
    }
    if !has_power_of_sum || degree > MAX_EXPANDED_DEGREE {
        return None;
    }
    let product = arena.mul(dependent);
    let expanded = crate::transforms::expand::expand(arena, product);
    (expanded != product).then_some(expanded)
}

/// The product of `dependent` with its exponential factors whose exponents
/// are polynomials in the variable merged into one, `exp(u)·exp(v) =
/// exp(u + v)` (true for all complex `u`, `v`); `None` when fewer than two
/// such factors.  The canonical form keeps `exp(x − 2)·exp(−x)` and
/// `exp(√5·x)·exp(−x/3 − 1)`, whose integrals stayed unevaluated up to 0.31
/// (the first is the constant `e⁻²`).  Exponentials of other exponents
/// stay apart: `exp(x)·exp(exp(x))` is the substitution `u = eˣ`.
fn merge_exponential_factors(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    let mut exponents: SmallVec<[ExprId; 4]> = SmallVec::new();
    let mut rest: SmallVec<[ExprId; 4]> = SmallVec::new();
    for &d in dependent {
        match arena.node(d) {
            ExprNode::Exp(u) if is_polynomial_in(arena, *u, var, var_sym) => exponents.push(*u),
            _ => rest.push(d),
        }
    }
    if exponents.len() < 2 {
        return None;
    }
    let sum = arena.add(&exponents);
    let merged = arena.exp(sum);
    rest.push(merged);
    Some(arena.mul(&rest))
}

/// `(c, f)` with `expr = c·f` when `expr` is a product with factors `c` free
/// of the variable and at least one factor `f` that depends on it (nested
/// powers flattened as in [`partition_factors`]); `None` otherwise.
fn split_constant_factors(
    arena: &mut Arena,
    expr: ExprId,
    var_sym: SymbolId,
) -> Option<(SmallVec<[ExprId; 4]>, ExprId)> {
    let ExprNode::Mul(children) = arena.node(expr).clone() else {
        return None;
    };
    let (constants, dependent) = partition_factors(arena, &children, var_sym);
    if constants.is_empty() || dependent.is_empty() {
        return None;
    }
    let f = if dependent.len() == 1 {
        dependent[0]
    } else {
        arena.mul(&dependent)
    };
    Some((constants, f))
}

/// `c·F` for the integrand `expr = c·f` whose constant factors `c` were
/// split off (linearity) and `F = ∫ f`; `None` when that is not an
/// antiderivative of `expr`.
///
/// `F` may differ from `f` where `f` is complex (the real-variable
/// convention: `F′ = f` is checked only where `f` is real), and for a
/// constant `c` that is not real, `c·f` is real exactly at some such
/// points.  `∫ √3·i·√x·|√x| dx` was `√3·i·x²·sign(√x)/2` (`∫ √x·|√x| dx =
/// x²·sign(√x)/2` differs from its integrand only at `x < 0`, where that
/// is imaginary): at `x = −5/7` the integrand is the real `−1.2372` and the
/// answer's derivative `+1.2372` (up to 0.46; the CI Fuzz workflow's
/// `fuzz_integrate`).  So `c·F` is checked as an antiderivative of `c·f` when
/// `c` is not known to be real and `f` or `F` is not analytic in the
/// variable ([`non_analytic_in`]); an `F` built from analytic functions has
/// `F′ = f` along the whole real line once it holds where `f` is real.
fn constants_times_antiderivative(
    arena: &mut Arena,
    expr: ExprId,
    f: ExprId,
    big_f: ExprId,
    constants: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    let result = wrap_with_constants(arena, big_f, constants);
    if !non_analytic_in(arena, f, var_sym) && !non_analytic_in(arena, big_f, var_sym) {
        return Some(result);
    }
    let c = arena.mul(constants);
    let mut reals = crate::base::assumptions::AssumptionCache::new();
    if constant_is_real(arena, c, &mut reals) {
        return Some(result);
    }
    (!candidate_rejected(arena, expr, result, var, var_sym)).then_some(result)
}

/// Multiply a result by constant factors (if any).
fn wrap_with_constants(arena: &mut Arena, result: ExprId, constants: &[ExprId]) -> ExprId {
    if constants.is_empty() {
        result
    } else {
        let mut all: SmallVec<[ExprId; 4]> = constants.iter().copied().collect();
        all.push(result);
        arena.mul(&all)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Nested powers over the reals
// ═══════════════════════════════════════════════════════════════════════════

/// Flattening of `Pow(Pow(g, m), n)` with rational `m`, `n` that keeps the
/// value — on the principal branch, and on the real line of the
/// integration variable.
///
/// The canon layer merges an integer outer exponent `n` (since 0.25; before,
/// only integer `m` and `n`); here also `−1 < m ≤ 1` and the real-line
/// `|g|` forms.  `(g^m)^n = g^{m·n}` holds
/// for every complex `g` when `n` is an integer or `−1 < m ≤ 1` (the
/// condition of simplify's `pow_pow`); for `g` real on the real line and `m`
/// an even integer, `g^m = |g|^m ≥ 0` and `(g^m)^n = |g|^{m·n}` (`√(x²) =
/// |x|`), which is `g^{m·n}` again when `m·n` is an even integer.  Anything
/// else stays nested (`None`): `(1/x)^{1/2} ≠ x^{−1/2}` for `x < 0` (`i/√|x|`
/// against `−i/√|x|`), and `(g^{2/3})` is not `|g|^{2/3}` for `g < 0` —
/// before 0.23 both were flattened (the second under the real-root
/// convention the evaluator no longer uses), and `∫ √x·√(1/x) dx` came out
/// `x`, whose derivative is wrong for `x < 0` (found by `fuzz_integrate`).
fn flatten_nested_pow(arena: &mut Arena, expr: ExprId, var_sym: SymbolId) -> Option<ExprId> {
    let ExprNode::Pow(base, exp) = arena.node(expr).clone() else {
        return None;
    };
    let ExprNode::Pow(inner_base, inner_exp) = arena.node(base).clone() else {
        return None;
    };
    let (m, n) = (arena.as_num(inner_exp)?.clone(), arena.as_num(exp)?.clone());
    let combined = &m * &n;
    let two = num_bigint::BigInt::from(2);
    let even_integer = |q: &crate::base::numeric::Q| q.is_integer() && (q.numer() % &two).is_zero();
    let principal =
        n.is_integer() || (m > crate::base::numeric::qi(-1) && m <= crate::base::numeric::qi(1));
    let abs_form = even_integer(&m) && real_on_reals(arena, inner_base, var_sym);
    let combined_id = arena.num_ratio(combined.clone());
    if principal || (abs_form && even_integer(&combined)) {
        Some(arena.pow(inner_base, combined_id))
    } else if abs_form {
        let abs_base = arena.abs(inner_base);
        Some(arena.pow(abs_base, combined_id))
    } else {
        None
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Numeric verification of candidate antiderivatives
// ═══════════════════════════════════════════════════════════════════════════

/// Sample points for [`check_antiderivative`]: small rationals of both
/// signs, none of them a root of the denominators met in practice.
const FTC_SAMPLE_POINTS: [(i64, i64); 6] = [(1, 3), (-5, 7), (13, 11), (-17, 5), (7, 5), (-2, 1)];

/// The second round of sample points, tried only when the first found no
/// point where `F′ = f`: nearer to 0 and farther out, for integrands that
/// are real only on part of the line (`√(x − 5)` is complex at every
/// point of the first round, and `F′` may legitimately differ from it
/// there under the real-variable convention).
const FTC_RETRY_POINTS: [(i64, i64); 6] =
    [(1, 19), (-3, 23), (37, 4), (-29, 3), (211, 7), (-307, 9)];

/// Decimal digits requested from `evalf` when checking `F′ − f`: the
/// working precision behind 30 digits is ~50 digits, which separates an
/// exact antiderivative (`F′ − f` at roundoff, ~1e-45) from one built on
/// pseudo-exact decimal coefficients (`nsimplify`'d roots, ~1e-13).
const FTC_CHECK_DIGITS: u32 = 30;

/// Tolerance on `|F′(x₀) − f(x₀)| / max(1, |f(x₀)|)` at [`FTC_CHECK_DIGITS`].
const FTC_CHECK_REL_TOL: f64 = 1e-20;

/// How far from the tolerance the distance of the separately evaluated
/// `F′(x₀)` and `f(x₀)` must lie to decide a point of [`ftc_point`] without
/// evaluating `F′ − f` as one expression.  Both values are certified to
/// [`FTC_CHECK_DIGITS`] digits, ten orders of magnitude below the
/// tolerance, so any factor above 1 is safe; this one leaves room for the
/// `f64` rounding of the distance and of the tolerance.
const FTC_DECISIVE_FACTOR: f64 = 16.0;

/// What [`check_antiderivative`] found out about a candidate `F` for `∫ f`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FtcVerdict {
    /// `F′ = f` at one sample point at least, where `F` evaluates too, and
    /// `F′ ≠ f` at none where `f` is real.
    Verified,
    /// `F′ ≠ f` at a sample point where `f` is real, or `F′ − f` is a
    /// non-zero constant.
    Wrong,
    /// `f` evaluates at some sample point, but `F′` agrees with it at none:
    /// `F′` does not evaluate there, it differs from `f` only where `f` is
    /// complex, or it agrees only where `F` itself does not evaluate.
    Undecided,
    /// No numeric test is possible: `F` or `F′` has an unevaluated node, a
    /// parameter has declared assumptions, or `f` itself evaluates at no
    /// sample point.
    Untestable,
}

/// `true` when the candidate `big_f` for `∫ f d(var)` must not be returned.
///
/// A candidate is accepted on **evidence**, not for the lack of it.
/// [`check_antiderivative`] compares `F′` with `f` at sample points, and:
///
/// - [`FtcVerdict::Verified`] — accepted.
/// - [`FtcVerdict::Wrong`] — rejected.
/// - [`FtcVerdict::Undecided`] — **rejected**.  `f` evaluates at some sample
///   point but `F′` agrees with it at none, even after a second round of
///   points ([`FTC_RETRY_POINTS`]).  That is what a quotient by a
///   difference that cancels to 0 looks like: since 0.26 `evalf` refuses
///   such a value with `PrecisionExhausted` instead of returning noise, and
///   up to 0.28.0 such candidates passed unchecked, because "no point
///   evaluates" counted as "not wrong".  An `F′` that is undefined at a
///   generic point where `f` is defined is not the derivative of an
///   antiderivative of `f` there; nor is an `F` that is undefined wherever
///   its `F′` agrees with `f` (the vanishing denominator cancelled in
///   `F′`) an antiderivative.  A difference only where `f` is complex is
///   excused by the real-variable convention but verifies nothing:
///   `ln|x − c|` for `∫ dx/(x − c)` with an imaginary `c` differs from
///   `f` at every point.  A retry at higher precision would
///   add nothing: `evalf` already re-evaluates up to twice the working
///   precision plus 256 bits before it refuses.
/// - [`FtcVerdict::Untestable`] — accepted.  When `f` itself evaluates at no
///   sample point, `F` contains an unevaluated `Integral` (a partial
///   answer, which says so), `F′` a derivative `diff` could not take, or a
///   parameter carries declared assumptions none of the generic values
///   satisfies, no candidate could ever be tested; the answer stands as
///   the route derived it.
///
/// Free parameters are bound to generic positive rationals
/// ([`FTC_PARAMETER_VALUES`]) before sampling; up to 0.28.0 a parameter
/// skipped the check altogether.  A verified candidate with parameters
/// that are not declared real then faces a second round with those
/// parameters at generic **complex** values (decision D4, 0.31; see
/// [`check_antiderivative`]), which rejects the real-parameter forms
/// (`ln|a·x + 1|`, `|a|`, `sign(a)`) that the first round cannot tell from
/// the principal ones.
///
/// Reached through [`candidate_rejected`], by the routes that check their own
/// candidates and by every stage of `integrate_impl`, so that no closed form
/// leaves `integrate` without facing this rule.  The tight tolerance
/// [`FTC_CHECK_REL_TOL`] relies on the certified digits of `evalf`, which
/// refuses a value it cannot certify (the point then says nothing).  On
/// the Rubi suite (0.28) the stage answers came out 8,952 verified and 752
/// untestable (670 of them partial answers), none wrong or undecided.
fn antiderivative_rejected(
    arena: &mut Arena,
    f: ExprId,
    big_f: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> bool {
    // A route's accepted candidate is usually its stage's answer too, and
    // is then judged again at the stage's exit: the verdict is remembered
    // for the rest of the top-level call (cleared with `INTEGRATE_MEMO`).
    let key = (arena.serial(), f, big_f, var);
    let verdict = match VERDICT_MEMO.with(|m| m.borrow().get(&key).copied()) {
        Some(verdict) => verdict,
        None => {
            let verdict = stage!(
                arena,
                "verify",
                big_f,
                check_antiderivative(arena, f, big_f, var, var_sym)
            );
            VERDICT_MEMO.with(|m| m.borrow_mut().insert(key, verdict));
            verdict
        }
    };
    match verdict {
        FtcVerdict::Verified => {
            let key = (arena.serial(), f, big_f, var);
            if let Some(jumps) = JUMP_MEMO.with(|m| m.borrow().get(&key).copied()) {
                return jumps;
            }
            let jumps = jumps_at_argument_poles(arena, f, big_f, var, var_sym);
            JUMP_MEMO.with(|m| m.borrow_mut().insert(key, jumps));
            if jumps {
                tracing::debug!("integrate: candidate jumps where the integrand is finite");
            }
            jumps
        }
        FtcVerdict::Untestable => false,
        FtcVerdict::Wrong | FtcVerdict::Undecided => {
            tracing::debug!(?verdict, "integrate: candidate antiderivative rejected");
            true
        }
    }
}

thread_local! {
    /// [`jumps_at_argument_poles`] verdicts during the current top-level
    /// `integrate` (cleared with [`VERDICT_MEMO`]).
    static JUMP_MEMO: std::cell::RefCell<FxHashMap<(u64, ExprId, ExprId, ExprId), bool>> =
        std::cell::RefCell::new(FxHashMap::default());
}

/// Does the candidate `big_f` jump at a real pole `p` of the argument of one
/// of its logarithms or dilogarithms (`ln(1/y)`, `Li₂(1 + 1/y)` at `y = 0`)
/// where the integrand `f` stays finite: `|F(p + ε) − F(p − ε)|` well above
/// `2ε·|f|`?  The FTC check samples `F′` away from such points and cannot see
/// it.  Up to 0.46 `∫ (ln(−1/y)/2 − ln(1/y)/2)/y − (ln(1 − 1/y)/2 − ln(1 +
/// 1/y)/2)/y dy` (SymPy's test_risch; its integrand tends to 1 at 0 from
/// both sides) came out with a step of `π²/2` there.  Only for parameter-free
/// candidates with such an argument; numeric poles only.
fn jumps_at_argument_poles(
    arena: &mut Arena,
    f: ExprId,
    big_f: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> bool {
    if crate::base::walk::free_symbols(arena, big_f)
        .iter()
        .any(|&s| s != var)
    {
        return false;
    }
    // A check that can only reject a candidate: its `f64` tolerances below
    // decide nothing else.
    let mut poles: Vec<f64> = Vec::new();
    for id in crate::base::walk::post_order_ids(arena, big_f) {
        let arg = match arena.node(id) {
            ExprNode::Ln(a) => *a,
            ExprNode::Lib(crate::base::libfn::LibFn::PolyLog, args) if args.len() == 2 => args[1],
            _ => continue,
        };
        if !contains_var(arena, arg, var_sym) {
            continue;
        }
        let (_, den) = crate::poly::polybridge::as_numer_denom(arena, arg);
        if !contains_var(arena, den, var_sym) {
            continue;
        }
        let Some(roots) = real_roots(arena, den, var) else {
            continue;
        };
        for r in roots {
            if let Some(v) = crate::transforms::evalf::eval_const_f64(arena, r)
                && !poles
                    .iter()
                    .any(|&w| (w - v).abs() <= 1e-12 * v.abs().max(1.0))
            {
                poles.push(v);
            }
        }
        if poles.len() > 4 {
            return false;
        }
    }
    for p in poles {
        let eps = 1e-7 * p.abs().max(1.0);
        let at = |arena: &mut Arena, e: ExprId, t: f64| -> Option<num_complex::Complex64> {
            let q = crate::base::numeric::f64_to_ratio_exact(t)?;
            let tv = arena.num_ratio(q);
            let s = crate::transforms::subs::subs(arena, e, var, tv);
            let s = crate::transforms::eval::eval(arena, s);
            crate::transforms::evalf::evalf_complex64(arena, s)
                .ok()
                .filter(|z| z.re.is_finite() && z.im.is_finite())
        };
        let (Some(f_lo), Some(f_hi)) = (at(arena, f, p - eps), at(arena, f, p + eps)) else {
            continue;
        };
        let scale = f_lo.norm().max(f_hi.norm());
        if scale.is_nan() || scale >= 1e6 || (f_lo - f_hi).norm() > 1e-3 * scale.max(1.0) {
            continue;
        }
        let (Some(big_lo), Some(big_hi)) = (at(arena, big_f, p - eps), at(arena, big_f, p + eps))
        else {
            continue;
        };
        if (big_hi - big_lo).norm() > 1e-4 * big_lo.norm().max(1.0) + 100.0 * eps * scale.max(1.0) {
            return true;
        }
    }
    false
}

thread_local! {
    /// [`check_antiderivative`] verdicts during the current top-level
    /// `integrate`, by (arena, `f`, `F`, variable).
    static VERDICT_MEMO: std::cell::RefCell<FxHashMap<(u64, ExprId, ExprId, ExprId), FtcVerdict>> =
        std::cell::RefCell::new(FxHashMap::default());
}

/// Generic values for the free parameters of a checked candidate, handed
/// out in the order of the parameters' names: distinct, positive,
/// non-integer, with pairwise different prime denominators, so that no
/// accidental relation (`a = b`, `m + 1 = n`, `a·b = 1`) holds between
/// them.  None is a value of the Rubi harness's parameter table, so the
/// harness stays an independent check.
const FTC_PARAMETER_VALUES: [(i64, i64); 8] = [
    (17, 13),
    (7, 11),
    (23, 17),
    (5, 19),
    (31, 23),
    (11, 29),
    (41, 31),
    (13, 37),
];

/// Imaginary parts of the complex values of the second round of
/// [`check_antiderivative`]: the k-th parameter that is not declared real
/// takes `FTC_PARAMETER_VALUES[k] + i·FTC_PARAMETER_IMAG[k]`.
/// Prime denominators unlike those of the real parts, so the values keep
/// clear of accidental relations; moderate imaginary parts (0.3 to 0.5)
/// and positive real parts, see [`complex_round_differs`].
const FTC_PARAMETER_IMAG: [(i64, i64); 8] = [
    (13, 41),
    (19, 43),
    (17, 47),
    (23, 53),
    (19, 59),
    (29, 61),
    (23, 67),
    (31, 71),
];

/// Which values [`bind_parameters`] gives the parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ParameterValues {
    /// The k-th parameter at the k-th value of [`FTC_PARAMETER_VALUES`]
    /// or its negative, whichever satisfies its declaration.
    Real,
    /// A parameter that is not declared real at that value plus
    /// `i·FTC_PARAMETER_IMAG[k]` where that satisfies its declaration (always,
    /// for one without declared assumptions); the others as in `Real`.
    Complex,
}

/// `f` and `big_f` with their parameters bound, and whether any of them
/// may be complex (is not declared real), and so is bound to a complex
/// value in the [`ParameterValues::Complex`] round.
struct BoundCandidate {
    f: ExprId,
    big_f: ExprId,
    may_be_complex: bool,
}

/// `f` and `big_f` with every free symbol other than `var` bound to a
/// value derived from [`FTC_PARAMETER_VALUES`] ([`ParameterValues`]): the
/// k-th parameter in name order takes the k-th value, its negative, or (in
/// the complex round, for a parameter not declared real) that value plus
/// `i·FTC_PARAMETER_IMAG[k]` — the first of these that provably satisfies
/// every declared property (a parameter without assumptions takes any; a
/// `real` or `positive` one is sampled at `17/13`, a `negative` one at
/// `−17/13`).  `None` when no candidate does (an `integer` symbol must not
/// be sampled at `17/13`) or there are more parameters than values.  Up to
/// 0.30 any declared parameter made the candidate untestable.  `subs`
/// replaces free occurrences only, so a `RootOf(a⁵ − a + 1, 0)` or a sum
/// over `a` keeps its bound `a` (0.28.0 renamed such binders here first,
/// when `subs` did not respect them).
fn bind_parameters(
    arena: &mut Arena,
    f: ExprId,
    big_f: ExprId,
    var: ExprId,
    mode: ParameterValues,
) -> Option<BoundCandidate> {
    let mut params: Vec<(String, ExprId)> = Vec::new();
    for root in [f, big_f] {
        for s in crate::base::walk::free_symbols(arena, root) {
            if s == var || params.iter().any(|&(_, p)| p == s) {
                continue;
            }
            let ExprNode::Symbol(sid) = *arena.node(s) else {
                return None;
            };
            params.push((arena.symbol_name(sid).to_owned(), s));
        }
    }
    if params.len() > FTC_PARAMETER_VALUES.len() {
        return None;
    }
    params.sort_by(|a, b| a.0.cmp(&b.0));
    let (mut f, mut big_f) = (f, big_f);
    let mut may_be_complex = false;
    let mut facts = crate::base::assumptions::AssumptionCache::new();
    for (k, &(_, s)) in params.iter().enumerate() {
        let ExprNode::Symbol(sid) = *arena.node(s) else {
            return None;
        };
        let mut declared = arena.symbol_assumptions(sid);
        declared.normalize_declared();
        let complex = !Truth::from(declared.query(crate::base::assumptions::Props::REAL)).is_true();
        may_be_complex |= complex;
        let (p, q) = FTC_PARAMETER_VALUES[k];
        let re = arena.rational(p, q);
        let mut candidates: SmallVec<[ExprId; 3]> = SmallVec::new();
        if mode == ParameterValues::Complex && complex {
            let (r, t) = FTC_PARAMETER_IMAG[k];
            let im = arena.rational(r, t);
            let i = arena.i_unit();
            let im = arena.mul(&[im, i]);
            candidates.push(arena.add(&[re, im]));
        }
        candidates.push(re);
        candidates.push(arena.rational(-p, q));
        // An integer parameter at the numerator (distinct odd primes, so
        // again no accidental relation), its negative, or twice either
        // for an even one.  Up to 0.46 no candidate satisfied `integer`,
        // and every antiderivative with an integer parameter was accepted
        // untested: `∫ sin(kx)·sin(|k|x) dx` as a quotient by `k − |k|`.
        for m in [1, -1, 2, -2] {
            candidates.push(arena.int(m * p));
        }
        let value = candidates
            .into_iter()
            .find(|&v| value_satisfies(arena, v, declared, &mut facts))?;
        f = crate::transforms::subs::subs(arena, f, s, value);
        big_f = crate::transforms::subs::subs(arena, big_f, s, value);
    }
    Some(BoundCandidate {
        f,
        big_f,
        may_be_complex,
    })
}

/// Does the constant `value` provably have every property `declared`
/// asserts and provably lack every one it denies?
fn value_satisfies(
    arena: &mut Arena,
    value: ExprId,
    declared: crate::base::assumptions::Assumptions,
    facts: &mut crate::base::assumptions::AssumptionCache,
) -> bool {
    declared
        .known_true
        .iter()
        .all(|flag| Truth::from(facts.query(arena, value, flag)).is_true())
        && declared
            .known_false
            .iter()
            .all(|flag| Truth::from(facts.query(arena, value, flag)).is_false())
}

/// The numeric FTC check behind [`antiderivative_rejected`]: `F′` against
/// `f` at [`FTC_SAMPLE_POINTS`], then, if no point agreed, at
/// [`FTC_RETRY_POINTS`].
///
/// At each point `f` is evaluated first; a point where it does not
/// evaluate says nothing.  Then `F′` on its own, which decides the point
/// when its distance from `f` is clear of the tolerance (see
/// [`ftc_point`]); otherwise `F′ − f` is evaluated as one expression (its
/// exact cancellation to 0 is what `evalf`'s zero detection is built for).
/// `|F′ − f| ≤ tol·max(1, |f|)` is an agreement, real or complex; a larger
/// difference is
/// [`FtcVerdict::Wrong`] where `f` is real and no evidence where it is
/// complex (the real-variable convention: `ln|x|` for `∫ dx/x` differs
/// from `1/x` for `x < 0` only in a complex continuation, and the
/// continuation of `f` is what is compared there).
///
/// That first round binds every parameter to a real rational
/// ([`ParameterValues::Real`]), where `ln|a·x + 1|/a` and `ln(a·x + 1)/a`
/// both pass.  A verified candidate with parameters that are not declared
/// real — complex numbers in the crate's model — faces a second round with
/// them at complex values ([`complex_round_differs`]); a difference there
/// makes it [`FtcVerdict::Wrong`] (decision D4, 0.31).
fn check_antiderivative(
    arena: &mut Arena,
    f: ExprId,
    big_f: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> FtcVerdict {
    if crate::base::walk::has_unevaluated(arena, big_f) {
        return FtcVerdict::Untestable;
    }
    let Some(bound) = bind_parameters(arena, f, big_f, var, ParameterValues::Real) else {
        return FtcVerdict::Untestable;
    };
    let mut real_domain: Vec<ExprId> = Vec::new();
    let verdict =
        check_bound_antiderivative(arena, bound.f, bound.big_f, var, var_sym, &mut real_domain);
    if verdict != FtcVerdict::Verified || !bound.may_be_complex {
        return verdict;
    }
    // The complex round tells apart forms that are not analytic in the
    // parameters; an `f` and `F` built from analytic functions of them
    // alone agree there as they do at the real values (the continuation
    // argument of [`complex_round_differs`]).  It was half of the cost of
    // the check (0.31: 4.5 of 9.7 s on a sample of the Rubi suite's
    // answers, and none of its 72,000 integrands met a candidate the
    // round rejected that is analytic in this sense).
    if analytic_in_parameters(arena, f, var_sym) && analytic_in_parameters(arena, big_f, var_sym) {
        return verdict;
    }
    let Some(complex) = bind_parameters(arena, f, big_f, var, ParameterValues::Complex) else {
        return verdict;
    };
    if real_domain.is_empty() {
        real_domain = real_domain_points(arena, bound.f, var);
    }
    if complex_round_differs(arena, complex.f, complex.big_f, var, var_sym, &real_domain) {
        tracing::debug!(
            "integrate: candidate antiderivative fails the FTC check at complex parameter values"
        );
        return FtcVerdict::Wrong;
    }
    verdict
}

/// The functions [`analytic_in_parameters`] counts as analytic: the
/// elementary functions, `Γ`, `erf`, `erfc`, `Si`, `Ci`, `Ei`, `li`, `ζ`.
fn analytic_off_cuts(f: crate::base::func::Func) -> bool {
    use crate::base::func::Func;
    f.elementary().is_some()
        || matches!(
            f,
            Func::Gamma
                | Func::Erf
                | Func::Erfc
                | Func::Si
                | Func::Ci
                | Func::Ei
                | Func::Li
                | Func::Zeta
        )
}

/// Is `e` analytic in its parameters (the free symbols other than the
/// variable): does every node that is not an analytic function of its
/// arguments — `|·|`, `sign`, `re`, `arg`, `Piecewise`, a relation, an
/// unknown function, … — have no parameter below it?  (`ln|x|` is;
/// `ln|a·x + 1|` and `sign(a)` are not.)  Elementary functions, `erf`,
/// `Ei`, `li`, `Si`, `Ci`, `Γ`, `ζ` and `RootSum` (a symmetric function of
/// the roots) count as analytic: their principal branches are analytic off
/// their cuts, and crossing a cut changes `F` by a constant, not `F′`.
fn analytic_in_parameters(arena: &Arena, e: ExprId, var_sym: SymbolId) -> bool {
    let mut has_param: FxHashMap<ExprId, bool> = FxHashMap::default();
    for id in crate::base::walk::post_order_ids(arena, e) {
        let node = arena.node(id);
        let mut below = matches!(node, ExprNode::Symbol(s) if *s != var_sym);
        node.for_each_child(|c| below |= has_param.get(&c).copied().unwrap_or(false));
        let analytic = matches!(
            node,
            ExprNode::Num(_)
                | ExprNode::Symbol(_)
                | ExprNode::Pi
                | ExprNode::E
                | ExprNode::ImaginaryUnit
                | ExprNode::EulerGamma
                | ExprNode::Catalan
                | ExprNode::GoldenRatio
                | ExprNode::Add(_)
                | ExprNode::Mul(_)
                | ExprNode::Pow(..)
                | ExprNode::Neg(_)
                | ExprNode::RootSum(..)
        ) || node.func_head().is_some_and(analytic_off_cuts);
        if below && !analytic {
            return false;
        }
        has_param.insert(id, below);
    }
    true
}

/// The points of [`FTC_SAMPLE_POINTS`] and [`FTC_RETRY_POINTS`] where the
/// integrand `f` (parameters bound to real values) evaluates to a real
/// number.
fn real_domain_points(arena: &mut Arena, f: ExprId, var: ExprId) -> Vec<ExprId> {
    let mut out = Vec::new();
    for &(p, q) in FTC_SAMPLE_POINTS.iter().chain(FTC_RETRY_POINTS.iter()) {
        let point = arena.rational(p, q);
        let f_at = crate::transforms::subs::subs(arena, f, var, point);
        if numeric_value(arena, f_at)
            .is_some_and(|v| crate::transforms::evalf::is_real_to_digits(&v, FTC_CHECK_DIGITS))
        {
            out.push(point);
        }
    }
    out
}

/// Points of the complex round that must agree before it gives up
/// looking for a difference.
const FTC_COMPLEX_AGREEMENTS: usize = 3;

/// The second round of [`check_antiderivative`]: `f` and `big_f` with the
/// parameters that are not declared real at complex values
/// ([`ParameterValues::Complex`]), compared at `real_domain` — the real
/// sample points where the integrand is real for the real values of the
/// first round.  `true` when `F′ ≠ f` at one of them.
///
/// Why the comparison is sound although `f` is complex at every point now,
/// where the first round excuses a difference (the real-variable
/// convention): `x` is real and only the parameters moved off the real
/// axis.  At a point where the integrand is real for real parameters,
/// `F′ = f` holds for the parameters on a real interval, and both sides
/// are analytic in the parameters there, so the identity continues to
/// complex parameter values near that interval — unless `F` is not
/// analytic in them, which is exactly what a real-parameter form is:
/// `d/dx ln|a·x + 1| = re(conj(u)·a)/|u|²` differs from `a/u` at every point
/// where `u` is not real; so do `|a|` and `sign(a)`.  The allowance of the
/// first round is kept where it belongs: a point where the integrand is
/// complex for real parameters (`√(x − 5)` at `x = 1/3`) is not in
/// `real_domain` and is not compared, so `ln|x + √(x² − 1)|`-type parts
/// that depend on `x` alone keep their convention whatever their
/// coefficients.  The complex values have positive real parts
/// ([`FTC_PARAMETER_IMAG`]): a principal form such as `√a` has its cut on
/// the negative axis, and values next to the positive reals continue the
/// identity from there without crossing it.  An undecidable point (`F′`
/// or `f` does not evaluate at the complex values) says nothing; the round
/// stops after [`FTC_COMPLEX_AGREEMENTS`] agreeing points.
fn complex_round_differs(
    arena: &mut Arena,
    f: ExprId,
    big_f: ExprId,
    var: ExprId,
    var_sym: SymbolId,
    real_domain: &[ExprId],
) -> bool {
    let d_big_f = crate::transforms::diff::diff(arena, big_f, var);
    if crate::base::walk::has_unevaluated(arena, d_big_f) {
        return false;
    }
    let residual = arena.sub(d_big_f, f);
    if !contains_var(arena, residual, var_sym) {
        return residual_abs(arena, residual).is_some_and(|r| r > FTC_CHECK_REL_TOL);
    }
    let mut agreements = 0;
    for &point in real_domain {
        match ftc_point(arena, f, d_big_f, residual, var, point) {
            FtcPoint::Differs { .. } => {
                tracing::debug!(
                    point = %arena.display(point),
                    "integrate: F′ ≠ f at complex parameter values"
                );
                return true;
            }
            FtcPoint::Agrees { .. } => {
                agreements += 1;
                if agreements >= FTC_COMPLEX_AGREEMENTS {
                    return false;
                }
            }
            FtcPoint::IntegrandUnevaluable | FtcPoint::DerivativeUnevaluable => {}
        }
    }
    false
}

/// The first round of [`check_antiderivative`], on `f` and `big_f` with
/// their parameters bound.  Every sample point where `f` is real and
/// `F′ = f` is appended to `real_domain`.
fn check_bound_antiderivative(
    arena: &mut Arena,
    f: ExprId,
    big_f: ExprId,
    var: ExprId,
    var_sym: SymbolId,
    real_domain: &mut Vec<ExprId>,
) -> FtcVerdict {
    let d_big_f = crate::transforms::diff::diff(arena, big_f, var);
    if crate::base::walk::has_unevaluated(arena, d_big_f) {
        return FtcVerdict::Untestable;
    }
    let residual = arena.sub(d_big_f, f);
    if !contains_var(arena, residual, var_sym)
        && let Some(r) = residual_abs(arena, residual)
    {
        // A constant residual: anything but zero is a wrong constant term
        // in F′.  Zero (`F′ ≡ f`, often structurally) still needs an `F`
        // that is defined somewhere.  One that does not evaluate is
        // sampled below.
        if r > FTC_CHECK_REL_TOL {
            return FtcVerdict::Wrong;
        }
        return if evaluates_at_a_sample_point(arena, big_f, var) {
            FtcVerdict::Verified
        } else if evaluates_at_a_sample_point(arena, f, var) {
            FtcVerdict::Undecided
        } else {
            FtcVerdict::Untestable
        };
    }
    let mut integrand_evaluates = false;
    for points in [&FTC_SAMPLE_POINTS, &FTC_RETRY_POINTS] {
        let mut agrees = false;
        for &(p, q) in points {
            let point = arena.rational(p, q);
            match ftc_point(arena, f, d_big_f, residual, var, point) {
                FtcPoint::IntegrandUnevaluable => {}
                FtcPoint::Agrees { f_real } => {
                    integrand_evaluates = true;
                    if f_real {
                        real_domain.push(point);
                    }
                    // `F′ = f` proves nothing about an `F` that is undefined
                    // (`x·e^{kx}/k − e^{kx}/k²` with a `k` that is 0 without
                    // being so structurally has `F′ = x·e^{kx}` exactly).
                    if !agrees {
                        let big_f_at = crate::transforms::subs::subs(arena, big_f, var, point);
                        agrees = numeric_value(arena, big_f_at).is_some();
                    }
                }
                FtcPoint::DerivativeUnevaluable | FtcPoint::Differs { f_real: false } => {
                    integrand_evaluates = true;
                }
                FtcPoint::Differs { f_real: true } => {
                    tracing::debug!(
                        point = %arena.display(point),
                        "integrate: candidate antiderivative fails the numeric FTC check"
                    );
                    return FtcVerdict::Wrong;
                }
            }
        }
        if agrees {
            return FtcVerdict::Verified;
        }
    }
    if integrand_evaluates {
        FtcVerdict::Undecided
    } else {
        FtcVerdict::Untestable
    }
}

/// The comparison of `F′` with `f` at one sample point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FtcPoint {
    /// `f` does not evaluate here: no information.
    IntegrandUnevaluable,
    /// `f` evaluates, `F′` (and `F′ − f`) does not.
    DerivativeUnevaluable,
    /// `|F′ − f| ≤ tol·max(1, |f|)`, with `f` real or complex here.
    Agrees { f_real: bool },
    /// `|F′ − f| > tol·max(1, |f|)`, with `f` real or complex here.
    Differs { f_real: bool },
}

fn ftc_point(
    arena: &mut Arena,
    f: ExprId,
    d_big_f: ExprId,
    residual: ExprId,
    var: ExprId,
    point: ExprId,
) -> FtcPoint {
    let f_at = crate::transforms::subs::subs(arena, f, var, point);
    let Some(f_val) = numeric_value(arena, f_at) else {
        return FtcPoint::IntegrandUnevaluable;
    };
    let Some(f_abs) = crate::transforms::evalf::abs_to_f64(&f_val).filter(|v| v.is_finite()) else {
        return FtcPoint::IntegrandUnevaluable;
    };
    let f_real = crate::transforms::evalf::is_real_to_digits(&f_val, FTC_CHECK_DIGITS);
    let tol = FTC_CHECK_REL_TOL * f_abs.max(1.0);

    // `F′` on its own first.  Both values are certified to
    // [`FTC_CHECK_DIGITS`] digits, so their distance is `|F′ − f|` to within
    // `10⁻³⁰·(|F′| + |f|)`, far below the tolerance: a distance clear of it
    // by [`FTC_DECISIVE_FACTOR`] either way decides the point as `F′ − f`
    // evaluated as one expression would.  That expression is a true zero
    // for a right answer, which `evalf` can only settle by re-evaluating
    // the whole of `F′` at up to two and a half times the working precision
    // (the zero search to its cap), where `F′` alone settles in one pass:
    // up to 0.38 those re-evaluations were half of the integrator's time on
    // the Rubi suite's trigonometric chapters.  A distance near the
    // tolerance, or an `F′` that does not evaluate on its own, still goes
    // to the residual.
    let d_at = crate::transforms::subs::subs(arena, d_big_f, var, point);
    let d_val = numeric_value(arena, d_at);
    if let Some(d) = &d_val
        && let Some(gap) = crate::transforms::evalf::distance_to_f64(d, &f_val)
        && !gap.is_nan()
    {
        if gap * FTC_DECISIVE_FACTOR <= tol {
            return FtcPoint::Agrees { f_real };
        }
        if gap >= tol * FTC_DECISIVE_FACTOR {
            return FtcPoint::Differs { f_real };
        }
    }

    let r_at = crate::transforms::subs::subs(arena, residual, var, point);
    let gap = match residual_abs(arena, r_at) {
        Some(gap) => gap,
        None => {
            let Some(d_val) = d_val else {
                return FtcPoint::DerivativeUnevaluable;
            };
            match crate::transforms::evalf::distance_to_f64(&d_val, &f_val) {
                Some(gap) if !gap.is_nan() => gap,
                _ => return FtcPoint::DerivativeUnevaluable,
            }
        }
    };
    if gap <= tol {
        FtcPoint::Agrees { f_real }
    } else {
        FtcPoint::Differs { f_real }
    }
}

/// Does `e` (a function of `var` only) evaluate at one of the sample
/// points?
fn evaluates_at_a_sample_point(arena: &mut Arena, e: ExprId, var: ExprId) -> bool {
    FTC_SAMPLE_POINTS
        .iter()
        .chain(FTC_RETRY_POINTS.iter())
        .any(|&(p, q)| {
            let point = arena.rational(p, q);
            let e_at = crate::transforms::subs::subs(arena, e, var, point);
            numeric_value(arena, e_at).is_some()
        })
}

/// `e` with every `RootSum` whose polynomial and body are numeric and whose
/// body is a rational function of the root replaced by its exact value
/// ([`crate::calculus::risch::log_to_real::numeric_rootsum_rational_body`]):
/// the derivative of a `RootSum` answer at a sample point.  `eval` would
/// solve the polynomial in radicals at every point instead.
fn exact_numeric_root_sums(arena: &mut Arena, e: ExprId) -> ExprId {
    let order = crate::base::walk::post_order_ids(arena, e);
    let mut out = e;
    for id in order {
        if let ExprNode::RootSum(poly, body, sumvar) = *arena.node(id)
            && let Some(value) = crate::calculus::risch::log_to_real::numeric_rootsum_rational_body(
                arena, poly, body, sumvar,
            )
        {
            out = arena.subs_structural(out, id, value);
        }
    }
    out
}

/// A variable-free expression evaluated to [`FTC_CHECK_DIGITS`] correct
/// digits (after exact simplification); `None` when it does not evaluate
/// to a finite complex number.
fn numeric_value(arena: &mut Arena, e: ExprId) -> Option<crate::base::bigcomplex::Complex> {
    let e = exact_numeric_root_sums(arena, e);
    let e = crate::transforms::eval::eval(arena, e);
    if crate::base::walk::has_unevaluated(arena, e) {
        return None;
    }
    crate::transforms::evalf::evalf_complex(arena, e, FTC_CHECK_DIGITS).ok()
}

/// `|r|` for a variable-free `F′ − f` sample, evaluated at
/// [`FTC_CHECK_DIGITS`] so that a ~1e-13 discrepancy is not lost; `None`
/// when it does not evaluate to a finite complex number.
fn residual_abs(arena: &mut Arena, residual: ExprId) -> Option<f64> {
    let r = exact_numeric_root_sums(arena, residual);
    let r = crate::transforms::eval::eval(arena, r);
    if crate::base::walk::has_unevaluated(arena, r) {
        return None;
    }
    let v = if let Some(q) = arena.as_num(r) {
        crate::base::numeric::ratio_to_f64(q)?.abs()
    } else {
        let z = crate::transforms::evalf::evalf_complex(arena, r, FTC_CHECK_DIGITS).ok()?;
        crate::transforms::evalf::abs_to_f64(&z)?
    };
    v.is_finite().then_some(v)
}

// ═══════════════════════════════════════════════════════════════════════════
// Rational functions with a biquadratic denominator
// ═══════════════════════════════════════════════════════════════════════════

/// `∫ N(x) / (x⁴ + p·x² + q) dx` with `deg N ≤ 3` and rational `p`, `q`,
/// for the two shapes without real roots:
///
/// ```text
///   p² < 4q:            x⁴ + p·x² + q = (x² − a·x + c)(x² + a·x + c),
///                                        c = √q,  a = √(2c − p)
///   p² > 4q, p, q > 0:  x⁴ + p·x² + q = (x² + r₁)(x² + r₂),
///                                        r₁,₂ = (p ∓ √(p² − 4q))/2 > 0
/// ```
///
/// followed by partial fractions over the corresponding real radical field
/// and the closed forms
/// `∫ (A·x + B)/(x² + a·x + c) = (A/2)·ln(x² + a·x + c)
///   + (2B − a·A)/√(4c − a²) · atan((2x + a)/√(4c − a²))`,
/// `∫ (A·x + B)/(x² + r) = (A/2)·ln(x² + r) + (B/√r)·atan(x/√r)`.
///
/// This is the route the general rational integrator gets wrong when the
/// Rothstein–Trager resultant has nested-radical roots (the `re(…)`/`im(…)`
/// closed forms of the 0.21 audit; also the denominators the Weierstrass
/// substitution produces for `∫ 1/(a·sin²x + b) dx`).  Here every constant
/// is an explicit real radical.  Returns `None` when `expr` is not of
/// either shape.
fn try_biquadratic_rational(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    let (numer_id, denom_id) = crate::poly::polybridge::as_numer_denom(arena, expr);
    if denom_id == arena.one {
        return None;
    }
    let numer_id = crate::transforms::expand::expand(arena, numer_id);
    let numer_id = crate::transforms::eval::eval(arena, numer_id);
    let denom_id = crate::transforms::expand::expand(arena, denom_id);
    let denom_id = crate::transforms::eval::eval(arena, denom_id);
    let numer = crate::poly::polybridge::expr_to_poly(arena, numer_id, var)?;
    let denom = crate::poly::polybridge::expr_to_poly(arena, denom_id, var)?;
    if denom.degree() != Some(4) || numer.is_zero() {
        return None;
    }
    if !denom.coeff(3).is_zero() || !denom.coeff(1).is_zero() {
        return None;
    }
    let lead = denom.coeff(4);
    if lead.is_zero() {
        return None;
    }
    // Improper fraction: N = Q·D + R, ∫ N/D = ∫ Q + ∫ R/D with ∫ Q by the
    // power rule (Σ qₖ x^{k+1}/(k+1)).
    let (quotient, numer) = if numer.degree().unwrap_or(0) > 3 {
        let (quot, rem) = numer.div_rem(&denom);
        (Some(quot), rem)
    } else {
        (None, numer)
    };
    let poly_part = quotient.filter(|q| !q.is_zero()).map(|q| {
        let deg = q.degree().unwrap_or(0);
        let mut coeffs: Vec<Q> = vec![Q::zero(); deg + 2];
        for (k, c) in coeffs.iter_mut().enumerate().skip(1) {
            *c = q.coeff(k - 1) / Q::from_integer(k.into());
        }
        let antiderivative = Poly::from_coeffs(coeffs);
        crate::poly::polybridge::poly_to_expr(arena, &antiderivative, var)
    });
    if numer.is_zero() {
        // N was a polynomial multiple of D.
        return poly_part;
    }
    let p = denom.coeff(2) / &lead;
    let q = denom.coeff(0) / &lead;
    let n_ids: [ExprId; 4] = [
        arena.num_ratio(numer.coeff(0) / &lead),
        arena.num_ratio(numer.coeff(1) / &lead),
        arena.num_ratio(numer.coeff(2) / &lead),
        arena.num_ratio(numer.coeff(3) / &lead),
    ];
    let disc = &p * &p - Q::from_integer(4.into()) * &q;
    let quadratics: SmallVec<[(ExprId, ExprId, ExprId, ExprId); 2]> = if disc.is_negative() {
        tracing::debug!(p = %p, q = %q, "integrate: biquadratic denominator, x² ± a·x + c");
        biquadratic_conjugate_quadratics(arena, &p, &q, &n_ids)
    } else if disc.is_positive() && p.is_positive() && q.is_positive() {
        tracing::debug!(p = %p, q = %q, "integrate: biquadratic denominator, (x² + r₁)(x² + r₂)");
        biquadratic_even_quadratics(arena, &p, &disc, &n_ids)
    } else {
        // Real roots (or a repeated factor): the general integrator's business.
        return None;
    };

    // ∫ (A x + B)/(x² + a x + c) = (A/2) ln(x² + a x + c)
    //                              + (2B − aA)/k · atan((2x + a)/k),  k = √(4c − a²)
    let two = arena.int(2);
    let four = arena.int(4);
    let half = arena.rational(1, 2);
    let x_sq = arena.pow(var, two);
    let two_x = arena.mul(&[two, var]);
    let mut terms: SmallVec<[ExprId; 4]> = SmallVec::new();
    for &(a, c, lin, cst) in quadratics.iter() {
        let a_x = arena.mul(&[a, var]);
        let quad = arena.add(&[x_sq, a_x, c]);
        // Both quadratics are positive on the whole line: no |·| in the ln.
        let ln_quad = arena.ln(quad);
        let half_lin = arena.mul(&[half, lin]);
        terms.push(arena.mul(&[half_lin, ln_quad]));
        let four_c = arena.mul(&[four, c]);
        let a_sq = arena.mul(&[a, a]);
        let k_sq = arena.sub(four_c, a_sq);
        let k = arena.sqrt(k_sq);
        let two_cst = arena.mul(&[two, cst]);
        let a_lin = arena.mul(&[a, lin]);
        let atan_coeff_num = arena.sub(two_cst, a_lin);
        let atan_coeff = arena.div(atan_coeff_num, k);
        let atan_arg_num = arena.add(&[two_x, a]);
        let atan_arg = arena.div(atan_arg_num, k);
        let atan_val = arena.atan(atan_arg);
        terms.push(arena.mul(&[atan_coeff, atan_val]));
    }
    if let Some(poly_part) = poly_part {
        terms.push(poly_part);
    }
    let result = arena.add(&terms);
    let result = crate::transforms::eval::eval(arena, result);
    // The closed form is exact by construction; the check guards the radical
    // arithmetic above (a wrong sign here would be a silent wrong answer).
    if candidate_rejected(arena, expr, result, var, var_sym) {
        tracing::debug!("integrate: biquadratic closed form failed verification");
        return None;
    }
    Some(result)
}

/// The two factors `(a, c, A, B)` of `N/(x⁴ + p x² + q)` = `Σ (A x + B)/(x² + a x + c)`
/// when `p² < 4q`: `c = √q`, `a = ∓√(2c − p)`, and from
/// `N = (A x + B)(x² + a x + c) + (C x + D)(x² − a x + c)`:
/// `A + C = n₃`, `a(A − C) + (B + D) = n₂`, `c(A + C) + a(B − D) = n₁`, `c(B + D) = n₀`.
fn biquadratic_conjugate_quadratics(
    arena: &mut Arena,
    p: &Q,
    q: &Q,
    n: &[ExprId; 4],
) -> SmallVec<[(ExprId, ExprId, ExprId, ExprId); 2]> {
    let p_id = arena.num_ratio(p.clone());
    let q_id = arena.num_ratio(q.clone());
    let two = arena.int(2);
    let half = arena.rational(1, 2);
    let c = arena.sqrt(q_id);
    let two_c = arena.mul(&[two, c]);
    let a_sq = arena.sub(two_c, p_id);
    let a = arena.sqrt(a_sq);
    let neg_a = arena.neg(a);

    let sum_bd = arena.div(n[0], c); // B + D = n₀/c
    let sum_ac = n[3]; // A + C = n₃
    let diff_ac = {
        let t = arena.sub(n[2], sum_bd);
        arena.div(t, a)
    }; // A − C = (n₂ − n₀/c)/a
    let diff_bd = {
        let c_n3 = arena.mul(&[c, n[3]]);
        let t = arena.sub(n[1], c_n3);
        arena.div(t, a)
    }; // B − D = (n₁ − c·n₃)/a
    let coeff_a = {
        let s = arena.add(&[sum_ac, diff_ac]);
        arena.mul(&[half, s])
    };
    let coeff_c = {
        let s = arena.sub(sum_ac, diff_ac);
        arena.mul(&[half, s])
    };
    let coeff_b = {
        let s = arena.add(&[sum_bd, diff_bd]);
        arena.mul(&[half, s])
    };
    let coeff_d = {
        let s = arena.sub(sum_bd, diff_bd);
        arena.mul(&[half, s])
    };
    // (A x + B)/(x² − a x + c) + (C x + D)/(x² + a x + c)
    let mut out = SmallVec::new();
    out.push((neg_a, c, coeff_a, coeff_b));
    out.push((a, c, coeff_c, coeff_d));
    out
}

/// The two factors `(0, r, A, B)` of `N/(x⁴ + p x² + q)` = `Σ (A x + B)/(x² + r)`
/// when `p² > 4q` and `p, q > 0`: `r₁,₂ = (p ∓ √disc)/2`, and from
/// `N = (A x + B)(x² + r₂) + (C x + D)(x² + r₁)`:
/// `A = (n₁ − n₃ r₁)/(r₂ − r₁)`, `C = n₃ − A`, `B = (n₀ − n₂ r₁)/(r₂ − r₁)`, `D = n₂ − B`.
fn biquadratic_even_quadratics(
    arena: &mut Arena,
    p: &Q,
    disc: &Q,
    n: &[ExprId; 4],
) -> SmallVec<[(ExprId, ExprId, ExprId, ExprId); 2]> {
    let p_id = arena.num_ratio(p.clone());
    let disc_id = arena.num_ratio(disc.clone());
    let half = arena.rational(1, 2);
    let sqrt_disc = arena.sqrt(disc_id);
    let r1 = {
        let t = arena.sub(p_id, sqrt_disc);
        arena.mul(&[half, t])
    };
    let r2 = {
        let t = arena.add(&[p_id, sqrt_disc]);
        arena.mul(&[half, t])
    };
    // r₂ − r₁ = √disc
    let coeff_a = {
        let n3_r1 = arena.mul(&[n[3], r1]);
        let t = arena.sub(n[1], n3_r1);
        arena.div(t, sqrt_disc)
    };
    let coeff_c = arena.sub(n[3], coeff_a);
    let coeff_b = {
        let n2_r1 = arena.mul(&[n[2], r1]);
        let t = arena.sub(n[0], n2_r1);
        arena.div(t, sqrt_disc)
    };
    let coeff_d = arena.sub(n[2], coeff_b);
    let zero = arena.zero;
    let mut out = SmallVec::new();
    out.push((zero, r1, coeff_a, coeff_b));
    out.push((zero, r2, coeff_c, coeff_d));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::arena::Arena;

    fn sym(a: &mut Arena, name: &str) -> ExprId {
        a.symbol(name)
    }

    fn display(a: &Arena, id: ExprId) -> String {
        a.display(id).to_string()
    }

    #[test]
    fn integrate_constant() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let five = a.int(5);
        let result = integrate(&mut a, five, x);
        assert_eq!(display(&a, result), "5*x");
    }

    #[test]
    fn integrate_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let result = integrate(&mut a, x, x);
        assert_eq!(display(&a, result), "1/2*x^2");
    }

    #[test]
    fn integrate_x_squared() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let result = integrate(&mut a, x2, x);
        assert_eq!(display(&a, result), "1/3*x^3");
    }

    #[test]
    fn integrate_x_inv() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let neg_one = a.int(-1);
        let x_inv = a.pow(x, neg_one);
        let result = integrate(&mut a, x_inv, x);
        assert_eq!(display(&a, result), "ln(abs(x))");
    }

    #[test]
    fn integrate_sin_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.sin(x);
        let result = integrate(&mut a, expr, x);
        assert_eq!(display(&a, result), "-cos(x)");
    }

    #[test]
    fn integrate_cos_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.cos(x);
        let result = integrate(&mut a, expr, x);
        assert_eq!(display(&a, result), "sin(x)");
    }

    #[test]
    fn integrate_exp_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.exp(x);
        let result = integrate(&mut a, expr, x);
        assert_eq!(display(&a, result), "exp(x)");
    }

    #[test]
    fn integrate_sum() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // ∫ (x + 1) dx = x^2/2 + x
        let one = a.one;
        let sum = a.add(&[x, one]);
        let result = integrate(&mut a, sum, x);
        let s = display(&a, result);
        assert!(s.contains("x^2"), "should contain x^2: {s}");
        assert!(s.contains("x"), "should contain x: {s}");
    }

    #[test]
    fn integrate_constant_times_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let three = a.int(3);
        let expr = a.mul(&[three, x]);
        let result = integrate(&mut a, expr, x);
        assert_eq!(display(&a, result), "3/2*x^2");
    }

    #[test]
    fn integrate_other_symbol_is_constant() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let result = integrate(&mut a, y, x);
        assert_eq!(display(&a, result), "x*y");
    }

    #[test]
    fn integrate_unevaluated_for_unknown() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // tan(x) is now integrable — verify we get the antiderivative
        let expr = a.tan(x);
        let result = integrate(&mut a, expr, x);
        assert_eq!(display(&a, result), "-ln(abs(cos(x)))");
    }

    #[test]
    fn integrate_x_sin_x_by_parts() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // ∫ x·sin(x) dx = sin(x) - x·cos(x)
        let sin_x = a.sin(x);
        let expr = a.mul(&[x, sin_x]);
        let result = integrate(&mut a, expr, x);
        let s = display(&a, result);
        // Should contain both sin(x) and cos(x) terms
        assert!(s.contains("sin(x)"), "should contain sin(x): {s}");
        assert!(s.contains("cos(x)"), "should contain cos(x): {s}");
    }

    #[test]
    fn integrate_x_exp_x_by_parts() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // ∫ x·exp(x) dx = x·exp(x) - exp(x) = (x-1)·exp(x)
        let exp_x = a.exp(x);
        let expr = a.mul(&[x, exp_x]);
        let result = integrate(&mut a, expr, x);
        let s = display(&a, result);
        assert!(s.contains("exp(x)"), "should contain exp(x): {s}");
    }

    #[test]
    fn integrate_x_cos_x_by_parts() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // ∫ x·cos(x) dx = x·sin(x) + cos(x)
        let cos_x = a.cos(x);
        let expr = a.mul(&[x, cos_x]);
        let result = integrate(&mut a, expr, x);
        let s = display(&a, result);
        assert!(s.contains("sin(x)"), "should contain sin(x): {s}");
        assert!(s.contains("cos(x)"), "should contain cos(x): {s}");
    }

    #[test]
    fn integrate_sin_2x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let two_x = a.mul(&[two, x]);
        let expr = a.sin(two_x);
        let result = integrate(&mut a, expr, x);
        let s = display(&a, result);
        // ∫ sin(2x) dx = -cos(2x)/2
        assert!(s.contains("cos"), "should contain cos: {s}");
        assert!(
            s.contains("1/2") || s.contains("2"),
            "should have factor of 1/2: {s}"
        );
    }

    #[test]
    fn integrate_cos_3x_plus_1() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let three = a.int(3);
        let one = a.one;
        let three_x = a.mul(&[three, x]);
        let inner = a.add(&[three_x, one]);
        let expr = a.cos(inner);
        let result = integrate(&mut a, expr, x);
        let s = display(&a, result);
        assert!(s.contains("sin"), "should contain sin: {s}");
    }

    #[test]
    fn integrate_exp_2x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let two_x = a.mul(&[two, x]);
        let expr = a.exp(two_x);
        let result = integrate(&mut a, expr, x);
        let s = display(&a, result);
        assert!(s.contains("exp"), "should contain exp: {s}");
    }

    // ═══════════════════════════════════════════════════════════════════
    // Sprint A – new tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn integrate_tan_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.tan(x);
        let result = integrate(&mut a, expr, x);
        assert_eq!(display(&a, result), "-ln(abs(cos(x)))");
    }

    #[test]
    fn integrate_ln_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.ln(x);
        let result = integrate(&mut a, expr, x);
        let s = display(&a, result);
        // ∫ ln(x) dx = x·ln(x) - x
        // Canonical form might be: -x + x*ln(x)  or  x*ln(x) + -x  etc.
        assert!(s.contains("ln(x)"), "should contain ln(x): {s}");
        assert!(s.contains("x"), "should contain x: {s}");
        // Verify both the x*ln(x) and the -x terms are present
        assert!(
            s.contains("x*ln(x)") || s.contains("ln(x)*x"),
            "should contain x*ln(x): {s}"
        );
    }

    #[test]
    fn integrate_atan_form() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // ∫ 1/(x²+1) dx = atan(x)
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let one = a.one;
        let base = a.add(&[x2, one]);
        let neg_one = a.int(-1);
        let expr = a.pow(base, neg_one);
        let result = integrate(&mut a, expr, x);
        let s = display(&a, result);
        assert!(s.contains("atan(x)"), "should be atan(x), got: {s}");
    }

    #[test]
    fn integrate_asin_form() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // ∫ 1/sqrt(1-x²) dx = (1-x²)^(-1/2) = asin(x)
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let one = a.one;
        let base = a.sub(one, x2); // 1 - x²
        let neg_half = a.rational(-1, 2);
        let expr = a.pow(base, neg_half);
        let result = integrate(&mut a, expr, x);
        let s = display(&a, result);
        assert!(s.contains("asin(x)"), "should be asin(x), got: {s}");
    }

    #[test]
    fn integrate_tan_2x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // ∫ tan(2x) dx = -ln|cos(2x)| / 2
        let two = a.int(2);
        let two_x = a.mul(&[two, x]);
        let expr = a.tan(two_x);
        let result = integrate(&mut a, expr, x);
        let s = display(&a, result);
        assert!(
            s.contains("ln") && s.contains("cos"),
            "should contain ln and cos: {s}"
        );
        assert!(
            s.contains("1/2") || s.contains("2"),
            "should have factor involving 2: {s}"
        );
    }

    #[test]
    fn integrate_ln_3x_plus_1() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // ∫ ln(3x+1) dx = ((3x+1)·ln(3x+1) - (3x+1)) / 3
        let three = a.int(3);
        let one = a.one;
        let three_x = a.mul(&[three, x]);
        let inner = a.add(&[three_x, one]);
        let expr = a.ln(inner);
        let result = integrate(&mut a, expr, x);
        let s = display(&a, result);
        assert!(s.contains("ln"), "should contain ln: {s}");
        assert!(!s.contains("Integral"), "should not be unevaluated: {s}");
    }

    #[test]
    fn integrate_tanh_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.tanh(x);
        let result = integrate(&mut a, expr, x);
        let s = display(&a, result);
        assert!(
            s.contains("ln") && s.contains("cosh"),
            "∫ tanh(x) dx should be ln(cosh(x)), got: {s}"
        );
    }

    #[test]
    fn integrate_tanh_2x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let two_x = a.mul(&[two, x]);
        let expr = a.tanh(two_x);
        let result = integrate(&mut a, expr, x);
        let s = display(&a, result);
        assert!(
            s.contains("ln") && s.contains("cosh"),
            "∫ tanh(2x) dx should involve ln(cosh(2x)), got: {s}"
        );
    }

    #[test]
    fn integrate_sinh_2x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let two_x = a.mul(&[two, x]);
        let expr = a.sinh(two_x);
        let result = integrate(&mut a, expr, x);
        let s = display(&a, result);
        // ∫ sinh(2x) dx = cosh(2x)/2
        assert!(
            s.contains("cosh"),
            "∫ sinh(2x) dx should involve cosh, got: {s}"
        );
    }

    #[test]
    fn integrate_cosh_3x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let three = a.int(3);
        let three_x = a.mul(&[three, x]);
        let expr = a.cosh(three_x);
        let result = integrate(&mut a, expr, x);
        let s = display(&a, result);
        // ∫ cosh(3x) dx = sinh(3x)/3
        assert!(
            s.contains("sinh"),
            "∫ cosh(3x) dx should involve sinh, got: {s}"
        );
    }

    #[test]
    fn integrate_2x_plus_1_cubed() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let one = a.one;
        let two_x = a.mul(&[two, x]);
        let inner = a.add(&[two_x, one]);
        let three = a.int(3);
        let expr = a.pow(inner, three);
        let result = integrate(&mut a, expr, x);
        let s = display(&a, result);
        // ∫ (2x+1)^3 dx = (2x+1)^4 / 8
        assert!(!s.contains("Integral"), "should not be unevaluated: {s}");
    }

    #[test]
    fn integrate_x_plus_1_squared() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.one;
        let inner = a.add(&[x, one]);
        let two = a.int(2);
        let expr = a.pow(inner, two);
        let result = integrate(&mut a, expr, x);
        let s = display(&a, result);
        // ∫ (x+1)^2 dx should be resolved (linear sub or expand)
        assert!(!s.contains("Integral"), "should not be unevaluated: {s}");
    }

    #[test]
    fn integrate_asin_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.asin(x);
        let result = integrate(&mut a, expr, x);
        let s = display(&a, result);
        assert!(s.contains("asin"), "should contain asin: {s}");
    }

    #[test]
    fn integrate_atan_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.atan(x);
        let result = integrate(&mut a, expr, x);
        let s = display(&a, result);
        assert!(s.contains("atan"), "should contain atan: {s}");
    }

    #[test]
    fn integrate_acos_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.acos(x);
        let result = integrate(&mut a, expr, x);
        let s = display(&a, result);
        assert!(s.contains("acos"), "should contain acos: {s}");
    }

    // ═══════════════════════════════════════════════════════════════════
    // u-substitution tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn integrate_2x_exp_x_squared_u_sub() {
        // ∫ 2x·exp(x²) dx = exp(x²)   [u = x², du = 2x dx]
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let two_x = a.mul(&[two, x]);
        let x2 = a.pow(x, two);
        let exp_x2 = a.exp(x2);
        let expr = a.mul(&[two_x, exp_x2]);
        let result = integrate(&mut a, expr, x);
        let s = display(&a, result);
        assert!(
            s.contains("exp"),
            "∫ 2x·exp(x²) dx should contain exp, got: {s}"
        );
        assert!(
            !s.contains("Integral"),
            "∫ 2x·exp(x²) dx should not be unevaluated, got: {s}"
        );
    }

    #[test]
    fn integrate_cos_x_exp_sin_x_u_sub() {
        // ∫ cos(x)·exp(sin(x)) dx = exp(sin(x))   [u = sin(x), du = cos(x) dx]
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let sin_x = a.sin(x);
        let cos_x = a.cos(x);
        let exp_sin_x = a.exp(sin_x);
        let expr = a.mul(&[cos_x, exp_sin_x]);
        let result = integrate(&mut a, expr, x);
        let s = display(&a, result);
        assert!(
            s.contains("exp"),
            "∫ cos(x)·exp(sin(x)) dx should contain exp, got: {s}"
        );
        assert!(
            !s.contains("Integral"),
            "∫ cos(x)·exp(sin(x)) dx should not be unevaluated, got: {s}"
        );
    }

    #[test]
    fn integrate_x_over_x2_plus_1_u_sub() {
        // ∫ x/(x²+1) dx = ½·ln(x²+1)   [u = x²+1, du = 2x dx]
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let one = a.one;
        let x2_plus_1 = a.add(&[x2, one]);
        let expr = a.div(x, x2_plus_1);
        let result = integrate(&mut a, expr, x);
        let s = display(&a, result);
        assert!(
            s.contains("ln"),
            "∫ x/(x²+1) dx should contain ln, got: {s}"
        );
        assert!(
            !s.contains("Integral"),
            "∫ x/(x²+1) dx should not be unevaluated, got: {s}"
        );
    }

    #[test]
    fn integrate_complete_square() {
        // ∫ 1/(x²+2x+5) dx — should use completing the square
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let five = a.int(5);
        let x2 = a.pow(x, two);
        let two_x = a.mul(&[two, x]);
        let quadratic = a.add(&[x2, two_x, five]);
        let neg_one = a.int(-1);
        let integrand = a.pow(quadratic, neg_one);
        let result = integrate(&mut a, integrand, x);
        let s = display(&a, result);
        assert!(s.contains("atan"), "should use atan: {s}");
    }

    #[test]
    fn integrate_complete_square_simple() {
        // ∫ 1/(x²+x+1) dx
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let one = a.one;
        let quadratic = a.add(&[x2, x, one]);
        let neg_one = a.int(-1);
        let integrand = a.pow(quadratic, neg_one);
        let result = integrate(&mut a, integrand, x);
        let s = display(&a, result);
        assert!(s.contains("atan"), "should use atan: {s}");
    }

    #[test]
    fn symbolic_linear_coeff_of_mul_a_x() {
        // a*x should be detected as linear in x with coefficient a
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let param_a = sym(&mut a, "a");
        let ax = a.mul(&[param_a, x]);
        let var_sym = match a.node(x) {
            ExprNode::Symbol(sid) => *sid,
            _ => panic!("x should be a symbol"),
        };
        let result = super::symbolic_linear_coeff_of(&mut a, ax, x, var_sym);
        assert!(
            result.is_some(),
            "a*x should be recognized as linear in x, node: {:?}",
            a.node(ax)
        );
        let (coeff, constant) = result.unwrap();
        assert_eq!(
            coeff,
            param_a,
            "coefficient should be a, got {}",
            display(&a, coeff)
        );
        assert_eq!(
            constant,
            a.zero,
            "constant should be 0, got {}",
            display(&a, constant)
        );
    }

    #[test]
    fn symbolic_linear_coeff_of_add_ax_b() {
        // a*x + b should be detected as linear in x with coefficient a, constant b
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let param_a = sym(&mut a, "a");
        let param_b = sym(&mut a, "b");
        let ax = a.mul(&[param_a, x]);
        let ax_plus_b = a.add(&[ax, param_b]);
        let var_sym = match a.node(x) {
            ExprNode::Symbol(sid) => *sid,
            _ => panic!("x should be a symbol"),
        };
        let result = super::symbolic_linear_coeff_of(&mut a, ax_plus_b, x, var_sym);
        assert!(
            result.is_some(),
            "a*x+b should be recognized as linear in x, expr: {}",
            display(&a, ax_plus_b)
        );
        let (coeff, constant) = result.unwrap();
        assert_eq!(
            coeff,
            param_a,
            "coefficient should be a, got {}",
            display(&a, coeff)
        );
        assert_eq!(
            constant,
            param_b,
            "constant should be b, got {}",
            display(&a, constant)
        );
    }

    #[test]
    fn symbolic_linear_coeff_of_bare_var() {
        // x alone should be linear with coefficient 1
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let var_sym = match a.node(x) {
            ExprNode::Symbol(sid) => *sid,
            _ => panic!("x should be a symbol"),
        };
        let result = super::symbolic_linear_coeff_of(&mut a, x, x, var_sym);
        assert!(result.is_some(), "x should be recognized as linear in x");
        let (coeff, constant) = result.unwrap();
        assert_eq!(coeff, a.one, "coefficient should be 1");
        assert_eq!(constant, a.zero, "constant should be 0");
    }

    #[test]
    fn integrate_sin_symbolic_coeff() {
        // ∫ sin(a*x) dx should not be unevaluated
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let param_a = sym(&mut a, "a");
        let ax = a.mul(&[param_a, x]);
        let sin_ax = a.sin(ax);
        let result = integrate(&mut a, sin_ax, x);
        let s = display(&a, result);
        assert!(
            !s.contains("Integral"),
            "∫sin(a*x)dx should not be unevaluated: {s}"
        );
        assert!(s.contains("cos"), "should contain cos: {s}");
    }

    #[test]
    fn integrate_exp_symbolic_coeff() {
        // ∫ exp(a*x) dx should not be unevaluated
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let param_a = sym(&mut a, "a");
        let ax = a.mul(&[param_a, x]);
        let exp_ax = a.exp(ax);
        let result = integrate(&mut a, exp_ax, x);
        let s = display(&a, result);
        assert!(
            !s.contains("Integral"),
            "∫exp(a*x)dx should not be unevaluated: {s}"
        );
        assert!(s.contains("exp"), "should contain exp: {s}");
    }

    #[test]
    fn integrate_cosh_symbolic_coeff() {
        // ∫ cosh(a*x) dx should not be unevaluated
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let param_a = sym(&mut a, "a");
        let ax = a.mul(&[param_a, x]);
        let cosh_ax = a.cosh(ax);
        let result = integrate(&mut a, cosh_ax, x);
        let s = display(&a, result);
        assert!(
            !s.contains("Integral"),
            "∫cosh(a*x)dx should not be unevaluated: {s}"
        );
    }

    // ── Inverse hyperbolic integration tests ───────────────────────

    /// Helper: substitute a rational value for a symbol and evaluate to f64.
    /// Returns None if evaluation fails.
    fn eval_at(
        arena: &mut Arena,
        expr: ExprId,
        var: ExprId,
        numer: i64,
        denom: i64,
    ) -> Option<f64> {
        let val = arena.rational(numer, denom);
        let substituted = crate::transforms::subs::subs(arena, expr, var, val);
        let evaled = crate::transforms::eval::eval(arena, substituted);
        let s = crate::transforms::evalf::evalf(arena, evaled, 15).ok()?;
        s.parse::<f64>().ok()
    }

    /// Helper: verify FTC at multiple points — d/dx(F(x)) ≈ f(x).
    /// `integrand` is f(x), `antideriv` is F(x) = ∫f(x)dx.
    /// Checks at each test point that |F'(point) - f(point)| < tol.
    fn assert_ftc(
        arena: &mut Arena,
        integrand: ExprId,
        antideriv: ExprId,
        var: ExprId,
        test_points: &[(i64, i64)],
        tol: f64,
        name: &str,
    ) {
        let deriv = crate::transforms::diff::diff(arena, antideriv, var);
        let deriv_simplified = crate::simplify::simplify_engine::smart_simplify(arena, deriv);
        for &(n, d) in test_points {
            let f_val = eval_at(arena, integrand, var, n, d);
            let fp_val = eval_at(arena, deriv_simplified, var, n, d);
            match (f_val, fp_val) {
                (Some(f), Some(fp)) => {
                    assert!(
                        (f - fp).abs() < tol,
                        "FTC failed for {name} at x={n}/{d}: f(x)={f}, F'(x)={fp}, diff={}",
                        (f - fp).abs()
                    );
                }
                _ => {
                    // If numerical eval fails at this point, skip it
                    // (e.g., acosh at x < 1 is undefined)
                }
            }
        }
    }

    #[test]
    fn integrate_asinh_direct() {
        // ∫ asinh(x) dx = x·asinh(x) - √(x²+1)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let asinh_x = a.asinh(x);
        let result = integrate(&mut a, asinh_x, x);

        // Structural: must not be unevaluated
        assert!(
            !matches!(a.node(result), ExprNode::Integral(_, _)),
            "asinh integration should return a closed form, not Integral"
        );
        let s = display(&a, result);
        assert!(s.contains("asinh"), "result should contain asinh: {s}");
        assert!(s.contains("sqrt"), "result should contain sqrt: {s}");

        // Numerical FTC: d/dx(result) ≈ asinh(x) at multiple points
        assert_ftc(
            &mut a,
            asinh_x,
            result,
            x,
            &[(1, 2), (3, 2), (5, 1)],
            1e-8,
            "∫asinh(x)dx",
        );
    }

    #[test]
    fn integrate_acosh_direct() {
        // ∫ acosh(x) dx = x·acosh(x) - √(x²-1)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let acosh_x = a.acosh(x);
        let result = integrate(&mut a, acosh_x, x);

        // Structural: must not be unevaluated
        assert!(
            !matches!(a.node(result), ExprNode::Integral(_, _)),
            "acosh integration should return a closed form, not Integral"
        );
        let s = display(&a, result);
        assert!(s.contains("acosh"), "result should contain acosh: {s}");

        // Numerical FTC: test at x > 1 only (acosh domain)
        assert_ftc(
            &mut a,
            acosh_x,
            result,
            x,
            &[(3, 2), (2, 1), (5, 1)],
            1e-8,
            "∫acosh(x)dx",
        );
    }

    #[test]
    fn integrate_atanh_direct() {
        // ∫ atanh(x) dx = x·atanh(x) + ½·ln(1-x²)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let atanh_x = a.atanh(x);
        let result = integrate(&mut a, atanh_x, x);

        // Structural: must not be unevaluated
        assert!(
            !matches!(a.node(result), ExprNode::Integral(_, _)),
            "atanh integration should return a closed form, not Integral"
        );
        let s = display(&a, result);
        assert!(s.contains("atanh"), "result should contain atanh: {s}");
        assert!(s.contains("ln"), "result should contain ln: {s}");

        // Numerical FTC: test at |x| < 1 only (atanh domain)
        assert_ftc(
            &mut a,
            atanh_x,
            result,
            x,
            &[(1, 4), (1, 2), (3, 4)],
            1e-8,
            "∫atanh(x)dx",
        );
    }

    /// [`check_antiderivative`] of `big_f` for `∫ f dx`, both parsed; `K`
    /// in either stands for `sin(1)² + cos(1)² − 1`, which is 0 without
    /// being so structurally.
    fn ftc_verdict(f: &str, big_f: &str) -> FtcVerdict {
        let k = "(sin(1)^2 + cos(1)^2 - 1)";
        let ctx = crate::api::context::Context::new();
        let f = ctx.parse(&f.replace('K', k)).unwrap().id();
        let big_f = ctx.parse(&big_f.replace('K', k)).unwrap().id();
        ctx.with_arena_mut(|a| {
            let x = a.symbol("x");
            let ExprNode::Symbol(x_sym) = *a.node(x) else {
                unreachable!()
            };
            check_antiderivative(a, f, big_f, x, x_sym)
        })
    }

    #[test]
    fn ftc_check_vanishing_denominator_is_undecided_not_passed() {
        // Up to 0.28.0 no sample point evaluated `F′` (a quotient by K = 0)
        // and "no evidence" counted as "not wrong".
        assert_eq!(
            ftc_verdict("x/(K*x + 1)", "(x - ln(abs(K*x + 1))/K)/K"),
            FtcVerdict::Undecided
        );
        // `F′ = x·e^{Kx}` exactly, but `F` is undefined everywhere.
        assert_eq!(
            ftc_verdict("x*exp(K*x)", "x*exp(K*x)/K - exp(K*x)/K^2"),
            FtcVerdict::Undecided
        );
        // `f` itself evaluates nowhere: nothing can be tested.
        assert_eq!(ftc_verdict("x/K", "x^2/(2*K)"), FtcVerdict::Untestable);
    }

    #[test]
    fn ftc_check_samples_parameters() {
        // Up to 0.28.0 a free parameter skipped the check.  Up to 0.30
        // `ln|x + a|` was verified here too (the parameter took real values
        // only); an unassumed `a` is complex (decision D4, 0.31), and the
        // complex round rejects it.
        assert_eq!(ftc_verdict("1/(x + a)", "ln(x + a)"), FtcVerdict::Verified);
        assert_eq!(
            ftc_verdict("1/(x + a)", "ln(abs(x + a))"),
            FtcVerdict::Wrong
        );
        assert_eq!(
            ftc_verdict("1/(x + a)", "ln(abs(x + a))/a"),
            FtcVerdict::Wrong
        );
    }

    #[test]
    fn ftc_check_complex_integrand() {
        // `c = √(1/2 − √5/2)` is imaginary: `f` is complex at every real
        // `x`, the principal `ln(x − c)` agrees with it, `ln|x − c|` does
        // not (its derivative is `Re f`).
        let f = "1/(x - sqrt(1/2 - sqrt(5)/2))";
        assert_eq!(
            ftc_verdict(f, "ln(x - sqrt(1/2 - sqrt(5)/2))"),
            FtcVerdict::Verified
        );
        assert_eq!(
            ftc_verdict(f, "ln(abs(x - sqrt(1/2 - sqrt(5)/2)))"),
            FtcVerdict::Undecided
        );
        // Real-variable convention: `ln|x|` differs from `1/x` nowhere
        // where it is real; `√x·x` against `(2/5)·x^{5/2}` agrees at
        // `x > 0` and in the principal continuation.
        assert_eq!(ftc_verdict("1/x", "ln(abs(x))"), FtcVerdict::Verified);
        assert_eq!(ftc_verdict("1/x", "ln(abs(x)) + x"), FtcVerdict::Wrong);
        assert_eq!(
            ftc_verdict("sqrt(x - 5)", "2/3*(x - 5)^(3/2)"),
            FtcVerdict::Verified
        );
    }

    #[test]
    fn ftc_check_samples_through_binders() {
        // `subs` alone turns `RootOf(x⁵ − x + 1, 1)` into `RootOf(163/243,
        // 1)` at `x = 1/3`, which does not evaluate; the check renames the
        // bound `x` first.
        let f = "1/(x - RootOf(x^5 - x + 1, 1))";
        assert_eq!(
            ftc_verdict(f, "ln(x - RootOf(x^5 - x + 1, 1))"),
            FtcVerdict::Verified
        );
        assert_eq!(
            ftc_verdict(f, "2*ln(x - RootOf(x^5 - x + 1, 1))"),
            FtcVerdict::Undecided
        );
    }

    #[test]
    fn contains_var_respects_binders() {
        let ctx = crate::api::context::Context::new();
        let cases = [
            ("RootOf(x^5 - x + 1, 0)", false),
            ("x - RootOf(x^5 - x + 1, 0)", true),
            // A parametric `RootOf` binds its variable only (up to 0.28 it
            // named none and bound nothing).
            ("RootOf(x^3 + a*x + 1, x, 0)", false),
            ("RootOf(x^3 + a*x + 1, a, 0)", true),
            ("Sum(x^k, k, 0, 3)", true),
            ("Sum(k*x, x, 0, 3)", false),
            ("Sum(k*x, x, 0, x)", true),
            ("Integral(x, x)", true),
        ];
        for (src, expected) in cases {
            let e = ctx.parse(src).unwrap().id();
            let got = ctx.with_arena_mut(|a| {
                let x = a.symbol("x");
                let ExprNode::Symbol(x_sym) = *a.node(x) else {
                    unreachable!()
                };
                contains_var(a, e, x_sym)
            });
            assert_eq!(got, expected, "contains_var({src}, x)");
        }
    }

    #[test]
    fn constant_realness_is_exact_or_certified() {
        let ctx = crate::api::context::Context::new();
        // Decision D4 (0.31): `a`, `b` are unassumed, hence complex
        // parameters (`a*b + 3` counted as real up to 0.30); `r`, `s` are
        // declared real.
        ctx.symbol_with("r", &[crate::base::assumptions::Assumption::Real])
            .unwrap();
        ctx.symbol_with("s", &[crate::base::assumptions::Assumption::Positive])
            .unwrap();
        let cases = [
            ("sqrt(2)", true),
            ("a*b + 3", false),
            ("abs(a) + 3", true),
            ("r*s + 3", true),
            ("ln(3)", true),
            ("sqrt(3 - sqrt(5))", true),
            ("sqrt(1/2 - sqrt(5)/2)", false),
            ("exp(pi*I/8)", false),
            ("RootOf(x^5 - x + 1, 1)", false),
            // A real parameter of unknown sign.
            ("sqrt(r)", false),
            ("sqrt(-r)", false),
            ("sqrt(s)", true),
            ("sqrt(a)", false),
        ];
        for (src, expected) in cases {
            let e = ctx.parse(src).unwrap().id();
            let got = ctx.with_arena_mut(|a| {
                let mut reals = crate::base::assumptions::AssumptionCache::new();
                constant_is_real(a, e, &mut reals)
            });
            assert_eq!(got, expected, "constant_is_real({src})");
        }
    }

    /// Which candidates the complex round of the self-check can tell apart
    /// from their real-parameter look-alikes: those with a non-analytic
    /// function of a parameter (`|a·x + 1|`, `sign(a)`); `|x|` alone, and
    /// analytic functions of the parameters, cannot differ there.
    #[test]
    fn analytic_in_parameters_finds_real_parameter_forms() {
        let ctx = crate::api::context::Context::new();
        for (src, analytic) in [
            ("ln(a*x + 1)/a", true),
            ("ln(abs(x)) + a*x", true),
            ("atan(x/sqrt(a))/sqrt(a) + exp(b*x)*sin(x)", true),
            ("ln(abs(a*x + 1))/a", false),
            ("sign(a)*x", false),
            ("abs(a)*x^2/2", false),
            ("re(a)*x", false),
        ] {
            let e = ctx.parse(src).unwrap().id();
            let got = ctx.with_arena_mut(|a| {
                let x = a.symbol("x");
                let ExprNode::Symbol(x_sym) = *a.node(x) else {
                    unreachable!()
                };
                analytic_in_parameters(a, e, x_sym)
            });
            assert_eq!(got, analytic, "{src}");
        }
    }

    /// The verifier's complex round (decision D4, 0.31).  Up to 0.30 the
    /// parameters were bound to positive rationals only, where
    /// `ln|a·x + 1|/a` passes as an antiderivative of `1/(a·x + 1)`; and a
    /// declared parameter made every candidate untestable.  `r` is declared
    /// declared real, `n` an integer (sampled at the integer `17` since 0.47;
    /// up to 0.46 no generic value satisfied that and every candidate with
    /// an integer parameter was untestable), `z` declared complex (which
    /// does not make it real), `m` negative.
    #[test]
    fn ftc_check_binds_unassumed_parameters_to_complex_values() {
        use crate::base::assumptions::Assumption;
        let ctx = crate::api::context::Context::new();
        ctx.symbol_with("r", &[Assumption::Real]).unwrap();
        ctx.symbol_with("n", &[Assumption::Integer]).unwrap();
        ctx.symbol_with("z", &[Assumption::Complex]).unwrap();
        ctx.symbol_with("m", &[Assumption::Negative]).unwrap();
        let cases = [
            ("1/(z*x + 1)", "ln(abs(z*x + 1))/z", FtcVerdict::Wrong),
            ("1/(z*x + 1)", "ln(z*x + 1)/z", FtcVerdict::Verified),
            ("1/(x - m)", "ln(abs(x - m))", FtcVerdict::Verified),
            // Wrong for a negative `m`, right for a positive one: up to 0.30
            // untestable, now sampled at `m = −17/13`.
            ("sqrt(m^2)", "m*x", FtcVerdict::Wrong),
            ("1/(a*x + 1)", "ln(abs(a*x + 1))/a", FtcVerdict::Wrong),
            ("1/(a*x + 1)", "ln(a*x + 1)/a", FtcVerdict::Verified),
            ("1/(r*x + 1)", "ln(abs(r*x + 1))/r", FtcVerdict::Verified),
            ("1/(x - 2)", "ln(abs(x - 2))", FtcVerdict::Verified),
            ("a*tan(x)", "-a*ln(abs(cos(x)))", FtcVerdict::Verified),
            // A part that depends on `x` alone keeps the real-variable
            // convention whatever its coefficient.
            (
                "a/sqrt(x^2 - 1)",
                "a*ln(abs(x + sqrt(x^2 - 1)))",
                FtcVerdict::Verified,
            ),
            ("a", "abs(a)*x", FtcVerdict::Wrong),
            ("exp(a*x)", "exp(a*x)/a", FtcVerdict::Verified),
            ("x^n", "x^(n + 1)/(n + 1)", FtcVerdict::Verified),
            ("x^n", "x^(n + 1)/n", FtcVerdict::Wrong),
            // A quotient by `n − |n|`, `0/0` at every positive integer `n`.
            (
                "sin(n*x)*sin(abs(n)*x)",
                "sin((n - abs(n))*x)/(2*(n - abs(n))) - sin((n + abs(n))*x)/(2*(n + abs(n)))",
                FtcVerdict::Wrong,
            ),
        ];
        for (f_src, big_f_src, expected) in cases {
            let f = ctx.parse(f_src).unwrap().id();
            let big_f = ctx.parse(big_f_src).unwrap().id();
            let got = ctx.with_arena_mut(|a| {
                let x = a.symbol("x");
                let ExprNode::Symbol(x_sym) = *a.node(x) else {
                    unreachable!()
                };
                check_antiderivative(a, f, big_f, x, x_sym)
            });
            assert_eq!(got, expected, "∫ {f_src} = {big_f_src}");
        }
    }

    /// `single_argument` and `analytic_in_parameters` select exactly the
    /// node-backed functions of their 0.49 or-patterns (W6d: the lists
    /// became `by_parts_unary` and `analytic_off_cuts`).
    #[test]
    fn function_predicates_are_the_old_lists() {
        use crate::base::func::{Func::*, selected};
        let parts = [
            Ei, Si, Ci, Li, Asin, Acos, Atan, Asinh, Acosh, Atanh, Erf, Erfc, Ln,
        ];
        assert_eq!(selected(by_parts_unary), parts.into_iter().collect());
        let analytic = [
            Sin, Cos, Tan, Exp, Ln, Asin, Acos, Atan, Sinh, Cosh, Tanh, Asinh, Acosh, Atanh, Gamma,
            Erf, Erfc, Si, Ci, Ei, Li, Zeta,
        ];
        assert_eq!(selected(analytic_off_cuts), analytic.into_iter().collect());
    }
}
