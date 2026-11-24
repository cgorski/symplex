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

mod algebraic;
mod biquadratic;
mod mobius;
mod piecewise;
mod products;
mod rational_constants;
mod special;
mod standard_forms;
mod substitution;
mod trig_rational;
mod u_substitution;
mod verify;
use algebraic::*;
use biquadratic::*;
use mobius::*;
use piecewise::*;
use products::*;
use rational_constants::*;
use special::*;
use standard_forms::*;
pub(crate) use substitution::*;
use trig_rational::*;
use u_substitution::*;
use verify::*;

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
    let result = if let ExprNode::Integral(_, _) = arena.node(result) {
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
    };

    // Last: a rational function over constants, as parameters (only where
    // every other stage left the integral, so their answers keep their form).
    if let ExprNode::Integral(_, _) = arena.node(result)
        && let Some(r) = stage!(
            arena,
            "rational_over_constants",
            expr,
            try_rational_over_constants(arena, expr, var, var_sym)
        )
    {
        return accept_or_unevaluated(arena, expr, r, var, var_sym);
    }
    result
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
/// An argument with an infinity in it is not linear: up to 0.53 `∫
/// besselj(1, x·∞) dx` and `∫ chebyshev_t(3, x·∞) dx` were `0` (an
/// antiderivative divided by the coefficient `∞`).
fn symbolic_linear_coeff_of(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<(ExprId, ExprId)> {
    let special = [
        arena.infinity,
        arena.neg_infinity,
        arena.complex_infinity,
        arena.nan,
    ];
    if special
        .iter()
        .any(|&s| crate::base::walk::contains(arena, expr, s))
    {
        return None;
    }
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
        crate::transforms::evalf::const_value(arena, evaled, 15)
            .ok()?
            .to_f64()
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
