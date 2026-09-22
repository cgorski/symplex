//! Inequality solving via the sign-chart method.
//!
//! # Algorithm
//!
//! Given `f(x) rel 0` (`rel` one of `>`, `>=`, `<`, `<=`):
//!
//! 1. Collect the **critical points** of `f` on the real line: the real
//!    zeros of the numerator and of the denominator of `f` written over a
//!    common denominator, the boundary points of the natural domain (zeros
//!    of radicands of even/fractional powers, of `ln` arguments, …), and
//!    the "kinks" of piecewise-defined pieces (`|·|`, `sign`, `Heaviside`,
//!    `min`/`max`, `Piecewise`).
//! 2. Order them exactly (rationals) or by a 30-digit approximation
//!    (algebraic / transcendental endpoints).
//! 3. Test the sign of `f` on every open interval between consecutive
//!    critical points at a **rational** sample point, exactly whenever the
//!    value is rational, and decide each critical point individually by
//!    evaluating `f` there (a point where `f` is undefined or not real is
//!    never part of the solution).
//! 4. Assemble the union of intervals / isolated points.
//!
//! Anything that cannot be decided exactly or to 30 digits is reported as
//! an error instead of guessing — the caller (`Ex::solve_gt` & co.) then
//! returns an unevaluated `ConditionSet`.

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Signed, Zero};
use rustc_hash::FxHashMap;

use crate::base::arena::Arena;
use crate::base::errors::SymplexError;
use crate::base::node::{
    ExprId, ExprNode, INTERVAL_BOTH_CLOSED, INTERVAL_BOTH_OPEN, INTERVAL_LEFT_OPEN,
    INTERVAL_RIGHT_OPEN,
};
use crate::base::numeric::Q;
use crate::base::walk;

/// Relation type for inequalities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Relation {
    /// Strictly greater than (`> 0`).
    Gt,
    /// Greater than or equal (`>= 0`).
    Ge,
    /// Strictly less than (`< 0`).
    Lt,
    /// Less than or equal (`<= 0`).
    Le,
}

impl Relation {
    /// Does a value with the given sign satisfy `value rel 0`?
    fn accepts(self, sign: std::cmp::Ordering) -> bool {
        use std::cmp::Ordering::*;
        matches!(
            (self, sign),
            (Relation::Gt, Greater)
                | (Relation::Lt, Less)
                | (Relation::Ge, Greater | Equal)
                | (Relation::Le, Less | Equal)
        )
    }
}

/// Decimal digits used for every numerical decision (sign of a value,
/// ordering of endpoints).
const DECISION_DIGITS: u32 = 30;

/// A numerical real part below `10^-AMBIGUOUS_EXP` is *not* trusted to be
/// nonzero without confirmation at higher precision (the value may be an
/// exact zero that radicals failed to cancel).
const AMBIGUOUS_EXP: usize = 24;

/// An imaginary part above `10^-NONREAL_EXP` (relative) makes a value
/// non-real.
const NONREAL_EXP: usize = 20;

/// Two critical points whose 30-digit approximations differ by less than
/// this (relative) are treated as the same point.
const MERGE_REL: f64 = 1e-24;

/// Upper bound on the number of critical points handled.
const MAX_CRITICAL_POINTS: usize = 256;

/// Bound on the number of nested branch-resolution passes for piecewise
/// pieces inside piecewise pieces.
const MAX_BRANCH_PASSES: usize = 8;

/// Solve `expr rel 0` for `var`, returning the solution as a set `ExprId`.
///
/// The result is a union of intervals (and possibly isolated points for
/// non-strict inequalities) that encodes the solution set on the real
/// line, restricted to the natural real domain of `expr`.
pub(crate) fn solve_inequality(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    rel: Relation,
) -> Result<ExprId, SymplexError> {
    tracing::debug!("solve_inequality: rel={:?}", rel);

    // ── Absolute-value fast path: c·|a·x + b| + d rel 0 → interval / union
    //    (also handles symbolic endpoints). ──
    if let Some(set) = try_solve_abs_inequality(arena, expr, var, rel) {
        return Ok(set);
    }

    // ── Sturm fast path: if the expression is polynomial, use Sturm
    //    chains to detect the no-real-roots case without solving. ──
    if let Some(poly) = crate::poly::polybridge::expr_to_poly(arena, expr, var) {
        let chain = crate::poly::sturm::SturmChain::new(&poly);
        if chain.has_no_real_roots() {
            // The polynomial has no real roots ⇒ constant sign on ℝ.
            let ls = chain.leading_sign_of_original();
            let sign = ls.cmp(&0);
            return if rel.accepts(sign) {
                Ok(arena.interval(arena.neg_infinity, arena.infinity, INTERVAL_BOTH_OPEN))
            } else {
                Ok(arena.empty_set)
            };
        }
    }

    sign_chart(arena, expr, var, rel)
}

/// Solve `expr = 0`, returning the solution set as an `ExprId`.
///
/// - Identity `0 = 0` → `UniversalSet`
/// - Contradiction / no roots found → `EmptySet`
/// - Otherwise a `FiniteSet` of the roots
pub(crate) fn solveset(arena: &mut Arena, expr: ExprId, var: ExprId) -> ExprId {
    use crate::transforms::solve::SolveOutcome;
    match crate::transforms::solve::solve_classified(arena, expr, var) {
        SolveOutcome::Identity => arena.universal_set,
        SolveOutcome::NoSolution(_) => arena.empty_set,
        SolveOutcome::Solutions(solutions) => {
            let root_ids: Vec<ExprId> = solutions.into_iter().map(|s| s.value).collect();
            if root_ids.is_empty() {
                arena.empty_set
            } else {
                arena.finite_set(&root_ids)
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Sign chart
// ═══════════════════════════════════════════════════════════════════════════

/// A critical point of the sign chart.
#[derive(Clone, Debug)]
struct Critical {
    /// Exact expression for the point (used as an interval endpoint).
    id: ExprId,
    /// Exact value when rational, otherwise a 30-digit approximation.
    approx: Q,
    /// Is `approx` exact?
    exact: bool,
    /// The point is a zero of the numerator of (a branch of) `expr`.
    numer_zero: bool,
    /// `expr` is undefined at the point (zero of a denominator or of the
    /// argument of `ln`, …).
    undefined: bool,
}

/// What a domain constraint demands of its expression.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ConstraintKind {
    /// `e ≠ 0` (a denominator).
    NonZero,
    /// `e ≥ 0` (radicand of a fractional power).
    NonNegative,
    /// `e > 0` (`ln` argument, base of a variable power, …).
    Positive,
}

#[derive(Clone, Debug)]
struct Constraint {
    expr: ExprId,
    kind: ConstraintKind,
}

/// One smooth branch of `expr`, valid on an open interval between two
/// consecutive kink points (or on the whole line when `expr` has no
/// piecewise-defined pieces).
#[derive(Clone, Debug)]
struct Branch {
    /// `expr` with every piecewise-defined node replaced by the branch
    /// that is active on this interval.
    resolved: ExprId,
    /// Natural-domain constraints of `resolved`.
    constraints: Vec<Constraint>,
}

/// Classification of the value of a variable-free expression.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ValueSign {
    Neg,
    Zero,
    Pos,
    /// Finite but with a nonzero imaginary part.
    NonReal,
    /// Infinite / NaN / boolean.
    Undefined,
    /// Symbolic, unevaluable, or numerically indistinguishable from zero.
    Unknown,
}

impl ValueSign {
    fn ordering(self) -> Option<std::cmp::Ordering> {
        match self {
            ValueSign::Neg => Some(std::cmp::Ordering::Less),
            ValueSign::Zero => Some(std::cmp::Ordering::Equal),
            ValueSign::Pos => Some(std::cmp::Ordering::Greater),
            _ => None,
        }
    }
}

fn failed(reason: String) -> SymplexError {
    SymplexError::computation_failed("solve_inequality", reason)
}

/// The general sign-chart solver (see the module docs).
fn sign_chart(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    rel: Relation,
) -> Result<ExprId, SymplexError> {
    check_supported(arena, expr, var)?;
    let expr = crate::transforms::eval::eval(arena, expr);

    // Phase 1: kink points of piecewise-defined pieces (zeros *and* poles
    // of the switching expressions: `min(x, 1/x)` switches at 0 too).
    let kinks = kink_expressions(arena, expr, var);
    let mut points: Vec<Critical> = Vec::new();
    for &k in &kinks {
        let (kn, kd) = numer_denom(arena, k);
        if let RootSet::Finite(roots) = real_roots(arena, kn, var)? {
            add_roots(arena, &mut points, &roots, false, false)?;
        }
        if kd != arena.one
            && let RootSet::Finite(roots) = real_roots(arena, kd, var)?
        {
            add_roots(arena, &mut points, &roots, false, false)?;
        }
    }
    sort_points(&mut points);
    let kink_points: Vec<Critical> = points.clone();

    // Phase 2: one smooth branch per kink interval; its numerator /
    // denominator zeros and domain boundaries become critical points.
    let has_piecewise = !kinks.is_empty();
    let mut branches: Vec<Branch> = Vec::with_capacity(kink_points.len() + 1);
    for i in 0..=kink_points.len() {
        let lo = if i == 0 {
            None
        } else {
            Some(&kink_points[i - 1])
        };
        let hi = kink_points.get(i);
        let resolved = if has_piecewise {
            let p = sample_between(arena, lo, hi)?;
            resolve_branches(arena, expr, var, p)?
        } else {
            expr
        };
        let constraints = domain_constraints(arena, resolved, var)?;
        let (numer, denom) = numer_denom(arena, resolved);

        let mut new_points: Vec<Critical> = Vec::new();
        match real_roots(arena, numer, var)? {
            // The branch vanishes identically: sampling reports `Zero`.
            RootSet::Everywhere => {}
            RootSet::Finite(roots) => add_roots(arena, &mut new_points, &roots, true, false)?,
        }
        if denom != arena.one {
            match real_roots(arena, denom, var)? {
                RootSet::Everywhere => {
                    return Err(failed("denominator is identically zero".into()));
                }
                RootSet::Finite(roots) => {
                    add_roots(arena, &mut new_points, &roots, false, true)?;
                }
            }
        }
        for c in &constraints {
            if let RootSet::Finite(roots) = real_roots(arena, c.expr, var)? {
                // At a zero of a `> 0` / `≠ 0` constraint the branch is
                // undefined; at a zero of a `≥ 0` constraint it is defined
                // and decided by evaluation.
                let undefined = c.kind != ConstraintKind::NonNegative;
                add_roots(arena, &mut new_points, &roots, false, undefined)?;
            }
        }
        // Only points inside (or on the boundary of) this kink interval
        // belong to the chart.
        for np in new_points {
            let inside_lo = lo.is_none_or(|l| approx_le(&l.approx, &np.approx));
            let inside_hi = hi.is_none_or(|h| approx_le(&np.approx, &h.approx));
            if inside_lo && inside_hi {
                merge_point(&mut points, np);
            }
        }
        branches.push(Branch {
            resolved,
            constraints,
        });
    }
    sort_points(&mut points);
    if points.len() > MAX_CRITICAL_POINTS {
        return Err(failed(format!(
            "too many critical points ({}); limit is {MAX_CRITICAL_POINTS}",
            points.len()
        )));
    }

    // Phase 3: decide every open interval and every critical point.  Each
    // interval is probed at several points; a disagreement means a zero
    // (or a pole) was missed and the chart cannot be trusted.
    let n = points.len();
    let mut interval_in: Vec<bool> = Vec::with_capacity(n + 1);
    for i in 0..=n {
        let lo = if i == 0 { None } else { Some(&points[i - 1]) };
        let hi = points.get(i);
        let probes = probe_ratios(lo, hi)?;
        let mut decision: Option<bool> = None;
        for p_ratio in probes {
            let p = arena.num_ratio(p_ratio.clone());
            let branch = branch_for(&branches, &kink_points, &p_ratio);
            let here = sample_satisfies(arena, branch, var, p, rel)?;
            match decision {
                None => decision = Some(here),
                Some(d) if d != here => {
                    return Err(failed(format!(
                        "the sign of {} changes between two consecutive critical points \
                         (a zero or a pole was not found)",
                        arena.display(expr)
                    )));
                }
                Some(_) => {}
            }
        }
        interval_in.push(decision.unwrap_or(false));
    }
    let mut point_in: Vec<bool> = Vec::with_capacity(n);
    for c in &points {
        point_in.push(point_satisfies(arena, expr, var, c, rel)?);
    }

    // Phase 4: assemble.
    let mut pieces: Vec<ExprId> = Vec::new();
    let mut isolated: Vec<ExprId> = Vec::new();
    for i in 0..=n {
        if !interval_in[i] {
            continue;
        }
        let (lo, lo_closed) = if i == 0 {
            (arena.neg_infinity, false)
        } else {
            (points[i - 1].id, point_in[i - 1])
        };
        let (hi, hi_closed) = if i == n {
            (arena.infinity, false)
        } else {
            (points[i].id, point_in[i])
        };
        let mut flags = INTERVAL_BOTH_CLOSED;
        if !lo_closed {
            flags |= INTERVAL_LEFT_OPEN;
        }
        if !hi_closed {
            flags |= INTERVAL_RIGHT_OPEN;
        }
        pieces.push(arena.interval(lo, hi, flags));
    }
    for (i, c) in points.iter().enumerate() {
        if point_in[i] && !interval_in[i] && !interval_in[i + 1] {
            isolated.push(c.id);
        }
    }
    if !isolated.is_empty() {
        pieces.push(arena.finite_set(&isolated));
    }
    Ok(match pieces.len() {
        0 => arena.empty_set,
        1 => pieces[0],
        _ => {
            let u = arena.set_union(&pieces);
            // Normal form: disjoint pieces in increasing order.
            crate::transforms::sets::simplify_set(arena, u)
        }
    })
}

/// Reject expressions the sign chart cannot handle honestly: periodic
/// functions (the equation solver only returns principal roots), step
/// functions with infinitely many jumps, special functions with poles, and
/// formal objects.
fn check_supported(arena: &Arena, expr: ExprId, var: ExprId) -> Result<(), SymplexError> {
    for id in walk::post_order_ids(arena, expr) {
        if !walk::contains(arena, id, var) {
            continue;
        }
        let ok = matches!(
            arena.node(id),
            ExprNode::Symbol(_)
                | ExprNode::Add(_)
                | ExprNode::Mul(_)
                | ExprNode::Pow(..)
                | ExprNode::Neg(_)
                | ExprNode::Exp(_)
                | ExprNode::Ln(_)
                | ExprNode::Abs(_)
                | ExprNode::Sign(_)
                | ExprNode::Heaviside(_)
                | ExprNode::Sinh(_)
                | ExprNode::Cosh(_)
                | ExprNode::Tanh(_)
                | ExprNode::Asinh(_)
                | ExprNode::Acosh(_)
                | ExprNode::Atanh(_)
                | ExprNode::Asin(_)
                | ExprNode::Acos(_)
                | ExprNode::Atan(_)
                | ExprNode::Erf(_)
                | ExprNode::Erfc(_)
                | ExprNode::Min(_)
                | ExprNode::Max(_)
                | ExprNode::Piecewise(_)
                | ExprNode::Gt(..)
                | ExprNode::Ge(..)
                | ExprNode::Eq_(..)
                | ExprNode::Ne(..)
                | ExprNode::And(_)
                | ExprNode::Or(_)
                | ExprNode::Not(_)
                | ExprNode::LambertW(_)
        );
        if !ok {
            return Err(failed(format!(
                "cannot build a sign chart for {}",
                arena.display(id)
            )));
        }
    }
    Ok(())
}

// ── critical points ────────────────────────────────────────────────────

/// Real zeros of an expression in `var`.
enum RootSet {
    /// The expression is identically zero.
    Everywhere,
    /// The (complete) list of real zeros.
    Finite(Vec<ExprId>),
}

/// `e` written over a common denominator, split into numerator and
/// denominator (denominator `1` when there is none).
fn numer_denom(arena: &mut Arena, e: ExprId) -> (ExprId, ExprId) {
    let combined = crate::poly::polybridge::together(arena, e);
    crate::poly::polybridge::as_numer_denom(arena, combined)
}

/// All real zeros of `e` as exact expressions.
///
/// Polynomials over ℚ are complete by construction: the number of real
/// roots returned by the equation solver is checked against an exact Sturm
/// count.  Rational functions reduce to their numerator.  Other
/// expressions rely on the equation solver and fail when it cannot find
/// the zeros.
fn real_roots(arena: &mut Arena, e: ExprId, var: ExprId) -> Result<RootSet, SymplexError> {
    use crate::transforms::solve::SolveOutcome;

    if !walk::contains(arena, e, var) {
        let ev = crate::transforms::eval::eval(arena, e);
        return Ok(if arena.is_zero_structural(ev) {
            RootSet::Everywhere
        } else {
            RootSet::Finite(Vec::new())
        });
    }

    // Rational function: zeros of the numerator that are not poles.
    if crate::poly::polybridge::expr_to_poly(arena, e, var).is_none() {
        let (n, d) = numer_denom(arena, e);
        if d != arena.one
            && n != e
            && crate::poly::polybridge::expr_to_poly(arena, n, var).is_some()
        {
            let numer_roots = match real_roots(arena, n, var)? {
                RootSet::Everywhere => return Ok(RootSet::Everywhere),
                RootSet::Finite(r) => r,
            };
            let mut out = Vec::new();
            for r in numer_roots {
                let at = crate::transforms::subs::subs(arena, d, var, r);
                if value_sign(arena, at) != ValueSign::Zero {
                    out.push(r);
                }
            }
            return Ok(RootSet::Finite(out));
        }
    }

    if let Some(poly) = crate::poly::polybridge::expr_to_poly(arena, e, var) {
        if poly.is_zero() {
            return Ok(RootSet::Everywhere);
        }
        if poly.is_constant() {
            return Ok(RootSet::Finite(Vec::new()));
        }
        let sqf = poly.square_free_part();
        let n_real = crate::poly::sturm::SturmChain::new(&sqf).count_real_roots();
        if n_real == 0 {
            return Ok(RootSet::Finite(Vec::new()));
        }
        let candidates = crate::transforms::solve::solve(arena, e, var);
        let mut reals: Vec<(ExprId, Q)> = Vec::new();
        for s in candidates {
            match value_sign(arena, s.value) {
                ValueSign::NonReal | ValueSign::Undefined => continue,
                _ => {}
            }
            let Some(a) = approx_real(arena, s.value) else {
                return Err(failed(format!(
                    "root {} cannot be evaluated numerically",
                    arena.display(s.value)
                )));
            };
            if !reals.iter().any(|(_, b)| approx_same(&a, b)) {
                reals.push((s.value, a));
            }
        }
        if reals.len() != n_real {
            return Err(failed(format!(
                "found {} of the {n_real} real roots of {}",
                reals.len(),
                arena.display(e)
            )));
        }
        return Ok(RootSet::Finite(
            reals.into_iter().map(|(id, _)| id).collect(),
        ));
    }

    match crate::transforms::solve::solve_classified(arena, e, var) {
        SolveOutcome::Identity => Ok(RootSet::Everywhere),
        SolveOutcome::NoSolution(_) => Ok(RootSet::Finite(Vec::new())),
        SolveOutcome::Solutions(s) if s.is_empty() => Err(failed(format!(
            "could not find the zeros of {}",
            arena.display(e)
        ))),
        SolveOutcome::Solutions(s) => {
            let mut out = Vec::new();
            for sol in s {
                match value_sign(arena, sol.value) {
                    ValueSign::NonReal | ValueSign::Undefined => continue,
                    ValueSign::Unknown if approx_real(arena, sol.value).is_none() => {
                        return Err(failed(format!(
                            "zero {} of {} cannot be located on the real line",
                            arena.display(sol.value),
                            arena.display(e)
                        )));
                    }
                    _ => out.push(sol.value),
                }
            }
            Ok(RootSet::Finite(out))
        }
    }
}

/// Add zeros to the critical-point list, merging numerically equal points.
fn add_roots(
    arena: &mut Arena,
    points: &mut Vec<Critical>,
    roots: &[ExprId],
    numer_zero: bool,
    undefined: bool,
) -> Result<(), SymplexError> {
    for &r in roots {
        let ev = crate::transforms::eval::eval(arena, r);
        let exact = arena.as_num(ev).is_some();
        let Some(approx) = approx_real(arena, ev) else {
            return Err(failed(format!(
                "endpoint {} cannot be ordered on the real line",
                arena.display(ev)
            )));
        };
        merge_point(
            points,
            Critical {
                id: ev,
                approx,
                exact,
                numer_zero,
                undefined,
            },
        );
    }
    Ok(())
}

/// Insert a point, merging it with an existing numerically equal one.
fn merge_point(points: &mut Vec<Critical>, np: Critical) {
    if let Some(existing) = points
        .iter_mut()
        .find(|c| approx_same(&c.approx, &np.approx))
    {
        existing.numer_zero |= np.numer_zero;
        existing.undefined |= np.undefined;
        if np.exact && !existing.exact {
            existing.id = np.id;
            existing.approx = np.approx;
            existing.exact = true;
        }
        return;
    }
    points.push(np);
}

fn sort_points(points: &mut [Critical]) {
    points.sort_by(|a, b| a.approx.cmp(&b.approx));
}

/// Are two approximations the same point (to [`MERGE_REL`])?
fn approx_same(a: &Q, b: &Q) -> bool {
    let scale = ratio_abs_f64(a).max(ratio_abs_f64(b)).max(1.0);
    let Some(tol) = crate::base::numeric::f64_to_ratio_exact(MERGE_REL * scale) else {
        return a == b;
    };
    (a - b).abs() <= tol
}

/// `a ≤ b` up to the merge tolerance.
fn approx_le(a: &Q, b: &Q) -> bool {
    a <= b || approx_same(a, b)
}

fn ratio_abs_f64(r: &Q) -> f64 {
    crate::base::numeric::ratio_to_f64(r)
        .map(f64::abs)
        .unwrap_or(f64::INFINITY)
}

// ── domain constraints and kinks ───────────────────────────────────────

/// Natural-domain constraints of `expr` (which must be free of
/// piecewise-defined nodes containing `var`).
fn domain_constraints(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
) -> Result<Vec<Constraint>, SymplexError> {
    let mut out: Vec<Constraint> = Vec::new();
    let mut push = |e: ExprId, kind: ConstraintKind| {
        if !out.iter().any(|c| c.expr == e && c.kind == kind) {
            out.push(Constraint { expr: e, kind });
        }
    };
    for id in walk::post_order_ids(arena, expr) {
        if !walk::contains(arena, id, var) {
            continue;
        }
        match arena.node(id).clone() {
            ExprNode::Pow(base, exp) => {
                let base_has_var = walk::contains(arena, base, var);
                if let Some(r) = arena.as_num(exp).cloned() {
                    if !base_has_var {
                        continue;
                    }
                    if r.is_negative() {
                        push(base, ConstraintKind::NonZero);
                    }
                    if !r.is_integer() {
                        // Principal branch: real only for a non-negative base.
                        push(
                            base,
                            if r.is_negative() {
                                ConstraintKind::Positive
                            } else {
                                ConstraintKind::NonNegative
                            },
                        );
                    }
                } else if walk::contains(arena, exp, var) {
                    if base_has_var {
                        push(base, ConstraintKind::Positive);
                    } else {
                        let positive_const = base == arena.e_const
                            || arena.as_num(base).is_some_and(|b| b.is_positive());
                        if !positive_const {
                            return Err(failed(format!(
                                "variable exponent with base {}",
                                arena.display(base)
                            )));
                        }
                    }
                } else {
                    // Constant, non-rational exponent (e.g. x^π).
                    match sign_of(arena, exp) {
                        Some(1) => push(base, ConstraintKind::NonNegative),
                        Some(_) => push(base, ConstraintKind::Positive),
                        None => {
                            return Err(failed(format!(
                                "exponent {} has unknown sign",
                                arena.display(exp)
                            )));
                        }
                    }
                }
            }
            ExprNode::Ln(a) => push(a, ConstraintKind::Positive),
            ExprNode::Asin(a) | ExprNode::Acos(a) => {
                let one = arena.one;
                let lo = arena.add(&[a, one]);
                let hi = arena.sub(one, a);
                push(lo, ConstraintKind::NonNegative);
                push(hi, ConstraintKind::NonNegative);
            }
            ExprNode::Acosh(a) => {
                let one = arena.one;
                let shifted = arena.sub(a, one);
                push(shifted, ConstraintKind::NonNegative);
            }
            ExprNode::Atanh(a) => {
                let one = arena.one;
                let lo = arena.add(&[a, one]);
                let hi = arena.sub(one, a);
                push(lo, ConstraintKind::Positive);
                push(hi, ConstraintKind::Positive);
            }
            ExprNode::LambertW(a) => {
                // Principal branch is real for a ≥ −1/e.
                let inv_e = arena.pow(arena.e_const, arena.neg_one);
                let shifted = arena.add(&[a, inv_e]);
                push(shifted, ConstraintKind::NonNegative);
            }
            _ => {}
        }
    }
    Ok(out)
}

/// Expressions whose zeros are the switching points of piecewise-defined
/// nodes in `expr`.
fn kink_expressions(arena: &mut Arena, expr: ExprId, var: ExprId) -> Vec<ExprId> {
    let mut out: Vec<ExprId> = Vec::new();
    for id in walk::post_order_ids(arena, expr) {
        if !walk::contains(arena, id, var) {
            continue;
        }
        match arena.node(id).clone() {
            ExprNode::Abs(a) | ExprNode::Sign(a) | ExprNode::Heaviside(a) => out.push(a),
            ExprNode::Min(args) | ExprNode::Max(args) => {
                for i in 0..args.len() {
                    for j in (i + 1)..args.len() {
                        let d = arena.sub(args[i], args[j]);
                        out.push(d);
                    }
                }
            }
            ExprNode::Piecewise(pairs) => {
                for (_, cond) in pairs.iter() {
                    for cid in walk::post_order_ids(arena, *cond) {
                        if let ExprNode::Gt(a, b)
                        | ExprNode::Ge(a, b)
                        | ExprNode::Eq_(a, b)
                        | ExprNode::Ne(a, b) = arena.node(cid).clone()
                        {
                            let d = arena.sub(a, b);
                            out.push(d);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    out.retain(|&e| walk::contains(arena, e, var));
    out.sort_by_key(|id| id.0);
    out.dedup();
    out
}

/// Replace every piecewise-defined node of `expr` by the branch that is
/// active at `var = p` (a point strictly inside a kink interval).
fn resolve_branches(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    p: ExprId,
) -> Result<ExprId, SymplexError> {
    let mut current = expr;
    for _ in 0..MAX_BRANCH_PASSES {
        let mut map: FxHashMap<ExprId, ExprId> = FxHashMap::default();
        for id in walk::post_order_ids(arena, current) {
            if !walk::contains(arena, id, var) {
                continue;
            }
            let replacement = match arena.node(id).clone() {
                ExprNode::Abs(a) => match sign_at(arena, a, var, p)? {
                    std::cmp::Ordering::Greater => a,
                    std::cmp::Ordering::Less => arena.neg(a),
                    std::cmp::Ordering::Equal => return Err(kink_at_sample(arena, id)),
                },
                ExprNode::Sign(a) => match sign_at(arena, a, var, p)? {
                    std::cmp::Ordering::Greater => arena.one,
                    std::cmp::Ordering::Less => arena.neg_one,
                    std::cmp::Ordering::Equal => return Err(kink_at_sample(arena, id)),
                },
                ExprNode::Heaviside(a) => match sign_at(arena, a, var, p)? {
                    std::cmp::Ordering::Greater => arena.one,
                    std::cmp::Ordering::Less => arena.zero,
                    std::cmp::Ordering::Equal => return Err(kink_at_sample(arena, id)),
                },
                ExprNode::Min(args) | ExprNode::Max(args) => {
                    let want_max = matches!(arena.node(id), ExprNode::Max(_));
                    let mut best: Option<(ExprId, Q)> = None;
                    for &a in args.iter() {
                        let at = crate::transforms::subs::subs(arena, a, var, p);
                        let Some(v) = approx_real(arena, at) else {
                            return Err(failed(format!(
                                "cannot evaluate {} at a sample point",
                                arena.display(a)
                            )));
                        };
                        best = Some(match best {
                            None => (a, v),
                            Some((b, bv)) => {
                                if approx_same(&v, &bv) {
                                    return Err(kink_at_sample(arena, id));
                                }
                                if (v > bv) == want_max {
                                    (a, v)
                                } else {
                                    (b, bv)
                                }
                            }
                        });
                    }
                    best.map(|(a, _)| a).unwrap_or(id)
                }
                ExprNode::Piecewise(pairs) => {
                    let mut chosen: Option<ExprId> = None;
                    for (val, cond) in pairs.iter() {
                        let c = crate::transforms::subs::subs(arena, *cond, var, p);
                        let c = crate::transforms::eval::eval(arena, c);
                        match arena.node(c) {
                            ExprNode::BoolTrue => {
                                chosen = Some(*val);
                                break;
                            }
                            ExprNode::BoolFalse => continue,
                            _ => {
                                return Err(failed(format!(
                                    "piecewise condition {} does not evaluate at a sample point",
                                    arena.display(*cond)
                                )));
                            }
                        }
                    }
                    chosen
                        .ok_or_else(|| failed("piecewise function has no active branch".into()))?
                }
                _ => continue,
            };
            map.insert(id, replacement);
        }
        if map.is_empty() {
            return Ok(current);
        }
        let next = walk::walk_and_rebuild(arena, current, &|_, id| map.get(&id).copied());
        current = crate::transforms::eval::eval(arena, next);
    }
    Err(failed("piecewise nesting too deep".into()))
}

fn kink_at_sample(arena: &Arena, id: ExprId) -> SymplexError {
    failed(format!(
        "switching point of {} coincides with a sample point",
        arena.display(id)
    ))
}

/// Exact sign of `e` at `var = p`, or an error when it cannot be decided.
fn sign_at(
    arena: &mut Arena,
    e: ExprId,
    var: ExprId,
    p: ExprId,
) -> Result<std::cmp::Ordering, SymplexError> {
    let at = crate::transforms::subs::subs(arena, e, var, p);
    value_sign(arena, at).ordering().ok_or_else(|| {
        failed(format!(
            "cannot decide the sign of {} at a sample point",
            arena.display(e)
        ))
    })
}

// ── sampling and evaluation ────────────────────────────────────────────

/// The branch active at the rational sample point `p`.
fn branch_for<'a>(branches: &'a [Branch], kinks: &[Critical], p: &Q) -> &'a Branch {
    let idx = kinks.iter().take_while(|k| k.approx < *p).count();
    &branches[idx.min(branches.len() - 1)]
}

/// A rational point strictly between two consecutive critical points (or
/// beyond the last / before the first one).
fn sample_between(
    arena: &mut Arena,
    lo: Option<&Critical>,
    hi: Option<&Critical>,
) -> Result<ExprId, SymplexError> {
    let r = probe_ratios(lo, hi)?.swap_remove(0);
    Ok(arena.num_ratio(r.clone()))
}

/// Rational probe points strictly inside the open interval between two
/// consecutive critical points.  The first one is the "main" sample
/// (midpoint / one unit beyond the end); the others spread over the
/// interval so that a missed sign change is detected.
fn probe_ratios(lo: Option<&Critical>, hi: Option<&Critical>) -> Result<Vec<Q>, SymplexError> {
    let int = |n: i64| Ratio::from_integer(BigInt::from(n));
    match (lo, hi) {
        (None, None) => Ok(vec![
            Ratio::zero(),
            int(10),
            int(-10),
            int(1000),
            int(-1000),
        ]),
        (None, Some(h)) => {
            let base = Ratio::from_integer(h.approx.floor().to_integer());
            Ok(vec![&base - int(1), &base - int(10), &base - int(1000)])
        }
        (Some(l), None) => {
            let base = Ratio::from_integer(l.approx.ceil().to_integer());
            Ok(vec![&base + int(1), &base + int(10), &base + int(1000)])
        }
        (Some(l), Some(h)) => {
            let gap = &h.approx - &l.approx;
            if !gap.is_positive() {
                return Err(failed("critical points too close to separate".into()));
            }
            let quarter = &gap / int(4);
            let mid = &l.approx + &gap / int(2);
            let targets = [mid.clone(), &l.approx + &quarter, &h.approx - &quarter];
            if l.exact && h.exact {
                return Ok(targets.to_vec());
            }
            // The approximations are accurate to ~1e-29 relative, so a
            // gap below the merge tolerance cannot be bridged reliably.
            let scale = ratio_abs_f64(&l.approx)
                .max(ratio_abs_f64(&h.approx))
                .max(1.0);
            let Some(min_gap) = crate::base::numeric::f64_to_ratio_exact(MERGE_REL * scale) else {
                return Err(failed("critical points too close to separate".into()));
            };
            if gap <= min_gap {
                return Err(failed("critical points too close to separate".into()));
            }
            // Simplify: any rational within gap/8 of a target is still
            // strictly inside (lo, hi).
            let max_denom = (int(8) / &gap).ceil().to_integer().max(BigInt::one());
            Ok(targets
                .iter()
                .map(|t| crate::base::numeric::best_rational_approx(t, &max_denom))
                .collect())
        }
    }
}

/// Does `expr rel 0` hold at the rational sample point `p` (which lies in
/// the kink interval of `branch`)?  Points where the branch leaves its
/// natural real domain never satisfy the inequality.
fn sample_satisfies(
    arena: &mut Arena,
    branch: &Branch,
    var: ExprId,
    p: ExprId,
    rel: Relation,
) -> Result<bool, SymplexError> {
    for c in &branch.constraints {
        let at = crate::transforms::subs::subs(arena, c.expr, var, p);
        let s = value_sign(arena, at);
        if s == ValueSign::Unknown {
            return Err(failed(format!(
                "cannot decide the domain constraint {} at a sample point",
                arena.display(c.expr)
            )));
        }
        let ok = match c.kind {
            ConstraintKind::NonZero => matches!(s, ValueSign::Pos | ValueSign::Neg),
            ConstraintKind::NonNegative => matches!(s, ValueSign::Pos | ValueSign::Zero),
            ConstraintKind::Positive => matches!(s, ValueSign::Pos),
        };
        if !ok {
            return Ok(false);
        }
    }
    let at = crate::transforms::subs::subs(arena, branch.resolved, var, p);
    match value_sign(arena, at) {
        ValueSign::NonReal | ValueSign::Undefined => Ok(false),
        ValueSign::Unknown => Err(failed(format!(
            "cannot decide the sign of {} at a sample point",
            arena.display(branch.resolved)
        ))),
        ValueSign::Neg => Ok(rel.accepts(std::cmp::Ordering::Less)),
        ValueSign::Zero => Ok(rel.accepts(std::cmp::Ordering::Equal)),
        ValueSign::Pos => Ok(rel.accepts(std::cmp::Ordering::Greater)),
    }
}

/// Does `expr rel 0` hold at the critical point `c`?
fn point_satisfies(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    c: &Critical,
    rel: Relation,
) -> Result<bool, SymplexError> {
    if c.undefined {
        return Ok(false);
    }
    let at = crate::transforms::subs::subs(arena, expr, var, c.id);
    match value_sign(arena, at) {
        ValueSign::NonReal | ValueSign::Undefined => Ok(false),
        ValueSign::Unknown if c.numer_zero => Ok(rel.accepts(std::cmp::Ordering::Equal)),
        ValueSign::Unknown => Err(failed(format!(
            "cannot decide the sign of the expression at {}",
            arena.display(c.id)
        ))),
        ValueSign::Neg => Ok(rel.accepts(std::cmp::Ordering::Less)),
        ValueSign::Zero => Ok(rel.accepts(std::cmp::Ordering::Equal)),
        ValueSign::Pos => Ok(rel.accepts(std::cmp::Ordering::Greater)),
    }
}

/// Classify the value of a variable-free expression: exactly when it
/// evaluates to a rational, otherwise to [`DECISION_DIGITS`] digits.
fn value_sign(arena: &mut Arena, e: ExprId) -> ValueSign {
    let ev = crate::transforms::eval::eval(arena, e);
    if let Some(r) = arena.as_num(ev) {
        return if r.is_zero() {
            ValueSign::Zero
        } else if r.is_positive() {
            ValueSign::Pos
        } else {
            ValueSign::Neg
        };
    }
    match arena.node(ev) {
        ExprNode::Infinity
        | ExprNode::NegInfinity
        | ExprNode::ComplexInfinity
        | ExprNode::NaN
        | ExprNode::BoolTrue
        | ExprNode::BoolFalse => return ValueSign::Undefined,
        _ => {}
    }
    if !walk::free_symbols(arena, ev).is_empty() || walk::has_unevaluated(arena, ev) {
        return ValueSign::Unknown;
    }
    let Ok(text) = crate::transforms::evalf::evalf(arena, ev, DECISION_DIGITS) else {
        return ValueSign::Unknown;
    };
    let Some(parsed) = parse_evalf_complex(&text) else {
        return ValueSign::Unknown;
    };
    if !is_negligible_imaginary(&parsed) {
        return ValueSign::NonReal;
    }
    let first = parsed.re;
    if first.is_zero() {
        return ValueSign::Zero;
    }
    if first.abs() >= pow10_inv(AMBIGUOUS_EXP) {
        return if first.is_positive() {
            ValueSign::Pos
        } else {
            ValueSign::Neg
        };
    }
    // Tiny: either a genuine small value (`exp(-1000)`) or cancellation
    // noise around an exact zero.  Re-evaluate at twice the precision: a
    // genuine value reproduces itself, noise shrinks by ~30 digits.
    let Ok(text2) = crate::transforms::evalf::evalf(arena, ev, 2 * DECISION_DIGITS) else {
        return ValueSign::Unknown;
    };
    let Some(second) = parse_evalf_complex(&text2).map(|p| p.re) else {
        return ValueSign::Unknown;
    };
    if second.is_zero() || first.is_positive() != second.is_positive() {
        return ValueSign::Unknown;
    }
    let diff = (&first - &second).abs();
    let tol = first.abs() / Ratio::from_integer(BigInt::from(1_000_000));
    if diff > tol {
        return ValueSign::Unknown;
    }
    if first.is_positive() {
        ValueSign::Pos
    } else {
        ValueSign::Neg
    }
}

/// `10^-k` as an exact rational.
fn pow10_inv(k: usize) -> Q {
    Ratio::new(BigInt::one(), num_traits::pow(BigInt::from(10), k))
}

/// Is the imaginary part negligible relative to `max(1, |re|)`?
fn is_negligible_imaginary(p: &ParsedComplex) -> bool {
    let scale = p.re.abs().max(Ratio::one());
    p.im.abs() <= scale * pow10_inv(NONREAL_EXP)
}

/// Real value of a variable-free expression as an exact rational (when
/// it evaluates to one) or a [`DECISION_DIGITS`]-digit approximation.
fn approx_real(arena: &mut Arena, e: ExprId) -> Option<Q> {
    let ev = crate::transforms::eval::eval(arena, e);
    if let Some(r) = arena.as_num(ev) {
        return Some(r.clone());
    }
    if !walk::free_symbols(arena, ev).is_empty() || walk::has_unevaluated(arena, ev) {
        return None;
    }
    let text = crate::transforms::evalf::evalf(arena, ev, DECISION_DIGITS).ok()?;
    let parsed = parse_evalf_complex(&text)?;
    if !is_negligible_imaginary(&parsed) {
        return None;
    }
    Some(parsed.re)
}

// ── evalf output parsing ───────────────────────────────────────────────

/// A parsed `evalf` string, as exact decimal rationals (no overflow or
/// underflow for huge / tiny magnitudes).
struct ParsedComplex {
    re: Q,
    im: Q,
}

/// Parse an `evalf` result: `"1.5"`, `"-2e-7"`, `"i"`, `"2.5*i"`,
/// `"1.0 + 2.0*i"`, `"1.0 - 2.0*i"`, `"0"`.
fn parse_evalf_complex(s: &str) -> Option<ParsedComplex> {
    let s = s.trim();
    if let Some(re) = decimal_to_ratio(s) {
        return Some(ParsedComplex {
            re,
            im: Ratio::zero(),
        });
    }
    let has_i = s.ends_with('i') || s.ends_with('I');
    if !has_i {
        return None;
    }
    let body = s[..s.len() - 1].trim_end_matches('*').trim();
    // Pure imaginary.
    let pure_im = match body {
        "" | "+" => Some(Ratio::one()),
        "-" => Some(-Ratio::<BigInt>::one()),
        t => decimal_to_ratio(t),
    };
    if let Some(im) = pure_im {
        return Some(ParsedComplex {
            re: Ratio::zero(),
            im,
        });
    }
    // Split at the last '+' / '-' that is not an exponent sign.
    let bytes = body.as_bytes();
    let mut split = None;
    for i in (1..bytes.len()).rev() {
        if (bytes[i] == b'+' || bytes[i] == b'-') && !matches!(bytes[i - 1], b'e' | b'E') {
            split = Some(i);
            break;
        }
    }
    let idx = split?;
    let re = decimal_to_ratio(body[..idx].trim())?;
    let im_text: String = body[idx..].chars().filter(|c| !c.is_whitespace()).collect();
    let im = match im_text.as_str() {
        "+" => Ratio::one(),
        "-" => -Ratio::<BigInt>::one(),
        t => decimal_to_ratio(t)?,
    };
    Some(ParsedComplex { re, im })
}

/// Exact rational value of a decimal literal such as `-12.345e-7`.
fn decimal_to_ratio(text: &str) -> Option<Q> {
    let t = text.trim();
    let (negative, t) = match t.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, t.strip_prefix('+').unwrap_or(t)),
    };
    let (mantissa, exp) = match t.find(['e', 'E']) {
        Some(pos) => (&t[..pos], t[pos + 1..].parse::<i64>().ok()?),
        None => (t, 0i64),
    };
    let (int_part, frac_part) = match mantissa.find('.') {
        Some(pos) => (&mantissa[..pos], &mantissa[pos + 1..]),
        None => (mantissa, ""),
    };
    if int_part.is_empty() && frac_part.is_empty() {
        return None;
    }
    if !int_part.chars().all(|c| c.is_ascii_digit())
        || !frac_part.chars().all(|c| c.is_ascii_digit())
    {
        return None;
    }
    let digits = format!("{int_part}{frac_part}");
    let digits = if digits.is_empty() { "0" } else { &digits };
    let n: BigInt = digits.parse().ok()?;
    let scale = exp - frac_part.len() as i64;
    let ten = BigInt::from(10);
    let mut r = if scale >= 0 {
        Ratio::from_integer(n * num_traits::pow(ten, scale.unsigned_abs() as usize))
    } else {
        Ratio::new(n, num_traits::pow(ten, scale.unsigned_abs() as usize))
    };
    if negative {
        r = -r;
    }
    Some(r)
}

// ═══════════════════════════════════════════════════════════════════════════
// Absolute-value inequalities: c·|f(x)| + d  rel  0  with f linear in x
// ═══════════════════════════════════════════════════════════════════════════

/// Try to solve `expr rel 0` when `expr` has the shape `c·|a·x + b| + d`
/// (a single absolute-value term plus var-free terms).
///
/// Rewrites to `|f| rel' k` and returns the interval / union directly,
/// which also works for **symbolic** `a`, `b`, `k` where numeric root
/// sorting would fail:
///
/// - `|f| < k`  → `-k < f < k`  → one interval in `x`
/// - `|f| > k`  → `f < -k ∪ f > k` → union of two rays
///
/// Returns `None` if the expression does not have this shape or the sign
/// of the leading coefficient `a` cannot be determined.
fn try_solve_abs_inequality(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    rel: Relation,
) -> Option<ExprId> {
    let terms: Vec<ExprId> = match arena.node(expr).clone() {
        ExprNode::Add(children) => children.to_vec(),
        _ => vec![expr],
    };

    // Locate the single |f(x)| term and its (var-free) coefficient.
    let mut abs_inner: Option<ExprId> = None;
    let mut abs_coeff: Option<ExprId> = None;
    let mut rest: Vec<ExprId> = Vec::new();
    for &t in &terms {
        if !walk::contains(arena, t, var) {
            rest.push(t);
            continue;
        }
        let (inner, coeff) = match arena.node(t).clone() {
            ExprNode::Abs(inner) => (inner, arena.one),
            ExprNode::Neg(n) => match arena.node(n).clone() {
                ExprNode::Abs(inner) => (inner, arena.neg_one),
                _ => return None,
            },
            ExprNode::Mul(children) => {
                let mut inner = None;
                let mut consts = Vec::new();
                for &c in &children {
                    if !walk::contains(arena, c, var) {
                        consts.push(c);
                    } else if let ExprNode::Abs(i) = arena.node(c).clone()
                        && inner.is_none()
                    {
                        inner = Some(i);
                    } else {
                        return None;
                    }
                }
                let coeff = match consts.len() {
                    0 => arena.one,
                    1 => consts[0],
                    _ => arena.mul(&consts),
                };
                (inner?, coeff)
            }
            _ => return None,
        };
        if abs_inner.is_some() {
            return None; // two abs terms
        }
        abs_inner = Some(inner);
        abs_coeff = Some(coeff);
    }
    let inner = abs_inner?;
    let coeff = abs_coeff?;

    // f = a·x + b must be linear in x.
    let coeffs = crate::transforms::solve::symbolic_poly_coeffs(arena, inner, var)?;
    if coeffs.len() != 2 {
        return None;
    }
    let a = coeffs[1];
    let b = coeffs[0];

    // c·|f| + d rel 0  ⇔  |f| rel'  (-d/c)   (flip if c < 0)
    let d = match rest.len() {
        0 => arena.zero,
        1 => rest[0],
        _ => arena.add(&rest),
    };
    let c_sign = sign_of(arena, coeff)?;
    let neg_d = arena.neg(d);
    let k = arena.div(neg_d, coeff);
    let k = crate::transforms::eval::eval(arena, k);
    let rel = if c_sign < 0 { flip(rel) } else { rel };

    // If k is a known negative number, |f| < k is empty and |f| > k is ℝ.
    if let Some(kv) = arena.as_num(k).cloned()
        && kv.is_negative()
    {
        return Some(match rel {
            Relation::Lt | Relation::Le => arena.empty_set,
            Relation::Gt | Relation::Ge => {
                arena.interval(arena.neg_infinity, arena.infinity, INTERVAL_BOTH_OPEN)
            }
        });
    }

    // Endpoints in x: f = ±k  ⇒  x = (±k - b)/a
    let a_sign = sign_of(arena, a)?;
    let k_minus_b = arena.sub(k, b);
    let neg_k = arena.neg(k);
    let neg_k_minus_b = arena.sub(neg_k, b);
    let x_hi = arena.div(k_minus_b, a);
    let x_lo = arena.div(neg_k_minus_b, a);
    let x_hi = crate::transforms::eval::eval(arena, x_hi);
    let x_lo = crate::transforms::eval::eval(arena, x_lo);
    let (lo, hi) = if a_sign > 0 {
        (x_lo, x_hi)
    } else {
        (x_hi, x_lo)
    };

    Some(match rel {
        Relation::Lt => arena.interval(lo, hi, INTERVAL_BOTH_OPEN),
        Relation::Le => arena.interval(lo, hi, INTERVAL_BOTH_CLOSED),
        Relation::Gt => {
            let left = arena.interval(arena.neg_infinity, lo, INTERVAL_BOTH_OPEN);
            let right = arena.interval(hi, arena.infinity, INTERVAL_BOTH_OPEN);
            arena.set_union(&[left, right])
        }
        Relation::Ge => {
            let left = arena.interval(arena.neg_infinity, lo, INTERVAL_LEFT_OPEN);
            let right = arena.interval(hi, arena.infinity, INTERVAL_RIGHT_OPEN);
            arena.set_union(&[left, right])
        }
    })
}

/// Reverse a relation (used when dividing by a negative coefficient).
fn flip(rel: Relation) -> Relation {
    match rel {
        Relation::Gt => Relation::Lt,
        Relation::Ge => Relation::Le,
        Relation::Lt => Relation::Gt,
        Relation::Le => Relation::Ge,
    }
}

/// Sign of a var-free expression: `Some(1)`, `Some(-1)`, or `None` if
/// unknown (symbolic without a determinable sign, or zero).
fn sign_of(arena: &mut Arena, e: ExprId) -> Option<i8> {
    let ev = crate::transforms::eval::eval(arena, e);
    if let Some(r) = arena.as_num(ev) {
        return if r.is_positive() {
            Some(1)
        } else if r.is_negative() {
            Some(-1)
        } else {
            None
        };
    }
    match value_sign(arena, ev) {
        ValueSign::Pos => return Some(1),
        ValueSign::Neg => return Some(-1),
        ValueSign::Zero | ValueSign::NonReal | ValueSign::Undefined => return None,
        ValueSign::Unknown => {}
    }
    // Symbolic: consult stored symbol assumptions for a bare symbol.
    if let ExprNode::Symbol(sid) = arena.node(ev) {
        let a = arena.symbol_assumptions(*sid);
        if a.known_true
            .contains(crate::base::assumptions::Props::POSITIVE)
        {
            return Some(1);
        }
        if a.known_true
            .contains(crate::base::assumptions::Props::NEGATIVE)
        {
            return Some(-1);
        }
    }
    None
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::arena::Arena;

    fn display(arena: &Arena, id: ExprId) -> String {
        arena.display(id).to_string()
    }

    /// Helper: build `x^2 - 4` in the arena and solve `> 0`.
    #[test]
    fn sign_chart_x2_minus_4_gt() {
        let mut arena = Arena::new();
        let x = arena.symbol("x");
        let two = arena.int(2);
        let x2 = arena.pow(x, two);
        let four = arena.int(4);
        let expr = arena.sub(x2, four); // x² - 4

        let result = solve_inequality(&mut arena, expr, x, Relation::Gt).unwrap();
        assert_eq!(display(&arena, result), "(-oo, -2) ∪ (2, oo)");
    }

    #[test]
    fn positive_constant_gt_zero() {
        let mut arena = Arena::new();
        let x = arena.symbol("x");
        let five = arena.int(5);

        let result = solve_inequality(&mut arena, five, x, Relation::Gt).unwrap();
        // 5 > 0 always true → should be (-∞, ∞), not EmptySet
        assert_ne!(result, arena.empty_set, "5 > 0 should not be EmptySet");
    }

    #[test]
    fn negative_constant_gt_zero() {
        let mut arena = Arena::new();
        let x = arena.symbol("x");
        let neg3 = arena.int(-3);

        let result = solve_inequality(&mut arena, neg3, x, Relation::Gt).unwrap();
        assert_eq!(result, arena.empty_set, "-3 > 0 should be EmptySet");
    }

    #[test]
    fn rational_inequality_excludes_pole_and_closes_root() {
        // (x - 1)/(x + 1) >= 0  →  (-oo, -1) ∪ [1, oo)
        let mut arena = Arena::new();
        let x = arena.symbol("x");
        let one = arena.one;
        let num = arena.sub(x, one);
        let den = arena.add(&[x, one]);
        let expr = arena.div(num, den);
        let result = solve_inequality(&mut arena, expr, x, Relation::Ge).unwrap();
        assert_eq!(display(&arena, result), "(-oo, -1) ∪ [1, oo)");
    }

    #[test]
    fn sqrt_inequality_respects_domain() {
        // sqrt(x) - 2 < 0  →  [0, 4)
        let mut arena = Arena::new();
        let x = arena.symbol("x");
        let s = arena.sqrt(x);
        let two = arena.int(2);
        let expr = arena.sub(s, two);
        let result = solve_inequality(&mut arena, expr, x, Relation::Lt).unwrap();
        assert_eq!(display(&arena, result), "[0, 4)");
    }

    #[test]
    fn solveset_quadratic_arena() {
        let mut arena = Arena::new();
        let x = arena.symbol("x");
        // x² - 5x + 6 = (x-2)(x-3)
        let two = arena.int(2);
        let x2 = arena.pow(x, two);
        let five = arena.int(5);
        let five_x = arena.mul(&[five, x]);
        let neg_five_x = arena.neg(five_x);
        let six = arena.int(6);
        let expr = arena.add(&[x2, neg_five_x, six]);

        let result = solveset(&mut arena, expr, x);
        assert_ne!(result, arena.empty_set, "should find roots of x²-5x+6");
    }

    #[test]
    fn parse_evalf_strings() {
        let r = |p: i64, q: i64| Ratio::new(BigInt::from(p), BigInt::from(q));
        let p = parse_evalf_complex("-1.5e-3").unwrap();
        assert_eq!((p.re, p.im), (r(-3, 2000), r(0, 1)));
        let p = parse_evalf_complex("1.25 - 2.5*i").unwrap();
        assert_eq!((p.re, p.im), (r(5, 4), r(-5, 2)));
        let p = parse_evalf_complex("-i").unwrap();
        assert_eq!((p.re, p.im), (r(0, 1), r(-1, 1)));
        let p = parse_evalf_complex("3*i").unwrap();
        assert_eq!((p.re, p.im), (r(0, 1), r(3, 1)));
        let p = parse_evalf_complex("2e-5 + i").unwrap();
        assert_eq!((p.re, p.im), (r(1, 50000), r(1, 1)));
        // Magnitudes beyond f64 stay exact.
        let p = parse_evalf_complex("1.0e500").unwrap();
        assert!(p.re > r(1, 1));
        let p = parse_evalf_complex("-2.5e-400").unwrap();
        assert!(p.re.is_negative() && p.re > r(-1, 1));
    }

    #[test]
    fn decimal_literals_to_ratio() {
        let r = |p: i64, q: i64| Ratio::new(BigInt::from(p), BigInt::from(q));
        assert_eq!(decimal_to_ratio("1.25"), Some(r(5, 4)));
        assert_eq!(decimal_to_ratio("-0.5"), Some(r(-1, 2)));
        assert_eq!(decimal_to_ratio("3"), Some(r(3, 1)));
        assert_eq!(decimal_to_ratio("1.5e2"), Some(r(150, 1)));
        assert_eq!(decimal_to_ratio("-12.5e-3"), Some(r(-1, 80)));
        assert_eq!(decimal_to_ratio("abc"), None);
    }
}
