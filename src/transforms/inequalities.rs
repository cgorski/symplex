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
use crate::base::interval::IntervalKind;
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::Q;
use crate::base::truth::{SignSet, Truth};
use crate::base::walk;
use crate::transforms::evalf::{ConstValue, Decimal};

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
    pub(crate) fn accepts(self, sign: std::cmp::Ordering) -> bool {
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
/// line, restricted to the natural real domain of `expr` and to the
/// declared domain of `var`
/// ([`restrict_real_set_to_domain`](crate::transforms::sets::restrict_real_set_to_domain)):
/// `x² − 4 < 0` is `(0, 2)` for a positive `x` and `{−1, 0, 1}` for an
/// integer one.  Up to 0.42 the assumptions of `var` were ignored
/// (`(−2, 2)` for both).  An error for a domain no set writes (rational,
/// prime, …).
pub(crate) fn solve_inequality(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    rel: Relation,
) -> Result<ExprId, SymplexError> {
    let set = solve_inequality_over_reals(arena, expr, var, rel)?;
    crate::transforms::sets::restrict_real_set_to_domain(arena, var, set).ok_or_else(|| {
        failed(
            "the declared domain of the variable is not a union of intervals or of integers in one"
                .into(),
        )
    })
}

/// [`solve_inequality`] over the whole real line, whatever the declared
/// domain of `var`.
pub(crate) fn solve_inequality_over_reals(
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

    // ── Non-real coefficients: real only where the imaginary part vanishes.
    if let Some(set) = try_solve_nonreal_inequality(arena, expr, var, rel) {
        return set;
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
                Ok(arena.interval(arena.neg_infinity, arena.infinity, IntervalKind::Open))
            } else {
                Ok(arena.empty_set)
            };
        }
    }

    // ── Periodic functions of the variable alone: one period, then all. ──
    if let Some(set) = try_solve_periodic_inequality(arena, expr, var, rel) {
        return set;
    }

    sign_chart(arena, expr, var, rel)
}

/// Most critical points in one period of a periodic inequality.
const MAX_PERIODIC_POINTS: usize = 128;

/// Largest `m` tried when reducing a common period `T` to `T/m`.
const MAX_PERIOD_DIVISOR: usize = 24;

/// The common period `T = 2π·r` (returned: `r`) of an expression in which
/// `var` occurs only in `sin`, `cos` and `tan` of `a·var + b` (rational
/// `a ≠ 0`; `cot`, `sec`, `csc` are their reciprocals): `r` is the least
/// common multiple of the `1/|a|`.  `None` otherwise.
fn trig_period_ratio(arena: &mut Arena, expr: ExprId, var: ExprId) -> Option<Q> {
    let mut atoms: Vec<ExprId> = Vec::new();
    let mut ratio: Option<Q> = None;
    for id in walk::post_order_ids(arena, expr) {
        let arg = match *arena.node(id) {
            ExprNode::Sin(a) | ExprNode::Cos(a) | ExprNode::Tan(a) => a,
            _ => continue,
        };
        if !walk::contains(arena, arg, var) {
            continue;
        }
        let slope = crate::transforms::diff::diff(arena, arg, var);
        let slope = crate::transforms::eval::eval(arena, slope);
        let a = arena.as_num(slope)?.clone();
        if a.is_zero() {
            return None;
        }
        let r = Ratio::one() / a.abs();
        ratio = Some(match ratio {
            None => r,
            Some(q) => {
                // lcm of p₁/q₁ and p₂/q₂ is lcm(p₁, p₂)/gcd(q₁, q₂)
                let num = num_integer::Integer::lcm(q.numer(), r.numer());
                let den = num_integer::Integer::gcd(q.denom(), r.denom());
                Ratio::new(num, den)
            }
        });
        if !atoms.contains(&id) {
            atoms.push(id);
        }
    }
    let ratio = ratio?;
    // `var` only inside the atoms, whose arguments are linear in it.
    let mut probe = expr;
    for (k, &a) in atoms.iter().enumerate() {
        let s = arena.symbol(&format!("__trig_atom_{k}"));
        probe = arena.subs_structural(probe, a, s);
    }
    if walk::contains(arena, probe, var) {
        return None;
    }
    for &a in &atoms {
        let arg = match *arena.node(a) {
            ExprNode::Sin(x) | ExprNode::Cos(x) | ExprNode::Tan(x) => x,
            _ => return None,
        };
        let slope = crate::transforms::diff::diff(arena, arg, var);
        let slope = crate::transforms::eval::eval(arena, slope);
        let at0 = crate::transforms::subs::subs(arena, arg, var, arena.zero);
        let at0 = crate::transforms::eval::eval(arena, at0);
        let lin = arena.mul(&[slope, var]);
        let lin = arena.add(&[lin, at0]);
        let d = arena.sub(arg, lin);
        let d = crate::transforms::expand::expand(arena, d);
        let d = crate::transforms::eval::eval(arena, d);
        if !arena.is_zero_structural(d) {
            return None;
        }
    }
    Some(ratio)
}

/// `expr rel 0` for an expression periodic in `var` ([`trig_period_ratio`]):
/// the critical points in one period `[0, T)` — the members there of the
/// families of zeros of the numerator, of the denominator (`tan` written
/// `sin/cos`) and of the domain constraints, from the general solver —
/// decide the sign on each arc between consecutive points (the last arc
/// wraps around to the first point plus `T`) and at each point, and the
/// maximal runs of accepted arcs and points repeat with the period:
/// `⋃ₙ ⟨lo + n·T, hi + n·T⟩` ([`crate::transforms::sets::periodic_interval_set`]),
/// with `T` reduced to `T/m` when the pattern repeats with it, each run
/// placed with its middle in `(−T/2, T/2]`.  `sin x > 1/2` is `⋃ₙ (π/6 +
/// 2nπ, 5π/6 + 2nπ)`; refused before 0.47 ("cannot build a sign chart"),
/// while SymPy's `solveset(sin(x) > 1/2, x, S.Reals)` gives the one period
/// `(π/6, 5π/6)`.  `None` when `expr` is not of that form; an error when
/// the zeros of a critical expression are not all found or a sign is not
/// decided.
fn try_solve_periodic_inequality(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    rel: Relation,
) -> Option<Result<ExprId, SymplexError>> {
    let ratio = trig_period_ratio(arena, expr, var)?;
    Some(solve_periodic_inequality(arena, expr, var, rel, &ratio))
}

/// [`try_solve_periodic_inequality`] for the period `2π·ratio`.
fn solve_periodic_inequality(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    rel: Relation,
    ratio: &Q,
) -> Result<ExprId, SymplexError> {
    let expr = crate::transforms::eval::eval(arena, expr);
    let two_r = arena.num_ratio(ratio * Ratio::from_integer(BigInt::from(2)));
    let period = arena.mul(&[two_r, arena.pi]);
    let period = crate::transforms::eval::eval(arena, period);
    let period_q =
        approx_real(arena, period).ok_or_else(|| failed("the period cannot be located".into()))?;
    // `tan u` as `sin u/cos u`: its poles are the zeros of a denominator.
    let mut rewritten = expr;
    for id in walk::post_order_ids(arena, expr) {
        if let ExprNode::Tan(u) = *arena.node(id)
            && walk::contains(arena, u, var)
        {
            let s = arena.sin(u);
            let c = arena.cos(u);
            let q = arena.div(s, c);
            rewritten = arena.subs_structural(rewritten, id, q);
        }
    }
    let rewritten = crate::transforms::eval::eval(arena, rewritten);
    let (numer, denom) = numer_denom(arena, rewritten);
    let constraints = domain_constraints(arena, rewritten, var)?;
    let mut critical: Vec<(ExprId, bool, bool)> = vec![(numer, true, false)];
    if denom != arena.one {
        critical.push((denom, false, true));
    }
    for c in &constraints {
        critical.push((c.expr, false, c.kind != ConstraintKind::NonNegative));
    }
    let mut points: Vec<Critical> = Vec::new();
    let mut identity = false;
    for (e, numer_zero, undefined) in critical {
        if !walk::contains(arena, e, var) {
            continue;
        }
        let members = match periodic_zeros_in_period(arena, e, var, period)? {
            Some(m) => m,
            None => {
                // `e` vanishes identically
                if numer_zero {
                    identity = true;
                    continue;
                }
                return Err(failed(format!("{} vanishes identically", arena.display(e))));
            }
        };
        add_roots(arena, &mut points, &members, numer_zero, undefined)?;
    }
    sort_points(&mut points);
    if points.len() > MAX_PERIODIC_POINTS {
        return Err(failed(format!(
            "too many critical points in one period ({})",
            points.len()
        )));
    }
    let branch = Branch {
        resolved: expr,
        constraints,
    };
    if identity {
        // `expr` is 0 wherever it is defined: the relation decides.
        if !rel.accepts(std::cmp::Ordering::Equal) {
            return Ok(arena.empty_set);
        }
    }
    let k = points.len();
    if k == 0 {
        // no zero, pole or boundary: one sign everywhere
        let p = arena.zero;
        let all = sample_satisfies(arena, &branch, var, p, rel)?;
        let (ninf, pinf) = (arena.neg_infinity, arena.infinity);
        return Ok(if all {
            arena.interval(ninf, pinf, IntervalKind::Open)
        } else {
            arena.empty_set
        });
    }
    // The arcs (pᵢ, pᵢ₊₁), the last one up to p₀ + T.
    let first_shifted = {
        let id = arena.add(&[points[0].id, period]);
        let id = crate::transforms::eval::eval(arena, id);
        Critical {
            id,
            approx: &points[0].approx + &period_q,
            exact: false,
            numer_zero: points[0].numer_zero,
            undefined: points[0].undefined,
        }
    };
    let mut arc_in: Vec<bool> = Vec::with_capacity(k);
    for i in 0..k {
        let lo = &points[i];
        let hi = if i + 1 < k {
            &points[i + 1]
        } else {
            &first_shifted
        };
        let mut decision: Option<bool> = None;
        for r in probe_ratios(Some(lo), Some(hi))? {
            let p = arena.num_ratio(r);
            let here = sample_satisfies(arena, &branch, var, p, rel)?;
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
        arc_in.push(decision.unwrap_or(false));
    }
    let mut point_in: Vec<bool> = Vec::with_capacity(k);
    for c in &points {
        let inside = point_in_domain(arena, &branch, var, c)?;
        point_in.push(inside && point_satisfies(arena, expr, var, c, rel)?);
    }
    // The smallest period `T/m` the pattern repeats with.
    let mut m_best = 1usize;
    for m in (2..=MAX_PERIOD_DIVISOR.min(k)).rev() {
        if !k.is_multiple_of(m) {
            continue;
        }
        let step = k / m;
        let shift = &period_q / Ratio::from_integer(BigInt::from(m as i64));
        let repeats = (0..k).all(|i| {
            let j = (i + step) % k;
            let target = if i + step >= k {
                &points[j].approx + &period_q
            } else {
                points[j].approx.clone()
            };
            approx_same(&(&points[i].approx + &shift), &target)
                && point_in[i] == point_in[j]
                && arc_in[i] == arc_in[j]
        });
        if repeats {
            m_best = m;
            break;
        }
    }
    let (k, period, period_q) = if m_best > 1 {
        let m_id = arena.int(m_best as i64);
        let p = arena.div(period, m_id);
        let p = crate::transforms::eval::eval(arena, p);
        (
            k / m_best,
            p,
            &period_q / Ratio::from_integer(BigInt::from(m_best as i64)),
        )
    } else {
        (k, period, period_q)
    };
    // Elements in cyclic order: point i is 2i, arc i is 2i + 1.
    let included = |e: usize| -> bool {
        let e = e % (2 * k);
        if e.is_multiple_of(2) {
            point_in[e / 2]
        } else {
            arc_in[e / 2]
        }
    };
    if (0..2 * k).all(included) {
        let (ninf, pinf) = (arena.neg_infinity, arena.infinity);
        return Ok(arena.interval(ninf, pinf, IntervalKind::Open));
    }
    // Every arc in, some points out: the line without their families.
    if (0..k).all(|i| arc_in[i]) {
        let (ninf, pinf) = (arena.neg_infinity, arena.infinity);
        let line = arena.interval(ninf, pinf, IntervalKind::Open);
        let mut holes: Vec<ExprId> = Vec::new();
        for (i, c) in points.iter().take(k).enumerate() {
            if !point_in[i] {
                holes.push(point_family(arena, c.id, period));
            }
        }
        let holes = arena.set_union(&holes);
        let set = arena.set_complement(line, holes);
        return Ok(crate::transforms::sets::simplify_set(arena, set));
    }
    // Start just after an excluded element, collect the maximal runs.
    let start = (0..2 * k).find(|&e| !included(e)).unwrap_or(0) + 1;
    let mut parts: Vec<ExprId> = Vec::new();
    let mut e = start;
    while e < start + 2 * k {
        if !included(e) {
            e += 1;
            continue;
        }
        let run_start = e;
        while e < start + 2 * k && included(e) {
            e += 1;
        }
        let run_end = e - 1;
        // the end of element `idx` (index into the doubled cycle) and its
        // number of whole periods past the base window
        let end_of = |idx: usize, upper: bool| -> (usize, i64) {
            let pt = if idx.is_multiple_of(2) {
                idx / 2
            } else if upper {
                idx / 2 + 1
            } else {
                idx / 2
            };
            (pt % k, (pt / k) as i64)
        };
        if run_start == run_end && run_start % 2 == 0 {
            let (pt, wraps) = end_of(run_start, false);
            let at = shifted(arena, points[pt].id, period, wraps);
            parts.push(point_family(arena, at, period));
            continue;
        }
        let (lo_pt, lo_wraps) = end_of(run_start, false);
        let (hi_pt, hi_wraps) = end_of(run_end, true);
        let lo_open = run_start % 2 == 1;
        let hi_open = run_end % 2 == 1;
        let mid = (&points[lo_pt].approx
            + &period_q * Ratio::from_integer(BigInt::from(lo_wraps))
            + &points[hi_pt].approx
            + &period_q * Ratio::from_integer(BigInt::from(hi_wraps)))
            / Ratio::from_integer(BigInt::from(2));
        // place the run with its middle in (−T/2, T/2]
        let half = &period_q / Ratio::from_integer(BigInt::from(2));
        let turns = ((&mid + &half) / &period_q).ceil() - Ratio::one();
        let turns = turns.to_integer();
        let turns = num_traits::ToPrimitive::to_i64(&turns).unwrap_or(0);
        let lo = shifted(arena, points[lo_pt].id, period, lo_wraps - turns);
        let hi = shifted(arena, points[hi_pt].id, period, hi_wraps - turns);
        let kind = IntervalKind::from_open_ends(lo_open, hi_open);
        parts.push(crate::transforms::sets::periodic_interval_set(
            arena, lo, hi, kind, period,
        ));
    }
    let union = arena.set_union(&parts);
    Ok(crate::transforms::sets::simplify_set(arena, union))
}

/// `p + k·period`, evaluated.
fn shifted(arena: &mut Arena, p: ExprId, period: ExprId, k: i64) -> ExprId {
    if k == 0 {
        return p;
    }
    let kk = arena.int(k);
    let s = arena.mul(&[kk, period]);
    let sum = arena.add(&[p, s]);
    crate::transforms::eval::eval(arena, sum)
}

/// The point family `{p + n·period : n ∈ ℤ}`.
fn point_family(arena: &mut Arena, p: ExprId, period: ExprId) -> ExprId {
    let scope = arena.finite_set(&[p, period]);
    let zero = arena.zero;
    let n = crate::transforms::sets::image_param(arena, zero, scope);
    let pn = arena.mul(&[period, n]);
    let body = arena.add(&[p, pn]);
    crate::transforms::sets::image_set(arena, n, body)
}

/// The real zeros of `e` (periodic in `var` with the period `period`) in
/// `[0, period)`, from the general solver: the members of its families
/// there and its other solutions reduced into the window, exactly (the
/// families located as image sets, every zero `z` reduced to `z −
/// ⌊z/period⌋·period` with a certified floor, its realness certified).
/// `Ok(None)` when `e` vanishes identically; an error when its zeros are
/// not found or one cannot be located.
///
/// Up to 0.53 the families were located by `f64` values cast with `as
/// i64`, and a zero with an imaginary part below `10⁻¹²` of its value was
/// real: `sin(x − 10²⁰) > 0` (members `n ≈ −3.18·10¹⁹`) found no zero in
/// a period and was "every real `x`".
fn periodic_zeros_in_period(
    arena: &mut Arena,
    e: ExprId,
    var: ExprId,
    period: ExprId,
) -> Result<Option<Vec<ExprId>>, SymplexError> {
    use crate::transforms::solve::{Reason, RootSet as Roots, Solutions};
    let scope = arena.finite_set(&[e, var]);
    let zero = arena.zero;
    let n = crate::transforms::sets::image_param(arena, zero, scope);
    let outcome = crate::transforms::solve::solve_general(arena, e, var, n);
    let sols = match outcome {
        Solutions::Complete(Roots::All) => return Ok(None),
        Solutions::Complete(Roots::Empty(_)) => return Ok(Some(Vec::new())),
        complete @ Solutions::Complete(_) => complete.claimed_points(),
        // every zero is needed: a partial outcome is refused
        Solutions::Refused(Reason::NotFound) | Solutions::Partial(..) => {
            return Err(failed(format!(
                "could not find the zeros of {}",
                arena.display(e)
            )));
        }
        Solutions::Refused(reason) => {
            return Err(failed(format!(
                "could not find the zeros of {}: {}",
                arena.display(e),
                reason.text()
            )));
        }
    };
    let window = crate::transforms::sets::Window::new(arena, zero, period)
        .ok_or_else(|| failed("the period cannot be located".into()))?;
    let mut out: Vec<ExprId> = Vec::new();
    for s in sols {
        let v = crate::transforms::eval::eval(arena, s);
        let family = walk::contains(arena, v, n);
        let zeros = if family {
            // its real members in [0, period], with a margin
            let members = crate::transforms::sets::image_set_points_in(arena, n, v, window)
                .ok_or_else(|| {
                    failed(format!(
                        "the family {} cannot be located on the real line",
                        arena.display(v)
                    ))
                })?;
            if members.len() > 4 * MAX_PERIODIC_POINTS {
                return Err(failed(format!(
                    "the family {} has too many members in one period",
                    arena.display(v)
                )));
            }
            members
        } else {
            match crate::transforms::realness::imaginary_part_sign(arena, v) {
                SignSet::ZERO => {}
                // not real (a real variable does not take it)
                s if s.is_nonzero().is_true() => continue,
                _ => {
                    return Err(failed(format!(
                        "the zero {} cannot be located on the real line",
                        arena.display(v)
                    )));
                }
            }
            vec![v]
        };
        for z in zeros {
            // z − ⌊z/period⌋·period, in [0, period)
            let q = arena.div(z, period);
            let turns = arena.floor(q);
            let turns = crate::transforms::eval::eval(arena, turns);
            let Some(t) = arena.as_num(turns).filter(|t| t.is_integer()).cloned() else {
                return Err(failed(format!(
                    "the zero {} cannot be reduced into one period",
                    arena.display(z)
                )));
            };
            if t.is_zero() {
                out.push(z);
            } else if !family {
                // a zero of a periodic function: its translate by the period
                let back = arena.mul(&[turns, period]);
                let back = arena.neg(back);
                let sum = arena.add(&[z, back]);
                out.push(crate::transforms::eval::eval(arena, sum));
            }
            // (a member outside the period: another member is its translate)
        }
    }
    Ok(Some(out))
}

/// Solve `expr = 0`, returning the solution set as an `ExprId`.
///
/// - Identity `0 = 0` → `UniversalSet`
/// - Provably no solution → `EmptySet`
/// - An algebraic equation (`var` only in sums, products and powers with
///   numeric exponents) → the `FiniteSet` of the roots `solve` returns
/// - Any other equation → its general solution ([`general_solution_set`]),
///   over the declared domain of `var` (ℂ for a plain symbol): the union
///   of a finite set and image sets `{f(n) : n ∈ ℤ}` for the periodic
///   families (`sin x = 0`: `{n·π : n ∈ ℤ}`; before 0.40 the principal
///   solutions `{0, π}`, which claimed there were no others), a family
///   minus the poles on it (`sin(x)/x`: `{n·π} \ {0}`)
/// - No root found, or the solver could not decide → `ConditionSet(x, Eq(expr,
///   0))`: the set is not known.  Before 0.39 it was `EmptySet`, a claim
///   that there is no solution: `(x − cos x).solve_as_set(x)` (`x ≈ 0.739`
///   solves it) and `x⁵ − x − 1` (five roots, none in radicals).
///
/// The set is restricted to the declared domain of `var`
/// ([`restrict_set_to_domain`](crate::transforms::sets::restrict_set_to_domain)):
/// for a positive `x`, `x² − 4` gives `{2}` and `sin x` the image set
/// `{nπ}` intersected with `(0, ∞)`.  Up to 0.42 the assumptions of `var`
/// other than realness were ignored (`{−2, 2}`).
pub(crate) fn solveset(arena: &mut Arena, expr: ExprId, var: ExprId) -> ExprId {
    let set = solveset_over_all_values(arena, expr, var);
    if walk::has_unevaluated(arena, set) {
        return set;
    }
    crate::transforms::sets::restrict_set_to_domain(arena, var, set).unwrap_or_else(|| {
        let zero = arena.zero;
        let cond = arena.eq_(expr, zero);
        arena.intern(ExprNode::ConditionSet(var, cond))
    })
}

/// [`solveset`] before the restriction to the declared domain of `var`.
fn solveset_over_all_values(arena: &mut Arena, expr: ExprId, var: ExprId) -> ExprId {
    let unknown = |arena: &mut Arena| {
        let zero = arena.zero;
        let cond = arena.eq_(expr, zero);
        arena.intern(ExprNode::ConditionSet(var, cond))
    };
    if !is_algebraic_in(arena, expr, var) && walk::has_free_var(arena, expr, var) {
        return general_solution_set(arena, expr, var, true).unwrap_or_else(|| unknown(arena));
    }
    // SymPy's convention: the set, the set ∪ ConditionSet when partial,
    // the ConditionSet when refused.
    crate::transforms::solve::solve_classified(arena, expr, var).into_set(arena, var, expr)
}

/// `expr rel 0` for an expression with non-real coefficients (`i`), which
/// holds only where `expr` is real: at the real zeros of its imaginary part
/// (for a real `var`) at which its real part satisfies the relation.  Up
/// to 0.46 the sign chart sampled `expr` between the zeros of `expr`
/// itself, found it non-real everywhere and answered `EmptySet`:
/// `((x − 1)(x − 2) + i)·((x − 1)(x − 2) + 2i) < 0` holds at `x = 1` and
/// `x = 2`, where it is `−2` (SymPy: `{1, 2}`).  `None` when `expr` has no
/// `i`, its parts are not found, or its imaginary part vanishes
/// identically.
fn try_solve_nonreal_inequality(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    rel: Relation,
) -> Option<Result<ExprId, SymplexError>> {
    let has_i = walk::post_order_ids(arena, expr)
        .into_iter()
        .any(|id| matches!(arena.node(id), ExprNode::ImaginaryUnit));
    if !has_i || !walk::contains(arena, expr, var) {
        return None;
    }
    // The parts for a real variable: a real stand-in for `var`.
    let t = crate::transforms::subs::fresh_symbol(arena, "__ineq_real", &[expr]);
    if let ExprNode::Symbol(sid) = *arena.node(t) {
        let mut real = crate::base::assumptions::Assumptions::default();
        real.assert_true(crate::base::assumptions::Props::REAL);
        arena.set_symbol_assumptions(sid, real);
    }
    let e_t = crate::transforms::subs::subs(arena, expr, var, t);
    let expanded = crate::transforms::expand::expand(arena, e_t);
    let (_, im) = crate::base::complex::as_real_imag(arena, expanded);
    let im = crate::transforms::eval::eval(arena, im);
    let opaque = walk::post_order_ids(arena, im).into_iter().any(|id| {
        matches!(
            arena.node(id),
            ExprNode::Re(_) | ExprNode::Im(_) | ExprNode::Arg(_) | ExprNode::Conjugate(_)
        )
    });
    if opaque || arena.is_zero_structural(im) {
        return None;
    }
    let (im_numer, _) = numer_denom(arena, im);
    // Only a polynomial imaginary part has its real zeros all found (a
    // Sturm count); `sin t` from `e^(i·x)` would give its principal zeros.
    crate::poly::polybridge::expr_to_poly(arena, im_numer, t)?;
    let roots = match real_roots(arena, im_numer, t) {
        Ok(RootSet::Finite(r)) => r,
        // the imaginary part vanishes identically: `expr` is real
        Ok(RootSet::Everywhere) => return None,
        Err(e) => return Some(Err(e)),
    };
    let mut kept: Vec<ExprId> = Vec::new();
    for r in roots {
        let at = crate::transforms::subs::subs(arena, expr, var, r);
        let at = crate::transforms::eval::eval(arena, at);
        let value = value_sign(arena, at);
        match value.ordering() {
            Some(o) => {
                if rel.accepts(o) {
                    kept.push(r);
                }
            }
            None if value.takes_no_real_value() => {}
            None => {
                return Some(Err(failed(format!(
                    "the sign of {} at {} is not decided",
                    arena.display(expr),
                    arena.display(r)
                ))));
            }
        }
    }
    Some(Ok(arena.finite_set(&kept)))
}

/// The set of real `var` where every expression of `sides` is defined
/// (denominators nonzero, radicands of fractional powers nonnegative,
/// `ln` arguments positive, …), when it may be smaller than that of
/// `combined` (their difference, in which singular terms can cancel:
/// `x + 1/x` and `−2 + 1/x` differ by `x + 2`).  `Ok(None)` when every
/// constraint of `sides` is one of `combined` (nothing is lost) or there
/// is none, also when a side has piecewise-defined pieces (not looked
/// into).  An error when a constraint cannot be solved.
pub(crate) fn lost_domain(
    arena: &mut Arena,
    sides: &[ExprId],
    combined: ExprId,
    var: ExprId,
) -> Result<Option<ExprId>, SymplexError> {
    if !kink_expressions(arena, combined, var).is_empty() {
        return Ok(None);
    }
    let have = domain_constraints(arena, combined, var)?;
    let mut lost: Vec<Constraint> = Vec::new();
    for &s in sides {
        if !walk::contains(arena, s, var) {
            continue;
        }
        if !kink_expressions(arena, s, var).is_empty() {
            return Ok(None);
        }
        for c in domain_constraints(arena, s, var)? {
            let known = have
                .iter()
                .chain(lost.iter())
                .any(|h| h.expr == c.expr && h.kind == c.kind);
            if !known {
                lost.push(c);
            }
        }
    }
    if lost.is_empty() {
        return Ok(None);
    }
    let mut parts: Vec<ExprId> = Vec::with_capacity(lost.len());
    for c in lost {
        let set = match c.kind {
            ConstraintKind::Positive => {
                solve_inequality_over_reals(arena, c.expr, var, Relation::Gt)?
            }
            ConstraintKind::NonNegative => {
                solve_inequality_over_reals(arena, c.expr, var, Relation::Ge)?
            }
            ConstraintKind::NonZero => {
                let above = solve_inequality_over_reals(arena, c.expr, var, Relation::Gt)?;
                let below = solve_inequality_over_reals(arena, c.expr, var, Relation::Lt)?;
                arena.set_union(&[above, below])
            }
        };
        parts.push(set);
    }
    let meet = arena.set_intersection(&parts);
    Ok(Some(crate::transforms::sets::simplify_set(arena, meet)))
}

/// Does `var` occur in `expr` only in sums, products and powers with a
/// numeric exponent (a radical of a rational function)?
fn is_algebraic_in(arena: &Arena, expr: ExprId, var: ExprId) -> bool {
    let mut stack = vec![expr];
    let mut seen: rustc_hash::FxHashSet<ExprId> = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) || !walk::has_free_var(arena, id, var) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Symbol(_) => {}
            ExprNode::Add(ch) | ExprNode::Mul(ch) => stack.extend(ch.iter().copied()),
            ExprNode::Neg(a) => stack.push(*a),
            ExprNode::Pow(b, e) if arena.as_num(*e).is_some() => stack.push(*b),
            _ => return false,
        }
    }
    true
}

/// The solution set of `expr = 0` from its general solution: over the
/// declared domain of `var` (ℂ for a plain symbol) when `over_c`
/// ([`solve_general_over_c`](crate::transforms::solve::solve_general_over_c)),
/// with the real-variable conventions of `solve_general` otherwise:
/// the finite set of the solutions free of the integer parameter united
/// with the image set of each family, simplified (families merged:
/// `2nπ ∪ 2nπ + π = nπ`).  Every member of a family solves the equation
/// (`solve_general` checks them).
///
/// When a family also holds poles of a polynomial denominator, which no
/// family over all integers can exclude (`sin(x)/x = 0`: `2nπ` holds `0`),
/// `expr = g·∏ qᵢ^(−kᵢ)` is split: the set of `g = 0` minus the zeros of
/// the `qᵢ` that are in it, `{nπ} \ {0}`, as SymPy's `solveset` removes
/// the zeros of denominators.  `None` (unknown) when nothing is found or
/// the general solve refuses otherwise.
pub(crate) fn general_solution_set(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    over_c: bool,
) -> Option<ExprId> {
    match general_solve_set(arena, expr, var, over_c) {
        Some(GeneralSet::Set(s)) => Some(s),
        Some(GeneralSet::Refused) => {
            let (g, denominators) = split_polynomial_denominators(arena, expr, var)?;
            let Some(GeneralSet::Set(g_set)) = general_solve_set(arena, g, var, over_c) else {
                return None;
            };
            // every pole is needed: the zeros of each denominator, complete
            let mut poles: Vec<ExprId> = Vec::new();
            for q in denominators {
                poles.extend(
                    crate::transforms::solve::solve_classified(arena, q, var)
                        .points_if_complete()?,
                );
            }
            let mut excluded = Vec::new();
            for p in poles {
                match crate::transforms::sets::set_contains(arena, g_set, p) {
                    // written as the set writes its member (`π`, not `√(π²)`)
                    Some(true) => excluded
                        .push(crate::transforms::sets::member_form(arena, g_set, p).unwrap_or(p)),
                    Some(false) => {}
                    None => return None,
                }
            }
            if excluded.is_empty() {
                return Some(g_set);
            }
            let excluded = arena.finite_set(&excluded);
            let set = arena.set_complement(g_set, excluded);
            Some(crate::transforms::sets::simplify_set(arena, set))
        }
        None => None,
    }
}

/// Outcome of [`general_solve_set`].
enum GeneralSet {
    /// The solution set.
    Set(ExprId),
    /// The general solve refused (a family mixing solutions and poles).
    Refused,
}

/// The general solution of `expr = 0` as a set (see
/// [`general_solution_set`]); `None` when nothing was found.
fn general_solve_set(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    over_c: bool,
) -> Option<GeneralSet> {
    use crate::transforms::solve::{Reason, Solutions};
    let scope = arena.finite_set(&[expr, var]);
    let zero = arena.zero;
    let n = crate::transforms::sets::image_param(arena, zero, scope);
    let outcome = if over_c {
        crate::transforms::solve::solve_general_over_c(arena, expr, var, n)
    } else {
        crate::transforms::solve::solve_general(arena, expr, var, n)
    };
    match outcome {
        Solutions::Refused(Reason::NotFound) => None,
        // roots under a condition on the parameters: an exact set
        conditional if conditional.is_conditional() => {
            Some(GeneralSet::Set(conditional.into_set(arena, var, expr)))
        }
        Solutions::Refused(_) => Some(GeneralSet::Refused),
        // (the union with the condition set; `Solutions::into_set`)
        partial @ Solutions::Partial(..) => {
            Some(GeneralSet::Set(partial.into_set(arena, var, expr)))
        }
        complete @ Solutions::Complete(_) => {
            Some(GeneralSet::Set(complete.into_set(arena, var, expr)))
        }
    }
}

/// Is `e` a polynomial in `var`, with coefficients free of it (`x − 10π`)?
fn is_polynomial_in(arena: &mut Arena, e: ExprId, var: ExprId) -> bool {
    if crate::poly::polybridge::expr_to_poly(arena, e, var).is_some() {
        return true;
    }
    crate::transforms::solve::symbolic_poly_coeffs(arena, e, var)
        .is_some_and(|cs| cs.iter().all(|&c| !walk::has_free_var(arena, c, var)))
}

/// `expr = g·∏ qᵢ^(−kᵢ)` with polynomials `qᵢ` in `var`: `(g, [qᵢ])`, or
/// `None` when `expr` is not a product with such a factor.
fn split_polynomial_denominators(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
) -> Option<(ExprId, Vec<ExprId>)> {
    let ExprNode::Mul(factors) = arena.node(expr).clone() else {
        return None;
    };
    let mut keep: Vec<ExprId> = Vec::new();
    let mut denominators: Vec<ExprId> = Vec::new();
    for f in factors {
        if let ExprNode::Pow(base, e) = *arena.node(f)
            && arena.as_num(e).is_some_and(|r| r.is_negative())
            && walk::has_free_var(arena, base, var)
            && is_polynomial_in(arena, base, var)
        {
            denominators.push(base);
        } else {
            keep.push(f);
        }
    }
    if denominators.is_empty() || keep.is_empty() {
        return None;
    }
    let g = arena.mul(&keep);
    Some((g, denominators))
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

impl ConstraintKind {
    /// The signs the constrained expression may take: real ones only (a
    /// variable on the real line takes no point where a denominator,
    /// radicand or logarithm argument is not real).
    fn allowed(self) -> SignSet {
        match self {
            ConstraintKind::NonZero => SignSet::NONZERO_REAL,
            ConstraintKind::NonNegative => SignSet::NONNEGATIVE,
            ConstraintKind::Positive => SignSet::POSITIVE,
        }
    }
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

/// The value of a variable-free expression ([`value_sign`]): a finite value
/// with the signs it may have, or no finite value.  A [`SignSet`] describes
/// finite values only (`base::truth`, "Infinities"), so `±∞`, `zoo`, NaN
/// and booleans are the case of their own, [`Undefined`](Self::Undefined).
/// Up to 0.49 a six-variant enum (`Neg`, `Zero`, `Pos`, `NonReal`,
/// `Undefined`, `Unknown`) compared with `==` at each use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ValueSign {
    /// A finite value: one atom when its sign is decided, [`SignSet::UNKNOWN`]
    /// when it is not (symbolic, unevaluable, or numerically
    /// indistinguishable from zero).
    Defined(SignSet),
    /// Infinite, NaN or a boolean.
    Undefined,
}

impl ValueSign {
    /// A value whose sign is not decided.
    const UNDECIDED: ValueSign = ValueSign::Defined(SignSet::UNKNOWN);

    /// The decided sign of a real value against 0.
    fn ordering(self) -> Option<std::cmp::Ordering> {
        match self {
            ValueSign::Defined(s) => s.to_ordering(),
            ValueSign::Undefined => None,
        }
    }

    /// Is the value finite with a sign in `property`?  `False` for no
    /// finite value.
    fn is_in(self, property: SignSet) -> Truth {
        match self {
            ValueSign::Defined(s) => s.is_in(property),
            ValueSign::Undefined => Truth::False,
        }
    }

    /// Is the value finite and certainly not real?
    fn is_nonreal(self) -> Truth {
        self.is_in(SignSet::NONREAL)
    }

    /// Is the value certainly not taken by a real variable: not real, or no
    /// finite value?
    fn takes_no_real_value(self) -> bool {
        self == ValueSign::Undefined || self.is_nonreal().is_true()
    }

    /// Is the value finite with an undecided sign?
    fn undecided(self) -> bool {
        matches!(self, ValueSign::Defined(s) if !s.is_single())
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
                merge_point(arena, &mut points, np)?;
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
        let branch = branch_for(&branches, &kink_points, &c.approx);
        let inside = point_in_domain(arena, branch, var, c)?;
        point_in.push(inside && point_satisfies(arena, expr, var, c, rel)?);
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
        let kind = IntervalKind::from_open_ends(!lo_closed, !hi_closed);
        pieces.push(arena.interval(lo, hi, kind));
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
/// expressions rely on the equation solver, asked for the zeros with `var`
/// real ([`Domain::Real`](crate::transforms::solve::Domain::Real)), and
/// fail when it cannot find them.  (Up to 0.48 it asked for the principal
/// roots in the declared domain of `var`, `Ex::solve`'s, and got the real
/// ones only because the inversions it reached took `var` real anyway.)
fn real_roots(arena: &mut Arena, e: ExprId, var: ExprId) -> Result<RootSet, SymplexError> {
    use crate::transforms::solve::{Domain, Missing, Reason, Solutions, SolveCx};

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
                if !value_sign(arena, at).is_in(SignSet::ZERO).is_true() {
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
        // (Checked against the Sturm count below.)
        let candidates = crate::transforms::solve::known_real_roots(arena, e, var);
        let mut reals: Vec<(ExprId, Q)> = Vec::new();
        for s in candidates {
            if value_sign(arena, s).takes_no_real_value() {
                continue;
            }
            let Some(a) = approx_real(arena, s) else {
                return Err(failed(format!(
                    "root {} cannot be evaluated numerically",
                    arena.display(s)
                )));
            };
            if !reals.iter().any(|(_, b)| approx_same(&a, b)) {
                reals.push((s, a));
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

    // A sign chart needs every zero: a partial outcome is an error.
    let because = |reason: &Reason| match reason {
        Reason::NotFound => String::new(),
        other => format!(": {}", other.text()),
    };
    match crate::transforms::solve::solutions(arena, e, var, &SolveCx::principal(Domain::Real)) {
        Solutions::Complete(crate::transforms::solve::RootSet::All) => Ok(RootSet::Everywhere),
        Solutions::Complete(crate::transforms::solve::RootSet::Empty(_)) => {
            Ok(RootSet::Finite(Vec::new()))
        }
        Solutions::Refused(reason) => Err(failed(format!(
            "could not find the zeros of {}{}",
            arena.display(e),
            because(&reason)
        ))),
        Solutions::Partial(_, Missing::Alternative(reason)) => Err(failed(format!(
            "could not find all the zeros of {}{}",
            arena.display(e),
            because(&reason)
        ))),
        complete @ Solutions::Complete(_) => {
            let mut out = Vec::new();
            for sol in complete.claimed_points() {
                let value = value_sign(arena, sol);
                if value.takes_no_real_value() {
                    continue;
                }
                if value.undecided() && approx_real(arena, sol).is_none() {
                    return Err(failed(format!(
                        "zero {} of {} cannot be located on the real line",
                        arena.display(sol),
                        arena.display(e)
                    )));
                }
                out.push(sol);
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
            arena,
            points,
            Critical {
                id: ev,
                approx,
                exact,
                numer_zero,
                undefined,
            },
        )?;
    }
    Ok(())
}

/// Insert a point, merging it with an existing one it equals.  Points whose
/// approximations agree to [`MERGE_REL`] are merged only when their
/// difference is proved 0 (`constant_zero_decision`), kept apart when it is
/// proved nonzero (the sample points between them then refuse if they
/// cannot be placed), and refused otherwise.  Up to 0.53 they were merged
/// on the approximations alone: `(x − 1)(x − 1 − e⁻⁷⁰) < 0` was `EmptySet`
/// (it is `(1, 1 + e⁻⁷⁰)`).
fn merge_point(
    arena: &mut Arena,
    points: &mut Vec<Critical>,
    np: Critical,
) -> Result<(), SymplexError> {
    let mut same: Option<usize> = None;
    for (i, c) in points.iter().enumerate() {
        if !approx_same(&c.approx, &np.approx) {
            continue;
        }
        if c.id == np.id {
            same = Some(i);
            break;
        }
        let d = arena.sub(c.id, np.id);
        match crate::transforms::eval::constant_zero_decision(arena, d) {
            Some(true) => {
                same = Some(i);
                break;
            }
            Some(false) => {}
            None => {
                return Err(failed(format!(
                    "cannot decide whether the critical points {} and {} are equal",
                    arena.display(c.id),
                    arena.display(np.id)
                )));
            }
        }
    }
    if let Some(existing) = same.and_then(|i| points.get_mut(i)) {
        existing.numer_zero |= np.numer_zero;
        existing.undefined |= np.undefined;
        if np.exact && !existing.exact {
            existing.id = np.id;
            existing.approx = np.approx;
            existing.exact = true;
        }
        return Ok(());
    }
    points.push(np);
    Ok(())
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
    out.sort_by(|&a, &b| arena.canonical_cmp(a, b));
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
                // `|a|` of an `a` that is not real at the sample point has no
                // switching point there and stays (`|x + i|`; up to 0.49 the
                // sign chart failed on "the sign of x + I").
                ExprNode::Abs(a) => {
                    let at = crate::transforms::subs::subs(arena, a, var, p);
                    if value_sign(arena, at).is_nonreal().is_true() {
                        continue;
                    }
                    match sign_at(arena, a, var, p)? {
                        std::cmp::Ordering::Greater => a,
                        std::cmp::Ordering::Less => arena.neg(a),
                        std::cmp::Ordering::Equal => return Err(kink_at_sample(arena, id)),
                    }
                }
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
        if s.undecided() {
            return Err(failed(format!(
                "cannot decide the domain constraint {} at a sample point",
                arena.display(c.expr)
            )));
        }
        if !s.is_in(c.kind.allowed()).is_true() {
            return Ok(false);
        }
    }
    let at = crate::transforms::subs::subs(arena, branch.resolved, var, p);
    let value = value_sign(arena, at);
    if value.takes_no_real_value() {
        return Ok(false);
    }
    match value.ordering() {
        Some(o) => Ok(rel.accepts(o)),
        None => Err(failed(format!(
            "cannot decide the sign of {} at a sample point",
            arena.display(branch.resolved)
        ))),
    }
}

/// Does the critical point `c` satisfy the natural-domain constraints of
/// its branch?  A zero of the numerator outside the domain is not a
/// solution even when the expression folds to `0` there: before 0.31
/// `(x + 2)/(ln x + 2) ≥ 0` contained `x = −2`, where `ln(−2)` is not
/// real and `0·(ln(−2) + 2)⁻¹` folded to `0` (sample points were checked,
/// critical points were not).
fn point_in_domain(
    arena: &mut Arena,
    branch: &Branch,
    var: ExprId,
    c: &Critical,
) -> Result<bool, SymplexError> {
    if c.undefined {
        return Ok(false);
    }
    for k in &branch.constraints {
        let at = crate::transforms::subs::subs(arena, k.expr, var, c.id);
        let s = value_sign(arena, at);
        if s.takes_no_real_value() {
            return Ok(false);
        }
        if s.undecided() {
            return Err(failed(format!(
                "cannot decide the domain constraint {} at {}",
                arena.display(k.expr),
                arena.display(c.id)
            )));
        }
        if !s.is_in(k.kind.allowed()).is_true() {
            return Ok(false);
        }
    }
    Ok(true)
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
    let value = value_sign(arena, at);
    if value.takes_no_real_value() {
        return Ok(false);
    }
    match value.ordering() {
        Some(o) => Ok(rel.accepts(o)),
        None if c.numer_zero => Ok(rel.accepts(std::cmp::Ordering::Equal)),
        None => Err(failed(format!(
            "cannot decide the sign of the expression at {}",
            arena.display(c.id)
        ))),
    }
}

/// Classify the value of a variable-free expression: exactly when it
/// evaluates to a rational, otherwise to [`DECISION_DIGITS`] digits.  A
/// value whose evaluation keeps an imaginary part is non-real, however small
/// the part next to the real one: a part that is kept has a ball that
/// excludes 0.  Up to 0.57 a part below `10⁻²⁰` of the value was taken for
/// 0 — `x + i·e⁻⁶⁰ > 0` was `(−i·e⁻⁶⁰, ∞)` (it holds for no real `x`).
fn value_sign(arena: &mut Arena, e: ExprId) -> ValueSign {
    let real_sign = |q: &Q| ValueSign::Defined(SignSet::from_ordering(q.cmp(&Q::zero())));
    let ev = crate::transforms::eval::eval(arena, e);
    if let Some(r) = arena.as_num(ev) {
        return real_sign(r);
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
        return ValueSign::UNDECIDED;
    }
    let Some((value, first)) = real_part(arena, ev, DECISION_DIGITS) else {
        return ValueSign::UNDECIDED;
    };
    if value.im_sign().is_nonzero().is_true() {
        return ValueSign::Defined(SignSet::NONREAL);
    }
    if first.is_zero() {
        return ValueSign::Defined(SignSet::ZERO);
    }
    // A part that is not 0 has its sign certified (its ball excludes 0).
    if first.magnitude() >= -(AMBIGUOUS_EXP as i64) {
        return ValueSign::Defined(value.re_sign());
    }
    // Tiny: either a genuine small value (`exp(-1000)`) or cancellation
    // noise around an exact zero.  Re-evaluate at twice the precision: a
    // genuine value reproduces itself, noise shrinks by ~30 digits.
    let Some((_, second)) = real_part(arena, ev, 2 * DECISION_DIGITS) else {
        return ValueSign::UNDECIDED;
    };
    if second.is_zero() || first.is_negative() != second.is_negative() {
        return ValueSign::UNDECIDED;
    }
    // |first − second| > |first|/10⁶
    if first.differs_relatively(&second, 6) {
        return ValueSign::UNDECIDED;
    }
    ValueSign::Defined(value.re_sign())
}

/// The value of the variable-free `ev` to `digits` digits
/// ([`const_value`](crate::transforms::evalf::const_value)) with the decimal
/// digits of its real part, exactly those `evalf` prints, kept unscaled:
/// the real part of a value at a far sample point has hundreds of thousands
/// of digits (`1001·exp(1001²)` for `x·e^(x²) − 1 > 0`), and up to 0.55
/// its exact rational was formed — `10^(4.3·10⁸)` for `exp(1000³)`, which
/// never finished (`exp(x³) ≥ 0` hung).  Up to 0.57 the printed string was
/// parsed.
fn real_part(arena: &Arena, ev: ExprId, digits: u32) -> Option<(ConstValue, Decimal)> {
    let value = crate::transforms::evalf::const_value(arena, ev, digits).ok()?;
    let re = value.re_decimal()?;
    Some((value, re))
}

/// The largest decimal exponent of an approximation [`approx_real`] turns
/// into an exact rational (`10^±10⁶` has 3.3 million bits).  Beyond it the
/// point is refused: up to 0.57 the power of ten was formed whatever its
/// size, and `x − exp(10⁹) > 0` (a critical point near `10^434294481`) never
/// finished.
const MAX_APPROX_EXP: u64 = 1_000_000;

/// Real value of a variable-free expression as an exact rational (when
/// it evaluates to one) or a [`DECISION_DIGITS`]-digit approximation
/// (the rational of the decimal `evalf` prints; `None` beyond
/// [`MAX_APPROX_EXP`]).
fn approx_real(arena: &mut Arena, e: ExprId) -> Option<Q> {
    let ev = crate::transforms::eval::eval(arena, e);
    if let Some(r) = arena.as_num(ev) {
        return Some(r.clone());
    }
    if !walk::free_symbols(arena, ev).is_empty() || walk::has_unevaluated(arena, ev) {
        return None;
    }
    let (value, re) = real_part(arena, ev, DECISION_DIGITS)?;
    if value.im_sign().is_nonzero().is_true() {
        return None;
    }
    re.to_ratio(MAX_APPROX_EXP)
}

// ═══════════════════════════════════════════════════════════════════════════
// Absolute-value inequalities: c·|f(x)| + d  rel  0  with f linear in x
// ═══════════════════════════════════════════════════════════════════════════

/// Is `e` certainly real, its symbols without assumptions taken real (the
/// parameters of an inequality)?
fn real_taking_symbols_real(arena: &Arena, e: ExprId) -> bool {
    use crate::base::assumptions::{AssumptionCache, Assumptions, Props};
    let mut cache = AssumptionCache::new();
    for s in walk::free_symbols(arena, e) {
        if let ExprNode::Symbol(sid) = *arena.node(s)
            && arena.symbol_assumptions(sid) == Assumptions::default()
        {
            let mut real = Assumptions::default();
            real.known_true |= Props::REAL;
            cache.set_symbol_assumptions(s, real);
        }
    }
    cache.query(arena, e, Props::REAL) == Some(true)
}

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
/// Returns `None` if the expression does not have this shape, `a·x + b`
/// is not certainly real, or the sign of the leading coefficient `a`
/// cannot be determined.
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
    // `|a·x + b| = ±(a·x + b)` only for a real `a·x + b`: `|x + i| > 2` is
    // `x² + 1 > 4` (the sign chart), not `x + i > 2` (up to 0.49 the set
    // `(2 − i, ∞) ∪ (−∞, −i − 2)`).  Plain parameters are taken real.
    if !real_taking_symbols_real(arena, a) || !real_taking_symbols_real(arena, b) {
        return None;
    }

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
                arena.interval(arena.neg_infinity, arena.infinity, IntervalKind::Open)
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
        Relation::Lt => arena.interval(lo, hi, IntervalKind::Open),
        Relation::Le => arena.interval(lo, hi, IntervalKind::Closed),
        Relation::Gt => {
            let left = arena.interval(arena.neg_infinity, lo, IntervalKind::Open);
            let right = arena.interval(hi, arena.infinity, IntervalKind::Open);
            arena.set_union(&[left, right])
        }
        Relation::Ge => {
            let left = arena.interval(arena.neg_infinity, lo, IntervalKind::LeftOpen);
            let right = arena.interval(hi, arena.infinity, IntervalKind::RightOpen);
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
        ValueSign::Defined(SignSet::POSITIVE) => return Some(1),
        ValueSign::Defined(SignSet::NEGATIVE) => return Some(-1),
        undecided if undecided.undecided() => {}
        _ => return None,
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

    /// The parts are read from the value, not from its printout (up to 0.57
    /// the `evalf` string was parsed); magnitudes beyond `f64` stay exact.
    #[test]
    fn value_parts_are_read_without_a_string() {
        let r = |p: i64, q: i64| Ratio::new(BigInt::from(p), BigInt::from(q));
        let mut arena = Arena::new();
        let parts = |arena: &mut Arena, e: ExprId| {
            let (value, re) = real_part(arena, e, DECISION_DIGITS).unwrap();
            (re.to_ratio(1000).unwrap(), value.im_sign())
        };
        let five_halves = arena.rational(5, 2);
        let neg_i = arena.neg(arena.i_unit);
        let e = arena.add(&[arena.one, five_halves, neg_i]);
        assert_eq!(parts(&mut arena, e), (r(7, 2), SignSet::NEGATIVE));
        let three = arena.int(3);
        let e = arena.mul(&[three, arena.i_unit]);
        assert_eq!(parts(&mut arena, e), (r(0, 1), SignSet::POSITIVE));
        let thousand = arena.int(1000);
        let e = arena.exp(thousand);
        assert!(parts(&mut arena, e).0 > r(1, 1));
        let neg = arena.int(-1000);
        let e = arena.exp(neg);
        let e = arena.neg(e);
        let (re, _) = parts(&mut arena, e);
        assert!(re.is_negative() && re > r(-1, 1));
    }

    /// Up to 0.55 a value at a far sample point was turned into its exact
    /// rational: `exp(1000³)` evaluates to `≈ 1.6·10^434294481`, whose power
    /// of ten never finished (`exp(x³) ≥ 0` hung).  The unscaled decimals
    /// compare exactly without it.
    #[test]
    fn huge_decimals_compare_without_their_powers() {
        let d = |s: &str| Decimal::parse(s, u64::MAX).unwrap();
        let huge = d("1.6e434294481");
        let tiny = d("-1.6e-434294481");
        assert!(!huge.is_negative() && tiny.is_negative());
        assert_eq!(huge.magnitude(), 434_294_481);
        assert_eq!(tiny.magnitude(), -434_294_481);
        assert!(tiny.magnitude() < -(AMBIGUOUS_EXP as i64));
        assert!(d("-0").is_zero() && !d("-0").is_negative());
        // Relative differences: one part in 10⁶ or more.
        assert!(!d("1.2345678e-400").differs_relatively(&d("1.23456780000001e-400"), 6));
        assert!(d("1.2345e-400").differs_relatively(&d("1.2346e-400"), 6));
        assert!(d("1e-400").differs_relatively(&d("1e-430"), 6));
        // The approximation of such a point is refused, not formed.
        assert!(huge.to_ratio(MAX_APPROX_EXP).is_none());
    }
}
