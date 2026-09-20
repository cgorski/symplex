//! Function-analysis utilities on [`Ex`]: singularities, stationary points,
//! monotonicity, extrema and periodicity (SymPy's `calculus.util`).  (0.9)
//!
//! * [`Ex::singularities`] — where the expression is undefined
//! * [`Ex::stationary_points`] — real zeros of the derivative
//! * [`Ex::maximum`] / [`Ex::minimum`] — supremum / infimum on a union of
//!   intervals
//! * [`Ex::is_increasing`], [`Ex::is_decreasing`],
//!   [`Ex::is_strictly_increasing`], [`Ex::is_strictly_decreasing`],
//!   [`Ex::is_monotonic`] — sign of the derivative on a domain
//! * [`Ex::is_convex`] — sign of the second derivative
//! * [`Ex::periodicity`] — a period of a trigonometric expression
//! * [`Ex::function_range`] — image of a continuous expression
//!
//! Every method works in the real domain.  Decisions are exact (Sturm
//! sequences for polynomial derivatives, the inequality solver, the
//! assumption system, exact set operations).  Numeric `f64` evaluation is
//! used in three documented places, none of which decides equality of two
//! symbolic values: to *order* two extremum candidates that are already
//! proven distinct ([`compare`]); to reject a constant solution as non-real
//! when its 16-digit imaginary part is clearly non-zero
//! ([`is_real_finite_point`]); and to *locate* the members of a periodic
//! solution family `a + n·p` inside a bounded domain, where the index range
//! is widened by one on each side and every candidate member is then
//! checked exactly against the domain ([`enumerate_family`]).
//!
//! Failures of the algorithms ("the zeros of `f'` cannot be found exactly",
//! "two candidates cannot be compared", …) are
//! [`SymplexError::ComputationFailed`]; `InvalidArgument` is reserved for
//! ill-formed input (a non-symbol variable, an empty or non-interval
//! domain).

use std::cmp::Ordering;

use crate::api::expr::{BoolEx, Ex, ExprType, SetEx, SetValued};
use crate::api::poly_ex::Poly;
use crate::base::errors::SymplexError;
use crate::base::node::ExprNode;
use crate::calculus::calculus_util as util;
use crate::calculus::limit::Direction;

// ═══════════════════════════════════════════════════════════════════════════
// Error helpers
// ═══════════════════════════════════════════════════════════════════════════

/// The algorithm behind `operation` could not complete on this input.
fn computation_failed(operation: &'static str, reason: impl Into<String>) -> SymplexError {
    SymplexError::ComputationFailed {
        operation,
        reason: reason.into(),
    }
}

fn invalid(operation: &'static str, reason: impl Into<String>) -> SymplexError {
    SymplexError::InvalidArgument {
        operation,
        reason: reason.into(),
    }
}

fn require_symbol(operation: &'static str, var: &Ex) -> Result<(), SymplexError> {
    if var.expr_type() == ExprType::Symbol {
        Ok(())
    } else {
        Err(invalid(operation, format!("`{var}` is not a symbol")))
    }
}

/// Largest number of members of one periodic solution family that are
/// enumerated inside a bounded domain.
const MAX_FAMILY_MEMBERS: i64 = 10_000;

/// Largest family index `|k|` for which the `f64` estimate of `k` in
/// [`enumerate_family`] is trusted to be within one of the true value
/// (a relative error of `1e-16` on `k` is then below `1e-7`).
const MAX_FAMILY_INDEX: f64 = 1e9;

/// Largest number of distinct `sign(h)` factors in a derivative that the
/// stationary-point analysis resolves by case-splitting on their signs
/// (`2^k` solver calls).
const MAX_SIGN_FACTORS: usize = 4;

// ═══════════════════════════════════════════════════════════════════════════
// Zero sets
// ═══════════════════════════════════════════════════════════════════════════

/// Is `p` (a solution returned by the solver) a real, finite point?
///
/// Non-real roots are rejected by the assumption system (`I`, `√−2`), by
/// structure (a constant mentioning `I` that is not proven real), and —
/// for constants only — numerically: a 16-digit imaginary part larger
/// than `1e-12·(1 + |re|)` rejects the point.  (A constant whose imaginary
/// part is non-zero but below that threshold is therefore kept; the
/// threshold is a tolerance, not an exact decision.)  Infinite or
/// undefined values are rejected.
fn is_real_finite_point(p: &Ex) -> bool {
    let ctx = p.context();
    if *p == ctx.infinity()
        || *p == ctx.neg_infinity()
        || *p == ctx.complex_infinity()
        || *p == ctx.nan()
    {
        return false;
    }
    if p.is_real() == Some(false) || p.is_finite() == Some(false) {
        return false;
    }
    if p.is_constant() {
        if p.contains(&ctx.i_unit()) && p.is_real() != Some(true) {
            return false;
        }
        if let Ok((re, im)) = p.eval_complex64()
            && im.abs() > 1e-12 * (1.0 + re.abs())
        {
            return false;
        }
    }
    true
}

/// Finite `f64` bounds of `domain`, or `None` when it is unbounded or its
/// bounds are not numeric.
fn numeric_bounds(domain: &SetEx) -> Option<(f64, f64)> {
    let ctx = domain.context();
    let lo = domain.inf()?;
    let hi = domain.sup()?;
    if lo == ctx.neg_infinity() || hi == ctx.infinity() {
        return None;
    }
    let lo = lo.eval_f64().ok()?;
    let hi = hi.eval_f64().ok()?;
    (lo.is_finite() && hi.is_finite()).then_some((lo, hi))
}

/// `{var | cond}` as a set expression.
fn condition_set(var: &Ex, cond: &BoolEx) -> SetEx {
    let var_id = var.raw_id();
    let cond_id = var.checked_id(cond);
    let id = var
        .inner
        .write()
        .arena
        .intern(ExprNode::ConditionSet(var_id, cond_id));
    var.wrap_as::<SetValued>(id)
}

/// The members of the one-parameter family `member(n)` (linear in the
/// integer parameter `n`) that lie in the bounded `domain`.
///
/// The range of indices is *estimated* in `f64` from the numeric values of
/// the offset, the step and the domain bounds, widened by one on each
/// side, and every candidate member is then tested exactly against the
/// domain.  The widening covers the `f64` error only while the estimated
/// index is below [`MAX_FAMILY_INDEX`]; larger indices are refused.
fn enumerate_family(
    member: &Ex,
    param: &Ex,
    (lo, hi): (f64, f64),
    domain: &SetEx,
    op: &'static str,
    out: &mut Vec<Ex>,
) -> Result<(), SymplexError> {
    let step = member.diff(param).eval();
    let offset = member.subs_i64(param, 0).eval();
    if step.contains(param) {
        return Err(computation_failed(
            op,
            format!("solution family `{member}` is not linear in `{param}`"),
        ));
    }
    // A non-real family (`atan(I) + nπ` from `tan² + 1 = 0`) has no real members.
    if !is_real_finite_point(&offset) {
        return Ok(());
    }
    let (Ok(step_f), Ok(offset_f)) = (step.eval_f64(), offset.eval_f64()) else {
        return Err(computation_failed(
            op,
            format!("cannot locate the members of the family `{member}` numerically"),
        ));
    };
    if step_f == 0.0 {
        out.push(offset);
        return Ok(());
    }
    let (k1, k2) = ((lo - offset_f) / step_f, (hi - offset_f) / step_f);
    let (k_lo, k_hi) = (k1.min(k2), k1.max(k2));
    // Bound the index range while still in `f64`: the ±1 widening below
    // only absorbs the rounding error of `k` while `|k|` is small, and the
    // casts must not saturate.
    if !k_lo.is_finite()
        || !k_hi.is_finite()
        || k_lo.abs() > MAX_FAMILY_INDEX
        || k_hi.abs() > MAX_FAMILY_INDEX
    {
        return Err(computation_failed(
            op,
            format!(
                "the members of `{member}` in the domain have indices beyond \
                 ±{MAX_FAMILY_INDEX:e}, where they cannot be located reliably"
            ),
        ));
    }
    if k_hi - k_lo > MAX_FAMILY_MEMBERS as f64 {
        return Err(computation_failed(
            op,
            format!("more than {MAX_FAMILY_MEMBERS} members of `{member}` may lie in the domain"),
        ));
    }
    let (k_min, k_max) = (k_lo.floor() as i64 - 1, k_hi.ceil() as i64 + 1);
    for k in k_min..=k_max {
        let point = member.subs_i64(param, k).eval();
        match domain.contains(&point) {
            Some(true) => out.push(point),
            Some(false) => {}
            None => {
                return Err(computation_failed(
                    op,
                    format!("cannot decide whether `{point}` lies in `{domain}`"),
                ));
            }
        }
    }
    Ok(())
}

/// The real zeros of `g` (with respect to `var`) inside `domain`.
///
/// Uses `solve` for non-periodic equations and `solve_general` when `g`
/// contains `sin`/`cos`/`tan` of `var`; periodic families are enumerated
/// on bounded domains and represented as `{var | g = 0}` otherwise.
fn zeros_in_domain(
    g: &Ex,
    var: &Ex,
    domain: &SetEx,
    op: &'static str,
) -> Result<SetEx, SymplexError> {
    let ctx = g.context();
    let var_id = g.checked_id(var);
    let periodic = {
        let inner = g.inner.read();
        util::has_trig_of(&inner.arena, g.raw_id(), var_id)
    };

    let mut points: Vec<Ex> = Vec::new();
    let mut families: Vec<SetEx> = Vec::new();

    let outcome = if periodic {
        g.solve_general(var).map(|fam| {
            let mut plain = Vec::new();
            let mut parametric = Vec::new();
            for s in fam.solutions {
                if fam.parameters.iter().any(|n| s.contains(n)) {
                    parametric.push(s);
                } else {
                    plain.push(s);
                }
            }
            (plain, parametric, fam.parameters)
        })
    } else {
        g.solve(var).map(|sols| (sols, Vec::new(), Vec::new()))
    };

    match outcome {
        Ok((plain, parametric, parameters)) => {
            points.extend(plain);
            if !parametric.is_empty() {
                let Some(param) = parameters.first() else {
                    return Err(computation_failed(
                        op,
                        "parametric solution without parameter",
                    ));
                };
                match numeric_bounds(domain) {
                    Some(bounds) => {
                        for member in &parametric {
                            enumerate_family(member, param, bounds, domain, op, &mut points)?;
                        }
                    }
                    None => families.push(condition_set(var, &g.eq_expr(&ctx.zero()))),
                }
            }
        }
        // No zeros at all.
        Err(SymplexError::NoSolution { .. }) => {}
        // `g ≡ 0`: every point of the domain.
        Err(SymplexError::InfiniteSolutions { .. }) => return Ok(domain.clone()),
        Err(e) => {
            return Err(computation_failed(
                op,
                format!("the zeros of `{g}` cannot be found exactly ({e})"),
            ));
        }
    }

    points.retain(is_real_finite_point);
    let mut result = ctx.finite_set(&points).intersection(domain);
    for fam in families {
        result = result.union(&fam.intersection(domain));
    }
    Ok(result.simplify())
}

// ═══════════════════════════════════════════════════════════════════════════
// Extremum candidates
// ═══════════════════════════════════════════════════════════════════════════

/// A value the expression takes (or approaches) on an interval.
struct Candidate {
    value: Ex,
    /// `true` when the value is attained at a point of the interval,
    /// `false` when it is only a one-sided limit at an open or infinite
    /// endpoint.
    attained: bool,
}

/// Exact comparison of two candidate values.
///
/// `±∞` compare as expected; finite values are compared through
/// [`Ex::equals`] and the sign of their difference (assumption system).
/// When the two values are proven distinct but the sign of the difference
/// is not decided symbolically, the 16-digit numeric value of the
/// difference orders them — it is never used to decide equality (see the
/// module notes for the other numeric steps).
fn compare(a: &Ex, b: &Ex) -> Option<Ordering> {
    let ctx = a.context();
    let (inf, ninf) = (ctx.infinity(), ctx.neg_infinity());
    if a == b {
        return Some(Ordering::Equal);
    }
    if *a == inf || *b == ninf {
        return Some(Ordering::Greater);
    }
    if *a == ninf || *b == inf {
        return Some(Ordering::Less);
    }
    if a.equals(b) == Some(true) {
        return Some(Ordering::Equal);
    }
    let d = (a - b).eval();
    if d.is_positive() == Some(true) {
        return Some(Ordering::Greater);
    }
    if d.is_negative() == Some(true) {
        return Some(Ordering::Less);
    }
    if a.equals(b) != Some(false) {
        return None;
    }
    // Proven distinct constants: order numerically.
    let v = d.eval_f64().ok()?;
    if v > 0.0 {
        Some(Ordering::Greater)
    } else if v < 0.0 {
        Some(Ordering::Less)
    } else {
        None
    }
}

/// Minimum and maximum of a non-empty candidate list.
fn min_max(
    cands: Vec<Candidate>,
    op: &'static str,
) -> Result<(Candidate, Candidate), SymplexError> {
    let mut iter = cands.into_iter();
    let Some(first) = iter.next() else {
        return Err(computation_failed(op, "no candidate values"));
    };
    let mut lo = Candidate {
        value: first.value.clone(),
        attained: first.attained,
    };
    let mut hi = first;
    for c in iter {
        let Some(ord) = compare(&c.value, &lo.value) else {
            return Err(computation_failed(
                op,
                format!(
                    "cannot compare the candidates `{}` and `{}`",
                    c.value, lo.value
                ),
            ));
        };
        match ord {
            Ordering::Less => {
                lo = Candidate {
                    value: c.value.clone(),
                    attained: c.attained,
                }
            }
            Ordering::Equal => lo.attained |= c.attained,
            Ordering::Greater => {}
        }
        let Some(ord) = compare(&c.value, &hi.value) else {
            return Err(computation_failed(
                op,
                format!(
                    "cannot compare the candidates `{}` and `{}`",
                    c.value, hi.value
                ),
            ));
        };
        match ord {
            Ordering::Greater => hi = c,
            Ordering::Equal => hi.attained |= c.attained,
            Ordering::Less => {}
        }
    }
    Ok((lo, hi))
}

/// An exact point strictly inside the non-degenerate interval `(lo, hi)`
/// (either bound may be infinite).
fn interior_point(ctx: &crate::api::context::Context, lo: &Ex, hi: &Ex) -> Ex {
    match (*lo == ctx.neg_infinity(), *hi == ctx.infinity()) {
        (true, true) => ctx.zero(),
        (true, false) => (hi - 1).eval(),
        (false, true) => (lo + 1).eval(),
        (false, false) => ((lo + hi) / 2).eval(),
    }
}

/// Sign of the constant `value`: `Some(true)` positive, `Some(false)`
/// negative, `None` zero or undecided.
fn constant_sign(value: &Ex) -> Option<bool> {
    if value.is_positive() == Some(true) {
        Some(true)
    } else if value.is_negative() == Some(true) {
        Some(false)
    } else {
        None
    }
}

/// The real zeros of the derivative `d` (of some function of `var`) in
/// `domain`, resolving the `sign(h)` factors that differentiating `|h|`
/// introduces — [`zeros_in_domain`] when there are none.
///
/// On each region where every `h` has a fixed sign, `d` equals the
/// specialisation with the matching `±1` substituted; its zeros are kept
/// where the assumed signs actually hold.  A specialisation that vanishes
/// identically (`|x| + x` for `x < 0`) makes the whole region stationary
/// and it is returned as a set of intervals.  Kinks `h = 0` at which `d`
/// itself evaluates to zero (`|x|` at `0`, since `sign(0) = 0`) are
/// included, as SymPy does.  At most [`MAX_SIGN_FACTORS`] distinct factors
/// are resolved.
fn stationary_set(
    d: &Ex,
    var: &Ex,
    domain: &SetEx,
    op: &'static str,
) -> Result<SetEx, SymplexError> {
    let signs = d.sign_factors(var);
    if signs.is_empty() {
        return zeros_in_domain(d, var, domain, op);
    }
    if signs.len() > MAX_SIGN_FACTORS {
        return Err(computation_failed(
            op,
            format!(
                "the derivative `{d}` has {} distinct sign(…) factors; at most \
                 {MAX_SIGN_FACTORS} are resolved",
                signs.len()
            ),
        ));
    }
    let ctx = d.context();
    let (one, minus_one) = (ctx.one(), ctx.int(-1));
    let mut result = ctx.empty_set();

    // Kinks at which the derivative itself vanishes.
    for (_, h) in &signs {
        let kinks = zeros_in_domain(h, var, domain, op)?;
        let Some(points) = kinks.as_finite_set() else {
            return Err(computation_failed(
                op,
                format!("the kinks {kinks} of `|{h}|` cannot be enumerated"),
            ));
        };
        let stationary: Vec<Ex> = points
            .into_iter()
            .filter(|p| d.subs(var, p).eval().is_zero_structural())
            .collect();
        if !stationary.is_empty() {
            result = result.union(&ctx.finite_set(&stationary));
        }
    }

    // Every sign pattern: bit `i` set means `sign(h_i) = +1`.
    for mask in 0..(1usize << signs.len()) {
        let mut spec = d.clone();
        let mut pattern: Vec<(&Ex, bool)> = Vec::with_capacity(signs.len());
        for (i, (s, h)) in signs.iter().enumerate() {
            let positive = mask & (1 << i) != 0;
            spec = spec.subs(s, if positive { &one } else { &minus_one });
            pattern.push((h, positive));
        }
        let spec = spec.eval();
        // The region where this pattern holds, as a set (built lazily).
        let region = || -> Result<SetEx, SymplexError> {
            let mut region = domain.clone();
            for &(h, positive) in &pattern {
                let side = if positive {
                    h.solve_gt(var)
                } else {
                    h.solve_lt(var)
                };
                if side.has_unevaluated() {
                    return Err(computation_failed(
                        op,
                        format!("cannot describe the region where `{h}` has a fixed sign"),
                    ));
                }
                region = region.intersection(&side);
            }
            Ok(region.simplify())
        };
        if spec.is_zero_structural() {
            // Stationary throughout the region.
            result = result.union(&region()?);
            continue;
        }
        let zeros = zeros_in_domain(&spec, var, domain, op)?;
        match zeros.as_finite_set() {
            Some(points) => {
                let mut kept: Vec<Ex> = Vec::new();
                for p in points {
                    let mut holds = true;
                    for &(h, positive) in &pattern {
                        let value = h.subs(var, &p).eval();
                        match constant_sign(&value) {
                            Some(sign) if sign == positive => {}
                            Some(_) => {
                                holds = false;
                                break;
                            }
                            None if value.is_zero_structural() => {
                                // A kink: handled above.
                                holds = false;
                                break;
                            }
                            None => {
                                return Err(computation_failed(
                                    op,
                                    format!("cannot decide the sign of `{h}` at `{var} = {p}`"),
                                ));
                            }
                        }
                    }
                    if holds {
                        kept.push(p);
                    }
                }
                if !kept.is_empty() {
                    result = result.union(&ctx.finite_set(&kept));
                }
            }
            // A periodic family: restrict it to the region as a set.
            None => result = result.union(&zeros.intersection(&region()?)),
        }
    }
    Ok(result.simplify())
}

impl Ex {
    /// Structural continuity information about `self` in `var`.
    fn continuity_scan(&self, var: &Ex) -> util::ContinuityScan {
        let var_id = self.checked_id(var);
        let mut inner = self.inner.write();
        util::continuity_scan(&mut inner.arena, self.raw_id(), var_id)
    }

    /// Refuse expressions that are not continuous on `domain`: opaque or
    /// discontinuous nodes, or singularities inside the domain.
    fn require_continuous(
        &self,
        var: &Ex,
        domain: &SetEx,
        op: &'static str,
    ) -> Result<(), SymplexError> {
        if let Some(name) = self.continuity_scan(var).opaque {
            return Err(computation_failed(
                op,
                format!("`{self}` contains {name} of `{var}`, which is not analysed"),
            ));
        }
        let sing = self.singularities(var, Some(domain))?;
        match sing.is_empty() {
            Some(true) => Ok(()),
            Some(false) => Err(computation_failed(
                op,
                format!("`{self}` has singularities {sing} inside the domain"),
            )),
            None => Err(computation_failed(
                op,
                format!("cannot decide whether the singularity set {sing} meets the domain"),
            )),
        }
    }

    /// The distinct `sign(h)` factors of `self` (a derivative) whose
    /// argument `h` depends on `var`, as `(sign(h), h)` pairs.
    fn sign_factors(&self, var: &Ex) -> Vec<(Ex, Ex)> {
        let var_id = self.checked_id(var);
        let ids = {
            let inner = self.inner.read();
            util::sign_nodes_of(&inner.arena, self.raw_id(), var_id)
        };
        ids.into_iter()
            .map(|(s, h)| (self.wrap(s), self.wrap(h)))
            .collect()
    }

    /// Value of `self` at `point`, or a one-sided limit when `limit` is
    /// given; rejects undefined / unevaluated results.
    fn value_at(
        &self,
        var: &Ex,
        point: &Ex,
        limit: Option<Direction>,
        op: &'static str,
    ) -> Result<Candidate, SymplexError> {
        let ctx = self.context();
        let (value, attained) = match limit {
            None => (self.subs(var, point).eval(), true),
            Some(dir) => {
                let v = self.try_limit_dir(var, point, dir).map_err(|e| {
                    computation_failed(op, format!("limit at the endpoint `{point}` failed: {e}"))
                })?;
                (v, false)
            }
        };
        let real = if attained {
            is_real_finite_point(&value)
        } else {
            // A limit may be ±∞, but not undefined or non-real.
            value == ctx.infinity() || value == ctx.neg_infinity() || is_real_finite_point(&value)
        };
        if value.has_unevaluated() || !real {
            return Err(computation_failed(
                op,
                format!("the value `{value}` at `{var} = {point}` is not a real number"),
            ));
        }
        Ok(Candidate { value, attained })
    }

    /// Every value that can be the supremum or infimum of `self` on the
    /// interval `(lo, hi)` with the given openness: stationary points,
    /// `abs` kinks, closed endpoints (attained) and one-sided limits at
    /// open or infinite endpoints (not attained).
    fn candidates_on(
        &self,
        var: &Ex,
        (lo, hi, lo_open, hi_open): (&Ex, &Ex, bool, bool),
        op: &'static str,
    ) -> Result<Vec<Candidate>, SymplexError> {
        let ctx = self.context();
        if lo == hi {
            return Ok(vec![self.value_at(var, lo, None, op)?]);
        }
        let part = ctx.interval(lo, hi, lo_open, hi_open);
        let mut cands = Vec::new();

        // Interior critical points: stationary points and |g| kinks.
        let derivative = self.diff(var);
        if derivative.has_unevaluated() {
            return Err(computation_failed(
                op,
                format!("the derivative `{derivative}` could not be evaluated"),
            ));
        }
        if derivative.is_zero_structural() {
            // Constant in `var`: one value, attained everywhere.  Simplified
            // so that `sin²x + cos²x` reports `1`, not itself.
            return Ok(vec![Candidate {
                value: self.simplify(),
                attained: true,
            }]);
        }
        let mut interior: Vec<SetEx> = vec![stationary_set(&derivative, var, &part, op)?];
        for g in self.continuity_scan(var).kinks {
            interior.push(zeros_in_domain(&self.wrap(g), var, &part, op)?);
        }
        for set in interior {
            if let Some(points) = set.as_finite_set() {
                for p in points {
                    cands.push(self.value_at(var, &p, None, op)?);
                }
            } else if let Some(pieces) = set.as_intervals() {
                // A whole interval of stationary points: `self` is
                // continuous with zero derivative on it, hence constant
                // there, and one interior point carries the (attained)
                // value.
                for (a, b, _, _) in pieces {
                    let p = if a == b {
                        a
                    } else {
                        interior_point(&ctx, &a, &b)
                    };
                    cands.push(self.value_at(var, &p, None, op)?);
                }
            } else {
                return Err(computation_failed(
                    op,
                    format!("the critical points {set} cannot be enumerated"),
                ));
            }
        }

        // Endpoints.
        let lo_limit = (lo_open || *lo == ctx.neg_infinity()).then_some(Direction::Right);
        let hi_limit = (hi_open || *hi == ctx.infinity()).then_some(Direction::Left);
        cands.push(self.value_at(var, lo, lo_limit, op)?);
        cands.push(self.value_at(var, hi, hi_limit, op)?);
        Ok(cands)
    }

    /// Shared body of [`maximum`](Ex::maximum) and [`minimum`](Ex::minimum).
    fn extremum(
        &self,
        var: &Ex,
        domain: &SetEx,
        want_max: bool,
        op: &'static str,
    ) -> Result<Ex, SymplexError> {
        self.checked_id(var);
        self.checked_id(domain);
        require_symbol(op, var)?;
        let Some(parts) = domain.as_intervals() else {
            return Err(invalid(
                op,
                format!("the domain `{domain}` is not a union of intervals"),
            ));
        };
        if parts.is_empty() {
            return Err(invalid(op, "the domain is empty"));
        }
        self.require_continuous(var, domain, op)?;
        let mut cands = Vec::new();
        for (lo, hi, lo_open, hi_open) in &parts {
            cands.extend(self.candidates_on(var, (lo, hi, *lo_open, *hi_open), op)?);
        }
        let (lo, hi) = min_max(cands, op)?;
        Ok(if want_max { hi.value } else { lo.value })
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Monotonicity
// ═══════════════════════════════════════════════════════════════════════════

/// Is the polynomial `p` strictly positive on the interval `(lo, hi)` with
/// the given openness?  Exact: a Sturm count of the roots inside the
/// closed interval, discounting roots at open endpoints.
fn poly_positive_on(
    p: &Poly,
    var: &Ex,
    (lo, hi, lo_open, hi_open): (&Ex, &Ex, bool, bool),
) -> Option<bool> {
    if p.is_positive_on(lo, hi) == Some(true) {
        return Some(true);
    }
    if !p.is_nonnegative_on(lo, hi)? {
        return Some(false);
    }
    // Non-negative on the closed interval with at least one root in it.
    let ctx = var.context();
    let mut roots = p.count_real_roots_in(lo, hi)?;
    let e = p.to_ex();
    let vanishes_at = |pt: &Ex| e.subs(var, pt).eval().is_zero_structural();
    if lo_open && *lo != ctx.neg_infinity() && vanishes_at(lo) {
        roots = roots.saturating_sub(1);
    }
    if hi_open && *hi != ctx.infinity() && lo != hi && vanishes_at(hi) {
        roots = roots.saturating_sub(1);
    }
    Some(roots == 0)
}

/// Exact decision of `d ≥ 0` on every interval of `parts` for a rational
/// function `d = num/den` in `var` with rational coefficients.  With
/// `strict`, additionally requires `d` not to vanish identically (its
/// zeros are then isolated).  `None` when the route does not apply.
fn rational_nonneg_on(
    d: &Ex,
    var: &Ex,
    parts: &[(Ex, Ex, bool, bool)],
    strict: bool,
) -> Option<bool> {
    let (num, den) = d.as_numer_denom();
    let pn = Poly::new(&num, &[var])?;
    let pd = Poly::new(&den, &[var])?;
    if !pn.has_rational_coeffs() || !pd.has_rational_coeffs() {
        return None;
    }
    let neg_pn = pn.neg();
    let neg_pd = pd.neg();
    for (lo, hi, lo_open, hi_open) in parts {
        let part = (lo, hi, *lo_open, *hi_open);
        let signed_num = if poly_positive_on(&pd, var, part)? {
            &pn
        } else if poly_positive_on(&neg_pd, var, part)? {
            &neg_pn
        } else {
            // The denominator changes sign or vanishes inside the interval.
            return None;
        };
        if !signed_num.is_nonnegative_on(lo, hi)? {
            return Some(false);
        }
    }
    if strict && pn.is_zero() {
        // Identically zero: not strict unless every part is a single point.
        return Some(parts.iter().all(|(lo, hi, _, _)| lo == hi));
    }
    Some(true)
}

/// Is `f` continuous at every finite closed endpoint of `parts`, i.e. is
/// `f(e)` a finite real value equal to the one-sided limit from inside the
/// interval?  (Isolated points need no check.)
fn continuous_at_closed_endpoints(f: &Ex, var: &Ex, parts: &[(Ex, Ex, bool, bool)]) -> bool {
    let ctx = f.context();
    let check = |e: &Ex, dir: Direction| -> bool {
        let value = f.subs(var, e).eval();
        if value.has_unevaluated() || !is_real_finite_point(&value) {
            return false;
        }
        f.try_limit_dir(var, e, dir)
            .is_ok_and(|lim| lim.equals(&value) == Some(true))
    };
    parts.iter().all(|(lo, hi, lo_open, hi_open)| {
        lo == hi
            || ((*lo_open || *lo == ctx.neg_infinity() || check(lo, Direction::Right))
                && (*hi_open || *hi == ctx.infinity() || check(hi, Direction::Left)))
    })
}

/// Does `f` have a singularity strictly inside `domain` that rules out
/// monotonicity?  `Some(false)` when every interior singularity is
/// removable (finite, equal one-sided limits), so the derivative-based
/// analysis stays valid on the continuous extension; `Some(true)` when some
/// one-sided limit at an interior singularity is infinite — a monotone
/// function has finite one-sided limits at every interior point of its
/// domain, so `f` is then neither non-decreasing nor non-increasing on the
/// domain; `None` when the singularities cannot be enumerated or a limit
/// is not decided (a finite jump, for instance, is compatible with
/// monotonicity and is not decided here).
fn interior_pole_breaks_monotonicity(f: &Ex, var: &Ex, domain: &SetEx) -> Option<bool> {
    let ctx = f.context();
    let interior = domain.interior()?;
    let sing = f.singularities(var, Some(&interior)).ok()?;
    if sing.is_empty() == Some(true) {
        return Some(false);
    }
    let points = sing.as_finite_set()?;
    let (inf, ninf) = (ctx.infinity(), ctx.neg_infinity());
    for p in points {
        let left = f.try_limit_dir(var, &p, Direction::Left).ok()?;
        let right = f.try_limit_dir(var, &p, Direction::Right).ok()?;
        if left == inf || left == ninf || right == inf || right == ninf {
            return Some(true);
        }
        if !(is_real_finite_point(&left)
            && is_real_finite_point(&right)
            && left.equals(&right) == Some(true))
        {
            return None;
        }
    }
    Some(false)
}

/// Three-valued decision "`f` is non-decreasing on `domain`" (`d = f'`;
/// pass `f = -g` to decide that `g` is non-increasing) — with `strict`,
/// "`f' ≥ 0` with only isolated zeros".
///
/// A pole strictly inside the domain (`1/x` on `[−1, 1]`, `tan x` on
/// `[0, π]`) refutes monotonicity outright, whatever the sign of `f'` on
/// either side (see [`interior_pole_breaks_monotonicity`]).  Otherwise the
/// routes, in order: exact Sturm-sequence test for polynomial / rational
/// `f'`, the assumption system, then the inequality solver
/// (`solve_ge` / `solve_gt`) with a subset test.  When `f'` fails only at
/// closed endpoints where it is undefined (`√x` at `0`), the test is
/// repeated on the interior of the domain and `f` is required to be
/// continuous at those endpoints (mean value theorem).  `None` whenever no
/// route decides.
fn monotone_on(f: &Ex, var: &Ex, domain: &SetEx, strict: bool) -> Option<bool> {
    // Discontinuous or opaque nodes (`floor`, `sign`, unknown functions, …):
    // the formal derivative says nothing about monotonicity.
    if f.continuity_scan(var).opaque.is_some() {
        return None;
    }
    let d = f.diff(var);
    if d.has_unevaluated() {
        return None;
    }
    let parts = domain.as_intervals()?;
    if parts.is_empty() {
        return Some(true);
    }
    if d.is_zero_structural() {
        return Some(!strict || parts.iter().all(|(lo, hi, _, _)| lo == hi));
    }
    // The sign of `f'` on either side of an interior pole says nothing
    // about monotonicity across it.
    match interior_pole_breaks_monotonicity(f, var, domain) {
        Some(true) => return Some(false),
        Some(false) => {}
        None => return None,
    }
    if let Some(answer) = rational_nonneg_on(&d, var, &parts, strict) {
        return Some(answer);
    }
    // Assumption system: a sign that holds for every real value of `var`.
    if d.is_negative() == Some(true) {
        return Some(false);
    }
    if (strict && d.is_positive() == Some(true)) || (!strict && d.is_nonnegative() == Some(true)) {
        return Some(true);
    }
    // Inequality solver.
    let ge = d.solve_ge(var);
    if ge.has_unevaluated() {
        return None;
    }
    let region = match domain.is_subset(&ge) {
        Some(true) => domain.clone(),
        Some(false) => {
            // `f'` may merely be undefined at a closed endpoint.
            if !continuous_at_closed_endpoints(f, var, &parts) {
                return Some(false);
            }
            let interior = domain.interior()?;
            match interior.is_subset(&ge) {
                Some(true) => interior,
                Some(false) => return Some(false),
                None => return None,
            }
        }
        None => return None,
    };
    if !strict {
        return Some(true);
    }
    let gt = d.solve_gt(var);
    if gt.has_unevaluated() {
        return None;
    }
    if region.is_subset(&gt) == Some(true) {
        return Some(true);
    }
    // `f' ≥ 0` with a finite zero set: still strictly monotonic.
    let zeros = ge.difference(&gt).intersection(&region).simplify();
    zeros.as_finite_set().map(|_| true)
}

/// Three-valued `a ∨ b`.
fn or3(a: Option<bool>, b: Option<bool>) -> Option<bool> {
    match (a, b) {
        (Some(true), _) | (_, Some(true)) => Some(true),
        (Some(false), Some(false)) => Some(false),
        _ => None,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════════════

impl Ex {
    /// The points of `domain` (default ℝ) where `self` is undefined —
    /// SymPy's `singularities`.
    ///
    /// The rule set is SymPy's: zeros of the base of every negative power
    /// (this covers denominators, `sec`, `csc` and `cot`), zeros of the
    /// argument of `ln`, poles of `tan`, and `atanh(g)` at `g = ±1`.  Only
    /// real points are reported.
    ///
    /// Zeros are found exactly with [`solve`](Ex::solve); equations with
    /// `sin`/`cos`/`tan` of `var` use [`solve_general`](Ex::solve_general)
    /// and the periodic families are enumerated inside a bounded domain
    /// (`tan(x)` on `[0, 10]` gives `{π/2, 3π/2, 5π/2}`).  On an unbounded
    /// domain such a family is returned as the condition set
    /// `{x | cos(x) = 0}` (intersected with the domain), since the
    /// infinite family has no interval / finite-set representation.
    ///
    /// # Errors
    ///
    /// * `InvalidArgument` — `var` is not a symbol.
    /// * `ComputationFailed` — the zeros of some source cannot be found
    ///   exactly (the solver fails on `g = 0`), or, on the periodic-family
    ///   path only, a family is not linear in its integer parameter, its
    ///   members cannot be located numerically, more than 10 000 of them
    ///   may lie in the domain, or the membership of a member in `domain`
    ///   cannot be decided.  Plain (non-periodic) zeros whose membership in
    ///   `domain` is undecided do **not** error: the result is then `Ok`
    ///   with the intersection `{p, …} ∩ domain` left unevaluated.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// // SymPy: singularities(1/(x**2 - 1), x) == {-1, 1}
    /// let s = (1 / (&x.powi(2) - 1)).singularities(&x, None).unwrap();
    /// assert_eq!(s.to_string(), "{-1, 1}");
    /// // SymPy: singularities(log(x), x) == {0}
    /// assert_eq!(x.ln().singularities(&x, None).unwrap().to_string(), "{0}");
    /// // Polynomials have none.
    /// assert_eq!(x.powi(2).singularities(&x, None).unwrap().is_empty(), Some(true));
    /// // Restricted to a domain.
    /// let dom = ctx.interval(&ctx.int(0), &ctx.int(5), false, false);
    /// assert_eq!((1 / (&x.powi(2) - 1)).singularities(&x, Some(&dom)).unwrap().to_string(), "{1}");
    /// ```
    pub fn singularities(&self, var: &Ex, domain: Option<&SetEx>) -> Result<SetEx, SymplexError> {
        const OP: &str = "singularities";
        self.checked_id(var);
        require_symbol(OP, var)?;
        let ctx = self.context();
        let domain = match domain {
            Some(d) => {
                self.checked_id(d);
                d.clone()
            }
            None => ctx.reals(),
        };
        let sources = self.continuity_scan(var).singular;
        let mut result = ctx.empty_set();
        for g in sources {
            let zeros = zeros_in_domain(&self.wrap(g), var, &domain, OP)?;
            result = result.union(&zeros);
        }
        Ok(result.simplify())
    }

    /// The real solutions of `d self / d var = 0` in `domain` (default ℝ)
    /// — SymPy's `stationary_points`.
    ///
    /// Periodic families of critical points (`sin`, `cos`, `tan`) are
    /// enumerated on a bounded domain and returned as a condition set
    /// `{x | f'(x) = 0}` on an unbounded one; see
    /// [`singularities`](Ex::singularities).  An expression that does not
    /// depend on `var` has derivative `0`, so every point of the domain is
    /// stationary and the domain itself is returned (as SymPy does).
    ///
    /// The `sign(h)` factors that differentiating `|h|` introduces are
    /// resolved by cases: on each region where every `h` has a fixed sign
    /// the derivative is a plain expression whose zeros are found and kept
    /// where the assumed signs hold (`|x − 1| + x²` → `{1/2}`).  A region
    /// on which the derivative vanishes identically is returned whole
    /// (`|x| + x` on `[−1, 2]` → `[−1, 0)`), and a kink at which the
    /// derivative evaluates to zero counts as stationary (`|x|` → `{0}`,
    /// since `sign(0) = 0`) — all as SymPy does.  At most four distinct
    /// `sign` factors are resolved.
    ///
    /// # Errors
    ///
    /// * `InvalidArgument` — `var` is not a symbol.
    /// * `ComputationFailed` — the derivative is a formal `Derivative`, the
    ///   zeros of the derivative cannot be found exactly, or more than four
    ///   `sign` factors would have to be resolved.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let f = &x.powi(3) - &x * 3;
    /// // SymPy: stationary_points(x**3 - 3*x, x) == {-1, 1}
    /// assert_eq!(f.stationary_points(&x, None).unwrap().to_string(), "{-1, 1}");
    /// // SymPy: stationary_points(x**3 - 3*x, x, Interval(0, 5)) == {1}
    /// let dom = ctx.interval(&ctx.int(0), &ctx.int(5), false, false);
    /// assert_eq!(f.stationary_points(&x, Some(&dom)).unwrap().to_string(), "{1}");
    /// // SymPy: stationary_points(sin(x), x, Interval(0, 2*pi)) == {pi/2, 3*pi/2}
    /// let two_pi = ctx.interval(&ctx.int(0), &(&ctx.pi() * 2), false, false);
    /// let sp = x.sin().stationary_points(&x, Some(&two_pi)).unwrap();
    /// assert_eq!(sp.as_finite_set().unwrap().len(), 2);
    /// // SymPy: stationary_points(Abs(x - 1) + x**2, x, Interval(-1, 2)) == {1/2}
    /// let g = (&x - 1).abs() + x.powi(2);
    /// let dom = ctx.interval(&ctx.int(-1), &ctx.int(2), false, false);
    /// assert_eq!(g.stationary_points(&x, Some(&dom)).unwrap().to_string(), "{1/2}");
    /// ```
    pub fn stationary_points(
        &self,
        var: &Ex,
        domain: Option<&SetEx>,
    ) -> Result<SetEx, SymplexError> {
        const OP: &str = "stationary_points";
        self.checked_id(var);
        require_symbol(OP, var)?;
        let ctx = self.context();
        let domain = match domain {
            Some(d) => {
                self.checked_id(d);
                d.clone()
            }
            None => ctx.reals(),
        };
        let derivative = self.diff(var);
        if derivative.has_unevaluated() {
            return Err(computation_failed(
                OP,
                format!("the derivative `{derivative}` could not be evaluated"),
            ));
        }
        if derivative.is_zero_structural() {
            return Ok(domain);
        }
        stationary_set(&derivative, var, &domain, OP)
    }

    /// Supremum of `self` (continuous in `var`) over `domain`, a union of
    /// intervals — SymPy's `maximum`.
    ///
    /// The candidates are the values at the stationary points and `abs`
    /// kinks inside the domain (see [`stationary_points`](Ex::stationary_points)
    /// for how `|h|` is handled: `maximum(|x|, [−1, 2]) = 2`,
    /// `minimum(|x − 1| + x², [−1, 2]) = 3/4`), at closed endpoints, and
    /// the one-sided limits at open or infinite endpoints; `+∞` / `−∞` are
    /// legitimate results.  Candidates are compared exactly (see the module
    /// notes for the numeric fallbacks).  The supremum need not be
    /// attained: `maximum(x, (0, 1)) = 1`.
    ///
    /// # Errors
    ///
    /// * `InvalidArgument` — `var` is not a symbol, or `domain` is empty
    ///   or not a union of intervals.
    /// * `ComputationFailed` — `self` has singularities inside the domain,
    ///   contains a discontinuous or opaque node (`floor`, `sign`,
    ///   `Piecewise`, an unknown function, …), its stationary points
    ///   cannot be enumerated, an endpoint limit cannot be computed, or two
    ///   candidates cannot be compared.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let f = &x.powi(3) - &x * 3;
    /// let dom = ctx.interval(&ctx.int(-2), &ctx.int(2), false, false);
    /// // SymPy: maximum(x**3 - 3*x, x, Interval(-2, 2)) == 2
    /// assert_eq!(f.maximum(&x, &dom).unwrap().to_string(), "2");
    /// // SymPy: maximum(x**2, x, S.Reals) == oo
    /// assert_eq!(x.powi(2).maximum(&x, &ctx.reals()).unwrap(), ctx.infinity());
    /// // SymPy: maximum(1/x, x, Interval(1, oo)) == 1
    /// let tail = ctx.interval(&ctx.int(1), &ctx.infinity(), false, true);
    /// assert_eq!((1 / &x).maximum(&x, &tail).unwrap().to_string(), "1");
    /// ```
    pub fn maximum(&self, var: &Ex, domain: &SetEx) -> Result<Ex, SymplexError> {
        self.extremum(var, domain, true, "maximum")
    }

    /// Infimum of `self` (continuous in `var`) over `domain` — SymPy's
    /// `minimum`.  Same method, candidates and errors as
    /// [`maximum`](Ex::maximum).
    ///
    /// # Errors
    ///
    /// See [`maximum`](Ex::maximum).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let f = &x.powi(3) - &x * 3;
    /// let dom = ctx.interval(&ctx.int(-2), &ctx.int(2), false, false);
    /// // SymPy: minimum(x**3 - 3*x, x, Interval(-2, 2)) == -2
    /// assert_eq!(f.minimum(&x, &dom).unwrap().to_string(), "-2");
    /// // SymPy: minimum(1/x, x, Interval(1, oo)) == 0   (a limit, not attained)
    /// let tail = ctx.interval(&ctx.int(1), &ctx.infinity(), false, true);
    /// assert_eq!((1 / &x).minimum(&x, &tail).unwrap().to_string(), "0");
    /// // SymPy: minimum(x**2, x, S.Reals) == 0
    /// assert_eq!(x.powi(2).minimum(&x, &ctx.reals()).unwrap().to_string(), "0");
    /// ```
    pub fn minimum(&self, var: &Ex, domain: &SetEx) -> Result<Ex, SymplexError> {
        self.extremum(var, domain, false, "minimum")
    }

    /// Is `self` non-decreasing in `var` on `domain` (`f' ≥ 0` there)?
    /// SymPy's `is_increasing`.
    ///
    /// Three-valued.  Polynomial and rational derivatives with rational
    /// coefficients are decided exactly (Sturm sequences on each interval of
    /// the domain, the denominator having constant sign there); otherwise
    /// the assumption system and the inequality solver
    /// ([`solve_ge`](Ex::solve_ge)) are consulted.  A derivative that is
    /// undefined at a closed endpoint (`√x` at `0`) is tested on the
    /// interior instead, provided the function is continuous there.
    /// `None` means undecided — never a guess; expressions with
    /// discontinuous or opaque nodes (`floor`, `sign`, unknown functions)
    /// are always `None`.  The domain must be a union of intervals (`None`
    /// otherwise); the empty domain is vacuously `Some(true)`.
    ///
    /// A pole strictly inside the domain refutes monotonicity regardless of
    /// the sign of `f'` on either side: `1/x` is not decreasing on
    /// `[−1, 1]` (`f(−1) = −1 < 1 = f(1)`) and `tan x` is not increasing
    /// on `[0, π]`, although `f' < 0` resp. `f' > 0` wherever it is
    /// defined.  (SymPy tests the derivative alone and answers `True` for
    /// `is_increasing(tan(x), Interval(0, pi))`.)  When the singularities
    /// inside the domain cannot be enumerated (`tan x` on ℝ) the answer is
    /// `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let reals = ctx.reals();
    /// // SymPy: is_increasing(x**3, S.Reals, x) is True
    /// assert_eq!(x.powi(3).is_increasing(&x, &reals), Some(true));
    /// // SymPy: is_increasing(x**2, S.Reals, x) is False
    /// assert_eq!(x.powi(2).is_increasing(&x, &reals), Some(false));
    /// // SymPy: is_increasing(x**2, Interval(0, oo), x) is True
    /// let half = ctx.interval(&ctx.int(0), &ctx.infinity(), false, true);
    /// assert_eq!(x.powi(2).is_increasing(&x, &half), Some(true));
    /// // SymPy: is_increasing(exp(x), S.Reals, x) is True
    /// assert_eq!(x.exp().is_increasing(&x, &reals), Some(true));
    /// ```
    #[must_use]
    pub fn is_increasing(&self, var: &Ex, domain: &SetEx) -> Option<bool> {
        self.checked_id(var);
        self.checked_id(domain);
        monotone_on(self, var, domain, false)
    }

    /// Is `self` non-increasing in `var` on `domain` (`f' ≤ 0` there)?
    /// SymPy's `is_decreasing`.  Same method as
    /// [`is_increasing`](Ex::is_increasing).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// // SymPy: is_decreasing(x**2, Interval(-oo, 0), x) is True
    /// let left = ctx.interval(&ctx.neg_infinity(), &ctx.int(0), true, false);
    /// assert_eq!(x.powi(2).is_decreasing(&x, &left), Some(true));
    /// // SymPy: is_decreasing(1/x, Interval.open(0, oo), x) is True
    /// let pos = ctx.interval(&ctx.int(0), &ctx.infinity(), true, true);
    /// assert_eq!((1 / &x).is_decreasing(&x, &pos), Some(true));
    /// assert_eq!(x.powi(3).is_decreasing(&x, &ctx.reals()), Some(false));
    /// ```
    #[must_use]
    pub fn is_decreasing(&self, var: &Ex, domain: &SetEx) -> Option<bool> {
        self.checked_id(var);
        self.checked_id(domain);
        monotone_on(&-self, var, domain, false)
    }

    /// Is `self` strictly increasing in `var` on `domain`?  SymPy's
    /// `is_strictly_increasing`.
    ///
    /// Decided as `f' ≥ 0` with only isolated zeros: for polynomial and
    /// rational derivatives this is exact (`x³` is strictly increasing on
    /// ℝ although `f'(0) = 0`, and so is `x²` on `[0, ∞)`); otherwise
    /// `Some(true)` needs `f' > 0` on the domain or a finite zero set from
    /// the inequality solver, `Some(false)` needs `f' < 0` somewhere, and
    /// anything else is `None`.  (SymPy tests `domain ⊆ {f' > 0}` and
    /// answers `None` / `False` for `x³` on ℝ; the mathematically correct
    /// answer is returned here.)
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// assert_eq!(x.powi(3).is_strictly_increasing(&x, &ctx.reals()), Some(true));
    /// assert_eq!(x.powi(2).is_strictly_increasing(&x, &ctx.reals()), Some(false));
    /// // A constant is increasing but not strictly.
    /// assert_eq!(ctx.int(3).is_increasing(&x, &ctx.reals()), Some(true));
    /// assert_eq!(ctx.int(3).is_strictly_increasing(&x, &ctx.reals()), Some(false));
    /// ```
    #[must_use]
    pub fn is_strictly_increasing(&self, var: &Ex, domain: &SetEx) -> Option<bool> {
        self.checked_id(var);
        self.checked_id(domain);
        monotone_on(self, var, domain, true)
    }

    /// Is `self` strictly decreasing in `var` on `domain`?  SymPy's
    /// `is_strictly_decreasing`; see
    /// [`is_strictly_increasing`](Ex::is_strictly_increasing).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// assert_eq!((-&x.powi(3)).is_strictly_decreasing(&x, &ctx.reals()), Some(true));
    /// let pos = ctx.interval(&ctx.int(0), &ctx.infinity(), true, true);
    /// assert_eq!((1 / &x).is_strictly_decreasing(&x, &pos), Some(true));
    /// ```
    #[must_use]
    pub fn is_strictly_decreasing(&self, var: &Ex, domain: &SetEx) -> Option<bool> {
        self.checked_id(var);
        self.checked_id(domain);
        monotone_on(&-self, var, domain, true)
    }

    /// Is `self` monotonic (non-decreasing or non-increasing) in `var` on
    /// `domain`?  SymPy's `is_monotonic`.
    ///
    /// The three-valued disjunction of [`is_increasing`](Ex::is_increasing)
    /// and [`is_decreasing`](Ex::is_decreasing): `Some(true)` when either
    /// is proven, `Some(false)` when both are refuted, `None` otherwise.
    /// (SymPy's `is_monotonic` instead asks whether `f'` has *no* zeros in
    /// the domain and therefore answers `False` for `x³` on ℝ.)
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// assert_eq!(x.powi(3).is_monotonic(&x, &ctx.reals()), Some(true));
    /// assert_eq!((-&x).is_monotonic(&x, &ctx.reals()), Some(true));
    /// assert_eq!(x.powi(2).is_monotonic(&x, &ctx.reals()), Some(false));
    /// ```
    #[must_use]
    pub fn is_monotonic(&self, var: &Ex, domain: &SetEx) -> Option<bool> {
        or3(
            self.is_increasing(var, domain),
            self.is_decreasing(var, domain),
        )
    }

    /// Is `self` convex in `var` on `domain` (`f'' ≥ 0` there)?  SymPy's
    /// `is_convex` for one variable.
    ///
    /// Same machinery as [`is_increasing`](Ex::is_increasing), applied to
    /// the first derivative.  Three-valued.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// // SymPy: is_convex(x**2, x) is True, is_convex(x**3, x) is False
    /// assert_eq!(x.powi(2).is_convex(&x, &ctx.reals()), Some(true));
    /// assert_eq!(x.powi(3).is_convex(&x, &ctx.reals()), Some(false));
    /// // SymPy: is_convex(x**3, x, domain=Interval(0, oo)) is True
    /// let half = ctx.interval(&ctx.int(0), &ctx.infinity(), false, true);
    /// assert_eq!(x.powi(3).is_convex(&x, &half), Some(true));
    /// assert_eq!(x.exp().is_convex(&x, &ctx.reals()), Some(true));
    /// ```
    #[must_use]
    pub fn is_convex(&self, var: &Ex, domain: &SetEx) -> Option<bool> {
        self.checked_id(var);
        self.checked_id(domain);
        monotone_on(&self.diff(var), var, domain, false)
    }

    /// A period of `self` in `var` — SymPy's `periodicity`.  Like
    /// SymPy's, the value is *a* period, not necessarily the fundamental
    /// one: composite expressions get the lcm of the periods of their
    /// pieces, and identities that shorten the period are not detected
    /// (`sin²x·cos²x = sin²(2x)/4` gives `π`, whose fundamental period is
    /// `π/2` — SymPy answers `π/2` here through its own simplification).
    ///
    /// * `Some(0)` when `self` does not depend on `var`.
    /// * `sin(a·x + b)`, `cos(a·x + b)` → `2π/|a|`; `tan(a·x + b)` →
    ///   `π/|a|`; `sec`, `csc`, `cot` (which are built from `sin`/`cos`)
    ///   follow, with products `sin(g)ᵖ·cos(g)ᵠ` of even exponent sum
    ///   (`sin·cos`, `cos/sin`, `sin²`) and `|sin g|`, `|cos g|` getting the
    ///   half period `π/|a|`.
    /// * Sums, products, powers and compositions (`exp(sin x)`,
    ///   `sin(2x) + cos(3x)`) take the lcm of the periods of their
    ///   `var`-dependent parts; the lcm needs pairwise rational ratios.
    /// * `None` when a `var`-dependent part is not recognised as periodic
    ///   (`x²`, `sin(x²)`, `sin(x) + x`, `sin(√2·x) + sin(x)`).
    ///
    /// The expression is simplified first (`sin²x + cos²x` → `1` →
    /// `Some(0)`); the original form is tried if the simplified one is not
    /// recognised.  Note that SymPy reports `2π` for `sin(x)²`; the
    /// half-period rule gives `π` here.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let p = |e: &Ex| e.periodicity(&x).map(|p| p.to_string());
    /// // SymPy: periodicity(sin(2*x) + cos(3*x), x) == 2*pi
    /// assert_eq!(p(&(&(&x * 2).sin() + &(&x * 3).cos())), Some("2*pi".into()));
    /// // SymPy: periodicity(tan(x), x) == pi
    /// assert_eq!(p(&x.tan()), Some("pi".into()));
    /// // SymPy: periodicity(sin(3*x + 1), x) == 2*pi/3
    /// assert_eq!(p(&(&x * 3 + 1).sin()), Some("2/3*pi".into()));
    /// // SymPy: periodicity(S(3), x) == 0; periodicity(x**2, x) is None
    /// assert_eq!(p(&ctx.int(3)), Some("0".into()));
    /// assert_eq!(p(&x.powi(2)), None);
    /// ```
    #[must_use]
    pub fn periodicity(&self, var: &Ex) -> Option<Ex> {
        let var_id = self.checked_id(var);
        if !self.contains(var) {
            return Some(self.context().zero());
        }
        let simplified = self.simplify();
        for candidate in [simplified.raw_id(), self.raw_id()] {
            let period = {
                let mut inner = self.inner.write();
                util::periodicity(&mut inner.arena, candidate, var_id)
            };
            if let Some(id) = period {
                return Some(self.wrap(id).eval());
            }
        }
        None
    }

    /// The image of `self` (continuous in `var`) over `domain`, a union of
    /// intervals — SymPy's `function_range`.
    ///
    /// On each interval of the domain the infimum and supremum are found
    /// as in [`minimum`](Ex::minimum) / [`maximum`](Ex::maximum); the image
    /// of that interval is `[inf, sup]` with an endpoint open exactly when
    /// the value is only approached (a one-sided limit at an open or
    /// infinite endpoint that is not also attained elsewhere) or infinite.
    /// The pieces are united and simplified.
    ///
    /// # Errors
    ///
    /// * `InvalidArgument` — `var` is not a symbol, or `domain` is not a
    ///   union of intervals (the empty domain gives the empty set).
    /// * `ComputationFailed` — as for [`maximum`](Ex::maximum).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let r = |f: &Ex, d: &SetEx| f.function_range(&x, d).unwrap().to_string();
    /// // SymPy: function_range(sin(x), x, Interval(0, pi)) == Interval(0, 1)
    /// assert_eq!(r(&x.sin(), &ctx.interval(&ctx.int(0), &ctx.pi(), false, false)), "[0, 1]");
    /// // SymPy: function_range(x**2, x, S.Reals) == Interval(0, oo)
    /// assert_eq!(r(&x.powi(2), &ctx.reals()), "[0, oo)");
    /// // SymPy: function_range(1/x, x, Interval(1, oo)) == Interval.Lopen(0, 1)
    /// let tail = ctx.interval(&ctx.int(1), &ctx.infinity(), false, true);
    /// assert_eq!(r(&(1 / &x), &tail), "(0, 1]");
    /// // SymPy: function_range(exp(x), x, S.Reals) == Interval.open(0, oo)
    /// assert_eq!(r(&x.exp(), &ctx.reals()), "(0, oo)");
    /// ```
    pub fn function_range(&self, var: &Ex, domain: &SetEx) -> Result<SetEx, SymplexError> {
        const OP: &str = "function_range";
        self.checked_id(var);
        self.checked_id(domain);
        require_symbol(OP, var)?;
        let ctx = self.context();
        let Some(parts) = domain.as_intervals() else {
            return Err(invalid(
                OP,
                format!("the domain `{domain}` is not a union of intervals"),
            ));
        };
        if parts.is_empty() {
            return Ok(ctx.empty_set());
        }
        self.require_continuous(var, domain, OP)?;
        let (inf, ninf) = (ctx.infinity(), ctx.neg_infinity());
        let mut result = ctx.empty_set();
        for (lo, hi, lo_open, hi_open) in &parts {
            let cands = self.candidates_on(var, (lo, hi, *lo_open, *hi_open), OP)?;
            let (min, max) = min_max(cands, OP)?;
            let piece = if min.value == max.value {
                ctx.finite_set(&[min.value])
            } else {
                let left_open = !min.attained || min.value == ninf;
                let right_open = !max.attained || max.value == inf;
                ctx.interval(&min.value, &max.value, left_open, right_open)
            };
            result = result.union(&piece);
        }
        Ok(result.simplify())
    }
}
