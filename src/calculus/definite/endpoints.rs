//! Endpoint values and one-sided limits (part of [`definite`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Endpoint values and one-sided limits
// ═══════════════════════════════════════════════════════════════════════════

/// Direction of approach to an endpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Side {
    /// `x → c⁺` (lower endpoint).
    FromRight,
    /// `x → c⁻` (upper endpoint).
    FromLeft,
}

/// Does `F` contain a node with a jump (so a direct substitution at a
/// singular endpoint might not equal the one-sided limit)?
fn has_jump_node(arena: &Arena, root: ExprId) -> bool {
    let mut stack = vec![root];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Sign(_)
            | ExprNode::Heaviside(_)
            | ExprNode::Floor(_)
            | ExprNode::Ceiling(_)
            | ExprNode::Piecewise(_)
            | ExprNode::Tan(_)
            | ExprNode::Atan2(_, _) => return true,
            node => node.for_each_child(|c| stack.push(c)),
        }
    }
    false
}

/// Value of `F` at endpoint `c`, approached from `side`: finite, `±∞`, or
/// `None` when unknown.
pub(super) fn endpoint_value(
    arena: &mut Arena,
    anti: ExprId,
    x: ExprId,
    c: ExprId,
    side: Side,
    singular: bool,
) -> Option<Extended<ExprId>> {
    if !is_infinite(arena, c) {
        // Direct substitution.  Valid when F is continuous at c; when c is
        // a singular endpoint we still accept a finite value provided F
        // has no jump-type nodes (its continuous extension equals the
        // one-sided limit for elementary F).
        let v = subs::subs(arena, anti, x, c);
        let v = safe_eval(arena, v);
        if let Some(numeric) = finite_value(arena, v, x)
            && (!singular || !has_jump_node(arena, anti))
        {
            // A value at a point of a cut may be the principal value from
            // the wrong side, a real one too: the principal `±iπ` of `ln u −
            // ln w` cancel when `u`, `w` are on the cut, while the path may
            // reach them from opposite sides (up to 0.52 only a value with
            // an `f64` imaginary part was checked).
            if numeric.is_some() && endpoint_on_a_cut(arena, anti, x, c, &[]) {
                return None;
            }
            return Some(Extended::Finite(v));
        }
        // One-sided limit: x = c ± 1/u, u → +∞.
        let u = fresh_symbol(arena, "defu", 0);
        let one = arena.one();
        let inv_u = arena.div(one, u);
        let repl = match side {
            Side::FromRight => arena.add(&[c, inv_u]),
            Side::FromLeft => arena.sub(c, inv_u),
        };
        let g = subs::subs(arena, anti, x, repl);
        let g = safe_eval(arena, g);
        return limit_pos_inf(arena, g, u);
    }
    if real_infinity(arena, c) == Some(Sign::Positive) {
        return limit_pos_inf(arena, anti, x);
    }
    // c = −∞: x = −u.
    let u = fresh_symbol(arena, "defu", 1);
    let neg_u = arena.neg(u);
    let g = subs::subs(arena, anti, x, neg_u);
    let g = safe_eval(arena, g);
    limit_pos_inf(arena, g, u)
}

/// Does a principal-branch argument `u` of `F` lie on its cut at the
/// finite endpoint `c` while `u` leaves the cut's axis inside the
/// interval?  Then `F(c)` (the principal value, the limit from one side of
/// the cut) need not be the limit of `F` along the path, which may come
/// from the other side, and the endpoint value is unknown.  Arguments that
/// are real for real `x`, or whose imaginary part does not depend on `x`,
/// run along the axis (on the cut or parallel to it) and are fine; so are
/// arguments with parameters, which [`check_branch_cuts`] has vetted, and
/// those in `skip` (continued from a known side, [`side_value`]).
pub(super) fn endpoint_on_a_cut(
    arena: &mut Arena,
    anti: ExprId,
    x: ExprId,
    c: ExprId,
    skip: &[ExprId],
) -> bool {
    let arguments = cut_arguments(arena, anti, x);
    if arguments.is_empty() {
        return false;
    }
    let mut reals = AssumptionCache::new();
    for (u, cut) in arguments {
        if skip.contains(&u)
            || real_for_real_x(arena, u, x, &mut reals)
            || (cut != Cut::ImaginaryOutside
                && imaginary_part_constant_polynomial(arena, u, x, &mut reals))
        {
            continue;
        }
        let at = subs::subs(arena, u, x, c);
        let at = safe_eval(arena, at);
        if !walk::free_symbols(arena, at).is_empty() {
            continue;
        }
        let Ok(z) = evalf::evalf_complex64(arena, at) else {
            return true;
        };
        // The tolerance only widens a refusal: an end value near a cut is
        // unknown.
        let tiny = 1e-12 * z.norm().max(1.0);
        let on_cut = match cut {
            Cut::RealBelow(b) => z.im.abs() <= tiny && z.re < b,
            Cut::RealAbove(b) => z.im.abs() <= tiny && z.re > b,
            Cut::RealOutside => z.im.abs() <= tiny && z.re.abs() > 1.0,
            Cut::ImaginaryOutside => z.re.abs() <= tiny && z.im.abs() > 1.0,
        };
        if on_cut {
            return true;
        }
    }
    false
}

/// `lim_{u → +∞} g(u)`: finite, `±∞`, or `None` when unknown.
///
/// Compositional rules (with assumptions on parameters) first; then the
/// black-box limit engine for parameter-free expressions, whose finite
/// answers are checked against numeric samples.
pub(super) fn limit_pos_inf(arena: &mut Arena, g: ExprId, u: ExprId) -> Option<Extended<ExprId>> {
    if !walk::contains(arena, g, u) {
        let v = safe_eval(arena, g);
        return if is_finite_value(arena, v, u) {
            Some(Extended::Finite(v))
        } else {
            None
        };
    }
    match compositional_limit(arena, g, u) {
        LimVal::Value(Extended::Finite(v)) => {
            if is_finite_value(arena, v, u) {
                return Some(Extended::Finite(v));
            }
        }
        LimVal::Value(infinite) => return Some(infinite),
        LimVal::Bounded | LimVal::Unknown => {}
    }
    if let Some(v) = log_polylog_limit(arena, g, u) {
        return Some(Extended::Finite(v));
    }

    // Black-box engine: only trusted when no parameters are present and
    // the expression has no unbounded oscillation (the engine can report
    // `∞` for `u·sin(u²)`, whose limit does not exist).
    let params: Vec<ExprId> = walk::free_symbols(arena, g)
        .into_iter()
        .filter(|&s| s != u)
        .collect();
    if params.is_empty() && !has_unbounded_oscillation(arena, g, u) {
        let inf = arena.infinity();
        if let Ok(l) = crate::calculus::limit::limit(arena, g, u, inf) {
            // The engine occasionally leaks its internal dummy symbols into
            // a half-finished result; a limit of a parameter-free
            // expression must itself be parameter-free.
            if !walk::free_symbols(arena, l).is_empty() {
                return None;
            }
            if let Some(direction) = real_infinity(arena, l) {
                let sign: i8 = if direction == Sign::Positive { 1 } else { -1 };
                if numeric_growth_sanity(arena, g, u, sign) {
                    return Some(if sign > 0 {
                        Extended::PosInf
                    } else {
                        Extended::NegInf
                    });
                }
                return None;
            }
            // A complex limit would escape the numeric sanity check (it
            // samples a compiled real function): only real ones are taken.
            if is_finite_value(arena, l, u)
                && calculus_util::expr_to_f64(arena, l).is_some()
                && numeric_limit_sanity(arena, g, u, l)
            {
                return Some(Extended::Finite(l));
            }
        }
    }
    None
}

/// `lim_{u→+∞} g` for a `g` that is (multiplied out) a sum of terms
/// `c·ln A`, `c·ln A·ln B`, `c·Li₂(C)` with `A`, `B`, `C` linear in `u` (numeric
/// slopes) and terms with finite limits: with `ℓ = ln u`, `ln(a·u + b) = ℓ +
/// λ + o(1)` ([`asymptotic_log_shift`]) and `Li₂(C) = −π²/6 − ½·ln²(−C) −
/// Li₂(1/C)` (DLMF 25.12.4, `C` off `[0, ∞)`), each term is `α·ℓ² + β·ℓ + γ +
/// o(1)`; when the `α` and the `β` add up to 0 the limit is the sum of the
/// `γ`.  The antiderivative of `ln(x + c)/D(x)` is such a sum of `Li₂` and
/// products of logarithms at the roots of `D`, whose `ln²u` terms cancel
/// only in the sum: `∫₀^∞ ln(x + 1)/(x² + 1) dx = π·ln 2/4 + G` was refused up
/// to 0.46.  The sums are decided exactly, or for a parameter-free `g` by a
/// certified zero test ([`calculus_util::hidden_zero`]); a sum that is 0
/// only to the precision reached is no proof.  Up to 0.52 a sum below
/// `10⁻²⁵` at 30 digits counted as 0, and `∫₀^∞ (ln(x + 1)/(x² + 1) +
/// e⁻⁶⁰·ln(x + 1)/(x + 2)) dx`, which diverges (its `ln²u` coefficient is
/// `e⁻⁶⁰/2`), was `1.4604`: the quadrature check of [`integrate_definite`]
/// does not run on an infinite range.
fn log_polylog_limit(arena: &mut Arena, g: ExprId, u: ExprId) -> Option<ExprId> {
    let has_target = walk::post_order_ids(arena, g).into_iter().any(|id| {
        matches!(arena.node(id), ExprNode::Lib(crate::base::libfn::LibFn::PolyLog, args)
            if args.len() == 2)
    });
    if !has_target {
        return None;
    }
    let e = expand::expand(arena, g);
    let terms: Vec<ExprId> = match arena.node(e) {
        ExprNode::Add(children) => children.to_vec(),
        _ => vec![e],
    };
    let mut sq: Vec<ExprId> = Vec::new();
    let mut lin: Vec<ExprId> = Vec::new();
    let mut cst: Vec<ExprId> = Vec::new();
    let mut rest: Vec<ExprId> = Vec::new();
    let order_two = arena.int(2);
    for t in terms {
        if !walk::contains(arena, t, u) {
            cst.push(t);
            continue;
        }
        let factors: Vec<ExprId> = match arena.node(t) {
            ExprNode::Mul(fs) => fs.to_vec(),
            _ => vec![t],
        };
        let (dep, coef): (Vec<ExprId>, Vec<ExprId>) = factors
            .into_iter()
            .partition(|&f| walk::contains(arena, f, u));
        let c = arena.mul(&coef);
        let mut logs: Vec<ExprId> = Vec::new();
        let mut li2: Option<ExprId> = None;
        let mut ok = true;
        for &d in &dep {
            match arena.node(d).clone() {
                ExprNode::Ln(a) => logs.push(a),
                ExprNode::Pow(b, n)
                    if matches!(arena.node(b), ExprNode::Ln(_))
                        && arena
                            .as_num(n)
                            .is_some_and(|q| *q == Q::from_integer(2.into())) =>
                {
                    let ExprNode::Ln(a) = *arena.node(b) else {
                        return None;
                    };
                    logs.push(a);
                    logs.push(a);
                }
                ExprNode::Lib(crate::base::libfn::LibFn::PolyLog, args)
                    if args.len() == 2 && args[0] == order_two && li2.is_none() =>
                {
                    li2 = Some(args[1]);
                }
                _ => ok = false,
            }
        }
        if !ok || (li2.is_some() && !logs.is_empty()) || logs.len() > 2 {
            rest.push(t);
            continue;
        }
        if let Some(z) = li2 {
            // −π²/6 − ½(ℓ + μ)² with μ the shift of ln(−C).
            let neg_z = arena.neg(z);
            let neg_z = safe_eval(arena, neg_z);
            let mu = asymptotic_log_shift(arena, neg_z, u, true)?;
            let minus_half = arena.rational(-1, 2);
            sq.push(arena.mul(&[c, minus_half]));
            let minus_one = arena.int(-1);
            lin.push(arena.mul(&[c, minus_one, mu]));
            let pi = arena.pi();
            let two = arena.int(2);
            let pi2 = arena.pow(pi, two);
            let sixth = arena.rational(-1, 6);
            let mu2 = arena.pow(mu, two);
            let g0 = arena.mul(&[sixth, pi2]);
            let g1 = arena.mul(&[minus_half, mu2]);
            let g = arena.add(&[g0, g1]);
            cst.push(arena.mul(&[c, g]));
            continue;
        }
        let lambdas: Vec<ExprId> = logs
            .iter()
            .map(|&a| asymptotic_log_shift(arena, a, u, false))
            .collect::<Option<_>>()?;
        match lambdas.len() {
            1 => {
                lin.push(c);
                cst.push(arena.mul(&[c, lambdas[0]]));
            }
            2 => {
                sq.push(c);
                let s = arena.add(&[lambdas[0], lambdas[1]]);
                lin.push(arena.mul(&[c, s]));
                cst.push(arena.mul(&[c, lambdas[0], lambdas[1]]));
            }
            _ => rest.push(t),
        }
    }
    let vanishes = |arena: &mut Arena, parts: &[ExprId]| -> bool {
        let s = arena.add(parts);
        let s = safe_eval(arena, s);
        if arena.is_zero_structural(s) {
            return true;
        }
        let s = expand::expand(arena, s);
        let s = safe_eval(arena, s);
        arena.is_zero_structural(s)
            || (walk::free_symbols(arena, s).is_empty()
                && calculus_util::hidden_zero(arena, s)
                    .proven()
                    .is_some_and(|t| t.is_true()))
    };
    if !vanishes(arena, &sq) || !vanishes(arena, &lin) {
        return None;
    }
    if !rest.is_empty() {
        let r = arena.add(&rest);
        match compositional_limit(arena, r, u) {
            LimVal::Value(Extended::Finite(v)) => cst.push(v),
            _ => return None,
        }
    }
    let total = arena.add(&cst);
    let total = safe_eval(arena, total);
    is_finite_value(arena, total, u).then_some(total)
}

/// `λ` with `ln(a·u + b) = ln u + λ + o(1)` as `u → +∞`, for a numeric
/// slope `a ≠ 0` and a constant `b`: `ln a` off the negative axis; on it
/// `ln|a| ± iπ` above or below the axis by the sign of `im b` (`+iπ` on the
/// axis itself, the principal value).  With `avoid_positive_axis` (the
/// argument `−C` of a `Li₂(C)` inversion), `None` when `a·u + b` runs along
/// the negative axis: `C` on `Li₂`'s cut.
fn asymptotic_log_shift(
    arena: &mut Arena,
    arg: ExprId,
    u: ExprId,
    avoid_positive_axis: bool,
) -> Option<ExprId> {
    let LinearCoeffs { slope, intercept } = calculus_util::linear_coeffs_exact(arena, arg, u)?;
    if walk::contains(arena, slope, u)
        || walk::contains(arena, intercept, u)
        || !walk::free_symbols(arena, slope).is_empty()
    {
        return None;
    }
    let a = evalf::evalf_complex64(arena, slope).ok()?;
    if a.norm().is_nan() || a.norm() <= 0.0 {
        return None;
    }
    // The axis and the side are decided by certified signs: up to 0.52 an
    // imaginary part below `10⁻¹²` of the value counted as 0, so a slope
    // `−1 − i·ε` was on the cut (`ln|a| + iπ` for the `ln|a| − iπ` it tends
    // to), and an intercept `−i·ε` on the axis.
    // A part of `evalf_complex64` other than `0.0` is certified nonzero.
    let slope_side = if a.im > 0.0 {
        SignSet::POSITIVE
    } else if a.im < 0.0 {
        SignSet::NEGATIVE
    } else {
        crate::transforms::realness::imaginary_part_sign(arena, slope)
    };
    if slope_side == SignSet::REAL {
        return None;
    }
    if slope_side == SignSet::ZERO && a.re < 0.0 {
        if !walk::free_symbols(arena, intercept).is_empty() {
            return None;
        }
        let side = crate::transforms::realness::imaginary_part_sign(arena, intercept);
        if side == SignSet::REAL {
            return None;
        }
        let on_axis = side == SignSet::ZERO;
        if avoid_positive_axis && on_axis {
            return None;
        }
        let abs_a = arena.neg(slope);
        let ln_abs = arena.ln(abs_a);
        let pi = arena.pi();
        let i = arena.i_unit();
        let sg = arena.int(if side == SignSet::NEGATIVE { -1 } else { 1 });
        let i_pi = arena.mul(&[sg, i, pi]);
        return Some(arena.add(&[ln_abs, i_pi]));
    }
    Some(arena.ln(slope))
}

/// Does `g` contain `sin`/`cos`/`tan` of an argument that does not
/// converge as `u → ∞`?
pub(super) fn has_unbounded_oscillation(arena: &mut Arena, g: ExprId, u: ExprId) -> bool {
    let order = walk::post_order_ids(arena, g);
    for id in order {
        let arg = match arena.node(id) {
            ExprNode::Sin(a) | ExprNode::Cos(a) | ExprNode::Tan(a) => *a,
            _ => continue,
        };
        if !walk::contains(arena, arg, u) {
            continue;
        }
        match compositional_limit(arena, arg, u) {
            LimVal::Value(Extended::Finite(_)) => {}
            _ => return true,
        }
    }
    false
}

/// Check a claimed `±∞` limit of `g(u)` against samples: `|g|` must grow
/// monotonically with the claimed sign at `u = 10², 10⁴, 10⁶`.
pub(super) fn numeric_growth_sanity(arena: &mut Arena, g: ExprId, u: ExprId, sign: i8) -> bool {
    let u_name = match arena.node(u) {
        ExprNode::Symbol(sid) => arena.symbol_name(*sid).to_string(),
        _ => return true,
    };
    let Ok(func) = crate::output::lambdify::compile(arena, g, &[&u_name]) else {
        return true;
    };
    let samples = [func(&[1e2]), func(&[1e4]), func(&[1e6])];
    if samples.iter().any(|v| v.is_nan()) {
        return true;
    }
    let mut prev = 0.0f64;
    for v in samples {
        if v.is_infinite() {
            return (v > 0.0) == (sign > 0);
        }
        if (v > 0.0) != (sign > 0) || v.abs() <= prev {
            return false;
        }
        prev = v.abs();
    }
    true
}

/// Check a claimed finite limit `l` of `g(u)` at `+∞` against samples at
/// `u = 10³` and `u = 10⁶`.  Accepts when either the far sample is close
/// or the discrepancy is clearly shrinking; rejects obvious contradictions.
fn numeric_limit_sanity(arena: &mut Arena, g: ExprId, u: ExprId, l: ExprId) -> bool {
    let u_name = match arena.node(u) {
        ExprNode::Symbol(sid) => arena.symbol_name(*sid).to_string(),
        _ => return true,
    };
    let Some(lv) = calculus_util::expr_to_f64(arena, l) else {
        return true;
    };
    let Ok(func) = crate::output::lambdify::compile(arena, g, &[&u_name]) else {
        return true;
    };
    let g3 = func(&[1e3]);
    let g6 = func(&[1e6]);
    if !g3.is_finite() || !g6.is_finite() {
        return true;
    }
    let scale = lv.abs().max(1.0);
    let d3 = (g3 - lv).abs();
    let d6 = (g6 - lv).abs();
    // A sanity check that can only reject the engine's limit.
    if d6 <= 1e-3 * scale {
        return true;
    }
    d6 < 0.5 * d3
}

/// Value classes for the compositional limit evaluator.
#[derive(Clone, Copy, Debug)]
enum LimVal {
    /// A limit on the extended line: a finite value or `±∞`.
    Value(Extended<ExprId>),
    /// Bounded but not convergent (e.g. `sin(u)`).
    Bounded,
    Unknown,
}

/// Bottom-up limit evaluation over the expression DAG (explicit
/// post-order, no recursion).  Handles sums, products, powers, exp, ln,
/// atan/erf/tanh saturation, and bounded oscillation.
fn compositional_limit(arena: &mut Arena, g: ExprId, u: ExprId) -> LimVal {
    let order = walk::post_order_ids(arena, g);
    let mut vals: FxHashMap<ExprId, LimVal> = FxHashMap::default();
    // Nodes that tend to 0 faster than any power of `u` grows
    // ([`exponentially_small`]).
    let mut small: FxHashSet<ExprId> = FxHashSet::default();

    // Leading terms `c·u^d` of the nodes that are algebraic in `u`.
    let mut lead: FxHashMap<ExprId, (Q, ExprId)> = FxHashMap::default();

    for id in order {
        let node = arena.node(id).clone();
        if exponentially_small(arena, &node, &vals, &small, u) {
            small.insert(id);
        }
        if let Some(t) = leading_term(arena, id, &node, &lead, u) {
            lead.insert(id, t);
        }
        let v = if id == u {
            LimVal::Value(Extended::PosInf)
        } else if !walk::contains(arena, id, u) {
            LimVal::Value(Extended::Finite(id))
        } else {
            match node {
                ExprNode::Mul(_) | ExprNode::Add(_) if small.contains(&id) => {
                    LimVal::Value(Extended::Finite(arena.zero()))
                }
                ExprNode::Add(children) => {
                    let mut finite: Vec<ExprId> = Vec::new();
                    let mut pos = 0usize;
                    let mut neg = 0usize;
                    let mut bounded = false;
                    let mut unknown = false;
                    let (grouped, group_value) = linear_log_group(arena, &children, u);
                    if let Some(v) = group_value {
                        finite.push(v);
                    }
                    for c in children.iter() {
                        if grouped.contains(c) {
                            continue;
                        }
                        match vals.get(c).copied().unwrap_or(LimVal::Unknown) {
                            LimVal::Value(Extended::Finite(v)) => finite.push(v),
                            LimVal::Value(Extended::PosInf) => pos += 1,
                            LimVal::Value(Extended::NegInf) => neg += 1,
                            LimVal::Bounded => bounded = true,
                            LimVal::Unknown => unknown = true,
                        }
                    }
                    if unknown || (pos > 0 && neg > 0) {
                        LimVal::Unknown
                    } else if pos > 0 {
                        LimVal::Value(Extended::PosInf)
                    } else if neg > 0 {
                        LimVal::Value(Extended::NegInf)
                    } else if bounded {
                        LimVal::Bounded
                    } else {
                        let s = arena.add(&finite);
                        LimVal::Value(Extended::Finite(safe_eval(arena, s)))
                    }
                }
                ExprNode::Mul(children) => {
                    let mut finite: Vec<ExprId> = Vec::new();
                    let mut infs = 0usize;
                    let mut neg_infs = 0usize;
                    let mut bounded = false;
                    let mut unknown = false;
                    for c in children.iter() {
                        match vals.get(c).copied().unwrap_or(LimVal::Unknown) {
                            LimVal::Value(Extended::Finite(v)) => finite.push(v),
                            LimVal::Value(Extended::PosInf) => infs += 1,
                            LimVal::Value(Extended::NegInf) => {
                                infs += 1;
                                neg_infs += 1;
                            }
                            LimVal::Bounded => bounded = true,
                            LimVal::Unknown => unknown = true,
                        }
                    }
                    if unknown {
                        LimVal::Unknown
                    } else {
                        let prod = if finite.is_empty() {
                            arena.one()
                        } else {
                            let m = arena.mul(&finite);
                            safe_eval(arena, m)
                        };
                        let psign = sign_of(arena, prod);
                        if infs > 0 {
                            match psign {
                                Some(Ordering::Equal) | None => LimVal::Unknown,
                                Some(s) => {
                                    if bounded {
                                        LimVal::Unknown
                                    } else {
                                        let negative = (s == Ordering::Less) ^ (neg_infs % 2 == 1);
                                        if negative {
                                            LimVal::Value(Extended::NegInf)
                                        } else {
                                            LimVal::Value(Extended::PosInf)
                                        }
                                    }
                                }
                            }
                        } else if bounded {
                            if psign == Some(Ordering::Equal) {
                                LimVal::Value(Extended::Finite(arena.zero()))
                            } else {
                                LimVal::Bounded
                            }
                        } else {
                            LimVal::Value(Extended::Finite(prod))
                        }
                    }
                }
                ExprNode::Neg(inner) => {
                    match vals.get(&inner).copied().unwrap_or(LimVal::Unknown) {
                        LimVal::Value(Extended::Finite(v)) => {
                            let n = arena.neg(v);
                            LimVal::Value(Extended::Finite(safe_eval(arena, n)))
                        }
                        LimVal::Value(Extended::PosInf) => LimVal::Value(Extended::NegInf),
                        LimVal::Value(Extended::NegInf) => LimVal::Value(Extended::PosInf),
                        other => other,
                    }
                }
                ExprNode::Pow(base, exp) => {
                    let bv = vals.get(&base).copied().unwrap_or(LimVal::Unknown);
                    let ev = vals.get(&exp).copied().unwrap_or(LimVal::Unknown);
                    pow_limit(arena, base, exp, bv, ev, u)
                }
                ExprNode::Exp(inner) => {
                    match vals.get(&inner).copied().unwrap_or(LimVal::Unknown) {
                        LimVal::Value(Extended::Finite(v)) => {
                            let e = arena.exp(v);
                            LimVal::Value(Extended::Finite(safe_eval(arena, e)))
                        }
                        LimVal::Value(Extended::PosInf) => LimVal::Value(Extended::PosInf),
                        LimVal::Value(Extended::NegInf) => {
                            LimVal::Value(Extended::Finite(arena.zero()))
                        }
                        LimVal::Bounded => LimVal::Bounded,
                        LimVal::Unknown => LimVal::Unknown,
                    }
                }
                ExprNode::Ln(inner) => match vals.get(&inner).copied().unwrap_or(LimVal::Unknown) {
                    LimVal::Value(Extended::Finite(v)) => match sign_of(arena, v) {
                        Some(Ordering::Equal) => LimVal::Value(Extended::NegInf),
                        Some(_) => {
                            let l = arena.ln(v);
                            LimVal::Value(Extended::Finite(safe_eval(arena, l)))
                        }
                        None => LimVal::Unknown,
                    },
                    LimVal::Value(Extended::PosInf) => LimVal::Value(Extended::PosInf),
                    _ => LimVal::Unknown,
                },
                ExprNode::Atan(inner) => match vals.get(&inner).copied().unwrap_or(LimVal::Unknown)
                {
                    LimVal::Value(Extended::Finite(v)) => {
                        let a = arena.atan(v);
                        LimVal::Value(Extended::Finite(safe_eval(arena, a)))
                    }
                    LimVal::Value(Extended::PosInf) => {
                        let half_pi = arena.rational(1, 2);
                        let pi = arena.pi();
                        LimVal::Value(Extended::Finite(arena.mul(&[half_pi, pi])))
                    }
                    LimVal::Value(Extended::NegInf) => {
                        let neg_half = arena.rational(-1, 2);
                        let pi = arena.pi();
                        LimVal::Value(Extended::Finite(arena.mul(&[neg_half, pi])))
                    }
                    _ => LimVal::Unknown,
                },
                ExprNode::Tanh(inner) | ExprNode::Erf(inner) => {
                    match vals.get(&inner).copied().unwrap_or(LimVal::Unknown) {
                        LimVal::Value(Extended::Finite(v)) => {
                            let r = if matches!(node, ExprNode::Tanh(_)) {
                                arena.tanh(v)
                            } else {
                                arena.erf(v)
                            };
                            LimVal::Value(Extended::Finite(safe_eval(arena, r)))
                        }
                        LimVal::Value(Extended::PosInf) => {
                            LimVal::Value(Extended::Finite(arena.one()))
                        }
                        LimVal::Value(Extended::NegInf) => {
                            LimVal::Value(Extended::Finite(arena.neg_one()))
                        }
                        _ => LimVal::Unknown,
                    }
                }
                ExprNode::Erfc(inner) => match vals.get(&inner).copied().unwrap_or(LimVal::Unknown)
                {
                    LimVal::Value(Extended::Finite(v)) => {
                        let r = arena.erfc(v);
                        LimVal::Value(Extended::Finite(safe_eval(arena, r)))
                    }
                    LimVal::Value(Extended::PosInf) => {
                        LimVal::Value(Extended::Finite(arena.zero()))
                    }
                    LimVal::Value(Extended::NegInf) => {
                        LimVal::Value(Extended::Finite(arena.int(2)))
                    }
                    _ => LimVal::Unknown,
                },
                ExprNode::Sin(inner) | ExprNode::Cos(inner) => {
                    match vals.get(&inner).copied().unwrap_or(LimVal::Unknown) {
                        LimVal::Value(Extended::Finite(v)) => {
                            let r = if matches!(node, ExprNode::Sin(_)) {
                                arena.sin(v)
                            } else {
                                arena.cos(v)
                            };
                            LimVal::Value(Extended::Finite(safe_eval(arena, r)))
                        }
                        LimVal::Value(Extended::PosInf)
                        | LimVal::Value(Extended::NegInf)
                        | LimVal::Bounded => LimVal::Bounded,
                        LimVal::Unknown => LimVal::Unknown,
                    }
                }
                ExprNode::Sinh(inner) | ExprNode::Asinh(inner) => {
                    match vals.get(&inner).copied().unwrap_or(LimVal::Unknown) {
                        LimVal::Value(Extended::Finite(v)) => {
                            let r = if matches!(node, ExprNode::Sinh(_)) {
                                arena.sinh(v)
                            } else {
                                arena.asinh(v)
                            };
                            LimVal::Value(Extended::Finite(safe_eval(arena, r)))
                        }
                        LimVal::Value(Extended::PosInf) => LimVal::Value(Extended::PosInf),
                        LimVal::Value(Extended::NegInf) => LimVal::Value(Extended::NegInf),
                        _ => LimVal::Unknown,
                    }
                }
                ExprNode::Cosh(inner) => match vals.get(&inner).copied().unwrap_or(LimVal::Unknown)
                {
                    LimVal::Value(Extended::Finite(v)) => {
                        let r = arena.cosh(v);
                        LimVal::Value(Extended::Finite(safe_eval(arena, r)))
                    }
                    LimVal::Value(Extended::PosInf) | LimVal::Value(Extended::NegInf) => {
                        LimVal::Value(Extended::PosInf)
                    }
                    _ => LimVal::Unknown,
                },
                ExprNode::Abs(inner) => {
                    match vals.get(&inner).copied().unwrap_or(LimVal::Unknown) {
                        LimVal::Value(Extended::Finite(v)) => {
                            let r = arena.abs(v);
                            LimVal::Value(Extended::Finite(safe_eval(arena, r)))
                        }
                        LimVal::Value(Extended::PosInf) | LimVal::Value(Extended::NegInf) => {
                            LimVal::Value(Extended::PosInf)
                        }
                        LimVal::Bounded => LimVal::Bounded,
                        LimVal::Unknown => LimVal::Unknown,
                    }
                }
                ExprNode::Sign(inner) => match vals.get(&inner).copied().unwrap_or(LimVal::Unknown)
                {
                    LimVal::Value(Extended::PosInf) => LimVal::Value(Extended::Finite(arena.one())),
                    LimVal::Value(Extended::NegInf) => {
                        LimVal::Value(Extended::Finite(arena.neg_one()))
                    }
                    LimVal::Value(Extended::Finite(v)) => match sign_of(arena, v) {
                        Some(Ordering::Greater) => LimVal::Value(Extended::Finite(arena.one())),
                        Some(Ordering::Less) => LimVal::Value(Extended::Finite(arena.neg_one())),
                        _ => LimVal::Unknown,
                    },
                    _ => LimVal::Unknown,
                },
                ExprNode::Acosh(inner) | ExprNode::Gamma(inner) => {
                    match vals.get(&inner).copied().unwrap_or(LimVal::Unknown) {
                        LimVal::Value(Extended::PosInf) => LimVal::Value(Extended::PosInf),
                        LimVal::Value(Extended::Finite(v)) if is_positive(arena, v) => {
                            let r = if matches!(node, ExprNode::Acosh(_)) {
                                arena.acosh(v)
                            } else {
                                arena.gamma(v)
                            };
                            LimVal::Value(Extended::Finite(safe_eval(arena, r)))
                        }
                        _ => LimVal::Unknown,
                    }
                }
                ExprNode::Asin(inner) | ExprNode::Acos(inner) | ExprNode::Atanh(inner) => {
                    match vals.get(&inner).copied().unwrap_or(LimVal::Unknown) {
                        LimVal::Value(Extended::Finite(v)) => {
                            let r = match node {
                                ExprNode::Asin(_) => arena.asin(v),
                                ExprNode::Acos(_) => arena.acos(v),
                                _ => arena.atanh(v),
                            };
                            let r = safe_eval(arena, r);
                            if is_finite_value(arena, r, u) {
                                LimVal::Value(Extended::Finite(r))
                            } else {
                                LimVal::Unknown
                            }
                        }
                        _ => LimVal::Unknown,
                    }
                }
                ExprNode::Si(inner) | ExprNode::Ci(inner) | ExprNode::Ei(inner) => {
                    let kind = match node {
                        ExprNode::Si(_) => SineIntegral::Si,
                        ExprNode::Ci(_) => SineIntegral::Ci,
                        _ => SineIntegral::Ei,
                    };
                    match vals.get(&inner).copied().unwrap_or(LimVal::Unknown) {
                        LimVal::Value(Extended::Finite(v)) => integral_function_at(arena, kind, v),
                        // A complex slope leaves the argument's limit
                        // unknown here; the linear form decides.
                        _ => integral_function_at_infinity(arena, kind, inner, u),
                    }
                }
                _ => LimVal::Unknown,
            }
        };
        let v = match (v, lead.get(&id)) {
            (LimVal::Unknown, Some((d, c))) => leading_term_limit(arena, d, *c),
            (v, _) => v,
        };
        vals.insert(id, v);
    }
    vals.get(&g).copied().unwrap_or(LimVal::Unknown)
}

/// The leading term `c·u^d` (`u → +∞`) of the node `id` (children done) when
/// it is algebraic in `u`: a constant, `u`, a sum (the top degree's
/// coefficients must not cancel structurally), a product, a constant
/// power (of a positive leading coefficient unless integer).
fn leading_term(
    arena: &mut Arena,
    id: ExprId,
    node: &ExprNode,
    lead: &FxHashMap<ExprId, (Q, ExprId)>,
    u: ExprId,
) -> Option<(Q, ExprId)> {
    if id == u {
        return Some((Q::one(), arena.one()));
    }
    if !walk::contains(arena, id, u) {
        return Some((Q::zero(), id));
    }
    match node {
        ExprNode::Add(children) => {
            let terms: Vec<(Q, ExprId)> = children
                .iter()
                .map(|c| lead.get(c).cloned())
                .collect::<Option<_>>()?;
            let top = terms.iter().map(|(d, _)| d.clone()).max()?;
            let coeffs: Vec<ExprId> = terms
                .iter()
                .filter(|(d, _)| *d == top)
                .map(|&(_, c)| c)
                .collect();
            let c = arena.add(&coeffs);
            let c = safe_eval(arena, c);
            if arena.is_zero_structural(c) || sign_of(arena, c) == Some(Ordering::Equal) {
                return None;
            }
            Some((top, c))
        }
        ExprNode::Mul(children) => {
            let mut d = Q::zero();
            let mut cs: Vec<ExprId> = Vec::with_capacity(children.len());
            for c in children.iter() {
                let (dc, cc) = lead.get(c).cloned()?;
                d += dc;
                cs.push(cc);
            }
            let c = arena.mul(&cs);
            Some((d, safe_eval(arena, c)))
        }
        ExprNode::Neg(inner) => {
            let (d, c) = lead.get(inner).cloned()?;
            let n = arena.neg(c);
            Some((d, safe_eval(arena, n)))
        }
        ExprNode::Pow(b, e) => {
            if walk::contains(arena, *e, u) {
                return None;
            }
            let r = arena.as_num(*e)?.clone();
            let (db, cb) = lead.get(b).cloned()?;
            if !r.is_integer() && sign_of(arena, cb) != Some(Ordering::Greater) {
                return None;
            }
            let p = arena.pow(cb, *e);
            Some((db * r, safe_eval(arena, p)))
        }
        _ => None,
    }
}

/// The limit at `+∞` of a leading term `c·u^d`.
fn leading_term_limit(arena: &mut Arena, d: &Q, c: ExprId) -> LimVal {
    if d.is_negative() {
        return LimVal::Value(Extended::Finite(arena.zero()));
    }
    if d.is_zero() {
        return LimVal::Value(Extended::Finite(c));
    }
    match sign_of(arena, c) {
        Some(Ordering::Greater) => LimVal::Value(Extended::PosInf),
        Some(Ordering::Less) => LimVal::Value(Extended::NegInf),
        _ => LimVal::Unknown,
    }
}

/// Does the node `node` (its children already visited) tend to 0 as `u → +∞`
/// faster than any power of `u` grows: `e^g` for an algebraic `g → −∞` (so
/// `|e^g| ≤ e^{−c·u^ε}`), a product of such a factor with factors that grow at
/// most like a power of `u` ([`at_most_polynomial`]), a sum of such terms?
/// Such a product tends to 0.  The compositional rules saw `∞·0` in
/// `x²·e^{−(x−μ)²/(2σ²)}` and in `x·(e^{−x}·sin x + e^{−x}·cos x)`, and the limit
/// engine is not consulted with parameters or oscillation:
/// `∫₀^∞ x³·e^{−x}·sin x dx` and the moments of a normal density with symbolic
/// `μ`, `σ` were refused up to 0.46 (SymPy's test_meijerint).
fn exponentially_small(
    arena: &Arena,
    node: &ExprNode,
    vals: &FxHashMap<ExprId, LimVal>,
    small: &FxHashSet<ExprId>,
    u: ExprId,
) -> bool {
    match node {
        ExprNode::Exp(g) => {
            matches!(vals.get(g), Some(LimVal::Value(Extended::NegInf)))
                && algebraic_in(arena, *g, u)
        }
        ExprNode::Mul(children) => {
            children.iter().any(|c| small.contains(c))
                && children.iter().all(|&c| {
                    small.contains(&c)
                        || !walk::contains(arena, c, u)
                        || at_most_polynomial(arena, c, u)
                })
        }
        ExprNode::Add(children) => children.iter().all(|c| small.contains(c)),
        ExprNode::Neg(c) => small.contains(c),
        _ => false,
    }
}

/// Is every node of `e` that depends on `u` a sum, product or constant
/// power of `u`?
fn algebraic_in(arena: &Arena, e: ExprId, u: ExprId) -> bool {
    walk::post_order_ids(arena, e).into_iter().all(|id| {
        !walk::contains(arena, id, u)
            || id == u
            || match arena.node(id) {
                ExprNode::Add(_) | ExprNode::Mul(_) | ExprNode::Neg(_) => true,
                ExprNode::Pow(_, ex) => !walk::contains(arena, *ex, u),
                _ => false,
            }
    })
}

/// Is `|e|` bounded by a power of `u` as `u → +∞`: built from `u`, sums,
/// products, constant powers of algebraic bases, logarithms, and the
/// bounded functions `sin`, `cos`, `atan`, `tanh`, `erf`, `erfc`, `sign`,
/// `Si` (of anything), with no exponential?
fn at_most_polynomial(arena: &Arena, e: ExprId, u: ExprId) -> bool {
    walk::post_order_ids(arena, e).into_iter().all(|id| {
        if !walk::contains(arena, id, u) || id == u {
            return true;
        }
        match arena.node(id) {
            ExprNode::Add(_)
            | ExprNode::Mul(_)
            | ExprNode::Neg(_)
            | ExprNode::Sin(_)
            | ExprNode::Cos(_)
            | ExprNode::Atan(_)
            | ExprNode::Tanh(_)
            | ExprNode::Erf(_)
            | ExprNode::Erfc(_)
            | ExprNode::Sign(_)
            | ExprNode::Si(_)
            | ExprNode::Abs(_) => true,
            ExprNode::Ln(g) => algebraic_in(arena, *g, u),
            ExprNode::Pow(b, ex) => {
                !walk::contains(arena, *ex, u)
                    && (algebraic_in(arena, *b, u)
                        || arena.as_num(*ex).is_some_and(|r| !r.is_negative()))
            }
            _ => false,
        }
    })
}

/// The terms `cⱼ·ln(aⱼ·u + bⱼ)` among `children` (constants `cⱼ`, numeric
/// `aⱼ ≠ 0`, constant `bⱼ`), each of modulus tending to `∞`, and the limit
/// of their sum as `u → +∞` when the coefficients `cⱼ` add up to 0:
/// `ln(a·u + b) = ln a + ln u + o(1)` (principal branches, `u > 0`) off the
/// negative axis, and `ln|a| + ln u ± iπ` along it, above or below by the
/// sign of `im b` (`+iπ` on the cut itself), so the `ln u` cancel and the
/// limit is `Σ cⱼ·ln aⱼ` (with those `±iπ`).  The continuous antiderivative
/// of `∫ e^{−x}·sin x/(x² + 1) dx` holds `ln(x + i) − ln((−1 + i)·x − 1 − i)`,
/// each of whose limits is a complex infinity (0.46 refused
/// `∫₀^∞ e^{−x}·sin x/(x² + 1) dx`).  Empty and `None` otherwise.
fn linear_log_group(
    arena: &mut Arena,
    children: &[ExprId],
    u: ExprId,
) -> (Vec<ExprId>, Option<ExprId>) {
    let none = (Vec::new(), None);
    let mut members: Vec<ExprId> = Vec::new();
    let mut coeffs: Vec<ExprId> = Vec::new();
    let mut values: Vec<ExprId> = Vec::new();
    for &c in children {
        let (coeff, log) = match arena.node(c).clone() {
            ExprNode::Ln(_) => (arena.one(), c),
            ExprNode::Mul(fs) => {
                let (logs, rest): (Vec<ExprId>, Vec<ExprId>) = fs
                    .iter()
                    .partition(|&&f| matches!(arena.node(f), ExprNode::Ln(_)));
                if logs.len() != 1 || rest.iter().any(|&f| walk::contains(arena, f, u)) {
                    continue;
                }
                (arena.mul(&rest), logs[0])
            }
            _ => continue,
        };
        let ExprNode::Ln(arg) = *arena.node(log) else {
            continue;
        };
        // `ln|a·u + b|` (real `a`, `b`): `ln|a| + ln u + o(1)`.
        let (arg, absolute) = match *arena.node(arg) {
            ExprNode::Abs(inner) => (inner, true),
            _ => (arg, false),
        };
        let Some(LinearCoeffs { slope, intercept }) =
            calculus_util::linear_coeffs_exact(arena, arg, u)
        else {
            continue;
        };
        if absolute {
            if walk::contains(arena, slope, u)
                || walk::contains(arena, intercept, u)
                || !walk::free_symbols(arena, slope).is_empty()
                || !is_real(arena, slope)
                || !is_real(arena, intercept)
            {
                continue;
            }
            let abs_slope = arena.abs(slope);
            let value = arena.ln(abs_slope);
            members.push(c);
            coeffs.push(coeff);
            values.push(arena.mul(&[coeff, value]));
            continue;
        }
        if walk::contains(arena, slope, u)
            || walk::contains(arena, intercept, u)
            || !walk::free_symbols(arena, slope).is_empty()
        {
            continue;
        }
        let Ok(a) = evalf::evalf_complex64(arena, slope) else {
            continue;
        };
        if a.norm().is_nan() || a.norm() <= 0.0 {
            continue;
        }
        // Along the negative axis (`im a` proved 0) or off it, decided by
        // certified signs (see [`asymptotic_log_shift`]).
        let slope_side = crate::transforms::realness::imaginary_part_sign(arena, slope);
        if slope_side == SignSet::REAL {
            continue;
        }
        let value = if slope_side == SignSet::ZERO && a.re < 0.0 {
            // Along the negative axis: ln|a| + ln u ± iπ by the side of b.
            if !walk::free_symbols(arena, intercept).is_empty() {
                return none;
            }
            let side = crate::transforms::realness::imaginary_part_sign(arena, intercept);
            if side == SignSet::REAL {
                return none;
            }
            let below = side == SignSet::NEGATIVE;
            let abs_a = arena.neg(slope);
            let ln_abs = arena.ln(abs_a);
            let pi = arena.pi();
            let i = arena.i_unit();
            let sg = arena.int(if below { -1 } else { 1 });
            let i_pi = arena.mul(&[sg, i, pi]);
            arena.add(&[ln_abs, i_pi])
        } else {
            arena.ln(slope)
        };
        members.push(c);
        coeffs.push(coeff);
        values.push(arena.mul(&[coeff, value]));
    }
    if members.len() < 2 {
        return none;
    }
    let total = arena.add(&coeffs);
    let total = safe_eval(arena, total);
    if !arena.is_zero_structural(total) {
        return none;
    }
    let sum = arena.add(&values);
    (members, Some(safe_eval(arena, sum)))
}

/// The exponential and trigonometric integrals that
/// [`compositional_limit`] knows at infinity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SineIntegral {
    Si,
    Ci,
    Ei,
}

/// `Si(v)`, `Ci(v)`, `Ei(v)` as the limit of `f(w)` for `w → v`: `Si` is
/// entire; `Ci` and `Ei` are continuous off their cut `(−∞, 0]`, so `v`
/// must be provably positive or provably non-real (a parameter-free value
/// whose imaginary part is not negligible).
fn integral_function_at(arena: &mut Arena, kind: SineIntegral, v: ExprId) -> LimVal {
    // An imaginary part above `10⁻⁹` of a 16-digit value is certified;
    // a smaller one refuses.
    let off_cut = kind == SineIntegral::Si
        || sign_of(arena, v) == Some(Ordering::Greater)
        || (walk::free_symbols(arena, v).is_empty()
            && evalf::evalf_complex64(arena, v)
                .is_ok_and(|z| z.im.is_finite() && z.im.abs() > 1e-9 * z.norm().max(1.0)));
    if !off_cut {
        return LimVal::Unknown;
    }
    let r = match kind {
        SineIntegral::Si => arena.si(v),
        SineIntegral::Ci => arena.ci(v),
        SineIntegral::Ei => arena.ei(v),
    };
    LimVal::Value(Extended::Finite(safe_eval(arena, r)))
}

/// The sign of `im c` for a constant `c`: `0` when `c` is provably real,
/// otherwise from the assumptions on the imaginary part of the real/imaginary
/// split, or for a parameter-free `c` by certified evaluation
/// ([`imaginary_part_sign`](crate::transforms::realness::imaginary_part_sign)).
///
/// A single atom when decided, otherwise [`SignSet::REAL`] (an imaginary
/// part is real; its sign is unknown).
fn imaginary_sign(arena: &mut Arena, c: ExprId) -> SignSet {
    if is_real(arena, c) {
        return SignSet::ZERO;
    }
    if walk::free_symbols(arena, c).is_empty() {
        return crate::transforms::realness::imaginary_part_sign(arena, c);
    }
    let parts = crate::base::complex::decompose(arena, c);
    if !parts.exact {
        return SignSet::REAL;
    }
    let im = safe_eval(arena, parts.im);
    sign_of(arena, im).map_or(SignSet::REAL, SignSet::from_ordering)
}

/// `lim_{u → ∞} f(αu + β)` for `f = Si, Ci, Ei`, a real slope `α ≠ 0` of
/// known sign and a constant `β` (the argument runs to `±∞` along a
/// horizontal line `im w = im β`): `Si → ±π/2`; `Ci → 0` at `+∞` and `±iπ`
/// at `−∞` above/below the axis (`+iπ` on the cut itself, the principal
/// value `Ci(−x) = Ci(x) + iπ`); `Ei → ±iπ` at `−∞` above/below the axis and
/// `0` on it, `+∞` at `+∞` on the axis (a complex infinity off it:
/// unknown).  Before 0.46 these were unknown here and in the limit engine,
/// so `∫₀^∞ sin x/(x² + 1) dx` — an antiderivative with `Si(x ± i)`,
/// `Ci(x ± i)` — was refused.  mpmath: `ci(-200+0.1j)` = `−0.0044 +
/// 3.14135i`, `ei(-200-3j)` = `−iπ + O(10⁻⁸⁹)`.
fn integral_function_at_infinity(
    arena: &mut Arena,
    kind: SineIntegral,
    w: ExprId,
    u: ExprId,
) -> LimVal {
    let Some(LinearCoeffs { slope, intercept }) = calculus_util::linear_coeffs_exact(arena, w, u)
    else {
        return LimVal::Unknown;
    };
    if walk::contains(arena, slope, u) || walk::contains(arena, intercept, u) {
        return LimVal::Unknown;
    }
    if !is_real(arena, slope) {
        // `Ei(w)` along a ray with `Re w` not growing and `Im w → ±∞`:
        // `Ei(w) = −E₁(−w) ± iπ` there and `E₁(−w) → 0` (as for
        // `e^{−x}·sin x` over rational functions, `Ei((−1 ± i)x + β)`).
        if kind != SineIntegral::Ei {
            return LimVal::Unknown;
        }
        let parts = crate::base::complex::decompose(arena, slope);
        if !parts.exact {
            return LimVal::Unknown;
        }
        let re = safe_eval(arena, parts.re);
        let im = safe_eval(arena, parts.im);
        let re_sign = sign_of(arena, re);
        let im_sign = sign_of(arena, im);
        return match (re_sign, im_sign) {
            (
                Some(Ordering::Less | Ordering::Equal),
                Some(s @ (Ordering::Greater | Ordering::Less)),
            ) => {
                let pi = arena.pi();
                let i = arena.i_unit();
                let sg = arena.int(if s == Ordering::Greater { 1 } else { -1 });
                let v = arena.mul(&[sg, i, pi]);
                LimVal::Value(Extended::Finite(safe_eval(arena, v)))
            }
            _ => LimVal::Unknown,
        };
    }
    let up = match sign_of(arena, slope) {
        Some(Ordering::Greater) => true,
        Some(Ordering::Less) => false,
        _ => return LimVal::Unknown,
    };
    let pi = arena.pi();
    let i = arena.i_unit();
    let i_pi = arena.mul(&[i, pi]);
    let finite =
        |arena: &mut Arena, v: ExprId| LimVal::Value(Extended::Finite(safe_eval(arena, v)));
    match (kind, up) {
        (SineIntegral::Si, _) => {
            let half = arena.rational(if up { 1 } else { -1 }, 2);
            let v = arena.mul(&[half, pi]);
            finite(arena, v)
        }
        (SineIntegral::Ci, true) => LimVal::Value(Extended::Finite(arena.zero())),
        (SineIntegral::Ci, false) => match imaginary_sign(arena, intercept) {
            SignSet::NEGATIVE => {
                let v = arena.neg(i_pi);
                finite(arena, v)
            }
            SignSet::ZERO | SignSet::POSITIVE => finite(arena, i_pi),
            _ => LimVal::Unknown,
        },
        (SineIntegral::Ei, false) => match imaginary_sign(arena, intercept) {
            SignSet::NEGATIVE => {
                let v = arena.neg(i_pi);
                finite(arena, v)
            }
            SignSet::POSITIVE => finite(arena, i_pi),
            SignSet::ZERO => LimVal::Value(Extended::Finite(arena.zero())),
            _ => LimVal::Unknown,
        },
        (SineIntegral::Ei, true) => match imaginary_sign(arena, intercept) {
            SignSet::ZERO => LimVal::Value(Extended::PosInf),
            _ => LimVal::Unknown,
        },
    }
}

/// Limit of `base^exp` from the limits of its parts.
fn pow_limit(
    arena: &mut Arena,
    base: ExprId,
    exp: ExprId,
    bv: LimVal,
    ev: LimVal,
    u: ExprId,
) -> LimVal {
    let exp_has_u = walk::contains(arena, exp, u);
    let base_has_u = walk::contains(arena, base, u);
    if !exp_has_u {
        // Constant exponent.
        let Some(e_sign) = sign_of(arena, exp) else {
            return LimVal::Unknown;
        };
        let e_int_parity: Option<bool> = arena.as_num(exp).and_then(|r| {
            if r.is_integer() {
                Some(r.to_integer().to_i64().unwrap_or(1) % 2 == 0)
            } else {
                None
            }
        });
        return match bv {
            LimVal::Value(Extended::Finite(b)) => {
                let b_zero = sign_of(arena, b) == Some(Ordering::Equal);
                if b_zero {
                    // 0^e: 0 for e > 0, undefined/infinite otherwise.
                    return if e_sign == Ordering::Greater {
                        LimVal::Value(Extended::Finite(arena.zero()))
                    } else {
                        LimVal::Unknown
                    };
                }
                let p = arena.pow(b, exp);
                let p = safe_eval(arena, p);
                if is_finite_value(arena, p, u) {
                    LimVal::Value(Extended::Finite(p))
                } else {
                    LimVal::Unknown
                }
            }
            LimVal::Value(Extended::PosInf) => match e_sign {
                Ordering::Greater => LimVal::Value(Extended::PosInf),
                Ordering::Less => LimVal::Value(Extended::Finite(arena.zero())),
                Ordering::Equal => LimVal::Value(Extended::Finite(arena.one())),
            },
            LimVal::Value(Extended::NegInf) => match e_sign {
                Ordering::Greater => match e_int_parity {
                    Some(true) => LimVal::Value(Extended::PosInf),
                    Some(false) => LimVal::Value(Extended::NegInf),
                    None => LimVal::Unknown,
                },
                Ordering::Less => LimVal::Value(Extended::Finite(arena.zero())),
                Ordering::Equal => LimVal::Value(Extended::Finite(arena.one())),
            },
            LimVal::Bounded => {
                if e_sign == Ordering::Greater {
                    LimVal::Bounded
                } else {
                    LimVal::Unknown
                }
            }
            LimVal::Unknown => LimVal::Unknown,
        };
    }
    if !base_has_u {
        // Constant base c > 0: c^{g(u)}.
        let one = arena.one();
        let c_minus_1 = arena.sub(base, one);
        let c_minus_1 = safe_eval(arena, c_minus_1);
        let Some(cmp1) = sign_of(arena, c_minus_1) else {
            return LimVal::Unknown;
        };
        if !is_positive(arena, base) {
            return LimVal::Unknown;
        }
        return match (ev, cmp1) {
            (LimVal::Value(Extended::Finite(e)), _) => {
                let p = arena.pow(base, e);
                LimVal::Value(Extended::Finite(safe_eval(arena, p)))
            }
            (_, Ordering::Equal) => LimVal::Value(Extended::Finite(arena.one())),
            (LimVal::Value(Extended::PosInf), Ordering::Greater)
            | (LimVal::Value(Extended::NegInf), Ordering::Less) => LimVal::Value(Extended::PosInf),
            (LimVal::Value(Extended::PosInf), Ordering::Less)
            | (LimVal::Value(Extended::NegInf), Ordering::Greater) => {
                LimVal::Value(Extended::Finite(arena.zero()))
            }
            _ => LimVal::Unknown,
        };
    }
    LimVal::Unknown
}
