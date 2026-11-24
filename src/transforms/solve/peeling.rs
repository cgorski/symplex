//! Transcendental solving via inversion peeling (part of [`solve`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Transcendental solving via inversion peeling
// ═══════════════════════════════════════════════════════════════════════════

/// `2π·i·n` for the integer parameter `n`: the period of `exp`.
pub(super) fn two_pi_i_n(arena: &mut Arena, n: ExprId) -> ExprId {
    let two = arena.int(2);
    let pi = arena.pi;
    let i = arena.i_unit;
    arena.mul(&[two, pi, i, n])
}

/// The outcome of a principal inversion of `e^u = c` (or `a^f = c`, `u =
/// f·ln a`) at an exponent `u` not known to be real for a real unknown, in
/// the crate's root searches ([`SolveCx::lists_every_real_root`]): the
/// principal branch of the logarithm is one of those that real values of
/// the unknown reach (`(−1)^x = 1` at every even `x`), so its roots are
/// only some of the roots — [`Partial`](Solutions::Partial), and no root on
/// it is no proof.  Up to 0.57 it was complete: the real zeros of `(−1)^x −
/// 1` were `[0]`, and `∫₁³ dx/((−1)^x − 1)` (a pole at 2) was `−2`,
/// `∫₂⁶ dx/(i^x − 1)` (a pole at 4) `−4`.  Other solves keep the principal
/// roots, which `Ex::solve` documents.
fn principal_branch_only(principal: Solutions, cx: &SolveCx) -> Solutions {
    if !cx.lists_every_real_root() {
        return principal;
    }
    principal.union(Solutions::refused(Reason::said(
        "the other branches of the logarithm (an exponent not real for a real unknown) may hold \
         real roots",
    )))
}

/// `e` (evaluated) with every `ln q`, `q` a positive rational, written
/// `k·ln r − k′·ln r′` for `q = r^k/r′^{k′}` with `r`, `r′` not perfect powers
/// ([`perfect_power_base`]), and logarithms expanded (`ln √2 = ln(2)/2`):
/// sums and quotients of such logarithms then cancel exactly (`ln 4 − ln 2 =
/// ln 2`).
fn logs_of_rationals_reduced(arena: &mut Arena, e: ExprId) -> ExprId {
    let e = arena.expand_log_expr(e);
    let mut cache: rustc_hash::FxHashMap<ExprId, ExprId> = rustc_hash::FxHashMap::default();
    for id in crate::base::walk::post_order_ids(arena, e) {
        let reduced = match *arena.node(id) {
            ExprNode::Ln(a) => arena
                .as_num(a)
                .filter(|q| q.is_positive() && !q.is_one())
                .cloned()
                .and_then(|q| {
                    let (n, d) = (
                        Q::from_integer(q.numer().clone()),
                        Q::from_integer(q.denom().clone()),
                    );
                    let mut terms = Vec::with_capacity(2);
                    for (part, sign) in [(n, 1i64), (d, -1i64)] {
                        if part.is_one() {
                            continue;
                        }
                        let (r, k) = perfect_power_base(&part)?;
                        let r = arena.num_ratio(Q::from_integer(r));
                        let ln_r = arena.ln(r);
                        let k = arena.int(sign * k);
                        terms.push(arena.mul(&[k, ln_r]));
                    }
                    Some(arena.add(&terms))
                }),
            _ => None,
        };
        let out = match reduced {
            Some(r) => r,
            None => crate::base::walk::rebuild_with_cache(arena, id, &cache),
        };
        cache.insert(id, out);
    }
    let out = cache.get(&e).copied().unwrap_or(e);
    crate::transforms::eval::eval(arena, out)
}

/// The real roots of `e^u = c` on every branch of the logarithm (`u = ln c
/// + 2πik`), for an unknown `x` real in `cx` (the crate's root searches, a
/// symbol declared real), `u = α·x + β` with symbol-free `α`, `β`, and a
/// symbol-free `c ≠ 0`.  For a real `x`, `|e^u| = e^{re α·x + re β}` must be
/// `|c|` and `im α·x + im β − arg c` a multiple of `2π`:
///
/// - `re α ≠ 0`: the one candidate `x₀ = (ln|c| − re β)/re α`, a root exactly
///   when `k = (im α·x₀ + im β − arg c)/2π` is an integer (its certified
///   integer part and the sign of the rest): `(−2)^x = 4` has the one real
///   root 2, on the branch `k = 1` (up to 0.57 `Ex::solve` for a real `x`
///   gave `[]`, the principal root `ln 4/(ln 2 + iπ)` not being real), and
///   `(−2)^x = 3` none;
/// - `re α = 0` (a real period `2π/im α`): no root when `ln|c| ≠ re β`
///   (`(−1)^x = 2`), infinitely many otherwise, which are not listed
///   (`None`).
///
/// `None` also when a sign or an integer part does not settle.
pub(super) fn real_roots_on_every_branch(
    arena: &mut Arena,
    u: ExprId,
    c: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Option<Solutions> {
    use crate::poly::algebraic::sign_checked;
    use crate::transforms::eval::eval;
    if cx.unknown_may_be_nonreal(arena, var) {
        return None;
    }
    let constant = |arena: &Arena, e: ExprId| {
        crate::base::walk::free_symbols(arena, e).is_empty()
            && !value_class::holds_special_atom(arena, e)
    };
    if !constant(arena, c) || arena.is_zero_structural(c) {
        return None;
    }
    let (alpha, beta) = linear_rate(arena, u, var)?;
    if !constant(arena, alpha) || !constant(arena, beta) {
        return None;
    }
    let part = |arena: &mut Arena, e: ExprId, imaginary: bool| {
        let p = if imaginary { arena.im(e) } else { arena.re(e) };
        eval(arena, p)
    };
    let (re_a, im_a) = (part(arena, alpha, false), part(arena, alpha, true));
    let (re_b, im_b) = (part(arena, beta, false), part(arena, beta, true));
    // (logarithms of rationals as multiples of the logarithms of their
    // bases, so that `(ln 4 − ln 2)/ln √2` is 2)
    let re_a = logs_of_rationals_reduced(arena, re_a);
    let abs_c = arena.abs(c);
    let ln_abs_c = arena.ln(abs_c);
    let modulus = arena.sub(ln_abs_c, re_b);
    let modulus = eval(arena, modulus);
    let modulus = logs_of_rationals_reduced(arena, modulus);
    if sign_checked(arena, re_a)? == 0 {
        return match sign_checked(arena, modulus)? {
            0 => None,
            _ => Some(Solutions::none(Reason::NoPreimage)),
        };
    }
    let x0 = arena.div(modulus, re_a);
    let x0 = eval(arena, x0);
    let arg_c = arena.arg(c);
    let neg_arg_c = arena.neg(arg_c);
    let phase = arena.mul(&[im_a, x0]);
    let theta = arena.add(&[phase, im_b, neg_arg_c]);
    let two = arena.int(2);
    let pi = arena.pi;
    let two_pi = arena.mul(&[two, pi]);
    let k = arena.div(theta, two_pi);
    let k = eval(arena, k);
    let n = arena.floor(k);
    let n = eval(arena, n);
    arena.as_num(n).filter(|q| q.is_integer())?;
    let frac = arena.sub(k, n);
    let frac = eval(arena, frac);
    match sign_checked(arena, frac)? {
        0 => Some(Solutions::found(vec![x0])),
        _ => Some(Solutions::none(Reason::NoPreimage)),
    }
}

/// Which members of a family `offset + step·k`, `k ∈ ℤ`, are real
/// ([`real_members_of_family`]).
pub(crate) enum RealMembers {
    /// The step is proved real: every member is real when the offset is, and
    /// none otherwise (the caller decides the offset).
    RealStep,
    /// The step is proved non-real and no member is real.
    None,
    /// The step is proved non-real and exactly one member is real: its value.
    One(ExprId),
    /// Not decided.
    Undecided,
}

/// The real members of the family `offset + step·k`, `k ∈ ℤ`, for a
/// symbol-free `step`.  A family with a non-real step crosses the real axis
/// at most once, at `k₀ = −im(offset)/im(step)`: it has a real member exactly
/// when `k₀` is an integer, which is decided exactly — a rational `k₀`, or
/// the sign of `k₀ − m` for the integer `m` its certified value is next to
/// ([`sign_checked`](crate::poly::algebraic::sign_checked): a proved 0 or a
/// certified sign, never a value 0 to the digits computed).  The member is
/// then `re(offset) + k₀·re(step)`.
///
/// The general solution of `(−2)^x = 4` is `x = (ln 4 + 2πik)/(ln 2 + iπ)`,
/// real at `k = 1` only (`x = 2`).  Up to 0.58 the crate's searches in a
/// window dropped such a family as soon as its offset (`k = 0`) was not
/// real: `Ex::singularities(1/((−2)^x − 4), x, [0, 10])` was `∅`.
pub(crate) fn real_members_of_family(
    arena: &mut Arena,
    offset: ExprId,
    step: ExprId,
) -> RealMembers {
    use crate::poly::algebraic::sign_checked;
    use crate::transforms::eval::eval;
    let constant = |arena: &Arena, e: ExprId| {
        crate::base::walk::free_symbols(arena, e).is_empty()
            && !value_class::holds_special_atom(arena, e)
    };
    if !constant(arena, step) {
        return RealMembers::Undecided;
    }
    // (logarithms of rationals over the logarithms of their bases, so that
    // `ln 4/(2 ln 2)` is 1)
    let part = |arena: &mut Arena, e: ExprId, imaginary: bool| {
        let p = if imaginary { arena.im(e) } else { arena.re(e) };
        let p = eval(arena, p);
        logs_of_rationals_reduced(arena, p)
    };
    let im_step = part(arena, step, true);
    match sign_checked(arena, im_step) {
        Some(0) => return RealMembers::RealStep,
        Some(_) => {}
        None => return RealMembers::Undecided,
    }
    if !constant(arena, offset) {
        return RealMembers::Undecided;
    }
    let im_offset = part(arena, offset, true);
    let neg = arena.neg(im_offset);
    let k = arena.div(neg, im_step);
    let k = logs_of_rationals_reduced(arena, k);
    let index = match arena.as_num(k) {
        Some(q) if q.is_integer() => q.to_integer(),
        Some(_) => return RealMembers::None,
        None => {
            // The one integer `k₀` can be: the certified 20-digit value
            // places it within 10⁻¹⁰ (below 2⁴⁰), and every other integer is
            // then at least 1/2 away.
            let Some(m) = crate::transforms::evalf::const_value(arena, k, 20)
                .ok()
                .and_then(|v| v.to_f64())
                .filter(|v| v.abs() < 1_099_511_627_776.0)
                .and_then(|v| crate::base::cast::f64_to_i64(v.round()))
            else {
                return RealMembers::Undecided;
            };
            let m_id = arena.int(m);
            let rest = arena.sub(k, m_id);
            let rest = eval(arena, rest);
            match sign_checked(arena, rest) {
                Some(0) => BigInt::from(m),
                Some(_) => return RealMembers::None,
                None => return RealMembers::Undecided,
            }
        }
    };
    let index = arena.num_ratio(Q::from_integer(index));
    let (re_offset, re_step) = (part(arena, offset, false), part(arena, step, false));
    let shift = arena.mul(&[index, re_step]);
    let value = arena.add(&[re_offset, shift]);
    let value = eval(arena, value);
    RealMembers::One(crate::simplify::ratsimp::ratsimp(arena, value))
}

/// Try to solve `expr = 0` by algebraic inversion.
/// Restructures as `f(x) = c` and inverts `f`.
///
/// No root ([`Reason::NoPreimage`]) when a range restriction proves there
/// is no real solution (e.g. `exp(x) = -1`), refused when the structure is
/// not invertible.
pub(super) fn try_solve_by_inversion(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Solutions {
    if !expr_contains_var(arena, expr, var) {
        return Solutions::refused(Reason::NotFound);
    }
    let zero = arena.zero;
    solve_by_peeling(arena, expr, zero, var, cx)
}

/// Peel layers off `lhs = rhs` to isolate `var`.
///
/// `rhs` never contains `var`.  In a general solve ([`SolveCx::period`])
/// periodic inversions (sin, cos, tan) emit full solution families in terms
/// of the integer parameter `n`; otherwise only principal branches are
/// produced.  A range restriction is used when [`SolveCx::range_restricted`]
/// allows it, and gives no root ([`Reason::NoPreimage`]); a structure that
/// is not inverted is refused, and a periodic layer whose right-hand side
/// is already a family ([`period_in_use`]) refuses with
/// [`Reason::NestedPeriods`] unless it handles it.
pub(super) fn solve_by_peeling(
    arena: &mut Arena,
    lhs: ExprId,
    rhs: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Solutions {
    let not_inverted = || Solutions::refused(Reason::NotFound);
    let no_preimage = || Solutions::none(Reason::NoPreimage);
    let period = cx.period();
    // Base case: lhs IS the variable
    if lhs == var {
        return Solutions::found(vec![rhs]);
    }

    let node = arena.node(lhs).clone();
    match node {
        // f(x) + c = rhs → f(x) = rhs - c ; several var-terms → polynomial fallback
        ExprNode::Add(ref children) => {
            let mut dep = Vec::new();
            let mut indep = Vec::new();
            for &child in children {
                if expr_contains_var(arena, child, var) {
                    dep.push(child);
                } else {
                    indep.push(child);
                }
            }
            if dep.len() == 1 {
                let new_rhs = if indep.is_empty() {
                    rhs
                } else {
                    let sum_indep = arena.add(&indep);
                    arena.sub(rhs, sum_indep)
                };
                return solve_by_peeling(arena, dep[0], new_rhs, var, cx);
            }
            // Multiple var-dependent terms: lhs - rhs = 0 may be polynomial.
            let diff = arena.sub(lhs, rhs);
            let diff = crate::transforms::eval::eval(arena, diff);
            if let Some(poly) = polybridge::expr_to_poly(arena, diff, var) {
                if poly.is_zero() || poly.is_constant() {
                    return not_inverted();
                }
                return Solutions::exactly(
                    values(solve_rational_poly(arena, var, &poly)),
                    Reason::NoPreimage,
                );
            }
            if let Some(sols) = try_solve_linear_symbolic(arena, diff, var) {
                return Solutions::exactly(values(sols), Reason::NoPreimage);
            }
            match try_solve_symbolic_poly(arena, diff, var) {
                Some(sols) => Solutions::exactly(values(sols), Reason::NoPreimage),
                None => not_inverted(),
            }
        }
        // c * f(x) = rhs → f(x) = rhs/c
        ExprNode::Mul(ref children) => {
            let mut dep = Vec::new();
            let mut indep = Vec::new();
            for &child in children {
                if expr_contains_var(arena, child, var) {
                    dep.push(child);
                } else {
                    indep.push(child);
                }
            }
            if dep.len() != 1 && is_rational_in(arena, lhs, var) {
                // A rational function (`(x − 3)/x = 0` under `√`, `(x + 1)(x − 1)
                // = 1` under `ln`; refused before 0.47), by the polynomial and
                // rational solvers directly (not `solve_raw`, which peels).
                let diff = arena.sub(lhs, rhs);
                let diff = crate::transforms::eval::eval(arena, diff);
                if let Some(poly) = polybridge::expr_to_poly(arena, diff, var) {
                    if poly.is_zero() || poly.is_constant() {
                        return not_inverted();
                    }
                    return Solutions::exactly(
                        values(solve_rational_poly(arena, var, &poly)),
                        Reason::NoPreimage,
                    );
                }
                if is_rational_function_of(arena, diff, var) {
                    let outcome = try_solve_rational(arena, diff, var, cx);
                    return if outcome.has_roots()
                        || outcome.is_none_proved()
                        || outcome.is_nested_periods()
                    {
                        outcome
                    } else {
                        not_inverted()
                    };
                }
                return match try_solve_linear_symbolic(arena, diff, var)
                    .filter(|s| !s.is_empty())
                    .or_else(|| try_solve_symbolic_poly(arena, diff, var))
                {
                    Some(sols) => Solutions::exactly(values(sols), Reason::NoPreimage),
                    None => not_inverted(),
                };
            }
            if dep.len() != 1 || indep.is_empty() {
                return not_inverted();
            }
            let coeff = arena.mul(&indep);
            let mut new_rhs = arena.div(rhs, coeff);
            // `i·x = 2πin + iπ` → `x = 2πn + π`, not `−(2πin + iπ)·i`.
            if crate::base::walk::contains(arena, coeff, arena.i_unit) {
                let e = crate::transforms::eval::eval(arena, new_rhs);
                let e = crate::transforms::expand::expand(arena, e);
                new_rhs = crate::transforms::eval::eval(arena, e);
            }
            solve_by_peeling(arena, dep[0], new_rhs, var, cx)
        }
        // exp(f(x)) = rhs → f(x) = ln(rhs) (+ 2πin for a non-real f)
        // Domain check: exp never vanishes; exp(f) > 0 for a real f, so
        // then rhs must be strictly positive.
        ExprNode::Exp(inner) => {
            if rhs == arena.zero {
                tracing::debug!("solve_by_peeling: exp domain error, rhs = 0");
                return no_preimage();
            }
            let real_inner = cx.real_for_real_var(arena, inner, var);
            if real_inner
                && let Some(c) = arena.as_num(rhs)
                && !c.is_positive()
                && cx.range_restricted(arena, inner, var)
            {
                tracing::debug!("solve_by_peeling: exp domain error, rhs <= 0");
                return no_preimage();
            }
            let ln_rhs = principal_ln(arena, rhs);
            // A non-real argument (`exp(i·x) = c`) reaches every branch of
            // the logarithm with real values of `var`: the general solution
            // carries the period `2πi`.
            if !real_inner && period_in_use(arena, rhs, cx) {
                return Solutions::refused(Reason::NestedPeriods);
            }
            if period.is_none() && !real_inner {
                if let Some(every) = real_roots_on_every_branch(arena, inner, rhs, var, cx) {
                    return every;
                }
                let principal = solve_by_peeling(arena, inner, ln_rhs, var, cx);
                return principal_branch_only(principal, cx);
            }
            let new_rhs = match period {
                Some(n) if !real_inner => {
                    let p = two_pi_i_n(arena, n);
                    arena.add(&[ln_rhs, p])
                }
                _ => ln_rhs,
            };
            solve_by_peeling(arena, inner, new_rhs, var, cx)
        }
        // ln(f(x)) = rhs → f(x) = exp(rhs)
        ExprNode::Ln(inner) => {
            let new_rhs = arena.exp(rhs);
            peel_principal_inverse(arena, PrincipalInverse::Ln, inner, rhs, new_rhs, var, cx)
        }
        // sin(f(x)) = rhs → f(x) ∈ {asin(rhs), π - asin(rhs)} (+ 2πn)
        ExprNode::Sin(inner) | ExprNode::Cos(inner) if period_in_use(arena, rhs, cx) => {
            // `sin f = cₙ` for a family `cₙ` (an outer periodic layer): over
            // the reals only the members with `|cₙ| ≤ 1` can be reached, each
            // giving families in a parameter of their own.  One refused
            // member refuses the whole.
            let Some(n) = period else {
                return not_inverted();
            };
            let members = if cx.real_for_real_var(arena, inner, var) {
                bounded_family_members(arena, rhs, n)
            } else {
                None
            };
            let Some(members) = members else {
                return Solutions::refused(Reason::NestedPeriods);
            };
            let mut out = no_preimage();
            for m in members {
                match solve_by_peeling(arena, lhs, m, var, cx) {
                    refused @ Solutions::Refused(_) => return refused,
                    member => out = out.union(member),
                }
            }
            out
        }
        ExprNode::Sin(inner) => {
            // Domain check: sin(x) = c has no real solutions when |c| > 1
            if let Some(c) = arena.as_num(rhs)
                && c.abs() > Ratio::one()
                && cx.range_restricted(arena, inner, var)
            {
                tracing::debug!("solve_by_peeling: sin domain error, |c| > 1");
                return no_preimage();
            }
            tracing::debug!("solve_by_peeling: inverting sin, two branches");
            let asin_rhs = arena.asin(rhs);
            let pi = arena.pi;
            let pi_minus_asin = arena.sub(pi, asin_rhs);
            let (b1, b2) = match period {
                Some(n) => {
                    let two = arena.int(2);
                    let two_pi_n = arena.mul(&[two, pi, n]);
                    (
                        arena.add(&[asin_rhs, two_pi_n]),
                        arena.add(&[pi_minus_asin, two_pi_n]),
                    )
                }
                None => (asin_rhs, pi_minus_asin),
            };
            peel_two_branches(arena, inner, b1, b2, var, cx)
        }
        // cos(f(x)) = rhs → f(x) ∈ {acos(rhs), -acos(rhs)} (+ 2πn)
        ExprNode::Cos(inner) => {
            // Domain check: cos(x) = c has no real solutions when |c| > 1
            if let Some(c) = arena.as_num(rhs)
                && c.abs() > Ratio::one()
                && cx.range_restricted(arena, inner, var)
            {
                tracing::debug!("solve_by_peeling: cos domain error, |c| > 1");
                return no_preimage();
            }
            tracing::debug!("solve_by_peeling: inverting cos, two branches");
            let acos_rhs = arena.acos(rhs);
            let neg_acos = arena.neg(acos_rhs);
            let (b1, b2) = match period {
                Some(n) => {
                    let two = arena.int(2);
                    let pi = arena.pi;
                    let two_pi_n = arena.mul(&[two, pi, n]);
                    (
                        arena.add(&[acos_rhs, two_pi_n]),
                        arena.add(&[neg_acos, two_pi_n]),
                    )
                }
                None => (acos_rhs, neg_acos),
            };
            peel_two_branches(arena, inner, b1, b2, var, cx)
        }
        // tan(f(x)) = rhs → f(x) = atan(rhs) (+ πn)
        ExprNode::Tan(inner) => {
            if period_in_use(arena, rhs, cx) {
                return Solutions::refused(Reason::NestedPeriods);
            }
            let atan_rhs = arena.atan(rhs);
            let new_rhs = match period {
                Some(n) => {
                    let pi = arena.pi;
                    let pi_n = arena.mul(&[pi, n]);
                    arena.add(&[atan_rhs, pi_n])
                }
                None => atan_rhs,
            };
            solve_by_peeling(arena, inner, new_rhs, var, cx)
        }
        // f(x)^n = rhs → f(x) = rhs^(1/n)
        // When n is a positive even integer, also consider f(x) = -(rhs^(1/n))
        //
        // a^f(x) = rhs → f(x) = ln(rhs) / ln(a)  (constant base, variable exponent)
        ExprNode::Pow(inner_base, inner_exp) => {
            if let Some(n) = arena.as_num(inner_exp) {
                let n = n.clone();
                if !n.is_zero() {
                    return match power_preimages(arena, rhs, &n, cx) {
                        None => not_inverted(),
                        Some(Preimages::Values(branches)) => {
                            peel_branches(arena, inner_base, &branches, var, cx)
                        }
                        Some(Preimages::Conditional { value, cond }) => {
                            peel_branches(arena, inner_base, &[value], var, cx).under(arena, cond)
                        }
                    };
                }
            }

            // f^c = 0 for an exponent free of `var`: `f = 0` exactly when
            // `re c > 0` (`0^c` is `0` there, undefined or `1` elsewhere, and
            // `f^c = e^(c·ln f) ≠ 0` for `f ≠ 0`).  Up to 0.52 the candidate
            // `0^(1/c)` was left unevaluated (`x^(1/erfi 3) = 0`: "cannot
            // decide whether 0^(1/erfi(3)) solves the equation").
            if arena.is_zero_structural(rhs)
                && expr_contains_var(arena, inner_base, var)
                && !expr_contains_var(arena, inner_exp, var)
            {
                return power_equals_zero(arena, inner_base, inner_exp, var, cx);
            }

            // a^f(x) = rhs where a is a constant (no var) and f(x) contains var.
            // Strategy: f(x) = ln(rhs) / ln(a).
            //
            // Integer shortcut: if a and rhs are positive integers and a^k == rhs
            // for some k, solve f(x) = k directly (gives exact answer like x = 3
            // instead of x = ln(8)/ln(2)).
            if !expr_contains_var(arena, inner_base, var)
                && expr_contains_var(arena, inner_exp, var)
            {
                tracing::debug!("solve_by_peeling: constant-base exponential a^f(x) = rhs");

                // a^f = exp(f·ln a) reaches every branch of the logarithm
                // with real values of `var` unless `f·ln a` is real (a > 0
                // and a real f): the general solution is then
                // f = (ln rhs + 2πin)/ln a.  Before 0.31 `(−1)^x = 1` gave
                // only `x = 0` from `solve_general` (the family is `2n`),
                // `(−1)^x = −1` only `1`.
                // A principal solve of `a^f = exp(f·ln a)` at a non-real
                // `f·ln a` (`(−1)^x = 1`): the principal branch of the
                // logarithm is one of those real values of `var` reach.
                let nonreal_exponent = if period.is_none() {
                    let ln_base = arena.ln(inner_base);
                    let ln_base = crate::transforms::eval::eval(arena, ln_base);
                    let prod = arena.mul(&[inner_exp, ln_base]);
                    (!cx.real_for_real_var(arena, prod, var)).then(|| {
                        let u = crate::transforms::expand::expand(arena, prod);
                        crate::transforms::eval::eval(arena, u)
                    })
                } else {
                    None
                };
                if let Some(u) = nonreal_exponent
                    && let Some(every) = real_roots_on_every_branch(arena, u, rhs, var, cx)
                {
                    return every;
                }
                let principal_only = |outcome: Solutions| match nonreal_exponent {
                    Some(_) => principal_branch_only(outcome, cx),
                    None => outcome,
                };
                if let Some(n) = period {
                    let ln_base = arena.ln(inner_base);
                    let ln_base = crate::transforms::eval::eval(arena, ln_base);
                    let prod = arena.mul(&[inner_exp, ln_base]);
                    if !cx.real_for_real_var(arena, prod, var) {
                        if period_in_use(arena, rhs, cx) {
                            return Solutions::refused(Reason::NestedPeriods);
                        }
                        let ln_rhs = principal_ln(arena, rhs);
                        let p = two_pi_i_n(arena, n);
                        let num = arena.add(&[ln_rhs, p]);
                        let new_rhs = arena.div(num, ln_base);
                        let new_rhs = crate::transforms::expand::expand(arena, new_rhs);
                        let new_rhs = crate::transforms::eval::eval(arena, new_rhs);
                        return solve_by_peeling(arena, inner_exp, new_rhs, var, cx);
                    }
                }

                // Integer shortcut: try to find k such that base^k == rhs
                if let (Some(b), Some(r)) = (
                    arena.as_num(inner_base).cloned(),
                    arena.as_num(rhs).cloned(),
                ) && b.is_integer()
                    && r.is_integer()
                    && b > Ratio::one()
                    && r.is_positive()
                {
                    let b_int = b.to_integer();
                    let r_int = r.to_integer();
                    // Try small powers: b^1, b^2, ... up to b^64
                    let mut power = BigInt::one();
                    for k in 0u32..65 {
                        if power == r_int {
                            let k_expr = arena.int(k as i64);
                            tracing::debug!(
                                "solve_by_peeling: integer log shortcut, base^{k} = rhs"
                            );
                            return principal_only(solve_by_peeling(
                                arena, inner_exp, k_expr, var, cx,
                            ));
                        }
                        if power > r_int {
                            break;
                        }
                        power *= &b_int;
                    }
                }

                // General case: f(x) = ln(rhs) / ln(base)
                let ln_base = arena.ln(inner_base);
                let ln_rhs = principal_ln(arena, rhs);
                let new_rhs = arena.div(ln_rhs, ln_base);
                return principal_only(solve_by_peeling(arena, inner_exp, new_rhs, var, cx));
            }

            // f^c = rhs for a constant irrational real c (`x^π = 2`): the
            // principal f = rhs^(1/c), the one real solution of a real f
            // (`f^c = exp(c·ln f)` is real only for f ≥ 0; refused before
            // 0.47, SymPy: `[2**(1/pi)]`).  Other branches for a complex f.
            if expr_contains_var(arena, inner_base, var)
                && !expr_contains_var(arena, inner_exp, var)
                && crate::base::walk::free_symbols(arena, inner_exp).is_empty()
                && arena.as_num(inner_exp).is_none()
                && crate::transforms::realness::constant_realness_as_declared(
                    arena,
                    inner_exp,
                    ROOT_CHECK_DIGITS,
                    &mut crate::base::assumptions::AssumptionCache::new(),
                ) == Some(true)
            {
                if period.is_some() && !cx.real_for_real_var(arena, inner_base, var) {
                    return not_inverted();
                }
                let one = arena.one;
                let inv = arena.div(one, inner_exp);
                let new_rhs = arena.pow(rhs, inv);
                let new_rhs = crate::transforms::eval::eval(arena, new_rhs);
                return solve_by_peeling(arena, inner_base, new_rhs, var, cx);
            }

            not_inverted()
        }
        // Neg(-f(x)) = rhs → f(x) = -rhs
        ExprNode::Neg(inner) => {
            let new_rhs = arena.neg(rhs);
            solve_by_peeling(arena, inner, new_rhs, var, cx)
        }
        // ── Inverse trig peeling ──────────────────────────────────
        // asin(f(x)) = rhs → f(x) = sin(rhs)
        ExprNode::Asin(inner) => {
            let new_rhs = arena.sin(rhs);
            peel_principal_inverse(arena, PrincipalInverse::Asin, inner, rhs, new_rhs, var, cx)
        }
        // acos(f(x)) = rhs → f(x) = cos(rhs)
        ExprNode::Acos(inner) => {
            let new_rhs = arena.cos(rhs);
            peel_principal_inverse(arena, PrincipalInverse::Acos, inner, rhs, new_rhs, var, cx)
        }
        // atan(f(x)) = rhs → f(x) = tan(rhs)
        ExprNode::Atan(inner) => {
            let new_rhs = arena.tan(rhs);
            peel_principal_inverse(arena, PrincipalInverse::Atan, inner, rhs, new_rhs, var, cx)
        }
        // ── Inverse hyperbolic peeling ────────────────────────────
        // sinh(f(x)) = rhs → f(x) = asinh(rhs); for a non-real f the
        // general solution is asinh(rhs) + 2πin, iπ − asinh(rhs) + 2πin.
        ExprNode::Sinh(inner) => {
            let asinh_rhs = arena.asinh(rhs);
            match period {
                Some(_)
                    if !cx.real_for_real_var(arena, inner, var)
                        && period_in_use(arena, rhs, cx) =>
                {
                    Solutions::refused(Reason::NestedPeriods)
                }
                Some(n) if !cx.real_for_real_var(arena, inner, var) => {
                    let p = two_pi_i_n(arena, n);
                    let i_pi = arena.mul(&[arena.i_unit, arena.pi]);
                    let other = arena.sub(i_pi, asinh_rhs);
                    let b1 = arena.add(&[asinh_rhs, p]);
                    let b2 = arena.add(&[other, p]);
                    peel_two_branches(arena, inner, b1, b2, var, cx)
                }
                Some(_) => solve_by_peeling(arena, inner, asinh_rhs, var, cx),
                None => {
                    let real_inner = cx.real_for_real_var(arena, inner, var);
                    let principal = solve_by_peeling(arena, inner, asinh_rhs, var, cx);
                    if real_inner {
                        principal
                    } else {
                        principal_branch_only(principal, cx)
                    }
                }
            }
        }
        // cosh(f(x)) = rhs → f(x) = ±acosh(rhs) (+ 2πin for a non-real f)
        // Domain check: cosh(f) >= 1 for a real f, so rhs must be >= 1.
        ExprNode::Cosh(inner) => {
            let real_inner = cx.real_for_real_var(arena, inner, var);
            if real_inner
                && let Some(c) = arena.as_num(rhs)
                && *c < Ratio::one()
                && cx.range_restricted(arena, inner, var)
            {
                tracing::debug!("solve_by_peeling: cosh domain error, rhs < 1");
                return no_preimage();
            }
            if !real_inner && period_in_use(arena, rhs, cx) {
                return Solutions::refused(Reason::NestedPeriods);
            }
            let acosh_rhs = arena.acosh(rhs);
            let neg_acosh = arena.neg(acosh_rhs);
            let (b1, b2) = match period {
                Some(n) if !real_inner => {
                    let p = two_pi_i_n(arena, n);
                    (arena.add(&[acosh_rhs, p]), arena.add(&[neg_acosh, p]))
                }
                _ => (acosh_rhs, neg_acosh),
            };
            let outcome = peel_two_branches(arena, inner, b1, b2, var, cx);
            // (`cosh(iπx) = 1` at every even `x`: up to 0.57 `∫₁³ dx/(cosh(iπx)
            // − 1)` was 0 across the double pole at 2)
            if period.is_none() && !real_inner {
                principal_branch_only(outcome, cx)
            } else {
                outcome
            }
        }
        // tanh(f(x)) = rhs → f(x) = atanh(rhs) (+ iπn for a non-real f)
        ExprNode::Tanh(inner) => {
            if !cx.real_for_real_var(arena, inner, var) && period_in_use(arena, rhs, cx) {
                return Solutions::refused(Reason::NestedPeriods);
            }
            let atanh_rhs = arena.atanh(rhs);
            let real_inner = cx.real_for_real_var(arena, inner, var);
            let new_rhs = match period {
                Some(n) if !real_inner => {
                    let i_pi_n = arena.mul(&[arena.i_unit, arena.pi, n]);
                    arena.add(&[atanh_rhs, i_pi_n])
                }
                _ => atanh_rhs,
            };
            let outcome = solve_by_peeling(arena, inner, new_rhs, var, cx);
            if period.is_none() && !real_inner {
                principal_branch_only(outcome, cx)
            } else {
                outcome
            }
        }
        // asinh(f) = rhs → f = sinh(rhs), acosh(f) = rhs → f = cosh(rhs),
        // atanh(f) = rhs → f = tanh(rhs): valid when rhs lies in the range
        // of the principal branch, which the final check of every candidate
        // decides (`acosh x = −1` has no solution: `cosh(−1)` gives
        // `acosh(cosh 1) = 1`).  Without these the kinks of `|asinh x|`
        // were invisible to the definite integrator.
        ExprNode::Asinh(inner) => {
            let new_rhs = arena.sinh(rhs);
            peel_principal_inverse(arena, PrincipalInverse::Asinh, inner, rhs, new_rhs, var, cx)
        }
        ExprNode::Acosh(inner) => {
            let new_rhs = arena.cosh(rhs);
            peel_principal_inverse(arena, PrincipalInverse::Acosh, inner, rhs, new_rhs, var, cx)
        }
        ExprNode::Atanh(inner) => {
            let new_rhs = arena.tanh(rhs);
            peel_principal_inverse(arena, PrincipalInverse::Atanh, inner, rhs, new_rhs, var, cx)
        }
        // ── Abs peeling ───────────────────────────────────────────
        // ── Abs peeling ───────────────────────────────────────────────────────────
        // erf(f) = rhs → f = erfinv(rhs), erfc(f) = rhs → f = erfcinv(rhs),
        // the principal solutions (erf is entire and takes each value
        // infinitely often).  A rhs outside the range (−1, 1) of erf on the
        // reals, (0, 2) of erfc — a real one beyond the ends, or a non-real
        // one — is reached by a complex f only, where the inverse has no
        // principal value: no solution when f is real in the domain of the
        // unknown, refused otherwise (a constant rhs not decided either way
        // too).  Up to 0.48 a plain `x` was taken real here (alone among the
        // range restrictions), and `Ex::solve` answered `erf(x) = 2` with "no
        // solution", while `erf(0.8694… + 1.3406…i) = 2` (mpmath
        // `findroot`); up to 0.51 a non-real rhs was inverted, and `erf(2x +
        // 1/3) = −2i` gave the one root `−erfinv(2i)/2 − 1/6` of infinitely
        // many.  Refused before 0.47 (SymPy: `solve(erf(x) - y)` →
        // `[erfinv(y)]`).
        ExprNode::Erf(inner) | ExprNode::Erfc(inner) => {
            use crate::base::truth::Truth;
            let is_erf = matches!(arena.node(lhs), ExprNode::Erf(_));
            let (lo, hi) = if is_erf {
                (-Q::one(), Q::one())
            } else {
                (Q::zero(), Q::from_integer(BigInt::from(2)))
            };
            // a constant `rhs` by its certified sign against the ends (`ln 3`:
            // up to 0.50 the real solve of `exp(erf x) = 3` gave `erfinv(ln 3)`,
            // which is not real); a value with parameters is inverted
            let in_range = match arena.as_num(rhs) {
                Some(c) => Truth::from(*c > lo && *c < hi),
                None if crate::base::walk::free_symbols(arena, rhs).is_empty() => {
                    let (lo_id, hi_id) = (arena.num_ratio(lo.clone()), arena.num_ratio(hi.clone()));
                    let above = arena.sub(rhs, hi_id);
                    let below = arena.sub(rhs, lo_id);
                    let above = crate::poly::algebraic::sign_checked(arena, above);
                    let below = crate::poly::algebraic::sign_checked(arena, below);
                    if above == Some(-1) && below == Some(1) {
                        Truth::True
                    } else if matches!(above, Some(0 | 1))
                        || matches!(below, Some(0 | -1))
                        || part_signs(arena, rhs).nonreal()
                    {
                        Truth::False
                    } else {
                        Truth::Unknown
                    }
                }
                None => Truth::True,
            };
            // `erf` and `erfc` are entire: never infinite (`π^(erfc x − 1) =
            // 0`); a `nan` (`0^(erfc f) = 0`) proves nothing
            if has_infinite_term(arena, rhs) && !matches!(arena.node(rhs), ExprNode::NaN) {
                return no_preimage();
            }
            let real_inner = cx.range_restricted(arena, inner, var);
            match in_range {
                Truth::True => {}
                Truth::False if real_inner => return no_preimage(),
                Truth::False | Truth::Unknown => return not_inverted(),
            }
            if period.is_some() && !real_inner {
                // the other preimages of a complex argument are not known
                return not_inverted();
            }
            let f = if is_erf {
                crate::base::libfn::LibFn::ErfInv
            } else {
                crate::base::libfn::LibFn::ErfcInv
            };
            let new_rhs = arena.lib_apply(f, &[rhs]);
            let new_rhs = crate::transforms::eval::eval(arena, new_rhs);
            solve_by_peeling(arena, inner, new_rhs, var, cx)
        }
        // W(f) = rhs → f = rhs·e^rhs (the principal branch reaches rhs only
        // in its range, which the check of the candidate decides; refused
        // before 0.47, SymPy: `solve(LambertW(2*x) - y)` → `[y*exp(y)/2]`).
        //
        // A rhs with parameters, which that check cannot evaluate, is a value
        // of `W₀` when its assumptions show `rhs ≥ −1`; a real one otherwise
        // exactly under that condition (`W₀(y·e^y) = y` for real `y` iff `y ≥
        // −1`), and a complex one is refused (the range of `W₀` is a region
        // bounded by `−η·cot η + iη`).  Up to 0.52 `W(2x) = y` gave `[y·e^y/2]`
        // for every `y`, wrong for `y = −5` (SymPy 1.14 likewise).
        ExprNode::LambertW(inner) => {
            use crate::base::assumptions::Props;
            use crate::base::truth::Truth;
            let e = arena.exp(rhs);
            let new_rhs = arena.mul(&[rhs, e]);
            let new_rhs = crate::transforms::eval::eval(arena, new_rhs);
            let parametric = crate::base::walk::free_symbols(arena, rhs)
                .into_iter()
                .any(|s| Some(s) != period);
            if parametric && !cx.conditional_roots_as_candidates() {
                let minus_one = arena.neg_one;
                let one = arena.one;
                let shifted = arena.add(&[rhs, one]);
                let mut cache = cx.parameter_cache(arena, rhs);
                let in_range = Truth::from(cache.query(arena, shifted, Props::NONNEGATIVE));
                let real = Truth::from(cache.query(arena, rhs, Props::REAL)).is_true();
                let with_period =
                    period.is_some_and(|p| crate::base::walk::contains(arena, rhs, p));
                match in_range {
                    Truth::True => {}
                    Truth::False if real => return no_preimage(),
                    _ if real && !with_period => {
                        let cond = arena.ge(rhs, minus_one);
                        return solve_by_peeling(arena, inner, new_rhs, var, cx).under(arena, cond);
                    }
                    _ => return not_inverted(),
                }
            }
            solve_by_peeling(arena, inner, new_rhs, var, cx)
        }
        // W_k(f) = rhs → f = rhs·e^rhs for the other branches too, when rhs is
        // a value of W_k — `W_k(rhs·e^rhs) = rhs`, which depends on rhs alone
        // and is decided here (the candidate check skips an equation with
        // parameters): exactly, else no preimage when certified otherwise
        // (`W₁(x) = 0`: `W₁(0) = −∞`; `W₋₁(x) = −asin(1/π)`: `W₋₁` is real
        // only ≤ −1), else refused (`W₋₁(x) = y`) unless roots may hold for
        // some parameter values only ([`SolveCx::conditional_roots_as_candidates`]).
        // `W₋₁(x) = −2` → `−2e⁻²`; refused before 0.52, so an initial value
        // on `W₋₁` could not be fitted (SymPy 1.14 raises NotImplementedError).
        ExprNode::Lib(crate::base::libfn::LibFn::LambertW, ref args)
            if args.len() == 2 && !expr_contains_var(arena, args[1], var) =>
        {
            let (inner, k) = (args[0], args[1]);
            let e = arena.exp(rhs);
            let new_rhs = arena.mul(&[rhs, e]);
            let new_rhs = crate::transforms::eval::eval(arena, new_rhs);
            let back = arena.lambertw_branch(new_rhs, k);
            let gap = arena.sub(back, rhs);
            let gap = crate::transforms::eval::eval(arena, gap);
            if !arena.is_zero_structural(gap) {
                // `gap` is free of `var`: the substitution is the identity.
                if certainly_not_a_root(arena, gap, var, var) {
                    return Solutions::none(Reason::said(
                        "the right-hand side is not a value of this branch of Lambert W",
                    ));
                }
                if !cx.conditional_roots_as_candidates() {
                    return not_inverted();
                }
            }
            solve_by_peeling(arena, inner, new_rhs, var, cx)
        }
        // atan2(f, c) = rhs for a constant c > 0 is atan(f/c) = rhs: f =
        // c·tan(rhs) (in the range when |rhs| < π/2, checked; refused before
        // 0.47, SymPy: `solve(atan2(x, 2) - pi/3)` → `[2*sqrt(3)]`).
        ExprNode::Atan2(f, c)
            if !expr_contains_var(arena, c, var)
                && crate::base::walk::free_symbols(arena, c).is_empty()
                && crate::poly::algebraic::sign_checked(arena, c) == Some(1) =>
        {
            let t = arena.tan(rhs);
            let new_rhs = arena.mul(&[c, t]);
            let new_rhs = crate::transforms::eval::eval(arena, new_rhs);
            solve_by_peeling(arena, f, new_rhs, var, cx)
        }
        // erfinv(f) = rhs → f = erf(rhs), erfcinv(f) = rhs → f = erfc(rhs)
        // (the inverses are injective; candidates are checked).
        ExprNode::Lib(
            f @ (crate::base::libfn::LibFn::ErfInv | crate::base::libfn::LibFn::ErfcInv),
            ref args,
        ) if args.len() == 1 => {
            let inner = args[0];
            let new_rhs = if f == crate::base::libfn::LibFn::ErfInv {
                arena.intern(ExprNode::Erf(rhs))
            } else {
                arena.intern(ExprNode::Erfc(rhs))
            };
            let new_rhs = crate::transforms::eval::eval(arena, new_rhs);
            solve_by_peeling(arena, inner, new_rhs, var, cx)
        }
        // |f(x)| = rhs → f(x) = rhs OR f(x) = -rhs (when rhs ≥ 0)
        ExprNode::Abs(inner) => {
            // |f(x)| = negative has no solutions
            if let Some(r) = arena.as_num(rhs)
                && r.is_negative()
            {
                return no_preimage();
            }
            // |f| = 0 exactly when f = 0, for any complex `f`.
            if arena.is_zero_structural(rhs) {
                return solve_by_peeling(arena, inner, rhs, var, cx);
            }
            // `f = ±rhs` are all the solutions only for a real `f`: a
            // complex `f` has modulus `rhs` on a whole circle.  Before 0.31
            // `|i·x| = 2` gave `x = ±2i` and missed the real `x = ±2`;
            // up to 0.48 a plain `x` was taken real here under `Ex::solve`
            // (`|x| = 1` gave `[1, −1]`; see `SolveCx::real`).
            if !cx.real(arena, inner, var) {
                // … but none when `rhs` is a constant that is not a
                // nonnegative real (`|x|² = −4` peels to `|x| = ±2i`).
                let not_a_modulus = crate::base::walk::free_symbols(arena, rhs).is_empty()
                    && (crate::transforms::realness::constant_realness_as_declared(
                        arena,
                        rhs,
                        ROOT_CHECK_DIGITS,
                        &mut crate::base::assumptions::AssumptionCache::new(),
                    ) == Some(false)
                        || crate::poly::algebraic::sign_checked(arena, rhs) == Some(-1));
                if not_a_modulus {
                    return no_preimage();
                }
                // A real unknown: `|f|² = (Re f)² + (Im f)² = rhs²`, an
                // equivalent real equation for `rhs ≥ 0` (`|x + i| = 2`:
                // `x² + 1 = 4`, `x = ±√3`; up to 0.49 refused).
                if !cx.unknown_may_be_nonreal(arena, var)
                    && crate::base::walk::free_symbols(arena, rhs).is_empty()
                    && crate::poly::algebraic::sign_checked(arena, rhs).is_some_and(|s| s >= 0)
                    && let Some(squared) = modulus_squared(arena, inner, var)
                {
                    let two = arena.int(2);
                    let rhs2 = arena.pow(rhs, two);
                    let eq = arena.sub(squared, rhs2);
                    let eq = crate::transforms::expand::expand(arena, eq);
                    let eq = crate::transforms::eval::eval(arena, eq);
                    if expr_contains_var(arena, eq, var) {
                        return solve_raw(arena, eq, var, cx);
                    }
                }
                return not_inverted();
            }
            let neg_rhs = arena.neg(rhs);
            peel_two_branches(arena, inner, rhs, neg_rhs, var, cx)
        }
        _ => not_inverted(),
    }
}

/// `|f|² = (Re f)² + (Im f)²` for a real `var`, expanded (`|x + i|² =
/// x² + 1`, `|a·x|² = (re(a)² + im(a)²)·x²`); `None` when a part of `f`
/// is not found free of `re`, `im`, `arg`, `conjugate` of `var`.
fn modulus_squared(arena: &mut Arena, f: ExprId, var: ExprId) -> Option<ExprId> {
    use crate::base::assumptions::{Assumptions, Props};
    // the parts for a real variable: a real stand-in for `var`
    let t = crate::transforms::subs::fresh_symbol(arena, "__abs_real", &[f]);
    if let ExprNode::Symbol(sid) = *arena.node(t) {
        let mut real = Assumptions::default();
        real.assert_true(Props::REAL);
        arena.set_symbol_assumptions(sid, real);
    }
    let f_t = crate::transforms::subs::subs(arena, f, var, t);
    let f_t = crate::transforms::expand::expand(arena, f_t);
    let (re, im) = crate::base::complex::as_real_imag(arena, f_t);
    let mut parts = Vec::with_capacity(2);
    for p in [re, im] {
        let p = crate::transforms::eval::eval(arena, p);
        let opaque = crate::base::walk::post_order_ids(arena, p)
            .into_iter()
            .any(|id| match *arena.node(id) {
                ExprNode::Re(u) | ExprNode::Im(u) | ExprNode::Arg(u) | ExprNode::Conjugate(u) => {
                    crate::base::walk::contains(arena, u, t)
                }
                _ => false,
            });
        if opaque {
            return None;
        }
        parts.push(p);
    }
    let two = arena.int(2);
    let re2 = arena.pow(parts[0], two);
    let im2 = arena.pow(parts[1], two);
    let sum = arena.add(&[re2, im2]);
    let sum = crate::transforms::subs::subs(arena, sum, t, var);
    let sum = crate::transforms::expand::expand(arena, sum);
    Some(crate::transforms::eval::eval(arena, sum))
}

/// Does `rhs` hold the integer parameter of a general solve (a family from
/// an outer periodic layer)?
pub(super) fn period_in_use(arena: &Arena, rhs: ExprId, cx: &SolveCx) -> bool {
    cx.period()
        .is_some_and(|n| crate::base::walk::contains(arena, rhs, n))
}

/// Most members of a family listed by [`bounded_family_members`].
const MAX_BOUNDED_MEMBERS: usize = 64;

/// The members `c(k)` of the family `c(n) = a + b·n` (constants `a`, `b`
/// proved real, `b ≠ 0`) that may lie in `[−1, 1]`, the range of `sin` and
/// `cos` on the reals: the integers `k` with `|a + b·k| ≤ 1`, and one more
/// on either side (a member just outside is answered with no real solution
/// by the next layer), from [`lattice_indices`].  `None` when `c` is not of
/// that form, the range is not decided or there are too many.
///
/// Up to 0.53 the range came from `f64` values of `a` and `b` (an imaginary
/// part below `10⁻¹²` read as 0) and was cast with `as i64`: for `tan(sin x
/// − 10²⁰) = 1` (`sin x = 10²⁰ + π/4 + kπ`, the member `k ≈ −3.18·10¹⁹` is
/// `0.0840…`) the index saturated, no member was listed, and a real `x` had
/// "no solution".
fn bounded_family_members(arena: &mut Arena, c: ExprId, n: ExprId) -> Option<Vec<ExprId>> {
    use crate::base::truth::SignSet;
    let expanded = crate::transforms::expand::expand(arena, c);
    let coeffs = symbolic_poly_coeffs(arena, expanded, n)?;
    let [a, b] = coeffs[..] else {
        return None;
    };
    if !crate::base::walk::free_symbols(arena, a).is_empty()
        || !crate::base::walk::free_symbols(arena, b).is_empty()
    {
        return None;
    }
    for part in [a, b] {
        if crate::transforms::realness::imaginary_part_sign(arena, part) != SignSet::ZERO {
            return None;
        }
    }
    let (lo, hi) = (arena.neg_one, arena.one);
    let indices = lattice_indices(arena, a, b, lo, hi, MAX_BOUNDED_MEMBERS)?;
    let mut out = Vec::with_capacity(indices.len());
    for k in indices {
        let kk = arena.big_int(k);
        let m = crate::transforms::subs::subs(arena, c, n, kk);
        out.push(crate::transforms::eval::eval(arena, m));
    }
    Some(out)
}

/// The integers `k` from `⌊min(q_lo, q_hi)⌋ − 1` to `⌈max(q_lo, q_hi)⌉ + 1`,
/// `q_end = (end − offset)/step`: every index of the lattice `offset +
/// step·k` (real constants, `step ≠ 0`) whose member lies between the real
/// constants `lo` and `hi`, and one more on either side.  The floors and
/// ceilings are exact (`eval`: of a rational, or the integers a certified
/// value lies strictly between), so the range is right however far the
/// window is from the lattice's origin.  `None` when one of them is not
/// decided or the range holds more than `max` integers.
///
/// Up to 0.53 such ranges were computed from 16-digit `f64` values and cast
/// with `as i64`: an index beyond `2⁵³` lost its last digits and one beyond
/// `2⁶³` saturated, and the members in the window were missed
/// (`solve_as_set(sin(x − 10²⁰)) ∩ [0, 10]` was empty).
pub(crate) fn lattice_indices(
    arena: &mut Arena,
    offset: ExprId,
    step: ExprId,
    lo: ExprId,
    hi: ExprId,
    max: usize,
) -> Option<Vec<BigInt>> {
    let mut quotients = [lo, hi];
    for q in &mut quotients {
        let d = arena.sub(*q, offset);
        *q = arena.div(d, step);
    }
    integers_around(arena, quotients, max)
}

/// The integers from `⌊min q⌋ − 1` to `⌈max q⌉ + 1` over the real constants
/// `q`, the floors and ceilings exact (see [`lattice_indices`]).  A
/// quotient within its error of an integer (`⌊q⌋` open: `π²/π² − 1` not
/// simplified, an index at a window's end) is bracketed by its certified
/// 20-digit value, widened by one more on either side.  `None` when a
/// quotient has no value or there are more than `max + 1` integers.
pub(crate) fn integers_around(
    arena: &mut Arena,
    quotients: [ExprId; 2],
    max: usize,
) -> Option<Vec<BigInt>> {
    let integer = |arena: &Arena, id: ExprId| {
        arena
            .as_num(id)
            .filter(|q| q.is_integer())
            .map(|q| q.to_integer())
    };
    let mut floors: Vec<BigInt> = Vec::with_capacity(2);
    let mut ceilings: Vec<BigInt> = Vec::with_capacity(2);
    for q in quotients {
        let q = crate::transforms::eval::eval(arena, q);
        let f = arena.floor(q);
        let f = crate::transforms::eval::eval(arena, f);
        let c = arena.ceiling(q);
        let c = crate::transforms::eval::eval(arena, c);
        match (integer(arena, f), integer(arena, c)) {
            (Some(f), Some(c)) => {
                floors.push(f);
                ceilings.push(c);
            }
            _ => {
                // `eval` leaves the floor open: the certified 20-digit value
                // places `q` within one of an integer (far below 2⁴⁰, its
                // error is far below 1), and the range is widened by one more
                let v = crate::transforms::evalf::const_value(arena, q, 20)
                    .ok()
                    .and_then(|v| v.to_f64())
                    .filter(|v| v.abs() < 1_099_511_627_776.0)?;
                let k = crate::base::cast::f64_to_i64(v.floor())?;
                floors.push(BigInt::from(k) - BigInt::one());
                ceilings.push(BigInt::from(k) + BigInt::from(2));
            }
        }
    }
    let first = floors.into_iter().min()? - BigInt::one();
    let last = ceilings.into_iter().max()? + BigInt::one();
    let count = num_traits::ToPrimitive::to_usize(&(&last - &first)).filter(|&c| c <= max)?;
    Some((0..=count).map(|i| &first + BigInt::from(i)).collect())
}

/// Largest `|p|` for which `f^(p/q) = c` with a numeric `c` is inverted
/// through all `|p|` roots of `u^p = c` (beyond it, the principal
/// candidate only).
const MAX_POWER_PREIMAGES: i64 = 24;

/// The values `f` with `f^n = c` (`n ≠ 0` rational) that peeling continues
/// with; `Values(vec![])` when there is none (`f^(−k) = 0`), `None` when
/// refused.
///
/// * Integer `n`: every root of `f^|n| = c^(sign n)` (`binomial_roots`), as
///   the polynomial route returns all roots of `xⁿ = c`.  Before 0.30 only
///   `c^(1/n)` (and its negative for a positive even `n`) came back:
///   `x^(−2) = 3` lost `−1/√3`, `(x²)^(−3) = 1/2` lost `−2^(1/6)`.
/// * `n = p/q` in lowest terms, `q > 1`: `f^(p/q) = (f^(1/q))^p` with the
///   principal `u = f^(1/q)` in the sector `−π/q < arg u ≤ π/q`, so the
///   candidates are `u^q` for the roots `u` of `u^p = c`; those outside the
///   sector are spurious (`√x = −2` has none, `x^(1/3) = −3/4` neither)
///   and are dropped by the check of every candidate against the equation
///   ([`certainly_not_a_root`]).  For a `c` with free symbols that check
///   cannot run, and only the principal candidate `c^(q/p)` can be a root
///   ([`principal_power_preimage`] decides whether it is one): it is kept
///   when it is, there is no preimage when `c` is nonzero and outside the
///   range of `f^(p/q)` (`√x = −√y`, `y > 0`), and otherwise the public
///   solvers get it under its exact condition (below) while the crate's
///   root searches keep the candidate
///   ([`SolveCx::conditional_roots_as_candidates`]).  Before 0.51 it was
///   always kept: `√(x + ∛x) + (x − y)^(1/5) = 0` gave `y = x + (x +
///   ∛x)^(5/2)`, a root nowhere but at `x + ∛x = 0` (`−√w` is never a
///   principal fifth root), as in SymPy 1.14.
///
/// In a general solve a `c` whose only symbol is the integer parameter
/// stands for constants, each member of the family of candidates checked
/// by [`check_family`] (`e^√x = −1`: `√x = (2n + 1)πi`, `x = −(2n +
/// 1)²π²`, all roots although half the `√x` are not principal).
///
/// When the sector test with parameters is undecided, a public solver
/// gets the principal candidate under its exact condition on the
/// parameters ([`Preimages::Conditional`], [`principal_power_condition`]):
/// `√x = y` is `x = y²` when `y ≥ 0` for a real `y`.  Up to 0.52 it
/// refused (since 0.51), and a `c` with the integer parameter of a general
/// solve still is (the condition would depend on the member).
fn power_preimages(arena: &mut Arena, c: ExprId, n: &Q, cx: &SolveCx) -> Option<Preimages> {
    let p = n.numer().clone();
    let q = n.denom().clone();
    let c_zero = arena.is_zero_structural(c);
    if c_zero {
        // f^n = 0: f = 0 for n > 0, nothing for n < 0.
        return Some(Preimages::Values(if n.is_positive() {
            vec![arena.zero]
        } else {
            vec![]
        }));
    }
    let symbolic = !crate::base::walk::free_symbols(arena, c).is_empty();
    let p_abs: i64 = num_traits::ToPrimitive::to_i64(&p.abs()).unwrap_or(i64::MAX);
    if q.is_one() || (!symbolic && p_abs <= MAX_POWER_PREIMAGES) {
        // u^|p| = c^(sign p), then f = u^q.
        let base = if p.is_negative() {
            let m1 = arena.neg_one;
            let inv = arena.pow(c, m1);
            crate::transforms::eval::eval(arena, inv)
        } else {
            c
        };
        if p_abs > MAX_INT_POWER_ROOTS {
            return None;
        }
        let roots = binomial_roots(arena, base, p_abs as usize);
        // The `k`-th root of `u^|p| = c^(sign p)` has argument
        // `π·(t + 2k)/|p|` with `t·π = arg c^(sign p)` (`binomial_roots`
        // multiplies the principal root by `e^(2πik/|p|)`, and writes a
        // negative real `c` as `|c|·e^(iπ)`).  When `t` is known exactly the
        // sector test is exact; otherwise the check of every candidate
        // against the equation decides.
        let exact_arg = if q.is_one() {
            None
        } else {
            gaussian_rational_arg(arena, c).map(|t| {
                if p.is_negative() && t != Ratio::one() {
                    -t
                } else {
                    t
                }
            })
        };
        // `binomial_roots` lists one root per `k = 0..|p|` (distinct for
        // `c ≠ 0`), so the position of a root is its `k`.
        let aligned = roots.len() as i64 == p_abs;
        let q_id = arena.big_int(q.clone());
        let mut out = Vec::with_capacity(roots.len());
        for (k, r) in roots.into_iter().enumerate() {
            if let Some(t) = &exact_arg
                && aligned
                && !in_principal_sector(t, k, p_abs, &q)
            {
                continue;
            }
            let f = arena.pow(r.value, q_id);
            let f = crate::transforms::expand::expand(arena, f);
            let f = crate::transforms::eval::eval(arena, f);
            if !out.contains(&f) {
                out.push(f);
            }
        }
        return Some(Preimages::Values(out));
    }
    // Principal candidate only: f = c^(1/n).
    let parametric = crate::base::walk::free_symbols(arena, c)
        .into_iter()
        .any(|s| Some(s) != cx.period());
    let inv_n = Ratio::one() / n.clone();
    let inv_n_id = arena.num_ratio(inv_n);
    if parametric {
        match principal_power_preimage(arena, c, n) {
            crate::base::truth::Truth::True => {}
            crate::base::truth::Truth::False => return Some(Preimages::Values(Vec::new())),
            crate::base::truth::Truth::Unknown if cx.conditional_roots_as_candidates() => {}
            crate::base::truth::Truth::Unknown => {
                if cx
                    .period()
                    .is_some_and(|p| crate::base::walk::contains(arena, c, p))
                {
                    return None;
                }
                let cond = principal_power_condition(arena, c, n)?;
                let value = arena.pow(c, inv_n_id);
                return Some(Preimages::Conditional { value, cond });
            }
        }
    }
    Some(Preimages::Values(vec![arena.pow(c, inv_n_id)]))
}

/// A principal inverse function peeled by its forward function, `inv(f) =
/// c → f = fwd(c)` ([`peel_principal_inverse`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PrincipalInverse {
    Ln,
    Asin,
    Acos,
    Atan,
    Asinh,
    Acosh,
    Atanh,
}

/// `inv(inner) = rhs` peeled as `inner = new_rhs` (`= fwd(rhs)`), which
/// holds exactly when `rhs` is a value of the principal `inv`.  For a
/// constant `rhs` the check of every candidate decides; one with
/// parameters, which that check cannot evaluate, is decided by its
/// assumptions where they suffice, and otherwise a public solver gets the
/// root under the exact condition ([`Solutions::under`]): `ln f = y` needs
/// `−π < im y ≤ π` (always for a real `y`), and for a real `y` `asin`
/// needs `|y| ≤ π/2`, `acos` `0 ≤ y ≤ π`, `atan` `|y| < π/2`, `acosh` `y ≥
/// 0`, `asinh` and `atanh` nothing; a complex `y` is refused for the
/// inverse trigonometric and hyperbolic functions (their ranges have
/// boundaries that depend on the branch-cut conventions).  Up to 0.52 the
/// root was returned for every value: `ln x = y` gave `[e^y]` (wrong for
/// `y = 4i`: `ln(e^{4i}) = (4 − 2π)i`), `asin x = y` for a real `y` gave
/// `[sin y]` (wrong for `y = 4`), as SymPy 1.14 does.
#[allow(clippy::too_many_arguments)]
fn peel_principal_inverse(
    arena: &mut Arena,
    inverse: PrincipalInverse,
    inner: ExprId,
    rhs: ExprId,
    new_rhs: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Solutions {
    use crate::base::assumptions::Props;
    let parametric = crate::base::walk::free_symbols(arena, rhs)
        .into_iter()
        .any(|s| Some(s) != cx.period());
    if !parametric || cx.conditional_roots_as_candidates() {
        return solve_by_peeling(arena, inner, new_rhs, var, cx);
    }
    // (plain parameters are real under `Params::Real`: `reduce_inequalities`)
    let mut cache = cx.parameter_cache(arena, rhs);
    let real = crate::base::truth::Truth::from(cache.query(arena, rhs, Props::REAL)).is_true();
    let (subject, bounds): (ExprId, Vec<RangeBound>) = if real {
        let pi = arena.pi;
        let half = arena.rational(1, 2);
        let half_pi = arena.mul(&[half, pi]);
        let minus_half_pi = arena.neg(half_pi);
        let zero = arena.zero;
        let b = |value: ExprId, lower: bool, strict: bool| RangeBound {
            value,
            lower,
            strict,
        };
        let bounds = match inverse {
            PrincipalInverse::Ln | PrincipalInverse::Asinh | PrincipalInverse::Atanh => Vec::new(),
            PrincipalInverse::Asin => {
                vec![b(minus_half_pi, true, false), b(half_pi, false, false)]
            }
            PrincipalInverse::Acos => vec![b(zero, true, false), b(pi, false, false)],
            PrincipalInverse::Atan => vec![b(minus_half_pi, true, true), b(half_pi, false, true)],
            PrincipalInverse::Acosh => vec![b(zero, true, false)],
        };
        (rhs, bounds)
    } else if inverse == PrincipalInverse::Ln {
        let im = arena.im(rhs);
        let im = crate::transforms::eval::eval(arena, im);
        // a principal argument is in `(−π, π]` (`ln(log(−3x, y)) = 0`:
        // `rhs = ln y`, `im = arg y`)
        if matches!(arena.node(im), ExprNode::Arg(_)) {
            return solve_by_peeling(arena, inner, new_rhs, var, cx);
        }
        let pi = arena.pi;
        let minus_pi = arena.neg(pi);
        let bounds = vec![
            RangeBound {
                value: minus_pi,
                lower: true,
                strict: true,
            },
            RangeBound {
                value: pi,
                lower: false,
                strict: false,
            },
        ];
        (im, bounds)
    } else {
        return Solutions::refused(Reason::NotFound);
    };
    // the bounds the assumptions do not decide
    let mut open: Vec<ExprId> = Vec::new();
    for bound in bounds {
        let (hi, lo) = if bound.lower {
            (subject, bound.value)
        } else {
            (bound.value, subject)
        };
        let gap = arena.sub(hi, lo);
        let gap = crate::transforms::eval::eval(arena, gap);
        let prop = if bound.strict {
            Props::POSITIVE
        } else {
            Props::NONNEGATIVE
        };
        match crate::base::truth::Truth::from(cache.query(arena, gap, prop)) {
            crate::base::truth::Truth::True => {}
            crate::base::truth::Truth::False => return Solutions::none(Reason::NoPreimage),
            crate::base::truth::Truth::Unknown => open.push(if bound.strict {
                arena.gt(hi, lo)
            } else {
                arena.ge(hi, lo)
            }),
        }
    }
    let peeled = solve_by_peeling(arena, inner, new_rhs, var, cx);
    if open.is_empty() {
        return peeled;
    }
    if cx
        .period()
        .is_some_and(|p| crate::base::walk::contains(arena, rhs, p))
    {
        return Solutions::refused(Reason::NotFound);
    }
    let cond = arena.and(&open);
    peeled.under(arena, cond)
}

/// One end of the range of a principal inverse ([`peel_principal_inverse`]).
struct RangeBound {
    /// The end.
    value: ExprId,
    /// A lower end (the value must be above it).
    lower: bool,
    /// The end itself is excluded.
    strict: bool,
}

/// The roots of `base^c = 0` for an exponent `c` free of `var` (not a
/// number): those of `base = 0` when `re c > 0`, none when `re c ≤ 0` —
/// decided by a certified sign for a constant `c`, by the assumptions of
/// its symbols otherwise, and else, in a public solver, under the
/// condition `re c > 0` (`c > 0` for a real `c`; [`Solutions::under`]).
pub(super) fn power_equals_zero(
    arena: &mut Arena,
    base: ExprId,
    c: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Solutions {
    use crate::base::assumptions::{AssumptionCache, Props};
    use crate::base::truth::Truth;
    let zero = arena.zero;
    let re = arena.re(c);
    let re = crate::transforms::eval::eval(arena, re);
    let constant = crate::base::walk::free_symbols(arena, c).is_empty();
    let positive = if constant {
        part_signs(arena, re).re.is_positive()
    } else {
        Truth::from(AssumptionCache::new().query(arena, re, Props::POSITIVE))
    };
    match positive {
        Truth::True => solve_by_peeling(arena, base, zero, var, cx),
        Truth::False => Solutions::none(Reason::NoPreimage),
        // (the crate's root searches keep refusing here: the integrator
        // would take `0` as a breakpoint of `x·δ(x^y)` for every `y`)
        Truth::Unknown
            if !constant
                && !cx.conditional_roots_as_candidates()
                && !cx
                    .period()
                    .is_some_and(|p| crate::base::walk::contains(arena, c, p)) =>
        {
            let at_zero = solve_by_peeling(arena, base, zero, var, cx);
            let real = Truth::from(AssumptionCache::new().query(arena, c, Props::REAL)).is_true();
            let cond = arena.gt(if real { c } else { re }, zero);
            at_zero.under(arena, cond)
        }
        Truth::Unknown => Solutions::refused(Reason::NotFound),
    }
}

/// The values `f` with `f^n = c` ([`power_preimages`]).
enum Preimages {
    /// Exactly these (none: no preimage).
    Values(Vec<ExprId>),
    /// `value` when `cond`, a condition on the parameters, holds, and none
    /// otherwise.
    Conditional {
        /// The principal candidate `c^(1/n)`.
        value: ExprId,
        /// The exact condition ([`principal_power_condition`]).
        cond: ExprId,
    },
}

/// The exact condition on the symbols of `c` under which the principal
/// candidate `c^(q/p)` solves `f^(p/q) = c` (`n = p/q` in lowest terms,
/// `|p| < q`; see [`principal_power_preimage`]): `arg c` in the range of
/// `f^(p/q)` — `(−|p|π/q, |p|π/q]` for `p > 0`, with `c = 0` (whose
/// preimage is `0`), `[−|p|π/q, |p|π/q)` without it for `p < 0`.  Written
/// `c ≥ 0` (`c > 0` for `p < 0`) for a `c` real by its symbols'
/// assumptions, `re c > 0 ∨ (re c = 0 ∧ im c ≥ 0)` (`im c < 0` for `p < 0`)
/// for `|p|/q = 1/2`, and with `arg c` otherwise.  `None` for `|p| ≥ q`
/// (every `c` has its preimage).
pub(super) fn principal_power_condition(arena: &mut Arena, c: ExprId, n: &Q) -> Option<ExprId> {
    use crate::base::assumptions::{AssumptionCache, Props};
    let (p, q) = (n.numer().abs(), n.denom().clone());
    if p >= q {
        return None;
    }
    let positive = n.is_positive();
    let zero = arena.zero;
    let real = crate::base::truth::Truth::from(AssumptionCache::new().query(arena, c, Props::REAL))
        .is_true();
    if real {
        return Some(if positive {
            arena.ge(c, zero)
        } else {
            arena.gt(c, zero)
        });
    }
    let b = Ratio::new(p, q);
    if b == Ratio::new(BigInt::one(), BigInt::from(2)) {
        let re = arena.re(c);
        let re = crate::transforms::eval::eval(arena, re);
        let im = arena.im(c);
        let im = crate::transforms::eval::eval(arena, im);
        let right = arena.gt(re, zero);
        let on_axis = arena.eq_(re, zero);
        let side = if positive {
            arena.ge(im, zero)
        } else {
            arena.gt(zero, im)
        };
        let edge = arena.and(&[on_axis, side]);
        return Some(arena.or(&[right, edge]));
    }
    let arg = arena.arg(c);
    let arg = crate::transforms::eval::eval(arena, arg);
    let b_id = arena.num_ratio(b);
    let pi = arena.pi;
    let hi = arena.mul(&[b_id, pi]);
    let hi = crate::transforms::eval::eval(arena, hi);
    let lo = arena.neg(hi);
    let lo = crate::transforms::eval::eval(arena, lo);
    Some(if positive {
        let above = arena.gt(arg, lo);
        let below = arena.ge(hi, arg);
        let sector = arena.and(&[above, below]);
        let at_zero = arena.eq_(c, zero);
        arena.or(&[sector, at_zero])
    } else {
        let above = arena.ge(arg, lo);
        let below = arena.gt(hi, arg);
        let nonzero = arena.ne_(c, zero);
        arena.and(&[above, below, nonzero])
    })
}

/// Does the principal candidate `F = c^(q/p)` solve `f^(p/q) = c` (`n =
/// p/q` in lowest terms, `q > 1`) for every value of the symbols in `c`?
///
/// `F^(p/q) = c` exactly when `θ·q/p` (`θ = arg c ∈ (−π, π]`) needs no
/// reduction into `(−π, π]`: always for `|p| > q`, and for `|p| < q` when
/// `θ ∈ (−|p|π/q, |p|π/q]` (`p > 0`) or `[−|p|π/q, |p|π/q)` (`p < 0`) — the
/// range of `f^(p/q)` — or `c = 0`.  No other preimage exists then (a root
/// `u` of `u^p = c` in the sector `|arg u| ≤ π/q` of principal `q`-th roots
/// has `|arg(u^p)| < π`).  `True` when the argument range of `c`
/// ([`arg_range`]) lies in that range, `False` when it misses it and `c` is
/// not 0, `Unknown` otherwise: `√x = y` for a plain `y` (`x = y²` only for
/// `−π/2 < arg y ≤ π/2`), or `−√w` when `w` may be 0.
pub(super) fn principal_power_preimage(
    arena: &mut Arena,
    c: ExprId,
    n: &Q,
) -> crate::base::truth::Truth {
    use crate::base::truth::Truth;
    let (p, q) = (n.numer().abs(), n.denom().clone());
    if p > q {
        return Truth::True;
    }
    let Some(range) = arg_range(arena, c) else {
        return Truth::Unknown;
    };
    let b = Ratio::new(p, q);
    let positive = n.is_positive();
    let (lo, hi) = (&range.center - &range.half, &range.center + &range.half);
    let point = range.half.is_zero();
    // the range of `f^(p/q)` shifted by `2k` turns: `(a, a + 2b]` for
    // `p > 0`, `[a, a + 2b)` for `p < 0`
    let meets = |k: i64| {
        let shift = Ratio::from_integer(BigInt::from(2 * k));
        let (a, e) = (&shift - &b, &shift + &b);
        match (point, positive) {
            (true, true) => a < lo && lo <= e,
            (true, false) => a <= lo && lo < e,
            (false, true) => lo < e && a < hi,
            (false, false) => a <= hi && lo < e,
        }
    };
    // in `(−b, b]` (`p > 0`) or `[−b, b)` (`p < 0`); an interval open
    // at `lo` may start at `−b`
    let minus_b = -b.clone();
    let inside = match (point, positive) {
        (true, true) => minus_b < lo && hi <= b,
        (false, true) => minus_b <= lo && hi <= b,
        (_, false) => minus_b <= lo && hi < b,
    };
    if inside {
        Truth::True
    } else if !range.maybe_zero && !(-1..=1).any(meets) {
        Truth::False
    } else {
        Truth::Unknown
    }
}

/// The arguments a value can take, in turns of `π`: `arg/π ∈ (center −
/// half, center + half]` (modulo 2; the point `center` when `half` is 0),
/// or the value is 0 (`maybe_zero`).
struct ArgRange {
    center: Q,
    half: Q,
    maybe_zero: bool,
}

/// The [`ArgRange`] of a product of factors whose arguments are known
/// exactly from the principal branch and the declared assumptions: a
/// nonzero rational or Gaussian rational with a rational argument
/// ([`gaussian_rational_arg`]), a positive or negative factor (`0`, `1`;
/// nonnegative and nonpositive ones may also be 0), `B^r` with a negative
/// `B` (`r`) and, for `0 < r < 1`, `B^r` with `B` of unknown sign (`r·arg B
/// ∈ (−rπ, rπ]`, or 0).  `None` when a factor is not of these kinds or the
/// range is the whole circle.  Exact: no evaluation decides an argument.
fn arg_range(arena: &mut Arena, value: ExprId) -> Option<ArgRange> {
    use crate::base::assumptions::{AssumptionCache, Props};
    let mut factors: Vec<ExprId> = vec![value];
    let mut out = ArgRange {
        center: Q::zero(),
        half: Q::zero(),
        maybe_zero: false,
    };
    let mut cache = AssumptionCache::new();
    while let Some(f) = factors.pop() {
        match arena.node(f) {
            ExprNode::Mul(ch) => {
                factors.extend(ch.iter().copied());
                continue;
            }
            ExprNode::Neg(inner) => {
                factors.push(*inner);
                out.center += Q::one();
                continue;
            }
            _ => {}
        }
        if crate::base::walk::free_symbols(arena, f).is_empty()
            && let Some(t) = gaussian_rational_arg(arena, f)
        {
            out.center += t;
            continue;
        }
        let mut known = |arena: &Arena, e: ExprId, p: Props| {
            crate::base::truth::Truth::from(cache.query(arena, e, p)).is_true()
        };
        if known(arena, f, Props::POSITIVE) {
            continue;
        }
        if known(arena, f, Props::NEGATIVE) {
            out.center += Q::one();
            continue;
        }
        if known(arena, f, Props::NONNEGATIVE) {
            out.maybe_zero = true;
            continue;
        }
        if known(arena, f, Props::NONPOSITIVE) {
            out.center += Q::one();
            out.maybe_zero = true;
            continue;
        }
        let ExprNode::Pow(base, e) = *arena.node(f) else {
            return None;
        };
        let r = arena.as_num(e)?.clone();
        if known(arena, base, Props::NEGATIVE) {
            out.center += r;
        } else if r.is_positive() && r < Q::one() {
            out.half += &r;
            out.maybe_zero |= !known(arena, base, Props::NONZERO);
        } else {
            return None;
        }
    }
    if out.half >= Q::one() {
        return None;
    }
    let two = Q::from_integer(BigInt::from(2));
    while out.center > Q::one() {
        out.center -= &two;
    }
    while out.center <= -Q::one() {
        out.center += &two;
    }
    Some(out)
}

/// Largest integer exponent `|n|` whose `n` roots are listed when peeling
/// `f^n = c`.
const MAX_INT_POWER_ROOTS: i64 = 64;

/// The principal logarithm of `c`, folded for a Gaussian rational whose
/// argument is a rational multiple of `π` (`ln i = iπ/2`,
/// `ln(1 + i) = ln 2/2 + iπ/4`), which `eval` leaves as `ln(i)`; a
/// candidate such as `(ln(i) − 2πi)²` (`exp(√x) = i`) is otherwise a
/// negative real the residual check cannot place on the branch cut of `√`.
pub(super) fn principal_ln(arena: &mut Arena, c: ExprId) -> ExprId {
    let Some(t) = gaussian_rational_arg(arena, c) else {
        return arena.ln(c);
    };
    if t.is_zero() || t.is_one() {
        return arena.ln(c);
    }
    let (re, im) = crate::base::complex::as_real_imag(arena, c);
    let re = crate::transforms::eval::eval(arena, re);
    let im = crate::transforms::eval::eval(arena, im);
    let (Some(a), Some(b)) = (arena.as_num(re).cloned(), arena.as_num(im).cloned()) else {
        return arena.ln(c);
    };
    // |c| = |b| (a = 0) or |a|·√2 (|a| = |b|)
    let modulus_ln = if a.is_zero() {
        let m = arena.num_ratio(b.abs());
        arena.ln(m)
    } else {
        let m = arena.num_ratio(a.abs());
        let ln_m = arena.ln(m);
        let two = arena.int(2);
        let ln2 = arena.ln(two);
        let half = arena.rational(1, 2);
        let half_ln2 = arena.mul(&[half, ln2]);
        arena.add(&[ln_m, half_ln2])
    };
    let t_id = arena.num_ratio(t);
    let angle = arena.mul(&[t_id, arena.pi, arena.i_unit]);
    let sum = arena.add(&[modulus_ln, angle]);
    crate::transforms::eval::eval(arena, sum)
}

/// `arg(c)/π ∈ (−1, 1]` for a nonzero Gaussian rational `c = a + b·i` whose
/// argument is a rational multiple of `π` (`b = 0`, `a = 0` or `|a| = |b|`:
/// by Niven's theorem these are the only ones with a rational tangent).
fn gaussian_rational_arg(arena: &mut Arena, c: ExprId) -> Option<Q> {
    let (re, im) = crate::base::complex::as_real_imag(arena, c);
    let re = crate::transforms::eval::eval(arena, re);
    let im = crate::transforms::eval::eval(arena, im);
    let a = arena.as_num(re)?.clone();
    let b = arena.as_num(im)?.clone();
    let q = |n: i64, d: i64| Ratio::new(BigInt::from(n), BigInt::from(d));
    Some(match (a.signum(), b.signum()) {
        (sa, sb) if sa.is_zero() && sb.is_zero() => return None,
        (sa, sb) if sb.is_zero() => {
            if sa.is_positive() {
                q(0, 1)
            } else {
                q(1, 1)
            }
        }
        (sa, sb) if sa.is_zero() => {
            if sb.is_positive() {
                q(1, 2)
            } else {
                q(-1, 2)
            }
        }
        (sa, sb) if a.abs() == b.abs() => match (sa.is_positive(), sb.is_positive()) {
            (true, true) => q(1, 4),
            (false, true) => q(3, 4),
            (false, false) => q(-3, 4),
            (true, false) => q(-1, 4),
        },
        _ => return None,
    })
}

/// Is the `k`-th root `u` of `u^n = c` (argument `π·(t + 2k)/n`, `t·π =
/// arg c`) in the sector `−π/q < arg u ≤ π/q` of principal `q`-th roots?
/// Exactly then `(u^q)^(1/q) = u`, and `u^q` solves `f^(p/q) = c`.
fn in_principal_sector(t: &Q, k: usize, n: i64, q: &BigInt) -> bool {
    let two = Ratio::from_integer(BigInt::from(2));
    let mut s = (t + Ratio::from_integer(BigInt::from(2 * k as i64)))
        / Ratio::from_integer(BigInt::from(n));
    while s > Ratio::one() {
        s -= &two;
    }
    while s <= -Ratio::<BigInt>::one() {
        s += &two;
    }
    let bound = Ratio::new(BigInt::one(), q.clone());
    -bound.clone() < s && s <= bound
}

/// Continue peeling `inner` against each of several right-hand sides (the
/// preimages of a power; none for `f^(−k) = 0`, which has no root) and
/// merge the results by [`Solutions::union`]: no root needs every branch
/// to prove it, and roots next to a refused branch are partial.
fn peel_branches(
    arena: &mut Arena,
    inner: ExprId,
    rhss: &[ExprId],
    var: ExprId,
    cx: &SolveCx,
) -> Solutions {
    let mut out = Solutions::none(Reason::NoPreimage);
    for &rhs in rhss {
        out = out.union(solve_by_peeling(arena, inner, rhs, var, cx));
    }
    out
}

/// Continue peeling `inner` against two alternative right-hand sides (`f =
/// c₁` or `f = c₂`) and merge the results by [`Solutions::union`]: up to
/// 0.48 roots next to a refused branch were all the roots (now partial),
/// and before that one branch proved empty next to a refused one gave "no
/// solution" — for a real `x`, `|x + i|² = 4` peels to `|x + i| = ±2`, the
/// branch `−2` is empty and `|x + i| = 2` (of a non-real argument) is
/// refused, while `±√3` solve it (SymPy's `solveset` too).
fn peel_two_branches(
    arena: &mut Arena,
    inner: ExprId,
    rhs1: ExprId,
    rhs2: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Solutions {
    let first = solve_by_peeling(arena, inner, rhs1, var, cx);
    first.union(solve_by_peeling(arena, inner, rhs2, var, cx))
}
