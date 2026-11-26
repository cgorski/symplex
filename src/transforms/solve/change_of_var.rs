//! Change-of-variable solving (part of [`solve`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Change-of-variable solving
// ═══════════════════════════════════════════════════════════════════════════

/// Try solving by detecting that the expression is polynomial in some f(x).
/// E.g., `exp(2x) - 3*exp(x) + 2` is polynomial in `t = exp(x)`: `t² - 3t + 2`.
pub(super) fn try_change_of_variable(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Solutions {
    let var_sym = match arena.node(var) {
        ExprNode::Symbol(sid) => *sid,
        _ => return Solutions::refused(Reason::NotFound),
    };

    // Collect candidate generators: inner functions applied to var, then
    // trigonometric functions and exponentials of one linear form `a·x + b`,
    // used only for a proof that there is no root (`cos(3x + 1)² + cos(3x +
    // 1) − 6` for a real `x`: up to 0.52 left to the half-angle substitution,
    // which found a partial list; the roots of the others are left to the
    // strategies that found them before).
    let mut candidates = collect_generators(arena, expr, var, var_sym);
    let bare = candidates.len();
    for id in crate::base::walk::post_order_ids(arena, expr) {
        if let ExprNode::Sin(u) | ExprNode::Cos(u) | ExprNode::Tan(u) | ExprNode::Exp(u) =
            *arena.node(id)
            && u != var
            && !candidates.contains(&id)
            && linear_rate(arena, u, var).is_some()
        {
            candidates.push(id);
        }
    }

    for (index, generator) in candidates.into_iter().enumerate() {
        let proofs_only = index >= bare;
        // Create a temporary variable for substitution
        let t = arena.symbol("__t_subst");

        // --- Simple case: direct structural substitution of generator → t ---
        // This handles e.g. sin(x)^2 - sin(x) where walk_and_rebuild
        // naturally replaces sin(x) inside Pow(sin(x), 2) as well.
        let substituted = arena.subs_structural(expr, generator, t);

        if !expr_contains_var(arena, substituted, var) {
            // All the roots in `t`, or none: a polynomial with radical
            // coefficients of which the solver finds some roots only is
            // solved over its number field, or this generator is skipped.
            // A periodic equation in `t` gives its families in a general
            // solve: up to 0.46 its principal roots alone, and for a real `x`
            // `cot(eˣ) = 2` had the solution set `{ln(2·atan(√5 − 2))}`
            // instead of `ln(acot 2 + nπ)` for `n ≥ 0`.
            let t_outcome = if cx.period().is_some() && has_periodic_atom(arena, substituted, t, cx)
            {
                solve_impl(arena, substituted, t, cx)
            } else {
                substitution_roots(arena, substituted, t, cx)
            };

            if let Ok((t_roots, t_missing)) = t_outcome.without_conditions().into_found() {
                // Back-substitute: for each t = c, solve generator(var) = c.
                // Use solve_by_peeling directly to avoid recursing into
                // try_change_of_variable again (which would infinite-loop).
                // Every root in `t` of a polynomial without a preimage is a
                // proof: for a real `x`, `cos(3x + 1)² + cos(3x + 1) − 6 = 0`
                // has `cos(3x + 1) ∈ {2, −3}`, outside `[−1, 1]` (up to 0.52
                // refused, after a half-angle substitution).  (The principal
                // roots of a periodic equation in `t` are not all its roots.)
                let back = back_substitute(arena, generator, t_roots, var, cx).lacking(t_missing);
                let polynomial_in_t = symbolic_poly_coeffs(arena, substituted, t)
                    .is_some_and(|cs| cs.iter().all(|&c| !expr_contains_var(arena, c, t)));

                if (polynomial_in_t && back.is_none_proved())
                    || (!proofs_only && (back.has_roots() || back.is_nested_periods()))
                {
                    return back;
                }
            }
        }

        // --- Advanced: detect exp(n*x) = exp(x)^n pattern ---
        // exp(2*x) is Exp(Mul([2, x])) which does NOT structurally
        // contain exp(x) = Exp(x), so simple substitution misses it.
        // We rewrite exp(k*x) → exp(x)^k first, then substitute.
        if let ExprNode::Exp(inner) = arena.node(generator).clone()
            && inner == var
        {
            let rewritten = rewrite_exp_powers(arena, expr, var, generator);
            if rewritten != expr {
                let substituted2 = arena.subs_structural(rewritten, generator, t);
                if !expr_contains_var(arena, substituted2, var)
                    && let Ok((t_roots, t_missing)) = substitution_roots(arena, substituted2, t, cx)
                        .without_conditions()
                        .into_found()
                {
                    let back =
                        back_substitute(arena, generator, t_roots, var, cx).lacking(t_missing);
                    if back.has_roots() || back.is_nested_periods() {
                        return back;
                    }
                }
            }
        }
    }

    Solutions::refused(Reason::NotFound)
}

/// The roots in `var` of `generator = tᵢ` for the roots `tᵢ` of a
/// substituted equation, by peeling, merged by [`Solutions::union`] (no
/// root when no `tᵢ` has a preimage; partial when one is refused).
pub(super) fn back_substitute(
    arena: &mut Arena,
    generator: ExprId,
    t_roots: Vec<ExprId>,
    var: ExprId,
    cx: &SolveCx,
) -> Solutions {
    let mut out = Solutions::none(Reason::NoPreimage);
    for t in t_roots {
        out = out.union(solve_by_peeling(arena, generator, t, var, cx));
    }
    out
}

/// An equation in `sin`, `cos`, `tan` of `aᵢ·x + bᵢ` alone with rational
/// `aᵢ` whose gcd `r` (gcd of the numerators over the lcm of the
/// denominators) is not 1, solved in `u = r·x` (multiples `aᵢ/r`, coprime
/// integers) and mapped back by `x = u/r`: `sin 300x + sin 600x = 0` and
/// `cos(x/5) + cos(x/15) = 0` were refused before 0.47 (the half-angle
/// substitution of `sin 600x` has degree 1200), SymPy solves both.  `u`
/// carries the assumptions of `x` (`r > 0`).  `None` for another shape or
/// `r = 1`.
pub(super) fn try_scale_trig_arguments(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Solutions {
    let not_found = || Solutions::refused(Reason::NotFound);
    let ExprNode::Symbol(var_sid) = *arena.node(var) else {
        return not_found();
    };
    // `var` only in the trigonometric atoms
    if walk_var_atoms(arena, expr, var, |id| {
        matches!(
            arena.node(id),
            ExprNode::Sin(_) | ExprNode::Cos(_) | ExprNode::Tan(_)
        )
    })
    .is_none()
    {
        return not_found();
    }
    let mut slopes: Vec<Q> = Vec::new();
    let mut ok = true;
    let atoms: Vec<ExprId> = crate::base::walk::post_order_ids(arena, expr)
        .into_iter()
        .filter(|&id| {
            matches!(arena.node(id), ExprNode::Sin(a) | ExprNode::Cos(a) | ExprNode::Tan(a) if expr_contains_var(arena, *a, var))
        })
        .collect();
    for id in atoms {
        let (ExprNode::Sin(a) | ExprNode::Cos(a) | ExprNode::Tan(a)) = *arena.node(id) else {
            continue;
        };
        match linear_rate(arena, a, var) {
            Some((rate, _)) => match arena.as_num(rate) {
                Some(q) if !q.is_zero() => slopes.push(q.abs()),
                _ => ok = false,
            },
            None => ok = false,
        }
    }
    if !ok || slopes.is_empty() {
        return not_found();
    }
    let mut gn = BigInt::zero();
    let mut gd = BigInt::one();
    for q in &slopes {
        gn = num_integer::Integer::gcd(&gn, q.numer());
        gd = num_integer::Integer::lcm(&gd, q.denom());
    }
    let r = Q::new(gn, gd);
    if r.is_zero() || r.is_one() {
        return not_found();
    }
    // A stand-in with the assumptions of the unknown, of the library's own
    // name space: up to 0.60.1 the symbol `__u_scaled` of the one name space
    // was re-declared at every call.
    let declared = arena.symbol_assumptions(var_sid);
    let u = arena.fresh_internal_symbol("__u_scaled", declared, &[expr]);
    let r_id = arena.num_ratio(r.clone());
    let x_of_u = arena.div(u, r_id);
    let sub = crate::transforms::subs::subs(arena, expr, var, x_of_u);
    let sub = crate::transforms::eval::eval(arena, sub);
    if expr_contains_var(arena, sub, var) {
        return not_found();
    }
    solve_raw(arena, sub, u, cx).map_roots(|v| {
        let x = arena.div(v, r_id);
        let x = crate::transforms::expand::expand(arena, x);
        crate::transforms::eval::eval(arena, x)
    })
}

/// An equation in which the unknown `x` occurs only through linear forms
/// `aᵢ·x + bᵢ` with one common shift `s = bᵢ/aᵢ ≠ 0`
/// (`sin(2x − 1) + cos(2x − 1) + 1/2`, `sinh(x + 1/3) − tanh(x + 1/3)`,
/// `cosh(2x + 1)² + 3·cosh(2x + 1) + 2`, `sin(2x − 1) + sin(4x − 2)`), solved in
/// `u = x + s`, in which every form is `aᵢ·u`, and mapped back by
/// `x = u − s` (a family `u₀ + n·P` becomes `u₀ − s + n·P`).  Before 0.50
/// these were refused while the same equations in `aᵢ·x` were solved: the
/// substitutions look for `sin(a·x)`, `e^(a·x)`, … of the unknown itself.
/// SymPy solves them.
///
/// `u` is real when `x` is declared real; the shift must be real (per the
/// context's [`Params`]), since a shift by
/// `i` would make `u` non-real for a real `x`.  Refused (`NotFound`) for
/// another shape (`x` bare or outside the forms: `x + sin(x + 1)`), for no
/// shift, or for two different shifts (`sin(x + 1) + cos(x + 2)`).  The
/// equation in `u` has no shifted form left, so this step does not repeat.
pub(super) fn try_shifted_argument(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Solutions {
    use crate::base::assumptions::{Assumptions, Props};
    let not_found = || Solutions::refused(Reason::NotFound);
    if !matches!(arena.node(var), ExprNode::Symbol(_)) {
        return not_found();
    }
    let Some(forms) = linear_forms_of(arena, expr, var) else {
        return not_found();
    };
    let mut shift = None;
    let mut rates: Vec<ExprId> = Vec::with_capacity(forms.len());
    for &form in &forms {
        let Some((a, b)) = linear_rate(arena, form, var) else {
            return not_found();
        };
        let s = arena.div(b, a);
        let s = crate::transforms::eval::eval(arena, s);
        match shift {
            None => shift = Some(s),
            Some(t) if t == s => {}
            Some(_) => return not_found(),
        }
        rates.push(a);
    }
    let Some(s) = shift else {
        return not_found();
    };
    if arena.is_zero_structural(s) || expr_contains_var(arena, s, var) || !cx.real(arena, s, var) {
        return not_found();
    }
    let mut declared = Assumptions::default();
    if Domain::declared(arena, var) == Domain::Real {
        declared.assert_true(Props::REAL);
    }
    let u = arena.fresh_internal_symbol("__u_shifted", declared, &[expr]);
    let mut sub = expr;
    for (&form, &a) in forms.iter().zip(&rates) {
        let au = arena.mul(&[a, u]);
        sub = crate::transforms::subs::subs(arena, sub, form, au);
    }
    if expr_contains_var(arena, sub, var) {
        return not_found();
    }
    let sub = crate::transforms::eval::eval(arena, sub);
    let outcome = solve_raw(arena, sub, u, cx);
    // A candidate at which the evaluated equation in `u` is undefined is
    // dropped here: evaluation can expose an infinite factor that the
    // original hides in an unevaluated function (`betainc_regularized(1, 1,
    // −u, ∞) = u + ∞`), where the root of another factor is `∞·0`, and the
    // check against the original (`solve_impl`), whose substitution makes
    // that product `0`, would pass it.  The other candidates are left to that
    // check.
    let outcome = match outcome.into_found() {
        Ok((roots, missing)) => {
            let mut kept = Vec::with_capacity(roots.len());
            for v in roots {
                let at = crate::transforms::subs::subs(arena, sub, u, v);
                let at = crate::transforms::eval::eval(arena, at);
                let undefined = matches!(
                    arena.node(at),
                    ExprNode::NaN
                        | ExprNode::ComplexInfinity
                        | ExprNode::Infinity
                        | ExprNode::NegInfinity
                );
                if !undefined {
                    kept.push(v);
                }
            }
            if kept.is_empty() {
                return not_found();
            }
            Solutions::with_missing(kept, missing)
        }
        Err(other) => other,
    };
    if !(outcome.has_roots()
        || outcome.is_none_proved()
        || outcome.is_nested_periods()
        || outcome.is_conditional())
    {
        return not_found();
    }
    tracing::debug!("solve: solved in the shifted unknown");
    // A reason in words names the unknown it was found for: `x`, not `u`.
    let (u_name, x_name) = (arena.display(u).to_string(), arena.display(var).to_string());
    let renamed = |reason: Reason| match reason {
        Reason::Said(text) if text.contains(&u_name) => {
            Reason::said(text.replace(&u_name, &x_name))
        }
        other => other,
    };
    let outcome = match outcome {
        Solutions::Complete(RootSet::Empty(reason)) => Solutions::none(renamed(reason)),
        Solutions::Partial(set, Missing::Alternative(reason)) => {
            Solutions::Partial(set, Missing::Alternative(renamed(reason)))
        }
        other => other,
    };
    outcome.map_roots(|v| {
        let x = arena.sub(v, s);
        let x = crate::transforms::expand::expand(arena, x);
        crate::transforms::eval::eval(arena, x)
    })
}

/// The maximal sums `a·x + b` (`x = var`, `a` and `b ≠ 0` free of it: an
/// `Add` of terms free of `var`, `var` and products of `var` with factors
/// free of it) through which `var` occurs in `expr`, or `None` when it
/// occurs elsewhere too (bare, or inside no such sum).
fn linear_forms_of(arena: &Arena, expr: ExprId, var: ExprId) -> Option<Vec<ExprId>> {
    let linear_term = |t: ExprId| {
        t == var
            || matches!(arena.node(t), ExprNode::Mul(fs)
                if fs.iter().filter(|&&f| f == var).count() == 1
                    && fs.iter().all(|&f| f == var || !expr_contains_var(arena, f, var)))
    };
    let mut forms = Vec::new();
    let mut stack = vec![expr];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) || !expr_contains_var(arena, id, var) {
            continue;
        }
        if id == var {
            return None;
        }
        if let ExprNode::Add(terms) = arena.node(id)
            && terms.iter().any(|&t| !expr_contains_var(arena, t, var))
            && terms
                .iter()
                .all(|&t| !expr_contains_var(arena, t, var) || linear_term(t))
        {
            forms.push(id);
            continue;
        }
        stack.extend(arena.children(id));
    }
    (!forms.is_empty()).then_some(forms)
}

/// Does `e` hold a function of `t` that is periodic for the variable of the
/// solve: `sin`, `cos`, `tan` always, exponentials and hyperbolic
/// functions for a variable that may be non-real
/// ([`SolveCx::variable_may_be_nonreal`]: [`solve_general_over_c`]).
pub(super) fn has_periodic_atom(arena: &Arena, e: ExprId, t: ExprId, cx: &SolveCx) -> bool {
    let complex = cx.variable_may_be_nonreal(arena, t);
    crate::base::walk::post_order_ids(arena, e)
        .into_iter()
        .any(|id| match *arena.node(id) {
            ExprNode::Sin(a) | ExprNode::Cos(a) | ExprNode::Tan(a) => {
                crate::base::walk::contains(arena, a, t)
            }
            ExprNode::Exp(a) | ExprNode::Sinh(a) | ExprNode::Cosh(a) | ExprNode::Tanh(a) => {
                complex && crate::base::walk::contains(arena, a, t)
            }
            ExprNode::Pow(b, x) => {
                complex
                    && !crate::base::walk::contains(arena, b, t)
                    && crate::base::walk::contains(arena, x, t)
            }
            _ => false,
        })
}

/// Equations in `x` and rational powers `x^(p/q)` of `x` itself
/// (`√x = x − 2`, `x^(2/3) + x^(1/3) = 2`): with `L` the lcm of the
/// denominators and `t = x^(1/L)` (principal), `x^(p/q) = t^(pL/q)` and
/// `x = t^L`, so the equation becomes one in `t` alone; each root `t₀`
/// gives the candidate `x = t₀^L`.  The identity holds only for `t` in the
/// principal sector (`−π/L < arg t ≤ π/L`), so a root outside it yields a
/// spurious `x` (`t = −1` gives `x = 1`, where `√1 = 1 ≠ 1 − 2`); the
/// check of every candidate against the equation removes those.  A root
/// `t₀` with parameters, which that check cannot evaluate, is decided by
/// its argument range ([`principal_power_preimage`]): kept in the sector,
/// dropped outside it, and otherwise kept by a public solver under the
/// exact sector condition ([`principal_power_condition`],
/// [`Reason::Conditional`]; partial up to 0.52, and when the root has the
/// integer parameter of a family) or as a candidate
/// ([`SolveCx::conditional_roots_as_candidates`]).  Before 0.51
/// it was kept: `√x + y + 4 = 0` gave `x = (y + 4)²` (no root for `y = 0`:
/// `√x = −4`), `x^(2/3) = y` gave `±y^(3/2)` (`−y^(3/2)` is no root for `y =
/// 1`: `(−1)^(2/3) = e^(2πi/3)`).
///
/// `None` when `x` occurs other than bare or as the base of a rational
/// power, or when no power has a denominator.
///
/// The radicand may also be one linear `u = a·x + b` (`√(x + 1) = x − 1`,
/// `x − 2√(x − 1) = 0`): `t = u^(1/L)` and `x = (t^L − b)/a`.  SymPy:
/// `solve(sqrt(x + 1) - (x - 1))` → `[3]` (the candidate `0` of the squared
/// equation fails), `solve(x - 2*sqrt(x - 1))` → `[2]`; both were refused
/// before 0.39.
pub(super) fn try_radical_substitution(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Solutions {
    match radical_substitution(arena, expr, var) {
        Some(sub) => sub.solve(arena, expr, var, cx),
        None => Solutions::refused(Reason::NotFound),
    }
}

/// The equation of [`try_radical_substitution`] in `t`, and `x` as a
/// function of `t`.
struct RadicalSubstitution {
    /// The new unknown `t`.
    t: ExprId,
    /// The equation in `t`.
    substituted: ExprId,
    /// `x = t^L`, or `(t^L − b)/a` for the radicand `a·x + b`.
    x_of_t: ExprId,
    /// The radicand is `a·x + b` rather than `x`: the candidates are
    /// checked strictly.
    linear: bool,
    /// `1/L`: `t` is the principal `L`-th root of the radicand.
    index: Q,
}

impl RadicalSubstitution {
    /// The roots in `x` from those of the equation in `t`.
    fn solve(&self, arena: &mut Arena, expr: ExprId, var: ExprId, cx: &SolveCx) -> Solutions {
        let Ok((t_roots, missing)) = solve_impl(arena, self.substituted, self.t, &cx.substituted())
            .without_conditions()
            .into_found()
        else {
            return Solutions::refused(Reason::NotFound);
        };
        let mut out: Vec<ExprId> = Vec::new();
        let mut off_branch = false;
        let mut undecided = false;
        // roots `t₀` in the principal sector only for some parameter values
        let mut guarded: Vec<Guarded> = Vec::new();
        for ts in t_roots {
            let parametric = crate::base::walk::free_symbols(arena, ts)
                .into_iter()
                .any(|s| Some(s) != cx.period());
            let mut cond = None;
            if parametric {
                match principal_power_preimage(arena, ts, &self.index) {
                    crate::base::truth::Truth::True => {}
                    crate::base::truth::Truth::False => {
                        off_branch = true;
                        continue;
                    }
                    crate::base::truth::Truth::Unknown if cx.conditional_roots_as_candidates() => {}
                    crate::base::truth::Truth::Unknown => {
                        let with_period = cx
                            .period()
                            .is_some_and(|p| crate::base::walk::contains(arena, ts, p));
                        match principal_power_condition(arena, ts, &self.index) {
                            Some(c) if !with_period => cond = Some(c),
                            _ => {
                                undecided = true;
                                continue;
                            }
                        }
                    }
                }
            }
            let x0 = crate::transforms::subs::subs(arena, self.x_of_t, self.t, ts);
            let x0 = crate::transforms::expand::expand(arena, x0);
            let x0 = crate::transforms::eval::eval(arena, x0);
            if let Some(cond) = cond {
                guarded.push(Guarded {
                    cond,
                    roots: RootSet::Points(vec![x0]),
                });
            } else if !out.contains(&x0) {
                out.push(x0);
            }
        }
        // no other root: every root in `t` was taken
        let only_guarded = out.is_empty() && missing.is_none() && !undecided;
        let outcome = self.unconditional(arena, expr, var, out, missing, off_branch, undecided);
        if guarded.is_empty() {
            return outcome;
        }
        let outcome = if only_guarded {
            Solutions::none(Reason::NoPreimage)
        } else {
            outcome
        };
        outcome.union(Solutions::refused(Reason::Conditional(guarded)))
    }

    /// The outcome of the roots `out` that are principal for every value of
    /// the parameters (see [`solve`](Self::solve)).
    #[allow(clippy::too_many_arguments)]
    fn unconditional(
        &self,
        arena: &mut Arena,
        expr: ExprId,
        var: ExprId,
        out: Vec<ExprId>,
        missing: Option<Missing>,
        off_branch: bool,
        undecided: bool,
    ) -> Solutions {
        let missing = if undecided {
            Missing::merged(
                missing,
                Some(Missing::Alternative(Reason::said(
                    "a root of the equation in the radical is a principal root only for some \
                     values of the parameters",
                ))),
            )
        } else {
            missing
        };
        if out.is_empty() && missing.is_none() && off_branch {
            return Solutions::none(Reason::said(
                "no root of the equation in the radical is a principal root",
            ));
        }
        if self.linear {
            return match strictly_checked(arena, expr, var, out) {
                Some(roots) => Solutions::with_missing(roots, missing),
                None => Solutions::refused(Reason::NotFound),
            };
        }
        Solutions::with_missing(out, missing)
    }
}

/// The substitution of [`try_radical_substitution`], or `None` for another
/// shape.
fn radical_substitution(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
) -> Option<RadicalSubstitution> {
    let post = crate::base::walk::post_order_ids(arena, expr);
    // The one base of the fractional powers.
    let mut radicand: Option<ExprId> = None;
    for &id in &post {
        if let ExprNode::Pow(b, e) = *arena.node(id)
            && expr_contains_var(arena, b, var)
        {
            let r = arena.as_num(e)?;
            if !r.is_integer() {
                match radicand {
                    None => radicand = Some(b),
                    Some(u) if u == b => {}
                    Some(_) => return None,
                }
            }
        }
    }
    let u = radicand?;
    let rate = if u == var {
        None
    } else {
        Some(linear_rate(arena, u, var)?)
    };
    let mut lcm = BigInt::one();
    let mut powers: Vec<(ExprId, Q)> = Vec::new();
    for &id in &post {
        if !expr_contains_var(arena, id, var) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Pow(b, e) if *b == u => {
                let r = arena.as_num(*e)?.clone();
                if !r.is_integer() {
                    lcm = num_integer::Integer::lcm(&lcm, r.denom());
                }
                powers.push((id, r));
            }
            // `x` elsewhere must be bare: a power's exponent or a function
            // argument that is not `x` or `x^(p/q)` is not handled here.
            ExprNode::Pow(_, e) if expr_contains_var(arena, *e, var) => return None,
            ExprNode::Symbol(_)
            | ExprNode::Add(_)
            | ExprNode::Mul(_)
            | ExprNode::Neg(_)
            | ExprNode::Pow(..) => {}
            _ => return None,
        }
    }
    if lcm.is_one() || lcm > BigInt::from(MAX_RADICAL_INDEX) {
        return None;
    }
    let t = arena.internal_symbol("__t_radical");
    let l_id = arena.big_int(lcm.clone());
    let mut substituted = expr;
    for (pow_id, r) in &powers {
        let k = r * Ratio::from_integer(lcm.clone());
        let k_id = arena.num_ratio(k);
        let tk = arena.pow(t, k_id);
        substituted = arena.subs_structural(substituted, *pow_id, tk);
    }
    let t_l = arena.pow(t, l_id);
    // x = t^L, or (t^L − b)/a for the radicand a·x + b
    let x_of_t = match rate {
        None => t_l,
        Some((a, b)) => {
            let num = arena.sub(t_l, b);
            arena.div(num, a)
        }
    };
    let index = Ratio::new(BigInt::one(), lcm);
    substituted = crate::transforms::subs::subs(arena, substituted, var, x_of_t);
    if rate.is_some() {
        substituted = crate::transforms::expand::expand(arena, substituted);
    }
    let substituted = crate::transforms::eval::eval(arena, substituted);
    if expr_contains_var(arena, substituted, var) {
        return None;
    }
    Some(RadicalSubstitution {
        t,
        substituted,
        x_of_t,
        linear: rate.is_some(),
        index,
    })
}

/// Largest root index `L` of [`try_radical_substitution`].
const MAX_RADICAL_INDEX: i64 = 12;

/// Equations polynomial in `x` and in the square roots `√A`, `√B` of one
/// or two polynomials (`√(x + 5) + √x = 3`, `√(x² + 1) = x + 2`), as
/// SymPy's `unrad`: with `s = √A`, `r = √B` the equation is
/// `P = a + b·s + c·r + d·s·r = 0` (`a, b, c, d` polynomials, `s² = A`,
/// `r² = B`), and every solution is a root of the norm
/// `P(s, r)·P(−s, −r)·P(−s, r)·P(s, −r) = E² − F²·A·B` with
/// `E = a² + d²AB − b²A − c²B`, `F = 2(ad − bc)`.  Its roots are the
/// candidates; those of the other sign choices fail the equation and are
/// removed by the check of every candidate.  SymPy:
/// `solve(sqrt(x + 5) + sqrt(x) - 3)` → `[4/9]`; refused before 0.39.
pub(super) fn try_square_root_elimination(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
) -> Option<Unradicated> {
    // The radicands, and the half-integer powers of them.
    let mut radicands: Vec<ExprId> = Vec::new();
    for id in crate::base::walk::post_order_ids(arena, expr) {
        if !expr_contains_var(arena, id, var) {
            continue;
        }
        match arena.node(id).clone() {
            ExprNode::Pow(b, e) => {
                let r = arena.as_num(e)?.clone();
                if r.is_integer() {
                    continue;
                }
                if *r.denom() != BigInt::from(2) || expr_contains_var(arena, e, var) {
                    return None;
                }
                polybridge::expr_to_poly(arena, b, var)?;
                if !radicands.contains(&b) {
                    radicands.push(b);
                }
            }
            ExprNode::Symbol(_) | ExprNode::Add(_) | ExprNode::Mul(_) | ExprNode::Neg(_) => {}
            _ => return None,
        }
    }
    if radicands.len() > 2 && radicands.len() <= MAX_SQRT_RADICANDS {
        return eliminate_many_square_roots(arena, expr, var, &radicands);
    }
    if radicands.is_empty() || radicands.len() > 2 {
        return None;
    }
    // s, r stand for √A, √B; A^(k/2) = A^((k−1)/2)·s for odd k.
    let s = arena.symbol("__s_unrad");
    let r = arena.symbol("__r_unrad");
    let syms = [s, r];
    let mut replaced = expr;
    for id in crate::base::walk::post_order_ids(arena, expr) {
        let ExprNode::Pow(b, e) = *arena.node(id) else {
            continue;
        };
        let Some(k) = arena.as_num(e).cloned() else {
            continue;
        };
        let Some(which) = radicands.iter().position(|&u| u == b) else {
            continue;
        };
        if k.is_integer() {
            continue;
        }
        // k = m + 1/2 with m = floor(k)
        let m = k.floor();
        let m_id = arena.num_ratio(m);
        let bm = arena.pow(b, m_id);
        let repl = arena.mul(&[bm, syms[which]]);
        replaced = arena.subs_structural(replaced, id, repl);
    }
    let replaced = crate::transforms::expand::expand(arena, replaced);
    let replaced = crate::transforms::eval::eval(arena, replaced);
    let (a_rad, b_rad) = (
        radicands[0],
        radicands.get(1).copied().unwrap_or(arena.zero),
    );
    // Coefficients of 1, s, r, s·r.
    let by_s = symbolic_poly_coeffs(arena, replaced, s)?;
    if by_s.len() > 2 {
        return None;
    }
    let zero = arena.zero;
    let s0 = by_s.first().copied().unwrap_or(zero);
    let s1 = by_s.get(1).copied().unwrap_or(zero);
    let split_r = |arena: &mut Arena, e: ExprId| -> Option<(ExprId, ExprId)> {
        let c = symbolic_poly_coeffs(arena, e, r)?;
        if c.len() > 2 {
            return None;
        }
        Some((
            c.first().copied().unwrap_or(zero),
            c.get(1).copied().unwrap_or(zero),
        ))
    };
    let (a, c) = split_r(arena, s0)?;
    let (b, d) = split_r(arena, s1)?;
    for &coef in &[a, b, c, d] {
        if crate::base::walk::contains(arena, coef, s)
            || crate::base::walk::contains(arena, coef, r)
        {
            return None;
        }
    }
    // E = a² + d²AB − b²A − c²B, F = 2(ad − bc), N = E² − F²AB
    let two = arena.int(2);
    let sq = |arena: &mut Arena, e: ExprId| arena.pow(e, two);
    let ab = arena.mul(&[a_rad, b_rad]);
    let a2 = sq(arena, a);
    let d2 = sq(arena, d);
    let b2 = sq(arena, b);
    let c2 = sq(arena, c);
    let d2ab = arena.mul(&[d2, ab]);
    let b2a = arena.mul(&[b2, a_rad]);
    let c2b = arena.mul(&[c2, b_rad]);
    let neg_b2a = arena.neg(b2a);
    let neg_c2b = arena.neg(c2b);
    let e_big = arena.add(&[a2, d2ab, neg_b2a, neg_c2b]);
    let ad = arena.mul(&[a, d]);
    let bc = arena.mul(&[b, c]);
    let ad_bc = arena.sub(ad, bc);
    let f_big = arena.mul(&[two, ad_bc]);
    let e2 = sq(arena, e_big);
    let f2 = sq(arena, f_big);
    let f2ab = arena.mul(&[f2, ab]);
    let norm = arena.sub(e2, f2ab);
    let norm = crate::transforms::expand::expand(arena, norm);
    let norm = crate::transforms::eval::eval(arena, norm);
    let poly = polybridge::expr_to_poly(arena, norm, var)?;
    if poly.is_constant() {
        if !poly.is_zero() {
            return Some(Unradicated::NormConstant);
        }
        return None;
    }
    let candidates = values(solve_rational_poly(arena, var, &poly));
    strictly_checked(arena, expr, var, candidates).map(Unradicated::Roots)
}

/// What [`try_square_root_elimination`] found.
pub(super) enum Unradicated {
    /// The candidates that certainly solve the equation (possibly none:
    /// not taken as a proof).
    Roots(Vec<ExprId>),
    /// The norm of the equation over the sign choices is a nonzero
    /// constant: the equation has no root.  (Up to 0.48 the thread-local
    /// flag `SQRT_NORM_CONSTANT` next to an empty list.)
    NormConstant,
}

/// Most square roots of polynomials eliminated by
/// [`eliminate_many_square_roots`].
const MAX_SQRT_RADICANDS: usize = 4;

/// Largest degree of the norm of [`eliminate_many_square_roots`].
const MAX_SQRT_NORM_DEGREE: usize = 64;

/// [`try_square_root_elimination`] for three or four radicands `Aᵢ`: with
/// `sᵢ = √Aᵢ` the equation is a polynomial `P` in `x` and the `sᵢ`, of
/// degree one in each after `sᵢ² = Aᵢ`; writing `P = P₀ + P₁·sₖ`, the
/// product with its conjugate `P₀ − P₁·sₖ` is `P₀² − P₁²·Aₖ`, free of `sₖ`
/// — repeated for every `sᵢ`, the norm over all sign choices, a polynomial
/// in `x` that every solution is a root of.  Its roots are the candidates,
/// each checked strictly.  `√(2x + 9) = √(x + 1) + √(x + 4)` gives `0`
/// (refused before 0.47; SymPy: `[0]`).
fn eliminate_many_square_roots(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    radicands: &[ExprId],
) -> Option<Unradicated> {
    let syms: Vec<ExprId> = (0..radicands.len())
        .map(|i| arena.symbol(&format!("__s_unrad_{i}")))
        .collect();
    let mut replaced = expr;
    for id in crate::base::walk::post_order_ids(arena, expr) {
        let ExprNode::Pow(b, e) = *arena.node(id) else {
            continue;
        };
        let Some(k) = arena.as_num(e).cloned() else {
            continue;
        };
        let Some(which) = radicands.iter().position(|&u| u == b) else {
            continue;
        };
        if k.is_integer() {
            continue;
        }
        let m = k.floor();
        let m_id = arena.num_ratio(m);
        let bm = arena.pow(b, m_id);
        let repl = arena.mul(&[bm, syms[which]]);
        replaced = arena.subs_structural(replaced, id, repl);
    }
    // `p` with every `sⱼ^m` reduced by `sⱼ² = Aⱼ` (degree ≤ 1 in each).
    let reduce = |arena: &mut Arena, p: ExprId| -> Option<ExprId> {
        let mut p = crate::transforms::expand::expand(arena, p);
        for (j, &s) in syms.iter().enumerate() {
            if !crate::base::walk::contains(arena, p, s) {
                continue;
            }
            let coeffs = symbolic_poly_coeffs(arena, p, s)?;
            let mut terms: Vec<ExprId> = Vec::with_capacity(coeffs.len());
            for (m, &c) in coeffs.iter().enumerate() {
                let half = arena.int((m / 2) as i64);
                let am = arena.pow(radicands[j], half);
                let t = if m % 2 == 1 {
                    arena.mul(&[c, am, s])
                } else {
                    arena.mul(&[c, am])
                };
                terms.push(t);
            }
            let sum = arena.add(&terms);
            p = crate::transforms::expand::expand(arena, sum);
        }
        Some(crate::transforms::eval::eval(arena, p))
    };
    let mut p = reduce(arena, replaced)?;
    for (i, &s) in syms.iter().enumerate().rev() {
        let coeffs = symbolic_poly_coeffs(arena, p, s)?;
        if coeffs.len() > 2 {
            return None;
        }
        let zero = arena.zero;
        let p0 = coeffs.first().copied().unwrap_or(zero);
        let p1 = coeffs.get(1).copied().unwrap_or(zero);
        let two = arena.int(2);
        let p0_2 = arena.pow(p0, two);
        let p1_2 = arena.pow(p1, two);
        let p1_2a = arena.mul(&[p1_2, radicands[i]]);
        let n = arena.sub(p0_2, p1_2a);
        p = reduce(arena, n)?;
        if crate::base::walk::contains(arena, p, s) {
            return None;
        }
    }
    let poly = polybridge::expr_to_poly(arena, p, var)?;
    if poly.is_constant() && !poly.is_zero() {
        return Some(Unradicated::NormConstant);
    }
    if poly.is_constant() || poly.degree().unwrap_or(0) > MAX_SQRT_NORM_DEGREE {
        return None;
    }
    let candidates = values(solve_rational_poly(arena, var, &poly));
    strictly_checked(arena, expr, var, candidates).map(Unradicated::Roots)
}

/// The candidates that certainly solve `expr = 0` (whose only symbol is
/// `var`), or `None` when one of them can be decided neither way: the
/// candidates of squaring away radicals include roots of the other sign
/// choices, and one whose residual sits on a branch cut (`√` of a negative
/// number written in nested radicals) must not be returned unverified.
/// (No candidate left is not taken as a proof that there is no root.)
fn strictly_checked(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    candidates: Vec<ExprId>,
) -> Option<Vec<ExprId>> {
    if crate::base::walk::free_symbols(arena, expr)
        .iter()
        .any(|&s| s != var)
    {
        return None;
    }
    let mut out = Vec::new();
    for c in candidates {
        if member_is_root(arena, expr, var, c) {
            out.push(c);
        } else if !certainly_not_a_root(arena, expr, var, c) {
            return None;
        }
    }
    Some(out)
}
