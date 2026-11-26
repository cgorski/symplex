//! nth-order reducible ODE: F(y, y', y'') = 0, no explicit x (part of [`ode`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// nth-order reducible ODE: F(y, y', y'') = 0, no explicit x
// ═══════════════════════════════════════════════════════════════════════════

/// Solve a second-order ODE where the independent variable doesn't appear:
///   F(y, y', y'') = 0
///
/// Substitution: p = y', y'' = p·dp/dy reduces to a first-order ODE in p(y).
pub(super) fn try_nth_order_reducible(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    _func_sym: SymbolId,
    var_sym: SymbolId,
) -> Option<OdeResult> {
    tracing::debug!("ode: trying nth-order reducible (missing x)");

    let dy_dx = arena.intern(ExprNode::Derivative(func, var));
    let d2y_dx2 = arena.intern(ExprNode::Derivative(dy_dx, var));

    // Must have y'' present.
    if !expr_contains(arena, expr, d2y_dx2) {
        return None;
    }

    // The independent variable x must NOT appear explicitly (only through
    // y and its derivatives).
    // We check: after removing derivative nodes, does x appear?
    // Strategy: substitute y''→__d2, y'→__d1, then check if var_sym remains.
    let d2_placeholder = arena.internal_symbol("__d2");
    let d1_placeholder = arena.internal_symbol("__d1");
    let stripped = crate::transforms::subs::subs(arena, expr, d2y_dx2, d2_placeholder);
    let stripped = crate::transforms::subs::subs(arena, stripped, dy_dx, d1_placeholder);
    let has_x = contains_sym(arena, stripped, var_sym);
    let has_y = contains_sym(arena, stripped, _func_sym);
    if has_x && has_y {
        return None; // neither reduction applies
    }
    let p = arena.internal_symbol("__p");
    let p_sym = match arena.node(p) {
        ExprNode::Symbol(s) => *s,
        _ => return None,
    };
    let c2 = arena.symbol("C2");

    // The reduced first-order solution `p = f(·, C1)` must be explicit and
    // carry its constant.  Before, an implicit one (still in `p`, and
    // without its constant) was integrated as if it were `p`: `y″ + 2y′²
    // = 0` gave `−1/4·ln(−2·__p²·(C2 + x))`.
    let usable = |arena: &Arena, r: &OdeResult| {
        !contains_sym(arena, r.solution, p_sym)
            && r.constants.len() == 1
            && expr_contains(arena, r.solution, r.constants[0])
            && !crate::base::walk::has_unevaluated(arena, r.solution)
    };

    // F(x, y′, y″) = 0 (no y): p(x) = y′ gives F(x, p, p′) = 0, and
    // y = ∫ p dx + C2 is explicit.
    if !has_y {
        let dp_dx = arena.intern(ExprNode::Derivative(p, var));
        let reduced = crate::transforms::subs::subs(arena, expr, d2y_dx2, dp_dx);
        let reduced = crate::transforms::subs::subs(arena, reduced, dy_dx, p);
        let reduced = crate::transforms::eval::eval(arena, reduced);
        if let Some(pr) = dsolve(arena, reduced, p, var)
            && usable(arena, &pr)
        {
            // One family of `y` per family of `p` (`y″ = y′³`: `p = ±(C1 −
            // 2x)^(−1/2)`; before 0.30 only the first).
            let antiderivative = |arena: &mut Arena, p_family: ExprId| {
                let integral = crate::transforms::integrate::integrate(arena, p_family, var);
                (!integral_failed(arena, integral)).then(|| {
                    let s = arena.add(&[integral, c2]);
                    crate::transforms::eval::eval(arena, s)
                })
            };
            if let Some(solution) = antiderivative(arena, pr.solution) {
                let mut branches = Vec::new();
                for &b in &pr.branches {
                    match antiderivative(arena, b) {
                        Some(y_b) => branches.push(y_b),
                        None => tracing::debug!("ode: a family of y' has no closed antiderivative"),
                    }
                }
                return Some(OdeResult {
                    solution,
                    constants: vec![pr.constants[0], c2],
                    branches,
                });
            }
        }
        if has_x {
            return None;
        }
    }

    // Now perform the reduction: let p = dy/dx, then d²y/dx² = p·dp/dy
    let dp_dy = arena.intern(ExprNode::Derivative(p, func));
    let p_dp_dy = arena.mul(&[p, dp_dy]);

    // Substitute: y'' → p·dp/dy,  y' → p
    let reduced = crate::transforms::subs::subs(arena, expr, d2y_dx2, p_dp_dy);
    let reduced = crate::transforms::subs::subs(arena, reduced, dy_dx, p);
    let reduced = crate::transforms::eval::eval(arena, reduced);
    // A factor `p` common to every term is divided out (it only carries
    // the constant solutions `y′ = 0`): `y·y″ − y′²` becomes `y·p′ − p`,
    // linear in `p(y)`.
    let reduced = divide_common_factor(arena, reduced, p).unwrap_or(reduced);

    // Now `reduced` is a first-order ODE in p(y) with independent var = y.
    let p_result = dsolve(arena, reduced, p, func)?;
    if !usable(arena, &p_result) {
        return None;
    }

    // p_result.solution gives p = f(y, C1).
    // Now solve dy/dx = p(y) — this is separable: ∫ dy/p(y) = x + C2.
    let p_sol = p_result.solution;

    // Set up: dy/dx - p_sol = 0  →  ∫ 1/p_sol dy = x + C2
    // We need to integrate 1/p_sol w.r.t. y.
    let neg_one_id = arena.int(-1);
    let inv_p = arena.pow(p_sol, neg_one_id);
    let inv_p = crate::transforms::eval::eval(arena, inv_p);
    let lhs_integral = crate::transforms::integrate::integrate(arena, inv_p, func);

    if integral_failed(arena, lhs_integral) {
        return None; // Can't integrate 1/p(y)
    }

    // Implicit solution: ∫ dy/p(y) = x + C2
    let rhs = arena.add(&[var, c2]);
    let implicit = arena.sub(lhs_integral, rhs);
    let implicit = crate::transforms::eval::eval(arena, implicit);

    // Try to solve explicitly for y (several roots: see `try_exact_ode`).
    let solutions = known_roots(arena, implicit, func);
    if solutions.len() == 1 {
        let sol = crate::transforms::eval::eval(arena, solutions[0].value);
        let mut constants = p_result.constants;
        constants.push(c2);
        return Some(OdeResult {
            solution: sol,
            constants,
            branches: Vec::new(),
        });
    }

    // Return implicit form.
    let mut constants = p_result.constants;
    constants.push(c2);
    Some(OdeResult {
        solution: implicit,
        constants,
        branches: Vec::new(),
    })
}

/// Does `sym` occur free in `expr`?  Bound occurrences (a `Sum` index, a
/// `RootOf` variable; see `walk::binder`) do not count.  Up to 0.28 this
/// was a structural walk with no visited set, exponential on a DAG with
/// shared sub-expressions.
pub(super) fn contains_sym(arena: &Arena, expr: ExprId, sym: SymbolId) -> bool {
    crate::base::walk::has_free_symbol(arena, expr, sym)
}

/// `∫ e d(var)`, trying equivalent forms of `e` until one integrates: as
/// given, with products of exponentials merged (`powsimp`: variation of
/// parameters builds `x·e⁻ˣ·e⁻ˣ/x²/e⁻²ˣ`, which is `1/x`), as one
/// fraction (`ratsimp`), expanded, and with the common factor of every sum
/// taken out (`factor_terms`: `∫ dv/(v·ln v − v)` is not found, `∫ dv/(v·(ln v
/// − 1)) = ln|ln v − 1|` is; the homogeneous equation `y′ = (y/x)·ln(y/x)`
/// was refused).  The first attempt's result (with its formal `Integral`)
/// when none does.
pub(super) fn integrate_forms(arena: &mut Arena, e: ExprId, var: ExprId) -> ExprId {
    let first = crate::transforms::integrate::integrate(arena, e, var);
    if !integral_failed(arena, first) {
        return first;
    }
    let merged = crate::simplify::powsimp::powsimp(arena, e);
    let merged = crate::transforms::eval::eval(arena, merged);
    let fraction = crate::simplify::ratsimp::ratsimp(arena, merged);
    let fraction = crate::transforms::eval::eval(arena, fraction);
    let expanded = crate::transforms::expand::expand(arena, merged);
    let expanded = crate::transforms::eval::eval(arena, expanded);
    let factored = {
        let post = crate::base::walk::post_order_ids(arena, merged);
        let mut cache: rustc_hash::FxHashMap<ExprId, ExprId> = rustc_hash::FxHashMap::default();
        for &id in &post {
            let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
            let new = if matches!(arena.node(rebuilt), ExprNode::Add(_)) {
                crate::simplify::factor_terms::factor_terms(arena, rebuilt)
            } else {
                rebuilt
            };
            cache.insert(id, new);
        }
        let f = cache.get(&merged).copied().unwrap_or(merged);
        crate::transforms::eval::eval(arena, f)
    };
    for form in [merged, fraction, expanded, factored] {
        if form == e {
            continue;
        }
        let r = crate::transforms::integrate::integrate(arena, form, var);
        if !integral_failed(arena, r) {
            return r;
        }
    }
    if let Some(r) = integrate_trig_as_exp(arena, merged, var) {
        return r;
    }
    first
}

/// `∫ e d(var)` with `sin`/`cos` of linear arguments written as complex
/// exponentials (and the result's complex exponentials back as
/// `cos + i·sin`): `∫ eˣ·cos x/x dx = (Ei((1 + i)x) + Ei((1 − i)x))/2`,
/// which the integrator does not find in the trigonometric form (the
/// linear equation `y′ + y = cos x/x` was left with a formal integral).
/// `None` when `e` has no such factor, has other trigonometric functions,
/// or the exponential form does not integrate either.
fn integrate_trig_as_exp(arena: &mut Arena, e: ExprId, var: ExprId) -> Option<ExprId> {
    let mut has_circular = false;
    for id in crate::base::walk::post_order_ids(arena, e) {
        match arena.node(id).clone() {
            ExprNode::Sin(a) | ExprNode::Cos(a) => {
                let c = crate::transforms::solve::symbolic_poly_coeffs(arena, a, var)?;
                if c.len() > 2 {
                    return None;
                }
                has_circular = true;
            }
            ExprNode::Tan(_) => {
                return None;
            }
            _ => {}
        }
    }
    if !has_circular {
        return None;
    }
    let x = crate::simplify::rewrite::rewrite_as_exp(arena, e);
    let x = crate::transforms::expand::expand(arena, x);
    let x = crate::transforms::eval::eval(arena, x);
    let x = crate::simplify::powsimp::powsimp(arena, x);
    let x = crate::transforms::eval::eval(arena, x);
    let r = crate::transforms::integrate::integrate(arena, x, var);
    if integral_failed(arena, r) {
        return None;
    }
    let r = complex_exp_as_trig(arena, r, var);
    let r = crate::transforms::expand::expand(arena, r);
    Some(crate::transforms::eval::eval(arena, r))
}

/// `e^{(a + i·b)·x + c + i·d} → e^{a·x + c}·(cos(b·x + d) + i·sin(b·x + d))`
/// for every exponential whose argument is linear in `var` with constant
/// coefficients (real parts `a`, `c` and imaginary parts `b`, `d` from
/// [`real_imag_parts`]); other exponentials, and constant ones, are kept
/// (`rewrite_as_trig` turned the constant `e^{−1 − i}` into
/// `cos(i − 1) + i·sin(i − 1)`).
pub(super) fn complex_exp_as_trig(arena: &mut Arena, e: ExprId, var: ExprId) -> ExprId {
    let i_unit = arena.i_unit;
    let post = crate::base::walk::post_order_ids(arena, e);
    let mut cache: rustc_hash::FxHashMap<ExprId, ExprId> = rustc_hash::FxHashMap::default();
    for &id in &post {
        let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
        let new = match *arena.node(rebuilt) {
            ExprNode::Exp(arg)
                if contains_sym_id(arena, arg, var)
                    && crate::base::walk::contains(arena, arg, i_unit) =>
            {
                exp_linear_as_trig(arena, arg, var).unwrap_or(rebuilt)
            }
            _ => rebuilt,
        };
        cache.insert(id, new);
    }
    let out = cache.get(&e).copied().unwrap_or(e);
    crate::transforms::eval::eval(arena, out)
}

/// The rewrite of [`complex_exp_as_trig`] for one exponential `e^{arg}`.
fn exp_linear_as_trig(arena: &mut Arena, arg: ExprId, var: ExprId) -> Option<ExprId> {
    let coeffs = crate::transforms::solve::symbolic_poly_coeffs(arena, arg, var)?;
    if coeffs.len() != 2 {
        return None;
    }
    let mut re = Vec::with_capacity(2);
    let mut im = Vec::with_capacity(2);
    for (k, &c) in coeffs.iter().enumerate() {
        if !crate::base::walk::free_symbols(arena, c).is_empty() {
            return None;
        }
        let (r, i) = real_imag_parts(arena, c);
        if !real_radical_form(arena, r) || !real_radical_form(arena, i) {
            return None;
        }
        let (r, i) = if k == 1 {
            (arena.mul(&[r, var]), arena.mul(&[i, var]))
        } else {
            (r, i)
        };
        re.push(r);
        im.push(i);
    }
    let re = arena.add(&re);
    let re = crate::transforms::eval::eval(arena, re);
    let im = arena.add(&im);
    let im = crate::transforms::eval::eval(arena, im);
    let envelope = arena.exp(re);
    let cos = arena.cos(im);
    let sin = arena.sin(im);
    let i_unit = arena.i_unit;
    let i_sin = arena.mul(&[i_unit, sin]);
    let osc = arena.add(&[cos, i_sin]);
    Some(arena.mul(&[envelope, osc]))
}

/// Is `e` zero as a rational function of its symbols and opaque
/// subexpressions (`ratsimp`)?  Catches `∂M/∂y − ∂N/∂x` of quotients
/// that `expand` leaves as unreduced fractions.
pub(super) fn rational_zero(arena: &mut Arena, e: ExprId) -> bool {
    let r = crate::simplify::ratsimp::ratsimp(arena, e);
    let r = crate::transforms::eval::eval(arena, r);
    arena.is_zero_structural(r)
}

/// `expr / p` when `p` is a factor of every term of `expand(expr)`
/// (`None` otherwise).
fn divide_common_factor(arena: &mut Arena, expr: ExprId, p: ExprId) -> Option<ExprId> {
    let expanded = crate::transforms::expand::expand(arena, expr);
    let terms: Vec<ExprId> = match arena.node(expanded) {
        ExprNode::Add(c) => c.to_vec(),
        _ => vec![expanded],
    };
    let has_factor = |arena: &Arena, t: ExprId| -> bool {
        let is_power_of_p = |f: ExprId| -> bool {
            f == p
                || matches!(arena.node(f), ExprNode::Pow(b, e) if *b == p
                    && arena.as_num(*e).is_some_and(|r| r.is_integer() && r.is_positive()))
        };
        match arena.node(t) {
            ExprNode::Mul(fs) => fs.iter().any(|&f| is_power_of_p(f)),
            ExprNode::Neg(inner) => match arena.node(*inner) {
                ExprNode::Mul(fs) => fs.iter().any(|&f| is_power_of_p(f)),
                _ => is_power_of_p(*inner),
            },
            _ => is_power_of_p(t),
        }
    };
    if terms.is_empty() || !terms.iter().all(|&t| has_factor(arena, t)) {
        return None;
    }
    let quotient = arena.div(expanded, p);
    let quotient = crate::transforms::eval::eval(arena, quotient);
    let quotient = crate::transforms::expand::expand(arena, quotient);
    Some(crate::transforms::eval::eval(arena, quotient))
}

/// Did an antiderivative fail anywhere in `id`?  The integrator returns a
/// formal `Integral` node for the part it cannot do, not necessarily at
/// the top: `∫ 2·sin(y)/cos(y) dy` came back as `2·Integral(…)`, which
/// the top-level checks let through into an integrating factor (and a
/// `Piecewise` of nested integrals was returned as the "solution").
pub(super) fn integral_failed(arena: &Arena, id: ExprId) -> bool {
    let mut stack = vec![id];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        let node = arena.node(n);
        if matches!(node, ExprNode::Integral(..)) {
            return true;
        }
        node.for_each_child(|c| stack.push(c));
    }
    false
}

/// `exp(Σ cᵢ·ln(fᵢ))` → `Π fᵢ^{cᵢ}` for integrating factors.
///
/// Integrating factors are only needed up to a constant factor, so
/// `ln|f|` is treated as `ln f` (the sign is absorbed into the constant
/// of integration).  Terms that are not logarithms stay inside `exp`.
pub(super) fn exp_of_log_sum(arena: &mut Arena, integral: ExprId) -> ExprId {
    let terms: Vec<ExprId> = match arena.node(integral).clone() {
        ExprNode::Add(c) => c.to_vec(),
        _ => vec![integral],
    };
    let mut factors: Vec<ExprId> = Vec::new();
    let mut leftover: Vec<ExprId> = Vec::new();
    for t in terms {
        let (coeff, term) = arena.as_coeff_term(t);
        let inner = match arena.node(term).clone() {
            ExprNode::Ln(inner) => Some(inner),
            _ => None,
        };
        match inner {
            Some(inner) => {
                let base = match arena.node(inner).clone() {
                    ExprNode::Abs(a) => a,
                    _ => inner,
                };
                if coeff.is_one() {
                    factors.push(base);
                } else {
                    let c_id = ode_ratio_to_expr(arena, &coeff);
                    factors.push(arena.pow(base, c_id));
                }
            }
            None => leftover.push(t),
        }
    }
    if factors.is_empty() {
        return arena.exp(integral);
    }
    if !leftover.is_empty() {
        let rest = arena.add(&leftover);
        factors.push(arena.exp(rest));
    }
    let prod = arena.mul(&factors);
    crate::transforms::eval::eval(arena, prod)
}
