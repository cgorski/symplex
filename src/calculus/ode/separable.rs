//! Full separable: y' = f(x) * g(y) (part of [`ode`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Full separable: y' = f(x) * g(y)
// ═══════════════════════════════════════════════════════════════════════════

/// Solve y' = f(x)*g(y) via separation of variables.
///
/// Algorithm:
/// 1. Extract dy/dx from the ODE expression
/// 2. Collect remaining terms as RHS (negated)
/// 3. If RHS factors into x-only × y-only parts, separate them
/// 4. For the common case g(y) = y: solution is y = C1·exp(∫f(x)dx)
/// 5. Otherwise: implicit solution ∫(1/g(y))dy = ∫f(x)dx + C1
pub(super) fn try_full_separable(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
    var_sym: SymbolId,
) -> Option<OdeResult> {
    tracing::debug!("ode: trying full separable");

    let dy_dx = arena.intern(ExprNode::Derivative(func, var));

    // Extract dy/dx term and collect the rest as RHS
    let children = match arena.node(expr).clone() {
        ExprNode::Add(children) => children,
        _ => return None,
    };

    let mut has_dy = false;
    let mut dy_coeff = num_rational::Ratio::<num_bigint::BigInt>::zero();
    let mut other_terms: Vec<ExprId> = Vec::new();

    for &child in &children {
        let (coeff, term) = arena.as_coeff_term(child);
        if term == dy_dx {
            has_dy = true;
            dy_coeff += coeff;
        } else {
            other_terms.push(child);
        }
    }

    use num_traits::Zero;
    if !has_dy || dy_coeff.is_zero() {
        return None;
    }

    // We need coefficient on y' to be 1 (or normalize)
    if !dy_coeff.is_one() {
        // Normalize: divide everything else by dy_coeff
        let inv_coeff = num_rational::Ratio::<num_bigint::BigInt>::one() / &dy_coeff;
        let inv_id = {
            let nid = arena.intern_num(inv_coeff);
            arena.intern(ExprNode::Num(nid))
        };
        let mut scaled = Vec::new();
        for &t in &other_terms {
            scaled.push(arena.mul(&[inv_id, t]));
        }
        other_terms = scaled;
    }

    // RHS = -(other_terms), i.e., y' = -other_terms
    let rhs = if other_terms.is_empty() {
        return None; // y' = 0 is handled by simple separable
    } else {
        let sum = arena.add(&other_terms);
        arena.neg(sum)
    };

    // Now we need: rhs = f(x) * g(y) where f depends only on var and g only on func
    // Check if rhs depends on func at all — if not, this is simple separable
    if !contains_sym(arena, rhs, func_sym) {
        return None; // Let simple separable handle it
    }

    // Try to factor rhs into x-only and y-only parts; `exp(a + b)` is
    // `exp(a)·exp(b)` (`y′ = e^{x+y}` was refused; SymPy gives
    // `−ln(C1 − eˣ)`).
    let mut factors = Vec::new();
    for f in collect_mul_factors(arena, rhs) {
        match arena.node(f).clone() {
            ExprNode::Exp(arg) if matches!(arena.node(arg), ExprNode::Add(_)) => {
                let ExprNode::Add(terms) = arena.node(arg).clone() else {
                    continue;
                };
                for &t in terms.iter() {
                    factors.push(arena.exp(t));
                }
            }
            _ => factors.push(f),
        }
    }

    let mut x_factors: Vec<ExprId> = Vec::new();
    let mut y_factors: Vec<ExprId> = Vec::new();

    for &factor in &factors {
        let has_x = contains_sym(arena, factor, var_sym);
        let has_y = contains_sym(arena, factor, func_sym);

        if has_x && has_y {
            // Factor depends on both x and y — cannot separate
            return None;
        } else if has_y {
            y_factors.push(factor);
        } else {
            // Pure x-factor or constant
            x_factors.push(factor);
        }
    }

    if y_factors.is_empty() {
        return None; // No y dependence — let simple separable handle it
    }

    // f(x) = product of x_factors
    let f_x = if x_factors.is_empty() {
        arena.one
    } else if x_factors.len() == 1 {
        x_factors[0]
    } else {
        arena.mul(&x_factors)
    };

    // g(y) = product of y_factors
    let g_y = if y_factors.len() == 1 {
        y_factors[0]
    } else {
        arena.mul(&y_factors)
    };

    // g(y) = k·y: y = C1·exp(k·∫f(x)dx).  Before, `exp(k·∫f dx + C1)`,
    // which is never zero or negative: the solutions y ≤ 0 were missing.
    {
        let (coeff, base) = arena.as_coeff_term(g_y);
        if base == func {
            let coeff_id = ode_ratio_to_expr(arena, &coeff);
            let scaled_fx = arena.mul(&[coeff_id, f_x]);
            let integral_fx = crate::transforms::integrate::integrate(arena, scaled_fx, var);
            let c1 = arena.symbol("C1");
            let growth = arena.exp(integral_fx);
            let solution = arena.mul(&[c1, growth]);
            return Some(OdeResult {
                solution,
                constants: vec![c1],
                branches: Vec::new(),
            });
        }
    }

    // General case: ∫ dy/g(y) = ∫ f(x) dx + C1, solved for y (every
    // explicit root is a family: `solution` and `branches`), else the
    // implicit `∫ dy/g(y) − ∫ f(x) dx − C1` (= 0).
    // Before, `∫ dy/g(y)` was never attempted (a formal `Integral` was
    // returned even for `y′ = 1 + y²`).
    let neg_one = arena.int(-1);
    let inv_gy = arena.pow(g_y, neg_one);
    let inv_gy = crate::transforms::eval::eval(arena, inv_gy);
    let lhs_attempt = integrate_forms(arena, inv_gy, func);
    let lhs_integral = if integral_failed(arena, lhs_attempt) {
        arena.intern(ExprNode::Integral(inv_gy, func))
    } else {
        lhs_attempt
    };
    let rhs_integral = crate::transforms::integrate::integrate(arena, f_x, var);
    let c1 = arena.symbol("C1");
    let neg_rhs = arena.neg(rhs_integral);
    let neg_c1 = arena.neg(c1);
    let implicit = arena.add(&[lhs_integral, neg_rhs, neg_c1]);
    let implicit = crate::transforms::eval::eval(arena, implicit);
    if !integral_failed(arena, implicit)
        && let Some((solution, branches)) = explicit_families(arena, expr, implicit, func, var, c1)
    {
        return Some(OdeResult {
            solution,
            constants: vec![c1],
            branches,
        });
    }
    Some(OdeResult {
        solution: implicit,
        constants: vec![c1],
        branches: Vec::new(),
    })
}

/// The families `y = …` of the implicit solution `implicit = 0` of the ODE
/// `expr = 0` (one arbitrary constant `c1`): `(first, others)`, or `None`.
///
/// 1. Every root of `implicit` in `func` (at most [`MAX_BRANCHES`], each
///    free of `func`) is a family: `y·y′ = 1` has `y²/2 − x = C1`, so
///    `y = ±√(2C1 + 2x)` (before, an exact or reduced equation whose
///    potential had two roots was left implicit).
/// 2. Failing that, with `ln|u|` replaced by `ln u` and the logarithms'
///    coefficients made integers (`ln|y − 1|/2 − ln|y + 1|/2 = x + C1`
///    as `ln(y − 1) − ln(y + 1) = 2x + 2C1`): on each interval where the
///    sign of `u` is fixed, `ln u − ln|u|` is a constant, absorbed by the
///    arbitrary constant, so each root still solves the ODE (SymPy's
///    `dsolve` integrates without `|·|` likewise).  An exponential
///    `e^{k·C1}` in every occurrence of the constant is then renamed `C1`
///    (its values are the non-zero numbers; `C1 = 0` is the limit family,
///    a solution too where defined): `y′ = y² − 1` gives
///    `(1 + C1·e^{2x})/(1 − C1·e^{2x})` (SymPy: `1/tanh(C1 − x)`; before,
///    the implicit `ln|y − 1|/2 − ln|y + 1|/2 − x − C1`).  Such families
///    are kept only if they verify ([`family_verifies`]).
pub(super) fn explicit_families(
    arena: &mut Arena,
    expr: ExprId,
    implicit: ExprId,
    func: ExprId,
    var: ExprId,
    c1: ExprId,
) -> Option<(ExprId, Vec<ExprId>)> {
    let ExprNode::Symbol(func_sym) = *arena.node(func) else {
        return None;
    };
    // Every root free of `func` that carries the constant (a root without
    // it is no member of the family: `y′ = 1 + sin y` got `y = ±π`, which
    // are not solutions, from spurious roots of `−2/(tan(y/2) + 1) = x + C1`)
    // and is not refuted numerically.
    let explicit_roots = |arena: &mut Arena, e: ExprId| -> Option<Vec<ExprId>> {
        let sols = known_roots(arena, e, func);
        if sols.is_empty()
            || sols.len() > MAX_BRANCHES
            || sols.iter().any(|s| contains_sym(arena, s.value, func_sym))
        {
            return None;
        }
        let mut fams = Vec::with_capacity(sols.len());
        for s in &sols {
            let f = crate::transforms::eval::eval(arena, s.value);
            if crate::base::walk::contains(arena, f, c1)
                && !residual_verdict(arena, expr, f, func, var, c1).is_false()
            {
                fams.push(f);
            }
        }
        (!fams.is_empty()).then_some(fams)
    };
    let stripped_families = |arena: &mut Arena| -> Option<Vec<ExprId>> {
        let stripped = strip_log_abs(arena, implicit)?;
        let fams = explicit_roots(arena, stripped)?;
        let mut kept: Vec<ExprId> = Vec::with_capacity(fams.len());
        for f in fams {
            let f = absorb_exp_constant(arena, f, c1);
            if !kept.contains(&f) && family_verifies(arena, expr, f, func, var, c1) {
                kept.push(f);
            }
        }
        (!kept.is_empty()).then_some(kept)
    };
    let mut fams = match explicit_roots(arena, implicit) {
        // Roots through `|x|` (`√(C1·x²·|x| − x²)`, from `ln|x|`) give way to
        // those of the form without `|·|` (`±x·√(C1·x − 1)`, SymPy's).
        Some(direct) if direct.iter().any(|&f| has_abs(arena, f)) => {
            match stripped_families(arena) {
                Some(s) if !s.iter().any(|&f| has_abs(arena, f)) => s,
                _ => direct,
            }
        }
        Some(direct) => direct,
        None => stripped_families(arena)?,
    };
    // A family `e^{k·C1 + r}` misses `C1·e^{r}` for `C1 ≤ 0`.
    for f in &mut fams {
        let renamed = absorb_exp_constant(arena, *f, c1);
        if renamed != *f && family_verifies(arena, expr, renamed, func, var, c1) {
            *f = renamed;
        }
        *f = exp_log_to_power(arena, *f);
        // `W(C1)/x` is the family `C1/x`.
        if let Some(renamed) = rename_constant_function(arena, *f, var, c1)
            && family_verifies(arena, expr, renamed, func, var, c1)
        {
            *f = renamed;
        }
    }
    // Families equal up to `C1 → −C1` are one (`y′ = tan y`:
    // `asin(C1·eˣ)` and `asin(−C1·eˣ)`).
    let neg_c1 = arena.neg(c1);
    let mut unique: Vec<ExprId> = Vec::with_capacity(fams.len());
    for f in fams {
        let mirrored = crate::transforms::subs::subs(arena, f, c1, neg_c1);
        let mirrored = crate::transforms::eval::eval(arena, mirrored);
        if !unique.contains(&f) && !unique.contains(&mirrored) {
            unique.push(f);
        }
    }
    let first = unique.remove(0);
    Some((first, unique))
}

/// The family `f` with its one principal Lambert W value `W₀(z)` whose
/// argument carries one of the `constants` replaced by `W₋₁(z)`, and that
/// constant; `None` when `f` has no such value or several.
///
/// Solving `w·e^w = z` for `w` takes the principal branch, but every branch
/// solves the ODE (`W′ = W/(z·(1 + W))` on each), and for `−1/e ≤ z < 0`
/// both `W₀(z) ≥ −1` and `W₋₁(z) ≤ −1` are real: a real initial value
/// on the lower branch has no constant in the principal family.  Before,
/// `y′ = 1/(2x + y + 2)`, `y(5/2) = 2/3` failed ("could not solve
/// −W(−2·C1·e⁻¹¹)/2 − 49/6 = 0 for C1": `W₀ ≥ −1` never reaches
/// `−49/3`); its solution is `−2x − W₋₁(−49/3·e^{−4x − 19/3})/2 − 5/2`.
/// SymPy's `dsolve` returns the principal family only (and its `ics`
/// fit fails).  Likewise the exact equation `y·e^{xy} + (x·e^{xy} + 1)·y′
/// = 0` has `C1 − W_k(x·e^{C1·x})/x`, `k = 0, −1`.
pub(super) fn lower_lambert_branch(
    arena: &mut Arena,
    f: ExprId,
    constants: &[ExprId],
) -> Option<(ExprId, ExprId)> {
    let mut principal: Option<(ExprId, ExprId, ExprId)> = None;
    for id in crate::base::walk::post_order_ids(arena, f) {
        let ExprNode::LambertW(z) = *arena.node(id) else {
            continue;
        };
        let Some(&c) = constants
            .iter()
            .find(|&&c| crate::base::walk::contains(arena, z, c))
        else {
            continue;
        };
        match principal {
            Some((w, _, _)) if w != id => return None,
            _ => principal = Some((id, z, c)),
        }
    }
    let (w, z, c) = principal?;
    let neg_one = arena.neg_one;
    let lower = arena.lambertw_branch(z, neg_one);
    let twin = crate::transforms::subs::subs(arena, f, w, lower);
    Some((crate::transforms::eval::eval(arena, twin), c))
}

/// `e` with `e^{k·ln u + r} → u^k·e^{r}` for rational `k` (`u^k` is
/// `e^{k·Log u}` by definition of the principal power): the renaming of
/// [`absorb_exp_constant`] leaves `W(C1·x·e^{−ln x})` for `y′ = −(1/x +
/// y)/(x + 1/y)`, whose initial-value fit then failed.
pub(super) fn exp_log_to_power(arena: &mut Arena, e: ExprId) -> ExprId {
    let post = crate::base::walk::post_order_ids(arena, e);
    let mut cache: rustc_hash::FxHashMap<ExprId, ExprId> = rustc_hash::FxHashMap::default();
    let mut changed = false;
    for &id in &post {
        let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
        let new = match *arena.node(rebuilt) {
            ExprNode::Exp(arg) => {
                let terms: Vec<ExprId> = match arena.node(arg) {
                    ExprNode::Add(c) => c.to_vec(),
                    _ => vec![arg],
                };
                let mut powers = Vec::new();
                let mut rest = Vec::new();
                for t in terms {
                    let (k, term) = arena.as_coeff_term(t);
                    match *arena.node(term) {
                        ExprNode::Ln(u) => {
                            let ke = ode_ratio_to_expr(arena, &k);
                            powers.push(arena.pow(u, ke));
                        }
                        _ => rest.push(t),
                    }
                }
                if powers.is_empty() {
                    rebuilt
                } else {
                    changed = true;
                    if !rest.is_empty() {
                        let r = arena.add(&rest);
                        powers.push(arena.exp(r));
                    }
                    arena.mul(&powers)
                }
            }
            _ => rebuilt,
        };
        cache.insert(id, new);
    }
    if !changed {
        return e;
    }
    let out = cache.get(&e).copied().unwrap_or(e);
    crate::transforms::eval::eval(arena, out)
}

/// `e` with its one subexpression `g(c1)` that contains `c1` but not `var`
/// (every occurrence of `c1` inside it, `g(c1) ≠ c1`) replaced by `c1`;
/// `None` when there is no such single subexpression.  The new constant
/// ranges over the values of `g`, and beyond them by analytic continuation
/// in the constant (the caller verifies the result).
fn rename_constant_function(
    arena: &mut Arena,
    e: ExprId,
    var: ExprId,
    c1: ExprId,
) -> Option<ExprId> {
    let ExprNode::Symbol(c_sym) = *arena.node(c1) else {
        return None;
    };
    let ExprNode::Symbol(v_sym) = *arena.node(var) else {
        return None;
    };
    // Maximal nodes free of `var` containing `c1`: walk from the top.
    let mut found: Option<ExprId> = None;
    let mut stack = vec![e];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) || !contains_sym(arena, n, c_sym) {
            continue;
        }
        if !contains_sym(arena, n, v_sym) {
            match found {
                Some(g) if g != n => return None,
                _ => found = Some(n),
            }
            continue;
        }
        arena.node(n).for_each_child(|c| stack.push(c));
    }
    let g = found?;
    if g == c1 || arena.as_num(g).is_some() {
        return None;
    }
    // A plain multiple or shift of the constant is left as it is.
    if matches!(
        arena.node(g),
        ExprNode::Add(_) | ExprNode::Mul(_) | ExprNode::Neg(_)
    ) {
        return None;
    }
    let out = crate::transforms::subs::subs(arena, e, g, c1);
    Some(crate::transforms::eval::eval(arena, out))
}

/// Does `e` contain an `Abs`?
fn has_abs(arena: &Arena, e: ExprId) -> bool {
    let mut stack = vec![e];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        let node = arena.node(n);
        if matches!(node, ExprNode::Abs(_)) {
            return true;
        }
        node.for_each_child(|c| stack.push(c));
    }
    false
}

/// `e` with every `ln|u|` replaced by `ln u` and multiplied by the least
/// common denominator of the coefficients of its logarithms (as terms of
/// the expanded sum); `None` if `e` has no `ln|u|`.
fn strip_log_abs(arena: &mut Arena, e: ExprId) -> Option<ExprId> {
    let out = ln_abs_to_ln(arena, e)?;
    let terms: Vec<ExprId> = match arena.node(out) {
        ExprNode::Add(c) => c.to_vec(),
        _ => vec![out],
    };
    let mut lcm = num_bigint::BigInt::one();
    for t in terms {
        let (c, term) = arena.as_coeff_term(t);
        if matches!(arena.node(term), ExprNode::Ln(_)) {
            lcm = num_integer::Integer::lcm(&lcm, c.denom());
        }
    }
    if lcm.is_one() {
        return Some(out);
    }
    let scale = arena.intern_num(Q::from_integer(lcm));
    let scale = arena.intern(ExprNode::Num(scale));
    let scaled = arena.mul(&[scale, out]);
    let scaled = crate::transforms::expand::expand(arena, scaled);
    Some(crate::transforms::eval::eval(arena, scaled))
}

/// `e` with every `ln|u|` replaced by `ln u`; `None` if it has none.
pub(super) fn ln_abs_to_ln(arena: &mut Arena, e: ExprId) -> Option<ExprId> {
    let post = crate::base::walk::post_order_ids(arena, e);
    let mut cache: rustc_hash::FxHashMap<ExprId, ExprId> = rustc_hash::FxHashMap::default();
    let mut changed = false;
    for &id in &post {
        let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
        let new = match arena.node(rebuilt).clone() {
            ExprNode::Ln(inner) => match arena.node(inner).clone() {
                ExprNode::Abs(u) => {
                    changed = true;
                    arena.ln(u)
                }
                _ => rebuilt,
            },
            _ => rebuilt,
        };
        cache.insert(id, new);
    }
    if !changed {
        return None;
    }
    let out = cache.get(&e).copied().unwrap_or(e);
    Some(crate::transforms::eval::eval(arena, out))
}

/// `e` with `c1` renamed in `e^{k·c1 + r} → c1^{k/k₀}·e^{r}` when every
/// occurrence of `c1` is in such an exponential with a rational `k`, `k₀`
/// the greatest rational dividing every `k` (the new `c1` is `e^{k₀·c1}`;
/// see [`explicit_families`]); `e` itself otherwise.
fn absorb_exp_constant(arena: &mut Arena, e: ExprId, c1: ExprId) -> ExprId {
    let ExprNode::Symbol(c_sym) = *arena.node(c1) else {
        return e;
    };
    // `e^{k·c1 + r}` (r free of c1) → `(k, r)`.
    let split = |arena: &mut Arena, arg: ExprId| -> Option<(Q, Vec<ExprId>)> {
        let terms: Vec<ExprId> = match arena.node(arg) {
            ExprNode::Add(c) => c.to_vec(),
            _ => vec![arg],
        };
        let mut k: Option<Q> = None;
        let mut rest = Vec::with_capacity(terms.len());
        for t in terms {
            let (c, term) = arena.as_coeff_term(t);
            if term == c1 && k.is_none() {
                k = Some(c);
            } else if contains_sym(arena, t, c_sym) {
                return None;
            } else {
                rest.push(t);
            }
        }
        Some((k?, rest))
    };
    let post = crate::base::walk::post_order_ids(arena, e);
    // Pass 1: the multiples `k`, and every occurrence of `c1` in one.
    let mut erased: rustc_hash::FxHashMap<ExprId, ExprId> = rustc_hash::FxHashMap::default();
    let mut hits: rustc_hash::FxHashMap<ExprId, (Q, Vec<ExprId>)> =
        rustc_hash::FxHashMap::default();
    let one = arena.one;
    for &id in &post {
        let gone = crate::base::walk::rebuild_with_cache(arena, id, &erased);
        let hit = match *arena.node(id) {
            ExprNode::Exp(arg) => split(arena, arg),
            _ => None,
        };
        let new_gone = match hit {
            Some(h) => {
                hits.insert(id, h);
                one
            }
            None => gone,
        };
        erased.insert(id, new_gone);
    }
    let rest = erased.get(&e).copied().unwrap_or(e);
    if hits.is_empty() || contains_sym(arena, rest, c_sym) {
        return e;
    }
    // k₀ = gcd(numerators)/lcm(denominators), with the sign of the first.
    let mut num_gcd = num_bigint::BigInt::from(0);
    let mut den_lcm = num_bigint::BigInt::one();
    let mut sign_neg = None;
    for &id in &post {
        if let Some((k, _)) = hits.get(&id) {
            num_gcd = num_integer::Integer::gcd(&num_gcd, k.numer());
            den_lcm = num_integer::Integer::lcm(&den_lcm, k.denom());
            sign_neg.get_or_insert(k.is_negative());
        }
    }
    if num_gcd == num_bigint::BigInt::from(0) {
        return e;
    }
    let mut k0 = Q::new(num_gcd, den_lcm);
    if Truth::from(sign_neg).is_true() {
        k0 = -k0;
    }
    // Pass 2: e^{k·c1 + r} → c1^{k/k₀}·e^{r}.
    let mut renamed: rustc_hash::FxHashMap<ExprId, ExprId> = rustc_hash::FxHashMap::default();
    for &id in &post {
        let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &renamed);
        let new = match hits.get(&id).cloned() {
            Some((k, rest)) => {
                let r = arena.add(&rest);
                let ex = arena.exp(r);
                let p = ode_ratio_to_expr(arena, &(k / &k0));
                let cp = arena.pow(c1, p);
                arena.mul(&[cp, ex])
            }
            None => rebuilt,
        };
        renamed.insert(id, new);
    }
    let out = renamed.get(&e).copied().unwrap_or(e);
    crate::transforms::eval::eval(arena, out)
}

/// Does `sol` solve `expr = 0`?  Its [`residual_verdict`], or [`checkodesol`]
/// when that is undecided.  Used for the families [`explicit_families`]
/// derives by dropping `|·|` and renaming the constant.
pub(super) fn family_verifies(
    arena: &mut Arena,
    expr: ExprId,
    sol: ExprId,
    func: ExprId,
    var: ExprId,
    c1: ExprId,
) -> bool {
    match residual_verdict(arena, expr, sol, func, var, c1) {
        Truth::True => true,
        Truth::False => false,
        Truth::Unknown => checkodesol(arena, expr, sol, func, var),
    }
}

/// The residual of the family `sol` (one constant `c1`; others are set to
/// fixed values) in `expr = 0` at three points and four values of `c1`,
/// evaluated over ℂ on principal branches: `True` when it is proved 0 at
/// two or more of them (a formula may be a solution only where its
/// radicals take the right branch: `(x/2 + C1/2)²` for `y′ = √y` needs `x +
/// C1 ≥ 0`), `False` when it is certified nonzero at every point where it
/// evaluates (two or more), `Unknown` otherwise.  Each point is decided
/// by [`residual_zero_at`] (exact or certified); up to 0.50 a residual
/// below `10⁻⁹` of the derivatives' scale in 12-digit arithmetic counted
/// as 0, so a family off by a tiny amount verified.
fn residual_verdict(
    arena: &mut Arena,
    expr: ExprId,
    sol: ExprId,
    func: ExprId,
    var: ExprId,
    c1: ExprId,
) -> Truth {
    let chain = derivative_chain(arena, func, var);
    let order = (1..chain.len())
        .rev()
        .find(|&k| expr_contains(arena, expr, chain[k]))
        .unwrap_or(0);
    let mut ders = vec![sol];
    for k in 1..=order {
        let d = crate::transforms::diff::diff(arena, ders[k - 1], var);
        ders.push(d);
    }
    let mut resid = expr;
    for k in (0..=order).rev() {
        resid = crate::transforms::subs::subs(arena, resid, chain[k], ders[k]);
    }
    // Other constants (C2, … of a reduced higher-order equation) fixed.
    let others: Vec<ExprId> = crate::base::walk::free_symbols(arena, sol)
        .into_iter()
        .filter(|&s| s != c1 && s != var && is_constant_name(arena, s))
        .collect();
    let (mut evaluated, mut passed, mut refuted) = (0usize, 0usize, 0usize);
    for (xn, xd) in [(3i64, 7i64), (11, 9), (-5, 4)] {
        for (cn, cd) in [(2i64, 3i64), (-5, 4), (7, 5), (-1, 3)] {
            let xv = arena.rational(xn, xd);
            let cv = arena.rational(cn, cd);
            let ov = arena.rational(3, 5);
            let mut e = crate::transforms::subs::subs(arena, resid, var, xv);
            e = crate::transforms::subs::subs(arena, e, c1, cv);
            for &o in &others {
                e = crate::transforms::subs::subs(arena, e, o, ov);
            }
            let r = crate::transforms::eval::eval(arena, e);
            let Some(zero) = residual_zero_at(arena, r) else {
                continue;
            };
            evaluated += 1;
            match zero {
                Truth::True => {
                    passed += 1;
                    if passed >= 2 {
                        return Truth::True;
                    }
                }
                Truth::False => refuted += 1,
                Truth::Unknown => {}
            }
        }
    }
    if passed >= 2 {
        Truth::True
    } else if evaluated >= 2 && refuted == evaluated {
        Truth::False
    } else {
        Truth::Unknown
    }
}

/// Is the residual `r` of [`residual_verdict`] at one sample point 0?
/// `True` when canonical forms or [`hidden_zero`](crate::calculus::calculus_util::hidden_zero)
/// prove it, `False` when a certified evaluation excludes 0, `Unknown` when
/// it is 0 to the precision reached but not proved; `None` when it has no
/// finite value there (a point outside the family's domain).
pub(super) fn residual_zero_at(arena: &mut Arena, r: ExprId) -> Option<Truth> {
    if arena.is_zero_structural(r) {
        return Some(Truth::True);
    }
    let v = crate::transforms::evalf::evalf_verdict(
        arena,
        r,
        15,
        crate::transforms::evalf::ZeroSearch::Cap,
    )
    .ok()?;
    let (re, im) = (&v.value.0, &v.value.1);
    if re.is_inf() || re.is_nan() || im.is_inf() || im.is_nan() {
        return None;
    }
    if let Some(z) = v.proven() {
        return Some(Truth::from_bool(z.0.is_zero() && z.1.is_zero()));
    }
    let proven = crate::calculus::calculus_util::hidden_zero(arena, r).proven();
    Some(proven.unwrap_or(Truth::Unknown))
}

/// Is the symbol `s` named like an integration constant (`C1`, `C2`, …)?
fn is_constant_name(arena: &Arena, s: ExprId) -> bool {
    match arena.node(s) {
        ExprNode::Symbol(sid) => {
            let name = arena.symbol_name(*sid);
            name.len() > 1 && name.starts_with('C') && name[1..].chars().all(|c| c.is_ascii_digit())
        }
        _ => false,
    }
}

/// Collect multiplicative factors from an expression.
/// If `expr` is `Mul(a, b, c)`, return `[a, b, c]`.
/// If `expr` is `Neg(inner)`, return factors of inner with a `-1` prepended.
/// Otherwise return `[expr]`.
pub(super) fn collect_mul_factors(arena: &mut Arena, expr: ExprId) -> Vec<ExprId> {
    let mut factors = Vec::new();
    let mut expr = expr;
    loop {
        match arena.node(expr).clone() {
            ExprNode::Mul(children) => {
                factors.extend(children.iter().copied());
                return factors;
            }
            ExprNode::Neg(inner) => {
                factors.push(arena.int(-1));
                expr = inner;
            }
            _ => {
                factors.push(expr);
                return factors;
            }
        }
    }
}
