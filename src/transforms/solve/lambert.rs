//! LambertW solving for mixed polynomial-exponential equations (part of [`solve`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// LambertW solving for mixed polynomial-exponential equations
// ═══════════════════════════════════════════════════════════════════════════

/// Try to solve equations involving mixed polynomial and exponential or
/// logarithmic terms with the Lambert W function (after SymPy's
/// `_solve_lambert` / `_lambert`, `sympy/solvers/bivariate.py`, BSD):
///
/// - `A·x·exp(B·x) + D = 0`  →  `x = W_k(−B·D/A)/B`
/// - `A·exp(B·x) + C·x + D = 0`  →  `x = −W_k(A·B/(C·exp(B·D/C)))/B − D/C`
/// - `A·x·ln(B·x) + C·x + D = 0`  →  `x = exp(W_k(−(B·D/A)·exp(C/A)) − C/A)/B`
/// - `A·ln(B·x) + C·x + D = 0`  →  `x = (A/C)·W_k((C/(A·B))·exp(−D/A))`
/// - `A·x^x + D = 0`  →  `x = exp(W_k(ln(−D/A)))`
///
/// (with `ln(B·x)` for a constant `B` solved in `B·x`).  The branches
/// `W_k` returned are those of [`lambert_branches`]: the principal one, and
/// `W₋₁` as well when the argument lies in `(−1/e, 0)`, where both are real
/// and the equation has two real roots.  Before 0.30 only `W₀` was taken:
/// `solve(exp(x) − 2x − π, x)` gave `−1.4540` and missed `1.9526`
/// (`−π/2 − W₋₁(−e^{−π/2}/2)`), and the logarithmic forms and `x^x` were
/// not recognised at all.
pub(super) fn try_solve_lambert(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
    cx: &SolveCx,
) -> Solutions {
    candidates_found(lambert_candidates(arena, expr, var, var_sym, cx))
}

/// The candidates of [`try_solve_lambert`]; `None` when the equation does
/// not have one of its shapes.
fn lambert_candidates(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    _var_sym: SymbolId,
    cx: &SolveCx,
) -> Option<Vec<Solution>> {
    tracing::debug!("solve: trying LambertW");

    // Get additive terms
    let terms: Vec<ExprId> = match arena.node(expr).clone() {
        ExprNode::Add(children) => children.to_vec(),
        _ => vec![expr],
    };

    // Classify each term into categories
    let mut constant_terms: Vec<ExprId> = Vec::new();
    let mut linear_coeffs: Vec<ExprId> = Vec::new(); // A for A*var
    let mut exp_terms: Vec<LambertExp> = Vec::new(); // A*exp(B*var)
    let mut var_exp_terms: Vec<LambertExp> = Vec::new(); // A*var*exp(B*var)
    let mut var_ln_terms: Vec<LambertLn> = Vec::new(); // A*var*ln(B*var)
    let mut ln_terms: Vec<LambertLn> = Vec::new(); // A*ln(B*var)
    let mut pow_terms: Vec<ExprId> = Vec::new(); // A for A*var^var

    for &term in &terms {
        if !expr_contains_var(arena, term, var) {
            constant_terms.push(term);
            continue;
        }

        match classify_lambert_term(arena, term, var) {
            Some(LambertTermClass::Linear(coeff)) => linear_coeffs.push(coeff),
            Some(LambertTermClass::ExpVar(e)) => exp_terms.push(e),
            Some(LambertTermClass::VarExpVar(e)) => var_exp_terms.push(e),
            Some(LambertTermClass::VarLnVar(ln)) => var_ln_terms.push(ln),
            Some(LambertTermClass::LnVar(ln)) => ln_terms.push(ln),
            Some(LambertTermClass::PowVarVar(coeff)) => pow_terms.push(coeff),
            None => return None,
        }
    }

    // Build constant sum (D)
    let d = match constant_terms.len() {
        0 => arena.zero,
        1 => constant_terms[0],
        _ => arena.add(&constant_terms),
    };
    let c = match linear_coeffs.len() {
        0 => None,
        1 => Some(linear_coeffs[0]),
        _ => Some(arena.add(&linear_coeffs)),
    };
    let counts = (
        exp_terms.len(),
        var_exp_terms.len(),
        var_ln_terms.len(),
        ln_terms.len(),
        pow_terms.len(),
    );

    let solutions: Vec<ExprId> = match (counts, c) {
        // ── A*var*exp(B*var) + D = 0 ─────────────────────────────────
        //   B*var*exp(B*var) = -B*D/A  →  var = W(-B*D/A)/B
        ((0, 1, 0, 0, 0), None) => {
            let LambertExp { coeff: a, rate: b } = var_exp_terms[0];
            let neg_d = arena.neg(d);
            let neg_d_over_a = arena.div(neg_d, a);
            let b_arg = arena.mul(&[b, neg_d_over_a]);
            lambert_branches(arena, b_arg, cx)
                .into_iter()
                .map(|w| arena.div(w, b))
                .collect()
        }
        // ── A*exp(B*var) + C*var + D = 0 ─────────────────────────────
        //   u = -(B*var + B*D/C):  u·exp(u) = A·B/(C·exp(B·D/C))
        //   var = -W(…)/B - D/C
        ((1, 0, 0, 0, 0), Some(c)) => {
            let LambertExp {
                coeff: a_exp,
                rate: b,
            } = exp_terms[0];
            let b_d = arena.mul(&[b, d]);
            let b_d_over_c = arena.div(b_d, c);
            let exp_bd_c = arena.exp(b_d_over_c);
            let c_exp_bd_c = arena.mul(&[c, exp_bd_c]);
            let a_b = arena.mul(&[a_exp, b]);
            let w_arg = arena.div(a_b, c_exp_bd_c);
            let d_over_c = arena.div(d, c);
            lambert_branches(arena, w_arg, cx)
                .into_iter()
                .map(|w| {
                    let neg_w = arena.neg(w);
                    let neg_w_over_b = arena.div(neg_w, b);
                    arena.sub(neg_w_over_b, d_over_c)
                })
                .collect()
        }
        // ── A*var*ln(B*var) + C*var + D = 0 ──────────────────────────
        //   B*var = e^t:  (t + C/A)·e^{t + C/A} = -(B·D/A)·e^{C/A}
        //   var = exp(W(-(B·D/A)·e^{C/A}) - C/A)/B
        ((0, 0, 1, 0, 0), c) if arena.is_zero_structural(var_ln_terms[0].shift) => {
            let LambertLn {
                coeff: a, scale: b, ..
            } = var_ln_terms[0];
            let c_over_a = match c {
                Some(c) => arena.div(c, a),
                None => arena.zero,
            };
            let neg_bd = arena.mul(&[arena.neg_one, b, d]);
            let neg_bd_over_a = arena.div(neg_bd, a);
            let exp_c_a = arena.exp(c_over_a);
            let w_arg = arena.mul(&[neg_bd_over_a, exp_c_a]);
            lambert_branches(arena, w_arg, cx)
                .into_iter()
                .map(|w| {
                    let t = arena.sub(w, c_over_a);
                    let y = arena.exp(t);
                    arena.div(y, b)
                })
                .collect()
        }
        // ── A*ln(B*var + E) + C*var + D = 0 ──────────────────────────
        //   u = B·var + E:  A·ln u + C′·u + D′ = 0,  C′ = C/B,  D′ = D − C·E/B
        //   u = (A/C′)·W((C′/A)·e^{-D′/A}),  var = (u − E)/B
        //   (E = 0: var = (A/C)·W((C/(A·B))·e^{-D/A}); a shift E was
        //   refused before 0.44: `ln(4x − 2) = 2x − 2`.)
        ((0, 0, 0, 1, 0), Some(c)) => {
            let LambertLn {
                coeff: a,
                scale: b,
                shift: e,
            } = ln_terms[0];
            let c1 = arena.div(c, b);
            let c1 = crate::transforms::eval::eval(arena, c1);
            let ce = arena.mul(&[c1, e]);
            let d1 = arena.sub(d, ce);
            let c1_over_a = arena.div(c1, a);
            let a_over_c1 = arena.div(a, c1);
            let neg_d = arena.neg(d1);
            let neg_d_over_a = arena.div(neg_d, a);
            let exp_part = arena.exp(neg_d_over_a);
            let w_arg = arena.mul(&[c1_over_a, exp_part]);
            lambert_branches(arena, w_arg, cx)
                .into_iter()
                .map(|w| {
                    let u = arena.mul(&[a_over_c1, w]);
                    let shifted = arena.sub(u, e);
                    arena.div(shifted, b)
                })
                .collect()
        }
        // ── A*var^var + D = 0 ────────────────────────────────────────
        //   var·ln(var) = ln(-D/A)  →  var = exp(W(ln(-D/A)))
        //   Besides, the integers with n^n = r: n ≥ 2 exactly (`x^x = 27`:
        //   `3`, not `exp(W(ln 27))`), the negative ones (`(−2)^(−2) = 1/4`,
        //   `(−1)^(−1) = −1`, real values of x^x off the positive axis that no
        //   real branch of W reaches; SymPy misses them), and `0` for
        //   `r = 1` (`0⁰ = 1`, as SymPy's `[0, 1]`).
        ((0, 0, 0, 0, 1), None) => {
            let a = pow_terms[0];
            let neg_d = arena.neg(d);
            let r = arena.div(neg_d, a);
            let r = crate::transforms::eval::eval(arena, r);
            let q = arena.as_num(r).cloned();
            let positive = q
                .as_ref()
                .filter(|q| q.is_integer() && q.is_positive())
                .and_then(|q| self_power_root(&q.to_integer()))
                .filter(|&n| n >= 2);
            let mut sols: Vec<ExprId> = match positive {
                Some(n) => vec![arena.int(i64::from(n))],
                None => {
                    let ln_r = arena.ln(r);
                    lambert_branches(arena, ln_r, cx)
                        .into_iter()
                        .map(|w| arena.exp(w))
                        .collect()
                }
            };
            if let Some(q) = q {
                // (−m)^(−m) = (−1)^m/m^m
                if q.numer().abs().is_one()
                    && let Some(m) = self_power_root(q.denom())
                    && (m % 2 == 0) == q.is_positive()
                {
                    sols.push(arena.int(-i64::from(m)));
                }
                if q.is_one() {
                    sols.push(arena.zero);
                }
            }
            sols
        }
        _ => return None,
    };
    Some(
        solutions
            .into_iter()
            .map(|s| Solution {
                value: crate::transforms::eval::eval(arena, s),
            })
            .collect(),
    )
}

/// Logarithmic Lambert forms in a power of the variable: `var` only in
/// `ln(var)` and in powers `varⁿ` of one rational exponent `n ≠ 0, 1`
/// (`x²·ln x = 1`, `ln(x)/x² = c`, `ln x = x³`).  With `u = xⁿ` and
/// `ln x = ln(u)/n` (for `x > 0`; a negative `x` makes `ln x` non-real)
/// they are forms of [`try_solve_lambert`] in `u`, and `x = u^(1/n)`;
/// [`solve_impl`] checks the candidates.  A logarithm `ln(B·x)` of a
/// constant multiple is first rescaled, `x = y/B`.  SymPy:
/// `solve(x**2*log(x) - 1)` → `[exp(LambertW(2)/2)]`, `solve(x**5*log(2*x)
/// - 3)` → `[exp(LambertW(480)/5)/2]`; refused before 0.44.
pub(super) fn try_lambert_power_log(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Solutions {
    candidates_found(power_log_candidates(arena, expr, var, cx))
}

/// The candidates of [`try_lambert_power_log`]; `None` when the equation
/// does not have its shape.
fn power_log_candidates(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Option<Vec<Solution>> {
    let logs: Vec<ExprId> = crate::base::walk::post_order_ids(arena, expr)
        .into_iter()
        .filter_map(|id| match *arena.node(id) {
            ExprNode::Ln(a) if expr_contains_var(arena, a, var) => Some(a),
            _ => None,
        })
        .collect();
    let [arg] = logs[..] else {
        return None;
    };
    if arg == var {
        return power_log_in(arena, expr, var, cx);
    }
    let scale = extract_var_coeff_in_product(arena, arg, var)?;
    // y = B·x: xⁿ = B⁻ⁿ·yⁿ for an integer n, or a positive B
    let powers: Vec<(ExprId, Q)> = crate::base::walk::post_order_ids(arena, expr)
        .into_iter()
        .filter_map(|id| match *arena.node(id) {
            ExprNode::Pow(b, e) if b == var => arena.as_num(e).map(|q| (id, q.clone())),
            _ => None,
        })
        .collect();
    let [(pow_node, ref n)] = powers[..] else {
        return None;
    };
    if !n.is_integer() && crate::poly::algebraic::sign_checked(arena, scale) != Some(1) {
        return None;
    }
    let y = arena.symbol("__y_lambert");
    let n_id = arena.num_ratio(n.clone());
    let neg_n = arena.num_ratio(-n);
    let y_n = arena.pow(y, n_id);
    let b_neg_n = arena.pow(scale, neg_n);
    let repl = arena.mul(&[b_neg_n, y_n]);
    let ln_arg = arena.ln(arg);
    let ln_y = arena.ln(y);
    let scaled = arena.subs_structural(expr, pow_node, repl);
    let scaled = arena.subs_structural(scaled, ln_arg, ln_y);
    let scaled = crate::transforms::eval::eval(arena, scaled);
    if expr_contains_var(arena, scaled, var) {
        return None;
    }
    let sols = power_log_in(arena, scaled, y, cx)?;
    Some(
        sols.into_iter()
            .map(|s| {
                let x = arena.div(s.value, scale);
                Solution {
                    value: crate::transforms::eval::eval(arena, x),
                }
            })
            .collect(),
    )
}

/// `K·x^a·(ln x)^m + d = 0` for an integer `m ≥ 2` and a rational `a ≠ 0`
/// (`x·ln²x = c`): for each `r` with `r^m = −d/K` ([`power_roots`]: both
/// `±√c` for an even `m`), `x^s·ln x = r` with `s = a/m`, and `y = s·ln x`
/// solves `y·e^y = s·r`: `x = exp(W_k(s·r)/s)` on the real branches.  Every
/// candidate is checked by [`solve_impl`].  SymPy: `solve(x*log(x)**2 -
/// 2)` → `[exp(2*LambertW(-sqrt(2)/2)), exp(2*LambertW(sqrt(2)/2))]`;
/// refused before 0.45.
pub(super) fn try_power_of_log(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Solutions {
    candidates_found(power_of_log_candidates(arena, expr, var, cx))
}

/// The candidates of [`try_power_of_log`]; `None` when the equation does
/// not have its shape.
fn power_of_log_candidates(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Option<Vec<Solution>> {
    let terms: Vec<ExprId> = match arena.node(expr) {
        ExprNode::Add(ch) => ch.to_vec(),
        _ => vec![expr],
    };
    let mut constants: Vec<ExprId> = Vec::new();
    let mut var_term: Option<ExprId> = None;
    for t in terms {
        if !expr_contains_var(arena, t, var) {
            constants.push(t);
        } else if var_term.replace(t).is_some() {
            return None;
        }
    }
    let term = var_term?;
    let factors: Vec<ExprId> = match arena.node(term) {
        ExprNode::Mul(ch) => ch.to_vec(),
        _ => vec![term],
    };
    let ln_x = arena.ln(var);
    let mut k_parts: Vec<ExprId> = Vec::new();
    let mut a = Q::zero();
    let mut m: Option<Q> = None;
    for f in factors {
        if !expr_contains_var(arena, f, var) {
            k_parts.push(f);
            continue;
        }
        if f == var {
            a += Q::one();
            continue;
        }
        match *arena.node(f) {
            ExprNode::Pow(b, e) if b == var => a += arena.as_num(e)?.clone(),
            ExprNode::Pow(b, e) if b == ln_x && m.is_none() => {
                let q = arena.as_num(e)?.clone();
                if !q.is_integer() || q < Q::from_integer(BigInt::from(2)) {
                    return None;
                }
                m = Some(q);
            }
            _ => return None,
        }
    }
    let m = m?;
    if a.is_zero() || constants.is_empty() {
        return None;
    }
    let k = if k_parts.is_empty() {
        arena.one
    } else {
        arena.mul(&k_parts)
    };
    let d = arena.add(&constants);
    let neg_d = arena.neg(d);
    let c = arena.div(neg_d, k);
    let c = crate::transforms::eval::eval(arena, c);
    if arena.is_zero_structural(c) {
        return None;
    }
    let s = &a / &m;
    let s_id = arena.num_ratio(s);
    let mut out: Vec<Solution> = Vec::new();
    for r in power_roots(arena, c, &m)? {
        let arg = arena.mul(&[s_id, r]);
        for w in lambert_branches(arena, arg, cx) {
            let y = arena.div(w, s_id);
            let x = arena.exp(y);
            let x = crate::transforms::eval::eval(arena, x);
            if !out.iter().any(|o| o.value == x) {
                out.push(Solution { value: x });
            }
        }
    }
    Some(out)
}

/// Equations in which the unknown occurs only through `xⁿ` (`n ≥ 2`, the
/// gcd of its integer exponents): `w = xⁿ` is substituted, the equation in
/// `w` solved, and each root `w₀` gives the `x` with `xⁿ = w₀` (the real
/// ones, or all `n` when there is none, [`real_nth_roots`]).  The
/// substitution is exact, so no root is lost, and [`solve_impl`] checks
/// the candidates against the equation in `x`.  `(x² + 1)·e^(x²) = 3` is
/// `(w + 1)·e^w = 3`: `x = ±√(W(3e) − 1)`;
/// refused before 0.45 (SymPy 1.14 solves it too).  `None` for a
/// polynomial, for families in `w` (periodic ones are peeled directly),
/// or when the equation in `w` is not solved.
pub(super) fn try_monomial_substitution(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Solutions {
    let not_found = || Solutions::refused(Reason::NotFound);
    let Some((w, sub, n)) = monomial_substitution(arena, expr, var) else {
        return not_found();
    };
    let (w_roots, missing) = match solve_raw(arena, sub, w, cx)
        .without_conditions()
        .into_found()
    {
        Ok(found) => found,
        Err(nested) if nested.is_nested_periods() => return nested,
        Err(_) => return not_found(),
    };
    let mut out: Vec<ExprId> = Vec::new();
    for r in w_roots {
        if cx
            .period()
            .is_some_and(|p| crate::base::walk::contains(arena, r, p))
        {
            return not_found();
        }
        for x in real_nth_roots(arena, r, n) {
            let x = crate::transforms::eval::eval(arena, x);
            if !out.contains(&x) {
                out.push(x);
            }
        }
    }
    Solutions::with_missing(out, missing)
}

/// The substitution `w = xⁿ` of [`try_monomial_substitution`]: `(w, the
/// equation in w, n)`, or `None` when the equation does not have its shape.
fn monomial_substitution(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
) -> Option<(ExprId, ExprId, u32)> {
    if polybridge::expr_to_poly(arena, expr, var).is_some() {
        return None;
    }
    // gcd of the exponents of `var`; a bare occurrence makes it 1
    let mut g = BigInt::zero();
    let mut powers: Vec<(ExprId, BigInt)> = Vec::new();
    let order = crate::base::walk::post_order_ids(arena, expr);
    let mut under_pow: rustc_hash::FxHashSet<ExprId> = rustc_hash::FxHashSet::default();
    for &id in &order {
        if let ExprNode::Pow(b, e) = *arena.node(id)
            && b == var
        {
            let q = arena.as_num(e)?;
            if !q.is_integer() {
                return None;
            }
            let k = q.to_integer();
            g = num_integer::Integer::gcd(&g, &k);
            powers.push((id, k));
            under_pow.insert(id);
        }
    }
    // `var` must not occur outside those powers
    for &id in &order {
        if under_pow.contains(&id) {
            continue;
        }
        if arena.node(id).children().contains(&var) && !under_pow.contains(&id) {
            return None;
        }
    }
    if expr == var || g < BigInt::from(2) {
        return None;
    }
    let n = num_traits::ToPrimitive::to_u32(&g)?;
    if u64::from(n) > MAX_LAMBERT_POWER {
        return None;
    }
    let w = arena.symbol("__w_monomial");
    let mut sub = expr;
    for (id, k) in &powers {
        let kk = arena.big_int(k / &g);
        let wk = arena.pow(w, kk);
        sub = arena.subs_structural(sub, *id, wk);
    }
    let sub = crate::transforms::eval::eval(arena, sub);
    if expr_contains_var(arena, sub, var) {
        return None;
    }
    Some((w, sub, n))
}

/// The outcome of a candidate search (Lambert W forms): the candidates
/// found, or refused when there are none or the shape does not apply —
/// such a search proves nothing.
fn candidates_found(candidates: Option<Vec<Solution>>) -> Solutions {
    match candidates {
        Some(found) => Solutions::found(values(found)),
        None => Solutions::refused(Reason::NotFound),
    }
}

/// The roots of a peeling inside a candidate search: `Some` (possibly
/// none) unless refused.  (The peelings of these searches invert `e^(b·x)`
/// or a power of `x` against a constant: they meet no periodic family.)
fn peeled_candidates(outcome: Solutions) -> Option<Vec<ExprId>> {
    match outcome.into_found() {
        Ok((roots, _)) => Some(roots),
        Err(Solutions::Complete(RootSet::Empty(_))) => Some(Vec::new()),
        Err(_) => None,
    }
}

/// [`try_lambert_power_log`] for `ln(var)` itself.
fn power_log_in(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Option<Vec<Solution>> {
    let ln_x = arena.ln(var);
    let mut power: Option<(ExprId, Q)> = None;
    let mut has_log = false;
    let mut stack = vec![expr];
    let mut seen: rustc_hash::FxHashSet<ExprId> = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) || !expr_contains_var(arena, id, var) {
            continue;
        }
        if id == ln_x {
            has_log = true;
            continue;
        }
        match arena.node(id) {
            ExprNode::Add(ch) | ExprNode::Mul(ch) => stack.extend(ch.iter().copied()),
            ExprNode::Neg(b) => stack.push(*b),
            ExprNode::Pow(b, e) if *b == var => {
                let q = arena.as_num(*e)?.clone();
                match &power {
                    Some((p, _)) if *p != id => return None,
                    _ => power = Some((id, q)),
                }
            }
            _ => return None,
        }
    }
    let (pow_node, n) = power?;
    // n = −1 (`ln(x)/x = c`) is left to the numerator `ln x − c·x`, which
    // gives SymPy's `−4·W(−1/4)` rather than the equal `exp(−W(−1/4))`.
    if !has_log || n.is_zero() || n.is_one() || n == -Q::one() {
        return None;
    }
    let u = arena.symbol("__u_lambert");
    let ExprNode::Symbol(u_sym) = *arena.node(u) else {
        return None;
    };
    let inv_n = arena.num_ratio(n.recip());
    let ln_u = arena.ln(u);
    let ln_u_over_n = arena.mul(&[inv_n, ln_u]);
    let sub = arena.subs_structural(expr, pow_node, u);
    let sub = arena.subs_structural(sub, ln_x, ln_u_over_n);
    let sub = crate::transforms::eval::eval(arena, sub);
    if expr_contains_var(arena, sub, var) {
        return None;
    }
    let sols = lambert_candidates(arena, sub, u, u_sym, cx)?;
    Some(
        sols.into_iter()
            .map(|s| {
                // exp(v)^(1/n) = exp(v/n) for a real v (`exp(W(2)/2)`, as
                // SymPy writes it, rather than `√(exp(W(2)))`)
                let node = arena.node(s.value).clone();
                let x = match node {
                    ExprNode::Exp(v)
                        if crate::base::walk::free_symbols(arena, v).is_empty()
                            && crate::poly::algebraic::sign_checked(arena, v).is_some() =>
                    {
                        let w = arena.mul(&[inv_n, v]);
                        arena.exp(w)
                    }
                    _ => arena.pow(s.value, inv_n),
                };
                Solution {
                    value: crate::transforms::eval::eval(arena, x),
                }
            })
            .collect(),
    )
}

/// A product with several `exp` factors with them merged into one
/// (`exp(−1)·exp(−3)` → `exp(−4)`, which the evaluator leaves apart).
fn merge_exp_factors(arena: &mut Arena, e: ExprId) -> ExprId {
    let ExprNode::Mul(factors) = arena.node(e).clone() else {
        return e;
    };
    let mut args: Vec<ExprId> = Vec::new();
    let mut rest: Vec<ExprId> = Vec::new();
    for f in factors.iter() {
        match *arena.node(*f) {
            ExprNode::Exp(a) => args.push(a),
            _ => rest.push(*f),
        }
    }
    if args.len() < 2 {
        return e;
    }
    let sum = arena.add(&args);
    let sum = crate::transforms::eval::eval(arena, sum);
    rest.push(arena.exp(sum));
    let merged = arena.mul(&rest);
    crate::transforms::eval::eval(arena, merged)
}

/// The integer `m ≥ 1` with `m^m = n`, if any.
fn self_power_root(n: &BigInt) -> Option<u32> {
    if !n.is_positive() {
        return None;
    }
    for m in 1u32..=64 {
        let p = num_traits::Pow::pow(BigInt::from(m), m);
        match p.cmp(n) {
            std::cmp::Ordering::Equal => return Some(m),
            std::cmp::Ordering::Greater => return None,
            std::cmp::Ordering::Less => {}
        }
    }
    None
}

/// The Lambert W values `W_k(arg)` a Lambert solution takes: the principal
/// branch, and `W₋₁(arg)` as well when `arg` is certainly in `(−1/e, 0)`,
/// where both branches are real (SymPy's `_lambert` keeps `k = −1` only
/// when `LambertW(arg, -1)` is known to be real; here the interval is
/// decided with certified signs).  At `arg = −1/e` the two coincide, and
/// for `arg ≥ 0` or a symbolic `arg` only `W₀` is taken — the other
/// branches give the complex roots, infinitely many, which `solve` does
/// not enumerate.
///
/// In a general solve (a solution set, [`SolveCx::all_branches`]) a
/// symbolic `arg` not known to be nonnegative also gives `W₋₁(arg)`, real for `−1/e ≤ arg < 0`: up to
/// 0.46 the real solutions of `a/x + e^(x/2) = 0` were `{2·W(−a/2)}`,
/// without `2·W₋₁(−a/2)` (for `0 < a < 2/e` both are real), and the set
/// cut to the real line drops it where it is not real.
fn lambert_branches(arena: &mut Arena, arg: ExprId, cx: &SolveCx) -> Vec<ExprId> {
    let arg = crate::transforms::eval::eval(arena, arg);
    let arg = merge_exp_factors(arena, arg);
    if cx.all_branches()
        && !crate::base::walk::free_symbols(arena, arg).is_empty()
        && crate::base::complex::query_prop(
            arena,
            arg,
            crate::base::assumptions::Props::NONNEGATIVE,
        ) != Some(true)
    {
        let principal = arena.lambertw(arg);
        let neg_one = arena.neg_one;
        let lower = arena.lambertw_branch(arg, neg_one);
        return vec![principal, lower];
    }
    if crate::base::walk::free_symbols(arena, arg).is_empty()
        && crate::poly::algebraic::sign_checked(arena, arg) == Some(-1)
    {
        let neg_one = arena.neg_one;
        let inv_e = arena.exp(neg_one);
        let shifted = arena.add(&[arg, inv_e]);
        match crate::poly::algebraic::sign_checked(arena, shifted) {
            Some(1) => {
                let principal = arena.lambertw(arg);
                let principal = exact_lambert_value(arena, arg, principal, true);
                let lower = arena.lambertw_branch(arg, neg_one);
                let lower = exact_lambert_value(arena, arg, lower, false);
                return vec![principal, lower];
            }
            // The branch point `−1/e` in a form the evaluator does not
            // recognise (`−√(4·e⁻²)/2`): `W(−1/e) = −1` on both branches,
            // rather than a `W` evaluated at its branch point.
            Some(0) => return vec![neg_one],
            _ => {}
        }
    }
    let principal = arena.lambertw(arg);
    vec![exact_lambert_value(arena, arg, principal, true)]
}

/// Largest denominator of the rational multiples `y = r·s` that
/// [`exact_lambert_value`] tries.
const LAMBERT_VALUE_MAX_DENOMINATOR: i64 = 12;

/// `w = W_k(arg)` (`principal`: `k = 0`, else `k = −1`) as the exact `y`
/// with `y·e^y = arg` when there is a simple one: `y = r·s` for a small
/// rational `r` and `s` either `1` or a logarithm `ln m` (`m` a positive
/// rational) in `arg`, `r` read off the value of `w` and then verified
/// exactly — `y·e^y − arg` is an exact zero, and `y ≥ −1` (`W₀`) or
/// `y ≤ −1` (`W₋₁`), the ranges on which `y ↦ y·e^y` is injective, so
/// `W_k(arg) = y`.  The evaluator knows `W(t·p^t·ln p)` for integers `t`,
/// `p`, not `W(√2·ln(2)/2) = ln(2)/2` or `W(−2·ln(5/2)/5) = −ln(5/2)`, and
/// before 0.45 such roots stayed as `W` values whose sign no evaluation
/// decides: `2·2ˣ + 4x − 2 = 0` for a positive `x` gave `[1/2 −
/// W(√2·ln(2)/2)/ln 2]`, which is `0`.  `w` itself when nothing is found.
fn exact_lambert_value(arena: &mut Arena, arg: ExprId, w: ExprId, principal: bool) -> ExprId {
    use crate::base::walk;
    if !matches!(
        arena.node(w),
        ExprNode::LambertW(_) | ExprNode::Apply(..) | ExprNode::Lib(..)
    ) || !walk::free_symbols(arena, arg).is_empty()
        || walk::has_unevaluated(arena, arg)
    {
        return w;
    }
    let Ok(value) = crate::transforms::evalf::evalf_complex64(arena, w) else {
        return w;
    };
    // (the values only guide the guess of `r`; `y` is verified exactly below)
    if value.im != 0.0 || !value.re.is_finite() {
        return w;
    }
    let mut scales: Vec<ExprId> = vec![arena.one];
    for id in walk::post_order_ids(arena, arg) {
        if let ExprNode::Ln(m) = *arena.node(id)
            && arena
                .as_num(m)
                .is_some_and(|q| q.is_positive() && !q.is_one())
            && !scales.contains(&id)
        {
            scales.push(id);
        }
    }
    for s in scales {
        let Ok(sv) = crate::transforms::evalf::evalf_complex64(arena, s) else {
            continue;
        };
        if sv.re == 0.0 || !sv.re.is_finite() {
            continue;
        }
        let Some(r) = small_rational_near(value.re / sv.re, LAMBERT_VALUE_MAX_DENOMINATOR) else {
            continue;
        };
        let r = arena.num_ratio(r);
        let y = arena.mul(&[r, s]);
        let y = crate::transforms::eval::eval(arena, y);
        let ey = arena.exp(y);
        let product = arena.mul(&[y, ey]);
        let diff = arena.sub(product, arg);
        let diff = crate::transforms::eval::eval(arena, diff);
        let exact = arena.is_zero_structural(diff)
            || crate::poly::algebraic::is_zero_checked(arena, diff) == Some(true);
        if !exact {
            continue;
        }
        let one = arena.one;
        let y_plus_one = arena.add(&[y, one]);
        let y_plus_one = crate::transforms::eval::eval(arena, y_plus_one);
        let on_branch = match crate::poly::algebraic::sign_checked(arena, y_plus_one) {
            Some(s) => (principal && s >= 0) || (!principal && s <= 0),
            None => false,
        };
        if on_branch {
            return y;
        }
    }
    w
}

/// The rational with denominator at most `max_den` within `10⁻¹²`
/// (relative) of `x`, the one with the least denominator (a guess that
/// [`exact_lambert_value`] verifies exactly).
fn small_rational_near(x: f64, max_den: i64) -> Option<Q> {
    if !x.is_finite() || x.abs() > 1e9 {
        return None;
    }
    for den in 1..=max_den {
        let num = (x * den as f64).round();
        if (x - num / den as f64).abs() <= 1e-12 * x.abs().max(1.0) {
            return Some(Q::new(BigInt::from(num as i64), BigInt::from(den)));
        }
    }
    None
}

/// `A·exp(B·var)` (and `A·var·exp(B·var)`): the coefficient `A` and the
/// rate `B`.
#[derive(Clone, Copy)]
struct LambertExp {
    coeff: ExprId,
    rate: ExprId,
}

/// `A·ln(B·var + E)` (and `A·var·ln(B·var + E)`): the coefficient `A`, the
/// scale `B` and the shift `E`.
#[derive(Clone, Copy)]
struct LambertLn {
    coeff: ExprId,
    scale: ExprId,
    shift: ExprId,
}

/// Classification of a single additive term for LambertW analysis.
enum LambertTermClass {
    /// `A * var` — linear in the solve variable.
    Linear(ExprId),
    /// `A * exp(B * var)` — exponential in the solve variable.
    ExpVar(LambertExp),
    /// `A * var * exp(B * var)` — mixed polynomial-exponential.
    VarExpVar(LambertExp),
    /// `A * var * ln(B * var)`.
    VarLnVar(LambertLn),
    /// `A * ln(B * var)`.
    LnVar(LambertLn),
    /// `A * var^var`.
    PowVarVar(ExprId),
}

/// Classify a single additive term (known to contain `var`) into a
/// LambertW-relevant category, or return `None` if unrecognizable.
fn classify_lambert_term(arena: &mut Arena, term: ExprId, var: ExprId) -> Option<LambertTermClass> {
    // `-(inner)`: classify `inner` and negate the coefficient.  This
    // handles cases like `-(x*exp(x))` that remain as Neg nodes rather than
    // being absorbed into a Mul with -1.
    let (term, negate) = match arena.node(term).clone() {
        ExprNode::Neg(inner) => (inner, true),
        _ => (term, false),
    };
    let factors: Vec<ExprId> = match arena.node(term).clone() {
        ExprNode::Mul(children) => children.to_vec(),
        _ => vec![term],
    };
    let mut const_factors: Vec<ExprId> = Vec::new();
    let mut has_var = false;
    let mut exp_inner: Option<ExprId> = None;
    let mut ln_inner: Option<ExprId> = None;
    let mut pow_var_var = false;
    for &child in &factors {
        if !expr_contains_var(arena, child, var) {
            const_factors.push(child);
            continue;
        }
        if child == var {
            if has_var {
                return None; // var appears twice → var²
            }
            has_var = true;
            continue;
        }
        match arena.node(child).clone() {
            ExprNode::Exp(inner) if exp_inner.is_none() => exp_inner = Some(inner),
            // a^f = exp(f·ln a) for a constant base (principal powers):
            // `2^x = 3x` as `exp(x·ln 2) = 3x` (SymPy: `solve(2**x - 3*x)` →
            // `[-LambertW(-log(2)/3)/log(2), -LambertW(-log(2)/3, -1)/log(2)]`).
            ExprNode::Pow(base, e)
                if exp_inner.is_none()
                    && !expr_contains_var(arena, base, var)
                    && !arena.is_zero_structural(base) =>
            {
                let ln_base = arena.ln(base);
                let ln_base = crate::transforms::eval::eval(arena, ln_base);
                exp_inner = Some(arena.mul(&[e, ln_base]));
            }
            ExprNode::Ln(inner) if ln_inner.is_none() => ln_inner = Some(inner),
            ExprNode::Pow(base, exp) if base == var && exp == var && !pow_var_var => {
                pow_var_var = true;
            }
            _ => return None, // unrecognized var-dependent factor
        }
    }
    let mut coeff = match const_factors.len() {
        0 => arena.one,
        1 => const_factors[0],
        _ => arena.mul(&const_factors),
    };
    if negate {
        coeff = arena.neg(coeff);
    }
    let class = match (has_var, exp_inner, ln_inner, pow_var_var) {
        (true, None, None, false) => LambertTermClass::Linear(coeff),
        (has_var, Some(inner), None, false) => {
            // exp(B·var + c) = e^c·exp(B·var) (`2^(x + 3) = 8·2^x`; refused
            // before 0.44)
            let rate = match extract_var_coeff_in_product(arena, inner, var) {
                Some(rate) => rate,
                None => {
                    let (rate, offset) = linear_rate(arena, inner, var)?;
                    let e_offset = arena.exp(offset);
                    coeff = arena.mul(&[coeff, e_offset]);
                    coeff = crate::transforms::eval::eval(arena, coeff);
                    rate
                }
            };
            let e = LambertExp { coeff, rate };
            if has_var {
                LambertTermClass::VarExpVar(e)
            } else {
                LambertTermClass::ExpVar(e)
            }
        }
        (has_var, None, Some(inner), false) => {
            let (scale, shift) = match extract_var_coeff_in_product(arena, inner, var) {
                Some(scale) => (scale, arena.zero),
                None => linear_rate(arena, inner, var)?,
            };
            let l = LambertLn {
                coeff,
                scale,
                shift,
            };
            if has_var {
                LambertTermClass::VarLnVar(l)
            } else {
                LambertTermClass::LnVar(l)
            }
        }
        (false, None, None, true) => LambertTermClass::PowVarVar(coeff),
        _ => return None,
    };
    Some(class)
}

// ═══════════════════════════════════════════════════════════════════════════
// Lambert W: a power times an exponential
// ═══════════════════════════════════════════════════════════════════════════

/// Largest `|p|` of a power `u^p` (integer `p`) whose `p`-th roots
/// [`try_solve_power_exp`] takes.
const MAX_LAMBERT_POWER: u64 = 12;

/// An additive term `K·L^a·e^(b·Mⁿ)` of [`try_solve_power_exp`]: `L` the
/// variable or a polynomial of degree 1 in it (`None` when `a = 0`), `a`
/// rational, `K` and the rate `b` free of the variable (`b = 0` without an
/// exponential), and the exponent `b·x` (`M = x`, `n = 1`) or `b·Mⁿ` with
/// `M = x + s` and `n ≥ 2` (`x·e^(x²)`, `(x + 1)·e^(−(x + 1)²)`).
#[derive(Clone)]
struct PowExpTerm {
    coeff: ExprId,
    base: Option<ExprId>,
    power: Q,
    rate: ExprId,
    /// `M`: the variable for a rate of degree 1.
    rate_base: ExprId,
    /// `n`: 1 for `e^(b·x)`, else an integer `≥ 2` or `1/m`.
    rate_power: Q,
}

/// `arg = b·Mⁿ + c₀` with `M = var + s` and `1 ≤ n ≤`
/// [`MAX_LAMBERT_POWER`] an integer or `n = 1/m` (`b·√(x + 1)`), `b ≠ 0`,
/// `b`, `s`, `c₀` free of `var`: `(b, c₀, M, n)`, with `M = var` for `n =
/// 1` (the linear rates of [`linear_rate`]).
fn monomial_rate(
    arena: &mut Arena,
    arg: ExprId,
    var: ExprId,
) -> Option<(ExprId, ExprId, ExprId, Q)> {
    let Some(coeffs) = symbolic_poly_coeffs(arena, arg, var) else {
        return radical_rate(arena, arg, var);
    };
    let n = coeffs.len().checked_sub(1)?;
    if n == 0 || n as u64 > MAX_LAMBERT_POWER {
        return None;
    }
    let lead = coeffs[n];
    if arena.is_zero_structural(lead) {
        return None;
    }
    if n == 1 {
        return Some((lead, coeffs[0], var, Q::one()));
    }
    // M = x + s with s = c_{n−1}/(n·b); every other coefficient must match.
    let n_id = arena.int(n as i64);
    let nb = arena.mul(&[n_id, lead]);
    let s = arena.div(coeffs[n - 1], nb);
    let s = crate::transforms::eval::eval(arena, s);
    let m = arena.add(&[var, s]);
    let m = crate::transforms::eval::eval(arena, m);
    let m_n = arena.pow(m, n_id);
    let model = arena.mul(&[lead, m_n]);
    let rest = arena.sub(arg, model);
    let rest = crate::transforms::expand::expand(arena, rest);
    let rest = crate::transforms::eval::eval(arena, rest);
    if expr_contains_var(arena, rest, var) {
        return None;
    }
    Some((lead, rest, m, Q::from_integer(BigInt::from(n))))
}

/// `arg = b·P^(1/m) + c₀` for one `P = α·(var + s)` with `α > 0` and an
/// integer `2 ≤ m ≤` [`MAX_LAMBERT_POWER`]: `(b·α^(1/m), c₀, var + s, 1/m)`
/// ([`monomial_rate`]).
fn radical_rate(
    arena: &mut Arena,
    arg: ExprId,
    var: ExprId,
) -> Option<(ExprId, ExprId, ExprId, Q)> {
    let mut radical: Option<(ExprId, ExprId, Q)> = None;
    for id in crate::base::walk::post_order_ids(arena, arg) {
        if let ExprNode::Pow(p, e) = *arena.node(id)
            && expr_contains_var(arena, p, var)
        {
            let r = arena.as_num(e)?.clone();
            match &radical {
                Some((node, _, _)) if *node != id => return None,
                _ => radical = Some((id, p, r)),
            }
        }
    }
    let (node, p, r) = radical?;
    if !r.numer().is_one() || r.denom() < &BigInt::from(2) {
        return None;
    }
    let m = num_traits::ToPrimitive::to_u64(r.denom())?;
    if m > MAX_LAMBERT_POWER {
        return None;
    }
    let (alpha, beta) = linear_rate(arena, p, var)?;
    if !crate::base::walk::free_symbols(arena, alpha).is_empty()
        || crate::poly::algebraic::sign_checked(arena, alpha) != Some(1)
    {
        return None;
    }
    let y = arena.symbol("__y_radical_rate");
    let sub = arena.subs_structural(arg, node, y);
    let sub = crate::transforms::eval::eval(arena, sub);
    if expr_contains_var(arena, sub, var) {
        return None;
    }
    let (b, c0) = linear_rate(arena, sub, y)?;
    let s = arena.div(beta, alpha);
    let rate_base = arena.add(&[var, s]);
    let rate_base = crate::transforms::eval::eval(arena, rate_base);
    let r_id = arena.num_ratio(r.clone());
    let alpha_r = arena.pow(alpha, r_id);
    let b = arena.mul(&[b, alpha_r]);
    let b = crate::transforms::eval::eval(arena, b);
    Some((b, c0, rate_base, r))
}

/// `b = content·L^m` with `L` of degree 1 in `var` (`var` itself, `2x + 1`,
/// `x + a`), or a polynomial over ℚ whose square-free decomposition is one
/// linear factor (`x² + 2x + 1 = (x + 1)²`).
fn linear_power_base(arena: &mut Arena, b: ExprId, var: ExprId) -> Option<(ExprId, u32, Q)> {
    if b == var {
        return Some((var, 1, Q::one()));
    }
    if let Some(p) = polybridge::expr_to_poly(arena, b, var) {
        match p.degree()? {
            0 => return None,
            1 => return Some((b, 1, Q::one())),
            _ => {}
        }
        let (content, factors) = p.sqf_list();
        let [(l, m)] = factors.as_slice() else {
            return None;
        };
        if l.degree() != Some(1) {
            return None;
        }
        let l_expr = polybridge::poly_to_expr(arena, l, var);
        let l_expr = crate::transforms::eval::eval(arena, l_expr);
        return Some((l_expr, *m, content));
    }
    linear_rate(arena, b, var)?;
    Some((b, 1, Q::one()))
}

/// Split a term containing `var` as `K·L^a·e^(b·x)` ([`PowExpTerm`]):
/// powers of the variable or of one linear polynomial, exponentials
/// `exp(a·x + c)` and constant-base powers `c^(a·x + d)` (`c` positive or
/// symbolic, as `exp((a·x + d)·ln c)`).  `None` for any other factor.
fn pow_exp_term(arena: &mut Arena, term: ExprId, var: ExprId) -> Option<PowExpTerm> {
    let (term, negate) = match *arena.node(term) {
        ExprNode::Neg(inner) => (inner, true),
        _ => (term, false),
    };
    let factors: Vec<ExprId> = match arena.node(term) {
        ExprNode::Mul(ch) => ch.to_vec(),
        _ => vec![term],
    };
    let mut consts: Vec<ExprId> = Vec::new();
    if negate {
        consts.push(arena.neg_one);
    }
    let mut base: Option<ExprId> = None;
    let mut power = Q::zero();
    let mut rates: Vec<ExprId> = Vec::new();
    // (M, n) of the exponents `b·Mⁿ`, one for all exponential factors
    let mut rate_shape: Option<(ExprId, Q)> = None;
    let mut same_shape = |shape: (ExprId, Q)| match &rate_shape {
        Some(s) => *s == shape,
        None => {
            rate_shape = Some(shape);
            true
        }
    };
    for f in factors {
        if !expr_contains_var(arena, f, var) {
            consts.push(f);
            continue;
        }
        let (b, e) = match arena.node(f).clone() {
            ExprNode::Exp(arg) => {
                let (a, c0, m, n) = monomial_rate(arena, arg, var)?;
                if !same_shape((m, n)) {
                    return None;
                }
                rates.push(a);
                consts.push(arena.exp(c0));
                continue;
            }
            ExprNode::Pow(c, e) if !expr_contains_var(arena, c, var) => {
                let positive = if crate::base::walk::free_symbols(arena, c).is_empty() {
                    crate::poly::algebraic::sign_checked(arena, c) == Some(1)
                } else {
                    true
                };
                if !positive {
                    return None;
                }
                let (a, c0, m, n) = monomial_rate(arena, e, var)?;
                if !same_shape((m, n)) {
                    return None;
                }
                let ln_c = arena.ln(c);
                let ln_c = crate::transforms::eval::eval(arena, ln_c);
                rates.push(arena.mul(&[a, ln_c]));
                consts.push(arena.pow(c, c0));
                continue;
            }
            ExprNode::Pow(b, e) if !expr_contains_var(arena, e, var) => {
                (b, arena.as_num(e)?.clone())
            }
            _ => (f, Q::one()),
        };
        let (l, m, content) = linear_power_base(arena, b, var)?;
        // (c·L^m)^e = c^e·L^(m·e) for an integer e only (principal powers)
        if (m != 1 || !content.is_one()) && !e.is_integer() {
            return None;
        }
        match base {
            Some(prev) if prev != l => return None,
            _ => base = Some(l),
        }
        power += &e * Q::from_integer(BigInt::from(m));
        if !content.is_one() {
            let c_id = arena.num_ratio(content);
            let e_id = arena.num_ratio(e);
            consts.push(arena.pow(c_id, e_id));
        }
    }
    let coeff = if consts.is_empty() {
        arena.one
    } else {
        let k = arena.mul(&consts);
        crate::transforms::eval::eval(arena, k)
    };
    let rate = if rates.is_empty() {
        arena.zero
    } else {
        let r = arena.add(&rates);
        crate::transforms::eval::eval(arena, r)
    };
    if power.is_zero() {
        base = None;
    }
    let (rate_base, rate_power) = rate_shape.unwrap_or((var, Q::one()));
    Some(PowExpTerm {
        coeff,
        base,
        power,
        rate,
        rate_base,
        rate_power,
    })
}

/// Merge the terms of one rate into one: `x·eˣ + eˣ = (x + 1)·eˣ`.  The
/// terms of a rate must then be `Kᵢ·xⁿⁱ` (nonnegative integers `nᵢ`,
/// rational ratios `Kᵢ/K₀`), and their polynomial `c·L^m` for one linear
/// `L`; `None` otherwise.
fn merge_by_rate(
    arena: &mut Arena,
    terms: Vec<PowExpTerm>,
    var: ExprId,
) -> Option<Vec<PowExpTerm>> {
    let mut groups: Vec<Vec<PowExpTerm>> = Vec::new();
    for t in terms {
        let same = |g: &&mut Vec<PowExpTerm>| {
            g[0].rate == t.rate
                && (arena.is_zero_structural(t.rate)
                    || (g[0].rate_base == t.rate_base && g[0].rate_power == t.rate_power))
        };
        match groups.iter_mut().find(same) {
            Some(g) => g.push(t),
            None => groups.push(vec![t]),
        }
    }
    let mut out = Vec::with_capacity(groups.len());
    for mut g in groups {
        if g.len() == 1 {
            out.push(g.pop()?);
            continue;
        }
        let k0 = g[0].coeff;
        let rate = g[0].rate;
        let (rate_base, rate_power) = (g[0].rate_base, g[0].rate_power.clone());
        if !rate_power.is_one() {
            return None;
        }
        let mut p = Poly::zero();
        for t in &g {
            if t.base.is_some_and(|b| b != var) || !t.power.is_integer() || t.power.is_negative() {
                return None;
            }
            let ratio = arena.div(t.coeff, k0);
            let ratio = crate::transforms::eval::eval(arena, ratio);
            let r = arena.as_num(ratio)?.clone();
            let deg = num_traits::ToPrimitive::to_usize(&t.power.to_integer())?;
            if deg > MAX_SUBSTITUTION_DEGREE as usize {
                return None;
            }
            p = p.add(&Poly::monomial(r, deg));
        }
        let (content, factors) = p.sqf_list();
        if content.is_zero() {
            return None;
        }
        let content_id = arena.num_ratio(content);
        let coeff = arena.mul(&[k0, content_id]);
        let coeff = crate::transforms::eval::eval(arena, coeff);
        match factors.as_slice() {
            [] => out.push(PowExpTerm {
                coeff,
                base: None,
                power: Q::zero(),
                rate,
                rate_base,
                rate_power: rate_power.clone(),
            }),
            [(l, m)] if l.degree() == Some(1) => {
                let l_expr = polybridge::poly_to_expr(arena, l, var);
                let l_expr = crate::transforms::eval::eval(arena, l_expr);
                out.push(PowExpTerm {
                    coeff,
                    base: Some(l_expr),
                    power: Q::from_integer(BigInt::from(*m)),
                    rate,
                    rate_base,
                    rate_power: rate_power.clone(),
                });
            }
            _ => return None,
        }
    }
    Some(out)
}

/// `K·L^a·e^(b·Mⁿ) = c` with `L = α·x + β`, `M = x + s`, `n ≥ 2` an
/// integer or `n = 1/m` ([`power_exp_monomial_solutions`]).
struct PowExpMonomial<'q> {
    k: ExprId,
    a: &'q Q,
    b: ExprId,
    c: ExprId,
    alpha: ExprId,
    beta: ExprId,
    rate_base: ExprId,
    n: &'q Q,
}

/// The candidates of `K·L^a·e^(b·Mⁿ) = c` ([`PowExpMonomial`]) when `L`
/// is a constant multiple of `M` (`x·e^(x²) = c`, `(2x + 2)²·e^(−(x +
/// 1)²) = c`): with `u = L`, `e^(b·Mⁿ) = e^(b′·uⁿ)` for `b′ = b/αⁿ`, and
/// [`power_exp_monomial_roots`] gives `u`; `x = (u − β)/α`.  For `a = 0`
/// the exponential alone is inverted.  `root_of_base` adds `L = 0` (the
/// two-term form with both powers positive).  Every candidate is checked
/// against the equation by [`solve_impl`].
fn power_exp_monomial_solutions(
    arena: &mut Arena,
    eq: &PowExpMonomial<'_>,
    root_of_base: bool,
    var: ExprId,
    cx: &SolveCx,
) -> Option<Vec<Solution>> {
    let mut out: Vec<Solution> = Vec::new();
    let mut push = |arena: &mut Arena, value: ExprId| {
        let value = crate::transforms::eval::eval(arena, value);
        if !out.iter().any(|s| s.value == value) {
            out.push(Solution { value });
        }
    };
    let n_id = arena.num_ratio(eq.n.clone());
    if eq.a.is_zero() {
        let m_n = arena.pow(eq.rate_base, n_id);
        let exponent = arena.mul(&[eq.b, m_n]);
        let generator = arena.exp(exponent);
        let c_over_k = arena.div(eq.c, eq.k);
        let c_over_k = crate::transforms::eval::eval(arena, c_over_k);
        for s in peeled_candidates(solve_by_peeling(arena, generator, c_over_k, var, cx))? {
            push(arena, s);
        }
    } else {
        // L = α·M exactly: β = α·s
        let s = arena.sub(eq.rate_base, var);
        let alpha_s = arena.mul(&[eq.alpha, s]);
        let gap = arena.sub(eq.beta, alpha_s);
        let gap = crate::transforms::expand::expand(arena, gap);
        let gap = crate::transforms::eval::eval(arena, gap);
        if !arena.is_zero_structural(gap) {
            return None;
        }
        // (α·u)^(1/m) = α^(1/m)·u^(1/m) for a positive α only
        if !eq.n.is_integer() && crate::poly::algebraic::sign_checked(arena, eq.alpha) != Some(1) {
            return None;
        }
        let alpha_n = arena.pow(eq.alpha, n_id);
        let b_u = arena.div(eq.b, alpha_n);
        let b_u = crate::transforms::eval::eval(arena, b_u);
        let c_u = arena.div(eq.c, eq.k);
        let c_u = crate::transforms::eval::eval(arena, c_u);
        for u in power_exp_monomial_roots(arena, eq.a, b_u, eq.n, c_u, cx)? {
            let shifted = arena.sub(u, eq.beta);
            let x = arena.div(shifted, eq.alpha);
            push(arena, x);
        }
    }
    if root_of_base {
        let neg_beta = arena.neg(eq.beta);
        let x0 = arena.div(neg_beta, eq.alpha);
        push(arena, x0);
    }
    Some(out)
}

/// The candidates `u` of `u^a·e^(b·uⁿ) = c` for a rational `a = p/q ≠ 0`
/// and an integer `n ≥ 2`.  Raised to the power `q·n` (integer powers, so
/// every root of the equation solves the result) it is `w^p·e^(b·q·n·w) =
/// c^(q·n)` in `w = uⁿ`, solved as in [`try_solve_power_exp`] (the values
/// `r` with `r^p = c^(q·n)` of [`power_roots`], then `w = (p/b″)·W_k((b″/p)·r)`
/// with `b″ = b·q·n`).  For `a = ±1` each `w` gives the one root `u =
/// (c·e^(−b·w))^(1/a)` (then `uⁿ = w` and `u^a·e^(b·uⁿ) = c` exactly):
/// SymPy's `solve(x*exp(x**2) - c)` → `[c*exp(-LambertW(2*c**2)/2)]`.
/// When `a = k·n` for an integer `k` no power is raised: `u^a = w^k` and
/// `w^k·e^(b·w) = c` is the linear form, every real `n`-th root of each
/// of its `w` a root (`x²·e^(x²) = c`: `±√W(c)`).  Otherwise the
/// candidates are the real `n`-th roots of `w` (both signs for an even
/// `n`), some of which may solve `u^a·e^(b·uⁿ) = ζ·c` for a root of unity
/// `ζ ≠ 1` instead; [`solve_impl`] checks them against the equation
/// (`x³·e^(x²) = 5` keeps `+√w` only), so that case needs a constant
/// `c` and `b` (`None` otherwise: a candidate with parameters is kept
/// unchecked).
///
/// For `n = 1/m` the unknown is `w = u^(1/m)` (principal), `u = w^m` and
/// `u^a = w^(m·a)` on the principal sector of `w`: `w^(m·a)·e^(b·w) = c`
/// is the linear form, `w = (m·a/b)·W_k((b/(m·a))·r)` for `r^(m·a) = c`,
/// and the candidates `u = w^m` are checked (`x²·e^(√x/2) = 2`); a `w`
/// outside the principal sector gives a non-root, so again `c` and `b`
/// must be constants.
///
/// Before 0.45 all these were refused ("not polynomial").
fn power_exp_monomial_roots(
    arena: &mut Arena,
    a: &Q,
    b: ExprId,
    n: &Q,
    c: ExprId,
    cx: &SolveCx,
) -> Option<Vec<ExprId>> {
    let constant = crate::base::walk::free_symbols(arena, b).is_empty()
        && crate::base::walk::free_symbols(arena, c).is_empty();
    if n.is_integer() && a.is_integer() {
        let n_int = n.to_integer();
        let a_int = a.to_integer();
        if !n_int.is_zero() && (&a_int % &n_int).is_zero() {
            // u^a = w^k exactly, k = a/n
            let k = Q::from_integer(&a_int / &n_int);
            let n = num_traits::ToPrimitive::to_u32(&n_int)?;
            let k_id = arena.num_ratio(k.clone());
            let b_over_k = arena.div(b, k_id);
            let k_over_b = arena.div(k_id, b);
            let mut out: Vec<ExprId> = Vec::new();
            for r in power_roots(arena, c, &k)? {
                let arg = arena.mul(&[b_over_k, r]);
                for wk in lambert_branches(arena, arg, cx) {
                    let w = arena.mul(&[k_over_b, wk]);
                    let w = crate::transforms::eval::eval(arena, w);
                    for u in real_nth_roots(arena, w, n) {
                        let u = crate::transforms::eval::eval(arena, u);
                        if !out.contains(&u) {
                            out.push(u);
                        }
                    }
                }
            }
            return Some(out);
        }
    }
    if !n.is_integer() {
        if !n.numer().is_one() || !constant {
            return None;
        }
        let m = n.denom().clone();
        let am = a * Q::from_integer(m.clone());
        let am_id = arena.num_ratio(am.clone());
        let b_over_am = arena.div(b, am_id);
        let am_over_b = arena.div(am_id, b);
        let m_id = arena.big_int(m);
        let mut out: Vec<ExprId> = Vec::new();
        for r in power_roots(arena, c, &am)? {
            let arg = arena.mul(&[b_over_am, r]);
            for wk in lambert_branches(arena, arg, cx) {
                let w = arena.mul(&[am_over_b, wk]);
                let u = arena.pow(w, m_id);
                let u = crate::transforms::eval::eval(arena, u);
                if !out.contains(&u) {
                    out.push(u);
                }
            }
        }
        return Some(out);
    }
    let n = num_traits::ToPrimitive::to_u32(n.numer())?;
    let p = num_traits::ToPrimitive::to_i64(a.numer())?;
    let q = num_traits::ToPrimitive::to_i64(a.denom())?;
    let qn = q.checked_mul(i64::from(n))?;
    if p == 0 || qn.unsigned_abs() > MAX_LAMBERT_POWER {
        return None;
    }
    let qn_id = arena.int(qn);
    let p_id = arena.int(p);
    let b2 = arena.mul(&[b, qn_id]);
    let b2 = crate::transforms::eval::eval(arena, b2);
    let c2 = arena.pow(c, qn_id);
    let c2 = crate::transforms::eval::eval(arena, c2);
    let b2_over_p = arena.div(b2, p_id);
    let p_over_b2 = arena.div(p_id, b2);
    let unit = q == 1 && p.abs() == 1;
    if !unit && !constant {
        return None;
    }
    let mut out: Vec<ExprId> = Vec::new();
    for r in power_roots(arena, c2, &Q::from_integer(BigInt::from(p)))? {
        let arg = arena.mul(&[b2_over_p, r]);
        for wk in lambert_branches(arena, arg, cx) {
            let w = arena.mul(&[p_over_b2, wk]);
            let w = crate::transforms::eval::eval(arena, w);
            let candidates: Vec<ExprId> = if unit {
                let bw = arena.mul(&[b, w]);
                let u = if p == 1 {
                    let minus_bw = arena.neg(bw);
                    let e = arena.exp(minus_bw);
                    arena.mul(&[c, e])
                } else {
                    let e = arena.exp(bw);
                    arena.div(e, c)
                };
                vec![u]
            } else {
                real_nth_roots(arena, w, n)
            };
            for u in candidates {
                let u = crate::transforms::eval::eval(arena, u);
                if !out.contains(&u) {
                    out.push(u);
                }
            }
        }
    }
    Some(out)
}

/// The `n`-th roots of `w` (`n ≥ 2`) that [`power_exp_monomial_roots`]
/// tries: the real ones of a real constant `w` that has some (`±w^(1/n)`
/// for a positive `w` and an even `n`, the one real root for an odd `n`),
/// all `n` complex ones of a constant without a real one (a negative `w`
/// and an even `n`, a non-real `w`), as [`power_roots`] does, and
/// `±w^(1/n)` (principal) for a symbolic `w`.
fn real_nth_roots(arena: &mut Arena, w: ExprId, n: u32) -> Vec<ExprId> {
    let even = n.is_multiple_of(2);
    let n_i64 = i64::from(n);
    let constant = crate::base::walk::free_symbols(arena, w).is_empty();
    let sign = if constant {
        crate::poly::algebraic::sign_checked(arena, w)
    } else {
        None
    };
    match sign {
        Some(0) => vec![arena.zero],
        Some(1) => {
            let r = positive_root(arena, w, n_i64);
            if even {
                let minus = arena.neg(r);
                vec![r, minus]
            } else {
                vec![r]
            }
        }
        Some(_) if !even => {
            let minus_w = arena.neg(w);
            let minus_w = crate::transforms::eval::eval(arena, minus_w);
            let r = positive_root(arena, minus_w, n_i64);
            vec![arena.neg(r)]
        }
        _ if constant => match usize::try_from(n) {
            Ok(k) => binomial_roots(arena, w, k)
                .into_iter()
                .map(|s| s.value)
                .collect(),
            Err(_) => Vec::new(),
        },
        _ => {
            let inv_n = arena.rational(1, n_i64);
            let r = arena.pow(w, inv_n);
            if even {
                let minus = arena.neg(r);
                vec![r, minus]
            } else {
                vec![r]
            }
        }
    }
}

/// The values `r` with `r^a = c` that [`try_solve_power_exp`] takes, for a
/// rational `a = p/q`: for `q = 1` every real `|p|`-th root of `c` (or of
/// `1/c` for `p < 0`) when `c` is a real constant with one (`±√c` for an
/// even `p` and `c > 0`, the real root for an odd `p`), all `|p|` complex
/// roots when it has none (`c < 0`, `p` even, or `c` not real), and `c^(1/p)`
/// (with `−c^(1/p)` for an even `p`) for a symbolic `c`; for `q > 1` the
/// principal `c^(1/a)`, the one positive value of a principal power.
/// `None` beyond [`MAX_LAMBERT_POWER`].
fn power_roots(arena: &mut Arena, c: ExprId, a: &Q) -> Option<Vec<ExprId>> {
    if !a.denom().is_one() {
        let inv = arena.num_ratio(a.recip());
        let r = arena.pow(c, inv);
        return Some(vec![crate::transforms::eval::eval(arena, r)]);
    }
    let p = num_traits::ToPrimitive::to_i64(a.numer())?;
    let m = p.unsigned_abs();
    if m == 0 || m > MAX_LAMBERT_POWER {
        return None;
    }
    let target = if p < 0 {
        let one = arena.one;
        arena.div(one, c)
    } else {
        c
    };
    let target = crate::transforms::eval::eval(arena, target);
    if m == 1 {
        return Some(vec![target]);
    }
    let m_i64 = i64::try_from(m).ok()?;
    let inv_m = arena.rational(1, m_i64);
    let even = m % 2 == 0;
    let constant = crate::base::walk::free_symbols(arena, target).is_empty();
    let sign = if constant {
        crate::poly::algebraic::sign_checked(arena, target)
    } else {
        None
    };
    let mut out = Vec::new();
    match sign {
        Some(1) => {
            let r = positive_root(arena, target, m_i64);
            out.push(r);
            if even {
                out.push(arena.neg(r));
            }
        }
        Some(-1) if !even => {
            let minus = arena.neg(target);
            let minus = crate::transforms::eval::eval(arena, minus);
            let r = positive_root(arena, minus, m_i64);
            out.push(arena.neg(r));
        }
        Some(0) => return None,
        _ if constant => {
            let n = usize::try_from(m).ok()?;
            out.extend(
                binomial_roots(arena, target, n)
                    .into_iter()
                    .map(|s| s.value),
            );
        }
        // A symbolic `c` and an odd `m ≥ 3`: every root (the real one of a
        // real `c` is not the principal one when `c < 0`): up to 0.46 the
        // real solutions of `e^(x/2)/4 + 2a/x³ = 0` were `{6·W(∛(−8a)/6)}`,
        // not real for `a > 0`, without the two real roots `6·Wₖ(−∛a/3)`
        // (SymPy's `solve` gives `W₀` of the three cube roots).
        _ if !even => {
            let n = usize::try_from(m).ok()?;
            out.extend(
                binomial_roots(arena, target, n)
                    .into_iter()
                    .map(|s| s.value),
            );
        }
        _ => {
            let r = arena.pow(target, inv_m);
            out.push(r);
            if even {
                out.push(arena.neg(r));
            }
        }
    }
    Some(
        out.into_iter()
            .map(|r| crate::transforms::eval::eval(arena, r))
            .collect(),
    )
}

/// `t^(1/m)` for a positive constant `t`, distributed over the factors of
/// a product of positive constants (`(4·e⁻²)^(1/2) = 2·e⁻¹`, which the
/// evaluator leaves as `√(4·e⁻²)`), `exp(u)^(1/m)` as `exp(u/m)`.
fn positive_root(arena: &mut Arena, t: ExprId, m: i64) -> ExprId {
    let inv_m = arena.rational(1, m);
    let factors: Vec<ExprId> = match arena.node(t) {
        ExprNode::Mul(ch) => ch.to_vec(),
        _ => vec![t],
    };
    let mut parts: Vec<ExprId> = Vec::with_capacity(factors.len());
    for f in factors {
        if crate::poly::algebraic::sign_checked(arena, f) != Some(1) {
            let r = arena.pow(t, inv_m);
            return crate::transforms::eval::eval(arena, r);
        }
        let node = arena.node(f).clone();
        parts.push(match node {
            ExprNode::Exp(u) => {
                let v = arena.mul(&[inv_m, u]);
                arena.exp(v)
            }
            ExprNode::Pow(b, e) if crate::poly::algebraic::sign_checked(arena, b) == Some(1) => {
                let v = arena.mul(&[inv_m, e]);
                arena.pow(b, v)
            }
            _ => arena.pow(f, inv_m),
        });
    }
    let r = arena.mul(&parts);
    crate::transforms::eval::eval(arena, r)
}

/// Equations `K·L^a·e^(b·x) = c` and `K₁·L^a₁·e^(b₁·x) + K₂·L^a₂·e^(b₂·x) =
/// 0` with `L` the variable or linear in it, `a` rational and `b ≠ 0` (an
/// exponential `e^(b·x)` or a power `p^x = e^(x·ln p)`): `x² = 2ˣ`,
/// `x²·eˣ = 3`, `xⁿ = c·eˣ`, `(x + 1)·eˣ = 2`, `√x = 2ˣ/4`, `x⁻²·eˣ = 3`.
/// Terms of one rate are merged first (`x·eˣ + eˣ = (x + 1)·eˣ`).  With
/// `u = L`, `u^a·e^(b′·u) = c′` gives `u·e^((b′/a)·u) = r` for each `r` with
/// `r^a = c′` ([`power_roots`]: both `±√c′` of `x²·eˣ = c′`), and
/// `u = (a/b′)·W_k((b′/a)·r)` on the branches of [`lambert_branches`] (SymPy's
/// `_solve_lambert` and `_lambert`, `sympy/solvers/bivariate.py`, BSD, take
/// the logarithm of both sides and the `p` roots of `t^p = e^(A)` the same
/// way).  In the two-term form the root `L = 0` is a candidate as well when
/// both powers are positive (`x²·eˣ = x³`).  Every candidate is checked
/// against the equation by [`solve_impl`].
///
/// Before 0.44 these were refused: `solve(x² − 2ˣ)` (SymPy `[2, 4,
/// −2·W(ln(2)/2)/ln 2]`), `solve(x²·eˣ − 3)` (SymPy `[2·W(−√3/2),
/// 2·W(√3/2)]`), `solve(x³ − 3ˣ)`, `solve((x + 1)·eˣ − 2)`.
pub(super) fn try_solve_power_exp(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Solutions {
    candidates_found(power_exp_candidates(arena, expr, var, cx))
}

/// The candidates of [`try_solve_power_exp`]; `None` when the equation
/// does not have its shape.
fn power_exp_candidates(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Option<Vec<Solution>> {
    let terms: Vec<ExprId> = match arena.node(expr) {
        ExprNode::Add(ch) => ch.to_vec(),
        _ => vec![expr],
    };
    let mut constants: Vec<ExprId> = Vec::new();
    let mut parts: Vec<PowExpTerm> = Vec::new();
    for t in terms {
        if expr_contains_var(arena, t, var) {
            parts.push(pow_exp_term(arena, t, var)?);
        } else {
            constants.push(t);
        }
    }
    let parts = merge_by_rate(arena, parts, var)?;
    let d = match constants.len() {
        0 => arena.zero,
        _ => {
            let s = arena.add(&constants);
            crate::transforms::eval::eval(arena, s)
        }
    };
    let d_zero = arena.is_zero_structural(d);
    // K·L^a·e^(b·x) = c, and the root L = 0 of the two-term form
    let shape: (ExprId, Q);
    let (k, base, a, b, c, root_of_base) = match (parts.as_slice(), d_zero) {
        ([t], false) => {
            shape = (t.rate_base, t.rate_power.clone());
            (
                t.coeff,
                t.base,
                t.power.clone(),
                t.rate,
                arena.neg(d),
                false,
            )
        }
        ([t1, t2], true) => {
            let base = match (t1.base, t2.base) {
                (Some(p), Some(q)) if p != q => return None,
                (p, q) => p.or(q),
            };
            let s1 = (t1.rate_base, t1.rate_power.clone());
            let s2 = (t2.rate_base, t2.rate_power.clone());
            shape = match (
                arena.is_zero_structural(t1.rate),
                arena.is_zero_structural(t2.rate),
            ) {
                (true, _) => s2,
                (false, true) => s1,
                (false, false) if s1 == s2 => s1,
                _ => return None,
            };
            let both_vanish = t1.power.is_positive() && t2.power.is_positive();
            let rate = arena.sub(t1.rate, t2.rate);
            let rate = crate::transforms::eval::eval(arena, rate);
            let ratio = arena.div(t2.coeff, t1.coeff);
            let c = arena.neg(ratio);
            (arena.one, base, &t1.power - &t2.power, rate, c, both_vanish)
        }
        _ => return None,
    };
    let c = crate::transforms::eval::eval(arena, c);
    let base = base?;
    if arena.is_zero_structural(b) || arena.is_zero_structural(c) {
        return None;
    }
    let (alpha, beta) = linear_rate(arena, base, var)?;
    if !shape.1.is_one() {
        let eq = PowExpMonomial {
            k,
            a: &a,
            b,
            c,
            alpha,
            beta,
            rate_base: shape.0,
            n: &shape.1,
        };
        return power_exp_monomial_solutions(arena, &eq, root_of_base, var, cx);
    }
    let root_of_base = root_of_base.then(|| {
        let neg_beta = arena.neg(beta);
        let x0 = arena.div(neg_beta, alpha);
        crate::transforms::eval::eval(arena, x0)
    });
    if a.is_zero() {
        // L^a·(K₁·e^(b₁·x) + K₂·e^(b₂·x)) = 0: e^(b·x) = c, and L = 0
        // (`x²·2ˣ − x²/10`).
        let bx = arena.mul(&[b, var]);
        let generator = arena.exp(bx);
        let mut out = peeled_candidates(solve_by_peeling(arena, generator, c, var, cx))?;
        if let Some(x0) = root_of_base
            && !out.contains(&x0)
        {
            out.push(x0);
        }
        return Some(out.into_iter().map(|value| Solution { value }).collect());
    }
    // u = L = α·x + β:  u^a·e^(b′·u) = c′,  b′ = b/α,  c′ = (c/K)·e^(b·β/α)
    let b1 = arena.div(b, alpha);
    let b1 = crate::transforms::eval::eval(arena, b1);
    let shift = arena.mul(&[b, beta]);
    let shift = arena.div(shift, alpha);
    let e_shift = arena.exp(shift);
    let c_over_k = arena.div(c, k);
    let c1 = arena.mul(&[c_over_k, e_shift]);
    let c1 = crate::transforms::eval::eval(arena, c1);
    let a_id = arena.num_ratio(a.clone());
    let b1_over_a = arena.div(b1, a_id);
    let a_over_b1 = arena.div(a_id, b1);
    let mut out: Vec<Solution> = Vec::new();
    let mut push = |arena: &mut Arena, value: ExprId| {
        let value = crate::transforms::eval::eval(arena, value);
        if !out.iter().any(|s| s.value == value) {
            out.push(Solution { value });
        }
    };
    for r in power_roots(arena, c1, &a)? {
        let arg = arena.mul(&[b1_over_a, r]);
        for w in lambert_branches(arena, arg, cx) {
            let u = arena.mul(&[a_over_b1, w]);
            let shifted = arena.sub(u, beta);
            let x = arena.div(shifted, alpha);
            push(arena, x);
        }
    }
    if let Some(x0) = root_of_base {
        push(arena, x0);
    }
    Some(out)
}
