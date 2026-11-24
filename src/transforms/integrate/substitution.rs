//! Substitution strategies (v0.2 gap-fill) (part of [`integrate`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Substitution strategies (v0.2 gap-fill)
// ═══════════════════════════════════════════════════════════════════════════

thread_local! {
    /// Nesting depth of substitution strategies that re-enter [`integrate`].
    pub(super) static SUBST_DEPTH: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}

/// Maximum nesting of substitution strategies.
const MAX_SUBST_DEPTH: u8 = 3;

thread_local! {
    /// Set while a rationalising substitution integrates its transformed
    /// integrand ([`integrate_nested_generic`]): no `Piecewise` cases there.
    pub(super) static GENERIC_ONLY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// [`integrate_nested`] for the integrand in the new variable of a
/// substitution whose answer is checked and wrapped in `x`: its generic
/// answer only.  The cases of the parameters there are cases of the
/// substitution (`b = 0` in `x = (t² − a)/b`), and the wrap of the outer
/// call finds the integrand's own; computing both nested a dozen
/// `Piecewise` levels and took seconds.
pub(super) fn integrate_nested_generic(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
) -> Option<ExprId> {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            let _ = GENERIC_ONLY.try_with(|g| g.set(self.0));
        }
    }
    let restore = Restore(GENERIC_ONLY.with(|g| g.replace(true)));
    let result = integrate_nested(arena, expr, var);
    drop(restore);
    result
}

/// Run the full integration pipeline on a transformed integrand from
/// inside a substitution strategy, with a re-entrancy bound.
pub(super) fn integrate_nested(arena: &mut Arena, expr: ExprId, var: ExprId) -> Option<ExprId> {
    /// Restores the substitution depth on drop, also when a panic unwinds.
    struct Restore(u8);
    impl Drop for Restore {
        fn drop(&mut self) {
            let _ = SUBST_DEPTH.try_with(|d| d.set(self.0));
        }
    }
    let depth = SUBST_DEPTH.with(|d| d.get());
    if depth >= MAX_SUBST_DEPTH {
        return None;
    }
    SUBST_DEPTH.with(|d| d.set(depth + 1));
    let restore = Restore(depth);
    let result = integrate(arena, expr, var);
    drop(restore);
    if crate::base::walk::has_unevaluated(arena, result) {
        None
    } else {
        Some(result)
    }
}

/// Try the substitution-based strategies in order.
pub(super) fn try_substitution_strategies(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    if let Some(r) = stage!(
        arena,
        "trig_rational",
        expr,
        try_trig_rational_substitution(arena, expr, var, var_sym)
    ) {
        return Some(r);
    }
    if let Some(r) = try_exp_rational_substitution(arena, expr, var, var_sym) {
        return Some(r);
    }
    if let Some(r) = try_exp_linear_substitution(arena, expr, var, var_sym) {
        return Some(r);
    }
    if let Some(r) = try_exp_imaginary_to_trig(arena, expr, var, var_sym) {
        return Some(r);
    }
    if let Some(r) = try_radical_substitution(arena, expr, var, var_sym) {
        return Some(r);
    }
    if let Some(r) = try_even_power_radicals(arena, expr, var, var_sym) {
        return Some(r);
    }
    if let Some(r) = try_mobius_radical_substitution(arena, expr, var, var_sym) {
        return Some(r);
    }
    if let Some(r) = try_exp_atanh_rewrite(arena, expr, var, var_sym) {
        return Some(r);
    }
    if let Some(r) = try_hyperbolic_to_exp(arena, expr, var, var_sym) {
        return Some(r);
    }
    if let Some(r) = stage!(
        arena,
        "log_substitution",
        expr,
        try_log_substitution(arena, expr, var, var_sym, true)
    ) {
        return Some(r);
    }
    if let Some(r) = stage!(
        arena,
        "exp_substitution",
        expr,
        try_exp_general_substitution(arena, expr, var, var_sym)
    ) {
        return Some(r);
    }
    // Last, and for the integrand itself only: a `Piecewise` answer is for
    // real values of the variable, and the routes above (or the caller of
    // a substitution, in its own variable) may have an analytic one
    // (`∫ √x·√(1/x) dx`).
    if SUBST_DEPTH.with(std::cell::Cell::get) == 0
        && let Some(r) = stage!(
            arena,
            "piecewise_split",
            expr,
            try_piecewise_split(arena, expr, var, var_sym)
        )
    {
        return Some(r);
    }
    None
}

/// `(c, n)` with `g = c·xⁿ` for `c`, `n` free of `x`, `n ≠ 0` (`x` itself,
/// `xⁿ`, `c·x`, `c·xⁿ`).
pub(super) fn monomial_parts(
    arena: &mut Arena,
    g: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<(ExprId, ExprId)> {
    let power = |arena: &Arena, f: ExprId| -> Option<ExprId> {
        if f == var {
            return Some(arena.one);
        }
        match arena.node(f) {
            ExprNode::Pow(b, e)
                if *b == var
                    && !contains_var(arena, *e, var_sym)
                    && !arena.is_zero_structural(*e) =>
            {
                Some(*e)
            }
            _ => None,
        }
    };
    if let Some(n) = power(arena, g) {
        return Some((arena.one, n));
    }
    let ExprNode::Mul(factors) = arena.node(g).clone() else {
        return None;
    };
    let mut n: Option<ExprId> = None;
    let mut rest: SmallVec<[ExprId; 4]> = SmallVec::new();
    for f in factors {
        if let Some(k) = power(arena, f) {
            if n.replace(k).is_some() {
                return None;
            }
        } else if contains_var(arena, f, var_sym) {
            return None;
        } else {
            rest.push(f);
        }
    }
    let c = match rest.len() {
        0 => arena.one,
        1 => rest[0],
        _ => arena.mul(&rest),
    };
    Some((c, n?))
}

/// `(c, k, d)` with `u = c·xᵏ + d` (`c`, `d` free of `x`, `k ≠ 0, 1` a
/// number); `None` otherwise (a linear `u` is a renaming that adds nothing).
pub(super) fn affine_monomial_parts(
    arena: &mut Arena,
    u: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<(ExprId, Q, ExprId)> {
    if u == var {
        return None;
    }
    let (term, d) = match arena.node(u).clone() {
        ExprNode::Add(terms) => {
            let (dep, free): (SmallVec<[ExprId; 4]>, SmallVec<[ExprId; 4]>) = terms
                .iter()
                .copied()
                .partition(|&t| contains_var(arena, t, var_sym));
            let [t] = dep[..] else {
                return None;
            };
            (t, arena.add(&free))
        }
        _ => (u, arena.zero),
    };
    let (c, k) = monomial_parts(arena, term, var, var_sym)?;
    let k = arena.as_num(k)?.clone();
    (!k.is_one()).then_some((c, k, d))
}

/// `∫ h(x, L) dx` for `L = ln(c·xⁿ)` inside a trigonometric, hyperbolic or
/// exponential function (`sin(ln x)`, `x²·cos(a + b·ln(c·xⁿ))²`,
/// `sinh(ln x)`), `x` occurring elsewhere only in powers: with `w = L`,
/// `x = K·e^{w/n}` (`K = c^{−1/n}`) and `dx = x/n·dw`, the integral is
/// `∫ h(K·e^{w/n}, w)·K·e^{w/n}/n dw`, an exponential times trigonometric
/// functions of `w`.  Every term of the integrand and of its answer
/// carries `Kᵐ` with `e^{m·w/n}`, and `Kᵐ·e^{m·w/n} = xᵐ`: the answer in
/// `x` and `L` is the same for every `K`, so it is computed with `K = 1`
/// and brought back with `e^{q·w} = x^{q·n}` (it satisfies `F′ = h` along
/// the curve of every `K`, whatever `c` and `n` are; the self-check
/// confirms it).  Up to 0.31 `∫ sin(ln x) dx` stayed unevaluated; SymPy:
/// `x·sin(log x)/2 − x·cos(log x)/2`.
///
/// Inside `integrate_node` (`anywhere` false) only for a logarithm inside a
/// trigonometric, hyperbolic or exponential function; as a stage of its own
/// for any integrand that is one in `x` and `L` alone: `∫ xᵐ/ln x` is
/// `Ei((m + 1)·ln x)`, `∫ dx/ln(x)²` is `Ei(ln x) − x/ln x`, `∫ sin(ln x)/(x·ln x)`
/// is `Si(ln x)` (unevaluated up to 0.43; SymPy gives the same).  `L` is
/// real and monotonic on `x > 0`, where the integrand is real, so the
/// answer is continuous there where the one in `L` is.  There the answer may
/// have terms without an `e^{q·w}` (`Ei(2w)` for `∫ x/ln(3x)`), and `K` is
/// kept (`x = K·e^{w/n}`, `e^{q·w} = (x/K)^{q·n}`); powers of the
/// logarithm's argument are `gᵐ = e^{m·w}` (`√(5x)/ln(5x)`).
pub(super) fn try_log_substitution(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
    anywhere: bool,
) -> Option<ExprId> {
    let order = crate::base::walk::post_order_ids(arena, expr);
    // The logarithm: one `ln(g)` of `x`, no other.
    let mut log: Option<(ExprId, ExprId)> = None;
    for &id in &order {
        if let ExprNode::Ln(g) = *arena.node(id)
            && contains_var(arena, g, var_sym)
        {
            match log {
                Some((l, _)) if l != id => return None,
                _ => log = Some((id, g)),
            }
        }
    }
    let (ln_node, g) = log?;
    let inside_function = order.iter().any(|&id| {
        matches!(arena.node(id),
            ExprNode::Sin(a) | ExprNode::Cos(a) | ExprNode::Tan(a) | ExprNode::Sinh(a)
            | ExprNode::Cosh(a) | ExprNode::Tanh(a) | ExprNode::Exp(a)
            if crate::base::walk::contains(arena, *a, ln_node))
    });
    if !inside_function && !anywhere {
        return None;
    }
    let w = arena.symbol("__lw");
    let ExprNode::Symbol(w_sym) = *arena.node(w) else {
        return None;
    };
    let Some((c, n)) = monomial_parts(arena, g, var, var_sym) else {
        return log_of_linear_substitution(arena, expr, ln_node, g, w, var, var_sym, anywhere);
    };
    // A real exponent only: `ln(c·xⁿ)` is real and monotonic on `x > 0` (up
    // to the constant `arg c`) only then.  For `n = i`, `xⁱ` turns around
    // the unit circle and `ln(c·xⁱ) = ln c + i·ln x` fails across the cut:
    // `∫ x·sec(a + 2·ln(c·xⁱ))³ dx` got an answer wrong for negative `c`
    // (Rubi 4.7.5 #181, `--negative-params`, 0.47 development).  A symbolic
    // `n` is generic (real), as every parameter of an antiderivative.
    let n_real = crate::base::assumptions::AssumptionCache::new().query(
        arena,
        n,
        crate::base::assumptions::Props::REAL,
    );
    if Truth::from(n_real).is_false()
        || (n_real.is_none() && crate::base::walk::free_symbols(arena, n).is_empty())
    {
        return None;
    }
    // K = c^{−1/n} where it is kept (see above).
    let scale = (anywhere && c != arena.one).then(|| {
        let minus_one = arena.int(-1);
        let e = arena.div(minus_one, n);
        let k = arena.pow(c, e);
        crate::transforms::eval::eval(arena, k)
    });
    // x^m → K^m·e^{m·w/n}, x → K·e^{w/n}.
    let x_power = |arena: &mut Arena, m: ExprId| -> ExprId {
        let mw = arena.mul(&[m, w]);
        let mwn = arena.div(mw, n);
        let e = arena.exp(mwn);
        match scale {
            Some(k) => {
                let k_m = arena.pow(k, m);
                arena.mul(&[k_m, e])
            }
            None => e,
        }
    };
    let mut sub = arena.subs_structural(expr, ln_node, w);
    if anywhere {
        if !algebraic_outside(arena, sub, g, var_sym) {
            return None;
        }
        sub = log_argument_powers(arena, sub, g, w, var_sym);
    }
    for id in crate::base::walk::post_order_ids(arena, sub) {
        if let ExprNode::Pow(b, m) = *arena.node(id)
            && b == var
            && !contains_var(arena, m, var_sym)
        {
            let r = x_power(arena, m);
            sub = arena.subs_structural(sub, id, r);
        }
    }
    let one = arena.one;
    let x_w = x_power(arena, one);
    sub = arena.subs_structural(sub, var, x_w);
    if contains_var(arena, sub, var_sym) {
        return None;
    }
    let integrand = arena.mul(&[sub, x_w]);
    let integrand = arena.div(integrand, n);
    let integrand = crate::transforms::eval::eval(arena, integrand);
    let res = integrate_nested(arena, integrand, w)?;
    // e^{q·w + r} → e^r·(x/K)^{q·n}.
    let x_over_k = match scale {
        Some(k) => arena.div(var, k),
        None => var,
    };
    let mut back = res;
    for id in crate::base::walk::post_order_ids(arena, res) {
        if let ExprNode::Exp(arg) = *arena.node(id)
            && crate::base::walk::has_free_symbol(arena, arg, w_sym)
        {
            let (q, r) = symbolic_linear_coeff_of(arena, arg, w, w_sym)?;
            let qn = arena.mul(&[q, n]);
            let x_qn = arena.pow(x_over_k, qn);
            let e_r = arena.exp(r);
            let repl = arena.mul(&[e_r, x_qn]);
            back = arena.subs_structural(back, id, repl);
        }
    }
    let back = crate::transforms::eval::eval(arena, back);
    let back = arena.subs_structural(back, w, ln_node);
    let back = crate::transforms::eval::eval(arena, back);
    if candidate_rejected(arena, expr, back, var, var_sym) {
        tracing::debug!("integrate: rejecting unverified logarithmic-substitution closed form");
        return None;
    }
    Some(back)
}

/// Does the variable occur in `e` (the integrand with its logarithm `ln g`
/// replaced) only in sums, products and powers with exponents free of it,
/// and in `g`?  The stage-level logarithmic substitution takes no other
/// integrand: `x = e^w` inside `sin x` gives `sin(e^w)`, no easier.
fn algebraic_outside(arena: &Arena, e: ExprId, g: ExprId, var_sym: SymbolId) -> bool {
    let mut stack = vec![e];
    let mut seen: FxHashSet<ExprId> = FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) || id == g || !contains_var(arena, id, var_sym) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Symbol(_) => {}
            ExprNode::Add(children) | ExprNode::Mul(children) => stack.extend(children.iter()),
            ExprNode::Neg(inner) => stack.push(*inner),
            ExprNode::Pow(b, k) if !contains_var(arena, *k, var_sym) => stack.push(*b),
            _ => return false,
        }
    }
    true
}

/// `e` with every power `gᵐ` (`m` free of the variable, `g ≠ x`) written
/// `e^{m·w}` for `w = ln g`, and `g` itself `e^w`: `e^{m·ln g} = gᵐ` for the
/// principal branches, whatever `g` is.
fn log_argument_powers(
    arena: &mut Arena,
    e: ExprId,
    g: ExprId,
    w: ExprId,
    var_sym: SymbolId,
) -> ExprId {
    if !matches!(arena.node(g), ExprNode::Add(_) | ExprNode::Mul(_)) {
        return e;
    }
    let mut out = e;
    for id in crate::base::walk::post_order_ids(arena, e) {
        if let ExprNode::Pow(b, m) = *arena.node(id)
            && b == g
            && !contains_var(arena, m, var_sym)
        {
            let mw = arena.mul(&[m, w]);
            let r = arena.exp(mw);
            out = arena.subs_structural(out, id, r);
        }
    }
    let e_w = arena.exp(w);
    arena.subs_structural(out, g, e_w)
}

/// [`try_log_substitution`] for `L = ln(g)` with `g = α·x + β` linear
/// (`β ≠ 0`) and `x` nowhere outside `L` (`sin(ln(a + b·x))`): `w = L`,
/// `dx = e^w/α·dw`, and the answer comes back with `e^{q·w} = g^q`.  As a
/// stage (`anywhere`), `x` may also occur in powers of `g`
/// ([`log_argument_powers`]): `∫ (2x + 1)/ln(2x + 1) dx = Ei(2·ln(2x + 1))/2`.
#[allow(clippy::too_many_arguments)]
fn log_of_linear_substitution(
    arena: &mut Arena,
    expr: ExprId,
    ln_node: ExprId,
    g: ExprId,
    w: ExprId,
    var: ExprId,
    var_sym: SymbolId,
    anywhere: bool,
) -> Option<ExprId> {
    let ExprNode::Symbol(w_sym) = *arena.node(w) else {
        return None;
    };
    let (alpha, beta) = symbolic_linear_coeff_of(arena, g, var, var_sym)?;
    if arena.is_zero_structural(beta) {
        return None;
    }
    let sub = arena.subs_structural(expr, ln_node, w);
    let sub = if anywhere {
        if !algebraic_outside(arena, sub, g, var_sym) {
            return None;
        }
        log_argument_powers(arena, sub, g, w, var_sym)
    } else {
        sub
    };
    if contains_var(arena, sub, var_sym) {
        return None;
    }
    let e_w = arena.exp(w);
    let integrand = arena.mul(&[sub, e_w]);
    let integrand = arena.div(integrand, alpha);
    let integrand = crate::transforms::eval::eval(arena, integrand);
    let res = integrate_nested(arena, integrand, w)?;
    let mut back = res;
    for id in crate::base::walk::post_order_ids(arena, res) {
        if let ExprNode::Exp(arg) = *arena.node(id)
            && crate::base::walk::has_free_symbol(arena, arg, w_sym)
        {
            let (q, r) = symbolic_linear_coeff_of(arena, arg, w, w_sym)?;
            let g_q = arena.pow(g, q);
            let e_r = arena.exp(r);
            let repl = arena.mul(&[e_r, g_q]);
            back = arena.subs_structural(back, id, repl);
        }
    }
    let back = crate::transforms::eval::eval(arena, back);
    let back = arena.subs_structural(back, w, ln_node);
    let back = crate::transforms::eval::eval(arena, back);
    if candidate_rejected(arena, expr, back, var, var_sym) {
        tracing::debug!("integrate: rejecting unverified logarithmic-substitution closed form");
        return None;
    }
    Some(back)
}

/// Collect the `exp(k·x)` nodes of `expr` (with numeric `k`).  Returns
/// `None` if some `exp` argument depends on `var` but is not of that form.
fn collect_exp_multiples(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<Vec<(ExprId, Q)>> {
    let order = crate::base::walk::post_order_ids(arena, expr);
    let mut out = Vec::new();
    for id in order {
        if let ExprNode::Exp(arg) = arena.node(id).clone()
            && contains_var(arena, arg, var_sym)
        {
            let (alpha, beta) = symbolic_linear_coeff_of(arena, arg, var, var_sym)?;
            if !arena.is_zero_structural(beta) {
                return None;
            }
            let k = arena.as_num(alpha)?.clone();
            out.push((id, k));
        }
    }
    Some(out)
}

/// `∫ R(e^{ax}) dx` via `u = e^{ax}`: `∫ R(u)/(a·u) du` for a rational `R`.
///
/// `a` is chosen as the greatest common divisor of all exponent
/// coefficients so that every `e^{kx}` becomes an integer power of `u`.
pub(super) fn try_exp_rational_substitution(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    use num_integer::Integer;
    let exps = collect_exp_multiples(arena, expr, var, var_sym)?;
    if exps.is_empty() {
        return None;
    }
    // a = gcd(k_i) for rationals: gcd(numerators)/lcm(denominators).
    let mut num_gcd = num_bigint::BigInt::zero();
    let mut den_lcm = num_bigint::BigInt::one();
    for (_, k) in &exps {
        num_gcd = num_gcd.gcd(k.numer());
        den_lcm = den_lcm.lcm(k.denom());
    }
    if num_gcd.is_zero() {
        return None;
    }
    let a = num_rational::Ratio::new(num_gcd, den_lcm);

    let u = arena.symbol("__eu");
    let u_sym = match arena.node(u) {
        ExprNode::Symbol(s) => *s,
        _ => return None,
    };
    let mut sub = expr;
    for (node, k) in &exps {
        let power = k / &a; // integer
        let power_id = arena.num_ratio(power.clone());
        let u_pow = arena.pow(u, power_id);
        sub = arena.subs_structural(sub, *node, u_pow);
    }
    if contains_var(arena, sub, var_sym) {
        return None;
    }
    // Integrand in u: R(u)/(a u)
    let a_id = arena.num_ratio(a.clone());
    let a_u = arena.mul(&[a_id, u]);
    let integrand_u = arena.div(sub, a_u);
    let integrand_u = clear_nested_fractions(arena, integrand_u, u);
    // Must be rational in u.
    let (n, d) = crate::poly::polybridge::as_numer_denom(arena, integrand_u);
    if crate::poly::polybridge::expr_to_poly(arena, n, u).is_none()
        || crate::poly::polybridge::expr_to_poly(arena, d, u).is_none()
    {
        return None;
    }
    let res_u = integrate_node(arena, integrand_u, u, u_sym, 20);
    if crate::base::walk::has_unevaluated(arena, res_u) {
        return None;
    }
    // Back-substitute u → e^{ax}.
    let ax = arena.mul(&[a_id, var]);
    let e_ax = arena.exp(ax);
    let result = arena.subs_structural(res_u, u, e_ax);
    Some(crate::transforms::eval::eval(arena, result))
}

/// `∫ R(e^{L₁}, …, e^{Lₖ}) dx` for linear `Lᵢ = rᵢ·L + γᵢ` with rational
/// `rᵢ`, symbolic `L = α·x + β` (the first exponent) and constants `γᵢ`:
/// `u = e^{g·L}` (`g` the gcd of the `rᵢ`), `du = g·α·u·dx`, every
/// `e^{Lᵢ} = e^{γᵢ}·u^{rᵢ/g}`, and the integrand is rational in `u` over
/// `ℚ(parameters, e^{γᵢ})` (each `e^{γᵢ} ≠ 1` a fresh symbol while the
/// rational integrator runs).  [`try_exp_rational_substitution`] for
/// symbolic rates and shifts: `sinh(a + b·x)`, `sech(c + d·x)²`,
/// `coth(a + b·x)·coth(c + b·x)` (Rubi 6.x) stayed unevaluated.  `u`
/// moves monotonically along a ray for real `x`, so the logarithms of the
/// rational answer jump only at poles of the integrand.
pub(super) fn try_exp_linear_substitution(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    use num_integer::Integer;
    let mut exps: Vec<(ExprId, ExprId, ExprId)> = Vec::new();
    for id in crate::base::walk::post_order_ids(arena, expr) {
        if let ExprNode::Exp(arg) = *arena.node(id)
            && contains_var(arena, arg, var_sym)
        {
            let (alpha, beta) = symbolic_linear_coeff_of(arena, arg, var, var_sym)?;
            exps.push((id, alpha, beta));
        }
    }
    let &(_, alpha0, beta0) = exps.first()?;
    let mut ratios: Vec<Q> = Vec::with_capacity(exps.len());
    for &(_, alpha, _) in &exps {
        ratios.push(rational_ratio(arena, alpha, alpha0)?);
    }
    let mut num_gcd = num_bigint::BigInt::zero();
    let mut den_lcm = num_bigint::BigInt::one();
    for r in &ratios {
        num_gcd = num_gcd.gcd(r.numer());
        den_lcm = den_lcm.lcm(r.denom());
    }
    if num_gcd.is_zero() {
        return None;
    }
    let g = Q::new(num_gcd, den_lcm);
    let u = arena.symbol("__elu");
    let ExprNode::Symbol(u_sym) = *arena.node(u) else {
        return None;
    };
    let mut constants: Vec<(ExprId, ExprId)> = Vec::new();
    let mut sub = expr;
    for (&(node, _, beta), r) in exps.iter().zip(&ratios) {
        let r_id = arena.num_ratio(r.clone());
        let r_beta0 = arena.mul(&[r_id, beta0]);
        let gamma = arena.sub(beta, r_beta0);
        let gamma = crate::transforms::eval::eval(arena, gamma);
        let k = arena.num_ratio(r / &g);
        let u_k = arena.pow(u, k);
        let repl = if arena.is_zero_structural(gamma) {
            u_k
        } else {
            let e_gamma = arena.exp(gamma);
            let kappa = match constants.iter().find(|&&(e, _)| e == e_gamma) {
                Some(&(_, s)) => s,
                None => {
                    let s = arena.symbol(&format!("__elk{}", constants.len()));
                    constants.push((e_gamma, s));
                    s
                }
            };
            arena.mul(&[kappa, u_k])
        };
        sub = arena.subs_structural(sub, node, repl);
    }
    if contains_var(arena, sub, var_sym) {
        return None;
    }
    let g_id = arena.num_ratio(g.clone());
    let du = arena.mul(&[g_id, alpha0, u]);
    let integrand = arena.div(sub, du);
    let integrand = crate::transforms::eval::eval(arena, integrand);
    if !is_rational_in(arena, integrand, u_sym) {
        return None;
    }
    tracing::debug!(integrand_u = %arena.display(integrand), "integrate: exponentials of linear arguments, integrand in u");
    let Some(res) = integrate_nested_generic(arena, integrand, u) else {
        tracing::debug!(
            "integrate: exponential substitution: the integrand in u has no antiderivative"
        );
        return None;
    };
    let Some(res) = analytic_logs(arena, res, u_sym) else {
        tracing::debug!("integrate: exponential substitution: a logarithm in u is not analytic");
        return None;
    };
    let l = arena.mul(&[alpha0, var]);
    let l = arena.add(&[l, beta0]);
    let gl = arena.mul(&[g_id, l]);
    let res = match exp_rate_is_nonreal_constant(arena, g_id, alpha0) {
        Truth::True => match circle_continuous_in_u(arena, res, u, u_sym, gl) {
            Some(res) => res,
            None => {
                let spiral = Spiral { u, u_sym, gl };
                return spiral_continuous(arena, expr, res, &spiral, &constants, var, var_sym);
            }
        },
        Truth::False => res,
        Truth::Unknown => {
            tracing::debug!(
                "integrate: exponential substitution: the rate is not decided real or not"
            );
            return None;
        }
    };
    let e_gl = arena.exp(gl);
    let mut back = arena.subs_structural(res, u, e_gl);
    for &(e_gamma, kappa) in &constants {
        back = arena.subs_structural(back, kappa, e_gamma);
    }
    let back = crate::transforms::eval::eval(arena, back);
    if candidate_rejected(arena, expr, back, var, var_sym) {
        tracing::debug!("integrate: rejecting unverified exponential-substitution closed form");
        return None;
    }
    Some(back)
}

/// `a/b` when it is a rational number: `eval` of the quotient, or else of its
/// real and imaginary parts (`eval` keeps `(1 + i)/(−i − 1)` a quotient, and
/// up to 0.55 `∫ dx/(e^{(1+i)x} + e^{−(1+i)x} − 3)` stayed unevaluated), or
/// else the rational with a small denominator next to its value for which
/// `a − q·b` is exactly 0 ([`confirmed_rational_ratio`]).
pub(crate) fn rational_ratio(arena: &mut Arena, a: ExprId, b: ExprId) -> Option<Q> {
    let r = arena.div(a, b);
    let r = crate::transforms::eval::eval(arena, r);
    if let Some(q) = arena.as_num(r) {
        return Some(q.clone());
    }
    if !crate::base::walk::free_symbols(arena, r).is_empty() {
        return None;
    }
    let (re, im) = crate::base::complex::as_real_imag(arena, r);
    let im = crate::transforms::eval::eval(arena, im);
    if arena.is_zero_structural(im) {
        let re = crate::transforms::eval::eval(arena, re);
        if let Some(q) = arena.as_num(re) {
            return Some(q.clone());
        }
    }
    confirmed_rational_ratio(arena, a, b, r)
}

/// Largest denominator [`confirmed_rational_ratio`] tries.
const MAX_RATIO_DENOMINATOR: u64 = 64;

/// The rational `q` (denominator at most [`MAX_RATIO_DENOMINATOR`]) nearest
/// the value of the constant `r = a/b` when `a − q·b` evaluates to an exact
/// 0: the value only proposes `q`, the exact difference decides.  `eval`
/// keeps a quotient of constant sums (`(−½ ln 2 − iπ/4)/(½ ln 2 + iπ/4)`)
/// as it is, and up to 0.57 `∫ dx/((1 + i)ˣ + (1 + i)⁻ˣ − 3)` and `∫ dx/(e^{(1 +
/// iπ)x} + e^{−(1 + iπ)x} − 3)` stayed unevaluated.
fn confirmed_rational_ratio(arena: &mut Arena, a: ExprId, b: ExprId, r: ExprId) -> Option<Q> {
    let z = crate::transforms::evalf::evalf_complex64(arena, r).ok()?;
    let q = crate::base::numeric::f64_to_ratio_approx(z.re, MAX_RATIO_DENOMINATOR)?;
    let q_id = arena.num_ratio(q.clone());
    let qb = arena.mul(&[q_id, b]);
    let d = arena.sub(a, qb);
    let d = crate::transforms::eval::eval(arena, d);
    arena.is_zero_structural(d).then_some(q)
}

/// Is the rate `g·α` of `u = e^{g·(α·x + β)}` a constant that is not real?
/// Then `u` does not run along a ray for real `x`: on the unit circle for an
/// imaginary rate, on a spiral otherwise.  `True` when its imaginary part
/// is certified nonzero, `False` when it is proved 0 or the rate has
/// parameters (read as real, as before 0.53), `Unknown` for a constant
/// whose imaginary part does not settle.  Up to 0.52 an imaginary part
/// below `10⁻¹²` of the rate counted as 0.
fn exp_rate_is_nonreal_constant(arena: &mut Arena, g: ExprId, alpha: ExprId) -> Truth {
    let rate = arena.mul(&[g, alpha]);
    let rate = crate::transforms::eval::eval(arena, rate);
    if !crate::base::walk::free_symbols(arena, rate).is_empty() {
        return Truth::False;
    }
    crate::transforms::realness::imaginary_part_sign(arena, rate)
        .is_zero()
        .not()
}

/// Is the rate `λ` of `e^{λ·x}` a constant that is not real (its imaginary
/// part certified nonzero)?  Then `e^{λ·x}` runs along a spiral (or a circle)
/// through every direction for real `x`.
pub(super) fn off_a_ray(arena: &mut Arena, lambda: ExprId) -> bool {
    if !crate::base::walk::free_symbols(arena, lambda).is_empty() {
        return false;
    }
    let lambda = crate::transforms::eval::eval(arena, lambda);
    crate::transforms::realness::imaginary_part_sign(arena, lambda)
        .is_nonzero()
        .is_true()
}

/// Does `g` hold a function of the variable that may have a branch cut?
/// Only sums, products, integer powers, constant bases, `exp`, the
/// circular and hyperbolic `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`,
/// `erf`, `erfc` and `Si` of it count as cut-free: entire or meromorphic,
/// so that `G(u(x))` is continuous wherever it is finite.
pub(super) fn has_cut_function(arena: &Arena, g: ExprId, var_sym: SymbolId) -> bool {
    crate::base::walk::post_order_ids(arena, g)
        .into_iter()
        .any(|id| match *arena.node(id) {
            ExprNode::Num(_)
            | ExprNode::Symbol(_)
            | ExprNode::Add(_)
            | ExprNode::Mul(_)
            | ExprNode::Neg(_)
            | ExprNode::Exp(_)
            | ExprNode::Sin(_)
            | ExprNode::Cos(_)
            | ExprNode::Tan(_)
            | ExprNode::Sinh(_)
            | ExprNode::Cosh(_)
            | ExprNode::Tanh(_)
            | ExprNode::Erf(_)
            | ExprNode::Erfc(_)
            | ExprNode::Si(_) => false,
            ExprNode::Pow(b, e) => {
                contains_var(arena, b, var_sym) && arena.as_num(e).is_none_or(|q| !q.is_integer())
            }
            _ => contains_var(arena, id, var_sym),
        })
}

/// The answer `res` in `u = e^{gl}` of [`try_exp_linear_substitution`] for a
/// non-real rate, where `u` leaves the ray along which its logarithms keep
/// off their cuts: `ln u` becomes `gl` (the continuous logarithm of
/// `e^{gl}`), and `None` when any other non-rational function of `u`
/// remains.  `∫ dt/(e^{it} − 2)` came out as `i·(ln e^{it} − ln(e^{it} − 2))/2`,
/// which jumps at `t = π` and at `t = 0` (SymPy's test_integrals; 0.46
/// returned `0` for the integral over `[0, 2π]`, which is `−π`).  Such an
/// integrand goes to [`try_exp_imaginary_to_trig`] instead.
fn circle_continuous_in_u(
    arena: &mut Arena,
    res: ExprId,
    u: ExprId,
    u_sym: SymbolId,
    gl: ExprId,
) -> Option<ExprId> {
    let ln_u = arena.ln(u);
    let res = arena.subs_structural(res, ln_u, gl);
    is_rational_in(arena, res, u_sym).then_some(res)
}

/// The substitution `u = e^{g·l}` of [`try_exp_linear_substitution`]: the
/// placeholder `u` (the symbol `u_sym`) and the exponent `g·l`, linear in
/// the variable.
struct Spiral {
    u: ExprId,
    u_sym: SymbolId,
    gl: ExprId,
}

/// Most distinct roots of the linear factors in `u` whose logarithms
/// [`spiral_continuous`] makes continuous.
const MAX_SPIRAL_LOGS: usize = 4;

/// The answer `res` in `u = e^{λ·x + μ}` of [`try_exp_linear_substitution`]
/// for a rate `λ` that is not real (its imaginary part certified nonzero),
/// with the constants `e^{γ}` of `constants` put back.
///
/// For `re λ ≠ 0`, `u` runs along a spiral from 0 to `∞`, and each `ln(c₁·u +
/// c₀)` with the root `r = −c₀/c₁ ≠ 0` crosses the cut of `ln` once per turn on
/// one side of the point `x_r` where `|u| = |r|`, `x_r = (ln|r| − re μ)/re λ`.
/// It is replaced by a logarithm with the same derivative that stays off the
/// cut on each side: `ln(1 − u/r)` where `|u| < |r|`, `λ·x + μ + ln(1 − r/u)`
/// where `|u| > |r|` (each argument has a positive real part there, `|u/r| ≤
/// 1` or `|r/u| ≤ 1`), joined continuously at `x_r` by the constant
/// `±ln(−r) + 2πi·m`.  With `u(x_r) = r·e^{iθ}`, `θ = im(λ·x_r + μ) − arg r`
/// and `k = ⌊θ/2π⌋` (an exact integer part: [`spiral_turns`]), the two sides
/// differ at `x_r` by `∓(ln|r| + i(π + arg r)) ∓ 2πi·k`, so
/// `m = ∓(k + [arg r > 0])`.  Where `θ/2π` is an integer, `u = r` at `x_r`:
/// the integrand has a pole on the real line there, and the same constant
/// joins two antiderivatives that no constant makes continuous.  Up to a constant
/// `ln(c₁·u)` is `λ·x + μ`.  The answer is `Piecewise((F₀, x < x₁), …, (Fₖ,
/// True))` over the points `xⱼ` in increasing order, as
/// [`try_piecewise_split`] joins its pieces.
///
/// For an imaginary `λ`, `u` runs around the circle `|u| = e^{re μ}` and no
/// point is needed: `ln(1 − u/r)` where `|r| ≥ e^{re μ}` (where they are
/// equal it is continuous between the poles at `u = r`), `λ·x + μ + ln(1 −
/// r/u)` where `|r| < e^{re μ}`.
///
/// `None` (refused) when another function of `u` remains, a logarithm's
/// argument is not linear in `u`, a constant has parameters, or a value or
/// an integer part does not settle.
///
/// `∫₀¹ dx/(e^{(1+10⁻²⁰i)x} − 2)` (`−0.6654466341020272668 −
/// 1.5707963267948966192i`) and `∫₀¹ dx/(e^{(1+i)x} − 2)` were refused up to
/// 0.54: no antiderivative continuous on the real line was known.  Up to 0.55
/// `∫ e^{ix}/(e^{ix} − 3) dx` was `−i·ln(e^{ix} − 3)`, which jumps at `x = π`
/// (now `−i·ln(1 − e^{ix}/3)`), and `∫ dx/(e^{(10⁻³⁰+i)x} − 3)` (`m ≈ −1.7·10²⁹`,
/// beyond an `f64` reading) and `∫ cˣ/(1 − cˣ) dx` (a pole at `x = 0`) were
/// refused.
#[allow(clippy::too_many_arguments)]
fn spiral_continuous(
    arena: &mut Arena,
    expr: ExprId,
    res: ExprId,
    spiral: &Spiral,
    constants: &[(ExprId, ExprId)],
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    use std::cmp::Ordering;
    let Spiral { u, u_sym, gl } = *spiral;
    let ln_u = arena.ln(u);
    let mut res = arena.subs_structural(res, ln_u, gl);
    for &(e_gamma, kappa) in constants {
        res = arena.subs_structural(res, kappa, e_gamma);
    }
    let res = crate::transforms::eval::eval(arena, res);
    if crate::base::walk::free_symbols(arena, res)
        .iter()
        .any(|&s| s != u && s != var)
    {
        tracing::debug!("integrate: spiral: the answer in u has parameters");
        return None;
    }
    let Some(res) = spiral_linear_logs(arena, res, u, u_sym) else {
        tracing::debug!("integrate: spiral: a logarithm or arctangent is not of linear factors");
        return None;
    };
    // `g·l = λ·x + μ`.
    let zero = arena.zero;
    let mu = crate::transforms::subs::subs(arena, gl, var, zero);
    let mu = crate::transforms::eval::eval(arena, mu);
    let lambda = arena.sub(gl, mu);
    let lambda = arena.div(lambda, var);
    let lambda = crate::transforms::eval::eval(arena, lambda);
    if contains_var(arena, lambda, var_sym) {
        return None;
    }
    let re_lambda = arena.re(lambda);
    let re_lambda = crate::transforms::eval::eval(arena, re_lambda);
    // `None` on a circle (an imaginary rate).
    let Some(sign) = crate::poly::algebraic::sign_checked(arena, re_lambda) else {
        tracing::debug!(re = %arena.display(re_lambda), "integrate: spiral: the sign of the rate's real part is not decided");
        return None;
    };
    let growing = match sign {
        0 => None,
        s => Some(s > 0),
    };
    let re_mu = arena.re(mu);
    let re_mu = crate::transforms::eval::eval(arena, re_mu);
    // The logarithms of `u`-dependent arguments, each `ln(c₁·u + c₀)`.
    let mut logs: Vec<(ExprId, Option<ExprId>)> = Vec::new();
    for id in crate::base::walk::post_order_ids(arena, res) {
        let ExprNode::Ln(w) = *arena.node(id) else {
            continue;
        };
        if !contains_var(arena, w, u_sym) || logs.iter().any(|&(l, _)| l == id) {
            continue;
        }
        // (`(u − 1)·i + 1`, the root `1 − i` of `(2 − i)ˣ − 1 + i`, is linear
        // once multiplied out; up to 0.58 it was refused.)
        let coeffs = match symbolic_linear_coeff_of(arena, w, u, u_sym) {
            Some(found) => Some(found),
            None => {
                let expanded = crate::transforms::expand::expand(arena, w);
                symbolic_linear_coeff_of(arena, expanded, u, u_sym)
            }
        };
        let Some((c1, c0)) = coeffs else {
            tracing::debug!(arg = %arena.display(w), "integrate: spiral: a logarithm's argument is not linear in u");
            return None;
        };
        if arena.is_zero_structural(c0) {
            logs.push((id, None));
            continue;
        }
        let r = arena.div(c0, c1);
        let r = arena.neg(r);
        logs.push((id, Some(crate::transforms::eval::eval(arena, r))));
    }
    let mut roots: Vec<ExprId> = Vec::new();
    for &(_, r) in &logs {
        if let Some(r) = r
            && !roots.contains(&r)
        {
            roots.push(r);
        }
    }
    if logs.is_empty() || roots.len() > MAX_SPIRAL_LOGS {
        tracing::debug!(
            logs = logs.len(),
            roots = roots.len(),
            "integrate: spiral: no logarithm of u, or too many roots"
        );
        return None;
    }
    // Nothing else of `u` but rational functions.
    let mut probe = res;
    for (k, &(id, _)) in logs.iter().enumerate() {
        let stand_in = arena.symbol(&format!("__spiral{k}"));
        probe = arena.subs_structural(probe, id, stand_in);
    }
    if !is_rational_in(arena, probe, u_sym) {
        tracing::debug!(res = %arena.display(res), "integrate: spiral: another function of u remains");
        return None;
    }
    // Per logarithm (`node`; in `by_root` its root): the forms left and
    // right of its point and the point.
    struct Sides {
        node: ExprId,
        left: ExprId,
        right: ExprId,
        at: Option<ExprId>,
    }
    let two_pi_i = {
        let two = arena.int(2);
        let pi = arena.pi();
        let i = arena.i_unit();
        arena.mul(&[two, pi, i])
    };
    let one = arena.one;
    // The forms of each root's logarithm, computed once per root.
    let mut by_root: Vec<Sides> = Vec::with_capacity(roots.len());
    for &r in &roots {
        let u_over_r = arena.div(u, r);
        let near = arena.sub(one, u_over_r);
        let near = arena.ln(near);
        let r_over_u = arena.div(r, u);
        let far = arena.sub(one, r_over_u);
        let far = arena.ln(far);
        let far = arena.add(&[gl, far]);
        let abs_r = arena.abs(r);
        let ln_abs_r = arena.ln(abs_r);
        let Some(growing) = growing else {
            // On the circle `|u| = e^{re μ}`: one form for the whole line.
            let d = arena.sub(ln_abs_r, re_mu);
            let d = crate::transforms::eval::eval(arena, d);
            let form = if crate::poly::algebraic::sign_checked(arena, d)? < 0 {
                far
            } else {
                near
            };
            by_root.push(Sides {
                node: r,
                left: form,
                right: form,
                at: None,
            });
            continue;
        };
        let gap = arena.sub(ln_abs_r, re_mu);
        let at = arena.div(gap, re_lambda);
        let mut at = crate::transforms::eval::eval(arena, at);
        // `ln 2/ln √2` (`(1+i)ˣ = 2i`) is 2 once `ln √2` is written `½ ln 2`,
        // as `definite::exp_rational_scan` reads it: the turn count at a
        // pole on the path is an exact integer only for the number (up to
        // 0.58 `∫ dx/((1+i)ˣ − 2i)` was refused: `⌊(π/4·ln 2/ln √2 −
        // π/2)/2π⌋` did not settle).
        if arena.as_num(at).is_none() {
            let n = ln_sqrt_as_half_ln(arena, gap);
            let d = ln_sqrt_as_half_ln(arena, re_lambda);
            let q = arena.div(n, d);
            let q = crate::transforms::eval::eval(arena, q);
            if arena.as_num(q).is_some() {
                at = q;
            }
        }
        if !crate::transforms::evalf::eval_const_f64(arena, at).is_some_and(f64::is_finite) {
            tracing::debug!(at = %arena.display(at), "integrate: spiral: a joining point has no value");
            return None;
        }
        let (left, right) = if growing { (near, far) } else { (far, near) };
        // `left − right` at the point is `∓ln(−r) + 2πi·m`.
        let neg_r = arena.neg(r);
        let ln_neg_r = arena.ln(neg_r);
        let base = if growing {
            arena.neg(ln_neg_r)
        } else {
            ln_neg_r
        };
        let Some(k) = spiral_turns(arena, lambda, mu, r, at) else {
            tracing::debug!(at = %arena.display(at), "integrate: spiral: the turn count does not settle");
            return None;
        };
        let above = match arg_is_positive(arena, r) {
            Truth::True => arena.one,
            Truth::False => arena.zero,
            Truth::Unknown => {
                tracing::debug!("integrate: spiral: the side of a root is not decided");
                return None;
            }
        };
        let m = arena.add(&[k, above]);
        let m = if growing { arena.neg(m) } else { m };
        let m = crate::transforms::eval::eval(arena, m);
        if !arena.as_num(m).is_some_and(|q| q.is_integer()) {
            tracing::debug!(
                "integrate: spiral: the joining constant is not an integer multiple of 2πi"
            );
            return None;
        }
        let jump = arena.mul(&[m, two_pi_i]);
        let join = arena.add(&[base, jump]);
        let right = arena.add(&[right, join]);
        by_root.push(Sides {
            node: r,
            left,
            right,
            at: Some(at),
        });
    }
    let mut sides: Vec<Sides> = Vec::with_capacity(logs.len());
    for &(node, root) in &logs {
        let (left, right, at) = match root {
            None => (gl, gl, None),
            Some(r) => {
                let s = by_root.iter().find(|s| s.node == r)?;
                (s.left, s.right, s.at)
            }
        };
        sides.push(Sides {
            node,
            left,
            right,
            at,
        });
    }
    // The distinct points in increasing order (certified comparisons).
    let mut points: Vec<ExprId> = Vec::new();
    for s in &sides {
        let Some(at) = s.at else {
            continue;
        };
        let mut pos = points.len();
        let mut known = false;
        for (k, &q) in points.iter().enumerate() {
            match cmp_roots(arena, at, q)? {
                Ordering::Equal => {
                    known = true;
                    break;
                }
                Ordering::Less => {
                    pos = k;
                    break;
                }
                Ordering::Greater => {}
            }
        }
        if !known {
            points.insert(pos, at);
        }
    }
    // The index of each logarithm's point in `points`.
    let mut index: Vec<usize> = Vec::with_capacity(sides.len());
    for s in &sides {
        let mut found = 0;
        if let Some(at) = s.at {
            let mut hit = None;
            for (k, &q) in points.iter().enumerate() {
                if cmp_roots(arena, at, q)? == Ordering::Equal {
                    hit = Some(k);
                    break;
                }
            }
            found = hit?;
        }
        index.push(found);
    }
    let e_gl = arena.exp(gl);
    let mut out: Vec<(ExprId, ExprId)> = Vec::with_capacity(points.len() + 1);
    for j in 0..=points.len() {
        let mut branch = res;
        for (s, &i) in sides.iter().zip(&index) {
            // The interval `j` ends at or before the point `i`: its left side
            // (both forms are `λ·x + μ` for `ln(c₁·u)`).
            let form = if j <= i { s.left } else { s.right };
            branch = arena.subs_structural(branch, s.node, form);
        }
        let branch = arena.subs_structural(branch, u, e_gl);
        let branch = crate::transforms::eval::eval(arena, branch);
        let cond = match points.get(j) {
            Some(&t) => arena.gt(t, var),
            None => arena.bool_true,
        };
        out.push((branch, cond));
    }
    let result = match out.as_slice() {
        [(only, _)] => *only,
        _ => arena.piecewise(&out),
    };
    if candidate_rejected(arena, expr, result, var, var_sym) {
        tracing::debug!("integrate: rejecting unverified spiral closed form");
        return None;
    }
    tracing::debug!("integrate: exponential substitution along a spiral, joined continuously");
    Some(result)
}

/// `expr` with every `ln √q` of a positive rational `q` written `½·ln q`
/// (the modulus of a Gaussian rational is such a root): the same value, in
/// the form in which a quotient of two such logarithms cancels to a number.
fn ln_sqrt_as_half_ln(arena: &mut Arena, expr: ExprId) -> ExprId {
    let mut out = expr;
    for id in crate::base::walk::post_order_ids(arena, expr) {
        let ExprNode::Ln(w) = *arena.node(id) else {
            continue;
        };
        let ExprNode::Pow(base, e) = *arena.node(w) else {
            continue;
        };
        let half = Q::new(1.into(), 2.into());
        if arena.as_num(e) != Some(&half)
            || !arena
                .as_num(base)
                .is_some_and(|q| q > &Q::from_integer(0.into()))
        {
            continue;
        }
        let ln_base = arena.ln(base);
        let half = arena.num_ratio(half);
        let repl = arena.mul(&[half, ln_base]);
        out = arena.subs_structural(out, id, repl);
    }
    crate::transforms::eval::eval(arena, out)
}

/// `⌊θ/2π⌋` for the angle `θ = im(λ·x_r + μ) − arg r` of `u(x_r)/r` in
/// [`spiral_continuous`], an exact integer: `eval` of `floor` reads it off
/// certified digits, or proves `θ/2π` an integer (`u = r` at `x_r`) by an
/// exact zero test.  `None` when it does not settle.  Up to 0.55 the turn
/// count was read off an `f64` value of the gap, refused beyond `10¹²` turns
/// (`∫ dx/(e^{(10⁻³⁰+i)x} − 3)` needs `≈ 1.7·10²⁹`) and undefined where
/// the integrand has a pole at `x_r`.
fn spiral_turns(
    arena: &mut Arena,
    lambda: ExprId,
    mu: ExprId,
    r: ExprId,
    at: ExprId,
) -> Option<ExprId> {
    let im_lambda = arena.im(lambda);
    let phase = arena.mul(&[im_lambda, at]);
    let im_mu = arena.im(mu);
    let arg_r = arena.arg(r);
    let neg_arg_r = arena.neg(arg_r);
    let theta = arena.add(&[phase, im_mu, neg_arg_r]);
    let two = arena.int(2);
    let pi = arena.pi();
    let two_pi = arena.mul(&[two, pi]);
    let q = arena.div(theta, two_pi);
    let q = crate::transforms::eval::eval(arena, q);
    let k = arena.floor(q);
    let k = crate::transforms::eval::eval(arena, k);
    arena.as_num(k).filter(|n| n.is_integer())?;
    Some(k)
}

/// Is `arg r > 0`: `r` above the real axis or on its negative half?
/// `Unknown` when that is not decided.
fn arg_is_positive(arena: &mut Arena, r: ExprId) -> Truth {
    let im = crate::transforms::realness::imaginary_part_sign(arena, r);
    if im.is_positive().is_true() {
        return Truth::True;
    }
    if im.is_negative().is_true() {
        return Truth::False;
    }
    if !im.is_zero().is_true() {
        return Truth::Unknown;
    }
    let re = arena.re(r);
    let re = crate::transforms::eval::eval(arena, re);
    match crate::poly::algebraic::sign_checked(arena, re) {
        Some(s) if s != 0 => Truth::from_bool(s < 0),
        _ => Truth::Unknown,
    }
}

/// `res` (in `u`, from the rational integrator) with every `atan(w)` of a
/// linear `w` written `(i/2)·(ln(1 − i·w) − ln(1 + i·w))` and every `ln(q)` of
/// a quadratic `q = a·u² + b·u + c` written `ln(u − r₁) + ln(u − r₂)`, its
/// roots: the same derivative, and logarithms of linear factors only, as
/// [`spiral_continuous`] takes them (`∫ dx/(e^{(1+i)x} − 2i)` came out of the
/// rational integrator with `ln(u² + 4)` and `atan(u/2)`).  `None` for a
/// coefficient with parameters.
fn spiral_linear_logs(
    arena: &mut Arena,
    res: ExprId,
    u: ExprId,
    u_sym: SymbolId,
) -> Option<ExprId> {
    let mut out = res;
    for id in crate::base::walk::post_order_ids(arena, res) {
        let replacement = match *arena.node(id) {
            ExprNode::Atan(w) if contains_var(arena, w, u_sym) => {
                symbolic_linear_coeff_of(arena, w, u, u_sym)?;
                let i = arena.i_unit();
                let iw = arena.mul(&[i, w]);
                let one = arena.one;
                let minus = arena.sub(one, iw);
                let plus = arena.add(&[one, iw]);
                let ln_minus = arena.ln(minus);
                let ln_plus = arena.ln(plus);
                let d = arena.sub(ln_minus, ln_plus);
                let half = arena.rational(1, 2);
                arena.mul(&[half, i, d])
            }
            ExprNode::Ln(q) if contains_var(arena, q, u_sym) => {
                let Some((a, b, c)) = symbolic_quadratic_coeffs(arena, q, u, u_sym) else {
                    continue;
                };
                if [a, b, c]
                    .iter()
                    .any(|&k| !crate::base::walk::free_symbols(arena, k).is_empty())
                {
                    return None;
                }
                let b2 = arena.mul(&[b, b]);
                let four = arena.int(4);
                let ac4 = arena.mul(&[four, a, c]);
                let disc = arena.sub(b2, ac4);
                let s = arena.sqrt(disc);
                let two = arena.int(2);
                let two_a = arena.mul(&[two, a]);
                let neg_b = arena.neg(b);
                let mut logs = Vec::with_capacity(2);
                for sign in [1, -1] {
                    let sgn = arena.int(sign);
                    let ss = arena.mul(&[sgn, s]);
                    let r = arena.add(&[neg_b, ss]);
                    let r = arena.div(r, two_a);
                    let r = crate::transforms::eval::eval(arena, r);
                    let f = arena.sub(u, r);
                    logs.push(arena.ln(f));
                }
                arena.add(&logs)
            }
            _ => continue,
        };
        out = arena.subs_structural(out, id, replacement);
    }
    Some(crate::transforms::eval::eval(arena, out))
}

/// `∫ R(e^{iθ₁}, …) dx` for real linear `θⱼ` (an imaginary rate, where
/// [`try_exp_linear_substitution`] cannot keep its logarithms continuous):
/// every `e^{iθ}` written `cos θ + i·sin θ`, and the trigonometric
/// integrand through the whole pipeline, whose routes correct the jumps
/// at the poles of `tan(θ/2)`.
fn try_exp_imaginary_to_trig(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    let mut any = false;
    for id in crate::base::walk::post_order_ids(arena, expr) {
        if let ExprNode::Exp(arg) = *arena.node(id)
            && contains_var(arena, arg, var_sym)
        {
            let (alpha, _) = symbolic_linear_coeff_of(arena, arg, var, var_sym)?;
            let one = arena.one;
            if !exp_rate_is_nonreal_constant(arena, one, alpha).is_true() {
                return None;
            }
            let re = arena.re(alpha);
            let re = crate::transforms::eval::eval(arena, re);
            if !arena.is_zero_structural(re) {
                return None;
            }
            any = true;
        }
    }
    if !any {
        return None;
    }
    let rewritten = arena.rewrite_as_trig_expr(expr);
    let rewritten = crate::transforms::eval::eval(arena, rewritten);
    if rewritten == expr
        || crate::base::walk::post_order_ids(arena, rewritten)
            .into_iter()
            .any(
                |id| matches!(*arena.node(id), ExprNode::Exp(a) if contains_var(arena, a, var_sym)),
            )
    {
        return None;
    }
    let res = integrate_nested(arena, rewritten, var)?;
    if candidate_rejected(arena, expr, res, var, var_sym) {
        tracing::debug!("integrate: rejecting unverified e^(iθ) → trig closed form");
        return None;
    }
    Some(res)
}

/// `∫ f(e^{a₁x+b₁}, …, e^{aₖx+bₖ}) dx` with rational rates `aᵢ`, `x` nowhere
/// else, that is not rational in the exponentials (those are
/// [`try_exp_rational_substitution`]'s and [`try_exp_linear_substitution`]'s):
/// `u = e^{g·x}` (`g` the gcd of the rates), `dx = du/(g·u)`, and the
/// integrand in `u` through the whole pipeline.  `∫ dy/√(2eʸ + 3)` is
/// `∫ du/(u·√(2u + 3))` (SymPy: `(ln(√(2eʸ+3) − √3) − ln(√(2eʸ+3) +
/// √3))/√3`), `∫ exp(eˣ) dx` is `Ei(eˣ)`, `∫ sin(eˣ) dx` is `Si(eˣ)`; up to
/// 0.43 they stayed unevaluated (the ODE solver worked around the first).
/// For real `x`, `u` runs once through `(0, ∞)`, where the answer in `u` is
/// continuous wherever its integrand is.  Rates with parameters are left
/// alone: `e^{a·x}` with a complex `a` does not keep to that ray.
fn try_exp_general_substitution(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    use num_integer::Integer;
    let mut exps: Vec<(ExprId, Q, ExprId)> = Vec::new();
    for id in crate::base::walk::post_order_ids(arena, expr) {
        if let ExprNode::Exp(arg) = *arena.node(id)
            && contains_var(arena, arg, var_sym)
            && let Some((alpha, beta)) = symbolic_linear_coeff_of(arena, arg, var, var_sym)
        {
            let a = arena.as_num(alpha)?.clone();
            exps.push((id, a, beta));
        }
    }
    if exps.is_empty() {
        return None;
    }
    let mut num_gcd = num_bigint::BigInt::zero();
    let mut den_lcm = num_bigint::BigInt::one();
    for (_, a, _) in &exps {
        num_gcd = num_gcd.gcd(a.numer());
        den_lcm = den_lcm.lcm(a.denom());
    }
    if num_gcd.is_zero() {
        return None;
    }
    // `u = e^{−x}` when every rate is negative: `∫ exp(e^{−x}) dx = −Ei(e^{−x})`.
    let g = if exps.iter().all(|(_, a, _)| a.is_negative()) {
        -Q::new(num_gcd, den_lcm)
    } else {
        Q::new(num_gcd, den_lcm)
    };
    let u = arena.symbol("__egu");
    let ExprNode::Symbol(u_sym) = *arena.node(u) else {
        return None;
    };
    let mut sub = expr;
    for (node, a, beta) in &exps {
        let k = arena.num_ratio(a / &g);
        let u_k = arena.pow(u, k);
        let repl = if arena.is_zero_structural(*beta) {
            u_k
        } else {
            let e_beta = arena.exp(*beta);
            arena.mul(&[e_beta, u_k])
        };
        sub = arena.subs_structural(sub, *node, repl);
    }
    if contains_var(arena, sub, var_sym) {
        return None;
    }
    let g_id = arena.num_ratio(g.clone());
    let du = arena.mul(&[g_id, u]);
    let integrand = arena.div(sub, du);
    let integrand = crate::transforms::eval::eval(arena, integrand);
    if is_rational_in(arena, integrand, u_sym) {
        return None;
    }
    // `u` is real and positive: the answer in `u` under the real-variable
    // convention (`ln|√(3 − 2u) − √3|`) is the one wanted (the stage exit
    // turns the logarithms of non-real arguments into principal ones).
    let res = integrate_nested_generic(arena, integrand, u)?;
    let gx = arena.mul(&[g_id, var]);
    let e_gx = arena.exp(gx);
    let back = arena.subs_structural(res, u, e_gx);
    let back = crate::transforms::eval::eval(arena, back);
    if candidate_rejected(arena, expr, back, var, var_sym) {
        tracing::debug!("integrate: rejecting unverified exponential-substitution closed form");
        return None;
    }
    Some(back)
}

/// `∫ f(x, (κ·x²ʲ)^r, …) dx` (`κ` free of `x`, `r` not an integer, `m = 2j·r`
/// an integer): every such radical is `C·x^m` with `C = (κ·x²ʲ)^r·x^(−m)`,
/// whose derivative is 0 (an identity of analytic functions, so for every
/// value of the parameters), so `F = C·G` with `G = ∫ f·x^m/(κ·x²ʲ)^r dx` —
/// Rubi's device for these radicals (`√(b·x⁴)·x⁻²·(…)`).  For real `x`, `C`
/// is constant on each side of `x = 0` (`(κ·p)^r = κ^r·p^r` for `p ≥ 0`): the
/// same constant `κ^r` for an even `m`; for an odd one it changes sign
/// there, and `F = C·(G − G(0))` is continuous.  The Möbius substitution took
/// `s = x/√(b·x⁴)` for `√(b·x⁴)`, which jumps from `−∞` to `+∞` at `x = 0`:
/// `∫ √(b·x⁴)/(1 + x²) dx` stepped by `π·√b` there (`F(1) − F(−1)` = 3.0924
/// for 0.3717 at `b = 3/4`; Rubi 4.3.0 `√(b·tan⁴(c + d·x))` through `t = tan
/// u`).  `None` when no radical is of that form or `G` is not found.
fn try_even_power_radicals(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    let two = num_bigint::BigInt::from(2);
    let mut out = expr;
    let mut constant: Vec<ExprId> = Vec::new();
    let mut odd = false;
    for id in crate::base::walk::post_order_ids(arena, expr) {
        let ExprNode::Pow(base, ex) = *arena.node(id) else {
            continue;
        };
        let Some(r) = arena.as_num(ex).cloned() else {
            continue;
        };
        if r.is_integer() || !contains_var(arena, base, var_sym) {
            continue;
        }
        let factors: Vec<ExprId> = match arena.node(base) {
            ExprNode::Mul(children) => children.to_vec(),
            _ => vec![base],
        };
        let mut power: Option<Q> = None;
        let mut kappa: Vec<ExprId> = Vec::new();
        for f in factors {
            if !contains_var(arena, f, var_sym) {
                kappa.push(f);
                continue;
            }
            match *arena.node(f) {
                ExprNode::Pow(b, e) if b == var && power.is_none() => {
                    power = arena.as_num(e).cloned();
                }
                _ => {
                    power = None;
                    break;
                }
            }
        }
        let Some(n) = power else {
            continue;
        };
        if !n.is_integer() || n.is_negative() || !(n.numer() % &two).is_zero() {
            continue;
        }
        let m = &n * &r;
        if !m.is_integer() {
            continue;
        }
        odd |= !(m.numer() % &two).is_zero();
        let m_id = arena.num_ratio(m.clone());
        let x_m = arena.pow(var, m_id);
        let minus_m = arena.num_ratio(-m);
        let x_minus_m = arena.pow(var, minus_m);
        constant.push(arena.mul(&[id, x_minus_m]));
        out = arena.subs_structural(out, id, x_m);
    }
    if constant.is_empty() {
        return None;
    }
    let out = crate::transforms::eval::eval(arena, out);
    tracing::debug!(rewritten = %arena.display(out), "integrate: radicals of even powers of the variable");
    let mut g = integrate_nested(arena, out, var)?;
    if odd {
        let at_zero = arena.subs_structural(g, var, arena.zero);
        let at_zero = crate::transforms::eval::eval(arena, at_zero);
        if contains_non_finite(arena, at_zero) || contains_var(arena, at_zero, var_sym) {
            return None;
        }
        let minus = arena.neg(at_zero);
        g = arena.add(&[g, minus]);
    }
    constant.push(g);
    let res = arena.mul(&constant);
    if candidate_rejected(arena, expr, res, var, var_sym) {
        return None;
    }
    Some(res)
}

/// `∫ f(x, x^{p/q}) dx` via `x = s^q`: `∫ q·s^{q−1} f(s^q, s^p) ds`.
pub(super) fn try_radical_substitution(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    use num_integer::Integer;
    let order = crate::base::walk::post_order_ids(arena, expr);
    let mut radicals: Vec<(ExprId, Q)> = Vec::new();
    let mut q_lcm = num_bigint::BigInt::one();
    for id in order {
        if let ExprNode::Pow(base, e) = arena.node(id).clone()
            && base == var
            && let Some(r) = arena.as_num(e).cloned()
            && !r.is_integer()
        {
            q_lcm = q_lcm.lcm(r.denom());
            radicals.push((id, r));
        }
    }
    if radicals.is_empty() {
        return None;
    }
    let q: i64 = q_lcm.to_string().parse().ok()?;
    if !(2..=6).contains(&q) {
        return None;
    }
    let s = arena.symbol("__rs");
    let mut sub = expr;
    for (node, r) in &radicals {
        let k = r * num_rational::Ratio::from_integer(num_bigint::BigInt::from(q));
        let k_id = arena.num_ratio(k.clone());
        let s_pow = arena.pow(s, k_id);
        sub = arena.subs_structural(sub, *node, s_pow);
    }
    let q_id = arena.int(q);
    let s_q = arena.pow(s, q_id);
    sub = arena.subs_structural(sub, var, s_q);
    if contains_var(arena, sub, var_sym) {
        return None;
    }
    // `s = x^{1/q}` is real only where `x ≥ 0` (for even `q`), so the
    // `s`-integrand may not contain anything whose value depends on `s`
    // being real — `|s|`, `sign(s)`, `H(s)`: `∫ |√x| dx` became
    // `∫ 2s·|s| ds = ⅔ s³ sign(s)`, wrong for x < 0 (fuzz_integrate).
    let s_sym = match arena.node(s) {
        ExprNode::Symbol(sid) => *sid,
        _ => return None,
    };
    let real_only = crate::base::walk::post_order_ids(arena, sub)
        .into_iter()
        .any(|id| match arena.node(id) {
            ExprNode::Abs(g) | ExprNode::Sign(g) | ExprNode::Heaviside(g) => {
                contains_var(arena, *g, s_sym)
            }
            _ => false,
        });
    if real_only {
        return None;
    }
    // dx = q s^{q−1} ds
    let qm1 = arena.int(q - 1);
    let s_qm1 = arena.pow(s, qm1);
    let integrand_s = arena.mul(&[q_id, s_qm1, sub]);
    let integrand_s = crate::transforms::eval::eval(arena, integrand_s);
    let res_s = integrate_nested(arena, integrand_s, s)?;
    // The `s`-integrand is analytic, but the rational integrator writes
    // `ln|v(s)|`, the *real* antiderivative — wrong once `s = x^{1/q}` is not
    // real (`∫ x^(−5/2)·atan(√x) dx` gained `−⅔ ln|√x|`, whose derivative
    // at x < 0 is not the integrand; found by `fuzz_integrate`).  Use the
    // analytic `ln v(s)`; any other `|·|`, `sign` or `H` of `s` is refused.
    let res_s = analytic_logs(arena, res_s, s_sym)?;
    // Back-substitute s → x^{1/q}.
    let inv_q = arena.rational(1, q);
    let root = arena.pow(var, inv_q);
    let result = arena.subs_structural(res_s, s, root);
    let result = crate::transforms::eval::eval(arena, result);
    // Verified like the other risky routes: the `s`-integral is computed by
    // the real-variable integrator, and a real-only step can still slip
    // through below the checks above — `(s⁻⁴)^{1/2}` flattened to `s⁻²` gave
    // `∫ cos(√x)/√(x⁻²) dx` a closed form whose derivative is wrong for
    // x < 0 (0.23, fuzz_integrate).
    if candidate_rejected(arena, expr, result, var, var_sym) {
        tracing::debug!("integrate: rejecting unverified radical-substitution closed form");
        return None;
    }
    Some(result)
}

/// `ln|g|` → `ln g` for every `g` depending on `s`; `None` when an `|·|`,
/// `sign` or `H` of `s` remains anywhere else.
pub(super) fn analytic_logs(arena: &mut Arena, e: ExprId, s_sym: SymbolId) -> Option<ExprId> {
    let order = crate::base::walk::post_order_ids(arena, e);
    let mut out = e;
    for id in order {
        if let ExprNode::Ln(inner) = arena.node(id).clone()
            && let ExprNode::Abs(g) = arena.node(inner).clone()
            && contains_var(arena, g, s_sym)
        {
            let plain = arena.ln(g);
            out = arena.subs_structural(out, id, plain);
        }
    }
    let leftover = crate::base::walk::post_order_ids(arena, out)
        .into_iter()
        .any(|id| match arena.node(id) {
            ExprNode::Abs(g) | ExprNode::Sign(g) | ExprNode::Heaviside(g) => {
                contains_var(arena, *g, s_sym)
            }
            _ => false,
        });
    (!leftover).then_some(out)
}
