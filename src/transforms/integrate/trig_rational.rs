//! trig rational (part of [`integrate`](super)).

use super::*;

/// Check whether `expr` is a rational function of `sin(var)` and `cos(var)`.
///
/// A rational trig function may contain sin(var), cos(var), numeric
/// constants, and arithmetic operations (+, ×, integer powers, negation).
/// The integration variable must appear **only** inside sin/cos.
fn is_rational_trig(arena: &Arena, expr: ExprId, var: ExprId, var_sym: SymbolId) -> bool {
    if !contains_var(arena, expr, var_sym) {
        return true; // constant → trivially rational
    }
    match arena.node(expr).clone() {
        ExprNode::Sin(inner) if inner == var => true,
        ExprNode::Cos(inner) if inner == var => true,
        ExprNode::Add(children) => children
            .iter()
            .all(|&c| is_rational_trig(arena, c, var, var_sym)),
        ExprNode::Mul(children) => children
            .iter()
            .all(|&c| is_rational_trig(arena, c, var, var_sym)),
        ExprNode::Neg(inner) => is_rational_trig(arena, inner, var, var_sym),
        ExprNode::Pow(base, exp) => {
            if !contains_var(arena, exp, var_sym)
                && let Some(e) = arena.as_num(exp)
                && e.is_integer()
            {
                return is_rational_trig(arena, base, var, var_sym);
            }
            false
        }
        _ => false,
    }
}

/// Apply the Weierstrass (half-angle tangent) substitution to integrate a
/// rational function of `sin(var)` and `cos(var)`.
///
/// Substitution: `t = tan(var/2)`, giving
///   - `sin(var) = 2t/(1+t²)`
///   - `cos(var) = (1−t²)/(1+t²)`
///   - `dx        = 2/(1+t²) dt`
///
/// After substitution the integrand becomes a rational function of `t`.
/// With parameters, all declared real, an integrand with `f(x + π) = f(x)`
/// first goes through `t = tan x` ([`try_bioche_tan_substitution`]).
pub(super) fn try_weierstrass_substitution(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    if depth < 3 {
        return None;
    }
    if !is_rational_trig(arena, expr, var, var_sym) {
        return None;
    }
    // The jump at the poles of `tan(x/2)` with parameters not declared
    // real, as for real ones (up to 0.37 the answer was kept uncorrected:
    // `∫ dx/(a + b·cos x)` dropped by `2π/√(a² − b²)` at every odd multiple
    // of `π` for `a > |b|`; the continuity hunter's 118 jumps).
    let _jumps = crate::transforms::trig_integ::RealParameterJumps::enable(false);
    if has_only_real_parameters(arena, expr, var_sym)
        && let Some(result) = try_bioche_tan_substitution(arena, expr, var, var_sym, depth, false)
    {
        return Some(result);
    }

    tracing::debug!("trying Weierstrass substitution");

    let t = arena.symbol("__wt");
    let ExprNode::Symbol(t_sym) = *arena.node(t) else {
        return None;
    };

    let two = arena.int(2);
    let one = arena.one;
    let t_sq = arena.pow(t, two);
    let one_plus_t_sq = arena.add(&[one, t_sq]);

    let sin_var = arena.sin(var);
    let cos_var = arena.cos(var);

    // sin(var) → 2t/(1+t²)
    let two_t = arena.mul(&[two, t]);
    let sin_sub = arena.div(two_t, one_plus_t_sq);

    // cos(var) → (1−t²)/(1+t²)
    let one_minus_t_sq = arena.sub(one, t_sq);
    let cos_sub = arena.div(one_minus_t_sq, one_plus_t_sq);

    // dx factor: 2/(1+t²)
    let dx_factor = arena.div(two, one_plus_t_sq);

    // Apply substitution
    let mut sub_expr = arena.subs_structural(expr, sin_var, sin_sub);
    sub_expr = arena.subs_structural(sub_expr, cos_var, cos_sub);

    // Multiply by dx factor
    let integrand_t = arena.mul(&[sub_expr, dx_factor]);

    // Aggressively simplify / cancel
    let integrand_t = crate::transforms::eval::eval(arena, integrand_t);
    let integrand_t = crate::transforms::expand::expand(arena, integrand_t);
    let integrand_t = crate::transforms::eval::eval(arena, integrand_t);
    let integrand_t = arena.cancel_expr(integrand_t, t);
    let integrand_t = crate::transforms::eval::eval(arena, integrand_t);
    // Clear nested fractions such as 2/((1+t²)(2 + (1−t²)/(1+t²))) by
    // normalising to a single numerator/denominator pair.
    let integrand_t = clear_nested_fractions(arena, integrand_t, t);
    tracing::debug!(integrand_t = %arena.display(integrand_t), "weierstrass: integrand in t");

    // Integrate w.r.t. t
    let integral_t = integrate_node(arena, integrand_t, t, t_sym, depth.saturating_sub(2));

    if crate::base::walk::has_unevaluated(arena, integral_t) {
        return None;
    }
    // The integrand in `t` is a rational function; check the closed form
    // before it is dressed up in tan(x/2) (the rational integrator's
    // nested-radical failures show up here as e.g. `atan(3·tan(x/2))` for
    // `∫ cos x/(sin²x + 1)`).
    if candidate_rejected(arena, integrand_t, integral_t, t, t_sym) {
        tracing::debug!("weierstrass: closed form in t failed verification");
        return None;
    }

    // Substitute back, t → tan(var/2), continuously across var = π + 2kπ
    // (up to 0.37 `∫ dx/(2 + cos x)` was `2/√3·atan(tan(x/2)/√3)`, which
    // drops by 2π/√3 there; definite integrals taken from it were wrong).
    let half = arena.rational(1, 2);
    let half_var = arena.mul(&[half, var]);
    match crate::transforms::trig_integ::continuous_through_tan(
        arena, integral_t, t, t_sym, half_var,
    ) {
        Some(result) => Some(result),
        None if undecided_jump_kept(arena, integrand_t, integral_t, t, t_sym) => {
            tracing::debug!("weierstrass: the jump at x = π is not decided; uncorrected form");
            let tan_half = arena.tan(half_var);
            let back = arena.subs_structural(integral_t, t, tan_half);
            Some(crate::transforms::trig_integ::atan_tan_terms_to_argument(
                arena, back, half_var,
            ))
        }
        None => {
            tracing::debug!("weierstrass: the jump of G(tan(x/2)) at x = π is not decided");
            None
        }
    }
}

/// Does `expr` have parameters (free symbols other than the variable), all
/// of them declared real?
fn has_only_real_parameters(arena: &Arena, expr: ExprId, var_sym: SymbolId) -> bool {
    let params: Vec<SymbolId> = crate::base::walk::free_symbols(arena, expr)
        .into_iter()
        .filter_map(|s| match *arena.node(s) {
            ExprNode::Symbol(sid) if sid != var_sym => Some(sid),
            _ => None,
        })
        .collect();
    !params.is_empty()
        && params
            .iter()
            .all(|&sid| crate::transforms::realness::symbol_declared_real(arena, sid))
}

/// Bioche's rule for a rational function `f` of `sin x`, `cos x` with
/// `f(x + π) = f(x)`: the substitution `t = tan x` (`sin x = t·c`, `cos x =
/// c`, `c² = 1/(1 + t²)`, `dx = dt/(1 + t²)`) gives a rational function of
/// `t` of half the degree the half-angle substitution gives.  For real
/// parameters that decides the jump of `G(tan x)` at the poles of `tan x`
/// where the half-angle route's cannot be: `∫ dx/(a + b·sin²x)` (`a`, `b > 0`)
/// is `∫ dt/(a + (a + b)·t²)`, an `atan` with a positive coefficient, where
/// `tan(x/2)` gives a quartic whose roots' half-planes depend on the
/// parameters (up to 0.37 the answer was a sum of four logarithms that
/// jumped at every odd multiple of `π`).  Rubi integrates these integrands
/// so (`atan(√(a + b)·tan x/√a)/√(a(a + b))`); the floor term is the
/// continuity correction of [`crate::transforms::trig_integ::continuous_through_tan`].
///
/// `f(tc, c)` is even in `c` exactly when `f(x + π) = f(x)`; its numerator
/// and denominator (cancelled) are then both even or both odd in `c`.
/// `None` when `f` is not so invariant, the integral in `t` is not found or
/// fails the check, or its jump at `t = ±∞` is not decided (the caller
/// then takes the half-angle route) — unless `keep_undecided`, when such a
/// jump is one [`jump_kept_undecided`] lets the half-angle route keep: the
/// answer is then `G(tan x)` uncorrected, as that route would give it (of
/// half the degree in `t`).
fn try_bioche_tan_substitution(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
    keep_undecided: bool,
) -> Option<ExprId> {
    let t = arena.symbol("__bt");
    let c = arena.symbol("__bc");
    let ExprNode::Symbol(t_sym) = *arena.node(t) else {
        return None;
    };
    let sin_var = arena.sin(var);
    let cos_var = arena.cos(var);
    let t_c = arena.mul(&[t, c]);
    let substituted = arena.subs_structural(expr, sin_var, t_c);
    let substituted = arena.subs_structural(substituted, cos_var, c);
    if contains_var(arena, substituted, var_sym) {
        return None;
    }
    let in_c = clear_nested_fractions(arena, substituted, c);
    let (num, den) = crate::poly::polybridge::as_numer_denom(arena, in_c);
    let num_c = crate::transforms::solve::symbolic_poly_coeffs(arena, num, c)?;
    let den_c = crate::transforms::solve::symbolic_poly_coeffs(arena, den, c)?;
    let vanishes = |arena: &mut Arena, cs: &[ExprId], parity: usize| {
        cs.iter()
            .skip(parity)
            .step_by(2)
            .all(|&k| Truth::from(crate::poly::algebraic::is_zero_checked(arena, k)).is_true())
    };
    // `shift` 0: both even; 1: both odd (divided by `c`).
    let shift = if vanishes(arena, &num_c, 1) && vanishes(arena, &den_c, 1) {
        0
    } else if vanishes(arena, &num_c, 0) && vanishes(arena, &den_c, 0) {
        1
    } else {
        return None;
    };
    let one = arena.one;
    let two = arena.int(2);
    let t_sq = arena.pow(t, two);
    let one_plus_t_sq = arena.add(&[one, t_sq]);
    let in_t = |arena: &mut Arena, cs: &[ExprId]| -> ExprId {
        let mut terms: Vec<ExprId> = Vec::new();
        for (j, &k) in cs.iter().enumerate().skip(shift).step_by(2) {
            let half_degree = i64::try_from((j - shift) / 2).unwrap_or(0);
            let e = arena.int(-half_degree);
            let c_pow = arena.pow(one_plus_t_sq, e);
            terms.push(arena.mul(&[k, c_pow]));
        }
        arena.add(&terms)
    };
    let num_t = in_t(arena, &num_c);
    let den_t = in_t(arena, &den_c);
    let dt = arena.mul(&[den_t, one_plus_t_sq]);
    let integrand_t = arena.div(num_t, dt);
    let integrand_t = crate::transforms::eval::eval(arena, integrand_t);
    let integrand_t = clear_nested_fractions(arena, integrand_t, t);
    if contains_var(arena, integrand_t, var_sym)
        || crate::base::walk::contains(arena, integrand_t, c)
    {
        return None;
    }
    tracing::debug!(integrand_t = %arena.display(integrand_t), "bioche: integrand in tan x");
    let integral_t = integrate_node(arena, integrand_t, t, t_sym, depth.saturating_sub(2));
    if crate::base::walk::has_unevaluated(arena, integral_t)
        || candidate_rejected(arena, integrand_t, integral_t, t, t_sym)
    {
        return None;
    }
    let result = match crate::transforms::trig_integ::continuous_through_tan(
        arena, integral_t, t, t_sym, var,
    ) {
        Some(result) => result,
        None if keep_undecided && undecided_jump_kept(arena, integrand_t, integral_t, t, t_sym) => {
            let tan_var = arena.tan(var);
            let back = arena.subs_structural(integral_t, t, tan_var);
            crate::transforms::trig_integ::atan_tan_terms_to_argument(arena, back, var)
        }
        None => return None,
    };
    if candidate_rejected(arena, expr, result, var, var_sym) {
        tracing::debug!("bioche: the antiderivative in x failed verification");
        return None;
    }
    Some(result)
}

/// Largest multiple `k` of the common angle in
/// [`try_trig_rational_substitution`]: `sin k·u` becomes a polynomial of
/// degree `k` in `sin u`, `cos u`.
const MAX_ANGLE_MULTIPLE: i64 = 12;

/// `(sin k·u, cos k·u)` as polynomials in `s = sin u`, `c = cos u` (`k ≥ 1`):
/// `cos k·u = Tₖ(c)`, `sin k·u = s·Uₖ₋₁(c)` (Chebyshev polynomials, DLMF
/// 18.5.1–18.5.2), by the recurrences `Tₙ₊₁ = 2c·Tₙ − Tₙ₋₁`, `Uₙ₊₁ = 2c·Uₙ −
/// Uₙ₋₁`.
fn multiple_angle(arena: &mut Arena, k: i64, s: ExprId, c: ExprId) -> (ExprId, ExprId) {
    let two = arena.int(2);
    let two_c = arena.mul(&[two, c]);
    let minus_one = arena.int(-1);
    let (mut t_prev, mut t_cur) = (arena.one, c);
    let (mut u_prev, mut u_cur) = (arena.zero, arena.one);
    for _ in 1..k {
        let t_lead = arena.mul(&[two_c, t_cur]);
        let t_back = arena.mul(&[minus_one, t_prev]);
        let t_next = arena.add(&[t_lead, t_back]);
        let t_next = crate::transforms::expand::expand(arena, t_next);
        let u_lead = arena.mul(&[two_c, u_cur]);
        let u_back = arena.mul(&[minus_one, u_prev]);
        let u_next = arena.add(&[u_lead, u_back]);
        let u_next = crate::transforms::expand::expand(arena, u_next);
        (t_prev, t_cur) = (t_cur, t_next);
        (u_prev, u_cur) = (u_cur, u_next);
    }
    let sin_k = arena.mul(&[s, u_cur]);
    let sin_k = crate::transforms::expand::expand(arena, sin_k);
    (sin_k, t_cur)
}

/// Is the polynomial `p` (in the symbol `v`, other symbols as coefficients)
/// even (`Some(0)`) or odd (`Some(1)`) in `v`?  `None` when it has both
/// kinds of powers, is zero, or is not a polynomial in `v`.  A coefficient
/// counts as zero when it is so structurally or proved so.
fn parity_in(arena: &mut Arena, p: ExprId, v: ExprId) -> Option<usize> {
    let coeffs = crate::transforms::solve::symbolic_poly_coeffs(arena, p, v)?;
    let mut nonzero = [false, false];
    for (j, &k) in coeffs.iter().enumerate() {
        let zero = arena.is_zero_structural(k)
            || (crate::base::walk::free_symbols(arena, k).is_empty()
                && Truth::from(crate::poly::algebraic::is_zero_checked(arena, k)).is_true());
        if !zero {
            nonzero[j % 2] = true;
        }
    }
    match nonzero {
        [true, false] => Some(0),
        [false, true] => Some(1),
        _ => None,
    }
}

/// `p = κ·p̃` for a polynomial `p` in the symbols `vars` whose coefficients are
/// all constant multiples of one of them, `κ`, which has free parameters:
/// `(κ, p̃)` with `p̃` free of parameters (`a + a·s` = `a·(1 + s)`).  `None`
/// otherwise.  The jump of an antiderivative through `tan w` is decided
/// for `p̃` where it is not for `p` (a parameter not declared real may be
/// complex), and it scales with `κ`.
fn parameter_content(arena: &mut Arena, p: ExprId, vars: &[ExprId]) -> Option<(ExprId, ExprId)> {
    let mut monomials: Vec<(ExprId, ExprId)> = vec![(p, arena.one)];
    for &v in vars {
        let mut next: Vec<(ExprId, ExprId)> = Vec::new();
        for (q, mono) in monomials {
            let coeffs = crate::transforms::solve::symbolic_poly_coeffs(arena, q, v)?;
            for (j, k) in coeffs.into_iter().enumerate() {
                if arena.is_zero_structural(k) {
                    continue;
                }
                let e = arena.int(i64::try_from(j).ok()?);
                let v_j = arena.pow(v, e);
                next.push((k, arena.mul(&[mono, v_j])));
            }
        }
        monomials = next;
    }
    let &(lead, _) = monomials.first()?;
    if crate::base::walk::free_symbols(arena, lead).is_empty() {
        return None;
    }
    let mut terms: Vec<ExprId> = Vec::with_capacity(monomials.len());
    for (k, mono) in monomials {
        let ratio = arena.div(k, lead);
        let ratio = crate::transforms::eval::eval(arena, ratio);
        if !crate::base::walk::free_symbols(arena, ratio).is_empty() {
            return None;
        }
        terms.push(arena.mul(&[ratio, mono]));
    }
    Some((lead, arena.add(&terms)))
}

/// `R/w` for a rational function `R = num/den` of `w` and `o` odd in `w`
/// (`num` and `den` of the parities `pn ≠ pd` in `w`), written in `t` with
/// `w² = 1 − t²`, `o = t`: the integrand in `t` of Bioche's substitutions
/// `t = cos u` (`w = sin u`, `dt = −w·du`) and `t = sin u` (`w = cos u`,
/// `dt = w·du`).  `R/w = w^(pn − pd − 1)·Ñ(w²)/D̃(w²)`, the power `w⁰` or
/// `w⁻²`.
fn odd_part_in_t(
    arena: &mut Arena,
    fraction: (ExprId, ExprId),
    parities: (usize, usize),
    w: ExprId,
    o: ExprId,
    t: ExprId,
) -> Option<ExprId> {
    let one = arena.one;
    let two = arena.int(2);
    let t_sq = arena.pow(t, two);
    let minus_t_sq = arena.neg(t_sq);
    let w_sq = arena.add(&[one, minus_t_sq]);
    let reduce = |arena: &mut Arena, p: ExprId, parity: usize| -> Option<ExprId> {
        let coeffs = crate::transforms::solve::symbolic_poly_coeffs(arena, p, w)?;
        let mut terms: Vec<ExprId> = Vec::new();
        for (j, &k) in coeffs.iter().enumerate().skip(parity).step_by(2) {
            let half = arena.int(i64::try_from((j - parity) / 2).ok()?);
            let k_t = arena.subs_structural(k, o, t);
            let power = arena.pow(w_sq, half);
            terms.push(arena.mul(&[k_t, power]));
        }
        Some(arena.add(&terms))
    };
    let n = reduce(arena, fraction.0, parities.0)?;
    let d = reduce(arena, fraction.1, parities.1)?;
    let d = if parities.0 == 1 {
        d
    } else {
        arena.mul(&[w_sq, d])
    };
    Some(arena.div(n, d))
}

/// Is `e` a rational function of the symbols `vars` (sums, products and
/// integer powers of them and of constants)?
fn is_rational_in_all(arena: &Arena, e: ExprId, vars: &[ExprId]) -> bool {
    vars.iter().all(|&v| match *arena.node(v) {
        ExprNode::Symbol(sid) => is_rational_in(arena, e, sid),
        _ => false,
    })
}

/// `∫ R(sin L₁, cos L₁, tan L₁, …, tan Lₖ) dx` for a rational function `R`
/// (with parameters) of trigonometric functions of linear arguments `Lᵢ =
/// αᵢ·x + βᵢ` whose rates are rational multiples of one another (`sec`,
/// `csc`, `cot` are written with `sin`, `cos`).  `u = g·(α₁·x + β₁)` (`g`
/// the gcd of the ratios `αᵢ/α₁`) makes every `Lᵢ = kᵢ·u + γᵢ` with an
/// integer `kᵢ`, and the multiple-angle and addition formulas give a
/// rational function `R̃(sin u, cos u)`; `du = g·α₁·dx`.  Bioche's rules
/// then pick the substitution that keeps the rational function in `t`
/// small:
///
/// - `R̃` odd in `sin u` (`R̃(u)·du` invariant under `u → −u`): `t = cos u`;
/// - odd in `cos u` (invariant under `u → π − u`): `t = sin u`;
/// - invariant under `u → π + u`: `t = tan u`
///   ([`try_bioche_tan_substitution`]);
/// - otherwise `t = tan(u/2)` ([`try_weierstrass_substitution`]).
///
/// The first two have no poles, so `G(sin u)` and `G(cos u)` are continuous
/// wherever the integrand is, whatever the parameters (`∫ cos u/(a +
/// b·sin u)² du = − 1/(b·(a + b·sin u))`); the last two add the floor
/// correction of [`crate::transforms::trig_integ::continuous_through_tan`].
/// Its jump `J` is decided with every parameter taken for real
/// ([`crate::transforms::trig_integ::RealParameterJumps`]: exact for real
/// values, `J·⌊w/π + 1/2⌋` locally constant for others), and an answer whose
/// jump is still not decided is returned only where the integrand is not
/// continuous across the poles ([`undecided_jump_kept`]).  A parameter
/// that multiplies the whole numerator or denominator (`(a + a·sin u)²`) is
/// split off first ([`parameter_content`]), and parameters only in the
/// numerator are taken term by term ([`trig_rational_in_u`]), which decides
/// `J` without signs.  Radicals: algebraic functions of `tan u`
/// ([`tan_algebraic_in_u`]) and radicals of `sin u` (`cos u`) times an odd
/// power of `cos u` (`sin u`) ([`odd_algebraic_in_u`]); a polynomial factor
/// in `x` goes through [`trig_polynomial_substitution`].  Before 0.38 a
/// trigonometric argument other than `x` itself went to the half-angle
/// substitution only for `sin`, `cos` powers, and `tan` never: `∫ dx/(5 +
/// 3·sin(2x + 1))`, `∫ dx/(2 + 3·tan x)` and 3,395 rational trigonometric
/// integrands of the Rubi suite (4.1–4.7, nearly all over `c + d·x`)
/// stayed unevaluated.  The substitutions follow the mathematics of Rubi's
/// rules for these integrands (4.1–4.7), not their form.  `None` when an
/// argument is not linear, the rates are not commensurable, a multiple
/// exceeds [`MAX_ANGLE_MULTIPLE`], the integrand is of none of these
/// forms, or no substitution gives a verified answer.
pub(super) fn try_trig_rational_substitution(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    use num_integer::Integer;
    let mut trig: Vec<(ExprId, ExprId, ExprId)> = Vec::new();
    for id in crate::base::walk::post_order_ids(arena, expr) {
        let arg = match *arena.node(id) {
            ExprNode::Sin(a) | ExprNode::Cos(a) | ExprNode::Tan(a) => a,
            _ => continue,
        };
        if !contains_var(arena, arg, var_sym) {
            continue;
        }
        let (alpha, beta) = symbolic_linear_coeff_of(arena, arg, var, var_sym)?;
        trig.push((id, alpha, beta));
    }
    let &(_, alpha0, beta0) = trig.first()?;
    let mut ratios: Vec<Q> = Vec::with_capacity(trig.len());
    for &(_, alpha, _) in &trig {
        let r = arena.div(alpha, alpha0);
        let r = crate::transforms::eval::eval(arena, r);
        ratios.push(arena.as_num(r)?.clone());
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
    let u = arena.symbol("__tru");
    let s = arena.symbol("__trs");
    let c = arena.symbol("__trc");
    let ExprNode::Symbol(u_sym) = *arena.node(u) else {
        return None;
    };
    let mut sub = expr;
    for (&(node, _, beta), r) in trig.iter().zip(&ratios) {
        let k = r / &g;
        if !k.is_integer() {
            return None;
        }
        let k = i64::try_from(k.to_integer()).ok()?;
        if k == 0 || k.abs() > MAX_ANGLE_MULTIPLE {
            return None;
        }
        let r_id = arena.num_ratio(r.clone());
        let r_beta0 = arena.mul(&[r_id, beta0]);
        let gamma = arena.sub(beta, r_beta0);
        let gamma = crate::transforms::eval::eval(arena, gamma);
        let (sin_k, cos_k) = multiple_angle(arena, k.abs(), s, c);
        let sin_k = if k < 0 { arena.neg(sin_k) } else { sin_k };
        let (sin_l, cos_l) = if arena.is_zero_structural(gamma) {
            (sin_k, cos_k)
        } else {
            let sin_g = arena.sin(gamma);
            let cos_g = arena.cos(gamma);
            let a = arena.mul(&[sin_k, cos_g]);
            let b = arena.mul(&[cos_k, sin_g]);
            let sin_l = arena.add(&[a, b]);
            let a = arena.mul(&[cos_k, cos_g]);
            let b = arena.mul(&[sin_k, sin_g]);
            let cos_l = arena.sub(a, b);
            (sin_l, cos_l)
        };
        let repl = match arena.node(node) {
            ExprNode::Sin(_) => sin_l,
            ExprNode::Cos(_) => cos_l,
            _ => arena.div(sin_l, cos_l),
        };
        sub = arena.subs_structural(sub, node, repl);
    }
    let rate = arena.num_ratio(g.clone());
    let rate = arena.mul(&[rate, alpha0]);
    let shift = arena.num_ratio(g);
    let shift = arena.mul(&[shift, beta0]);
    let shift = crate::transforms::eval::eval(arena, shift);
    if contains_var(arena, sub, var_sym) {
        return trig_polynomial_substitution(arena, expr, sub, (s, c), (rate, shift), var, var_sym);
    }
    // The jumps of answers through `tan w` with parameters, as for real
    // ones (exact for real values, locally constant otherwise), of the
    // answer with the logarithms the stage exit writes `ln u`.
    let undeclared = |arena: &Arena, e: ExprId| {
        crate::base::walk::free_symbols(arena, e)
            .into_iter()
            .any(|v| match *arena.node(v) {
                ExprNode::Symbol(sid) => {
                    !crate::transforms::realness::symbol_declared_real(arena, sid)
                }
                _ => true,
            })
    };
    let plain_logs = undeclared(arena, rate) || undeclared(arena, shift);
    let _jumps = crate::transforms::trig_integ::RealParameterJumps::enable(plain_logs);
    let _strict = crate::transforms::trig_integ::StrictJumps::enable();
    let (kappa, result_u) = if is_rational_in_all(arena, sub, &[s, c]) {
        tracing::debug!(
            "integrate: a rational function of the trigonometric functions of a linear argument"
        );
        let r = clear_nested_fractions(arena, sub, s);
        let (num, den) = crate::poly::polybridge::as_numer_denom(arena, r);
        let num = crate::transforms::expand::expand(arena, num);
        let den = crate::transforms::expand::expand(arena, den);
        // A parameter that multiplies the whole numerator or denominator.
        let (num_k, num) = parameter_content(arena, num, &[s, c]).unwrap_or((arena.one, num));
        let (den_k, den) = parameter_content(arena, den, &[s, c]).unwrap_or((arena.one, den));
        let kappa = arena.div(num_k, den_k);
        (
            kappa,
            trig_rational_in_u(arena, (num, den), (s, c), u, u_sym)?,
        )
    } else {
        let result = match odd_algebraic_in_u(arena, sub, (s, c), u) {
            Some(r) => r,
            None => match tan_algebraic_in_u(arena, sub, (s, c), u) {
                Some(r) => r,
                None => half_angle_radical_in_u(arena, sub, (s, c), u)?,
            },
        };
        (arena.one, result)
    };
    let l = arena.mul(&[rate, var]);
    let l = arena.add(&[l, shift]);
    let back = arena.subs_structural(result_u, u, l);
    let minus_one = arena.int(-1);
    let inv_rate = arena.pow(rate, minus_one);
    let back = arena.mul(&[kappa, inv_rate, back]);
    let back = crate::transforms::eval::eval(arena, back);
    if candidate_rejected(arena, expr, back, var, var_sym) {
        tracing::debug!(
            "integrate: rejecting an unverified trigonometric-substitution closed form"
        );
        return None;
    }
    Some(back)
}

/// The terms of the polynomial `p` in the symbols `vars` grouped by their
/// constant factors with free parameters: `p = Σ πₖ·pₖ` with every `pₖ` free
/// of parameters (`A + B·s + C·s²` = `A·1 + B·s + C·s²`).
fn parameter_terms(arena: &mut Arena, p: ExprId, vars: &[ExprId]) -> Vec<(ExprId, ExprId)> {
    let p = crate::transforms::expand::expand(arena, p);
    let terms: Vec<ExprId> = match arena.node(p) {
        ExprNode::Add(children) => children.to_vec(),
        _ => vec![p],
    };
    let mut groups: Vec<(ExprId, Vec<ExprId>)> = Vec::new();
    for term in terms {
        let factors: Vec<ExprId> = match arena.node(term) {
            ExprNode::Mul(children) => children.to_vec(),
            _ => vec![term],
        };
        let mut key: Vec<ExprId> = Vec::new();
        let mut part: Vec<ExprId> = Vec::new();
        for f in factors {
            let dependent = vars.iter().any(|&v| contains_var_id(arena, f, v));
            if !dependent && !crate::base::walk::free_symbols(arena, f).is_empty() {
                key.push(f);
            } else {
                part.push(f);
            }
        }
        let key = arena.mul(&key);
        let part = arena.mul(&part);
        match groups.iter_mut().find(|(k, _)| *k == key) {
            Some((_, parts)) => parts.push(part),
            None => groups.push((key, vec![part])),
        }
    }
    groups
        .into_iter()
        .map(|(k, parts)| (k, arena.add(&parts)))
        .collect()
}

/// [`try_trig_rational_substitution`] for an integrand with the variable
/// also outside the trigonometric functions, polynomially (`(c + d·x)²·
/// cos²(a + b·x)·sin³(a + b·x)`, Rubi 4.x.10, 4.7.3): `sub` is the
/// integrand with the trigonometric functions written in `s = sin u`, `c =
/// cos u` (`sc`), `u = rate·x + shift` (`linear`).  With `x = (u −
/// shift)/rate` the integrand in `u` has `u` as every argument (multiple
/// angles expanded: `csc(a + b·x)·sin(3a + 3b·x)` is `3 − 4·sin²u`), and
/// goes through the whole pipeline (by parts takes linear arguments only
/// as `x` itself).  `None` when the substitution is `u = x`, the variable
/// is not polynomial in `sub`, or no answer is found.
fn trig_polynomial_substitution(
    arena: &mut Arena,
    expr: ExprId,
    sub: ExprId,
    sc: (ExprId, ExprId),
    linear: (ExprId, ExprId),
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    let (s, c) = sc;
    let (rate, shift) = linear;
    if !is_rational_in_all(arena, sub, &[s, c])
        || crate::transforms::solve::symbolic_poly_coeffs(arena, sub, var).is_none()
    {
        return None;
    }
    let u = arena.symbol("__trp");
    let minus_shift = arena.neg(shift);
    let u_minus = arena.add(&[u, minus_shift]);
    let x_of_u = arena.div(u_minus, rate);
    let in_u = arena.subs_structural(sub, var, x_of_u);
    let sin_u = arena.sin(u);
    let cos_u = arena.cos(u);
    let in_u = arena.subs_structural(in_u, s, sin_u);
    let in_u = arena.subs_structural(in_u, c, cos_u);
    let in_u = arena.div(in_u, rate);
    let in_u = crate::transforms::eval::eval(arena, in_u);
    // `u = x` with nothing rewritten: the pipeline has seen this integrand.
    let same = arena.subs_structural(in_u, u, var);
    if same == expr {
        return None;
    }
    tracing::debug!(integrand_u = %arena.display(in_u), "integrate: polynomial times trigonometric functions of a linear argument");
    let res = integrate_nested_generic(arena, in_u, u)?;
    let l = arena.mul(&[rate, var]);
    let l = arena.add(&[l, shift]);
    let back = arena.subs_structural(res, u, l);
    let back = crate::transforms::eval::eval(arena, back);
    if candidate_rejected(arena, expr, back, var, var_sym) {
        return None;
    }
    Some(back)
}

/// `∫ R(sin u, cos u) du` for `R = num/den`, polynomials in the symbols
/// `s = sin u`, `c = cos u` (`sc`), by Bioche's rules (see
/// [`try_trig_rational_substitution`]).  With a denominator free of
/// parameters, a numerator `Σ πₖ·pₖ` ([`parameter_terms`]) is integrated term
/// by term: each `∫ pₖ/den` is free of parameters, so the jump of an
/// answer through `tan w` is decided for it (`∫ (A + B·cos u)/(1 + cos u)²
/// du`; the jump of the whole depends on whether `A`, `B` are real).
fn trig_rational_in_u(
    arena: &mut Arena,
    fraction: (ExprId, ExprId),
    sc: (ExprId, ExprId),
    u: ExprId,
    u_sym: SymbolId,
) -> Option<ExprId> {
    let (num, den) = fraction;
    let (s, c) = sc;
    let parametric = |arena: &Arena, e: ExprId| {
        crate::base::walk::free_symbols(arena, e)
            .into_iter()
            .any(|v| v != s && v != c)
    };
    let groups = if !parametric(arena, den) && parametric(arena, num) {
        parameter_terms(arena, num, &[s, c])
    } else {
        Vec::new()
    };
    if groups.len() < 2 {
        return trig_rational_single(arena, fraction, sc, u, u_sym);
    }
    let mut sum: Vec<ExprId> = Vec::with_capacity(groups.len());
    for (key, part) in groups {
        let r = trig_rational_single(arena, (part, den), sc, u, u_sym)?;
        sum.push(arena.mul(&[key, r]));
    }
    Some(arena.add(&sum))
}

/// [`trig_rational_in_u`] for one fraction.
fn trig_rational_single(
    arena: &mut Arena,
    fraction: (ExprId, ExprId),
    sc: (ExprId, ExprId),
    u: ExprId,
    u_sym: SymbolId,
) -> Option<ExprId> {
    let (num, den) = fraction;
    let (s, c) = sc;
    let ratio = arena.div(num, den);
    if !contains_var_id(arena, ratio, s) && !contains_var_id(arena, ratio, c) {
        return Some(arena.mul(&[ratio, u]));
    }
    let t = arena.symbol("__trt");
    let ExprNode::Symbol(t_sym) = *arena.node(t) else {
        return None;
    };
    let sin_u = arena.sin(u);
    let cos_u = arena.cos(u);
    // `t = cos u` when `R` is odd in `sin u`, `t = sin u` when odd in `cos u`.
    for (w, o, back, sign) in [(s, c, cos_u, -1), (c, s, sin_u, 1)] {
        let (Some(pn), Some(pd)) = (parity_in(arena, num, w), parity_in(arena, den, w)) else {
            continue;
        };
        if pn == pd {
            continue;
        }
        let Some(in_t) = odd_part_in_t(arena, (num, den), (pn, pd), w, o, t) else {
            continue;
        };
        let sign = arena.int(sign);
        let in_t = arena.mul(&[sign, in_t]);
        let in_t = clear_nested_fractions(arena, in_t, t);
        tracing::debug!(integrand_t = %arena.display(in_t), "bioche: integrand in sin u or cos u");
        let g = integrate_node(arena, in_t, t, t_sym, 18);
        if crate::base::walk::has_unevaluated(arena, g)
            || candidate_rejected(arena, in_t, g, t, t_sym)
        {
            continue;
        }
        return Some(arena.subs_structural(g, t, back));
    }
    let r_s = arena.subs_structural(ratio, s, sin_u);
    let r_u = arena.subs_structural(r_s, c, cos_u);
    let r_u = crate::transforms::eval::eval(arena, r_u);
    let lam = arena.symbol("__trl");
    let joint = |arena: &mut Arena, p: ExprId| -> Option<usize> {
        let ls = arena.mul(&[lam, s]);
        let lc = arena.mul(&[lam, c]);
        let p = arena.subs_structural(p, s, ls);
        let p = arena.subs_structural(p, c, lc);
        parity_in(arena, p, lam)
    };
    if let (Some(pn), Some(pd)) = (joint(arena, num), joint(arena, den))
        && pn == pd
        && let Some(result) = try_bioche_tan_substitution(arena, r_u, u, u_sym, 20, true)
    {
        return Some(result);
    }
    try_weierstrass_substitution(arena, r_u, u, u_sym, 20)
}
/// `∫ A(sin u, cos u) du` for `A = R(sin u, cos u)·√(1 + ε·w)^p` (`w` = `sin u`
/// or `cos u`, `ε = ±1`, every radical `(κ·(1 + ε·w))^(k/2)` of the
/// integrand of that one base: Rubi 4.1.1.1, 4.1.1.2, 4.2.1.1 `(a + a·sin(c +
/// d·x))^(7/2)`, `cos²(c + d·x)·√(a + a·sin(c + d·x))`), `R` rational with
/// poles only where `1 + ε·w = 0`.  `(κ·p)^r = κ^r·p^r` for real `p ≥ 0`, and
/// with `v = u/2 + φ` (`φ = ε·π/4` for `sin u`, `π/2` or `0` for `1 ± cos u`)
/// `1 + ε·w = 2·sin²v`, so `√(1 + ε·w) = √2·σ·sin v` with `σ = sign(sin v)`:
/// the integrand is `σ·g(v)·dv` with `g` rational in `sin v`, `cos v` and
/// `g(v + π) = −g(v)`.
///
/// Where `g` is a polynomial (the integrand is continuous), its parts odd
/// in `sin v` and in `cos v` integrate (`t = cos v`, `t = sin v`) to an
/// antiderivative `G` with `G(v + π) = −G(v)`, and `F = σ·G − 2·G(0)·⌊v/π⌋`
/// is continuous at the zeros `v = kπ` of `sin v`, where `σ` flips (`σ·G`
/// steps by `2·G(0)` at each).  `σ·G` is written without `σ`, with
/// `σ·sin v = √(1 + ε·w)/√2` and `σ·cos v = √(1 + ε·w)·sin 2v/(√2·(1 + ε·w))`:
/// `∫ √(1 + sin x) dx = −2·cos x/√(1 + sin x) + 4√2·⌊(x/2 + π/4)/π⌋`.
/// Rubi's answers are the first term alone and jump by `4√2` at every
/// zero of `1 + sin x`; SymPy 1.14 leaves these integrands unevaluated.
/// Where `g` has poles (at `sin v = 0` only) `F = σ·G`.  `None` for other
/// radicals, a part of `A` with an even power of the radical, poles of `R`
/// elsewhere, or no answer for `G`.
fn half_angle_radical_in_u(
    arena: &mut Arena,
    e: ExprId,
    sc: (ExprId, ExprId),
    u: ExprId,
) -> Option<ExprId> {
    let (s, c) = sc;
    // The radicals `B^(k/2)` and their common base `κ·(1 + ε·w)`.
    let mut base_kind: Option<(ExprId, ExprId)> = None;
    let mut radicals: Vec<(ExprId, ExprId, i64, ExprId)> = Vec::new();
    for id in crate::base::walk::post_order_ids(arena, e) {
        let ExprNode::Pow(base, ex) = *arena.node(id) else {
            continue;
        };
        let Some(r) = arena.as_num(ex).cloned() else {
            continue;
        };
        if r.is_integer() || !(contains_var_id(arena, base, s) || contains_var_id(arena, base, c)) {
            continue;
        }
        if *r.denom() != num_bigint::BigInt::from(2) {
            return None;
        }
        let k = i64::try_from(r.numer()).ok()?;
        let (w, other) = if contains_var_id(arena, base, c) {
            (c, s)
        } else {
            (s, c)
        };
        if contains_var_id(arena, base, other) {
            return None;
        }
        let coeffs = crate::transforms::solve::symbolic_poly_coeffs(arena, base, w)?;
        let [kappa, b1] = coeffs.as_slice() else {
            return None;
        };
        let eps = arena.div(*b1, *kappa);
        let eps = crate::transforms::eval::eval(arena, eps);
        let eps_one = arena.as_num(eps).filter(|q| q.abs().is_one())?.clone();
        let eps = arena.num_ratio(eps_one);
        match base_kind {
            None => base_kind = Some((w, eps)),
            Some(kind) if kind == (w, eps) => {}
            Some(_) => return None,
        }
        radicals.push((id, *kappa, k, w));
    }
    let (w, eps) = base_kind?;
    let eps_positive = arena.as_num(eps).is_some_and(|q| q.is_positive());
    // `√(1 + ε·w)` as the symbol `rad`: `B^(k/2) = κ^(k/2)·rad^k`.
    let rad = arena.symbol("__hrad");
    let mut sub = e;
    for &(id, kappa, k, _) in &radicals {
        let half_k = arena.rational(k, 2);
        let kappa_pow = arena.pow(kappa, half_k);
        let k_id = arena.int(k);
        let rad_pow = arena.pow(rad, k_id);
        let repl = arena.mul(&[kappa_pow, rad_pow]);
        sub = arena.subs_structural(sub, id, repl);
    }
    let ExprNode::Symbol(rad_sym) = *arena.node(rad) else {
        return None;
    };
    if !is_rational_in_all(arena, sub, &[s, c, rad]) || !is_rational_in(arena, sub, rad_sym) {
        return None;
    }
    // `sub = rad·E₁` with `E₁` free of `rad` after `rad² = 1 + ε·w`.
    let (num, den) = crate::poly::polybridge::as_numer_denom(arena, sub);
    let one = arena.one;
    let eps_w = arena.mul(&[eps, w]);
    let rad_sq = arena.add(&[one, eps_w]);
    let reduce_rad = |arena: &mut Arena, p: ExprId| -> Option<(ExprId, ExprId)> {
        let cs = crate::transforms::solve::symbolic_poly_coeffs(arena, p, rad)?;
        let mut even: Vec<ExprId> = Vec::new();
        let mut odd: Vec<ExprId> = Vec::new();
        for (j, &k) in cs.iter().enumerate() {
            let e = arena.int(i64::try_from(j / 2).ok()?);
            let power = arena.pow(rad_sq, e);
            let term = arena.mul(&[k, power]);
            if j % 2 == 0 {
                even.push(term);
            } else {
                odd.push(term);
            }
        }
        let even = arena.add(&even);
        let odd = arena.add(&odd);
        Some((
            crate::transforms::expand::expand(arena, even),
            crate::transforms::expand::expand(arena, odd),
        ))
    };
    let (n_even, n_odd) = reduce_rad(arena, num)?;
    let (d_even, d_odd) = reduce_rad(arena, den)?;
    // `(n₀ + rad·n₁)/(d₀ + rad·d₁)`: only `rad·n₁/d₀` and `n₀/(rad·d₁)` are
    // odd in `rad`.
    let e1 = if arena.is_zero_structural(n_even) && arena.is_zero_structural(d_odd) {
        arena.div(n_odd, d_even)
    } else if arena.is_zero_structural(n_odd) && arena.is_zero_structural(d_even) {
        // `n₀/(rad·d₁) = rad·n₀/((1 + ε·w)·d₁)`.
        let d = arena.mul(&[rad_sq, d_odd]);
        arena.div(n_even, d)
    } else {
        return None;
    };
    // `v = u/2 + φ`; `sin u`, `cos u` in `S = sin v`, `C = cos v`.
    let big_s = arena.symbol("__hs");
    let big_c = arena.symbol("__hc");
    let two = arena.int(2);
    let two_sc = arena.mul(&[two, big_s, big_c]);
    let s_sq = arena.pow(big_s, two);
    let c_sq = arena.pow(big_c, two);
    let c2_minus_s2 = arena.sub(c_sq, s_sq);
    let neg = |arena: &mut Arena, z: ExprId| arena.neg(z);
    let w_is_sin = w == s;
    let (sin_u, cos_u, phi) = match (w_is_sin, eps_positive) {
        (true, true) => (neg(arena, c2_minus_s2), two_sc, arena.rational(1, 4)),
        (true, false) => (c2_minus_s2, neg(arena, two_sc), arena.rational(-1, 4)),
        (false, true) => (
            neg(arena, two_sc),
            neg(arena, c2_minus_s2),
            arena.rational(1, 2),
        ),
        (false, false) => (two_sc, c2_minus_s2, arena.zero),
    };
    let pi = arena.pi();
    let phi = arena.mul(&[phi, pi]);
    // `g(v) = 2·√2·S·E₁(sin u, cos u)` (`du = 2·dv`, `rad = √2·σ·S`).
    let g = arena.subs_structural(e1, s, sin_u);
    let g = arena.subs_structural(g, c, cos_u);
    let sqrt2 = arena.sqrt(two);
    let g = arena.mul(&[two, sqrt2, big_s, g]);
    let g = clear_nested_fractions(arena, g, big_s);
    let (g_num, g_den) = crate::poly::polybridge::as_numer_denom(arena, g);
    // Reduce `C² = 1 − S²` in both: `A(S) + C·B(S)`.
    let reduce_c = |arena: &mut Arena, p: ExprId| -> Option<(ExprId, ExprId)> {
        let cs = crate::transforms::solve::symbolic_poly_coeffs(arena, p, big_c)?;
        let one = arena.one;
        let s_sq = arena.pow(big_s, two);
        let minus = arena.neg(s_sq);
        let one_minus = arena.add(&[one, minus]);
        let mut even: Vec<ExprId> = Vec::new();
        let mut odd: Vec<ExprId> = Vec::new();
        for (j, &k) in cs.iter().enumerate() {
            let e = arena.int(i64::try_from(j / 2).ok()?);
            let power = arena.pow(one_minus, e);
            let term = arena.mul(&[k, power]);
            if j % 2 == 0 {
                even.push(term);
            } else {
                odd.push(term);
            }
        }
        let even = arena.add(&even);
        let odd = arena.add(&odd);
        Some((
            crate::transforms::expand::expand(arena, even),
            crate::transforms::expand::expand(arena, odd),
        ))
    };
    let (a_part, b_part) = reduce_c(arena, g_num)?;
    let (den_s, den_c) = reduce_c(arena, g_den)?;
    if !arena.is_zero_structural(den_c) {
        return None;
    }
    // The denominator: a constant times `Sᵐ`.
    let den_coeffs = crate::transforms::solve::symbolic_poly_coeffs(arena, den_s, big_s)?;
    let nonzero: Vec<usize> = (0..den_coeffs.len())
        .filter(|&j| !arena.is_zero_structural(den_coeffs[j]))
        .collect();
    let [m] = nonzero.as_slice() else {
        return None;
    };
    let mut m = *m;
    // `A + C·B` vanishes to order `k` at `S = 0` (`C = ±1` there) exactly when
    // `Sᵏ` divides `A` and `B`: those powers cancel against `Sᵐ` (`cos⁶u/√(1 +
    // sin u)` has no pole where `σ` flips).
    let lowest = |arena: &mut Arena, p: ExprId| -> Option<usize> {
        if arena.is_zero_structural(p) {
            return Some(usize::MAX);
        }
        let cs = crate::transforms::solve::symbolic_poly_coeffs(arena, p, big_s)?;
        cs.iter().position(|&k| !arena.is_zero_structural(k))
    };
    let k = lowest(arena, a_part)?.min(lowest(arena, b_part)?).min(m);
    let (a_part, b_part, den_s) = if k > 0 {
        let shift = |arena: &mut Arena, p: ExprId| -> Option<ExprId> {
            if arena.is_zero_structural(p) {
                return Some(p);
            }
            let cs = crate::transforms::solve::symbolic_poly_coeffs(arena, p, big_s)?;
            let mut terms: Vec<ExprId> = Vec::new();
            for (j, &c) in cs.iter().enumerate().skip(k) {
                let e = arena.int(i64::try_from(j - k).ok()?);
                let power = arena.pow(big_s, e);
                terms.push(arena.mul(&[c, power]));
            }
            Some(arena.add(&terms))
        };
        let a = shift(arena, a_part)?;
        let b = shift(arena, b_part)?;
        let d = shift(arena, den_s)?;
        m -= k;
        (a, b, d)
    } else {
        (a_part, b_part, den_s)
    };
    let v = arena.symbol("__hv");
    let ExprNode::Symbol(v_sym) = *arena.node(v) else {
        return None;
    };
    let b_times_c = arena.mul(&[big_c, b_part]);
    let half = arena.rational(1, 2);
    let half_u = arena.mul(&[half, u]);
    let v_of_u = arena.add(&[half_u, phi]);
    if m > 0 {
        // Poles where `σ` flips: `F = σ·G`, with `σ = √2·sin v/√(1 + ε·w)`
        // (analytic in the parameters of `u`, as `sign(sin v)` is not).
        let n = arena.add(&[a_part, b_times_c]);
        let big_g = trig_rational_in_u(arena, (n, den_s), (big_s, big_c), v, v_sym)?;
        let back = arena.subs_structural(big_g, v, v_of_u);
        let sin_v = arena.sin(v_of_u);
        let sqrt2 = arena.sqrt(two);
        let w_u = if w_is_sin { arena.sin(u) } else { arena.cos(u) };
        let eps_w_u = arena.mul(&[eps, w_u]);
        let one_eps_w = arena.add(&[one, eps_w_u]);
        let rad_u = arena.sqrt(one_eps_w);
        let minus_one = arena.int(-1);
        let inv_rad = arena.pow(rad_u, minus_one);
        tracing::debug!("integrate: a radical of 1 ± sin u (cos u) through the half angle");
        let big_f = arena.mul(&[sqrt2, sin_v, inv_rad, back]);
        return Some(crate::transforms::eval::eval(arena, big_f));
    }
    // `g(v + π) = −g(v)`: `A` odd and `B` even in `S`.  Each part goes
    // through Bioche's parity routes (`t = cos v`, `t = sin v`) to a
    // polynomial odd in `cos v` (`sin v`), with no constant: `G(v + π) =
    // −G(v)`.
    let odd_or_zero = |arena: &mut Arena, p: ExprId, parity: usize| {
        arena.is_zero_structural(p) || parity_in(arena, p, big_s) == Some(parity)
    };
    if !odd_or_zero(arena, a_part, 1) || !odd_or_zero(arena, b_part, 0) {
        return None;
    }
    // `σ·G` without `σ`: `σ·sin v = rad/√2` and `σ·cos v = rad·sin 2v/(√2·(1 +
    // ε·w))` (`sin²v = (1 + ε·w)/2`), with `rad = √(1 + ε·w)` of `u`: a form
    // analytic in the parameters of `u`, as Rubi writes these answers.
    let sin_u_node = arena.sin(u);
    let cos_u_node = arena.cos(u);
    let w_u = if w_is_sin { sin_u_node } else { cos_u_node };
    let eps_w_u = arena.mul(&[eps, w_u]);
    let one_eps_w = arena.add(&[one, eps_w_u]);
    let s2_u = arena.mul(&[half, one_eps_w]);
    let minus_s2 = arena.neg(s2_u);
    let c2_u = arena.add(&[one, minus_s2]);
    let sin_2v = match (w_is_sin, eps_positive) {
        (true, true) => cos_u_node,
        (true, false) => arena.neg(cos_u_node),
        (false, true) => arena.neg(sin_u_node),
        (false, false) => sin_u_node,
    };
    let mut over_s: Vec<ExprId> = Vec::new();
    let mut over_c: Vec<ExprId> = Vec::new();
    let mut at_zero: Vec<ExprId> = Vec::new();
    let sin_v = arena.sin(v);
    let cos_v = arena.cos(v);
    for (part, fv, sym, sq, out) in [
        (b_times_c, sin_v, big_s, s2_u, &mut over_s),
        (a_part, cos_v, big_c, c2_u, &mut over_c),
    ] {
        if arena.is_zero_structural(part) {
            continue;
        }
        let big_g = trig_rational_in_u(arena, (part, den_s), (big_s, big_c), v, v_sym)?;
        let p = arena.subs_structural(big_g, fv, sym);
        let p = crate::transforms::eval::eval(arena, p);
        if contains_var_id(arena, p, v) || parity_in(arena, p, sym) != Some(1) {
            return None;
        }
        let coeffs = crate::transforms::solve::symbolic_poly_coeffs(arena, p, sym)?;
        for (j, &k) in coeffs.iter().enumerate().skip(1).step_by(2) {
            let e = arena.int(i64::try_from(j / 2).ok()?);
            let power = arena.pow(sq, e);
            out.push(arena.mul(&[k, power]));
            if sym == big_c {
                at_zero.push(k);
            }
        }
    }
    let rad_u = arena.sqrt(one_eps_w);
    let inv_sqrt2 = {
        let r = arena.sqrt(two);
        let minus_one = arena.int(-1);
        arena.pow(r, minus_one)
    };
    let p_s = arena.add(&over_s);
    let p_c = arena.add(&over_c);
    let ratio = arena.div(sin_2v, one_eps_w);
    let c_term = arena.mul(&[ratio, p_c]);
    let bracket = arena.add(&[p_s, c_term]);
    let main = arena.mul(&[rad_u, inv_sqrt2, bracket]);
    // `σ·G` steps by `2·G(0)` at every `v = kπ`; `G(0)` is the sum of the
    // coefficients of the odd polynomial in `cos v` (`sin 0 = 0`).
    let g0 = arena.add(&at_zero);
    let g0 = crate::transforms::eval::eval(arena, g0);
    let minus_two = arena.int(-2);
    let minus_one = arena.int(-1);
    let inv_pi = arena.pow(pi, minus_one);
    let v_over_pi = arena.mul(&[v_of_u, inv_pi]);
    let fl = arena.floor(v_over_pi);
    let correction = arena.mul(&[minus_two, g0, fl]);
    tracing::debug!("integrate: a radical of 1 ± sin u (cos u) through the half angle");
    let big_f = arena.add(&[main, correction]);
    Some(crate::transforms::eval::eval(arena, big_f))
}

/// `∫ A(sin u, cos u) du` for an algebraic `A` odd in `cos u` with every
/// radical a function of `sin u` alone (`t = sin u`), or odd in `sin u`
/// with radicals of `cos u` (`t = cos u`): Bioche's first two rules of
/// [`try_trig_rational_substitution`] for radicals (`cos⁵u·√(a + b·sin u)`,
/// Rubi 4.1.1.2, is `∫ (1 − t²)²·√(a + b·t) dt`).  The integrand in `t`
/// goes through the whole pipeline; `G(sin u)`, `G(cos u)` need no
/// correction (the substitutions have no poles).
fn odd_algebraic_in_u(
    arena: &mut Arena,
    e: ExprId,
    sc: (ExprId, ExprId),
    u: ExprId,
) -> Option<ExprId> {
    let (s, c) = sc;
    let t = arena.symbol("__toa");
    // The bases of the radicals: every one must be free of `w`.
    let radical_bases: Vec<ExprId> = crate::base::walk::post_order_ids(arena, e)
        .into_iter()
        .filter_map(|id| match *arena.node(id) {
            ExprNode::Pow(base, ex) if !arena.as_num(ex).is_some_and(|r| r.is_integer()) => {
                Some(base)
            }
            _ => None,
        })
        .collect();
    let sin_u = arena.sin(u);
    let cos_u = arena.cos(u);
    for (w, o, back, sign) in [(c, s, sin_u, 1), (s, c, cos_u, -1)] {
        let ExprNode::Symbol(w_sym) = *arena.node(w) else {
            continue;
        };
        if radical_bases.iter().any(|&b| contains_var_id(arena, b, w))
            || !is_rational_in(arena, e, w_sym)
        {
            continue;
        }
        let (num, den) = crate::poly::polybridge::as_numer_denom(arena, e);
        let (Some(pn), Some(pd)) = (parity_in(arena, num, w), parity_in(arena, den, w)) else {
            continue;
        };
        if pn == pd {
            continue;
        }
        let Some(in_t) = odd_part_in_t(arena, (num, den), (pn, pd), w, o, t) else {
            continue;
        };
        let sign = arena.int(sign);
        let in_t = arena.mul(&[sign, in_t]);
        let in_t = crate::transforms::eval::eval(arena, in_t);
        if contains_var_id(arena, in_t, s) || contains_var_id(arena, in_t, c) {
            continue;
        }
        tracing::debug!(integrand_t = %arena.display(in_t), "integrate: an algebraic function through t = sin u or cos u");
        let Some(g) = integrate_nested_generic(arena, in_t, t) else {
            continue;
        };
        return Some(arena.subs_structural(g, t, back));
    }
    None
}

/// `∫ A(sin u, cos u) du` for an algebraic function `A` of `tan u` alone
/// (`s = sin u`, `c = cos u` in `sc`): radicals of `k·tan u`, `a + b·tan u`,
/// `a + b·tan²u` (Rubi 4.3, 4.4: `√(d·tan u)`, `(a + i·a·tan u)^(7/2)`) times a
/// rational function of `sin u`, `cos u` invariant under `u → u + π`.
/// With `s = t·c`, `A` is a function of `t` and of `c` in even (or odd)
/// integer powers outside the radicals, `c² = 1/(1 + t²)`, `du = dt/(1 + t²)`;
/// the integrand in `t` goes through the whole pipeline (its radical
/// substitutions).  The answer `G(tan u)` gets the floor correction of
/// [`crate::transforms::trig_integ::continuous_through_tan`] where its
/// jump is decided; otherwise it is kept only when the integrand is not
/// continuous across the poles of `tan u` ([`tan_pole_continuous`]: `√(tan
/// u)` is real on one side only), so that no answer jumps where the
/// integrand is continuous.
fn tan_algebraic_in_u(
    arena: &mut Arena,
    e: ExprId,
    sc: (ExprId, ExprId),
    u: ExprId,
) -> Option<ExprId> {
    let (s, c) = sc;
    let ExprNode::Symbol(c_sym) = *arena.node(c) else {
        return None;
    };
    let t = arena.symbol("__tat");
    let ExprNode::Symbol(t_sym) = *arena.node(t) else {
        return None;
    };
    let tc = arena.mul(&[t, c]);
    let e = arena.subs_structural(e, s, tc);
    let e = crate::transforms::eval::eval(arena, e);
    if contains_var_id(arena, e, s) || !is_rational_in(arena, e, c_sym) {
        return None;
    }
    let (num, den) = crate::poly::polybridge::as_numer_denom(arena, e);
    let num_c = crate::transforms::solve::symbolic_poly_coeffs(arena, num, c)?;
    let den_c = crate::transforms::solve::symbolic_poly_coeffs(arena, den, c)?;
    let parity = |arena: &mut Arena, cs: &[ExprId]| -> Option<usize> {
        let mut nonzero = [false, false];
        for (j, &k) in cs.iter().enumerate() {
            if !arena.is_zero_structural(k) {
                nonzero[j % 2] = true;
            }
        }
        match nonzero {
            [true, false] => Some(0),
            [false, true] => Some(1),
            _ => None,
        }
    };
    let shift = parity(arena, &num_c)?;
    if parity(arena, &den_c)? != shift {
        return None;
    }
    tracing::debug!("integrate: an algebraic function of tan u, through t = tan u");
    let one = arena.one;
    let two = arena.int(2);
    let t_sq = arena.pow(t, two);
    let one_plus_t_sq = arena.add(&[one, t_sq]);
    let in_t = |arena: &mut Arena, cs: &[ExprId]| -> Option<ExprId> {
        let mut terms: Vec<ExprId> = Vec::new();
        for (j, &k) in cs.iter().enumerate().skip(shift).step_by(2) {
            let e = arena.int(-i64::try_from((j - shift) / 2).ok()?);
            let c_pow = arena.pow(one_plus_t_sq, e);
            terms.push(arena.mul(&[k, c_pow]));
        }
        Some(arena.add(&terms))
    };
    let num_t = in_t(arena, &num_c)?;
    let den_t = in_t(arena, &den_c)?;
    let dt = arena.mul(&[den_t, one_plus_t_sq]);
    let integrand_t = arena.div(num_t, dt);
    let integrand_t = crate::transforms::eval::eval(arena, integrand_t);
    if contains_var_id(arena, integrand_t, c) {
        return None;
    }
    tracing::debug!(integrand_t = %arena.display(integrand_t), "tan u: algebraic integrand in t");
    let g = integrate_nested_generic(arena, integrand_t, t)?;
    if let Some(r) = crate::transforms::trig_integ::continuous_through_tan(arena, g, t, t_sym, u) {
        return Some(r);
    }
    if !tan_pole_continuous(arena, integrand_t, t).is_false() {
        tracing::debug!("tan u: the jump at the poles of tan u is not decided");
        return None;
    }
    let tan_u = arena.tan(u);
    let back = arena.subs_structural(g, t, tan_u);
    Some(crate::transforms::trig_integ::atan_tan_terms_to_argument(
        arena, back, u,
    ))
}

/// Is the integrand `f(u) = h(t)·(1 + t²)` (`t = tan u`) continuous across
/// the poles of `tan u`, for its parameters at the generic values of
/// [`FTC_PARAMETER_VALUES`]?  By the limits of `(1 + t²)·h(t)` at `t = ±∞`:
/// `False` when one is infinite or they differ (`√(tan u)` is real on one
/// side only), `True` when they agree (to `10⁻⁹`: only `False` is acted on,
/// by refusing), `Unknown` when a limit is not found.
fn tan_pole_continuous(arena: &mut Arena, h: ExprId, t: ExprId) -> Truth {
    let one = arena.one;
    let two = arena.int(2);
    let t_sq = arena.pow(t, two);
    let one_plus_t_sq = arena.add(&[one, t_sq]);
    let mut k = arena.mul(&[h, one_plus_t_sq]);
    let mut params: Vec<(String, ExprId)> = Vec::new();
    for s in crate::base::walk::free_symbols(arena, k) {
        if s == t {
            continue;
        }
        let ExprNode::Symbol(sid) = *arena.node(s) else {
            return Truth::Unknown;
        };
        params.push((arena.symbol_name(sid).to_owned(), s));
    }
    if params.len() > FTC_PARAMETER_VALUES.len() {
        return Truth::Unknown;
    }
    params.sort_by(|a, b| a.0.cmp(&b.0));
    for (j, &(_, s)) in params.iter().enumerate() {
        let (p, d) = FTC_PARAMETER_VALUES[j];
        let value = arena.rational(p, d);
        k = crate::transforms::subs::subs(arena, k, s, value);
    }
    let k = crate::transforms::eval::eval(arena, k);
    let mut values = Vec::with_capacity(2);
    for point in [arena.infinity(), arena.neg_infinity()] {
        let Ok(v) = arena.limit_expr(k, t, point) else {
            return Truth::Unknown;
        };
        let v = crate::transforms::eval::eval(arena, v);
        if crate::base::walk::has_unevaluated(arena, v) || contains_var_id(arena, v, t) {
            return Truth::Unknown;
        }
        if contains_non_finite(arena, v) {
            return Truth::False;
        }
        let Ok(z) = crate::transforms::evalf::evalf_complex64(arena, v) else {
            return Truth::Unknown;
        };
        if !z.re.is_finite() || !z.im.is_finite() {
            return Truth::False;
        }
        values.push(z);
    }
    let gap = (values[0] - values[1]).norm();
    Truth::from_bool(gap <= 1e-9 * values[0].norm().max(1.0))
}

/// Does `e` contain the symbol `v` (free)?
fn contains_var_id(arena: &Arena, e: ExprId, v: ExprId) -> bool {
    match *arena.node(v) {
        ExprNode::Symbol(sid) => contains_var(arena, e, sid),
        _ => true,
    }
}

/// [`jump_kept_undecided`] for the answer `g` of the integrand `h` in `t =
/// tan w`, and inside the trigonometric substitutions of 0.38
/// ([`crate::transforms::trig_integ::strict_jumps_active`]) only
/// where the integrand is not continuous across the poles of `tan w`
/// ([`tan_pole_continuous`] at generic values of the parameters): those
/// routes return no answer that jumps where the integrand is continuous
/// (`∫ cos²u/(a − b·sin⁴u) du`, a `RootSum` over a polynomial with parameters,
/// is refused).
fn undecided_jump_kept(
    arena: &mut Arena,
    h: ExprId,
    g: ExprId,
    t: ExprId,
    t_sym: SymbolId,
) -> bool {
    jump_kept_undecided(arena, g, t_sym)
        && (!crate::transforms::trig_integ::strict_jumps_active()
            || tan_pole_continuous(arena, h, t).is_false())
}

/// May an antiderivative `G(tan w)` (`g` = `G(t)`, `t` the substitution's
/// symbol) whose jump `G(+∞) − G(−∞)` at the poles of `tan w` is not
/// decided be returned uncorrected, continuous only between those poles?
/// Yes when `G` has parameters — the jump depends on their values and, for
/// parameters not declared real, on the sides of the cuts their logarithms
/// approach (`∫ dx/(a + b·cos x)`; SymPy and Rubi answer such integrands so
/// too, and refusing them lost 435 answers of the Rubi suite) — or a
/// `RootSum` whose end values are not found (a logarithm of degree ≥ 2 in
/// `t`).  Any other undecided jump makes the substitution refuse.
pub(super) fn jump_kept_undecided(arena: &Arena, g: ExprId, t_sym: SymbolId) -> bool {
    crate::base::walk::free_symbols(arena, g)
        .iter()
        .any(|&s| !matches!(arena.node(s), ExprNode::Symbol(sid) if *sid == t_sym))
        || crate::base::walk::post_order_ids(arena, g)
            .iter()
            .any(|&id| matches!(arena.node(id), ExprNode::RootSum(..)))
}
