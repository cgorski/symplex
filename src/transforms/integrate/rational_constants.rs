//! rational constants (part of [`integrate`](super)).

use super::*;

/// `∫ R(x) dx` for a rational function `R` with `i` in its coefficients
/// (Gaussian rationals, parameters allowed).  With `i` taken for a symbol
/// `ι` (`ι² = −1`), `R = (N₀ + ι·N₁)/(D₀ + ι·D₁)` with `N₀`, `N₁`, `D₀`, `D₁`
/// free of `i`, and `R = ((N₀D₀ + N₁D₁) + i·(N₁D₀ − N₀D₁))/(D₀² + D₁²)`:
/// two rational functions free of `i`, integrated separately.  This is an
/// identity of rational functions, so no parameter has to be real.  The
/// rational integrators work over ℚ and ℚ(parameters): up to 0.37 `∫ (1 +
/// i·t)⁵/(1 + t²)⁶ dt` stayed unevaluated, and with it the Rubi suite's
/// `(a + i·a·tan(c + d·x))ⁿ` families through `t = tan x`.  For a real
/// variable both parts come out as real antiderivatives (`ln|·|`, `atan`),
/// continuous wherever `R` is.
pub(super) fn try_gaussian_rational(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    let i = arena.i_unit();
    if depth < 2
        || !crate::base::walk::contains(arena, expr, i)
        || !is_rational_in(arena, expr, var_sym)
    {
        return None;
    }
    let iota = arena.symbol("__gi");
    let e = arena.subs_structural(expr, i, iota);
    if crate::base::walk::contains(arena, e, i) {
        return None;
    }
    let e = clear_nested_fractions(arena, e, var);
    let (n, d) = crate::poly::polybridge::as_numer_denom(arena, e);
    let split = |arena: &mut Arena, p: ExprId| -> Option<(ExprId, ExprId)> {
        let coeffs = crate::transforms::solve::symbolic_poly_coeffs(arena, p, iota)?;
        let mut re: Vec<ExprId> = Vec::new();
        let mut im: Vec<ExprId> = Vec::new();
        for (k, &c) in coeffs.iter().enumerate() {
            let c = if (k / 2) % 2 == 0 { c } else { arena.neg(c) };
            if k % 2 == 0 {
                re.push(c);
            } else {
                im.push(c);
            }
        }
        let re = arena.add(&re);
        let im = arena.add(&im);
        Some((
            crate::transforms::eval::eval(arena, re),
            crate::transforms::eval::eval(arena, im),
        ))
    };
    let (n0, n1) = split(arena, n)?;
    let (d0, d1) = split(arena, d)?;
    let (p, q, den) = if arena.is_zero_structural(d1) {
        (n0, n1, d0)
    } else {
        let a = arena.mul(&[n0, d0]);
        let b = arena.mul(&[n1, d1]);
        let p = arena.add(&[a, b]);
        let a = arena.mul(&[n1, d0]);
        let b = arena.mul(&[n0, d1]);
        let q = arena.sub(a, b);
        let two = arena.int(2);
        let a = arena.pow(d0, two);
        let b = arena.pow(d1, two);
        (p, q, arena.add(&[a, b]))
    };
    let mut parts: Vec<ExprId> = Vec::with_capacity(2);
    for (num, factor) in [(p, arena.one), (q, i)] {
        let num = crate::transforms::expand::expand(arena, num);
        if arena.is_zero_structural(num) {
            continue;
        }
        let f = arena.div(num, den);
        let f = clear_nested_fractions(arena, f, var);
        if crate::base::walk::contains(arena, f, iota) {
            return None;
        }
        let big_f = integrate_node(arena, f, var, var_sym, depth - 1);
        if crate::base::walk::has_unevaluated(arena, big_f) {
            return None;
        }
        // Term by term, so that the terms `c·atan(tan w)` of a substitution
        // through `tan w` stay terms (`add_tan_floor` writes them `c·w`).
        let terms: Vec<ExprId> = match arena.node(big_f) {
            ExprNode::Add(children) => children.to_vec(),
            _ => vec![big_f],
        };
        for term in terms {
            parts.push(arena.mul(&[factor, term]));
        }
    }
    tracing::debug!(
        "integrate: a rational function with Gaussian coefficients, by real and imaginary parts"
    );
    let result = arena.add(&parts);
    if candidate_rejected(arena, expr, result, var, var_sym) {
        return None;
    }
    Some(result)
}

/// At most this many distinct constants in the numerator for
/// [`try_rational_with_constant_coefficients`]: each costs one rational
/// integration over the same denominator.
const MAX_CONSTANT_GROUPS: usize = 8;

/// The largest numerator degree [`try_rational_with_constant_coefficients`]
/// takes (the rational integrator's own bound on exponents): its pieces are
/// built as dense polynomials, and `x^1000000000/(x + 1)` would ask for a
/// billion coefficients.
const MAX_CONSTANT_COEFF_DEGREE: usize = 1000;

/// `∫ N(x)/D(x) dx` for `D ∈ ℚ[x]` and a polynomial numerator whose
/// coefficients are constants, not all rational: `N = Σ κₖ·Pₖ(x)` with
/// distinct constants `κₖ` and `Pₖ ∈ ℚ[x]`, and `∫ N/D = Σ κₖ·∫ Pₖ/D`, each
/// piece the rational integrator's over ℚ (Hermite, Rothstein–Trager),
/// complete there.  Linearity is exact, so the pieces' checked answers need
/// no further check.  Up to 0.46 the integrator over ℚ refused such an
/// integrand (a constant like `1/e` is not in ℚ), the one over ℚ(p₁, …)
/// as well (`e` is no symbol), and the search went down nested by parts and
/// sum distributions that found the same answer piece by piece: `∫ (t⁸ +
/// t⁷/e)/(t¹² + 1/2) dt` took 6.4 s, and through it (`x = t²`, then real and
/// imaginary parts) `∫ (√x + 1/e)/(x³ + √2·i/2) dx` 3.6 s in 0.46 and 5.2 s
/// in the 0.47 development code (the slow `fuzz_integrate` unit of its
/// campaign: 10 s in the instrumented build, against a 30 s CI limit);
/// now 0.012 s and 0.041 s.
pub(super) fn try_rational_with_constant_coefficients(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    let expr = arena.mul(dependent);
    if depth < 2 || !is_rational_in(arena, expr, var_sym) {
        return None;
    }
    let (num, den) = crate::poly::polybridge::as_numer_denom(arena, expr);
    if !contains_var(arena, den, var_sym)
        || !is_polynomial_in(arena, num, var, var_sym)
        || crate::poly::polybridge::expr_to_poly(arena, den, var).is_none()
        || crate::poly::polybridge::expr_to_poly(arena, num, var).is_some()
    {
        return None;
    }
    let terms = crate::poly::polybridge::symbolic_multipoly_terms(arena, num, &[var])?;
    // The terms `q·κ·xᵈ` of the numerator, `q` rational.
    let mut monomials: Vec<(usize, Q, ExprId)> = Vec::new();
    for (exps, c) in terms {
        let degree = usize::try_from(exps[0]).ok()?;
        if degree > MAX_CONSTANT_COEFF_DEGREE {
            return None;
        }
        let parts: SmallVec<[ExprId; 6]> = match arena.node(c) {
            ExprNode::Add(children) => children.clone(),
            _ => SmallVec::from_slice(&[c]),
        };
        for part in parts {
            let (q, kappa) = arena.as_coeff_term(part);
            if contains_var(arena, kappa, var_sym) {
                return None;
            }
            monomials.push((degree, q, kappa));
        }
    }
    if monomials.iter().all(|(_, _, kappa)| *kappa == arena.one) {
        return None;
    }
    // κₖ ↦ the coefficients of Pₖ, by degree.
    let mut groups: Vec<(ExprId, Vec<Q>)> = Vec::new();
    for (degree, q, kappa) in monomials {
        let k = match groups.iter().position(|(g, _)| *g == kappa) {
            Some(k) => k,
            None => {
                if groups.len() == MAX_CONSTANT_GROUPS {
                    return None;
                }
                groups.push((kappa, Vec::new()));
                groups.len() - 1
            }
        };
        let coeffs = &mut groups[k].1;
        if coeffs.len() <= degree {
            coeffs.resize(degree + 1, Q::zero());
        }
        coeffs[degree] += q;
    }
    let mut parts: Vec<ExprId> = Vec::with_capacity(groups.len());
    for (kappa, coeffs) in groups {
        let p = Poly::from_coeffs(coeffs);
        if p.is_zero() {
            continue;
        }
        let p = crate::poly::polybridge::poly_to_expr(arena, &p, var);
        let piece = arena.div(p, den);
        let big_f = integrate_node(arena, piece, var, var_sym, depth - 1);
        if crate::base::walk::has_unevaluated(arena, big_f) {
            return None;
        }
        // Term by term, as `try_gaussian_rational` writes its parts.
        let terms: Vec<ExprId> = match arena.node(big_f) {
            ExprNode::Add(children) => children.to_vec(),
            _ => vec![big_f],
        };
        for term in terms {
            parts.push(arena.mul(&[kappa, term]));
        }
    }
    tracing::debug!("integrate: a rational function over ℚ times constants, piece by piece");
    Some(arena.add(&parts))
}

/// Most distinct constants [`try_rational_over_constants`] turns into
/// parameters.
const MAX_CONSTANT_PARAMETERS: usize = 4;

/// `∫ R(x) dx` for a rational function with constant coefficients that are
/// not all rational in its denominator (`1/(x·(x − √2))`, `1/((x − e)(x +
/// π))`): each maximal subterm free of symbols that is not a number (`√2`,
/// `π`) becomes a fresh parameter, the integrator over `ℚ(p₁, …)` integrates,
/// and the constants are put back.  Its answer `F` has `F′ = R` as an
/// identity of rational functions of `x` and the parameters, so the
/// constants may replace the parameters wherever no denominator of `F`
/// vanishes at them (relations such as `√2² = 2` are lost, never used); an
/// answer that becomes undefined there (a `nan`, `zoo`), or that the numeric
/// check of every candidate rejects, gives `None`.  Up to 0.57 `∫ dx/(x·(x −
/// √2))` was unevaluated (the integrator over `ℚ` refuses `√2`, the one over
/// `ℚ(p₁, …)` takes symbols only), and with it `∫ dx/(eˣ − √2)`; SymPy
/// 1.14: `√2·ln(x − √2)/2 − √2·ln(x)/2`.
pub(super) fn try_rational_over_constants(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    if !is_rational_in(arena, expr, var_sym) {
        return None;
    }
    let mut constants: Vec<ExprId> = Vec::new();
    let mut stack = vec![expr];
    while let Some(id) = stack.pop() {
        if arena.as_num(id).is_some() {
            continue;
        }
        if crate::base::walk::free_symbols(arena, id).is_empty() {
            if !constants.contains(&id) {
                if constants.len() == MAX_CONSTANT_PARAMETERS {
                    return None;
                }
                constants.push(id);
            }
            continue;
        }
        match arena.node(id) {
            ExprNode::Add(cs) | ExprNode::Mul(cs) => stack.extend(cs.iter().copied()),
            ExprNode::Pow(b, e) => stack.extend([*b, *e]),
            ExprNode::Neg(c) => stack.push(*c),
            _ => {}
        }
    }
    if constants.is_empty() {
        return None;
    }
    let params: Vec<ExprId> = (0..constants.len())
        .map(|i| arena.symbol(&format!("__rational_constant{i}")))
        .collect();
    let mut in_params = expr;
    for (&c, &p) in constants.iter().zip(&params) {
        in_params = arena.subs_structural(in_params, c, p);
    }
    if constants
        .iter()
        .any(|&c| crate::base::walk::contains(arena, in_params, c))
    {
        return None;
    }
    let big_f = crate::calculus::risch::try_risch_rational_params(arena, in_params, var)?;
    if crate::base::walk::has_unevaluated(arena, big_f) {
        return None;
    }
    let mut back = big_f;
    for (&c, &p) in constants.iter().zip(&params) {
        back = arena.subs_structural(back, p, c);
    }
    let back = crate::transforms::eval::eval(arena, back);
    if crate::base::value_class::holds_special_atom(arena, back)
        || params
            .iter()
            .any(|&p| crate::base::walk::contains(arena, back, p))
        || candidate_rejected(arena, expr, back, var, var_sym)
    {
        return None;
    }
    tracing::debug!("integrate: a rational function over constants as parameters");
    Some(back)
}
