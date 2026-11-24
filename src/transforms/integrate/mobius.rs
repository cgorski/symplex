//! Radicals of a Möbius function: t = B^{1/q}, t = B₁^{1/q}·B₂^{−1/q} (part of [`integrate`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Radicals of a Möbius function: t = B^{1/q}, t = B₁^{1/q}·B₂^{−1/q}
// ═══════════════════════════════════════════════════════════════════════════

/// Largest root index `q` [`try_mobius_radical_substitution`] takes (that
/// of [`try_radical_substitution`]).
const MAX_MOBIUS_ROOT: i64 = 6;

/// `m = (a·x + b)/(c·x + d)` with `a`, `b`, `c`, `d` free of `x`.
struct Mobius {
    a: ExprId,
    b: ExprId,
    c: ExprId,
    d: ExprId,
}

/// The coefficients of `m` as a Möbius function of `var` (after its
/// fractions are combined and cancelled), or `None`.
fn mobius_parts(arena: &mut Arena, m: ExprId, var: ExprId, var_sym: SymbolId) -> Option<Mobius> {
    let m = clear_nested_fractions(arena, m, var);
    let (numer, denom) = crate::poly::polybridge::as_numer_denom(arena, m);
    let mut linear = |e: ExprId| -> Option<(ExprId, ExprId)> {
        if contains_var(arena, e, var_sym) {
            symbolic_linear_coeff_of(arena, e, var, var_sym)
        } else {
            Some((arena.zero, e))
        }
    };
    let (a, b) = linear(numer)?;
    let (c, d) = linear(denom)?;
    if contains_var(arena, a, var_sym)
        || contains_var(arena, b, var_sym)
        || contains_var(arena, c, var_sym)
        || contains_var(arena, d, var_sym)
    {
        return None;
    }
    Some(Mobius { a, b, c, d })
}

/// `(node, base, e)` for every power `base^e` in `expr` whose base depends
/// on the variable and whose exponent is a rational number that is not an
/// integer; `None` when a power of such a base has a symbolic exponent or
/// one that depends on the variable.
fn radical_powers(
    arena: &Arena,
    expr: ExprId,
    var_sym: SymbolId,
) -> Option<Vec<(ExprId, ExprId, Q)>> {
    let mut out = Vec::new();
    for id in crate::base::walk::post_order_ids(arena, expr) {
        if let ExprNode::Pow(base, e) = *arena.node(id)
            && contains_var(arena, base, var_sym)
        {
            let r = arena.as_num(e)?;
            if !r.is_integer() {
                out.push((id, base, r.clone()));
            }
        } else if let ExprNode::Pow(_, e) = *arena.node(id)
            && contains_var(arena, e, var_sym)
        {
            return None;
        }
    }
    Some(out)
}

/// Denominators of at most this degree go to the rational integrators
/// cancelled and expanded ([`rational_in_factored_form`]); the one over
/// `ℚ(p₁, …)` takes up to this degree in that form.
const FACTORED_FORM_DEGREE: i64 = 10;

/// The largest exponent of `v` in the expanded polynomial `p`.
fn expanded_degree_in(arena: &Arena, p: ExprId, v: ExprId) -> i64 {
    let mut deg = 0;
    for id in crate::base::walk::post_order_ids(arena, p) {
        if id == v {
            deg = deg.max(1);
        } else if let ExprNode::Pow(b, k) = *arena.node(id)
            && b == v
            && let Some(k) = arena
                .as_num(k)
                .and_then(|k| i64::try_from(k.to_integer()).ok())
        {
            deg = deg.max(k);
        }
    }
    deg
}

/// A rational function `e` of `v` as one quotient: with parameters, the
/// numerator multiplied out over the denominator's factors as they come
/// (`(a − c·t²)⁸`), which the rational integrator over `ℚ(p₁, …)` uses
/// as its square-free decomposition; without, cancelled and expanded
/// ([`clear_nested_fractions`]).
fn rational_in_factored_form(arena: &mut Arena, e: ExprId, v: ExprId) -> ExprId {
    let e = crate::transforms::eval::eval(arena, e);
    let cleared = clear_nested_fractions(arena, e, v);
    let has_parameters = crate::base::walk::free_symbols(arena, cleared)
        .into_iter()
        .any(|s| s != v);
    let (_, d) = crate::poly::polybridge::as_numer_denom(arena, cleared);
    if !has_parameters || expanded_degree_in(arena, d, v) <= FACTORED_FORM_DEGREE {
        return cleared;
    }
    let (n, d) = crate::poly::polybridge::fraction_parts(arena, e);
    let n = crate::transforms::expand::expand(arena, n);
    let n = crate::transforms::eval::eval(arena, n);
    // Each factor multiplied out on its own: a Möbius image such as
    // `a·b·t² + a·(1 − b·t²)` is the constant `a`.
    let factors: SmallVec<[ExprId; 6]> = match arena.node(d) {
        ExprNode::Mul(children) => children.clone(),
        _ => SmallVec::from_slice(&[d]),
    };
    let mut expanded: SmallVec<[ExprId; 6]> = SmallVec::new();
    for f in factors {
        let (base, k) = match *arena.node(f) {
            ExprNode::Pow(b, k) => (b, Some(k)),
            _ => (f, None),
        };
        let base = crate::transforms::expand::expand(arena, base);
        let base = crate::transforms::eval::eval(arena, base);
        expanded.push(match k {
            Some(k) => arena.pow(base, k),
            None => base,
        });
    }
    let d = arena.mul(&expanded);
    arena.div(n, d)
}

/// Is `e` a rational function of the symbol `v` (sums, products and
/// integer powers of `v` and of constants)?
pub(super) fn is_rational_in(arena: &Arena, e: ExprId, v: SymbolId) -> bool {
    let mut depends: FxHashMap<ExprId, bool> = FxHashMap::default();
    for id in crate::base::walk::post_order_ids(arena, e) {
        let node = arena.node(id);
        let mut below = matches!(node, ExprNode::Symbol(s) if *s == v);
        node.for_each_child(|c| below |= depends.get(&c).copied().unwrap_or(false));
        if below {
            let ok = match node {
                ExprNode::Symbol(_) | ExprNode::Add(_) | ExprNode::Mul(_) | ExprNode::Neg(_) => {
                    true
                }
                ExprNode::Pow(_, ex) => arena.as_num(*ex).is_some_and(|r| r.is_integer()),
                _ => false,
            };
            if !ok {
                return false;
            }
        }
        depends.insert(id, below);
    }
    true
}

/// `expr` with every product `B₁^{e₁}·B₂^{e₂}·r` (`e₁ + e₂` an integer)
/// written `t^{q·e₁}·B₂^{e₁+e₂}·r` — exact for `t = B₁^{1/q}·B₂^{−1/q}`,
/// since `B₁^{e₁} = (B₁^{1/q})^{q·e₁}` and `B₂^{e₂} = (B₂^{−1/q})^{−q·e₂}`
/// for principal powers, and `(B₂^{−1/q})^{−q·k} = B₂^k` for an integer
/// `k`.  `None` when a radical of `B₁` or `B₂` is left over (one without
/// its partner, or a pair whose exponents do not sum to an integer:
/// `∛(1 + x)·∛(1 − x)` is elliptic).
fn pair_radicals(
    arena: &mut Arena,
    expr: ExprId,
    b1: ExprId,
    b2: ExprId,
    t: ExprId,
    q: i64,
) -> Option<ExprId> {
    let radical_of = |arena: &Arena, id: ExprId, base: ExprId| -> Option<(ExprId, Q)> {
        match *arena.node(id) {
            ExprNode::Pow(b, e) if b == base => {
                let r = arena.as_num(e)?;
                (!r.is_integer()).then(|| (id, r.clone()))
            }
            _ => None,
        }
    };
    let q_ratio = Q::from_integer(q.into());
    let mut cur = expr;
    for _ in 0..64 {
        let mut hit = None;
        for id in crate::base::walk::post_order_ids(arena, cur) {
            if let ExprNode::Mul(children) = arena.node(id) {
                let p1 = children.iter().find_map(|&c| radical_of(arena, c, b1));
                let p2 = children.iter().find_map(|&c| radical_of(arena, c, b2));
                if let (Some(p1), Some(p2)) = (p1, p2) {
                    hit = Some((id, children.clone(), p1, p2));
                    break;
                }
            }
        }
        let Some((id, children, (n1, e1), (n2, e2))) = hit else {
            break;
        };
        let sum = &e1 + &e2;
        if !sum.is_integer() {
            return None;
        }
        let mut rest: SmallVec<[ExprId; 6]> = children
            .iter()
            .copied()
            .filter(|&c| c != n1 && c != n2)
            .collect();
        let k = arena.num_ratio(&e1 * &q_ratio);
        rest.push(arena.pow(t, k));
        let s = arena.num_ratio(sum);
        rest.push(arena.pow(b2, s));
        let replacement = arena.mul(&rest);
        cur = arena.subs_structural(cur, id, replacement);
    }
    let leftover = crate::base::walk::post_order_ids(arena, cur)
        .into_iter()
        .any(|id| radical_of(arena, id, b1).is_some() || radical_of(arena, id, b2).is_some());
    (!leftover).then_some(cur)
}

/// `∫ R(x, B^{p₁/q₁}, …) dx` for one base `B = (a·x + b)/(c·x + d)` (a
/// linear `a + b·x` among them), and `∫ R(x, B₁^{e₁}·B₂^{e₂}, …) dx` for
/// two bases whose quotient is such a function, with the radicals in
/// products whose exponents sum to integers (`√(1 + x)/√(1 − x)`,
/// `x^{−7/2}/√(a + b·x)`, `(a + b/x)^{3/2}`): the substitution
/// `t = B^{1/q}` (`t = B₁^{1/q}·B₂^{−1/q}`), `q` the least common
/// denominator of the exponents, makes the integrand rational,
/// `x = (d·tᵠ − b)/(a − c·tᵠ)` and `dx = q·t^{q−1}·(a·d − b·c)/(a − c·tᵠ)²·dt`
/// (SymPy's `sqrt_fractional_linear_rule` for one base; the pairing is the
/// classical one of Chebyshev's binomial integrals).
///
/// Exact for principal powers wherever `t` is: `tᵠ = B` (`B₁/B₂`) holds
/// as an identity of complex numbers, and every radical of the integrand is
/// an integer power of `t` times a rational function of `x` (see
/// [`pair_radicals`]), so the answer `G(t(x))` is an antiderivative at
/// complex values of the integrand too, and needs no `|·|` or `sign`
/// (SymPy writes `√(x + 1)/√(1 − x)` as `√((x + 1)/(1 − x))`, which differs
/// in sign for `x > 1`).  Where the integrand is real, `t` is real, and the
/// rational antiderivative `G` (logarithms at its poles, `atan` of
/// polynomials) is continuous between the poles of the `t`-integrand,
/// which are singularities of the integrand or `x = ∞`.
///
/// An integrand `xᵐ·h(xⁿ)` (`n ≥ 2` the gcd of the exponents of `x` in
/// `h`, `m` rational) is first written in `w = xⁿ`, `xᵐ·dx =
/// w^{(m+1−n)/n}·dw/n` ([`power_substituted`]): the binomial
/// `xᵐ·(a + b·xⁿ)ᵖ` becomes a radical of a linear function of `w` when
/// `(m + 1)/n` is an integer, and a pair with the radical of `w` itself when
/// `(m + 1)/n + p` is — Chebyshev's two elementary cases besides an
/// integer `p` (Gradshteyn–Ryzhik 2.202; Rubi 1.1.3.2).  The radicals of
/// `w` stand for powers of `x` (`w^{1/q}` is `x^{n/q}`, not `(xⁿ)^{1/q}`),
/// and the identities used hold for those as they do for principal powers
/// of `w`.  `t` then has the radical of `w` in its numerator, so that
/// `x = 0`, where `x^{−1}`-type powers change sign but the integrand may
/// be continuous (`√(a + b·x²)`), maps to the finite `t = 0`, not across
/// `t = ∞` (where an `atan` or a pair of complex logarithms of `G` jumps).
pub(super) fn try_mobius_radical_substitution(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    let back = match power_substituted(arena, expr, var, var_sym) {
        Some((f_w, w, n)) => {
            let ExprNode::Symbol(w_sym) = *arena.node(w) else {
                return None;
            };
            let g = mobius_core(arena, f_w, w, w_sym, true)?;
            unmap_power(arena, g, w, var, n)
        }
        None => match mobius_core(arena, expr, var, var_sym, false) {
            Some(g) => g,
            None => mobius_core_split(arena, expr, var, var_sym)?,
        },
    };
    let back = crate::transforms::eval::eval(arena, back);
    if candidate_rejected(arena, expr, back, var, var_sym) {
        tracing::debug!("integrate: rejecting unverified Möbius-radical closed form");
        return None;
    }
    Some(back)
}

/// [`mobius_core`] with every radical base written as monic linear factors
/// times a locally constant coefficient: `(a + b·x)ᵉ = K·(x + a/b)ᵉ` and,
/// for a quadratic `Q = α·x² + β·x + γ` whose discriminant is a square
/// `s²` in `ℚ[parameters]`, `Qᵉ = K·(x − r₁)ᵉ·(x − r₂)ᵉ` with
/// `r = (−β ± s)/(2α)` (`√(b·x + c·x²)`, `√(d² − e²·x²)`,
/// `√(c − a²·c·x²)`; Rubi 1.2.1, 7.3.6).  `K = Qᵉ·∏(x − rᵢ)^{−e}` has
/// derivative 0: it is integrated as a parameter and put back (Rubi's
/// "piecewise constant extraction").  For real parameters it changes
/// value only at a root, where the integrand passes from real to complex
/// — except at a double root (`√(a² + 2a·b·x + b²·x²) = K·(x + a/b)`,
/// real on both sides): there the antiderivative `K·G` is made continuous
/// by `G(r) = 0`, as the `P·|g|` rule does, or refused when `G(r)` is not
/// finite.
fn mobius_core_split(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    let radicals = radical_powers(arena, expr, var_sym)?;
    if radicals.is_empty() {
        return None;
    }
    let mut replaced: Vec<(ExprId, ExprId)> = Vec::new();
    let mut constants: Vec<(ExprId, ExprId)> = Vec::new();
    let mut double_roots: Vec<(ExprId, ExprId)> = Vec::new();
    let two = arena.int(2);
    for (node, base, e) in &radicals {
        if replaced.iter().any(|&(n, _)| n == *node) {
            continue;
        }
        // Monic linear factors of the base, with multiplicities.
        let mut factors: SmallVec<[(ExprId, i64); 2]> = SmallVec::new();
        let mut double: Option<ExprId> = None;
        if let Some((alpha, beta)) = symbolic_linear_coeff_of(arena, *base, var, var_sym) {
            let r = arena.div(beta, alpha);
            let r = arena.neg(r);
            let r = crate::transforms::eval::eval(arena, r);
            factors.push((arena.sub(var, r), 1));
        } else if let Some((a2, a1, a0)) = symbolic_quadratic_coeffs(arena, *base, var, var_sym) {
            let a1_sq = arena.pow(a1, two);
            let four = arena.int(4);
            let prod = arena.mul(&[four, a2, a0]);
            let disc = arena.sub(a1_sq, prod);
            let disc = crate::transforms::expand::expand(arena, disc);
            let disc = crate::transforms::eval::eval(arena, disc);
            let s = crate::calculus::risch::polynomial_sqrt(arena, disc)?;
            let two_a2 = arena.mul(&[two, a2]);
            let neg_a1 = arena.neg(a1);
            if arena.is_zero_structural(s) {
                let r = arena.div(neg_a1, two_a2);
                let r = crate::transforms::eval::eval(arena, r);
                factors.push((arena.sub(var, r), 2));
                double = Some(r);
            } else {
                for sign_s in [arena.neg(s), s] {
                    let num = arena.add(&[neg_a1, sign_s]);
                    let r = arena.div(num, two_a2);
                    let r = crate::transforms::eval::eval(arena, r);
                    factors.push((arena.sub(var, r), 1));
                }
            }
        } else {
            return None;
        }
        let e_id = arena.num_ratio(e.clone());
        let mut k_parts: SmallVec<[ExprId; 3]> = SmallVec::new();
        k_parts.push(arena.pow(*base, e_id));
        let mut new_parts: SmallVec<[ExprId; 3]> = SmallVec::new();
        for &(l, mult) in &factors {
            let em = e * Q::from_integer(mult.into());
            let em_id = arena.num_ratio(em.clone());
            new_parts.push(arena.pow(l, em_id));
            let neg_em = arena.num_ratio(-em);
            k_parts.push(arena.pow(l, neg_em));
        }
        let k = arena.mul(&k_parts);
        let k = crate::transforms::eval::eval(arena, k);
        if k != arena.one {
            let kappa = arena.symbol(&format!("__mk{}", constants.len()));
            constants.push((kappa, k));
            if let Some(r) = double {
                double_roots.push((kappa, r));
            }
            new_parts.push(kappa);
        } else if double.is_some() {
            return None;
        }
        replaced.push((*node, arena.mul(&new_parts)));
    }
    if constants.is_empty() {
        return None;
    }
    let rewritten = arena.subs_map_structural(expr, &replaced);
    let rewritten = crate::transforms::eval::eval(arena, rewritten);
    let mut g = mobius_core(arena, rewritten, var, var_sym, false)?;
    for &(kappa, r) in &double_roots {
        let g1 = crate::transforms::diff::diff(arena, g, kappa);
        let at = arena.subs_structural(g1, var, r);
        let at = crate::transforms::eval::eval(arena, at);
        if contains_var(arena, at, var_sym)
            || contains_non_finite(arena, at)
            || crate::base::walk::has_unevaluated(arena, at)
        {
            return None;
        }
        let shift = arena.mul(&[kappa, at]);
        g = arena.sub(g, shift);
    }
    for &(kappa, k) in &constants {
        g = arena.subs_structural(g, kappa, k);
    }
    Some(g)
}

/// `(f_w, w, n)` with `∫ f dx = ∫ f_w dw` at `w = xⁿ`, `n ≥ 2`, for
/// `f = xᵐ·h` where `x` occurs in `h` only in integer powers `xᵏ` with
/// `n | k` (`m` rational, the exponent of a top-level factor `xᵐ`, or 0):
/// `f_w = w^{(m+1−n)/n}·h(w)/n`.  `None` when `n < 2` or `h` has `x`
/// elsewhere.
fn power_substituted(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<(ExprId, ExprId, i64)> {
    use num_integer::Integer;
    let monomial_exponent = |arena: &Arena, f: ExprId| -> Option<Q> {
        if f == var {
            return Some(Q::one());
        }
        match *arena.node(f) {
            ExprNode::Pow(b, e) if b == var => arena.as_num(e).cloned(),
            _ => None,
        }
    };
    let (m, rest): (Q, SmallVec<[ExprId; 6]>) = match arena.node(expr) {
        ExprNode::Mul(children) => {
            let children = children.clone();
            let mut m = Q::zero();
            let mut rest = SmallVec::new();
            for c in children {
                match monomial_exponent(arena, c) {
                    Some(k) => m += k,
                    None => rest.push(c),
                }
            }
            (m, rest)
        }
        _ => (Q::zero(), SmallVec::from_slice(&[expr])),
    };
    let mut g = num_bigint::BigInt::zero();
    let mut found = false;
    for &r in &rest {
        for id in crate::base::walk::post_order_ids(arena, r) {
            let node = arena.node(id);
            if let ExprNode::Pow(b, e) = *node
                && b == var
            {
                let k = arena.as_num(e)?;
                if !k.is_integer() {
                    return None;
                }
                g = g.gcd(&k.to_integer());
                found = true;
                continue;
            }
            let mut bare = false;
            node.for_each_child(|c| bare |= c == var);
            if bare {
                return None;
            }
        }
    }
    let n = i64::try_from(&g).ok()?;
    if !found || !(2..=12).contains(&n) {
        return None;
    }
    let w = arena.symbol("__mw");
    let n_q = Q::from_integer(n.into());
    let mut pairs: Vec<(ExprId, ExprId)> = Vec::new();
    for &r in &rest {
        for id in crate::base::walk::post_order_ids(arena, r) {
            if let ExprNode::Pow(b, e) = *arena.node(id)
                && b == var
                && let Some(k) = arena.as_num(e).cloned()
                && !pairs.iter().any(|&(old, _)| old == id)
            {
                let k_w = arena.num_ratio(k / &n_q);
                let w_k = arena.pow(w, k_w);
                pairs.push((id, w_k));
            }
        }
    }
    let mut factors: SmallVec<[ExprId; 6]> = SmallVec::new();
    for &r in &rest {
        factors.push(arena.subs_map_structural(r, &pairs));
    }
    let e_w = arena.num_ratio((&m + Q::one() - &n_q) / &n_q);
    factors.push(arena.pow(w, e_w));
    factors.push(arena.rational(1, n));
    let f_w = arena.mul(&factors);
    if contains_var(arena, f_w, var_sym) {
        return None;
    }
    Some((f_w, w, n))
}

/// `e` in `w = xⁿ` written in `x`: `wʳ → x^{n·r}` (the radicals of
/// [`power_substituted`]'s `w` are powers of `x`), then `w → xⁿ`.
fn unmap_power(arena: &mut Arena, e: ExprId, w: ExprId, var: ExprId, n: i64) -> ExprId {
    let n_q = Q::from_integer(n.into());
    let mut pairs: Vec<(ExprId, ExprId)> = Vec::new();
    for id in crate::base::walk::post_order_ids(arena, e) {
        if let ExprNode::Pow(b, r) = *arena.node(id)
            && b == w
            && let Some(r) = arena.as_num(r).cloned()
            && !pairs.iter().any(|&(old, _)| old == id)
        {
            let nr = arena.num_ratio(r * &n_q);
            let x_nr = arena.pow(var, nr);
            pairs.push((id, x_nr));
        }
    }
    let e = arena.subs_map_structural(e, &pairs);
    let n_id = arena.int(n);
    let x_n = arena.pow(var, n_id);
    arena.subs_structural(e, w, x_n)
}

/// The substitution of [`try_mobius_radical_substitution`] in the variable
/// `var` (`x`, or `w = xⁿ`): the antiderivative in `var`, not yet checked.
/// `w_first`: in a pair, the radical of `var` itself goes into the
/// numerator of `t`.
fn mobius_core(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
    w_first: bool,
) -> Option<ExprId> {
    use num_integer::Integer;
    let radicals = radical_powers(arena, expr, var_sym)?;
    let mut bases: SmallVec<[ExprId; 2]> = SmallVec::new();
    let mut q_lcm = num_bigint::BigInt::one();
    for (_, b, e) in &radicals {
        if !bases.contains(b) {
            bases.push(*b);
        }
        q_lcm = q_lcm.lcm(e.denom());
    }
    if bases.is_empty() || bases.len() > 2 {
        return None;
    }
    let q = i64::try_from(&q_lcm).ok()?;
    if !(2..=MAX_MOBIUS_ROOT).contains(&q) {
        return None;
    }
    let t = arena.symbol("__mt");
    let ExprNode::Symbol(t_sym) = *arena.node(t) else {
        return None;
    };
    let q_ratio = Q::from_integer(q.into());
    let inv_q = arena.rational(1, q);
    let (ratio, sub, t_of_x) = if let [b] = bases[..] {
        let mut sub = expr;
        for (node, _, e) in &radicals {
            let k = arena.num_ratio(e * &q_ratio);
            let tk = arena.pow(t, k);
            sub = arena.subs_structural(sub, *node, tk);
        }
        (b, sub, arena.pow(b, inv_q))
    } else {
        let (b1, b2) = if w_first && bases[1] == var {
            (bases[1], bases[0])
        } else {
            (bases[0], bases[1])
        };
        let sub = pair_radicals(arena, expr, b1, b2, t, q)?;
        let neg_inv_q = arena.rational(-1, q);
        let r1 = arena.pow(b1, inv_q);
        let r2 = arena.pow(b2, neg_inv_q);
        (arena.div(b1, b2), sub, arena.mul(&[r1, r2]))
    };
    let Mobius { a, b, c, d } = mobius_parts(arena, ratio, var, var_sym)?;
    let ad = arena.mul(&[a, d]);
    let bc = arena.mul(&[b, c]);
    let det = arena.sub(ad, bc);
    let det = crate::transforms::eval::eval(arena, det);
    if arena.is_zero_structural(det) {
        return None;
    }
    let q_id = arena.int(q);
    let t_q = arena.pow(t, q_id);
    let d_tq = arena.mul(&[d, t_q]);
    let x_num = arena.sub(d_tq, b);
    let c_tq = arena.mul(&[c, t_q]);
    let x_den = arena.sub(a, c_tq);
    let x_of_t = arena.div(x_num, x_den);
    let qm1 = arena.int(q - 1);
    let t_qm1 = arena.pow(t, qm1);
    let two = arena.int(2);
    let den_sq = arena.pow(x_den, two);
    let dx_num = arena.mul(&[q_id, t_qm1, det]);
    let dx_dt = arena.div(dx_num, den_sq);
    let sub = arena.subs_structural(sub, var, x_of_t);
    if contains_var(arena, sub, var_sym) {
        return None;
    }
    let integrand = arena.mul(&[sub, dx_dt]);
    let integrand = rational_in_factored_form(arena, integrand, t);
    if !is_rational_in(arena, integrand, t_sym) {
        return None;
    }
    tracing::debug!(integrand_t = %arena.display(integrand), "integrate: Möbius radical substitution");
    let res = integrate_nested_generic(arena, integrand, t)?;
    let res = analytic_logs(arena, res, t_sym)?;
    let res = logs_off_the_cut_at(arena, res, t, t_of_x, var);
    Some(arena.subs_structural(res, t, t_of_x))
}

/// `g` with every `ln(p(t))` whose argument is a negative number at
/// `t₀ = t(0)` (the image of `x = 0`, when that is a number) written
/// `ln(−p(t))`: the same derivative, and continuous where the path `t(x)`
/// passes `t₀`.  `analytic_logs` writes the `ln|t − 1|` of `∫ dt/(1 − t²)`
/// as `ln(t − 1)`, whose argument at `t = 0` is on the cut: for
/// `∫ dx/√(x² + a)` (`t = x/√(x² + a)`, `t(0) = 0`) the answer jumped by
/// `iπ` at `x = 0` for every non-real `a` and for real `a < 0` (where the
/// integrand is imaginary on `|x| < √−a`); on the real path of a real
/// integrand only a constant changes.
fn logs_off_the_cut_at(
    arena: &mut Arena,
    g: ExprId,
    t: ExprId,
    t_of_x: ExprId,
    var: ExprId,
) -> ExprId {
    let zero = arena.zero;
    let t0 = arena.subs_structural(t_of_x, var, zero);
    let t0 = crate::transforms::eval::eval(arena, t0);
    if arena.as_num(t0).is_none() {
        return g;
    }
    let mut out = g;
    for id in crate::base::walk::post_order_ids(arena, g) {
        let ExprNode::Ln(p) = *arena.node(id) else {
            continue;
        };
        if !crate::base::walk::contains(arena, p, t) {
            continue;
        }
        let at = arena.subs_structural(p, t, t0);
        let at = crate::transforms::eval::eval(arena, at);
        if arena.as_num(at).is_some_and(|v| v.is_negative()) {
            let minus_p = arena.neg(p);
            let flipped = arena.ln(minus_p);
            out = arena.subs_structural(out, id, flipped);
        }
    }
    out
}

/// `∫ f` with every `e^{k·atanh(u)}` (`k` rational) written
/// `(1 + u)^{k/2}·(1 − u)^{−k/2}`: `atanh u = ½·(ln(1 + u) − ln(1 − u))`
/// (DLMF 4.37.24; the principal branches symplex and SymPy evaluate, on
/// the cuts `|u| > 1` too), so the two agree as complex numbers.  The
/// rewritten integrand is algebraic — rational for an even `k` — and goes
/// through the whole pipeline (Rubi 7.3.6, and 7.4.2 with `acoth u =
/// atanh(1/u)`; Rubi's rules rewrite `e^{n·atanh(a·x)}` the same way).
pub(super) fn try_exp_atanh_rewrite(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    let mut rewritten = expr;
    for id in crate::base::walk::post_order_ids(arena, expr) {
        let ExprNode::Exp(arg) = *arena.node(id) else {
            continue;
        };
        if !contains_var(arena, arg, var_sym) {
            continue;
        }
        let (k, u) = match arena.node(arg).clone() {
            ExprNode::Atanh(u) => (Q::one(), u),
            ExprNode::Mul(fs) if fs.len() == 2 => {
                let (k, u) = match (arena.as_num(fs[0]), arena.node(fs[1])) {
                    (Some(k), ExprNode::Atanh(u)) => (k.clone(), *u),
                    _ => continue,
                };
                (k, u)
            }
            _ => continue,
        };
        let half_k = arena.num_ratio(&k / Q::from_integer(2.into()));
        let neg_half_k = arena.num_ratio(-&k / Q::from_integer(2.into()));
        let one = arena.one;
        let one_plus = arena.add(&[one, u]);
        let one_minus = arena.sub(one, u);
        let p1 = arena.pow(one_plus, half_k);
        let p2 = arena.pow(one_minus, neg_half_k);
        let repl = arena.mul(&[p1, p2]);
        rewritten = arena.subs_structural(rewritten, id, repl);
    }
    if rewritten == expr {
        return None;
    }
    let rewritten = crate::transforms::eval::eval(arena, rewritten);
    let res = integrate_nested(arena, rewritten, var)?;
    if candidate_rejected(arena, expr, res, var, var_sym) {
        tracing::debug!("integrate: rejecting unverified e^(k·atanh u) closed form");
        return None;
    }
    Some(res)
}

/// Rewrite `sinh`/`cosh`/`tanh` in terms of `exp` and retry (e.g.
/// `1/cosh x = 2e^x/(e^{2x}+1)` → `2 atan(e^x)`).
pub(super) fn try_hyperbolic_to_exp(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    let order = crate::base::walk::post_order_ids(arena, expr);
    let has_hyp = order.iter().any(|&id| {
        matches!(
            arena.node(id),
            ExprNode::Sinh(_) | ExprNode::Cosh(_) | ExprNode::Tanh(_)
        ) && contains_var(arena, id, var_sym)
    });
    if !has_hyp {
        return None;
    }
    let rewritten = arena.rewrite_as_exp_expr(expr);
    if rewritten == expr {
        return None;
    }
    let rewritten = crate::transforms::eval::eval(arena, rewritten);
    try_exp_rational_substitution(arena, rewritten, var, var_sym).or_else(|| {
        let r = try_exp_linear_substitution(arena, rewritten, var, var_sym)?;
        (!candidate_rejected(arena, expr, r, var, var_sym)).then_some(r)
    })
}
