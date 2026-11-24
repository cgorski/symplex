//! |g|, sign(g), Heaviside(g), Piecewise (part of [`integrate`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// |g|, sign(g), Heaviside(g), Piecewise
// ═══════════════════════════════════════════════════════════════════════════

/// The real roots of `g` (as expressions) when all of them are known;
/// `None` otherwise.  [`try_abs_sign_product`] takes the sign of `g`
/// constant between the roots it is given, so a root missing here makes
/// an antiderivative that jumps there:
///
/// * `g` with a `sin` or `cos` of the variable is refused.  Such a `g` has
///   as a rule infinitely many real roots, and the solver returns the
///   principal branches only: `sin(eˣ) = 0` gave `x = ln π` alone, and
///   `∫ sign(sin eˣ) dx` was `(x − ln π)·sign(sin eˣ)` — its derivative is
///   the integrand, but it jumps at `x = ln(kπ)` for every `k ≥ 2`, so
///   `F(b) − F(a)` was wrong across any of them (also `cos(eˣ)`,
///   `eˣ·|sin(eˣ)|`, `sin(e⁻ˣ)`).
/// * For a polynomial over `ℚ` the roots found must be as many as its
///   distinct real roots (Sturm).
/// * A root that does not evaluate is not taken for a complex one: it is
///   refused unless it evaluates to a value off the real axis.
pub(super) fn real_roots(arena: &mut Arena, g: ExprId, var: ExprId) -> Option<Vec<ExprId>> {
    let var_sym = match arena.node(var) {
        ExprNode::Symbol(s) => *s,
        _ => return None,
    };
    let poly = crate::poly::polybridge::expr_to_poly(arena, g, var);
    if poly.is_none()
        && crate::base::walk::post_order_ids(arena, g)
            .into_iter()
            .any(|id| match arena.node(id) {
                ExprNode::Sin(a) | ExprNode::Cos(a) => contains_var(arena, *a, var_sym),
                _ => false,
            })
    {
        return None;
    }
    let sols = crate::transforms::solve::known_real_roots(arena, g, var);
    if sols.is_empty() && poly.is_none() {
        return None;
    }
    let mut roots = Vec::new();
    for s in sols {
        if !crate::base::walk::free_symbols(arena, s).is_empty() {
            return None; // parametric root: cannot decide
        }
        match crate::transforms::evalf::eval_const_f64(arena, s) {
            Some(v) if v.is_finite() => roots.push(s),
            Some(_) => return None,
            None => {
                // A complex root only when its imaginary part is certified
                // nonzero (an `f64` one may be the residue of a real root).
                let value = crate::transforms::eval::eval(arena, s);
                if !crate::transforms::realness::imaginary_part_sign(arena, value)
                    .is_nonzero()
                    .is_true()
                {
                    return None;
                }
            }
        }
    }
    if let Some(p) = &poly
        && p.degree().unwrap_or(0) > 0
        && crate::poly::sturm::SturmChain::new(p).count_real_roots() != roots.len()
    {
        return None;
    }
    Some(roots)
}

/// `∫ N(x)/(a·x + b) dx` for a polynomial `N` with rational coefficients and
/// a linear denominator whose `a`, `b` are constants that are not both
/// rational (`x + π`, `x + √2`, `x + c`, `x + sin 2`): with `x₀ = −b/a`,
/// synthetic division gives `N(x) = (x − x₀)·Q(x) + N(x₀)`, so the
/// antiderivative is `(1/a)·[∫Q + N(x₀)·ln|a·x + b|]`.  The rational
/// integrator handles all-rational denominators and is left alone for them;
/// 0.22 returned `∫ x/(x + π) dx` unevaluated (and searched for 12 s on
/// `∫ x·ln(x + sin 2) dx`, whose by-parts step needs it).
pub(super) fn try_poly_over_symbolic_linear(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    let (num, den) = crate::poly::polybridge::as_numer_denom(arena, expr);
    if !contains_var(arena, den, var_sym) {
        return None;
    }
    let (a, b) = symbolic_linear_coeff_of(arena, den, var, var_sym)?;
    if contains_var(arena, a, var_sym)
        || contains_var(arena, b, var_sym)
        || arena.is_zero_structural(a)
        || arena.as_num(a).is_some() && arena.as_num(b).is_some()
    {
        return None;
    }
    let p = crate::poly::polybridge::expr_to_poly(arena, num, var)?;
    let coeffs: Vec<ExprId> = p
        .coeffs()
        .iter()
        .map(|c| arena.num_ratio(c.clone()))
        .collect();
    let n = coeffs.len().checked_sub(1)?;
    // x₀ = −b/a; Horner from the top: q_{n−1} = c_n, q_{k−1} = c_k + x₀·q_k.
    let neg_b = arena.neg(b);
    let x0 = arena.div(neg_b, a);
    let mut q = vec![arena.zero; n];
    let mut carry = coeffs[n];
    for k in (1..=n).rev() {
        q[k - 1] = carry;
        let prod = arena.mul(&[x0, carry]);
        carry = arena.add(&[coeffs[k - 1], prod]);
    }
    let remainder = crate::transforms::eval::eval(arena, carry);
    // ∫Q = Σ q_k x^{k+1}/(k+1).
    let mut terms = Vec::with_capacity(n + 1);
    for (k, &qk) in q.iter().enumerate() {
        let e = arena.int(k as i64 + 1);
        let xp = arena.pow(var, e);
        let inv = arena.rational(1, k as i64 + 1);
        terms.push(arena.mul(&[inv, qk, xp]));
    }
    if !arena.is_zero_structural(remainder) {
        let abs_den = arena.abs(den);
        let log = arena.ln(abs_den);
        terms.push(arena.mul(&[remainder, log]));
    }
    let sum = arena.add(&terms);
    let inv_a = arena.pow(a, arena.neg_one());
    let result = arena.mul(&[inv_a, sum]);
    let result = crate::transforms::eval::eval(arena, result);
    (!candidate_rejected(arena, expr, result, var, var_sym)).then_some(result)
}

/// `∫ N(x)/(c·x² + d·x + e) dx` with `deg N ≥ 2` and constant coefficients
/// that are not all rational (`∫ 2x⁴/(sin(−1) − 2x²) dx`, the radical
/// substitution of `∫ x^(3/2)/(sin(−1) − 2x) dx`; `∫ x²·cos 4/(1 + cos²4·x²)`,
/// by parts from `∫ x·atan(x·cos 4) dx`).  Divides `N` by the quadratic over
/// the field of constant expressions, integrates the quotient termwise and
/// hands the linear remainder over the quadratic back to the integrator
/// (`try_linear_over_quadratic`).  The rational-coefficient case belongs
/// to the rational integrator and is left to it.  Checked by
/// differentiation.
pub(super) fn try_poly_over_symbolic_quadratic(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    /// Quotients of higher degree are left to the other routes.
    const MAX_NUMERATOR_DEGREE: usize = 16;
    let (num, den) = crate::poly::polybridge::as_numer_denom(arena, expr);
    if !contains_var(arena, den, var_sym) {
        return None;
    }
    if crate::poly::polybridge::expr_to_poly(arena, den, var).is_some()
        && crate::poly::polybridge::expr_to_poly(arena, num, var).is_some()
    {
        return None;
    }
    let dense = |arena: &mut Arena, e: ExprId| -> Option<Vec<ExprId>> {
        let terms = crate::poly::polybridge::symbolic_multipoly_terms(arena, e, &[var])?;
        let degree = terms.iter().map(|(exps, _)| exps[0] as usize).max()?;
        if degree > MAX_NUMERATOR_DEGREE {
            return None;
        }
        let mut coeffs = vec![arena.zero; degree + 1];
        for (exps, c) in terms {
            if contains_var(arena, c, var_sym) {
                return None;
            }
            coeffs[exps[0] as usize] = c;
        }
        Some(coeffs)
    };
    let d = dense(arena, den)?;
    if d.len() != 3 || arena.is_zero_structural(d[2]) {
        return None;
    }
    let mut rem = dense(arena, num)?;
    if rem.len() < 3 {
        return None;
    }
    // rem ← rem − q_k·x^k·(c x² + d x + e), from the top.
    let mut quotient = vec![arena.zero; rem.len() - 2];
    for k in (2..rem.len()).rev() {
        let qk = arena.div(rem[k], d[2]);
        let qk = crate::transforms::eval::eval(arena, qk);
        quotient[k - 2] = qk;
        for (j, &dj) in d.iter().enumerate().take(2) {
            let prod = arena.mul(&[qk, dj]);
            let diff = arena.sub(rem[k - 2 + j], prod);
            rem[k - 2 + j] = crate::transforms::eval::eval(arena, diff);
        }
        rem[k] = arena.zero;
    }
    let mut terms: Vec<ExprId> = Vec::with_capacity(quotient.len() + 1);
    for (j, &qj) in quotient.iter().enumerate() {
        let e = arena.int(i64::try_from(j).ok()? + 1);
        let xp = arena.pow(var, e);
        let inv = arena.rational(1, i64::try_from(j).ok()? + 1);
        terms.push(arena.mul(&[inv, qj, xp]));
    }
    let r1x = arena.mul(&[rem[1], var]);
    let linear = arena.add(&[r1x, rem[0]]);
    let linear = crate::transforms::eval::eval(arena, linear);
    if !arena.is_zero_structural(linear) {
        let inv_den = arena.pow(den, arena.neg_one());
        let piece = arena.mul(&[linear, inv_den]);
        let piece_integral = integrate_node(arena, piece, var, var_sym, depth.saturating_sub(1));
        if crate::base::walk::has_unevaluated(arena, piece_integral) {
            return None;
        }
        terms.push(piece_integral);
    }
    let result = arena.add(&terms);
    let result = crate::transforms::eval::eval(arena, result);
    (!candidate_rejected(arena, expr, result, var, var_sym)).then_some(result)
}

/// Does `e` contain `nan`, `zoo` or `±∞`?
pub(super) fn contains_non_finite(arena: &Arena, e: ExprId) -> bool {
    crate::base::walk::post_order_ids(arena, e)
        .into_iter()
        .any(|id| {
            matches!(
                arena.node(id),
                ExprNode::NaN
                    | ExprNode::ComplexInfinity
                    | ExprNode::Infinity
                    | ExprNode::NegInfinity
            )
        })
}

/// Does `e` apply a function that is not analytic (`|·|`, `sign`, `H`,
/// `re`, `im`, `conjugate`, `arg`, `min`, `max`, `Piecewise`) to something
/// that depends on the variable?  A `ln|g|` does not count: the stage exit
/// writes it `ln g` where `g` is not real ([`analytic_log_of_complex_arguments`]),
/// and `floor`/`ceiling` do not either (the continuity corrections
/// `J·⌊w/π + 1/2⌋` of trigonometric antiderivatives): they are constant
/// between their jumps, so their derivative is 0 along any path.
pub(super) fn non_analytic_in(arena: &Arena, e: ExprId, var_sym: SymbolId) -> bool {
    let ids = crate::base::walk::post_order_ids(arena, e);
    let in_log: FxHashSet<ExprId> = ids
        .iter()
        .filter_map(|&id| match arena.node(id) {
            ExprNode::Ln(a) if matches!(arena.node(*a), ExprNode::Abs(_)) => Some(*a),
            _ => None,
        })
        .collect();
    ids.into_iter().any(|id| {
        let non_analytic = match arena.node(id) {
            ExprNode::Abs(_) => !in_log.contains(&id),
            ExprNode::Sign(_)
            | ExprNode::Heaviside(_)
            | ExprNode::Re(_)
            | ExprNode::Im(_)
            | ExprNode::Conjugate(_)
            | ExprNode::Arg(_)
            | ExprNode::Min(_)
            | ExprNode::Max(_)
            | ExprNode::Piecewise(_) => true,
            _ => false,
        };
        non_analytic && contains_var(arena, id, var_sym)
    })
}

/// Is `e` real *and continuous* for every real value of the integration
/// variable, under the declared assumptions of the other symbols?  The
/// `|g|`/`sign(g)` route needs both: `g` may change sign only at its real
/// roots.  Conservative: sums, products, non-negative integer powers (a
/// negative power only of a variable-free base), and `sin`/`cos`/`exp`/
/// `atan`/`sinh`/`cosh`/`tanh`/`abs` of such; roots, logarithms, `tan`,
/// reciprocals of the variable and `sign` are refused.  (`∫ |π/(4 cos x)| dx`
/// came back as `(π/4)·ln|sec x + tan x|`, wrong wherever `cos x < 0` —
/// `c/cos x` has no roots but changes sign at its poles; found by
/// `fuzz_integrate`.)  A symbol other than the variable counts only when
/// it is declared real (decision D4, 0.31): up to 0.30 every symbol did,
/// and [`flatten_nested_pow`] rewrote `√((x + a)²)` to `|x + a|` for an
/// unassumed `a` — `∫ √((x + a)²) dx` came back as the unevaluated
/// `∫ |x + a| dx`, a different integral for a non-real `a`.
pub(super) fn real_on_reals(arena: &Arena, e: ExprId, var_sym: SymbolId) -> bool {
    crate::base::walk::post_order_ids(arena, e)
        .into_iter()
        .all(|id| match arena.node(id) {
            ExprNode::Symbol(s) => {
                *s == var_sym || crate::transforms::realness::symbol_declared_real(arena, *s)
            }
            ExprNode::Num(_)
            | ExprNode::Pi
            | ExprNode::E
            | ExprNode::EulerGamma
            | ExprNode::Catalan
            | ExprNode::GoldenRatio
            | ExprNode::Add(_)
            | ExprNode::Mul(_)
            | ExprNode::Sin(_)
            | ExprNode::Cos(_)
            | ExprNode::Exp(_)
            | ExprNode::Atan(_)
            | ExprNode::Sinh(_)
            | ExprNode::Cosh(_)
            | ExprNode::Tanh(_)
            | ExprNode::Abs(_) => true,
            ExprNode::Pow(base, exp) => arena.as_num(*exp).is_some_and(|r| {
                r.is_integer() && (!r.is_negative() || !contains_var(arena, *base, var_sym))
            }),
            _ => false,
        })
}

/// `∫ P(x)·|g(x)| dx`, `∫ P(x)·sign(g(x)) dx`, `∫ P(x)·H(g(x)) dx` where
/// `g` has at most one simple real root `r` (or none):
///
/// * one root:  `sign(g)·(G(x) − G(r))` with `G = ∫ P·g` (for `|g|`),
///   `sign(g)·(F(x) − F(r))` with `F = ∫ P` (for `sign`), and
///   `H(g)·(F(x) − F(r))` for `H` — each continuous at `r`;
/// * no root: `sign(g(x₀))` is constant and the factor is replaced by
///   `±g`, `±1` or `0/1` respectively.
pub(super) fn try_abs_sign_product(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    let idx = dependent.iter().position(|&d| {
        matches!(
            arena.node(d),
            ExprNode::Abs(_) | ExprNode::Sign(_) | ExprNode::Heaviside(_)
        ) && contains_var(arena, d, var_sym)
    })?;
    let node = arena.node(dependent[idx]).clone();
    let g = match node {
        ExprNode::Abs(g) | ExprNode::Sign(g) | ExprNode::Heaviside(g) => g,
        _ => return None,
    };
    // `|g| = sign(g)·g` and the root-shift below assume `g` is real at every
    // real `x`; `|√x|` is `√|x|`, not `sign(√x)·√x` (0.22.2 returned
    // `2/3·x^(3/2)·sign(√x)`, found by `fuzz_integrate`).
    if !real_on_reals(arena, g, var_sym) {
        return None;
    }
    let rest = remaining_product(arena, dependent, idx);
    let roots = real_roots(arena, g, var)?;
    if roots.len() > 1 {
        return None;
    }
    let root = roots.first().copied();
    // The "smooth" integrand whose antiderivative gets the sign factor.
    let smooth = match node {
        ExprNode::Abs(_) => arena.mul(&[rest, g]),
        _ => rest,
    };
    let smooth_int = integrate_node(arena, smooth, var, var_sym, depth - 1);
    if crate::base::walk::has_unevaluated(arena, smooth_int) {
        return None;
    }
    match root {
        Some(r) => {
            let at_r = crate::transforms::subs::subs(arena, smooth_int, var, r);
            let at_r = crate::transforms::eval::eval(arena, at_r);
            // The shift needs a finite G(r); an antiderivative singular at
            // the root (`ln|x|` at 0) made `nan` of the whole answer.
            if contains_non_finite(arena, at_r) {
                return None;
            }
            let shifted = arena.sub(smooth_int, at_r);
            let factor = match node {
                ExprNode::Heaviside(_) => arena.heaviside(g),
                _ => arena.sign(g),
            };
            Some(arena.mul(&[factor, shifted]))
        }
        None => {
            // Constant sign: sample g at a point.
            let sample = crate::transforms::subs::subs(arena, g, var, arena.zero);
            let sample = crate::transforms::eval::eval(arena, sample);
            let sgn = crate::transforms::evalf::eval_const_f64(arena, sample)?;
            // A nonzero printed digit is certified; 0 refuses.
            if sgn == 0.0 || !sgn.is_finite() {
                return None;
            }
            let positive = sgn > 0.0;
            match node {
                ExprNode::Abs(_) | ExprNode::Sign(_) => {
                    if positive {
                        Some(smooth_int)
                    } else {
                        Some(arena.neg(smooth_int))
                    }
                }
                _ => {
                    if positive {
                        Some(smooth_int)
                    } else {
                        Some(arena.zero)
                    }
                }
            }
        }
    }
}

/// Most real kinks [`try_piecewise_split`] splits the line at.
const MAX_SPLIT_POINTS: usize = 6;

/// The order of the real constants `r` and `w`: equal when they are the
/// same node, otherwise by the certified sign of `w − r`
/// ([`sign_checked`](crate::poly::algebraic::sign_checked), exact 0
/// included); `None` when that does not settle.
pub(super) fn cmp_roots(arena: &mut Arena, r: ExprId, w: ExprId) -> Option<std::cmp::Ordering> {
    if r == w {
        return Some(std::cmp::Ordering::Equal);
    }
    let d = arena.sub(w, r);
    let d = crate::transforms::eval::eval(arena, d);
    crate::poly::algebraic::sign_checked(arena, d).map(|s| 0.cmp(&s))
}

/// `∫ f dx` for an `f` holding `|g|`, `sign(g)`, `H(g)`, `min(g₁, g₂)`,
/// `max(g₁, g₂)` of functions real on the real line whose sign changes
/// only at finitely many numeric real points `r₁ < … < rₖ` (all of them
/// found, or one symbolic root of a linear `g` with a leading coefficient
/// of known sign): on each interval every such factor is `±g`, `±1`, `0/1`,
/// a branch, the plain integrand goes through the whole pipeline, and the
/// pieces are joined continuously, `Piecewise((F₀, x < r₁), (F₁ + k₁, x <
/// r₂), …)` with `kⱼ = kⱼ₋₁ + Fⱼ₋₁(rⱼ) − Fⱼ(rⱼ)`.  The product rule
/// ([`try_abs_sign_product`]) takes at most one root and a polynomial
/// factor: `∫ |x² − 1| dx`, `∫ e^{−|t|} dt`, `∫ e^{|x − 2|} dx`,
/// `∫ min(e^{−x}, eˣ)² dx`, `∫ H(3x − y)·sin(x + y) dx` stayed unevaluated up
/// to 0.46 (SymPy's test_integrals and test_manual answer them as such a
/// `Piecewise`).
pub(super) fn try_piecewise_split(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    // The kinked nodes and the function whose sign decides each.
    let mut kinks: Vec<(ExprId, ExprId)> = Vec::new();
    for id in crate::base::walk::post_order_ids(arena, expr) {
        let g = match arena.node(id) {
            ExprNode::Abs(g) | ExprNode::Sign(g) | ExprNode::Heaviside(g) => *g,
            ExprNode::Min(args) | ExprNode::Max(args) if args.len() == 2 => {
                let (a, b) = (args[0], args[1]);
                arena.sub(a, b)
            }
            _ => continue,
        };
        if !contains_var(arena, g, var_sym) {
            continue;
        }
        if !real_on_reals(arena, g, var_sym) {
            return None;
        }
        kinks.push((id, g));
    }
    if kinks.is_empty() || kinks.len() > 3 {
        return None;
    }
    // Breakpoints: numeric roots, sorted; or one symbolic root.
    // The numeric roots, distinct and in increasing order.
    let mut points: Vec<ExprId> = Vec::new();
    let mut symbolic: Option<(ExprId, ExprId)> = None;
    for &(_, g) in &kinks {
        match real_roots(arena, g, var) {
            Some(rs) => {
                // Placed by exact or certified comparison ([`cmp_roots`]):
                // up to 0.52 roots within `10⁻¹²` of each other were
                // merged and the rest sorted by `f64` values, and `∫ sign((x
                // − 1)(x − 1 − 10⁻¹⁵)) dx` lost the interval between them.
                for r in rs {
                    crate::transforms::evalf::eval_const_f64(arena, r)?;
                    let mut at = points.len();
                    let mut known = false;
                    for (k, &w) in points.iter().enumerate() {
                        match cmp_roots(arena, r, w)? {
                            std::cmp::Ordering::Equal => {
                                known = true;
                                break;
                            }
                            std::cmp::Ordering::Less => {
                                at = k;
                                break;
                            }
                            std::cmp::Ordering::Greater => {}
                        }
                    }
                    if !known {
                        points.insert(at, r);
                    }
                }
            }
            None => {
                let (a, b) = symbolic_linear_coeff_of(arena, g, var, var_sym)?;
                if symbolic.is_some() {
                    return None;
                }
                let r = arena.div(b, a);
                let r = arena.neg(r);
                symbolic = Some((crate::transforms::eval::eval(arena, r), a));
            }
        }
    }
    if symbolic.is_some() && (!points.is_empty() || kinks.len() > 1) {
        return None;
    }
    if points.len() > MAX_SPLIT_POINTS {
        return None;
    }
    let breaks: Vec<ExprId> = match symbolic {
        Some((r, _)) => vec![r],
        None => points,
    };
    if breaks.is_empty() {
        return None;
    }
    // The sign of each kink's function on each interval.
    let n = breaks.len();
    let mut pieces: Vec<ExprId> = Vec::with_capacity(n + 1);
    for j in 0..=n {
        let mut piece = expr;
        for &(node, g) in &kinks {
            let positive = match symbolic {
                Some((_, a)) => {
                    let s = crate::poly::algebraic::sign_checked(arena, a)?;
                    if s == 0 {
                        return None;
                    }
                    (s > 0) == (j == 1)
                }
                None => {
                    let one = arena.one;
                    let sample = if j == 0 {
                        arena.sub(breaks[0], one)
                    } else if j == n {
                        arena.add(&[breaks[n - 1], one])
                    } else {
                        let half = arena.rational(1, 2);
                        let s = arena.add(&[breaks[j - 1], breaks[j]]);
                        arena.mul(&[half, s])
                    };
                    let at = crate::transforms::subs::subs(arena, g, var, sample);
                    let at = crate::transforms::eval::eval(arena, at);
                    let v = crate::transforms::evalf::eval_const_f64(arena, at)?;
                    // A nonzero printed digit is certified; 0 refuses.
                    if v == 0.0 || !v.is_finite() {
                        return None;
                    }
                    v > 0.0
                }
            };
            let rep = match arena.node(node).clone() {
                ExprNode::Abs(_) => {
                    if positive {
                        g
                    } else {
                        arena.neg(g)
                    }
                }
                ExprNode::Sign(_) => arena.int(if positive { 1 } else { -1 }),
                ExprNode::Heaviside(_) => arena.int(i64::from(positive)),
                ExprNode::Min(args) => args[usize::from(positive)],
                ExprNode::Max(args) => args[usize::from(!positive)],
                _ => return None,
            };
            piece = arena.subs_structural(piece, node, rep);
        }
        pieces.push(crate::transforms::eval::eval(arena, piece));
    }
    let mut antis: Vec<ExprId> = Vec::with_capacity(n + 1);
    for &p in &pieces {
        antis.push(integrate_nested(arena, p, var)?);
    }
    let mut out: Vec<(ExprId, ExprId)> = Vec::with_capacity(n + 1);
    let mut k = arena.zero;
    for j in 0..=n {
        if j > 0 {
            let r = breaks[j - 1];
            let prev = crate::transforms::subs::subs(arena, antis[j - 1], var, r);
            let cur = crate::transforms::subs::subs(arena, antis[j], var, r);
            let diff = arena.sub(prev, cur);
            k = arena.add(&[k, diff]);
            k = crate::transforms::eval::eval(arena, k);
            if contains_non_finite(arena, k) {
                return None;
            }
        }
        let branch = arena.add(&[antis[j], k]);
        let cond = if j < n {
            arena.gt(breaks[j], var)
        } else {
            arena.bool_true
        };
        out.push((branch, cond));
    }
    tracing::debug!("integrate: split at the kinks, joined continuously");
    Some(arena.piecewise(&out))
}

/// `∫ Piecewise((fᵢ, cᵢ)) dx = Piecewise((Fᵢ + kᵢ, cᵢ))`.
///
/// When the conditions form a chain `x < c₁, x < c₂, …, otherwise` (any
/// of `<`, `≤`) with increasing constants, the constants `kᵢ` are chosen so
/// that the antiderivative is continuous at every breakpoint.  For other
/// condition shapes the branch antiderivatives are returned without
/// matching constants (still a valid antiderivative on each branch).
pub(super) fn integrate_piecewise(
    arena: &mut Arena,
    pairs: &[(ExprId, ExprId)],
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    let mut antis: Vec<ExprId> = Vec::with_capacity(pairs.len());
    for &(val, _) in pairs {
        let f = integrate_node(arena, val, var, var_sym, depth - 1);
        if crate::base::walk::has_unevaluated(arena, f) {
            return None;
        }
        antis.push(f);
    }
    // Chain detection: condition i (< last) is `x < c_i` or `x ≤ c_i`,
    // i.e. Gt(c_i, x) / Ge(c_i, x) with c_i free of var.
    let mut breakpoints: Vec<ExprId> = Vec::new();
    let mut chain = true;
    for &(_, cond) in &pairs[..pairs.len().saturating_sub(1)] {
        match arena.node(cond).clone() {
            ExprNode::Gt(c, v) | ExprNode::Ge(c, v)
                if v == var && !contains_var(arena, c, var_sym) =>
            {
                breakpoints.push(c);
            }
            _ => {
                chain = false;
                break;
            }
        }
    }
    let mut out: Vec<(ExprId, ExprId)> = Vec::with_capacity(pairs.len());
    if chain && pairs.len() >= 2 {
        let mut k = arena.zero;
        out.push((antis[0], pairs[0].1));
        for i in 1..pairs.len() {
            let c = breakpoints[i - 1];
            // k_i = k_{i-1} + F_{i-1}(c) − F_i(c)
            let prev_at_c = crate::transforms::subs::subs(arena, antis[i - 1], var, c);
            let cur_at_c = crate::transforms::subs::subs(arena, antis[i], var, c);
            let diff = arena.sub(prev_at_c, cur_at_c);
            k = arena.add(&[k, diff]);
            k = crate::transforms::eval::eval(arena, k);
            let branch = arena.add(&[antis[i], k]);
            out.push((branch, pairs[i].1));
        }
    } else {
        for (i, &(_, cond)) in pairs.iter().enumerate() {
            out.push((antis[i], cond));
        }
    }
    Some(arena.piecewise(&out))
}

// ═══════════════════════════════════════════════════════════════════════════
// Piecewise parametric wrapping
// ═══════════════════════════════════════════════════════════════════════════

/// Attempt to wrap the integration result in a `Piecewise` for parametric
/// degenerate cases.
///
/// When the result contains denominators involving free symbols (parameters
/// other than the integration variable), we check if setting those parameters
/// to specific values would cause division by zero.  For each such degenerate
/// value, we substitute back into the original integrand, re-integrate the
/// simplified form, and build a `Piecewise` node with explicit conditions.
pub(super) fn try_piecewise_wrap(
    arena: &mut Arena,
    result: ExprId,
    original_integrand: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> ExprId {
    // If result is an unevaluated Integral, nothing to wrap.
    if matches!(arena.node(result), ExprNode::Integral(_, _)) {
        return result;
    }
    // The integrand of a rationalising substitution: the degenerate cases
    // are those of the integrand in `x`, which the outer call wraps.
    if GENERIC_ONLY.with(std::cell::Cell::get) {
        return result;
    }
    // An integrand free of the variable has the antiderivative `c·x`: where
    // a denominator of `c` vanishes the integrand itself is undefined, and
    // there is no degenerate case to integrate.  In 0.44 the branch `b = 0`
    // of `∫ (cosh⁴(a + bx) − sinh⁴(a + bx))/(cosh⁴(a + bx) + sinh⁴(a + bx))`
    // was wrapped in 8 nested cases `a ≠ −ln(RootOf(t⁸ + 6t⁴ + 1, k))` (Rubi
    // 6.7.1 #1052: 6 ms in 0.43, 0.2 s in 0.44).
    if !contains_var(arena, original_integrand, var_sym) {
        return result;
    }

    // A result that is already such a case analysis — from the wrap of a
    // nested `integrate`, as a substitution re-enters the pipeline — keeps
    // it: its degenerate values count as handled, and only its generic
    // branch is searched for denominators.  `∫ sin(a·ln x) dx` came out with
    // the cases `a = i` and `a = −i` nested twice each (0.31).
    let mut generic = result;
    let mut handled: Vec<(ExprId, ExprId)> = Vec::new();
    while let ExprNode::Piecewise(pairs) = arena.node(generic).clone()
        && pairs.len() == 2
        && pairs[1].1 == arena.bool_true
        && let ExprNode::Ne(p, v) = *arena.node(pairs[0].1)
    {
        handled.push((p, v));
        handled.push((v, p));
        generic = pairs[0].0;
    }

    // Collect denominator expressions from the result.
    let denoms = collect_denominators(arena, generic);
    if denoms.is_empty() {
        return result;
    }

    let mut wrapped = result;
    // The handled pairs with the denominator they came from.
    let mut handled_by_denom: Vec<(ExprId, ExprId, ExprId)> = Vec::new();
    // The `(parameter, value)` cases whose re-integration found no closed
    // form.  The same case met again through another denominator (`a = b`
    // from `a − b` and from `a² − b²`) is the same integrand in the same
    // context and fails again (on the Rubi suite's 4.5.4.2, 3,852 of 8,948
    // failed cases were such repeats; 0.38).
    let mut failed: Vec<(ExprId, ExprId)> = Vec::new();

    for denom in &denoms {
        let denom_syms = crate::base::walk::free_symbols(arena, *denom);
        for sym_expr in &denom_syms {
            // Skip the integration variable.
            if let ExprNode::Symbol(sid) = arena.node(*sym_expr)
                && *sid == var_sym
            {
                continue;
            }
            // Only parameters of the integrand can be degenerate.  A symbol
            // that occurs in the result alone (the bound variable of a
            // `RootSum`, a substitution symbol) would leave the integrand
            // unchanged and re-integrate the same problem forever.
            if !crate::base::walk::contains(arena, original_integrand, *sym_expr) {
                continue;
            }

            // Solve denom = 0 for this parameter symbol.
            let solutions = degenerate_values(arena, *denom, *sym_expr);
            for &degen_val in &solutions {
                // A degenerate value is a constant.  A denominator in both
                // the parameter and the variable (`(a + 1)·x + a`, from a
                // partial answer) has the "solution" `a = −x/(x + 1)`, and
                // substituting it made a zero denominator in disguise that
                // panicked the rational integrator (found by the
                // complex-parameter hunter, 0.31).
                if contains_var(arena, degen_val, var_sym) || contains_non_finite(arena, degen_val)
                {
                    continue;
                }
                // Nor a value in a symbol the integrand does not have: the
                // bound variable of a `RootSum` in a denominator of its body.
                if crate::base::walk::free_symbols(arena, degen_val)
                    .iter()
                    .any(|&s| !crate::base::walk::contains(arena, original_integrand, s))
                {
                    continue;
                }

                // Avoid duplicate wrapping for the same (param, value) pair.
                if handled
                    .iter()
                    .any(|&(p, v)| p == *sym_expr && v == degen_val)
                    || failed.contains(&(*sym_expr, degen_val))
                {
                    continue;
                }
                // Nor for a value whose points are already covered: `b = a`
                // after `a = b` (the denominator `a − b` solved for either
                // symbol), `b = a²/4` after `a = ±2√b`.  Each such wrap nested
                // an unreachable copy of the whole case analysis: the
                // parametric rational integrator's `∫ dx/((x + a)(x + b)(x + c))`
                // came out six levels deep (0.31).
                if degenerate_value_covered(arena, &handled_by_denom, *denom, *sym_expr, degen_val)
                {
                    continue;
                }
                // Nor for a value that an enclosing degenerate case rules
                // out: inside the branch `c = b²/(4a)` the parameter `a` is
                // not 0.  Its integrand at `a = 0` came out as `0` (the
                // `1/(b²x⁴/(4·0) + …)²` of the substituted denominator is
                // `1/zoo²`), and `∫ x²(d + e·x² + f·x⁴ + g·x⁶)/(a + b·x² +
                // c·x⁴)² dx` carried an unreachable `Piecewise(…, (0, True))`
                // (0.31).
                if excluded_by_enclosing_case(arena, *sym_expr, degen_val) {
                    continue;
                }
                // Nor for a non-real constant value of a parameter declared
                // real: `a = ±i√2` in `∫ dx/(a² + 1 + cos x)` (an unreachable
                // branch, and one `piecewise_simplify` cannot drop: `i√2 ≠ −3`
                // after substituting `a = −3` stayed undecided).
                if nonreal_value_of_real_symbol(arena, *sym_expr, degen_val) {
                    continue;
                }
                // Nor for a value the other assumptions of the parameter
                // exclude: `a = 0` and `b = −a` for positive `a`, `b` in
                // `∫ dx/(a + b·sin²x)` (0.42 carried both unreachable
                // branches).  An undecided value keeps its branch.
                if Truth::from(crate::transforms::solve::assumptions_hold(
                    arena, *sym_expr, degen_val,
                ))
                .is_false()
                {
                    continue;
                }

                // Filter: skip if substituting this value makes the original
                // integrand singular (these are poles of the problem, not
                // artifacts of the antiderivative formula).
                let integrand_at_degen =
                    crate::transforms::subs::subs(arena, original_integrand, *sym_expr, degen_val);
                let integrand_at_degen = crate::transforms::eval::eval(arena, integrand_at_degen);
                if has_zero_denominator(arena, integrand_at_degen) {
                    continue;
                }

                // Re-integrate the simplified integrand at the degenerate value.
                tracing::debug!(
                    integrand = %arena.display(integrand_at_degen),
                    "try_piecewise_wrap: integrating a degenerate case"
                );
                let nonzero = NonzeroInCase::enter(arena, collect_denominators(arena, degen_val));
                // A value that real parameters cannot take (`b = 3i/n`, the
                // resonance of `∫ x²·sin(a + b·ln(c·xⁿ)) dx`) gets a small
                // step budget: on the Rubi suite (0.31) such cases were 2,932
                // failed re-integrations taking 34 s (SymPy leaves them
                // unevaluated too) and 223 successes of at most 18 steps.
                let _cap = (!real_at_generic_values(arena, degen_val))
                    .then(|| StepCap::enter(NONREAL_CASE_STEPS));
                let degen_result = stage!(
                    arena,
                    "degenerate_case",
                    integrand_at_degen,
                    integrate(arena, integrand_at_degen, var)
                );
                drop(nonzero);
                let degen_result = crate::transforms::eval::eval(arena, degen_result);

                // Skip if re-integration returned unevaluated, also in part:
                // `x²/2 + ∫ 1/(−x/(x + 1) + x·(−x/(x + 1) + 1)) dx` (an integrand
                // undefined at the degenerate value) made the whole answer
                // unevaluated (0.31, the parametric-rational hunt).
                // Nor a branch that is not finite: the generic formula at
                // the degenerate value (`(Pₙ₊₁ − Pₙ₋₁)/(2n + 1)` at `n = −½`)
                // came back as `zoo`, which is no antiderivative.
                if crate::base::walk::has_unevaluated(arena, degen_result)
                    || contains_non_finite(arena, degen_result)
                {
                    failed.push((*sym_expr, degen_val));
                    continue;
                }

                // Build Piecewise: [(generic, Ne(param, degen)), (degen_result, True)]
                let condition = arena.ne_(*sym_expr, degen_val);
                let true_cond = arena.bool_true;
                wrapped = arena.piecewise(&[(wrapped, condition), (degen_result, true_cond)]);

                handled.push((*sym_expr, degen_val));
                handled_by_denom.push((*denom, *sym_expr, degen_val));
            }
        }
    }

    wrapped
}

/// The values of the parameter `sym` at which `denom` vanishes: the roots
/// of each irreducible factor (over the other parameters) of its numerator
/// as a fraction, or of `denom` itself when that numerator does not split.
/// A factor linear in `sym` gives its root without radicals: `1 − k²/m²`
/// vanishes at `m = k` and `m = −k`, where `solve` on the whole gives
/// `m = ±√(k²)`, which is `±|k|` for a real `k`; the branch `m = |k|` of
/// `∫ sin(kx)·sin(mx) dx` then integrated `sin(kx)·sin(|k|x)` by product to
/// sum into a quotient by `k − |k|`, `0/0` for every `k > 0` (SymPy's
/// test_integrals, `k`, `m` integers; 0.46 returned `0` for the definite
/// integral over `[0, π]` at `m = k`, which is `π/2`).
fn degenerate_values(arena: &mut Arena, denom: ExprId, sym: ExprId) -> Vec<ExprId> {
    let (num, _) = crate::poly::polybridge::fraction_parts(arena, denom);
    let num = crate::transforms::expand::expand(arena, num);
    let factored = arena.factor_expr(num, sym);
    let factors: Vec<ExprId> = match arena.node(factored) {
        ExprNode::Mul(children) => children
            .iter()
            .map(|&c| match *arena.node(c) {
                ExprNode::Pow(b, e) if arena.as_num(e).is_some_and(|r| r.is_integer()) => b,
                _ => c,
            })
            .collect(),
        _ => Vec::new(),
    };
    let split: Vec<ExprId> = factors
        .into_iter()
        .filter(|&f| crate::base::walk::contains(arena, f, sym))
        .collect();
    if split.len() < 2 {
        return crate::transforms::solve::known_real_roots(arena, denom, sym);
    }
    let mut values: Vec<ExprId> = Vec::new();
    for f in split {
        for s in crate::transforms::solve::known_real_roots(arena, f, sym) {
            if !values.contains(&s) {
                values.push(s);
            }
        }
    }
    values
}

/// The step budget ([`StepCap`]) of a degenerate case whose value is not
/// real at real parameter values.
const NONREAL_CASE_STEPS: u64 = 20;

/// Is the parameter value `val` real for some real values of its own
/// parameters: the generic values of [`FTC_PARAMETER_VALUES`] (in name
/// order) with every combination of signs (`sqrt(−a/b)` is real for
/// `a/b < 0`; `3i/n`, `(−1/2 + i√3/2)·∛(−a/b)` are real for none)?
/// `true` when it does not evaluate, or has more than three parameters.
fn real_at_generic_values(arena: &mut Arena, val: ExprId) -> bool {
    let mut params: Vec<(String, ExprId)> = Vec::new();
    for s in crate::base::walk::free_symbols(arena, val) {
        let ExprNode::Symbol(sid) = *arena.node(s) else {
            return true;
        };
        params.push((arena.symbol_name(sid).to_owned(), s));
    }
    if params.len() > 3 {
        return true;
    }
    params.sort_by(|a, b| a.0.cmp(&b.0));
    for signs in 0u32..(1 << params.len()) {
        let mut bound = val;
        for (k, &(_, s)) in params.iter().enumerate() {
            let (p, q) = FTC_PARAMETER_VALUES[k];
            let p = if signs & (1 << k) == 0 { p } else { -p };
            let value = arena.rational(p, q);
            bound = crate::transforms::subs::subs(arena, bound, s, value);
        }
        let bound = crate::transforms::eval::eval(arena, bound);
        // A guide (the step budget of the degenerate case), right either
        // way: an imaginary part `evalf` keeps is certified nonzero, and one
        // negligible next to the real part (`e⁻⁵⁰·i` next to 1) reads as real
        // here, which only leaves the case its full budget.
        match crate::transforms::evalf::evalf_complex(arena, bound, 15) {
            Ok(v) if !crate::transforms::evalf::is_real_to_digits(&v, 12) => {}
            _ => return true,
        }
    }
    false
}

thread_local! {
    /// The denominators of the degenerate values of the enclosing
    /// [`try_piecewise_wrap`] cases on this thread (`a` for `c = b²/(4a)`):
    /// non-zero wherever the case's answer applies.
    static NONZERO_IN_CASE: std::cell::RefCell<Vec<(u64, ExprId)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// The denominators of a degenerate value pushed onto [`NONZERO_IN_CASE`]
/// for the re-integration of its case; popped on drop, also when a panic
/// unwinds.
struct NonzeroInCase(usize);

impl NonzeroInCase {
    fn enter(arena: &Arena, denominators: Vec<ExprId>) -> Self {
        let key = arena.serial();
        NONZERO_IN_CASE.with(|s| {
            let mut s = s.borrow_mut();
            let len = s.len();
            s.extend(denominators.into_iter().map(|d| (key, d)));
            NonzeroInCase(len)
        })
    }
}

impl Drop for NonzeroInCase {
    fn drop(&mut self) {
        let _ = NONZERO_IN_CASE.try_with(|s| {
            if let Ok(mut s) = s.try_borrow_mut() {
                s.truncate(self.0);
            }
        });
    }
}

/// Does `sym = val` make one of the enclosing cases' non-zero
/// denominators ([`NONZERO_IN_CASE`]) vanish?
fn excluded_by_enclosing_case(arena: &mut Arena, sym: ExprId, val: ExprId) -> bool {
    let key = arena.serial();
    let nonzero: Vec<ExprId> = NONZERO_IN_CASE.with(|s| {
        s.borrow()
            .iter()
            .filter(|&&(k, _)| k == key)
            .map(|&(_, d)| d)
            .collect()
    });
    nonzero.into_iter().any(|d| {
        crate::base::walk::contains(arena, d, sym) && {
            let at = crate::transforms::subs::subs(arena, d, sym, val);
            let at = crate::transforms::expand::expand(arena, at);
            let at = crate::transforms::eval::eval(arena, at);
            arena.is_zero_structural(at)
        }
    })
}

/// Is `sym` declared real and `val` a constant that is not real?
pub(super) fn nonreal_value_of_real_symbol(arena: &mut Arena, sym: ExprId, val: ExprId) -> bool {
    let ExprNode::Symbol(sid) = *arena.node(sym) else {
        return false;
    };
    crate::transforms::realness::symbol_declared_real(arena, sid)
        && crate::base::walk::free_symbols(arena, val).is_empty()
        // Certified beyond `10⁻⁹` of a 16-digit value; a smaller imaginary
        // part keeps the case.
        && crate::transforms::evalf::evalf_complex64(arena, val)
            .is_ok_and(|z| z.im.is_finite() && z.im.abs() > 1e-9 * z.norm().max(1.0))
}

/// Is every point with `sym = val` already a handled degenerate case of
/// [`try_piecewise_wrap`]: does `p − v` vanish there for one handled pair
/// `(p, v)`, or the product of `p − v` over the pairs handled for the same
/// denominator `denom` (its solutions for one symbol cover its zero set)?
fn degenerate_value_covered(
    arena: &mut Arena,
    handled: &[(ExprId, ExprId, ExprId)],
    denom: ExprId,
    sym: ExprId,
    val: ExprId,
) -> bool {
    let vanishes_at = |arena: &mut Arena, e: ExprId| {
        let at = crate::transforms::subs::subs(arena, e, sym, val);
        let at = crate::transforms::expand::expand(arena, at);
        let at = crate::transforms::eval::eval(arena, at);
        arena.is_zero_structural(at)
    };
    let mut same_denom: Vec<ExprId> = Vec::new();
    for &(d, p, v) in handled {
        let diff = arena.sub(p, v);
        if vanishes_at(arena, diff) {
            return true;
        }
        if d == denom {
            same_denom.push(diff);
        }
    }
    if same_denom.len() >= 2 {
        let product = arena.mul(&same_denom);
        return vanishes_at(arena, product);
    }
    false
}

/// Collect all denominator sub-expressions from an expression tree.
///
/// A "denominator" is the base of any `Pow(base, exp)` node where `exp`
/// is a negative rational number.
fn collect_denominators(arena: &Arena, expr: ExprId) -> Vec<ExprId> {
    let mut denoms = Vec::new();
    let mut stack: Vec<ExprId> = vec![expr];
    let mut visited: FxHashSet<ExprId> = FxHashSet::default();

    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        if let ExprNode::Pow(base, exp) = arena.node(id).clone()
            && let Some(r) = arena.as_num(exp)
            && r.is_negative()
        {
            denoms.push(base);
        }
        arena.node(id).for_each_child(|c| stack.push(c));
    }

    denoms
}

/// Check if an expression contains a sub-expression that evaluates to
/// division by zero (a denominator that is structurally zero, or NaN /
/// ComplexInfinity atoms).
pub(super) fn has_zero_denominator(arena: &Arena, expr: ExprId) -> bool {
    let mut stack: Vec<ExprId> = vec![expr];
    let mut visited: FxHashSet<ExprId> = FxHashSet::default();

    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        if matches!(arena.node(id), ExprNode::NaN | ExprNode::ComplexInfinity) {
            return true;
        }
        if let ExprNode::Pow(base, exp) = arena.node(id).clone()
            && let Some(r) = arena.as_num(exp)
            && r.is_negative()
            && arena.is_zero_structural(base)
        {
            return true;
        }
        arena.node(id).for_each_child(|c| stack.push(c));
    }

    false
}
