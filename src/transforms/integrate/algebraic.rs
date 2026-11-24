//! Algebraic reductions: P(x)·Q(x)^{k/2} (part of [`integrate`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Algebraic reductions: P(x)·Q(x)^{k/2}
// ═══════════════════════════════════════════════════════════════════════════

/// Split the dependent factors into a rational-coefficient polynomial `P`
/// and a single factor `Q^{k/2}` (`k` odd, `Q` of degree 1 or 2).
fn split_poly_and_half_power(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
) -> Option<(crate::poly::Poly, crate::poly::Poly, ExprId, Q)> {
    let mut half: Option<(ExprId, Q)> = None;
    let mut poly = crate::poly::Poly::from_int(1);
    for &d in dependent {
        if let ExprNode::Pow(base, e) = arena.node(d).clone()
            && let Some(r) = arena.as_num(e).cloned()
            && *r.denom() == num_bigint::BigInt::from(2)
        {
            if half.is_some() {
                return None;
            }
            half = Some((base, r));
            continue;
        }
        let p = crate::poly::polybridge::expr_to_poly(arena, d, var)?;
        poly = poly.mul(&p);
    }
    let (q_expr, k) = half?;
    let q = crate::poly::polybridge::expr_to_poly(arena, q_expr, var)?;
    let qd = q.degree()?;
    if !(1..=2).contains(&qd) {
        return None;
    }
    Some((poly, q, q_expr, k))
}

/// `∫ P(x)·Q(x)^{k/2} dx` for `k ∈ {−1, 1, 3}`: write `P·Q^{(k+1)/2} = P̃`, then
/// find a polynomial `R` and constant `c` with
/// `P̃ = (2R'Q + RQ')/2 + c`, so that the integral is `R√Q + c∫ dx/√Q`.
pub(super) fn try_poly_times_half_power(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    let (p, q, q_expr, k) = split_poly_and_half_power(arena, dependent, var)?;
    let two = Q::from_integer(2.into());
    let k2 = &k * &two; // odd integer as rational
    let k2: i64 = k2.to_integer().to_string().parse().ok()?;
    if !matches!(k2, -1 | 1 | 3) {
        return None;
    }
    // P̃ = P · Q^{(k+1)/2}
    let mut p_tilde = p;
    for _ in 0..((k2 + 1) / 2) {
        p_tilde = p_tilde.mul(&q);
    }
    let pd = p_tilde.degree()?;
    let qd = q.degree()?;
    if pd == 0 && qd == 2 {
        return None; // plain 1/√Q: handled by the standard forms
    }
    // Unknowns: R = r_0 + … + r_m x^m (m = pd − qd + 1, or pd for linear Q), c.
    let m: i64 = if qd == 2 { pd as i64 - 1 } else { pd as i64 };
    let with_c = qd == 2;
    // The dense system has n² entries and costs n³ steps: refused beyond
    // 2¹⁰ unknowns (up to 0.57 its rows were sized by the degree of `P̃`).
    let n_unknowns = crate::base::cast::bounded_len(m + 1, 1usize << 10)? + usize::from(with_c);
    let n_eq = pd + 1;
    if n_unknowns != n_eq {
        return None;
    }
    // Build the linear system: coefficient of x^j in (2R'Q + RQ')/2 + c.
    let q_prime = q.derivative();
    let mut rows: Vec<Vec<Q>> = vec![vec![Q::zero(); n_unknowns + 1]; n_eq];
    for i in 0..=(m.max(-1)) {
        if i < 0 {
            break;
        }
        let iu = crate::base::cast::bounded_len(i, n_unknowns)?;
        // basis monomial x^i for R
        let mut mono = vec![Q::zero(); iu + 1];
        mono[iu] = Q::one();
        let r_i = crate::poly::Poly::from_coeffs(mono);
        let term = r_i
            .derivative()
            .mul(&q)
            .scale(&two)
            .add(&r_i.mul(&q_prime))
            .scale(&Q::new(1.into(), 2.into()));
        for (j, row) in rows.iter_mut().enumerate() {
            row[iu] = term.coeff(j);
        }
    }
    if with_c {
        rows[0][n_unknowns - 1] = Q::one();
    }
    for (j, row) in rows.iter_mut().enumerate() {
        row[n_unknowns] = p_tilde.coeff(j);
    }
    let sol = solve_linear_system(rows, n_unknowns)?;
    // Assemble R√Q + c∫1/√Q.
    let r_coeffs: Vec<Q> = sol[..(m.max(-1) + 1) as usize].to_vec();
    let r_poly = crate::poly::Poly::from_coeffs(r_coeffs);
    let r_expr = crate::poly::polybridge::poly_to_expr(arena, &r_poly, var);
    let half = arena.rational(1, 2);
    let sqrt_q = arena.pow(q_expr, half);
    let mut result = arena.mul(&[r_expr, sqrt_q]);
    if with_c {
        let c = sol[n_unknowns - 1].clone();
        if !c.is_zero() {
            let neg_half = arena.rational(-1, 2);
            let inv_sqrt = arena.pow(q_expr, neg_half);
            let base_int = integrate_node(arena, inv_sqrt, var, var_sym, depth - 1);
            if crate::base::walk::has_unevaluated(arena, base_int) {
                return None;
            }
            let c_id = arena.num_ratio(c.clone());
            let c_term = arena.mul(&[c_id, base_int]);
            result = arena.add(&[result, c_term]);
        }
    }
    Some(result)
}

/// Largest degree of `P` in [`try_symbolic_quadratic_radical`].
const MAX_RADICAL_POLY_DEGREE: usize = 6;

/// `∫ P(x)·Q^{±1/2} dx` for `Q = A·x² + B·x + C` whose coefficients are not
/// all rational (parameters, `√2`) and a polynomial `P` over them, by the
/// reduction `Iₙ = ∫ xⁿ/√Q = (xⁿ⁻¹·√Q − (n − ½)·B·Iₙ₋₁ − (n − 1)·C·Iₙ₋₂)/(n·A)`
/// (the derivative of `xⁿ⁻¹·√Q`, as in Gradshteyn–Ryzhik 2.260) down to
/// `I₀ = ln(2√A·√Q + 2A·x + B)/√A`, an antiderivative for every complex
/// `A ≠ 0`, `B`, `C` (its derivative is `√A·(2A·x + B + 2√A·√Q)/(√Q·(…))`
/// whatever the branches), and `P·√Q = P·Q/√Q`.  `A = 0` is a degenerate
/// case of the wrap.  Up to 0.46 `∫ dx/√(a + b·x + c·x²)`, `∫ √(a + b·x +
/// c·x²) dx` and `∫ x·asin(a·x) dx` (by parts) stayed unevaluated (SymPy's
/// test_integrals and test_manual; SymPy gives the same logarithm).
pub(super) fn try_symbolic_quadratic_radical(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    if depth < 2 {
        return None;
    }
    let mut radical: Option<(ExprId, bool)> = None;
    let mut rest: SmallVec<[ExprId; 4]> = SmallVec::new();
    for &d in dependent {
        if let ExprNode::Pow(b, e) = *arena.node(d)
            && contains_var(arena, b, var_sym)
            && let Some(r) = arena.as_num(e)
            && *r.denom() == num_bigint::BigInt::from(2)
            && (*r.numer() == num_bigint::BigInt::from(1)
                || *r.numer() == num_bigint::BigInt::from(-1))
        {
            if radical.is_some() {
                return None;
            }
            radical = Some((b, r.is_positive()));
        } else {
            rest.push(d);
        }
    }
    let (q, positive) = radical?;
    if crate::poly::polybridge::expr_to_poly(arena, q, var).is_some() {
        return None;
    }
    let (a2, a1, a0) = symbolic_quadratic_coeffs(arena, q, var, var_sym)?;
    if arena.is_zero_structural(a2) {
        return None;
    }
    // A perfect square (`√((a + x)²)`): the logarithm degenerates (its
    // argument vanishes where `a + x` is negative and real).
    let disc = {
        let b2 = arena.mul(&[a1, a1]);
        let four = arena.int(4);
        let ac = arena.mul(&[four, a2, a0]);
        let d = arena.sub(b2, ac);
        let d = crate::transforms::expand::expand(arena, d);
        crate::transforms::eval::eval(arena, d)
    };
    if arena.is_zero_structural(disc) {
        return None;
    }
    let p = if rest.is_empty() {
        arena.one
    } else {
        arena.mul(&rest)
    };
    let mut coeffs = poly_coeffs_in(arena, p, var, var_sym, MAX_RADICAL_POLY_DEGREE)?;
    if positive {
        // P·Q as the numerator over √Q.
        let mut pq = vec![arena.zero; coeffs.len() + 2];
        for (i, &c) in coeffs.iter().enumerate() {
            for (j, &qc) in [a0, a1, a2].iter().enumerate() {
                let t = arena.mul(&[c, qc]);
                pq[i + j] = arena.add(&[pq[i + j], t]);
            }
        }
        coeffs = pq;
    }
    let half = arena.rational(1, 2);
    let sqrt_q = arena.pow(q, half);
    let sqrt_a = arena.pow(a2, half);
    let two = arena.int(2);
    let ln_arg = {
        let t1 = arena.mul(&[two, sqrt_a, sqrt_q]);
        let t2 = arena.mul(&[two, a2, var]);
        arena.add(&[t1, t2, a1])
    };
    let ln = arena.ln(ln_arg);
    let i0 = arena.div(ln, sqrt_a);
    let mut ints: Vec<ExprId> = vec![i0];
    for n in 1..coeffs.len() {
        let n_i = i64::try_from(n).ok()?;
        let n_minus_1 = arena.int(n_i - 1);
        let x_pow = arena.pow(var, n_minus_1);
        let lead = arena.mul(&[x_pow, sqrt_q]);
        let k1 = arena.rational(2 * n_i - 1, 2);
        let t1 = arena.mul(&[k1, a1, ints[n - 1]]);
        let mut num = arena.sub(lead, t1);
        if n >= 2 {
            let k2 = arena.int(n_i - 1);
            let t2 = arena.mul(&[k2, a0, ints[n - 2]]);
            num = arena.sub(num, t2);
        }
        let n_e = arena.int(n_i);
        let den = arena.mul(&[n_e, a2]);
        ints.push(arena.div(num, den));
    }
    let mut terms: Vec<ExprId> = Vec::with_capacity(coeffs.len());
    for (&c, &i) in coeffs.iter().zip(&ints) {
        if !arena.is_zero_structural(c) {
            terms.push(arena.mul(&[c, i]));
        }
    }
    tracing::debug!("integrate: P(x)·Q^(±1/2) for a symbolic quadratic Q");
    let sum = arena.add(&terms);
    Some(crate::transforms::eval::eval(arena, sum))
}

/// Gaussian elimination over ℚ on an augmented matrix (`n` unknowns).
fn solve_linear_system(mut rows: Vec<Vec<Q>>, n: usize) -> Option<Vec<Q>> {
    let m = rows.len();
    let mut pivot_row = 0;
    let mut pivot_cols: Vec<usize> = Vec::new();
    for col in 0..n {
        let Some(p) = (pivot_row..m).find(|&r| !rows[r][col].is_zero()) else {
            continue;
        };
        rows.swap(pivot_row, p);
        let inv = Q::one() / rows[pivot_row][col].clone();
        for v in rows[pivot_row].iter_mut() {
            *v = &*v * &inv;
        }
        let pivot = rows[pivot_row].clone();
        for (r, row) in rows.iter_mut().enumerate() {
            if r != pivot_row && !row[col].is_zero() {
                let f = row[col].clone();
                for (c, cell) in row.iter_mut().enumerate() {
                    let sub = &pivot[c] * &f;
                    *cell = &*cell - &sub;
                }
            }
        }
        pivot_cols.push(col);
        pivot_row += 1;
        if pivot_row == m {
            break;
        }
    }
    // Inconsistent rows?
    for row in rows.iter().skip(pivot_row) {
        if !row[n].is_zero() {
            return None;
        }
    }
    let mut sol = vec![Q::zero(); n];
    for (r, &col) in pivot_cols.iter().enumerate() {
        sol[col] = rows[r][n].clone();
    }
    Some(sol)
}

/// `∫ x^{−n} Q(x)^{−1/2} dx` (`n ≥ 1`, `Q` quadratic) via `x = 1/t`:
/// `= −∫ t^{n−2} (c t² + b t + a)^{−1/2} dt` for `Q = a x² + b x + c`.
pub(super) fn try_reciprocal_sqrt_substitution(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    if dependent.len() != 2 {
        return None;
    }
    let mut n_neg: Option<i64> = None;
    let mut half: Option<(ExprId, Q)> = None;
    for &d in dependent {
        if let ExprNode::Pow(base, e) = arena.node(d).clone()
            && let Some(r) = arena.as_num(e).cloned()
        {
            if base == var && r.is_integer() && r.is_negative() {
                n_neg = Some(-r.to_integer().to_string().parse::<i64>().ok()?);
                continue;
            }
            if *r.denom() == num_bigint::BigInt::from(2) && contains_var(arena, base, var_sym) {
                half = Some((base, r));
                continue;
            }
        }
        return None;
    }
    let n = n_neg?;
    let (q_expr, k) = half?;
    if k != Q::new((-1).into(), 2.into()) || !(1..=4).contains(&n) {
        return None;
    }
    let q = crate::poly::polybridge::expr_to_poly(arena, q_expr, var)?;
    if q.degree()? != 2 {
        return None;
    }
    // Q(1/t) = (a + b t + c t²)/t²  ⇒  Q^{-1/2} = |t| (c t² + b t + a)^{-1/2};
    // x^{-n} = t^n, dx = −dt/t²  ⇒  integrand −|t|·t^{n−2}·(…)^{-1/2}
    //   = −sign(x)·t^{n−1}·(c t² + b t + a)^{-1/2}.
    // sign(x) is locally constant, so F(x) = sign(x)·G(1/x) with
    // G(t) = ∫ −t^{n−1} (c t² + b t + a)^{-1/2} dt.
    let t = arena.symbol("__rt");
    let t_sym = match arena.node(t) {
        ExprNode::Symbol(s) => *s,
        _ => return None,
    };
    let coeffs = q.coeffs().to_vec(); // [c, b, a]
    let reversed = crate::poly::Poly::from_coeffs(coeffs.iter().rev().cloned().collect());
    let q_rev = crate::poly::polybridge::poly_to_expr(arena, &reversed, t);
    let neg_half = arena.rational(-1, 2);
    let q_rev_pow = arena.pow(q_rev, neg_half);
    let t_pow_id = arena.int(n - 1);
    let t_pow = arena.pow(t, t_pow_id);
    let integrand_t = arena.mul(&[t_pow, q_rev_pow]);
    let neg_integrand = arena.neg(integrand_t);
    let res_t = integrate_node(arena, neg_integrand, t, t_sym, depth - 1);
    if crate::base::walk::has_unevaluated(arena, res_t) {
        return None;
    }
    let inv_x = arena.pow(var, arena.neg_one);
    let g_of_x = arena.subs_structural(res_t, t, inv_x);
    let sgn = arena.sign(var);
    let result = arena.mul(&[sgn, g_of_x]);
    Some(crate::transforms::eval::eval(arena, result))
}

/// `tanᵐ(g)·sec²(g)` → `tanᵐ⁺¹(g)/((m+1)·g')` and `secⁿ(g)·tan(g)` →
/// `secⁿ(g)/(n·g')` for linear `g`.
pub(super) fn try_tan_sec_patterns(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    if dependent.len() != 2 {
        return None;
    }
    // Identify tan^m(g) and cos^{-n}(g).
    let mut tan_part: Option<(ExprId, Q)> = None;
    let mut sec_part: Option<(ExprId, Q)> = None;
    for &d in dependent {
        let (base, e) = if let ExprNode::Pow(b, e) = arena.node(d).clone() {
            (b, arena.as_num(e)?.clone())
        } else {
            (d, Q::one())
        };
        match arena.node(base).clone() {
            ExprNode::Tan(g) if e.is_integer() && e.is_positive() => tan_part = Some((g, e)),
            ExprNode::Cos(g) if e.is_integer() && e.is_negative() => sec_part = Some((g, -e)),
            _ => return None,
        }
    }
    let (g, m) = tan_part?;
    let (g2, n) = sec_part?;
    if g != g2 {
        return None;
    }
    let (a_expr, _) = symbolic_linear_coeff_of(arena, g, var, var_sym)?;
    let two = Q::from_integer(2.into());
    if n == two {
        // ∫ tan^m sec² = tan^{m+1}/(m+1)
        let m1 = &m + &Q::one();
        let m1_id = arena.num_ratio(m1.clone());
        let tan_g = arena.tan(g);
        let tp = arena.pow(tan_g, m1_id);
        let denom = arena.mul(&[m1_id, a_expr]);
        return Some(arena.div(tp, denom));
    }
    if m == Q::one() {
        // ∫ secⁿ tan = secⁿ/n = cos^{-n}/n
        let cos_g = arena.cos(g);
        let neg_n = arena.num_ratio(-n.clone());
        let sec_n = arena.pow(cos_g, neg_n);
        let n_id = arena.num_ratio(n.clone());
        let denom = arena.mul(&[n_id, a_expr]);
        return Some(arena.div(sec_n, denom));
    }
    None
}
