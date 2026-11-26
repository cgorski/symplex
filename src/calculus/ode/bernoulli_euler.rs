//! Bernoulli equations: y' + P(x)·y = Q(x)·y^n  (n ≠ 0, 1) (part of [`ode`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Bernoulli equations: y' + P(x)·y = Q(x)·y^n  (n ≠ 0, 1)
// ═══════════════════════════════════════════════════════════════════════════

/// Solve Bernoulli equations: y' + P(x)·y = Q(x)·y^n (n ≠ 0, 1).
///
/// Substitution v = y^(1−n) transforms to a first-order linear ODE:
///   v' + (1−n)·P(x)·v = (1−n)·Q(x)
/// Solve for v, then recover y = v^(1/(1−n)).
pub(super) fn try_bernoulli(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
    var_sym: SymbolId,
) -> Option<OdeResult> {
    tracing::debug!("ode: trying Bernoulli");

    let dy_dx = arena.intern(ExprNode::Derivative(func, var));
    let d2y_dx2 = arena.intern(ExprNode::Derivative(dy_dx, var));

    // Must be first-order (no second derivatives)
    if expr_contains(arena, expr, d2y_dx2) {
        return None;
    }

    let children = match arena.node(expr).clone() {
        ExprNode::Add(children) => children,
        _ => return None,
    };

    use num_traits::Zero;

    let mut dy_coeff = num_rational::Ratio::<num_bigint::BigInt>::zero();
    let mut p_x_terms: Vec<ExprId> = Vec::new();
    let mut q_x_terms: Vec<(ExprId, Q)> = Vec::new();

    for &child in &children {
        let (coeff, term) = arena.as_coeff_term(child);
        if term == dy_dx {
            dy_coeff += coeff;
        } else if !contains_sym(arena, child, func_sym) {
            return None; // Free terms not allowed in standard Bernoulli
        } else if let Some(px) = extract_coeff_of_func(arena, child, func, func_sym, var_sym) {
            p_x_terms.push(px);
        } else if let Some((qx, n)) = extract_bernoulli_term(arena, child, func, func_sym, var_sym)
        {
            q_x_terms.push((qx, n));
        } else {
            return None;
        }
    }

    if dy_coeff.is_zero() || q_x_terms.is_empty() {
        return None;
    }

    // All y^n terms must share the same exponent n ≠ 0, 1
    let n_val = q_x_terms[0].1.clone();
    if n_val.is_zero() || n_val.is_one() {
        return None;
    }
    for (_, n) in &q_x_terms[1..] {
        if *n != n_val {
            return None;
        }
    }

    // Normalize by dy_coeff
    let inv_dy = num_rational::Ratio::<num_bigint::BigInt>::one() / &dy_coeff;
    let inv_dy_id = ode_ratio_to_expr(arena, &inv_dy);

    let p_raw = if p_x_terms.is_empty() {
        arena.zero
    } else if p_x_terms.len() == 1 {
        p_x_terms[0]
    } else {
        arena.add(&p_x_terms)
    };
    let p_x = if dy_coeff.is_one() {
        p_raw
    } else {
        let s = arena.mul(&[inv_dy_id, p_raw]);
        crate::transforms::eval::eval(arena, s)
    };

    // Q_raw·y^n appears on the LHS: y' + P·y + Q_raw·y^n = 0
    // So actual Q in y' + P·y = Q·y^n is −Q_raw
    let q_sum: Vec<ExprId> = q_x_terms.iter().map(|(qx, _)| *qx).collect();
    let q_raw = if q_sum.len() == 1 {
        q_sum[0]
    } else {
        arena.add(&q_sum)
    };
    let neg_q_raw = arena.neg(q_raw);
    let q_x = if dy_coeff.is_one() {
        neg_q_raw
    } else {
        let s = arena.mul(&[inv_dy_id, neg_q_raw]);
        crate::transforms::eval::eval(arena, s)
    };

    if contains_sym(arena, p_x, func_sym) || contains_sym(arena, q_x, func_sym) {
        return None;
    }

    // Substitution: v = y^(1−n)
    // Transformed ODE: v' + (1−n)·P·v = (1−n)·Q
    let one_minus_n = num_rational::Ratio::<num_bigint::BigInt>::one() - &n_val;
    let one_minus_n_id = ode_ratio_to_expr(arena, &one_minus_n);

    let new_p = arena.mul(&[one_minus_n_id, p_x]);
    let new_p = crate::transforms::eval::eval(arena, new_p);
    let new_q = arena.mul(&[one_minus_n_id, q_x]);
    let new_q = crate::transforms::eval::eval(arena, new_q);

    // Build linear ODE for v: v' + new_p·v − new_q = 0
    let v = arena.internal_symbol("__v");
    let dv = arena.intern(ExprNode::Derivative(v, var));
    let pv = arena.mul(&[new_p, v]);
    let neg_new_q = arena.neg(new_q);
    let linear_expr = arena.add(&[dv, pv, neg_new_q]);

    let v_sym = match arena.node(v) {
        ExprNode::Symbol(sid) => *sid,
        _ => return None,
    };

    // Solve the linear ODE for v (fall back to simple separable if P=0)
    let v_result = try_first_order_linear_general(arena, linear_expr, v, var, v_sym, var_sym)
        .or_else(|| try_simple_separable(arena, linear_expr, v, var, v_sym, var_sym))?;

    // Recover y = v^(1/(1−n)); for an even denominator (`y⁻² = v`) the
    // negative root `−v^(1/(1−n))` is a family too (before 0.30 it was
    // missing: `y′ = y³` with `y(0) = −1` had no solution).
    let inv_one_minus_n = num_rational::Ratio::<num_bigint::BigInt>::one() / &one_minus_n;
    let inv_id = ode_ratio_to_expr(arena, &inv_one_minus_n);
    let solution = arena.pow(v_result.solution, inv_id);
    let solution = crate::transforms::eval::eval(arena, solution);
    let two = num_bigint::BigInt::from(2);
    let branches = if inv_one_minus_n.denom() % &two == num_bigint::BigInt::from(0) {
        let negative = arena.neg(solution);
        vec![crate::transforms::eval::eval(arena, negative)]
    } else if !inv_one_minus_n.is_integer() && inv_one_minus_n.numer().abs().is_one() {
        // `1 − n` an odd integer `m` (`y′ = y/x + y⁴/x²`, `y⁻³ = v`): the
        // principal `v^{1/m}` is real only for `v > 0`, and `−(−v)^{1/m}`
        // (`(−(−v)^{1/m})^m = v`) is the real solution where `v < 0`.
        // Before, initial values `y(x₀) < 0` had no real family.
        let neg_v = arena.neg(v_result.solution);
        let root = arena.pow(neg_v, inv_id);
        let negative = arena.neg(root);
        vec![crate::transforms::eval::eval(arena, negative)]
    } else {
        Vec::new()
    };

    Some(OdeResult {
        solution,
        constants: v_result.constants,
        branches,
    })
}

/// Try to extract a Bernoulli term Q(x)·y^n from an expression.
///
/// Returns `Some((Q(x), n))` where Q(x) is free of y and n is a rational
/// constant.
pub(super) fn extract_bernoulli_term(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    func_sym: SymbolId,
    _var_sym: SymbolId,
) -> Option<(ExprId, Q)> {
    // Case 1: expr is y^n
    if let ExprNode::Pow(base, exp) = arena.node(expr).clone()
        && base == func
    {
        let n = arena.as_num(exp)?.clone();
        return Some((arena.one, n));
    }

    // Case 2: expr is Mul([..., y^n, ...])
    if let ExprNode::Mul(ref factors) = arena.node(expr).clone() {
        let mut yn_idx = None;
        let mut yn_exp = None;
        for (i, &f) in factors.iter().enumerate() {
            if let ExprNode::Pow(base, exp) = arena.node(f).clone()
                && base == func
                && let Some(n) = arena.as_num(exp)
            {
                yn_idx = Some(i);
                yn_exp = Some(n.clone());
                break;
            }
        }
        if let (Some(idx), Some(n)) = (yn_idx, yn_exp) {
            let mut other: Vec<ExprId> = Vec::new();
            for (i, &f) in factors.iter().enumerate() {
                if i != idx {
                    if contains_sym(arena, f, func_sym) {
                        return None;
                    }
                    other.push(f);
                }
            }
            let qx = match other.len() {
                0 => arena.one,
                1 => other[0],
                _ => arena.mul(&other),
            };
            return Some((qx, n));
        }
    }

    // Case 3: numeric coefficient × something
    {
        let (coeff, term) = arena.as_coeff_term(expr);
        if !coeff.is_one()
            && term != expr
            && let Some((inner_qx, n)) =
                extract_bernoulli_term(arena, term, func, func_sym, _var_sym)
        {
            let coeff_id = ode_ratio_to_expr(arena, &coeff);
            let qx = arena.mul(&[coeff_id, inner_qx]);
            return Some((qx, n));
        }
    }

    // Case 4: Neg(something)
    if let ExprNode::Neg(inner) = arena.node(expr).clone()
        && let Some((inner_qx, n)) = extract_bernoulli_term(arena, inner, func, func_sym, _var_sym)
    {
        let neg_qx = arena.neg(inner_qx);
        return Some((neg_qx, n));
    }

    None
}

// ═══════════════════════════════════════════════════════════════════════════
// Euler-Cauchy equations: a·x²·y'' + b·x·y' + c·y = 0
// ═══════════════════════════════════════════════════════════════════════════

/// Solve Euler-Cauchy equations: a·x²·y'' + b·x·y' + c·y = 0.
///
/// The characteristic equation is a·r(r−1) + b·r + c = 0, equivalently
/// a·r² + (b−a)·r + c = 0.
///
/// - Distinct real roots r₁, r₂: y = C1·x^r₁ + C2·x^r₂
/// - Repeated root r: y = (C1 + C2·ln(x))·x^r
/// - Complex roots α ± βi: y = x^α·(C1·cos(β·ln(x)) + C2·sin(β·ln(x)))
///
/// A forcing term `g(x)` is handled by `x = eᵗ` (for `x > 0`, like the
/// homogeneous solutions): `a·Y″ + (b − a)·Y′ + c·Y = g(eᵗ)` has constant
/// coefficients, and its solution with `t = ln x` is the answer.  Before,
/// every forced Cauchy–Euler equation was refused.
pub(super) fn try_euler_cauchy(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
    var_sym: SymbolId,
) -> Option<OdeResult> {
    tracing::debug!("ode: trying Euler-Cauchy");

    let dy_dx = arena.intern(ExprNode::Derivative(func, var));
    let d2y_dx2 = arena.intern(ExprNode::Derivative(dy_dx, var));

    if !expr_contains(arena, expr, d2y_dx2) {
        return None;
    }

    let children = match arena.node(expr).clone() {
        ExprNode::Add(children) => children,
        _ => return None,
    };

    use num_traits::Zero;
    let mut a_coeff = num_rational::Ratio::<num_bigint::BigInt>::zero();
    let mut b_coeff = num_rational::Ratio::<num_bigint::BigInt>::zero();
    let mut c_coeff = num_rational::Ratio::<num_bigint::BigInt>::zero();

    let two_id = arena.int(2);
    let x_sq = arena.pow(var, two_id);
    let mut forcing: Vec<ExprId> = Vec::new();

    for &child in &children {
        let (coeff, term) = arena.as_coeff_term(child);
        if term == func {
            c_coeff += coeff;
        } else if !contains_sym(arena, child, func_sym) {
            forcing.push(child);
        } else if let ExprNode::Mul(ref factors) = arena.node(term).clone() {
            let has_d2 = factors.contains(&d2y_dx2);
            let has_d1 = factors.contains(&dy_dx);
            let has_xsq = factors.contains(&x_sq);
            let has_xvar = factors.contains(&var);

            // Collect factors that are not x²/x/y''/y'
            let other: Vec<ExprId> = factors
                .iter()
                .copied()
                .filter(|&f| f != d2y_dx2 && f != dy_dx && f != x_sq && f != var)
                .collect();
            for &of in &other {
                if contains_sym(arena, of, func_sym) || contains_sym(arena, of, var_sym) {
                    return None;
                }
            }
            let extra = if other.is_empty() {
                num_rational::Ratio::<num_bigint::BigInt>::from_integer(1.into())
            } else {
                let e = if other.len() == 1 {
                    other[0]
                } else {
                    arena.mul(&other)
                };
                arena.as_num(e)?.clone()
            };

            if has_d2 && has_xsq && !has_d1 && !has_xvar {
                a_coeff += &coeff * &extra;
            } else if has_d1 && has_xvar && !has_d2 && !has_xsq {
                b_coeff += &coeff * &extra;
            } else {
                return None;
            }
        } else {
            return None;
        }
    }

    if a_coeff.is_zero() {
        return None;
    }

    if !forcing.is_empty() {
        return euler_cauchy_forced(arena, &a_coeff, &b_coeff, &c_coeff, &forcing, var, var_sym);
    }

    // Characteristic equation: a·r² + (b−a)·r + c = 0
    // Normalize: r² + ((b−a)/a)·r + c/a = 0
    let p = (&b_coeff - &a_coeff) / &a_coeff;
    let q = &c_coeff / &a_coeff;

    let four = num_rational::Ratio::<num_bigint::BigInt>::from_integer(4.into());
    let disc = &p * &p - &four * &q;

    let c1 = arena.symbol("C1");
    let c2 = arena.symbol("C2");

    if disc.is_positive() {
        // Distinct real roots — solve r² + p·r + q = 0
        let r_var = arena.internal_symbol("__r");
        let r_sq = arena.pow(r_var, two_id);
        let p_id = ode_ratio_to_expr(arena, &p);
        let q_id = ode_ratio_to_expr(arena, &q);
        let p_r = arena.mul(&[p_id, r_var]);
        let char_eq = arena.add(&[r_sq, p_r, q_id]);
        let roots = known_roots(arena, char_eq, r_var);

        if roots.len() >= 2 {
            let r1 = roots[0].value;
            let r2 = roots[1].value;
            let x_r1 = arena.pow(var, r1);
            let x_r2 = arena.pow(var, r2);
            let t1 = arena.mul(&[c1, x_r1]);
            let t2 = arena.mul(&[c2, x_r2]);
            let solution = arena.add(&[t1, t2]);
            Some(OdeResult {
                solution,
                constants: vec![c1, c2],
                branches: Vec::new(),
            })
        } else {
            None
        }
    } else if disc.is_zero() {
        // Repeated root: r = −p/2
        let two_r = num_rational::Ratio::<num_bigint::BigInt>::from_integer(2.into());
        let r = -(&p) / &two_r;
        let r_id = ode_ratio_to_expr(arena, &r);
        let x_r = arena.pow(var, r_id);
        let ln_x = arena.ln(var);
        let c2_ln = arena.mul(&[c2, ln_x]);
        let inner = arena.add(&[c1, c2_ln]);
        let solution = arena.mul(&[inner, x_r]);
        Some(OdeResult {
            solution,
            constants: vec![c1, c2],
            branches: Vec::new(),
        })
    } else {
        // Complex roots α ± βi: α = −p/2, β = √(−disc)/2
        let two_r = num_rational::Ratio::<num_bigint::BigInt>::from_integer(2.into());
        let alpha = -(&p) / &two_r;
        let neg_disc = -disc;
        let neg_disc_id = ode_ratio_to_expr(arena, &neg_disc);
        let half = arena.rational(1, 2);
        let sqrt_neg_disc = arena.pow(neg_disc_id, half);
        let two_expr = arena.int(2);
        let beta = arena.div(sqrt_neg_disc, two_expr);
        let beta = crate::transforms::eval::eval(arena, beta);

        let ln_x = arena.ln(var);
        let beta_ln_x = arena.mul(&[beta, ln_x]);
        let cos_part = arena.cos(beta_ln_x);
        let sin_part = arena.sin(beta_ln_x);
        let c1_cos = arena.mul(&[c1, cos_part]);
        let c2_sin = arena.mul(&[c2, sin_part]);
        let trig_part = arena.add(&[c1_cos, c2_sin]);

        let solution = if alpha.is_zero() {
            trig_part
        } else {
            let alpha_id = ode_ratio_to_expr(arena, &alpha);
            let x_alpha = arena.pow(var, alpha_id);
            arena.mul(&[x_alpha, trig_part])
        };

        Some(OdeResult {
            solution,
            constants: vec![c1, c2],
            branches: Vec::new(),
        })
    }
}

/// `a·x²·y″ + b·x·y′ + c·y + Σ forcing = 0` by `x = eᵗ` (see
/// [`try_euler_cauchy`]).
fn euler_cauchy_forced(
    arena: &mut Arena,
    a: &Q,
    b: &Q,
    c: &Q,
    forcing: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
) -> Option<OdeResult> {
    euler_cauchy_by_exp(arena, &[c.clone(), b - a, a.clone()], forcing, var, var_sym)
}

/// Cauchy–Euler of any order ≥ 3, `Σₖ aₖ·xᵏ·y⁽ᵏ⁾ + g(x) = 0` with rational
/// `aₖ`, by `x = eᵗ`: `xᵏ·y⁽ᵏ⁾ = D(D − 1)⋯(D − k + 1)·Y` with `D = d/dt`,
/// a constant-coefficient equation in `t` (solutions for `x > 0`, as for
/// order 2).  SymPy's `nth_linear_euler_eq_*`; order 3 was refused.
pub(super) fn try_euler_cauchy_n(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
    var_sym: SymbolId,
) -> Option<OdeResult> {
    let euler = euler_operator(arena, expr, func, var, func_sym, 3)?;
    euler_cauchy_by_exp(arena, &euler.coeffs, &euler.forcing, var, var_sym)
}

/// A Cauchy–Euler equation `Σₖ aₖ·xᵏ·y⁽ᵏ⁾ + g(x) = 0` after `x = eᵗ`:
/// `Σⱼ coeffsⱼ·Y⁽ʲ⁾(t) + g = 0`.
struct EulerOperator {
    /// The coefficients of `Dʲ` (`D = d/dt`) of `Σ aₖ·D(D − 1)⋯(D − k + 1)`.
    coeffs: Vec<Q>,
    /// The forcing terms (in `x`).
    forcing: Vec<ExprId>,
}

/// The signed Stirling numbers of the first kind `s(k, j)`, `j ≤ k`: the
/// coefficients of `D(D − 1)⋯(D − k + 1) = Σⱼ s(k, j)·Dʲ`.
fn falling_operator(k: usize) -> Vec<Q> {
    let mut falling = vec![Q::from_integer(1.into())];
    for i in 0..k {
        let shift = Q::from_integer((i as i64).into());
        let mut next = vec![Q::from_integer(0.into()); falling.len() + 1];
        for (j, v) in falling.iter().enumerate() {
            next[j + 1] += v;
            next[j] -= v * &shift;
        }
        falling = next;
    }
    falling
}

/// Recognise a Cauchy–Euler equation of order `≥ min_order` with rational
/// `aₖ` (see [`EulerOperator`]).
fn euler_operator(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
    min_order: usize,
) -> Option<EulerOperator> {
    let chain = derivative_chain(arena, func, var);
    let order = (1..chain.len())
        .rev()
        .find(|&k| expr_contains(arena, expr, chain[k]))?;
    if order < min_order {
        return None;
    }
    let ExprNode::Add(children) = arena.node(expr).clone() else {
        return None;
    };
    let mut a = vec![Q::from_integer(0.into()); order + 1];
    let mut forcing = Vec::new();
    for &child in children.iter() {
        if !contains_sym(arena, child, func_sym) {
            forcing.push(child);
            continue;
        }
        let (coeff, term) = arena.as_coeff_term(child);
        let factors: Vec<ExprId> = match arena.node(term) {
            ExprNode::Mul(fs) => fs.to_vec(),
            _ => vec![term],
        };
        let mut k_deriv: Option<usize> = None;
        let mut x_power: i64 = 0;
        for f in factors {
            if let Some(k) = chain.iter().position(|&d| d == f) {
                if k_deriv.replace(k).is_some() {
                    return None;
                }
            } else if f == var {
                x_power += 1;
            } else if let ExprNode::Pow(b, e) = *arena.node(f)
                && b == var
                && let Some(p) = arena.as_num(e).filter(|p| p.is_integer())
            {
                x_power += num_traits::ToPrimitive::to_i64(&p.to_integer())?;
            } else {
                return None;
            }
        }
        let k = k_deriv?;
        if x_power != i64::try_from(k).ok()? {
            return None;
        }
        a[k] += coeff;
    }
    if num_traits::Zero::is_zero(&a[order]) {
        return None;
    }
    // Σₖ aₖ·D(D − 1)⋯(D − k + 1) as coefficients of Dʲ.
    let mut cd = vec![Q::from_integer(0.into()); order + 1];
    for (k, ak) in a.iter().enumerate() {
        for (j, v) in falling_operator(k).iter().enumerate() {
            cd[j] += ak * v;
        }
    }
    Some(EulerOperator {
        coeffs: cd,
        forcing,
    })
}

/// [`linear_cc_ivp_lagrange`] for a Cauchy–Euler equation with all
/// conditions at one point `x₀ > 0`: with `x = x₀·eᵗ` the equation is
/// `Σ cdⱼ·Y⁽ʲ⁾(t) + g(x₀eᵗ) = 0` and the conditions become
/// `Y⁽ʲ⁾(0) = Wⱼ` through `x₀ᵏ·y⁽ᵏ⁾(x₀) = Σⱼ s(k, j)·Wⱼ`; the
/// constant-coefficient fit, mapped back by `t = ln(x/x₀)`.  The generic fit
/// over `RootOf(m⁴ + m + 1, k)` took 5 s (debug build) for
/// `x⁴y⁗ + 6x³y‴ + 7x²y″ + 2xy′ + y = x²`.
pub(super) fn euler_ivp_lagrange(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    x0: ExprId,
    values: &[ExprId],
) -> Option<ExprId> {
    let ExprNode::Symbol(func_sym) = *arena.node(func) else {
        return None;
    };
    let ExprNode::Symbol(var_sym) = *arena.node(var) else {
        return None;
    };
    if !is_real_constant(arena, x0)
        || !crate::transforms::evalf::evalf_complex64(arena, x0).is_ok_and(|z| z.re > 0.0)
    {
        return None;
    }
    let euler = euler_operator(arena, expr, func, var, func_sym, 2)?;
    let n = euler.coeffs.len() - 1;
    if values.len() != n {
        return None;
    }
    let t = arena.internal_symbol("__t_euler_ivp");
    let big_y = arena.internal_symbol("__Y_euler_ivp");
    // Σ cdⱼ·Y⁽ʲ⁾ + g(x₀·eᵗ)
    let mut parts = Vec::with_capacity(n + 1 + euler.forcing.len());
    let mut d = big_y;
    for (j, c) in euler.coeffs.iter().enumerate() {
        if j > 0 {
            d = arena.intern(ExprNode::Derivative(d, t));
        }
        if !num_traits::Zero::is_zero(c) {
            let c = ode_ratio_to_expr(arena, c);
            parts.push(arena.mul(&[c, d]));
        }
    }
    let ln_x = arena.ln(var);
    let ln_x0 = arena.ln(x0);
    let ln_shifted = arena.add(&[ln_x0, t]);
    for &f in &euler.forcing {
        let g = crate::transforms::subs::subs(arena, f, ln_x, ln_shifted);
        let post = crate::base::walk::post_order_ids(arena, g);
        let mut cache: rustc_hash::FxHashMap<ExprId, ExprId> = rustc_hash::FxHashMap::default();
        for &id in &post {
            let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
            let new = match *arena.node(rebuilt) {
                ExprNode::Pow(b, e) if b == var && !contains_sym(arena, e, var_sym) => {
                    let scale = arena.pow(x0, e);
                    let et = arena.mul(&[e, t]);
                    let growth = arena.exp(et);
                    arena.mul(&[scale, growth])
                }
                _ => rebuilt,
            };
            cache.insert(id, new);
        }
        let g = cache.get(&g).copied().unwrap_or(g);
        let exp_t = arena.exp(t);
        let x_of_t = arena.mul(&[x0, exp_t]);
        let g = crate::transforms::subs::subs(arena, g, var, x_of_t);
        let g = crate::transforms::eval::eval(arena, g);
        let g = crate::transforms::expand::expand(arena, g);
        let g = crate::transforms::eval::eval(arena, g);
        if contains_sym(arena, g, var_sym) {
            return None;
        }
        parts.push(g);
    }
    let ode_t = arena.add(&parts);
    let ode_t = crate::transforms::eval::eval(arena, ode_t);
    // W_k = x₀ᵏ·vₖ − Σ_{j<k} s(k, j)·Wⱼ
    let mut w: Vec<ExprId> = Vec::with_capacity(n);
    for (k, &v) in values.iter().enumerate() {
        let kk = arena.int(i64::try_from(k).ok()?);
        let x0k = arena.pow(x0, kk);
        let mut acc = vec![arena.mul(&[x0k, v])];
        for (j, s) in falling_operator(k).iter().enumerate().take(k) {
            if num_traits::Zero::is_zero(s) {
                continue;
            }
            let c = ode_ratio_to_expr(arena, &-s.clone());
            acc.push(arena.mul(&[c, w[j]]));
        }
        let wk = arena.add(&acc);
        w.push(crate::transforms::eval::eval(arena, wk));
    }
    let zero = arena.zero;
    let sol_t = linear_cc_ivp_lagrange(arena, ode_t, big_y, t, zero, &w)?;
    // t = ln(x/x₀)
    let ratio = arena.div(var, x0);
    let back = arena.ln(ratio);
    let sol = crate::transforms::subs::subs(arena, sol_t, t, back);
    let sol = crate::transforms::eval::eval(arena, sol);
    let sol = exp_log_to_power(arena, sol);
    if contains_sym(arena, sol, func_sym) || crate::base::walk::contains(arena, sol, t) {
        return None;
    }
    Some(sol)
}

/// `Σⱼ cdⱼ·Y⁽ʲ⁾(t) + Σ forcing(x = eᵗ) = 0` solved and mapped back to `x`
/// (`e^{k·t + m} → xᵏ·eᵐ`, `t → ln x`): the Cauchy–Euler equation whose
/// operator in `D = d/dt` has the coefficients `cd`.
fn euler_cauchy_by_exp(
    arena: &mut Arena,
    cd: &[Q],
    forcing: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
) -> Option<OdeResult> {
    let order = cd.len().checked_sub(1)?;
    let t = arena.internal_symbol("__t");
    let big_y = arena.internal_symbol("__Y");
    let (t_sym, y_sym) = match (arena.node(t), arena.node(big_y)) {
        (ExprNode::Symbol(ts), ExprNode::Symbol(ys)) => (*ts, *ys),
        _ => return None,
    };
    let exp_t = arena.exp(t);
    let ln_x = arena.ln(var);
    let mut parts = Vec::with_capacity(cd.len() + forcing.len());
    let mut d = big_y;
    for (j, c) in cd.iter().enumerate() {
        if j > 0 {
            d = arena.intern(ExprNode::Derivative(d, t));
        }
        if num_traits::Zero::is_zero(c) {
            continue;
        }
        let c_id = ode_ratio_to_expr(arena, c);
        parts.push(arena.mul(&[c_id, d]));
    }
    for &f in forcing {
        // ln x = t first (x > 0), then xᵖ = e^{p·t} (`√x` became
        // `(eᵗ)^(1/2)`, which is not `e^{t/2}` for complex `t`, and the
        // forcing was not recognised), then x = eᵗ.
        let g = crate::transforms::subs::subs(arena, f, ln_x, t);
        let post = crate::base::walk::post_order_ids(arena, g);
        let mut cache: rustc_hash::FxHashMap<ExprId, ExprId> = rustc_hash::FxHashMap::default();
        for &id in &post {
            let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
            let new = match *arena.node(rebuilt) {
                ExprNode::Pow(b, e) if b == var && !contains_sym(arena, e, var_sym) => {
                    let et = arena.mul(&[e, t]);
                    arena.exp(et)
                }
                _ => rebuilt,
            };
            cache.insert(id, new);
        }
        let g = cache.get(&g).copied().unwrap_or(g);
        let g = crate::transforms::subs::subs(arena, g, var, exp_t);
        parts.push(crate::transforms::eval::eval(arena, g));
    }
    let ode_t = arena.add(&parts);
    let ode_t = crate::transforms::eval::eval(arena, ode_t);
    if contains_sym(arena, ode_t, var_sym) {
        return None;
    }
    let res = dsolve(arena, ode_t, big_y, t)?;
    if contains_sym(arena, res.solution, y_sym)
        || res.constants.len() != order
        || crate::base::walk::has_unevaluated(arena, res.solution)
    {
        return None;
    }
    // e^{k·t + m} = x^k·e^m, then t = ln x.
    let mut replacements: Vec<(ExprId, ExprId)> = Vec::new();
    for id in crate::base::walk::post_order_ids(arena, res.solution) {
        let ExprNode::Exp(arg) = arena.node(id).clone() else {
            continue;
        };
        let arg = crate::transforms::expand::expand(arena, arg);
        let arg = crate::transforms::eval::eval(arena, arg);
        let Some(coeffs) = arena.coefficients_of(arg, t) else {
            continue;
        };
        if coeffs.len() != 2 || contains_sym(arena, coeffs[1], t_sym) {
            continue;
        }
        let x_k = arena.pow(var, coeffs[1]);
        let rest = arena.exp(coeffs[0]);
        let r = arena.mul(&[x_k, rest]);
        let r = crate::transforms::eval::eval(arena, r);
        replacements.push((id, r));
    }
    let solution = crate::transforms::subs::subs_map(arena, res.solution, &replacements);
    let solution = crate::transforms::subs::subs(arena, solution, t, ln_x);
    let solution = crate::transforms::eval::eval(arena, solution);
    if contains_sym(arena, solution, t_sym) {
        return None;
    }
    Some(OdeResult {
        solution,
        constants: res.constants,
        branches: Vec::new(),
    })
}
