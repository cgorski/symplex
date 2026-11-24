//! Variation of parameters: y'' + p·y' + q·y = g(x) (part of [`ode`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Variation of parameters: y'' + p·y' + q·y = g(x)
// ═══════════════════════════════════════════════════════════════════════════

/// Solve y'' + p·y' + q·y = g(x) via variation of parameters.
///
/// Used as a fallback when undetermined coefficients fails (e.g., forcing
/// function is tan(x), sec(x), etc.).
///
/// Given homogeneous solutions y₁, y₂:
/// - Wronskian W computed via differentiation (with Abel's identity fallback)
/// - Particular: y_p = −y₁·∫(y₂·g/W)dx + y₂·∫(y₁·g/W)dx
pub(super) fn try_variation_of_parameters(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
    var_sym: SymbolId,
) -> Option<OdeResult> {
    tracing::debug!("ode: trying variation of parameters");

    use num_traits::Zero;

    let dy_dx = arena.intern(ExprNode::Derivative(func, var));
    let d2y_dx2 = arena.intern(ExprNode::Derivative(dy_dx, var));

    let children = match arena.node(expr).clone() {
        ExprNode::Add(children) => children,
        _ => return None,
    };

    let mut a_coeff = num_rational::Ratio::<num_bigint::BigInt>::zero();
    let mut b_coeff = num_rational::Ratio::<num_bigint::BigInt>::zero();
    let mut c_coeff = num_rational::Ratio::<num_bigint::BigInt>::zero();
    let mut f_terms: Vec<ExprId> = Vec::new();

    for &child in &children {
        let (coeff, term) = arena.as_coeff_term(child);
        if term == d2y_dx2 {
            a_coeff += coeff;
        } else if term == dy_dx {
            b_coeff += coeff;
        } else if term == func {
            c_coeff += coeff;
        } else if !contains_sym(arena, child, func_sym) {
            f_terms.push(child);
        } else {
            return None;
        }
    }

    if a_coeff.is_zero() || f_terms.is_empty() {
        return None;
    }

    let b = &b_coeff / &a_coeff;
    let c = &c_coeff / &a_coeff;

    // g(x) = −f_sum / a
    let f_sum = if f_terms.len() == 1 {
        f_terms[0]
    } else {
        arena.add(&f_terms)
    };
    let neg_f = arena.neg(f_sum);
    let a_id = ode_ratio_to_expr(arena, &a_coeff);
    let g_x = arena.div(neg_f, a_id);
    let g_x = crate::transforms::eval::eval(arena, g_x);

    // Solve homogeneous equation
    let homo = solve_characteristic_equation(arena, b.clone(), c.clone(), var)?;
    let (y1, y2) = extract_fundamental_solutions(arena, homo.solution, &homo.constants)?;

    // Wronskian via symbolic differentiation
    let y1_prime = crate::transforms::diff::diff(arena, y1, var);
    let y2_prime = crate::transforms::diff::diff(arena, y2, var);
    let w_term1 = arena.mul(&[y1, y2_prime]);
    let w_term2 = arena.mul(&[y2, y1_prime]);
    let wronskian = arena.sub(w_term1, w_term2);
    let wronskian = crate::transforms::eval::eval(arena, wronskian);
    let wronskian = crate::transforms::expand::expand(arena, wronskian);
    let wronskian = crate::transforms::eval::eval(arena, wronskian);

    // If W still depends on var (e.g. cos²+sin² unsimplified), use Abel's
    // identity: W(x) = W(0)·exp(−b·x).
    let wronskian = if contains_sym(arena, wronskian, var_sym) {
        let w0 = crate::transforms::subs::subs(arena, wronskian, var, arena.zero);
        let w0 = crate::transforms::eval::eval(arena, w0);
        if w0 == arena.zero {
            return None;
        }
        if b.is_zero() {
            w0
        } else {
            let neg_b_id = ode_ratio_to_expr(arena, &(-b.clone()));
            let neg_bx = arena.mul(&[neg_b_id, var]);
            let exp_nbx = arena.exp(neg_bx);
            arena.mul(&[w0, exp_nbx])
        }
    } else {
        wronskian
    };

    if wronskian == arena.zero {
        return None;
    }

    // y_p = −y₁·∫(y₂·g/W)dx + y₂·∫(y₁·g/W)dx
    let y2_g = arena.mul(&[y2, g_x]);
    let integrand1 = arena.div(y2_g, wronskian);
    let integrand1 = crate::transforms::eval::eval(arena, integrand1);
    let integral1 = integrate_forms(arena, integrand1, var);
    if integral_failed(arena, integral1) {
        return None;
    }

    let y1_g = arena.mul(&[y1, g_x]);
    let integrand2 = arena.div(y1_g, wronskian);
    let integrand2 = crate::transforms::eval::eval(arena, integrand2);
    let integral2 = integrate_forms(arena, integrand2, var);
    if integral_failed(arena, integral2) {
        return None;
    }

    let term1 = arena.mul(&[y1, integral1]);
    let neg_term1 = arena.neg(term1);
    let term2 = arena.mul(&[y2, integral2]);
    let y_p = arena.add(&[neg_term1, term2]);
    let y_p = crate::transforms::eval::eval(arena, y_p);

    let solution = arena.add(&[homo.solution, y_p]);
    let solution = crate::transforms::eval::eval(arena, solution);

    Some(OdeResult {
        solution,
        constants: homo.constants,
        branches: Vec::new(),
    })
}

/// Highest order [`try_linear_cc_green`] handles (one antiderivative per
/// root and multiplicity).
const MAX_GREEN_ORDER: usize = 6;

/// A root of a characteristic polynomial for [`try_linear_cc_green`]: its
/// multiplicity `mult` and the cofactor `p(s)/(s − root)^mult` (an
/// expression in `s`).
struct GreenRoot {
    root: ExprId,
    mult: usize,
    cofactor: ExprId,
}

/// The roots of the rational polynomial `p` with their cofactors, when
/// every irreducible factor of `p` over ℚ has degree 1 or 2 (rational
/// roots, real quadratic surds, complex quadratic pairs); `None` otherwise.
fn green_roots(arena: &mut Arena, p: &crate::poly::Poly, s: ExprId) -> Option<Vec<GreenRoot>> {
    let (content, factors) = p.factor_over_z();
    let factor_exprs: Vec<ExprId> = factors
        .iter()
        .map(|(g, _)| crate::poly::polybridge::poly_to_expr(arena, g, s))
        .collect();
    let mut out = Vec::new();
    for (fi, (g, mult)) in factors.iter().enumerate() {
        let mult = *mult as usize;
        let lead = g.leading_coeff()?.clone();
        let roots: Vec<ExprId> = match g.degree()? {
            1 => vec![ode_ratio_to_expr(arena, &(-g.coeff(0) / &lead))],
            2 => {
                let (a, b, c) = (lead.clone(), g.coeff(1), g.coeff(0));
                let disc = &b * &b - Q::from_integer(4.into()) * &a * &c;
                let half = Q::new(1.into(), 2.into());
                let mag = ode_ratio_to_expr(arena, &disc.abs());
                let half_id = ode_ratio_to_expr(arena, &half);
                let sqrt = arena.pow(mag, half_id);
                let sqrt = if disc.is_negative() {
                    let i_unit = arena.i_unit;
                    arena.mul(&[i_unit, sqrt])
                } else {
                    sqrt
                };
                let two_a = ode_ratio_to_expr(arena, &(Q::from_integer(2.into()) * &a));
                let neg_b = ode_ratio_to_expr(arena, &(-b));
                let neg_sqrt = arena.neg(sqrt);
                [sqrt, neg_sqrt]
                    .into_iter()
                    .map(|d| {
                        let num = arena.add(&[neg_b, d]);
                        let r = arena.div(num, two_a);
                        let r = crate::transforms::expand::expand(arena, r);
                        crate::transforms::eval::eval(arena, r)
                    })
                    .collect()
            }
            _ => return None,
        };
        for (ri, &root) in roots.iter().enumerate() {
            // p(s)/(s − r)^m = content · lead^m · (s − r′)^m · Π_{other} h(s)^{m_h}
            let mut parts = vec![ode_ratio_to_expr(arena, &content)];
            let m_id = arena.int(i64::try_from(mult).ok()?);
            let lead_id = ode_ratio_to_expr(arena, &lead);
            parts.push(arena.pow(lead_id, m_id));
            if roots.len() == 2 {
                let other = roots[1 - ri];
                let lin = arena.sub(s, other);
                parts.push(arena.pow(lin, m_id));
            }
            for (hi, (_, mh)) in factors.iter().enumerate() {
                if hi != fi {
                    let e = arena.int(i64::from(*mh));
                    parts.push(arena.pow(factor_exprs[hi], e));
                }
            }
            let cofactor = arena.mul(&parts);
            out.push(GreenRoot {
                root,
                mult,
                cofactor,
            });
        }
    }
    Some(out)
}

/// `Σ aₖ·y⁽ᵏ⁾ = g(x)` with rational `aₖ` (order `2 ≤ n ≤ 6`) and a forcing
/// term outside undetermined coefficients, by the Green's function of
/// `p(D)`: with `1/p(s) = Σ_r Σ_{j≤m_r} A_{r,j}/(s − r)ʲ`,
/// `yₚ(x) = Σ A_{r,j}/(j−1)! · ∫ˣ (x − t)^{j−1}·e^{r(x−t)}·g(t) dt`, the
/// binomial expansion of `(x − t)^{j−1}` leaving the antiderivatives
/// `∫ tⁱ·e^{−rt}·g(t) dt` (trigonometric and hyperbolic functions in `g`
/// written as exponentials).  The roots are those of factors of degree ≤ 2
/// over ℚ, complex ones as complex exponentials: `y″ + y = eˣ/x` gets
/// `Ei((1 ∓ i)x)` (a real function written with `i`), `y‴ − y′ = 1/x` gets
/// `ln x`, `Ei(±x)`.  Before, variation of parameters was only tried for
/// order 2 and with the real basis (`∫ eˣ sin x/x dx` is not found), and
/// these equations were refused.  The homogeneous part is the real basis of
/// [`homogeneous_basis_cc`].
pub(super) fn try_linear_cc_green(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
) -> Option<OdeResult> {
    let ode = extract_linear_cc(arena, expr, func, var, func_sym)?;
    let n = ode.coeffs.len() - 1;
    if !(2..=MAX_GREEN_ORDER).contains(&n) || ode.forcing.is_empty() {
        return None;
    }
    let p = crate::poly::Poly::from_coeffs(ode.coeffs.clone());
    let s = arena.symbol("__s_green");
    let roots = green_roots(arena, &p, s)?;
    let basis = homogeneous_basis_cc(arena, &ode.coeffs, var)?;
    if basis.len() != n {
        return None;
    }
    // g = −(forcing terms), circular and hyperbolic functions as exponentials
    let g = {
        let f = arena.add(&ode.forcing);
        let neg = arena.neg(f);
        let g = crate::simplify::rewrite::rewrite_as_exp(arena, neg);
        crate::transforms::eval::eval(arena, g)
    };
    let mut particular = Vec::new();
    for gr in &roots {
        // F_i = ∫ xⁱ·e^{−rx}·g dx, i < m
        let neg_r = arena.neg(gr.root);
        let neg_rx = arena.mul(&[neg_r, var]);
        let decay = arena.exp(neg_rx);
        let mut antiderivatives = Vec::with_capacity(gr.mult);
        for i in 0..gr.mult {
            let xi = arena.int(i64::try_from(i).ok()?);
            let xi = arena.pow(var, xi);
            let integrand = arena.mul(&[xi, decay, g]);
            let integrand = crate::transforms::expand::expand(arena, integrand);
            let integrand = crate::transforms::eval::eval(arena, integrand);
            let f = integrate_forms(arena, integrand, var);
            if integral_failed(arena, f) {
                return None;
            }
            antiderivatives.push(f);
        }
        let rx = arena.mul(&[gr.root, var]);
        let growth = arena.exp(rx);
        // A_{r,j} = (1/(m−j)!)·[d^{m−j}/ds^{m−j} 1/cofactor](r)
        let mut inv = {
            let m1 = arena.int(-1);
            arena.pow(gr.cofactor, m1)
        };
        let mut derivs = vec![inv];
        for _ in 1..gr.mult {
            inv = crate::transforms::diff::diff(arena, inv, s);
            derivs.push(inv);
        }
        for j in 1..=gr.mult {
            let k = gr.mult - j;
            let at = crate::transforms::subs::subs(arena, derivs[k], s, gr.root);
            let fact = ode_ratio_to_expr(arena, &falling_factorial_rat(k, k));
            let a = arena.div(at, fact);
            let a = crate::transforms::expand::expand(arena, a);
            let a = crate::transforms::eval::eval(arena, a);
            if arena.is_zero_structural(a) {
                continue;
            }
            // Σ_i C(j−1, i)·x^{j−1−i}·(−1)^i·F_i
            let mut inner = Vec::with_capacity(j);
            for (i, &fi) in antiderivatives.iter().enumerate().take(j) {
                let mut c = Q::from_integer(binomial((j - 1) as u64, i as u64));
                if i % 2 == 1 {
                    c = -c;
                }
                let c = ode_ratio_to_expr(arena, &c);
                let xp = arena.int(i64::try_from(j - 1 - i).ok()?);
                let xp = arena.pow(var, xp);
                inner.push(arena.mul(&[c, xp, fi]));
            }
            let inner = arena.add(&inner);
            let jf = ode_ratio_to_expr(arena, &falling_factorial_rat(j - 1, j - 1));
            let term = arena.mul(&[a, growth, inner]);
            particular.push(arena.div(term, jf));
        }
    }
    // Products of exponentials merged, complex exponentials as cos + i·sin
    // (the imaginary parts of the elementary terms then cancel).
    let y_p = arena.add(&particular);
    let y_p = crate::simplify::powsimp::powsimp(arena, y_p);
    let y_p = crate::transforms::eval::eval(arena, y_p);
    let y_p = complex_exp_as_trig(arena, y_p, var);
    let y_p = crate::transforms::expand::expand(arena, y_p);
    let y_p = crate::transforms::eval::eval(arena, y_p);
    if crate::base::walk::has_unevaluated(arena, y_p) {
        return None;
    }
    let mut constants = Vec::with_capacity(n);
    let mut terms = Vec::with_capacity(n + 1);
    for (i, &b) in basis.iter().enumerate() {
        let c = arena.symbol(&format!("C{}", i + 1));
        constants.push(c);
        terms.push(arena.mul(&[c, b]));
    }
    terms.push(y_p);
    let solution = arena.add(&terms);
    let solution = crate::transforms::eval::eval(arena, solution);
    Some(OdeResult {
        solution,
        constants,
        branches: Vec::new(),
    })
}

/// Extract fundamental solutions y₁ and y₂ from a homogeneous solution
/// of the form C1·y₁ + C2·y₂ (or multiplied by a common factor).
///
/// Substitutes C1=1,C2=0 and C1=0,C2=1 to recover the individual solutions.
fn extract_fundamental_solutions(
    arena: &mut Arena,
    homo_solution: ExprId,
    constants: &[ExprId],
) -> Option<(ExprId, ExprId)> {
    if constants.len() != 2 {
        return None;
    }
    let c1 = constants[0];
    let c2 = constants[1];
    let zero = arena.zero;
    let one = arena.one;

    let y1 = crate::transforms::subs::subs(arena, homo_solution, c1, one);
    let y1 = crate::transforms::subs::subs(arena, y1, c2, zero);
    let y1 = crate::transforms::eval::eval(arena, y1);

    let y2 = crate::transforms::subs::subs(arena, homo_solution, c1, zero);
    let y2 = crate::transforms::subs::subs(arena, y2, c2, one);
    let y2 = crate::transforms::eval::eval(arena, y2);

    if y1 == arena.zero || y2 == arena.zero {
        return None;
    }

    Some((y1, y2))
}
