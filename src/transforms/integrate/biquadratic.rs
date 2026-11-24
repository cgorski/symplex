//! Rational functions with a biquadratic denominator (part of [`integrate`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Rational functions with a biquadratic denominator
// ═══════════════════════════════════════════════════════════════════════════

/// `∫ N(x) / (x⁴ + p·x² + q) dx` with `deg N ≤ 3` and rational `p`, `q`,
/// for the two shapes without real roots:
///
/// ```text
///   p² < 4q:            x⁴ + p·x² + q = (x² − a·x + c)(x² + a·x + c),
///                                        c = √q,  a = √(2c − p)
///   p² > 4q, p, q > 0:  x⁴ + p·x² + q = (x² + r₁)(x² + r₂),
///                                        r₁,₂ = (p ∓ √(p² − 4q))/2 > 0
/// ```
///
/// followed by partial fractions over the corresponding real radical field
/// and the closed forms
/// `∫ (A·x + B)/(x² + a·x + c) = (A/2)·ln(x² + a·x + c)
///   + (2B − a·A)/√(4c − a²) · atan((2x + a)/√(4c − a²))`,
/// `∫ (A·x + B)/(x² + r) = (A/2)·ln(x² + r) + (B/√r)·atan(x/√r)`.
///
/// This is the route the general rational integrator gets wrong when the
/// Rothstein–Trager resultant has nested-radical roots (the `re(…)`/`im(…)`
/// closed forms of the 0.21 audit; also the denominators the Weierstrass
/// substitution produces for `∫ 1/(a·sin²x + b) dx`).  Here every constant
/// is an explicit real radical.  Returns `None` when `expr` is not of
/// either shape.
pub(super) fn try_biquadratic_rational(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    let (numer_id, denom_id) = crate::poly::polybridge::as_numer_denom(arena, expr);
    if denom_id == arena.one {
        return None;
    }
    let numer_id = crate::transforms::expand::expand(arena, numer_id);
    let numer_id = crate::transforms::eval::eval(arena, numer_id);
    let denom_id = crate::transforms::expand::expand(arena, denom_id);
    let denom_id = crate::transforms::eval::eval(arena, denom_id);
    let numer = crate::poly::polybridge::expr_to_poly(arena, numer_id, var)?;
    let denom = crate::poly::polybridge::expr_to_poly(arena, denom_id, var)?;
    if denom.degree() != Some(4) || numer.is_zero() {
        return None;
    }
    if !denom.coeff(3).is_zero() || !denom.coeff(1).is_zero() {
        return None;
    }
    let lead = denom.coeff(4);
    if lead.is_zero() {
        return None;
    }
    // Improper fraction: N = Q·D + R, ∫ N/D = ∫ Q + ∫ R/D with ∫ Q by the
    // power rule (Σ qₖ x^{k+1}/(k+1)).
    let (quotient, numer) = if numer.degree().unwrap_or(0) > 3 {
        let (quot, rem) = numer.div_rem(&denom);
        (Some(quot), rem)
    } else {
        (None, numer)
    };
    let poly_part = quotient.filter(|q| !q.is_zero()).map(|q| {
        let deg = q.degree().unwrap_or(0);
        let mut coeffs: Vec<Q> = vec![Q::zero(); deg + 2];
        for (k, c) in coeffs.iter_mut().enumerate().skip(1) {
            *c = q.coeff(k - 1) / Q::from_integer(k.into());
        }
        let antiderivative = Poly::from_coeffs(coeffs);
        crate::poly::polybridge::poly_to_expr(arena, &antiderivative, var)
    });
    if numer.is_zero() {
        // N was a polynomial multiple of D.
        return poly_part;
    }
    let p = denom.coeff(2) / &lead;
    let q = denom.coeff(0) / &lead;
    let n_ids: [ExprId; 4] = [
        arena.num_ratio(numer.coeff(0) / &lead),
        arena.num_ratio(numer.coeff(1) / &lead),
        arena.num_ratio(numer.coeff(2) / &lead),
        arena.num_ratio(numer.coeff(3) / &lead),
    ];
    let disc = &p * &p - Q::from_integer(4.into()) * &q;
    let quadratics: SmallVec<[(ExprId, ExprId, ExprId, ExprId); 2]> = if disc.is_negative() {
        tracing::debug!(p = %p, q = %q, "integrate: biquadratic denominator, x² ± a·x + c");
        biquadratic_conjugate_quadratics(arena, &p, &q, &n_ids)
    } else if disc.is_positive() && p.is_positive() && q.is_positive() {
        tracing::debug!(p = %p, q = %q, "integrate: biquadratic denominator, (x² + r₁)(x² + r₂)");
        biquadratic_even_quadratics(arena, &p, &disc, &n_ids)
    } else {
        // Real roots (or a repeated factor): the general integrator's business.
        return None;
    };

    // ∫ (A x + B)/(x² + a x + c) = (A/2) ln(x² + a x + c)
    //                              + (2B − aA)/k · atan((2x + a)/k),  k = √(4c − a²)
    let two = arena.int(2);
    let four = arena.int(4);
    let half = arena.rational(1, 2);
    let x_sq = arena.pow(var, two);
    let two_x = arena.mul(&[two, var]);
    let mut terms: SmallVec<[ExprId; 4]> = SmallVec::new();
    for &(a, c, lin, cst) in quadratics.iter() {
        let a_x = arena.mul(&[a, var]);
        let quad = arena.add(&[x_sq, a_x, c]);
        // Both quadratics are positive on the whole line: no |·| in the ln.
        let ln_quad = arena.ln(quad);
        let half_lin = arena.mul(&[half, lin]);
        terms.push(arena.mul(&[half_lin, ln_quad]));
        let four_c = arena.mul(&[four, c]);
        let a_sq = arena.mul(&[a, a]);
        let k_sq = arena.sub(four_c, a_sq);
        let k = arena.sqrt(k_sq);
        let two_cst = arena.mul(&[two, cst]);
        let a_lin = arena.mul(&[a, lin]);
        let atan_coeff_num = arena.sub(two_cst, a_lin);
        let atan_coeff = arena.div(atan_coeff_num, k);
        let atan_arg_num = arena.add(&[two_x, a]);
        let atan_arg = arena.div(atan_arg_num, k);
        let atan_val = arena.atan(atan_arg);
        terms.push(arena.mul(&[atan_coeff, atan_val]));
    }
    if let Some(poly_part) = poly_part {
        terms.push(poly_part);
    }
    let result = arena.add(&terms);
    let result = crate::transforms::eval::eval(arena, result);
    // The closed form is exact by construction; the check guards the radical
    // arithmetic above (a wrong sign here would be a silent wrong answer).
    if candidate_rejected(arena, expr, result, var, var_sym) {
        tracing::debug!("integrate: biquadratic closed form failed verification");
        return None;
    }
    Some(result)
}

/// The two factors `(a, c, A, B)` of `N/(x⁴ + p x² + q)` = `Σ (A x + B)/(x² + a x + c)`
/// when `p² < 4q`: `c = √q`, `a = ∓√(2c − p)`, and from
/// `N = (A x + B)(x² + a x + c) + (C x + D)(x² − a x + c)`:
/// `A + C = n₃`, `a(A − C) + (B + D) = n₂`, `c(A + C) + a(B − D) = n₁`, `c(B + D) = n₀`.
fn biquadratic_conjugate_quadratics(
    arena: &mut Arena,
    p: &Q,
    q: &Q,
    n: &[ExprId; 4],
) -> SmallVec<[(ExprId, ExprId, ExprId, ExprId); 2]> {
    let p_id = arena.num_ratio(p.clone());
    let q_id = arena.num_ratio(q.clone());
    let two = arena.int(2);
    let half = arena.rational(1, 2);
    let c = arena.sqrt(q_id);
    let two_c = arena.mul(&[two, c]);
    let a_sq = arena.sub(two_c, p_id);
    let a = arena.sqrt(a_sq);
    let neg_a = arena.neg(a);

    let sum_bd = arena.div(n[0], c); // B + D = n₀/c
    let sum_ac = n[3]; // A + C = n₃
    let diff_ac = {
        let t = arena.sub(n[2], sum_bd);
        arena.div(t, a)
    }; // A − C = (n₂ − n₀/c)/a
    let diff_bd = {
        let c_n3 = arena.mul(&[c, n[3]]);
        let t = arena.sub(n[1], c_n3);
        arena.div(t, a)
    }; // B − D = (n₁ − c·n₃)/a
    let coeff_a = {
        let s = arena.add(&[sum_ac, diff_ac]);
        arena.mul(&[half, s])
    };
    let coeff_c = {
        let s = arena.sub(sum_ac, diff_ac);
        arena.mul(&[half, s])
    };
    let coeff_b = {
        let s = arena.add(&[sum_bd, diff_bd]);
        arena.mul(&[half, s])
    };
    let coeff_d = {
        let s = arena.sub(sum_bd, diff_bd);
        arena.mul(&[half, s])
    };
    // (A x + B)/(x² − a x + c) + (C x + D)/(x² + a x + c)
    let mut out = SmallVec::new();
    out.push((neg_a, c, coeff_a, coeff_b));
    out.push((a, c, coeff_c, coeff_d));
    out
}

/// The two factors `(0, r, A, B)` of `N/(x⁴ + p x² + q)` = `Σ (A x + B)/(x² + r)`
/// when `p² > 4q` and `p, q > 0`: `r₁,₂ = (p ∓ √disc)/2`, and from
/// `N = (A x + B)(x² + r₂) + (C x + D)(x² + r₁)`:
/// `A = (n₁ − n₃ r₁)/(r₂ − r₁)`, `C = n₃ − A`, `B = (n₀ − n₂ r₁)/(r₂ − r₁)`, `D = n₂ − B`.
fn biquadratic_even_quadratics(
    arena: &mut Arena,
    p: &Q,
    disc: &Q,
    n: &[ExprId; 4],
) -> SmallVec<[(ExprId, ExprId, ExprId, ExprId); 2]> {
    let p_id = arena.num_ratio(p.clone());
    let disc_id = arena.num_ratio(disc.clone());
    let half = arena.rational(1, 2);
    let sqrt_disc = arena.sqrt(disc_id);
    let r1 = {
        let t = arena.sub(p_id, sqrt_disc);
        arena.mul(&[half, t])
    };
    let r2 = {
        let t = arena.add(&[p_id, sqrt_disc]);
        arena.mul(&[half, t])
    };
    // r₂ − r₁ = √disc
    let coeff_a = {
        let n3_r1 = arena.mul(&[n[3], r1]);
        let t = arena.sub(n[1], n3_r1);
        arena.div(t, sqrt_disc)
    };
    let coeff_c = arena.sub(n[3], coeff_a);
    let coeff_b = {
        let n2_r1 = arena.mul(&[n[2], r1]);
        let t = arena.sub(n[0], n2_r1);
        arena.div(t, sqrt_disc)
    };
    let coeff_d = arena.sub(n[2], coeff_b);
    let zero = arena.zero;
    let mut out = SmallVec::new();
    out.push((zero, r1, coeff_a, coeff_b));
    out.push((zero, r2, coeff_c, coeff_d));
    out
}
