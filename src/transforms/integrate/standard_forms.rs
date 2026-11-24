//! standard forms (part of [`integrate`](super)).

use super::*;

/// Try to recognise standard‑form integrals of the shape
/// `(x² ± a²)^n` where `n` is −1 or −1/2.
///
/// Returns `Some(antiderivative)` on success.
pub(super) fn try_standard_form_integral(
    arena: &mut Arena,
    _expr: ExprId,
    base: ExprId,
    exp: ExprId,
    var: ExprId,
    _var_sym: SymbolId,
) -> Option<ExprId> {
    // ── Check exponent ─────────────────────────────────────────────
    let exp_val = arena.as_num(exp)?.clone();
    let neg_one = num_rational::Ratio::<num_bigint::BigInt>::from_integer((-1).into());
    let neg_half = num_rational::Ratio::<num_bigint::BigInt>::new((-1).into(), 2.into());

    let is_neg_one = exp_val == neg_one;
    let is_neg_half = exp_val == neg_half;
    if !is_neg_one && !is_neg_half {
        return None;
    }

    // ── Check base is Add with exactly 2 children ──────────────────
    let children = match arena.node(base).clone() {
        ExprNode::Add(c) if c.len() == 2 => c,
        _ => return None,
    };

    // ── Classify each child ────────────────────────────────────────
    let mut const_val: Option<Q> = None;
    let mut has_pos_x2 = false;
    let mut has_neg_x2 = false;

    for &child in children.iter() {
        if let Some(n) = arena.as_num(child) {
            const_val = Some(n.clone());
        } else if is_var_squared(arena, child, var) {
            has_pos_x2 = true;
        } else if is_neg_var_squared(arena, child, var) {
            has_neg_x2 = true;
        } else {
            return None;
        }
    }

    let c_val = const_val?;
    if !has_pos_x2 && !has_neg_x2 {
        return None;
    }

    let a_squared = c_val.abs();
    if a_squared.is_zero() {
        return None;
    }

    // ── Build x/a and 1/a (simplify when a²=1) ────────────────────
    let a_sq_is_one = a_squared == num_rational::Ratio::<num_bigint::BigInt>::one();

    let x_over_a = if a_sq_is_one {
        var
    } else {
        let a_sq_id = arena.num_ratio(a_squared.clone());
        let nh = arena.rational(-1, 2);
        let a_inv = arena.pow(a_sq_id, nh); // (a²)^{-1/2} = 1/a
        arena.mul(&[var, a_inv])
    };

    // 1/a   (only needed for exp == -1 forms)
    let one_over_a = if a_sq_is_one {
        None
    } else {
        let a_sq_id = arena.num_ratio(a_squared.clone());
        let nh = arena.rational(-1, 2);
        Some(arena.pow(a_sq_id, nh))
    };

    // ── Match patterns ─────────────────────────────────────────────

    // Pattern: x² + a²  (c_val > 0, positive x²)
    if has_pos_x2 && c_val.is_positive() {
        if is_neg_one {
            // A3: ∫ (x²+a²)^{-1} dx = (1/a)·atan(x/a)
            let atan_val = arena.atan(x_over_a);
            return Some(match one_over_a {
                Some(inv_a) => arena.mul(&[inv_a, atan_val]),
                None => atan_val,
            });
        }
        if is_neg_half {
            // A5: ∫ (x²+a²)^{-1/2} dx = asinh(x/a)
            let asinh_val = arena.asinh(x_over_a);
            return Some(asinh_val);
        }
    }

    // Pattern: a² − x²  (c_val > 0, negative x²)
    if has_neg_x2 && c_val.is_positive() {
        if is_neg_half {
            // A4: ∫ (a²−x²)^{-1/2} dx = asin(x/a)
            let asin_val = arena.asin(x_over_a);
            return Some(asin_val);
        }
        if is_neg_one {
            // A7: ∫ (a²−x²)^{-1} dx = (1/a)·atanh(x/a)
            let atanh_val = arena.atanh(x_over_a);
            return Some(match one_over_a {
                Some(inv_a) => arena.mul(&[inv_a, atanh_val]),
                None => atanh_val,
            });
        }
    }

    // Pattern: x² − a²  (c_val < 0, positive x², a² = |c_val|)
    if has_pos_x2 && c_val.is_negative() && is_neg_half {
        // A6: ∫ (x²−a²)^{-1/2} dx = ln|x + √(x²−a²)|, the real-variable
        // antiderivative on both x > a and x < −a.  `acosh(x/a)` (SymPy's
        // answer, and ours before 0.24) differs from it by a constant for
        // x > a but has the wrong derivative for x < −a.
        return Some(ln_abs_x_plus_sqrt(arena, var, base));
    }

    None
}

/// `ln|v + √q|`, the real-variable antiderivative of `v′/√q` when
/// `q = v² − c` for a constant `c > 0` (`q` is `x² − a²`, or `a·u² + d`
/// with `v = √a·u`): `d/dx ln|v + √(v² − c)| = v′/√(v² − c)` on both
/// `v > √c` and `v < −√c`.
fn ln_abs_x_plus_sqrt(arena: &mut Arena, v: ExprId, q: ExprId) -> ExprId {
    let half = arena.rational(1, 2);
    let sqrt_q = arena.pow(q, half);
    let sum = arena.add(&[v, sqrt_q]);
    let abs_sum = arena.abs(sum);
    arena.ln(abs_sum)
}

/// Try integrating `(ax²+bx+c)^exp` via completing the square.
///
/// Handles two exponent values:
/// - `exp = -1`:   `∫ 1/(ax²+bx+c) dx` → atan form
/// - `exp = -1/2`: `∫ 1/√(ax²+bx+c) dx` → asinh / acosh / asin form
pub(super) fn try_complete_square_integral(
    arena: &mut Arena,
    base: ExprId,
    exp: ExprId,
    var: ExprId,
    _var_sym: SymbolId,
) -> Option<ExprId> {
    let exp_r = arena.as_num(exp)?.clone();
    let neg_one = num_rational::Ratio::<num_bigint::BigInt>::from_integer((-1).into());
    let neg_half = num_rational::Ratio::<num_bigint::BigInt>::new((-1).into(), 2.into());

    let is_neg_one = exp_r == neg_one;
    let is_neg_half = exp_r == neg_half;
    if !is_neg_one && !is_neg_half {
        return None;
    }

    // base must be a quadratic in var: ax² + bx + c
    let poly_opt = crate::poly::polybridge::expr_to_poly(arena, base, var);

    // ── Symbolic fallback for irrational coefficients ──────────────
    // When expr_to_poly fails (e.g., coefficients contain √5 or ∛2),
    // try symbolic_quadratic_coeffs and arena-level completing the square.
    // This enables integration of terms produced by apart for cyclotomic
    // denominators like x⁵−1 and x⁸−1.
    if poly_opt.is_none() && is_neg_one {
        tracing::debug!(
            "try_complete_square: expr_to_poly failed, trying symbolic quadratic coefficients"
        );
        if let Some((c_id, d_id, e_id)) = symbolic_quadratic_coeffs(arena, base, var, _var_sym) {
            tracing::debug!(
                "try_complete_square: symbolic quadratic coefficients extracted, completing the square"
            );

            // Normalize to monic: b_sym = d/c, c_norm = e/c
            let b_sym = arena.div(d_id, c_id);
            let c_norm = arena.div(e_id, c_id);

            // Complete the square: x² + bx + c_norm = (x + b/2)² + (c_norm − b²/4)
            let two = arena.int(2);
            let half_b = arena.div(b_sym, two);
            let half_b_sq = arena.mul(&[half_b, half_b]);
            let disc = arena.sub(c_norm, half_b_sq);
            let disc = crate::transforms::eval::eval(arena, disc);

            // Sign test: discriminant must be positive for the atan form.
            let disc_sign = crate::poly::algebraic::sign_checked(arena, disc);
            tracing::debug!(?disc_sign, "try_complete_square: discriminant sign");
            if disc_sign != Some(1) {
                tracing::debug!(
                    "try_complete_square: discriminant non-positive, atan form not applicable"
                );
                return None;
            }

            // Result: (1/(a·√d)) · atan((x + b/2) / √d)
            let half = arena.rational(1, 2);
            let sqrt_disc = arena.pow(disc, half);
            let shifted = arena.add(&[var, half_b]);
            let ratio = arena.div(shifted, sqrt_disc);
            let atan_val = arena.atan(ratio);

            let sqrt_disc2 = arena.pow(disc, half);
            let a_sqrt_d = arena.mul(&[c_id, sqrt_disc2]);
            tracing::debug!(
                "try_complete_square: symbolic completing-the-square succeeded → atan form"
            );
            return Some(arena.div(atan_val, a_sqrt_d));
        }
    }

    // ── Rational-coefficient path ──────────────────────────────────
    let poly = poly_opt?;
    if poly.degree()? != 2 {
        return None;
    }

    let a_coeff = poly.coeff(2);
    let b_coeff = poly.coeff(1);
    let c_coeff = poly.coeff(0);

    if a_coeff.is_zero() {
        return None;
    }

    // ── exp = -1: ∫ 1/(ax²+bx+c) dx ───────────────────────────────
    if is_neg_one {
        // Normalize to monic: divide by a
        let b = &b_coeff / &a_coeff;
        let c = &c_coeff / &a_coeff;

        // If there's no linear term, this is a standard form — let the other helper handle it.
        if b.is_zero() {
            return None;
        }

        // Complete the square: x² + bx + c = (x + b/2)² + (c - b²/4)
        let half_b = &b / &num_rational::Ratio::from_integer(2.into());
        let d = &c - &(&half_b * &half_b); // d = c - b²/4

        if d.is_zero() || d.is_negative() {
            return None; // Can't use atan form if d ≤ 0
        }

        // Build (x + b/2)
        let half_b_id = {
            let nid = arena.intern_num(half_b.clone());
            arena.intern(crate::base::node::ExprNode::Num(nid))
        };
        let shifted = arena.add(&[var, half_b_id]);

        // Build √d
        let d_id = {
            let nid = arena.intern_num(d.clone());
            arena.intern(crate::base::node::ExprNode::Num(nid))
        };
        let half = arena.rational(1, 2);
        let sqrt_d = arena.pow(d_id, half);

        // Result: (1/(a·√d)) · atan((x+b/2)/√d)
        let ratio = arena.div(shifted, sqrt_d);
        let atan_result = arena.atan(ratio);

        // Divide by a·√d
        let a_id = {
            let nid = arena.intern_num(a_coeff);
            arena.intern(crate::base::node::ExprNode::Num(nid))
        };
        // Rebuild √d for the denominator (arena IDs are Copy, but let's be explicit)
        let sqrt_d2 = arena.pow(d_id, half);
        let a_sqrt_d = arena.mul(&[a_id, sqrt_d2]);

        return Some(arena.div(atan_result, a_sqrt_d));
    }

    // ── exp = -1/2: ∫ 1/√(ax²+bx+c) dx ───────────────────────────
    // Complete the square: ax²+bx+c = a·(x + b/(2a))² + (c − b²/(4a))
    // Let u = x + b/(2a),  d = c − b²/(4a).
    // Then ∫ 1/√(a·u² + d) du.
    let two_r = num_rational::Ratio::<num_bigint::BigInt>::from_integer(2.into());
    let four_r = num_rational::Ratio::<num_bigint::BigInt>::from_integer(4.into());

    let shift = &b_coeff / &(&a_coeff * &two_r); // b/(2a)
    let d = &c_coeff - &(&b_coeff * &b_coeff / &(&a_coeff * &four_r)); // c − b²/(4a)

    // Build u = x + b/(2a)
    let u_expr = if shift.is_zero() {
        var
    } else {
        let shift_id = arena.num_ratio(shift.clone());
        arena.add(&[var, shift_id])
    };

    let half = arena.rational(1, 2);

    if a_coeff.is_positive() {
        // a > 0
        let a_id = arena.num_ratio(a_coeff.clone());
        let sqrt_a = arena.pow(a_id, half); // √a
        let inv_sqrt_a = {
            let neg_half_e = arena.rational(-1, 2);
            arena.pow(a_id, neg_half_e)
        }; // 1/√a

        if d.is_positive() {
            // a > 0, d > 0: (1/√a) · asinh(u·√a / √d)
            let d_id = arena.num_ratio(d.clone());
            let sqrt_d = arena.pow(d_id, half);
            let u_sqrt_a = arena.mul(&[u_expr, sqrt_a]);
            let arg = arena.div(u_sqrt_a, sqrt_d);
            let asinh_val = arena.asinh(arg);
            return Some(arena.mul(&[inv_sqrt_a, asinh_val]));
        } else if d.is_negative() {
            // a > 0, d < 0: (1/√a) · ln|u·√a + √(a u² + d)|, valid on both
            // sides of the gap (acosh(u·√a/√|d|) is wrong for u < 0).
            let u_sqrt_a = arena.mul(&[u_expr, sqrt_a]);
            let ln_val = ln_abs_x_plus_sqrt(arena, u_sqrt_a, base);
            return Some(arena.mul(&[inv_sqrt_a, ln_val]));
        } else {
            // a > 0, d = 0: √(a·u²) = √a·|u| for the real variable, so the
            // antiderivative is sign(u)·ln|u|/√a (0.23 dropped the sign:
            // ∫ 1/√((3x − 2)²) dx was ln|x − 2/3|/3, wrong for x < 2/3;
            // found by the Rubi suite).
            let abs_u = arena.abs(u_expr);
            let ln_u = arena.ln(abs_u);
            let sign_u = arena.sign(u_expr);
            return Some(arena.mul(&[inv_sqrt_a, sign_u, ln_u]));
        }
    } else if a_coeff.is_negative() && d.is_positive() {
        // a < 0, d > 0: (1/√|a|) · asin(u·√|a| / √d)
        let abs_a = a_coeff.abs();
        let abs_a_id = arena.num_ratio(abs_a.clone());
        let sqrt_abs_a = arena.pow(abs_a_id, half);
        let inv_sqrt_abs_a = {
            let neg_half_e = arena.rational(-1, 2);
            arena.pow(abs_a_id, neg_half_e)
        };
        let d_id = arena.num_ratio(d.clone());
        let sqrt_d = arena.pow(d_id, half);
        let u_sqrt_abs_a = arena.mul(&[u_expr, sqrt_abs_a]);
        let arg = arena.div(u_sqrt_abs_a, sqrt_d);
        let asin_val = arena.asin(arg);
        return Some(arena.mul(&[inv_sqrt_abs_a, asin_val]));
    }

    None
}

/// Detect `sec(x)·tan(x)` and `csc(x)·cot(x)` patterns in a product.
///
/// - `sin(x) · cos(x)^{-2}` → `cos(x)^{-1}`   (∫ sec·tan dx = sec)
/// - `cos(x) · sin(x)^{-2}` → `-sin(x)^{-1}`   (∫ csc·cot dx = −csc)
pub(super) fn try_trig_recip_product(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    if dependent.len() != 2 {
        return None;
    }

    let neg_two = num_rational::Ratio::<num_bigint::BigInt>::from_integer((-2).into());

    for (i, j) in [(0usize, 1usize), (1, 0)] {
        let node_i = arena.node(dependent[i]).clone();
        let node_j = arena.node(dependent[j]).clone();

        // Pattern: sin(g) · cos(g)^{-2} → cos(g)^{-1} [/ chain coeff]
        if let ExprNode::Sin(inner_sin) = node_i
            && let ExprNode::Pow(base_j, exp_j) = node_j
            && let ExprNode::Cos(inner_cos) = arena.node(base_j).clone()
            && inner_sin == inner_cos
            && let Some(e) = arena.as_num(exp_j)
            && *e == neg_two
        {
            let neg_one_e = arena.int(-1);
            if inner_sin == var {
                return Some(arena.pow(base_j, neg_one_e));
            } else if let Some((a_expr, _)) =
                symbolic_linear_coeff_of(arena, inner_sin, var, var_sym)
            {
                let recip = arena.pow(base_j, neg_one_e);
                return Some(arena.div(recip, a_expr));
            }
        }

        // Pattern: cos(g) · sin(g)^{-2} → −sin(g)^{-1} [/ chain coeff]
        if let ExprNode::Cos(inner_cos) = node_i
            && let ExprNode::Pow(base_j, exp_j) = node_j
            && let ExprNode::Sin(inner_sin) = arena.node(base_j).clone()
            && inner_cos == inner_sin
            && let Some(e) = arena.as_num(exp_j)
            && *e == neg_two
        {
            let neg_one_e = arena.int(-1);
            if inner_cos == var {
                let recip = arena.pow(base_j, neg_one_e);
                return Some(arena.neg(recip));
            } else if let Some((a_expr, _)) =
                symbolic_linear_coeff_of(arena, inner_cos, var, var_sym)
            {
                let recip = arena.pow(base_j, neg_one_e);
                let neg_recip = arena.neg(recip);
                return Some(arena.div(neg_recip, a_expr));
            }
        }
    }

    None
}

/// Detect `x / √(ax²+bx+c)` and integrate using the decomposition:
///
///   `∫ x/√R dx = √R/a − (b/(2a))·∫ 1/√R dx`
///
/// where `R = ax²+bx+c`.
pub(super) fn try_x_over_sqrt_quadratic(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    if dependent.len() != 2 {
        return None;
    }

    // Find which factor is var and which is Pow(quadratic, -1/2)
    let pow_idx = if dependent[0] == var {
        1
    } else if dependent[1] == var {
        0
    } else {
        return None;
    };

    // Check the other factor is Pow(base, -1/2)
    let (base, exp_id) = match arena.node(dependent[pow_idx]).clone() {
        ExprNode::Pow(b, e) => (b, e),
        _ => return None,
    };

    let exp_r = arena.as_num(exp_id)?.clone();
    let neg_half = num_rational::Ratio::<num_bigint::BigInt>::new((-1).into(), 2.into());
    if exp_r != neg_half {
        return None;
    }

    // base must be quadratic in var
    let poly = crate::poly::polybridge::expr_to_poly(arena, base, var)?;
    if poly.degree()? != 2 {
        return None;
    }

    let a_coeff = poly.coeff(2);
    let b_coeff = poly.coeff(1);

    if a_coeff.is_zero() {
        return None;
    }

    // √R = base^{1/2}
    let half = arena.rational(1, 2);
    let sqrt_r = arena.pow(base, half);

    // First term: √R / a
    let a_id = arena.num_ratio(a_coeff.clone());
    let first_term = arena.div(sqrt_r, a_id);

    if b_coeff.is_zero() {
        // No linear term: ∫ x/√(ax²+c) dx = √(ax²+c)/a
        return Some(first_term);
    }

    // Need I_0 = ∫ 1/√R dx
    let i_0 = integrate_node(
        arena,
        dependent[pow_idx],
        var,
        var_sym,
        depth.saturating_sub(1),
    );
    if matches!(arena.node(i_0), ExprNode::Integral(_, _)) {
        return None;
    }

    // b/(2a)
    let two_r = num_rational::Ratio::<num_bigint::BigInt>::from_integer(2.into());
    let b_over_2a = &b_coeff / &(&a_coeff * &two_r);
    let b_over_2a_id = arena.num_ratio(b_over_2a.clone());

    // Result: √R/a − (b/(2a))·I_0
    let second_term = arena.mul(&[b_over_2a_id, i_0]);
    Some(arena.sub(first_term, second_term))
}

/// Try to integrate `(a² ± x²)^{1/2}` forms using trig substitution results.
///
/// Handles the three standard trig substitution patterns with positive
/// half-exponent:
/// - `∫ √(a²−x²) dx = ½(x·√(a²−x²) + a²·asin(x/a))`
/// - `∫ √(x²+a²) dx = ½(x·√(x²+a²) + a²·asinh(x/a))`
/// - `∫ √(x²−a²) dx = ½(x·√(x²−a²) − a²·acosh(x/a))`
pub(super) fn try_trig_sub_sqrt_integral(
    arena: &mut Arena,
    base: ExprId,
    exp: ExprId,
    var: ExprId,
    _var_sym: SymbolId,
) -> Option<ExprId> {
    // ── Check exponent is 1/2 ──────────────────────────────────────
    let exp_val = arena.as_num(exp)?.clone();
    let pos_half = num_rational::Ratio::<num_bigint::BigInt>::new(1.into(), 2.into());
    if exp_val != pos_half {
        return None;
    }

    // ── Check base is Add with exactly 2 children ──────────────────
    let children = match arena.node(base).clone() {
        ExprNode::Add(c) if c.len() == 2 => c,
        _ => return None,
    };

    // ── Classify each child ────────────────────────────────────────
    let mut const_val: Option<Q> = None;
    let mut has_pos_x2 = false;
    let mut has_neg_x2 = false;

    for &child in children.iter() {
        if let Some(n) = arena.as_num(child) {
            const_val = Some(n.clone());
        } else if is_var_squared(arena, child, var) {
            has_pos_x2 = true;
        } else if is_neg_var_squared(arena, child, var) {
            has_neg_x2 = true;
        } else {
            return None;
        }
    }

    let c_val = const_val?;
    if !has_pos_x2 && !has_neg_x2 {
        return None;
    }

    let a_squared = c_val.abs();
    if a_squared.is_zero() {
        return None;
    }

    let half = arena.rational(1, 2);
    let a_sq_is_one = a_squared == num_rational::Ratio::<num_bigint::BigInt>::one();

    // √(base) for reuse in the result
    let sqrt_base = arena.pow(base, half);

    // a² as an expression
    let a_sq_expr = if a_sq_is_one {
        arena.one
    } else {
        arena.num_ratio(a_squared.clone())
    };

    // x/a = x · (a²)^{-1/2}
    let x_over_a = if a_sq_is_one {
        var
    } else {
        let neg_half = arena.rational(-1, 2);
        let a_inv = arena.pow(a_sq_expr, neg_half);
        arena.mul(&[var, a_inv])
    };

    // ── Pattern: a² − x²  (c_val > 0, negative x²) ───────────────
    // ∫ √(a²−x²) dx = ½(x·√(a²−x²) + a²·asin(x/a))
    if has_neg_x2 && c_val.is_positive() {
        let x_sqrt = arena.mul(&[var, sqrt_base]);
        let asin_term = arena.asin(x_over_a);
        let a_sq_asin = if a_sq_is_one {
            asin_term
        } else {
            arena.mul(&[a_sq_expr, asin_term])
        };
        let sum = arena.add(&[x_sqrt, a_sq_asin]);
        return Some(arena.mul(&[half, sum]));
    }

    // ── Pattern: x² + a²  (c_val > 0, positive x²) ───────────────
    // ∫ √(x²+a²) dx = ½(x·√(x²+a²) + a²·asinh(x/a))
    if has_pos_x2 && c_val.is_positive() {
        let x_sqrt = arena.mul(&[var, sqrt_base]);
        let asinh_term = arena.asinh(x_over_a);
        let a_sq_asinh = if a_sq_is_one {
            asinh_term
        } else {
            arena.mul(&[a_sq_expr, asinh_term])
        };
        let sum = arena.add(&[x_sqrt, a_sq_asinh]);
        return Some(arena.mul(&[half, sum]));
    }

    // ── Pattern: x² − a²  (c_val < 0, positive x²) ───────────────
    // ∫ √(x²−a²) dx = ½(x·√(x²−a²) − a²·ln|x + √(x²−a²)|), valid for
    // x > a and x < −a (the acosh(x/a) form is wrong for x < −a).
    if has_pos_x2 && c_val.is_negative() {
        let x_sqrt = arena.mul(&[var, sqrt_base]);
        let ln_term = ln_abs_x_plus_sqrt(arena, var, base);
        let a_sq_ln = if a_sq_is_one {
            ln_term
        } else {
            arena.mul(&[a_sq_expr, ln_term])
        };
        let diff = arena.sub(x_sqrt, a_sq_ln);
        return Some(arena.mul(&[half, diff]));
    }

    None
}

/// Integrate a product of a linear polynomial times `(quadratic)^{-1}`:
///
///   `∫ (ax+b) / (cx²+dx+e) dx`
///
/// Decomposes the linear numerator as a multiple of the derivative of the
/// quadratic denominator plus a constant remainder:
///
///   `ax+b = (a/(2c))·(2cx+d) + (b − ad/(2c))`
///
/// Then:
///   - First part:  `(a/(2c)) · ln|cx²+dx+e|`
///   - Second part: `(b − ad/(2c)) · ∫ 1/(cx²+dx+e) dx`  (completing the square)
pub(super) fn try_linear_over_quadratic(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    if dependent.len() != 2 {
        return None;
    }

    // Identify linear factor and Pow(quadratic, -1) factor.
    let (linear_idx, pow_idx) = {
        let mut li = None;
        let mut pi = None;
        for (i, &d) in dependent.iter().enumerate() {
            if let ExprNode::Pow(_, _) = arena.node(d) {
                if pi.is_none() {
                    pi = Some(i);
                }
            } else if li.is_none() {
                li = Some(i);
            }
        }
        (li?, pi?)
    };

    let linear = dependent[linear_idx];
    let (pow_base, pow_exp) = match arena.node(dependent[pow_idx]).clone() {
        ExprNode::Pow(b, e) => (b, e),
        _ => return None,
    };

    // Exponent must be exactly −1.
    let exp_val = arena.as_num(pow_exp)?.clone();
    let neg_one = num_rational::Ratio::<num_bigint::BigInt>::from_integer((-1).into());
    if exp_val != neg_one {
        return None;
    }

    // ── Try rational-coefficient path first ────────────────────────
    let lin_poly = crate::poly::polybridge::expr_to_poly(arena, linear, var);
    let quad_poly = crate::poly::polybridge::expr_to_poly(arena, pow_base, var);

    if let (Some(lp), Some(qp)) = (&lin_poly, &quad_poly) {
        tracing::trace!("try_linear_over_quadratic: rational coefficient path");
        if lp.degree() == Some(1) && qp.degree() == Some(2) {
            let a_coeff = lp.coeff(1);
            let b_coeff = lp.coeff(0);
            let c_coeff = qp.coeff(2);
            let d_coeff = qp.coeff(1);

            use num_traits::Zero;
            if !c_coeff.is_zero() && !a_coeff.is_zero() {
                // Decompose: ax+b = (a/(2c))·(2cx+d) + (b − ad/(2c))
                let two_r = num_rational::Ratio::<num_bigint::BigInt>::from_integer(2.into());
                let a_over_2c = &a_coeff / &(&c_coeff * &two_r);
                let remainder = &b_coeff - &(&a_coeff * &d_coeff / &(&c_coeff * &two_r));

                let mut terms: Vec<ExprId> = Vec::new();

                // First term: (a/(2c)) · ln|cx²+dx+e|
                if !a_over_2c.is_zero() {
                    let coeff_id = arena.num_ratio(a_over_2c.clone());
                    let abs_quad = arena.abs(pow_base);
                    let ln_quad = arena.ln(abs_quad);
                    terms.push(arena.mul(&[coeff_id, ln_quad]));
                }

                // Second term: remainder · ∫ 1/(cx²+dx+e) dx
                if !remainder.is_zero() {
                    let inv_quad = arena.pow(pow_base, pow_exp); // (quad)^{-1}
                    let inv_integral =
                        integrate_node(arena, inv_quad, var, var_sym, depth.saturating_sub(1));
                    if matches!(arena.node(inv_integral), ExprNode::Integral(_, _)) {
                        return None;
                    }
                    let rem_id = arena.num_ratio(remainder.clone());
                    terms.push(arena.mul(&[rem_id, inv_integral]));
                }

                return match terms.len() {
                    0 => Some(arena.zero),
                    1 => Some(terms[0]),
                    _ => Some(arena.add(&terms)),
                };
            }
        }

        // Rational polys exist but don't match expected shape.
        return None;
    }

    // ── Symbolic fallback for irrational coefficients ──────────────
    // When expr_to_poly fails (e.g., coefficients contain √5 or √2),
    // extract symbolic coefficients and decompose using arena arithmetic.
    tracing::debug!(
        "try_linear_over_quadratic: expr_to_poly failed, trying symbolic coefficient path"
    );
    let (a_id, b_id) = symbolic_linear_coeff_of(arena, linear, var, var_sym)?;
    let (c_id, d_id, _e_id) = symbolic_quadratic_coeffs(arena, pow_base, var, var_sym)?;
    tracing::debug!(
        "try_linear_over_quadratic: symbolic coefficients extracted for linear/quadratic decomposition"
    );

    // log_coeff = a / (2c)
    let two = arena.int(2);
    let two_c = arena.mul(&[two, c_id]);
    let log_coeff = arena.div(a_id, two_c);
    let log_coeff = crate::transforms::eval::eval(arena, log_coeff);

    // remainder = b − a·d/(2c)
    let a_d = arena.mul(&[a_id, d_id]);
    let a_d_over_2c = arena.div(a_d, two_c);
    let remainder_expr = arena.sub(b_id, a_d_over_2c);
    let remainder_expr = crate::transforms::eval::eval(arena, remainder_expr);

    let mut terms: Vec<ExprId> = Vec::new();

    // A term is dropped only when its coefficient is *proved* to vanish:
    // structurally zero, or a constant shown 0 exactly
    // (`constant_zero_decision`: `√5·√5/5 − 1`).  Up to 0.50 a constant whose
    // `f64` value was below `10⁻¹⁴` counted as 0, so a coefficient `e⁻⁴⁰/2`
    // was dropped; keeping a coefficient that is 0 but not proved is
    // harmless.  A coefficient with a free parameter has no value, and
    // treating that as "zero" returned `∫ (a·x + b)/(x² + 1) dx = 0` — 56
    // wrong answers on the Rubi suite (0.24).
    let known_zero = |arena: &mut Arena, c: ExprId| {
        arena.is_zero_structural(c)
            || Truth::from(crate::transforms::eval::constant_zero_decision(arena, c)).is_true()
    };

    // First term: log_coeff · ln|quadratic|
    if !known_zero(arena, log_coeff) {
        let abs_quad = arena.abs(pow_base);
        let ln_quad = arena.ln(abs_quad);
        terms.push(arena.mul(&[log_coeff, ln_quad]));
    }

    // Second term: remainder · ∫ 1/(cx²+dx+e) dx
    if !known_zero(arena, remainder_expr) {
        let inv_quad = arena.pow(pow_base, pow_exp); // (quad)^{-1}
        let inv_integral = integrate_node(arena, inv_quad, var, var_sym, depth.saturating_sub(1));
        if crate::base::walk::has_unevaluated(arena, inv_integral) {
            return None;
        }
        terms.push(arena.mul(&[remainder_expr, inv_integral]));
    }

    match terms.len() {
        0 => Some(arena.zero),
        1 => Some(terms[0]),
        _ => Some(arena.add(&terms)),
    }
}
