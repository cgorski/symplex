//! Symbolic integration (antiderivatives).
//!
//! This module implements [`integrate`], which computes the indefinite
//! integral of an expression with respect to a symbol.
//!
//! # Supported integrands
//!
//! - **Power rule:** `∫ x^n dx = x^(n+1)/(n+1)` for n ≠ -1
//! - **Logarithmic:** `∫ x^(-1) dx = ln(|x|)`
//! - **Trigonometric:** `∫ sin(x) dx = -cos(x)`, `∫ cos(x) dx = sin(x)`
//! - **Tangent:** `∫ tan(x) dx = -ln|cos(x)|`
//! - **Natural log:** `∫ ln(x) dx = x·ln(x) - x`
//! - **Exponential:** `∫ exp(x) dx = exp(x)`
//! - **Linearity:** `∫ (f + g) dx = ∫f dx + ∫g dx`
//! - **Constant factor:** `∫ c·f dx = c · ∫f dx` (when c is independent of x)
//! - **Constants:** `∫ c dx = c·x`
//! - **Standard forms:** `∫ 1/(x²+a²) dx = (1/a)·atan(x/a)`, etc.
//! - **Partial fractions:** apart→integrate pipeline for rational integrands
//!
//! For integrands that don't match any rule, an unevaluated
//! `Integral(body, var)` node is returned.
//!
//! # Design
//!
//! Integration is computed bottom-up using an iterative post-order
//! traversal, mirroring the design of `diff.rs`. Results are
//! constructed through canonical arena constructors to preserve
//! invariants.

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use num_traits::One;
use num_traits::Signed;
use num_traits::Zero;

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode, SymbolId};

/// Integrate `expr` with respect to `var`.
///
/// Returns the antiderivative. If integration cannot be performed,
/// returns an unevaluated `Integral(expr, var)` node.
pub(crate) fn integrate(arena: &mut Arena, expr: ExprId, var: ExprId) -> ExprId {
    let var_sym = match arena.node(var) {
        ExprNode::Symbol(sid) => *sid,
        _ => {
            // var is not a symbol — can't integrate w.r.t. a non-symbol.
            return arena.intern(ExprNode::Integral(expr, var));
        }
    };

    let result = integrate_node(arena, expr, var, var_sym, 20);

    // If the rule-based integrator returned an unevaluated Integral node,
    // try the heuristic Risch integrator as a fallback.
    let result = if let ExprNode::Integral(_, _) = arena.node(result)
        && let Some(heurisch_result) =
            crate::transforms::heurisch::heurisch_integrate(arena, expr, var, var_sym)
    {
        heurisch_result
    } else {
        result
    };

    // Piecewise wrapping for parametric degenerate cases
    try_piecewise_wrap(arena, result, expr, var, var_sym)
}

/// Check whether `expr` is a suitable candidate for the `u` factor in
/// integration by parts.  Returns `true` when `expr` is a polynomial
/// in `var` **or** when it is `ln(inner)` with `inner` depending on `var`.
fn is_by_parts_candidate(arena: &Arena, expr: ExprId, var: ExprId, var_sym: SymbolId) -> bool {
    if is_polynomial_in(arena, expr, var, var_sym) {
        return true;
    }
    match arena.node(expr) {
        ExprNode::Ln(inner)
        | ExprNode::Asin(inner)
        | ExprNode::Acos(inner)
        | ExprNode::Atan(inner) => contains_var(arena, *inner, var_sym),
        _ => false,
    }
}

/// LIATE priority for integration by parts: lower = better choice for u.
/// L(og) = 1, I(nverse trig) = 2, A(lgebraic/polynomial) = 3,
/// T(rig) = 4, E(xponential) = 5, other = 6.
fn liate_rank(arena: &Arena, expr: ExprId, var: ExprId, var_sym: SymbolId) -> u8 {
    match arena.node(expr) {
        ExprNode::Ln(_) => 1,
        ExprNode::Asin(_) | ExprNode::Acos(_) | ExprNode::Atan(_) => 2,
        _ if is_polynomial_in(arena, expr, var, var_sym) => 3,
        ExprNode::Sin(_) | ExprNode::Cos(_) | ExprNode::Tan(_) => 4,
        ExprNode::Exp(_) | ExprNode::Sinh(_) | ExprNode::Cosh(_) | ExprNode::Tanh(_) => 5,
        _ => 6,
    }
}

/// Check whether `expr` is `Pow(var, 2)`, i.e. `x²`.
fn is_var_squared(arena: &Arena, expr: ExprId, var: ExprId) -> bool {
    if let ExprNode::Pow(b, e) = arena.node(expr)
        && *b == var
        && let Some(n) = arena.as_num(*e)
    {
        return *n == num_rational::Ratio::from_integer(2.into());
    }
    false
}

/// Check whether `expr` represents `-x²` in canonical form.
///
/// Handles both `Neg(Pow(var, 2))` and `Mul([-1, Pow(var, 2)])`.
fn is_neg_var_squared(arena: &Arena, expr: ExprId, var: ExprId) -> bool {
    match arena.node(expr).clone() {
        ExprNode::Neg(inner) => is_var_squared(arena, inner, var),
        ExprNode::Mul(ref children) => {
            if children.len() == 2 {
                let neg_one_val =
                    num_rational::Ratio::<num_bigint::BigInt>::from_integer((-1).into());
                let has_neg_one = children
                    .iter()
                    .any(|&c| arena.as_num(c).is_some_and(|n| *n == neg_one_val));
                let has_var_sq = children.iter().any(|&c| is_var_squared(arena, c, var));
                has_neg_one && has_var_sq
            } else {
                false
            }
        }
        _ => false,
    }
}

/// Try to recognise standard‑form integrals of the shape
/// `(x² ± a²)^n` where `n` is −1 or −1/2.
///
/// Returns `Some(antiderivative)` on success.
fn try_standard_form_integral(
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
    let mut const_val: Option<num_rational::Ratio<num_bigint::BigInt>> = None;
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
        let a_sq_id = rational_to_expr(arena, &a_squared);
        let nh = arena.rational(-1, 2);
        let a_inv = arena.pow(a_sq_id, nh); // (a²)^{-1/2} = 1/a
        arena.mul(&[var, a_inv])
    };

    // 1/a   (only needed for exp == -1 forms)
    let one_over_a = if a_sq_is_one {
        None
    } else {
        let a_sq_id = rational_to_expr(arena, &a_squared);
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
        // A6: ∫ (x²−a²)^{-1/2} dx = acosh(x/a)
        let acosh_val = arena.acosh(x_over_a);
        return Some(acosh_val);
    }

    None
}

/// Try integrating `(ax²+bx+c)^exp` via completing the square.
///
/// Handles two exponent values:
/// - `exp = -1`:   `∫ 1/(ax²+bx+c) dx` → atan form
/// - `exp = -1/2`: `∫ 1/√(ax²+bx+c) dx` → asinh / acosh / asin form
fn try_complete_square_integral(
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
    let poly = crate::poly::polybridge::expr_to_poly(arena, base, var)?;
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
        let shift_id = rational_to_expr(arena, &shift);
        arena.add(&[var, shift_id])
    };

    let half = arena.rational(1, 2);

    if a_coeff.is_positive() {
        // a > 0
        let a_id = rational_to_expr(arena, &a_coeff);
        let sqrt_a = arena.pow(a_id, half); // √a
        let inv_sqrt_a = {
            let neg_half_e = arena.rational(-1, 2);
            arena.pow(a_id, neg_half_e)
        }; // 1/√a

        if d.is_positive() {
            // a > 0, d > 0: (1/√a) · asinh(u·√a / √d)
            let d_id = rational_to_expr(arena, &d);
            let sqrt_d = arena.pow(d_id, half);
            let u_sqrt_a = arena.mul(&[u_expr, sqrt_a]);
            let arg = arena.div(u_sqrt_a, sqrt_d);
            let asinh_val = arena.asinh(arg);
            return Some(arena.mul(&[inv_sqrt_a, asinh_val]));
        } else if d.is_negative() {
            // a > 0, d < 0: (1/√a) · acosh(u·√a / √|d|)
            let abs_d = d.abs();
            let abs_d_id = rational_to_expr(arena, &abs_d);
            let sqrt_abs_d = arena.pow(abs_d_id, half);
            let u_sqrt_a = arena.mul(&[u_expr, sqrt_a]);
            let arg = arena.div(u_sqrt_a, sqrt_abs_d);
            let acosh_val = arena.acosh(arg);
            return Some(arena.mul(&[inv_sqrt_a, acosh_val]));
        } else {
            // a > 0, d = 0: (1/√a) · ln|u|
            let abs_u = arena.abs(u_expr);
            let ln_u = arena.ln(abs_u);
            return Some(arena.mul(&[inv_sqrt_a, ln_u]));
        }
    } else if a_coeff.is_negative() && d.is_positive() {
        // a < 0, d > 0: (1/√|a|) · asin(u·√|a| / √d)
        let abs_a = a_coeff.abs();
        let abs_a_id = rational_to_expr(arena, &abs_a);
        let sqrt_abs_a = arena.pow(abs_a_id, half);
        let inv_sqrt_abs_a = {
            let neg_half_e = arena.rational(-1, 2);
            arena.pow(abs_a_id, neg_half_e)
        };
        let d_id = rational_to_expr(arena, &d);
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
fn try_trig_recip_product(
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
                            && *e == neg_two {
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
                            && *e == neg_two {
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
fn try_x_over_sqrt_quadratic(
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
    let a_id = rational_to_expr(arena, &a_coeff);
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
    let b_over_2a_id = rational_to_expr(arena, &b_over_2a);

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
fn try_trig_sub_sqrt_integral(
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
    let mut const_val: Option<num_rational::Ratio<num_bigint::BigInt>> = None;
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
        rational_to_expr(arena, &a_squared)
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
    // ∫ √(x²−a²) dx = ½(x·√(x²−a²) − a²·acosh(x/a))
    if has_pos_x2 && c_val.is_negative() {
        let x_sqrt = arena.mul(&[var, sqrt_base]);
        let acosh_term = arena.acosh(x_over_a);
        let a_sq_acosh = if a_sq_is_one {
            acosh_term
        } else {
            arena.mul(&[a_sq_expr, acosh_term])
        };
        let diff = arena.sub(x_sqrt, a_sq_acosh);
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
fn try_linear_over_quadratic(
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

    // linear must be degree 1 in var.
    let lin_poly = crate::poly::polybridge::expr_to_poly(arena, linear, var)?;
    if lin_poly.degree()? != 1 {
        return None;
    }
    let a_coeff = lin_poly.coeff(1);
    let b_coeff = lin_poly.coeff(0);

    // pow_base must be degree 2 in var.
    let quad_poly = crate::poly::polybridge::expr_to_poly(arena, pow_base, var)?;
    if quad_poly.degree()? != 2 {
        return None;
    }
    let c_coeff = quad_poly.coeff(2);
    let d_coeff = quad_poly.coeff(1);

    use num_traits::Zero;
    if c_coeff.is_zero() || a_coeff.is_zero() {
        return None;
    }

    // Decompose: ax+b = (a/(2c))·(2cx+d) + (b − ad/(2c))
    let two_r = num_rational::Ratio::<num_bigint::BigInt>::from_integer(2.into());
    let a_over_2c = &a_coeff / &(&c_coeff * &two_r);
    let remainder = &b_coeff - &(&a_coeff * &d_coeff / &(&c_coeff * &two_r));

    let mut terms: Vec<ExprId> = Vec::new();

    // First term: (a/(2c)) · ln|cx²+dx+e|
    if !a_over_2c.is_zero() {
        let coeff_id = rational_to_expr(arena, &a_over_2c);
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
        let rem_id = rational_to_expr(arena, &remainder);
        terms.push(arena.mul(&[rem_id, inv_integral]));
    }

    match terms.len() {
        0 => Some(arena.zero),
        1 => Some(terms[0]),
        _ => Some(arena.add(&terms)),
    }
}

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
fn try_weierstrass_substitution(
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

    tracing::debug!("trying Weierstrass substitution");

    let t = arena.symbol("__wt");
    let t_sym = match arena.node(t) {
        ExprNode::Symbol(sid) => *sid,
        _ => unreachable!(),
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

    // Integrate w.r.t. t
    let integral_t = integrate_node(arena, integrand_t, t, t_sym, depth.saturating_sub(2));

    if matches!(arena.node(integral_t), ExprNode::Integral(_, _)) {
        return None;
    }

    // Substitute back: t → tan(var/2)
    let half = arena.rational(1, 2);
    let half_var = arena.mul(&[half, var]);
    let tan_half = arena.tan(half_var);
    let result = arena.subs_structural(integral_t, t, tan_half);

    Some(result)
}

/// Attempt cyclic integration by parts for integrals like `∫ exp(x)·sin(x) dx`.
///
/// After two IBP rounds (with u₂ = du₁, dv₂ = v₁), if the remaining
/// integral is a constant multiple `c` of the original integrand we
/// solve algebraically:
///
/// ```text
///   I = boundary₁ − (boundary₂ − c·I)
///   I(1 − c) = boundary₁ − boundary₂
///   I = (boundary₁ − boundary₂) / (1 − c)
/// ```
fn try_cyclic_ibp(
    arena: &mut Arena,
    factors: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    if factors.len() != 2 || depth < 2 {
        return None;
    }

    // Use LIATE ordering: lower rank = u (trig before exp)
    let (u_idx, dv_idx) = {
        let r0 = liate_rank(arena, factors[0], var, var_sym);
        let r1 = liate_rank(arena, factors[1], var, var_sym);
        if r0 <= r1 { (0, 1) } else { (1, 0) }
    };
    let u1 = factors[u_idx];
    let dv1 = factors[dv_idx];

    // Both must depend on var
    if !contains_var(arena, u1, var_sym) || !contains_var(arena, dv1, var_sym) {
        return None;
    }

    // ── Round 1: ∫ u1·dv1 dx = u1·v1 − ∫ v1·du1 dx ──────────────
    let v1 = integrate_node(arena, dv1, var, var_sym, depth - 1);
    if matches!(arena.node(v1), ExprNode::Integral(_, _)) {
        return None;
    }
    let du1 = crate::transforms::diff::diff(arena, u1, var);
    let boundary1 = arena.mul(&[u1, v1]); // u1·v1

    // ── Round 2: ∫ v1·du1 dx  with u₂ = du1, dv₂ = v1 ───────────
    let v2 = integrate_node(arena, v1, var, var_sym, depth - 1);
    if matches!(arena.node(v2), ExprNode::Integral(_, _)) {
        return None;
    }
    let du2 = crate::transforms::diff::diff(arena, du1, var); // u1''
    let boundary2 = arena.mul(&[du1, v2]); // du1·v2

    // remaining₂ body = v2 · du2
    let remaining2 = arena.mul(&[v2, du2]);
    let original = arena.mul(&[factors[0], factors[1]]);

    // ── Check remaining₂ = c · original for some constant c ──────

    // Fast path: c = −1 (covers exp·sin, exp·cos and similar)
    let sum = arena.add(&[remaining2, original]);
    if sum == arena.zero {
        tracing::debug!("cyclic IBP detected (c = -1)");
        let numerator = arena.sub(boundary1, boundary2);
        let two = arena.int(2);
        return Some(arena.div(numerator, two));
    }

    // Fast path: c = +1 would be degenerate (1−c = 0), skip.
    let diff_check = arena.sub(remaining2, original);
    if diff_check == arena.zero {
        return None;
    }

    // General path: try polynomial cancellation on the ratio.
    let ratio = arena.div(remaining2, original);
    let cancelled = arena.cancel_expr(ratio, var);
    if !contains_var(arena, cancelled, var_sym) && cancelled != arena.one {
        tracing::debug!("cyclic IBP detected (general c)");
        let numerator = arena.sub(boundary1, boundary2);
        let one = arena.one;
        let one_minus_c = arena.sub(one, cancelled);
        return Some(arena.div(numerator, one_minus_c));
    }

    None
}

/// Integrate a single node with respect to `var`.
fn integrate_node(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> ExprId {
    tracing::trace!(depth = depth, "integrate_node entered");

    if depth == 0 {
        return arena.intern(ExprNode::Integral(expr, var));
    }

    // Try trig power/product integration first (sin^n, cos^n, sin^m*cos^n)
    if let Some(result) = crate::transforms::trig_integ::try_trig_power_integral(arena, expr, var, var_sym) {
        // Only use the trig-power result when it is fully evaluated;
        // negative-exponent cases (e.g. sin·cos^{-2}) come back as
        // unevaluated Integral nodes — fall through so the Mul handler
        // can try sec·tan / csc·cot patterns and u-substitution.
        if !matches!(arena.node(result), ExprNode::Integral(_, _)) {
            return result;
        }
    }

    // ── Trig identity rewrites ────────────────────────────────────
    // Rewrite squared trig identities to forms with known antiderivatives.
    if let ExprNode::Pow(trig_base, trig_exp) = arena.node(expr).clone()
        && let Some(n_val) = arena.as_num(trig_exp)
    {
        let two_r = num_rational::Ratio::<num_bigint::BigInt>::from_integer(2.into());
        if *n_val == two_r {
            // tan²(g) → sec²(g) − 1 = cos(g)^{-2} − 1
            if let ExprNode::Tan(inner) = arena.node(trig_base).clone() {
                let cos_inner = arena.cos(inner);
                let neg_two = arena.int(-2);
                let sec_sq = arena.pow(cos_inner, neg_two);
                let rewritten = arena.sub(sec_sq, arena.one);
                return integrate_node(arena, rewritten, var, var_sym, depth - 1);
            }
            // tanh²(g) → 1 − sech²(g) = 1 − cosh(g)^{-2}
            if let ExprNode::Tanh(inner) = arena.node(trig_base).clone() {
                let cosh_inner = arena.cosh(inner);
                let neg_two = arena.int(-2);
                let sech_sq = arena.pow(cosh_inner, neg_two);
                let rewritten = arena.sub(arena.one, sech_sq);
                return integrate_node(arena, rewritten, var, var_sym, depth - 1);
            }
        }
    }

    // ── Type dispatch: rational function detection ────────────────
    // Before dispatching on node type, check if the expression is a
    // rational function P(x)/Q(x).  If so, route to the complete
    // Hermite + Rothstein-Trager algorithm.  This handles ALL structural
    // variants (Mul with negative powers, Pow with negative exponent,
    // etc.) because as_numer_denom normalizes them all to (numer, denom).
    //
    // This is the standard CAS architecture: rational function integration
    // is a solved problem with efficient algorithms, and it should run
    // before any heuristic pattern matching.
    if let Some(result) = crate::calculus::risch::try_risch_rational(arena, expr, var) {
        if !matches!(arena.node(result), ExprNode::Integral(_, _)) {
            return result;
        }
    }

    let node = arena.node(expr).clone();

    match node {
        // ── Constants (independent of var) → c * var ───────────────
        ExprNode::Num(_)
        | ExprNode::Pi
        | ExprNode::E
        | ExprNode::ImaginaryUnit
        | ExprNode::Infinity
        | ExprNode::NegInfinity
        | ExprNode::ComplexInfinity
        | ExprNode::NaN => {
            // ∫ c dx = c * x
            arena.mul(&[expr, var])
        }

        ExprNode::Symbol(sid) => {
            if sid == var_sym {
                // ∫ x dx = x^2 / 2
                let two = arena.int(2);
                let x_sq = arena.pow(var, two);
                let half = arena.rational(1, 2);
                arena.mul(&[half, x_sq])
            } else {
                // ∫ c dx = c * x (c is independent of var)
                arena.mul(&[expr, var])
            }
        }

        // ── Add: linearity ─────────────────────────────────────────
        ExprNode::Add(ref children) => {
            let integrals: SmallVec<[ExprId; 6]> = children
                .iter()
                .map(|&child| integrate_node(arena, child, var, var_sym, depth - 1))
                .collect();
            arena.add(&integrals)
        }

        // ── Mul: factor out constants ──────────────────────────────
        ExprNode::Mul(ref children) => {
            // Separate constant factors (independent of var) from the rest.
            let mut constants: SmallVec<[ExprId; 4]> = SmallVec::new();
            let mut dependent: SmallVec<[ExprId; 4]> = SmallVec::new();

            for &child in children {
                if contains_var(arena, child, var_sym) {
                    dependent.push(child);
                } else {
                    constants.push(child);
                }
            }

            // Normalize dependent factors: flatten Pow(Pow(a, m), n) → Pow(a, m·n)
            // for rational exponents.  The canon layer only flattens when both
            // exponents are integers (branch-cut safety), but for integration we
            // need e.g. Pow(Pow(x²+1, 1/2), -1) → Pow(x²+1, -1/2).
            for d in dependent.iter_mut() {
                if let ExprNode::Pow(pow_base, pow_exp) = arena.node(*d).clone()
                    && let ExprNode::Pow(inner_base, inner_exp) = arena.node(pow_base).clone()
                        && let (Some(m), Some(n)) = (arena.as_num(inner_exp), arena.as_num(pow_exp))
                        {
                            let combined = m.clone() * n.clone();
                            let combined_id = rational_to_expr(arena, &combined);
                            let flattened = arena.pow(inner_base, combined_id);
                            *d = flattened;
                        }
            }

            if dependent.is_empty() {
                // All constant: ∫ c dx = c * x
                return arena.mul(&[expr, var]);
            }

            if !constants.is_empty() && dependent.len() == 1 {
                // c * f(x) → c * ∫ f(x) dx
                let inner_integral = integrate_node(arena, dependent[0], var, var_sym, depth - 1);
                // Check if the inner integral is unevaluated
                if let ExprNode::Integral(_, _) = arena.node(inner_integral) {
                    // Can't integrate the inner part — return unevaluated for whole
                    return arena.intern(ExprNode::Integral(expr, var));
                }
                constants.push(inner_integral);
                return arena.mul(&constants);
            }

            // ── sec(x)·tan(x) and csc(x)·cot(x) forms ──────────────
            if dependent.len() == 2
                && let Some(result) = try_trig_recip_product(arena, &dependent, var, var_sym) {
                    if constants.is_empty() {
                        return result;
                    } else {
                        let mut all = constants.clone();
                        all.push(result);
                        return arena.mul(&all);
                    }
                }

            // ── x/√(ax²+bx+c) form ──────────────────────────────────
            if dependent.len() == 2
                && let Some(result) =
                    try_x_over_sqrt_quadratic(arena, &dependent, var, var_sym, depth)
                {
                    if constants.is_empty() {
                        return result;
                    } else {
                        let mut all = constants.clone();
                        all.push(result);
                        return arena.mul(&all);
                    }
                }

            // ── (linear) / (irreducible quadratic) ───────────────────
            if dependent.len() == 2
                && let Some(result) =
                    try_linear_over_quadratic(arena, &dependent, var, var_sym, depth)
                {
                    if constants.is_empty() {
                        return result;
                    } else {
                        let mut all = constants.clone();
                        all.push(result);
                        return arena.mul(&all);
                    }
                }

            // ── DiracDelta sifting property: ∫ f(x)·δ(g(x)) dx = f(root)·H(g(x)) ──
            // Check if any factor in the product is a DiracDelta.
            {
                let all_children: SmallVec<[ExprId; 6]> = children.clone();
                for (i, &child) in all_children.iter().enumerate() {
                    if let ExprNode::DiracDelta(delta_arg) = arena.node(child).clone() {
                        tracing::debug!(
                            "integrate: detected DiracDelta factor in Mul, attempting sifting property"
                        );

                        // Collect the remaining factors as f(x)
                        let other_factors: SmallVec<[ExprId; 4]> = all_children
                            .iter()
                            .enumerate()
                            .filter(|&(j, _)| j != i)
                            .map(|(_, &c)| c)
                            .collect();
                        let f_expr = if other_factors.len() == 1 {
                            other_factors[0]
                        } else if other_factors.is_empty() {
                            arena.one
                        } else {
                            arena.mul(&other_factors)
                        };

                        // Simple case: δ(x) → root = 0
                        if delta_arg == var {
                            let f_at_0 = crate::transforms::subs::subs(arena, f_expr, var, arena.zero);
                            let f_at_0_eval = crate::transforms::eval::eval(arena, f_at_0);
                            let heaviside = arena.intern(ExprNode::Heaviside(var));
                            return arena.mul(&[f_at_0_eval, heaviside]);
                        }

                        // General case: solve δ(g(x)) = 0, i.e. g(x) = 0 for x
                        let solutions = crate::transforms::solve::solve(arena, delta_arg, var);
                        if solutions.len() == 1 {
                            let root = solutions[0].value;
                            let f_at_root = crate::transforms::subs::subs(arena, f_expr, var, root);
                            let f_at_root_eval = crate::transforms::eval::eval(arena, f_at_root);
                            let heaviside = arena.intern(ExprNode::Heaviside(delta_arg));
                            return arena.mul(&[f_at_root_eval, heaviside]);
                        }

                        // If we can't solve, fall through to other strategies
                        break;
                    }
                }
            }

            // ── Integration by parts: ∫ u·dv = u·v - ∫ v·du ───────────
            // Try when there are exactly 2 dependent factors:
            // one that's a by-parts candidate (u), and one that's directly
            // integrable (dv).
            if dependent.len() == 2 {
                // Try LIATE-preferred ordering: factor with lower LIATE rank as u first.
                let orderings = {
                    let r0 = liate_rank(arena, dependent[0], var, var_sym);
                    let r1 = liate_rank(arena, dependent[1], var, var_sym);
                    tracing::debug!(u_rank = r0, dv_rank = r1, "by-parts LIATE ordering");
                    if r0 <= r1 {
                        [(0usize, 1usize), (1, 0)]
                    } else {
                        [(1, 0), (0, 1)]
                    }
                };
                for (u_idx, dv_idx) in orderings {
                    let u = dependent[u_idx];
                    let dv = dependent[dv_idx];

                    // Check that u is a by-parts candidate (polynomial or ln)
                    if !is_by_parts_candidate(arena, u, var, var_sym) {
                        continue;
                    }

                    // Check that dv is directly integrable
                    let v = integrate_node(arena, dv, var, var_sym, depth - 1);
                    if let ExprNode::Integral(_, _) = arena.node(v) {
                        continue; // dv not integrable
                    }

                    // Compute du = d(u)/dx
                    let du = crate::transforms::diff::diff(arena, u, var);

                    // Compute ∫ v·du dx
                    let v_du = arena.mul(&[v, du]);
                    let integral_v_du = integrate_node(arena, v_du, var, var_sym, depth - 1);

                    // Check if the remaining integral was resolved
                    if let ExprNode::Integral(_, _) = arena.node(integral_v_du) {
                        continue; // Remaining integral not solvable
                    }

                    // Success: ∫ u·dv = u·v - ∫ v·du
                    tracing::debug!("integration by parts succeeded");
                    let u_v = arena.mul(&[u, v]);
                    let result = arena.sub(u_v, integral_v_du);

                    // Re-include constant factors if any
                    if constants.is_empty() {
                        return result;
                    } else {
                        let mut all = constants.clone();
                        all.push(result);
                        return arena.mul(&all);
                    }
                }
            }

            // ── Cyclic IBP: ∫ exp·sin, ∫ exp·cos, etc. ────────────
            if dependent.len() == 2
                && let Some(result) = try_cyclic_ibp(arena, &dependent, var, var_sym, depth)
            {
                if constants.is_empty() {
                    return result;
                } else {
                    let mut all = constants.clone();
                    all.push(result);
                    return arena.mul(&all);
                }
            }

            // ── Try partial fraction decomposition for rational integrands ──
            {
                let (_numer, denom) = crate::poly::polybridge::as_numer_denom(arena, expr);
                if denom != arena.one {
                    let decomposed = crate::transforms::apart::apart(arena, expr, var);
                    if decomposed != expr {
                        let result = integrate_node(arena, decomposed, var, var_sym, depth - 1);
                        if !matches!(arena.node(result), ExprNode::Integral(_, _)) {
                            return result;
                        }
                    }
                }
            }

            // ── Try general u-substitution ──────────────────────────────
            if let Some(result) = try_u_substitution(arena, &dependent, var, var_sym, depth - 1) {
                if constants.is_empty() {
                    return result;
                } else {
                    let mut all = constants.clone();
                    all.push(result);
                    return arena.mul(&all);
                }
            }

            // ── Weierstrass substitution for rational trig functions ──
            if let Some(result) =
                try_weierstrass_substitution(arena, expr, var, var_sym, depth)
            {
                if constants.is_empty() {
                    return result;
                } else {
                    let mut all = constants.clone();
                    all.push(result);
                    return arena.mul(&all);
                }
            }

            // ── Special function integration table ──────────────────
            if let Some(result) = try_special_function_integral(arena, &dependent, &constants, var, var_sym) {
                return result;
            }

            // General product of var-dependent terms — can't integrate without
            // further techniques.
            tracing::debug!("integration: no strategy succeeded, returning unevaluated");
            arena.intern(ExprNode::Integral(expr, var))
        }

        // ── Neg ────────────────────────────────────────────────────
        ExprNode::Neg(inner) => {
            let inner_int = integrate_node(arena, inner, var, var_sym, depth - 1);
            arena.neg(inner_int)
        }

        // ── Pow: power rule ────────────────────────────────────────
        ExprNode::Pow(base, exp) => {
            // Flatten Pow(Pow(a, m), n) → Pow(a, m·n) when both m and n
            // are rational.  The canon layer only does this for integer
            // exponents (to avoid complex branch-cut issues), but for
            // real-valued integration it is safe and necessary so that
            // e.g.  1/√(x²+1) = Pow(Pow(x²+1, 1/2), -1) becomes
            // Pow(x²+1, -1/2) and hits the standard-form / completing-
            // the-square handlers.
            if let ExprNode::Pow(inner_base, inner_exp) = arena.node(base).clone()
                && let (Some(m), Some(n)) = (arena.as_num(inner_exp), arena.as_num(exp)) {
                    let m = m.clone();
                    let n = n.clone();
                    let combined = &m * &n;
                    let combined_id = rational_to_expr(arena, &combined);
                    let flattened = arena.pow(inner_base, combined_id);
                    if flattened != expr {
                        return integrate_node(arena, flattened, var, var_sym, depth - 1);
                    }
                }

            let base_is_var = base == var;
            let exp_has_var = contains_var(arena, exp, var_sym);
            let base_has_var = contains_var(arena, base, var_sym);

            if base_is_var && !exp_has_var {
                // ∫ x^n dx
                if let Some(n) = arena.as_num(exp) {
                    let n = n.clone();
                    if n == num_rational::Ratio::from_integer((-1).into()) {
                        // ∫ x^(-1) dx = ln(|x|)
                        let abs_x = arena.abs(var);
                        return arena.ln(abs_x);
                    }
                    // ∫ x^n dx = x^(n+1) / (n+1) for n ≠ -1
                    let one = num_rational::Ratio::<num_bigint::BigInt>::one();
                    let n_plus_1 = &n + &one;
                    let n_plus_1_id = {
                        let nid = arena.intern_num(n_plus_1.clone());
                        arena.intern(ExprNode::Num(nid))
                    };
                    let x_pow = arena.pow(var, n_plus_1_id);
                    let recip = {
                        let inv = one / n_plus_1;
                        let nid = arena.intern_num(inv);
                        arena.intern(ExprNode::Num(nid))
                    };
                    return arena.mul(&[recip, x_pow]);
                } else {
                    // Symbolic exponent: ∫ x^n dx = x^(n+1)/(n+1)
                    let one_id = arena.one;
                    let n_plus_1 = arena.add(&[exp, one_id]);
                    let x_pow = arena.pow(var, n_plus_1);
                    return arena.div(x_pow, n_plus_1);
                }
            }

            if !base_has_var && !exp_has_var {
                // Constant: ∫ c dx = c * x
                return arena.mul(&[expr, var]);
            }

            // ── sech²(g) = cosh(g)^{-2} → tanh(g) / chain_coeff ──
            if let ExprNode::Cosh(inner) = arena.node(base).clone()
                && let Some(e_val) = arena.as_num(exp)
            {
                let neg_two_r = num_rational::Ratio::<num_bigint::BigInt>::from_integer((-2).into());
                if *e_val == neg_two_r {
                    if inner == var {
                        return arena.tanh(var);
                    }
                    if let Some((a_expr, _)) = symbolic_linear_coeff_of(arena, inner, var, var_sym) {
                        let tanh_inner = arena.tanh(inner);
                        return arena.div(tanh_inner, a_expr);
                    }
                }
            }

            // ── csch²(g) = sinh(g)^{-2} → −coth(g) / chain_coeff ──
            if let ExprNode::Sinh(inner) = arena.node(base).clone()
                && let Some(e_val) = arena.as_num(exp)
            {
                let neg_two_r = num_rational::Ratio::<num_bigint::BigInt>::from_integer((-2).into());
                if *e_val == neg_two_r {
                    if inner == var {
                        let cosh_v = arena.cosh(var);
                        let sinh_v = arena.sinh(var);
                        let neg1 = arena.int(-1);
                        let sinh_inv = arena.pow(sinh_v, neg1);
                        let coth_v = arena.mul(&[cosh_v, sinh_inv]);
                        return arena.neg(coth_v);
                    }
                    if let Some((a_expr, _)) = symbolic_linear_coeff_of(arena, inner, var, var_sym) {
                        let cosh_i = arena.cosh(inner);
                        let sinh_i = arena.sinh(inner);
                        let neg1 = arena.int(-1);
                        let sinh_inv = arena.pow(sinh_i, neg1);
                        let coth_i = arena.mul(&[cosh_i, sinh_inv]);
                        let neg_coth = arena.neg(coth_i);
                        return arena.div(neg_coth, a_expr);
                    }
                }
            }

            // ── ln(x)^n by parts: ∫ ln(x)^n dx = x·ln(x)^n − n·∫ ln(x)^(n−1) dx ──
            if let ExprNode::Ln(inner) = arena.node(base).clone()
                && inner == var && !exp_has_var
                && let Some(n_val) = arena.as_num(exp)
            {
                let n_val = n_val.clone();
                if n_val.is_integer() && n_val.is_positive() {
                    let n_i64: i64 = n_val.to_integer().try_into().unwrap_or(0);
                    if n_i64 >= 2 {
                        let x_ln_n = arena.mul(&[var, expr]);
                        let n_id = rational_to_expr(arena, &n_val);
                        let n_minus_1 = {
                            let v = &n_val - &num_rational::Ratio::<num_bigint::BigInt>::one();
                            rational_to_expr(arena, &v)
                        };
                        let ln_x = arena.ln(var);
                        let ln_nm1 = if n_i64 == 2 { ln_x } else { arena.pow(ln_x, n_minus_1) };
                        let sub_int = integrate_node(arena, ln_nm1, var, var_sym, depth - 1);
                        let n_times_sub = arena.mul(&[n_id, sub_int]);
                        return arena.sub(x_ln_n, n_times_sub);
                    }
                }
            }

            // General linear substitution: ∫ (ax+b)^n dx = (ax+b)^(n+1) / (a*(n+1))
            if !exp_has_var
                && base_has_var
                && let Some((a_expr, _b_expr)) = symbolic_linear_coeff_of(arena, base, var, var_sym)
                && let Some(n) = arena.as_num(exp)
            {
                let n = n.clone();
                let neg_one = num_rational::Ratio::from_integer((-1).into());
                if n != neg_one {
                    // ∫ (ax+b)^n dx = (ax+b)^(n+1) / (a*(n+1))
                    let one = num_rational::Ratio::<num_bigint::BigInt>::one();
                    let n_plus_1 = &n + &one;
                    let n_plus_1_id = rational_to_expr(arena, &n_plus_1);
                    let base_pow = arena.pow(base, n_plus_1_id);
                    let denom = arena.mul(&[a_expr, n_plus_1_id]);
                    return arena.div(base_pow, denom);
                } else {
                    // ∫ (ax+b)^(-1) dx = ln|ax+b| / a
                    let abs_base = arena.abs(base);
                    let ln_base = arena.ln(abs_base);
                    return arena.div(ln_base, a_expr);
                }
            }

            // ── Standard form integrals (A3–A7) ───────────────────────
            if base_has_var
                && !exp_has_var
                && let Some(result) =
                    try_standard_form_integral(arena, expr, base, exp, var, var_sym)
            {
                return result;
            }

            // ── Symbolic standard form: ∫ (x² + k)^{-1} dx ──────────
            // where k is free of var (handles e.g. ∫ 1/(x²+a²) dx)
            if base_has_var && !exp_has_var
                && let Some(exp_val) = arena.as_num(exp)
            {
                let neg_one_r = num_rational::Ratio::<num_bigint::BigInt>::from_integer((-1).into());
                if *exp_val == neg_one_r
                    && let ExprNode::Add(ref ac) = arena.node(base).clone()
                    && ac.len() == 2
                {
                    let (mut x2_found, mut k_id) = (false, None);
                    for &ch in ac.iter() {
                        if is_var_squared(arena, ch, var) {
                            x2_found = true;
                        } else if !contains_var(arena, ch, var_sym) {
                            k_id = Some(ch);
                        }
                    }
                    if x2_found
                        && let Some(k) = k_id
                        && arena.as_num(k).is_none()
                    {
                        // Only use symbolic path when k is NOT pure numeric
                        // (numeric case is handled by try_standard_form_integral)
                        // ∫ 1/(x²+k) dx = (1/√k)·atan(x/√k)
                        let half = arena.rational(1, 2);
                        let sqrt_k = arena.pow(k, half);
                        let x_over_sk = arena.div(var, sqrt_k);
                        let atan_val = arena.atan(x_over_sk);
                        let neg_half = arena.rational(-1, 2);
                        let inv_sk = arena.pow(k, neg_half);
                        return arena.mul(&[inv_sk, atan_val]);
                    }
                }
            }

            // ── Completing the square for 1/(ax²+bx+c) ───────────────
            if base_has_var
                && !exp_has_var
                && let Some(result) = try_complete_square_integral(arena, base, exp, var, var_sym)
            {
                return result;
            }

            // ── Trig substitution: √(a²±x²), √(x²±a²) ──────────────
            if base_has_var
                && !exp_has_var
                && let Some(result) =
                    try_trig_sub_sqrt_integral(arena, base, exp, var, var_sym)
            {
                return result;
            }

            // ── Try partial fraction decomposition ────────────────────
            {
                let (_numer, denom) = crate::poly::polybridge::as_numer_denom(arena, expr);
                if denom != arena.one {
                    let decomposed = crate::transforms::apart::apart(arena, expr, var);
                    if decomposed != expr {
                        let result = integrate_node(arena, decomposed, var, var_sym, depth - 1);
                        if !matches!(arena.node(result), ExprNode::Integral(_, _)) {
                            return result;
                        }
                    }
                }
            }

            // Fallback: if base is an Add and exp is a small positive integer, expand and retry
            if let ExprNode::Add(_) = arena.node(base)
                && let Some(n) = arena.as_num(exp)
                && n.is_integer()
                && n.is_positive()
            {
                let n_i64: i64 = n.to_integer().try_into().unwrap_or(0);
                if (2..=10).contains(&n_i64) {
                    let expanded = crate::transforms::expand::expand(arena, expr);
                    if expanded != expr {
                        let result = integrate_node(arena, expanded, var, var_sym, depth - 1);
                        if !matches!(arena.node(result), ExprNode::Integral(_, _)) {
                            return result;
                        }
                    }
                }
            }

            // ── Weierstrass substitution (Pow arm) ────────────────────
            if let Some(result) =
                try_weierstrass_substitution(arena, expr, var, var_sym, depth)
            {
                return result;
            }

            // ── 1/ln(x) → li(x) (logarithmic integral) ──────────────
            if let ExprNode::Ln(inner) = arena.node(base).clone()
                && inner == var
                && let Some(n_val) = arena.as_num(exp)
            {
                let neg_one_r = num_rational::Ratio::<num_bigint::BigInt>::from_integer((-1).into());
                if *n_val == neg_one_r {
                    return make_apply(arena, "li", &[var]);
                }
            }

            // ── 1/(ax+b) with symbolic coefficients → ln|ax+b|/a ────
            if let Some(n_val) = arena.as_num(exp) {
                let neg_one_r = num_rational::Ratio::<num_bigint::BigInt>::from_integer((-1).into());
                if *n_val == neg_one_r && base_has_var
                    && let Some((a_expr, _b_expr)) = symbolic_linear_coeff_of(arena, base, var, var_sym)
                {
                    let abs_base = arena.abs(base);
                    let ln_abs = arena.ln(abs_base);
                    return arena.div(ln_abs, a_expr);
                }
            }

            // ── Distribute inverse over Mul: 1/(a·b) → a^(-1)·b^(-1) ──
            // When the base is a Mul and the exponent is a negative integer,
            // distribute the power over each factor.  This transforms
            // Pow(Mul(x, ln(x)), -1) into Mul(x^(-1), ln(x)^(-1)), which
            // lets the Mul arm's u-substitution logic find candidates.
            if let ExprNode::Mul(ref children) = arena.node(base).clone() {
                if let Some(e_val) = arena.as_num(exp) {
                    if e_val.is_negative() && e_val.is_integer() {
                        let factors: SmallVec<[ExprId; 6]> = children
                            .iter()
                            .map(|&child| arena.pow(child, exp))
                            .collect();
                        let distributed = arena.mul(&factors);
                        if distributed != expr {
                            let result = integrate_node(arena, distributed, var, var_sym, depth - 1);
                            if !matches!(arena.node(result), ExprNode::Integral(_, _)) {
                                return result;
                            }
                        }
                    }
                }
            }

            // General case: unevaluated.
            tracing::debug!("integration: no strategy succeeded, returning unevaluated");
            arena.intern(ExprNode::Integral(expr, var))
        }

        // ── Elementary functions ────────────────────────────────────
        ExprNode::Sin(inner) => {
            if inner == var {
                // ∫ sin(x) dx = -cos(x)
                let cos_x = arena.cos(var);
                return arena.neg(cos_x);
            }
            // Try u-substitution: if inner = a*x + b, ∫ sin(a*x+b) dx = -cos(a*x+b)/a
            if let Some((a_expr, _)) = symbolic_linear_coeff_of(arena, inner, var, var_sym) {
                let cos_inner = arena.cos(inner);
                let neg_cos = arena.neg(cos_inner);
                return arena.div(neg_cos, a_expr);
            }
            arena.intern(ExprNode::Integral(expr, var))
        }

        ExprNode::Cos(inner) => {
            if inner == var {
                // ∫ cos(x) dx = sin(x)
                return arena.sin(var);
            }
            // Try u-substitution: if inner = a*x + b, ∫ cos(a*x+b) dx = sin(a*x+b)/a
            if let Some((a_expr, _)) = symbolic_linear_coeff_of(arena, inner, var, var_sym) {
                let sin_inner = arena.sin(inner);
                return arena.div(sin_inner, a_expr);
            }
            arena.intern(ExprNode::Integral(expr, var))
        }

        // ── Tan: ∫ tan(x) dx = -ln|cos(x)| ───────────────────────
        ExprNode::Tan(inner) => {
            if inner == var {
                // ∫ tan(x) dx = -ln(|cos(x)|)
                let cos_x = arena.cos(var);
                let abs_cos = arena.abs(cos_x);
                let ln_abs_cos = arena.ln(abs_cos);
                return arena.neg(ln_abs_cos);
            }
            // Try u-substitution: if inner = a*x + b,
            // ∫ tan(a*x+b) dx = -ln|cos(a*x+b)| / a
            if let Some((a_expr, _)) = symbolic_linear_coeff_of(arena, inner, var, var_sym) {
                let cos_inner = arena.cos(inner);
                let abs_cos = arena.abs(cos_inner);
                let ln_abs_cos = arena.ln(abs_cos);
                let neg_ln = arena.neg(ln_abs_cos);
                return arena.div(neg_ln, a_expr);
            }
            arena.intern(ExprNode::Integral(expr, var))
        }

        ExprNode::Exp(inner) => {
            if inner == var {
                // ∫ exp(x) dx = exp(x)
                return arena.exp(var);
            }

            // ── Gaussian form: ∫ exp(a·x²+b·x+c) dx where a < 0 ──
            // Result: √π/(2√(−a)) · exp(c − b²/(4a)) · erf((−2a·x−b)/(2√(−a)))
            if let Some(poly) = crate::poly::polybridge::expr_to_poly(arena, inner, var)
                && poly.degree() == Some(2)
            {
                let coeffs = poly.coeffs(); // [c, b, a]
                let a_coeff = &coeffs[2];
                let b_coeff = &coeffs[1];
                let c_coeff = &coeffs[0];

                if a_coeff.is_negative() {
                    // neg_a = -a (positive)
                    let neg_a = -a_coeff.clone();
                    let neg_a_expr = rational_to_expr(arena, &neg_a);

                    // sqrt(-a)
                    let sqrt_neg_a = arena.sqrt(neg_a_expr);

                    let two = arena.int(2);

                    // front = √π / (2·√(-a))
                    let pi_id = arena.pi;
                    let sqrt_pi = arena.sqrt(pi_id);
                    let two_sqrt_neg_a = arena.mul(&[two, sqrt_neg_a]);
                    let front = arena.div(sqrt_pi, two_sqrt_neg_a);

                    // exp_factor = exp(c - b²/(4a))
                    let b_expr = rational_to_expr(arena, b_coeff);
                    let a_expr = rational_to_expr(arena, a_coeff);
                    let c_expr = rational_to_expr(arena, c_coeff);

                    let b_sq = arena.mul(&[b_expr, b_expr]);
                    let four = arena.int(4);
                    let four_a = arena.mul(&[four, a_expr]);
                    let b_sq_over_4a = arena.div(b_sq, four_a);
                    let exp_arg = arena.sub(c_expr, b_sq_over_4a);
                    let exp_factor = arena.exp(exp_arg);

                    // erf_arg = (-2a·x - b) / (2·√(-a))
                    // Note: -2a is positive since a < 0
                    let neg_two_a = {
                        let two_r =
                            num_rational::Ratio::<num_bigint::BigInt>::from_integer(2.into());
                        let val = -two_r * a_coeff;
                        rational_to_expr(arena, &val)
                    };
                    let neg_2ax = arena.mul(&[neg_two_a, var]);
                    let erf_numer = arena.sub(neg_2ax, b_expr);
                    // recompute sqrt_neg_a fresh (the prior one may have been consumed)
                    let sqrt_neg_a2 = arena.sqrt(neg_a_expr);
                    let erf_denom = arena.mul(&[two, sqrt_neg_a2]);
                    let erf_arg = arena.div(erf_numer, erf_denom);
                    let erf_term = arena.erf(erf_arg);

                    return arena.mul(&[front, exp_factor, erf_term]);
                }
            }

            // Try u-substitution: if inner = a*x + b, ∫ exp(a*x+b) dx = exp(a*x+b)/a
            if let Some((a_expr, _)) = symbolic_linear_coeff_of(arena, inner, var, var_sym) {
                let exp_inner = arena.exp(inner);
                return arena.div(exp_inner, a_expr);
            }
            arena.intern(ExprNode::Integral(expr, var))
        }

        // ── Ln: ∫ ln(x) dx = x·ln(x) - x ─────────────────────────
        ExprNode::Ln(inner) => {
            if inner == var {
                // ∫ ln(x) dx = x·ln(x) - x
                let ln_var = arena.ln(var);
                let x_ln_x = arena.mul(&[var, ln_var]);
                return arena.sub(x_ln_x, var);
            }
            // ∫ ln(ln(x)) dx = x·ln(ln(x)) − li(x)  (by parts:
            // u = ln(ln(x)), dv = dx  →  du = 1/(x·ln(x)) dx, v = x)
            if let ExprNode::Ln(ln_inner) = arena.node(inner).clone()
                && ln_inner == var
            {
                let ln_x = arena.ln(var);
                let ln_ln_x = arena.ln(ln_x);
                let x_ln_ln_x = arena.mul(&[var, ln_ln_x]);
                let li_x = make_apply(arena, "li", &[var]);
                return arena.sub(x_ln_ln_x, li_x);
            }
            // Try u-substitution: if inner = a*x + b (linear),
            // ∫ ln(a*x+b) dx = ((a*x+b)·ln(a*x+b) - (a*x+b)) / a
            if let Some((a_expr, _)) = symbolic_linear_coeff_of(arena, inner, var, var_sym) {
                let ln_inner = arena.ln(inner);
                let inner_times_ln = arena.mul(&[inner, ln_inner]);
                let diff = arena.sub(inner_times_ln, inner);
                return arena.div(diff, a_expr);
            }
            arena.intern(ExprNode::Integral(expr, var))
        }

        ExprNode::Sinh(inner) => {
            if inner == var {
                // ∫ sinh(x) dx = cosh(x)
                return arena.intern(ExprNode::Cosh(var));
            }
            // u-sub: ∫ sinh(ax+b) dx = cosh(ax+b)/a
            if let Some((a_expr, _)) = symbolic_linear_coeff_of(arena, inner, var, var_sym) {
                let cosh_inner = arena.cosh(inner);
                return arena.div(cosh_inner, a_expr);
            }
            arena.intern(ExprNode::Integral(expr, var))
        }

        ExprNode::Cosh(inner) => {
            if inner == var {
                // ∫ cosh(x) dx = sinh(x)
                return arena.intern(ExprNode::Sinh(var));
            }
            // u-sub: ∫ cosh(ax+b) dx = sinh(ax+b)/a
            if let Some((a_expr, _)) = symbolic_linear_coeff_of(arena, inner, var, var_sym) {
                let sinh_inner = arena.sinh(inner);
                return arena.div(sinh_inner, a_expr);
            }
            arena.intern(ExprNode::Integral(expr, var))
        }

        ExprNode::Tanh(inner) => {
            if inner == var {
                // ∫ tanh(x) dx = ln(cosh(x))
                let cosh_x = arena.cosh(var);
                return arena.ln(cosh_x);
            }
            // u-sub: if inner = a*x + b, ∫ tanh(a*x+b) dx = ln(cosh(a*x+b))/a
            if let Some((a_expr, _)) = symbolic_linear_coeff_of(arena, inner, var, var_sym) {
                let cosh_inner = arena.cosh(inner);
                let ln_cosh = arena.ln(cosh_inner);
                return arena.div(ln_cosh, a_expr);
            }
            arena.intern(ExprNode::Integral(expr, var))
        }

        ExprNode::Asin(inner) => {
            if inner == var {
                // ∫ asin(x) dx = x*asin(x) + sqrt(1-x²)
                let asin_var = arena.asin(var);
                let x_asin = arena.mul(&[var, asin_var]);
                let one = arena.one;
                let two = arena.int(2);
                let x2 = arena.pow(var, two);
                let one_minus_x2 = arena.sub(one, x2);
                let half = arena.rational(1, 2);
                let sqrt_term = arena.pow(one_minus_x2, half);
                return arena.add(&[x_asin, sqrt_term]);
            }
            arena.intern(ExprNode::Integral(expr, var))
        }

        ExprNode::Acos(inner) => {
            if inner == var {
                // ∫ acos(x) dx = x*acos(x) - sqrt(1-x²)
                let acos_var = arena.acos(var);
                let x_acos = arena.mul(&[var, acos_var]);
                let one = arena.one;
                let two = arena.int(2);
                let x2 = arena.pow(var, two);
                let one_minus_x2 = arena.sub(one, x2);
                let half = arena.rational(1, 2);
                let sqrt_term = arena.pow(one_minus_x2, half);
                return arena.sub(x_acos, sqrt_term);
            }
            arena.intern(ExprNode::Integral(expr, var))
        }

        ExprNode::Atan(inner) => {
            if inner == var {
                // ∫ atan(x) dx = x*atan(x) - 1/2*ln(1+x²)
                let atan_var = arena.atan(var);
                let x_atan = arena.mul(&[var, atan_var]);
                let two = arena.int(2);
                let x2 = arena.pow(var, two);
                let one = arena.one;
                let one_plus_x2 = arena.add(&[one, x2]);
                let half = arena.rational(1, 2);
                let ln_term = arena.ln(one_plus_x2);
                let half_ln = arena.mul(&[half, ln_term]);
                return arena.sub(x_atan, half_ln);
            }
            arena.intern(ExprNode::Integral(expr, var))
        }

        // ── DiracDelta: ∫δ(f(x))dx ────────────────────────────────
        ExprNode::DiracDelta(inner) => {
            // ∫δ(x)dx = H(x)
            if inner == var {
                return arena.intern(ExprNode::Heaviside(var));
            }
            // Linear case: ∫δ(ax+b)dx = H(ax+b) / |a|
            if let Some((a_expr, _)) = symbolic_linear_coeff_of(arena, inner, var, var_sym) {
                let h = arena.intern(ExprNode::Heaviside(inner));
                let abs_a = arena.abs(a_expr);
                return arena.div(h, abs_a);
            }
            // Leave unevaluated
            arena.intern(ExprNode::Integral(expr, var))
        }

        // ── Heaviside: ∫ H(g(x)) dx ───────────────────────────────
        ExprNode::Heaviside(inner) => {
            // ∫ H(x) dx = x·H(x)
            if inner == var {
                return arena.mul(&[var, expr]);
            }
            // Linear case: ∫ H(ax+b) dx = (ax+b)·H(ax+b) / a
            if let Some((a_expr, _b_expr)) = symbolic_linear_coeff_of(arena, inner, var, var_sym) {
                let h = arena.intern(ExprNode::Heaviside(inner));
                let product = arena.mul(&[inner, h]);
                return arena.div(product, a_expr);
            }
            // Leave unevaluated
            arena.intern(ExprNode::Integral(expr, var))
        }

        // ── Asinh: ∫ asinh(x) dx = x·asinh(x) - √(x²+1) ─────────
        ExprNode::Asinh(inner) => {
            if inner == var {
                tracing::debug!("integrate: matched asinh(x) direct");
                let asinh_var = arena.asinh(var);
                let x_asinh = arena.mul(&[var, asinh_var]);
                let two = arena.int(2);
                let x2 = arena.pow(var, two);
                let one = arena.one;
                let x2_plus_1 = arena.add(&[x2, one]);
                let half = arena.rational(1, 2);
                let sqrt_term = arena.pow(x2_plus_1, half);
                return arena.sub(x_asinh, sqrt_term);
            }
            // Linear chain rule: ∫ asinh(ax+b) dx = (ax+b)·asinh(ax+b)/a - √((ax+b)²+1)/a
            if let Some((a_expr, _b_expr)) = symbolic_linear_coeff_of(arena, inner, var, var_sym)
            {
                tracing::debug!("integrate: matched asinh(ax+b) linear");
                let asinh_g = arena.asinh(inner);
                let g_asinh = arena.mul(&[inner, asinh_g]);
                let two = arena.int(2);
                let g2 = arena.pow(inner, two);
                let one = arena.one;
                let g2_plus_1 = arena.add(&[g2, one]);
                let half = arena.rational(1, 2);
                let sqrt_term = arena.pow(g2_plus_1, half);
                let numer = arena.sub(g_asinh, sqrt_term);
                return arena.div(numer, a_expr);
            }
            arena.intern(ExprNode::Integral(expr, var))
        }

        // ── Acosh: ∫ acosh(x) dx = x·acosh(x) - √(x²-1) ─────────
        ExprNode::Acosh(inner) => {
            if inner == var {
                tracing::debug!("integrate: matched acosh(x) direct");
                let acosh_var = arena.acosh(var);
                let x_acosh = arena.mul(&[var, acosh_var]);
                let two = arena.int(2);
                let x2 = arena.pow(var, two);
                let one = arena.one;
                let x2_minus_1 = arena.sub(x2, one);
                let half = arena.rational(1, 2);
                let sqrt_term = arena.pow(x2_minus_1, half);
                return arena.sub(x_acosh, sqrt_term);
            }
            // Linear chain rule: ∫ acosh(ax+b) dx
            if let Some((a_expr, _b_expr)) = symbolic_linear_coeff_of(arena, inner, var, var_sym)
            {
                tracing::debug!("integrate: matched acosh(ax+b) linear");
                let acosh_g = arena.acosh(inner);
                let g_acosh = arena.mul(&[inner, acosh_g]);
                let two = arena.int(2);
                let g2 = arena.pow(inner, two);
                let one = arena.one;
                let g2_minus_1 = arena.sub(g2, one);
                let half = arena.rational(1, 2);
                let sqrt_term = arena.pow(g2_minus_1, half);
                let numer = arena.sub(g_acosh, sqrt_term);
                return arena.div(numer, a_expr);
            }
            arena.intern(ExprNode::Integral(expr, var))
        }

        // ── Atanh: ∫ atanh(x) dx = x·atanh(x) + ½·ln(1-x²) ──────
        ExprNode::Atanh(inner) => {
            if inner == var {
                tracing::debug!("integrate: matched atanh(x) direct");
                let atanh_var = arena.atanh(var);
                let x_atanh = arena.mul(&[var, atanh_var]);
                let two = arena.int(2);
                let x2 = arena.pow(var, two);
                let one = arena.one;
                let one_minus_x2 = arena.sub(one, x2);
                let half = arena.rational(1, 2);
                let ln_term = arena.ln(one_minus_x2);
                let half_ln = arena.mul(&[half, ln_term]);
                return arena.add(&[x_atanh, half_ln]);
            }
            // Linear chain rule: ∫ atanh(ax+b) dx
            if let Some((a_expr, _b_expr)) = symbolic_linear_coeff_of(arena, inner, var, var_sym)
            {
                tracing::debug!("integrate: matched atanh(ax+b) linear");
                let atanh_g = arena.atanh(inner);
                let g_atanh = arena.mul(&[inner, atanh_g]);
                let two = arena.int(2);
                let g2 = arena.pow(inner, two);
                let one = arena.one;
                let one_minus_g2 = arena.sub(one, g2);
                let half = arena.rational(1, 2);
                let ln_term = arena.ln(one_minus_g2);
                let half_ln = arena.mul(&[half, ln_term]);
                let numer = arena.add(&[g_atanh, half_ln]);
                return arena.div(numer, a_expr);
            }
            arena.intern(ExprNode::Integral(expr, var))
        }

        // Sign: leave as unevaluated integral
        ExprNode::Sign(_) => arena.intern(ExprNode::Integral(expr, var)),

        // Everything else: unevaluated integral.
        _ => {
            tracing::debug!("integration: no strategy succeeded, returning unevaluated");
            arena.intern(ExprNode::Integral(expr, var))
        }
    }
}

/// Check if an expression contains the given symbol.
fn contains_var(arena: &Arena, expr: ExprId, var: SymbolId) -> bool {
    let mut stack: Vec<ExprId> = vec![expr];
    let mut visited: FxHashMap<ExprId, ()> = FxHashMap::default();
    while let Some(id) = stack.pop() {
        if visited.contains_key(&id) {
            continue;
        }
        visited.insert(id, ());
        if let ExprNode::Symbol(sid) = arena.node(id)
            && *sid == var
        {
            return true;
        }
        let children = arena.node(id).children();
        stack.extend_from_slice(&children);
    }
    false
}

/// Check if an expression is a polynomial in the given variable.
/// A polynomial is: the variable itself, a power of the variable with a
/// non-negative integer exponent, a numeric constant, or sums/products of these.
fn is_polynomial_in(arena: &Arena, expr: ExprId, var: ExprId, var_sym: SymbolId) -> bool {
    if expr == var {
        return true;
    }
    if !contains_var(arena, expr, var_sym) {
        return true; // constant
    }
    match arena.node(expr).clone() {
        ExprNode::Pow(base, exp) => {
            if base == var {
                // x^n where n is a non-negative integer
                if let Some(r) = arena.as_num(exp) {
                    return r.is_integer() && !r.is_negative();
                }
            }
            false
        }
        ExprNode::Mul(children) => children
            .iter()
            .all(|&c| is_polynomial_in(arena, c, var, var_sym)),
        ExprNode::Add(children) => children
            .iter()
            .all(|&c| is_polynomial_in(arena, c, var, var_sym)),
        ExprNode::Neg(inner) => is_polynomial_in(arena, inner, var, var_sym),
        ExprNode::Num(_) => true,
        _ => false,
    }
}

/// Check if `expr` is a linear function of `var` with **numeric** coefficients.
/// Returns `Some(a)` (the leading coefficient as `Ratio<BigInt>`) if linear, `None` otherwise.
#[allow(dead_code)]
fn linear_coeff_of(
    arena: &Arena,
    expr: ExprId,
    _var: ExprId,
    _var_sym: SymbolId,
) -> Option<num_rational::Ratio<num_bigint::BigInt>> {
    // Try to convert to polynomial in var.
    let poly = crate::poly::polybridge::expr_to_poly(arena, expr, _var)?;
    // Must be degree exactly 1.
    if poly.degree()? != 1 {
        return None;
    }
    let a = poly.coeff(1);
    if a.is_zero() {
        return None;
    }
    Some(a)
}

/// Check if `expr` is a linear function of `var`: `a*var + b` where a ≠ 0,
/// with **symbolic** (possibly non-numeric) coefficients.
///
/// Returns `Some((a_expr, b_expr))` where `a_expr` is the coefficient of `var`
/// and `b_expr` is the constant term — both as `ExprId`s that are free of `var`.
fn symbolic_linear_coeff_of(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<(ExprId, ExprId)> {
    // Fast path: try numeric first (covers the common numeric-coefficient case)
    if let Some(a) = linear_coeff_of(arena, expr, var, var_sym) {
        let a_id = rational_to_expr(arena, &a);
        // Also extract constant term
        if let Some(poly) = crate::poly::polybridge::expr_to_poly(arena, expr, var) {
            let b = poly.coeff(0);
            let b_id = rational_to_expr(arena, &b);
            return Some((a_id, b_id));
        }
        return Some((a_id, arena.zero));
    }

    // Case 1: expr == var → coefficient is 1, constant is 0
    if expr == var {
        return Some((arena.one, arena.zero));
    }

    // Case 2: Neg(inner) → negate coefficient and constant
    if let ExprNode::Neg(inner) = arena.node(expr).clone() {
        if let Some((coeff, constant)) = symbolic_linear_coeff_of(arena, inner, var, var_sym) {
            let neg_coeff = arena.neg(coeff);
            let neg_const = arena.neg(constant);
            return Some((neg_coeff, neg_const));
        }
        return None;
    }

    // Case 3: Mul containing var exactly once, all other factors free of var
    if let ExprNode::Mul(ref children) = arena.node(expr).clone() {
        let mut has_var = false;
        let mut other_factors: SmallVec<[ExprId; 4]> = SmallVec::new();
        let mut var_count = 0u32;

        for &child in children {
            if child == var {
                var_count += 1;
                if var_count > 1 {
                    return None; // var² or higher
                }
                has_var = true;
            } else if contains_var(arena, child, var_sym) {
                return None; // Non-trivial var dependence
            } else {
                other_factors.push(child);
            }
        }

        if has_var && var_count == 1 {
            let coeff = match other_factors.len() {
                0 => arena.one,
                1 => other_factors[0],
                _ => arena.mul(&other_factors),
            };
            return Some((coeff, arena.zero));
        }
    }

    // Case 4: Add → separate var-containing and var-free terms
    if let ExprNode::Add(ref children) = arena.node(expr).clone() {
        let mut var_terms: SmallVec<[ExprId; 4]> = SmallVec::new();
        let mut const_terms: SmallVec<[ExprId; 4]> = SmallVec::new();

        for &child in children {
            if contains_var(arena, child, var_sym) {
                var_terms.push(child);
            } else {
                const_terms.push(child);
            }
        }

        if var_terms.is_empty() {
            return None; // No var dependence — not linear in var
        }

        // The var-containing part should be a single term of the form c*var
        let var_part = if var_terms.len() == 1 {
            var_terms[0]
        } else {
            arena.add(&var_terms)
        };

        // Try to extract coefficient from var_part (should be c*var)
        let coeff = if var_part == var {
            arena.one
        } else if let ExprNode::Mul(ref mul_children) = arena.node(var_part).clone() {
            let mut has_v = false;
            let mut other: SmallVec<[ExprId; 4]> = SmallVec::new();
            let mut vc = 0u32;
            for &mc in mul_children {
                if mc == var {
                    vc += 1;
                    if vc > 1 {
                        return None;
                    }
                    has_v = true;
                } else if contains_var(arena, mc, var_sym) {
                    return None;
                } else {
                    other.push(mc);
                }
            }
            if !has_v || vc != 1 {
                return None;
            }
            match other.len() {
                0 => arena.one,
                1 => other[0],
                _ => arena.mul(&other),
            }
        } else {
            return None; // Can't decompose
        };

        // Verify coefficient is free of var
        if contains_var(arena, coeff, var_sym) {
            return None;
        }

        let constant = match const_terms.len() {
            0 => arena.zero,
            1 => const_terms[0],
            _ => arena.add(&const_terms),
        };

        // Verify constant is free of var
        if contains_var(arena, constant, var_sym) {
            return None;
        }

        return Some((coeff, constant));
    }

    None
}

/// Convert a Ratio<BigInt> to an ExprId.
fn rational_to_expr(arena: &mut Arena, r: &num_rational::Ratio<num_bigint::BigInt>) -> ExprId {
    let nid = arena.intern_num(r.clone());
    arena.intern(crate::base::node::ExprNode::Num(nid))
}

// ═══════════════════════════════════════════════════════════════════════════
// General u-substitution
// ═══════════════════════════════════════════════════════════════════════════

/// Try general u-substitution on a product integrand.
///
/// For each dependent factor `g(u)` where `u = u(x)`, this checks
/// whether the remaining factors equal `du/dx` times a constant `c`.
/// When they do, `∫ c · (du/dx) · g(u) dx = c · G(u)` where `G` is
/// the antiderivative of `g` with respect to `u`.
///
/// The technique works by:
/// 1. Extracting candidate inner arguments from function / power nodes.
/// 2. Computing `du/dx` via [`crate::transforms::diff::diff`].
/// 3. Forming `remaining / du` and checking it is free of `var`.
/// 4. Substituting `u → var` in the factor, integrating, then
///    substituting back.
fn try_u_substitution(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    for (i, &factor) in dependent.iter().enumerate() {
        let candidates = u_sub_candidates(arena, factor, var_sym);

        for u_expr in candidates {
            // Skip the trivial u = var case (already handled elsewhere)
            if u_expr == var {
                continue;
            }

            // Compute du/dx
            let du = crate::transforms::diff::diff(arena, u_expr, var);
            if du == arena.zero {
                continue;
            }

            // Product of the remaining dependent factors
            let remaining_expr = remaining_product(arena, dependent, i);

            // quotient = remaining / du — if free of var, we have our constant
            let quotient = arena.div(remaining_expr, du);

            // Try the raw quotient first; fall back to polynomial cancellation
            let coeff = if !contains_var(arena, quotient, var_sym) {
                quotient
            } else {
                let cancelled = arena.cancel_expr(quotient, var);
                if !contains_var(arena, cancelled, var_sym) {
                    cancelled
                } else {
                    continue;
                }
            };

            // Replace u(x) → var inside the factor to get g(var),
            // integrate g(var) w.r.t. var, then substitute var → u(x) back.
            let g_of_var = arena.subs_structural(factor, u_expr, var);
            let g_integrated = integrate_node(arena, g_of_var, var, var_sym, depth);

            // If the inner integral is unevaluated, this candidate didn't help
            if matches!(arena.node(g_integrated), ExprNode::Integral(_, _)) {
                continue;
            }

            // G(u) — substitute var back to u(x)
            let antideriv = arena.subs_structural(g_integrated, var, u_expr);
            tracing::debug!("u-substitution succeeded");
            return Some(arena.mul(&[coeff, antideriv]));
        }
    }
    None
}

/// Collect candidate `u`-expressions from a single factor.
///
/// For function nodes (`sin`, `cos`, `exp`, …) the inner argument is
/// returned.  For `Pow(base, exp)` the base is returned (enabling
/// e.g. `u = x² + 1` inside `(x²+1)^{-1}`).
fn u_sub_candidates(arena: &Arena, factor: ExprId, var_sym: SymbolId) -> SmallVec<[ExprId; 4]> {
    let mut out: SmallVec<[ExprId; 4]> = SmallVec::new();
    match arena.node(factor).clone() {
        ExprNode::Sin(inner)
        | ExprNode::Cos(inner)
        | ExprNode::Tan(inner)
        | ExprNode::Exp(inner)
        | ExprNode::Ln(inner)
        | ExprNode::Sinh(inner)
        | ExprNode::Cosh(inner)
        | ExprNode::Tanh(inner)
        | ExprNode::Asin(inner)
        | ExprNode::Acos(inner)
        | ExprNode::Atan(inner)
        | ExprNode::Asinh(inner)
        | ExprNode::Acosh(inner)
        | ExprNode::Atanh(inner)
        | ExprNode::Abs(inner) => {
            if contains_var(arena, inner, var_sym) {
                out.push(inner);
                // Also try the function node itself as a candidate.
                // E.g., for ln(x), try u = ln(x) (not just u = x).
                // This enables ∫ 1/(x·ln(x)) dx via u = ln(x), du = 1/x dx.
                out.push(factor);
            }
        }
        ExprNode::Pow(base, _exp) => {
            if contains_var(arena, base, var_sym) {
                out.push(base);
            }
        }
        _ => {}
    }
    out
}

/// Build the product of all elements in `children` except index `skip`.
fn remaining_product(arena: &mut Arena, children: &[ExprId], skip: usize) -> ExprId {
    let parts: SmallVec<[ExprId; 4]> = children
        .iter()
        .enumerate()
        .filter(|&(j, _)| j != skip)
        .map(|(_, &c)| c)
        .collect();
    match parts.len() {
        0 => arena.one,
        1 => parts[0],
        _ => arena.mul(&parts),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

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
fn try_piecewise_wrap(
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

    // Collect denominator expressions from the result.
    let denoms = collect_denominators(arena, result);
    if denoms.is_empty() {
        return result;
    }

    let mut wrapped = result;
    let mut handled: Vec<(ExprId, ExprId)> = Vec::new();

    for denom in &denoms {
        let denom_syms = crate::base::walk::free_symbols(arena, *denom);
        for sym_expr in &denom_syms {
            // Skip the integration variable.
            if let ExprNode::Symbol(sid) = arena.node(*sym_expr)
                && *sid == var_sym
            {
                continue;
            }

            // Solve denom = 0 for this parameter symbol.
            let solutions = crate::transforms::solve::solve(arena, *denom, *sym_expr);
            for sol in &solutions {
                let degen_val = sol.value;

                // Avoid duplicate wrapping for the same (param, value) pair.
                if handled.iter().any(|&(p, v)| p == *sym_expr && v == degen_val) {
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
                let degen_result = integrate(arena, integrand_at_degen, var);
                let degen_result = crate::transforms::eval::eval(arena, degen_result);

                // Skip if re-integration returned unevaluated.
                if matches!(arena.node(degen_result), ExprNode::Integral(_, _)) {
                    continue;
                }

                // Build Piecewise: [(generic, Ne(param, degen)), (degen_result, True)]
                let condition = arena.ne_(*sym_expr, degen_val);
                let true_cond = arena.bool_true;
                wrapped = arena.piecewise(&[
                    (wrapped, condition),
                    (degen_result, true_cond),
                ]);

                handled.push((*sym_expr, degen_val));
            }
        }
    }

    wrapped
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
fn has_zero_denominator(arena: &Arena, expr: ExprId) -> bool {
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
            && r.is_negative() && arena.is_zero_structural(base)
        {
            return true;
        }
        arena.node(id).for_each_child(|c| stack.push(c));
    }

    false
}

// ═══════════════════════════════════════════════════════════════════════════
// Special function integration table
// ═══════════════════════════════════════════════════════════════════════════

/// Try to match the integrand against known special function patterns.
///
/// Handles:
/// - `sin(x)/x` → `Si(x)` (sine integral)
/// - `cos(x)/x` → `Ci(x)` (cosine integral)
/// - `exp(x)/x` → `Ei(x)` (exponential integral)
/// - `exp(-x)/x` → `-Ei(-x)`
fn try_special_function_integral(
    arena: &mut Arena,
    dependent: &[ExprId],
    constants: &[ExprId],
    var: ExprId,
    _var_sym: SymbolId,
) -> Option<ExprId> {
    if dependent.len() != 2 {
        return None;
    }

    // Identify which factor is Pow(var, -1) and which is the function.
    let (func_factor, _inv_factor) = if is_inv_of_var(arena, dependent[0], var) {
        (dependent[1], dependent[0])
    } else if is_inv_of_var(arena, dependent[1], var) {
        (dependent[0], dependent[1])
    } else {
        return None;
    };

    // Match the function factor against known special functions.
    let sf_result = match arena.node(func_factor).clone() {
        ExprNode::Sin(inner) if inner == var => {
            Some(make_apply(arena, "Si", &[var]))
        }
        ExprNode::Cos(inner) if inner == var => {
            Some(make_apply(arena, "Ci", &[var]))
        }
        ExprNode::Exp(inner) if inner == var => {
            Some(make_apply(arena, "Ei", &[var]))
        }
        ExprNode::Exp(inner) => {
            // exp(-x)/x → -Ei(-x)
            if let ExprNode::Neg(neg_inner) = arena.node(inner).clone()
                && neg_inner == var
            {
                let neg_var = arena.neg(var);
                let ei = make_apply(arena, "Ei", &[neg_var]);
                let neg_ei = arena.neg(ei);
                return Some(wrap_with_constants(arena, neg_ei, constants));
            }
            None
        }
        _ => None,
    };

    sf_result.map(|r| wrap_with_constants(arena, r, constants))
}

/// Check if `expr` is `Pow(var, -1)`.
fn is_inv_of_var(arena: &Arena, expr: ExprId, var: ExprId) -> bool {
    if let ExprNode::Pow(base, exp) = arena.node(expr)
        && *base == var
        && let Some(r) = arena.as_num(*exp)
    {
        return *r
            == num_rational::Ratio::<num_bigint::BigInt>::from_integer((-1).into());
    }
    false
}

/// Multiply a result by constant factors (if any).
fn wrap_with_constants(arena: &mut Arena, result: ExprId, constants: &[ExprId]) -> ExprId {
    if constants.is_empty() {
        result
    } else {
        let mut all: SmallVec<[ExprId; 4]> = constants.iter().copied().collect();
        all.push(result);
        arena.mul(&all)
    }
}

/// Create an `Apply` node for a named special function.
fn make_apply(arena: &mut Arena, name: &str, args: &[ExprId]) -> ExprId {
    let sym_node = arena.symbol(name);
    let sid = match arena.node(sym_node) {
        ExprNode::Symbol(s) => *s,
        _ => unreachable!(),
    };
    let args_sv: SmallVec<[ExprId; 2]> = args.iter().copied().collect();
    arena.intern(ExprNode::Apply(sid, args_sv))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::arena::Arena;

    fn sym(a: &mut Arena, name: &str) -> ExprId {
        a.symbol(name)
    }

    fn display(a: &Arena, id: ExprId) -> String {
        a.display(id).to_string()
    }

    #[test]
    fn integrate_constant() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let five = a.int(5);
        let result = integrate(&mut a, five, x);
        assert_eq!(display(&a, result), "5*x");
    }

    #[test]
    fn integrate_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let result = integrate(&mut a, x, x);
        assert_eq!(display(&a, result), "1/2*x^2");
    }

    #[test]
    fn integrate_x_squared() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let result = integrate(&mut a, x2, x);
        assert_eq!(display(&a, result), "1/3*x^3");
    }

    #[test]
    fn integrate_x_inv() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let neg_one = a.int(-1);
        let x_inv = a.pow(x, neg_one);
        let result = integrate(&mut a, x_inv, x);
        let s = display(&a, result);
        // The Risch rational path returns ln(x) (via Rothstein-Trager),
        // while the direct power-rule path returns ln(abs(x)).
        // Both are correct for real x ≠ 0.
        assert!(
            s == "ln(abs(x))" || s == "ln(x)",
            "∫ 1/x dx should be ln(x) or ln(abs(x)), got: {s}"
        );
    }

    #[test]
    fn integrate_sin_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.sin(x);
        let result = integrate(&mut a, expr, x);
        assert_eq!(display(&a, result), "-cos(x)");
    }

    #[test]
    fn integrate_cos_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.cos(x);
        let result = integrate(&mut a, expr, x);
        assert_eq!(display(&a, result), "sin(x)");
    }

    #[test]
    fn integrate_exp_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.exp(x);
        let result = integrate(&mut a, expr, x);
        assert_eq!(display(&a, result), "exp(x)");
    }

    #[test]
    fn integrate_sum() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // ∫ (x + 1) dx = x^2/2 + x
        let one = a.one;
        let sum = a.add(&[x, one]);
        let result = integrate(&mut a, sum, x);
        let s = display(&a, result);
        assert!(s.contains("x^2"), "should contain x^2: {s}");
        assert!(s.contains("x"), "should contain x: {s}");
    }

    #[test]
    fn integrate_constant_times_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let three = a.int(3);
        let expr = a.mul(&[three, x]);
        let result = integrate(&mut a, expr, x);
        assert_eq!(display(&a, result), "3/2*x^2");
    }

    #[test]
    fn integrate_other_symbol_is_constant() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let result = integrate(&mut a, y, x);
        assert_eq!(display(&a, result), "x*y");
    }

    #[test]
    fn integrate_unevaluated_for_unknown() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // tan(x) is now integrable — verify we get the antiderivative
        let expr = a.tan(x);
        let result = integrate(&mut a, expr, x);
        assert_eq!(display(&a, result), "-ln(abs(cos(x)))");
    }

    #[test]
    fn integrate_x_sin_x_by_parts() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // ∫ x·sin(x) dx = sin(x) - x·cos(x)
        let sin_x = a.sin(x);
        let expr = a.mul(&[x, sin_x]);
        let result = integrate(&mut a, expr, x);
        let s = display(&a, result);
        // Should contain both sin(x) and cos(x) terms
        assert!(s.contains("sin(x)"), "should contain sin(x): {s}");
        assert!(s.contains("cos(x)"), "should contain cos(x): {s}");
    }

    #[test]
    fn integrate_x_exp_x_by_parts() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // ∫ x·exp(x) dx = x·exp(x) - exp(x) = (x-1)·exp(x)
        let exp_x = a.exp(x);
        let expr = a.mul(&[x, exp_x]);
        let result = integrate(&mut a, expr, x);
        let s = display(&a, result);
        assert!(s.contains("exp(x)"), "should contain exp(x): {s}");
    }

    #[test]
    fn integrate_x_cos_x_by_parts() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // ∫ x·cos(x) dx = x·sin(x) + cos(x)
        let cos_x = a.cos(x);
        let expr = a.mul(&[x, cos_x]);
        let result = integrate(&mut a, expr, x);
        let s = display(&a, result);
        assert!(s.contains("sin(x)"), "should contain sin(x): {s}");
        assert!(s.contains("cos(x)"), "should contain cos(x): {s}");
    }

    #[test]
    fn integrate_sin_2x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let two_x = a.mul(&[two, x]);
        let expr = a.sin(two_x);
        let result = integrate(&mut a, expr, x);
        let s = display(&a, result);
        // ∫ sin(2x) dx = -cos(2x)/2
        assert!(s.contains("cos"), "should contain cos: {s}");
        assert!(
            s.contains("1/2") || s.contains("2"),
            "should have factor of 1/2: {s}"
        );
    }

    #[test]
    fn integrate_cos_3x_plus_1() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let three = a.int(3);
        let one = a.one;
        let three_x = a.mul(&[three, x]);
        let inner = a.add(&[three_x, one]);
        let expr = a.cos(inner);
        let result = integrate(&mut a, expr, x);
        let s = display(&a, result);
        assert!(s.contains("sin"), "should contain sin: {s}");
    }

    #[test]
    fn integrate_exp_2x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let two_x = a.mul(&[two, x]);
        let expr = a.exp(two_x);
        let result = integrate(&mut a, expr, x);
        let s = display(&a, result);
        assert!(s.contains("exp"), "should contain exp: {s}");
    }

    // ═══════════════════════════════════════════════════════════════════
    // Sprint A – new tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn integrate_tan_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.tan(x);
        let result = integrate(&mut a, expr, x);
        assert_eq!(display(&a, result), "-ln(abs(cos(x)))");
    }

    #[test]
    fn integrate_ln_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.ln(x);
        let result = integrate(&mut a, expr, x);
        let s = display(&a, result);
        // ∫ ln(x) dx = x·ln(x) - x
        // Canonical form might be: -x + x*ln(x)  or  x*ln(x) + -x  etc.
        assert!(s.contains("ln(x)"), "should contain ln(x): {s}");
        assert!(s.contains("x"), "should contain x: {s}");
        // Verify both the x*ln(x) and the -x terms are present
        assert!(
            s.contains("x*ln(x)") || s.contains("ln(x)*x"),
            "should contain x*ln(x): {s}"
        );
    }

    #[test]
    fn integrate_atan_form() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // ∫ 1/(x²+1) dx = atan(x)
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let one = a.one;
        let base = a.add(&[x2, one]);
        let neg_one = a.int(-1);
        let expr = a.pow(base, neg_one);
        let result = integrate(&mut a, expr, x);
        let s = display(&a, result);
        assert!(s.contains("atan(x)"), "should be atan(x), got: {s}");
    }

    #[test]
    fn integrate_asin_form() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // ∫ 1/sqrt(1-x²) dx = (1-x²)^(-1/2) = asin(x)
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let one = a.one;
        let base = a.sub(one, x2); // 1 - x²
        let neg_half = a.rational(-1, 2);
        let expr = a.pow(base, neg_half);
        let result = integrate(&mut a, expr, x);
        let s = display(&a, result);
        assert!(s.contains("asin(x)"), "should be asin(x), got: {s}");
    }

    #[test]
    fn integrate_tan_2x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // ∫ tan(2x) dx = -ln|cos(2x)| / 2
        let two = a.int(2);
        let two_x = a.mul(&[two, x]);
        let expr = a.tan(two_x);
        let result = integrate(&mut a, expr, x);
        let s = display(&a, result);
        assert!(
            s.contains("ln") && s.contains("cos"),
            "should contain ln and cos: {s}"
        );
        assert!(
            s.contains("1/2") || s.contains("2"),
            "should have factor involving 2: {s}"
        );
    }

    #[test]
    fn integrate_ln_3x_plus_1() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // ∫ ln(3x+1) dx = ((3x+1)·ln(3x+1) - (3x+1)) / 3
        let three = a.int(3);
        let one = a.one;
        let three_x = a.mul(&[three, x]);
        let inner = a.add(&[three_x, one]);
        let expr = a.ln(inner);
        let result = integrate(&mut a, expr, x);
        let s = display(&a, result);
        assert!(s.contains("ln"), "should contain ln: {s}");
        assert!(!s.contains("Integral"), "should not be unevaluated: {s}");
    }

    #[test]
    fn integrate_tanh_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.tanh(x);
        let result = integrate(&mut a, expr, x);
        let s = display(&a, result);
        assert!(
            s.contains("ln") && s.contains("cosh"),
            "∫ tanh(x) dx should be ln(cosh(x)), got: {s}"
        );
    }

    #[test]
    fn integrate_tanh_2x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let two_x = a.mul(&[two, x]);
        let expr = a.tanh(two_x);
        let result = integrate(&mut a, expr, x);
        let s = display(&a, result);
        assert!(
            s.contains("ln") && s.contains("cosh"),
            "∫ tanh(2x) dx should involve ln(cosh(2x)), got: {s}"
        );
    }

    #[test]
    fn integrate_sinh_2x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let two_x = a.mul(&[two, x]);
        let expr = a.sinh(two_x);
        let result = integrate(&mut a, expr, x);
        let s = display(&a, result);
        // ∫ sinh(2x) dx = cosh(2x)/2
        assert!(
            s.contains("cosh"),
            "∫ sinh(2x) dx should involve cosh, got: {s}"
        );
    }

    #[test]
    fn integrate_cosh_3x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let three = a.int(3);
        let three_x = a.mul(&[three, x]);
        let expr = a.cosh(three_x);
        let result = integrate(&mut a, expr, x);
        let s = display(&a, result);
        // ∫ cosh(3x) dx = sinh(3x)/3
        assert!(
            s.contains("sinh"),
            "∫ cosh(3x) dx should involve sinh, got: {s}"
        );
    }

    #[test]
    fn integrate_2x_plus_1_cubed() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let one = a.one;
        let two_x = a.mul(&[two, x]);
        let inner = a.add(&[two_x, one]);
        let three = a.int(3);
        let expr = a.pow(inner, three);
        let result = integrate(&mut a, expr, x);
        let s = display(&a, result);
        // ∫ (2x+1)^3 dx = (2x+1)^4 / 8
        assert!(!s.contains("Integral"), "should not be unevaluated: {s}");
    }

    #[test]
    fn integrate_x_plus_1_squared() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.one;
        let inner = a.add(&[x, one]);
        let two = a.int(2);
        let expr = a.pow(inner, two);
        let result = integrate(&mut a, expr, x);
        let s = display(&a, result);
        // ∫ (x+1)^2 dx should be resolved (linear sub or expand)
        assert!(!s.contains("Integral"), "should not be unevaluated: {s}");
    }

    #[test]
    fn integrate_asin_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.asin(x);
        let result = integrate(&mut a, expr, x);
        let s = display(&a, result);
        assert!(s.contains("asin"), "should contain asin: {s}");
    }

    #[test]
    fn integrate_atan_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.atan(x);
        let result = integrate(&mut a, expr, x);
        let s = display(&a, result);
        assert!(s.contains("atan"), "should contain atan: {s}");
    }

    #[test]
    fn integrate_acos_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.acos(x);
        let result = integrate(&mut a, expr, x);
        let s = display(&a, result);
        assert!(s.contains("acos"), "should contain acos: {s}");
    }

    // ═══════════════════════════════════════════════════════════════════
    // u-substitution tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn integrate_2x_exp_x_squared_u_sub() {
        // ∫ 2x·exp(x²) dx = exp(x²)   [u = x², du = 2x dx]
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let two_x = a.mul(&[two, x]);
        let x2 = a.pow(x, two);
        let exp_x2 = a.exp(x2);
        let expr = a.mul(&[two_x, exp_x2]);
        let result = integrate(&mut a, expr, x);
        let s = display(&a, result);
        assert!(
            s.contains("exp"),
            "∫ 2x·exp(x²) dx should contain exp, got: {s}"
        );
        assert!(
            !s.contains("Integral"),
            "∫ 2x·exp(x²) dx should not be unevaluated, got: {s}"
        );
    }

    #[test]
    fn integrate_cos_x_exp_sin_x_u_sub() {
        // ∫ cos(x)·exp(sin(x)) dx = exp(sin(x))   [u = sin(x), du = cos(x) dx]
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let sin_x = a.sin(x);
        let cos_x = a.cos(x);
        let exp_sin_x = a.exp(sin_x);
        let expr = a.mul(&[cos_x, exp_sin_x]);
        let result = integrate(&mut a, expr, x);
        let s = display(&a, result);
        assert!(
            s.contains("exp"),
            "∫ cos(x)·exp(sin(x)) dx should contain exp, got: {s}"
        );
        assert!(
            !s.contains("Integral"),
            "∫ cos(x)·exp(sin(x)) dx should not be unevaluated, got: {s}"
        );
    }

    #[test]
    fn integrate_x_over_x2_plus_1_u_sub() {
        // ∫ x/(x²+1) dx = ½·ln(x²+1)   [u = x²+1, du = 2x dx]
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let one = a.one;
        let x2_plus_1 = a.add(&[x2, one]);
        let expr = a.div(x, x2_plus_1);
        let result = integrate(&mut a, expr, x);
        let s = display(&a, result);
        assert!(
            s.contains("ln"),
            "∫ x/(x²+1) dx should contain ln, got: {s}"
        );
        assert!(
            !s.contains("Integral"),
            "∫ x/(x²+1) dx should not be unevaluated, got: {s}"
        );
    }

    #[test]
    fn integrate_complete_square() {
        // ∫ 1/(x²+2x+5) dx — should use completing the square
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let five = a.int(5);
        let x2 = a.pow(x, two);
        let two_x = a.mul(&[two, x]);
        let quadratic = a.add(&[x2, two_x, five]);
        let neg_one = a.int(-1);
        let integrand = a.pow(quadratic, neg_one);
        let result = integrate(&mut a, integrand, x);
        let s = display(&a, result);
        assert!(s.contains("atan"), "should use atan: {s}");
    }

    #[test]
    fn integrate_complete_square_simple() {
        // ∫ 1/(x²+x+1) dx
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let one = a.one;
        let quadratic = a.add(&[x2, x, one]);
        let neg_one = a.int(-1);
        let integrand = a.pow(quadratic, neg_one);
        let result = integrate(&mut a, integrand, x);
        let s = display(&a, result);
        assert!(s.contains("atan"), "should use atan: {s}");
    }

    #[test]
    fn symbolic_linear_coeff_of_mul_a_x() {
        // a*x should be detected as linear in x with coefficient a
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let param_a = sym(&mut a, "a");
        let ax = a.mul(&[param_a, x]);
        let var_sym = match a.node(x) {
            ExprNode::Symbol(sid) => *sid,
            _ => panic!("x should be a symbol"),
        };
        let result = super::symbolic_linear_coeff_of(&mut a, ax, x, var_sym);
        assert!(result.is_some(), "a*x should be recognized as linear in x, node: {:?}", a.node(ax));
        let (coeff, constant) = result.unwrap();
        assert_eq!(coeff, param_a, "coefficient should be a, got {}", display(&a, coeff));
        assert_eq!(constant, a.zero, "constant should be 0, got {}", display(&a, constant));
    }

    #[test]
    fn symbolic_linear_coeff_of_add_ax_b() {
        // a*x + b should be detected as linear in x with coefficient a, constant b
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let param_a = sym(&mut a, "a");
        let param_b = sym(&mut a, "b");
        let ax = a.mul(&[param_a, x]);
        let ax_plus_b = a.add(&[ax, param_b]);
        let var_sym = match a.node(x) {
            ExprNode::Symbol(sid) => *sid,
            _ => panic!("x should be a symbol"),
        };
        let result = super::symbolic_linear_coeff_of(&mut a, ax_plus_b, x, var_sym);
        assert!(result.is_some(), "a*x+b should be recognized as linear in x, expr: {}", display(&a, ax_plus_b));
        let (coeff, constant) = result.unwrap();
        assert_eq!(coeff, param_a, "coefficient should be a, got {}", display(&a, coeff));
        assert_eq!(constant, param_b, "constant should be b, got {}", display(&a, constant));
    }

    #[test]
    fn symbolic_linear_coeff_of_bare_var() {
        // x alone should be linear with coefficient 1
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let var_sym = match a.node(x) {
            ExprNode::Symbol(sid) => *sid,
            _ => panic!("x should be a symbol"),
        };
        let result = super::symbolic_linear_coeff_of(&mut a, x, x, var_sym);
        assert!(result.is_some(), "x should be recognized as linear in x");
        let (coeff, constant) = result.unwrap();
        assert_eq!(coeff, a.one, "coefficient should be 1");
        assert_eq!(constant, a.zero, "constant should be 0");
    }

    #[test]
    fn integrate_sin_symbolic_coeff() {
        // ∫ sin(a*x) dx should not be unevaluated
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let param_a = sym(&mut a, "a");
        let ax = a.mul(&[param_a, x]);
        let sin_ax = a.sin(ax);
        let result = integrate(&mut a, sin_ax, x);
        let s = display(&a, result);
        assert!(!s.contains("Integral"), "∫sin(a*x)dx should not be unevaluated: {s}");
        assert!(s.contains("cos"), "should contain cos: {s}");
    }

    #[test]
    fn integrate_exp_symbolic_coeff() {
        // ∫ exp(a*x) dx should not be unevaluated
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let param_a = sym(&mut a, "a");
        let ax = a.mul(&[param_a, x]);
        let exp_ax = a.exp(ax);
        let result = integrate(&mut a, exp_ax, x);
        let s = display(&a, result);
        assert!(!s.contains("Integral"), "∫exp(a*x)dx should not be unevaluated: {s}");
        assert!(s.contains("exp"), "should contain exp: {s}");
    }

    #[test]
    fn integrate_cosh_symbolic_coeff() {
        // ∫ cosh(a*x) dx should not be unevaluated
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let param_a = sym(&mut a, "a");
        let ax = a.mul(&[param_a, x]);
        let cosh_ax = a.cosh(ax);
        let result = integrate(&mut a, cosh_ax, x);
        let s = display(&a, result);
        assert!(!s.contains("Integral"), "∫cosh(a*x)dx should not be unevaluated: {s}");
    }

    // ── Inverse hyperbolic integration tests ───────────────────────

    /// Helper: substitute a rational value for a symbol and evaluate to f64.
    /// Returns None if evaluation fails.
    fn eval_at(arena: &mut Arena, expr: ExprId, var: ExprId, numer: i64, denom: i64) -> Option<f64> {
        let val = arena.rational(numer, denom);
        let substituted = crate::transforms::subs::subs(arena, expr, var, val);
        let evaled = crate::transforms::eval::eval(arena, substituted);
        let s = crate::transforms::evalf::evalf(arena, evaled, 15).ok()?;
        s.parse::<f64>().ok()
    }

    /// Helper: verify FTC at multiple points — d/dx(F(x)) ≈ f(x).
    /// `integrand` is f(x), `antideriv` is F(x) = ∫f(x)dx.
    /// Checks at each test point that |F'(point) - f(point)| < tol.
    fn assert_ftc(
        arena: &mut Arena,
        integrand: ExprId,
        antideriv: ExprId,
        var: ExprId,
        test_points: &[(i64, i64)],
        tol: f64,
        name: &str,
    ) {
        let deriv = crate::transforms::diff::diff(arena, antideriv, var);
        let deriv_simplified = crate::simplify::simplify_engine::smart_simplify(arena, deriv);
        for &(n, d) in test_points {
            let f_val = eval_at(arena, integrand, var, n, d);
            let fp_val = eval_at(arena, deriv_simplified, var, n, d);
            match (f_val, fp_val) {
                (Some(f), Some(fp)) => {
                    assert!(
                        (f - fp).abs() < tol,
                        "FTC failed for {name} at x={n}/{d}: f(x)={f}, F'(x)={fp}, diff={}",
                        (f - fp).abs()
                    );
                }
                _ => {
                    // If numerical eval fails at this point, skip it
                    // (e.g., acosh at x < 1 is undefined)
                }
            }
        }
    }

    #[test]
    fn integrate_asinh_direct() {
        // ∫ asinh(x) dx = x·asinh(x) - √(x²+1)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let asinh_x = a.asinh(x);
        let result = integrate(&mut a, asinh_x, x);

        // Structural: must not be unevaluated
        assert!(
            !matches!(a.node(result), ExprNode::Integral(_, _)),
            "asinh integration should return a closed form, not Integral"
        );
        let s = display(&a, result);
        assert!(s.contains("asinh"), "result should contain asinh: {s}");
        assert!(s.contains("sqrt"), "result should contain sqrt: {s}");

        // Numerical FTC: d/dx(result) ≈ asinh(x) at multiple points
        assert_ftc(
            &mut a, asinh_x, result, x,
            &[(1, 2), (3, 2), (5, 1)],
            1e-8,
            "∫asinh(x)dx",
        );
    }

    #[test]
    fn integrate_acosh_direct() {
        // ∫ acosh(x) dx = x·acosh(x) - √(x²-1)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let acosh_x = a.acosh(x);
        let result = integrate(&mut a, acosh_x, x);

        // Structural: must not be unevaluated
        assert!(
            !matches!(a.node(result), ExprNode::Integral(_, _)),
            "acosh integration should return a closed form, not Integral"
        );
        let s = display(&a, result);
        assert!(s.contains("acosh"), "result should contain acosh: {s}");

        // Numerical FTC: test at x > 1 only (acosh domain)
        assert_ftc(
            &mut a, acosh_x, result, x,
            &[(3, 2), (2, 1), (5, 1)],
            1e-8,
            "∫acosh(x)dx",
        );
    }

    #[test]
    fn integrate_atanh_direct() {
        // ∫ atanh(x) dx = x·atanh(x) + ½·ln(1-x²)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let atanh_x = a.atanh(x);
        let result = integrate(&mut a, atanh_x, x);

        // Structural: must not be unevaluated
        assert!(
            !matches!(a.node(result), ExprNode::Integral(_, _)),
            "atanh integration should return a closed form, not Integral"
        );
        let s = display(&a, result);
        assert!(s.contains("atanh"), "result should contain atanh: {s}");
        assert!(s.contains("ln"), "result should contain ln: {s}");

        // Numerical FTC: test at |x| < 1 only (atanh domain)
        assert_ftc(
            &mut a, atanh_x, result, x,
            &[(1, 4), (1, 2), (3, 4)],
            1e-8,
            "∫atanh(x)dx",
        );
    }
}
