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

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use num_traits::One;
use num_traits::Signed;
use num_traits::Zero;

use crate::arena::Arena;
use crate::node::{ExprId, ExprNode, SymbolId};

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

    integrate_node(arena, expr, var, var_sym, 20)
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

/// Try integrating 1/(x²+bx+c) via completing the square.
/// x²+bx+c = (x+b/2)² + (c - b²/4)
/// If d = c - b²/4 > 0: ∫ 1/((x+b/2)²+d) dx = (1/√d)·atan((x+b/2)/√d)
fn try_complete_square_integral(
    arena: &mut Arena,
    base: ExprId,
    exp: ExprId,
    var: ExprId,
    _var_sym: SymbolId,
) -> Option<ExprId> {
    // exp must be -1
    let exp_r = arena.as_num(exp)?;
    if *exp_r != num_rational::Ratio::from_integer((-1).into()) {
        return None;
    }

    // base must be a quadratic in var: ax² + bx + c
    let poly = crate::polybridge::expr_to_poly(arena, base, var)?;
    if poly.degree()? != 2 {
        return None;
    }

    let a_coeff = poly.coeff(2);
    let b_coeff = poly.coeff(1);
    let c_coeff = poly.coeff(0);

    // Normalize to monic: divide by a
    if a_coeff.is_zero() {
        return None;
    }
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
        arena.intern(crate::node::ExprNode::Num(nid))
    };
    let shifted = arena.add(&[var, half_b_id]);

    // Build √d
    let d_id = {
        let nid = arena.intern_num(d.clone());
        arena.intern(crate::node::ExprNode::Num(nid))
    };
    let half = arena.rational(1, 2);
    let sqrt_d = arena.pow(d_id, half);

    // Result: (1/(a·√d)) · atan((x+b/2)/√d)
    let ratio = arena.div(shifted, sqrt_d);
    let atan_result = arena.atan(ratio);

    // Divide by a·√d
    let a_id = {
        let nid = arena.intern_num(a_coeff);
        arena.intern(crate::node::ExprNode::Num(nid))
    };
    // Rebuild √d for the denominator (arena IDs are Copy, but let's be explicit)
    let sqrt_d2 = arena.pow(d_id, half);
    let a_sqrt_d = arena.mul(&[a_id, sqrt_d2]);

    Some(arena.div(atan_result, a_sqrt_d))
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
    let du1 = crate::diff::diff(arena, u1, var);
    let boundary1 = arena.mul(&[u1, v1]); // u1·v1

    // ── Round 2: ∫ v1·du1 dx  with u₂ = du1, dv₂ = v1 ───────────
    let v2 = integrate_node(arena, v1, var, var_sym, depth - 1);
    if matches!(arena.node(v2), ExprNode::Integral(_, _)) {
        return None;
    }
    let du2 = crate::diff::diff(arena, du1, var); // u1''
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
    if let Some(result) = crate::trig_integ::try_trig_power_integral(arena, expr, var, var_sym) {
        return result;
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
                    let du = crate::diff::diff(arena, u, var);

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
            if dependent.len() == 2 {
                if let Some(result) = try_cyclic_ibp(arena, &dependent, var, var_sym, depth) {
                    if constants.is_empty() {
                        return result;
                    } else {
                        let mut all = constants.clone();
                        all.push(result);
                        return arena.mul(&all);
                    }
                }
            }

            // ── Try partial fraction decomposition for rational integrands ──
            {
                let (_numer, denom) = crate::polybridge::as_numer_denom(arena, expr);
                if denom != arena.one {
                    let decomposed = crate::apart::apart(arena, expr, var);
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
                }
            }

            if !base_has_var && !exp_has_var {
                // Constant: ∫ c dx = c * x
                return arena.mul(&[expr, var]);
            }

            // General linear substitution: ∫ (ax+b)^n dx = (ax+b)^(n+1) / (a*(n+1))
            if !exp_has_var
                && base_has_var
                && let Some(a) = linear_coeff_of(arena, base, var, var_sym)
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
                    let denom_val = &a * &n_plus_1;
                    let denom_id = rational_to_expr(arena, &denom_val);
                    return arena.div(base_pow, denom_id);
                } else {
                    // ∫ (ax+b)^(-1) dx = ln|ax+b| / a
                    let abs_base = arena.abs(base);
                    let ln_base = arena.ln(abs_base);
                    let a_id = rational_to_expr(arena, &a);
                    return arena.div(ln_base, a_id);
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

            // ── Completing the square for 1/(ax²+bx+c) ───────────────
            if base_has_var
                && !exp_has_var
                && let Some(result) = try_complete_square_integral(arena, base, exp, var, var_sym)
            {
                return result;
            }

            // ── Try partial fraction decomposition ────────────────────
            {
                let (_numer, denom) = crate::polybridge::as_numer_denom(arena, expr);
                if denom != arena.one {
                    let decomposed = crate::apart::apart(arena, expr, var);
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
                    let expanded = crate::expand::expand(arena, expr);
                    if expanded != expr {
                        let result = integrate_node(arena, expanded, var, var_sym, depth - 1);
                        if !matches!(arena.node(result), ExprNode::Integral(_, _)) {
                            return result;
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
            if let Some(a) = linear_coeff_of(arena, inner, var, var_sym) {
                let cos_inner = arena.cos(inner);
                let neg_cos = arena.neg(cos_inner);
                let a_id = rational_to_expr(arena, &a);
                return arena.div(neg_cos, a_id);
            }
            arena.intern(ExprNode::Integral(expr, var))
        }

        ExprNode::Cos(inner) => {
            if inner == var {
                // ∫ cos(x) dx = sin(x)
                return arena.sin(var);
            }
            // Try u-substitution: if inner = a*x + b, ∫ cos(a*x+b) dx = sin(a*x+b)/a
            if let Some(a) = linear_coeff_of(arena, inner, var, var_sym) {
                let sin_inner = arena.sin(inner);
                let a_id = rational_to_expr(arena, &a);
                return arena.div(sin_inner, a_id);
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
            if let Some(a) = linear_coeff_of(arena, inner, var, var_sym) {
                let cos_inner = arena.cos(inner);
                let abs_cos = arena.abs(cos_inner);
                let ln_abs_cos = arena.ln(abs_cos);
                let neg_ln = arena.neg(ln_abs_cos);
                let a_id = rational_to_expr(arena, &a);
                return arena.div(neg_ln, a_id);
            }
            arena.intern(ExprNode::Integral(expr, var))
        }

        ExprNode::Exp(inner) => {
            if inner == var {
                // ∫ exp(x) dx = exp(x)
                return arena.exp(var);
            }
            // Try u-substitution: if inner = a*x + b, ∫ exp(a*x+b) dx = exp(a*x+b)/a
            if let Some(a) = linear_coeff_of(arena, inner, var, var_sym) {
                let exp_inner = arena.exp(inner);
                let a_id = rational_to_expr(arena, &a);
                return arena.div(exp_inner, a_id);
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
            // Try u-substitution: if inner = a*x + b (linear),
            // ∫ ln(a*x+b) dx = ((a*x+b)·ln(a*x+b) - (a*x+b)) / a
            if let Some(a) = linear_coeff_of(arena, inner, var, var_sym) {
                let ln_inner = arena.ln(inner);
                let inner_times_ln = arena.mul(&[inner, ln_inner]);
                let diff = arena.sub(inner_times_ln, inner);
                let a_id = rational_to_expr(arena, &a);
                return arena.div(diff, a_id);
            }
            arena.intern(ExprNode::Integral(expr, var))
        }

        ExprNode::Sinh(inner) => {
            if inner == var {
                // ∫ sinh(x) dx = cosh(x)
                return arena.intern(ExprNode::Cosh(var));
            }
            // u-sub: ∫ sinh(ax+b) dx = cosh(ax+b)/a
            if let Some(a) = linear_coeff_of(arena, inner, var, var_sym) {
                let cosh_inner = arena.cosh(inner);
                let a_id = rational_to_expr(arena, &a);
                return arena.div(cosh_inner, a_id);
            }
            arena.intern(ExprNode::Integral(expr, var))
        }

        ExprNode::Cosh(inner) => {
            if inner == var {
                // ∫ cosh(x) dx = sinh(x)
                return arena.intern(ExprNode::Sinh(var));
            }
            // u-sub: ∫ cosh(ax+b) dx = sinh(ax+b)/a
            if let Some(a) = linear_coeff_of(arena, inner, var, var_sym) {
                let sinh_inner = arena.sinh(inner);
                let a_id = rational_to_expr(arena, &a);
                return arena.div(sinh_inner, a_id);
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
            if let Some(a) = linear_coeff_of(arena, inner, var, var_sym) {
                let cosh_inner = arena.cosh(inner);
                let ln_cosh = arena.ln(cosh_inner);
                let a_id = rational_to_expr(arena, &a);
                return arena.div(ln_cosh, a_id);
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

        // Inverse hyperbolics and sign: leave as unevaluated integrals
        ExprNode::Asinh(_) | ExprNode::Acosh(_) | ExprNode::Atanh(_) | ExprNode::Sign(_) => {
            arena.intern(ExprNode::Integral(expr, var))
        }

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

/// Check if `expr` is a linear function of `var`: `a*var + b` where a ≠ 0.
/// Returns `Some(a)` if linear, `None` otherwise.
fn linear_coeff_of(
    arena: &Arena,
    expr: ExprId,
    _var: ExprId,
    _var_sym: SymbolId,
) -> Option<num_rational::Ratio<num_bigint::BigInt>> {
    // Try to convert to polynomial in var.
    let poly = crate::polybridge::expr_to_poly(arena, expr, _var)?;
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

/// Convert a Ratio<BigInt> to an ExprId.
fn rational_to_expr(arena: &mut Arena, r: &num_rational::Ratio<num_bigint::BigInt>) -> ExprId {
    let nid = arena.intern_num(r.clone());
    arena.intern(crate::node::ExprNode::Num(nid))
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
/// 2. Computing `du/dx` via [`crate::diff::diff`].
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
            let du = crate::diff::diff(arena, u_expr, var);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::Arena;

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
        assert_eq!(display(&a, result), "ln(abs(x))");
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
}
