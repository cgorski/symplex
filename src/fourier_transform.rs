//! Symbolic Fourier transform and inverse Fourier transform (table-based).
//!
//! Forward: F{f(t)}(ω) = ∫_{-∞}^{∞} f(t) e^{-iωt} dt
//! Inverse: F⁻¹{F(ω)}(t) = (1/2π) ∫_{-∞}^{∞} F(ω) e^{iωt} dω
//!
//! # Supported transforms (forward)
//!
//! | Time domain              | Frequency domain                    |
//! |--------------------------|-------------------------------------|
//! | δ(t)                     | 1                                   |
//! | 1                        | 2π·δ(ω)                             |
//! | exp(a·t)·H(t)            | 1/(iω − a)                         |
//! | t^n·exp(a·t)·H(t)        | n! / (iω − a)^(n+1)               |
//! | sin(ω₀·t)               | iπ[δ(ω+ω₀) − δ(ω−ω₀)]            |
//! | cos(ω₀·t)               | π[δ(ω−ω₀) + δ(ω+ω₀)]             |
//! | H(t)                     | π·δ(ω) + 1/(iω)                   |
//!
//! Plus linearity (sums, constant factors).

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{Signed, ToPrimitive, Zero};

use crate::arena::Arena;
use crate::errors::SymplexError;
use crate::node::{ExprId, ExprNode, SymbolId};

// ═══════════════════════════════════════════════════════════════════════════
// Forward Fourier transform
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the Fourier transform of `expr` with respect to time variable `t`,
/// producing a function of frequency variable `omega`.
pub(crate) fn fourier_transform(
    arena: &mut Arena,
    expr: ExprId,
    t: ExprId,
    omega: ExprId,
) -> Result<ExprId, SymplexError> {
    let t_sym = match arena.node(t) {
        ExprNode::Symbol(sid) => *sid,
        _ => {
            return Err(SymplexError::ComputationFailed {
                operation: "fourier_transform",
                reason: "t must be a symbol".to_string(),
            });
        }
    };
    match arena.node(omega) {
        ExprNode::Symbol(_) => {}
        _ => {
            return Err(SymplexError::ComputationFailed {
                operation: "fourier_transform",
                reason: "omega must be a symbol".to_string(),
            });
        }
    };

    do_forward(arena, expr, t, t_sym, omega)
}

/// Internal recursive forward transform.
fn do_forward(
    arena: &mut Arena,
    expr: ExprId,
    t: ExprId,
    t_sym: SymbolId,
    omega: ExprId,
) -> Result<ExprId, SymplexError> {
    // ── Linearity: if expr is Add, transform each term ──
    let node = arena.node(expr).clone();
    if let ExprNode::Add(ref children) = node {
        let kids = children.clone();
        let mut terms = Vec::with_capacity(kids.len());
        for child in kids {
            terms.push(do_forward(arena, child, t, t_sym, omega)?);
        }
        return Ok(arena.add(&terms));
    }

    // ── Handle Neg: F{-f} = -F{f} ──
    if let ExprNode::Neg(inner) = node {
        let transformed = do_forward(arena, inner, t, t_sym, omega)?;
        return Ok(arena.neg(transformed));
    }

    // ── Factor out constants (terms not depending on t) ──
    let (coeff, body) = split_independent(arena, expr, t);
    if coeff != arena.one {
        let transformed = do_forward(arena, body, t, t_sym, omega)?;
        return Ok(arena.mul(&[coeff, transformed]));
    }

    // ── Try table rules ──
    if let Some(result) = try_table_forward(arena, expr, t, t_sym, omega) {
        return Ok(result);
    }

    Err(SymplexError::ComputationFailed {
        operation: "fourier_transform",
        reason: "cannot transform expression".to_string(),
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Forward table rules
// ═══════════════════════════════════════════════════════════════════════════

fn try_table_forward(
    arena: &mut Arena,
    expr: ExprId,
    t: ExprId,
    t_sym: SymbolId,
    omega: ExprId,
) -> Option<ExprId> {
    let node = arena.node(expr).clone();

    match node {
        // Rule 0: constant c (no t) → c · 2π · δ(ω)
        _ if !contains_var(arena, expr, t) => {
            let two = arena.int(2);
            let pi = arena.pi;
            let two_pi = arena.mul(&[two, pi]);
            let delta_omega = arena.dirac_delta(omega);
            Some(arena.mul(&[expr, two_pi, delta_omega]))
        }

        // Rule 1: δ(t) → 1
        ExprNode::DiracDelta(inner) if inner == t => Some(arena.one),

        // Rule 1b: δ(a*t) → 1/|a|
        ExprNode::DiracDelta(inner) => {
            if let Some(a) = extract_linear_coeff(arena, inner, t) {
                let abs_a = arena.abs(a);
                Some(arena.div(arena.one, abs_a))
            } else {
                None
            }
        }

        // Rule 2: Heaviside(t) → π·δ(ω) + 1/(iω)
        ExprNode::Heaviside(inner) if inner == t => {
            let pi = arena.pi;
            let delta_omega = arena.dirac_delta(omega);
            let pi_delta = arena.mul(&[pi, delta_omega]);
            let i = arena.i_unit;
            let i_omega = arena.mul(&[i, omega]);
            let recip = arena.div(arena.one, i_omega);
            Some(arena.add(&[pi_delta, recip]))
        }

        // Rule 3: sin(ω₀·t) → iπ[δ(ω+ω₀) − δ(ω−ω₀)]
        ExprNode::Sin(arg) => {
            if let Some(omega0) = extract_linear_coeff(arena, arg, t) {
                let i = arena.i_unit;
                let pi = arena.pi;
                let i_pi = arena.mul(&[i, pi]);
                let omega_plus = arena.add(&[omega, omega0]);
                let omega_minus = arena.sub(omega, omega0);
                let delta_plus = arena.dirac_delta(omega_plus);
                let delta_minus = arena.dirac_delta(omega_minus);
                let diff = arena.sub(delta_plus, delta_minus);
                Some(arena.mul(&[i_pi, diff]))
            } else {
                None
            }
        }

        // Rule 4: cos(ω₀·t) → π[δ(ω−ω₀) + δ(ω+ω₀)]
        ExprNode::Cos(arg) => {
            if let Some(omega0) = extract_linear_coeff(arena, arg, t) {
                let pi = arena.pi;
                let omega_plus = arena.add(&[omega, omega0]);
                let omega_minus = arena.sub(omega, omega0);
                let delta_plus = arena.dirac_delta(omega_plus);
                let delta_minus = arena.dirac_delta(omega_minus);
                let sum = arena.add(&[delta_minus, delta_plus]);
                Some(arena.mul(&[pi, sum]))
            } else {
                None
            }
        }

        // Rule 5: t — F{t} = 2πi·δ'(ω)
        // Cannot represent δ' easily; skip.
        ExprNode::Symbol(sid) if sid == t_sym => None,

        // Rule 6: Mul — look for exp(a*t)·H(t) or t^n·exp(a*t)·H(t)
        ExprNode::Mul(ref children) => {
            try_exp_heaviside_mul(arena, &children.clone(), t, omega)
        }

        _ => None,
    }
}

/// Try to match a Mul node as [t^n ·] exp(a*t) · H(t) [· const_factors].
///
/// F{t^n · exp(a·t) · H(t)} = n! / (iω − a)^{n+1}
fn try_exp_heaviside_mul(
    arena: &mut Arena,
    children: &[ExprId],
    t: ExprId,
    omega: ExprId,
) -> Option<ExprId> {
    let mut has_heaviside = false;
    let mut exp_coeff: Option<ExprId> = None;
    let mut t_power: u64 = 0;
    let mut other_factors: Vec<ExprId> = Vec::new();

    for &child in children {
        let child_node = arena.node(child).clone();
        match child_node {
            ExprNode::Heaviside(inner) if inner == t => {
                has_heaviside = true;
            }
            ExprNode::Exp(arg) => {
                if let Some(a) = extract_linear_coeff(arena, arg, t) {
                    exp_coeff = Some(a);
                } else {
                    // Exp of something we don't understand
                    return None;
                }
            }
            ExprNode::Pow(base, exp) if base == t => {
                if let Some(r) = arena.as_num(exp).cloned() {
                    if r.is_integer() && r.is_positive() {
                        if let Some(n) = r.to_integer().to_u64() {
                            t_power += n;
                        } else {
                            return None;
                        }
                    } else {
                        return None;
                    }
                } else {
                    return None;
                }
            }
            _ if child == t => {
                t_power += 1;
            }
            _ => {
                if !contains_var(arena, child, t) {
                    other_factors.push(child);
                } else {
                    return None; // unrecognized t-dependent factor
                }
            }
        }
    }

    if !has_heaviside {
        return None;
    }

    let a = exp_coeff.unwrap_or(arena.zero);

    // F{t^n · exp(a·t) · H(t)} = n! / (iω − a)^{n+1}
    let i = arena.i_unit;
    let i_omega = arena.mul(&[i, omega]);
    let denom_base = arena.sub(i_omega, a);
    let n_plus_1 = arena.int(t_power as i64 + 1);
    let denom = if t_power == 0 {
        denom_base
    } else {
        arena.pow(denom_base, n_plus_1)
    };

    let numer = factorial_expr(arena, t_power);

    let mut result = arena.div(numer, denom);

    if !other_factors.is_empty() {
        other_factors.push(result);
        result = arena.mul(&other_factors);
    }

    Some(result)
}

// ═══════════════════════════════════════════════════════════════════════════
// Inverse Fourier transform
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the inverse Fourier transform of `expr` (function of `omega`)
/// back to a function of time variable `t`.
///
/// F⁻¹{F(ω)}(t) = (1/2π) ∫_{-∞}^{∞} F(ω) e^{iωt} dω
pub(crate) fn inverse_fourier_transform(
    arena: &mut Arena,
    expr: ExprId,
    omega: ExprId,
    t: ExprId,
) -> Result<ExprId, SymplexError> {
    let omega_sym = match arena.node(omega) {
        ExprNode::Symbol(sid) => *sid,
        _ => {
            return Err(SymplexError::ComputationFailed {
                operation: "inverse_fourier_transform",
                reason: "omega must be a symbol".to_string(),
            });
        }
    };
    match arena.node(t) {
        ExprNode::Symbol(_) => {}
        _ => {
            return Err(SymplexError::ComputationFailed {
                operation: "inverse_fourier_transform",
                reason: "t must be a symbol".to_string(),
            });
        }
    };

    do_inverse(arena, expr, omega, omega_sym, t)
}

fn do_inverse(
    arena: &mut Arena,
    expr: ExprId,
    omega: ExprId,
    omega_sym: SymbolId,
    t: ExprId,
) -> Result<ExprId, SymplexError> {
    // ── Linearity: handle Add ──
    let node = arena.node(expr).clone();
    if let ExprNode::Add(ref children) = node {
        let kids = children.clone();
        let mut terms = Vec::with_capacity(kids.len());
        for child in kids {
            terms.push(do_inverse(arena, child, omega, omega_sym, t)?);
        }
        return Ok(arena.add(&terms));
    }

    // ── Handle Neg ──
    if let ExprNode::Neg(inner) = node {
        let result = do_inverse(arena, inner, omega, omega_sym, t)?;
        return Ok(arena.neg(result));
    }

    // ── Factor out constants ──
    let (coeff, body) = split_independent(arena, expr, omega);
    if coeff != arena.one {
        let result = do_inverse(arena, body, omega, omega_sym, t)?;
        return Ok(arena.mul(&[coeff, result]));
    }

    // ── Table rules ──
    if let Some(result) = try_table_inverse(arena, expr, omega, omega_sym, t) {
        return Ok(result);
    }

    Err(SymplexError::ComputationFailed {
        operation: "inverse_fourier_transform",
        reason: "cannot invert expression".to_string(),
    })
}

fn try_table_inverse(
    arena: &mut Arena,
    expr: ExprId,
    omega: ExprId,
    _omega_sym: SymbolId,
    t: ExprId,
) -> Option<ExprId> {
    let node = arena.node(expr).clone();

    match node {
        // Rule 0: constant (no omega) → c · δ(t)
        // F⁻¹{c} = c·δ(t) (with our convention F⁻¹{1} = δ(t))
        _ if !contains_var(arena, expr, omega) => {
            let delta_t = arena.dirac_delta(t);
            Some(arena.mul(&[expr, delta_t]))
        }

        // Rule 1: δ(ω) → 1/(2π)
        ExprNode::DiracDelta(inner) if inner == omega => {
            let two = arena.int(2);
            let pi = arena.pi;
            let two_pi = arena.mul(&[two, pi]);
            Some(arena.div(arena.one, two_pi))
        }

        // Rule 2: δ(ω − ω₀) → (1/2π)·exp(iω₀t)
        ExprNode::DiracDelta(inner) => {
            // Check if inner = omega - omega0 (i.e. Add([omega, Neg(omega0)]))
            // or inner = omega + omega0
            if let Some(omega0) = extract_delta_shift(arena, inner, omega) {
                let two = arena.int(2);
                let pi = arena.pi;
                let two_pi = arena.mul(&[two, pi]);
                let i = arena.i_unit;
                let i_omega0_t = arena.mul(&[i, omega0, t]);
                let exp_term = arena.exp(i_omega0_t);
                Some(arena.div(exp_term, two_pi))
            } else {
                None
            }
        }

        // Rule 3: 1/(iω − a) → exp(a·t)·H(t)
        // Represented as Pow(iω − a, -1) or as a Mul with Pow
        ExprNode::Pow(base, exp) => {
            if let Some(r) = arena.as_num(exp).cloned()
                && r.is_negative() && r.is_integer()
                && let Some(a) = extract_i_omega_minus_a(arena, base, omega)
            {
                let n_val = (-r.to_integer()).to_u64()?;
                if n_val >= 1 {
                    let at = arena.mul(&[a, t]);
                    let exp_at = arena.exp(at);
                    let heaviside = arena.heaviside(t);

                    if n_val == 1 {
                        // 1/(iω − a) → exp(a·t)·H(t)
                        return Some(arena.mul(&[exp_at, heaviside]));
                    } else {
                        // 1/(iω − a)^n → t^{n-1}·exp(a·t)·H(t) / (n-1)!
                        let n_minus_1 = n_val - 1;
                        let t_pow = if n_minus_1 == 1 {
                            t
                        } else {
                            let exp_id = arena.int(n_minus_1 as i64);
                            arena.pow(t, exp_id)
                        };
                        let fact = factorial_expr(arena, n_minus_1);
                        let numer = arena.mul(&[t_pow, exp_at, heaviside]);
                        return Some(arena.div(numer, fact));
                    }
                }
            }
            None
        }

        _ => None,
    }
}

/// Try to match `base` as `iω − a` and return `a` (using `&mut Arena`).
///
/// Recognizes patterns:
/// - `i*ω` → a = 0
/// - `Add([i*ω, c])` → a = -c  (where c is a constant or expression)
fn extract_i_omega_minus_a(arena: &mut Arena, base: ExprId, omega: ExprId) -> Option<ExprId> {
    // Check for just i*omega (meaning a = 0)
    if is_i_times_omega(arena, base, omega) {
        return Some(arena.zero);
    }

    let node = arena.node(base).clone();
    if let ExprNode::Add(ref children) = node {
        if children.len() != 2 {
            return None;
        }
        let c0 = children[0];
        let c1 = children[1];

        // Try both orderings: (i*omega, -a) or (-a, i*omega)
        if is_i_times_omega(arena, c0, omega) {
            // c1 = -a, so a = -c1
            let a = arena.neg(c1);
            return Some(a);
        }
        if is_i_times_omega(arena, c1, omega) {
            // c0 = -a, so a = -c0
            let a = arena.neg(c0);
            return Some(a);
        }
    }

    None
}

/// Check if `expr` is `i * omega`.
fn is_i_times_omega(arena: &Arena, expr: ExprId, omega: ExprId) -> bool {
    let node = arena.node(expr);
    if let ExprNode::Mul(children) = node
        && children.len() == 2
    {
        return (children[0] == arena.i_unit && children[1] == omega)
            || (children[1] == arena.i_unit && children[0] == omega);
    }
    false
}

/// Given `inner` from `δ(inner)`, try to extract the shift ω₀ such that
/// `inner = ω − ω₀`. Returns ω₀ if found.
fn extract_delta_shift(arena: &mut Arena, inner: ExprId, omega: ExprId) -> Option<ExprId> {
    let node = arena.node(inner).clone();

    // inner = omega + c  →  δ(ω + c) corresponds to ω₀ = -c
    // inner = omega - c  →  δ(ω − c) corresponds to ω₀ = c
    if let ExprNode::Add(ref children) = node {
        if children.len() != 2 {
            return None;
        }
        let c0 = children[0];
        let c1 = children[1];

        if c0 == omega && !contains_var(arena, c1, omega) {
            // inner = omega + c1, so shift is -c1 for the exp(iω₀t) form
            // Actually δ(ω − ω₀) → e^{iω₀t}/(2π)
            // If inner = ω + c1, then ω₀ = −c1
            let omega0 = arena.neg(c1);
            return Some(omega0);
        }
        if c1 == omega && !contains_var(arena, c0, omega) {
            let omega0 = arena.neg(c0);
            return Some(omega0);
        }
    }

    // inner = omega itself → ω₀ = 0
    if inner == omega {
        return Some(arena.zero);
    }

    None
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Check whether `expr` contains the variable `t`.
fn contains_var(arena: &Arena, expr: ExprId, t: ExprId) -> bool {
    crate::walk::contains(arena, expr, t)
}

/// Split `expr` into `(coefficient_independent_of_t, rest_depending_on_t)`.
fn split_independent(arena: &mut Arena, expr: ExprId, t: ExprId) -> (ExprId, ExprId) {
    let node = arena.node(expr).clone();
    if let ExprNode::Mul(ref children) = node {
        let mut indep = Vec::new();
        let mut dep = Vec::new();
        for &child in children {
            if contains_var(arena, child, t) {
                dep.push(child);
            } else {
                indep.push(child);
            }
        }
        if !indep.is_empty() && !dep.is_empty() {
            let coeff = if indep.len() == 1 {
                indep[0]
            } else {
                arena.mul(&indep)
            };
            let body = if dep.len() == 1 {
                dep[0]
            } else {
                arena.mul(&dep)
            };
            return (coeff, body);
        }
    }
    (arena.one, expr)
}

/// Extract the coefficient `a` from an expression of the form `a*t`.
fn extract_linear_coeff(arena: &mut Arena, expr: ExprId, t: ExprId) -> Option<ExprId> {
    if expr == t {
        return Some(arena.one);
    }

    let node = arena.node(expr).clone();
    if let ExprNode::Mul(ref children) = node {
        let mut t_count = 0usize;
        let mut others = Vec::new();
        for &child in children {
            if child == t {
                t_count += 1;
            } else if contains_var(arena, child, t) {
                return None;
            } else {
                others.push(child);
            }
        }
        if t_count == 1 {
            if others.is_empty() {
                return Some(arena.one);
            } else if others.len() == 1 {
                return Some(others[0]);
            } else {
                return Some(arena.mul(&others));
            }
        }
    }

    // Fallback: try polynomial approach
    let poly = crate::polybridge::expr_to_poly(arena, expr, t)?;
    if poly.degree()? != 1 {
        return None;
    }
    let c0 = poly.coeff(0);
    if !c0.is_zero() {
        return None;
    }
    let a = poly.coeff(1);
    if a.is_zero() {
        return None;
    }
    let nid = arena.intern_num(a);
    Some(arena.intern(ExprNode::Num(nid)))
}

/// Build n! as an expression.
fn factorial_expr(arena: &mut Arena, n: u64) -> ExprId {
    let mut result = BigInt::from(1u64);
    for i in 2..=n {
        result *= BigInt::from(i);
    }
    let ratio = Ratio::from_integer(result);
    let nid = arena.intern_num(ratio);
    arena.intern(ExprNode::Num(nid))
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
    fn forward_delta() {
        let mut a = Arena::new();
        let t = sym(&mut a, "t");
        let omega = sym(&mut a, "omega");
        let delta_t = a.dirac_delta(t);
        let result = fourier_transform(&mut a, delta_t, t, omega).unwrap();
        // F{δ(t)} = 1
        assert_eq!(result, a.one, "F{{δ(t)}} should be 1");
    }

    #[test]
    fn forward_constant() {
        let mut a = Arena::new();
        let t = sym(&mut a, "t");
        let omega = sym(&mut a, "omega");
        let five = a.int(5);
        let result = fourier_transform(&mut a, five, t, omega).unwrap();
        let d = display(&a, result);
        // F{5} = 5 · 2π · δ(ω) = 10π·δ(ω)
        assert!(
            d.contains("delta") || d.contains("pi"),
            "F{{5}} should contain delta and pi: {d}"
        );
    }

    #[test]
    fn forward_sin() {
        let mut a = Arena::new();
        let t = sym(&mut a, "t");
        let omega = sym(&mut a, "omega");
        let three = a.int(3);
        let three_t = a.mul(&[three, t]);
        let sin_3t = a.sin(three_t);
        let result = fourier_transform(&mut a, sin_3t, t, omega).unwrap();
        let d = display(&a, result);
        assert!(
            d.contains("DiracDelta") || d.contains("delta"),
            "F{{sin(3t)}} should have delta terms: {d}"
        );
    }

    #[test]
    fn forward_cos() {
        let mut a = Arena::new();
        let t = sym(&mut a, "t");
        let omega = sym(&mut a, "omega");
        let two = a.int(2);
        let two_t = a.mul(&[two, t]);
        let cos_2t = a.cos(two_t);
        let result = fourier_transform(&mut a, cos_2t, t, omega).unwrap();
        let d = display(&a, result);
        assert!(
            d.contains("DiracDelta") || d.contains("delta"),
            "F{{cos(2t)}} should have delta terms: {d}"
        );
    }

    #[test]
    fn forward_exp_heaviside() {
        let mut a = Arena::new();
        let t = sym(&mut a, "t");
        let omega = sym(&mut a, "omega");
        // F{exp(-2t)·H(t)} = 1/(iω − (−2)) = 1/(iω + 2)
        let neg2 = a.int(-2);
        let neg2_t = a.mul(&[neg2, t]);
        let exp_neg2t = a.exp(neg2_t);
        let heaviside_t = a.heaviside(t);
        let expr = a.mul(&[exp_neg2t, heaviside_t]);
        let result = fourier_transform(&mut a, expr, t, omega).unwrap();
        let d = display(&a, result);
        assert!(
            d.contains("omega") && d.contains("I"),
            "F{{exp(-2t)·H(t)}} should contain iω: {d}"
        );
    }

    #[test]
    fn forward_linearity() {
        let mut a = Arena::new();
        let t = sym(&mut a, "t");
        let omega = sym(&mut a, "omega");
        let delta_t = a.dirac_delta(t);
        let three = a.int(3);
        let expr = a.mul(&[three, delta_t]);
        let result = fourier_transform(&mut a, expr, t, omega).unwrap();
        // F{3·δ(t)} = 3
        assert_eq!(display(&a, result), "3", "F{{3·δ(t)}} should be 3");
    }

    #[test]
    fn forward_t_must_be_symbol() {
        let mut a = Arena::new();
        let t = a.int(5); // not a symbol!
        let omega = sym(&mut a, "omega");
        let result = fourier_transform(&mut a, t, t, omega);
        assert!(result.is_err());
    }

    #[test]
    fn forward_heaviside() {
        let mut a = Arena::new();
        let t = sym(&mut a, "t");
        let omega = sym(&mut a, "omega");
        let heaviside_t = a.heaviside(t);
        let result = fourier_transform(&mut a, heaviside_t, t, omega).unwrap();
        let d = display(&a, result);
        // F{H(t)} = π·δ(ω) + 1/(iω)
        assert!(
            (d.contains("DiracDelta") || d.contains("delta")) && d.contains("pi"),
            "F{{H(t)}} should contain π·δ(ω): {d}"
        );
    }

    #[test]
    fn inverse_constant() {
        let mut a = Arena::new();
        let t = sym(&mut a, "t");
        let omega = sym(&mut a, "omega");
        let three = a.int(3);
        let result = inverse_fourier_transform(&mut a, three, omega, t).unwrap();
        let d = display(&a, result);
        // F⁻¹{3} = 3·δ(t)
        assert!(
            d.contains("DiracDelta") || d.contains("delta"),
            "F⁻¹{{3}} should contain δ(t): {d}"
        );
    }

    #[test]
    fn inverse_delta_omega() {
        let mut a = Arena::new();
        let t = sym(&mut a, "t");
        let omega = sym(&mut a, "omega");
        let delta_omega = a.dirac_delta(omega);
        let result = inverse_fourier_transform(&mut a, delta_omega, omega, t).unwrap();
        let d = display(&a, result);
        // F⁻¹{δ(ω)} = 1/(2π)
        assert!(
            d.contains("pi"),
            "F⁻¹{{δ(ω)}} should be 1/(2π): {d}"
        );
    }
}
