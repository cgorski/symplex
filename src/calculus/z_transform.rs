//! Z-transform and inverse z-transform for discrete-time signal analysis.
//!
//! Table-based implementation covering common discrete-time sequences.
//! The z-transform of a sequence x\[n\] is X(z) = Σ x\[n\] z⁻ⁿ.
//!
//! # Supported transforms (forward)
//!
//! | Time domain       | Z domain                                  |
//! |---|---|
//! | c (constant)      | c·z/(z−1)                                 |
//! | aⁿ                | z/(z−a)                                   |
//! | n·aⁿ              | a·z/(z−a)²                                |
//! | sin(ωn)           | z·sin(ω) / (z²−2z·cos(ω)+1)              |
//! | cos(ωn)           | z·(z−cos(ω)) / (z²−2z·cos(ω)+1)          |
//! | aⁿ·sin(ωn)        | a·z·sin(ω) / (z²−2a·z·cos(ω)+a²)        |
//! | aⁿ·cos(ωn)        | z·(z−a·cos(ω)) / (z²−2a·z·cos(ω)+a²)    |
//!
//! Plus linearity (sum of terms) and constant factor extraction.

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Signed, Zero};

use crate::base::arena::Arena;
use crate::base::errors::SymplexError;
use crate::base::node::{ExprId, ExprNode, SymbolId};
use crate::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Forward Z-transform
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the z-transform of a discrete-time expression.
///
/// Transforms x(n) → X(z) using a table of known transforms.
///
/// Returns the z-domain expression, or an error if no table entry matches.
pub(crate) fn z_transform(
    arena: &mut Arena,
    expr: ExprId,
    n_var: ExprId,
    z_var: ExprId,
) -> Result<ExprId, SymplexError> {
    let n_sym = match arena.node(n_var) {
        ExprNode::Symbol(sid) => *sid,
        _ => {
            return Err(SymplexError::ComputationFailed {
                operation: "z_transform",
                reason: "n must be a symbol".to_string(),
            });
        }
    };
    match arena.node(z_var) {
        ExprNode::Symbol(_) => {}
        _ => {
            return Err(SymplexError::ComputationFailed {
                operation: "z_transform",
                reason: "z must be a symbol".to_string(),
            });
        }
    };

    do_forward(arena, expr, n_var, n_sym, z_var)
}

/// Internal recursive forward transform.
fn do_forward(
    arena: &mut Arena,
    expr: ExprId,
    n_var: ExprId,
    n_sym: SymbolId,
    z_var: ExprId,
) -> Result<ExprId, SymplexError> {
    // ── Linearity: if expr is Add, transform each term ──
    let node = arena.node(expr).clone();
    if let ExprNode::Add(ref children) = node {
        let kids = children.clone();
        let mut terms = Vec::with_capacity(kids.len());
        for child in kids {
            terms.push(do_forward(arena, child, n_var, n_sym, z_var)?);
        }
        return Ok(arena.add(&terms));
    }

    // ── Handle Neg: Z{-f} = -Z{f} ──
    if let ExprNode::Neg(inner) = node {
        let transformed = do_forward(arena, inner, n_var, n_sym, z_var)?;
        return Ok(arena.neg(transformed));
    }

    // ── Factor out constants (terms not depending on n) ──
    let (coeff, body) = split_independent(arena, expr, n_var);
    if coeff != arena.one {
        let transformed = do_forward(arena, body, n_var, n_sym, z_var)?;
        return Ok(arena.mul(&[coeff, transformed]));
    }

    // ── Try table rules ──
    if let Some(result) = try_table_forward(arena, expr, n_var, n_sym, z_var) {
        return Ok(result);
    }

    Err(SymplexError::ComputationFailed {
        operation: "z_transform",
        reason: "cannot transform expression".to_string(),
    })
}

// ─── Helpers ─────────────────────────────────────────────────────────────

/// Check whether `expr` contains the variable `var`.
fn contains_var(arena: &Arena, expr: ExprId, var: ExprId) -> bool {
    crate::base::walk::contains(arena, expr, var)
}

/// Split `expr` into `(coefficient_independent_of_var, rest_depending_on_var)`.
///
/// If expr is a `Mul` node, factors not containing `var` are pulled out.
/// Otherwise returns `(1, expr)`.
fn split_independent(arena: &mut Arena, expr: ExprId, var: ExprId) -> (ExprId, ExprId) {
    let node = arena.node(expr).clone();
    if let ExprNode::Mul(ref children) = node {
        let mut indep = Vec::new();
        let mut dep = Vec::new();
        for &child in children {
            if contains_var(arena, child, var) {
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

/// Given an expression that should be of the form `a*n` (or just `n`),
/// extract the coefficient `a`.
///
/// Returns `None` if the expression is not linear in `n`.
fn extract_linear_coeff(arena: &mut Arena, expr: ExprId, n_var: ExprId) -> Option<ExprId> {
    if expr == n_var {
        return Some(arena.one);
    }

    let node = arena.node(expr).clone();
    if let ExprNode::Mul(ref children) = node {
        let mut n_count = 0usize;
        let mut others = Vec::new();
        for &child in children {
            if child == n_var {
                n_count += 1;
            } else if contains_var(arena, child, n_var) {
                return None;
            } else {
                others.push(child);
            }
        }
        if n_count == 1 {
            if others.is_empty() {
                return Some(arena.one);
            } else if others.len() == 1 {
                return Some(others[0]);
            } else {
                return Some(arena.mul(&others));
            }
        }
    }

    // Fallback: polynomial approach
    let poly = crate::poly::polybridge::expr_to_poly(arena, expr, n_var)?;
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

/// Convert a `Ratio<BigInt>` to an `ExprId`.
fn rational_to_expr(arena: &mut Arena, r: &Ratio<BigInt>) -> ExprId {
    let nid = arena.intern_num(r.clone());
    arena.intern(ExprNode::Num(nid))
}

/// Build the denominator z² − 2·c·z + d  where c and d are pre-built ExprIds.
/// Returns the denominator ExprId.
fn build_quadratic_denom(
    arena: &mut Arena,
    z_var: ExprId,
    two_coeff_z: ExprId,
    constant_term: ExprId,
) -> ExprId {
    let z_sq = arena.mul(&[z_var, z_var]);
    let neg_mid = arena.neg(two_coeff_z);
    arena.add(&[z_sq, neg_mid, constant_term])
}

// ─── Forward table rules ─────────────────────────────────────────────────

fn try_table_forward(
    arena: &mut Arena,
    expr: ExprId,
    n_var: ExprId,
    n_sym: SymbolId,
    z_var: ExprId,
) -> Option<ExprId> {
    let node = arena.node(expr).clone();

    // ── Rule 0: constant c (no n) → c·z/(z−1) ──
    if !contains_var(arena, expr, n_var) {
        let numer = arena.mul(&[expr, z_var]);
        let z_minus_1 = arena.sub(z_var, arena.one);
        return Some(arena.div(numer, z_minus_1));
    }

    match node {
        // ── Rule 1: n itself → z/(z−1)² ──
        ExprNode::Symbol(sid) if sid == n_sym => {
            let z_minus_1 = arena.sub(z_var, arena.one);
            let two = arena.int(2);
            let denom = arena.pow(z_minus_1, two);
            Some(arena.div(z_var, denom))
        }

        // ── Rule 2: a^n → z/(z−a) ──
        ExprNode::Pow(base, exp) if exp == n_var && !contains_var(arena, base, n_var) => {
            let z_minus_a = arena.sub(z_var, base);
            Some(arena.div(z_var, z_minus_a))
        }

        // ── Rule 3: sin(ω·n) → z·sin(ω)/(z²−2z·cos(ω)+1) ──
        ExprNode::Sin(arg) => {
            if let Some(omega) = extract_linear_coeff(arena, arg, n_var) {
                let sin_omega = arena.sin(omega);
                let cos_omega = arena.cos(omega);
                let two = arena.int(2);

                // numerator: z·sin(ω)
                let numer = arena.mul(&[z_var, sin_omega]);

                // denominator: z² − 2z·cos(ω) + 1
                let two_z_cos = arena.mul(&[two, z_var, cos_omega]);
                let one = arena.one;
                let denom = build_quadratic_denom(arena, z_var, two_z_cos, one);

                return Some(arena.div(numer, denom));
            }
            None
        }

        // ── Rule 4: cos(ω·n) → z·(z−cos(ω))/(z²−2z·cos(ω)+1) ──
        ExprNode::Cos(arg) => {
            if let Some(omega) = extract_linear_coeff(arena, arg, n_var) {
                let cos_omega = arena.cos(omega);
                let two = arena.int(2);

                // numerator: z·(z − cos(ω))
                let z_minus_cos = arena.sub(z_var, cos_omega);
                let numer = arena.mul(&[z_var, z_minus_cos]);

                // denominator: z² − 2z·cos(ω) + 1
                let two_z_cos = arena.mul(&[two, z_var, cos_omega]);
                let one = arena.one;
                let denom = build_quadratic_denom(arena, z_var, two_z_cos, one);

                return Some(arena.div(numer, denom));
            }
            None
        }

        // ── Rule 5: Mul — check for compound patterns ──
        ExprNode::Mul(ref children) => {
            let kids = children.clone();
            try_mul_patterns(arena, &kids, n_var, n_sym, z_var)
        }

        _ => None,
    }
}

/// Match compound patterns inside a Mul node:
///   - n · a^n  → a·z/(z−a)²
///   - a^n · sin(ω·n) → a·z·sin(ω)/(z²−2a·z·cos(ω)+a²)
///   - a^n · cos(ω·n) → z·(z−a·cos(ω))/(z²−2a·z·cos(ω)+a²)
fn try_mul_patterns(
    arena: &mut Arena,
    children: &[ExprId],
    n_var: ExprId,
    n_sym: SymbolId,
    z_var: ExprId,
) -> Option<ExprId> {
    // Separate factors into categories:
    //   - pow_base: base `a` from a^n factors (exactly one expected)
    //   - has_n: whether bare n_var is among the factors
    //   - trig_kind: sin or cos with its omega argument
    //   - constants: factors independent of n

    let mut pow_base: Option<ExprId> = None;
    let mut has_n = false;
    let mut trig: Option<(TrigKind, ExprId)> = None; // (kind, omega)
    let mut constants: Vec<ExprId> = Vec::new();
    let mut unmatched = false;

    for &child in children {
        let cnode = arena.node(child).clone();
        match cnode {
            // bare n
            ExprNode::Symbol(sid) if sid == n_sym => {
                if has_n {
                    // n² or higher — not supported
                    unmatched = true;
                    break;
                }
                has_n = true;
            }
            // a^n
            ExprNode::Pow(base, exp) if exp == n_var && !contains_var(arena, base, n_var) => {
                if pow_base.is_some() {
                    unmatched = true;
                    break;
                }
                pow_base = Some(base);
            }
            // sin(ω·n)
            ExprNode::Sin(arg) => {
                if let Some(omega) = extract_linear_coeff(arena, arg, n_var) {
                    if trig.is_some() {
                        unmatched = true;
                        break;
                    }
                    trig = Some((TrigKind::Sin, omega));
                } else {
                    unmatched = true;
                    break;
                }
            }
            // cos(ω·n)
            ExprNode::Cos(arg) => {
                if let Some(omega) = extract_linear_coeff(arena, arg, n_var) {
                    if trig.is_some() {
                        unmatched = true;
                        break;
                    }
                    trig = Some((TrigKind::Cos, omega));
                } else {
                    unmatched = true;
                    break;
                }
            }
            // constant factor (no n)
            _ if !contains_var(arena, child, n_var) => {
                constants.push(child);
            }
            // anything else depending on n that we don't recognize
            _ => {
                unmatched = true;
                break;
            }
        }
    }

    if unmatched {
        return None;
    }

    let const_factor = if constants.is_empty() {
        arena.one
    } else if constants.len() == 1 {
        constants[0]
    } else {
        arena.mul(&constants)
    };

    // ── Pattern: n · a^n → a·z/(z−a)² ──
    if has_n
        && let Some(a) = pow_base
        && trig.is_none()
    {
        let z_minus_a = arena.sub(z_var, a);
        let two = arena.int(2);
        let denom = arena.pow(z_minus_a, two);
        let numer = arena.mul(&[a, z_var]);
        let result = arena.div(numer, denom);
        if const_factor != arena.one {
            return Some(arena.mul(&[const_factor, result]));
        }
        return Some(result);
    }

    // ── Pattern: n alone (inside a Mul with only constants) ──
    if has_n && pow_base.is_none() && trig.is_none() {
        // n → z/(z−1)²
        let z_minus_1 = arena.sub(z_var, arena.one);
        let two = arena.int(2);
        let denom = arena.pow(z_minus_1, two);
        let result = arena.div(z_var, denom);
        if const_factor != arena.one {
            return Some(arena.mul(&[const_factor, result]));
        }
        return Some(result);
    }

    // ── Pattern: a^n · sin(ω·n) → a·z·sin(ω)/(z²−2a·z·cos(ω)+a²) ──
    if let Some((TrigKind::Sin, omega)) = trig
        && !has_n
    {
        let a = pow_base.unwrap_or(arena.one);
        let sin_omega = arena.sin(omega);
        let cos_omega = arena.cos(omega);
        let two = arena.int(2);

        // numerator: a·z·sin(ω)
        let numer = arena.mul(&[a, z_var, sin_omega]);

        // denominator: z² − 2a·z·cos(ω) + a²
        let two_az_cos = arena.mul(&[two, a, z_var, cos_omega]);
        let a_sq = arena.mul(&[a, a]);
        let denom = build_quadratic_denom(arena, z_var, two_az_cos, a_sq);

        let result = arena.div(numer, denom);
        if const_factor != arena.one {
            return Some(arena.mul(&[const_factor, result]));
        }
        return Some(result);
    }

    // ── Pattern: a^n · cos(ω·n) → z·(z−a·cos(ω))/(z²−2a·z·cos(ω)+a²) ──
    if let Some((TrigKind::Cos, omega)) = trig
        && !has_n
    {
        let a = pow_base.unwrap_or(arena.one);
        let cos_omega = arena.cos(omega);
        let two = arena.int(2);

        // numerator: z·(z − a·cos(ω))
        let a_cos = arena.mul(&[a, cos_omega]);
        let z_minus_a_cos = arena.sub(z_var, a_cos);
        let numer = arena.mul(&[z_var, z_minus_a_cos]);

        // denominator: z² − 2a·z·cos(ω) + a²
        let two_az_cos = arena.mul(&[two, a, z_var, cos_omega]);
        let a_sq = arena.mul(&[a, a]);
        let denom = build_quadratic_denom(arena, z_var, two_az_cos, a_sq);

        let result = arena.div(numer, denom);
        if const_factor != arena.one {
            return Some(arena.mul(&[const_factor, result]));
        }
        return Some(result);
    }

    None
}

#[derive(Clone, Copy)]
enum TrigKind {
    Sin,
    Cos,
}

// ═══════════════════════════════════════════════════════════════════════════
// Inverse Z-transform
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the inverse z-transform via partial fraction decomposition.
///
/// Transforms X(z) → x(n) by:
/// 1. Attempting direct table lookup
/// 2. Partial fraction decomposition
/// 3. Table lookup for each term
pub(crate) fn inverse_z_transform(
    arena: &mut Arena,
    expr: ExprId,
    z_var: ExprId,
    n_var: ExprId,
) -> Result<ExprId, SymplexError> {
    match arena.node(z_var) {
        ExprNode::Symbol(_) => {}
        _ => {
            return Err(SymplexError::ComputationFailed {
                operation: "inverse_z_transform",
                reason: "z must be a symbol".to_string(),
            });
        }
    };
    match arena.node(n_var) {
        ExprNode::Symbol(_) => {}
        _ => {
            return Err(SymplexError::ComputationFailed {
                operation: "inverse_z_transform",
                reason: "n must be a symbol".to_string(),
            });
        }
    };

    do_inverse(arena, expr, z_var, n_var, 0)
}

/// Internal recursive inverse transform with a recursion guard.
fn do_inverse(
    arena: &mut Arena,
    expr: ExprId,
    z_var: ExprId,
    n_var: ExprId,
    depth: u32,
) -> Result<ExprId, SymplexError> {
    if depth > 20 {
        return Err(SymplexError::ComputationFailed {
            operation: "inverse_z_transform",
            reason: "recursion limit reached".to_string(),
        });
    }

    // ── Linearity: handle Add ──
    let node = arena.node(expr).clone();
    if let ExprNode::Add(ref children) = node {
        let kids = children.clone();
        let mut terms = Vec::with_capacity(kids.len());
        for child in kids {
            terms.push(do_inverse(arena, child, z_var, n_var, depth + 1)?);
        }
        return Ok(arena.add(&terms));
    }

    // ── Handle Neg: Z⁻¹{-F} = -Z⁻¹{F} ──
    if let ExprNode::Neg(inner) = node {
        let result = do_inverse(arena, inner, z_var, n_var, depth + 1)?;
        return Ok(arena.neg(result));
    }

    // ── Factor out constants (not containing z) ──
    let (coeff, body) = split_independent(arena, expr, z_var);
    if coeff != arena.one {
        let result = do_inverse(arena, body, z_var, n_var, depth + 1)?;
        return Ok(arena.mul(&[coeff, result]));
    }

    // ── Try table lookup ──
    if let Some(result) = try_table_inverse(arena, expr, z_var, n_var) {
        return Ok(result);
    }

    // ── Try partial fraction decomposition ──
    let decomposed = crate::transforms::apart::apart(arena, expr, z_var);
    if decomposed != expr {
        return do_inverse(arena, decomposed, z_var, n_var, depth + 1);
    }

    Err(SymplexError::ComputationFailed {
        operation: "inverse_z_transform",
        reason: "cannot invert expression".to_string(),
    })
}

// ─── Inverse table rules ─────────────────────────────────────────────────

fn try_table_inverse(
    arena: &mut Arena,
    expr: ExprId,
    z_var: ExprId,
    n_var: ExprId,
) -> Option<ExprId> {
    // Decompose as numerator / denominator
    let (numer, denom) = crate::poly::polybridge::as_numer_denom(arena, expr);

    // If denominator is 1, check for trivial cases
    if denom == arena.one {
        // A constant (no z) is not a standard Z-domain form.
        return None;
    }

    // Try to convert denominator to a polynomial in z
    let denom_poly = crate::poly::polybridge::expr_to_poly(arena, denom, z_var)?;
    let deg = denom_poly.degree()?;

    // ── Degree 1 denominator: c₁·z + c₀ ──
    if deg == 1 {
        return inverse_degree1(arena, numer, &denom_poly, z_var, n_var);
    }

    // ── Degree 2 denominator ──
    if deg == 2 {
        return inverse_degree2(arena, numer, &denom_poly, z_var, n_var);
    }

    None
}

/// Inverse for degree-1 denominator.
///
/// Forms: N(z) / (c₁·z + c₀) where we look at what N(z) is.
///
/// Key transforms:
///   z/(z−a) → aⁿ  so  z / (z−a) with N=z, denom=(z−a)
///   c·z/(z−1) → c  (unit step scaled by c)
fn inverse_degree1(
    arena: &mut Arena,
    numer: ExprId,
    denom_poly: &crate::poly::Poly,
    z_var: ExprId,
    n_var: ExprId,
) -> Option<ExprId> {
    let c0 = denom_poly.coeff(0);
    let c1 = denom_poly.coeff(1);
    if c1.is_zero() {
        return None;
    }

    // The root of the denominator: z = -c₀/c₁ = a
    let a_rat = -&c0 / &c1;

    // Try to express numerator as a polynomial in z
    let numer_poly = crate::poly::polybridge::expr_to_poly(arena, numer, z_var)?;
    let numer_deg = numer_poly.degree()?;

    // ── Form: z / (c₁·(z − a))  →  (1/c₁)·aⁿ ──
    // Numerator is degree 1: n₁·z + n₀
    if numer_deg == 1 {
        let n0 = numer_poly.coeff(0);
        let n1 = numer_poly.coeff(1);

        // If n₀ = 0, this is n₁·z / (c₁·z + c₀) = (n₁/c₁) · z/(z−a) → (n₁/c₁)·aⁿ
        if n0.is_zero() {
            let scale = &n1 / &c1;
            let a_id = rational_to_expr(arena, &a_rat);
            let result = arena.pow(a_id, n_var);
            if scale.is_one() {
                return Some(result);
            }
            let scale_id = rational_to_expr(arena, &scale);
            return Some(arena.mul(&[scale_id, result]));
        }
    }

    // ── Form: constant / (c₁·z + c₀) ──
    // Skip non-standard forms for now.
    if numer_deg == 0 {
        let _n0 = numer_poly.coeff(0);
        // 1/(z-a) = z^{-1} · 1/(1 - a·z^{-1}), which maps to a^{n-1}·u[n-1].
        // This is a one-sided form that requires careful handling. Skip.
        return None;
    }

    None
}

/// Inverse for degree-2 denominator.
///
/// Key transforms:
///   z/(z−a)²  → n·aⁿ⁻¹
///   a·z/(z−a)² → n·aⁿ
///   z·sin(ω)/(z²−2z·cos(ω)+1) → sin(ωn)
///   z·(z−cos(ω))/(z²−2z·cos(ω)+1) → cos(ωn)
fn inverse_degree2(
    arena: &mut Arena,
    numer: ExprId,
    denom_poly: &crate::poly::Poly,
    z_var: ExprId,
    n_var: ExprId,
) -> Option<ExprId> {
    let c0 = denom_poly.coeff(0);
    let c1 = denom_poly.coeff(1);
    let c2 = denom_poly.coeff(2);

    if c2.is_zero() {
        return None;
    }

    // Normalize: divide through by c₂ so denominator = z² + (c₁/c₂)·z + (c₀/c₂)
    let p = &c1 / &c2; // coefficient of z
    let q = &c0 / &c2; // constant term

    // Check if this is a perfect square: (z − a)² = z² − 2a·z + a²
    // That means p = −2a, q = a², so a = −p/2 and a² should equal q.
    let two = Ratio::from_integer(BigInt::from(2));
    let candidate_a = -&p / &two;
    let a_sq = &candidate_a * &candidate_a;

    if a_sq == q {
        // Denominator is c₂·(z − a)²
        // Look for forms: N(z) / (c₂·(z − a)²)
        let numer_poly = crate::poly::polybridge::expr_to_poly(arena, numer, z_var)?;
        let numer_deg = numer_poly.degree()?;

        if numer_deg == 1 {
            let n0 = numer_poly.coeff(0);
            let n1 = numer_poly.coeff(1);

            // Form: n₁·z / (c₂·(z−a)²) when n₀ = 0
            // z/(z−a)² → n·a^{n−1}
            // So n₁·z / (c₂·(z−a)²) → (n₁/c₂)·n·a^{n−1}
            if n0.is_zero() {
                let scale = &n1 / &c2;
                let a_id = rational_to_expr(arena, &candidate_a);

                if candidate_a.is_zero() {
                    // z/(z−0)² = 1/z → δ[n−1], skip
                    return None;
                }

                // n·a^{n−1}
                let n_minus_1 = arena.sub(n_var, arena.one);
                let a_pow_nm1 = arena.pow(a_id, n_minus_1);
                let n_times_pow = arena.mul(&[n_var, a_pow_nm1]);
                if scale.is_one() {
                    return Some(n_times_pow);
                }
                let scale_id = rational_to_expr(arena, &scale);
                return Some(arena.mul(&[scale_id, n_times_pow]));
            }
        }
    }

    // ── Check for trig form: z² + p·z + q where discriminant < 0 ──
    // z²−2a·z·cos(ω)+a²  matches z² + p·z + q  with p = −2a·cos(ω), q = a²
    // For the unit-amplitude case (a=1): z²−2cos(ω)z+1, so p = −2cos(ω), q = 1

    // Check discriminant: p² − 4q
    let four = Ratio::from_integer(BigInt::from(4));
    let discriminant = &p * &p - &four * &q;

    // Trig form requires complex roots (negative discriminant)
    if discriminant.is_negative() && q.is_positive() {
        // Handle q = 1 (unit amplitude) case
        if q.is_one() {
            // Numerator may contain trig functions of omega — try structural matching.
            return try_trig_inverse_structural(arena, numer, denom_poly, z_var, n_var);
        }
    }

    None
}

/// Try structural matching for trig inverse z-transforms.
///
/// Matches numerator patterns like z·sin(ω) or z·(z−cos(ω)) against
/// the denominator z²−2z·cos(ω)+1.
fn try_trig_inverse_structural(
    arena: &mut Arena,
    numer: ExprId,
    _denom_poly: &crate::poly::Poly,
    z_var: ExprId,
    n_var: ExprId,
) -> Option<ExprId> {
    // Try to decompose the numerator as a product involving z and trig functions.
    let numer_node = arena.node(numer).clone();

    match numer_node {
        // Numerator is Mul([z, sin(ω)]) — this is z·sin(ω)
        ExprNode::Mul(ref children) => {
            let kids = children.clone();
            let mut has_z = false;
            let mut sin_arg: Option<ExprId> = None;
            let mut constants: Vec<ExprId> = Vec::new();

            for &child in &kids {
                if child == z_var {
                    has_z = true;
                    continue;
                }
                let cnode = arena.node(child).clone();
                match cnode {
                    ExprNode::Sin(arg) if !contains_var(arena, arg, z_var) => {
                        sin_arg = Some(arg);
                    }
                    _ if !contains_var(arena, child, z_var) => {
                        constants.push(child);
                    }
                    _ => {
                        return None;
                    }
                }
            }

            if has_z && let Some(omega) = sin_arg {
                // z·sin(ω) / denom → sin(ω·n)
                let omega_n = arena.mul(&[omega, n_var]);
                let result = arena.sin(omega_n);
                if constants.is_empty() {
                    return Some(result);
                }
                let scale = if constants.len() == 1 {
                    constants[0]
                } else {
                    arena.mul(&constants)
                };
                return Some(arena.mul(&[scale, result]));
            }

            None
        }
        _ => None,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Public API on Ex
// ═══════════════════════════════════════════════════════════════════════════

impl Ex {
    /// Compute the z-transform of this expression.
    ///
    /// Transforms x(n) → X(z) for discrete-time sequences using a
    /// table of known transforms.
    ///
    /// # Arguments
    ///
    /// * `n` — the discrete-time index variable
    /// * `z` — the z-domain variable
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let n = ctx.symbol("n");
    /// let z = ctx.symbol("z");
    /// let half = ctx.rational(1, 2);
    /// // Z{(1/2)^n} = z/(z − 1/2)
    /// let result = half.pow(&n).z_transform(&n, &z).unwrap();
    /// ```
    #[must_use = "returns the z-transform; does not modify in place"]
    pub fn z_transform(&self, n: &Ex, z: &Ex) -> Result<Ex, SymplexError> {
        let id = {
            let mut guard = self.inner.write();
            z_transform(&mut guard.arena, self.raw_id(), n.raw_id(), z.raw_id())?
        };
        Ok(self.wrap(id))
    }

    /// Compute the inverse z-transform.
    ///
    /// Transforms X(z) → x(n) for z-domain expressions using table
    /// lookup and partial fraction decomposition.
    ///
    /// # Arguments
    ///
    /// * `z` — the z-domain variable
    /// * `n` — the discrete-time index variable
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let n = ctx.symbol("n");
    /// let z = ctx.symbol("z");
    /// // Z⁻¹{z/(z−2)} = 2ⁿ
    /// let xz = &z / &(&z - 2);
    /// let result = xz.inverse_z_transform(&z, &n).unwrap();
    /// ```
    #[must_use = "returns the inverse z-transform; does not modify in place"]
    pub fn inverse_z_transform(&self, z: &Ex, n: &Ex) -> Result<Ex, SymplexError> {
        let id = {
            let mut guard = self.inner.write();
            inverse_z_transform(&mut guard.arena, self.raw_id(), z.raw_id(), n.raw_id())?
        };
        Ok(self.wrap(id))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests
// ═══════════════════════════════════════════════════════════════════════════

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
    fn forward_constant() {
        let mut a = Arena::new();
        let n = sym(&mut a, "n");
        let z = sym(&mut a, "z");
        let five = a.int(5);
        let result = z_transform(&mut a, five, n, z).unwrap();
        let d = display(&a, result);
        // Z{5} = 5z/(z-1)
        assert!(
            d.contains("5") && d.contains("z"),
            "Z{{5}} should be 5z/(z-1), got: {d}"
        );
    }

    #[test]
    fn forward_exponential() {
        let mut a = Arena::new();
        let n = sym(&mut a, "n");
        let z = sym(&mut a, "z");
        let half = a.rational(1, 2);
        let expr = a.pow(half, n); // (1/2)^n
        let result = z_transform(&mut a, expr, n, z).unwrap();
        let d = display(&a, result);
        // Z{(1/2)^n} = z/(z - 1/2)
        assert!(d.contains("z"), "Z{{(1/2)^n}} should involve z, got: {d}");
    }

    #[test]
    fn forward_sin() {
        let mut a = Arena::new();
        let n = sym(&mut a, "n");
        let z = sym(&mut a, "z");
        let three = a.int(3);
        let three_n = a.mul(&[three, n]);
        let sin_3n = a.sin(three_n);
        let result = z_transform(&mut a, sin_3n, n, z).unwrap();
        let d = display(&a, result);
        assert!(
            d.contains("sin") && d.contains("z"),
            "Z{{sin(3n)}} should involve sin and z, got: {d}"
        );
    }

    #[test]
    fn forward_cos() {
        let mut a = Arena::new();
        let n = sym(&mut a, "n");
        let z = sym(&mut a, "z");
        let cos_n = a.cos(n);
        let result = z_transform(&mut a, cos_n, n, z).unwrap();
        let d = display(&a, result);
        assert!(
            d.contains("cos") && d.contains("z"),
            "Z{{cos(n)}} should involve cos and z, got: {d}"
        );
    }

    #[test]
    fn forward_linearity() {
        let mut a = Arena::new();
        let n = sym(&mut a, "n");
        let z = sym(&mut a, "z");
        let half = a.rational(1, 2);
        let exp_term = a.pow(half, n);
        let sin_n = a.sin(n);
        let sum = a.add(&[exp_term, sin_n]);
        let result = z_transform(&mut a, sum, n, z);
        assert!(result.is_ok(), "linearity should work");
    }

    #[test]
    fn forward_n_must_be_symbol() {
        let mut a = Arena::new();
        let n = a.int(5); // not a symbol
        let z = a.symbol("z");
        let one = a.one;
        let result = z_transform(&mut a, one, n, z);
        assert!(result.is_err());
    }

    #[test]
    fn forward_n_var() {
        let mut a = Arena::new();
        let n = sym(&mut a, "n");
        let z = sym(&mut a, "z");
        // Z{n} = z/(z-1)^2
        let result = z_transform(&mut a, n, n, z).unwrap();
        let d = display(&a, result);
        assert!(d.contains("z"), "Z{{n}} should involve z, got: {d}");
    }

    #[test]
    fn forward_n_times_a_n() {
        let mut a = Arena::new();
        let n = sym(&mut a, "n");
        let z = sym(&mut a, "z");
        let two = a.int(2);
        let two_n = a.pow(two, n); // 2^n
        let n_times_2n = a.mul(&[n, two_n]); // n * 2^n
        let result = z_transform(&mut a, n_times_2n, n, z).unwrap();
        let d = display(&a, result);
        // Z{n·2^n} = 2z/(z-2)^2
        assert!(
            d.contains("z") && d.contains("2"),
            "Z{{n·2^n}} should involve z and 2, got: {d}"
        );
    }
}
