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
//! | nᵏ·x\[n\]         | (−z d/dz)ᵏ X(z)  (so n²aⁿ, n³, …)         |
//! | δ\[n−k\]           | z⁻ᵏ                                       |
//! | H(n−k)            | z⁻ᵏ·z/(z−1)                              |
//! | C(n, k)           | z/(z−1)^(k+1)                             |
//! | 1/n!              | e^(1/z)                                   |
//! | aⁿ·x\[n\]         | X(z/a)                                    |
//! | x\[n−k\]·H(n−k)  | z⁻ᵏ·X(z)                                 |
//!
//! Plus linearity (sum of terms) and constant factor extraction.
//!
//! The inverse handles rational `X(z)` through partial fractions (terms
//! `z/(z−a)ᵐ → C(n, m−1) a^(n−m+1)`, `1/(z−a)ᵐ` via the delay rule),
//! constants (`δ\[n\]`), `z⁻ᵏ` (`δ\[n−k\]`), `z⁻ᵏ X(z)` (delay) and the
//! trigonometric forms.
//!
//! Unit samples in inverse results are `KroneckerDelta(n, k)`; on input both
//! `KroneckerDelta(n, k)` and `DiracDelta(n − k)` are accepted.
//! Discrete unit steps in inverse results are written `H(n − k + 1/2)`:
//! for integer `n` this is exactly `u[n − k]` (`1` for `n ≥ k`, else `0`)
//! under every convention for `H(0)`. Both `H(n − k)` and `H(n − k + 1/2)`
//! are accepted on input, with `H(0)` read as `1`.

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

    // ── Extended entries and rules ──
    if let Some(result) = try_extended_forward(arena, expr, n_var, n_sym, z_var)? {
        return Ok(result);
    }

    Err(SymplexError::ComputationFailed {
        operation: "z_transform",
        reason: format!("cannot transform {}", arena.display(expr)),
    })
}

/// `expr = a·n + b` with `a`, `b` free of `n`.
fn linear_in(arena: &mut Arena, expr: ExprId, n_var: ExprId) -> Option<(ExprId, ExprId)> {
    if expr == n_var {
        return Some((arena.one, arena.zero));
    }
    if !contains_var(arena, expr, n_var) {
        return Some((arena.zero, expr));
    }
    match arena.node(expr).clone() {
        ExprNode::Neg(inner) => {
            let (a, b) = linear_in(arena, inner, n_var)?;
            Some((arena.neg(a), arena.neg(b)))
        }
        ExprNode::Mul(children) => {
            let mut coeff = Vec::new();
            let mut seen = false;
            for &c in &children {
                if c == n_var {
                    if seen {
                        return None;
                    }
                    seen = true;
                } else if contains_var(arena, c, n_var) {
                    return None;
                } else {
                    coeff.push(c);
                }
            }
            if !seen {
                return None;
            }
            let a = if coeff.is_empty() {
                arena.one
            } else {
                arena.mul(&coeff)
            };
            Some((a, arena.zero))
        }
        ExprNode::Add(children) => {
            let mut a_terms = Vec::new();
            let mut b_terms = Vec::new();
            for &c in &children {
                let (a, b) = linear_in(arena, c, n_var)?;
                if !arena.is_zero_structural(a) {
                    a_terms.push(a);
                }
                if !arena.is_zero_structural(b) {
                    b_terms.push(b);
                }
            }
            let a = arena.add(&a_terms);
            let b = arena.add(&b_terms);
            Some((a, b))
        }
        _ => None,
    }
}

/// Non-negative integer shift `k` from a step argument `n − k` or
/// `n − k + 1/2` (the form produced by [`discrete_step`]).
fn shift_of(arena: &mut Arena, arg: ExprId, n_var: ExprId) -> Option<u64> {
    let (a, b) = linear_in(arena, arg, n_var)?;
    if a != arena.one {
        return None;
    }
    let b = arena.as_num(b)?.clone();
    // b = −k  or  b = −k + 1/2
    let half = Ratio::new(BigInt::from(1), BigInt::from(2));
    let k = if b.is_integer() { -b } else { half - b };
    if !k.is_integer() || k.is_negative() {
        return None;
    }
    k.to_integer().try_into().ok()
}

/// The unit sample `δ[n − k]` as `KroneckerDelta(n, k)`.
fn kronecker(arena: &mut Arena, n_var: ExprId, k: i64) -> ExprId {
    let k_id = arena.int(k);
    arena.kronecker_delta(n_var, k_id)
}

/// The discrete unit step `u[n − k]` (`1` for `n ≥ k`, `0` otherwise),
/// written as `H(n − k + 1/2)` so that it evaluates to exactly `0` or `1`
/// at every integer regardless of the `H(0) = 1/2` convention.
fn discrete_step(arena: &mut Arena, n_var: ExprId, k: i64) -> ExprId {
    let off = arena.rational(1 - 2 * k, 2);
    let arg = arena.add(&[n_var, off]);
    arena.heaviside(arg)
}

fn zfail(reason: impl Into<String>) -> SymplexError {
    SymplexError::ComputationFailed {
        operation: "z_transform",
        reason: reason.into(),
    }
}

/// Extended forward entries: `δ[n−k]`, `H(n−k)`, `C(n, k)`, `1/n!`, the
/// `n·x[n] → −z X′(z)` rule (powers of `n`), scaling `aⁿ x[n] → X(z/a)` and
/// the delay `x[n−k] H(n−k) → z^{−k} X(z)`.
fn try_extended_forward(
    arena: &mut Arena,
    expr: ExprId,
    n_var: ExprId,
    n_sym: SymbolId,
    z_var: ExprId,
) -> Result<Option<ExprId>, SymplexError> {
    let one = arena.one;
    match arena.node(expr).clone() {
        // δ[n − k] → z^{−k}  (as `DiracDelta(n − k)` or `KroneckerDelta(n, k)`)
        ExprNode::DiracDelta(arg) => {
            let Some(k) = shift_of(arena, arg, n_var) else {
                return Ok(None);
            };
            let neg_k = arena.int(-(k as i64));
            Ok(Some(arena.pow(z_var, neg_k)))
        }
        ExprNode::KroneckerDelta(i, j) => {
            let k = if i == n_var {
                j
            } else if j == n_var {
                i
            } else {
                return Ok(None);
            };
            let Some(k) = arena.as_num(k).cloned() else {
                return Ok(None);
            };
            if !k.is_integer() || k.is_negative() {
                return Ok(None);
            }
            let neg_k = rational_to_expr(arena, &-k);
            Ok(Some(arena.pow(z_var, neg_k)))
        }
        // H(n − k) → z^{−k} z/(z − 1)
        ExprNode::Heaviside(arg) => {
            let Some(k) = shift_of(arena, arg, n_var) else {
                return Ok(None);
            };
            let one_minus_k = arena.int(1 - k as i64);
            let zp = arena.pow(z_var, one_minus_k);
            let z_minus_1 = arena.sub(z_var, one);
            Ok(Some(arena.div(zp, z_minus_1)))
        }
        // C(n, k) → z/(z − 1)^{k+1}
        ExprNode::Binomial(top, k) if top == n_var && !contains_var(arena, k, n_var) => {
            let k_p1 = arena.add(&[k, one]);
            let z_minus_1 = arena.sub(z_var, one);
            let den = arena.pow(z_minus_1, k_p1);
            Ok(Some(arena.div(z_var, den)))
        }
        // 1/n! → e^{1/z}
        ExprNode::Pow(base, e) if e == arena.neg_one => {
            if let ExprNode::Factorial(arg) = arena.node(base).clone()
                && arg == n_var
            {
                let inv_z = arena.div(one, z_var);
                return Ok(Some(arena.exp(inv_z)));
            }
            Ok(None)
        }
        // nᵏ (k ≥ 2) → (−z d/dz)^{k−1} Z{n}
        ExprNode::Pow(base, e) if base == n_var => {
            let Some(k) = arena.as_num(e).cloned() else {
                return Ok(None);
            };
            if !k.is_integer() || !k.is_positive() {
                return Ok(None);
            }
            let k: u32 = k
                .to_integer()
                .try_into()
                .map_err(|_| zfail("power too large"))?;
            let z_minus_1 = arena.sub(z_var, one);
            let two = arena.int(2);
            let den = arena.pow(z_minus_1, two);
            let mut x = arena.div(z_var, den); // Z{n}
            for _ in 1..k {
                x = neg_z_derivative(arena, x, z_var);
            }
            Ok(Some(x))
        }
        ExprNode::Mul(children) => {
            let kids: Vec<ExprId> = children.iter().copied().collect();
            // Delay: x[n − k]·H(n − k) → z^{−k} X(z)
            for (i, &c) in kids.iter().enumerate() {
                if let ExprNode::Heaviside(arg) = arena.node(c).clone()
                    && let Some(k) = shift_of(arena, arg, n_var)
                {
                    let rest: Vec<ExprId> = kids
                        .iter()
                        .enumerate()
                        .filter(|&(j, _)| j != i)
                        .map(|(_, &c)| c)
                        .collect();
                    let g = if rest.is_empty() {
                        one
                    } else {
                        arena.mul(&rest)
                    };
                    let k_id = arena.int(k as i64);
                    let n_plus_k = arena.add(&[n_var, k_id]);
                    let g_shifted = crate::transforms::subs::subs(arena, g, n_var, n_plus_k);
                    let g_shifted = crate::transforms::eval::eval(arena, g_shifted);
                    let gz = do_forward(arena, g_shifted, n_var, n_sym, z_var)?;
                    let neg_k = arena.int(-(k as i64));
                    let zk = arena.pow(z_var, neg_k);
                    return Ok(Some(arena.mul(&[zk, gz])));
                }
            }
            // Powers of n: nᵏ·g[n] → (−z d/dz)ᵏ G(z)
            let mut n_power: u32 = 0;
            let mut rest: Vec<ExprId> = Vec::new();
            for &c in &kids {
                if c == n_var {
                    n_power += 1;
                } else if let ExprNode::Pow(base, e) = arena.node(c).clone()
                    && base == n_var
                    && let Some(k) = arena.as_num(e).cloned()
                    && k.is_integer()
                    && k.is_positive()
                {
                    let k: u32 = k
                        .to_integer()
                        .try_into()
                        .map_err(|_| zfail("power too large"))?;
                    n_power += k;
                } else {
                    rest.push(c);
                }
            }
            if n_power > 0 && !rest.is_empty() {
                let g = arena.mul(&rest);
                let mut x = do_forward(arena, g, n_var, n_sym, z_var)?;
                for _ in 0..n_power {
                    x = neg_z_derivative(arena, x, z_var);
                }
                return Ok(Some(x));
            }
            // Scaling: aⁿ·g[n] → G(z/a)
            for (i, &c) in kids.iter().enumerate() {
                if let ExprNode::Pow(base, e) = arena.node(c).clone()
                    && e == n_var
                    && !contains_var(arena, base, n_var)
                {
                    let rest: Vec<ExprId> = kids
                        .iter()
                        .enumerate()
                        .filter(|&(j, _)| j != i)
                        .map(|(_, &c)| c)
                        .collect();
                    if rest.is_empty() {
                        return Ok(None);
                    }
                    let g = arena.mul(&rest);
                    let gz = do_forward(arena, g, n_var, n_sym, z_var)?;
                    let z_over_a = arena.div(z_var, base);
                    let scaled = crate::transforms::subs::subs(arena, gz, z_var, z_over_a);
                    return Ok(Some(scaled));
                }
            }
            Ok(None)
        }
        _ => Ok(None),
    }
}

/// `−z·dX/dz`, kept as a single fraction.
///
/// For `X = N(z)/B(z)ᵐ` with `B` linear in `z` (every table entry except
/// the trigonometric ones) the quotient rule is applied by hand so that the
/// result is `−z(N′B − mB′N)/B^{m+1}` with an expanded numerator — the form
/// the inverse transform recognises. Otherwise the generic derivative is
/// combined with `together`/`cancel`.
fn neg_z_derivative(arena: &mut Arena, x: ExprId, z_var: ExprId) -> ExprId {
    let (num, den) = crate::poly::polybridge::as_numer_denom(arena, x);
    let (base, m) = match arena.node(den).clone() {
        ExprNode::Pow(b, e)
            if arena
                .as_num(e)
                .is_some_and(|r| r.is_integer() && r.is_positive()) =>
        {
            (b, e)
        }
        _ => (den, arena.one),
    };
    if den != arena.one && linear_in(arena, base, z_var).is_some() {
        let n_prime = crate::transforms::diff::diff(arena, num, z_var);
        let b_prime = crate::transforms::diff::diff(arena, base, z_var);
        let t1 = arena.mul(&[n_prime, base]);
        let t2 = arena.mul(&[m, b_prime, num]);
        let diff = arena.sub(t1, t2);
        let new_num = arena.mul(&[z_var, diff]);
        let new_num = arena.neg(new_num);
        let new_num = crate::transforms::expand::expand(arena, new_num);
        let new_num = crate::transforms::eval::eval(arena, new_num);
        let m_p1 = arena.add(&[m, arena.one]);
        let new_den = arena.pow(base, m_p1);
        return arena.div(new_num, new_den);
    }
    let d = crate::transforms::diff::diff(arena, x, z_var);
    let zd = arena.mul(&[z_var, d]);
    let r = arena.neg(zd);
    let r = crate::poly::polybridge::together(arena, r);
    let r = crate::poly::polybridge::cancel(arena, r, z_var);
    crate::transforms::eval::eval(arena, r)
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

    // ── Constant c → c·δ[n] ──
    if !contains_var(arena, expr, z_var) {
        let d = kronecker(arena, n_var, 0);
        return Ok(arena.mul(&[expr, d]));
    }

    // ── Factor out constants (not containing z) ──
    let (coeff, body) = split_independent(arena, expr, z_var);
    if coeff != arena.one {
        let result = do_inverse(arena, body, z_var, n_var, depth + 1)?;
        return Ok(arena.mul(&[coeff, result]));
    }

    // ── Trigonometric forms with symbolic/irrational parameters ──
    if let Some(result) = try_trig_inverse_general(arena, expr, z_var, n_var) {
        return Ok(result);
    }

    // ── Extended entries: z^{−k}, delay, z/(z−a)^m, 1/(z−a)^m, e^{1/z} ──
    if let Some(result) = try_extended_inverse(arena, expr, z_var, n_var, depth)? {
        return Ok(result);
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
        reason: format!("cannot invert {}", arena.display(expr)),
    })
}

/// `(αz² + βz)/(z² − 2az·cosω + a²)` → `α·aⁿcos(ωn) + (β + αa·cosω)/(a·sinω)·aⁿsin(ωn)`,
/// recognising `cos ω` structurally in the denominator (so `ω` and `a` may
/// be symbolic or irrational).
fn try_trig_inverse_general(
    arena: &mut Arena,
    expr: ExprId,
    z_var: ExprId,
    n_var: ExprId,
) -> Option<ExprId> {
    let (numer, denom) = crate::poly::polybridge::as_numer_denom(arena, expr);
    let one = arena.one;
    let two = arena.int(2);
    let z2 = arena.pow(z_var, two);
    let ExprNode::Add(terms) = arena.node(denom).clone() else {
        return None;
    };
    if terms.len() != 3 || !terms.contains(&z2) {
        return None;
    }
    // Linear term c·z and constant q.
    let mut lin: Option<ExprId> = None;
    let mut q: Option<ExprId> = None;
    for &t in &terms {
        if t == z2 {
            continue;
        }
        if !contains_var(arena, t, z_var) {
            q = Some(t);
        } else if let Some((c, b)) = linear_in(arena, t, z_var)
            && arena.is_zero_structural(b)
        {
            lin = Some(c);
        } else {
            return None;
        }
    }
    let (c, q) = (lin?, q?);
    // c = −2·a·cos(ω): find the Cos factor.
    let factors: Vec<ExprId> = match arena.node(c).clone() {
        ExprNode::Mul(ch) => ch.iter().copied().collect(),
        ExprNode::Neg(inner) => {
            let mut v = vec![arena.neg_one];
            match arena.node(inner).clone() {
                ExprNode::Mul(ch) => v.extend(ch.iter().copied()),
                _ => v.push(inner),
            }
            v
        }
        _ => return None,
    };
    let mut omega: Option<ExprId> = None;
    let mut rest: Vec<ExprId> = Vec::new();
    for &f in &factors {
        if let ExprNode::Cos(w) = arena.node(f).clone()
            && omega.is_none()
        {
            omega = Some(w);
        } else {
            rest.push(f);
        }
    }
    let omega = omega?;
    // rest = −2a
    let rest_prod = arena.mul(&rest);
    let neg_two = arena.int(-2);
    let a = arena.div(rest_prod, neg_two);
    let a = crate::transforms::eval::eval(arena, a);
    // Consistency: q must equal a².
    let a2 = arena.pow(a, two);
    let a2 = crate::transforms::eval::eval(arena, a2);
    let q_e = crate::transforms::eval::eval(arena, q);
    if a2 != q_e {
        let d = arena.sub(a2, q_e);
        let d = crate::transforms::expand::expand(arena, d);
        let d = crate::transforms::eval::eval(arena, d);
        if !arena.is_zero_structural(d) {
            return None;
        }
    }
    // Numerator: αz² + βz (γ = 0).
    let numer_x = crate::transforms::expand::expand(arena, numer);
    let numer_terms: Vec<ExprId> = match arena.node(numer_x).clone() {
        ExprNode::Add(ch) => ch.iter().copied().collect(),
        _ => vec![numer_x],
    };
    let mut alpha = arena.zero;
    let mut beta = arena.zero;
    for &t in &numer_terms {
        if !contains_var(arena, t, z_var) {
            return None;
        }
        // t = k·z² or k·z
        let (k, is_sq) = match arena.node(t).clone() {
            _ if t == z_var => (one, false),
            _ if t == z2 => (one, true),
            ExprNode::Mul(ch) => {
                let mut coeff = Vec::new();
                let mut kind: Option<bool> = None;
                for &f in &ch {
                    if f == z_var && kind.is_none() {
                        kind = Some(false);
                    } else if f == z2 && kind.is_none() {
                        kind = Some(true);
                    } else if contains_var(arena, f, z_var) {
                        return None;
                    } else {
                        coeff.push(f);
                    }
                }
                (arena.mul(&coeff), kind?)
            }
            _ => return None,
        };
        if is_sq {
            alpha = arena.add(&[alpha, k]);
        } else {
            beta = arena.add(&[beta, k]);
        }
    }
    let wn = arena.mul(&[omega, n_var]);
    let cos_wn = arena.cos(wn);
    let sin_wn = arena.sin(wn);
    let an = arena.pow(a, n_var);
    let cos_w = arena.cos(omega);
    let sin_w = arena.sin(omega);
    // β + α·a·cosω
    let acos = arena.mul(&[alpha, a, cos_w]);
    let sin_coeff_num = arena.add(&[beta, acos]);
    let a_sin = arena.mul(&[a, sin_w]);
    let sin_coeff = arena.div(sin_coeff_num, a_sin);
    let cos_term = arena.mul(&[alpha, an, cos_wn]);
    let sin_term = arena.mul(&[sin_coeff, an, sin_wn]);
    let r = arena.add(&[cos_term, sin_term]);
    Some(crate::transforms::eval::eval(arena, r))
}

/// Extended inverse entries.
///
/// * `z^{−k} → δ[n − k]` and `z^{−k} X(z) → x[n−k] H(n−k)` (delay);
/// * `z/(z − a)^m → C(n, m−1) a^{n−m+1}` and `1/(z − a)^m` via the delay
///   rule, with symbolic `a` allowed;
/// * `e^{1/z} → 1/n!`.
fn try_extended_inverse(
    arena: &mut Arena,
    expr: ExprId,
    z_var: ExprId,
    n_var: ExprId,
    depth: u32,
) -> Result<Option<ExprId>, SymplexError> {
    let one = arena.one;
    // Factor classification.
    let kids: Vec<ExprId> = match arena.node(expr).clone() {
        ExprNode::Mul(ch) => ch.iter().copied().collect(),
        _ => vec![expr],
    };
    let mut z_power: i64 = 0; // net power of z among plain z factors
    let mut pole: Option<(ExprId, u64)> = None; // (a, m) from (z − a)^{−m}
    let mut others: Vec<ExprId> = Vec::new();
    for &k in &kids {
        if k == z_var {
            z_power += 1;
            continue;
        }
        match arena.node(k).clone() {
            ExprNode::Pow(base, e) if base == z_var => {
                if let Some(r) = arena.as_num(e)
                    && r.is_integer()
                {
                    z_power += i64::try_from(r.to_integer()).map_err(|_| {
                        SymplexError::ComputationFailed {
                            operation: "inverse_z_transform",
                            reason: "power too large".into(),
                        }
                    })?;
                    continue;
                }
                others.push(k);
            }
            ExprNode::Pow(base, e)
                if pole.is_none()
                    && let Some(r) = arena.as_num(e).cloned()
                    && r.is_integer()
                    && r.is_negative()
                    && let Some((c1, c0)) = linear_in(arena, base, z_var)
                    && c1 == one =>
            {
                let a = arena.neg(c0);
                let a = crate::transforms::eval::eval(arena, a);
                let m: u64 =
                    (-r.to_integer())
                        .try_into()
                        .map_err(|_| SymplexError::ComputationFailed {
                            operation: "inverse_z_transform",
                            reason: "power too large".into(),
                        })?;
                pole = Some((a, m));
            }
            ExprNode::Exp(arg) if others.is_empty() && pole.is_none() => {
                // e^{1/z} → 1/n!
                let inv_z = arena.pow(z_var, arena.neg_one);
                if arg == inv_z && kids.len() == 1 {
                    let f = arena.factorial(n_var);
                    return Ok(Some(arena.div(one, f)));
                }
                others.push(k);
            }
            _ => others.push(k),
        }
    }

    // Pure z^{−k} → δ[n − k]
    if pole.is_none() && others.is_empty() {
        if z_power <= 0 {
            return Ok(Some(kronecker(arena, n_var, -z_power)));
        }
        return Err(SymplexError::ComputationFailed {
            operation: "inverse_z_transform",
            reason: "positive powers of z correspond to non-causal sequences".into(),
        });
    }

    // z^{1−m}·… with a pole: z/(z − a)^m → C(n, m−1) a^{n−m+1}
    if let Some((a, m)) = pole
        && others.is_empty()
    {
        // expr = z^{p} (z − a)^{−m}; write as z^{p−1} · [z/(z−a)^m].
        let delay = 1 - z_power; // z^{p−1} = z^{−delay}
        if delay < 0 {
            return Ok(None); // improper: leave to partial fractions
        }
        // C(n, m−1) written as the falling factorial n(n−1)…(n−m+2)/(m−1)!,
        // which evaluates to 0 for 0 ≤ n < m−1 (a `Binomial` node with
        // k > n does not currently evaluate).
        let m_minus_1 = arena.int(m as i64 - 1);
        let exp = arena.sub(n_var, m_minus_1);
        let a_pow = arena.pow(a, exp);
        let base = if m == 1 {
            arena.pow(a, n_var)
        } else {
            let mut factors = Vec::with_capacity(m as usize);
            for j in 0..(m - 1) {
                let jj = arena.int(j as i64);
                factors.push(arena.sub(n_var, jj));
            }
            let fact: i64 = (1..m as i64).product();
            let inv_fact = arena.rational(1, fact);
            factors.push(inv_fact);
            factors.push(a_pow);
            arena.mul(&factors)
        };
        let base = crate::transforms::eval::eval(arena, base);
        if delay == 0 {
            return Ok(Some(base));
        }
        let k = arena.int(delay);
        let n_minus_k = arena.sub(n_var, k);
        let shifted = crate::transforms::subs::subs(arena, base, n_var, n_minus_k);
        let h = discrete_step(arena, n_var, delay);
        return Ok(Some(arena.mul(&[shifted, h])));
    }

    // General delay: z^{−k}·X(z) with X in the table.
    if z_power < 0 {
        let mut rest = others.clone();
        if let Some((a, m)) = pole {
            let base = arena.sub(z_var, a);
            let neg_m = arena.int(-(m as i64));
            rest.push(arena.pow(base, neg_m));
        }
        // Keep one factor of z with X if the rest is a proper `z/(…)` form.
        let x = arena.mul(&rest);
        let xz = arena.mul(&[z_var, x]);
        let (k, inner) = if let Ok(r) = do_inverse(arena, xz, z_var, n_var, depth + 1) {
            (-z_power + 1, r)
        } else {
            (-z_power, do_inverse(arena, x, z_var, n_var, depth + 1)?)
        };
        let k_id = arena.int(k);
        let n_minus_k = arena.sub(n_var, k_id);
        let shifted = crate::transforms::subs::subs(arena, inner, n_var, n_minus_k);
        let h = discrete_step(arena, n_var, k);
        return Ok(Some(arena.mul(&[shifted, h])));
    }

    Ok(None)
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
    /// Unilateral Z-transform `X(z) = Σ_{n≥0} x[n] z^{−n}` of this sequence
    /// (a function of the integer index `n`).
    ///
    /// Table: constants, `aⁿ`, `nᵏ aⁿ` (via `Z{n x[n]} = −z X′(z)`),
    /// `sin(ωn)`, `cos(ωn)`, `aⁿ sin(ωn)`, `aⁿ cos(ωn)`, `H(n − k)`,
    /// `δ[n − k]`, `C(n, k)`, `1/n!`; rules: linearity, scaling
    /// `aⁿ x[n] → X(z/a)`, delay `x[n − k] H(n − k) → z^{−k} X(z)`.
    ///
    /// # Errors
    ///
    /// `ComputationFailed` if `n`/`z` are not symbols or no rule applies.
    /// There is no unevaluated Z-transform node, so this API is
    /// `Result`-only.
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
    /// assert_eq!(result, &z / (&z - half));
    /// // n² → z(z + 1)/(z − 1)³
    /// let x = n.powi(2).z_transform(&n, &z).unwrap();
    /// let expected = &z * (&z + 1) / (&z - 1).powi(3);
    /// assert!((&x - &expected).simplify().is_zero_structural(), "{x}");
    /// // δ[n − 3] → z⁻³
    /// assert_eq!((&n - 3).dirac_delta().z_transform(&n, &z).unwrap(), z.powi(-3));
    /// ```
    #[must_use = "returns the z-transform; does not modify in place"]
    pub fn z_transform(&self, n: &Ex, z: &Ex) -> Result<Ex, SymplexError> {
        let n_id = self.checked_id(n);
        let z_id = self.checked_id(z);
        let id = {
            let mut guard = self.inner.write();
            z_transform(&mut guard.arena, self.raw_id(), n_id, z_id)?
        };
        Ok(self.wrap(id))
    }

    /// Inverse (unilateral) Z-transform of this expression (a function of
    /// `z`) as a sequence in `n`.
    ///
    /// Rational `X(z)` is handled through partial fractions in `z`
    /// (`z/(z − a)ᵐ → C(n, m−1) a^{n−m+1}`, `1/(z − a)ᵐ` through the delay
    /// rule), together with constants (`δ[n]`), `z^{−k}` (`δ[n − k]`),
    /// `z^{−k} X(z)` (`x[n−k] H(n−k)`), `e^{1/z}` (`1/n!`) and the
    /// trigonometric forms.
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
    /// assert_eq!(format!("{}", xz.inverse_z_transform(&z, &n).unwrap()), "2^n");
    /// // Z⁻¹{z⁻²} = δ[n − 2]
    /// let d = (1 / z.powi(2)).inverse_z_transform(&z, &n).unwrap();
    /// assert_eq!(format!("{d}"), "KroneckerDelta(2, n)");
    /// ```
    #[must_use = "returns the inverse z-transform; does not modify in place"]
    pub fn inverse_z_transform(&self, z: &Ex, n: &Ex) -> Result<Ex, SymplexError> {
        let z_id = self.checked_id(z);
        let n_id = self.checked_id(n);
        let id = {
            let mut guard = self.inner.write();
            inverse_z_transform(&mut guard.arena, self.raw_id(), z_id, n_id)?
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
