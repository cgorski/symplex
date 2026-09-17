//! Laplace transform and inverse Laplace transform.
//!
//! Forward: L{f(t)}(s) = ∫₀^∞ f(t)·e^(-st) dt
//! Inverse: table-lookup after partial fraction decomposition.
//!
//! # Supported transforms (forward)
//!
//! | Time domain | Frequency domain |
//! |---|---|
//! | 1 | 1/s |
//! | t^n | n!/s^(n+1) |
//! | exp(at) | 1/(s-a) |
//! | t^n·exp(at) | n!/(s-a)^(n+1) |
//! | sin(ωt) | ω/(s²+ω²) |
//! | cos(ωt) | s/(s²+ω²) |
//! | sinh(at) | a/(s²-a²) |
//! | cosh(at) | s/(s²-a²) |
//!
//! Plus linearity and frequency shift (exp(at)·f(t) → F(s-a)).

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Signed, Zero};

use crate::base::arena::Arena;
use crate::base::errors::SymplexError;
use crate::base::node::{ExprId, ExprNode, SymbolId};

// ═══════════════════════════════════════════════════════════════════════════
// Forward Laplace transform
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the Laplace transform of `expr` with respect to time variable `t`,
/// producing a function of frequency variable `s`.
pub(crate) fn laplace_transform(
    arena: &mut Arena,
    expr: ExprId,
    t: ExprId,
    s: ExprId,
) -> Result<ExprId, SymplexError> {
    let t_sym = match arena.node(t) {
        ExprNode::Symbol(sid) => *sid,
        _ => {
            return Err(SymplexError::ComputationFailed {
                operation: "laplace_transform",
                reason: "t must be a symbol".to_string(),
            });
        }
    };
    // Validate s is a symbol (only at the top-level entry).
    match arena.node(s) {
        ExprNode::Symbol(_) => {}
        _ => {
            return Err(SymplexError::ComputationFailed {
                operation: "laplace_transform",
                reason: "s must be a symbol".to_string(),
            });
        }
    };

    do_forward(arena, expr, t, t_sym, s)
}

/// Internal recursive forward transform (s is always the original symbol).
fn do_forward(
    arena: &mut Arena,
    expr: ExprId,
    t: ExprId,
    t_sym: SymbolId,
    s: ExprId,
) -> Result<ExprId, SymplexError> {
    // ── Linearity: if expr is Add, transform each term ──
    let node = arena.node(expr).clone();
    if let ExprNode::Add(ref children) = node {
        let kids = children.clone();
        let mut terms = Vec::with_capacity(kids.len());
        for child in kids {
            terms.push(do_forward(arena, child, t, t_sym, s)?);
        }
        return Ok(arena.add(&terms));
    }

    // ── Handle Neg: L{-f} = -L{f} ──
    if let ExprNode::Neg(inner) = node {
        let transformed = do_forward(arena, inner, t, t_sym, s)?;
        return Ok(arena.neg(transformed));
    }

    // ── Derivative rule: L{f'(t)} = s·F(s) - f(0) ──
    tracing::debug!("laplace: trying derivative rule");
    if let ExprNode::Derivative(body, deriv_var) = node
        && deriv_var == t
    {
        // Transform the body: F(s) = L{f(t)}
        let f_transform = do_forward(arena, body, t, t_sym, s)?;
        let s_times_f = arena.mul(&[s, f_transform]);

        // f(0): substitute t = 0 into the body and evaluate
        let zero = arena.zero;
        let f_at_0 = crate::transforms::subs::subs(arena, body, t, zero);
        let f_at_0_eval = crate::transforms::eval::eval(arena, f_at_0);

        return Ok(arena.sub(s_times_f, f_at_0_eval));
    }

    // ── Factor out constants (terms not depending on t) ──
    let (coeff, body) = split_independent(arena, expr, t);
    if coeff != arena.one {
        let transformed = do_forward(arena, body, t, t_sym, s)?;
        return Ok(arena.mul(&[coeff, transformed]));
    }

    // ── Try table rules ──
    tracing::debug!("laplace: trying table forward");
    if let Some(result) = try_table_forward(arena, expr, t, t_sym, s) {
        return Ok(result);
    }

    // ── Time-shift rule: L{H(t-a)·f(t)} = exp(-a·s)·L{f(t+a)} ──
    // Must come before frequency shift so H(t-a)*exp(t) isn't consumed by freq_shift first.
    if let Some(result) = try_time_shift(arena, expr, t, t_sym, s)? {
        return Ok(result);
    }

    // ── Try frequency shift: expr = exp(a*t) * g(t) ──
    if let Some(result) = try_freq_shift(arena, expr, t, t_sym, s)? {
        return Ok(result);
    }

    // ── Try t^n * exp(at) pattern ──
    if let Some(result) = try_tn_exp(arena, expr, t, t_sym, s) {
        return Ok(result);
    }

    // ── Frequency differentiation: L{t^n·f(t)} = (-1)^n · d^n/ds^n L{f(t)} ──
    if let Some(result) = try_freq_diff(arena, expr, t, t_sym, s)? {
        return Ok(result);
    }

    Err(SymplexError::ComputationFailed {
        operation: "laplace_transform",
        reason: "cannot transform expression".to_string(),
    })
}

// ─── Helpers ─────────────────────────────────────────────────────────────

/// Check whether `expr` contains the variable `t`.
fn contains_var(arena: &Arena, expr: ExprId, t: ExprId) -> bool {
    crate::base::walk::contains(arena, expr, t)
}

/// Split `expr` into `(coefficient_independent_of_t, rest_depending_on_t)`.
///
/// If expr is a `Mul` node, factors not containing `t` are pulled out.
/// Otherwise returns `(1, expr)`.
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

/// Given an expression that should be of the form `a*t` (or just `t`),
/// extract the coefficient `a` (returning it as an ExprId).
///
/// Returns `None` if the expression is not linear in `t`.
fn extract_linear_coeff(arena: &mut Arena, expr: ExprId, t: ExprId) -> Option<ExprId> {
    // expr == t → coefficient is 1
    if expr == t {
        return Some(arena.one);
    }

    // expr = Mul([..., t, ...]) where t appears exactly once
    // and other factors don't contain t
    let node = arena.node(expr).clone();
    if let ExprNode::Mul(ref children) = node {
        let mut t_count = 0usize;
        let mut others = Vec::new();
        for &child in children {
            if child == t {
                t_count += 1;
            } else if contains_var(arena, child, t) {
                // A factor that contains t but isn't t itself
                // (e.g. t^2) — not a simple linear coefficient
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

    // Fallback: try polynomial approach for numeric coefficients
    let poly = crate::poly::polybridge::expr_to_poly(arena, expr, t)?;
    if poly.degree()? != 1 {
        return None;
    }
    // Check that constant term is zero
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

/// Compute n! as a BigInt.
fn factorial_bigint(n: u64) -> BigInt {
    let mut result = BigInt::from(1u64);
    for i in 2..=n {
        result *= BigInt::from(i);
    }
    result
}

/// Convert a `Ratio<BigInt>` to an `ExprId`.
fn rational_to_expr(arena: &mut Arena, r: &Ratio<BigInt>) -> ExprId {
    let nid = arena.intern_num(r.clone());
    arena.intern(ExprNode::Num(nid))
}

// ─── Forward table rules ─────────────────────────────────────────────────

fn try_table_forward(
    arena: &mut Arena,
    expr: ExprId,
    t: ExprId,
    t_sym: SymbolId,
    s: ExprId,
) -> Option<ExprId> {
    let node = arena.node(expr).clone();

    match node {
        // Rule 0: constant c (no t) → c/s
        _ if !contains_var(arena, expr, t) => Some(arena.div(expr, s)),

        // Rule 1: t → 1/s²
        ExprNode::Symbol(sid) if sid == t_sym => {
            let two = arena.int(2);
            let s2 = arena.pow(s, two);
            Some(arena.div(arena.one, s2))
        }

        // Rule 2: t^n (positive integer n) → n!/s^(n+1)
        ExprNode::Pow(base, exp) if base == t => {
            if let Some(r) = arena.as_num(exp).cloned()
                && r.is_integer()
                && r.is_positive()
            {
                let n_val = r.to_integer().try_into().ok()?;
                let fact = factorial_bigint(n_val);
                let fact_rat = Ratio::from_integer(fact);
                let fact_id = rational_to_expr(arena, &fact_rat);

                let n_plus_1_rat = r + Ratio::one();
                let n_plus_1_id = rational_to_expr(arena, &n_plus_1_rat);
                let s_pow = arena.pow(s, n_plus_1_id);
                return Some(arena.div(fact_id, s_pow));
            }
            None
        }

        // Rule 3: exp(a*t) → 1/(s-a)
        //         exp(a*t + b) → exp(b)/(s-a)  (affine extension)
        ExprNode::Exp(arg) => {
            if let Some(a) = extract_linear_coeff(arena, arg, t) {
                let s_minus_a = arena.sub(s, a);
                return Some(arena.div(arena.one, s_minus_a));
            }
            // Affine case: arg = a*t + b where b ≠ 0
            if let Some(poly) = crate::poly::polybridge::expr_to_poly(arena, arg, t)
                && poly.degree() == Some(1)
            {
                let a_coeff = poly.coeff(1);
                let b_coeff = poly.coeff(0);
                if !a_coeff.is_zero() && !b_coeff.is_zero() {
                    let a_id = rational_to_expr(arena, &a_coeff);
                    let b_id = rational_to_expr(arena, &b_coeff);
                    let s_minus_a = arena.sub(s, a_id);
                    let exp_b = arena.exp(b_id);
                    return Some(arena.div(exp_b, s_minus_a));
                }
            }
            None
        }

        // Rule 4: sin(ω*t) → ω/(s²+ω²)
        ExprNode::Sin(arg) => {
            if let Some(omega) = extract_linear_coeff(arena, arg, t) {
                let omega_sq = arena.mul(&[omega, omega]);
                let s_sq = arena.mul(&[s, s]);
                let denom = arena.add(&[s_sq, omega_sq]);
                return Some(arena.div(omega, denom));
            }
            None
        }

        // Rule 5: cos(ω*t) → s/(s²+ω²)
        ExprNode::Cos(arg) => {
            if let Some(omega) = extract_linear_coeff(arena, arg, t) {
                let omega_sq = arena.mul(&[omega, omega]);
                let s_sq = arena.mul(&[s, s]);
                let denom = arena.add(&[s_sq, omega_sq]);
                return Some(arena.div(s, denom));
            }
            None
        }

        // Rule 6: sinh(a*t) → a/(s²-a²)
        ExprNode::Sinh(arg) => {
            if let Some(a) = extract_linear_coeff(arena, arg, t) {
                let a_sq = arena.mul(&[a, a]);
                let s_sq = arena.mul(&[s, s]);
                let denom = arena.sub(s_sq, a_sq);
                return Some(arena.div(a, denom));
            }
            None
        }

        // Rule 7: cosh(a*t) → s/(s²-a²)
        ExprNode::Cosh(arg) => {
            if let Some(a) = extract_linear_coeff(arena, arg, t) {
                let a_sq = arena.mul(&[a, a]);
                let s_sq = arena.mul(&[s, s]);
                let denom = arena.sub(s_sq, a_sq);
                return Some(arena.div(s, denom));
            }
            None
        }

        _ => None,
    }
}

// ─── Time-shift rule ─────────────────────────────────────────────────────

/// Try the time-shift rule: L{H(t-a)·f(t)} = exp(-a·s)·L{f(t+a)}
///
/// Detects a `Heaviside(t - a)` factor in a Mul node, shifts f(t) → f(t+a),
/// transforms the shifted function, and multiplies by exp(-a·s).
fn try_time_shift(
    arena: &mut Arena,
    expr: ExprId,
    t: ExprId,
    t_sym: SymbolId,
    s: ExprId,
) -> Result<Option<ExprId>, SymplexError> {
    let node = arena.node(expr).clone();

    if let ExprNode::Mul(ref children) = node {
        let kids = children.clone();
        for (i, &child) in kids.iter().enumerate() {
            if let ExprNode::Heaviside(h_arg) = arena.node(child).clone() {
                // Check if h_arg is (t - a) for some constant a
                if let Some(a) = extract_shift(arena, h_arg, t, t_sym) {
                    tracing::debug!("laplace: time-shift detected with a={:?}", a);

                    // Collect remaining factors as g(t)
                    let remaining: Vec<ExprId> = kids
                        .iter()
                        .enumerate()
                        .filter(|&(j, _)| j != i)
                        .map(|(_, &c)| c)
                        .collect();
                    let g = if remaining.len() == 1 {
                        remaining[0]
                    } else if remaining.is_empty() {
                        arena.one
                    } else {
                        arena.mul(&remaining)
                    };

                    // Substitute t → t + a in g
                    let t_plus_a = arena.add(&[t, a]);
                    let g_shifted = crate::transforms::subs::subs(arena, g, t, t_plus_a);

                    // Recursively transform g_shifted
                    let g_transform = do_forward(arena, g_shifted, t, t_sym, s)?;

                    // Multiply by exp(-a*s)
                    let a_s = arena.mul(&[a, s]);
                    let neg_a_s = arena.neg(a_s);
                    let exp_factor = arena.exp(neg_a_s);

                    return Ok(Some(arena.mul(&[exp_factor, g_transform])));
                }
            }
        }
    }
    Ok(None)
}

/// Extract the shift amount `a` from an expression of the form `t - a`.
///
/// - If `expr == t`, returns `Some(0)` (shift by 0).
/// - If `expr` is `Add(t, Neg(a))` or similar polynomial `t - a`, returns `Some(a)`.
/// - Returns `None` if `expr` is not a simple linear shift of `t`.
fn extract_shift(arena: &mut Arena, expr: ExprId, t: ExprId, _t_sym: SymbolId) -> Option<ExprId> {
    // expr == t → shift by 0
    if expr == t {
        return Some(arena.zero);
    }

    // Try polynomial approach: expr should be t - a, i.e. linear with coeff 1
    let poly = crate::poly::polybridge::expr_to_poly(arena, expr, t)?;
    if poly.degree()? != 1 {
        return None;
    }
    // Coefficient of t must be 1
    let a1 = poly.coeff(1);
    if !a1.is_one() {
        return None;
    }
    // Constant term is -a, so a = -constant
    let c0 = poly.coeff(0);
    let a_val = -c0;
    let a_id = rational_to_expr(arena, &a_val);
    Some(a_id)
}

// ─── Frequency differentiation ───────────────────────────────────────────

/// Try the frequency differentiation rule: L{t^n·f(t)} = (-1)^n · d^n/ds^n L{f(t)}
///
/// Detects a `t` or `t^n` factor in a Mul node, transforms the remaining
/// factors, then differentiates with respect to `s` `n` times, applying the
/// (-1)^n sign.
fn try_freq_diff(
    arena: &mut Arena,
    expr: ExprId,
    t: ExprId,
    t_sym: SymbolId,
    s: ExprId,
) -> Result<Option<ExprId>, SymplexError> {
    let node = arena.node(expr).clone();

    if let ExprNode::Mul(ref children) = node {
        let kids = children.clone();
        for (i, &child) in kids.iter().enumerate() {
            let n = check_t_power(arena, child, t, t_sym);
            if n > 0 {
                tracing::debug!("laplace: frequency differentiation detected, t^{}", n);

                // Collect remaining factors as g(t)
                let remaining: Vec<ExprId> = kids
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != i)
                    .map(|(_, &c)| c)
                    .collect();
                let g = if remaining.len() == 1 {
                    remaining[0]
                } else if remaining.is_empty() {
                    arena.one
                } else {
                    arena.mul(&remaining)
                };

                // L{t^n · g(t)} = (-1)^n · d^n/ds^n L{g(t)}
                let g_transform = do_forward(arena, g, t, t_sym, s)?;
                let mut result = g_transform;
                for _ in 0..n {
                    result = crate::transforms::diff::diff(arena, result, s);
                    result = arena.neg(result);
                }
                return Ok(Some(result));
            }
        }
    }
    Ok(None)
}

/// Check if `expr` is `t` or `t^n` for a positive integer n.
/// Returns 0 if not a power of t, otherwise returns n.
fn check_t_power(arena: &Arena, expr: ExprId, t: ExprId, t_sym: SymbolId) -> u64 {
    // expr == t → n = 1
    if expr == t {
        return 1;
    }
    if let ExprNode::Symbol(sid) = arena.node(expr)
        && *sid == t_sym
    {
        return 1;
    }

    // expr == Pow(t, n)
    if let ExprNode::Pow(base, exp) = arena.node(expr)
        && *base == t
        && let Some(r) = arena.as_num(*exp)
        && r.is_integer()
        && r.is_positive()
    {
        return r.to_integer().try_into().unwrap_or(0);
    }

    0
}

// ─── Frequency shift ─────────────────────────────────────────────────────

/// Try the frequency-shift rule: L{exp(a*t)·g(t)} = G(s − a)
///
/// where G(s) = L{g(t)}.
fn try_freq_shift(
    arena: &mut Arena,
    expr: ExprId,
    t: ExprId,
    t_sym: SymbolId,
    s: ExprId,
) -> Result<Option<ExprId>, SymplexError> {
    let node = arena.node(expr).clone();

    if let ExprNode::Mul(ref children) = node {
        let kids = children.clone();
        for (i, &child) in kids.iter().enumerate() {
            let child_node = arena.node(child).clone();
            if let ExprNode::Exp(arg) = child_node
                && let Some(a) = extract_linear_coeff(arena, arg, t)
            {
                // Collect remaining factors (everything except this exp)
                let remaining: Vec<ExprId> = kids
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != i)
                    .map(|(_, &c)| c)
                    .collect();
                let g = if remaining.len() == 1 {
                    remaining[0]
                } else {
                    arena.mul(&remaining)
                };

                // Compute G(s) = L{g(t)}
                let g_of_s = do_forward(arena, g, t, t_sym, s)?;

                // Substitute s → (s − a) in G(s) to get G(s − a)
                let s_shifted = arena.sub(s, a);
                let result = crate::transforms::subs::subs(arena, g_of_s, s, s_shifted);
                return Ok(Some(result));
            }
        }
    }
    Ok(None)
}

// ─── t^n * exp(a*t) pattern ──────────────────────────────────────────────

/// Try to match t^n * exp(a*t) → n! / (s-a)^(n+1)
fn try_tn_exp(
    arena: &mut Arena,
    expr: ExprId,
    t: ExprId,
    t_sym: SymbolId,
    s: ExprId,
) -> Option<ExprId> {
    let node = arena.node(expr).clone();

    if let ExprNode::Mul(ref children) = node {
        let kids = children.clone();
        let mut exp_idx = None;
        let mut pow_idx = None;
        let mut t_idx = None;

        for (i, &child) in kids.iter().enumerate() {
            let cn = arena.node(child).clone();
            match cn {
                ExprNode::Exp(_) => {
                    if exp_idx.is_some() {
                        return None;
                    }
                    exp_idx = Some(i);
                }
                ExprNode::Pow(base, _) if base == t => {
                    if pow_idx.is_some() {
                        return None;
                    }
                    pow_idx = Some(i);
                }
                ExprNode::Symbol(sid) if sid == t_sym => {
                    if t_idx.is_some() {
                        return None;
                    }
                    t_idx = Some(i);
                }
                _ => {}
            }
        }

        // Need exp part
        let exp_i = exp_idx?;
        let exp_node = arena.node(kids[exp_i]).clone();
        let ExprNode::Exp(exp_arg) = exp_node else {
            return None;
        };
        let a = extract_linear_coeff(arena, exp_arg, t)?;

        // Determine n from t^n or plain t
        let n_val: u64;
        if let Some(pi) = pow_idx {
            let pow_node = arena.node(kids[pi]).clone();
            if let ExprNode::Pow(_, exp) = pow_node {
                let r = arena.as_num(exp).cloned()?;
                if !r.is_integer() || !r.is_positive() {
                    return None;
                }
                n_val = r.to_integer().try_into().ok()?;
            } else {
                return None;
            }
        } else if let Some(_ti) = t_idx {
            n_val = 1;
        } else {
            return None;
        }

        // Check no other dependent factors remain
        let skip: Vec<usize> = {
            let mut v = vec![exp_i];
            if let Some(pi) = pow_idx {
                v.push(pi);
            }
            if let Some(ti) = t_idx {
                v.push(ti);
            }
            v
        };
        for (i, &child) in kids.iter().enumerate() {
            if skip.contains(&i) {
                continue;
            }
            if contains_var(arena, child, t) {
                return None;
            }
        }

        // Collect constant factors (those not skipped)
        let const_factors: Vec<ExprId> = kids
            .iter()
            .enumerate()
            .filter(|(i, _)| !skip.contains(i))
            .map(|(_, &c)| c)
            .collect();

        // n! / (s - a)^(n+1)
        let fact = factorial_bigint(n_val);
        let fact_rat = Ratio::from_integer(fact);
        let fact_id = rational_to_expr(arena, &fact_rat);

        let s_minus_a = arena.sub(s, a);
        let n_plus_1 = arena.int(n_val as i64 + 1);
        let denom = arena.pow(s_minus_a, n_plus_1);
        let mut result = arena.div(fact_id, denom);

        if !const_factors.is_empty() {
            let mut all = const_factors;
            all.push(result);
            result = arena.mul(&all);
        }

        return Some(result);
    }

    None
}

// ═══════════════════════════════════════════════════════════════════════════
// Inverse Laplace transform
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the inverse Laplace transform of `expr` (function of `s`)
/// back to a function of `t`.
pub(crate) fn inverse_laplace_transform(
    arena: &mut Arena,
    expr: ExprId,
    s: ExprId,
    t: ExprId,
) -> Result<ExprId, SymplexError> {
    match arena.node(s) {
        ExprNode::Symbol(_) => {}
        _ => {
            return Err(SymplexError::ComputationFailed {
                operation: "inverse_laplace_transform",
                reason: "s must be a symbol".to_string(),
            });
        }
    };
    match arena.node(t) {
        ExprNode::Symbol(_) => {}
        _ => {
            return Err(SymplexError::ComputationFailed {
                operation: "inverse_laplace_transform",
                reason: "t must be a symbol".to_string(),
            });
        }
    };

    do_inverse(arena, expr, s, t, 0)
}

/// Internal recursive inverse transform with a recursion guard.
fn do_inverse(
    arena: &mut Arena,
    expr: ExprId,
    s: ExprId,
    t: ExprId,
    depth: u32,
) -> Result<ExprId, SymplexError> {
    if depth > 20 {
        return Err(SymplexError::ComputationFailed {
            operation: "inverse_laplace_transform",
            reason: "recursion limit reached".to_string(),
        });
    }

    // ── Linearity: handle Add ──
    let node = arena.node(expr).clone();
    if let ExprNode::Add(ref children) = node {
        let kids = children.clone();
        let mut terms = Vec::with_capacity(kids.len());
        for child in kids {
            terms.push(do_inverse(arena, child, s, t, depth + 1)?);
        }
        return Ok(arena.add(&terms));
    }

    // ── Handle Neg: L⁻¹{-F} = -L⁻¹{F} ──
    if let ExprNode::Neg(inner) = node {
        let result = do_inverse(arena, inner, s, t, depth + 1)?;
        return Ok(arena.neg(result));
    }

    // ── Factor out constants (not containing s) ──
    let (coeff, body) = split_independent(arena, expr, s);
    if coeff != arena.one {
        let result = do_inverse(arena, body, s, t, depth + 1)?;
        return Ok(arena.mul(&[coeff, result]));
    }

    // ── Try table lookup ──
    if let Some(result) = try_table_inverse(arena, expr, s, t) {
        return Ok(result);
    }

    // ── Try partial fraction decomposition ──
    let decomposed = crate::transforms::apart::apart(arena, expr, s);
    if decomposed != expr {
        // apart produced something different — recurse on the decomposition
        return do_inverse(arena, decomposed, s, t, depth + 1);
    }

    Err(SymplexError::ComputationFailed {
        operation: "inverse_laplace_transform",
        reason: "cannot invert expression".to_string(),
    })
}

// ─── Inverse table rules ─────────────────────────────────────────────────

fn try_table_inverse(arena: &mut Arena, expr: ExprId, s: ExprId, t: ExprId) -> Option<ExprId> {
    // If expr doesn't contain s at all, it can't be a valid F(s).
    // But a constant / something with s could still be valid.

    // Decompose as numerator / denominator
    let (numer, denom) = crate::poly::polybridge::as_numer_denom(arena, expr);

    // If denominator is 1, this expression has no s in the denominator.
    // It's not a standard inverse Laplace form (unless it's 0).
    if denom == arena.one {
        return None;
    }

    // Try to convert denominator to a polynomial in s.
    let denom_poly = crate::poly::polybridge::expr_to_poly(arena, denom, s)?;
    let deg = denom_poly.degree()?;

    // ── Degree 1 denominator: c₁·s + c₀ ──
    // Represents (up to scale) 1/(s − a) where a = −c₀/c₁
    if deg == 1 {
        return inverse_degree1(arena, numer, &denom_poly, s, t);
    }

    // ── Degree 2 denominator ──
    if deg == 2 {
        return inverse_degree2(arena, numer, &denom_poly, s, t);
    }

    // ── Higher-degree denominators: try to detect (s-a)^n form ──
    if deg >= 2
        && let Some(result) = inverse_power_form(arena, numer, denom, &denom_poly, s, t)
    {
        return Some(result);
    }

    None
}

/// Inverse for degree-1 denominator: L⁻¹{N(s) / (c₁·s + c₀)}
///
/// If N is constant: result = (N/c₁) · exp(a·t) where a = −c₀/c₁
fn inverse_degree1(
    arena: &mut Arena,
    numer: ExprId,
    denom_poly: &crate::poly::Poly,
    s: ExprId,
    t: ExprId,
) -> Option<ExprId> {
    let c0 = denom_poly.coeff(0);
    let c1 = denom_poly.coeff(1);
    if c1.is_zero() {
        return None;
    }

    // a = -c₀ / c₁
    let a_rat = -&c0 / &c1;

    // If numerator doesn't contain s, it's a constant
    if !contains_var(arena, numer, s) {
        // L⁻¹{N / (c₁(s - a))} = (N/c₁) · exp(a·t)
        let a_id = rational_to_expr(arena, &a_rat);
        let at = arena.mul(&[a_id, t]);
        let exp_at = arena.exp(at);

        if a_rat.is_zero() {
            // L⁻¹{N / (c₁·s)} = N/c₁  (constant function)
            let c1_id = rational_to_expr(arena, &c1);
            return Some(arena.div(numer, c1_id));
        }

        let c1_id = rational_to_expr(arena, &c1);
        let coeff = arena.div(numer, c1_id);
        return Some(arena.mul(&[coeff, exp_at]));
    }

    // If numerator is a polynomial in s of degree 0 or higher, we could
    // do polynomial long division, but for now handle the simple case where
    // numer = s (occurs in some forms).
    let numer_poly = crate::poly::polybridge::expr_to_poly(arena, numer, s)?;
    let numer_deg = numer_poly.degree()?;

    if numer_deg == 0 {
        let n0 = numer_poly.coeff(0);
        let a_id = rational_to_expr(arena, &a_rat);
        let at = arena.mul(&[a_id, t]);
        let exp_at = arena.exp(at);

        if a_rat.is_zero() {
            let result_rat = n0 / c1;
            return Some(rational_to_expr(arena, &result_rat));
        }

        let scale = n0 / c1;
        let scale_id = rational_to_expr(arena, &scale);
        return Some(arena.mul(&[scale_id, exp_at]));
    }

    None
}

/// Inverse for degree-2 denominator: L⁻¹{N(s) / (c₂·s² + c₁·s + c₀)}
fn inverse_degree2(
    arena: &mut Arena,
    numer: ExprId,
    denom_poly: &crate::poly::Poly,
    s: ExprId,
    t: ExprId,
) -> Option<ExprId> {
    let c0 = denom_poly.coeff(0);
    let c1 = denom_poly.coeff(1);
    let c2 = denom_poly.coeff(2);
    if c2.is_zero() {
        return None;
    }

    // Normalize: s² + (c₁/c₂)·s + (c₀/c₂)
    let b = &c1 / &c2; // linear coefficient after normalization
    let c = &c0 / &c2; // constant term after normalization

    // Convert numerator to polynomial
    let numer_poly = crate::poly::polybridge::expr_to_poly(arena, numer, s);

    // Case 1: No linear term in denom (b == 0) → s² + ω² form
    if b.is_zero() {
        // Special case: b=0, c=0 → denominator is c₂·s² (Bug 8)
        if c.is_zero() {
            if let Some(np) = &numer_poly {
                let nd = np.degree().unwrap_or(0);
                if nd == 0 {
                    // L⁻¹{k / (c₂·s²)} = (k/c₂)·t
                    let k = np.coeff(0);
                    let k_id = rational_to_expr(arena, &k);
                    let c2_id = rational_to_expr(arena, &c2);
                    let scale = arena.div(k_id, c2_id);
                    return Some(arena.mul(&[scale, t]));
                }
            }
            return None;
        }

        // ω² = c (must be positive for sin/cos)
        if !c.is_positive() {
            // s² - |c| → could be sinh/cosh, but partial fractions handle it
            return None;
        }
        let omega_sq = c.clone();

        // Try to find ω such that ω² = omega_sq
        let omega_id = {
            let omega_sq_id = rational_to_expr(arena, &omega_sq);
            let raw_sqrt = arena.sqrt(omega_sq_id);
            crate::transforms::eval::eval(arena, raw_sqrt)
        };

        if let Some(np) = &numer_poly {
            let nd = np.degree().unwrap_or(0);

            // Numer is constant: L⁻¹{k / (s² + ω²)} = (k/ω) · sin(ωt)
            if nd == 0 {
                let k = np.coeff(0);
                let k_id = rational_to_expr(arena, &k);

                let c2_id = rational_to_expr(arena, &c2);

                let omega_t = arena.mul(&[omega_id, t]);
                let sin_omega_t = arena.sin(omega_t);

                // result = (k / (c₂ · ω)) · sin(ωt)
                let c2_omega = arena.mul(&[c2_id, omega_id]);
                let scale = arena.div(k_id, c2_omega);
                return Some(arena.mul(&[scale, sin_omega_t]));
            }

            // Numer is linear (a₁·s + a₀):
            // L⁻¹{(a₁·s + a₀)/(c₂(s²+ω²))} = (a₁/c₂)·cos(ωt) + (a₀/(c₂·ω))·sin(ωt)
            if nd == 1 {
                let a0 = np.coeff(0);
                let a1 = np.coeff(1);

                let c2_id = rational_to_expr(arena, &c2);
                let omega_t = arena.mul(&[omega_id, t]);
                let sin_omega_t = arena.sin(omega_t);
                let cos_omega_t = arena.cos(omega_t);

                let mut terms = Vec::new();

                // cos term: (a₁/c₂) · cos(ωt)
                if !a1.is_zero() {
                    let a1_id = rational_to_expr(arena, &a1);
                    let cos_coeff = arena.div(a1_id, c2_id);
                    terms.push(arena.mul(&[cos_coeff, cos_omega_t]));
                }

                // sin term: (a₀/(c₂·ω)) · sin(ωt)
                if !a0.is_zero() {
                    let a0_id = rational_to_expr(arena, &a0);
                    let c2_omega = arena.mul(&[c2_id, omega_id]);
                    let sin_coeff = arena.div(a0_id, c2_omega);
                    terms.push(arena.mul(&[sin_coeff, sin_omega_t]));
                }

                if terms.is_empty() {
                    return Some(arena.zero);
                } else if terms.len() == 1 {
                    return Some(terms[0]);
                } else {
                    return Some(arena.add(&terms));
                }
            }
        }

        // Numerator is not polynomial in s — can't handle
        return None;
    }

    // Case 2: Non-zero linear term → complete the square
    // s² + b·s + c = (s + b/2)² + (c - b²/4)
    // Let α = -b/2, β² = c - b²/4
    let alpha = -&b / Ratio::from_integer(BigInt::from(2));
    let beta_sq = &c - &(&b * &b) / Ratio::from_integer(BigInt::from(4));

    if beta_sq.is_zero() {
        // Repeated root: denominator is c₂·(s - α)² where α = -b/2 (Bug 9)
        let alpha_id = rational_to_expr(arena, &alpha);

        if let Some(np) = &numer_poly {
            let nd = np.degree().unwrap_or(0);
            let c2_id = rational_to_expr(arena, &c2);

            if nd == 0 {
                // L⁻¹{k / (c₂·(s-α)²)} = (k/c₂)·t·exp(α·t)
                let k = np.coeff(0);
                let k_id = rational_to_expr(arena, &k);
                let scale = arena.div(k_id, c2_id);
                if alpha.is_zero() {
                    return Some(arena.mul(&[scale, t]));
                }
                let alpha_t = arena.mul(&[alpha_id, t]);
                let exp_alpha_t = arena.exp(alpha_t);
                return Some(arena.mul(&[scale, t, exp_alpha_t]));
            }

            if nd == 1 {
                let a0 = np.coeff(0);
                let a1 = np.coeff(1);

                // Decompose: a₁·s + a₀ = a₁·(s - α) + (a₁·α + a₀)
                let d_const = &a1 * &alpha + &a0;

                let alpha_t = arena.mul(&[alpha_id, t]);
                let exp_alpha_t = arena.exp(alpha_t);

                let mut terms = Vec::new();

                // exp term: (a₁/c₂) · exp(α·t)
                if !a1.is_zero() {
                    let a1_id = rational_to_expr(arena, &a1);
                    let exp_coeff = arena.div(a1_id, c2_id);
                    terms.push(arena.mul(&[exp_coeff, exp_alpha_t]));
                }

                // t·exp term: (d_const/c₂) · t · exp(α·t)
                if !d_const.is_zero() {
                    let d_id = rational_to_expr(arena, &d_const);
                    let t_exp_coeff = arena.div(d_id, c2_id);
                    terms.push(arena.mul(&[t_exp_coeff, t, exp_alpha_t]));
                }

                if terms.is_empty() {
                    return Some(arena.zero);
                } else if terms.len() == 1 {
                    return Some(terms[0]);
                } else {
                    return Some(arena.add(&terms));
                }
            }
        }

        return None;
    }

    // ── Case 2b: β² < 0 → overdamped / real exponential forms ──
    // (s-α)² - γ² where γ² = -β² = b²/4 - c
    // L⁻¹{ k / (c₂·((s-α)² - γ²)) } = (k/(c₂·γ)) · exp(α·t) · sinh(γ·t)
    if beta_sq.is_negative() {
        let gamma_sq = -&beta_sq;
        let gamma_sq_id = rational_to_expr(arena, &gamma_sq);
        let gamma_id = {
            let raw = arena.sqrt(gamma_sq_id);
            crate::transforms::eval::eval(arena, raw)
        };
        let alpha_id = rational_to_expr(arena, &alpha);

        if let Some(np) = &numer_poly {
            let nd = np.degree().unwrap_or(0);
            let c2_id = rational_to_expr(arena, &c2);

            if nd == 0 {
                let k = np.coeff(0);
                let k_id = rational_to_expr(arena, &k);

                let alpha_t = arena.mul(&[alpha_id, t]);
                let exp_alpha_t = arena.exp(alpha_t);
                let gamma_t = arena.mul(&[gamma_id, t]);
                let sinh_gamma_t = arena.sinh(gamma_t);

                let c2_gamma = arena.mul(&[c2_id, gamma_id]);
                let scale = arena.div(k_id, c2_gamma);
                return Some(arena.mul(&[scale, exp_alpha_t, sinh_gamma_t]));
            }

            if nd == 1 {
                let a0 = np.coeff(0);
                let a1 = np.coeff(1);

                // Decompose: a₁·s + a₀ = a₁·(s - α) + (a₁·α + a₀)
                let d_const = &a1 * &alpha + &a0;

                let alpha_t = arena.mul(&[alpha_id, t]);
                let exp_alpha_t = arena.exp(alpha_t);
                let gamma_t = arena.mul(&[gamma_id, t]);
                let sinh_gamma_t = arena.sinh(gamma_t);
                let cosh_gamma_t = arena.cosh(gamma_t);

                let mut terms = Vec::new();

                // cosh term: (a₁/c₂) · exp(α·t) · cosh(γ·t)
                if !a1.is_zero() {
                    let a1_id = rational_to_expr(arena, &a1);
                    let cosh_coeff = arena.div(a1_id, c2_id);
                    terms.push(arena.mul(&[cosh_coeff, exp_alpha_t, cosh_gamma_t]));
                }

                // sinh term: (d_const/(c₂·γ)) · exp(α·t) · sinh(γ·t)
                if !d_const.is_zero() {
                    let d_id = rational_to_expr(arena, &d_const);
                    let c2_gamma = arena.mul(&[c2_id, gamma_id]);
                    let sinh_coeff = arena.div(d_id, c2_gamma);
                    terms.push(arena.mul(&[sinh_coeff, exp_alpha_t, sinh_gamma_t]));
                }

                if terms.is_empty() {
                    return Some(arena.zero);
                } else if terms.len() == 1 {
                    return Some(terms[0]);
                } else {
                    return Some(arena.add(&terms));
                }
            }
        }

        return None;
    }

    // β = sqrt(β²)
    let beta_sq_id = rational_to_expr(arena, &beta_sq);
    let beta_id = {
        let raw = arena.sqrt(beta_sq_id);
        crate::transforms::eval::eval(arena, raw)
    };
    let alpha_id = rational_to_expr(arena, &alpha);

    if let Some(np) = &numer_poly {
        let nd = np.degree().unwrap_or(0);
        let c2_id = rational_to_expr(arena, &c2);

        // Rewrite numer in terms of (s - α): N(s) = N(α) + N'(α)·(s - α)
        // For a₁·s + a₀ with substitution s = (s' + α):
        // a₁(s' + α) + a₀ = a₁·s' + (a₁·α + a₀)
        // where s' = s - α = s + b/2

        if nd == 0 {
            // L⁻¹{k / (c₂·((s-α)² + β²))} = (k/(c₂·β)) · exp(α·t) · sin(β·t)
            let k = np.coeff(0);
            let k_id = rational_to_expr(arena, &k);

            let alpha_t = arena.mul(&[alpha_id, t]);
            let exp_alpha_t = arena.exp(alpha_t);
            let beta_t = arena.mul(&[beta_id, t]);
            let sin_beta_t = arena.sin(beta_t);

            let c2_beta = arena.mul(&[c2_id, beta_id]);
            let scale = arena.div(k_id, c2_beta);
            return Some(arena.mul(&[scale, exp_alpha_t, sin_beta_t]));
        }

        if nd == 1 {
            let a0 = np.coeff(0);
            let a1 = np.coeff(1);

            // Decompose: a₁·s + a₀ = a₁·(s - α) + (a₁·α + a₀)
            let d_const = &a1 * &alpha + &a0;

            let alpha_t = arena.mul(&[alpha_id, t]);
            let exp_alpha_t = arena.exp(alpha_t);
            let beta_t = arena.mul(&[beta_id, t]);
            let sin_beta_t = arena.sin(beta_t);
            let cos_beta_t = arena.cos(beta_t);

            let mut terms = Vec::new();

            // cos term: (a₁/c₂) · exp(α·t) · cos(β·t)
            if !a1.is_zero() {
                let a1_id = rational_to_expr(arena, &a1);
                let cos_coeff = arena.div(a1_id, c2_id);
                terms.push(arena.mul(&[cos_coeff, exp_alpha_t, cos_beta_t]));
            }

            // sin term: (d_const/(c₂·β)) · exp(α·t) · sin(β·t)
            if !d_const.is_zero() {
                let d_id = rational_to_expr(arena, &d_const);
                let c2_beta = arena.mul(&[c2_id, beta_id]);
                let sin_coeff = arena.div(d_id, c2_beta);
                terms.push(arena.mul(&[sin_coeff, exp_alpha_t, sin_beta_t]));
            }

            if terms.is_empty() {
                return Some(arena.zero);
            } else if terms.len() == 1 {
                return Some(terms[0]);
            } else {
                return Some(arena.add(&terms));
            }
        }
    }

    None
}

/// Try to detect denominator of the form (s − a)^n and invert.
///
/// L⁻¹{1/(s−a)^n} = t^(n−1) · exp(a·t) / (n−1)!
fn inverse_power_form(
    arena: &mut Arena,
    numer: ExprId,
    denom: ExprId,
    _denom_poly: &crate::poly::Poly,
    s: ExprId,
    t: ExprId,
) -> Option<ExprId> {
    // Check if denom is Pow(base, exp) with positive integer exp
    let denom_node = arena.node(denom).clone();
    if let ExprNode::Pow(base, exp) = denom_node
        && let Some(r) = arena.as_num(exp).cloned()
        && r.is_integer()
        && r.is_positive()
    {
        let n_val: u64 = r.to_integer().try_into().ok()?;

        // Check if base is linear in s: base = s - a
        let base_poly = crate::poly::polybridge::expr_to_poly(arena, base, s)?;
        if base_poly.degree()? != 1 {
            return None;
        }
        let bc0 = base_poly.coeff(0);
        let bc1 = base_poly.coeff(1);
        if !bc1.is_one() {
            return None; // Only handle monic for now
        }
        let a_rat = -bc0;

        // Numerator must be constant (no s)
        if contains_var(arena, numer, s) {
            return None;
        }

        // L⁻¹{N / (s-a)^n} = N · t^(n-1) · exp(a·t) / (n-1)!
        let a_id = rational_to_expr(arena, &a_rat);
        let at = arena.mul(&[a_id, t]);
        let exp_at = arena.exp(at);

        if n_val == 1 {
            // Simple: N · exp(a·t)
            if a_rat.is_zero() {
                return Some(numer);
            }
            return Some(arena.mul(&[numer, exp_at]));
        }

        // t^(n-1)
        let n_minus_1 = arena.int(n_val as i64 - 1);
        let t_pow = arena.pow(t, n_minus_1);

        // (n-1)!
        let fact = factorial_bigint(n_val - 1);
        let fact_rat = Ratio::from_integer(fact);
        let fact_id = rational_to_expr(arena, &fact_rat);

        // result = numer * t^(n-1) * exp(at) / (n-1)!
        let numer_t = arena.mul(&[numer, t_pow]);
        let scaled = arena.div(numer_t, fact_id);
        if a_rat.is_zero() {
            return Some(scaled);
        }
        return Some(arena.mul(&[scaled, exp_at]));
    }

    None
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
    fn forward_constant() {
        let mut a = Arena::new();
        let t = sym(&mut a, "t");
        let s = sym(&mut a, "s");
        let five = a.int(5);
        let result = laplace_transform(&mut a, five, t, s).unwrap();
        let d = display(&a, result);
        // L{5} = 5/s = 5*s^(-1)
        assert!(
            d.contains("5") && d.contains("s"),
            "L{{5}} should be 5/s, got: {d}"
        );
    }

    #[test]
    fn forward_t() {
        let mut a = Arena::new();
        let t = sym(&mut a, "t");
        let s = sym(&mut a, "s");
        // L{t} = 1/s²
        let result = laplace_transform(&mut a, t, t, s).unwrap();
        let d = display(&a, result);
        assert!(d.contains("s"), "L{{t}} should be 1/s², got: {d}");
    }

    #[test]
    fn forward_exp() {
        let mut a = Arena::new();
        let t = sym(&mut a, "t");
        let s = sym(&mut a, "s");
        let two = a.int(2);
        let two_t = a.mul(&[two, t]);
        let exp_2t = a.exp(two_t);
        let result = laplace_transform(&mut a, exp_2t, t, s).unwrap();
        let d = display(&a, result);
        // L{exp(2t)} = 1/(s-2)
        assert!(
            d.contains("s") && d.contains("2"),
            "L{{exp(2t)}} should involve s and 2, got: {d}"
        );
    }

    #[test]
    fn forward_sin() {
        let mut a = Arena::new();
        let t = sym(&mut a, "t");
        let s = sym(&mut a, "s");
        let three = a.int(3);
        let three_t = a.mul(&[three, t]);
        let sin_3t = a.sin(three_t);
        let result = laplace_transform(&mut a, sin_3t, t, s).unwrap();
        let d = display(&a, result);
        // L{sin(3t)} = 3/(s²+9)
        assert!(
            d.contains("3") && d.contains("s"),
            "L{{sin(3t)}} should involve 3 and s, got: {d}"
        );
    }

    #[test]
    fn forward_cos() {
        let mut a = Arena::new();
        let t = sym(&mut a, "t");
        let s = sym(&mut a, "s");
        let cos_t = a.cos(t);
        let result = laplace_transform(&mut a, cos_t, t, s).unwrap();
        let d = display(&a, result);
        // L{cos(t)} = s/(s²+1)
        assert!(d.contains("s"), "L{{cos(t)}} should involve s, got: {d}");
    }

    #[test]
    fn forward_linearity() {
        let mut a = Arena::new();
        let t = sym(&mut a, "t");
        let s = sym(&mut a, "s");
        // L{exp(t) + sin(t)}
        let exp_t = a.exp(t);
        let sin_t = a.sin(t);
        let sum = a.add(&[exp_t, sin_t]);
        let result = laplace_transform(&mut a, sum, t, s);
        assert!(result.is_ok(), "linearity should work");
    }

    #[test]
    fn forward_t_must_be_symbol() {
        let mut a = Arena::new();
        let t = a.int(5); // not a symbol
        let s = a.symbol("s");
        let one = a.one;
        let result = laplace_transform(&mut a, one, t, s);
        assert!(result.is_err());
    }
}
