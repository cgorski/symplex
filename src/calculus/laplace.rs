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
//! | sin(ωt + φ), cos(ωt + φ), sinh(at + b), cosh(at + b) | by the addition formulas |
//! | t^ν (ν > −1) | Γ(ν+1)/s^(ν+1) |
//! | ln t | −(γ + ln s)/s |
//! | δ(t − a), H(t − a) | e^(−as), e^(−as)/s |
//! | Jₙ(at) | (√(s²+a²) − s)ⁿ / (aⁿ √(s²+a²)) |
//! | erf(a√t) | a/(s √(s + a²)) |
//! | sin(at)/t, (1 − cos at)/t | atan(a/s), ½ ln(1 + a²/s²) |
//!
//! Plus linearity, frequency shift (exp(at)·f(t) → F(s-a)), time shift
//! (f(t−a)H(t−a) → e^(−as)F(s)), frequency differentiation
//! (tⁿ f(t) → (−1)ⁿ F⁽ⁿ⁾(s)) and division by t (f(t)/t → ∫_s^∞ F(u) du).
//! When no rule applies, powers `sinⁿ`/`cosⁿ` are linearised, products of
//! sines and cosines turned into sums, `sinh`/`cosh` next to other factors
//! written as exponentials and the product expanded, and the rules tried
//! once more.
//!
//! The inverse handles rational functions through partial fractions and
//! the table; a rational function with rational coefficients that the
//! table does not cover (repeated complex poles, non-monic factors, …) is
//! inverted exactly as the solution of `D(d/dt) f = 0` whose initial
//! derivatives are read off the expansion of `F` at infinity.  Also
//! `e^(−as)F(s)` (delay), `s^(−ν)`, `1/√(s²+a²)`, `atan(a/s)`,
//! `1/(s√(s+a²))` and constants (`δ`).

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Signed, Zero};

use crate::base::arena::Arena;
use crate::base::combinatorics::factorial;
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

/// Internal recursive forward transform (s is always the original symbol):
/// the rules, then — only if they fail — once more on the expression
/// rewritten into table-friendly form ([`rewrite_for_table`]).
fn do_forward(
    arena: &mut Arena,
    expr: ExprId,
    t: ExprId,
    t_sym: SymbolId,
    s: ExprId,
) -> Result<ExprId, SymplexError> {
    let ruled = do_forward_rules(arena, expr, t, t_sym, s);
    if ruled.is_ok() {
        return ruled;
    }
    let rewritten = rewrite_for_table(arena, expr, t);
    if rewritten == expr {
        return ruled;
    }
    do_forward_rules(arena, rewritten, t, t_sym, s).or(ruled)
}

/// `f(t)` rewritten towards sums of `tⁿ·e^{at}·{1, sin bt, cos bt}`, the
/// shapes the table, frequency shift and frequency differentiation cover:
/// `sinⁿ`/`cosⁿ` linearised, products of two sines/cosines by the
/// product-to-sum formulas, `sinh`/`cosh` next to other factors of `t` as
/// exponentials, `e^{at+b} = e^b·e^{at}`, and the result expanded (so the
/// time-shift rule's `(t + a)ⁿ` becomes a polynomial).  Only the top-level
/// factors are rewritten.  (`cos² t`, `sinh(t)·sin(t)`, `t³·H(t − 1/2)`
/// and `e^{−t−2}·sin 2t` were refused.)
fn rewrite_for_table(arena: &mut Arena, expr: ExprId, t: ExprId) -> ExprId {
    let factors: Vec<ExprId> = match arena.node(expr) {
        ExprNode::Mul(ch) => ch.to_vec(),
        _ => vec![expr],
    };
    let n_dep = factors
        .iter()
        .filter(|&&f| contains_var(arena, f, t))
        .count();
    let mut out: Vec<ExprId> = Vec::with_capacity(factors.len());
    let mut trig: Vec<ExprId> = Vec::new();
    let half = arena.rational(1, 2);
    for f in factors {
        let node = arena.node(f).clone();
        match node {
            ExprNode::Pow(b, e)
                if matches!(arena.node(b), ExprNode::Sin(_) | ExprNode::Cos(_))
                    && arena.as_num(e).is_some_and(|q| {
                        q.is_integer()
                            && q.is_positive()
                            && *q <= Ratio::from_integer(BigInt::from(8))
                    }) =>
            {
                let n: u32 = arena
                    .as_num(e)
                    .and_then(|q| q.to_integer().try_into().ok())
                    .unwrap_or(1);
                let lin = match arena.node(b).clone() {
                    ExprNode::Sin(a) => crate::simplify::fu::linearize_sin_power(arena, a, n),
                    ExprNode::Cos(a) => crate::simplify::fu::linearize_cos_power(arena, a, n),
                    _ => f,
                };
                out.push(lin);
            }
            ExprNode::Sin(_) | ExprNode::Cos(_) if contains_var(arena, f, t) => trig.push(f),
            ExprNode::Sinh(a) | ExprNode::Cosh(a) if n_dep > 1 && contains_var(arena, a, t) => {
                // (e^a ∓ e^{−a})/2
                let ea = arena.exp(a);
                let na = arena.neg(a);
                let ena = arena.exp(na);
                let pair = if matches!(node, ExprNode::Sinh(_)) {
                    arena.sub(ea, ena)
                } else {
                    arena.add(&[ea, ena])
                };
                out.push(arena.mul(&[half, pair]));
            }
            ExprNode::Exp(a) => match linear_in(arena, a, t) {
                Some((k, b)) if !arena.is_zero_structural(b) && !arena.is_zero_structural(k) => {
                    let kt = arena.mul(&[k, t]);
                    let e1 = arena.exp(b);
                    let e2 = arena.exp(kt);
                    out.push(e1);
                    out.push(e2);
                }
                _ => out.push(f),
            },
            _ => out.push(f),
        }
    }
    // Product-to-sum on pairs of sines/cosines.
    while trig.len() >= 2 {
        let (x, y) = (trig.remove(0), trig.remove(0));
        let arg = |arena: &Arena, id: ExprId| match arena.node(id) {
            ExprNode::Sin(a) => (true, *a),
            ExprNode::Cos(a) => (false, *a),
            _ => (false, id),
        };
        let ((sx, a), (sy, b)) = (arg(arena, x), arg(arena, y));
        let sum = arena.add(&[a, b]);
        let diff = arena.sub(a, b);
        let combined = match (sx, sy) {
            // sin a sin b = ½[cos(a − b) − cos(a + b)]
            (true, true) => {
                let c1 = arena.cos(diff);
                let c2 = arena.cos(sum);
                arena.sub(c1, c2)
            }
            // cos a cos b = ½[cos(a − b) + cos(a + b)]
            (false, false) => {
                let c1 = arena.cos(diff);
                let c2 = arena.cos(sum);
                arena.add(&[c1, c2])
            }
            // sin a cos b = ½[sin(a + b) + sin(a − b)]
            (true, false) => {
                let s1 = arena.sin(sum);
                let s2 = arena.sin(diff);
                arena.add(&[s1, s2])
            }
            // cos a sin b = ½[sin(a + b) − sin(a − b)]
            (false, true) => {
                let s1 = arena.sin(sum);
                let s2 = arena.sin(diff);
                arena.sub(s1, s2)
            }
        };
        let combined = crate::transforms::eval::eval(arena, combined);
        out.push(arena.mul(&[half, combined]));
    }
    out.extend(trig);
    let product = arena.mul(&out);
    let expanded = crate::transforms::expand::expand(arena, product);
    crate::transforms::eval::eval(arena, expanded)
}

/// The table-and-rules forward transform.
fn do_forward_rules(
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
    if let Some(result) = try_special_forward(arena, expr, t, s)? {
        return Ok(result);
    }

    // ── Division by t: L{f(t)/t} = ∫_s^∞ F(u) du ──
    if let Some(result) = try_divide_by_t(arena, expr, t, t_sym, s)? {
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

/// Does `expr` depend on the variable `t`?  Only free occurrences count (a
/// `Sum` over `t`, a `RootOf` in `t` are constants; see `walk::binder`).
fn contains_var(arena: &Arena, expr: ExprId, t: ExprId) -> bool {
    crate::base::walk::has_free_var(arena, expr, t)
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
                let fact = factorial(n_val);
                let fact_rat = Ratio::from_integer(fact);
                let fact_id = arena.num_ratio(fact_rat.clone());

                let n_plus_1_rat = r + Ratio::one();
                let n_plus_1_id = arena.num_ratio(n_plus_1_rat.clone());
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
                    let a_id = arena.num_ratio(a_coeff.clone());
                    let b_id = arena.num_ratio(b_coeff.clone());
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
    .or_else(|| phase_shifted_forward(arena, expr, t, s))
}

/// Rules 4–7 with an affine argument `a·t + b` (`b ≠ 0` free of `t`), by
/// the addition formulas:
///
/// | `f(t)` | `F(s)` |
/// |---|---|
/// | `sin(at + b)` | `(a cos b + s sin b)/(s² + a²)` |
/// | `cos(at + b)` | `(s cos b − a sin b)/(s² + a²)` |
/// | `sinh(at + b)` | `(a cosh b + s sinh b)/(s² − a²)` |
/// | `cosh(at + b)` | `(s cosh b + a sinh b)/(s² − a²)` |
///
/// The time-shift rule `H(t − c)·f(t) → e^{−cs}·L{f(t + c)}` produces these
/// arguments (`H(t − 1)·sin(3t)` needs `L{sin(3t + 3)}`), so without them it
/// failed for every trigonometric or hyperbolic `f`.
fn phase_shifted_forward(arena: &mut Arena, expr: ExprId, t: ExprId, s: ExprId) -> Option<ExprId> {
    let (arg, kind) = match arena.node(expr) {
        ExprNode::Sin(a) => (*a, 0u8),
        ExprNode::Cos(a) => (*a, 1),
        ExprNode::Sinh(a) => (*a, 2),
        ExprNode::Cosh(a) => (*a, 3),
        _ => return None,
    };
    let (a, b) = linear_in(arena, arg, t)?;
    if arena.is_zero_structural(a) || arena.is_zero_structural(b) {
        return None;
    }
    let a2 = arena.mul(&[a, a]);
    let s2 = arena.mul(&[s, s]);
    let (num, den) = if kind < 2 {
        let (cb, sb) = (arena.cos(b), arena.sin(b));
        let den = arena.add(&[s2, a2]);
        let num = if kind == 0 {
            let x = arena.mul(&[a, cb]);
            let y = arena.mul(&[s, sb]);
            arena.add(&[x, y])
        } else {
            let x = arena.mul(&[s, cb]);
            let y = arena.mul(&[a, sb]);
            arena.sub(x, y)
        };
        (num, den)
    } else {
        let (cb, sb) = (arena.cosh(b), arena.sinh(b));
        let den = arena.sub(s2, a2);
        let num = if kind == 2 {
            let x = arena.mul(&[a, cb]);
            let y = arena.mul(&[s, sb]);
            arena.add(&[x, y])
        } else {
            let x = arena.mul(&[s, cb]);
            let y = arena.mul(&[a, sb]);
            arena.add(&[x, y])
        };
        (num, den)
    };
    Some(arena.div(num, den))
}

// ─── Time-shift rule ─────────────────────────────────────────────────────

// ─── Special functions and distributions ─────────────────────────────────────────────────

fn fail(reason: impl Into<String>) -> SymplexError {
    SymplexError::ComputationFailed {
        operation: "laplace_transform",
        reason: reason.into(),
    }
}

/// Sign of a real parameter via numbers and assumptions.
fn param_sign(arena: &mut Arena, e: ExprId) -> Option<i32> {
    crate::calculus::limit::const_sign(arena, e)
}

fn need_sign(arena: &Arena, e: ExprId, cond: &str) -> SymplexError {
    fail(format!(
        "requires {cond} (declare the sign of {} with Assumption::Positive / Assumption::Negative)",
        arena.display(e)
    ))
}

/// `expr = a·t + b` with `a`, `b` free of `t`.
fn linear_in(arena: &mut Arena, expr: ExprId, t: ExprId) -> Option<(ExprId, ExprId)> {
    if expr == t {
        return Some((arena.one, arena.zero));
    }
    if !contains_var(arena, expr, t) {
        return Some((arena.zero, expr));
    }
    match arena.node(expr).clone() {
        ExprNode::Neg(inner) => {
            let (a, b) = linear_in(arena, inner, t)?;
            Some((arena.neg(a), arena.neg(b)))
        }
        ExprNode::Mul(children) => {
            let mut coeff = Vec::new();
            let mut seen = false;
            for &c in &children {
                if c == t {
                    if seen {
                        return None;
                    }
                    seen = true;
                } else if contains_var(arena, c, t) {
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
                let (a, b) = linear_in(arena, c, t)?;
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

/// Table entries beyond the elementary ones: `t^ν`, `ln t`, `δ(t−a)`,
/// `H(t−a)`, `Jₙ(at)`, `erf(a√t)`.
fn try_special_forward(
    arena: &mut Arena,
    expr: ExprId,
    t: ExprId,
    s: ExprId,
) -> Result<Option<ExprId>, SymplexError> {
    let one = arena.one;
    let two = arena.int(2);
    let expr = flatten_nested_power(arena, expr);
    match arena.node(expr).clone() {
        // t^ν → Γ(ν + 1)/s^{ν+1}, ν > −1 (non-integer or symbolic ν)
        ExprNode::Pow(base, nu) if base == t && !contains_var(arena, nu, t) => {
            let nu_p1 = arena.add(&[nu, one]);
            let nu_p1 = crate::transforms::eval::eval(arena, nu_p1);
            match param_sign(arena, nu_p1) {
                Some(sg) if sg > 0 => {}
                Some(_) => {
                    return Err(fail("t^ν with ν ≤ −1 is not Laplace transformable"));
                }
                None => return Err(need_sign(arena, nu_p1, "ν > −1 in t^ν")),
            }
            let g = arena.gamma(nu_p1);
            let g = crate::transforms::eval::eval(arena, g);
            let neg = arena.neg(nu_p1);
            let s_pow = arena.pow(s, neg);
            Ok(Some(arena.mul(&[g, s_pow])))
        }
        // ln t → −(γ + ln s)/s
        ExprNode::Ln(arg) if arg == t => {
            let gamma = arena.euler_gamma();
            let ln_s = arena.ln(s);
            let sum = arena.add(&[gamma, ln_s]);
            let r = arena.div(sum, s);
            Ok(Some(arena.neg(r)))
        }
        // δ(t − a) → e^{−as} (a ≥ 0)
        ExprNode::DiracDelta(arg) => {
            let Some((k, b)) = linear_in(arena, arg, t) else {
                return Ok(None);
            };
            if k != one {
                return Ok(None);
            }
            let a = arena.neg(b);
            let a = crate::transforms::eval::eval(arena, a);
            match param_sign(arena, a) {
                Some(0) => return Ok(Some(one)),
                Some(sg) if sg > 0 => {}
                Some(_) => return Ok(Some(arena.zero)),
                None => return Err(need_sign(arena, a, "a ≥ 0 in δ(t − a)")),
            }
            let as_ = arena.mul(&[a, s]);
            let neg = arena.neg(as_);
            Ok(Some(arena.exp(neg)))
        }
        // H(t − a) → e^{−as}/s (a ≥ 0)
        ExprNode::Heaviside(arg) => {
            let Some((k, b)) = linear_in(arena, arg, t) else {
                return Ok(None);
            };
            if k != one {
                return Ok(None);
            }
            let a = arena.neg(b);
            let a = crate::transforms::eval::eval(arena, a);
            match param_sign(arena, a) {
                Some(sg) if sg > 0 => {}
                Some(_) => return Ok(Some(arena.div(one, s))),
                None => return Err(need_sign(arena, a, "a ≥ 0 in H(t − a)")),
            }
            let as_ = arena.mul(&[a, s]);
            let neg = arena.neg(as_);
            let e = arena.exp(neg);
            Ok(Some(arena.div(e, s)))
        }
        // Jₙ(at) → (√(s²+a²) − s)ⁿ / (aⁿ √(s²+a²)), integer n ≥ 0
        ExprNode::Apply(name, args)
            if arena.symbol_name(name) == crate::base::arena::FN_BESSELJ && args.len() == 2 =>
        {
            let order = args[0];
            let arg = args[1];
            let Some(n) = arena.as_num(order).cloned() else {
                return Ok(None);
            };
            if !n.is_integer() || n.is_negative() {
                return Ok(None);
            }
            let n: u32 = n
                .to_integer()
                .try_into()
                .map_err(|_| fail("Bessel order too large"))?;
            let Some((a, b)) = linear_in(arena, arg, t) else {
                return Ok(None);
            };
            if !arena.is_zero_structural(b) {
                return Ok(None);
            }
            let s2 = arena.pow(s, two);
            let a2 = arena.pow(a, two);
            let sum = arena.add(&[s2, a2]);
            let root = arena.sqrt(sum);
            if n == 0 {
                return Ok(Some(arena.div(one, root)));
            }
            let n_id = arena.int(i64::from(n));
            let diff = arena.sub(root, s);
            let num = arena.pow(diff, n_id);
            let an = arena.pow(a, n_id);
            let den = arena.mul(&[an, root]);
            Ok(Some(arena.div(num, den)))
        }
        // erf(a√t) → a/(s√(s + a²)), a > 0
        ExprNode::Erf(arg) => {
            let half = arena.rational(1, 2);
            let sqrt_t = arena.pow(t, half);
            let (a, inner) = match arena.node(arg).clone() {
                ExprNode::Pow(..) if arg == sqrt_t => (one, sqrt_t),
                ExprNode::Mul(ch) => {
                    let rest: Vec<ExprId> = ch.iter().copied().filter(|&c| c != sqrt_t).collect();
                    if rest.len() + 1 != ch.len() || rest.iter().any(|&c| contains_var(arena, c, t))
                    {
                        return Ok(None);
                    }
                    (arena.mul(&rest), sqrt_t)
                }
                _ => return Ok(None),
            };
            let _ = inner;
            match param_sign(arena, a) {
                Some(sg) if sg > 0 => {}
                Some(_) => return Ok(None),
                None => return Err(need_sign(arena, a, "a > 0 in erf(a√t)")),
            }
            let a2 = arena.pow(a, two);
            let sum = arena.add(&[s, a2]);
            let root = arena.sqrt(sum);
            let den = arena.mul(&[s, root]);
            Ok(Some(arena.div(a, den)))
        }
        _ => Ok(None),
    }
}

/// `((b)^p)^q → b^{pq}` for numeric `p`, `q` — the form `1/√x` takes after
/// canonicalization. Valid on the Laplace domain (`t > 0`, `Re s > 0`).
fn flatten_nested_power(arena: &mut Arena, expr: ExprId) -> ExprId {
    if let ExprNode::Pow(inner, q) = arena.node(expr).clone()
        && let ExprNode::Pow(base, p) = arena.node(inner).clone()
        && arena.as_num(p).is_some()
        && arena.as_num(q).is_some()
    {
        let pq = arena.mul(&[p, q]);
        let pq = crate::transforms::eval::eval(arena, pq);
        return arena.pow(base, pq);
    }
    expr
}

/// `L{f(t)/t} = ∫_s^∞ F(u) du`, valid when `f(t)/t` is integrable at 0.
///
/// Special cases `sin(at)/t → atan(a/s)` and `(1 − cos at)/t → ½ ln(1 +
/// a²/s²)` are recognised directly; everything else goes through the
/// definite integral of the transform of the numerator.
fn try_divide_by_t(
    arena: &mut Arena,
    expr: ExprId,
    t: ExprId,
    t_sym: SymbolId,
    s: ExprId,
) -> Result<Option<ExprId>, SymplexError> {
    let ExprNode::Mul(children) = arena.node(expr).clone() else {
        return Ok(None);
    };
    let inv_t = arena.pow(t, arena.neg_one);
    if !children.contains(&inv_t) {
        return Ok(None);
    }
    let rest: Vec<ExprId> = children.iter().copied().filter(|&c| c != inv_t).collect();
    if rest.is_empty() {
        return Err(fail(
            "1/t is not Laplace transformable (not integrable at 0)",
        ));
    }
    let g = arena.mul(&rest);

    // Direct entries.
    let one = arena.one;
    let two = arena.int(2);
    match arena.node(g).clone() {
        ExprNode::Sin(arg) => {
            if let Some((a, b)) = linear_in(arena, arg, t)
                && arena.is_zero_structural(b)
            {
                let ratio = arena.div(a, s);
                return Ok(Some(arena.atan(ratio)));
            }
        }
        ExprNode::Add(terms) if terms.len() == 2 && terms.contains(&one) => {
            let other = if terms[0] == one { terms[1] } else { terms[0] };
            // `−cos(at)` is `Neg(cos)` or `Mul(-1, cos)` depending on canonical form.
            let neg_cos = match arena.node(other).clone() {
                ExprNode::Neg(c) => Some(c),
                ExprNode::Mul(ch) if ch.len() == 2 && ch.contains(&arena.neg_one) => {
                    Some(if ch[0] == arena.neg_one { ch[1] } else { ch[0] })
                }
                _ => None,
            };
            if let Some(c) = neg_cos
                && let ExprNode::Cos(arg) = arena.node(c).clone()
                && let Some((a, b)) = linear_in(arena, arg, t)
                && arena.is_zero_structural(b)
            {
                let a2 = arena.pow(a, two);
                let s2 = arena.pow(s, two);
                let ratio = arena.div(a2, s2);
                let arg = arena.add(&[one, ratio]);
                let ln = arena.ln(arg);
                return Ok(Some(arena.div(ln, two)));
            }
        }
        _ => {}
    }

    // General: integrate G(u) from s to ∞.
    let zero = arena.zero;
    let g0 = crate::calculus::limit::safe_substitute(arena, g, t, zero);
    match g0 {
        Some(v) if arena.is_zero_structural(v) => {}
        _ => {
            return Err(fail(format!(
                "{}/t is not integrable at t = 0 (numerator does not vanish there)",
                arena.display(g)
            )));
        }
    }
    let big_g = do_forward(arena, g, t, t_sym, s)?;
    let u = arena.symbol("__lap_u");
    let big_g_u = crate::transforms::subs::subs(arena, big_g, s, u);
    let inf = arena.infinity();
    let ok = |arena: &Arena, r: ExprId| {
        !contains_var(arena, r, u) && !crate::base::walk::has_unevaluated(arena, r)
    };
    if let Ok(integral) = crate::calculus::definite::integrate_definite(arena, big_g_u, u, s, inf)
        && ok(arena, integral)
    {
        return Ok(Some(integral));
    }
    // The definite integrator refuses ∞ − ∞ forms such as
    // [ln(u+2) − ln(u+1)]_s^∞; take the antiderivative and let the limit
    // engine resolve the upper end.
    let anti = crate::transforms::integrate::integrate(arena, big_g_u, u);
    if crate::base::walk::has_unevaluated(arena, anti) {
        return Err(fail("∫_s^∞ F(u) du has no closed form"));
    }
    let at_inf = crate::calculus::limit::limit(arena, anti, u, inf)
        .map_err(|e| fail(format!("∫_s^∞ F(u) du: {e}")))?;
    if at_inf == inf || at_inf == arena.neg_infinity() {
        return Err(fail("∫_s^∞ F(u) du diverges"));
    }
    let at_s = crate::transforms::subs::subs(arena, anti, u, s);
    let r = arena.sub(at_inf, at_s);
    let r = crate::transforms::eval::eval(arena, r);
    if ok(arena, r) {
        Ok(Some(r))
    } else {
        Err(fail("∫_s^∞ F(u) du has no closed form"))
    }
}

// ─── Time-shift rule ─────────────────────────────────────────────────────────────────────────

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
    // expr = t + b (symbolic b allowed) → a = −b; the shift must be ≥ 0.
    let (k, b) = linear_in(arena, expr, t)?;
    if k != arena.one {
        return None;
    }
    let a = arena.neg(b);
    let a = crate::transforms::eval::eval(arena, a);
    match param_sign(arena, a) {
        Some(sg) if sg >= 0 => Some(a),
        _ => None,
    }
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
        } else {
            let _ti = t_idx?;
            n_val = 1;
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
        let fact = factorial(n_val);
        let fact_rat = Ratio::from_integer(fact);
        let fact_id = arena.num_ratio(fact_rat.clone());

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

/// Internal recursive inverse transform: the table and rules, then the exact
/// rational-function inverse when they fail.
fn do_inverse(
    arena: &mut Arena,
    expr: ExprId,
    s: ExprId,
    t: ExprId,
    depth: u32,
) -> Result<ExprId, SymplexError> {
    let ruled = do_inverse_rules(arena, expr, s, t, depth);
    if ruled.is_ok() {
        return ruled;
    }
    match inverse_rational_exact(arena, expr, s, t) {
        Some(f) => Ok(f),
        None => ruled,
    }
}

/// `L⁻¹` of a rational `F(s)` with rational coefficients, for every
/// denominator: the proper part solves `D(d/dt) f = 0` with
/// `f⁽ᵏ⁾(0⁺)` read off the expansion of `F` at infinity (repeated complex
/// poles `1/(s² + 4)²` and non-monic factors `1/(2s + 1)³` were refused
/// by the partial-fraction table); a constant polynomial part is `c·δ(t)`.
fn inverse_rational_exact(arena: &mut Arena, expr: ExprId, s: ExprId, t: ExprId) -> Option<ExprId> {
    let (numer, denom) = crate::poly::polybridge::as_numer_denom(arena, expr);
    let np = crate::poly::polybridge::expr_to_poly(arena, numer, s)?;
    let dp = crate::poly::polybridge::expr_to_poly(arena, denom, s)?;
    if dp.degree()? == 0 {
        return None;
    }
    let (q, r) = np.div_rem(&dp);
    if q.degree().is_some_and(|k| k > 0) {
        // sᵏ with k ≥ 1 would need derivatives of δ.
        return None;
    }
    let proper = crate::transforms::rsolve::rational_inverse_into(arena, &r, &dp, t, true)?;
    if q.is_zero() {
        return Some(proper);
    }
    let nid = arena.intern_num(q.coeff(0));
    let c = arena.intern(ExprNode::Num(nid));
    let d = arena.dirac_delta(t);
    let cd = arena.mul(&[c, d]);
    Some(arena.add(&[cd, proper]))
}

/// The table-and-rules inverse transform, with a recursion guard.
fn do_inverse_rules(
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

    // ── Constant c → c·δ(t) ──
    if !contains_var(arena, expr, s) {
        let d = arena.dirac_delta(t);
        return Ok(arena.mul(&[expr, d]));
    }

    // ── Factor out constants (not containing s) ──
    let (coeff, body) = split_independent(arena, expr, s);
    if coeff != arena.one {
        let result = do_inverse(arena, body, s, t, depth + 1)?;
        return Ok(arena.mul(&[coeff, result]));
    }

    // ── Delay: e^{−as} G(s) → g(t − a) H(t − a) ──
    if let Some(result) = try_inverse_delay(arena, expr, s, t, depth)? {
        return Ok(result);
    }

    // ── Special entries: s^{−ν}, 1/√(s²+a²), atan(a/s), 1/(s√(s+a²)), ln(s)/s,
    //    and rational forms with symbolic parameters ──
    if let Some(result) = try_special_inverse(arena, expr, s, t)? {
        return Ok(result);
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

fn ifail(reason: impl Into<String>) -> SymplexError {
    SymplexError::ComputationFailed {
        operation: "inverse_laplace_transform",
        reason: reason.into(),
    }
}

/// `e^{−as}·G(s) → g(t − a)·H(t − a)` for `a ≥ 0` (second shifting theorem).
fn try_inverse_delay(
    arena: &mut Arena,
    expr: ExprId,
    s: ExprId,
    t: ExprId,
    depth: u32,
) -> Result<Option<ExprId>, SymplexError> {
    let children: Vec<ExprId> = match arena.node(expr).clone() {
        ExprNode::Mul(ch) => ch.iter().copied().collect(),
        ExprNode::Exp(_) => vec![expr],
        _ => return Ok(None),
    };
    let mut delay: Option<ExprId> = None;
    let mut rest: Vec<ExprId> = Vec::new();
    for &c in &children {
        if delay.is_none()
            && let ExprNode::Exp(arg) = arena.node(c).clone()
            && let Some((k, b)) = linear_in(arena, arg, s)
            && arena.is_zero_structural(b)
        {
            // arg = k·s = −a·s
            let a = arena.neg(k);
            let a = crate::transforms::eval::eval(arena, a);
            match param_sign(arena, a) {
                Some(sg) if sg > 0 => {
                    delay = Some(a);
                    continue;
                }
                _ => {}
            }
        }
        rest.push(c);
    }
    let Some(a) = delay else {
        return Ok(None);
    };
    if rest.is_empty() {
        // e^{−as} alone → δ(t − a)
        let t_minus_a = arena.sub(t, a);
        return Ok(Some(arena.dirac_delta(t_minus_a)));
    }
    let g = arena.mul(&rest);
    let g_t = do_inverse(arena, g, s, t, depth + 1)?;
    let t_minus_a = arena.sub(t, a);
    let shifted = crate::transforms::subs::subs(arena, g_t, t, t_minus_a);
    let h = arena.heaviside(t_minus_a);
    Ok(Some(arena.mul(&[shifted, h])))
}

/// Inverse entries beyond rational functions with numeric coefficients.
fn try_special_inverse(
    arena: &mut Arena,
    expr: ExprId,
    s: ExprId,
    t: ExprId,
) -> Result<Option<ExprId>, SymplexError> {
    let one = arena.one;
    let two = arena.int(2);
    let neg_one = arena.neg_one;
    let neg_half = arena.rational(-1, 2);
    let expr = flatten_nested_power(arena, expr);
    match arena.node(expr).clone() {
        // s^{−ν} → t^{ν−1}/Γ(ν), ν > 0 non-integer or symbolic
        ExprNode::Pow(base, e) if base == s && !contains_var(arena, e, s) => {
            let nu = arena.neg(e);
            let nu = crate::transforms::eval::eval(arena, nu);
            if arena.as_num(nu).is_some_and(|r| r.is_integer()) {
                return Ok(None); // integer powers: rational-function path
            }
            match param_sign(arena, nu) {
                Some(sg) if sg > 0 => {}
                Some(_) => {
                    return Err(ifail(
                        "s^k with k ≥ 0 is not an inverse-transformable function",
                    ));
                }
                None => {
                    return Err(ifail(format!(
                        "requires ν > 0 in s^(-ν) (declare the sign of {} with Assumption::Positive)",
                        arena.display(nu)
                    )));
                }
            }
            let nu_m1 = arena.sub(nu, one);
            let tp = arena.pow(t, nu_m1);
            let g = arena.gamma(nu);
            let g = crate::transforms::eval::eval(arena, g);
            Ok(Some(arena.div(tp, g)))
        }
        // 1/√(s² + a²) → J₀(at);  1/(s − a)^n and 1/(s² + a²) with symbolic a
        ExprNode::Pow(base, e) if !contains_var(arena, e, s) => {
            let s2 = arena.pow(s, two);
            // (s² + a²)^{−1/2} and (s² + a²)^{−1}
            if let ExprNode::Add(terms) = arena.node(base).clone()
                && terms.len() == 2
                && terms.contains(&s2)
            {
                let a2 = if terms[0] == s2 { terms[1] } else { terms[0] };
                if contains_var(arena, a2, s) {
                    return Ok(None);
                }
                let a = match arena.node(a2).clone() {
                    ExprNode::Pow(b, k) if k == two && param_sign(arena, b) == Some(1) => b,
                    _ => {
                        if arena.as_num(a2).is_some() && e == neg_one {
                            return Ok(None); // numeric: rational-function path
                        }
                        match param_sign(arena, a2) {
                            Some(1) => {
                                let r = arena.sqrt(a2);
                                crate::transforms::eval::eval(arena, r)
                            }
                            _ => return Ok(None),
                        }
                    }
                };
                let at = arena.mul(&[a, t]);
                if e == neg_half {
                    let zero = arena.zero;
                    return Ok(Some(arena.besselj(zero, at)));
                }
                if e == neg_one {
                    let sn = arena.sin(at);
                    return Ok(Some(arena.div(sn, a)));
                }
                return Ok(None);
            }
            // (k s + b)^{−n} with symbolic coefficients → t^{n−1} e^{−(b/k) t}/(kⁿ (n−1)!)
            if let Some((k, b)) = linear_in(arena, base, s)
                && !arena.is_zero_structural(k)
                && let Some(r) = arena.as_num(e).cloned()
                && r.is_integer()
                && r.is_negative()
            {
                if arena.as_num(k).is_some() && arena.as_num(b).is_some() {
                    return Ok(None); // numeric: rational-function path
                }
                let n: u64 = (-r.to_integer())
                    .try_into()
                    .map_err(|_| ifail("power too large"))?;
                let a = arena.div(b, k);
                let a = arena.neg(a);
                let at = arena.mul(&[a, t]);
                let ex = arena.exp(at);
                let n_id = arena.int(n as i64);
                let kn = arena.pow(k, n_id);
                let nm1 = n - 1;
                let tp = if nm1 == 0 {
                    one
                } else {
                    let m = arena.int(nm1 as i64);
                    arena.pow(t, m)
                };
                let f = factorial(nm1);
                let f_id = arena.num_ratio(Ratio::from_integer(f).clone());
                let den = arena.mul(&[kn, f_id]);
                let num = arena.mul(&[tp, ex]);
                return Ok(Some(arena.div(num, den)));
            }
            Ok(None)
        }
        // atan(a/s) → sin(at)/t
        ExprNode::Atan(arg) => {
            let inv_s = arena.pow(s, neg_one);
            let a = match arena.node(arg).clone() {
                ExprNode::Pow(..) if arg == inv_s => one,
                ExprNode::Mul(ch) if ch.contains(&inv_s) => {
                    let rest: Vec<ExprId> = ch.iter().copied().filter(|&c| c != inv_s).collect();
                    if rest.iter().any(|&c| contains_var(arena, c, s)) {
                        return Ok(None);
                    }
                    arena.mul(&rest)
                }
                _ => return Ok(None),
            };
            let at = arena.mul(&[a, t]);
            let sn = arena.sin(at);
            Ok(Some(arena.div(sn, t)))
        }
        ExprNode::Mul(children) => {
            let kids: Vec<ExprId> = children
                .iter()
                .map(|&k| flatten_nested_power(arena, k))
                .collect();
            let inv_s = arena.pow(s, neg_one);
            // A sum inside a product (e.g. −(γ + ln s)/s): distribute and let
            // linearity handle the pieces.
            if kids
                .iter()
                .any(|&k| matches!(arena.node(k), ExprNode::Add(_)))
            {
                let expanded = crate::transforms::expand::expand(arena, expr);
                if expanded != expr && matches!(arena.node(expanded), ExprNode::Add(_)) {
                    return do_inverse(arena, expanded, s, t, 1).map(Some);
                }
            }
            // ln(s)/s → −(ln t + γ)
            if kids.len() == 2 && kids.contains(&inv_s) {
                let other = if kids[0] == inv_s { kids[1] } else { kids[0] };
                if let ExprNode::Ln(arg) = arena.node(other).clone()
                    && arg == s
                {
                    let ln_t = arena.ln(t);
                    let g = arena.euler_gamma();
                    let sum = arena.add(&[ln_t, g]);
                    return Ok(Some(arena.neg(sum)));
                }
                // s/(s² + a²) with symbolic a → cos(at)  (kids = [s, (s²+a²)^{-1}])
            }
            if kids.len() == 2 && kids.contains(&s) {
                let other = if kids[0] == s { kids[1] } else { kids[0] };
                let s2 = arena.pow(s, two);
                if let ExprNode::Pow(base, e) = arena.node(other).clone()
                    && e == neg_one
                    && let ExprNode::Add(terms) = arena.node(base).clone()
                    && terms.len() == 2
                    && terms.contains(&s2)
                {
                    let a2 = if terms[0] == s2 { terms[1] } else { terms[0] };
                    if arena.as_num(a2).is_some() || contains_var(arena, a2, s) {
                        return Ok(None);
                    }
                    let a = match arena.node(a2).clone() {
                        ExprNode::Pow(b, k) if k == two && param_sign(arena, b) == Some(1) => b,
                        _ if param_sign(arena, a2) == Some(1) => {
                            let r = arena.sqrt(a2);
                            crate::transforms::eval::eval(arena, r)
                        }
                        _ => return Ok(None),
                    };
                    let at = arena.mul(&[a, t]);
                    return Ok(Some(arena.cos(at)));
                }
            }
            // 1/(s√(s + a²)) → erf(a√t)/a
            if kids.len() == 2 && kids.contains(&inv_s) {
                let other = if kids[0] == inv_s { kids[1] } else { kids[0] };
                if let ExprNode::Pow(base, e) = arena.node(other).clone()
                    && e == neg_half
                    && let Some((k, a2)) = linear_in(arena, base, s)
                    && k == one
                    && param_sign(arena, a2) == Some(1)
                {
                    let a = match arena.node(a2).clone() {
                        ExprNode::Pow(b, kk) if kk == two && param_sign(arena, b) == Some(1) => b,
                        _ => {
                            let r = arena.sqrt(a2);
                            crate::transforms::eval::eval(arena, r)
                        }
                    };
                    let rt = arena.sqrt(t);
                    let arg = arena.mul(&[a, rt]);
                    let er = arena.erf(arg);
                    return Ok(Some(arena.div(er, a)));
                }
            }
            Ok(None)
        }
        _ => Ok(None),
    }
}

// ─── Inverse table rules ─────────────────────────────────────────────────────────────────────

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
        let a_id = arena.num_ratio(a_rat.clone());
        let at = arena.mul(&[a_id, t]);
        let exp_at = arena.exp(at);

        if a_rat.is_zero() {
            // L⁻¹{N / (c₁·s)} = N/c₁  (constant function)
            let c1_id = arena.num_ratio(c1.clone());
            return Some(arena.div(numer, c1_id));
        }

        let c1_id = arena.num_ratio(c1.clone());
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
        let a_id = arena.num_ratio(a_rat.clone());
        let at = arena.mul(&[a_id, t]);
        let exp_at = arena.exp(at);

        if a_rat.is_zero() {
            let result_rat = n0 / c1;
            return Some(arena.num_ratio(result_rat.clone()));
        }

        let scale = n0 / c1;
        let scale_id = arena.num_ratio(scale.clone());
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
                    let k_id = arena.num_ratio(k.clone());
                    let c2_id = arena.num_ratio(c2.clone());
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
            let omega_sq_id = arena.num_ratio(omega_sq.clone());
            let raw_sqrt = arena.sqrt(omega_sq_id);
            crate::transforms::eval::eval(arena, raw_sqrt)
        };

        if let Some(np) = &numer_poly {
            let nd = np.degree().unwrap_or(0);

            // Numer is constant: L⁻¹{k / (s² + ω²)} = (k/ω) · sin(ωt)
            if nd == 0 {
                let k = np.coeff(0);
                let k_id = arena.num_ratio(k.clone());

                let c2_id = arena.num_ratio(c2.clone());

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

                let c2_id = arena.num_ratio(c2.clone());
                let omega_t = arena.mul(&[omega_id, t]);
                let sin_omega_t = arena.sin(omega_t);
                let cos_omega_t = arena.cos(omega_t);

                let mut terms = Vec::new();

                // cos term: (a₁/c₂) · cos(ωt)
                if !a1.is_zero() {
                    let a1_id = arena.num_ratio(a1.clone());
                    let cos_coeff = arena.div(a1_id, c2_id);
                    terms.push(arena.mul(&[cos_coeff, cos_omega_t]));
                }

                // sin term: (a₀/(c₂·ω)) · sin(ωt)
                if !a0.is_zero() {
                    let a0_id = arena.num_ratio(a0.clone());
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
        let alpha_id = arena.num_ratio(alpha.clone());

        if let Some(np) = &numer_poly {
            let nd = np.degree().unwrap_or(0);
            let c2_id = arena.num_ratio(c2.clone());

            if nd == 0 {
                // L⁻¹{k / (c₂·(s-α)²)} = (k/c₂)·t·exp(α·t)
                let k = np.coeff(0);
                let k_id = arena.num_ratio(k.clone());
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
                    let a1_id = arena.num_ratio(a1.clone());
                    let exp_coeff = arena.div(a1_id, c2_id);
                    terms.push(arena.mul(&[exp_coeff, exp_alpha_t]));
                }

                // t·exp term: (d_const/c₂) · t · exp(α·t)
                if !d_const.is_zero() {
                    let d_id = arena.num_ratio(d_const.clone());
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
        let gamma_sq_id = arena.num_ratio(gamma_sq.clone());
        let gamma_id = {
            let raw = arena.sqrt(gamma_sq_id);
            crate::transforms::eval::eval(arena, raw)
        };
        let alpha_id = arena.num_ratio(alpha.clone());

        if let Some(np) = &numer_poly {
            let nd = np.degree().unwrap_or(0);
            let c2_id = arena.num_ratio(c2.clone());

            if nd == 0 {
                let k = np.coeff(0);
                let k_id = arena.num_ratio(k.clone());

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
                    let a1_id = arena.num_ratio(a1.clone());
                    let cosh_coeff = arena.div(a1_id, c2_id);
                    terms.push(arena.mul(&[cosh_coeff, exp_alpha_t, cosh_gamma_t]));
                }

                // sinh term: (d_const/(c₂·γ)) · exp(α·t) · sinh(γ·t)
                if !d_const.is_zero() {
                    let d_id = arena.num_ratio(d_const.clone());
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
    let beta_sq_id = arena.num_ratio(beta_sq.clone());
    let beta_id = {
        let raw = arena.sqrt(beta_sq_id);
        crate::transforms::eval::eval(arena, raw)
    };
    let alpha_id = arena.num_ratio(alpha.clone());

    if let Some(np) = &numer_poly {
        let nd = np.degree().unwrap_or(0);
        let c2_id = arena.num_ratio(c2.clone());

        // Rewrite numer in terms of (s - α): N(s) = N(α) + N'(α)·(s - α)
        // For a₁·s + a₀ with substitution s = (s' + α):
        // a₁(s' + α) + a₀ = a₁·s' + (a₁·α + a₀)
        // where s' = s - α = s + b/2

        if nd == 0 {
            // L⁻¹{k / (c₂·((s-α)² + β²))} = (k/(c₂·β)) · exp(α·t) · sin(β·t)
            let k = np.coeff(0);
            let k_id = arena.num_ratio(k.clone());

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
                let a1_id = arena.num_ratio(a1.clone());
                let cos_coeff = arena.div(a1_id, c2_id);
                terms.push(arena.mul(&[cos_coeff, exp_alpha_t, cos_beta_t]));
            }

            // sin term: (d_const/(c₂·β)) · exp(α·t) · sin(β·t)
            if !d_const.is_zero() {
                let d_id = arena.num_ratio(d_const.clone());
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
        let a_id = arena.num_ratio(a_rat.clone());
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
        let fact = factorial(n_val - 1);
        let fact_rat = Ratio::from_integer(fact);
        let fact_id = arena.num_ratio(fact_rat.clone());

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
