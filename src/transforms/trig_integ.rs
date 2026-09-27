//! Trigonometric power integration.
//!
//! Handles integrals of the form:
//! - `∫ sin^n(x) dx` via recursive reduction formula
//! - `∫ cos^n(x) dx` via recursive reduction formula
//! - `∫ sin^m(x)·cos^n(x) dx` when one exponent is odd (Pythagorean substitution)
//!   or both even (reduction formula)

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode, SymbolId};

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Zero};

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Compute binomial coefficient C(n, k) using BigInt to avoid overflow.
fn binom(n: u64, k: u64) -> BigInt {
    if k > n {
        return BigInt::zero();
    }
    if k == 0 || k == n {
        return BigInt::one();
    }
    // Use the smaller of k and n-k for efficiency
    let k = k.min(n - k);
    let mut result = BigInt::one();
    for i in 0..k {
        result = result * BigInt::from(n - i) / BigInt::from(i + 1);
    }
    result
}

/// Does `expr` depend on `var_sym` — does it occur *free* in it?  The
/// binder-aware test the integrator uses
/// ([`crate::base::walk::has_free_symbol`]); up to 0.28.0 this one was
/// structural (and, without a visited set, exponential on a shared DAG).
fn contains_var(arena: &Arena, expr: ExprId, var_sym: SymbolId) -> bool {
    crate::base::walk::has_free_symbol(arena, expr, var_sym)
}

/// Try to extract a [`Ratio<BigInt>`] as an `i64`.
///
/// Returns `Some(n)` when the expression is a numeric literal whose value
/// is an integer that fits in `i64`.
fn as_i64(arena: &Arena, id: ExprId) -> Option<i64> {
    let r = arena.as_num(id)?;
    if !r.is_integer() {
        return None;
    }
    let big = r.to_integer();
    i64::try_from(&big).ok()
}

/// A trigonometric or hyperbolic power `f(u)^n` of an argument `u`.
#[derive(Clone, Copy)]
struct TrigPower {
    arg: ExprId,
    exp: i64,
}

/// Decompose `factor` into a power of the function `func` picks out:
/// `f(u)` → `(u, 1)`, `Pow(f(u), n)` → `(u, n)` for an integer `n`.
fn extract_power(
    arena: &Arena,
    factor: ExprId,
    func: fn(&ExprNode) -> Option<ExprId>,
) -> Option<TrigPower> {
    if let Some(arg) = func(arena.node(factor)) {
        return Some(TrigPower { arg, exp: 1 });
    }
    match arena.node(factor) {
        ExprNode::Pow(base, exp) => {
            let arg = func(arena.node(*base))?;
            Some(TrigPower {
                arg,
                exp: as_i64(arena, *exp)?,
            })
        }
        _ => None,
    }
}

fn sin_arg(node: &ExprNode) -> Option<ExprId> {
    match node {
        ExprNode::Sin(u) => Some(*u),
        _ => None,
    }
}

fn cos_arg(node: &ExprNode) -> Option<ExprId> {
    match node {
        ExprNode::Cos(u) => Some(*u),
        _ => None,
    }
}

fn tan_arg(node: &ExprNode) -> Option<ExprId> {
    match node {
        ExprNode::Tan(u) => Some(*u),
        _ => None,
    }
}

fn sinh_arg(node: &ExprNode) -> Option<ExprId> {
    match node {
        ExprNode::Sinh(u) => Some(*u),
        _ => None,
    }
}

fn cosh_arg(node: &ExprNode) -> Option<ExprId> {
    match node {
        ExprNode::Cosh(u) => Some(*u),
        _ => None,
    }
}

/// `du/dx` for an argument `u = a·x + b` (`a` free of `x`, not
/// structurally zero), or `None` when `u` is not linear in `x`.
fn linear_rate(arena: &mut Arena, u: ExprId, var: ExprId, var_sym: SymbolId) -> Option<ExprId> {
    if u == var {
        return Some(arena.one);
    }
    if !contains_var(arena, u, var_sym) {
        return None;
    }
    let a = crate::transforms::diff::diff(arena, u, var);
    (a != arena.zero && !contains_var(arena, a, var_sym)).then_some(a)
}

/// Largest `|m|`, `|n|` for which `∫ sin^m·cos^n` is expanded: the closed
/// forms have `O(|m| + |n|)` terms (`O(|m|·|n|)` intermediate pairs when both
/// are negative) and the reduction formulas recurse once per step of 2.
const MAX_TRIG_POWER: i64 = 256;

/// Are both exponents within [`MAX_TRIG_POWER`]?
fn within_power_bound(m: i64, n: i64) -> bool {
    m.checked_abs().is_some_and(|a| a <= MAX_TRIG_POWER)
        && n.checked_abs().is_some_and(|a| a <= MAX_TRIG_POWER)
}

/// `sin(var)^n` → `n` (the tests' view of [`extract_power`]).
#[cfg(test)]
fn extract_sin_power(arena: &Arena, factor: ExprId, var: ExprId) -> Option<i64> {
    extract_power(arena, factor, sin_arg)
        .filter(|p| p.arg == var)
        .map(|p| p.exp)
}

/// `cos(var)^n` → `n` (the tests' view of [`extract_power`]).
#[cfg(test)]
fn extract_cos_power(arena: &Arena, factor: ExprId, var: ExprId) -> Option<i64> {
    extract_power(arena, factor, cos_arg)
        .filter(|p| p.arg == var)
        .map(|p| p.exp)
}

/// Largest `m + n` that [`linearize_sin_cos`] expands.
const MAX_LINEARIZE_DEGREE: u32 = 12;

/// `sin(u)^m·cos(u)^n` (`m, n ≥ 0`, `m + n ≤` [`MAX_LINEARIZE_DEGREE`]) as
/// `Σₖ (αₖ·cos(k·u) + βₖ·sin(k·u))` with rational `αₖ`, `βₖ`: one factor
/// at a time by the product-to-sum identities `cos(ku)·cos u =
/// (cos((k+1)u) + cos((k−1)u))/2` and its three siblings (Fu's `TRpower`
/// for a product).  `None` beyond the degree bound.
pub(crate) fn linearize_sin_cos(arena: &mut Arena, u: ExprId, m: u32, n: u32) -> Option<ExprId> {
    use std::collections::BTreeMap;
    if m + n > MAX_LINEARIZE_DEGREE {
        return None;
    }
    let half = Ratio::new(BigInt::one(), BigInt::from(2));
    // k ↦ (αₖ, βₖ); β₀ is always 0.
    let mut terms: BTreeMap<u32, (Ratio<BigInt>, Ratio<BigInt>)> = BTreeMap::new();
    terms.insert(0, (Ratio::one(), Ratio::zero()));
    let zero = || (Ratio::<BigInt>::zero(), Ratio::<BigInt>::zero());
    for step in 0..m + n {
        let by_sin = step < m;
        let mut next: BTreeMap<u32, (Ratio<BigInt>, Ratio<BigInt>)> = BTreeMap::new();
        for (k, (a, b)) in terms {
            let (ah, bh) = (&a * &half, &b * &half);
            if by_sin {
                // cos(ku)·sin u = (sin((k+1)u) − sin((k−1)u))/2,
                // sin(ku)·sin u = (cos((k−1)u) − cos((k+1)u))/2.
                let up = next.entry(k + 1).or_insert_with(zero);
                up.1 += &ah;
                up.0 -= &bh;
                let down = next.entry(k.abs_diff(1)).or_insert_with(zero);
                // sin(−u) = −sin u; sin 0 = 0.
                if k >= 1 {
                    down.1 -= &ah;
                } else {
                    down.1 += &ah;
                }
                down.0 += &bh;
            } else {
                // cos(ku)·cos u = (cos((k+1)u) + cos((k−1)u))/2,
                // sin(ku)·cos u = (sin((k+1)u) + sin((k−1)u))/2.
                let up = next.entry(k + 1).or_insert_with(zero);
                up.0 += &ah;
                up.1 += &bh;
                let down = next.entry(k.abs_diff(1)).or_insert_with(zero);
                down.0 += &ah;
                if k >= 1 {
                    down.1 += &bh;
                } else {
                    down.1 -= &bh;
                }
            }
        }
        terms = next;
    }
    let mut out: Vec<ExprId> = Vec::new();
    for (k, (a, b)) in terms {
        if k == 0 {
            if !a.is_zero() {
                out.push(arena.num_ratio(a));
            }
            continue;
        }
        let k_id = arena.int(i64::from(k));
        let ku = arena.mul(&[k_id, u]);
        for (c, is_sin) in [(a, false), (b, true)] {
            if c.is_zero() {
                continue;
            }
            let g = if is_sin { arena.sin(ku) } else { arena.cos(ku) };
            let c_id = arena.num_ratio(c);
            out.push(arena.mul(&[c_id, g]));
        }
    }
    Some(match out.len() {
        0 => arena.zero,
        1 => out[0],
        _ => arena.add(&out),
    })
}

/// Is `e` a polynomial in the variable and in `exp(u)`, `sin(u)`, `cos(u)`
/// for arguments `u` linear in the variable (non-negative integer powers,
/// sums and products only)?  Products of such factors become sums of
/// terms `xᵏ·e^{ℓ(x)}·sin/cos(ℓ′(x))` by power reduction and
/// product-to-sum, each of which by parts and the cyclic rule integrate.
fn trig_exp_polynomial(arena: &mut Arena, e: ExprId, var: ExprId, var_sym: SymbolId) -> bool {
    let mut arguments: Vec<ExprId> = Vec::new();
    let mut stack = vec![e];
    let mut seen: rustc_hash::FxHashSet<ExprId> = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) || !contains_var(arena, id, var_sym) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Symbol(_) => {}
            ExprNode::Add(ch) | ExprNode::Mul(ch) => stack.extend(ch.iter().copied()),
            ExprNode::Pow(b, n) if as_i64(arena, *n).is_some_and(|k| k >= 0) => stack.push(*b),
            ExprNode::Exp(u) | ExprNode::Sin(u) | ExprNode::Cos(u) => arguments.push(*u),
            _ => return false,
        }
    }
    arguments
        .into_iter()
        .all(|u| linear_rate(arena, u, var, var_sym).is_some())
}

/// The product of `factors` with its `sin(u)^m·cos(u)^n` factors (`m, n ≥
/// 0`, `m + n ≥ 2`) for one argument `u` linear in the variable replaced
/// by their [`linearize_sin_cos`] form; `None` when there is no such group,
/// a sine or cosine factor has a negative power, or a factor is not a
/// polynomial in the variable, `exp`, `sin` and `cos` of linear arguments
/// ([`trig_exp_polynomial`]: with a radical or a denominator the reduced
/// powers rarely help, and on the Rubi suite the attempts cost more than
/// the whole rest of the step).
pub(crate) fn linearize_trig_powers(
    arena: &mut Arena,
    factors: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    let mut groups: Vec<(ExprId, u32, u32)> = Vec::new();
    for &f in factors {
        let found = match extract_power(arena, f, sin_arg) {
            Some(p) => Some((p, true)),
            None => extract_power(arena, f, cos_arg).map(|p| (p, false)),
        };
        let Some((p, is_sin)) = found else {
            continue;
        };
        let e = u32::try_from(p.exp).ok()?;
        match groups.iter_mut().find(|g| g.0 == p.arg) {
            Some(g) if is_sin => g.1 += e,
            Some(g) => g.2 += e,
            None => groups.push(if is_sin { (p.arg, e, 0) } else { (p.arg, 0, e) }),
        }
    }
    let &(u, m, n) = groups.iter().find(|g| g.1 + g.2 >= 2)?;
    for &f in factors {
        if !trig_exp_polynomial(arena, f, var, var_sym) {
            return None;
        }
    }
    let is_group_member = |arena: &Arena, f: ExprId| {
        extract_power(arena, f, sin_arg)
            .or_else(|| extract_power(arena, f, cos_arg))
            .is_some_and(|p| p.arg == u)
    };
    let mut rest: Vec<ExprId> = factors
        .iter()
        .copied()
        .filter(|&f| !is_group_member(arena, f))
        .collect();
    let linear = linearize_sin_cos(arena, u, m, n)?;
    rest.push(linear);
    Some(arena.mul(&rest))
}

/// `F(u)/a`: an antiderivative in `u = a·x + b` as one in `x`.
fn over_rate(arena: &mut Arena, big_f: ExprId, rate: ExprId) -> ExprId {
    if rate == arena.one {
        big_f
    } else {
        arena.div(big_f, rate)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// sin^n integration
// ═══════════════════════════════════════════════════════════════════════════

/// Compute `∫ sin^n(x) dx` using the recursive reduction formula.
///
/// - n = 0 → `x`
/// - n = 1 → `−cos(x)`
/// - n = −1 → `−ln|csc(x)+cot(x)|`
/// - n ≤ −2 → upward reduction toward 0:
///   `(1/(n+1))·cos(x)·sin^(n+1)(x) + (n+2)/(n+1)·∫sin^(n+2)(x)dx`
/// - n ≥ 2 → `−(1/n)·cos(x)·sin^(n−1)(x) + (n−1)/n · ∫ sin^(n−2)(x) dx`
pub(crate) fn sin_pow_integrate(arena: &mut Arena, n: i64, var: ExprId) -> ExprId {
    if n == 0 {
        return var; // ∫ 1 dx = x
    }
    if n == 1 {
        // ∫ sin(x) dx = −cos(x)
        let cos_x = arena.cos(var);
        return arena.neg(cos_x);
    }
    if n == -1 {
        // ∫ csc(x) dx = −ln|csc(x) + cot(x)|
        let sin_x = arena.sin(var);
        let cos_x = arena.cos(var);
        let neg_one_id = arena.int(-1);
        let csc_x = arena.pow(sin_x, neg_one_id); // 1/sin(x)
        let cot_x = arena.mul(&[cos_x, csc_x]); // cos(x)/sin(x)
        let sum = arena.add(&[csc_x, cot_x]);
        let abs_sum = arena.abs(sum);
        let ln_val = arena.ln(abs_sum);
        return arena.neg(ln_val);
    }
    if n < -1 {
        // Upward reduction formula (recurse toward 0):
        //   ∫ sin^n(x) dx = (1/(n+1))·cos(x)·sin^(n+1)(x)
        //                  + (n+2)/(n+1) · ∫ sin^(n+2)(x) dx
        let cos_x = arena.cos(var);
        let sin_x = arena.sin(var);

        // sin^(n+1)(x)  — note n+1 ≤ −1 here, never 0 or 1
        let exp_id = arena.int(n + 1);
        let sin_pow = arena.pow(sin_x, exp_id);

        // First term: (1/(n+1)) · cos(x) · sin^(n+1)(x)
        let inv = arena.rational(1, n + 1);
        let first_term = arena.mul(&[inv, cos_x, sin_pow]);

        // Second term: (n+2)/(n+1) · ∫ sin^(n+2)(x) dx
        let coeff = arena.rational(n + 2, n + 1);
        let recursive = sin_pow_integrate(arena, n + 2, var);
        let second_term = arena.mul(&[coeff, recursive]);

        return arena.add(&[first_term, second_term]);
    }

    // Recursive reduction:
    //   ∫ sin^n(x) dx = −(1/n)·cos(x)·sin^(n−1)(x)
    //                  + (n−1)/n · ∫ sin^(n−2)(x) dx

    let cos_x = arena.cos(var);
    let sin_x = arena.sin(var);

    // sin^(n−1)(x)
    let sin_pow = if n - 1 == 1 {
        sin_x
    } else {
        let n_minus_1 = arena.int(n - 1);
        arena.pow(sin_x, n_minus_1)
    };

    // First term: −(1/n) · cos(x) · sin^(n−1)(x)
    let neg_inv_n = arena.rational(-1, n);
    let first_term = arena.mul(&[neg_inv_n, cos_x, sin_pow]);

    // Second term: (n−1)/n · ∫ sin^(n−2)(x) dx
    let coeff = arena.rational(n - 1, n);
    let recursive = sin_pow_integrate(arena, n - 2, var);
    let second_term = arena.mul(&[coeff, recursive]);

    arena.add(&[first_term, second_term])
}

// ═══════════════════════════════════════════════════════════════════════════
// cos^n integration
// ═══════════════════════════════════════════════════════════════════════════

/// Compute `∫ cos^n(x) dx` using the recursive reduction formula.
///
/// - n = 0 → `x`
/// - n = 1 → `sin(x)`
/// - n = −1 → `ln|sec(x)+tan(x)|`
/// - n ≤ −2 → upward reduction toward 0:
///   `−(1/(n+1))·sin(x)·cos^(n+1)(x) + (n+2)/(n+1)·∫cos^(n+2)(x)dx`
/// - n ≥ 2 → `(1/n)·sin(x)·cos^(n−1)(x) + (n−1)/n · ∫ cos^(n−2)(x) dx`
pub(crate) fn cos_pow_integrate(arena: &mut Arena, n: i64, var: ExprId) -> ExprId {
    if n == 0 {
        return var; // ∫ 1 dx = x
    }
    if n == 1 {
        // ∫ cos(x) dx = sin(x)
        return arena.sin(var);
    }
    if n == -1 {
        // ∫ sec(x) dx = ln|sec(x) + tan(x)|
        let cos_x = arena.cos(var);
        let neg_one_id = arena.int(-1);
        let sec_x = arena.pow(cos_x, neg_one_id); // 1/cos(x)
        let tan_x = arena.tan(var);
        let sum = arena.add(&[sec_x, tan_x]);
        let abs_sum = arena.abs(sum);
        return arena.ln(abs_sum);
    }
    if n < -1 {
        // Upward reduction formula (recurse toward 0):
        //   ∫ cos^n(x) dx = −(1/(n+1))·sin(x)·cos^(n+1)(x)
        //                  + (n+2)/(n+1) · ∫ cos^(n+2)(x) dx
        let sin_x = arena.sin(var);
        let cos_x = arena.cos(var);

        // cos^(n+1)(x)  — note n+1 ≤ −1 here, never 0 or 1
        let exp_id = arena.int(n + 1);
        let cos_pow = arena.pow(cos_x, exp_id);

        // First term: −(1/(n+1)) · sin(x) · cos^(n+1)(x)
        let neg_inv = arena.rational(-1, n + 1);
        let first_term = arena.mul(&[neg_inv, sin_x, cos_pow]);

        // Second term: (n+2)/(n+1) · ∫ cos^(n+2)(x) dx
        let coeff = arena.rational(n + 2, n + 1);
        let recursive = cos_pow_integrate(arena, n + 2, var);
        let second_term = arena.mul(&[coeff, recursive]);

        return arena.add(&[first_term, second_term]);
    }

    // Recursive reduction:
    //   ∫ cos^n(x) dx = (1/n)·sin(x)·cos^(n−1)(x)
    //                  + (n−1)/n · ∫ cos^(n−2)(x) dx

    let sin_x = arena.sin(var);
    let cos_x = arena.cos(var);

    // cos^(n−1)(x)
    let cos_pow = if n - 1 == 1 {
        cos_x
    } else {
        let n_minus_1 = arena.int(n - 1);
        arena.pow(cos_x, n_minus_1)
    };

    // First term: (1/n) · sin(x) · cos^(n−1)(x)
    let inv_n = arena.rational(1, n);
    let first_term = arena.mul(&[inv_n, sin_x, cos_pow]);

    // Second term: (n−1)/n · ∫ cos^(n−2)(x) dx
    let coeff = arena.rational(n - 1, n);
    let recursive = cos_pow_integrate(arena, n - 2, var);
    let second_term = arena.mul(&[coeff, recursive]);

    arena.add(&[first_term, second_term])
}

// ═══════════════════════════════════════════════════════════════════════════
// sin^m · cos^n integration
// ═══════════════════════════════════════════════════════════════════════════

/// Compute `∫ sin^m(x) · cos^n(x) dx`.
///
/// Strategy:
/// - m = 0 → `cos_pow_integrate(n)`
/// - n = 0 → `sin_pow_integrate(m)`
/// - m odd → Pythagorean substitution with `u = cos(x)`
/// - n odd → Pythagorean substitution with `u = sin(x)`
/// - both even → reduction formula on `m`
pub(crate) fn sin_cos_integrate(arena: &mut Arena, m: i64, n: i64, var: ExprId) -> ExprId {
    // Degenerate cases: one exponent is zero.
    if m == 0 {
        return cos_pow_integrate(arena, n, var);
    }
    if n == 0 {
        return sin_pow_integrate(arena, m, var);
    }

    // ── m is odd and positive: u = cos(x), sin²(x) = 1 − u² ───────────────
    //
    // ∫ sin^(2k+1)(x) · cos^n(x) dx          (any integer n)
    //   = −∫ (1−u²)^k · u^n du       where u = cos(x)
    //   = −Σ_{j=0}^{k} C(k,j)·(−1)^j · ∫ u^(n+2j) du
    if m > 0 && m % 2 != 0 {
        let cos_x = arena.cos(var);
        return odd_power_substitution(arena, (m - 1) / 2, n, cos_x, true);
    }

    // ── n is odd and positive: u = sin(x), cos²(x) = 1 − u² ───────────────
    //
    // ∫ sin^m(x) · cos^(2k+1)(x) dx          (any integer m)
    //   = ∫ u^m · (1−u²)^k du        where u = sin(x)
    //   = Σ_{j=0}^{k} C(k,j)·(−1)^j · ∫ u^(m+2j) du
    if n > 0 && n % 2 != 0 {
        let sin_x = arena.sin(var);
        return odd_power_substitution(arena, (n - 1) / 2, m, sin_x, false);
    }

    // ── One exponent even and positive, the other negative ──────────────
    //
    // cos^n = (1 − sin²)^(n/2):  ∫ sin^m·cos^n = Σ_j C(n/2, j)(−1)^j ∫ sin^(m+2j)
    // sin^m = (1 − cos²)^(m/2):  ∫ sin^m·cos^n = Σ_j C(m/2, j)(−1)^j ∫ cos^(n+2j)
    // (the single powers are integrated for every integer exponent).
    if n >= 2 && m < 0 {
        return even_power_expansion(arena, n / 2, m, var, sin_pow_integrate);
    }
    if m >= 2 && n < 0 {
        return even_power_expansion(arena, m / 2, n, var, cos_pow_integrate);
    }

    // ── Both negative: sin² + cos² = 1 raises one exponent at a time ────
    //
    // ∫ sin^m·cos^n = ∫ sin^(m+2)·cos^n + ∫ sin^m·cos^(n+2), until one
    // exponent is 0 or 1 (a case above).
    if m < 0 && n < 0 {
        return both_negative_powers(arena, m, n, var);
    }

    // ── Both m and n are even: use reduction formula on m ──────────
    //
    // ∫ sin^m(x)·cos^n(x) dx
    //   = −sin^(m−1)(x)·cos^(n+1)(x) / (m+n)
    //     + (m−1)/(m+n) · ∫ sin^(m−2)(x)·cos^n(x) dx
    //
    // Base cases are handled by the m==0 / n==0 checks at the top via
    // recursion reducing m by 2 each step.
    debug_assert!(m % 2 == 0 && n % 2 == 0 && m >= 2 && n >= 2);

    let sin_x = arena.sin(var);
    let cos_x = arena.cos(var);
    let mn = m + n;

    // sin^(m−1)(x)
    let sin_pow = if m - 1 == 1 {
        sin_x
    } else {
        let exp = arena.int(m - 1);
        arena.pow(sin_x, exp)
    };

    // cos^(n+1)(x)
    let cos_pow = {
        let exp = arena.int(n + 1);
        arena.pow(cos_x, exp)
    };

    // First term: −sin^(m−1)(x) · cos^(n+1)(x) / (m+n)
    let neg_inv_mn = arena.rational(-1, mn);
    let first_term = arena.mul(&[neg_inv_mn, sin_pow, cos_pow]);

    // Second term: (m−1)/(m+n) · ∫ sin^(m−2)(x)·cos^n(x) dx
    let coeff = arena.rational(m - 1, mn);
    let recursive = sin_cos_integrate(arena, m - 2, n, var);
    let second_term = arena.mul(&[coeff, recursive]);

    arena.add(&[first_term, second_term])
}

/// `s·Σ_{j=0}^{k} C(k, j)·(−1)^j · ∫ u^(e+2j) du` for `u = cos x`
/// (`negate`, `s = −1`) or `u = sin x` (`s = 1`): the substitution for an
/// odd positive power `2k + 1` of the other function.  A term with
/// `e + 2j = −1` integrates to `ln|u|` (before 0.30 it divided by zero,
/// and every negative exponent was left unevaluated).
fn odd_power_substitution(arena: &mut Arena, k: i64, e: i64, u: ExprId, negate: bool) -> ExprId {
    let mut terms: Vec<ExprId> = Vec::new();
    for j in 0..=k {
        let mut c = binom(k as u64, j as u64);
        if (j % 2 != 0) != negate {
            c = -c;
        }
        let power = e + 2 * j + 1;
        let term = if power == 0 {
            let abs_u = arena.abs(u);
            let ln = arena.ln(abs_u);
            let coeff = arena.intern_num(Ratio::from_integer(c));
            let coeff = arena.intern(ExprNode::Num(coeff));
            arena.mul(&[coeff, ln])
        } else {
            let ratio = Ratio::new(c, BigInt::from(power));
            let coeff = arena.intern_num(ratio);
            let coeff = arena.intern(ExprNode::Num(coeff));
            let pow_id = arena.int(power);
            let u_pow = arena.pow(u, pow_id);
            arena.mul(&[coeff, u_pow])
        };
        terms.push(term);
    }
    arena.add(&terms)
}

/// `Σ_{j=0}^{h} C(h, j)·(−1)^j · single(e + 2j)`: `∫ sin^e·cos^(2h)` with
/// `single = sin_pow_integrate`, or `∫ sin^(2h)·cos^e` with
/// `single = cos_pow_integrate`.
fn even_power_expansion(
    arena: &mut Arena,
    h: i64,
    e: i64,
    var: ExprId,
    single: fn(&mut Arena, i64, ExprId) -> ExprId,
) -> ExprId {
    let mut terms: Vec<ExprId> = Vec::new();
    for j in 0..=h {
        let mut c = binom(h as u64, j as u64);
        if j % 2 != 0 {
            c = -c;
        }
        let integral = single(arena, e + 2 * j, var);
        let coeff = arena.intern_num(Ratio::from_integer(c));
        let coeff = arena.intern(ExprNode::Num(coeff));
        terms.push(arena.mul(&[coeff, integral]));
    }
    arena.add(&terms)
}

/// `∫ sin^m·cos^n` for `m, n < 0`: `1 = sin² + cos²` gives
/// `I(m, n) = I(m + 2, n) + I(m, n + 2)`; the pairs are expanded with
/// multiplicities (a worklist, no recursion) until one exponent is `0` or
/// `1`, where [`sin_cos_integrate`] has a closed form.  `∫ 1/(sin x·cos x)`
/// is `ln|sin x| − ln|cos x|`.
fn both_negative_powers(arena: &mut Arena, m: i64, n: i64, var: ExprId) -> ExprId {
    use std::collections::BTreeMap;
    let mut open: BTreeMap<(i64, i64), BigInt> = BTreeMap::new();
    let mut done: BTreeMap<(i64, i64), BigInt> = BTreeMap::new();
    open.insert((m, n), BigInt::one());
    while let Some(((a, b), c)) = open.pop_first() {
        if a >= 0 || b >= 0 {
            *done.entry((a, b)).or_insert_with(BigInt::zero) += c;
            continue;
        }
        *open.entry((a + 2, b)).or_insert_with(BigInt::zero) += &c;
        *open.entry((a, b + 2)).or_insert_with(BigInt::zero) += c;
    }
    let mut terms: Vec<ExprId> = Vec::new();
    for ((a, b), c) in done {
        let integral = sin_cos_integrate(arena, a, b, var);
        let coeff = arena.intern_num(Ratio::from_integer(c));
        let coeff = arena.intern(ExprNode::Num(coeff));
        terms.push(arena.mul(&[coeff, integral]));
    }
    arena.add(&terms)
}

// ═══════════════════════════════════════════════════════════════════════════
// Hyperbolic half-angle integration
// ═══════════════════════════════════════════════════════════════════════════

/// Compute `∫ sinh²(x) dx` using the half-angle identity:
///   `sinh²(x) = (cosh(2x) − 1) / 2`
///   `∫ sinh²(x) dx = sinh(2x)/4 − x/2`
pub(crate) fn sinh_squared_integrate(arena: &mut Arena, var: ExprId) -> ExprId {
    let two = arena.int(2);
    let two_x = arena.mul(&[two, var]);
    let sinh_2x = arena.sinh(two_x);
    let quarter = arena.rational(1, 4);
    let first_term = arena.mul(&[quarter, sinh_2x]);
    let half = arena.rational(1, 2);
    let second_term = arena.mul(&[half, var]);
    arena.sub(first_term, second_term)
}

/// Compute `∫ cosh²(x) dx` using the half-angle identity:
///   `cosh²(x) = (cosh(2x) + 1) / 2`
///   `∫ cosh²(x) dx = sinh(2x)/4 + x/2`
pub(crate) fn cosh_squared_integrate(arena: &mut Arena, var: ExprId) -> ExprId {
    let two = arena.int(2);
    let two_x = arena.mul(&[two, var]);
    let sinh_2x = arena.sinh(two_x);
    let quarter = arena.rational(1, 4);
    let first_term = arena.mul(&[quarter, sinh_2x]);
    let half = arena.rational(1, 2);
    let second_term = arena.mul(&[half, var]);
    arena.add(&[first_term, second_term])
}

// ═══════════════════════════════════════════════════════════════════════════
// Pattern-matching entry point
// ═══════════════════════════════════════════════════════════════════════════

/// Try to recognise a trigonometric power integrand and compute the
/// antiderivative.
///
/// Recognises, for one argument `u = a·x + b` linear in the variable
/// (integrated in `u`, then divided by `a`):
/// - `sin(u)^n`  (or bare `sin(u)` → n=1)
/// - `cos(u)^n`  (or bare `cos(u)` → n=1)
/// - products containing `sin(u)^m`, `cos(u)^n` and `tan(u)^k` (for every
///   integer exponent, `|m|, |n| ≤` [`MAX_TRIG_POWER`] after `tan^k` is
///   written `sin^k·cos^(−k)`), and `tan(u)^k` for `k < 0`
/// - `sinh(u)²`, `cosh(u)²`
///
/// Up to 0.31 only `u = x` was recognised: `∫ cos(2x)² dx`,
/// `∫ sin(x/3)³ dx` and `∫ (cos²(3x/2) − sin²(3x/2)) dx` stayed
/// unevaluated (normal-form hunt).
///
/// Returns `None` when the expression does not match any trig-power pattern.
pub(crate) fn try_trig_power_integral(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    let node = arena.node(expr).clone();

    // ── Single factor: Pow(Sin(u), n) or Pow(Cos(u), n) ────────
    if let Some(p) = extract_power(arena, expr, sin_arg) {
        // n == 1 is handled by the main integrator already.
        if (0..2).contains(&p.exp) || !within_power_bound(p.exp, 0) {
            return None;
        }
        let rate = linear_rate(arena, p.arg, var, var_sym)?;
        let big_f = sin_pow_integrate(arena, p.exp, p.arg);
        return Some(over_rate(arena, big_f, rate));
    }

    if let Some(p) = extract_power(arena, expr, cos_arg) {
        if (0..2).contains(&p.exp) || !within_power_bound(p.exp, 0) {
            return None;
        }
        let rate = linear_rate(arena, p.arg, var, var_sym)?;
        let big_f = cos_pow_integrate(arena, p.exp, p.arg);
        return Some(over_rate(arena, big_f, rate));
    }

    // ── Single factor: sinh(u)², cosh(u)² via half-angle ──
    // (other hyperbolic powers are not handled here).
    if let Some(p) = extract_power(arena, expr, sinh_arg)
        && p.exp == 2
    {
        let rate = linear_rate(arena, p.arg, var, var_sym)?;
        let big_f = sinh_squared_integrate(arena, p.arg);
        return Some(over_rate(arena, big_f, rate));
    }
    if let Some(p) = extract_power(arena, expr, cosh_arg)
        && p.exp == 2
    {
        let rate = linear_rate(arena, p.arg, var, var_sym)?;
        let big_f = cosh_squared_integrate(arena, p.arg);
        return Some(over_rate(arena, big_f, rate));
    }

    // ── Single factor: tan(u)^k for k < 0, i.e. cot^|k| = sin^k·cos^(−k) ──
    // (`∫ 1/tan v dv = ln|sin v|`; positive powers of tan have their own
    // rules in the integrator).
    if let Some(p) = extract_power(arena, expr, tan_arg)
        && p.exp < 0
        && within_power_bound(p.exp, -p.exp)
    {
        let rate = linear_rate(arena, p.arg, var, var_sym)?;
        let big_f = sin_cos_integrate(arena, p.exp, -p.exp, p.arg);
        return Some(over_rate(arena, big_f, rate));
    }

    // ── Product: Mul(...) containing sin/cos/tan powers of one argument ──
    if let ExprNode::Mul(ref children) = node {
        let mut sin_exp: i64 = 0;
        let mut cos_exp: i64 = 0;
        let mut other_factors: Vec<ExprId> = Vec::new();
        let mut found_tan = false;
        let mut arg: Option<ExprId> = None;
        let mut same_arg = |p: TrigPower| -> bool { *arg.get_or_insert(p.arg) == p.arg };

        for &child in children.iter() {
            if let Some(p) = extract_power(arena, child, sin_arg) {
                if !same_arg(p) {
                    return None;
                }
                sin_exp = sin_exp.checked_add(p.exp)?;
            } else if let Some(p) = extract_power(arena, child, cos_arg) {
                if !same_arg(p) {
                    return None;
                }
                cos_exp = cos_exp.checked_add(p.exp)?;
            } else if let Some(p) = extract_power(arena, child, tan_arg) {
                if !same_arg(p) {
                    return None;
                }
                // tan^t = sin^t·cos^(−t): `∫ sin x·tan x dx` (before 0.30
                // unevaluated) is `∫ sin²x/cos x`.
                sin_exp = sin_exp.checked_add(p.exp)?;
                cos_exp = cos_exp.checked_sub(p.exp)?;
                found_tan = true;
            } else if contains_var(arena, child, var_sym) {
                // A var-dependent factor that isn't a sin/cos power — bail.
                return None;
            } else {
                // Constant factor (independent of var): keep aside.
                other_factors.push(child);
            }
        }

        let u = arg?;

        // We need at least one exponent ≥ 2 to be interesting, or a
        // mixed sin·cos product.  The main integrator already handles
        // single sin(x) and cos(x), so only fire when the combined
        // problem is genuinely a "power" integral.
        let dominated_by_basic = sin_exp == 0 && cos_exp == 1 || sin_exp == 1 && cos_exp == 0;
        if dominated_by_basic && !found_tan {
            return None;
        }
        if !within_power_bound(sin_exp, cos_exp) {
            return None;
        }
        let rate = linear_rate(arena, u, var, var_sym)?;

        // Compute the trig-power integral.
        let trig_result = sin_cos_integrate(arena, sin_exp, cos_exp, u);
        let trig_result = over_rate(arena, trig_result, rate);

        // Re-attach any constant prefactors.
        if other_factors.is_empty() {
            return Some(trig_result);
        }
        other_factors.push(trig_result);
        return Some(arena.mul(&other_factors));
    }

    None
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::arena::Arena;

    // ------------------------------------------------------------------
    // sin^n integration
    // ------------------------------------------------------------------

    #[test]
    fn sin_zero() {
        // ∫ sin^0(x) dx = ∫ 1 dx = x
        let mut a = Arena::new();
        let x = a.symbol("x");
        let result = sin_pow_integrate(&mut a, 0, x);
        assert_eq!(result, x);
    }

    #[test]
    fn sin_first() {
        // ∫ sin(x) dx = −cos(x)
        let mut a = Arena::new();
        let x = a.symbol("x");
        let result = sin_pow_integrate(&mut a, 1, x);
        let s = a.display(result).to_string();
        assert!(s.contains("cos"), "expected −cos(x), got: {s}");
    }

    #[test]
    fn sin_squared() {
        // ∫ sin²(x) dx = −½·sin(x)·cos(x) + ½·x
        let mut a = Arena::new();
        let x = a.symbol("x");
        let result = sin_pow_integrate(&mut a, 2, x);
        let s = a.display(result).to_string();
        assert!(
            s.contains("cos") && s.contains("sin"),
            "expected reduction formula terms, got: {s}"
        );
    }

    #[test]
    fn sin_cubed() {
        // ∫ sin³(x) dx = −⅓·cos(x)·sin²(x) + ⅔·(−cos(x))
        let mut a = Arena::new();
        let x = a.symbol("x");
        let result = sin_pow_integrate(&mut a, 3, x);
        let s = a.display(result).to_string();
        assert!(s.contains("cos"), "expected cos terms, got: {s}");
    }

    #[test]
    fn sin_fourth() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let result = sin_pow_integrate(&mut a, 4, x);
        let s = a.display(result).to_string();
        // Should produce some combination of x, sin, cos
        assert!(!s.is_empty(), "got: {s}");
    }

    #[test]
    fn sin_neg2_is_neg_cot() {
        // ∫ sin^(-2)(x) dx = −cot(x) = −cos(x)/sin(x)
        let mut a = Arena::new();
        let x = a.symbol("x");
        let result = sin_pow_integrate(&mut a, -2, x);
        let s = a.display(result).to_string();
        // Should NOT be unevaluated — should contain cos and sin
        assert!(
            !s.contains("Integral"),
            "expected evaluated result, got unevaluated: {s}"
        );
        assert!(
            s.contains("cos") && s.contains("sin"),
            "expected cos/sin terms for −cot(x), got: {s}"
        );
    }

    // ------------------------------------------------------------------
    // cos^n integration
    // ------------------------------------------------------------------

    #[test]
    fn cos_zero() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let result = cos_pow_integrate(&mut a, 0, x);
        assert_eq!(result, x);
    }

    #[test]
    fn cos_first() {
        // ∫ cos(x) dx = sin(x)
        let mut a = Arena::new();
        let x = a.symbol("x");
        let result = cos_pow_integrate(&mut a, 1, x);
        let s = a.display(result).to_string();
        assert!(s.contains("sin"), "expected sin(x), got: {s}");
    }

    #[test]
    fn cos_squared() {
        // ∫ cos²(x) dx = ½·sin(x)·cos(x) + ½·x
        let mut a = Arena::new();
        let x = a.symbol("x");
        let result = cos_pow_integrate(&mut a, 2, x);
        let s = a.display(result).to_string();
        assert!(
            s.contains("cos") && s.contains("sin"),
            "expected reduction formula terms, got: {s}"
        );
    }

    #[test]
    fn cos_cubed() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let result = cos_pow_integrate(&mut a, 3, x);
        let s = a.display(result).to_string();
        assert!(s.contains("sin"), "expected sin terms, got: {s}");
    }

    #[test]
    fn cos_neg1_is_ln_sec_tan() {
        // ∫ cos^(-1)(x) dx = ∫ sec(x) dx = ln|sec(x)+tan(x)|
        let mut a = Arena::new();
        let x = a.symbol("x");
        let result = cos_pow_integrate(&mut a, -1, x);
        let s = a.display(result).to_string();
        assert!(
            !s.contains("Integral"),
            "expected evaluated result, got unevaluated: {s}"
        );
        assert!(
            s.contains("ln"),
            "expected ln term for ln|sec+tan|, got: {s}"
        );
    }

    #[test]
    fn cos_neg2_is_tan() {
        // ∫ cos^(-2)(x) dx = ∫ sec²(x) dx = tan(x)
        let mut a = Arena::new();
        let x = a.symbol("x");
        let result = cos_pow_integrate(&mut a, -2, x);
        let s = a.display(result).to_string();
        assert!(
            !s.contains("Integral"),
            "expected evaluated result, got unevaluated: {s}"
        );
        assert!(
            s.contains("sin") && s.contains("cos"),
            "expected sin/cos terms for tan(x), got: {s}"
        );
    }

    // ------------------------------------------------------------------
    // sin^m · cos^n integration
    // ------------------------------------------------------------------

    #[test]
    fn sin1_cos1() {
        // ∫ sin(x)·cos(x) dx  (m=1 odd)
        // = −cos²(x)/2
        let mut a = Arena::new();
        let x = a.symbol("x");
        let result = sin_cos_integrate(&mut a, 1, 1, x);
        let s = a.display(result).to_string();
        assert!(s.contains("cos"), "expected cos term, got: {s}");
    }

    #[test]
    fn sin2_cos1() {
        // ∫ sin²(x)·cos(x) dx  (n=1 odd)
        // = sin³(x)/3
        let mut a = Arena::new();
        let x = a.symbol("x");
        let result = sin_cos_integrate(&mut a, 2, 1, x);
        let s = a.display(result).to_string();
        assert!(s.contains("sin"), "expected sin term, got: {s}");
    }

    #[test]
    fn sin1_cos2() {
        // ∫ sin(x)·cos²(x) dx  (m=1 odd)
        // = −cos³(x)/3
        let mut a = Arena::new();
        let x = a.symbol("x");
        let result = sin_cos_integrate(&mut a, 1, 2, x);
        let s = a.display(result).to_string();
        assert!(s.contains("cos"), "expected cos term, got: {s}");
    }

    #[test]
    fn sin3_cos2() {
        // ∫ sin³(x)·cos²(x) dx  (m=3 odd)
        // = −cos³(x)/3 + cos⁵(x)/5
        let mut a = Arena::new();
        let x = a.symbol("x");
        let result = sin_cos_integrate(&mut a, 3, 2, x);
        let s = a.display(result).to_string();
        assert!(s.contains("cos"), "expected cos terms, got: {s}");
    }

    #[test]
    fn sin2_cos3() {
        // ∫ sin²(x)·cos³(x) dx  (n=3 odd)
        // = sin³(x)/3 − sin⁵(x)/5
        let mut a = Arena::new();
        let x = a.symbol("x");
        let result = sin_cos_integrate(&mut a, 2, 3, x);
        let s = a.display(result).to_string();
        assert!(s.contains("sin"), "expected sin terms, got: {s}");
    }

    #[test]
    fn sin2_cos2_both_even() {
        // ∫ sin²(x)·cos²(x) dx — both even, uses reduction
        let mut a = Arena::new();
        let x = a.symbol("x");
        let result = sin_cos_integrate(&mut a, 2, 2, x);
        let s = a.display(result).to_string();
        // Should contain both sin and cos terms
        assert!(!s.is_empty(), "got: {s}");
    }

    #[test]
    fn sin_cos_m_zero_delegates() {
        // m=0 should delegate to cos_pow_integrate
        let mut a = Arena::new();
        let x = a.symbol("x");
        let result = sin_cos_integrate(&mut a, 0, 3, x);
        let direct = cos_pow_integrate(&mut a, 3, x);
        assert_eq!(result, direct);
    }

    #[test]
    fn sin_cos_n_zero_delegates() {
        // n=0 should delegate to sin_pow_integrate
        let mut a = Arena::new();
        let x = a.symbol("x");
        let result = sin_cos_integrate(&mut a, 3, 0, x);
        let direct = sin_pow_integrate(&mut a, 3, x);
        assert_eq!(result, direct);
    }

    // ------------------------------------------------------------------
    // Pattern detection via try_trig_power_integral
    // ------------------------------------------------------------------

    #[test]
    fn try_detect_sin_squared() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let var_sym = match a.node(x) {
            ExprNode::Symbol(sid) => *sid,
            _ => unreachable!(),
        };
        let sin_x = a.sin(x);
        let two = a.int(2);
        let expr = a.pow(sin_x, two);
        let result = try_trig_power_integral(&mut a, expr, x, var_sym);
        assert!(result.is_some(), "should detect sin²(x)");
        let s = a.display(result.unwrap()).to_string();
        assert!(
            s.contains("cos") && s.contains("sin"),
            "expected reduction formula, got: {s}"
        );
    }

    #[test]
    fn try_detect_cos_cubed() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let var_sym = match a.node(x) {
            ExprNode::Symbol(sid) => *sid,
            _ => unreachable!(),
        };
        let cos_x = a.cos(x);
        let three = a.int(3);
        let expr = a.pow(cos_x, three);
        let result = try_trig_power_integral(&mut a, expr, x, var_sym);
        assert!(result.is_some(), "should detect cos³(x)");
        let s = a.display(result.unwrap()).to_string();
        assert!(s.contains("sin"), "expected sin terms, got: {s}");
    }

    #[test]
    fn try_detect_sin_cos_product() {
        // sin(x)^2 * cos(x)^3 as Mul
        let mut a = Arena::new();
        let x = a.symbol("x");
        let var_sym = match a.node(x) {
            ExprNode::Symbol(sid) => *sid,
            _ => unreachable!(),
        };
        let sin_x = a.sin(x);
        let cos_x = a.cos(x);
        let two = a.int(2);
        let three = a.int(3);
        let sin2 = a.pow(sin_x, two);
        let cos3 = a.pow(cos_x, three);
        let expr = a.mul(&[sin2, cos3]);
        let result = try_trig_power_integral(&mut a, expr, x, var_sym);
        assert!(result.is_some(), "should detect sin²·cos³");
    }

    #[test]
    fn try_detect_bare_sin_cos_product() {
        // sin(x) * cos(x) — implicit power 1 for each
        let mut a = Arena::new();
        let x = a.symbol("x");
        let var_sym = match a.node(x) {
            ExprNode::Symbol(sid) => *sid,
            _ => unreachable!(),
        };
        let sin_x = a.sin(x);
        let cos_x = a.cos(x);
        let expr = a.mul(&[sin_x, cos_x]);
        let result = try_trig_power_integral(&mut a, expr, x, var_sym);
        assert!(result.is_some(), "should detect sin(x)·cos(x)");
    }

    #[test]
    fn try_detect_with_constant_factor() {
        // 3 * sin(x)^2 * cos(x)^3
        let mut a = Arena::new();
        let x = a.symbol("x");
        let var_sym = match a.node(x) {
            ExprNode::Symbol(sid) => *sid,
            _ => unreachable!(),
        };
        let sin_x = a.sin(x);
        let cos_x = a.cos(x);
        let two = a.int(2);
        let three_exp = a.int(3);
        let sin2 = a.pow(sin_x, two);
        let cos3 = a.pow(cos_x, three_exp);
        let three = a.int(3);
        let expr = a.mul(&[three, sin2, cos3]);
        let result = try_trig_power_integral(&mut a, expr, x, var_sym);
        assert!(
            result.is_some(),
            "should detect 3·sin²·cos³ with constant factor"
        );
    }

    #[test]
    fn try_returns_none_for_non_trig() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let var_sym = match a.node(x) {
            ExprNode::Symbol(sid) => *sid,
            _ => unreachable!(),
        };
        // x^2 is not a trig power
        let two = a.int(2);
        let expr = a.pow(x, two);
        let result = try_trig_power_integral(&mut a, expr, x, var_sym);
        assert!(result.is_none(), "x² is not a trig power integral");
    }

    #[test]
    fn try_returns_none_for_mixed_non_trig() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let var_sym = match a.node(x) {
            ExprNode::Symbol(sid) => *sid,
            _ => unreachable!(),
        };
        // x * sin(x)^2 — has a var-dependent non-trig factor
        let sin_x = a.sin(x);
        let two = a.int(2);
        let sin2 = a.pow(sin_x, two);
        let expr = a.mul(&[x, sin2]);
        let result = try_trig_power_integral(&mut a, expr, x, var_sym);
        assert!(
            result.is_none(),
            "x·sin²(x) has a non-trig dependent factor"
        );
    }

    // ------------------------------------------------------------------
    // Helper tests
    // ------------------------------------------------------------------

    #[test]
    fn binom_values() {
        assert_eq!(binom(0, 0), BigInt::from(1));
        assert_eq!(binom(1, 0), BigInt::from(1));
        assert_eq!(binom(1, 1), BigInt::from(1));
        assert_eq!(binom(4, 2), BigInt::from(6));
        assert_eq!(binom(5, 0), BigInt::from(1));
        assert_eq!(binom(5, 5), BigInt::from(1));
        assert_eq!(binom(5, 3), BigInt::from(10));
        assert_eq!(binom(6, 3), BigInt::from(20));
        assert_eq!(binom(3, 5), BigInt::from(0)); // k > n
    }

    #[test]
    fn extract_sin_power_bare() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let sin_x = a.sin(x);
        assert_eq!(extract_sin_power(&a, sin_x, x), Some(1));
    }

    #[test]
    fn extract_sin_power_squared() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let sin_x = a.sin(x);
        let two = a.int(2);
        let sin2 = a.pow(sin_x, two);
        assert_eq!(extract_sin_power(&a, sin2, x), Some(2));
    }

    #[test]
    fn extract_cos_power_bare() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let cos_x = a.cos(x);
        assert_eq!(extract_cos_power(&a, cos_x, x), Some(1));
    }

    #[test]
    fn extract_cos_power_cubed() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let cos_x = a.cos(x);
        let three = a.int(3);
        let cos3 = a.pow(cos_x, three);
        assert_eq!(extract_cos_power(&a, cos3, x), Some(3));
    }

    #[test]
    fn extract_returns_none_for_non_trig() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let two = a.int(2);
        let x2 = a.pow(x, two);
        assert_eq!(extract_sin_power(&a, x2, x), None);
        assert_eq!(extract_cos_power(&a, x2, x), None);
    }
}
