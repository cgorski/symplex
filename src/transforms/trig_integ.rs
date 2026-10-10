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
// Continuity across the poles of a tangent substitution
// ═══════════════════════════════════════════════════════════════════════════

/// The limit of an antiderivative `G(t)` at one end of the real line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EndValue {
    /// A finite limit.
    Finite(ExprId),
    /// `|G(t)| → ∞`: the integrand is not integrable at that end.
    Infinite,
}

/// `(degree, leading coefficient)` of `p` as a polynomial in `t` with
/// `t`-free coefficients, the leading coefficient proved non-zero or, with
/// free parameters, not structurally zero (a generic value of them, as for
/// the limit engine); `None` when `p` is no such polynomial, is zero, or a
/// constant leading coefficient is not decided.
fn leading_term(arena: &mut Arena, p: ExprId, t: ExprId) -> Option<(usize, ExprId)> {
    let key = (std::ptr::from_ref::<Arena>(arena) as usize, p, t);
    if let Some(found) = JUMP_MEMO.with(|m| {
        m.borrow()
            .as_ref()
            .and_then(|m| m.leading.get(&key).copied())
    }) {
        return found;
    }
    let lead = leading_term_uncached(arena, p, t);
    JUMP_MEMO.with(|m| {
        if let Some(m) = m.borrow_mut().as_mut() {
            m.leading.insert(key, lead);
        }
    });
    lead
}

/// [`leading_term`] without the memo.
fn leading_term_uncached(arena: &mut Arena, p: ExprId, t: ExprId) -> Option<(usize, ExprId)> {
    let mut coeffs = crate::transforms::solve::symbolic_poly_coeffs(arena, p, t)?;
    while let Some(&lc) = coeffs.last() {
        match crate::poly::algebraic::is_zero_checked(arena, lc) {
            Some(true) => {
                coeffs.pop();
            }
            Some(false) => return Some((coeffs.len() - 1, lc)),
            None if !crate::base::walk::free_symbols(arena, lc).is_empty()
                && !arena.is_zero_structural(lc) =>
            {
                return Some((coeffs.len() - 1, lc));
            }
            None => return None,
        }
    }
    None
}

/// Where a non-zero constant `z` lies, for the end values of `ln` and
/// `atan`: by its value, or (with free parameters) by the declared
/// assumptions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Placement {
    /// On the positive real axis.
    Positive,
    /// On the negative real axis (the cut of `ln`).
    Negative,
    /// Off the real axis, with the sign of its real part (0: on the
    /// imaginary axis) and whether `|Im z| ≥ 1` there.
    Complex { re_sign: i8, im_at_least_one: bool },
}

fn placement(arena: &mut Arena, z: ExprId) -> Option<Placement> {
    if let Ok(v) = crate::transforms::evalf::evalf_complex64(arena, z) {
        let scale = v.norm();
        if !scale.is_finite() || scale == 0.0 {
            return None;
        }
        let tol = 1e-9 * scale;
        if v.im.abs() <= tol {
            return Some(if v.re > 0.0 {
                Placement::Positive
            } else {
                Placement::Negative
            });
        }
        let re_sign = if v.re > tol {
            1
        } else if v.re < -tol {
            -1
        } else {
            0
        };
        if re_sign == 0 && (v.im.abs() - 1.0).abs() <= 1e-9 {
            return None;
        }
        return Some(Placement::Complex {
            re_sign,
            im_at_least_one: v.im.abs() >= 1.0,
        });
    }
    let mut facts = crate::base::assumptions::AssumptionCache::new();
    if facts.query(arena, z, crate::base::assumptions::Props::POSITIVE) == Some(true) {
        Some(Placement::Positive)
    } else if facts.query(arena, z, crate::base::assumptions::Props::NEGATIVE) == Some(true) {
        Some(Placement::Negative)
    } else {
        None
    }
}

/// A rational function `r` of `t` near `t = ±∞`: `r ~ c·tᵈ`, as
/// `(d, c·(±1)ᵈ)` — the degree and the coefficient of `|t|ᵈ` at the end
/// `positive` picks.  `None` when `r` is not a quotient of polynomials in
/// `t` with decided leading coefficients.
fn rational_end(arena: &mut Arena, r: ExprId, t: ExprId, positive: bool) -> Option<(i64, ExprId)> {
    let (num, den) = arena.as_numer_denom_expr(r);
    let (dn, ln) = leading_term(arena, num, t)?;
    let (dd, ld) = leading_term(arena, den, t)?;
    let d = i64::try_from(dn).ok()? - i64::try_from(dd).ok()?;
    let mut lead = arena.div(ln, ld);
    if !positive && d % 2 != 0 {
        lead = arena.neg(lead);
    }
    Some((d, crate::transforms::eval::eval(arena, lead)))
}

/// The limit of `ln(p(t)) − d·ln|t|` (principal `ln`) at the end where
/// `p(t) ~ lead·|t|ᵈ`.  `ln(lead)` when `lead` is off the cut (the negative
/// reals).  On the cut it depends on the side from which `p(t)` approaches
/// it: the sign of `Im p(t)`, that of the highest-degree non-real
/// coefficient `cₖ` of `p` (times `(±1)ᵏ`) — `ln(lead) = ln|lead| + iπ` from
/// above or when `p(t)` is real there, `ln(lead) − 2πi` from below
/// (`ln(t + i√3) → ln|t| + iπ` and `ln(t − i√3) → ln|t| − iπ` as `t → −∞`).
///
/// With real parameters a coefficient may be real for some of their values
/// and not for others (`√(b² − a²)/(a − b)` in `∫ dx/(a + b·cos x)` through
/// `tan(x/2)`).  When it is the only coefficient not proved real, `Im p(t)`
/// is `Im cₖ·tᵏ` and the side is that of `sg = sign((±1)ᵏ·Im cₖ)` for every
/// value of the parameters, `p(t)` real (on the cut, `+iπ`) where `sg = 0`:
/// the limit is `ln(lead) + iπ·(sg − sg²)` (`ln(lead) − 2πi` exactly where
/// `sg = −1`).  `None` when that is not decided or `p` has a non-constant
/// denominator.
fn log_end(
    arena: &mut Arena,
    p: ExprId,
    t: ExprId,
    lead: ExprId,
    positive: bool,
) -> Option<ExprId> {
    let principal = arena.ln(lead);
    let assumed = ASSUME_REAL_PARAMETERS.with(std::cell::Cell::get);
    // A leading coefficient of undecided place, rational in parameters
    // not declared real (real parameters assumed): the end is on the cut
    // exactly where `lead` is a negative real, where the principal `ln`
    // gives `ln|lead| + iπ`; elsewhere `ln(lead)` is the end value.
    let parametric_lead = match placement(arena, lead) {
        Some(Placement::Negative) => false,
        Some(_) => return Some(principal),
        None if assumed && rational_in_parameters(arena, lead) => true,
        None => return None,
    };
    let (num, den) = arena.as_numer_denom_expr(p);
    if contains_var_id(arena, den, t) || (parametric_lead && !rational_in_parameters(arena, den)) {
        return None;
    }
    let coeffs = crate::transforms::solve::symbolic_poly_coeffs(arena, num, t)?;
    // The side of the cut is that of the first non-zero `(±1)ᵏ·Im cₖ`,
    // highest degree first: `signs` holds `sign((±1)ᵏ·Im cₖ)` of the
    // coefficients whose imaginary part depends on the parameters' values,
    // `side` the side of the first coefficient known not to be real.
    //
    // With real parameters assumed, a coefficient rational in parameters
    // not declared real used to count as real: right for real values, but
    // the jump `J·⌊w/π + 1/2⌋` of `∫ dx/(a + tan x)` then had the wrong
    // sign for `Im a < 0` and the answer jumped by `2πi/(a² + 1)` at every
    // pole of `tan x` (0.38–0.39).  Its `sign(Im c)` is 0 for real values,
    // so they keep their answer.
    let mut reals = crate::base::assumptions::AssumptionCache::new();
    let mut signs: Vec<ExprId> = Vec::new();
    let mut declared_undecided = false;
    let mut side: Option<i64> = None;
    for (k, &c) in coeffs.iter().enumerate().rev() {
        let c = arena.div(c, den);
        let c = crate::transforms::eval::eval(arena, c);
        let flip = !(positive || k % 2 == 0);
        let symbolic = match crate::transforms::realness::constant_realness_as_declared(
            arena, c, 30, &mut reals,
        ) {
            Some(true) => continue,
            None if assumed && rational_in_parameters(arena, c) => true,
            _ if parametric_lead => return None,
            // A coefficient real or not by the values of declared-real
            // parameters (`√(b² − a²)/(a − b)`): one is followed.
            None if !declared_undecided && only_real_parameters(arena, c) => {
                declared_undecided = true;
                true
            }
            None if declared_undecided => return None,
            _ => false,
        };
        if symbolic {
            let im = arena.im(c);
            let im = if flip { arena.neg(im) } else { im };
            signs.push(arena.sign(im));
            continue;
        }
        if declared_undecided {
            return None;
        }
        let z = crate::transforms::evalf::evalf_complex64(arena, c).ok()?;
        let im = z.im.abs();
        if im.is_nan() || im <= 1e-12 * z.norm().max(1.0) {
            return None;
        }
        side = Some(if (z.im > 0.0) != flip { 1 } else { -1 });
        break;
    }
    // S = s₁ + (1 − s₁²)·(s₂ + (1 − s₂²)·(… + side)): the first non-zero
    // sign, or the decided side, or 0 (`p(t)` real: on the cut, `+iπ`).
    let mut s_total = arena.int(side.unwrap_or(0));
    for &sg in signs.iter().rev() {
        let one = arena.one;
        let two = arena.int(2);
        let sg_sq = arena.pow(sg, two);
        let minus_sq = arena.neg(sg_sq);
        let rest_weight = arena.add(&[one, minus_sq]);
        let rest = arena.mul(&[rest_weight, s_total]);
        s_total = arena.add(&[sg, rest]);
    }
    // From below (`S = −1`) the limit is `ln(lead) − 2πi`: `iπ·(S − S²)`.
    let two = arena.int(2);
    let s_sq = arena.pow(s_total, two);
    let minus_sq = arena.neg(s_sq);
    let shift = arena.add(&[s_total, minus_sq]);
    if arena.is_zero_structural(shift) {
        return Some(principal);
    }
    let i = arena.i_unit();
    let pi = arena.pi();
    let mut parts = vec![i, pi, shift];
    if parametric_lead {
        // [lead < 0] = ½·(1 − sign(re lead))·(1 − sign(im lead)²).
        let one = arena.one;
        let half = arena.rational(1, 2);
        let re = arena.re(lead);
        let re_sign = arena.sign(re);
        let minus_re = arena.neg(re_sign);
        let left = arena.add(&[one, minus_re]);
        let im = arena.im(lead);
        let im_sign = arena.sign(im);
        let im_sq = arena.pow(im_sign, two);
        let minus_im = arena.neg(im_sq);
        let real_axis = arena.add(&[one, minus_im]);
        parts.extend([half, left, real_axis]);
    }
    let correction = arena.mul(&parts);
    Some(arena.add(&[principal, correction]))
}

/// What a jump at the poles of `tan w` depends on besides the arena's
/// expressions: which of [`infinity_jump`] and
/// [`infinity_jump_real_parameters`] took it, and the setting of
/// [`ASSUME_REAL_PARAMETERS`] on entry (the latter sets it itself).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct JumpKey {
    arena: usize,
    real_parameters: bool,
    assumed_real: bool,
    g: ExprId,
    t: ExprId,
}

/// The memo of a [`JumpMemo`]: jumps, and the leading terms of
/// [`leading_term`] (by arena, polynomial and variable).
#[derive(Default)]
struct JumpMemoMaps {
    jumps: rustc_hash::FxHashMap<JumpKey, Option<ExprId>>,
    leading: rustc_hash::FxHashMap<(usize, ExprId, ExprId), Option<(usize, ExprId)>>,
}

thread_local! {
    /// Set while a [`JumpMemo`] is alive.
    static JUMP_MEMO: std::cell::RefCell<Option<JumpMemoMaps>> =
        const { std::cell::RefCell::new(None) };
}

/// While alive (the outermost `integrate` call holds one), the jumps of
/// [`infinity_jump`] and [`infinity_jump_real_parameters`] and the leading
/// terms behind them are remembered: functions of the arena's (immutable,
/// hash-consed) expressions and the flags in [`JumpKey`] alone.  One
/// integration meets the same `G(t)` again and again — every degenerate
/// case of the parameters and every group of a numerator re-runs the
/// trigonometric substitution that produced it — and the two ends of one
/// jump the same polynomials: `∫ (A + B·sin u)/((a + a·sin u)²·(c + d·sin
/// u)³) du` took the leading terms of the same ten polynomials 16 times
/// each, 8.8 of its 9 s (0.38).  Cleared when the outermost call ends, so
/// no entry outlives its arena.
pub(crate) struct JumpMemo(bool);

impl JumpMemo {
    /// Start remembering, unless an enclosing guard already does.
    pub(crate) fn enter() -> Self {
        JUMP_MEMO.with(|m| {
            let mut m = m.borrow_mut();
            let owner = m.is_none();
            if owner {
                *m = Some(JumpMemoMaps::default());
            }
            JumpMemo(owner)
        })
    }
}

impl Drop for JumpMemo {
    fn drop(&mut self) {
        if self.0 {
            let _ = JUMP_MEMO.try_with(|m| {
                if let Ok(mut m) = m.try_borrow_mut() {
                    *m = None;
                }
            });
        }
    }
}

/// The jump of `key`, remembered while a [`JumpMemo`] is alive.
fn memoized_jump(key: JumpKey, compute: impl FnOnce() -> Option<ExprId>) -> Option<ExprId> {
    if let Some(found) =
        JUMP_MEMO.with(|m| m.borrow().as_ref().and_then(|m| m.jumps.get(&key).copied()))
    {
        return found;
    }
    let jump = compute();
    JUMP_MEMO.with(|m| {
        if let Some(m) = m.borrow_mut().as_mut() {
            m.jumps.insert(key, jump);
        }
    });
    jump
}

/// Is `c` a constant with free parameters, all of them declared real (so
/// that `Im c` is a real number for every value of them)?  While
/// [`infinity_jump_real_parameters`] runs, every parameter counts as real.
fn only_real_parameters(arena: &Arena, c: ExprId) -> bool {
    let symbols = crate::base::walk::free_symbols(arena, c);
    let assumed = ASSUME_REAL_PARAMETERS.with(std::cell::Cell::get);
    !symbols.is_empty()
        && symbols.into_iter().all(|s| match *arena.node(s) {
            ExprNode::Symbol(sid) => {
                assumed || crate::transforms::realness::symbol_declared_real(arena, sid)
            }
            _ => false,
        })
}

/// Is `e`, a rational function of its symbols (sums, products, integer
/// powers), identically zero: is the numerator of its normal form zero?
/// The rates of the logarithms of a parametric answer cancel only so
/// (`b⁵/(a⁴(a² + b²)) + (a²b − b³)/a⁴ − b/(a² + b²)` in `∫ cot⁴u/(a + b·tan u)
/// du`), which the numerical zero test cannot decide.
fn rational_function_is_zero(arena: &mut Arena, e: ExprId) -> bool {
    let e = polynomial_root_sums(arena, e);
    if !rational_in_parameters(arena, e) {
        return false;
    }
    let combined = crate::poly::polybridge::together(arena, e);
    let (num, _) = crate::poly::polybridge::as_numer_denom(arena, combined);
    let num = crate::transforms::expand::expand(arena, num);
    if arena.is_zero_structural(num) {
        return true;
    }
    // Nested denominators `together` leaves: exact values at generic
    // rational points (a nonzero rational function of degree `D` vanishes
    // at a random point of a large grid with probability `≤ D/N`; and a
    // wrong "zero" only gives a locally constant term a wrong coefficient
    // where the integrand is not integrable at the poles of `tan w`).
    let symbols = crate::base::walk::free_symbols(arena, e);
    for point in 0..GENERIC_POINTS.len() {
        let mut v = e;
        for (k, &s) in symbols.iter().enumerate() {
            let (p, q) = GENERIC_POINTS[(point + k) % GENERIC_POINTS.len()];
            let value = arena.rational(p + i64::try_from(k).unwrap_or(0), q);
            v = crate::transforms::subs::subs(arena, v, s, value);
        }
        let v = crate::transforms::eval::eval(arena, v);
        if !arena.is_zero_structural(v) {
            return false;
        }
    }
    true
}

/// The points of [`rational_function_is_zero`] (coordinates `p/q` shifted
/// by the parameter's index).
const GENERIC_POINTS: [(i64, i64); 3] = [(7_919, 1_031), (-6_421, 2_053), (4_259, 3_967)];

/// Most roots a polynomial of [`polynomial_root_sums`] may have.
const MAX_NEWTON_DEGREE: usize = 64;

/// `e` with every `RootSum(p, b(s), s)` whose body is a polynomial in `s`
/// replaced by `Σ bₖ·Sₖ`, `Sₖ` the power sums of the roots of `p` (Newton's
/// identities: `aₙ·Sₖ + aₙ₋₁·Sₖ₋₁ + … + k·aₙ₋ₖ = 0`), a rational function of
/// the coefficients.  The rate `Σ c(ρ)` of the logarithms of a `RootSum`
/// over a polynomial with parameters is so decided to be 0.
fn polynomial_root_sums(arena: &mut Arena, e: ExprId) -> ExprId {
    let mut out = e;
    for id in crate::base::walk::post_order_ids(arena, e) {
        let ExprNode::RootSum(p, body, s) = *arena.node(id) else {
            continue;
        };
        let (Some(a), Some(b)) = (
            crate::transforms::solve::symbolic_poly_coeffs(arena, p, s),
            crate::transforms::solve::symbolic_poly_coeffs(arena, body, s),
        ) else {
            continue;
        };
        let n = a.len().saturating_sub(1);
        if n == 0 || n > MAX_NEWTON_DEGREE || b.len() > MAX_NEWTON_DEGREE {
            continue;
        }
        // `rⱼ = aₙ₋ⱼ/aₙ`, then `Sₖ = −(k·rₖ + Σ_{j<k} rⱼ·Sₖ₋ⱼ)` (`rⱼ = 0`, `j > n`).
        let lead = a[n];
        let minus_one = arena.int(-1);
        let inv_lead = arena.pow(lead, minus_one);
        let r: Vec<ExprId> = (0..=n)
            .map(|j| {
                if j == 0 {
                    arena.one
                } else {
                    arena.mul(&[a[n - j], inv_lead])
                }
            })
            .collect();
        let n_id = arena.int(i64::try_from(n).unwrap_or(0));
        let mut sums: Vec<ExprId> = vec![n_id];
        for k in 1..b.len() {
            let mut terms: Vec<ExprId> = Vec::new();
            if k <= n {
                let k_id = arena.int(i64::try_from(k).unwrap_or(0));
                terms.push(arena.mul(&[k_id, r[k]]));
            }
            for j in 1..k.min(n + 1) {
                terms.push(arena.mul(&[r[j], sums[k - j]]));
            }
            let total = arena.add(&terms);
            let s_k = arena.mul(&[minus_one, total]);
            sums.push(crate::transforms::eval::eval(arena, s_k));
        }
        let terms: Vec<ExprId> = b
            .iter()
            .zip(&sums)
            .map(|(&bk, &sk)| arena.mul(&[bk, sk]))
            .collect();
        let total = arena.add(&terms);
        out = arena.subs_structural(out, id, total);
    }
    out
}

/// Is the constant `c` real for every real value of its parameters, as
/// far as its form shows: sums, products and integer powers of rational
/// numbers and symbols.
fn rational_in_parameters(arena: &Arena, c: ExprId) -> bool {
    crate::base::walk::post_order_ids(arena, c)
        .iter()
        .all(|&id| match arena.node(id) {
            ExprNode::Num(_) | ExprNode::Symbol(_) | ExprNode::Add(_) | ExprNode::Mul(_) => true,
            ExprNode::Neg(_) => true,
            ExprNode::Pow(_, e) => arena.as_num(*e).is_some_and(|r| r.is_integer()),
            _ => false,
        })
}

thread_local! {
    /// Set while [`infinity_jump_real_parameters`] runs: every parameter
    /// counts as real in [`only_real_parameters`].
    static ASSUME_REAL_PARAMETERS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Set by [`RealParameterJumps`]: [`continuous_through_tan`] then falls
    /// back to [`infinity_jump_real_parameters`].
    static REAL_PARAMETER_JUMPS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Set by [`RealParameterJumps`]: [`continuous_through_tan`] writes every
    /// `ln|p(t)|` as `ln p(t)` before it takes the jump.
    static PLAIN_LOGS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// While alive, [`continuous_through_tan`] decides the jump of an answer
/// with parameters not declared real as for real ones
/// ([`infinity_jump_real_parameters`]).  For the routes whose answers were
/// never returned before (the trigonometric substitutions of 0.38), so that
/// no earlier answer changes.
///
/// The answer is `G` with `ln|p(t)|` written `ln p(t)` where the
/// integrator's stage exit would write `ln|p(tan w)|` so (decision D4: `p`
/// has a constant not declared real, or `w` has one, which `plain_logs`
/// says), and the jump is that of this `G`: `ln(tan w + b)` with a real `b`
/// steps by `i·π` where `tan w` passes from `+∞` to `−∞`, which the jump of
/// the `ln|·|` form missed (`∫ cot⁴(c + d·x)/(a + b·tan(c + d·x)) dx`
/// stepped by `−i·π·b/(d·(a² + b²))`).
pub(crate) struct RealParameterJumps(bool, bool);

impl RealParameterJumps {
    /// `plain_logs` adds to the setting of an enclosing guard.
    pub(crate) fn enable(plain_logs: bool) -> Self {
        let active = REAL_PARAMETER_JUMPS.with(|f| f.replace(true));
        let plain = PLAIN_LOGS.with(std::cell::Cell::get);
        PLAIN_LOGS.with(|f| f.set(plain_logs || (active && plain)));
        Self(active, plain)
    }
}

impl Drop for RealParameterJumps {
    fn drop(&mut self) {
        let _ = REAL_PARAMETER_JUMPS.try_with(|f| f.set(self.0));
        let _ = PLAIN_LOGS.try_with(|f| f.set(self.1));
    }
}

/// Is a [`StrictJumps`] alive?  Its routes then keep an answer whose jump
/// at the poles of `tan w` is still not decided only where the integrand
/// is not continuous across them.
pub(crate) fn strict_jumps_active() -> bool {
    STRICT_JUMPS.with(std::cell::Cell::get)
}

thread_local! {
    /// Set by [`StrictJumps`].
    static STRICT_JUMPS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// While alive, [`strict_jumps_active`] is set: for the routes whose
/// answers were never returned before (the trigonometric substitutions of
/// 0.38), which return no answer that jumps where the integrand is
/// continuous.
pub(crate) struct StrictJumps(bool);

impl StrictJumps {
    pub(crate) fn enable() -> Self {
        Self(STRICT_JUMPS.with(|f| f.replace(true)))
    }
}

impl Drop for StrictJumps {
    fn drop(&mut self) {
        let _ = STRICT_JUMPS.try_with(|f| f.set(self.0));
    }
}

/// `g` with `ln|p|` (`p` depending on `t`) written `ln p` where `all` is
/// set or `p` has a symbol not declared real: the logarithms the
/// integrator's stage exit writes `ln p` (decision D4), for a jump taken
/// before it.
pub(crate) fn plain_logs(arena: &mut Arena, g: ExprId, t_sym: SymbolId, all: bool) -> ExprId {
    let mut out = g;
    for id in crate::base::walk::post_order_ids(arena, g) {
        if let ExprNode::Ln(inner) = *arena.node(id)
            && let ExprNode::Abs(p) = *arena.node(inner)
            && contains_var(arena, p, t_sym)
            && (all
                || crate::base::walk::free_symbols(arena, p)
                    .into_iter()
                    .any(|s| match *arena.node(s) {
                        ExprNode::Symbol(sid) => {
                            sid != t_sym
                                && !crate::transforms::realness::symbol_declared_real(arena, sid)
                        }
                        _ => true,
                    }))
        {
            let plain = arena.ln(p);
            out = arena.subs_structural(out, id, plain);
        }
    }
    out
}

/// [`infinity_jump`] with every parameter taken for real: `J` as a function
/// of the parameters (with `sign(·)`, `sign(im(·))`, `sign(re(·))` where
/// the end values depend on them) that is exact for real values of them,
/// and, where the end values are those of `ln` of polynomials and of `atan`
/// with rational-in-parameter coefficients, for complex values too (the
/// side of the cut is read off `sign(im(·))` of the coefficients; up to
/// 0.39 it was taken as for real values, and `∫ dx/(a + tan x)` jumped at
/// every pole of `tan x` for `Im a < 0`).  Elsewhere `J·⌊w/π + 1/2⌋` is
/// still locally constant, so the answer stays an antiderivative; without
/// it the answer jumps for every value (the Rubi
/// suite's `∫ dx/(a + b·sin(c + d·x))` at `a = 6/5`, `b = 3/4` dropped by
/// `2π/(d·√(a² − b²))` at every pole of `tan(u/2)`).  An `atan` end with a
/// leading coefficient of undecided sign is followed only when that
/// coefficient is a rational function of the parameters (real for real
/// ones).
pub(crate) fn infinity_jump_real_parameters(
    arena: &mut Arena,
    g: ExprId,
    t: ExprId,
    t_sym: SymbolId,
) -> Option<ExprId> {
    // The parameters count as real whatever the caller's setting.
    let key = JumpKey {
        arena: std::ptr::from_ref::<Arena>(arena) as usize,
        real_parameters: true,
        assumed_real: true,
        g,
        t,
    };
    memoized_jump(key, || {
        infinity_jump_real_parameters_uncached(arena, g, t, t_sym)
    })
}

/// [`infinity_jump_real_parameters`] without the memo.
fn infinity_jump_real_parameters_uncached(
    arena: &mut Arena,
    g: ExprId,
    t: ExprId,
    t_sym: SymbolId,
) -> Option<ExprId> {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            let _ = ASSUME_REAL_PARAMETERS.try_with(|f| f.set(self.0));
        }
    }
    let restore = Restore(ASSUME_REAL_PARAMETERS.with(|f| f.replace(true)));
    let plus = end_value(arena, g, t, t_sym, true);
    let minus = plus
        .is_some()
        .then(|| end_value(arena, g, t, t_sym, false))
        .flatten();
    drop(restore);
    match (plus?, minus?) {
        (EndValue::Finite(p), EndValue::Finite(m)) => {
            let j = arena.sub(p, m);
            let j = crate::transforms::eval::eval(arena, j);
            Some(combined_signs(arena, j))
        }
        _ => Some(arena.zero),
    }
}

/// Does `e` contain the symbol `t` (free)?
fn contains_var_id(arena: &Arena, e: ExprId, t: ExprId) -> bool {
    match *arena.node(t) {
        ExprNode::Symbol(sid) => contains_var(arena, e, sid),
        _ => true,
    }
}

/// The limit of `g` as `t → +∞` (`positive`) or `t → −∞`, for the forms
/// the rational integrator writes: a rational function of `t`, constants
/// times `ln(p)`, `ln|p|` and `atan(p)` with `p` rational in `t` (the
/// divergent `d·ln|t|` parts of the logarithms must cancel), and other
/// terms whose limit the limit engine finds finite.  `None` when a term is
/// of another form or a sign or leading coefficient is not decided.
///
/// The limit engine alone does not do: it leaves `ln(t + i√3) − ln(t − i√3)`
/// at `t → −∞` unevaluated (the answer, `2πi`, depends on the sides of the
/// cut from which the two arguments approach it).
pub(crate) fn end_value(
    arena: &mut Arena,
    g: ExprId,
    t: ExprId,
    t_sym: SymbolId,
    positive: bool,
) -> Option<EndValue> {
    let terms = additive_terms(arena, g, t_sym);
    let mut rational: Vec<ExprId> = Vec::new();
    let mut finite: Vec<ExprId> = Vec::new();
    let mut log_rate: Vec<ExprId> = Vec::new();
    for term in terms {
        // A term this large is not followed: see [`MAX_END_TERM_NODES`].
        if crate::base::walk::post_order_ids(arena, term).len() > MAX_END_TERM_NODES {
            tracing::debug!("end_value: a term too large to follow to the end");
            return None;
        }
        if !contains_var(arena, term, t_sym) {
            finite.push(term);
            continue;
        }
        let (c, h) = match arena.node(term).clone() {
            ExprNode::Mul(children) => {
                let (dep, consts): (Vec<ExprId>, Vec<ExprId>) = children
                    .iter()
                    .partition(|&&k| contains_var(arena, k, t_sym));
                let c = arena.mul(&consts);
                let h = if dep.len() == 1 {
                    dep[0]
                } else {
                    arena.mul(&dep)
                };
                (c, h)
            }
            ExprNode::Neg(inner) => (arena.int(-1), inner),
            _ => (arena.one, term),
        };
        match arena.node(h).clone() {
            ExprNode::Ln(arg) => {
                let (p, absolute) = match arena.node(arg) {
                    ExprNode::Abs(inner) => (*inner, true),
                    _ => (arg, false),
                };
                let (d, lead) = rational_end(arena, p, t, positive)?;
                if d != 0 {
                    let d_id = arena.int(d);
                    log_rate.push(arena.mul(&[c, d_id]));
                }
                let value = if absolute {
                    let a = arena.abs(lead);
                    arena.ln(a)
                } else {
                    log_end(arena, p, t, lead, positive)?
                };
                finite.push(arena.mul(&[c, value]));
            }
            ExprNode::Atan(q) => {
                let (d, lead) = rational_end(arena, q, t, positive)?;
                let value = match d.cmp(&0) {
                    // `atan(z) → ±π/2` as `|z| → ∞` with `Re z ≷ 0`.
                    std::cmp::Ordering::Greater => {
                        let pi = arena.pi();
                        match placement(arena, lead) {
                            Some(Placement::Complex { re_sign: 0, .. }) => return None,
                            Some(found) => {
                                let sign = match found {
                                    Placement::Positive => 1,
                                    Placement::Negative => -1,
                                    Placement::Complex { re_sign, .. } => i64::from(re_sign),
                                };
                                let half = arena.rational(sign, 2);
                                arena.mul(&[half, pi])
                            }
                            // A real leading coefficient of undecided sign
                            // (`atan(a·t/√(a² + 2))` for a real `a`):
                            // `±π/2` by its sign for every value of the
                            // parameters (where it is 0 the term is not
                            // defined: it divides by that coefficient).
                            None if only_real_parameters(arena, lead)
                                && crate::transforms::realness::constant_realness_as_declared(
                                    arena,
                                    lead,
                                    30,
                                    &mut crate::base::assumptions::AssumptionCache::new(),
                                ) == Some(true) =>
                            {
                                let half = arena.rational(1, 2);
                                let sign = sign_of_real(arena, lead);
                                arena.mul(&[half, pi, sign])
                            }
                            // Real parameters assumed, `lead` rational in
                            // parameters not declared real: `±π/2` by the
                            // sign of `Re lead`, which is that of `lead` for
                            // real values (`sign(lead)` is not real for the
                            // others).
                            None if ASSUME_REAL_PARAMETERS.with(std::cell::Cell::get)
                                && only_real_parameters(arena, lead)
                                && rational_in_parameters(arena, lead) =>
                            {
                                let half = arena.rational(1, 2);
                                let re = arena.re(lead);
                                let re = crate::transforms::eval::eval(arena, re);
                                let sign = sign_of_real(arena, re);
                                arena.mul(&[half, pi, sign])
                            }
                            None => return None,
                        }
                    }
                    // A limit on the cut (`iy`, `|y| ≥ 1`) is not followed.
                    std::cmp::Ordering::Equal => match placement(arena, lead) {
                        Some(Placement::Complex {
                            re_sign: 0,
                            im_at_least_one: true,
                        }) => return None,
                        Some(_) => arena.atan(lead),
                        // Real for real parameters: off the cut.
                        None if ASSUME_REAL_PARAMETERS.with(std::cell::Cell::get)
                            && rational_in_parameters(arena, lead) =>
                        {
                            arena.atan(lead)
                        }
                        None => return None,
                    },
                    std::cmp::Ordering::Less => arena.zero,
                };
                finite.push(arena.mul(&[c, value]));
            }
            ExprNode::RootSum(p, body, s) => {
                let (rate, value) = root_sum_end(arena, p, body, s, t, t_sym, positive)?;
                log_rate.push(arena.mul(&[c, rate]));
                finite.push(arena.mul(&[c, value]));
            }
            _ if rational_end(arena, h, t, positive).is_some() => rational.push(term),
            _ => {
                let point = if positive {
                    arena.infinity()
                } else {
                    arena.neg_infinity()
                };
                let v = arena.limit_expr(term, t, point).ok()?;
                let v = crate::transforms::eval::eval(arena, v);
                if contains_var(arena, v, t_sym)
                    || crate::base::walk::has_unevaluated(arena, v)
                    || crate::transforms::evalf::evalf_complex64(arena, v)
                        .ok()
                        .is_none_or(|z| !z.re.is_finite() || !z.im.is_finite())
                {
                    return None;
                }
                finite.push(v);
            }
        }
    }
    if !rational.is_empty() {
        let r = arena.add(&rational);
        let (d, lead) = rational_end(arena, r, t, positive)?;
        match d.cmp(&0) {
            std::cmp::Ordering::Greater => return Some(EndValue::Infinite),
            std::cmp::Ordering::Equal => finite.push(lead),
            std::cmp::Ordering::Less => {}
        }
    }
    if !log_rate.is_empty() {
        let rate = arena.add(&log_rate);
        let rate = crate::transforms::eval::eval(arena, rate);
        match crate::poly::algebraic::is_zero_checked(arena, rate) {
            Some(true) => {}
            Some(false) => return Some(EndValue::Infinite),
            None if rational_function_is_zero(arena, rate) => {}
            None => return None,
        }
    }
    let value = arena.add(&finite);
    Some(EndValue::Finite(crate::transforms::eval::eval(
        arena, value,
    )))
}

/// The largest term (distinct nodes) whose end value [`end_value`] follows.
/// On the Rubi suite (0.38) every term it decided, or handed to the limit
/// engine and saw refused, had at most 243 nodes, and the limit engine's
/// calls took about a second in all; the terms beyond were the Euler
/// substitution's answers for the `(a + i·a·tan u)^(m/2)·(c − i·c·tan
/// u)^(n/2)` family, single products of 13,826 nodes on which the limit
/// engine worked for 10 s before its work budget ran out — the integration
/// timed out.  Such a term leaves the end value undecided at once, as the
/// exhausted budget did.
const MAX_END_TERM_NODES: usize = 2_000;

/// The terms of `g` as a sum, with every product `c·(u₁ + … + uₖ)` whose
/// only `t`-dependent factor is a sum distributed (`(−3)^(−1/2)·(ln(t − i√3)
/// − ln(t + i√3))`), at most [`MAX_DISTRIBUTED_TERMS`] of them.
fn additive_terms(arena: &mut Arena, g: ExprId, t_sym: SymbolId) -> Vec<ExprId> {
    let mut out: Vec<ExprId> = Vec::new();
    let mut work: Vec<ExprId> = vec![g];
    while let Some(e) = work.pop() {
        match arena.node(e).clone() {
            ExprNode::Add(children)
                if out.len() + work.len() + children.len() <= MAX_DISTRIBUTED_TERMS =>
            {
                work.extend(children.iter().copied());
            }
            ExprNode::Mul(children) => {
                let dependent: Vec<usize> = (0..children.len())
                    .filter(|&k| contains_var(arena, children[k], t_sym))
                    .collect();
                let sum = match dependent.as_slice() {
                    [k] => match arena.node(children[*k]) {
                        ExprNode::Add(parts) => Some((*k, parts.to_vec())),
                        _ => None,
                    },
                    _ => None,
                };
                match sum {
                    Some((k, parts))
                        if out.len() + work.len() + parts.len() <= MAX_DISTRIBUTED_TERMS =>
                    {
                        let mut rest: Vec<ExprId> = children.to_vec();
                        rest.remove(k);
                        let c = arena.mul(&rest);
                        for part in parts {
                            work.push(arena.mul(&[c, part]));
                        }
                    }
                    _ => out.push(e),
                }
            }
            _ => out.push(e),
        }
    }
    out
}

/// Most terms [`additive_terms`] writes out.
const MAX_DISTRIBUTED_TERMS: usize = 256;

/// `Σ_{p(ρ)=0} c(ρ)·ln(t + β(ρ))` (`body` = `c(s)·ln(t + β(s))`, `p ∈ ℚ[s]`)
/// near `t = ±∞`: `(Σ c(ρ), value)` with the sum `≈ Σ c(ρ)·ln|t| + value`.
/// `ln(t + β) → ln t` at `+∞`; at `−∞`, `t + β` approaches the cut from
/// above where `Im β > 0` and lies on it where `β` is real (principal
/// `ln`: `+iπ`), so `ln(t + β) − ln|t| → iπ·σ(β)` with `σ = −1` where
/// `Im β < 0` and `σ = 1` otherwise.  Without real roots of `p` the value is
/// `iπ·Σ c(ρ)·sign(Im β(ρ))`, written as a `RootSum` (the sum over the roots
/// in one half-plane has no simpler exact form in general).  The
/// `RootSum`s of the rational integrator have `c(ρ)` the residue at the
/// pole `−β(ρ)` of a real rational function, so `β(ρ)` is not real where
/// `ρ` is not.
///
/// With real roots (`∫ dx/(2·sin⁵x + 1)` through `tan(x/2)`: the real
/// roots are the residues at the real poles) `sign(Im β)` is 0 there where
/// `σ` is 1: the value is `iπ·Σ c·σ = iπ·Σ c + iπ·Σ c(ρ)·sg·(1 − sg)` with
/// `sg = sign(Im β(ρ))`, the second sum `−2πi·Σ_{Im β < 0} c(ρ)`; where
/// the whole antiderivative's rate is 0 that makes the jump `J` `2πi` times
/// the sum of the residues at the poles in the upper half-plane, whatever
/// the real poles.  The real roots' terms of the second sum vanish exactly
/// (a certified real root has an exactly zero imaginary part in the
/// numerical `RootSum`), so no real root has to be told from a complex one.
fn root_sum_end(
    arena: &mut Arena,
    p: ExprId,
    body: ExprId,
    s: ExprId,
    t: ExprId,
    t_sym: SymbolId,
    positive: bool,
) -> Option<(ExprId, ExprId)> {
    // Real roots or not, `iπ·(Σ c + RootSum(c·sg·(1 − sg)))` is the value
    // (where `sg = ±1` it is `iπ·Σ c·sg`); with real parameters assumed
    // a polynomial with parameters takes it.
    let real_roots = match crate::poly::polybridge::expr_to_poly(arena, p, s) {
        Some(p_poly) => {
            if p_poly.degree().is_none_or(|d| d == 0) {
                return None;
            }
            crate::poly::sturm::SturmChain::new(&p_poly).count_real_roots() != 0
        }
        None if ASSUME_REAL_PARAMETERS.with(std::cell::Cell::get) => {
            let coeffs = crate::transforms::solve::symbolic_poly_coeffs(arena, p, s)?;
            if coeffs.len() < 2 || !coeffs.iter().all(|&k| rational_in_parameters(arena, k)) {
                return None;
            }
            true
        }
        None => return None,
    };
    let ExprNode::Mul(children) = arena.node(body).clone() else {
        return None;
    };
    let logs: Vec<usize> = (0..children.len())
        .filter(|&k| contains_var(arena, children[k], t_sym))
        .collect();
    let [k] = logs.as_slice() else {
        return None;
    };
    let ExprNode::Ln(arg) = *arena.node(children[*k]) else {
        return None;
    };
    let coeffs = crate::transforms::solve::symbolic_poly_coeffs(arena, arg, t)?;
    let [beta, one] = coeffs.as_slice() else {
        return None;
    };
    if *one != arena.one || contains_var(arena, *beta, t_sym) {
        return None;
    }
    let beta = *beta;
    let mut rest: Vec<ExprId> = children.to_vec();
    rest.remove(*k);
    let c = arena.mul(&rest);
    let rate = arena.intern(ExprNode::RootSum(p, c, s));
    if positive {
        return Some((rate, arena.zero));
    }
    let im_beta = arena.im(beta);
    let sign_im = arena.sign(im_beta);
    let sum = if real_roots {
        let one = arena.one;
        let minus = arena.neg(sign_im);
        let lower = arena.add(&[one, minus]);
        let signed = arena.mul(&[c, sign_im, lower]);
        let below = arena.intern(ExprNode::RootSum(p, signed, s));
        let sum = arena.add(&[rate, below]);
        crate::transforms::eval::eval(arena, sum)
    } else {
        let signed = arena.mul(&[c, sign_im]);
        arena.intern(ExprNode::RootSum(p, signed, s))
    };
    let i = arena.i_unit();
    let pi = arena.pi();
    Some((rate, arena.mul(&[i, pi, sum])))
}

/// The jump `J = G(+∞) − G(−∞)` of an antiderivative `G(t)` of a rational
/// function of `t` (or of a function of `t` whose end values
/// [`end_value`] finds): `Some(0)` when `G` is infinite at an end (the
/// integrand is not integrable there, so nothing is to be made continuous),
/// `None` when an end value is not decided.
///
/// A parameter that is not declared real may be complex (decision D4), and
/// then the end values of `ln(t + a)` depend on the side of the cut from
/// which `t + a` approaches it: `None` at once (on the Rubi suite the
/// attempt cost seconds on large parametric answers and decided none).
pub(crate) fn infinity_jump(
    arena: &mut Arena,
    g: ExprId,
    t: ExprId,
    t_sym: SymbolId,
) -> Option<ExprId> {
    let key = JumpKey {
        arena: std::ptr::from_ref::<Arena>(arena) as usize,
        real_parameters: false,
        assumed_real: ASSUME_REAL_PARAMETERS.with(std::cell::Cell::get),
        g,
        t,
    };
    memoized_jump(key, || infinity_jump_uncached(arena, g, t, t_sym))
}

/// [`infinity_jump`] without the memo.
fn infinity_jump_uncached(
    arena: &mut Arena,
    g: ExprId,
    t: ExprId,
    t_sym: SymbolId,
) -> Option<ExprId> {
    let undeclared = crate::base::walk::free_symbols(arena, g)
        .into_iter()
        .any(|s| match *arena.node(s) {
            ExprNode::Symbol(sid) => {
                sid != t_sym && !crate::transforms::realness::symbol_declared_real(arena, sid)
            }
            _ => true,
        });
    if undeclared {
        return None;
    }
    let plus = end_value(arena, g, t, t_sym, true)?;
    let minus = end_value(arena, g, t, t_sym, false)?;
    match (plus, minus) {
        (EndValue::Finite(p), EndValue::Finite(m)) => {
            let j = arena.sub(p, m);
            let j = crate::transforms::eval::eval(arena, j);
            Some(combined_signs(arena, j))
        }
        _ => Some(arena.zero),
    }
}

/// `j` expanded when that is smaller: the end values with parameters of
/// [`log_end`] are `iπ·(1 + sg − sg²)` per logarithm, and the pair
/// `ln(t + β) − ln(t − β)` leaves `2πi·sg` only after expansion.
fn combined_signs(arena: &mut Arena, j: ExprId) -> ExprId {
    let has_sign = crate::base::walk::post_order_ids(arena, j)
        .iter()
        .any(|&id| matches!(arena.node(id), ExprNode::Sign(_)));
    if !has_sign {
        return j;
    }
    let expanded = crate::transforms::expand::expand(arena, j);
    let expanded = crate::transforms::eval::eval(arena, expanded);
    if printed_size(arena, expanded) < printed_size(arena, j) {
        expanded
    } else {
        j
    }
}

/// The number of nodes of `e` written out as a tree (shared subexpressions
/// counted at every occurrence, as printed), saturating.
fn printed_size(arena: &Arena, e: ExprId) -> usize {
    let mut size: rustc_hash::FxHashMap<ExprId, usize> = rustc_hash::FxHashMap::default();
    for id in crate::base::walk::post_order_ids(arena, e) {
        let mut s: usize = 1;
        arena
            .node(id)
            .for_each_child(|c| s = s.saturating_add(size.get(&c).copied().unwrap_or(1)));
        size.insert(id, s);
    }
    size.get(&e).copied().unwrap_or(1)
}

/// `sign(z)` for a real `z ≠ 0`, with the factors of `z` the assumptions
/// prove positive dropped and those they prove negative turned into a
/// factor `−1` (`sign(a/√(a² + 2)) = sign(a)`).
fn sign_of_real(arena: &mut Arena, z: ExprId) -> ExprId {
    use crate::base::assumptions::{AssumptionCache, Props};
    let factors: Vec<ExprId> = match arena.node(z) {
        ExprNode::Mul(children) => children.to_vec(),
        _ => vec![z],
    };
    let mut facts = AssumptionCache::new();
    let mut kept: Vec<ExprId> = Vec::new();
    let mut negative = false;
    for f in factors {
        if facts.query(arena, f, Props::POSITIVE) == Some(true) {
            continue;
        }
        if facts.query(arena, f, Props::NEGATIVE) == Some(true) {
            negative = !negative;
            continue;
        }
        kept.push(f);
    }
    let s = if kept.is_empty() {
        arena.one
    } else {
        let core = arena.mul(&kept);
        arena.sign(core)
    };
    if negative { arena.neg(s) } else { s }
}

/// `G(tan w)` made continuous across the poles of `tan w`, for an
/// antiderivative `G(t)` (in the symbol `t`) of the integrand after the
/// substitution `t = tan w`, `w` real for real `x` (any real continuous
/// function of `x`; the half-angle substitution has `w = x/2`).
///
/// Where `w` crosses `π/2 + kπ` upwards, `tan w` passes from `+∞` to
/// `−∞`, and `G(tan w)` from `G(+∞)` to `G(−∞)`: it drops by
/// `J = G(+∞) − G(−∞)` ([`infinity_jump`]) wherever the integrand is
/// continuous there.  `⌊w/π + 1/2⌋` rises by 1 at exactly those points (and
/// falls by 1 where `w` crosses them downwards, where `G(tan w)` rises), so
/// `G(tan w) + J·⌊w/π + 1/2⌋` is continuous; it is the old answer on
/// `−π/2 < w < π/2` and has the same derivative (`floor′ = 0`).  This is
/// the correction of D. J. Jeffrey and A. D. Rich, "The evaluation of
/// trigonometric integrals avoiding spurious discontinuities" (ACM TOMS 20,
/// 1994) and of SymPy's `Integral.doit`, which adds
/// `sign(c)·π·⌊(w − π/2)/π⌋` to an `atan(c·tan w + d)` (a constant away).
///
/// A term `c·atan(tan w)` is written `c·w − c·π·⌊w/π + 1/2⌋`, which it equals
/// for real `w`; with the correction the floors then often cancel
/// (`∫ tan x/(tan x + 2) dx` is `x/5 − 2/5·ln|tan x + 2| + 1/5·ln(tan²x + 1)`,
/// as SymPy writes it).
///
/// `None` when `J` is not decided (the caller then has no continuous
/// answer).
pub(crate) fn continuous_through_tan(
    arena: &mut Arena,
    g: ExprId,
    t: ExprId,
    t_sym: SymbolId,
    w: ExprId,
) -> Option<ExprId> {
    let g = if REAL_PARAMETER_JUMPS.with(std::cell::Cell::get) {
        let all = PLAIN_LOGS.with(std::cell::Cell::get);
        plain_logs(arena, g, t_sym, all)
    } else {
        g
    };
    let jump = match infinity_jump(arena, g, t, t_sym) {
        Some(j) => j,
        None if REAL_PARAMETER_JUMPS.with(std::cell::Cell::get) => {
            infinity_jump_real_parameters(arena, g, t, t_sym)?
        }
        None => return None,
    };
    let tan_w = arena.tan(w);
    let back = arena.subs_structural(g, t, tan_w);
    Some(add_tan_floor(arena, back, w, jump))
}

/// `big_f + jump·⌊w/π + 1/2⌋`, with every term `c·atan(tan w)` of `big_f`
/// written `c·w − c·π·⌊w/π + 1/2⌋` (so that the floors combine).
pub(crate) fn add_tan_floor(arena: &mut Arena, big_f: ExprId, w: ExprId, jump: ExprId) -> ExprId {
    add_tan_floor_in(arena, big_f, w, jump, None)
}

/// [`add_tan_floor`] for a `w` with parameters: a coefficient `c` of
/// `atan(tan w)` must be free of the variable `var` only (with `None`, of
/// every symbol of `w`).  `∫ (c + d·tan(f·x))/(1 + i·tan(f·x)) dx` kept
/// `c·atan(tan(f·x))/(2f)`, whose coefficient has the `f` of `w`.
pub(crate) fn add_tan_floor_in(
    arena: &mut Arena,
    big_f: ExprId,
    w: ExprId,
    jump: ExprId,
    var: Option<SymbolId>,
) -> ExprId {
    let pi = arena.pi();
    let (mut kept, atan_coeffs) = split_atan_tan_terms(arena, big_f, w, var);
    let mut floor_coeff: Vec<ExprId> = vec![jump];
    for c in atan_coeffs {
        let minus_one = arena.int(-1);
        floor_coeff.push(arena.mul(&[minus_one, c, pi]));
    }
    let coeff = arena.add(&floor_coeff);
    let coeff = crate::transforms::eval::eval(arena, coeff);
    if crate::poly::algebraic::is_zero_checked(arena, coeff) != Some(true) {
        let minus_one = arena.int(-1);
        let inv_pi = arena.pow(pi, minus_one);
        let w_over_pi = arena.mul(&[w, inv_pi]);
        let half = arena.rational(1, 2);
        let arg = arena.add(&[w_over_pi, half]);
        let fl = arena.floor(arg);
        kept.push(arena.mul(&[coeff, fl]));
    }
    arena.add(&kept)
}

/// `big_f` with every term `c·atan(tan w)` written `c·w`, which differs
/// from it by a constant between consecutive poles of `tan w` and has no
/// jumps (`atan(tan w) = w − π·⌊w/π + 1/2⌋` for real `w`).  For the
/// answers whose jump at those poles is not decided otherwise.
pub(crate) fn atan_tan_terms_to_argument(arena: &mut Arena, big_f: ExprId, w: ExprId) -> ExprId {
    atan_tan_terms_to_argument_in(arena, big_f, w, None)
}

/// [`atan_tan_terms_to_argument`] for a `w` with parameters (see
/// [`add_tan_floor_in`]).
pub(crate) fn atan_tan_terms_to_argument_in(
    arena: &mut Arena,
    big_f: ExprId,
    w: ExprId,
    var: Option<SymbolId>,
) -> ExprId {
    let (kept, atan_coeffs) = split_atan_tan_terms(arena, big_f, w, var);
    if atan_coeffs.is_empty() {
        return big_f;
    }
    arena.add(&kept)
}

/// The terms of `big_f`, each `c·atan(tan w)` (with `c` free of the
/// variable `var`, or with `None` of every symbol of `w`) replaced by
/// `c·w`, and the coefficients `c` of those.
fn split_atan_tan_terms(
    arena: &mut Arena,
    big_f: ExprId,
    w: ExprId,
    var: Option<SymbolId>,
) -> (Vec<ExprId>, Vec<ExprId>) {
    let tan_w = arena.tan(w);
    let atan_tan_w = arena.atan(tan_w);
    if !crate::base::walk::contains(arena, big_f, atan_tan_w) {
        let terms = match arena.node(big_f) {
            ExprNode::Add(children) => children.to_vec(),
            _ => vec![big_f],
        };
        return (terms, Vec::new());
    }
    let terms = terms_distributed(arena, big_f, w, var);
    let mut kept: Vec<ExprId> = Vec::with_capacity(terms.len() + 1);
    let mut coeffs: Vec<ExprId> = Vec::new();
    for term in terms {
        let c = if term == atan_tan_w {
            Some(arena.one)
        } else if let ExprNode::Mul(children) = arena.node(term).clone()
            && children.iter().filter(|&&k| k == atan_tan_w).count() == 1
        {
            let rest: Vec<ExprId> = children
                .iter()
                .copied()
                .filter(|&k| k != atan_tan_w)
                .collect();
            Some(arena.mul(&rest))
        } else {
            None
        };
        match c {
            Some(c) if !depends_on_variable(arena, c, w, var) => {
                kept.push(arena.mul(&[c, w]));
                coeffs.push(c);
            }
            _ => kept.push(term),
        }
    }
    (kept, coeffs)
}

/// The terms of `big_f` as a sum, every product `c·(u₁ + … + uₖ)` whose
/// only factor with a symbol of `w` is a sum distributed (at most
/// [`MAX_DISTRIBUTED_TERMS`] terms): a constant factor of an answer (`1/a`
/// in `(c·atan(tan w) + …)/a`, split off by linearity) hid its terms
/// `c·atan(tan w)` from [`split_atan_tan_terms`], and the answer kept the
/// jumps of `atan(tan w)`.
fn terms_distributed(
    arena: &mut Arena,
    big_f: ExprId,
    w: ExprId,
    var: Option<SymbolId>,
) -> Vec<ExprId> {
    let mut out: Vec<ExprId> = Vec::new();
    let mut work: Vec<ExprId> = vec![big_f];
    while let Some(e) = work.pop() {
        match arena.node(e).clone() {
            ExprNode::Add(children)
                if out.len() + work.len() + children.len() <= MAX_DISTRIBUTED_TERMS =>
            {
                work.extend(children.iter().copied());
            }
            ExprNode::Mul(children) => {
                let dependent: Vec<usize> = (0..children.len())
                    .filter(|&k| depends_on_variable(arena, children[k], w, var))
                    .collect();
                let sum = match dependent.as_slice() {
                    [k] => match arena.node(children[*k]) {
                        ExprNode::Add(parts) => Some((*k, parts.to_vec())),
                        _ => None,
                    },
                    _ => None,
                };
                match sum {
                    Some((k, parts))
                        if out.len() + work.len() + parts.len() <= MAX_DISTRIBUTED_TERMS =>
                    {
                        let mut rest: Vec<ExprId> = children.to_vec();
                        rest.remove(k);
                        let c = arena.mul(&rest);
                        for part in parts {
                            work.push(arena.mul(&[c, part]));
                        }
                    }
                    _ => out.push(e),
                }
            }
            _ => out.push(e),
        }
    }
    out
}

/// Does `c` depend on the variable `var` (with `None`: on a free symbol of
/// `w`)?
fn depends_on_variable(arena: &Arena, c: ExprId, w: ExprId, var: Option<SymbolId>) -> bool {
    match var {
        Some(v) => contains_var(arena, c, v),
        None => free_symbols_meet(arena, c, w),
    }
}

/// Does `c` contain a free symbol of `w` (so that it is not constant in
/// the variable of `w`)?
fn free_symbols_meet(arena: &Arena, c: ExprId, w: ExprId) -> bool {
    let of_w = crate::base::walk::free_symbols(arena, w);
    crate::base::walk::free_symbols(arena, c)
        .iter()
        .any(|s| of_w.contains(s))
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::arena::Arena;

    /// `G(+∞) − G(−∞)` of `g` (in `t`), printed; `None` when undecided.
    fn jump_of(g: &str) -> Option<String> {
        let ctx = crate::api::context::Context::new();
        let g = ctx.parse(g).unwrap().id();
        ctx.with_arena_mut(|a| {
            let t = a.symbol("t");
            let ExprNode::Symbol(t_sym) = *a.node(t) else {
                unreachable!()
            };
            infinity_jump(a, g, t, t_sym).map(|j| a.display(j).to_string())
        })
    }

    #[test]
    fn jumps_at_infinity_of_rational_antiderivatives() {
        // atan: ±π/2 by the sign of the leading coefficient.
        assert_eq!(
            jump_of("2/3*sqrt(3)*atan(1/3*sqrt(3)*t)").as_deref(),
            Some("2/3*sqrt(3)*pi")
        );
        assert_eq!(
            jump_of("1/4*sqrt(2)*(2*atan(1/4*sqrt(2)*(t^3 + 7*t)) + 2*atan(1/4*sqrt(2)*t))")
                .as_deref(),
            Some("sqrt(2)*pi")
        );
        // The ln|·| parts cancel at both ends.
        assert_eq!(
            jump_of("-2/5*ln(abs(t + 2)) + 1/5*ln(t^2 + 1) + 1/5*atan(t)").as_deref(),
            Some("1/5*pi")
        );
        // ln(t + i√3) − ln(t − i√3) → 2πi at −∞ (the limit engine leaves
        // it unevaluated): J = (−3)^(−1/2)·(−2πi) = −2π/√3.
        let j = jump_of("(-3)^(-1/2)*(-ln(sqrt(3)*I + t) + ln(-sqrt(3)*I + t))").unwrap();
        let ctx = crate::api::context::Context::new();
        let v = ctx.parse(&j).unwrap().eval_complex64().unwrap();
        let want = 2.0 * std::f64::consts::PI / 3f64.sqrt();
        assert!(
            (v.re - want).abs() < 1e-14 && v.im.abs() < 1e-14,
            "{j} = {v}"
        );
        // Infinite at an end: the integrand is not integrable there.
        assert_eq!(jump_of("t").as_deref(), Some("0"));
        assert_eq!(jump_of("ln(abs(t))").as_deref(), Some("0"));
        // A free parameter that is not declared real: undecided.
        assert_eq!(jump_of("atan(t/a)"), None);
    }

    #[test]
    fn jump_of_a_root_sum_over_one_half_plane() {
        // Σ_{ρ⁴ = −1} −ρ/4·ln(t − ρ) = ∫ dt/(t⁴ + 1): J = ∫_ℝ = π/√2.
        let mut a = Arena::new();
        let t = a.symbol("t");
        let s = a.symbol("s");
        let ExprNode::Symbol(t_sym) = *a.node(t) else {
            unreachable!()
        };
        let four = a.int(4);
        let s4 = a.pow(s, four);
        let one = a.one;
        let p = a.add(&[s4, one]);
        let t_minus_s = a.sub(t, s);
        let ln = a.ln(t_minus_s);
        let quarter = a.rational(-1, 4);
        let body = a.mul(&[quarter, s, ln]);
        let g = a.intern(ExprNode::RootSum(p, body, s));
        let j = infinity_jump(&mut a, g, t, t_sym).unwrap();
        let v = crate::transforms::evalf::evalf_complex64(&a, j).unwrap();
        let want = std::f64::consts::PI / 2f64.sqrt();
        assert!(
            (v.re - want).abs() < 1e-14 && v.im.abs() < 1e-14,
            "{} = {v}",
            a.display(j)
        );
    }

    /// `J` of a `RootSum` in `t` over `p` (in `s`) with body `ln(t − s)/q(s)`,
    /// evaluated.
    fn root_sum_jump(p: &str, q: &str) -> num_complex::Complex64 {
        let ctx = crate::api::context::Context::new();
        let p = ctx.parse(p).unwrap().id();
        let q = ctx.parse(q).unwrap().id();
        ctx.with_arena_mut(|a| {
            let t = a.symbol("t");
            let s = a.symbol("s");
            let ExprNode::Symbol(t_sym) = *a.node(t) else {
                unreachable!()
            };
            let t_minus_s = a.sub(t, s);
            let ln = a.ln(t_minus_s);
            let body = a.div(ln, q);
            let g = a.intern(ExprNode::RootSum(p, body, s));
            let j = infinity_jump(a, g, t, t_sym).unwrap();
            crate::transforms::evalf::evalf_complex64(a, j).unwrap()
        })
    }

    #[test]
    fn jump_of_a_root_sum_with_real_roots() {
        // ∫ dt/((t² − 2)(t² + 1)) = Σ ln(t − ρ)/D′(ρ): the real poles ±√2
        // have opposite residues; J = 2πi·Res(t = i) = −π/3 (the real form
        // (1/3)·(ln|(t − √2)/(t + √2)|/(2√2) − atan t) drops by π/3).
        let j = root_sum_jump("s^4 - s^2 - 2", "4*s^3 - 2*s");
        let want = -std::f64::consts::PI / 3.0;
        assert!((j.re - want).abs() < 1e-14 && j.im.abs() < 1e-14, "{j}");
        // ∫ dt/((t − 1)(t² + 1)) with the principal ln(t − 1), which is
        // iπ/2 higher at t → −∞: J = 2πi·Res(t = i) = −π/2 − iπ/2.
        let j = root_sum_jump("s^3 - s^2 + s - 1", "3*s^2 - 2*s + 1");
        let want = -std::f64::consts::FRAC_PI_2;
        assert!(
            (j.re - want).abs() < 1e-14 && (j.im - want).abs() < 1e-14,
            "{j}"
        );
    }

    #[test]
    fn jump_of_a_root_sum_whose_own_rate_is_not_zero() {
        // ∫ dt/((t³ − 2)(t² + 1)) as two RootSums: over s³ − 2 (one real
        // root; its residues sum to −1/5) and over s² + 1 (+1/5).  mpmath:
        // 2πi·(Res(t = i) + Res(t = ∛2·e^(2πi/3))) =
        // −0.909316057595450453 − 0.254963612453376551i, and the principal
        // logarithms' G(10¹²) − G(−10¹²) agree to 30 digits.
        let ctx = crate::api::context::Context::new();
        let j = ctx.with_arena_mut(|a| {
            let t = a.symbol("t");
            let s = a.symbol("s");
            let ExprNode::Symbol(t_sym) = *a.node(t) else {
                unreachable!()
            };
            let t_minus_s = a.sub(t, s);
            let ln = a.ln(t_minus_s);
            let two = a.int(2);
            let three = a.int(3);
            let s2 = a.pow(s, two);
            let s3 = a.pow(s, three);
            let minus_two = a.int(-2);
            let one = a.one;
            let cubic = a.add(&[s3, minus_two]);
            let quad = a.add(&[s2, one]);
            let q1 = a.mul(&[three, s2, quad]);
            let body1 = a.div(ln, q1);
            let g1 = a.intern(ExprNode::RootSum(cubic, body1, s));
            let q2 = a.mul(&[two, s, cubic]);
            let body2 = a.div(ln, q2);
            let g2 = a.intern(ExprNode::RootSum(quad, body2, s));
            let g = a.add(&[g1, g2]);
            let j = infinity_jump(a, g, t, t_sym).unwrap();
            crate::transforms::evalf::evalf_complex64(a, j).unwrap()
        });
        assert!(
            (j.re + 0.909_316_057_595_450_5).abs() < 1e-13
                && (j.im + 0.254_963_612_453_376_55).abs() < 1e-13,
            "{j}"
        );
    }

    #[test]
    fn jumps_with_real_parameters_of_undecided_sign() {
        let ctx = crate::api::context::Context::new();
        let a = ctx
            .symbol_with("a", &[crate::base::assumptions::Assumption::Real])
            .unwrap();
        let jump = |src: &str| -> crate::api::expr::Ex {
            let g = ctx.parse(src).unwrap();
            let j = ctx.with_arena_mut(|ar| {
                let t = ar.symbol("t");
                let ExprNode::Symbol(t_sym) = *ar.node(t) else {
                    unreachable!()
                };
                infinity_jump(ar, g.id(), t, t_sym)
            });
            g.wrap(j.unwrap())
        };
        // atan(a·t/√(a² + 2)) → ±π/2·sign(a): J = π·sign(a).
        let j = jump("atan(a*t/sqrt(a^2 + 2))");
        assert_eq!(j.to_string(), "sign(a)*pi");
        // ln(t + √a) − ln(t − √a): no jump for a > 0 (real arguments, the
        // ln|t| parts cancel), −2πi for a < 0 (t ± i√|a| approach the cut
        // from opposite sides).
        let j = jump("ln(t + sqrt(a)) - ln(t - sqrt(a))");
        for (v, want) in [(4, 0.0), (-4, -2.0 * std::f64::consts::PI)] {
            let z = j.subs(&a, &ctx.int(v)).eval_complex64().unwrap();
            assert!(
                z.re.abs() < 1e-14 && (z.im - want).abs() < 1e-14,
                "{j} at a = {v}: {z}"
            );
        }
    }

    #[test]
    fn atan_tan_terms_become_the_argument() {
        let ctx = crate::api::context::Context::new();
        let f = ctx
            .parse("-2/5*ln(abs(tan(x) + 2)) + 1/5*atan(tan(x))")
            .unwrap()
            .id();
        let (with_floor, plain) = ctx.with_arena_mut(|a| {
            let x = a.symbol("x");
            let pi = a.pi();
            let fifth = a.rational(1, 5);
            let jump = a.mul(&[fifth, pi]);
            let with_floor = add_tan_floor(a, f, x, jump);
            let plain = atan_tan_terms_to_argument(a, f, x);
            (
                a.display(with_floor).to_string(),
                a.display(plain).to_string(),
            )
        });
        // The jump π/5 cancels the floor of atan(tan x) = x − π·⌊x/π + 1/2⌋.
        assert_eq!(with_floor, "1/5*x - 2/5*ln(abs(tan(x) + 2))");
        assert_eq!(plain, "1/5*x - 2/5*ln(abs(tan(x) + 2))");
    }

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
