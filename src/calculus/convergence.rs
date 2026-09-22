//! Convergence tests for infinite series `Σ_{k≥k₀} a_k`.
//!
//! Every test returns `Some(true)` (converges), `Some(false)` (diverges)
//! or `None` (inconclusive) — never a guess.
//!
//! # Tests
//!
//! 1. **Constant terms** — `Σ c` diverges unless `c = 0`.
//! 2. **Rational functions** `N(k)/D(k)` — converges iff
//!    `deg D − deg N ≥ 2` (covers p-series with integer `p`, `k/(k+1)`, …).
//! 3. **Exact growth analysis** for hypergeometric-type terms
//!    `c · (−1)^k · rᵏ · Π(k+β)^p · Π((αk+β)!)^e · Π(αk+β)^(ck+d)`: Stirling's
//!    formula gives `ln|a_k| = A·k ln k + B·k + C·ln k + O(1)` with *exact*
//!    rational `A`, `C` and `B ∈ ℚ + Σ ℚ·ln p` (primes `p`).  The sign of
//!    `A`, then `B`, then `C` decides — this subsumes the ratio test, the
//!    root test, the p-series test and the alternating-series test (for
//!    `A = B = 0` the terms are eventually monotone, so `Σ(−1)^k a_k`
//!    converges iff `C < 0`).
//! 4. **Bounded factors** — `sin(·)`, `cos(·)` are stripped and the rest is
//!    tested for absolute convergence (direct comparison `|a_k| ≤ C/k^p`).
//! 5. **Integral test** for log-exp ("Hardy field") terms such as
//!    `1/(k ln² k)`: these are eventually monotone of constant sign, so
//!    `Σ a_k` converges iff `∫^∞ a(x) dx` does.  The antiderivative's limit
//!    is cross-checked numerically before it is trusted.
//!
//! [`is_absolutely_convergent`] runs the same machinery on `|a_k|`
//! (sign-alternating factors removed).

use std::collections::BTreeMap;

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Signed, ToPrimitive, Zero};
use tracing::trace;

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::Q;
use crate::base::walk;
use crate::calculus::summation::{self, TermShape};
use crate::poly::polybridge;
use crate::transforms::eval;

/// Result of a convergence test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Convergence {
    /// The series converges.
    Converges,
    /// The series diverges.
    Diverges,
    /// The test was inconclusive.
    Inconclusive,
}

impl Convergence {
    fn to_option(self) -> Option<bool> {
        match self {
            Convergence::Converges => Some(true),
            Convergence::Diverges => Some(false),
            Convergence::Inconclusive => None,
        }
    }
}

/// Determine whether the infinite series `Σ_{k≥1} body(var)` converges.
///
/// Returns `Some(true)` if convergent (absolutely or conditionally),
/// `Some(false)` if divergent, `None` if inconclusive.
pub(crate) fn is_convergent(arena: &mut Arena, body: ExprId, var: ExprId) -> Option<bool> {
    test_convergence(arena, body, var, false).to_option()
}

/// Determine whether `Σ_{k≥1} |body(var)|` converges.
pub(crate) fn is_absolutely_convergent(
    arena: &mut Arena,
    body: ExprId,
    var: ExprId,
) -> Option<bool> {
    test_convergence(arena, body, var, true).to_option()
}

fn test_convergence(arena: &mut Arena, body: ExprId, var: ExprId, absolute: bool) -> Convergence {
    if !matches!(arena.node(var), ExprNode::Symbol(_)) {
        return Convergence::Inconclusive;
    }
    // 0. Constant terms.
    if !walk::contains(arena, body, var) {
        let v = eval::eval(arena, body);
        trace!("convergence: body is constant w.r.t. var");
        return if arena.is_zero_structural(v) || arena.as_num(v).is_some_and(|r| r.is_zero()) {
            Convergence::Converges
        } else {
            Convergence::Diverges
        };
    }

    // 1. Rational functions (after combining fractions).
    if let Some(c) = rational_test(arena, body, var) {
        return c;
    }

    // 2. Sums: term-wise.
    if let ExprNode::Add(ref terms) = arena.node(body).clone() {
        let terms: Vec<ExprId> = terms.to_vec();
        let mut convergent = 0usize;
        let mut divergent = 0usize;
        for &t in &terms {
            match test_convergence(arena, t, var, absolute) {
                Convergence::Converges => convergent += 1,
                Convergence::Diverges => divergent += 1,
                Convergence::Inconclusive => return Convergence::Inconclusive,
            }
        }
        if divergent == 0 {
            return Convergence::Converges;
        }
        if divergent == 1 && convergent == terms.len() - 1 {
            return Convergence::Diverges;
        }
        return Convergence::Inconclusive;
    }

    // 3. Exact growth analysis.
    if let Some(g) = growth_exponents(arena, body, var) {
        return g.decide(absolute);
    }

    // 4. Bounded factors → absolute comparison on the rest.
    if let Some(rest) = strip_bounded_factors(arena, body, var)
        && test_convergence(arena, rest, var, true) == Convergence::Converges
    {
        return Convergence::Converges;
    }

    // 5. Bertrand series  c · k^a · ln(k)^b.
    if let Some(c) = bertrand_test(arena, body, var) {
        return c;
    }

    // 6. Integral test for log-exp terms.
    if let Some(c) = integral_test(arena, body, var) {
        return c;
    }

    Convergence::Inconclusive
}

// ═══════════════════════════════════════════════════════════════════════════
// Bertrand series
// ═══════════════════════════════════════════════════════════════════════════

/// `Σ c · (αk+β)^a · ln(α'k+β')^b` converges iff `a < −1`, or `a = −1` and `b < −1`.
fn bertrand_test(arena: &mut Arena, body: ExprId, var: ExprId) -> Option<Convergence> {
    let factors: Vec<ExprId> = match arena.node(body) {
        ExprNode::Mul(ch) => ch.to_vec(),
        _ => vec![body],
    };
    let mut a = Q::zero();
    let mut b = Q::zero();
    let mut saw_log = false;
    for f in factors {
        if !walk::contains(arena, f, var) {
            continue;
        }
        let (base, exp) = arena.as_base_exp(f);
        let e = arena.as_num(exp).cloned()?;
        match arena.node(base).clone() {
            ExprNode::Ln(arg) => {
                let (alpha, _) = linear_in(arena, arg, var)?;
                if !alpha.is_positive() {
                    return None;
                }
                saw_log = true;
                b += e;
            }
            _ => {
                let (alpha, _) = linear_in(arena, base, var)?;
                if !alpha.is_positive() {
                    return None;
                }
                a += e;
            }
        }
    }
    if !saw_log {
        return None; // pure powers are handled by the growth analysis
    }
    let minus_one = -Q::one();
    trace!("convergence: Bertrand series with a = {a}, b = {b}");
    Some(if a < minus_one || (a == minus_one && b < minus_one) {
        Convergence::Converges
    } else {
        Convergence::Diverges
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Rational functions
// ═══════════════════════════════════════════════════════════════════════════

fn rational_test(arena: &mut Arena, body: ExprId, var: ExprId) -> Option<Convergence> {
    let combined = if matches!(arena.node(body), ExprNode::Add(_)) {
        let t = polybridge::together(arena, body);
        eval::eval(arena, t)
    } else {
        body
    };
    let (n, d) = polybridge::as_numer_denom(arena, combined);
    let np = polybridge::expr_to_poly(arena, n, var)?;
    let dp = polybridge::expr_to_poly(arena, d, var)?;
    if dp.is_zero() {
        return None;
    }
    if np.is_zero() {
        return Some(Convergence::Converges);
    }
    let dn = np.degree()? as i64;
    let dd = dp.degree()? as i64;
    trace!("convergence: rational function, deg N = {dn}, deg D = {dd}");
    Some(if dd - dn >= 2 {
        Convergence::Converges
    } else {
        Convergence::Diverges
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Exact growth analysis
// ═══════════════════════════════════════════════════════════════════════════

/// `ln|a_k| = a·k ln k + b·k + c·ln k + O(1)` with `b = b_rat + Σ b_logs[p]·ln p`.
#[derive(Debug, Clone)]
pub(crate) struct Growth {
    a: Q,
    b_rat: Q,
    b_logs: BTreeMap<u64, Q>,
    /// Contribution `Σ a·ln|base|` of constant, non-rational bases (e.g. `π^k`).
    b_numeric: f64,
    b_has_numeric: bool,
    /// `b` contains `ln|x|` for a symbolic base `x` (sign unknown).
    b_symbolic: bool,
    c: Q,
    alternating: bool,
}

impl Growth {
    /// Sign of the exponential coefficient `b`: `Some(true)` positive,
    /// `Some(false)` negative, `None` zero or unknown.
    fn b_sign(&self) -> Option<Option<bool>> {
        if self.b_symbolic {
            return None;
        }
        let logs_zero = self.b_logs.values().all(|q| q.is_zero());
        if !self.b_has_numeric {
            if logs_zero {
                if self.b_rat.is_zero() {
                    return Some(None);
                }
                return Some(Some(self.b_rat.is_positive()));
            }
            // {1} ∪ {ln p : p prime} is ℚ-linearly independent (Lindemann–
            // Weierstrass + unique factorisation), so b ≠ 0 here and its sign
            // is safely decided numerically.
            let mut v = self.b_rat.to_f64().unwrap_or(0.0);
            for (p, q) in &self.b_logs {
                v += q.to_f64().unwrap_or(0.0) * (*p as f64).ln();
            }
            return Some(Some(v > 0.0));
        }
        // A transcendental constant base is involved: decide numerically
        // only when clearly away from zero.
        let mut v = self.b_rat.to_f64().unwrap_or(0.0) + self.b_numeric;
        for (p, q) in &self.b_logs {
            v += q.to_f64().unwrap_or(0.0) * (*p as f64).ln();
        }
        if v.abs() < 1e-9 {
            return None;
        }
        Some(Some(v > 0.0))
    }

    /// Does `|a_k| → 0`?  `Some(false)` when `|a_k| → ∞`, `None` when the
    /// terms stay bounded away from zero or the sign of `b` is unknown.
    pub(crate) fn tends_to_zero(&self) -> Option<bool> {
        if self.a.is_negative() {
            return Some(true);
        }
        if self.a.is_positive() {
            return Some(false);
        }
        match self.b_sign()? {
            Some(false) => Some(true),
            Some(true) => Some(false),
            None => {
                if self.c.is_negative() {
                    Some(true)
                } else if self.c.is_positive() {
                    Some(false)
                } else {
                    None
                }
            }
        }
    }

    fn decide(&self, absolute: bool) -> Convergence {
        if self.a.is_positive() {
            return Convergence::Diverges;
        }
        if self.a.is_negative() {
            return Convergence::Converges;
        }
        match self.b_sign() {
            None => return Convergence::Inconclusive,
            Some(Some(true)) => return Convergence::Diverges,
            Some(Some(false)) => return Convergence::Converges,
            Some(None) => {}
        }
        // Purely algebraic decay/growth k^c.
        let minus_one = -Q::one();
        if self.alternating && !absolute {
            // Alternating-series test: terms are eventually monotone, so
            // convergence ⇔ |a_k| → 0 ⇔ c < 0.
            if self.c.is_negative() {
                Convergence::Converges
            } else {
                Convergence::Diverges
            }
        } else if self.c < minus_one {
            Convergence::Converges
        } else {
            Convergence::Diverges
        }
    }
}

/// Add `q·ln|r|` for a rational `r ≠ 0` to the exact log-sum.
fn add_log_rational(logs: &mut BTreeMap<u64, Q>, r: &Q, q: &Q) -> Option<()> {
    if r.is_zero() {
        return None;
    }
    let numer = r.numer().abs().to_u64()?;
    let denom = r.denom().abs().to_u64()?;
    for (n, sign) in [(numer, 1i64), (denom, -1i64)] {
        for (p, e) in factor_small(n) {
            let entry = logs.entry(p).or_insert_with(Q::zero);
            *entry += q * Ratio::from_integer(BigInt::from(sign * e as i64));
        }
    }
    Some(())
}

/// Trial-division prime factorisation for the small integers that appear
/// as coefficients.
fn factor_small(mut n: u64) -> Vec<(u64, u32)> {
    let mut out = Vec::new();
    if n < 2 {
        return out;
    }
    let mut p = 2u64;
    while p * p <= n {
        let mut e = 0u32;
        while n.is_multiple_of(p) {
            n /= p;
            e += 1;
        }
        if e > 0 {
            out.push((p, e));
        }
        p += if p == 2 { 1 } else { 2 };
    }
    if n > 1 {
        out.push((n, 1));
    }
    out
}

/// Compute the growth exponents of `body`, or `None` if a factor is not of a
/// recognised shape.
pub(crate) fn growth_exponents(arena: &mut Arena, body: ExprId, var: ExprId) -> Option<Growth> {
    let factors: Vec<ExprId> = match arena.node(body) {
        ExprNode::Mul(ch) => ch.to_vec(),
        _ => vec![body],
    };
    // Split off (αk+β)^(ck+d) factors, which `term_shape` does not model.
    let mut pow_pows: Vec<(Q, Q, Q, Q)> = Vec::new();
    let mut rest: Vec<ExprId> = Vec::new();
    for f in factors {
        // Flatten (b^p)^q with integer q (e.g. (k^k)^(-1) → k^(-k)).
        let f = if let ExprNode::Pow(base, exp) = arena.node(f).clone()
            && let ExprNode::Pow(b2, e2) = arena.node(base).clone()
            && arena.as_num(exp).is_some_and(|q| q.is_integer())
        {
            let pq = arena.mul(&[e2, exp]);
            let pq = eval::eval(arena, pq);
            arena.pow(b2, pq)
        } else {
            f
        };
        if let ExprNode::Pow(base, exp) = arena.node(f).clone()
            && walk::contains(arena, base, var)
            && walk::contains(arena, exp, var)
        {
            let (alpha, beta) = linear_in(arena, base, var)?;
            let (c, d) = linear_in(arena, exp, var)?;
            if !alpha.is_positive() {
                return None;
            }
            pow_pows.push((alpha, beta, c, d));
        } else {
            rest.push(f);
        }
    }
    let rest_expr = match rest.len() {
        0 => arena.one,
        1 => rest[0],
        _ => arena.mul(&rest),
    };
    let shape: TermShape = summation::term_shape(arena, rest_expr, var)?;
    if !shape.binomials.is_empty() {
        return None;
    }
    let mut g = Growth {
        a: Q::zero(),
        b_rat: Q::zero(),
        b_logs: BTreeMap::new(),
        b_numeric: 0.0,
        b_has_numeric: false,
        b_symbolic: false,
        c: Q::zero(),
        alternating: shape.alternating,
    };
    // Algebraic part.
    for (_, p) in &shape.lin_pows {
        g.c += p;
    }
    // Geometric part: r^k → b += ln|r|.
    if !shape.numeric_base.is_one() {
        if shape.numeric_base.is_negative() {
            g.alternating = !g.alternating;
        }
        add_log_rational(&mut g.b_logs, &shape.numeric_base, &Q::one())?;
    }
    for (base, a) in &shape.bases {
        if *base == arena.e_const {
            g.b_rat += a;
        } else if let Some(v) = numeric_abs(arena, *base) {
            // |base| is a known constant but not rational (e.g. π): the
            // contribution a·ln|base| can only be assessed numerically.
            if v <= 0.0 || !v.is_finite() {
                return None;
            }
            g.b_numeric += a.to_f64()? * v.ln();
            g.b_has_numeric = true;
        } else {
            g.b_symbolic = true;
        }
    }
    // Factorials: ((αk+β)!)^e → a += eα, b += e(α ln α − α), c += e(β + 1/2).
    let half = Q::new(BigInt::one(), BigInt::from(2));
    for (alpha, beta, e) in &shape.facts {
        if !alpha.is_positive() {
            return None;
        }
        let er = Ratio::from_integer(BigInt::from(*e));
        g.a += &er * alpha;
        g.b_rat -= &er * alpha;
        add_log_rational(&mut g.b_logs, alpha, &(&er * alpha))?;
        g.c += &er * (beta + &half);
    }
    // (αk+β)^(ck+d) → a += c, b += c ln α, c += d.
    for (alpha, _beta, c, d) in &pow_pows {
        g.a += c;
        add_log_rational(&mut g.b_logs, alpha, c)?;
        g.c += d;
    }
    trace!(?g, "convergence: growth exponents");
    Some(g)
}

fn numeric_abs(arena: &mut Arena, e: ExprId) -> Option<f64> {
    if !walk::free_symbols(arena, e).is_empty() {
        return None;
    }
    crate::transforms::evalf::eval_const_f64(arena, e).map(f64::abs)
}

fn linear_in(arena: &Arena, e: ExprId, var: ExprId) -> Option<(Q, Q)> {
    let p = polybridge::expr_to_poly(arena, e, var)?;
    match p.degree() {
        Some(1) => Some((p.coeff(1), p.coeff(0))),
        Some(0) => Some((Q::zero(), p.coeff(0))),
        _ => None,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Bounded factors
// ═══════════════════════════════════════════════════════════════════════════

/// Remove `sin(·)` / `cos(·)` factors (bounded by 1 for real arguments).
/// Returns `None` if there was nothing to strip.
fn strip_bounded_factors(arena: &mut Arena, body: ExprId, var: ExprId) -> Option<ExprId> {
    let ExprNode::Mul(ref factors) = arena.node(body).clone() else {
        return None;
    };
    let factors: Vec<ExprId> = factors.to_vec();
    let mut kept = Vec::new();
    let mut stripped = false;
    for f in factors {
        let bounded = match arena.node(f).clone() {
            ExprNode::Sin(_) | ExprNode::Cos(_) => true,
            ExprNode::Pow(base, exp) => {
                matches!(arena.node(base), ExprNode::Sin(_) | ExprNode::Cos(_))
                    && arena.as_num(exp).is_some_and(|r| r.is_positive())
            }
            _ => false,
        };
        if bounded && walk::contains(arena, f, var) {
            stripped = true;
        } else {
            kept.push(f);
        }
    }
    if !stripped {
        return None;
    }
    Some(match kept.len() {
        0 => arena.one,
        1 => kept[0],
        _ => arena.mul(&kept),
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Integral test
// ═══════════════════════════════════════════════════════════════════════════

/// Is `e` built only from `var`, numbers, `+`, `×`, rational powers, `ln`
/// and `exp`?  Such functions lie in a Hardy field: they are eventually
/// monotone and of constant sign.
fn is_log_exp(arena: &Arena, e: ExprId, var: ExprId) -> bool {
    let mut stack = vec![e];
    let mut has_log = false;
    while let Some(id) = stack.pop() {
        match arena.node(id) {
            ExprNode::Num(_) => {}
            ExprNode::Symbol(_) => {
                if id != var {
                    return false;
                }
            }
            ExprNode::Add(ch) | ExprNode::Mul(ch) => stack.extend(ch.iter().copied()),
            ExprNode::Pow(b, x) => {
                if arena.as_num(*x).is_none() {
                    return false;
                }
                stack.push(*b);
            }
            ExprNode::Ln(a) => {
                has_log = true;
                stack.push(*a);
            }
            ExprNode::Exp(a) => stack.push(*a),
            ExprNode::Neg(a) => stack.push(*a),
            _ => return false,
        }
    }
    has_log
}

fn integral_test(arena: &mut Arena, body: ExprId, var: ExprId) -> Option<Convergence> {
    if !is_log_exp(arena, body, var) {
        return None;
    }
    let anti = crate::transforms::integrate::integrate(arena, body, var);
    if walk::has_unevaluated(arena, anti) {
        return None;
    }
    let inf = arena.infinity;
    let lim = crate::calculus::limit::limit(arena, anti, var, inf).ok()?;
    if lim == arena.infinity || lim == arena.neg_infinity {
        // Cross-check: |F(N)| must grow.
        let f1 = value_at(arena, anti, var, 1_000)?;
        let f2 = value_at(arena, anti, var, 1_000_000_000)?;
        if f2.abs() > f1.abs() {
            return Some(Convergence::Diverges);
        }
        return None;
    }
    if walk::has_unevaluated(arena, lim) || !walk::free_symbols(arena, lim).is_empty() {
        return None;
    }
    let l = crate::transforms::evalf::eval_const_f64(arena, lim)?;
    if !l.is_finite() {
        return None;
    }
    // Cross-check the claimed limit numerically.
    let f1 = value_at(arena, anti, var, 1_000)?;
    let f2 = value_at(arena, anti, var, 1_000_000_000)?;
    let d1 = (f1 - l).abs();
    let d2 = (f2 - l).abs();
    if d2 <= d1 + 1e-12 && d2 < 0.5 * f64::max(1.0, l.abs()) {
        trace!("convergence: integral test → converges (limit {l})");
        Some(Convergence::Converges)
    } else {
        None
    }
}

fn value_at(arena: &mut Arena, e: ExprId, var: ExprId, n: i64) -> Option<f64> {
    let ne = arena.int(n);
    let s = crate::transforms::subs::subs(arena, e, var, ne);
    let v = crate::transforms::evalf::eval_const_f64(arena, s)?;
    if v.is_finite() { Some(v) } else { None }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (Arena, ExprId) {
        let mut arena = Arena::new();
        let k = arena.symbol("k");
        (arena, k)
    }

    #[test]
    fn p_series_converges_p2() {
        let (mut arena, k) = setup();
        let neg2 = arena.int(-2);
        let body = arena.pow(k, neg2);
        assert_eq!(is_convergent(&mut arena, body, k), Some(true));
        assert_eq!(is_absolutely_convergent(&mut arena, body, k), Some(true));
    }

    #[test]
    fn p_series_diverges_harmonic() {
        let (mut arena, k) = setup();
        let neg1 = arena.neg_one;
        let body = arena.pow(k, neg1);
        assert_eq!(is_convergent(&mut arena, body, k), Some(false));
    }

    #[test]
    fn p_series_diverges_p_half() {
        let (mut arena, k) = setup();
        let neg_half = arena.rational(-1, 2);
        let body = arena.pow(k, neg_half);
        assert_eq!(is_convergent(&mut arena, body, k), Some(false));
    }

    #[test]
    fn geometric_converges_half() {
        let (mut arena, k) = setup();
        let half = arena.rational(1, 2);
        let body = arena.pow(half, k);
        assert_eq!(is_convergent(&mut arena, body, k), Some(true));
    }

    #[test]
    fn geometric_diverges_two() {
        let (mut arena, k) = setup();
        let two = arena.int(2);
        let body = arena.pow(two, k);
        assert_eq!(is_convergent(&mut arena, body, k), Some(false));
    }

    #[test]
    fn constant_zero_converges() {
        let (mut arena, k) = setup();
        let body = arena.zero;
        assert_eq!(is_convergent(&mut arena, body, k), Some(true));
    }

    #[test]
    fn constant_nonzero_diverges() {
        let (mut arena, k) = setup();
        let body = arena.int(5);
        assert_eq!(is_convergent(&mut arena, body, k), Some(false));
    }

    #[test]
    fn growing_terms_diverge() {
        let (mut arena, k) = setup();
        let two = arena.int(2);
        let body = arena.pow(k, two);
        assert_eq!(is_convergent(&mut arena, body, k), Some(false));
    }

    #[test]
    fn sin_over_k_inconclusive_but_sin_over_k_squared_converges() {
        let (mut arena, k) = setup();
        let sin_k = arena.sin(k);
        let neg1 = arena.neg_one;
        let k_inv = arena.pow(k, neg1);
        let body = arena.mul(&[sin_k, k_inv]);
        assert_eq!(is_convergent(&mut arena, body, k), None);
        let neg2 = arena.int(-2);
        let k_inv2 = arena.pow(k, neg2);
        let body = arena.mul(&[sin_k, k_inv2]);
        assert_eq!(is_convergent(&mut arena, body, k), Some(true));
    }

    #[test]
    fn alternating_harmonic_conditionally_convergent() {
        let (mut arena, k) = setup();
        let m1 = arena.neg_one;
        let sgn = arena.pow(m1, k);
        let body = arena.div(sgn, k);
        assert_eq!(is_convergent(&mut arena, body, k), Some(true));
        assert_eq!(is_absolutely_convergent(&mut arena, body, k), Some(false));
        // (−1)^k alone diverges
        assert_eq!(is_convergent(&mut arena, sgn, k), Some(false));
    }

    #[test]
    fn factorial_ratio_tests() {
        let (mut arena, k) = setup();
        let kf = arena.factorial(k);
        // k!/k^k converges
        let kk = arena.pow(k, k);
        let body = arena.div(kf, kk);
        assert_eq!(is_convergent(&mut arena, body, k), Some(true));
        // 2^k/k! converges
        let two = arena.int(2);
        let tk = arena.pow(two, k);
        let body = arena.div(tk, kf);
        assert_eq!(is_convergent(&mut arena, body, k), Some(true));
        // k!/2^k diverges
        let body = arena.div(kf, tk);
        assert_eq!(is_convergent(&mut arena, body, k), Some(false));
        // k^10/2^k converges
        let ten = arena.int(10);
        let k10 = arena.pow(k, ten);
        let body = arena.div(k10, tk);
        assert_eq!(is_convergent(&mut arena, body, k), Some(true));
        // C(2k,k)/4^k diverges (~ 1/√(πk)); C(2k,k)/5^k converges
        let two_k = arena.mul(&[two, k]);
        let c2k = arena.binomial(two_k, k);
        let four = arena.int(4);
        let fk = arena.pow(four, k);
        let body = arena.div(c2k, fk);
        assert_eq!(is_convergent(&mut arena, body, k), Some(false));
        let five = arena.int(5);
        let fk5 = arena.pow(five, k);
        let body = arena.div(c2k, fk5);
        assert_eq!(is_convergent(&mut arena, body, k), Some(true));
        // C(2k,k)/(4^k k) converges (~ k^(-3/2))
        let den = arena.mul(&[fk, k]);
        let body = arena.div(c2k, den);
        assert_eq!(is_convergent(&mut arena, body, k), Some(true));
    }

    #[test]
    fn rational_function_terms() {
        let (mut arena, k) = setup();
        let one = arena.one;
        let k1 = arena.add(&[k, one]);
        let body = arena.div(k, k1);
        assert_eq!(is_convergent(&mut arena, body, k), Some(false));
        // 1/k − 1/(k+1) converges (telescoping, ~1/k²)
        let a = arena.div(one, k);
        let b = arena.div(one, k1);
        let body = arena.sub(a, b);
        assert_eq!(is_convergent(&mut arena, body, k), Some(true));
    }

    #[test]
    fn integral_test_log_terms() {
        let (mut arena, k) = setup();
        let one = arena.one;
        let lnk = arena.ln(k);
        let two = arena.int(2);
        let ln2 = arena.pow(lnk, two);
        let den = arena.mul(&[k, ln2]);
        let body = arena.div(one, den);
        assert_eq!(is_convergent(&mut arena, body, k), Some(true));
    }

    #[test]
    fn symbolic_parameter_decided_by_factorial() {
        let (mut arena, k) = setup();
        let x = arena.symbol("x");
        let xk = arena.pow(x, k);
        let kf = arena.factorial(k);
        let body = arena.div(xk, kf);
        assert_eq!(is_convergent(&mut arena, body, k), Some(true));
        // x^k alone: depends on x
        assert_eq!(is_convergent(&mut arena, xk, k), None);
    }

    #[test]
    fn factor_small_primes() {
        assert_eq!(factor_small(360), vec![(2, 3), (3, 2), (5, 1)]);
        assert_eq!(factor_small(1), vec![]);
        assert_eq!(factor_small(97), vec![(97, 1)]);
    }
}
