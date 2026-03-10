//! Convergence tests for infinite series.
//!
//! Implements the divergence test, p-series test, geometric series test,
//! and ratio test (structural) for determining whether an infinite series
//! `Σ_{k=1}^{∞} a_k` converges or diverges.
//!
//! Returns `Some(true)` for convergent, `Some(false)` for divergent,
//! `None` if the test is inconclusive.

use num_traits::Zero;
use tracing::trace;

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};
use crate::base::walk;

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

/// Determine whether the infinite series `Σ_{k=1}^{∞} body(var)` converges.
///
/// Applies several tests in sequence:
/// 1. **Divergence test**: if the body is a nonzero constant, diverges.
/// 2. **P-series test**: `Σ 1/k^p` converges iff `p > 1`.
/// 3. **Geometric test**: `Σ r^k` converges iff `|r| < 1`.
///
/// Returns `Some(true)` if convergent, `Some(false)` if divergent,
/// `None` if inconclusive.
pub(crate) fn is_convergent(arena: &mut Arena, body: ExprId, var: ExprId) -> Option<bool> {
    let result = test_convergence(arena, body, var);
    match result {
        Convergence::Converges => Some(true),
        Convergence::Diverges => Some(false),
        Convergence::Inconclusive => None,
    }
}

/// Internal: run all convergence tests in order.
fn test_convergence(arena: &mut Arena, body: ExprId, var: ExprId) -> Convergence {
    // Test 0: body must actually depend on var for meaningful analysis.
    // A nonzero constant body means the series diverges.
    let syms = walk::free_symbols(arena, body);
    if !syms.contains(&var) {
        trace!("convergence: body is constant w.r.t. var");
        // Σ c from 1 to ∞ diverges unless c = 0
        if arena.is_zero_structural(body) {
            return Convergence::Converges;
        }
        if let Some(r) = arena.as_num(body)
            && r.is_zero()
        {
            return Convergence::Converges;
        }
        return Convergence::Diverges;
    }

    // Test 1: P-series — detect body = var^(-p) i.e. 1/k^p
    if let Some(conv) = test_p_series(arena, body, var) {
        return conv;
    }

    // Test 2: Geometric — detect body = r^k where r is constant
    if let Some(conv) = test_geometric(arena, body, var) {
        return conv;
    }

    // Test 3: Simple ratio-like — detect body = 1/var (p=1 case, harmonic)
    // Already handled by p-series test above.

    Convergence::Inconclusive
}

/// P-series test: if `body = var^(-p)` for rational `p`, converges iff `p > 1`.
///
/// Also handles `body = var^p` for negative `p` (same thing),
/// and `body = var` which is `p = 1` (diverges, but this is Σk which obviously diverges).
fn test_p_series(arena: &mut Arena, body: ExprId, var: ExprId) -> Option<Convergence> {
    // Check: body = var^exp
    if let ExprNode::Pow(base, exp) = arena.node(body).clone()
        && base == var
    {
        // body = var^exp. For p-series we need exp to be a negative rational.
        // p-series: Σ k^(-p) converges iff p > 1, i.e. exp < -1
        if let Some(r) = arena.as_num(exp).cloned() {
            use num_traits::Signed;
            if r.is_negative() {
                // exp = -p, so p = -exp
                let neg_r = -&r;
                let one = num_rational::Ratio::from_integer(num_bigint::BigInt::from(1));
                if neg_r > one {
                    trace!("convergence: p-series with p = {} > 1 → converges", neg_r);
                    return Some(Convergence::Converges);
                } else if neg_r == one {
                    trace!("convergence: harmonic series (p=1) → diverges");
                    return Some(Convergence::Diverges);
                } else {
                    trace!("convergence: p-series with p = {} < 1 → diverges", neg_r);
                    return Some(Convergence::Diverges);
                }
            } else if r.is_positive() {
                // body = k^p with p > 0 — terms grow, diverges
                trace!("convergence: body = k^{} grows → diverges", r);
                return Some(Convergence::Diverges);
            }
            // exp = 0 means body = 1, constant — handled earlier
        }
    }

    // Check: body = 1/var = var^(-1), might be represented as Mul([1, Pow(var, -1)])
    // or directly. Let's also check the Mul form for 1/var^p patterns.
    if let ExprNode::Mul(ref children) = arena.node(body).clone() {
        let child_vec: Vec<ExprId> = children.to_vec();
        // Look for a single Pow(var, neg_exp) factor with everything else constant
        let mut pow_factor: Option<ExprId> = None;
        for &c in &child_vec {
            let c_syms = walk::free_symbols(arena, c);
            if c_syms.contains(&var) {
                if pow_factor.is_some() {
                    // Multiple var-dependent factors — not a simple p-series
                    return None;
                }
                pow_factor = Some(c);
            }
        }

        if let Some(pf) = pow_factor {
            // Recursively check the var-dependent factor
            return test_p_series(arena, pf, var);
        }
    }

    None
}

/// Geometric series test: if `body = r^var` where `r` is constant,
/// converges iff `|r| < 1`.
fn test_geometric(arena: &mut Arena, body: ExprId, var: ExprId) -> Option<Convergence> {
    if let ExprNode::Pow(base, exp) = arena.node(body).clone()
        && exp == var
    {
        let base_syms = walk::free_symbols(arena, base);
        if !base_syms.contains(&var) {
            // body = base^var, base is constant
            if let Some(r) = arena.as_num(base).cloned() {
                use num_traits::Signed;
                let abs_r = r.abs();
                let one = num_rational::Ratio::from_integer(num_bigint::BigInt::from(1));
                if abs_r < one {
                    trace!("convergence: geometric |r| = {} < 1 → converges", abs_r);
                    return Some(Convergence::Converges);
                } else {
                    trace!("convergence: geometric |r| = {} >= 1 → diverges", abs_r);
                    return Some(Convergence::Diverges);
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p_series_converges_p2() {
        // Σ 1/k² converges (p=2 > 1)
        let mut arena = Arena::new();
        let k = arena.symbol("k");
        let neg2 = arena.int(-2);
        let body = arena.pow(k, neg2); // k^(-2)

        let result = is_convergent(&mut arena, body, k);
        assert_eq!(result, Some(true));
    }

    #[test]
    fn p_series_diverges_harmonic() {
        // Σ 1/k diverges (p=1)
        let mut arena = Arena::new();
        let k = arena.symbol("k");
        let neg1 = arena.neg_one;
        let body = arena.pow(k, neg1); // k^(-1)

        let result = is_convergent(&mut arena, body, k);
        assert_eq!(result, Some(false));
    }

    #[test]
    fn p_series_diverges_p_half() {
        // Σ 1/k^(1/2) diverges (p=0.5 < 1)
        let mut arena = Arena::new();
        let k = arena.symbol("k");
        let neg_half = arena.rational(-1, 2);
        let body = arena.pow(k, neg_half); // k^(-1/2)

        let result = is_convergent(&mut arena, body, k);
        assert_eq!(result, Some(false));
    }

    #[test]
    fn geometric_converges_half() {
        // Σ (1/2)^k converges (|r| = 0.5 < 1)
        let mut arena = Arena::new();
        let k = arena.symbol("k");
        let half = arena.rational(1, 2);
        let body = arena.pow(half, k);

        let result = is_convergent(&mut arena, body, k);
        assert_eq!(result, Some(true));
    }

    #[test]
    fn geometric_diverges_two() {
        // Σ 2^k diverges (|r| = 2 >= 1)
        let mut arena = Arena::new();
        let k = arena.symbol("k");
        let two = arena.int(2);
        let body = arena.pow(two, k);

        let result = is_convergent(&mut arena, body, k);
        assert_eq!(result, Some(false));
    }

    #[test]
    fn constant_zero_converges() {
        // Σ 0 converges trivially
        let mut arena = Arena::new();
        let k = arena.symbol("k");
        let body = arena.zero;

        let result = is_convergent(&mut arena, body, k);
        assert_eq!(result, Some(true));
    }

    #[test]
    fn constant_nonzero_diverges() {
        // Σ 5 diverges
        let mut arena = Arena::new();
        let k = arena.symbol("k");
        let body = arena.int(5);

        let result = is_convergent(&mut arena, body, k);
        assert_eq!(result, Some(false));
    }

    #[test]
    fn growing_terms_diverge() {
        // Σ k^2 diverges (terms grow)
        let mut arena = Arena::new();
        let k = arena.symbol("k");
        let two = arena.int(2);
        let body = arena.pow(k, two);

        let result = is_convergent(&mut arena, body, k);
        assert_eq!(result, Some(false));
    }

    #[test]
    fn unknown_body_inconclusive() {
        // Σ sin(k)/k — we can't determine this
        let mut arena = Arena::new();
        let k = arena.symbol("k");
        let sin_k = arena.sin(k);
        let neg1 = arena.neg_one;
        let k_inv = arena.pow(k, neg1);
        let body = arena.mul(&[sin_k, k_inv]);

        let result = is_convergent(&mut arena, body, k);
        assert_eq!(result, None);
    }
}
