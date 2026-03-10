//! Combinatorial functions: Stirling numbers, multinomial coefficients,
//! and integer partition counting.
//!
//! **Unified API** — every function accepts arbitrary-precision integers via
//! `impl Into<BigInt>`.  Returns `Option<BigInt>` where `None` means the
//! input doesn't fit in a machine-sized integer (and thus the computation
//! cannot proceed).  For inputs that do fit, the function always returns
//! the exact result — there are no artificial resource limits.
//!
//! At the CAS expression layer, `None` causes the node to remain in
//! unevaluated symbolic form (e.g. the user sees `stirling2(n, k)`),
//! which is the mathematically honest response when computation isn't
//! feasible.
//!
//! # Examples
//!
//! ```
//! use symplex::combinatorics::{stirling2, stirling1, multinomial, partition_count};
//! use num_bigint::BigInt;
//!
//! // Stirling number of the second kind: S(4, 2) = 7
//! assert_eq!(stirling2(4, 2), Some(BigInt::from(7)));
//!
//! // Stirling number of the first kind (signed): s(4, 1) = -6
//! assert_eq!(stirling1(4, 1), Some(BigInt::from(-6)));
//!
//! // Multinomial coefficient: 6! / (2! * 3! * 1!) = 60
//! assert_eq!(multinomial(6, &[2, 3, 1]), Some(BigInt::from(60)));
//!
//! // Number of integer partitions: p(5) = 7
//! assert_eq!(partition_count(5), Some(BigInt::from(7)));
//! ```

use num_bigint::BigInt;
use num_traits::{One, Signed, ToPrimitive, Zero};

// ═══════════════════════════════════════════════════════════════════════════
// Stirling numbers of the second kind: S(n, k)
// ═══════════════════════════════════════════════════════════════════════════

/// Stirling number of the second kind `S(n, k)`.
///
/// Counts the number of ways to partition a set of `n` elements into
/// exactly `k` non-empty subsets.
///
/// Uses the recurrence `S(n, k) = k·S(n−1, k) + S(n−1, k−1)` with
/// base cases `S(0, 0) = 1` and `S(n, 0) = S(0, k) = 0` for `n, k > 0`.
///
/// Returns `Some(0)` for negative inputs or when `k > n`.
/// Returns `None` if the inputs don't fit in `u64`.
///
/// # Examples
///
/// ```
/// use symplex::combinatorics::stirling2;
/// use num_bigint::BigInt;
///
/// assert_eq!(stirling2(0, 0), Some(BigInt::from(1)));
/// assert_eq!(stirling2(4, 2), Some(BigInt::from(7)));
/// assert_eq!(stirling2(5, 3), Some(BigInt::from(25)));
/// ```
pub fn stirling2(n: impl Into<BigInt>, k: impl Into<BigInt>) -> Option<BigInt> {
    let n = n.into();
    let k = k.into();
    if n.is_negative() || k.is_negative() {
        return Some(BigInt::zero());
    }
    let n: u64 = n.try_into().ok()?;
    let k: u64 = k.try_into().ok()?;
    Some(stirling2_u64(n, k))
}

fn stirling2_u64(n: u64, k: u64) -> BigInt {
    // Base cases
    if k > n {
        return BigInt::zero();
    }
    if n == 0 && k == 0 {
        return BigInt::one();
    }
    if k == 0 || n == 0 {
        return BigInt::zero();
    }

    // Fast paths (closed-form, O(1) or O(n) — work for any u64 n)
    if k == 1 || k == n {
        return BigInt::one();
    }
    if k == 2 {
        // S(n, 2) = 2^(n-1) - 1
        return (BigInt::one() << (n - 1) as usize) - BigInt::one();
    }
    if k == n - 1 {
        // S(n, n-1) = C(n, 2) = n*(n-1)/2
        return BigInt::from(n) * BigInt::from(n - 1) / BigInt::from(2);
    }

    // General case: build triangle row by row.
    // Only need two rows at a time: O(k) space, O(n*k) time.
    let n = n as usize;
    let k = k as usize;

    let mut prev = vec![BigInt::zero(); k + 1];
    prev[0] = BigInt::one(); // S(0, 0) = 1

    for i in 1..=n {
        let mut curr = vec![BigInt::zero(); k + 1];
        for j in 1..=k.min(i) {
            // S(i, j) = j * S(i-1, j) + S(i-1, j-1)
            curr[j] = BigInt::from(j as u64) * &prev[j] + &prev[j - 1];
        }
        prev = curr;
    }

    prev[k].clone()
}

// ═══════════════════════════════════════════════════════════════════════════
// Stirling numbers of the first kind: s(n, k)  (signed)
// ═══════════════════════════════════════════════════════════════════════════

/// Signed Stirling number of the first kind `s(n, k)`.
///
/// Related to the number of permutations of `n` elements with exactly
/// `k` cycles.  The unsigned Stirling number `|s(n, k)|` counts those
/// permutations; the signed version satisfies `s(n, k) = (-1)^{n-k} |s(n, k)|`.
///
/// Uses the recurrence `s(n, k) = -(n−1)·s(n−1, k) + s(n−1, k−1)` with
/// base cases `s(0, 0) = 1` and `s(n, 0) = s(0, k) = 0` for `n, k > 0`.
///
/// The signed Stirling numbers connect falling factorials to ordinary
/// powers: `x^{(n)} = Σ_k s(n, k) x^k`.
///
/// Returns `Some(0)` for negative inputs or when `k > n`.
/// Returns `None` if the inputs don't fit in `u64`.
///
/// # Examples
///
/// ```
/// use symplex::combinatorics::stirling1;
/// use num_bigint::BigInt;
///
/// assert_eq!(stirling1(0, 0), Some(BigInt::from(1)));
/// assert_eq!(stirling1(3, 1), Some(BigInt::from(2)));
/// assert_eq!(stirling1(4, 1), Some(BigInt::from(-6)));
/// ```
pub fn stirling1(n: impl Into<BigInt>, k: impl Into<BigInt>) -> Option<BigInt> {
    let n = n.into();
    let k = k.into();
    if n.is_negative() || k.is_negative() {
        return Some(BigInt::zero());
    }
    let n: u64 = n.try_into().ok()?;
    let k: u64 = k.try_into().ok()?;
    Some(stirling1_u64(n, k))
}

fn stirling1_u64(n: u64, k: u64) -> BigInt {
    // Base cases
    if k > n {
        return BigInt::zero();
    }
    if n == 0 && k == 0 {
        return BigInt::one();
    }
    if k == 0 || n == 0 {
        return BigInt::zero();
    }

    // Fast paths (closed-form — work for any u64 n)
    if k == n {
        return BigInt::one();
    }
    if k == n - 1 {
        // s(n, n-1) = -C(n, 2) = -n*(n-1)/2
        return -(BigInt::from(n) * BigInt::from(n - 1) / BigInt::from(2));
    }
    if k == 1 {
        // s(n, 1) = (-1)^{n-1} * (n-1)!
        let mut fact = BigInt::one();
        for i in 1..n {
            fact *= BigInt::from(i);
        }
        if (n - 1) % 2 == 0 {
            return fact;
        } else {
            return -fact;
        }
    }

    // General case: build triangle row by row.
    let n = n as usize;
    let k = k as usize;

    let mut prev = vec![BigInt::zero(); k + 1];
    prev[0] = BigInt::one(); // s(0, 0) = 1

    for i in 1..=n {
        let mut curr = vec![BigInt::zero(); k + 1];
        for j in 1..=k.min(i) {
            // s(i, j) = -(i-1) * s(i-1, j) + s(i-1, j-1)
            curr[j] = -BigInt::from((i - 1) as u64) * &prev[j] + &prev[j - 1];
        }
        prev = curr;
    }

    prev[k].clone()
}

// ═══════════════════════════════════════════════════════════════════════════
// Multinomial coefficient
// ═══════════════════════════════════════════════════════════════════════════

/// Multinomial coefficient `n! / (k₁! · k₂! · … · kₘ!)`.
///
/// This is the number of ways to divide `n` objects into groups of sizes
/// `k₁, k₂, …, kₘ`.
///
/// Returns `Some(0)` if the `kᵢ` don't sum to `n` or if any `kᵢ` is negative.
/// Returns `None` if `n` or any `kᵢ` doesn't fit in `u64`.
///
/// The computation avoids computing full factorials by using incremental
/// products and divisions, keeping intermediate values small.
///
/// # Examples
///
/// ```
/// use symplex::combinatorics::multinomial;
/// use num_bigint::BigInt;
///
/// // 6! / (2! * 3! * 1!) = 60
/// assert_eq!(multinomial(6, &[2, 3, 1]), Some(BigInt::from(60)));
///
/// // Reduces to binomial: C(10, 3) = 120
/// assert_eq!(multinomial(10, &[3, 7]), Some(BigInt::from(120)));
/// ```
pub fn multinomial(n: impl Into<BigInt>, ks: &[impl Into<BigInt> + Clone]) -> Option<BigInt> {
    let n = n.into();
    if n.is_negative() {
        return Some(BigInt::zero());
    }

    let ks_big: Vec<BigInt> = ks.iter().map(|k| k.clone().into()).collect();

    // Check all k_i are non-negative and sum to n.
    let mut sum = BigInt::zero();
    for k in &ks_big {
        if k.is_negative() {
            return Some(BigInt::zero());
        }
        sum += k;
    }
    if sum != n {
        return Some(BigInt::zero());
    }

    // Convert to u64 for the computation loop (consistent with
    // eval_rising_factorial, eval_falling_factorial, etc.)
    let ks_u64: Vec<u64> = ks_big
        .iter()
        .map(|k| k.try_into().ok())
        .collect::<Option<Vec<u64>>>()?;

    Some(multinomial_u64(&ks_u64))
}

fn multinomial_u64(ks: &[u64]) -> BigInt {
    // Compute n! / (k1! * k2! * ... * km!) incrementally.
    // Use the identity: multinomial(n; k1, k2, ...) = C(n, k1) * C(n-k1, k2) * ...
    // This keeps intermediate values bounded by C(n, k_i).
    let mut result = BigInt::one();
    let mut remaining: u64 = ks.iter().sum();

    for &k in ks {
        if k == 0 {
            continue;
        }
        // Incremental binomial: C(remaining, k) = ∏_{i=0}^{k-1} (remaining - i) / (i + 1)
        let mut binom = BigInt::one();
        for i in 0..k {
            binom *= BigInt::from(remaining - i);
            binom /= BigInt::from(i + 1);
        }
        result *= binom;
        remaining -= k;
    }

    result
}

// ═══════════════════════════════════════════════════════════════════════════
// Integer partition count: p(n)
// ═══════════════════════════════════════════════════════════════════════════

/// Number of integer partitions of `n`.
///
/// An integer partition of `n` is a way to write `n` as a sum of positive
/// integers (order doesn't matter).  For example, `p(5) = 7` because
/// 5 = 5 = 4+1 = 3+2 = 3+1+1 = 2+2+1 = 2+1+1+1 = 1+1+1+1+1.
///
/// Uses Euler's pentagonal number theorem recurrence:
/// `p(n) = Σ_{k≠0} (-1)^{k+1} p(n − k(3k−1)/2)`
/// with `p(0) = 1` and `p(n) = 0` for `n < 0`.
///
/// Returns `Some(0)` for negative inputs, `Some(1)` for `n = 0`.
/// Returns `None` if `n` doesn't fit in `usize`.
///
/// # Examples
///
/// ```
/// use symplex::combinatorics::partition_count;
/// use num_bigint::BigInt;
///
/// assert_eq!(partition_count(0), Some(BigInt::from(1)));
/// assert_eq!(partition_count(5), Some(BigInt::from(7)));
/// assert_eq!(partition_count(10), Some(BigInt::from(42)));
/// ```
pub fn partition_count(n: impl Into<BigInt>) -> Option<BigInt> {
    let n = n.into();
    if n.is_negative() {
        return Some(BigInt::zero());
    }
    if n.is_zero() {
        return Some(BigInt::one());
    }
    let n: usize = n.try_into().ok()?;
    Some(partition_count_usize(n))
}

fn partition_count_usize(n: usize) -> BigInt {
    // Build table bottom-up using Euler's pentagonal recurrence.
    // p(n) = Σ_{k=1,2,3,...} (-1)^{k+1} * [p(n - g1(k)) + p(n - g2(k))]
    // where g1(k) = k*(3k-1)/2 and g2(k) = k*(3k+1)/2 are the
    // generalized pentagonal numbers.
    let mut table = vec![BigInt::zero(); n + 1];
    table[0] = BigInt::one();

    for i in 1..=n {
        // Use i128 for pentagonal number arithmetic to avoid overflow.
        // k can reach ~sqrt(2n/3); for n up to usize::MAX (~1.8e19),
        // k*(3k+1) can exceed i64::MAX but fits comfortably in i128.
        let mut k: i128 = 1;
        loop {
            let g1 = (k * (3 * k - 1) / 2) as usize;
            let g2 = (k * (3 * k + 1) / 2) as usize;

            if g1 > i {
                break;
            }

            if k % 2 == 1 {
                // sign = +1
                let term1 = table[i - g1].clone();
                table[i] += term1;
                if g2 <= i {
                    let term2 = table[i - g2].clone();
                    table[i] += term2;
                }
            } else {
                // sign = -1
                let term1 = table[i - g1].clone();
                table[i] -= term1;
                if g2 <= i {
                    let term2 = table[i - g2].clone();
                    table[i] -= term2;
                }
            }

            k += 1;
        }
    }

    table[n].clone()
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;

    fn bi(n: i64) -> BigInt {
        BigInt::from(n)
    }

    // ── Stirling2: known values ─────────────────────────────────────

    #[test]
    fn stirling2_base_cases() {
        assert_eq!(stirling2(0, 0), Some(bi(1)));
        assert_eq!(stirling2(1, 0), Some(bi(0)));
        assert_eq!(stirling2(0, 1), Some(bi(0)));
        assert_eq!(stirling2(1, 1), Some(bi(1)));
    }

    #[test]
    fn stirling2_known_values() {
        // OEIS A008277: S(n,k) triangle
        assert_eq!(stirling2(3, 1), Some(bi(1)));
        assert_eq!(stirling2(3, 2), Some(bi(3)));
        assert_eq!(stirling2(3, 3), Some(bi(1)));
        assert_eq!(stirling2(4, 1), Some(bi(1)));
        assert_eq!(stirling2(4, 2), Some(bi(7)));
        assert_eq!(stirling2(4, 3), Some(bi(6)));
        assert_eq!(stirling2(4, 4), Some(bi(1)));
        assert_eq!(stirling2(5, 2), Some(bi(15)));
        assert_eq!(stirling2(5, 3), Some(bi(25)));
        assert_eq!(stirling2(5, 4), Some(bi(10)));
        assert_eq!(stirling2(5, 5), Some(bi(1)));
        assert_eq!(stirling2(6, 3), Some(bi(90)));
        assert_eq!(stirling2(7, 4), Some(bi(350)));
        assert_eq!(stirling2(8, 4), Some(bi(1701)));
        assert_eq!(stirling2(10, 5), Some(bi(42525)));
    }

    #[test]
    fn stirling2_k_equals_1_always_1() {
        for n in 1..=15u64 {
            assert_eq!(stirling2(n, 1u64), Some(bi(1)), "S({n}, 1) should be 1");
        }
    }

    #[test]
    fn stirling2_k_equals_n_always_1() {
        for n in 0..=15u64 {
            assert_eq!(stirling2(n, n), Some(bi(1)), "S({n}, {n}) should be 1");
        }
    }

    #[test]
    fn stirling2_k_equals_2() {
        // S(n, 2) = 2^(n-1) - 1
        for n in 2..=12u64 {
            let expected = bi(1 << (n - 1)) - bi(1);
            assert_eq!(stirling2(n, 2u64), Some(expected), "S({n}, 2)");
        }
    }

    #[test]
    fn stirling2_k_equals_n_minus_1() {
        // S(n, n-1) = C(n, 2) = n*(n-1)/2
        for n in 2..=15u64 {
            let expected = bi((n * (n - 1) / 2) as i64);
            assert_eq!(stirling2(n, n - 1), Some(expected), "S({n}, {n}-1)");
        }
    }

    #[test]
    fn stirling2_k_greater_than_n_is_zero() {
        assert_eq!(stirling2(3, 5), Some(bi(0)));
        assert_eq!(stirling2(0, 1), Some(bi(0)));
        assert_eq!(stirling2(5, 10), Some(bi(0)));
    }

    #[test]
    fn stirling2_negative_inputs() {
        assert_eq!(stirling2(-1, 0), Some(bi(0)));
        assert_eq!(stirling2(0, -1), Some(bi(0)));
        assert_eq!(stirling2(-3, -2), Some(bi(0)));
    }

    #[test]
    fn stirling2_recurrence_identity() {
        // S(n, k) = k * S(n-1, k) + S(n-1, k-1)
        for n in 2..=10u64 {
            for k in 1..=n {
                let lhs = stirling2(n, k).unwrap();
                let rhs =
                    bi(k as i64) * stirling2(n - 1, k).unwrap() + stirling2(n - 1, k - 1).unwrap();
                assert_eq!(lhs, rhs, "recurrence failed for S({n}, {k})");
            }
        }
    }

    #[test]
    fn stirling2_row_sum_equals_bell() {
        // Σ_k S(n, k) = B(n) (Bell number)
        // Known Bell numbers: B(0)=1, B(1)=1, B(2)=2, B(3)=5, B(4)=15,
        //   B(5)=52, B(6)=203, B(7)=877, B(8)=4140
        let bells = [1i64, 1, 2, 5, 15, 52, 203, 877, 4140];
        for (n, &b) in bells.iter().enumerate() {
            let mut sum = BigInt::zero();
            for k in 0..=n {
                sum += stirling2(n as u64, k as u64).unwrap();
            }
            assert_eq!(sum, bi(b), "Σ S({n}, k) should equal B({n}) = {b}");
        }
    }

    #[test]
    fn stirling2_huge_n_none() {
        // Inputs that don't fit in u64 return None
        let huge = BigInt::from(u64::MAX) + BigInt::one();
        assert_eq!(stirling2(huge, 2), None);
    }

    // ── Stirling1: known values ─────────────────────────────────────

    #[test]
    fn stirling1_base_cases() {
        assert_eq!(stirling1(0, 0), Some(bi(1)));
        assert_eq!(stirling1(1, 0), Some(bi(0)));
        assert_eq!(stirling1(0, 1), Some(bi(0)));
        assert_eq!(stirling1(1, 1), Some(bi(1)));
    }

    #[test]
    fn stirling1_known_values() {
        // OEIS A008275: signed Stirling numbers of the first kind
        assert_eq!(stirling1(2, 1), Some(bi(-1)));
        assert_eq!(stirling1(2, 2), Some(bi(1)));
        assert_eq!(stirling1(3, 1), Some(bi(2)));
        assert_eq!(stirling1(3, 2), Some(bi(-3)));
        assert_eq!(stirling1(3, 3), Some(bi(1)));
        assert_eq!(stirling1(4, 1), Some(bi(-6)));
        assert_eq!(stirling1(4, 2), Some(bi(11)));
        assert_eq!(stirling1(4, 3), Some(bi(-6)));
        assert_eq!(stirling1(4, 4), Some(bi(1)));
        assert_eq!(stirling1(5, 1), Some(bi(24)));
        assert_eq!(stirling1(5, 2), Some(bi(-50)));
        assert_eq!(stirling1(5, 3), Some(bi(35)));
        assert_eq!(stirling1(5, 4), Some(bi(-10)));
        assert_eq!(stirling1(5, 5), Some(bi(1)));
    }

    #[test]
    fn stirling1_k_equals_n_always_1() {
        for n in 0..=15u64 {
            assert_eq!(stirling1(n, n), Some(bi(1)), "s({n}, {n}) should be 1");
        }
    }

    #[test]
    fn stirling1_k_equals_n_minus_1() {
        // s(n, n-1) = -C(n, 2) = -n*(n-1)/2
        for n in 2..=15u64 {
            let expected = -bi((n * (n - 1) / 2) as i64);
            assert_eq!(stirling1(n, n - 1), Some(expected), "s({n}, {n}-1)");
        }
    }

    #[test]
    fn stirling1_k_equals_1() {
        // s(n, 1) = (-1)^{n-1} * (n-1)!
        // 0!=1, 1!=1, 2!=2, 3!=6, 4!=24, 5!=120, 6!=720, 7!=5040, 8!=40320
        let factorials = [1i64, 1, 2, 6, 24, 120, 720, 5040, 40320];
        for n in 1..=9u64 {
            let expected = if (n - 1) % 2 == 0 {
                bi(factorials[n as usize - 1])
            } else {
                bi(-factorials[n as usize - 1])
            };
            assert_eq!(stirling1(n, 1u64), Some(expected), "s({n}, 1)");
        }
    }

    #[test]
    fn stirling1_k_greater_than_n_is_zero() {
        assert_eq!(stirling1(3, 5), Some(bi(0)));
        assert_eq!(stirling1(0, 1), Some(bi(0)));
    }

    #[test]
    fn stirling1_negative_inputs() {
        assert_eq!(stirling1(-1, 0), Some(bi(0)));
        assert_eq!(stirling1(0, -1), Some(bi(0)));
    }

    #[test]
    fn stirling1_recurrence_identity() {
        // s(n, k) = -(n-1) * s(n-1, k) + s(n-1, k-1)
        for n in 2..=10u64 {
            for k in 1..=n {
                let lhs = stirling1(n, k).unwrap();
                let rhs = -bi((n - 1) as i64) * stirling1(n - 1, k).unwrap()
                    + stirling1(n - 1, k - 1).unwrap();
                assert_eq!(lhs, rhs, "recurrence failed for s({n}, {k})");
            }
        }
    }

    #[test]
    fn stirling1_row_sum_is_zero_for_n_ge_2() {
        // x^{(n)} evaluated at x=1: 1*0*(-1)*... = 0 for n >= 2.
        // So Σ_k s(n, k) * 1^k = 0 for n >= 2.
        for n in 2..=10u64 {
            let mut sum = BigInt::zero();
            for k in 0..=n {
                sum += stirling1(n, k).unwrap();
            }
            assert_eq!(sum, bi(0), "Σ s({n}, k) should be 0");
        }
    }

    #[test]
    fn stirling1_unsigned_row_sum_is_n_factorial() {
        // Σ_k |s(n, k)| = n!
        let factorials = [1i64, 1, 2, 6, 24, 120, 720, 5040, 40320];
        for n in 0..=8u64 {
            let mut sum = BigInt::zero();
            for k in 0..=n {
                let s = stirling1(n, k).unwrap();
                sum += if s.is_negative() { -s } else { s };
            }
            assert_eq!(
                sum,
                bi(factorials[n as usize]),
                "Σ |s({n}, k)| should be {n}!"
            );
        }
    }

    #[test]
    fn stirling1_huge_n_none() {
        let huge = BigInt::from(u64::MAX) + BigInt::one();
        assert_eq!(stirling1(huge, 2), None);
    }

    // ── Stirling orthogonality ──────────────────────────────────────

    #[test]
    fn stirling_orthogonality() {
        // Σ_j s(n, j) * S(j, k) = δ(n, k)  (Kronecker delta)
        // This is the fundamental inverse relationship.
        for n in 0..=7u64 {
            for k in 0..=7u64 {
                let mut sum = BigInt::zero();
                for j in 0..=n.max(k) {
                    sum += stirling1(n, j).unwrap() * stirling2(j, k).unwrap();
                }
                let expected = if n == k { bi(1) } else { bi(0) };
                assert_eq!(
                    sum, expected,
                    "orthogonality failed: Σ_j s({n},j)*S(j,{k}) = {sum}, expected {expected}"
                );
            }
        }
    }

    // ── Multinomial: known values ───────────────────────────────────

    #[test]
    fn multinomial_basic() {
        // 6! / (2! * 3! * 1!) = 720 / 12 = 60
        assert_eq!(multinomial(6, &[2, 3, 1]), Some(bi(60)));
    }

    #[test]
    fn multinomial_reduces_to_binomial() {
        // multinomial(n, [k, n-k]) = C(n, k)
        assert_eq!(multinomial(10, &[3, 7]), Some(bi(120)));
        assert_eq!(multinomial(6, &[2, 4]), Some(bi(15)));
        assert_eq!(multinomial(20, &[10, 10]), Some(bi(184756)));
    }

    #[test]
    fn multinomial_all_ones() {
        // multinomial(n, [1, 1, ..., 1]) = n!
        assert_eq!(multinomial(4, &[1, 1, 1, 1]), Some(bi(24)));
        assert_eq!(multinomial(5, &[1, 1, 1, 1, 1]), Some(bi(120)));
    }

    #[test]
    fn multinomial_single_group() {
        // multinomial(n, [n]) = 1
        assert_eq!(multinomial(5, &[5]), Some(bi(1)));
        assert_eq!(multinomial(10, &[10]), Some(bi(1)));
    }

    #[test]
    fn multinomial_with_zeros() {
        assert_eq!(multinomial(6, &[2, 0, 3, 0, 1]), Some(bi(60)));
    }

    #[test]
    fn multinomial_sum_mismatch_returns_zero() {
        assert_eq!(multinomial(5, &[2, 2]), Some(bi(0)));
        assert_eq!(multinomial(5, &[3, 3]), Some(bi(0)));
    }

    #[test]
    fn multinomial_negative_k_returns_zero() {
        assert_eq!(multinomial(5, &[-1, 6]), Some(bi(0)));
    }

    #[test]
    fn multinomial_zero() {
        let empty: &[i64] = &[];
        assert_eq!(multinomial(0, empty), Some(bi(1)));
        assert_eq!(multinomial(0, &[0]), Some(bi(1)));
    }

    #[test]
    fn multinomial_larger() {
        // 12! / (3! * 4! * 5!) = 479001600 / (6 * 24 * 120) = 27720
        assert_eq!(multinomial(12, &[3, 4, 5]), Some(bi(27720)));
    }

    #[test]
    fn multinomial_huge_k_none() {
        let huge = BigInt::from(u64::MAX) + BigInt::one();
        assert_eq!(multinomial(huge.clone(), &[huge]), None);
    }

    // ── Partition count: known values ───────────────────────────────

    #[test]
    fn partition_count_small() {
        // OEIS A000041: 1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42, ...
        let expected = [1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42];
        for (n, &p) in expected.iter().enumerate() {
            assert_eq!(
                partition_count(n as u64),
                Some(bi(p)),
                "p({n}) should be {p}"
            );
        }
    }

    #[test]
    fn partition_count_medium() {
        assert_eq!(partition_count(20u64), Some(bi(627)));
        assert_eq!(partition_count(30u64), Some(bi(5604)));
        assert_eq!(partition_count(50u64), Some(bi(204226)));
    }

    #[test]
    fn partition_count_100() {
        assert_eq!(partition_count(100u64), Some(bi(190569292)));
    }

    #[test]
    fn partition_count_200() {
        assert_eq!(
            partition_count(200u64),
            Some(BigInt::parse_bytes(b"3972999029388", 10).unwrap())
        );
    }

    #[test]
    fn partition_count_negative() {
        assert_eq!(partition_count(-1), Some(bi(0)));
        assert_eq!(partition_count(-100), Some(bi(0)));
    }

    #[test]
    fn partition_count_zero() {
        assert_eq!(partition_count(0), Some(bi(1)));
    }
}
