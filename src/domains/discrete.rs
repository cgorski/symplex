//! Discrete transforms on exact sequences (SymPy's `sympy.discrete`):
//! linear, cyclic and subset convolutions, the number-theoretic transform,
//! the Walsh–Hadamard transform and the Möbius (subset/superset-sum)
//! transform.  (0.9)
//!
//! Everything here is **exact**.  Sequences are `Ratio<BigInt>` (or
//! `BigInt` residues for the NTT) and every result is an exact rational or
//! residue.  There is deliberately no floating-point FFT
//! (`sympy.discrete.transforms.fft`) and no symbolic DFT over `Ex` roots of
//! unity: the exact tool for polynomial multiplication is [`convolution`]
//! (or [`convolution_ntt`] modulo a prime), and a symbolic result is
//! obtained by mapping the rationals through `Context::from_ratio` or by
//! using [`convolution_ex`] directly on `Ex` coefficients.
//!
//! Transforms that need a power-of-two length ([`ntt`], [`fwht`],
//! [`mobius_transform`] and their inverses, [`convolution_subset`]) zero-pad
//! their input to the next power of two, exactly as SymPy does; the empty
//! sequence is returned unchanged.
//!
//! # Examples
//!
//! ```
//! use symplex::discrete::{convolution, fwht, ifwht, mobius_transform};
//! use num_bigint::BigInt;
//! use num_rational::Ratio;
//!
//! let q = |v: &[i64]| -> Vec<Ratio<BigInt>> {
//!     v.iter().map(|&t| Ratio::from_integer(BigInt::from(t))).collect()
//! };
//! // (1 + 2x + 3x²)(4 + 5x + 6x²) = 4 + 13x + 28x² + 27x³ + 18x⁴
//! assert_eq!(convolution(&q(&[1, 2, 3]), &q(&[4, 5, 6])), q(&[4, 13, 28, 27, 18]));
//! assert_eq!(fwht(&q(&[1, 2, 3, 4])), q(&[10, -2, -4, 0]));
//! assert_eq!(ifwht(&fwht(&q(&[1, 2, 3, 4]))), q(&[1, 2, 3, 4]));
//! assert_eq!(mobius_transform(&q(&[1, 2, 3, 4])), q(&[1, 3, 4, 10]));
//! ```

use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::Ratio;
use num_traits::{One, Zero};

use crate::api::expr::Ex;
use crate::base::errors::SymplexError;
use crate::base::numeric::Q;
use crate::domains::ntheory::{isprime, mod_inverse, primitive_root};

/// The power of two at or above `len`.  A `Vec` holds at most `isize::MAX`
/// bytes, so `len ≤ isize::MAX` and the next power of two always fits in
/// `usize`; the fallback is unreachable and kept only so the arithmetic is
/// total.
fn pow2_at_least(len: usize) -> usize {
    len.checked_next_power_of_two().unwrap_or(len)
}

/// Zero-pad `a` to at least `min_len` and then to a power of two (the
/// empty sequence stays empty when `min_len` is 0).
fn pad_pow2_to<T: Clone + Zero>(a: &[T], min_len: usize) -> Vec<T> {
    let mut v = a.to_vec();
    let len = v.len().max(min_len);
    if len == 0 {
        return v;
    }
    v.resize(pow2_at_least(len), T::zero());
    v
}

/// Zero-pad `a` to the next power of two (the empty sequence stays empty).
fn pad_pow2<T: Clone + Zero>(a: &[T]) -> Vec<T> {
    pad_pow2_to(a, 0)
}

// ═══════════════════════════════════════════════════════════════════════════
// Convolutions
// ═══════════════════════════════════════════════════════════════════════════

/// Linear convolution `c[k] = Σᵢ a[i]·b[k−i]` (SymPy `convolution(a, b)`):
/// the coefficients of the product of the polynomials with coefficient
/// lists `a` and `b` (index = degree).
///
/// The result has `a.len() + b.len() − 1` entries, or none if either input
/// is empty.  Exact over `ℚ`; `O(|a|·|b|)`.
///
/// # Examples
///
/// ```
/// use symplex::discrete::convolution;
/// use num_bigint::BigInt;
/// use num_rational::Ratio;
///
/// let q = |v: &[i64]| -> Vec<Ratio<BigInt>> {
///     v.iter().map(|&t| Ratio::from_integer(BigInt::from(t))).collect()
/// };
/// // SymPy: convolution([1, 2, 3], [4, 5, 6]) == [4, 13, 28, 27, 18]
/// assert_eq!(convolution(&q(&[1, 2, 3]), &q(&[4, 5, 6])), q(&[4, 13, 28, 27, 18]));
/// // SymPy: convolution([1/2, 1/3], [3, 4]) == [3/2, 3, 4/3]
/// let r = |p: i64, d: i64| Ratio::new(BigInt::from(p), BigInt::from(d));
/// assert_eq!(convolution(&[r(1, 2), r(1, 3)], &q(&[3, 4])), vec![r(3, 2), r(3, 1), r(4, 3)]);
/// assert!(convolution(&[], &q(&[1, 2])).is_empty());
/// ```
pub fn convolution(a: &[Q], b: &[Q]) -> Vec<Q> {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }
    let mut out = vec![Ratio::zero(); a.len() + b.len() - 1];
    for (i, ai) in a.iter().enumerate() {
        if ai.is_zero() {
            continue;
        }
        for (j, bj) in b.iter().enumerate() {
            out[i + j] += ai * bj;
        }
    }
    out
}

/// Cyclic convolution of length `n` (SymPy `convolution(a, b, cycle=n)`):
/// the linear convolution folded modulo `n`, `c[k] = Σ_{i ≡ k (mod n)} lin[i]`.
///
/// Returns `n` entries (zero-padded when `n` exceeds the linear length);
/// `n = 0` or an empty input gives an empty vector.
///
/// # Examples
///
/// ```
/// use symplex::discrete::convolution_cyclic;
/// use num_bigint::BigInt;
/// use num_rational::Ratio;
///
/// let q = |v: &[i64]| -> Vec<Ratio<BigInt>> {
///     v.iter().map(|&t| Ratio::from_integer(BigInt::from(t))).collect()
/// };
/// // SymPy: convolution([1, 2, 3], [4, 5, 6], cycle=3) == [31, 31, 28]
/// assert_eq!(convolution_cyclic(&q(&[1, 2, 3]), &q(&[4, 5, 6]), 3), q(&[31, 31, 28]));
/// // SymPy: convolution([1, 2, 3], [4, 5, 6], cycle=4) == [22, 13, 28, 27]
/// assert_eq!(convolution_cyclic(&q(&[1, 2, 3]), &q(&[4, 5, 6]), 4), q(&[22, 13, 28, 27]));
/// // SymPy: convolution([1, 2, 3], [4, 5, 6], cycle=6) == [4, 13, 28, 27, 18, 0]
/// assert_eq!(convolution_cyclic(&q(&[1, 2, 3]), &q(&[4, 5, 6]), 6), q(&[4, 13, 28, 27, 18, 0]));
/// ```
pub fn convolution_cyclic(a: &[Q], b: &[Q], n: usize) -> Vec<Q> {
    if n == 0 || a.is_empty() || b.is_empty() {
        return vec![];
    }
    let mut out = vec![Ratio::zero(); n];
    for (i, v) in convolution(a, b).into_iter().enumerate() {
        out[i % n] += v;
    }
    out
}

/// Subset convolution (SymPy `convolution_subset`): indices are bitmasks
/// and `c[k] = Σ_{s ⊆ k} a[s]·b[k ∖ s]` — the sum over all ways of
/// writing `k` as a disjoint union of two subsets.
///
/// Both inputs are zero-padded to the same power-of-two length `2ᵐ`; the
/// result has that length.  Direct enumeration of sub-masks, `O(3ᵐ)`.
///
/// # Examples
///
/// ```
/// use symplex::discrete::convolution_subset;
/// use num_bigint::BigInt;
/// use num_rational::Ratio;
///
/// let q = |v: &[i64]| -> Vec<Ratio<BigInt>> {
///     v.iter().map(|&t| Ratio::from_integer(BigInt::from(t))).collect()
/// };
/// // SymPy: convolution_subset([1, 2, 3, 4], [5, 6, 7, 8]) == [5, 16, 22, 60]
/// assert_eq!(convolution_subset(&q(&[1, 2, 3, 4]), &q(&[5, 6, 7, 8])), q(&[5, 16, 22, 60]));
/// // SymPy: convolution_subset([1, 2], [3, 4, 5]) == [3, 10, 5, 10]
/// assert_eq!(convolution_subset(&q(&[1, 2]), &q(&[3, 4, 5])), q(&[3, 10, 5, 10]));
/// ```
pub fn convolution_subset(a: &[Q], b: &[Q]) -> Vec<Q> {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }
    let n = pow2_at_least(a.len().max(b.len()));
    let a = pad_pow2_to(a, n);
    let b = pad_pow2_to(b, n);
    let mut out = vec![Ratio::zero(); n];
    for (mask, slot) in out.iter_mut().enumerate() {
        let mut acc = &a[0] * &b[mask];
        // Enumerate the non-empty sub-masks s of mask.
        let mut s = mask;
        while s > 0 {
            if !a[s].is_zero() {
                acc += &a[s] * &b[mask ^ s];
            }
            s = (s - 1) & mask;
        }
        *slot = acc;
    }
    out
}

/// Linear convolution of symbolic coefficient lists: `c[k] = Σᵢ a[i]·b[k−i]`
/// as `Ex` (the coefficients of the product of two polynomials given by
/// their `Ex` coefficients).  Empty if either input is empty.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::discrete::convolution_ex;
///
/// let ctx = Context::new();
/// symplex::syms!(ctx; a, b);
/// // (a + x)(b + x): coefficient lists [a, 1] and [b, 1]
/// let c = convolution_ex(&[a.clone(), ctx.one()], &[b.clone(), ctx.one()]);
/// assert_eq!(c.len(), 3);
/// assert_eq!(c[0], &a * &b);
/// assert_eq!(c[1], &a + &b);
/// assert_eq!(c[2], ctx.one());
/// ```
pub fn convolution_ex(a: &[Ex], b: &[Ex]) -> Vec<Ex> {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }
    let mut out: Vec<Option<Ex>> = vec![None; a.len() + b.len() - 1];
    for (i, ai) in a.iter().enumerate() {
        for (j, bj) in b.iter().enumerate() {
            let term = ai * bj;
            out[i + j] = Some(match out[i + j].take() {
                Some(acc) => acc + term,
                None => term,
            });
        }
    }
    // Every slot k receives at least one term (i = min(k, |a|−1)).
    out.into_iter().flatten().collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// Number-theoretic transform
// ═══════════════════════════════════════════════════════════════════════════

/// Everything an NTT of one length over one prime needs, computed once:
/// the validated prime, the power-of-two length `n` and the primitive
/// `n`-th root of unity `ω = g^{(p−1)/n}` for the smallest primitive root
/// `g` (the root SymPy uses).  A convolution builds one plan and runs
/// three transforms on it instead of re-validating the prime and
/// re-deriving the root each time.
struct NttPlan {
    prime: BigInt,
    n: usize,
    root: BigInt,
}

impl NttPlan {
    /// A plan for sequences of length `len` (padded to a power of two).
    fn new(prime: BigInt, len: usize, operation: &'static str) -> Result<Self, SymplexError> {
        let invalid = |reason: String| SymplexError::InvalidArgument { operation, reason };
        if !isprime(prime.clone()) {
            return Err(invalid(format!("modulus {prime} is not prime")));
        }
        let n = pow2_at_least(len);
        if n <= 1 {
            // A length-0/1 transform is the identity; no root is needed.
            return Ok(NttPlan {
                prime,
                n,
                root: BigInt::one(),
            });
        }
        let p_minus_1 = &prime - BigInt::one();
        if !(&p_minus_1 % n).is_zero() {
            return Err(invalid(format!(
                "{prime} has no primitive {n}-th root of unity ({n} does not divide p − 1); \
                 use a prime of the form m·2ᵏ + 1 with 2ᵏ ≥ {n}"
            )));
        }
        // p is prime, so (ℤ/pℤ)× is cyclic and a generator exists.
        let g = primitive_root(prime.clone()).ok_or_else(|| SymplexError::ComputationFailed {
            operation,
            reason: format!("no primitive root modulo the prime {prime}"),
        })?;
        let root = g.modpow(&(&p_minus_1 / n), &prime);
        Ok(NttPlan { prime, n, root })
    }

    /// The root for the inverse transform, `ω⁻¹`.
    fn inverse_root(&self, operation: &'static str) -> Result<BigInt, SymplexError> {
        mod_inverse(self.root.clone(), self.prime.clone()).ok_or_else(|| {
            SymplexError::ComputationFailed {
                operation,
                reason: "root of unity not invertible".to_string(),
            }
        })
    }
}

/// One-shot transform: validate, plan and run.
fn ntt_core(
    a: &[BigInt],
    prime: BigInt,
    inverse: bool,
    operation: &'static str,
) -> Result<Vec<BigInt>, SymplexError> {
    let plan = NttPlan::new(prime, a.len(), operation)?;
    ntt_with_plan(a, &plan, inverse, operation)
}

/// Forward/inverse NTT (iterative radix-2 Cooley–Tukey over `ℤ/pℤ`) on a
/// prepared plan.  `a` is reduced modulo `p` and zero-padded to `plan.n`.
fn ntt_with_plan(
    a: &[BigInt],
    plan: &NttPlan,
    inverse: bool,
    operation: &'static str,
) -> Result<Vec<BigInt>, SymplexError> {
    let p = &plan.prime;
    let mut v: Vec<BigInt> = a.iter().map(|x| x.mod_floor(p)).collect();
    if plan.n <= 1 {
        return Ok(v);
    }
    let n = plan.n;
    v.resize(n, BigInt::zero());
    let root = if inverse {
        plan.inverse_root(operation)?
    } else {
        plan.root.clone()
    };
    // Bit-reversal permutation.
    let bits = n.trailing_zeros();
    for i in 1..n {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        if i < j {
            v.swap(i, j);
        }
    }
    // w[i] = root^i for 0 ≤ i < n/2.
    let mut w = Vec::with_capacity(n / 2);
    w.push(BigInt::one());
    for i in 1..n / 2 {
        let next = (&w[i - 1] * &root) % p;
        w.push(next);
    }
    let mut h = 2;
    while h <= n {
        let half = h / 2;
        let stride = n / h;
        for start in (0..n).step_by(h) {
            for j in 0..half {
                let u = v[start + j].clone();
                let t = (&v[start + j + half] * &w[stride * j]) % p;
                v[start + j] = (&u + &t) % p;
                v[start + j + half] = (&u - &t).mod_floor(p);
            }
        }
        h *= 2;
    }
    if inverse {
        let inv_n = mod_inverse(BigInt::from(n), p.clone()).ok_or_else(|| {
            SymplexError::ComputationFailed {
                operation,
                reason: "transform length not invertible modulo p".to_string(),
            }
        })?;
        for x in &mut v {
            *x = (&*x * &inv_n) % p;
        }
    }
    Ok(v)
}

/// Number-theoretic transform (SymPy `ntt`): the discrete Fourier
/// transform of `a` over `ℤ/pℤ`, `A[k] = Σⱼ a[j]·ωʲᵏ mod p`, with
/// `ω = g^{(p−1)/n}` for the smallest primitive root `g` (the same root
/// SymPy uses, so the outputs agree entry for entry).
///
/// The input is zero-padded to a power-of-two length `n`; `p` must be
/// prime with `n | p − 1` (e.g. `998244353 = 119·2²³ + 1`,
/// `469762049 = 7·2²⁶ + 1`).  Inputs are reduced modulo `p`.
///
/// # Errors
///
/// `InvalidArgument` if `prime` is not prime or has no primitive `n`-th
/// root of unity.
///
/// # Examples
///
/// ```
/// use symplex::discrete::{ntt, intt};
/// use num_bigint::BigInt;
///
/// let b = |v: &[i64]| -> Vec<BigInt> { v.iter().map(|&t| BigInt::from(t)).collect() };
/// // SymPy: ntt([1, 2, 3, 4], prime=998244353) == [10, 173167434, 998244351, 825076915]
/// let t = ntt(&b(&[1, 2, 3, 4]), 998_244_353).unwrap();
/// assert_eq!(t, b(&[10, 173_167_434, 998_244_351, 825_076_915]));
/// assert_eq!(intt(&t, 998_244_353).unwrap(), b(&[1, 2, 3, 4]));
/// // SymPy: ntt([1, 2, 3, 4], prime=769) == [10, 643, 767, 122]
/// assert_eq!(ntt(&b(&[1, 2, 3, 4]), 769).unwrap(), b(&[10, 643, 767, 122]));
/// // 7 − 1 = 6 is not divisible by 4
/// assert!(ntt(&b(&[1, 2, 3, 4]), 7).is_err());
/// ```
pub fn ntt(a: &[BigInt], prime: impl Into<BigInt>) -> Result<Vec<BigInt>, SymplexError> {
    ntt_core(a, prime.into(), false, "ntt")
}

/// Inverse number-theoretic transform (SymPy `intt`):
/// `a[j] = n⁻¹·Σₖ A[k]·ω⁻ʲᵏ mod p`, so that `intt(ntt(a)) == a` for a
/// power-of-two length.
///
/// # Errors
///
/// `InvalidArgument` if `prime` is not prime or has no primitive `n`-th
/// root of unity.
///
/// # Examples
///
/// ```
/// use symplex::discrete::intt;
/// use num_bigint::BigInt;
///
/// let b = |v: &[i64]| -> Vec<BigInt> { v.iter().map(|&t| BigInt::from(t)).collect() };
/// // SymPy: intt([1, 2, 3, 4], prime=998244353) == [499122179, 455830317, 499122176, 542414035]
/// assert_eq!(
///     intt(&b(&[1, 2, 3, 4]), 998_244_353).unwrap(),
///     b(&[499_122_179, 455_830_317, 499_122_176, 542_414_035])
/// );
/// ```
pub fn intt(a: &[BigInt], prime: impl Into<BigInt>) -> Result<Vec<BigInt>, SymplexError> {
    ntt_core(a, prime.into(), true, "intt")
}

/// Linear convolution modulo a prime through the NTT (SymPy
/// `convolution(a, b, prime=p)` / `convolution_ntt`): `ntt` both inputs at
/// a power-of-two length `≥ |a| + |b| − 1`, multiply pointwise, `intt`.
///
/// # Errors
///
/// `InvalidArgument` if `prime` is not prime or is too small to carry the
/// transform length (`2ᵏ ∤ p − 1`).
///
/// # Examples
///
/// ```
/// use symplex::discrete::convolution_ntt;
/// use num_bigint::BigInt;
///
/// let b = |v: &[i64]| -> Vec<BigInt> { v.iter().map(|&t| BigInt::from(t)).collect() };
/// // SymPy: convolution_ntt([1, 2, 3], [4, 5, 6], prime=998244353) == [4, 13, 28, 27, 18]
/// assert_eq!(
///     convolution_ntt(&b(&[1, 2, 3]), &b(&[4, 5, 6]), 998_244_353).unwrap(),
///     b(&[4, 13, 28, 27, 18])
/// );
/// ```
pub fn convolution_ntt(
    a: &[BigInt],
    b: &[BigInt],
    prime: impl Into<BigInt>,
) -> Result<Vec<BigInt>, SymplexError> {
    const OP: &str = "convolution_ntt";
    let prime: BigInt = prime.into();
    if a.is_empty() || b.is_empty() {
        // Validate the modulus even for the trivial product, as `ntt` does.
        NttPlan::new(prime, 0, OP)?;
        return Ok(vec![]);
    }
    let len = a.len() + b.len() - 1;
    let plan = NttPlan::new(prime, len, OP)?;
    let ta = ntt_with_plan(a, &plan, false, OP)?;
    let tb = ntt_with_plan(b, &plan, false, OP)?;
    let prod: Vec<BigInt> = ta
        .iter()
        .zip(&tb)
        .map(|(x, y)| (x * y) % &plan.prime)
        .collect();
    let mut out = ntt_with_plan(&prod, &plan, true, OP)?;
    out.truncate(len);
    Ok(out)
}

// ═══════════════════════════════════════════════════════════════════════════
// Walsh–Hadamard transform
// ═══════════════════════════════════════════════════════════════════════════

/// In-place butterflies `(u, v) ← (u + v, u − v)` over every stride.
fn fwht_in_place(v: &mut [Q]) {
    let n = v.len();
    let mut h = 1;
    while h < n {
        for start in (0..n).step_by(2 * h) {
            for j in start..start + h {
                let u = v[j].clone();
                let w = v[j + h].clone();
                v[j] = &u + &w;
                v[j + h] = u - w;
            }
        }
        h *= 2;
    }
}

/// Walsh–Hadamard transform (SymPy `fwht`), Hadamard ordering:
/// `A[k] = Σⱼ (−1)^{popcount(j & k)} a[j]`.  The input is zero-padded to a
/// power of two.
///
/// # Examples
///
/// ```
/// use symplex::discrete::fwht;
/// use num_bigint::BigInt;
/// use num_rational::Ratio;
///
/// let q = |v: &[i64]| -> Vec<Ratio<BigInt>> {
///     v.iter().map(|&t| Ratio::from_integer(BigInt::from(t))).collect()
/// };
/// // SymPy: fwht([1, 2, 3, 4]) == [10, -2, -4, 0]
/// assert_eq!(fwht(&q(&[1, 2, 3, 4])), q(&[10, -2, -4, 0]));
/// // SymPy: fwht([1, 2, 3]) == [6, 2, 0, -4]   (padded to length 4)
/// assert_eq!(fwht(&q(&[1, 2, 3])), q(&[6, 2, 0, -4]));
/// ```
pub fn fwht(a: &[Q]) -> Vec<Q> {
    let mut v = pad_pow2(a);
    fwht_in_place(&mut v);
    v
}

/// Inverse Walsh–Hadamard transform (SymPy `ifwht`): [`fwht`] divided by
/// the (padded) length, so `ifwht(fwht(a)) == a` for a power-of-two length.
///
/// # Examples
///
/// ```
/// use symplex::discrete::ifwht;
/// use num_bigint::BigInt;
/// use num_rational::Ratio;
///
/// let q = |v: &[i64]| -> Vec<Ratio<BigInt>> {
///     v.iter().map(|&t| Ratio::from_integer(BigInt::from(t))).collect()
/// };
/// let r = |p: i64, d: i64| Ratio::new(BigInt::from(p), BigInt::from(d));
/// // SymPy: ifwht([10, -2, -4, 0]) == [1, 2, 3, 4]
/// assert_eq!(ifwht(&q(&[10, -2, -4, 0])), q(&[1, 2, 3, 4]));
/// // SymPy: ifwht([1, 2, 3, 4]) == [5/2, -1/2, -1, 0]
/// assert_eq!(ifwht(&q(&[1, 2, 3, 4])), vec![r(5, 2), r(-1, 2), r(-1, 1), r(0, 1)]);
/// ```
pub fn ifwht(a: &[Q]) -> Vec<Q> {
    let mut v = pad_pow2(a);
    fwht_in_place(&mut v);
    if v.is_empty() {
        return v;
    }
    let inv_n = Ratio::new(BigInt::one(), BigInt::from(v.len()));
    for x in &mut v {
        *x = &*x * &inv_n;
    }
    v
}

// ═══════════════════════════════════════════════════════════════════════════
// Möbius (subset-sum / superset-sum) transform
// ═══════════════════════════════════════════════════════════════════════════

/// Sum-over-subsets (`subset = true`) or sum-over-supersets transform and
/// its inverse (`sign = −1`), in place on a power-of-two length.
fn mobius_in_place(v: &mut [Q], subset: bool, inverse: bool) {
    let n = v.len();
    let mut bit = 1;
    while bit < n {
        for mask in 0..n {
            // Visit the pair (mask without bit, mask with bit) once, from the
            // element that carries the bit.
            if mask & bit == 0 {
                continue;
            }
            let (lo, hi) = (mask ^ bit, mask);
            let (target, source) = if subset { (hi, lo) } else { (lo, hi) };
            let delta = v[source].clone();
            if inverse {
                v[target] -= delta;
            } else {
                v[target] += delta;
            }
        }
        bit <<= 1;
    }
}

/// Möbius transform over subsets (SymPy `mobius_transform(seq)` with
/// `subset=True`): indices are bitmasks and `A[k] = Σ_{s ⊆ k} a[s]`.  The
/// input is zero-padded to a power of two.
///
/// # Examples
///
/// ```
/// use symplex::discrete::mobius_transform;
/// use num_bigint::BigInt;
/// use num_rational::Ratio;
///
/// let q = |v: &[i64]| -> Vec<Ratio<BigInt>> {
///     v.iter().map(|&t| Ratio::from_integer(BigInt::from(t))).collect()
/// };
/// // SymPy: mobius_transform([1, 2, 3, 4]) == [1, 3, 4, 10]
/// assert_eq!(mobius_transform(&q(&[1, 2, 3, 4])), q(&[1, 3, 4, 10]));
/// // SymPy: mobius_transform([1, 2, 3, 4, 5, 6, 7, 8]) == [1, 3, 4, 10, 6, 14, 16, 36]
/// assert_eq!(
///     mobius_transform(&q(&[1, 2, 3, 4, 5, 6, 7, 8])),
///     q(&[1, 3, 4, 10, 6, 14, 16, 36])
/// );
/// ```
pub fn mobius_transform(a: &[Q]) -> Vec<Q> {
    let mut v = pad_pow2(a);
    mobius_in_place(&mut v, true, false);
    v
}

/// Inverse of [`mobius_transform`] (SymPy `inverse_mobius_transform(seq)`):
/// recovers `a` from its subset sums, `a[k] = Σ_{s ⊆ k} (−1)^{|k∖s|} A[s]`.
///
/// # Examples
///
/// ```
/// use symplex::discrete::inverse_mobius_transform;
/// use num_bigint::BigInt;
/// use num_rational::Ratio;
///
/// let q = |v: &[i64]| -> Vec<Ratio<BigInt>> {
///     v.iter().map(|&t| Ratio::from_integer(BigInt::from(t))).collect()
/// };
/// // SymPy: inverse_mobius_transform([1, 3, 4, 10]) == [1, 2, 3, 4]
/// assert_eq!(inverse_mobius_transform(&q(&[1, 3, 4, 10])), q(&[1, 2, 3, 4]));
/// // SymPy: inverse_mobius_transform([1, 2, 3, 4]) == [1, 1, 2, 0]
/// assert_eq!(inverse_mobius_transform(&q(&[1, 2, 3, 4])), q(&[1, 1, 2, 0]));
/// ```
pub fn inverse_mobius_transform(a: &[Q]) -> Vec<Q> {
    let mut v = pad_pow2(a);
    mobius_in_place(&mut v, true, true);
    v
}

/// Möbius transform over supersets (SymPy `mobius_transform(seq,
/// subset=False)`): `A[k] = Σ_{s ⊇ k} a[s]` over the padded index range.
///
/// # Examples
///
/// ```
/// use symplex::discrete::mobius_transform_superset;
/// use num_bigint::BigInt;
/// use num_rational::Ratio;
///
/// let q = |v: &[i64]| -> Vec<Ratio<BigInt>> {
///     v.iter().map(|&t| Ratio::from_integer(BigInt::from(t))).collect()
/// };
/// // SymPy: mobius_transform([1, 2, 3, 4], subset=False) == [10, 6, 7, 4]
/// assert_eq!(mobius_transform_superset(&q(&[1, 2, 3, 4])), q(&[10, 6, 7, 4]));
/// ```
pub fn mobius_transform_superset(a: &[Q]) -> Vec<Q> {
    let mut v = pad_pow2(a);
    mobius_in_place(&mut v, false, false);
    v
}

/// Inverse of [`mobius_transform_superset`] (SymPy
/// `inverse_mobius_transform(seq, subset=False)`).
///
/// # Examples
///
/// ```
/// use symplex::discrete::{inverse_mobius_transform_superset, mobius_transform_superset};
/// use num_bigint::BigInt;
/// use num_rational::Ratio;
///
/// let q = |v: &[i64]| -> Vec<Ratio<BigInt>> {
///     v.iter().map(|&t| Ratio::from_integer(BigInt::from(t))).collect()
/// };
/// let a = q(&[1, 2, 3, 4]);
/// assert_eq!(inverse_mobius_transform_superset(&mobius_transform_superset(&a)), a);
/// ```
pub fn inverse_mobius_transform_superset(a: &[Q]) -> Vec<Q> {
    let mut v = pad_pow2(a);
    mobius_in_place(&mut v, false, true);
    v
}
