//! Diophantine equations: linear `ax + by = c`, Pell `x² − Dy² = 1`,
//! sums of two squares, and Pythagorean triples.
//!
//! All arithmetic is exact (`BigInt`).  Functions accept any integer type
//! via `impl Into<BigInt>` like the rest of the number-theory API.
//!
//! # Examples
//!
//! ```
//! use symplex::diophantine::{linear_diophantine, pell, sum_of_two_squares};
//! use num_bigint::BigInt;
//!
//! // 12x + 18y = 30 has solutions x = -5 + 3k, y = 5 - 2k  (up to the sign
//! // convention of the particular solution)
//! let sol = linear_diophantine(12, 18, 30).unwrap();
//! assert_eq!(BigInt::from(12) * &sol.x + BigInt::from(18) * &sol.y, BigInt::from(30));
//! assert_eq!(BigInt::from(12) * &sol.x_step + BigInt::from(18) * &sol.y_step, BigInt::from(0));
//!
//! // Fundamental solution of x² − 61y² = 1
//! let (x, y) = pell(61).unwrap();
//! assert_eq!(x, BigInt::from(1_766_319_049u64));
//! assert_eq!(y, BigInt::from(226_153_980u64));
//!
//! // 25 = 3² + 4²
//! assert_eq!(sum_of_two_squares(25), Some((BigInt::from(3), BigInt::from(4))));
//! ```

use num_bigint::BigInt;
use num_integer::{ExtendedGcd, Integer};
use num_traits::{One, Signed, Zero};

use crate::domains::ntheory;

// ═══════════════════════════════════════════════════════════════════════════
// Linear Diophantine equations
// ═══════════════════════════════════════════════════════════════════════════

/// The solution family of a linear Diophantine equation `a·x + b·y = c`,
/// as returned by [`linear_diophantine`].
///
/// `(x, y)` is one particular solution; the general solution is
/// `(x + k·x_step, y + k·y_step)` for all `k ∈ ℤ`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LinearDiophantine {
    /// `x` of the particular solution.
    pub x: BigInt,
    /// `y` of the particular solution.
    pub y: BigInt,
    /// The change in `x` between consecutive solutions (`b / gcd(a, b)`).
    pub x_step: BigInt,
    /// The change in `y` between consecutive solutions (`−a / gcd(a, b)`).
    pub y_step: BigInt,
}

/// Solve `a·x + b·y = c` over the integers.
///
/// Returns `Some(`[`LinearDiophantine`]` { x, y, x_step, y_step })` where
/// `(x, y)` is a particular solution and the general solution is
/// `(x + k·x_step, y + k·y_step)` for all `k ∈ ℤ`, with `x_step = b/g`,
/// `y_step = −a/g`, `g = gcd(a, b)`.
/// Returns `None` if `g ∤ c` (no solution), or if `a = b = 0` and
/// `c ≠ 0`.  When `a = b = c = 0` every pair is a solution; this is
/// reported with all four fields zero.
///
/// The particular solution is normalised so that `0 ≤ x < |x_step|`
/// whenever `x_step ≠ 0`.
///
/// # Examples
///
/// ```
/// use symplex::diophantine::linear_diophantine;
/// use num_bigint::BigInt;
///
/// let sol = linear_diophantine(3, 5, 1).unwrap();
/// assert_eq!((sol.x, sol.y), (BigInt::from(2), BigInt::from(-1)));       // 3·2 − 5 = 1
/// assert_eq!((sol.x_step, sol.y_step), (BigInt::from(5), BigInt::from(-3)));
/// assert!(linear_diophantine(4, 6, 7).is_none());                        // gcd 2 ∤ 7
/// ```
pub fn linear_diophantine(
    a: impl Into<BigInt>,
    b: impl Into<BigInt>,
    c: impl Into<BigInt>,
) -> Option<LinearDiophantine> {
    let a: BigInt = a.into();
    let b: BigInt = b.into();
    let c: BigInt = c.into();
    if a.is_zero() && b.is_zero() {
        return if c.is_zero() {
            Some(LinearDiophantine {
                x: BigInt::zero(),
                y: BigInt::zero(),
                x_step: BigInt::zero(),
                y_step: BigInt::zero(),
            })
        } else {
            None
        };
    }
    let ExtendedGcd { gcd: g, x: s, y: t } = ntheory::gcdex(a.clone(), b.clone());
    let (q, r) = c.div_rem(&g);
    if !r.is_zero() {
        return None;
    }
    let mut x0 = &s * &q;
    let mut y0 = &t * &q;
    let dx = &b / &g;
    let dy = -(&a / &g);
    // Normalise 0 ≤ x0 < |dx|.
    if !dx.is_zero() {
        let k = x0.div_floor(&dx.abs());
        x0 -= &k * &dx.abs();
        // Adjust y0 consistently: moving x by ±|dx| moves y by ∓ sign(dx)·dy... keep the
        // pair on the solution line by recomputing from the equation.
        y0 = if b.is_zero() {
            y0
        } else {
            (&c - &a * &x0) / &b
        };
    }
    debug_assert_eq!(&a * &x0 + &b * &y0, c);
    Some(LinearDiophantine {
        x: x0,
        y: y0,
        x_step: dx,
        y_step: dy,
    })
}

/// Solve `Σ coeffsᵢ · xᵢ = c` over the integers for any number of
/// variables.
///
/// Returns one particular solution vector, or `None` if
/// `gcd(coeffs) ∤ c` (or if all coefficients are zero and `c ≠ 0`).  The
/// solution is found by iterated extended GCD: the equation is reduced to
/// two variables at a time.  For the full solution family use the
/// two-variable [`linear_diophantine`] on the reduced problems.
///
/// # Examples
///
/// ```
/// use symplex::diophantine::linear_diophantine_n;
/// use num_bigint::BigInt;
///
/// let sol = linear_diophantine_n(&[6, 10, 15], 1).unwrap();
/// let lhs: BigInt = [6, 10, 15].iter().zip(&sol).map(|(a, x)| BigInt::from(*a) * x).sum();
/// assert_eq!(lhs, BigInt::from(1));
/// assert!(linear_diophantine_n(&[4, 6], 3).is_none());
/// ```
pub fn linear_diophantine_n(
    coeffs: &[impl Into<BigInt> + Clone],
    c: impl Into<BigInt>,
) -> Option<Vec<BigInt>> {
    let coeffs: Vec<BigInt> = coeffs.iter().map(|a| a.clone().into()).collect();
    let c: BigInt = c.into();
    if coeffs.is_empty() {
        return if c.is_zero() { Some(vec![]) } else { None };
    }
    if coeffs.len() == 1 {
        if coeffs[0].is_zero() {
            return if c.is_zero() {
                Some(vec![BigInt::zero()])
            } else {
                None
            };
        }
        let (q, r) = c.div_rem(&coeffs[0]);
        return if r.is_zero() { Some(vec![q]) } else { None };
    }
    // Prefix gcds: g_i = gcd(a_0, …, a_i) with Bezout data so that
    // g_i = g_{i-1}·s_i + a_i·t_i.
    let mut g = coeffs[0].clone();
    let mut bezout: Vec<(BigInt, BigInt)> = Vec::with_capacity(coeffs.len());
    for a in &coeffs[1..] {
        let ExtendedGcd {
            gcd: gi,
            x: s,
            y: t,
        } = ntheory::gcdex(g.clone(), a.clone());
        bezout.push((s, t));
        g = gi;
    }
    if g.is_zero() {
        return if c.is_zero() {
            Some(vec![BigInt::zero(); coeffs.len()])
        } else {
            None
        };
    }
    let (q, r) = c.div_rem(&g);
    if !r.is_zero() {
        return None;
    }
    // Back-substitute: target for the last prefix is q·g = c.
    let n = coeffs.len();
    let mut xs = vec![BigInt::zero(); n];
    let mut target = q; // multiplier of g_{n-1}
    for i in (1..n).rev() {
        let (s, t) = &bezout[i - 1];
        // g_i · target = g_{i-1}·(s·target) + a_i·(t·target)
        xs[i] = t * &target;
        target = s * &target;
    }
    xs[0] = target;
    debug_assert_eq!(
        coeffs.iter().zip(&xs).map(|(a, x)| a * x).sum::<BigInt>(),
        c
    );
    Some(xs)
}

// ═══════════════════════════════════════════════════════════════════════════
// Pell equations
// ═══════════════════════════════════════════════════════════════════════════

/// Fundamental (smallest positive) solution of the Pell equation
/// `x² − D·y² = 1`, via the continued fraction of `√D`.
///
/// Returns `None` if `D ≤ 0` or `D` is a perfect square (no non-trivial
/// solutions).
///
/// # Examples
///
/// ```
/// use symplex::diophantine::pell;
/// use num_bigint::BigInt;
///
/// assert_eq!(pell(2), Some((BigInt::from(3), BigInt::from(2))));
/// assert_eq!(pell(3), Some((BigInt::from(2), BigInt::from(1))));
/// assert_eq!(pell(5), Some((BigInt::from(9), BigInt::from(4))));
/// assert_eq!(pell(61).unwrap().0, BigInt::from(1_766_319_049u64));
/// assert_eq!(pell(4), None);
/// ```
pub fn pell(d: impl Into<BigInt>) -> Option<(BigInt, BigInt)> {
    let d: BigInt = d.into();
    if !d.is_positive() {
        return None;
    }
    let a0 = d.sqrt();
    if &a0 * &a0 == d {
        return None;
    }
    // Walk the continued fraction convergents of √D; the first convergent
    // h/k with h² − D k² = 1 is the fundamental solution.
    let (mut h_prev, mut h) = (BigInt::one(), a0.clone()); // h_{-1}, h_0
    let (mut k_prev, mut k) = (BigInt::zero(), BigInt::one()); // k_{-1}, k_0
    let mut m = BigInt::zero();
    let mut dd = BigInt::one();
    let mut a = a0.clone();
    loop {
        if &h * &h - &d * &k * &k == BigInt::one() {
            return Some((h, k));
        }
        m = &dd * &a - &m;
        dd = (&d - &m * &m) / &dd;
        a = (&a0 + &m) / &dd;
        let h_next = &a * &h + &h_prev;
        let k_next = &a * &k + &k_prev;
        h_prev = std::mem::replace(&mut h, h_next);
        k_prev = std::mem::replace(&mut k, k_next);
    }
}

/// The first `count` positive solutions of `x² − D·y² = 1`, generated from
/// the fundamental solution by Brahmagupta composition
/// `(x₁, y₁)·(x, y) = (x₁x + D y₁y, x₁y + y₁x)`.
///
/// Returns an empty vector when [`pell`] has no solution.
///
/// # Examples
///
/// ```
/// use symplex::diophantine::pell_solutions;
/// use num_bigint::BigInt;
///
/// let sols = pell_solutions(2, 4);
/// let as_i64: Vec<(i64, i64)> = sols.iter()
///     .map(|(x, y)| (x.try_into().unwrap(), y.try_into().unwrap()))
///     .collect();
/// assert_eq!(as_i64, vec![(3, 2), (17, 12), (99, 70), (577, 408)]);
/// ```
pub fn pell_solutions(d: impl Into<BigInt>, count: usize) -> Vec<(BigInt, BigInt)> {
    let d: BigInt = d.into();
    let Some((x1, y1)) = pell(d.clone()) else {
        return vec![];
    };
    let mut out = Vec::with_capacity(count);
    let (mut x, mut y) = (x1.clone(), y1.clone());
    for _ in 0..count {
        out.push((x.clone(), y.clone()));
        let nx = &x1 * &x + &d * &y1 * &y;
        let ny = &x1 * &y + &y1 * &x;
        x = nx;
        y = ny;
    }
    out
}

/// Fundamental solution of the negative Pell equation `x² − D·y² = −1`, if
/// one exists (it does iff the continued-fraction period of `√D` has odd
/// length).
///
/// # Examples
///
/// ```
/// use symplex::diophantine::pell_negative;
/// use num_bigint::BigInt;
///
/// assert_eq!(pell_negative(2), Some((BigInt::from(1), BigInt::from(1))));
/// assert_eq!(pell_negative(5), Some((BigInt::from(2), BigInt::from(1))));
/// assert_eq!(pell_negative(3), None);
/// ```
pub fn pell_negative(d: impl Into<BigInt>) -> Option<(BigInt, BigInt)> {
    let d: BigInt = d.into();
    if !d.is_positive() {
        return None;
    }
    let a0 = d.sqrt();
    if &a0 * &a0 == d {
        return None;
    }
    let cf = ntheory::continued_fraction_periodic(d.clone())?;
    if cf.period.len() % 2 == 0 {
        return None;
    }
    // The convergent just before the end of the first period gives the
    // solution: walk convergents and test.
    let mut terms = cf.pre_period;
    terms.extend(cf.period.iter().cloned());
    let neg_one = -BigInt::one();
    for c in ntheory::continued_fraction_convergents(&terms) {
        let (h, k) = (c.numer().clone(), c.denom().clone());
        if &h * &h - &d * &k * &k == neg_one {
            return Some((h, k));
        }
    }
    None
}

// ═══════════════════════════════════════════════════════════════════════════
// Sums of squares
// ═══════════════════════════════════════════════════════════════════════════

/// Write `n ≥ 0` as `a² + b²` with `0 ≤ a ≤ b`, if possible.
///
/// By Fermat's theorem a representation exists iff every prime `≡ 3
/// (mod 4)` divides `n` to an even power.  The representation is built
/// multiplicatively from the prime factorization: primes `p ≡ 1 (mod 4)`
/// are split with Cornacchia's algorithm (Hermite–Serret), `2 = 1² + 1²`,
/// and `p ≡ 3 (mod 4)` contribute `p^{e/2}` to both coordinates via
/// scaling.  Returns `None` for negative `n` or when no representation
/// exists.
///
/// # Examples
///
/// ```
/// use symplex::diophantine::sum_of_two_squares;
/// use num_bigint::BigInt;
///
/// let b = |v: i64| BigInt::from(v);
/// assert_eq!(sum_of_two_squares(0), Some((b(0), b(0))));
/// assert_eq!(sum_of_two_squares(2), Some((b(1), b(1))));
/// assert_eq!(sum_of_two_squares(13), Some((b(2), b(3))));
/// assert_eq!(sum_of_two_squares(45), Some((b(3), b(6))));
/// assert_eq!(sum_of_two_squares(21), None);
/// ```
pub fn sum_of_two_squares(n: impl Into<BigInt>) -> Option<(BigInt, BigInt)> {
    let n: BigInt = n.into();
    if n.is_negative() {
        return None;
    }
    if n.is_zero() {
        return Some((BigInt::zero(), BigInt::zero()));
    }
    // Gaussian-integer product (a + bi)(c + di).
    let mut a = BigInt::one();
    let mut b = BigInt::zero();
    for (p, e) in ntheory::factorint(n) {
        let (pa, pb) = if p == BigInt::from(2) {
            (BigInt::one(), BigInt::one())
        } else if p.mod_floor(&BigInt::from(4)) == BigInt::from(3) {
            if e % 2 == 1 {
                return None;
            }
            // p^e = (p^{e/2})² + 0²
            let s = p.pow(e / 2);
            a *= &s;
            b *= &s;
            continue;
        } else {
            cornacchia_prime(&p)?
        };
        for _ in 0..e {
            let na = &a * &pa - &b * &pb;
            let nb = &a * &pb + &b * &pa;
            a = na;
            b = nb;
        }
    }
    let (a, b) = (a.abs(), b.abs());
    Some(if a <= b { (a, b) } else { (b, a) })
}

/// Cornacchia / Hermite–Serret: write a prime `p ≡ 1 (mod 4)` as
/// `a² + b²`.
fn cornacchia_prime(p: &BigInt) -> Option<(BigInt, BigInt)> {
    // x with x² ≡ −1 (mod p), taken in (p/2, p).
    let x = ntheory::sqrt_mod(-BigInt::one(), p.clone())?;
    let mut x = if &x * 2 < *p { p - &x } else { x };
    // Euclid on (p, x) until the remainder drops below √p.
    let root = p.sqrt();
    let mut a = p.clone();
    while a > root {
        let r = a.mod_floor(&x);
        a = std::mem::replace(&mut x, r);
    }
    let rem = p - &a * &a;
    let b = rem.sqrt();
    if &b * &b == rem { Some((a, b)) } else { None }
}

/// Every non-negative integer is a sum of four squares (Lagrange).
/// Returns one representation `n = a² + b² + c² + d²` with
/// `a ≤ b ≤ c ≤ d`, or `None` for negative `n`.
///
/// Strategy: strip factors of 4 (`n = 4ᵏ m`), then for `m` search a
/// small `a` such that `m − a²` is a sum of two squares (using
/// [`sum_of_two_squares`]); when `m ≡ 7 (mod 8)` a sum of three squares
/// is impossible so `d² ` is chosen first with `m − d² ≢ 7 (mod 8)`.
///
/// # Examples
///
/// ```
/// use symplex::diophantine::sum_of_four_squares;
/// use num_bigint::BigInt;
///
/// let (a, b, c, d) = sum_of_four_squares(7).unwrap();
/// assert_eq!(&a * &a + &b * &b + &c * &c + &d * &d, BigInt::from(7));
/// let (a, b, c, d) = sum_of_four_squares(123_456_789).unwrap();
/// assert_eq!(&a * &a + &b * &b + &c * &c + &d * &d, BigInt::from(123_456_789));
/// ```
pub fn sum_of_four_squares(n: impl Into<BigInt>) -> Option<(BigInt, BigInt, BigInt, BigInt)> {
    let n: BigInt = n.into();
    if n.is_negative() {
        return None;
    }
    if n.is_zero() {
        let z = BigInt::zero();
        return Some((z.clone(), z.clone(), z.clone(), z));
    }
    // n = 4^k · m
    let mut m = n.clone();
    let mut scale = BigInt::one();
    let four = BigInt::from(4);
    while (&m % &four).is_zero() {
        m /= &four;
        scale *= 2;
    }
    // Try d = ⌊√m⌋, ⌊√m⌋−1, … and write m − d² as a sum of three squares by
    // a further search on c with a two-square check.  Terminates quickly
    // in practice (density of sums of two squares is high).
    let root = m.sqrt();
    let mut d = root.clone();
    while d >= BigInt::zero() {
        let rest = &m - &d * &d;
        // Sum of three squares needs rest ≢ 7 (mod 8) (after removing 4^j).
        let mut r3 = rest.clone();
        while !r3.is_zero() && (&r3 % &four).is_zero() {
            r3 /= &four;
        }
        if (&r3 % BigInt::from(8)) != BigInt::from(7) {
            let mut c = rest.sqrt();
            while c >= BigInt::zero() {
                let rest2 = &rest - &c * &c;
                if let Some((a, b)) = sum_of_two_squares(rest2) {
                    let mut v = [a * &scale, b * &scale, &c * &scale, &d * &scale];
                    v.sort();
                    let [a, b, c, d] = v;
                    return Some((a, b, c, d));
                }
                c -= 1;
            }
        }
        d -= 1;
    }
    None
}

// ═══════════════════════════════════════════════════════════════════════════
// Pythagorean triples
// ═══════════════════════════════════════════════════════════════════════════

/// All primitive Pythagorean triples `(a, b, c)` with `a < b < c ≤ limit`,
/// generated by Euclid's formula `a = m² − n², b = 2mn, c = m² + n²` over
/// coprime `m > n` of opposite parity, sorted by `(c, a)`.
///
/// # Examples
///
/// ```
/// use symplex::diophantine::pythagorean_triples;
///
/// assert_eq!(pythagorean_triples(30), vec![(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25), (20, 21, 29)]);
/// ```
pub fn pythagorean_triples(limit: u64) -> Vec<(u64, u64, u64)> {
    let mut out = Vec::new();
    let mut m = 2u64;
    while m * m < limit {
        let start = if m.is_multiple_of(2) { 1 } else { 2 };
        let mut n = start;
        while n < m && m * m + n * n <= limit {
            if ntheory::gcd(m, n).is_one() {
                let a = m * m - n * n;
                let b = 2 * m * n;
                let c = m * m + n * n;
                out.push((a.min(b), a.max(b), c));
            }
            n += 2;
        }
        m += 1;
    }
    out.sort_by_key(|&(a, _, c)| (c, a));
    out
}

/// Frobenius number of a set of positive integers with `gcd = 1`: the
/// largest integer **not** representable as a non-negative integer
/// combination of them.
///
/// For two arguments the closed form `ab − a − b` is used; otherwise a
/// round-robin (shortest-path) dynamic program over residues modulo the
/// smallest element, which needs `O(min · k)` time and `O(min)` memory.
/// Returns `None` if the set is empty, contains a non-positive value, or
/// has `gcd > 1` (infinitely many non-representable integers), or if the
/// smallest element exceeds `10⁷` (memory cap).  Sets containing `1` give
/// `Some(-1)`.
///
/// # Examples
///
/// ```
/// use symplex::diophantine::frobenius_number;
///
/// assert_eq!(frobenius_number(&[3, 5]), Some(7));
/// assert_eq!(frobenius_number(&[6, 9, 20]), Some(43));   // McNuggets
/// assert_eq!(frobenius_number(&[2, 4]), None);
/// assert_eq!(frobenius_number(&[1, 7]), Some(-1));
/// ```
pub fn frobenius_number(values: &[u64]) -> Option<i128> {
    if values.is_empty() || values.contains(&0) {
        return None;
    }
    let mut vals: Vec<u64> = values.to_vec();
    vals.sort_unstable();
    vals.dedup();
    let g = vals
        .iter()
        .fold(0u64, |acc, &v| ntheory::gcd(acc, v).try_into().unwrap_or(0));
    if g != 1 {
        return None;
    }
    if vals[0] == 1 {
        return Some(-1);
    }
    if vals.len() == 2 {
        return Some(vals[0] as i128 * vals[1] as i128 - vals[0] as i128 - vals[1] as i128);
    }
    let a = vals[0];
    if a > 10_000_000 {
        return None;
    }
    // Round-robin algorithm (Böcker–Lipták): dist[r] = smallest representable
    // number ≡ r (mod a).
    let a_us = a as usize;
    let mut dist: Vec<u128> = vec![u128::MAX; a_us];
    dist[0] = 0;
    for &v in &vals[1..] {
        let d = ntheory::gcd(a, v).try_into().unwrap_or(1u64) as usize;
        for r0 in 0..d {
            // Walk the cycle r0 → r0 + v → … (mod a), twice to propagate.
            let mut r = r0;
            let mut best = dist[r];
            let cycle_len = a_us / d;
            for _ in 0..(2 * cycle_len) {
                let next = (r + v as usize) % a_us;
                let cand = best.saturating_add(v as u128);
                if cand < dist[next] {
                    dist[next] = cand;
                }
                best = dist[next];
                r = next;
            }
        }
    }
    let max = dist.iter().copied().max()?;
    if max == u128::MAX {
        return None;
    }
    Some(max as i128 - a as i128)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn bi(n: i64) -> BigInt {
        BigInt::from(n)
    }

    #[test]
    fn linear_two_variables() {
        for (a, b, c) in [
            (3i64, 5i64, 1i64),
            (12, 18, 30),
            (-4, 6, 10),
            (7, 0, 21),
            (0, 5, -15),
            (5, 7, 0),
        ] {
            let sol = linear_diophantine(a, b, c).expect("solvable");
            assert_eq!(
                bi(a) * &sol.x + bi(b) * &sol.y,
                bi(c),
                "particular ({a},{b},{c})"
            );
            assert_eq!(
                bi(a) * &sol.x_step + bi(b) * &sol.y_step,
                bi(0),
                "direction ({a},{b},{c})"
            );
            // Neighbouring solutions.
            for k in -3..=3i64 {
                let x = &sol.x + bi(k) * &sol.x_step;
                let y = &sol.y + bi(k) * &sol.y_step;
                assert_eq!(bi(a) * x + bi(b) * y, bi(c));
            }
            if !sol.x_step.is_zero() {
                assert!(sol.x >= bi(0) && sol.x < sol.x_step.abs());
            }
        }
        assert!(linear_diophantine(4, 6, 7).is_none());
        assert!(linear_diophantine(0, 0, 1).is_none());
        assert_eq!(
            linear_diophantine(0, 0, 0),
            Some(LinearDiophantine {
                x: bi(0),
                y: bi(0),
                x_step: bi(0),
                y_step: bi(0),
            })
        );
    }

    #[test]
    fn linear_n_variables() {
        let cases: Vec<(Vec<i64>, i64)> = vec![
            (vec![6, 10, 15], 1),
            (vec![6, 10, 15], 100),
            (vec![2, 3], 7),
            (vec![4, 6, 9, 10], -13),
            (vec![5], 15),
            (vec![0, 0, 3], 9),
        ];
        for (coeffs, c) in cases {
            let sol = linear_diophantine_n(&coeffs, c).expect("solvable");
            let lhs: BigInt = coeffs.iter().zip(&sol).map(|(a, x)| bi(*a) * x).sum();
            assert_eq!(lhs, bi(c), "{coeffs:?} = {c}");
        }
        assert!(linear_diophantine_n(&[4, 6], 3).is_none());
        assert!(linear_diophantine_n(&[0, 0], 3).is_none());
        assert_eq!(linear_diophantine_n(&[5], 7), None);
        assert_eq!(linear_diophantine_n(&Vec::<i64>::new(), 0), Some(vec![]));
    }

    #[test]
    fn pell_fundamental_solutions() {
        let expect = [
            (2i64, 3i64, 2i64),
            (3, 2, 1),
            (5, 9, 4),
            (6, 5, 2),
            (7, 8, 3),
            (13, 649, 180),
            (61, 1_766_319_049, 226_153_980),
            (109, 158_070_671_986_249, 15_140_424_455_100),
        ];
        for (d, x, y) in expect {
            assert_eq!(pell(d), Some((bi(x), bi(y))), "D = {d}");
        }
        assert_eq!(pell(1), None);
        assert_eq!(pell(4), None);
        assert_eq!(pell(0), None);
        assert_eq!(pell(-2), None);
        // Every non-square D up to 200 has a solution that checks out.
        for d in 2..200i64 {
            if let Some((x, y)) = pell(d) {
                assert_eq!(&x * &x - bi(d) * &y * &y, bi(1), "D = {d}");
            } else {
                assert!(ntheory::is_square(d));
            }
        }
    }

    #[test]
    fn pell_solution_sequences() {
        let s = pell_solutions(3, 5);
        let v: Vec<(i64, i64)> = s
            .iter()
            .map(|(x, y)| (x.try_into().unwrap(), y.try_into().unwrap()))
            .collect();
        assert_eq!(v, vec![(2, 1), (7, 4), (26, 15), (97, 56), (362, 209)]);
        for (x, y) in pell_solutions(61, 3) {
            assert_eq!(&x * &x - bi(61) * &y * &y, bi(1));
        }
        assert!(pell_solutions(9, 3).is_empty());
    }

    #[test]
    fn negative_pell() {
        assert_eq!(pell_negative(2), Some((bi(1), bi(1))));
        assert_eq!(pell_negative(5), Some((bi(2), bi(1))));
        assert_eq!(pell_negative(10), Some((bi(3), bi(1))));
        assert_eq!(pell_negative(13), Some((bi(18), bi(5))));
        assert_eq!(pell_negative(3), None);
        assert_eq!(pell_negative(7), None);
        for d in 2..120i64 {
            if let Some((x, y)) = pell_negative(d) {
                assert_eq!(&x * &x - bi(d) * &y * &y, bi(-1), "D = {d}");
            }
        }
    }

    #[test]
    fn two_squares_brute_force() {
        for n in 0..500i64 {
            let brute = (0..=n)
                .flat_map(|a| (a..=n).map(move |b| (a, b)))
                .find(|(a, b)| a * a + b * b == n);
            let got = sum_of_two_squares(n);
            assert_eq!(got.is_some(), brute.is_some(), "n = {n}");
            if let Some((a, b)) = got {
                assert_eq!(&a * &a + &b * &b, bi(n));
                assert!(a <= b);
            }
        }
    }

    #[test]
    fn two_squares_large() {
        // 1000000007 ≡ 3 mod 4 → impossible; 1000000009 ≡ 1 mod 4 → possible.
        assert!(sum_of_two_squares(1_000_000_007).is_none());
        let (a, b) = sum_of_two_squares(1_000_000_009).unwrap();
        assert_eq!(&a * &a + &b * &b, bi(1_000_000_009));
        // (2^61 − 1) is ≡ 3 mod 4 (Mersenne) → squared it is representable.
        let m: BigInt = (BigInt::one() << 61usize) - 1;
        let (a, b) = sum_of_two_squares(&m * &m).unwrap();
        assert_eq!(&a * &a + &b * &b, &m * &m);
    }

    #[test]
    fn four_squares_brute_force() {
        for n in 0..300i64 {
            let (a, b, c, d) = sum_of_four_squares(n).unwrap();
            assert_eq!(&a * &a + &b * &b + &c * &c + &d * &d, bi(n), "n = {n}");
            assert!(a <= b && b <= c && c <= d);
        }
        for n in [7i64, 15, 23, 28, 31, 60, 112, 240, 10_000_007, 987_654_321] {
            let (a, b, c, d) = sum_of_four_squares(n).unwrap();
            assert_eq!(&a * &a + &b * &b + &c * &c + &d * &d, bi(n), "n = {n}");
        }
        assert!(sum_of_four_squares(-1).is_none());
    }

    #[test]
    fn pythagorean_triples_are_primitive_and_complete() {
        let t = pythagorean_triples(100);
        for &(a, b, c) in &t {
            assert_eq!(a * a + b * b, c * c);
            assert!(a < b && b < c && c <= 100);
            assert!(ntheory::gcd(a, b).is_one());
        }
        // Brute-force count of primitive triples with c ≤ 100 is 16.
        let mut brute = 0;
        for a in 1..100u64 {
            for b in a + 1..100 {
                let c2 = a * a + b * b;
                let c = num_integer::Roots::sqrt(&c2);
                if c * c == c2 && c <= 100 && ntheory::gcd(a, b).is_one() {
                    brute += 1;
                }
            }
        }
        assert_eq!(t.len(), brute);
        assert_eq!(t[0], (3, 4, 5));
    }

    #[test]
    fn frobenius_numbers() {
        assert_eq!(frobenius_number(&[3, 5]), Some(7));
        assert_eq!(frobenius_number(&[6, 9, 20]), Some(43));
        assert_eq!(frobenius_number(&[4, 7, 12]), Some(17));
        assert_eq!(frobenius_number(&[12, 16, 20, 27]), Some(89));
        assert_eq!(frobenius_number(&[2, 4]), None);
        assert_eq!(frobenius_number(&[]), None);
        assert_eq!(frobenius_number(&[1, 9]), Some(-1));
        // Brute-force check for random small sets.
        for set in [[3u64, 7, 11], [5, 8, 9], [7, 10, 13], [11, 13, 17]] {
            let f = frobenius_number(&set).unwrap();
            let bound = (f + 1) as usize + 200;
            let mut reachable = vec![false; bound];
            reachable[0] = true;
            for i in 1..bound {
                for &v in &set {
                    if i >= v as usize && reachable[i - v as usize] {
                        reachable[i] = true;
                        break;
                    }
                }
            }
            assert!(!reachable[f as usize], "{set:?}: {f} should be unreachable");
            assert!(reachable[f as usize + 1..].iter().all(|&r| r), "{set:?}");
        }
    }
}
