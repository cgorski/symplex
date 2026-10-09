//! Integer gcds and modular images behind the polynomial gcds.
//!
//! - [`int_gcd`]: the gcd of two integers by Lehmer's algorithm (Knuth,
//!   TAOCP vol. 2, §4.5.2, Algorithm L).  `num-integer`'s gcd is the binary
//!   (Stein) algorithm, about one bit per pass over the operands; Lehmer's
//!   algorithm runs the Euclidean quotients on the leading 62 bits and
//!   applies them to the full operands about 30 bits per pass.  Both are
//!   quadratic; the integer gcds at the bottom of a heuristic gcd in four
//!   variables have hundreds of thousands of digits.
//! - [`content`], [`ratio_reduced`], [`int_lcm`], [`rat_add`], [`rat_mul`]:
//!   contents, reduced fractions and fraction arithmetic with it.
//! - [`heu_gcd`]: the heuristic gcd in `ℤ[x₁, …, xₙ]` (SymPy's
//!   `dmp_zz_heu_gcd`) on integer coefficients.
//! - [`zx_gcd`]: the gcd in `ℤ[x]` by Brown's modular algorithm (W. S.
//!   Brown, *J. ACM* 18 (1971) 478–504; the univariate form of SymPy's
//!   `modgcd_univariate`, `sympy/polys/modulargcd.py`, BSD-3): images over
//!   `𝔽ₚ` for word-size primes, combined by the Chinese remainder theorem
//!   until stable, verified by exact division.  A single image of degree 0
//!   proves the inputs coprime, which is the common case and costs one
//!   reduction of the coefficients.
//! - [`coprime_certified`]: the same certificate for sparse multivariate
//!   polynomials, one variable at a time through univariate images.
//!
//! Primes are taken below `2⁶²` (so [`Fp64`] applies) in descending order
//! and cached.  Everything here is deterministic.

use std::sync::{Mutex, OnceLock};

use num_bigint::{BigInt, BigUint, Sign};
use num_integer::Integer;
use num_traits::{One, Signed, ToPrimitive, Zero};

use super::modpoly::{Fp64, PolyIn, RingOps};
use crate::base::rng::SplitMix64;

// ═══════════════════════════════════════════════════════════════════════════
// Integer gcd
// ═══════════════════════════════════════════════════════════════════════════

/// `gcd(a, b) ≥ 0` (`gcd(0, 0) = 0`), equal to `num_integer::Integer::gcd`.
/// Operands of one or two words take a machine gcd.
pub(crate) fn int_gcd(a: &BigInt, b: &BigInt) -> BigInt {
    let (x, y) = (a.magnitude(), b.magnitude());
    if let (Some(x), Some(y)) = (x.to_u64(), y.to_u64()) {
        return BigInt::from(gcd_u64(x, y));
    }
    if x.bits() <= 128 && y.bits() <= 128 {
        let (x, y) = (x.to_u128().unwrap_or(0), y.to_u128().unwrap_or(0));
        return BigInt::from(gcd_u128(x.max(y), x.min(y)));
    }
    BigInt::from_biguint(Sign::Plus, uint_gcd(x, y))
}

/// Binary gcd on words.
fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    if a == 0 {
        return b;
    }
    if b == 0 {
        return a;
    }
    let shift = (a | b).trailing_zeros();
    a >>= a.trailing_zeros();
    loop {
        b >>= b.trailing_zeros();
        if a > b {
            std::mem::swap(&mut a, &mut b);
        }
        b -= a;
        if b == 0 {
            return a << shift;
        }
    }
}

/// The gcd of the integers in `xs`, non-negative (`0` when there are none or
/// all are zero).  The shortest operand is taken first, so that every
/// further gcd starts with one division of a long operand by a short one;
/// stops as soon as the gcd is `1`.
pub(crate) fn content<'a>(xs: impl IntoIterator<Item = &'a BigInt>) -> BigInt {
    let xs: Vec<&BigInt> = xs.into_iter().filter(|x| !x.is_zero()).collect();
    let Some(start) = (0..xs.len()).min_by_key(|&i| xs[i].bits()) else {
        return BigInt::zero();
    };
    let mut g = xs[start].abs();
    for (i, x) in xs.iter().enumerate() {
        if g.is_one() {
            break;
        }
        if i != start {
            g = int_gcd(&g, x);
        }
    }
    g
}

/// `n/d` in lowest terms with a positive denominator (`d ≠ 0`): the value
/// and representation of `Ratio::new(n, d)`, reduced with [`int_gcd`].
pub(crate) fn ratio_reduced(n: BigInt, d: BigInt) -> num_rational::Ratio<BigInt> {
    if d.is_zero() {
        return num_rational::Ratio::new(n, d);
    }
    let g = int_gcd(&n, &d);
    let (mut n, mut d) = if g.is_one() { (n, d) } else { (n / &g, d / &g) };
    if d.is_negative() {
        n = -n;
        d = -d;
    }
    num_rational::Ratio::new_raw(n, d)
}

/// `a + b` for fractions in lowest terms with positive denominators (as
/// every `Ratio` operation leaves them): `numeric::q_add` (SymPy's
/// `Rational.__add__`, Henrici's sum for two fractions) with Lehmer's gcd
/// [`int_gcd`] for two long operands.  The same reduced fraction as `a + b`.
pub(crate) fn rat_add(
    a: &num_rational::Ratio<BigInt>,
    b: &num_rational::Ratio<BigInt>,
) -> num_rational::Ratio<BigInt> {
    use num_rational::Ratio;
    if a.is_zero() {
        return b.clone();
    }
    if b.is_zero() {
        return a.clone();
    }
    match (a.denom().is_one(), b.denom().is_one()) {
        (true, true) => Ratio::from_integer(a.numer() + b.numer()),
        // p/q + k = (p + k·q)/q is in lowest terms.
        (false, true) => Ratio::new_raw(a.numer() + b.numer() * a.denom(), a.denom().clone()),
        (true, false) => Ratio::new_raw(b.numer() + a.numer() * b.denom(), b.denom().clone()),
        (false, false) => {
            let (u, u1, v, v1) = (a.numer(), a.denom(), b.numer(), b.denom());
            let d1 = int_gcd(u1, v1);
            if d1.is_one() {
                return Ratio::new_raw(u * v1 + v * u1, u1 * v1);
            }
            let t = u * (v1 / &d1) + v * (u1 / &d1);
            if t.is_zero() {
                return Ratio::zero();
            }
            let d2 = int_gcd(&t, &d1);
            if d2.is_one() {
                Ratio::new_raw(t, (u1 / &d1) * v1)
            } else {
                Ratio::new_raw(t / &d2, (u1 / &d1) * (v1 / &d2))
            }
        }
    }
}

/// `a·b` for fractions in lowest terms with positive denominators:
/// `numeric::q_mul` (cross-cancellation) with [`int_gcd`].  The same
/// reduced fraction as `a * b`.
pub(crate) fn rat_mul(
    a: &num_rational::Ratio<BigInt>,
    b: &num_rational::Ratio<BigInt>,
) -> num_rational::Ratio<BigInt> {
    use num_rational::Ratio;
    if a.is_zero() || b.is_zero() {
        return Ratio::zero();
    }
    if a.denom().is_one() && b.denom().is_one() {
        return Ratio::from_integer(a.numer() * b.numer());
    }
    let div = |x: &BigInt, g: &BigInt| if g.is_one() { x.clone() } else { x / g };
    let g1 = int_gcd(a.numer(), b.denom());
    let g2 = int_gcd(b.numer(), a.denom());
    let n = div(a.numer(), &g1) * div(b.numer(), &g2);
    let d = div(a.denom(), &g2) * div(b.denom(), &g1);
    if d.is_negative() {
        Ratio::new_raw(-n, -d)
    } else {
        Ratio::new_raw(n, d)
    }
}

/// `lcm(a, b) ≥ 0` with [`int_gcd`].
pub(crate) fn int_lcm(a: &BigInt, b: &BigInt) -> BigInt {
    if a.is_zero() || b.is_zero() {
        return BigInt::zero();
    }
    (a / int_gcd(a, b) * b).abs()
}

/// Lehmer's gcd of two natural numbers.
///
/// The operands are kept as little-endian `u64` limb vectors, and each
/// cosequence matrix is applied to both in one pass in place
/// ([`lehmer_apply`]); a quotient too long for the leading word is one
/// multiprecision division.
fn uint_gcd(a: &BigUint, b: &BigUint) -> BigUint {
    let (mut u, mut v) = if a >= b {
        (a.to_u64_digits(), b.to_u64_digits())
    } else {
        (b.to_u64_digits(), a.to_u64_digits())
    };
    loop {
        limbs_trim(&mut u);
        limbs_trim(&mut v);
        if v.len() <= 2 {
            // Two words: finish with a division and machine gcds.
            let (ub, vb) = (limbs_to_big(&u), limbs_to_big(&v));
            if vb.is_zero() {
                return ub;
            }
            let r = (&ub % &vb).to_u128().unwrap_or(0);
            return BigUint::from(gcd_u128(vb.to_u128().unwrap_or(0), r));
        }
        let (ubits, vbits) = (limbs_bits(&u), limbs_bits(&v));
        let division_step = |u: &mut Vec<u64>, v: &mut Vec<u64>| {
            let (ub, vb) = (limbs_to_big(u), limbs_to_big(v));
            let r = &ub % &vb;
            *u = std::mem::take(v);
            *v = r.to_u64_digits();
        };
        // A quotient longer than half a word: one multiprecision division.
        if ubits - vbits > 32 {
            division_step(&mut u, &mut v);
            continue;
        }
        let shift = ubits - 62;
        let x = limbs_window(&u, shift) & ((1 << 62) - 1);
        let y = limbs_window(&v, shift) & ((1 << 62) - 1);
        let coeffs = lehmer_cosequence(x, y);
        let bounded = coeffs.iter().all(|c| c.unsigned_abs() < 1u64 << 62);
        if coeffs[1] == 0 || !bounded {
            division_step(&mut u, &mut v);
            continue;
        }
        // (u, v) ← (A·u + B·v, C·u + D·v): two consecutive remainders of
        // Euclid's sequence (the matrix is a product of the true quotient
        // steps, unimodular, so the gcd is unchanged in any case).
        v.resize(u.len(), 0);
        if !lehmer_apply(&mut u, &mut v, coeffs) {
            // Not reached (the quotients are exact); start over from the
            // values with BigUint arithmetic, which is always correct.
            return uint_gcd_simple(a, b);
        }
        limbs_trim(&mut u);
        limbs_trim(&mut v);
        if limbs_cmp(&u, &v) == std::cmp::Ordering::Less {
            std::mem::swap(&mut u, &mut v);
        }
    }
}

/// The Euclidean algorithm with `num-integer`'s gcd, the fallback of
/// [`uint_gcd`].
fn uint_gcd_simple(a: &BigUint, b: &BigUint) -> BigUint {
    a.gcd(b)
}

/// `(u, v) ← (A·u + B·v, C·u + D·v)` in place, for limb vectors of the same
/// length and cofactors below `2⁶²` in magnitude; `false` (operands
/// unspecified) if a result would be negative or overflow, which a Lehmer
/// cosequence rules out.
fn lehmer_apply(u: &mut [u64], v: &mut [u64], [a, b, c, d]: [i64; 4]) -> bool {
    let (a, b, c, d) = (i128::from(a), i128::from(b), i128::from(c), i128::from(d));
    let (mut cu, mut cv) = (0i128, 0i128);
    for (ui, vi) in u.iter_mut().zip(v.iter_mut()) {
        let (x, y) = (i128::from(*ui), i128::from(*vi));
        let su = a * x + b * y + cu;
        let sv = c * x + d * y + cv;
        *ui = su as u64;
        *vi = sv as u64;
        cu = su >> 64;
        cv = sv >> 64;
    }
    cu == 0 && cv == 0
}

fn limbs_trim(x: &mut Vec<u64>) {
    while x.last() == Some(&0) {
        x.pop();
    }
}

fn limbs_bits(x: &[u64]) -> u64 {
    x.last().map_or(0, |&top| {
        64 * (x.len() as u64 - 1) + (64 - u64::from(top.leading_zeros()))
    })
}

/// The 64 bits of `x` starting at bit `shift`.
fn limbs_window(x: &[u64], shift: u64) -> u64 {
    let (w, off) = ((shift / 64) as usize, (shift % 64) as u32);
    let lo = x.get(w).copied().unwrap_or(0) >> off;
    let hi = if off == 0 {
        0
    } else {
        x.get(w + 1).copied().unwrap_or(0) << (64 - off)
    };
    lo | hi
}

fn limbs_cmp(x: &[u64], y: &[u64]) -> std::cmp::Ordering {
    x.len()
        .cmp(&y.len())
        .then_with(|| x.iter().rev().cmp(y.iter().rev()))
}

fn limbs_to_big(x: &[u64]) -> BigUint {
    let digits: Vec<u32> = x
        .iter()
        .flat_map(|&w| [w as u32, (w >> 32) as u32])
        .collect();
    BigUint::new(digits)
}

/// The cosequence `[A, B, C, D]` of Lehmer's inner loop for the leading
/// 62 bits `x ≥ y` (taken with the same shift) of `u ≥ v`: Euclid's
/// quotients of `x/y` are emulated as long as they provably agree with
/// those of `u/v` (Knuth's test with `(x + A)/(y + C)` and
/// `(x + B)/(y + D)`).  `B = 0` means no quotient was certain.  Every
/// cosequence value is at most `x < 2⁶²` in magnitude (Knuth), so the
/// sums fit an `i64` and the updates, whose results do, are computed with
/// wrapping products.
fn lehmer_cosequence(x: u64, y: u64) -> [i64; 4] {
    let (mut a, mut b, mut c, mut d) = (1i64, 0i64, 0i64, 1i64);
    let (mut x, mut y) = (x as i64, y as i64);
    loop {
        let (yc, yd) = (y + c, y + d);
        if yc <= 0 || yd <= 0 {
            break;
        }
        let (xa, xb) = (x + a, x + b);
        if xa < 0 || xb < 0 {
            break;
        }
        let q = xa / yc;
        if q != xb / yd {
            break;
        }
        (a, c) = (c, a.wrapping_sub(q.wrapping_mul(c)));
        (b, d) = (d, b.wrapping_sub(q.wrapping_mul(d)));
        (x, y) = (y, x.wrapping_sub(q.wrapping_mul(y)));
        if [a, b, c, d].iter().any(|t| t.unsigned_abs() >= 1u64 << 62) {
            // Outside the proven range (not reached): stop with the last
            // step undone by the caller's bound check.
            break;
        }
    }
    [a, b, c, d]
}

fn gcd_u128(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

// ═══════════════════════════════════════════════════════════════════════════
// Word-size primes
// ═══════════════════════════════════════════════════════════════════════════

/// The primes below `2⁶²` found so far, descending.
fn prime_cache() -> &'static Mutex<Vec<u64>> {
    static CACHE: OnceLock<Mutex<Vec<u64>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(Vec::new()))
}

/// The largest primes below `2⁶²`, descending (SymPy `prevprime`; the
/// cache below continues the sequence from the last one).
const FIRST_PRIMES: [u64; 8] = [
    4_611_686_018_427_387_847,
    4_611_686_018_427_387_817,
    4_611_686_018_427_387_787,
    4_611_686_018_427_387_761,
    4_611_686_018_427_387_751,
    4_611_686_018_427_387_737,
    4_611_686_018_427_387_733,
    4_611_686_018_427_387_709,
];

/// The `i`-th prime below `2⁶²` in descending order.
pub(crate) fn prime(i: usize) -> u64 {
    if let Some(&p) = FIRST_PRIMES.get(i) {
        return p;
    }
    let mut cache = match prime_cache().lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    let j = i - FIRST_PRIMES.len();
    while cache.len() <= j {
        let last = cache
            .last()
            .copied()
            .unwrap_or(FIRST_PRIMES[FIRST_PRIMES.len() - 1]);
        let mut n = last - 1;
        if n.is_multiple_of(2) {
            n -= 1;
        }
        while !is_prime_u64(n) {
            n -= 2;
        }
        cache.push(n);
    }
    cache[j]
}

/// Deterministic Miller–Rabin for `n < 2⁶⁴` (the seven bases of Jaeschke and
/// Sinclair are a certificate in that range).
fn is_prime_u64(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for p in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if n.is_multiple_of(p) {
            return n == p;
        }
    }
    let mul = |a: u64, b: u64| ((u128::from(a) * u128::from(b)) % u128::from(n)) as u64;
    let pow = |mut b: u64, mut e: u64| {
        let mut r = 1u64;
        while e > 0 {
            if e & 1 == 1 {
                r = mul(r, b);
            }
            b = mul(b, b);
            e >>= 1;
        }
        r
    };
    let s = (n - 1).trailing_zeros();
    let d = (n - 1) >> s;
    'bases: for a in [2u64, 325, 9375, 28178, 450_775, 9_780_504, 1_795_265_022] {
        let a = a % n;
        if a == 0 {
            continue;
        }
        let mut x = pow(a, d);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 1..s {
            x = mul(x, x);
            if x == n - 1 {
                continue 'bases;
            }
        }
        return false;
    }
    true
}

/// `c mod p` in `[0, p)`.
pub(crate) fn residue(c: &BigInt, p: u64) -> u64 {
    let r = (c.magnitude() % p).to_u64().unwrap_or(0);
    if r != 0 && c.is_negative() { p - r } else { r }
}

// ═══════════════════════════════════════════════════════════════════════════
// ℤ[x]
// ═══════════════════════════════════════════════════════════════════════════

/// Drop trailing zeros.
fn trim(c: &mut Vec<BigInt>) {
    while c.last().is_some_and(Zero::is_zero) {
        c.pop();
    }
}

/// `f / h` in `ℤ[x]` (ascending, normalised, `h ≠ 0`), `None` unless the
/// division is exact with an integer quotient.
pub(crate) fn zx_div_exact(f: &[BigInt], h: &[BigInt]) -> Option<Vec<BigInt>> {
    let lc = h.last()?;
    let m = h.len() - 1;
    let mut r = f.to_vec();
    trim(&mut r);
    if r.is_empty() {
        return Some(Vec::new());
    }
    if r.len() < h.len() {
        return None;
    }
    // Cheap necessary conditions first: the constant terms.
    if !h[0].is_zero() && !(&r[0] % &h[0]).is_zero() {
        return None;
    }
    let mut q = vec![BigInt::zero(); r.len() - m];
    while r.len() > m {
        let k = r.len() - 1;
        let (qk, rem) = r[k].div_rem(lc);
        if !rem.is_zero() {
            return None;
        }
        r.pop();
        let s = k - m;
        if !qk.is_zero() {
            for (rj, hj) in r[s..].iter_mut().zip(&h[..m]) {
                *rj -= &qk * hj;
            }
        }
        q[s] = qk;
    }
    trim(&mut r);
    r.is_empty().then(|| {
        trim(&mut q);
        q
    })
}

/// The monic gcd over `𝔽ₚ` of the images of `f` and `g` (ascending,
/// normalised), or `None` when `p` divides both leading coefficients.
fn gcd_image(f: &[BigInt], g: &[BigInt], p: u64) -> Option<PolyIn<Fp64>> {
    let field = Fp64::new(p);
    let fp = PolyIn::from_coeffs(field, f.iter().map(|c| residue(c, p)).collect());
    let gp = PolyIn::from_coeffs(field, g.iter().map(|c| residue(c, p)).collect());
    if fp.degree() != Some(f.len() - 1) && gp.degree() != Some(g.len() - 1) {
        return None;
    }
    Some(fp.gcd(&gp))
}

/// Primitive part with a positive leading coefficient.
fn primitive_positive(mut c: Vec<BigInt>) -> Vec<BigInt> {
    trim(&mut c);
    let g = content(&c);
    if !g.is_zero() && !g.is_one() {
        for x in &mut c {
            *x /= &g;
        }
    }
    if c.last().is_some_and(Signed::is_negative) {
        for x in &mut c {
            *x = -std::mem::take(x);
        }
    }
    c
}

/// `gcd(f, g)` in `ℤ[x]` for non-zero `f`, `g` (ascending): primitive with
/// a positive leading coefficient, `[1]` when they are coprime — exactly
/// the gcd any algorithm returns under that normalisation — by Brown's
/// modular algorithm (see the module documentation).  `None` when the
/// number of primes exceeds a generous cap without a verified result
/// (which the theory rules out: only finitely many primes are unlucky);
/// the caller then runs its remainder sequence.
pub(crate) fn zx_gcd(f: &[BigInt], g: &[BigInt]) -> Option<Vec<BigInt>> {
    let f = primitive_positive(f.to_vec());
    let g = primitive_positive(g.to_vec());
    if f.is_empty() || g.is_empty() {
        return None;
    }
    if f.len() == 1 || g.len() == 1 {
        return Some(vec![BigInt::one()]);
    }
    let lcf = f.last()?;
    let lcg = g.last()?;
    let gamma = int_gcd(lcf, lcg);
    // Landau–Mignotte: a factor `h` of `f` of degree `d` has
    // `‖h‖∞ ≤ 2^d·|lc h/lc f|·‖f‖₂`; the images reconstruct `(γ/lc h)·h`,
    // so its coefficients have at most `bits(γ) + d + bits(‖f‖₂) − bits(lc f)`
    // bits (taking the smaller of the two inputs' bounds).
    let norm_bits = |c: &[BigInt]| {
        let lc_bits = c.last().map_or(0, BigInt::bits);
        let max_bits = c.iter().map(BigInt::bits).max().unwrap_or(0);
        // ‖c‖₂ ≤ √(n+1)·‖c‖∞.
        let sqrt_len = (64 - (c.len() as u64).leading_zeros()) as u64;
        max_bits + sqrt_len - lc_bits.min(max_bits)
    };
    let d_max = (f.len().min(g.len()) - 1) as u64;
    let h_bits = gamma.bits() + d_max + norm_bits(&f).min(norm_bits(&g)) + 2;
    let needed = (h_bits / 61 + 2) as usize;
    let cap = 2 * needed + 64;

    // CRT state: the symmetric residues `hs` modulo `m` of the scaled gcd.
    let mut bound = usize::MAX;
    let mut hs: Vec<BigInt> = Vec::new();
    let mut m = BigInt::one();
    let mut tried = 0usize;
    let mut i = 0usize;
    while tried < cap {
        let p = prime(i);
        i += 1;
        if residue(&gamma, p) == 0 {
            continue;
        }
        tried += 1;
        let Some(hp) = gcd_image(&f, &g, p) else {
            continue;
        };
        let Some(deg) = hp.degree() else {
            continue;
        };
        if deg == 0 {
            return Some(vec![BigInt::one()]);
        }
        if deg > bound {
            continue;
        }
        // The image of `(γ/lc h)·h`: the monic image times `γ`.
        let field = *hp.ring();
        let gp = residue(&gamma, p);
        let img: Vec<u64> = hp.coeffs().iter().map(|&c| field.mul(&c, &gp)).collect();
        if deg < bound {
            bound = deg;
            m = BigInt::from(p);
            hs = img.iter().map(|&r| symmetric(r, p)).collect();
            if let Some(h) = verified(&f, &g, &hs) {
                return Some(h);
            }
            continue;
        }
        // Combine: x ≡ hs (mod m), x ≡ img (mod p).
        let m_inv = field.inv(&residue(&m, p)).unwrap_or(0);
        let mp = &m * BigInt::from(p);
        let half = &mp >> 1u32;
        let mut stable = true;
        for (h, &r) in hs.iter_mut().zip(&img) {
            let delta = field.mul(&field.sub(&r, &residue(h, p)), &m_inv);
            if delta != 0 {
                stable = false;
                *h += &m * BigInt::from(delta);
                if *h > half {
                    *h -= &mp;
                }
            }
        }
        m = mp;
        if stable && let Some(h) = verified(&f, &g, &hs) {
            return Some(h);
        }
    }
    None
}

/// `r` in `(−p/2, p/2]`.
fn symmetric(r: u64, p: u64) -> BigInt {
    if r > p / 2 {
        BigInt::from(r) - BigInt::from(p)
    } else {
        BigInt::from(r)
    }
}

/// The primitive part of the candidate `hs` when it divides `f` and `g`.
fn verified(f: &[BigInt], g: &[BigInt], hs: &[BigInt]) -> Option<Vec<BigInt>> {
    let h = primitive_positive(hs.to_vec());
    if h.len() < 2 {
        return None;
    }
    zx_div_exact(f, &h)?;
    zx_div_exact(g, &h)?;
    Some(h)
}

// ═══════════════════════════════════════════════════════════════════════════
// Multivariate coprimality
// ═══════════════════════════════════════════════════════════════════════════

/// Number of primes [`coprime_certified`] tries before giving up.
const CERTIFY_PRIMES: usize = 2;
/// Evaluation points per variable and prime whose leading coefficients
/// vanish before [`coprime_certified`] gives up on that prime.
const CERTIFY_POINTS: usize = 3;

/// Is `gcd(f, g)` in `ℤ[x₁, …, xₙ]` certainly a constant?  `f` and `g` are
/// non-zero, given as `(exponents, coefficient)` terms in `nv` variables.
///
/// A non-constant gcd `h` involves some variable `v` that occurs in both
/// inputs.  For such a `v`, substitute values `a` for the other variables
/// and reduce modulo a prime `p`: when the coefficient of the top power of
/// `v` in `f` (or `g`) does not vanish at `a` modulo `p`, neither does the
/// leading coefficient of `h` in `v` (leading coefficients multiply), so
/// the image of `h` is a common factor of the images of `f` and `g` with
/// the same positive degree in `v`.  An image gcd of degree 0 therefore
/// proves `deg_v h = 0`; when that holds for every common variable, `h` is
/// a constant (Brown's degree argument, one variable at a time).  `false`
/// means "not certified" — a common factor, or unlucky images.
pub(crate) fn coprime_certified<'a>(
    nv: usize,
    f: &[(&'a [u32], &'a BigInt)],
    g: &[(&'a [u32], &'a BigInt)],
) -> bool {
    let degs = |t: &[(&'a [u32], &'a BigInt)]| -> Vec<u32> {
        (0..nv)
            .map(|v| t.iter().map(|(e, _)| e[v]).max().unwrap_or(0))
            .collect()
    };
    let (df, dg) = (degs(f), degs(g));
    let common: Vec<usize> = (0..nv).filter(|&v| df[v] > 0 && dg[v] > 0).collect();
    if common.is_empty() {
        return true;
    }
    let mut rng = SplitMix64::new(0x5EED_6CD0_0000_0001);
    'primes: for i in 0..CERTIFY_PRIMES {
        let p = prime(i);
        let field = Fp64::new(p);
        let fr: Vec<u64> = f.iter().map(|(_, c)| residue(c, p)).collect();
        let gr: Vec<u64> = g.iter().map(|(_, c)| residue(c, p)).collect();
        'vars: for &v in &common {
            for _ in 0..CERTIFY_POINTS {
                let point: Vec<u64> = (0..nv).map(|_| 1 + rng.next_u64() % (p - 1)).collect();
                let image = |t: &[(&'a [u32], &'a BigInt)], r: &[u64], deg: u32| -> Vec<u64> {
                    let mut out = vec![0u64; deg as usize + 1];
                    for ((e, _), &c) in t.iter().zip(r) {
                        if c == 0 {
                            continue;
                        }
                        let mut val = c;
                        for (j, &ej) in e.iter().enumerate() {
                            if j != v && ej > 0 {
                                val = field.mul(&val, &field.pow(point[j], u64::from(ej)));
                            }
                        }
                        let slot = &mut out[e[v] as usize];
                        *slot = field.add(slot, &val);
                    }
                    out
                };
                let fi = image(f, &fr, df[v]);
                let gi = image(g, &gr, dg[v]);
                if fi.last() == Some(&0) && gi.last() == Some(&0) {
                    continue;
                }
                let fp = PolyIn::from_coeffs(field, fi);
                let gp = PolyIn::from_coeffs(field, gi);
                if fp.gcd(&gp).degree() == Some(0) {
                    continue 'vars;
                }
                // A common factor of the images with a valid leading
                // coefficient: very likely a true common factor.
                return false;
            }
            continue 'primes;
        }
        return true;
    }
    false
}

// ═══════════════════════════════════════════════════════════════════════════
// Heuristic gcd in ℤ[x₁, …, xₙ]
// ═══════════════════════════════════════════════════════════════════════════

/// A sparse polynomial with integer coefficients, keyed by exponent
/// vectors in lexicographic order (the last variable least significant);
/// no zero coefficients.  The working form of [`heu_gcd`].
pub(crate) type ZMap = std::collections::BTreeMap<Vec<u32>, BigInt>;

/// Maximum number of evaluation points tried by [`heu_gcd`] at one level.
const HEU_GCD_MAX_TRIES: usize = 6;

/// A gcd with its cofactors: `f = h·cff`, `g = h·cfg`.
#[derive(Debug, PartialEq)]
pub(crate) struct HeuGcd {
    /// The gcd.
    pub(crate) h: ZMap,
    /// `f / h`.
    pub(crate) cff: ZMap,
    /// `g / h`.
    pub(crate) cfg: ZMap,
}

/// Outcome of one evaluation point of the heuristic gcd.
enum Attempt {
    /// The gcd and cofactors, verified.
    Found(HeuGcd),
    /// This point did not work; try the next one.
    Retry,
    /// The recursion below failed: give up the heuristic.
    Failed,
}

/// The heuristic gcd of non-zero `f`, `g` in `nv` variables: a [`HeuGcd`]
/// whose `h` is the gcd including the gcd of the integer contents (its sign
/// is not normalised), or `None` when the heuristic fails.
///
/// SymPy's `dmp_zz_heu_gcd` (`sympy/polys/euclidtools.py`, BSD-3; Char,
/// Geddes & Gonnet, "GCDHEU", *J. Symbolic Comput.* 7 (1989)) on integer
/// coefficients throughout: evaluate the last variable at
/// `ξ = max(min(B, 99√B), 2·min(‖f‖/|lc f|, ‖g‖/|lc g|) + 4)` with
/// `B = 2·min(‖f‖, ‖g‖) + 29`, recurse down to an integer gcd, reconstruct
/// by symmetric `ξ`-adic expansion and verify each of the three candidates
/// (the gcd and the quotients by the two cofactors) by exact division; a
/// failure below fails the whole attempt.  The recursion depth is `nv`.
pub(crate) fn heu_gcd(nv: usize, f: &ZMap, g: &ZMap) -> Option<HeuGcd> {
    let cf = content(f.values());
    let cg = content(g.values());
    if cf.is_zero() || cg.is_zero() {
        return None;
    }
    let c = int_gcd(&cf, &cg);
    let f = zmap_div_scalar(f, &c);
    let g = zmap_div_scalar(g, &c);
    let one = || ZMap::from([(vec![0u32; nv], BigInt::one())]);
    let constant_of = |p: &ZMap| -> Option<BigInt> {
        match p.len() {
            1 => p.get(&vec![0u32; nv]).cloned(),
            _ => None,
        }
    };
    if nv == 0 || constant_of(&f).is_some() || constant_of(&g).is_some() {
        // An integer gcd: of the values, or of the contents when one input
        // is a constant.
        let k = int_gcd(&content(f.values()), &content(g.values()));
        if k.is_zero() {
            return None;
        }
        let h = ZMap::from([(vec![0u32; nv], &k * &c)]);
        return Some(HeuGcd {
            h,
            cff: zmap_div_scalar(&f, &k),
            cfg: zmap_div_scalar(&g, &k),
        });
    }
    // Cheap exact-division shortcuts.
    if let Some(q) = zmap_div_exact(&g, &f) {
        return Some(HeuGcd {
            h: zmap_scale(&f, &c),
            cff: one(),
            cfg: q,
        });
    }
    if let Some(q) = zmap_div_exact(&f, &g) {
        return Some(HeuGcd {
            h: zmap_scale(&g, &c),
            cff: q,
            cfg: one(),
        });
    }
    let f_norm = max_norm(&f);
    let g_norm = max_norm(&g);
    let two = BigInt::from(2);
    let b = &two * f_norm.clone().min(g_norm.clone()) + BigInt::from(29);
    let lc_ratio = |norm: &BigInt, p: &ZMap| -> BigInt {
        let lc = p
            .last_key_value()
            .map(|(_, c)| c.abs())
            .filter(|c| !c.is_zero())
            .unwrap_or_else(BigInt::one);
        norm / lc
    };
    let mut xi = (b.clone().min(BigInt::from(99) * b.sqrt()))
        .max(&two * lc_ratio(&f_norm, &f).min(lc_ratio(&g_norm, &g)) + BigInt::from(4));
    for _ in 0..HEU_GCD_MAX_TRIES {
        match heu_attempt(nv, &f, &g, &xi) {
            Attempt::Found(found) => {
                return Some(HeuGcd {
                    h: zmap_scale(&found.h, &c),
                    ..found
                });
            }
            Attempt::Retry => {}
            Attempt::Failed => return None,
        }
        // `73794·ξ·ξ^(1/4)/27011` (grows like `ξ^1.25`).
        let root4 = xi.sqrt().sqrt().max(BigInt::from(2));
        xi = (BigInt::from(73794) * &xi * root4) / BigInt::from(27011);
    }
    None
}

/// One evaluation/interpolation round of [`heu_gcd`] at `ξ = xi` for inputs
/// without common integer content.
fn heu_attempt(nv: usize, f: &ZMap, g: &ZMap, xi: &BigInt) -> Attempt {
    let ff = eval_last(nv, f, xi);
    let gg = eval_last(nv, g, xi);
    if ff.is_empty() || gg.is_empty() {
        return Attempt::Retry;
    }
    let Some(HeuGcd { h, cff, cfg }) = heu_gcd(nv - 1, &ff, &gg) else {
        return Attempt::Failed;
    };
    // Candidate 1: the interpolated gcd, made primitive.
    let h = interpolate(&h, xi);
    let ch = content(h.values());
    if !h.is_empty() && !ch.is_zero() {
        let h = zmap_div_scalar(&h, &ch);
        if let Some(cf) = zmap_div_exact(f, &h)
            && let Some(cg) = zmap_div_exact(g, &h)
        {
            return Attempt::Found(HeuGcd {
                h,
                cff: cf,
                cfg: cg,
            });
        }
    }
    // Candidate 2: f divided by the interpolated cofactor of f.
    let cff = interpolate(&cff, xi);
    if !cff.is_empty()
        && let Some(h) = zmap_div_exact(f, &cff)
        && let Some(cg) = zmap_div_exact(g, &h)
    {
        return Attempt::Found(HeuGcd { h, cff, cfg: cg });
    }
    // Candidate 3: g divided by the interpolated cofactor of g.
    let cfg = interpolate(&cfg, xi);
    if !cfg.is_empty()
        && let Some(h) = zmap_div_exact(g, &cfg)
        && let Some(cf) = zmap_div_exact(f, &h)
    {
        return Attempt::Found(HeuGcd { h, cff: cf, cfg });
    }
    Attempt::Retry
}

/// `p(…, ξ)`: the last of the `nv ≥ 1` variables evaluated at `xi` (the
/// result has `nv − 1` variables).
fn eval_last(nv: usize, p: &ZMap, xi: &BigInt) -> ZMap {
    let top = p.keys().map(|e| e[nv - 1]).max().unwrap_or(0) as usize;
    let mut pows = Vec::with_capacity(top + 1);
    let mut acc = BigInt::one();
    for _ in 0..=top {
        pows.push(acc.clone());
        acc *= xi;
    }
    let mut out = ZMap::new();
    for (e, c) in p {
        let term = c * &pows[e[nv - 1] as usize];
        *out.entry(e[..nv - 1].to_vec()).or_default() += term;
    }
    out.retain(|_, c| !c.is_zero());
    out
}

/// The polynomial in one more (last) variable whose coefficients are the
/// symmetric `ξ`-adic digits of those of `h`: `h = Σᵢ gᵢ·ξⁱ` with every
/// coefficient of `gᵢ` in `(−ξ/2, ξ/2]`.
fn interpolate(h: &ZMap, xi: &BigInt) -> ZMap {
    let mut out = ZMap::new();
    for (e, c) in h {
        let mut rest = c.clone();
        let mut i = 0u32;
        while !rest.is_zero() {
            let mut d = rest.mod_floor(xi);
            if &d + &d > *xi {
                d -= xi;
            }
            rest -= &d;
            rest /= xi;
            if !d.is_zero() {
                let mut k = Vec::with_capacity(e.len() + 1);
                k.extend_from_slice(e);
                k.push(i);
                out.insert(k, d);
            }
            i += 1;
        }
    }
    out
}

fn max_norm(p: &ZMap) -> BigInt {
    p.values().map(Signed::abs).max().unwrap_or_default()
}

fn zmap_scale(p: &ZMap, c: &BigInt) -> ZMap {
    if c.is_one() {
        return p.clone();
    }
    p.iter().map(|(e, x)| (e.clone(), x * c)).collect()
}

/// `p / c` for a divisor `c ≠ 0` of every coefficient.
fn zmap_div_scalar(p: &ZMap, c: &BigInt) -> ZMap {
    if c.is_one() {
        return p.clone();
    }
    p.iter().map(|(e, x)| (e.clone(), x / c)).collect()
}

/// `f / h` in `ℤ[x₁, …, xₙ]` (`h ≠ 0`), `None` unless exact with integer
/// coefficients.  Lexicographic division by the leading term of `h`; the
/// trailing terms and the partial degrees are checked first, and a
/// remainder term above a partial degree of `f` ends the division (every
/// term of `q·h` for a part `q` of an exact quotient stays within them).
pub(crate) fn zmap_div_exact(f: &ZMap, h: &ZMap) -> Option<ZMap> {
    let (he, hc) = h.last_key_value()?;
    let Some((fe0, fc0)) = f.first_key_value() else {
        return Some(ZMap::new());
    };
    let (he0, hc0) = h.first_key_value()?;
    if fe0.iter().zip(he0).any(|(a, b)| a < b) || !(fc0 % hc0).is_zero() {
        return None;
    }
    let nv = he.len();
    let deg = |p: &ZMap| -> Vec<u32> {
        (0..nv)
            .map(|v| p.keys().map(|e| e[v]).max().unwrap_or(0))
            .collect()
    };
    let (df, dh) = (deg(f), deg(h));
    if df.iter().zip(&dh).any(|(a, b)| a < b) {
        return None;
    }
    let mut r = f.clone();
    let mut q = ZMap::new();
    while let Some((e, c)) = r.last_key_value() {
        if e.iter().zip(he).any(|(a, b)| a < b) {
            return None;
        }
        let qe: Vec<u32> = e.iter().zip(he).map(|(a, b)| a - b).collect();
        let (qc, rem) = c.div_rem(hc);
        if !rem.is_zero() {
            return None;
        }
        for (e2, c2) in h {
            let k: Vec<u32> = e2.iter().zip(&qe).map(|(a, b)| a + b).collect();
            if k.iter().zip(&df).any(|(a, b)| a > b) {
                return None;
            }
            let t = &qc * c2;
            match r.entry(k) {
                std::collections::btree_map::Entry::Occupied(mut slot) => {
                    *slot.get_mut() -= t;
                    if slot.get().is_zero() {
                        slot.remove();
                    }
                }
                std::collections::btree_map::Entry::Vacant(slot) => {
                    slot.insert(-t);
                }
            }
        }
        q.insert(qe, qc);
    }
    Some(q)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b(s: &str) -> BigInt {
        s.parse().unwrap()
    }

    #[test]
    fn lehmer_matches_binary_gcd() {
        let mut rng = SplitMix64::new(7);
        let mut big = |words: usize| -> BigInt {
            let mut x = BigInt::zero();
            for _ in 0..words {
                x = (x << 64u32) + BigInt::from(rng.next_u64());
            }
            x
        };
        for (wa, wb, wg) in [
            (1, 1, 0),
            (3, 2, 1),
            (5, 5, 2),
            (40, 40, 7),
            (60, 3, 1),
            (2, 70, 30),
        ] {
            let g: BigInt = big(wg) + 1;
            let a: BigInt = big(wa) * &g;
            let bb: BigInt = -(big(wb) * &g);
            assert_eq!(int_gcd(&a, &bb), a.gcd(&bb), "{wa} {wb} {wg}");
        }
        assert_eq!(int_gcd(&b("0"), &b("-12")), b("12"));
        assert_eq!(int_gcd(&b("0"), &b("0")), b("0"));
        let f = b("2").pow(4000u32) - 1;
        let g = b("2").pow(2600u32) - 1;
        assert_eq!(int_gcd(&f, &g), b("2").pow(200u32) - 1);
        // Consecutive Fibonacci numbers: all quotients 1.
        let (mut x, mut y) = (b("1"), b("1"));
        for _ in 0..3000 {
            (x, y) = (y.clone(), x + y);
        }
        assert_eq!(int_gcd(&x, &y), b("1"));
    }

    #[test]
    fn rational_arithmetic_matches_ratio() {
        use num_rational::Ratio;
        let mut rng = SplitMix64::new(11);
        let mut num = |words: usize| -> BigInt {
            let mut x = BigInt::from(rng.next_u64() % 1000);
            for _ in 0..words {
                x = (x << 64u32) + BigInt::from(rng.next_u64());
            }
            if rng.next_u64().is_multiple_of(2) {
                -x
            } else {
                x
            }
        };
        let mut vals: Vec<Ratio<BigInt>> = Vec::new();
        for w in [0usize, 1, 3, 9] {
            let common = num(w).abs() + 1;
            vals.push(Ratio::from_integer(num(w)));
            vals.push(Ratio::new(num(w) * &common, num(w).abs() + 1));
            vals.push(Ratio::new(num(w), (num(w).abs() + 1) * &common));
        }
        vals.push(Ratio::zero());
        for a in &vals {
            for b in &vals {
                let (s, p) = (rat_add(a, b), rat_mul(a, b));
                let (s0, p0) = (a + b, a * b);
                assert_eq!(
                    (s.numer(), s.denom()),
                    (s0.numer(), s0.denom()),
                    "{a} + {b}"
                );
                assert_eq!(
                    (p.numer(), p.denom()),
                    (p0.numer(), p0.denom()),
                    "{a} * {b}"
                );
            }
        }
        assert_eq!(
            ratio_reduced(BigInt::from(6), BigInt::from(-4)),
            Ratio::new(BigInt::from(-3), BigInt::from(2))
        );
        assert_eq!(
            int_lcm(&BigInt::from(-6), &BigInt::from(4)),
            BigInt::from(12)
        );
    }

    #[test]
    fn content_and_primes() {
        assert_eq!(content(&[b("12"), b("-18"), b("0"), b("30")]), b("6"));
        assert_eq!(content(&[]), b("0"));
        assert_eq!(content(&[b("0")]), b("0"));
        assert_eq!(prime(0), (1u64 << 62) - 57);
        // The table and the cache continue each other without a gap.
        for i in 0..12 {
            assert!(is_prime_u64(prime(i)), "{i}");
            assert!(prime(i + 1) < prime(i));
            assert!(
                (prime(i + 1) + 1..prime(i))
                    .step_by(2)
                    .all(|n| !is_prime_u64(n))
            );
        }
        assert_eq!(gcd_u64(0, 12), 12);
        assert_eq!(gcd_u64(48, 180), 12);
        assert_eq!(
            int_gcd(
                &b("-340282366920938463463374607431768211455"),
                &b("18446744073709551615")
            ),
            b("18446744073709551615")
        );
        assert!(!is_prime_u64(3_215_031_751) && is_prime_u64(2_305_843_009_213_693_951));
    }

    fn zx(cs: &[i64]) -> Vec<BigInt> {
        cs.iter().map(|&c| BigInt::from(c)).collect()
    }

    #[test]
    fn modular_gcd_small() {
        // (x − 1)(2x + 3) and (x − 1)(x + 5)(x²+1)
        let f = zx(&[-3, 1, 2]);
        let g = zx(&[-5, 4, -4, 4, 1]);
        assert_eq!(zx_gcd(&f, &g), Some(zx(&[-1, 1])));
        assert_eq!(zx_gcd(&zx(&[1, 1]), &zx(&[2, 1])), Some(zx(&[1])));
        // gcd(6x² − 6, −4x − 4) = x + 1 (primitive, positive).
        assert_eq!(zx_gcd(&zx(&[-6, 0, 6]), &zx(&[-4, -4])), Some(zx(&[1, 1])));
        assert_eq!(
            zx_div_exact(&zx(&[-1, 0, 1]), &zx(&[1, 1])),
            Some(zx(&[-1, 1]))
        );
        assert_eq!(zx_div_exact(&zx(&[-1, 0, 1]), &zx(&[1, 2])), None);
    }

    #[test]
    fn modular_gcd_huge_coefficients() {
        let a = b("3").pow(3000u32) + 7;
        let c = b("5").pow(2000u32) - 3;
        let d = b("7").pow(1500u32) + 1;
        // (x + a)(c·x − d)(x + 1) and (x + a)(c·x − d)(x − 1)²
        let lin = |k: &BigInt, l: &BigInt| vec![k.clone(), l.clone()];
        let mul = |p: &[BigInt], q: &[BigInt]| {
            let mut out = vec![BigInt::zero(); p.len() + q.len() - 1];
            for (i, x) in p.iter().enumerate() {
                for (j, y) in q.iter().enumerate() {
                    out[i + j] += x * y;
                }
            }
            out
        };
        let h = mul(&lin(&a, &b("1")), &lin(&-&d, &c));
        let f = mul(&h, &zx(&[1, 1]));
        let g = mul(&mul(&h, &zx(&[-1, 1])), &zx(&[-1, 1]));
        assert_eq!(zx_gcd(&f, &g), Some(primitive_positive(h)));
    }

    /// `(x + 1)^k` as a [`ZMap`] in one variable.
    fn zmap_univariate(cs: &[i64]) -> ZMap {
        cs.iter()
            .enumerate()
            .filter(|(_, c)| **c != 0)
            .map(|(i, &c)| (vec![i as u32], BigInt::from(c)))
            .collect()
    }

    #[test]
    fn heuristic_retry() {
        // gcd = (x + 1)⁸ has a coefficient 70, but the inputs have max-norms
        // 28 and 112, so at the evaluation point ξ = 2·28 + 29 = 85 the GCD
        // cannot be recovered from its value (70 is not a symmetric digit
        // mod 85), nor can the cofactors (gcd(f(85), g(85)) carries the
        // spurious factor gcd(84, 7226) = 2): the attempt must fail and the
        // next ξ must recover the answer.
        let h = zmap_univariate(&[1, 8, 28, 56, 70, 56, 28, 8, 1]);
        let cf = zmap_univariate(&[-1, 1]);
        let cg = zmap_univariate(&[1, 0, 1]);
        let mul = |a: &ZMap, b: &ZMap| -> ZMap {
            let mut out = ZMap::new();
            for (ea, ca) in a {
                for (eb, cb) in b {
                    *out.entry(vec![ea[0] + eb[0]]).or_default() += ca * cb;
                }
            }
            out.retain(|_, c| !c.is_zero());
            out
        };
        let f = mul(&h, &cf);
        let g = mul(&h, &cg);
        assert_eq!(max_norm(&f), BigInt::from(28));
        assert_eq!(max_norm(&g), BigInt::from(112));
        assert!(matches!(
            heu_attempt(1, &f, &g, &BigInt::from(85)),
            Attempt::Retry
        ));
        match heu_attempt(1, &f, &g, &BigInt::from(696)) {
            Attempt::Found(found) => {
                let expected = HeuGcd {
                    h: h.clone(),
                    cff: cf.clone(),
                    cfg: cg.clone(),
                };
                assert_eq!(found, expected);
            }
            _ => panic!("ξ = 696 should recover the GCD"),
        }
        assert_eq!(heu_gcd(1, &f, &g).unwrap().h, h);
        assert_eq!(
            zmap_div_exact(&f, &h),
            Some(mul(&cf, &zmap_univariate(&[1])))
        );
        assert_eq!(zmap_div_exact(&f, &cg), None);
    }

    #[test]
    fn multivariate_certificate() {
        // f = x·y + 1, g = x + y: coprime; f·(x − y), g·(x − y): not.
        let e = |a: u32, b: u32| vec![a, b];
        let one = b("1");
        let m1 = b("-1");
        let (e11, e00, e10, e01) = (e(1, 1), e(0, 0), e(1, 0), e(0, 1));
        let f = [(&e11[..], &one), (&e00[..], &one)];
        let g = [(&e10[..], &one), (&e01[..], &one)];
        assert!(coprime_certified(2, &f, &g));
        let (e20, e02) = (e(2, 0), e(0, 2));
        // g·(x − y) = x² − y²; f and x² − y² are coprime; x + y and x² − y² are not.
        let g2 = [(&e20[..], &one), (&e02[..], &m1)];
        assert!(coprime_certified(2, &f, &g2));
        assert!(!coprime_certified(2, &g, &g2));
        // Disjoint variables.
        let fx = [(&e10[..], &one), (&e00[..], &one)];
        let gy = [(&e01[..], &one), (&e00[..], &one)];
        assert!(coprime_certified(2, &fx, &gy));
    }
}
