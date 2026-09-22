//! The fraction-free (Bareiss) integer kernel shared by the exact matrices
//! ([`exact_matrix`](super::exact_matrix)), the polytope code and the
//! simplex tableau ([`linprog`](super::linprog)).
//!
//! Everything here works on a **cell** type ([`Cell`]): a machine integer
//! with overflow detection (`i64`, `i128`, the 256-bit [`W256`]) or
//! `BigInt`.  The one operation every fraction-free algorithm repeats is
//! the pivot row update `row ← (row·p − row[s]·pivot_row) / d`
//! ([`pivot_row_update`]); every division in it is exact by Sylvester's
//! identity, and every cell type **verifies** that in release builds — the
//! fixed-width cells by one checking multiplication of the quotient
//! candidate produced by the modular-inverse scheme (Jebelean), `BigInt`
//! by the remainder of its `div_rem`.  A cell operation that fails
//! (`None`) means, for a fixed-width cell, that a value did not fit and the
//! caller should retry with wider cells ([`KernelError::Overflow`]); for
//! `BigInt`, which never overflows, it means the fraction-free invariant
//! was violated ([`KernelError::Inexact`]) — an internal error that the
//! callers surface as `ComputationFailed` instead of a truncated result.
//!
//! On top of the row update sit the two elimination kernels:
//!
//! * [`Elimination`] / [`fraction_free_gauss_jordan`] — Gauss–Jordan to
//!   (`d ×`) reduced row-echelon form, resumable one pivot at a time so
//!   that the drivers [`scaled_rref`] / [`try_scaled_rref`] can start on
//!   `i64` cells and move the *current state* to wider cells when a value
//!   does not fit, instead of starting over;
//! * [`Bareiss`] / [`bareiss_det`] — the forward-only determinant, with
//!   the same escalation in [`try_det`].
//!
//! The simplex tableau does not eliminate to RREF — it pivots one chosen
//! `(row, column)` at a time on a maintained tableau with an extra
//! objective row and its own restart-based escalation — so it shares only
//! the [`Cell`] trait and [`pivot_row_update`].

use std::fmt;

use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::Ratio;
use num_traits::{One, Signed, Zero};

use crate::base::numeric::Q;

// ═══════════════════════════════════════════════════════════════════════════
// Cells
// ═══════════════════════════════════════════════════════════════════════════

/// A tableau cell: an integer type with the handful of exact operations
/// the fraction-free algorithms need.  Fallible operations return `None`
/// when the result does not fit (or, for `BigInt`, when a division that
/// should have been exact was not).
pub(crate) trait Cell: Clone + PartialEq + Eq + Ord + fmt::Debug {
    /// `true` for the fixed-width cells, whose operations fail when a value
    /// does not fit; `false` for `BigInt`, whose only failure is an inexact
    /// division.  Decides which [`KernelError`] a failed operation becomes.
    const BOUNDED: bool;
    /// The outgoing common denominator `d` of one pivot, prepared once for
    /// the `O(m·n)` exact divisions of that pivot (for the fixed-width
    /// cells: its odd part's inverse modulo the word size, so that each
    /// division is a wrapping multiplication instead of a long division).
    type Divisor;
    fn divisor(d: &Self) -> Self::Divisor;
    fn cell_zero() -> Self;
    fn cell_one() -> Self;
    fn from_big(v: &BigInt) -> Option<Self>;
    /// `n / d` as a cell (`None` if it does not fit) — the entries of the
    /// initial tableau, `numer · (s / denom)`, for a row scale `s`.
    fn from_ratio_scaled(q: &Q, s: &BigInt) -> Option<Self>;
    fn to_big(&self) -> BigInt;
    fn is_zero(&self) -> bool;
    fn is_negative(&self) -> bool;
    /// `Ordering` of `self` against zero.
    fn signum(&self) -> std::cmp::Ordering;
    fn neg(&self) -> Option<Self>;
    fn mul(&self, o: &Self) -> Option<Self>;
    /// `(v · p − f · pr) / d`, the fraction-free pivot update; the division
    /// is exact.
    fn pivot_update(v: &Self, p: &Self, f: &Self, pr: &Self, d: &Self::Divisor) -> Option<Self>;
    /// `(v · p) / d` (exact) — the update of a row with a zero pivot-column entry.
    fn rescale(v: &Self, p: &Self, d: &Self::Divisor) -> Option<Self>;
    /// `self − f · r`, checked.
    fn sub_mul(&self, f: &Self, r: &Self) -> Option<Self>;
    /// `a · b` compared with `c · d`, exactly.
    fn cmp_products(a: &Self, b: &Self, c: &Self, d: &Self) -> std::cmp::Ordering;
}

/// The outgoing denominator of a `BigInt` pivot: the value itself, with
/// `±1` (the first pivot of every elimination) flagged so that its `O(m·n)`
/// divisions are skipped.
#[derive(Clone, Debug)]
pub(crate) struct BigDivisor {
    d: BigInt,
    unit: bool,
}

/// `t / d` when the division is exact, `None` otherwise.  The fraction-free
/// update is exact by Sylvester's identity, so a remainder means the
/// invariant was violated; the fixed-width cells verify their quotient by
/// one multiplication (Jebelean), and `BigInt` — whose division computes
/// the remainder anyway — checks it here, so that a violation surfaces as
/// `ComputationFailed` in release builds instead of a truncated tableau.
fn big_div_exact(t: BigInt, d: &BigDivisor) -> Option<BigInt> {
    if d.unit {
        return Some(if Signed::is_negative(&d.d) { -t } else { t });
    }
    let (q, r) = t.div_rem(&d.d);
    Zero::is_zero(&r).then_some(q)
}

impl Cell for BigInt {
    const BOUNDED: bool = false;
    type Divisor = BigDivisor;
    fn divisor(d: &Self) -> BigDivisor {
        BigDivisor {
            d: d.clone(),
            unit: d.magnitude().is_one(),
        }
    }
    fn cell_zero() -> Self {
        <BigInt as Zero>::zero()
    }
    fn cell_one() -> Self {
        <BigInt as One>::one()
    }
    fn from_big(v: &BigInt) -> Option<Self> {
        Some(v.clone())
    }
    fn from_ratio_scaled(q: &Q, s: &BigInt) -> Option<Self> {
        Some(q.numer() * (s / q.denom()))
    }
    fn to_big(&self) -> BigInt {
        self.clone()
    }
    fn is_zero(&self) -> bool {
        Zero::is_zero(self)
    }
    fn is_negative(&self) -> bool {
        Signed::is_negative(self)
    }
    fn signum(&self) -> std::cmp::Ordering {
        match self.sign() {
            num_bigint::Sign::Minus => std::cmp::Ordering::Less,
            num_bigint::Sign::NoSign => std::cmp::Ordering::Equal,
            num_bigint::Sign::Plus => std::cmp::Ordering::Greater,
        }
    }
    fn neg(&self) -> Option<Self> {
        Some(-self)
    }
    fn mul(&self, o: &Self) -> Option<Self> {
        Some(self * o)
    }
    fn pivot_update(v: &Self, p: &Self, f: &Self, pr: &Self, d: &BigDivisor) -> Option<Self> {
        let t = if Zero::is_zero(pr) {
            v * p
        } else if Zero::is_zero(v) {
            -(f * pr)
        } else {
            v * p - f * pr
        };
        big_div_exact(t, d)
    }
    fn rescale(v: &Self, p: &Self, d: &BigDivisor) -> Option<Self> {
        big_div_exact(v * p, d)
    }
    fn sub_mul(&self, f: &Self, r: &Self) -> Option<Self> {
        Some(self - f * r)
    }
    fn cmp_products(a: &Self, b: &Self, c: &Self, d: &Self) -> std::cmp::Ordering {
        (a * b).cmp(&(c * d))
    }
}

/// The outgoing denominator of an `i64` pivot, prepared for exact division
/// of the 128-bit intermediates: `d = 2^shift · d_odd`, with `inv` the
/// inverse of `d_odd` modulo `2^64` (two's complement; the sign of `d` rides
/// along, since an odd negative number is odd as a `u64` too).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Div64 {
    d: i64,
    shift: u32,
    inv: u64,
}

impl Div64 {
    fn new(d: i64) -> Self {
        debug_assert!(d != 0, "pivot on a zero entry");
        let shift = d.trailing_zeros();
        // `d >> shift` is odd; `i64::MIN >> 63 == -1`, also odd.
        let d_odd = (d >> shift) as u64;
        Div64 {
            d,
            shift,
            inv: inverse_mod_2_64(d_odd),
        }
    }

    /// Exact `t / d` (`t` is a multiple of `d`), or `None` when the
    /// quotient does not fit an `i64`.
    ///
    /// Jebelean's exact division: `t / d = (t >> shift) · inv (mod 2^64)`,
    /// so the low 64 bits of the shifted dividend times `inv` are the
    /// quotient modulo `2^64` — which *is* the quotient whenever the
    /// quotient is an `i64`.  One widening multiplication checks that: if
    /// the true quotient does not fit, the candidate differs from it by a
    /// non-zero multiple of `2^64`, so `candidate · d ≠ t` (the product is
    /// exact in an `i128`, so this cannot be fooled by wrapping).  Two
    /// multiplications in place of a 128-bit long division (`__divti3`,
    /// which otherwise dominates the pivot loop), with the identical value.
    #[inline]
    fn div_exact(&self, t: i128) -> Option<i64> {
        let q = ((t >> self.shift) as u64).wrapping_mul(self.inv) as i64;
        (i128::from(q) * i128::from(self.d) == t).then_some(q)
    }
}

/// The inverse of an odd `d` modulo `2^64` (see [`inverse_mod_2_128`]):
/// three correct bits to start, doubled five times.
fn inverse_mod_2_64(d: u64) -> u64 {
    debug_assert!(d & 1 == 1);
    let mut x = d;
    for _ in 0..5 {
        x = x.wrapping_mul(2u64.wrapping_sub(d.wrapping_mul(x)));
    }
    debug_assert_eq!(d.wrapping_mul(x), 1);
    x
}

/// `numer · (s / denom)` of a small rational without touching the heap:
/// `None` if `numer`, `denom` or `s` does not fit an `i64` (the caller then
/// takes the `BigInt` route).  `denom` divides `s`.
#[inline]
fn ratio_scaled_i128(q: &Q, s: &BigInt) -> Option<i128> {
    let numer = i64::try_from(q.numer()).ok()?;
    let denom = i64::try_from(q.denom()).ok()?;
    let s = i64::try_from(s).ok()?;
    Some(i128::from(numer) * i128::from(s / denom))
}

impl Cell for i64 {
    const BOUNDED: bool = true;
    type Divisor = Div64;
    fn divisor(d: &Self) -> Div64 {
        Div64::new(*d)
    }
    fn cell_zero() -> Self {
        0
    }
    fn cell_one() -> Self {
        1
    }
    fn from_big(v: &BigInt) -> Option<Self> {
        i64::try_from(v).ok()
    }
    fn from_ratio_scaled(q: &Q, s: &BigInt) -> Option<Self> {
        match ratio_scaled_i128(q, s) {
            Some(v) => i64::try_from(v).ok(),
            None => i64::try_from(q.numer() * (s / q.denom())).ok(),
        }
    }
    fn to_big(&self) -> BigInt {
        BigInt::from(*self)
    }
    fn is_zero(&self) -> bool {
        *self == 0
    }
    fn is_negative(&self) -> bool {
        *self < 0
    }
    fn signum(&self) -> std::cmp::Ordering {
        self.cmp(&0)
    }
    fn neg(&self) -> Option<Self> {
        self.checked_neg()
    }
    fn mul(&self, o: &Self) -> Option<Self> {
        self.checked_mul(*o)
    }
    fn pivot_update(v: &Self, p: &Self, f: &Self, pr: &Self, d: &Div64) -> Option<Self> {
        // Two i64 products and their difference always fit an i128.
        let t = i128::from(*v) * i128::from(*p) - i128::from(*f) * i128::from(*pr);
        d.div_exact(t)
    }
    fn rescale(v: &Self, p: &Self, d: &Div64) -> Option<Self> {
        d.div_exact(i128::from(*v) * i128::from(*p))
    }
    fn sub_mul(&self, f: &Self, r: &Self) -> Option<Self> {
        i64::try_from(i128::from(*self) - i128::from(*f) * i128::from(*r)).ok()
    }
    fn cmp_products(a: &Self, b: &Self, c: &Self, d: &Self) -> std::cmp::Ordering {
        (i128::from(*a) * i128::from(*b)).cmp(&(i128::from(*c) * i128::from(*d)))
    }
}

/// A signed 256-bit integer as `(hi, lo)` in two's complement, just enough
/// for the exact intermediates of `i128` cells: products of two `i128`s,
/// their differences, exact division by an `i128`, and comparison.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct I256 {
    hi: i128,
    lo: u128,
}

impl I256 {
    fn from_i128(v: i128) -> Self {
        I256 {
            hi: if v < 0 { -1 } else { 0 },
            lo: v as u128,
        }
    }

    /// Exact `a · b`.
    fn mul(a: i128, b: i128) -> Self {
        let neg = (a < 0) != (b < 0);
        let (ua, ub) = (a.unsigned_abs(), b.unsigned_abs());
        // 128×128 → 256 by 64-bit limbs.
        let (a0, a1) = (ua as u64 as u128, ua >> 64);
        let (b0, b1) = (ub as u64 as u128, ub >> 64);
        let p00 = a0 * b0;
        let p01 = a0 * b1;
        let p10 = a1 * b0;
        let p11 = a1 * b1;
        let mid = (p00 >> 64) + (p01 as u64 as u128) + (p10 as u64 as u128);
        let lo = (p00 as u64 as u128) | (mid << 64);
        let hi = p11 + (p01 >> 64) + (p10 >> 64) + (mid >> 64);
        let mag = I256 { hi: hi as i128, lo };
        if neg { mag.neg() } else { mag }
    }

    fn neg(self) -> Self {
        let lo = (!self.lo).wrapping_add(1);
        let hi = (!self.hi).wrapping_add(u128::from(lo == 0) as i128);
        I256 { hi, lo }
    }

    fn sub(self, o: Self) -> Self {
        let (lo, borrow) = self.lo.overflowing_sub(o.lo);
        let hi = self.hi.wrapping_sub(o.hi).wrapping_sub(i128::from(borrow));
        I256 { hi, lo }
    }

    fn is_negative(self) -> bool {
        self.hi < 0
    }

    /// The value as an `i128`, if it fits.
    fn to_i128(self) -> Option<i128> {
        let lo = self.lo as i128;
        let fits = (self.hi == 0 && lo >= 0) || (self.hi == -1 && lo < 0);
        fits.then_some(lo)
    }

    /// Exact division of a non-negative `self` by a positive `d` (the
    /// remainder is known to be zero); `None` if the quotient does not fit
    /// a `u128`.
    ///
    /// Jebelean's exact division: write `d = 2^k · d'` with `d'` odd, shift
    /// the dividend right by `k` (exact), and multiply its low 128 bits by
    /// the inverse of `d'` modulo `2^128` — which is the quotient whenever
    /// the quotient fits, i.e. whenever the shifted high half is below
    /// `d'`.  A handful of wrapping multiplications instead of a long
    /// division (there is no hardware `u128` division either, so the
    /// small-dividend case takes the same route).
    fn div_exact_unsigned(self, d: &Div128) -> Option<u128> {
        debug_assert!(!self.is_negative());
        let hi = self.hi as u128;
        let k = d.shift;
        let lo = if k == 0 {
            self.lo
        } else {
            (self.lo >> k) | (hi << (128 - k))
        };
        let hi = hi >> k;
        if hi >= d.odd {
            return None;
        }
        Some(lo.wrapping_mul(d.inv))
    }

    /// Exact `self / d` for any signs, `None` if the quotient does not fit
    /// an `i128` — or if the division was not exact after all: the
    /// modular-inverse candidate is only the quotient when `d` divides
    /// `self`, so it is verified by one multiplication (as `Div64` and
    /// `Div256` do), which turns a violated fraction-free invariant into a
    /// reported failure instead of a wrong cell.
    fn div_exact_by(self, d: &Div128) -> Option<i128> {
        let neg = self.is_negative() != (d.d < 0);
        let mag = if self.is_negative() { self.neg() } else { self };
        let q = mag.div_exact_unsigned(d)?;
        let q = if neg {
            // −2^127 is representable; q up to 2^127 is allowed then.
            if q > (1u128 << 127) {
                return None;
            }
            (q as i128).wrapping_neg()
        } else {
            i128::try_from(q).ok()?
        };
        (I256::mul(q, d.d) == self).then_some(q)
    }

    /// [`div_exact_by`](Self::div_exact_by) with the divisor prepared on
    /// the spot.
    #[cfg(test)]
    fn div_exact(self, d: i128) -> Option<i128> {
        self.div_exact_by(&Div128::new(d))
    }
}

/// The outgoing denominator of an `i128` pivot: `|d| = 2^shift · odd`, with
/// `inv` the inverse of `odd` modulo `2^128`, computed once per pivot.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Div128 {
    d: i128,
    shift: u32,
    odd: u128,
    inv: u128,
}

impl Div128 {
    fn new(d: i128) -> Self {
        debug_assert!(d != 0, "pivot on a zero entry");
        let abs = d.unsigned_abs();
        let shift = abs.trailing_zeros();
        let odd = abs >> shift;
        Div128 {
            d,
            shift,
            odd,
            inv: inverse_mod_2_128(odd),
        }
    }
}

/// The inverse of an odd `d` modulo `2^128`, by Newton's iteration
/// `x ← x·(2 − d·x)`, which doubles the number of correct low bits each
/// step (an odd `d` is its own inverse modulo 8, so three bits to start).
fn inverse_mod_2_128(d: u128) -> u128 {
    debug_assert!(d & 1 == 1);
    let mut x = d;
    for _ in 0..6 {
        x = x.wrapping_mul(2u128.wrapping_sub(d.wrapping_mul(x)));
    }
    debug_assert_eq!(d.wrapping_mul(x), 1);
    x
}

impl PartialOrd for I256 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for I256 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.hi.cmp(&other.hi).then(self.lo.cmp(&other.lo))
    }
}

impl Cell for i128 {
    const BOUNDED: bool = true;
    type Divisor = Div128;
    fn divisor(d: &Self) -> Div128 {
        Div128::new(*d)
    }
    fn cell_zero() -> Self {
        0
    }
    fn cell_one() -> Self {
        1
    }
    fn from_big(v: &BigInt) -> Option<Self> {
        i128::try_from(v).ok()
    }
    fn from_ratio_scaled(q: &Q, s: &BigInt) -> Option<Self> {
        match ratio_scaled_i128(q, s) {
            Some(v) => Some(v),
            None => i128::try_from(q.numer() * (s / q.denom())).ok(),
        }
    }
    fn to_big(&self) -> BigInt {
        BigInt::from(*self)
    }
    fn is_zero(&self) -> bool {
        *self == 0
    }
    fn is_negative(&self) -> bool {
        *self < 0
    }
    fn signum(&self) -> std::cmp::Ordering {
        self.cmp(&0)
    }
    fn neg(&self) -> Option<Self> {
        self.checked_neg()
    }
    fn mul(&self, o: &Self) -> Option<Self> {
        self.checked_mul(*o)
    }
    fn pivot_update(v: &Self, p: &Self, f: &Self, pr: &Self, d: &Div128) -> Option<Self> {
        let t = I256::mul(*v, *p).sub(I256::mul(*f, *pr));
        t.div_exact_by(d)
    }
    fn rescale(v: &Self, p: &Self, d: &Div128) -> Option<Self> {
        I256::mul(*v, *p).div_exact_by(d)
    }
    fn sub_mul(&self, f: &Self, r: &Self) -> Option<Self> {
        I256::from_i128(*self).sub(I256::mul(*f, *r)).to_i128()
    }
    fn cmp_products(a: &Self, b: &Self, c: &Self, d: &Self) -> std::cmp::Ordering {
        I256::mul(*a, *b).cmp(&I256::mul(*c, *d))
    }
}

// ── 256-bit cells ───────────────────────────────────────────────────────────
//
// Fraction-free entries are minors of the scaled system, so a 20-row
// certificate LP with three-digit coefficients peaks around 130–200 bits:
// past `i128`, but far from needing heap integers.  `W256` is a signed
// 256-bit two's-complement integer in four little-endian `u64` limbs, with
// exact 512-bit intermediates (`W512`), so that such problems stay on the
// stack; the arithmetic is the plain schoolbook kind on limbs.

/// Unsigned little-endian limb arithmetic shared by [`W256`] and [`W512`].
/// All operations wrap (two's complement) unless they report a carry.
mod limbs {
    use std::cmp::Ordering;

    #[inline]
    pub(super) fn sub<const N: usize>(a: &[u64; N], b: &[u64; N]) -> ([u64; N], bool) {
        let mut out = [0u64; N];
        let mut borrow = false;
        for i in 0..N {
            let (s, b1) = a[i].overflowing_sub(b[i]);
            let (s, b2) = s.overflowing_sub(u64::from(borrow));
            out[i] = s;
            borrow = b1 || b2;
        }
        (out, borrow)
    }

    /// Two's-complement negation (wrapping; the minimum maps to itself).
    #[inline]
    pub(super) fn neg<const N: usize>(a: &[u64; N]) -> [u64; N] {
        sub(&[0u64; N], a).0
    }

    #[inline]
    pub(super) fn is_negative<const N: usize>(a: &[u64; N]) -> bool {
        a[N - 1] >> 63 == 1
    }

    /// Signed comparison of two's-complement values.
    #[inline]
    pub(super) fn cmp_signed<const N: usize>(a: &[u64; N], b: &[u64; N]) -> Ordering {
        match (is_negative(a), is_negative(b)) {
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            // Same sign: unsigned order agrees with signed order.
            _ => a.iter().rev().cmp(b.iter().rev()),
        }
    }

    /// Low `N` limbs of `a · b` (wrapping multiplication).
    #[inline]
    pub(super) fn mul_low<const N: usize>(a: &[u64; N], b: &[u64; N]) -> [u64; N] {
        let mut out = [0u64; N];
        for i in 0..N {
            let mut carry = 0u128;
            for j in 0..N - i {
                let t = u128::from(a[i]) * u128::from(b[j]) + u128::from(out[i + j]) + carry;
                out[i + j] = t as u64;
                carry = t >> 64;
            }
        }
        out
    }

    /// Full unsigned product of two four-limb values.
    #[inline]
    pub(super) fn mul_4x4(a: &[u64; 4], b: &[u64; 4]) -> [u64; 8] {
        let mut out = [0u64; 8];
        for i in 0..4 {
            let mut carry = 0u128;
            for j in 0..4 {
                let t = u128::from(a[i]) * u128::from(b[j]) + u128::from(out[i + j]) + carry;
                out[i + j] = t as u64;
                carry = t >> 64;
            }
            out[i + 4] = carry as u64;
        }
        out
    }

    /// Arithmetic shift right by `k < 64·N` bits.
    #[inline]
    pub(super) fn shr_signed<const N: usize>(a: &[u64; N], k: u32) -> [u64; N] {
        let fill = if is_negative(a) { u64::MAX } else { 0 };
        let limbs = (k / 64) as usize;
        let bits = k % 64;
        let mut out = [fill; N];
        for i in 0..N - limbs {
            let lo = a[i + limbs] >> bits;
            let hi = if bits == 0 {
                0
            } else if i + limbs + 1 < N {
                a[i + limbs + 1] << (64 - bits)
            } else {
                fill << (64 - bits)
            };
            out[i] = lo | hi;
        }
        out
    }

    #[inline]
    pub(super) fn trailing_zeros<const N: usize>(a: &[u64; N]) -> u32 {
        let mut n = 0;
        for limb in a {
            if *limb == 0 {
                n += 64;
            } else {
                return n + limb.trailing_zeros();
            }
        }
        n
    }
}

/// Signed 256-bit integer (two's complement, little-endian `u64` limbs).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct W256([u64; 4]);

/// Signed 512-bit integer: the exact intermediates of [`W256`] cells.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct W512([u64; 8]);

impl W256 {
    const ZERO: W256 = W256([0; 4]);
    const ONE: W256 = W256([1, 0, 0, 0]);

    fn from_i128(v: i128) -> Self {
        let fill = if v < 0 { u64::MAX } else { 0 };
        let u = v as u128;
        W256([u as u64, (u >> 64) as u64, fill, fill])
    }

    /// `None` if `|v| ≥ 2^255` (the minimum itself is not accepted — it is
    /// merely reported as not fitting, which only widens the cells).
    fn from_big(v: &BigInt) -> Option<Self> {
        let mut mag = [0u64; 4];
        for (i, digit) in v.iter_u64_digits().enumerate() {
            *mag.get_mut(i)? = digit;
        }
        if mag[3] >> 63 == 1 {
            return None;
        }
        Some(W256(if Signed::is_negative(v) {
            limbs::neg(&mag)
        } else {
            mag
        }))
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.0 == [0; 4]
    }

    #[inline]
    fn is_negative(&self) -> bool {
        limbs::is_negative(&self.0)
    }

    /// Exact signed product, as 512 bits.
    #[inline]
    fn mul_full(a: &W256, b: &W256) -> W512 {
        let neg = a.is_negative() != b.is_negative();
        let ua = if a.is_negative() {
            limbs::neg(&a.0)
        } else {
            a.0
        };
        let ub = if b.is_negative() {
            limbs::neg(&b.0)
        } else {
            b.0
        };
        let prod = limbs::mul_4x4(&ua, &ub);
        W512(if neg { limbs::neg(&prod) } else { prod })
    }

    /// Sign-extend to 512 bits.
    #[inline]
    fn widen(&self) -> W512 {
        let fill = if self.is_negative() { u64::MAX } else { 0 };
        let a = &self.0;
        W512([a[0], a[1], a[2], a[3], fill, fill, fill, fill])
    }
}

impl PartialOrd for W256 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for W256 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        limbs::cmp_signed(&self.0, &other.0)
    }
}

impl W512 {
    #[inline]
    fn sub(&self, o: &W512) -> W512 {
        W512(limbs::sub(&self.0, &o.0).0)
    }

    /// The value as a [`W256`], `None` if it does not fit (the high half
    /// must be the sign extension of the low half).
    #[inline]
    fn narrow(&self) -> Option<W256> {
        let a = &self.0;
        let fill = if a[3] >> 63 == 1 { u64::MAX } else { 0 };
        (a[4] == fill && a[5] == fill && a[6] == fill && a[7] == fill)
            .then(|| W256([a[0], a[1], a[2], a[3]]))
    }

    #[inline]
    fn cmp(&self, o: &W512) -> std::cmp::Ordering {
        limbs::cmp_signed(&self.0, &o.0)
    }
}

/// The outgoing denominator of a [`W256`] pivot: `d = 2^shift · d_odd` with
/// `inv` the inverse of `d_odd` modulo `2^256` (see [`Div64`]).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Div256 {
    d: W256,
    shift: u32,
    inv: W256,
}

impl Div256 {
    fn new(d: W256) -> Self {
        debug_assert!(!d.is_zero(), "pivot on a zero entry");
        let shift = limbs::trailing_zeros(&d.0);
        let d_odd = limbs::shr_signed(&d.0, shift);
        // Newton's iteration doubles the correct low bits: 3 → 6 → … → 384.
        let mut x = d_odd;
        for _ in 0..7 {
            let dx = limbs::mul_low(&d_odd, &x);
            let two_minus = limbs::sub(&[2, 0, 0, 0], &dx).0;
            x = limbs::mul_low(&x, &two_minus);
        }
        debug_assert_eq!(limbs::mul_low(&d_odd, &x), [1, 0, 0, 0]);
        Div256 {
            d,
            shift,
            inv: W256(x),
        }
    }

    /// Exact `t / d`, `None` if the quotient does not fit 256 bits — the
    /// same modular-inverse-plus-checking-multiplication scheme as
    /// [`Div64::div_exact`], on limbs.
    #[inline]
    fn div_exact(&self, t: &W512) -> Option<W256> {
        let shifted = limbs::shr_signed(&t.0, self.shift);
        let low = [shifted[0], shifted[1], shifted[2], shifted[3]];
        let q = W256(limbs::mul_low(&low, &self.inv.0));
        (W256::mul_full(&q, &self.d) == *t).then_some(q)
    }
}

impl Cell for W256 {
    const BOUNDED: bool = true;
    type Divisor = Div256;
    fn divisor(d: &Self) -> Div256 {
        Div256::new(*d)
    }
    fn cell_zero() -> Self {
        W256::ZERO
    }
    fn cell_one() -> Self {
        W256::ONE
    }
    fn from_big(v: &BigInt) -> Option<Self> {
        W256::from_big(v)
    }
    fn from_ratio_scaled(q: &Q, s: &BigInt) -> Option<Self> {
        match ratio_scaled_i128(q, s) {
            Some(v) => Some(W256::from_i128(v)),
            None => W256::from_big(&(q.numer() * (s / q.denom()))),
        }
    }
    fn to_big(&self) -> BigInt {
        let neg = self.is_negative();
        let mag = if neg { limbs::neg(&self.0) } else { self.0 };
        let mut bytes = [0u8; 32];
        for (i, limb) in mag.iter().enumerate() {
            bytes[8 * i..8 * i + 8].copy_from_slice(&limb.to_le_bytes());
        }
        BigInt::from_bytes_le(
            if neg {
                num_bigint::Sign::Minus
            } else {
                num_bigint::Sign::Plus
            },
            &bytes,
        )
    }
    fn is_zero(&self) -> bool {
        W256::is_zero(self)
    }
    fn is_negative(&self) -> bool {
        W256::is_negative(self)
    }
    fn signum(&self) -> std::cmp::Ordering {
        if self.is_zero() {
            std::cmp::Ordering::Equal
        } else if self.is_negative() {
            std::cmp::Ordering::Less
        } else {
            std::cmp::Ordering::Greater
        }
    }
    fn neg(&self) -> Option<Self> {
        let n = W256(limbs::neg(&self.0));
        // Only the minimum is its own negation.
        (n.is_negative() != self.is_negative() || self.is_zero()).then_some(n)
    }
    fn mul(&self, o: &Self) -> Option<Self> {
        W256::mul_full(self, o).narrow()
    }
    fn pivot_update(v: &Self, p: &Self, f: &Self, pr: &Self, d: &Div256) -> Option<Self> {
        let t = W256::mul_full(v, p).sub(&W256::mul_full(f, pr));
        d.div_exact(&t)
    }
    fn rescale(v: &Self, p: &Self, d: &Div256) -> Option<Self> {
        d.div_exact(&W256::mul_full(v, p))
    }
    fn sub_mul(&self, f: &Self, r: &Self) -> Option<Self> {
        self.widen().sub(&W256::mul_full(f, r)).narrow()
    }
    fn cmp_products(a: &Self, b: &Self, c: &Self, d: &Self) -> std::cmp::Ordering {
        W256::mul_full(a, b).cmp(&W256::mul_full(c, d))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// The shared pivot step
// ═══════════════════════════════════════════════════════════════════════════

/// Why a fraction-free kernel stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum KernelError {
    /// A value did not fit the (fixed-width) cell type: retry with wider
    /// cells.  Only [`Cell::BOUNDED`] types report this.
    Overflow,
    /// A division that Sylvester's identity guarantees to be exact was not:
    /// the fraction-free invariant was violated.  Reported by `BigInt` cells
    /// (which cannot overflow); an internal error — the caller maps it to
    /// `ComputationFailed` or a slower, invariant-free algorithm, never to
    /// a truncated result.
    Inexact,
}

impl KernelError {
    /// The error a failed `C` operation stands for.
    #[inline]
    fn of<C: Cell>() -> Self {
        if C::BOUNDED {
            KernelError::Overflow
        } else {
            KernelError::Inexact
        }
    }
}

/// One fraction-free pivot applied to a single row:
/// `row ← (row·p − row[s]·prow) / d` with `row[s]` cleared, where `p =
/// prow[s]` is the pivot and `d` the prepared outgoing denominator.  A row
/// with `row[s] = 0` keeps its rational value and is only rescaled from `d`
/// to `p`.  `None` when a cell operation fails (see [`KernelError`]); the
/// row is then partially updated and unusable.
///
/// This is the inner loop of every fraction-free algorithm here — the
/// Gauss–Jordan kernel, the Bareiss determinant and the simplex tableau's
/// pivot — so the arithmetic (and hence the values) is identical across
/// them.
#[inline]
pub(crate) fn pivot_row_update<C: Cell>(
    row: &mut [C],
    prow: &[C],
    s: usize,
    p: &C,
    d: &C::Divisor,
) -> Option<()> {
    let f = std::mem::replace(&mut row[s], C::cell_zero());
    if f.is_zero() {
        for v in row.iter_mut() {
            if !v.is_zero() {
                *v = C::rescale(v, p, d)?;
            }
        }
        return Some(());
    }
    for (j, (v, pr)) in row.iter_mut().zip(prow.iter()).enumerate() {
        if j == s {
            continue;
        }
        if v.is_zero() && pr.is_zero() {
            continue;
        }
        *v = C::pivot_update(v, p, &f, pr, d)?;
    }
    Some(())
}

/// Swap rows `i < j` of a row-major buffer with `ncols` columns.
#[inline]
fn swap_rows<C>(a: &mut [C], ncols: usize, i: usize, j: usize) {
    let (head, tail) = a.split_at_mut(j * ncols);
    head[i * ncols..(i + 1) * ncols].swap_with_slice(&mut tail[..ncols]);
}

// ═══════════════════════════════════════════════════════════════════════════
// Fraction-free Gauss–Jordan
// ═══════════════════════════════════════════════════════════════════════════

/// Bareiss's fraction-free Gauss–Jordan elimination in progress on a
/// row-major `nrows × ncols` buffer, one pivot per [`step`](Self::step).
///
/// Pivots are taken from columns `0..pivot_limit` only, in column order,
/// each on the first row (from the next pivot row down) with a non-zero
/// entry, which is swapped into place.  When finished every row equals
/// `d · (row of the rational reduced row-echelon form)` for the final `d`
/// (the last pivot; it may be negative), so the RREF entry `(i, j)` is
/// `a[i][j] / d`.  Rows without a pivot are zero in columns `< pivot_limit`;
/// in the remaining columns they hold `d` times the reduced right-hand side
/// (the inconsistency residual of an augmented system).  Every division is
/// exact (Sylvester's identity: all intermediate entries are minors of the
/// input).
///
/// The state between two pivots is just the buffer, `d`, the pivot columns
/// and the position, so it can be moved to a wider cell type with
/// [`convert`](Self::convert) and continued — the drivers below start on
/// `i64` cells and widen on overflow without redoing any pivot.
#[derive(Clone, Debug)]
pub(crate) struct Elimination<C: Cell> {
    a: Vec<C>,
    nrows: usize,
    ncols: usize,
    /// `pivot_limit.min(ncols)`.
    limit: usize,
    d: C,
    pivots: Vec<usize>,
    /// Next pivot row.
    pr: usize,
    /// Next column to examine.
    col: usize,
}

/// The finished elimination: `cells = d · RREF`, the pivot columns, `d`.
pub(crate) struct Reduced<C> {
    pub(crate) cells: Vec<C>,
    pub(crate) pivots: Vec<usize>,
    pub(crate) d: C,
}

impl<C: Cell> Elimination<C> {
    pub(crate) fn new(a: Vec<C>, nrows: usize, ncols: usize, pivot_limit: usize) -> Self {
        debug_assert_eq!(a.len(), nrows * ncols);
        Elimination {
            a,
            nrows,
            ncols,
            limit: pivot_limit.min(ncols),
            d: C::cell_one(),
            pivots: Vec::new(),
            pr: 0,
            col: 0,
        }
    }

    /// Perform the next pivot.  `Ok(true)` when there is none left (the
    /// elimination is finished), `Ok(false)` after a pivot.  On `Err` the
    /// buffer is half-updated and the state must be discarded.
    pub(crate) fn step(&mut self) -> Result<bool, KernelError> {
        let (nrows, ncols) = (self.nrows, self.ncols);
        while self.col < self.limit && self.pr < nrows {
            let c = self.col;
            self.col += 1;
            let pr = self.pr;
            let Some(p) = (pr..nrows).find(|&i| !self.a[i * ncols + c].is_zero()) else {
                continue;
            };
            if p != pr {
                swap_rows(&mut self.a, ncols, pr, p);
            }
            // Take the pivot row out so the other rows can be updated against it.
            let prow: Vec<C> = self.a[pr * ncols..(pr + 1) * ncols]
                .iter_mut()
                .map(|v| std::mem::replace(v, C::cell_zero()))
                .collect();
            let pv = prow[c].clone();
            let d = C::divisor(&self.d);
            for i in 0..nrows {
                if i == pr {
                    continue;
                }
                pivot_row_update(&mut self.a[i * ncols..(i + 1) * ncols], &prow, c, &pv, &d)
                    .ok_or_else(KernelError::of::<C>)?;
            }
            self.a[pr * ncols..(pr + 1) * ncols]
                .iter_mut()
                .zip(prow)
                .for_each(|(slot, v)| *slot = v);
            self.d = pv;
            self.pivots.push(c);
            self.pr += 1;
            return Ok(false);
        }
        Ok(true)
    }

    /// Run to completion.
    pub(crate) fn run(&mut self) -> Result<(), KernelError> {
        while !self.step()? {}
        Ok(())
    }

    /// The same elimination on another cell type (between two pivots).
    pub(crate) fn convert<D: Cell>(self, mut f: impl FnMut(C) -> D) -> Elimination<D> {
        Elimination {
            a: self.a.into_iter().map(&mut f).collect(),
            nrows: self.nrows,
            ncols: self.ncols,
            limit: self.limit,
            d: f(self.d),
            pivots: self.pivots,
            pr: self.pr,
            col: self.col,
        }
    }

    pub(crate) fn finish(self) -> Reduced<C> {
        Reduced {
            cells: self.a,
            pivots: self.pivots,
            d: self.d,
        }
    }
}

/// Fraction-free Gauss–Jordan elimination of `a` in place on one cell type
/// (see [`Elimination`] for the contract).  Returns the pivot columns
/// (their count is the rank of the first `pivot_limit` columns) and `d`.
/// On `Err` the buffer is unspecified.
pub(crate) fn fraction_free_gauss_jordan<C: Cell>(
    a: &mut [C],
    nrows: usize,
    ncols: usize,
    pivot_limit: usize,
) -> Result<(Vec<usize>, C), KernelError> {
    let cells: Vec<C> = a
        .iter_mut()
        .map(|v| std::mem::replace(v, C::cell_zero()))
        .collect();
    let mut e = Elimination::new(cells, nrows, ncols, pivot_limit);
    let r = e.run();
    let done = e.finish();
    a.iter_mut().zip(done.cells).for_each(|(slot, v)| *slot = v);
    r.map(|()| (done.pivots, done.d))
}

// ═══════════════════════════════════════════════════════════════════════════
// Bareiss determinant
// ═══════════════════════════════════════════════════════════════════════════

/// Bareiss's fraction-free determinant of a square `n × n` buffer in
/// progress: forward elimination only, one column per [`step`](Self::step);
/// the last diagonal entry is `±det` when finished.
#[derive(Clone, Debug)]
pub(crate) struct Bareiss<C: Cell> {
    a: Vec<C>,
    n: usize,
    prev: C,
    /// Odd number of row swaps so far.
    sign: bool,
    /// Next elimination column.
    k: usize,
    singular: bool,
}

impl<C: Cell> Bareiss<C> {
    pub(crate) fn new(a: Vec<C>, n: usize) -> Self {
        debug_assert_eq!(a.len(), n * n);
        Bareiss {
            a,
            n,
            prev: C::cell_one(),
            sign: false,
            k: 0,
            singular: false,
        }
    }

    /// Eliminate the next column.  `Ok(true)` when finished.
    pub(crate) fn step(&mut self) -> Result<bool, KernelError> {
        let n = self.n;
        if self.singular || n == 0 || self.k + 1 >= n {
            return Ok(true);
        }
        let k = self.k;
        if self.a[k * n + k].is_zero() {
            let Some(p) = ((k + 1)..n).find(|&i| !self.a[i * n + k].is_zero()) else {
                self.singular = true;
                return Ok(true);
            };
            swap_rows(&mut self.a, n, k, p);
            self.sign = !self.sign;
        }
        let pivot = self.a[k * n + k].clone();
        let d = C::divisor(&self.prev);
        // Rows below k are zero in columns < k, as is row k, so the shared
        // row update touches only columns > k (and clears column k).
        let (head, tail) = self.a.split_at_mut((k + 1) * n);
        let prow = &head[k * n..(k + 1) * n];
        for row in tail.chunks_exact_mut(n) {
            pivot_row_update(row, prow, k, &pivot, &d).ok_or_else(KernelError::of::<C>)?;
        }
        self.prev = pivot;
        self.k += 1;
        Ok(false)
    }

    pub(crate) fn run(&mut self) -> Result<(), KernelError> {
        while !self.step()? {}
        Ok(())
    }

    pub(crate) fn convert<D: Cell>(self, mut f: impl FnMut(C) -> D) -> Bareiss<D> {
        Bareiss {
            a: self.a.into_iter().map(&mut f).collect(),
            n: self.n,
            prev: f(self.prev),
            sign: self.sign,
            k: self.k,
            singular: self.singular,
        }
    }

    /// The determinant (after [`run`](Self::run)).  `None` only if the
    /// sign flip of a fixed-width result does not fit.
    pub(crate) fn finish(mut self) -> Option<C> {
        if self.singular {
            return Some(C::cell_zero());
        }
        if self.n == 0 {
            return Some(C::cell_one());
        }
        let d = std::mem::replace(&mut self.a[self.n * self.n - 1], C::cell_zero());
        if self.sign { d.neg() } else { Some(d) }
    }
}

/// Bareiss fraction-free determinant of a square row-major buffer on one
/// cell type (consumes the buffer).
pub(crate) fn bareiss_det<C: Cell>(a: Vec<C>, n: usize) -> Result<C, KernelError> {
    let mut b = Bareiss::new(a, n);
    b.run()?;
    b.finish().ok_or_else(KernelError::of::<C>)
}

// ═══════════════════════════════════════════════════════════════════════════
// Fraction-free LU
// ═══════════════════════════════════════════════════════════════════════════

/// Bareiss forward elimination of a square `n × n` buffer that also records
/// what an LU decomposition needs: the row permutation, each step's pivot
/// and the column entries it eliminated.  Gaussian elimination's entries
/// are ratios of consecutive leading minors, and Bareiss's intermediates
/// *are* those minors, so `L[i][k] = col[i][k] / pivot[k]` and
/// `U[k][·] = row k / pivot[k − 1]` come out with one exact division per
/// entry instead of a normalised fraction per operation.  The pivot of a
/// column is its first non-zero entry at or below the diagonal — the rule
/// of the symbolic `Matrix::lu`, so the same factors.
#[derive(Clone, Debug)]
pub(crate) struct FractionFreeLu<C: Cell> {
    a: Vec<C>,
    n: usize,
    /// `col[i · n + k]` for `i > k`: entry `(i, k)` just before step `k`
    /// cleared it (zero elsewhere).
    col: Vec<C>,
    /// The pivot of every completed step.
    pivots: Vec<C>,
    perm: Vec<usize>,
    /// Next elimination column.
    k: usize,
    singular: bool,
}

/// The finished elimination (non-singular): see [`FractionFreeLu`].
pub(crate) struct LuParts<C> {
    /// Row `k` is `pivot[k − 1]` times row `k` of `U` (`pivot[−1] = 1`).
    pub(crate) rows: Vec<C>,
    /// `col[i · n + k]` is `pivot[k]` times `L[i][k]` for `i > k`.
    pub(crate) col: Vec<C>,
    pub(crate) pivots: Vec<C>,
    pub(crate) perm: Vec<usize>,
}

impl<C: Cell> FractionFreeLu<C> {
    pub(crate) fn new(a: Vec<C>, n: usize) -> Self {
        debug_assert_eq!(a.len(), n * n);
        FractionFreeLu {
            col: vec![C::cell_zero(); n * n],
            a,
            n,
            pivots: Vec::with_capacity(n),
            perm: (0..n).collect(),
            k: 0,
            singular: false,
        }
    }

    /// Eliminate the next column.  `Ok(true)` when finished — every column
    /// pivoted, or one had no non-zero candidate (singular).
    pub(crate) fn step(&mut self) -> Result<bool, KernelError> {
        let n = self.n;
        if self.singular || self.k >= n {
            return Ok(true);
        }
        let k = self.k;
        let Some(p) = (k..n).find(|&i| !self.a[i * n + k].is_zero()) else {
            self.singular = true;
            return Ok(true);
        };
        if p != k {
            swap_rows(&mut self.a, n, k, p);
            // Columns < k of `col` are the multipliers of rows k and p;
            // columns ≥ k are still zero in both.
            swap_rows(&mut self.col, n, k, p);
            self.perm.swap(k, p);
        }
        let pivot = self.a[k * n + k].clone();
        if k + 1 < n {
            let prev = self.pivots.last().cloned().unwrap_or_else(C::cell_one);
            let d = C::divisor(&prev);
            let (head, tail) = self.a.split_at_mut((k + 1) * n);
            let prow = &head[k * n..(k + 1) * n];
            for (below, row) in tail.chunks_exact_mut(n).enumerate() {
                let i = k + 1 + below;
                self.col[i * n + k] = row[k].clone();
                pivot_row_update(row, prow, k, &pivot, &d).ok_or_else(KernelError::of::<C>)?;
            }
        }
        self.pivots.push(pivot);
        self.k += 1;
        Ok(self.k >= n)
    }

    pub(crate) fn run(&mut self) -> Result<(), KernelError> {
        while !self.step()? {}
        Ok(())
    }

    pub(crate) fn convert<D: Cell>(self, mut f: impl FnMut(C) -> D) -> FractionFreeLu<D> {
        FractionFreeLu {
            a: self.a.into_iter().map(&mut f).collect(),
            n: self.n,
            col: self.col.into_iter().map(&mut f).collect(),
            pivots: self.pivots.into_iter().map(&mut f).collect(),
            perm: self.perm,
            k: self.k,
            singular: self.singular,
        }
    }

    /// The factors' integer parts (after [`run`](Self::run)); `None` if the
    /// matrix is singular.
    pub(crate) fn finish(self) -> Option<LuParts<C>> {
        if self.singular {
            return None;
        }
        Some(LuParts {
            rows: self.a,
            col: self.col,
            pivots: self.pivots,
            perm: self.perm,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Berkowitz characteristic polynomial
// ═══════════════════════════════════════════════════════════════════════════

/// Berkowitz's division-free characteristic polynomial of a square `n × n`
/// buffer in progress: [`vec`](Self::vec) holds the coefficients of
/// `det(λI − A_k)` for the trailing `k × k` block, highest degree first,
/// and each [`step`](Self::step) grows the block by one row and column.
/// Only multiplications and additions are used, so the state between two
/// steps moves to a wider cell type exactly like [`Bareiss`].
#[derive(Clone, Debug)]
pub(crate) struct Berkowitz<C: Cell> {
    a: Vec<C>,
    /// `−A` entry-wise: every accumulation `acc + x·y` is written as
    /// `acc − (−x)·y` so that the one checked fused operation of [`Cell`]
    /// ([`sub_mul`](Cell::sub_mul)) covers it.
    neg_a: Vec<C>,
    n: usize,
    /// Coefficients for the trailing block of size `vec.len() − 1`.
    vec: Vec<C>,
}

impl<C: Cell> Berkowitz<C> {
    /// `Err` only if an entry cannot be negated in `C` (the minimum of a
    /// fixed-width cell): the caller then widens the cells.
    pub(crate) fn new(a: Vec<C>, n: usize) -> Result<Self, KernelError> {
        debug_assert_eq!(a.len(), n * n);
        let neg_a = a
            .iter()
            .map(Cell::neg)
            .collect::<Option<Vec<C>>>()
            .ok_or_else(KernelError::of::<C>)?;
        // The trailing 1×1 block: det(λ − a_{n−1,n−1}).
        let vec = match neg_a.last() {
            Some(last) => vec![C::cell_one(), last.clone()],
            None => vec![C::cell_one()],
        };
        Ok(Berkowitz { a, neg_a, n, vec })
    }

    /// Extend the coefficients to the next larger trailing block.
    /// `Ok(true)` when the whole matrix is covered.
    pub(crate) fn step(&mut self) -> Result<bool, KernelError> {
        let n = self.n;
        let k = self.vec.len(); // block size after this step
        if k > n {
            return Ok(true);
        }
        let s = n - k; // top-left index of the trailing k×k block
        let err = KernelError::of::<C>;
        let zero = C::cell_zero();

        // Negated Toeplitz diagonals: [−1, a_ss, R·C, R·A'·C, …, R·A'^{k−2}·C]
        // with R = row s (columns s+1..n), C = column s (rows s+1..n) and
        // A' the trailing (k−1)×(k−1) block.
        let mut nd: Vec<C> = Vec::with_capacity(k + 1);
        nd.push(C::cell_one().neg().ok_or_else(err)?);
        nd.push(self.a[s * n + s].clone());
        let mut c: Vec<C> = (s + 1..n).map(|i| self.a[i * n + s].clone()).collect();
        for pass in 0..k - 1 {
            if pass > 0 {
                // c ← A'·c
                let mut next = Vec::with_capacity(k - 1);
                for i in s + 1..n {
                    let mut acc = zero.clone();
                    for (cj, j) in c.iter().zip(s + 1..n) {
                        if !cj.is_zero() {
                            acc = acc.sub_mul(&self.neg_a[i * n + j], cj).ok_or_else(err)?;
                        }
                    }
                    next.push(acc);
                }
                c = next;
            }
            let mut rc = zero.clone();
            for (cj, j) in c.iter().zip(s + 1..n) {
                if !cj.is_zero() {
                    rc = rc.sub_mul(&self.neg_a[s * n + j], cj).ok_or_else(err)?;
                }
            }
            nd.push(rc);
        }

        // vec ← T·vec, T the (k+1)×k lower-triangular Toeplitz matrix with
        // T[i][j] = −nd[i − j].
        let mut next_vec: Vec<C> = Vec::with_capacity(k + 1);
        for i in 0..=k {
            let mut acc = zero.clone();
            for (j, v) in self.vec.iter().enumerate().take(i + 1) {
                if !v.is_zero() {
                    acc = acc.sub_mul(&nd[i - j], v).ok_or_else(err)?;
                }
            }
            next_vec.push(acc);
        }
        self.vec = next_vec;
        Ok(self.vec.len() > n)
    }

    pub(crate) fn run(&mut self) -> Result<(), KernelError> {
        while !self.step()? {}
        Ok(())
    }

    pub(crate) fn convert<D: Cell>(self, mut f: impl FnMut(C) -> D) -> Berkowitz<D> {
        Berkowitz {
            a: self.a.into_iter().map(&mut f).collect(),
            neg_a: self.neg_a.into_iter().map(&mut f).collect(),
            n: self.n,
            vec: self.vec.into_iter().map(&mut f).collect(),
        }
    }

    /// The coefficients of `det(λI − A)`, highest degree first (length
    /// `n + 1`), after [`run`](Self::run).
    pub(crate) fn finish(self) -> Vec<C> {
        self.vec
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Width escalation: i64 → i128 → W256 → BigInt, resuming where the narrower
// cells overflowed
// ═══════════════════════════════════════════════════════════════════════════

/// Run a fixed-width elimination to completion (`Ok`), or return the state
/// just before the pivot that overflowed (`Err`) so that wider cells can
/// take over from there.  One snapshot per pivot: a `memcpy` of the buffer,
/// small against the `O(rows · cols)` cell operations of the pivot itself.
fn run_bounded<S: Clone>(
    mut s: S,
    mut step: impl FnMut(&mut S) -> Result<bool, KernelError>,
) -> Result<S, S> {
    let mut snapshot = s.clone();
    loop {
        match step(&mut s) {
            Ok(true) => return Ok(s),
            Ok(false) => snapshot.clone_from(&s),
            Err(_) => return Err(snapshot),
        }
    }
}

/// Where the fixed-width stages of an escalating run left off.
enum Staged<T> {
    /// Finished on fixed-width cells; the result is converted to `BigInt`.
    Done(T),
    /// The 256-bit cells overflowed too: continue on `BigInt` from here.
    Resume(T),
    /// The input itself does not fit 256-bit cells.
    TooWide,
}

/// The fixed-width stages of [`try_scaled_rref`]: start on the narrowest
/// cells that hold the input, widen on overflow.  Reads `a` only.
fn gauss_jordan_fixed(
    a: &[BigInt],
    nrows: usize,
    ncols: usize,
    pivot_limit: usize,
) -> Staged<Elimination<BigInt>> {
    let e128 = match a
        .iter()
        .map(|v| i64::try_from(v).ok())
        .collect::<Option<Vec<i64>>>()
    {
        Some(cells) => match run_bounded(
            Elimination::new(cells, nrows, ncols, pivot_limit),
            Elimination::step,
        ) {
            Ok(done) => return Staged::Done(done.convert(BigInt::from)),
            Err(snap) => Some(snap.convert(i128::from)),
        },
        None => a
            .iter()
            .map(|v| i128::try_from(v).ok())
            .collect::<Option<Vec<i128>>>()
            .map(|cells| Elimination::new(cells, nrows, ncols, pivot_limit)),
    };
    let e256 = match e128 {
        Some(e) => match run_bounded(e, Elimination::step) {
            Ok(done) => return Staged::Done(done.convert(BigInt::from)),
            Err(snap) => Some(snap.convert(W256::from_i128)),
        },
        None => a
            .iter()
            .map(W256::from_big)
            .collect::<Option<Vec<W256>>>()
            .map(|cells| Elimination::new(cells, nrows, ncols, pivot_limit)),
    };
    match e256 {
        Some(e) => match run_bounded(e, Elimination::step) {
            Ok(done) => Staged::Done(done.convert(|w| w.to_big())),
            Err(snap) => Staged::Resume(snap.convert(|w| w.to_big())),
        },
        None => Staged::TooWide,
    }
}

/// Fraction-free Gauss–Jordan of integer rows with width escalation: the
/// contract of [`fraction_free_gauss_jordan`] (on return `a` holds `d ·
/// RREF`; the pivot columns and `d` are returned), computed on `i64`,
/// `i128` and 256-bit cells as long as they hold the values and on `BigInt`
/// beyond.  The only error is [`KernelError::Inexact`] — the `BigInt`
/// stage found a division inexact — after which `a` is unspecified.
pub(crate) fn try_scaled_rref(
    a: &mut Vec<BigInt>,
    nrows: usize,
    ncols: usize,
    pivot_limit: usize,
) -> Result<(Vec<usize>, BigInt), KernelError> {
    let mut e = match gauss_jordan_fixed(a, nrows, ncols, pivot_limit) {
        Staged::Done(done) => {
            let r = done.finish();
            *a = r.cells;
            return Ok((r.pivots, r.d));
        }
        Staged::Resume(e) => e,
        Staged::TooWide => return fraction_free_gauss_jordan(a, nrows, ncols, pivot_limit),
    };
    e.run()?;
    let r = e.finish();
    *a = r.cells;
    Ok((r.pivots, r.d))
}

/// [`try_scaled_rref`] for the infallible callers (`rank`, `rref`, …).
/// Should the `BigInt` stage ever report an inexact division — a violated
/// internal invariant, never observed — the result is recomputed by plain
/// Gauss–Jordan over `Ratio<BigInt>` ([`rational_scaled_rref`]), which
/// does not rely on the invariant, so the answer is still correct; the
/// event is logged at `error` level.
pub(crate) fn scaled_rref(
    a: &mut Vec<BigInt>,
    nrows: usize,
    ncols: usize,
    pivot_limit: usize,
) -> (Vec<usize>, BigInt) {
    // `a` is read but not written until a stage has succeeded, so the
    // fallback always sees the original rows.
    let mut e = match gauss_jordan_fixed(a, nrows, ncols, pivot_limit) {
        Staged::Done(done) => {
            let r = done.finish();
            *a = r.cells;
            return (r.pivots, r.d);
        }
        Staged::Resume(e) => e,
        Staged::TooWide => Elimination::new(a.clone(), nrows, ncols, pivot_limit),
    };
    match e.run() {
        Ok(()) => {
            let r = e.finish();
            *a = r.cells;
            (r.pivots, r.d)
        }
        Err(err) => {
            debug_assert!(false, "fraction-free Gauss–Jordan: {err:?}");
            tracing::error!(
                target: "symplex::exact_kernel",
                ?err,
                nrows,
                ncols,
                "fraction-free Gauss–Jordan violated its exactness invariant; recomputing over Q"
            );
            rational_scaled_rref(a, nrows, ncols, pivot_limit)
        }
    }
}

/// Plain Gauss–Jordan over `Ratio<BigInt>` with the result brought to the
/// common denominator `d > 0` (the lcm of the entries' denominators), so
/// that it satisfies the contract of [`fraction_free_gauss_jordan`].  The
/// invariant-free fallback of [`scaled_rref`]; about an order of magnitude
/// slower than the fraction-free kernel.
pub(crate) fn rational_scaled_rref(
    a: &mut [BigInt],
    nrows: usize,
    ncols: usize,
    pivot_limit: usize,
) -> (Vec<usize>, BigInt) {
    let mut q: Vec<Q> = a.iter().map(|v| Ratio::from_integer(v.clone())).collect();
    let mut pivots = Vec::new();
    let mut pr = 0usize;
    for c in 0..pivot_limit.min(ncols) {
        if pr >= nrows {
            break;
        }
        let Some(p) = (pr..nrows).find(|&i| !q[i * ncols + c].is_zero()) else {
            continue;
        };
        if p != pr {
            swap_rows(&mut q, ncols, pr, p);
        }
        let pv = q[pr * ncols + c].clone();
        for v in &mut q[pr * ncols..(pr + 1) * ncols] {
            if !v.is_zero() {
                *v = &*v / &pv;
            }
        }
        let prow: Vec<Q> = q[pr * ncols..(pr + 1) * ncols].to_vec();
        for i in 0..nrows {
            if i == pr {
                continue;
            }
            let f = std::mem::replace(&mut q[i * ncols + c], Q::zero());
            if f.is_zero() {
                continue;
            }
            for (j, (v, p)) in q[i * ncols..(i + 1) * ncols]
                .iter_mut()
                .zip(&prow)
                .enumerate()
            {
                if j != c && !p.is_zero() {
                    *v = &*v - &f * p;
                }
            }
        }
        pivots.push(c);
        pr += 1;
    }
    let d = q.iter().fold(BigInt::one(), |l, v| l.lcm(v.denom()));
    for (slot, v) in a.iter_mut().zip(q) {
        *slot = v.numer() * (&d / v.denom());
    }
    (pivots, d)
}

/// The fixed-width stages of [`try_det`].
fn bareiss_fixed(a: &[BigInt], n: usize) -> Staged<Bareiss<BigInt>> {
    let b128 = match a
        .iter()
        .map(|v| i64::try_from(v).ok())
        .collect::<Option<Vec<i64>>>()
    {
        Some(cells) => match run_bounded(Bareiss::new(cells, n), Bareiss::step) {
            Ok(done) => return Staged::Done(done.convert(BigInt::from)),
            Err(snap) => Some(snap.convert(i128::from)),
        },
        None => a
            .iter()
            .map(|v| i128::try_from(v).ok())
            .collect::<Option<Vec<i128>>>()
            .map(|cells| Bareiss::new(cells, n)),
    };
    let b256 = match b128 {
        Some(b) => match run_bounded(b, Bareiss::step) {
            Ok(done) => return Staged::Done(done.convert(BigInt::from)),
            Err(snap) => Some(snap.convert(W256::from_i128)),
        },
        None => a
            .iter()
            .map(W256::from_big)
            .collect::<Option<Vec<W256>>>()
            .map(|cells| Bareiss::new(cells, n)),
    };
    match b256 {
        Some(b) => match run_bounded(b, Bareiss::step) {
            Ok(done) => Staged::Done(done.convert(|w| w.to_big())),
            Err(snap) => Staged::Resume(snap.convert(|w| w.to_big())),
        },
        None => Staged::TooWide,
    }
}

/// Bareiss determinant of a square row-major integer buffer with width
/// escalation (see [`try_scaled_rref`]).  `Err(Inexact)` only.
pub(crate) fn try_det(a: Vec<BigInt>, n: usize) -> Result<BigInt, KernelError> {
    let mut b = match bareiss_fixed(&a, n) {
        Staged::Done(done) => return done.finish().ok_or(KernelError::Inexact),
        Staged::Resume(b) => b,
        Staged::TooWide => return bareiss_det(a, n),
    };
    b.run()?;
    b.finish().ok_or(KernelError::Inexact)
}

/// The fixed-width stages of [`try_lu`].
fn lu_fixed(a: &[BigInt], n: usize) -> Staged<FractionFreeLu<BigInt>> {
    let b128 = match a
        .iter()
        .map(|v| i64::try_from(v).ok())
        .collect::<Option<Vec<i64>>>()
    {
        Some(cells) => match run_bounded(FractionFreeLu::new(cells, n), FractionFreeLu::step) {
            Ok(done) => return Staged::Done(done.convert(BigInt::from)),
            Err(snap) => Some(snap.convert(i128::from)),
        },
        None => a
            .iter()
            .map(|v| i128::try_from(v).ok())
            .collect::<Option<Vec<i128>>>()
            .map(|cells| FractionFreeLu::new(cells, n)),
    };
    let b256 = match b128 {
        Some(b) => match run_bounded(b, FractionFreeLu::step) {
            Ok(done) => return Staged::Done(done.convert(BigInt::from)),
            Err(snap) => Some(snap.convert(W256::from_i128)),
        },
        None => a
            .iter()
            .map(W256::from_big)
            .collect::<Option<Vec<W256>>>()
            .map(|cells| FractionFreeLu::new(cells, n)),
    };
    match b256 {
        Some(b) => match run_bounded(b, FractionFreeLu::step) {
            Ok(done) => Staged::Done(done.convert(|w| w.to_big())),
            Err(snap) => Staged::Resume(snap.convert(|w| w.to_big())),
        },
        None => Staged::TooWide,
    }
}

/// Fraction-free LU of a square row-major integer buffer with width
/// escalation (see [`try_scaled_rref`]): `Ok(None)` for a singular matrix,
/// `Err(Inexact)` only for a violated exactness invariant.
pub(crate) fn try_lu(a: Vec<BigInt>, n: usize) -> Result<Option<LuParts<BigInt>>, KernelError> {
    let mut b = match lu_fixed(&a, n) {
        Staged::Done(done) => return Ok(done.finish()),
        Staged::Resume(b) => b,
        Staged::TooWide => FractionFreeLu::new(a, n),
    };
    b.run()?;
    Ok(b.finish())
}

/// The fixed-width stages of [`try_char_poly`].
fn berkowitz_fixed(a: &[BigInt], n: usize) -> Staged<Berkowitz<BigInt>> {
    let b128 = match a
        .iter()
        .map(|v| i64::try_from(v).ok())
        .collect::<Option<Vec<i64>>>()
        .and_then(|cells| Berkowitz::new(cells, n).ok())
    {
        Some(b) => match run_bounded(b, Berkowitz::step) {
            Ok(done) => return Staged::Done(done.convert(BigInt::from)),
            Err(snap) => Some(snap.convert(i128::from)),
        },
        None => a
            .iter()
            .map(|v| i128::try_from(v).ok())
            .collect::<Option<Vec<i128>>>()
            .and_then(|cells| Berkowitz::new(cells, n).ok()),
    };
    let b256 = match b128 {
        Some(b) => match run_bounded(b, Berkowitz::step) {
            Ok(done) => return Staged::Done(done.convert(BigInt::from)),
            Err(snap) => Some(snap.convert(W256::from_i128)),
        },
        None => a
            .iter()
            .map(W256::from_big)
            .collect::<Option<Vec<W256>>>()
            .and_then(|cells| Berkowitz::new(cells, n).ok()),
    };
    match b256 {
        Some(b) => match run_bounded(b, Berkowitz::step) {
            Ok(done) => Staged::Done(done.convert(|w| w.to_big())),
            Err(snap) => Staged::Resume(snap.convert(|w| w.to_big())),
        },
        None => Staged::TooWide,
    }
}

/// Berkowitz characteristic polynomial `det(λI − A)` of a square row-major
/// integer buffer, highest degree first (length `n + 1`), with width
/// escalation (see [`try_scaled_rref`]).  The algorithm is division-free,
/// so the `BigInt` stage cannot fail; the `Result` is the shape shared with
/// the other drivers.
pub(crate) fn try_char_poly(a: Vec<BigInt>, n: usize) -> Result<Vec<BigInt>, KernelError> {
    let mut b = match berkowitz_fixed(&a, n) {
        Staged::Done(done) => return Ok(done.finish()),
        Staged::Resume(b) => b,
        Staged::TooWide => Berkowitz::new(a, n)?,
    };
    b.run()?;
    Ok(b.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::numeric::q;

    // ── Kernel ──────────────────────────────────────────────────────────────

    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 11
        }

        fn small(&mut self) -> BigInt {
            BigInt::from((self.next() % 19) as i64 - 9)
        }

        /// A random integer of about `bits` bits, either sign.
        fn wide(&mut self, bits: u32) -> BigInt {
            let mut v = BigInt::from(0);
            let mut left = bits;
            while left > 0 {
                let take = left.min(50);
                v = (v << take) + BigInt::from(self.next() & ((1u64 << take) - 1));
                left -= take;
            }
            if self.next() & 1 == 1 { -v } else { v }
        }
    }

    /// The rational matrix `cells / d`.
    fn rational(cells: &[BigInt], d: &BigInt) -> Vec<Q> {
        cells
            .iter()
            .map(|v| Ratio::new(v.clone(), d.clone()))
            .collect()
    }

    /// A cell that computes correctly on `i64` but reports every division
    /// by a divisor other than `±1` as failed — i.e. from the second pivot
    /// on.  `BOUNDED` decides which error that becomes.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
    struct Faulty<const BOUNDED: bool>(i64);

    impl<const B: bool> Cell for Faulty<B> {
        const BOUNDED: bool = B;
        type Divisor = i64;
        fn divisor(d: &Self) -> i64 {
            d.0
        }
        fn cell_zero() -> Self {
            Faulty(0)
        }
        fn cell_one() -> Self {
            Faulty(1)
        }
        fn from_big(v: &BigInt) -> Option<Self> {
            i64::try_from(v).ok().map(Faulty)
        }
        fn from_ratio_scaled(_: &Q, _: &BigInt) -> Option<Self> {
            None
        }
        fn to_big(&self) -> BigInt {
            BigInt::from(self.0)
        }
        fn is_zero(&self) -> bool {
            self.0 == 0
        }
        fn is_negative(&self) -> bool {
            self.0 < 0
        }
        fn signum(&self) -> std::cmp::Ordering {
            self.0.cmp(&0)
        }
        fn neg(&self) -> Option<Self> {
            Some(Faulty(-self.0))
        }
        fn mul(&self, o: &Self) -> Option<Self> {
            Some(Faulty(self.0 * o.0))
        }
        fn pivot_update(v: &Self, p: &Self, f: &Self, pr: &Self, d: &i64) -> Option<Self> {
            (d.abs() == 1).then(|| Faulty((v.0 * p.0 - f.0 * pr.0) / d))
        }
        fn rescale(v: &Self, p: &Self, d: &i64) -> Option<Self> {
            (d.abs() == 1).then(|| Faulty(v.0 * p.0 / d))
        }
        fn sub_mul(&self, f: &Self, r: &Self) -> Option<Self> {
            Some(Faulty(self.0 - f.0 * r.0))
        }
        fn cmp_products(a: &Self, b: &Self, c: &Self, d: &Self) -> std::cmp::Ordering {
            (a.0 * b.0).cmp(&(c.0 * d.0))
        }
    }

    /// A failed cell operation stops the kernels with the error the cell
    /// type stands for — `Inexact` for an unbounded cell, `Overflow` for a
    /// bounded one — instead of a truncated result; the first pivot (whose
    /// divisor is 1) goes through, the second fails.
    #[test]
    fn kernels_report_a_failed_cell_operation() {
        let rows = [[2i64, 1, 1], [1, 3, 2], [1, 0, 0]];
        let cells = |_: ()| -> Vec<i64> { rows.iter().flatten().copied().collect() };
        let mut a: Vec<Faulty<false>> = cells(()).into_iter().map(Faulty).collect();
        assert_eq!(
            fraction_free_gauss_jordan(&mut a, 3, 3, 3),
            Err(KernelError::Inexact)
        );
        let mut a: Vec<Faulty<true>> = cells(()).into_iter().map(Faulty).collect();
        assert_eq!(
            fraction_free_gauss_jordan(&mut a, 3, 3, 3),
            Err(KernelError::Overflow)
        );
        let a: Vec<Faulty<false>> = cells(()).into_iter().map(Faulty).collect();
        assert_eq!(bareiss_det(a, 3), Err(KernelError::Inexact));
        let a: Vec<Faulty<true>> = cells(()).into_iter().map(Faulty).collect();
        assert_eq!(bareiss_det(a, 3), Err(KernelError::Overflow));
        // Step-wise: the first pivot succeeds, the second fails.
        let a: Vec<Faulty<true>> = cells(()).into_iter().map(Faulty).collect();
        let mut e = Elimination::new(a, 3, 3, 3);
        assert_eq!(e.step(), Ok(false));
        assert_eq!(e.step(), Err(KernelError::Overflow));
        // A single-pivot problem (divisor 1 throughout) is fine on either.
        let mut a = vec![Faulty::<false>(3), Faulty(6)];
        assert_eq!(
            fraction_free_gauss_jordan(&mut a, 1, 2, 2),
            Ok((vec![0], Faulty(3)))
        );
        assert_eq!(a, vec![Faulty(3), Faulty(6)]);
    }

    /// The `i128` cells' exact division now verifies its candidate: an
    /// operand that is *not* a multiple of the divisor is reported as
    /// failed instead of yielding the wrapped garbage of the modular
    /// inverse, on both the "fits" and the sign paths.
    #[test]
    fn i128_division_rejects_inexact_operands() {
        for &(t, d) in &[
            (7i128, 2i128),
            (-7, 2),
            (7, -2),
            (1 << 100, 3),
            (i128::MAX, 2),
            ((1i128 << 90) + 1, 1 << 40),
        ] {
            assert_eq!(I256::from_i128(t).div_exact(d), None, "{t} / {d}");
            // … while the neighbouring exact quotients still go through.
            let exact = t - t.rem_euclid(d);
            assert_eq!(I256::from_i128(exact).div_exact(d), Some(exact / d));
        }
        // The pivot update itself: (v·p − f·pr) / d with an inexact sum.
        let dv = <i128 as Cell>::divisor(&3);
        assert_eq!(<i128 as Cell>::pivot_update(&4, &2, &1, &1, &dv), None); // 7 / 3
        assert_eq!(<i128 as Cell>::pivot_update(&4, &2, &1, &2, &dv), Some(2)); // 6 / 3
        assert_eq!(<i128 as Cell>::rescale(&5, &2, &dv), None); // 10 / 3
    }

    /// Width escalation resumes the very same elimination, so on random
    /// matrices whose minors outgrow `i64`, `i128` and 256 bits in turn the
    /// escalating drivers return the `BigInt` kernel's result bit for bit —
    /// the same pivots, the same (signed) `d`, the same buffer.
    #[test]
    fn escalating_drivers_match_the_bigint_kernel() {
        let mut g = Lcg(7);
        let cases: Vec<(usize, usize, u32)> = vec![
            (3, 4, 3),    // stays on i64
            (8, 10, 3),   // i64 throughout
            (14, 17, 8),  // into i128
            (20, 24, 3),  // into i128
            (30, 36, 3),  // into 256-bit cells
            (12, 14, 40), // i64 overflows at once, i128 soon, then 256-bit
            (10, 12, 90), // starts on 256-bit cells
            (6, 8, 300),  // does not fit 256-bit cells: BigInt from the start
            (9, 9, 60),   // square, into 256-bit cells
        ];
        for (nrows, ncols, bits) in cases {
            for round in 0..3 {
                let a: Vec<BigInt> = (0..nrows * ncols)
                    .map(|k| {
                        // A few zeros so that row swaps and zero pivots occur.
                        if (k + round) % 7 == 0 {
                            BigInt::from(0)
                        } else if bits <= 3 {
                            g.small()
                        } else {
                            g.wide(bits)
                        }
                    })
                    .collect();
                for &limit in &[ncols, ncols - 1, nrows.min(ncols)] {
                    let mut reference = a.clone();
                    let want = fraction_free_gauss_jordan(&mut reference, nrows, ncols, limit)
                        .expect("BigInt kernel is exact");
                    let mut escalated = a.clone();
                    let got = try_scaled_rref(&mut escalated, nrows, ncols, limit)
                        .expect("escalating driver is exact");
                    assert_eq!(got, want, "{nrows}×{ncols}, {bits} bits, limit {limit}");
                    assert_eq!(escalated, reference);
                    let mut infallible = a.clone();
                    assert_eq!(scaled_rref(&mut infallible, nrows, ncols, limit), want);
                    assert_eq!(infallible, reference);
                }
                if nrows == ncols {
                    let want = bareiss_det(a.clone(), nrows).expect("exact");
                    assert_eq!(try_det(a.clone(), nrows), Ok(want));
                }
            }
        }
    }

    /// The determinant with escalation on matrices that need a row swap,
    /// are singular, or are empty.
    #[test]
    fn escalating_det_edge_cases() {
        let m = |rows: &[&[i64]]| -> Vec<BigInt> {
            rows.iter()
                .flat_map(|r| r.iter().map(|&v| BigInt::from(v)))
                .collect()
        };
        assert_eq!(try_det(Vec::new(), 0), Ok(BigInt::from(1)));
        assert_eq!(try_det(m(&[&[0, 1], &[1, 0]]), 2), Ok(BigInt::from(-1)));
        assert_eq!(try_det(m(&[&[1, 2], &[2, 4]]), 2), Ok(BigInt::from(0)));
        assert_eq!(
            try_det(m(&[&[0, 0, 1], &[0, 1, 0], &[1, 0, 0]]), 3),
            Ok(BigInt::from(-1))
        );
        assert_eq!(
            try_det(m(&[&[2, 1, 0], &[1, 3, 1], &[0, 1, 4]]), 3),
            Ok(BigInt::from(18))
        );
        // Entries around 2^40: the i64 stage overflows on the first pivot.
        let big = BigInt::from(1i64 << 40);
        let a: Vec<BigInt> = m(&[&[3, 1, 2], &[1, 5, 1], &[2, 1, 3]])
            .into_iter()
            .map(|v| v * &big)
            .collect();
        assert_eq!(
            try_det(a, 3),
            Ok(BigInt::from(3 * (15 - 1) - (3 - 2) + 2 * (1 - 10)) * &big * &big * &big)
        );
    }

    /// Berkowitz on `BigInt` cells, written out with plain `+`/`*` (no
    /// `Cell` operations): `det(λI − A)`, highest degree first.
    fn reference_char_poly(a: &[BigInt], n: usize) -> Vec<BigInt> {
        if n == 0 {
            return vec![BigInt::from(1)];
        }
        let at = |i: usize, j: usize| &a[i * n + j];
        let mut vec = vec![BigInt::from(1), -at(n - 1, n - 1)];
        for k in 2..=n {
            let s = n - k;
            let mut c: Vec<BigInt> = (s + 1..n).map(|i| at(i, s).clone()).collect();
            let mut diags = vec![BigInt::from(1), -at(s, s)];
            for step in 0..k - 1 {
                if step > 0 {
                    c = (s + 1..n)
                        .map(|i| (s + 1..n).zip(&c).map(|(j, cj)| at(i, j) * cj).sum())
                        .collect();
                }
                let rc: BigInt = (s + 1..n).zip(&c).map(|(j, cj)| at(s, j) * cj).sum();
                diags.push(-rc);
            }
            vec = (0..=k)
                .map(|i| {
                    vec.iter()
                        .enumerate()
                        .take(i + 1)
                        .map(|(j, v)| &diags[i - j] * v)
                        .sum()
                })
                .collect();
        }
        vec
    }

    /// The escalating Berkowitz driver returns the plain `BigInt`
    /// computation exactly, whichever cell widths the coefficients pass
    /// through, and the small cases by hand: `det(λI − A)` of the empty
    /// matrix is `1`, of `[a]` is `λ − a`, of a 2×2 is `λ² − tr·λ + det`.
    #[test]
    fn escalating_char_poly_matches_the_bigint_computation() {
        let m = |rows: &[&[i64]]| -> Vec<BigInt> {
            rows.iter()
                .flat_map(|r| r.iter().map(|&v| BigInt::from(v)))
                .collect()
        };
        let ints = |v: &[i64]| -> Vec<BigInt> { v.iter().map(|&x| BigInt::from(x)).collect() };
        assert_eq!(try_char_poly(Vec::new(), 0), Ok(ints(&[1])));
        assert_eq!(try_char_poly(m(&[&[7]]), 1), Ok(ints(&[1, -7])));
        assert_eq!(
            try_char_poly(m(&[&[1, 2], &[3, 4]]), 2),
            Ok(ints(&[1, -5, -2]))
        );
        assert_eq!(
            try_char_poly(m(&[&[2, 0, 0], &[0, 3, 0], &[0, 0, 5]]), 3),
            Ok(ints(&[1, -10, 31, -30]))
        );
        // i64::MIN cannot be negated on i64 cells: the driver widens.
        assert_eq!(
            try_char_poly(vec![BigInt::from(i64::MIN)], 1),
            Ok(vec![BigInt::from(1), -BigInt::from(i64::MIN)])
        );
        let mut g = Lcg(5);
        let cases: Vec<(usize, u32)> = vec![
            (4, 3),   // stays on i64
            (8, 3),   // i64 throughout
            (7, 12),  // into i128
            (9, 24),  // into 256-bit cells
            (6, 45),  // i64 overflows at once
            (5, 100), // starts on 256-bit cells, overflows into BigInt
            (4, 300), // does not fit 256-bit cells: BigInt from the start
        ];
        for (n, bits) in cases {
            for round in 0..3 {
                let a: Vec<BigInt> = (0..n * n)
                    .map(|k| {
                        if (k + round) % 5 == 0 {
                            BigInt::from(0)
                        } else if bits <= 3 {
                            g.small()
                        } else {
                            g.wide(bits)
                        }
                    })
                    .collect();
                let want = reference_char_poly(&a, n);
                assert_eq!(want.len(), n + 1);
                assert_eq!(want[0], BigInt::from(1));
                assert_eq!(
                    try_char_poly(a.clone(), n),
                    Ok(want.clone()),
                    "{n}×{n}, {bits} bits"
                );
                // Against the fraction-free determinant: c_0 = (−1)ⁿ det A.
                let det = try_det(a, n).expect("exact");
                assert_eq!(want[n], if n % 2 == 1 { -det } else { det });
            }
        }
    }

    /// The invariant-free fallback agrees with the fraction-free kernel as
    /// a rational matrix (its `d` is the positive lcm of the denominators,
    /// not the last pivot) and honours `pivot_limit`.
    #[test]
    fn rational_fallback_matches_the_kernel() {
        let mut g = Lcg(11);
        for &(nrows, ncols) in &[(1usize, 1usize), (3, 5), (6, 4), (7, 9), (5, 5)] {
            for round in 0..4 {
                let a: Vec<BigInt> = (0..nrows * ncols)
                    .map(|k| {
                        if (k + round) % 5 == 0 {
                            BigInt::from(0)
                        } else {
                            g.small()
                        }
                    })
                    .collect();
                for &limit in &[ncols, ncols / 2, 1] {
                    let mut kernel = a.clone();
                    let (pivots, d) = fraction_free_gauss_jordan(&mut kernel, nrows, ncols, limit)
                        .expect("exact");
                    let mut fallback = a.clone();
                    let (fp, fd) = rational_scaled_rref(&mut fallback, nrows, ncols, limit);
                    assert_eq!(fp, pivots);
                    assert!(Signed::is_positive(&fd));
                    assert_eq!(rational(&fallback, &fd), rational(&kernel, &d));
                }
            }
        }
    }

    /// `pivot_row_update` is the LP's row update verbatim: on `BigInt`
    /// cells it reproduces `(v·p − f·pr) / d` entry by entry, clears the
    /// pivot column, and only rescales a row with a zero pivot-column entry.
    #[test]
    fn pivot_row_update_is_the_fraction_free_rule() {
        let b = |v: i64| BigInt::from(v);
        let prow = vec![b(6), b(4), b(-2), b(10)];
        let d = <BigInt as Cell>::divisor(&b(2));
        let p = b(4); // pivot column s = 1
        let mut row = vec![b(2), b(3), b(8), b(-4)];
        pivot_row_update(&mut row, &prow, 1, &p, &d).expect("exact");
        // (row·4 − 3·prow) / 2
        assert_eq!(
            row,
            vec![b((8 - 18) / 2), b(0), b((32 + 6) / 2), b((-16 - 30) / 2)]
        );
        let mut zero_col = vec![b(2), b(0), b(8), b(-4)];
        pivot_row_update(&mut zero_col, &prow, 1, &p, &d).expect("exact");
        assert_eq!(zero_col, vec![b(4), b(0), b(16), b(-8)]);
        // The `BigInt` cells check the remainder: an inexact operand fails.
        let d3 = <BigInt as Cell>::divisor(&b(3));
        let mut bad = vec![b(1), b(1), b(0), b(0)];
        assert_eq!(pivot_row_update(&mut bad, &prow, 1, &p, &d3), None);
        // … unless the divisor is ±1, which is skipped.
        let unit = <BigInt as Cell>::divisor(&b(-1));
        let mut r = vec![b(1), b(0), b(0), b(0)];
        pivot_row_update(&mut r, &prow, 1, &p, &unit).expect("exact");
        assert_eq!(r, vec![b(-4), b(0), b(0), b(0)]);
    }

    // ── Cells (moved from `linprog`, where the types used to live) ─────────

    /// The `i64` cells' exact division (modular inverse plus one checking
    /// multiplication) agrees with the long division it replaced on random
    /// operands and on the edges of the range: quotients of exactly
    /// `i64::MIN`/`i64::MAX`, one past them, even and negative divisors.
    #[test]
    fn i64_exact_division_matches_bigint() {
        let check = |t: i128, d: i64| {
            let want = i64::try_from(t / i128::from(d)).ok();
            let got = Div64::new(d).div_exact(t);
            assert_eq!(got, want, "{t} / {d}");
        };
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = |bits: u32| -> i64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let v = (state & ((1u64 << bits) - 1)) as i64;
            if state & (1 << 5) != 0 { -v } else { v }
        };
        for _ in 0..20_000 {
            let d = next(40);
            if d == 0 {
                continue;
            }
            // Quotients that fit (≤ 62 bits) and quotients that do not.
            let q_small = next(62);
            check(i128::from(q_small) * i128::from(d), d);
            let q_big = i128::from(next(62)) * 4 + i128::from(i64::MAX);
            check(q_big * i128::from(d), d);
            // The general update `v·p − f·pr` with both products multiples of d.
            let (v, p, f, pr) = (next(20) * d, next(32), next(20) * d, next(32));
            let dv = Div64::new(d);
            let t = i128::from(v) * i128::from(p) - i128::from(f) * i128::from(pr);
            assert_eq!(
                <i64 as Cell>::pivot_update(&v, &p, &f, &pr, &dv),
                i64::try_from(t / i128::from(d)).ok()
            );
            assert_eq!(
                <i64 as Cell>::rescale(&v, &p, &dv),
                i64::try_from(i128::from(v) * i128::from(p) / i128::from(d)).ok()
            );
        }
        for &d in &[
            1i64,
            -1,
            2,
            -2,
            6,
            -6,
            1 << 40,
            -(1 << 40),
            i64::MAX,
            i64::MIN,
            3,
            -3,
        ] {
            for &q in &[
                0i64,
                1,
                -1,
                i64::MAX,
                i64::MIN,
                i64::MAX - 1,
                i64::MIN + 1,
                12345,
            ] {
                check(i128::from(q) * i128::from(d), d);
            }
            // One past the range in both directions.
            check((i128::from(i64::MAX) + 1) * i128::from(d), d);
            check((i128::from(i64::MIN) - 1) * i128::from(d), d);
            // Odd part of |d| is inverted correctly (d·inv ≡ 1 mod 2^64).
            let dv = Div64::new(d);
            assert_eq!(
                ((d >> dv.shift) as u64).wrapping_mul(dv.inv),
                1,
                "inverse of the odd part of {d}"
            );
        }
    }

    /// Every [`Cell`] operation of the 256-bit cells agrees with `BigInt`
    /// on random operands up to the full width, including the fit/no-fit
    /// boundaries of the checked operations and the exact division.
    #[test]
    fn w256_cells_match_bigint() {
        let mut state: u128 = 0x243F_6A88_85A3_08D3_1319_8A2E_0370_7344;
        let mut word = || -> u64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as u64
        };
        // A random BigInt of at most `bits` bits, either sign.
        let mut rand_big = |bits: u32| -> BigInt {
            let mut v = BigInt::from(0);
            let mut left = bits;
            while left > 0 {
                let take = left.min(64);
                let w = word() & (u64::MAX >> (64 - take));
                v = (v << take) + BigInt::from(w);
                left -= take;
            }
            if word() & 1 == 1 { -v } else { v }
        };
        let min256: BigInt = -(BigInt::from(1) << 255u32);
        let fits256 = |v: &BigInt| v.bits() <= 255 && *v != min256;
        let to_opt = |v: &BigInt| -> Option<BigInt> { fits256(v).then(|| v.clone()) };
        for round in 0..4000 {
            // Sizes chosen so that products and quotients straddle 256 bits.
            let bits = [60, 120, 127, 128, 200, 250, 254, 255][round % 8];
            let (a, b, c, d) = (
                rand_big(bits),
                rand_big(bits),
                rand_big(bits),
                rand_big(255 - bits.min(254)),
            );
            let (wa, wb, wc, wd) = (
                W256::from_big(&a).expect("fits"),
                W256::from_big(&b).expect("fits"),
                W256::from_big(&c).expect("fits"),
                W256::from_big(&d).expect("fits"),
            );
            assert_eq!(wa.to_big(), a);
            assert_eq!(<W256 as Cell>::signum(&wa), a.cmp(&BigInt::from(0)));
            assert_eq!(wa.cmp(&wb), a.cmp(&b), "{a} vs {b}");
            assert_eq!(<W256 as Cell>::neg(&wa).map(|v| v.to_big()), to_opt(&-&a));
            assert_eq!(
                <W256 as Cell>::mul(&wa, &wb).map(|v| v.to_big()),
                to_opt(&(&a * &b)),
                "{a} · {b}"
            );
            assert_eq!(
                <W256 as Cell>::cmp_products(&wa, &wb, &wc, &wd),
                (&a * &b).cmp(&(&c * &d))
            );
            assert_eq!(
                <W256 as Cell>::sub_mul(&wa, &wb, &wc).map(|v| v.to_big()),
                to_opt(&(&a - &b * &c))
            );
            // Exact division: (v·p − f·pr)/d with v and f multiples of d.
            if !Zero::is_zero(&d) {
                let dv = <W256 as Cell>::divisor(&wd);
                let k = rand_big(bits.min(250));
                let v = &k * &d;
                if let Some(wv) = W256::from_big(&v) {
                    let want = &k * &b;
                    assert_eq!(
                        <W256 as Cell>::rescale(&wv, &wb, &dv).map(|q| q.to_big()),
                        to_opt(&want),
                        "({v} · {b}) / {d}"
                    );
                    let f = &rand_big(20) * &d;
                    let wf = W256::from_big(&f).expect("fits");
                    let want2 = &k * &b - (&f / &d) * &c;
                    assert_eq!(
                        <W256 as Cell>::pivot_update(&wv, &wb, &wf, &wc, &dv).map(|q| q.to_big()),
                        to_opt(&want2),
                        "({v} · {b} − {f} · {c}) / {d}"
                    );
                }
            }
        }
        // Edges: the minimum is rejected on input and by negation, and
        // values just inside the range round-trip.
        let two255: BigInt = BigInt::from(1) << 255u32;
        assert!(W256::from_big(&two255).is_none());
        assert!(W256::from_big(&-&two255).is_none());
        let max = &two255 - 1;
        let wmax = W256::from_big(&max).expect("fits");
        assert_eq!(wmax.to_big(), max);
        assert_eq!(W256::from_big(&(-&max)).map(|v| v.to_big()), Some(-&max));
        assert_eq!(<W256 as Cell>::neg(&wmax).map(|v| v.to_big()), Some(-&max));
        // The minimum can arise as an exact quotient and is then handled.
        let dv = <W256 as Cell>::divisor(&W256::from_i128(-1));
        let wmin =
            <W256 as Cell>::pivot_update(&wmax, &W256::ONE, &W256::from_i128(-1), &W256::ONE, &dv)
                .expect("(max − (−1)·1) / −1 = −(max + 1) = min");
        assert_eq!(wmin.to_big(), -&two255);
        assert!(<W256 as Cell>::neg(&wmin).is_none());
        assert_eq!(<W256 as Cell>::signum(&wmin), std::cmp::Ordering::Less);
        // Row-scaled entries take the small path and the big path alike.
        assert_eq!(
            <W256 as Cell>::from_ratio_scaled(&q(5, 3), &BigInt::from(6)).map(|v| v.to_big()),
            Some(BigInt::from(10))
        );
        let huge = Ratio::new(&two255 / 2, BigInt::from(3)); // 2^254 / 3
        assert_eq!(
            <W256 as Cell>::from_ratio_scaled(&huge, &BigInt::from(3)).map(|v| v.to_big()),
            Some(&two255 / 2)
        );
        // … and 2^254 · 2 = 2^255 does not fit.
        assert!(<W256 as Cell>::from_ratio_scaled(&huge, &BigInt::from(6)).is_none());
    }

    /// The 256-bit intermediates of the `i128` cells agree with `BigInt`
    /// on random operands spanning the whole range, including the exact
    /// division's fit/no-fit boundary.
    #[test]
    fn i256_intermediates_match_bigint() {
        let mut state: u128 = 0x9E37_79B9_7F4A_7C15_F39C_C060_5CED_C834;
        let mut next = |bits: u32| -> i128 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let mask = if bits >= 128 {
                u128::MAX
            } else {
                (1u128 << bits) - 1
            };
            let v = (state & mask) as i128;
            if state & (1 << 5) != 0 { -v } else { v }
        };
        for _ in 0..3000 {
            let (a, b, c, d) = (next(120), next(120), next(120), next(120));
            let big = |x: i128| BigInt::from(x);
            // Products and comparisons.
            let ab = I256::mul(a, b);
            let cd = I256::mul(c, d);
            assert_eq!(
                ab.cmp(&cd),
                (big(a) * big(b)).cmp(&(big(c) * big(d))),
                "{a} {b} {c} {d}"
            );
            assert_eq!(
                <i128 as Cell>::cmp_products(&a, &b, &c, &d),
                (big(a) * big(b)).cmp(&(big(c) * big(d)))
            );
            // Exact division: v = d₀·k (both ≤ 60 bits, so v fits), then
            // (v·b)/d₀ = k·b exactly; it fits an i128 for a 64-bit b and
            // usually not for a 120-bit one.
            let d0 = next(60);
            if d0 != 0 {
                let k = next(60);
                let v = d0 * k;
                let b2 = if k & 1 == 0 { next(64) } else { b };
                let exp = big(k) * big(b2);
                let dv = <i128 as Cell>::divisor(&d0);
                assert_eq!(
                    <i128 as Cell>::rescale(&v, &b2, &dv),
                    i128::try_from(&exp).ok(),
                    "{v} {b2} {d0}"
                );
                // The general update (v·p − f·pr)/d with f·pr also a multiple of d₀.
                let f = d0 * next(30);
                let pr = next(64);
                let exp2 = big(k) * big(b2) - big(f) / big(d0) * big(pr);
                assert_eq!(
                    <i128 as Cell>::pivot_update(&v, &b2, &f, &pr, &dv),
                    i128::try_from(&exp2).ok(),
                    "{v} {b2} {f} {pr} {d0}"
                );
            }
            let diff = ab.sub(cd);
            let exp = big(a) * big(b) - big(c) * big(d);
            assert_eq!(diff.to_i128(), i128::try_from(&exp).ok(), "{exp}");
            assert_eq!(diff.is_negative(), Signed::is_negative(&exp));
        }
        // Exact quotients that fill the whole i128 range.
        for &(x, d) in &[
            (i128::MAX, 1i128),
            (i128::MIN, 1),
            (i128::MIN, -1),
            (i128::MAX, i128::MAX),
            (1 << 100, 1 << 40),
        ] {
            let prod = I256::mul(x, d);
            let got = prod.div_exact(d);
            let expected =
                i128::try_from(&(BigInt::from(x) * BigInt::from(d) / BigInt::from(d))).ok();
            assert_eq!(got, expected, "{x} · {d} / {d}");
        }
        // A quotient one past the range is reported as not fitting.
        let over = I256::mul(i128::MAX, 4).sub(I256::from_i128(0));
        assert_eq!(over.div_exact(2), None);
        assert_eq!(I256::mul(i128::MAX, 4).div_exact(4), Some(i128::MAX));
    }
}
