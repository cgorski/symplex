//! Univariate factorization over ℤ via Berlekamp–Zassenhaus (mod-p factoring, Hensel lifting, recombination).
//!
//! This module implements the classical Zassenhaus algorithm for factoring a
//! square-free primitive polynomial `f ∈ ℤ[x]` into irreducibles:
//!
//! 1. **Choose primes.**  Small odd primes `p` with `p ∤ lc(f)` and `f mod p`
//!    square-free are tried; the one producing the fewest modular factors is
//!    kept.  The sets of achievable factor degrees are intersected across all
//!    tried primes, which often proves irreducibility outright.
//! 2. **Factor mod p** with Cantor–Zassenhaus (distinct-degree splitting
//!    followed by equal-degree splitting).  Arithmetic in `GF(p)[x]` uses
//!    `u64` coefficients with `p < 2³¹`.
//! 3. **Hensel lift** the modular factorization to `p^k` where
//!    `p^k > 2 · 2ⁿ · ‖f‖₂ · |lc(f)|` (the Mignotte bound) using linear
//!    multifactor lifting.
//! 4. **Recombine** by trying subsets of the lifted factors of increasing
//!    size.  Each candidate is `lc · ∏ gᵢ` reduced to symmetric residues,
//!    made primitive, and tested by exact trial division.  Cheap filters
//!    (achievable degree, constant-term divisibility) reject almost all
//!    spurious subsets without a division.
//! 5. **Verify** the final factorization by multiplying back.  If anything
//!    is inconsistent the input is returned unfactored — the result is
//!    never wrong, at worst incomplete.
//!
//! The subset search is exponential in the number of modular factors, so
//! it is bounded by [`MAX_RECOMBINATION_SUBSETS`].  When the budget is
//! exhausted the remaining (possibly reducible) cofactor is returned as a
//! single factor and a `tracing::warn!` is emitted.
//!
//! # Examples
//!
//! ```
//! use symplex::factor_zassenhaus::{factor_zassenhaus, Poly};
//! use num_bigint::BigInt;
//! use num_rational::Ratio;
//!
//! let c = |n: i64| Ratio::from_integer(BigInt::from(n));
//! // x^4 + 4 = (x^2 - 2x + 2)(x^2 + 2x + 2)   (Sophie Germain identity)
//! let f = Poly::from_coeffs(vec![c(4), c(0), c(0), c(0), c(1)]);
//! let factors = factor_zassenhaus(&f);
//! assert_eq!(factors.len(), 2);
//! assert!(factors.iter().all(|(g, m)| g.degree() == Some(2) && *m == 1));
//! ```

use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use num_rational::Ratio;
use num_traits::{One, Signed, ToPrimitive, Zero};

use super::multipoly::{MonomialOrd, MultiPoly};

// The dense polynomial types live in crate-private modules; re-export them
// (and the coefficient traits their documentation refers to) so that callers
// of this public module can construct inputs.
pub use super::dense::Poly;
pub use super::generic::GenPoly;

/// Coefficient-ring traits used by [`GenPoly`] (re-exported so that the
/// polynomial types above are fully documented and usable generically).
pub mod traits {
    pub use crate::poly::traits::{
        BindingStrength, CoeffDisplay, EuclideanDomain, Field, IntegralCoeff, Ring,
    };
}

// ═══════════════════════════════════════════════════════════════════════════
// Tunables
// ═══════════════════════════════════════════════════════════════════════════

/// A factorization over `GF(p)`: the leading coefficient and the monic
/// irreducible factors (ascending coefficients in `[0, p)`) with
/// multiplicities.  Returned by [`factor_mod_p`].
pub type ModPFactorization = (u64, Vec<(Vec<u64>, u32)>);

/// A factorization over ℤ of a multivariate polynomial: rational content
/// and `(factor, multiplicity)` pairs.  Returned by [`factor_multivariate`].
pub type MultiFactorization<O> = (Ratio<BigInt>, Vec<(MultiPoly<O>, u32)>);

/// Number of usable primes whose modular factorizations are compared; the
/// one with the fewest factors is used for lifting.
pub const PRIMES_TO_TRY: usize = 5;

/// Upper bound on the number of factor subsets examined during
/// recombination.  Beyond this the remaining cofactor is returned as-is.
pub const MAX_RECOMBINATION_SUBSETS: usize = 400_000;

/// Maximum number of candidate primes examined before giving up on finding
/// one for which `f mod p` is square-free.
const MAX_PRIMES_EXAMINED: usize = 600;

/// Largest prime allowed for the `u64` finite-field arithmetic
/// (products of two residues must fit in `u64`).
const MAX_PRIME: u64 = 1 << 31;

/// Largest univariate degree the Kronecker substitution in
/// [`factor_multivariate`] is allowed to produce.
pub const MAX_KRONECKER_SUBSTITUTION_DEGREE: usize = 96;

/// Upper bound on the factor subsets examined when regrouping the
/// univariate factors of a Kronecker image back into multivariate factors.
const MAX_MULTIVARIATE_SUBSETS: usize = 20_000;

// ═══════════════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════════════

/// Factor a polynomial over ℤ into primitive irreducible factors with
/// multiplicities.
///
/// The rational content (GCD of the coefficients, with sign) is removed and
/// **not** returned; use [`factor_zassenhaus_with_content`] to get it.  Each
/// returned factor is primitive with positive leading coefficient, and the
/// list is sorted by degree, then coefficients.  The zero polynomial and
/// constants produce an empty list.
///
/// # Examples
///
/// ```
/// use symplex::factor_zassenhaus::{factor_zassenhaus, Poly};
/// use num_bigint::BigInt;
/// use num_rational::Ratio;
///
/// let c = |n: i64| Ratio::from_integer(BigInt::from(n));
/// // 2·(x − 1)²·(x + 1) = 2x³ − 2x² − 2x + 2
/// let f = Poly::from_coeffs(vec![c(2), c(-2), c(-2), c(2)]);
/// let factors = factor_zassenhaus(&f);
/// let x_plus_1 = Poly::from_coeffs(vec![c(1), c(1)]);
/// let x_minus_1 = Poly::from_coeffs(vec![c(-1), c(1)]);
/// assert_eq!(factors, vec![(x_minus_1, 2), (x_plus_1, 1)]);
/// ```
#[must_use]
pub fn factor_zassenhaus(f: &Poly) -> Vec<(Poly, u32)> {
    factor_zassenhaus_with_content(f).1
}

/// Factor a polynomial over ℤ, returning `(content, factors)`.
///
/// `content` is the rational GCD of the coefficients with sign chosen so
/// that every factor has a positive leading coefficient, and
/// `f = content · ∏ factorᵢ^multᵢ` exactly.  Constants return
/// `(c, [])`; the zero polynomial returns `(0, [])`.
///
/// # Examples
///
/// ```
/// use symplex::factor_zassenhaus::{factor_zassenhaus_with_content, Poly};
/// use num_bigint::BigInt;
/// use num_rational::Ratio;
///
/// let c = |n: i64| Ratio::from_integer(BigInt::from(n));
/// // -3x² + 3 = -3·(x − 1)(x + 1)
/// let f = Poly::from_coeffs(vec![c(3), c(0), c(-3)]);
/// let (content, factors) = factor_zassenhaus_with_content(&f);
/// assert_eq!(content, c(-3));
/// assert_eq!(factors.len(), 2);
/// ```
#[must_use]
pub fn factor_zassenhaus_with_content(f: &Poly) -> (Ratio<BigInt>, Vec<(Poly, u32)>) {
    if f.is_zero() {
        return (Ratio::zero(), vec![]);
    }
    if f.is_constant() {
        return (f.coeff(0), vec![]);
    }

    let mut content = f.content();
    let mut prim = f.primitive_part();
    if prim.leading_coeff().is_some_and(|lc| lc.is_negative()) {
        content = -content;
        prim = -&prim;
    }

    let mut all: Vec<(Poly, u32)> = Vec::new();
    for (sf, mult) in super::dense::square_free_decomposition(&prim) {
        let coeffs = poly_to_z(&sf);
        for g in factor_squarefree_z(&coeffs) {
            all.push((z_to_poly(&g), mult));
        }
    }

    // Verify by multiplying back; fall back to the unfactored primitive part
    // if anything went wrong (this should never happen, but the contract is
    // "never wrong").
    let mut check = Poly::from_int(1);
    for (g, m) in &all {
        for _ in 0..*m {
            check = &check * g;
        }
    }
    if check != prim {
        tracing::error!(
            "factor_zassenhaus: verification failed (product of factors ≠ input); returning unfactored"
        );
        return (content, vec![(prim, 1)]);
    }

    all.sort_by(|a, b| cmp_poly(&a.0, &b.0));
    (content, all)
}

/// Factor a square-free primitive polynomial `f ∈ ℤ[x]` (coefficients in
/// ascending degree order) into irreducible integer factors.
///
/// Each factor is primitive with positive leading coefficient; the list is
/// sorted by degree then coefficients, and the product of the factors
/// equals `±f` (the sign is the sign of `lc(f)`).  Inputs that are not
/// square-free are still handled correctly (repeated factors simply appear
/// multiple times), though the caller should prefer a square-free
/// decomposition first for speed.
///
/// Constants (including zero) return an empty list.
///
/// # Examples
///
/// ```
/// use symplex::factor_zassenhaus::factor_squarefree_z;
/// use num_bigint::BigInt;
///
/// let b = |n: i64| BigInt::from(n);
/// // 6x⁴ − 7x³ − 8x² + 7x + 2 = (x − 1)(x + 1)(6x² − 7x − 2)
/// let f = vec![b(2), b(7), b(-8), b(-7), b(6)];
/// let factors = factor_squarefree_z(&f);
/// assert_eq!(factors, vec![
///     vec![b(-1), b(1)],
///     vec![b(1), b(1)],
///     vec![b(-2), b(-7), b(6)],
/// ]);
/// ```
#[must_use]
pub fn factor_squarefree_z(f: &[BigInt]) -> Vec<Vec<BigInt>> {
    let mut f = f.to_vec();
    z_normalize(&mut f);
    let Some(n) = z_degree(&f) else {
        return vec![];
    };
    if n == 0 {
        return vec![];
    }
    let mut f = z_primitive_part(&f);
    if f.last().is_some_and(|lc| lc.is_negative()) {
        for c in &mut f {
            *c = -std::mem::take(c);
        }
    }

    let mut factors: Vec<Vec<BigInt>> = Vec::new();

    // Powers of x.
    let low_zeros = f.iter().take_while(|c| c.is_zero()).count();
    if low_zeros > 0 {
        for _ in 0..low_zeros {
            factors.push(vec![BigInt::zero(), BigInt::one()]);
        }
        f.drain(..low_zeros);
    }

    if z_degree(&f).unwrap_or(0) == 0 {
        factors.sort_by(cmp_z);
        return factors;
    }

    // Not square-free?  Decompose first (Berlekamp–Zassenhaus needs a
    // square-free image mod p) and expand multiplicities into repeats.
    let fq = z_to_poly(&f);
    if fq.is_squarefree() == Some(false) {
        for (part, mult) in super::dense::square_free_decomposition(&fq) {
            let sub = factor_squarefree_z(&poly_to_z(&part));
            for _ in 0..mult {
                factors.extend(sub.iter().cloned());
            }
        }
        factors.sort_by(cmp_z);
        return factors;
    }

    // Cheap pre-pass: rational roots (linear factors) via the Rational Root
    // Theorem.  Capped internally, so this never dominates.
    let (remaining, linear) = super::dense::extract_rational_roots(&fq);
    for l in linear {
        factors.push(poly_to_z(&l));
    }
    let mut remaining = poly_to_z(&remaining);
    if remaining.last().is_some_and(|lc| lc.is_negative()) {
        for c in &mut remaining {
            *c = -std::mem::take(c);
        }
    }

    match z_degree(&remaining) {
        None | Some(0) => {}
        Some(1) => factors.push(remaining),
        Some(_) => match zassenhaus_core(&remaining) {
            Some(fs) => factors.extend(fs),
            None => {
                tracing::warn!(
                    "factor_squarefree_z: no usable prime found; returning cofactor unfactored"
                );
                factors.push(remaining);
            }
        },
    }

    factors.sort_by(cmp_z);
    factors
}

/// Decide whether a non-constant polynomial with rational coefficients is
/// irreducible over ℚ.
///
/// Returns `None` for the zero polynomial and for constants (irreducibility
/// is undefined there).  A polynomial that is not primitive is judged by
/// its primitive part, so `2x + 2` counts as irreducible.
///
/// # Examples
///
/// ```
/// use symplex::factor_zassenhaus::{is_irreducible_z, Poly};
/// use num_bigint::BigInt;
/// use num_rational::Ratio;
///
/// let c = |n: i64| Ratio::from_integer(BigInt::from(n));
/// let x2_plus_1 = Poly::from_coeffs(vec![c(1), c(0), c(1)]);
/// let x2_minus_1 = Poly::from_coeffs(vec![c(-1), c(0), c(1)]);
/// assert_eq!(is_irreducible_z(&x2_plus_1), Some(true));
/// assert_eq!(is_irreducible_z(&x2_minus_1), Some(false));
/// assert_eq!(is_irreducible_z(&Poly::from_int(7)), None);
/// ```
#[must_use]
pub fn is_irreducible_z(f: &Poly) -> Option<bool> {
    if f.is_zero() || f.is_constant() {
        return None;
    }
    let factors = factor_zassenhaus(f);
    Some(factors.len() == 1 && factors[0].1 == 1)
}

/// Factor a polynomial over the prime field `GF(p)`.
///
/// Returns `Some((lc, factors))` where `lc` is the leading coefficient of
/// `f mod p` and `factors` lists monic irreducible polynomials (coefficients
/// in ascending degree order, each in `[0, p)`) with multiplicities, so that
/// `f ≡ lc · ∏ gᵢ^mᵢ (mod p)`.  Constants return `(c, [])`; a polynomial
/// that vanishes identically mod `p` returns `(0, [])`.
///
/// Returns `None` if `p` is not an odd prime below `2³¹`, or if some
/// coefficient of `f` has a denominator divisible by `p`.
///
/// # Examples
///
/// ```
/// use symplex::factor_zassenhaus::{factor_mod_p, Poly};
/// use num_bigint::BigInt;
/// use num_rational::Ratio;
///
/// let c = |n: i64| Ratio::from_integer(BigInt::from(n));
/// // x² + 1 ≡ (x + 2)(x + 3) (mod 5)
/// let f = Poly::from_coeffs(vec![c(1), c(0), c(1)]);
/// let (lc, factors) = factor_mod_p(&f, 5).unwrap();
/// assert_eq!(lc, 1);
/// assert_eq!(factors, vec![(vec![2, 1], 1), (vec![3, 1], 1)]);
///
/// // x² + 1 is irreducible mod 3.
/// let (_, factors) = factor_mod_p(&f, 3).unwrap();
/// assert_eq!(factors, vec![(vec![1, 0, 1], 1)]);
/// ```
#[must_use]
pub fn factor_mod_p(f: &Poly, p: u64) -> Option<ModPFactorization> {
    if !is_small_odd_prime(p) {
        return None;
    }
    let pb = BigInt::from(p);
    let mut fp: FpPoly = Vec::with_capacity(f.coeffs().len());
    for c in f.coeffs() {
        let d = c.denom().mod_floor(&pb).to_u64()?;
        if d == 0 {
            return None;
        }
        let nmod = c.numer().mod_floor(&pb).to_u64()?;
        fp.push(mod_mul(nmod, mod_inv(d, p), p));
    }
    fp_normalize(&mut fp);
    let Some(deg) = fp_degree(&fp) else {
        return Some((0, vec![]));
    };
    let lc = fp[deg];
    if deg == 0 {
        return Some((lc, vec![]));
    }
    let monic = fp_monic(&fp, p);
    let mut out: Vec<(Vec<u64>, u32)> = Vec::new();
    for (g, m) in fp_squarefree_factorization(&monic, p) {
        for h in fp_factor_squarefree_monic(&g, p) {
            out.push((h, m));
        }
    }
    out.sort_by(|a, b| cmp_fp(&a.0, &b.0).then(a.1.cmp(&b.1)));
    Some((lc, out))
}

// ═══════════════════════════════════════════════════════════════════════════
// Multivariate factorization via Kronecker substitution
// ═══════════════════════════════════════════════════════════════════════════

/// Factor a multivariate polynomial over ℤ by Kronecker substitution.
///
/// Returns `Some((content, factors))` with `f = content · ∏ gᵢ^mᵢ`, every
/// `gᵢ` a primitive integer polynomial with positive leading coefficient
/// (in the monomial order `O`), sorted for determinism.  Returns `None` if
/// the polynomial is too large for the method (see
/// [`MAX_KRONECKER_SUBSTITUTION_DEGREE`]) or if the result could not be
/// verified — a result is never returned without being multiplied back.
///
/// # Algorithm
///
/// 1. Remove rational and monomial content.
/// 2. If only one variable remains, factor with [`factor_zassenhaus`].
/// 3. Otherwise substitute `xⱼ → t^{Dⱼ}` with mixed radices
///    `Dⱼ₊₁ = Dⱼ · (deg_{xⱼ} f + 1)`, which is injective on the monomials
///    of `f`, factor the univariate image, and regroup its irreducible
///    factors (with multiplicity) into subsets whose products map back to
///    genuine divisors of `f` (checked by exact multivariate division).
///    All variable orderings are tried, cheapest first.
///
/// # Examples
///
/// ```
/// use symplex::factor_zassenhaus::factor_multivariate;
/// use symplex::multipoly::MultiPoly;
///
/// let [x, y]: [MultiPoly; 2] = [MultiPoly::var(2, 0), MultiPoly::var(2, 1)];
/// // x³ − y³ = (x − y)(x² + xy + y²)
/// let f = x.mul(&x).mul(&x).sub(&y.mul(&y).mul(&y));
/// let (content, factors) = factor_multivariate(&f).unwrap();
/// assert!(content.is_integer());
/// assert_eq!(factors.len(), 2);
/// let mut back = MultiPoly::constant(2, content);
/// for (g, m) in &factors {
///     for _ in 0..*m { back = back.mul(g); }
/// }
/// assert_eq!(back, f);
/// ```
#[must_use]
pub fn factor_multivariate<O: MonomialOrd>(f: &MultiPoly<O>) -> Option<MultiFactorization<O>> {
    let nv = f.num_vars();
    if f.is_zero() {
        return Some((Ratio::zero(), vec![]));
    }

    // 1. Rational content and sign.
    let mut prim = f.primitive_part_q();
    let mut content = f.leading_coeff()? / prim.leading_coeff()?;
    if prim.leading_coeff().is_some_and(|c| c.is_negative()) {
        prim = prim.neg();
        content = -content;
    }

    let mut factors: Vec<(MultiPoly<O>, u32)> = Vec::new();

    // Monomial content.
    let mono = prim.monomial_content();
    if mono.iter().any(|&e| e > 0) {
        let mut stripped = MultiPoly::zero(nv);
        for (exp, c) in prim.terms() {
            let new_exp: Vec<u32> = exp.iter().zip(&mono).map(|(e, m)| e - m).collect();
            stripped = stripped.add(&MultiPoly::monomial(c.clone(), new_exp));
        }
        for (i, &e) in mono.iter().enumerate() {
            if e > 0 {
                factors.push((MultiPoly::var(nv, i), e));
            }
        }
        prim = stripped;
    }

    let present = prim.variables_present();
    match present.len() {
        0 => {
            // Constant left over after stripping monomials.
            if let Some(c) = prim.leading_coeff() {
                content *= c;
            }
        }
        1 => {
            let v = present[0];
            let uni = multipoly_to_uni(&prim, v);
            let (c, fs) = factor_zassenhaus_with_content(&uni);
            content *= c;
            for (g, m) in fs {
                factors.push((uni_to_multipoly(&g, v, nv), m));
            }
        }
        _ => {
            let found = kronecker_factor_all_orders(&prim)?;
            for g in found {
                match factors.iter_mut().find(|(h, _)| *h == g) {
                    Some(entry) => entry.1 += 1,
                    None => factors.push((g, 1)),
                }
            }
        }
    }

    // Verify by multiplying back.
    let mut back = MultiPoly::constant(nv, content.clone());
    for (g, m) in &factors {
        for _ in 0..*m {
            back = back.mul(g);
        }
    }
    if back != *f {
        tracing::error!("factor_multivariate: verification failed; returning None");
        return None;
    }

    factors.sort_by(|a, b| cmp_multipoly(&a.0, &b.0));
    Some((content, factors))
}

/// Try Kronecker substitution over all orderings of the variables present
/// in `f`, cheapest (lowest univariate degree) first.
fn kronecker_factor_all_orders<O: MonomialOrd>(f: &MultiPoly<O>) -> Option<Vec<MultiPoly<O>>> {
    let present = f.variables_present();
    let mut orders: Vec<(usize, Vec<usize>)> = Vec::new();
    for perm in permutations(&present) {
        let mut radix = 1usize;
        let mut total = 0usize;
        let mut ok = true;
        for &v in &perm {
            let d = f.degree_in(v) as usize;
            total = total.saturating_add(d.saturating_mul(radix));
            radix = radix.saturating_mul(d + 1);
            if total > MAX_KRONECKER_SUBSTITUTION_DEGREE {
                ok = false;
                break;
            }
        }
        if ok {
            orders.push((total, perm));
        }
    }
    orders.sort();
    for (_, order) in orders {
        if let Some(result) = kronecker_factor_with_order(f, &order) {
            return Some(result);
        }
    }
    None
}

/// All permutations of a small slice.
fn permutations(items: &[usize]) -> Vec<Vec<usize>> {
    if items.len() <= 1 {
        return vec![items.to_vec()];
    }
    let mut out = Vec::new();
    for i in 0..items.len() {
        let mut rest = items.to_vec();
        let head = rest.remove(i);
        for mut tail in permutations(&rest) {
            tail.insert(0, head);
            out.push(tail);
        }
    }
    out
}

/// Kronecker substitution with a fixed variable ordering.  `f` must be a
/// primitive integer polynomial involving at least two variables.
fn kronecker_factor_with_order<O: MonomialOrd>(
    f: &MultiPoly<O>,
    order: &[usize],
) -> Option<Vec<MultiPoly<O>>> {
    let nv = f.num_vars();
    let mut radices = Vec::with_capacity(order.len());
    let mut radix = 1usize;
    for &v in order {
        radices.push(radix);
        radix *= f.degree_in(v) as usize + 1;
    }
    let total_slots = radix;

    // Univariate image.
    let mut coeffs = vec![Ratio::zero(); total_slots];
    for (exp, c) in f.terms() {
        let k: usize = order
            .iter()
            .zip(&radices)
            .map(|(&v, &r)| exp[v] as usize * r)
            .sum();
        coeffs[k] = c.clone();
    }
    let image = Poly::from_coeffs(coeffs);
    let (_content, ufactors) = factor_zassenhaus_with_content(&image);

    // Multiset of univariate irreducibles.
    let mut pool: Vec<Poly> = Vec::new();
    for (g, m) in ufactors {
        for _ in 0..m {
            pool.push(g.clone());
        }
    }

    let mut remaining = f.clone();
    let mut found: Vec<MultiPoly<O>> = Vec::new();
    let mut budget = MAX_MULTIVARIATE_SUBSETS;
    let mut s = 1usize;
    'outer: while s <= pool.len() {
        let r = pool.len();
        let mut idx: Vec<usize> = (0..s).collect();
        loop {
            if budget == 0 {
                break 'outer;
            }
            budget -= 1;
            let mut prod = Poly::from_int(1);
            for &i in &idx {
                prod = &prod * &pool[i];
            }
            let cand = normalize_sign(map_back(&prod, order, &radices, nv));
            if !cand.is_zero()
                && cand.total_degree().unwrap_or(0) > 0
                && let Some(q) = remaining.div_exact(&cand)
            {
                found.push(cand);
                remaining = q;
                for &i in idx.iter().rev() {
                    pool.remove(i);
                }
                continue 'outer;
            }
            if !next_combination(&mut idx, r) {
                break;
            }
        }
        s += 1;
    }

    if remaining.total_degree().unwrap_or(0) > 0 {
        // Could not be regrouped completely (budget) — keep the cofactor.
        found.push(normalize_sign(remaining.primitive_part_q()));
    }
    Some(found)
}

/// Inverse Kronecker map: exponent `k` of `t` becomes the mixed-radix
/// digits placed at `order[j]`.
fn map_back<O: MonomialOrd>(
    p: &Poly,
    order: &[usize],
    radices: &[usize],
    num_vars: usize,
) -> MultiPoly<O> {
    let mut out = MultiPoly::zero(num_vars);
    for (k, c) in p.coeffs().iter().enumerate() {
        if c.is_zero() {
            continue;
        }
        let mut exp = vec![0u32; num_vars];
        let mut rest = k;
        for j in (0..order.len()).rev() {
            let digit = rest / radices[j];
            rest %= radices[j];
            exp[order[j]] = digit as u32;
        }
        out = out.add(&MultiPoly::monomial(c.clone(), exp));
    }
    out
}

/// Make the leading coefficient positive.
fn normalize_sign<O: MonomialOrd>(p: MultiPoly<O>) -> MultiPoly<O> {
    if p.leading_coeff().is_some_and(|c| c.is_negative()) {
        p.neg()
    } else {
        p
    }
}

/// View a polynomial that only involves variable `v` as a univariate `Poly`.
fn multipoly_to_uni<O: MonomialOrd>(p: &MultiPoly<O>, v: usize) -> Poly {
    let deg = p.degree_in(v) as usize;
    let mut coeffs = vec![Ratio::zero(); deg + 1];
    for (exp, c) in p.terms() {
        coeffs[exp[v] as usize] += c;
    }
    Poly::from_coeffs(coeffs)
}

/// Embed a univariate polynomial as a `MultiPoly` in variable `v`.
fn uni_to_multipoly<O: MonomialOrd>(p: &Poly, v: usize, num_vars: usize) -> MultiPoly<O> {
    let mut out = MultiPoly::zero(num_vars);
    for (k, c) in p.coeffs().iter().enumerate() {
        if c.is_zero() {
            continue;
        }
        let mut exp = vec![0u32; num_vars];
        exp[v] = k as u32;
        out = out.add(&MultiPoly::monomial(c.clone(), exp));
    }
    out
}

fn cmp_multipoly<O: MonomialOrd>(a: &MultiPoly<O>, b: &MultiPoly<O>) -> std::cmp::Ordering {
    let ta: Vec<(Vec<u32>, Ratio<BigInt>)> =
        a.terms().map(|(e, c)| (e.to_vec(), c.clone())).collect();
    let tb: Vec<(Vec<u32>, Ratio<BigInt>)> =
        b.terms().map(|(e, c)| (e.to_vec(), c.clone())).collect();
    a.total_degree()
        .cmp(&b.total_degree())
        .then_with(|| ta.len().cmp(&tb.len()))
        .then_with(|| ta.cmp(&tb))
}

// ═══════════════════════════════════════════════════════════════════════════
// Core algorithm
// ═══════════════════════════════════════════════════════════════════════════

/// Integer polynomial, ascending degree order, normalized (no trailing zeros).
type ZPoly = Vec<BigInt>;
/// Polynomial over `GF(p)`, ascending degree order, coefficients in `[0, p)`.
type FpPoly = Vec<u64>;

/// Berlekamp–Zassenhaus on a square-free primitive `f` of degree ≥ 2 with
/// positive leading coefficient and `f(0) ≠ 0`.
///
/// Returns `None` if no usable prime could be found.
fn zassenhaus_core(f: &ZPoly) -> Option<Vec<ZPoly>> {
    let n = z_degree(f)?;
    let lc = f[n].clone();

    // ── 1. Choose a prime ──────────────────────────────────────────────
    let mut degree_mask = vec![true; n + 1];
    let mut best: Option<(u64, Vec<FpPoly>)> = None;
    let mut usable = 0usize;
    let mut examined = 0usize;

    for p in odd_primes() {
        examined += 1;
        if examined > MAX_PRIMES_EXAMINED {
            break;
        }
        if (&lc % p).is_zero() {
            continue;
        }
        let fp = z_to_fp(f, p);
        if fp_degree(&fp) != Some(n) || !fp_is_squarefree(&fp, p) {
            continue;
        }
        let monic = fp_monic(&fp, p);
        let factors = fp_factor_squarefree_monic(&monic, p);
        if factors.len() <= 1 {
            return Some(vec![f.clone()]);
        }

        // Intersect the achievable degree set.
        let degs: Vec<usize> = factors.iter().map(|g| fp_degree(g).unwrap_or(0)).collect();
        let achievable = subset_sums(&degs, n);
        for (m, a) in degree_mask.iter_mut().zip(achievable.iter()) {
            *m &= *a;
        }
        if degree_mask
            .iter()
            .enumerate()
            .all(|(d, &ok)| !ok || d == 0 || d == n)
        {
            tracing::debug!("factor_zassenhaus: degree analysis proves irreducibility");
            return Some(vec![f.clone()]);
        }

        let better = match &best {
            None => true,
            Some((bp, bf)) => factors.len() < bf.len() || (factors.len() == bf.len() && p > *bp),
        };
        if better {
            best = Some((p, factors));
        }
        usable += 1;
        if usable >= PRIMES_TO_TRY {
            break;
        }
    }

    let (p, modular_factors) = best?;
    let r = modular_factors.len();
    tracing::debug!(
        prime = p,
        modular_factors = r,
        degree = n,
        "factor_zassenhaus: selected prime"
    );

    // ── 2. Lifting target from the Mignotte bound ──────────────────────
    let bound = mignotte_bound(f);
    let two_bound = &bound * 2;
    let pb = BigInt::from(p);
    let mut pk = pb.clone();
    let mut k = 1u32;
    while pk <= two_bound {
        pk *= &pb;
        k += 1;
    }

    // ── 3. Hensel lift ─────────────────────────────────────────────────
    let lifted = hensel_lift(f, p, &modular_factors, k);
    debug_assert_eq!(lifted.len(), r);

    // ── 4. Recombine ───────────────────────────────────────────────────
    let (mut found, remainder, complete) = recombine(f, &pk, lifted, &degree_mask);

    if let Some(rem) = remainder {
        if complete {
            found.push(rem);
        } else {
            tracing::warn!(
                degree = z_degree(&rem).unwrap_or(0),
                "factor_zassenhaus: recombination budget exhausted; trying Kronecker fallback on cofactor"
            );
            found.extend(kronecker_fallback(&rem));
        }
    }

    Some(found)
}

/// Fallback used when the recombination budget is exhausted: Kronecker's
/// method for factors of small degree.  Whatever cannot be split is
/// returned as a single (possibly reducible) factor.
fn kronecker_fallback(f: &ZPoly) -> Vec<ZPoly> {
    let mut remaining = z_to_poly(f);
    let mut out: Vec<ZPoly> = Vec::new();
    let max_trial = (remaining.degree().unwrap_or(0) / 2).min(super::MAX_KRONECKER_DEGREE);
    for trial_deg in 2..=max_trial {
        loop {
            let rem_deg = remaining.degree().unwrap_or(0);
            if rem_deg < 2 * trial_deg {
                break;
            }
            match super::dense::kronecker_find_factor(&remaining, trial_deg) {
                Some((fac, quot)) => {
                    out.extend(factor_squarefree_z(&poly_to_z(&fac)));
                    remaining = quot;
                }
                None => break,
            }
        }
    }
    if remaining.degree().unwrap_or(0) >= 1 {
        out.push(poly_to_z(&super::dense::ensure_positive_lc(&remaining)));
    }
    out
}

/// Mignotte-style bound `2ⁿ · ⌈‖f‖₂⌉ · |lc(f)|` on the coefficients of
/// `lc(f) · g` for any monic-scaled factor `g` of `f`.
fn mignotte_bound(f: &ZPoly) -> BigInt {
    let n = z_degree(f).unwrap_or(0);
    let sum_sq: BigInt = f.iter().map(|c| c * c).sum();
    let norm = isqrt_ceil(&sum_sq);
    let lc_abs = f[n].abs();
    (BigInt::one() << n) * norm * lc_abs
}

/// Smallest integer `s` with `s² ≥ n` (for `n ≥ 0`).
fn isqrt_ceil(n: &BigInt) -> BigInt {
    if n.is_zero() {
        return BigInt::zero();
    }
    let s = n.sqrt();
    if &s * &s < *n { s + 1 } else { s }
}

/// Bitset of achievable subset sums of `degs`, capped at `n`.
fn subset_sums(degs: &[usize], n: usize) -> Vec<bool> {
    let mut reach = vec![false; n + 1];
    reach[0] = true;
    for &d in degs {
        for s in (d..=n).rev() {
            if reach[s - d] {
                reach[s] = true;
            }
        }
    }
    reach
}

// ═══════════════════════════════════════════════════════════════════════════
// Hensel lifting (linear, multifactor)
// ═══════════════════════════════════════════════════════════════════════════

/// Lift `f ≡ lc(f) · ∏ gᵢ (mod p)` (with `gᵢ` monic and pairwise coprime
/// mod `p`) to a factorization modulo `p^k`.
///
/// Returns the lifted monic factors with coefficients in `[0, p^k)`.
fn hensel_lift(f: &ZPoly, p: u64, factors: &[FpPoly], k: u32) -> Vec<ZPoly> {
    let n = z_degree(f).unwrap_or(0);
    let lc_p = f[n].mod_floor(&BigInt::from(p)).to_u64().unwrap_or(0);
    let lc_inv = mod_inv(lc_p, p);
    let r = factors.len();

    // Bezout coefficients: sᵢ with Σ sᵢ · ∏_{j≠i} gⱼ ≡ 1 (mod p), deg sᵢ < deg gᵢ.
    let mut s_coeffs: Vec<FpPoly> = Vec::with_capacity(r);
    for i in 0..r {
        let mut others = vec![1u64];
        for (j, g) in factors.iter().enumerate() {
            if j != i {
                others = fp_mul(&others, g, p);
            }
        }
        let (u, _v, _g) = fp_extended_gcd(&others, &factors[i], p);
        let (_, s) = fp_div_rem(&u, &factors[i], p);
        s_coeffs.push(s);
    }

    let pb = BigInt::from(p);
    let mut lifted: Vec<ZPoly> = factors
        .iter()
        .map(|g| g.iter().map(|&c| BigInt::from(c)).collect())
        .collect();
    let mut modulus = pb.clone();

    for _step in 1..k {
        let next = &modulus * &pb;

        // prod = lc · ∏ gᵢ  (mod next)
        let mut prod: ZPoly = vec![f[n].mod_floor(&next)];
        for g in &lifted {
            prod = z_mul_mod(&prod, g, &next);
        }

        // e = (f − prod) / modulus, reduced mod p.
        let mut e_p: FpPoly = vec![0; n.max(1)];
        let mut any = false;
        for (i, slot) in e_p.iter_mut().enumerate().take(n) {
            let fi = f.get(i).cloned().unwrap_or_else(BigInt::zero);
            let pi = prod.get(i).cloned().unwrap_or_else(BigInt::zero);
            let diff = (fi - pi).mod_floor(&next);
            debug_assert!((&diff % &modulus).is_zero());
            let q = diff / &modulus;
            let c = q.mod_floor(&pb).to_u64().unwrap_or(0);
            if c != 0 {
                any = true;
            }
            *slot = c;
        }
        fp_normalize(&mut e_p);
        if any {
            let e_scaled = fp_scale(&e_p, lc_inv, p);
            for (i, g) in lifted.iter_mut().enumerate() {
                let se = fp_mul(&s_coeffs[i], &e_scaled, p);
                let (_, delta) = fp_div_rem(&se, &factors[i], p);
                for (j, &d) in delta.iter().enumerate() {
                    if d != 0 {
                        g[j] += &modulus * BigInt::from(d);
                    }
                }
            }
        }
        modulus = next;
    }

    lifted
}

// ═══════════════════════════════════════════════════════════════════════════
// Recombination
// ═══════════════════════════════════════════════════════════════════════════

/// Try subsets of the lifted factors to find true factors of `f`.
///
/// Returns `(found_factors, remaining_cofactor, complete)`.  When `complete`
/// is `true` the remaining cofactor (if any) is irreducible; otherwise the
/// search budget was exhausted and it may still be reducible.
fn recombine(
    f: &ZPoly,
    pk: &BigInt,
    mut g: Vec<ZPoly>,
    degree_mask: &[bool],
) -> (Vec<ZPoly>, Option<ZPoly>, bool) {
    let mut f_cur = f.clone();
    let mut found: Vec<ZPoly> = Vec::new();
    let mut budget = MAX_RECOMBINATION_SUBSETS;
    let mut s = 1usize;

    'outer: while 2 * s <= g.len() {
        let r = g.len();
        let n_cur = z_degree(&f_cur).unwrap_or(0);
        let lc_cur = f_cur[n_cur].clone();
        let lc_f0 = &lc_cur * &f_cur[0];
        let degs: Vec<usize> = g.iter().map(|gi| z_degree(gi).unwrap_or(0)).collect();

        let mut idx: Vec<usize> = (0..s).collect();
        loop {
            if budget == 0 {
                break 'outer;
            }
            budget -= 1;

            let deg_sum: usize = idx.iter().map(|&i| degs[i]).sum();
            let degree_ok = deg_sum <= n_cur && degree_mask.get(deg_sum).copied().unwrap_or(false);

            if degree_ok {
                // Constant-term filter.
                let mut c = lc_cur.mod_floor(pk);
                for &i in &idx {
                    c = (&c * &g[i][0]).mod_floor(pk);
                }
                let c = symmetric(c, pk);
                if !c.is_zero() && (&lc_f0 % &c).is_zero() {
                    // Full candidate.
                    let mut cand: ZPoly = vec![lc_cur.mod_floor(pk)];
                    for &i in &idx {
                        cand = z_mul_mod(&cand, &g[i], pk);
                    }
                    for coeff in &mut cand {
                        *coeff = symmetric(std::mem::take(coeff), pk);
                    }
                    z_normalize(&mut cand);
                    let cand = z_primitive_part(&cand);
                    if z_degree(&cand).is_some_and(|d| d >= 1)
                        && let Some(quot) = z_div_exact(&f_cur, &cand)
                    {
                        found.push(cand);
                        f_cur = quot;
                        for &i in idx.iter().rev() {
                            g.remove(i);
                        }
                        continue 'outer;
                    }
                }
            }

            if !next_combination(&mut idx, r) {
                break;
            }
        }
        s += 1;
    }

    let complete = 2 * s > g.len() || g.is_empty();
    let remainder = if z_degree(&f_cur).is_some_and(|d| d >= 1) {
        let mut rem = z_primitive_part(&f_cur);
        if rem.last().is_some_and(|lc| lc.is_negative()) {
            for c in &mut rem {
                *c = -std::mem::take(c);
            }
        }
        Some(rem)
    } else {
        None
    };
    (found, remainder, complete)
}

/// Advance `idx` to the next `k`-combination of `0..n` in lexicographic
/// order.  Returns `false` when exhausted.
fn next_combination(idx: &mut [usize], n: usize) -> bool {
    let k = idx.len();
    if k == 0 {
        return false;
    }
    let mut i = k;
    while i > 0 {
        i -= 1;
        if idx[i] < n - k + i {
            idx[i] += 1;
            for j in i + 1..k {
                idx[j] = idx[j - 1] + 1;
            }
            return true;
        }
    }
    false
}

/// Symmetric residue of `x ∈ [0, m)` in `(−m/2, m/2]`.
fn symmetric(x: BigInt, m: &BigInt) -> BigInt {
    if &x * 2 > *m { x - m } else { x }
}

// ═══════════════════════════════════════════════════════════════════════════
// ℤ[x] helpers
// ═══════════════════════════════════════════════════════════════════════════

fn z_normalize(f: &mut ZPoly) {
    while f.last().is_some_and(|c| c.is_zero()) {
        f.pop();
    }
}

fn z_degree(f: &ZPoly) -> Option<usize> {
    if f.is_empty() {
        None
    } else {
        Some(f.len() - 1)
    }
}

/// Product with all coefficients reduced modulo `m` into `[0, m)`.
fn z_mul_mod(a: &ZPoly, b: &ZPoly, m: &BigInt) -> ZPoly {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }
    let mut out = vec![BigInt::zero(); a.len() + b.len() - 1];
    for (i, ai) in a.iter().enumerate() {
        if ai.is_zero() {
            continue;
        }
        for (j, bj) in b.iter().enumerate() {
            out[i + j] += ai * bj;
        }
    }
    for c in &mut out {
        *c = c.mod_floor(m);
    }
    z_normalize(&mut out);
    out
}

fn z_content(f: &ZPoly) -> BigInt {
    let mut g = BigInt::zero();
    for c in f {
        g = g.gcd(c);
    }
    g
}

fn z_primitive_part(f: &ZPoly) -> ZPoly {
    let c = z_content(f);
    if c.is_zero() || c.is_one() {
        return f.clone();
    }
    f.iter().map(|x| x / &c).collect()
}

/// Exact division in ℤ\[x\]; `None` if `g ∤ f` (or `g = 0`).
fn z_div_exact(f: &ZPoly, g: &ZPoly) -> Option<ZPoly> {
    let dg = z_degree(g)?;
    let Some(df) = z_degree(f) else {
        return Some(vec![]);
    };
    if df < dg {
        return None;
    }
    let lc_g = &g[dg];
    let mut rem = f.clone();
    let mut quot = vec![BigInt::zero(); df - dg + 1];
    while let Some(dr) = z_degree(&rem) {
        if dr < dg {
            return None;
        }
        let (c, r) = rem[dr].div_rem(lc_g);
        if !r.is_zero() {
            return None;
        }
        let shift = dr - dg;
        for (j, gj) in g.iter().enumerate() {
            rem[shift + j] -= &c * gj;
        }
        quot[shift] = c;
        z_normalize(&mut rem);
    }
    Some(quot)
}

fn z_to_fp(f: &ZPoly, p: u64) -> FpPoly {
    let pb = BigInt::from(p);
    let mut out: FpPoly = f
        .iter()
        .map(|c| c.mod_floor(&pb).to_u64().unwrap_or(0))
        .collect();
    fp_normalize(&mut out);
    out
}

fn poly_to_z(p: &Poly) -> ZPoly {
    // Callers pass polynomials that are already primitive integer
    // polynomials; clear any stray denominators defensively.
    let mut denom = BigInt::one();
    for c in p.coeffs() {
        denom = denom.lcm(c.denom());
    }
    let mut out: ZPoly = p
        .coeffs()
        .iter()
        .map(|c| c.numer() * (&denom / c.denom()))
        .collect();
    z_normalize(&mut out);
    out
}

fn z_to_poly(f: &ZPoly) -> Poly {
    Poly::from_coeffs(f.iter().cloned().map(Ratio::from_integer).collect())
}

fn cmp_z(a: &ZPoly, b: &ZPoly) -> std::cmp::Ordering {
    a.len().cmp(&b.len()).then_with(|| a.cmp(b))
}

fn cmp_poly(a: &Poly, b: &Poly) -> std::cmp::Ordering {
    a.coeffs()
        .len()
        .cmp(&b.coeffs().len())
        .then_with(|| a.coeffs().cmp(b.coeffs()))
}

fn cmp_fp(a: &FpPoly, b: &FpPoly) -> std::cmp::Ordering {
    a.len().cmp(&b.len()).then_with(|| a.cmp(b))
}

// ═══════════════════════════════════════════════════════════════════════════
// GF(p) scalar helpers
// ═══════════════════════════════════════════════════════════════════════════

#[inline]
fn mod_mul(a: u64, b: u64, p: u64) -> u64 {
    ((a as u128 * b as u128) % p as u128) as u64
}

fn mod_pow(mut base: u64, mut exp: u64, p: u64) -> u64 {
    let mut result = 1u64 % p;
    base %= p;
    while exp > 0 {
        if exp & 1 == 1 {
            result = mod_mul(result, base, p);
        }
        base = mod_mul(base, base, p);
        exp >>= 1;
    }
    result
}

/// Inverse of `a` modulo the prime `p` (Fermat).  `a` must be nonzero mod `p`.
fn mod_inv(a: u64, p: u64) -> u64 {
    mod_pow(a % p, p - 2, p)
}

/// Deterministic primality test for odd `p < 2³¹` by trial division.
fn is_small_odd_prime(p: u64) -> bool {
    if !(3..MAX_PRIME).contains(&p) || p.is_multiple_of(2) {
        return false;
    }
    let mut d = 3u64;
    while d * d <= p {
        if p.is_multiple_of(d) {
            return false;
        }
        d += 2;
    }
    true
}

/// Iterator over odd primes `3, 5, 7, 11, …` (below [`MAX_PRIME`]).
fn odd_primes() -> impl Iterator<Item = u64> {
    (3u64..MAX_PRIME)
        .step_by(2)
        .filter(|&p| is_small_odd_prime(p))
}

/// Tiny deterministic xorshift generator for Cantor–Zassenhaus splitting.
struct XorShift(u64);

impl XorShift {
    fn new(seed: u64) -> Self {
        XorShift(seed.max(1) ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GF(p)[x] arithmetic
// ═══════════════════════════════════════════════════════════════════════════

fn fp_normalize(f: &mut FpPoly) {
    while f.last() == Some(&0) {
        f.pop();
    }
}

fn fp_degree(f: &FpPoly) -> Option<usize> {
    if f.is_empty() {
        None
    } else {
        Some(f.len() - 1)
    }
}

#[cfg(test)]
fn fp_add(a: &FpPoly, b: &FpPoly, p: u64) -> FpPoly {
    let n = a.len().max(b.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let x = a.get(i).copied().unwrap_or(0);
        let y = b.get(i).copied().unwrap_or(0);
        out.push((x + y) % p);
    }
    fp_normalize(&mut out);
    out
}

fn fp_sub(a: &FpPoly, b: &FpPoly, p: u64) -> FpPoly {
    let n = a.len().max(b.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let x = a.get(i).copied().unwrap_or(0);
        let y = b.get(i).copied().unwrap_or(0);
        out.push((x + p - y) % p);
    }
    fp_normalize(&mut out);
    out
}

fn fp_scale(a: &FpPoly, c: u64, p: u64) -> FpPoly {
    let mut out: FpPoly = a.iter().map(|&x| mod_mul(x, c, p)).collect();
    fp_normalize(&mut out);
    out
}

fn fp_mul(a: &FpPoly, b: &FpPoly, p: u64) -> FpPoly {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }
    let mut out = vec![0u64; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        if ai == 0 {
            continue;
        }
        for (j, &bj) in b.iter().enumerate() {
            out[i + j] = (out[i + j] + mod_mul(ai, bj, p)) % p;
        }
    }
    fp_normalize(&mut out);
    out
}

/// Euclidean division in `GF(p)[x]`.  `b` must be nonzero.
fn fp_div_rem(a: &FpPoly, b: &FpPoly, p: u64) -> (FpPoly, FpPoly) {
    let db = fp_degree(b).expect("fp_div_rem: division by zero polynomial");
    let Some(da) = fp_degree(a) else {
        return (vec![], vec![]);
    };
    if da < db {
        return (vec![], a.clone());
    }
    let inv_lc = mod_inv(b[db], p);
    let mut rem = a.clone();
    let mut quot = vec![0u64; da - db + 1];
    while let Some(dr) = fp_degree(&rem) {
        if dr < db {
            break;
        }
        let c = mod_mul(rem[dr], inv_lc, p);
        let shift = dr - db;
        quot[shift] = c;
        for (j, &bj) in b.iter().enumerate() {
            let sub = mod_mul(c, bj, p);
            rem[shift + j] = (rem[shift + j] + p - sub) % p;
        }
        fp_normalize(&mut rem);
    }
    fp_normalize(&mut quot);
    (quot, rem)
}

fn fp_rem(a: &FpPoly, b: &FpPoly, p: u64) -> FpPoly {
    fp_div_rem(a, b, p).1
}

fn fp_monic(a: &FpPoly, p: u64) -> FpPoly {
    match fp_degree(a) {
        None => vec![],
        Some(d) => {
            if a[d] == 1 {
                a.clone()
            } else {
                fp_scale(a, mod_inv(a[d], p), p)
            }
        }
    }
}

/// Monic GCD in `GF(p)[x]`.
fn fp_gcd(a: &FpPoly, b: &FpPoly, p: u64) -> FpPoly {
    let mut a = a.clone();
    let mut b = b.clone();
    while !b.is_empty() {
        let r = fp_rem(&a, &b, p);
        a = b;
        b = r;
    }
    fp_monic(&a, p)
}

/// Extended Euclid: returns `(u, v, g)` with `u·a + v·b = g`, `g` monic.
fn fp_extended_gcd(a: &FpPoly, b: &FpPoly, p: u64) -> (FpPoly, FpPoly, FpPoly) {
    let (mut r0, mut r1) = (a.clone(), b.clone());
    let (mut s0, mut s1) = (vec![1u64], vec![]);
    let (mut t0, mut t1) = (vec![], vec![1u64]);
    while !r1.is_empty() {
        let (q, r) = fp_div_rem(&r0, &r1, p);
        let s = fp_sub(&s0, &fp_mul(&q, &s1, p), p);
        let t = fp_sub(&t0, &fp_mul(&q, &t1, p), p);
        r0 = r1;
        r1 = r;
        s0 = s1;
        s1 = s;
        t0 = t1;
        t1 = t;
    }
    if let Some(d) = fp_degree(&r0) {
        let inv = mod_inv(r0[d], p);
        return (
            fp_scale(&s0, inv, p),
            fp_scale(&t0, inv, p),
            fp_scale(&r0, inv, p),
        );
    }
    (s0, t0, r0)
}

fn fp_derivative(a: &FpPoly, p: u64) -> FpPoly {
    if a.len() <= 1 {
        return vec![];
    }
    let mut out: FpPoly = a
        .iter()
        .enumerate()
        .skip(1)
        .map(|(i, &c)| mod_mul(c, (i as u64) % p, p))
        .collect();
    fp_normalize(&mut out);
    out
}

fn fp_is_squarefree(a: &FpPoly, p: u64) -> bool {
    let d = fp_derivative(a, p);
    if d.is_empty() {
        return fp_degree(a) == Some(0);
    }
    fp_degree(&fp_gcd(a, &d, p)) == Some(0)
}

/// `base^exp mod m` in `GF(p)[x]` with a `u64` exponent.
fn fp_powmod_u64(base: &FpPoly, mut exp: u64, m: &FpPoly, p: u64) -> FpPoly {
    let mut result = vec![1u64];
    let mut b = fp_rem(base, m, p);
    while exp > 0 {
        if exp & 1 == 1 {
            result = fp_rem(&fp_mul(&result, &b, p), m, p);
        }
        exp >>= 1;
        if exp > 0 {
            b = fp_rem(&fp_mul(&b, &b, p), m, p);
        }
    }
    result
}

/// `base^exp mod m` in `GF(p)[x]` with an arbitrary-precision exponent.
fn fp_powmod_big(base: &FpPoly, exp: &BigUint, m: &FpPoly, p: u64) -> FpPoly {
    let mut result = vec![1u64];
    let mut b = fp_rem(base, m, p);
    let bits = exp.bits();
    for i in 0..bits {
        if exp.bit(i) {
            result = fp_rem(&fp_mul(&result, &b, p), m, p);
        }
        if i + 1 < bits {
            b = fp_rem(&fp_mul(&b, &b, p), m, p);
        }
    }
    result
}

/// `p`-th root of a polynomial in `GF(p)[x]` whose derivative vanishes
/// (i.e. `a(x) = b(x^p)`): returns `b`.
fn fp_pth_root(a: &FpPoly, p: u64) -> FpPoly {
    let step = p as usize;
    let mut out: FpPoly = a.iter().step_by(step).copied().collect();
    fp_normalize(&mut out);
    out
}

/// Square-free factorization of a monic polynomial over `GF(p)`.
///
/// Returns `[(g₁, m₁), …]` with `a = ∏ gᵢ^mᵢ`, each `gᵢ` monic square-free
/// and pairwise coprime.  Recursion depth is bounded by `log_p(deg a)`.
fn fp_squarefree_factorization(a: &FpPoly, p: u64) -> Vec<(FpPoly, u32)> {
    let mut result: Vec<(FpPoly, u32)> = Vec::new();
    let Some(d) = fp_degree(a) else {
        return result;
    };
    if d == 0 {
        return result;
    }

    let da = fp_derivative(a, p);
    if da.is_empty() {
        // a = b(x^p) = b(x)^p.
        let b = fp_pth_root(a, p);
        for (g, m) in fp_squarefree_factorization(&b, p) {
            result.push((g, m * (p as u32)));
        }
        return result;
    }

    let mut c = fp_gcd(a, &da, p);
    let mut w = fp_div_rem(a, &c, p).0;
    let mut i = 1u32;
    while fp_degree(&w).unwrap_or(0) > 0 {
        let y = fp_gcd(&w, &c, p);
        let z = fp_div_rem(&w, &y, p).0;
        if fp_degree(&z).unwrap_or(0) > 0 {
            result.push((fp_monic(&z, p), i));
        }
        i += 1;
        w = y;
        c = fp_div_rem(&c, &w, p).0;
    }
    if fp_degree(&c).unwrap_or(0) > 0 {
        let b = fp_pth_root(&c, p);
        for (g, m) in fp_squarefree_factorization(&b, p) {
            result.push((g, m * (p as u32)));
        }
    }
    result
}

/// Factor a monic square-free polynomial over `GF(p)` into monic
/// irreducibles (Cantor–Zassenhaus).  Sorted by degree, then coefficients.
fn fp_factor_squarefree_monic(a: &FpPoly, p: u64) -> Vec<FpPoly> {
    let mut out: Vec<FpPoly> = Vec::new();
    let Some(d) = fp_degree(a) else {
        return out;
    };
    if d == 0 {
        return out;
    }
    let mut rng = XorShift::new(p ^ ((d as u64) << 32));
    for (g, deg) in fp_distinct_degree(a, p) {
        fp_equal_degree(&g, deg, p, &mut rng, &mut out);
    }
    out.sort_by(cmp_fp);
    out
}

/// Distinct-degree factorization: returns `[(g, d)]` where every irreducible
/// factor of `g` has degree exactly `d`.
fn fp_distinct_degree(a: &FpPoly, p: u64) -> Vec<(FpPoly, usize)> {
    let mut result = Vec::new();
    let mut f = a.clone();
    let x: FpPoly = vec![0, 1];
    let mut h = x.clone();
    let mut i = 1usize;
    while fp_degree(&f).unwrap_or(0) >= 2 * i {
        h = fp_powmod_u64(&h, p, &f, p);
        let g = fp_gcd(&fp_sub(&h, &x, p), &f, p);
        if fp_degree(&g).unwrap_or(0) > 0 {
            result.push((g.clone(), i));
            f = fp_div_rem(&f, &g, p).0;
            h = fp_rem(&h, &f, p);
        }
        i += 1;
    }
    if fp_degree(&f).unwrap_or(0) > 0 {
        let d = fp_degree(&f).unwrap_or(0);
        result.push((f, d));
    }
    result
}

/// Equal-degree splitting (Cantor–Zassenhaus) of a monic `g` all of whose
/// irreducible factors have degree `d`.  Appends the factors to `out`.
fn fp_equal_degree(g: &FpPoly, d: usize, p: u64, rng: &mut XorShift, out: &mut Vec<FpPoly>) {
    // Explicit work stack instead of recursion.
    let mut stack: Vec<FpPoly> = vec![g.clone()];
    // (p^d − 1) / 2
    let exp = (BigUint::from(p).pow(d as u32) - BigUint::one()) / BigUint::from(2u32);
    let one: FpPoly = vec![1];

    while let Some(cur) = stack.pop() {
        let deg = fp_degree(&cur).unwrap_or(0);
        if deg == 0 {
            continue;
        }
        if deg == d {
            out.push(fp_monic(&cur, p));
            continue;
        }
        // Try random splitting polynomials until a proper divisor appears.
        let mut split: Option<FpPoly> = None;
        for _attempt in 0..256 {
            let mut a: FpPoly = (0..deg).map(|_| rng.next_u64() % p).collect();
            fp_normalize(&mut a);
            if fp_degree(&a).unwrap_or(0) == 0 {
                continue;
            }
            let g1 = fp_gcd(&a, &cur, p);
            let dg1 = fp_degree(&g1).unwrap_or(0);
            if dg1 > 0 && dg1 < deg {
                split = Some(g1);
                break;
            }
            let b = fp_powmod_big(&a, &exp, &cur, p);
            let h = fp_gcd(&fp_sub(&b, &one, p), &cur, p);
            let dh = fp_degree(&h).unwrap_or(0);
            if dh > 0 && dh < deg {
                split = Some(h);
                break;
            }
        }
        match split {
            Some(h) => {
                let q = fp_div_rem(&cur, &h, p).0;
                stack.push(h);
                stack.push(q);
            }
            None => {
                // Astronomically unlikely (each attempt succeeds with
                // probability ≥ 1/2); keep the block unsplit rather than loop.
                tracing::error!("factor_mod_p: equal-degree splitting failed to split a block");
                out.push(fp_monic(&cur, p));
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn zp(c: &[i64]) -> ZPoly {
        c.iter().map(|&x| BigInt::from(x)).collect()
    }

    fn qp(c: &[i64]) -> Poly {
        Poly::from_coeffs(
            c.iter()
                .map(|&x| Ratio::from_integer(BigInt::from(x)))
                .collect(),
        )
    }

    fn z_mul(a: &ZPoly, b: &ZPoly) -> ZPoly {
        if a.is_empty() || b.is_empty() {
            return vec![];
        }
        let mut out = vec![BigInt::zero(); a.len() + b.len() - 1];
        for (i, ai) in a.iter().enumerate() {
            for (j, bj) in b.iter().enumerate() {
                out[i + j] += ai * bj;
            }
        }
        z_normalize(&mut out);
        out
    }

    fn product(fs: &[ZPoly]) -> ZPoly {
        let mut acc = vec![BigInt::one()];
        for f in fs {
            acc = z_mul(&acc, f);
        }
        acc
    }

    /// `x^n − 1`
    fn xn_minus_1(n: usize) -> ZPoly {
        let mut v = vec![BigInt::zero(); n + 1];
        v[0] = BigInt::from(-1);
        v[n] = BigInt::one();
        v
    }

    // ── GF(p) primitives ────────────────────────────────────────────

    #[test]
    fn fp_div_rem_roundtrip() {
        let p = 7;
        let a = vec![3, 1, 4, 1, 5];
        let b = vec![2, 0, 1];
        let (q, r) = fp_div_rem(&a, &b, p);
        let back = fp_add(&fp_mul(&q, &b, p), &r, p);
        assert_eq!(back, a);
        assert!(fp_degree(&r).unwrap_or(0) < 2);
    }

    #[test]
    fn fp_extended_gcd_bezout() {
        let p = 13;
        let a = vec![1, 2, 3, 1];
        let b = vec![5, 1, 1];
        let (u, v, g) = fp_extended_gcd(&a, &b, p);
        let lhs = fp_add(&fp_mul(&u, &a, p), &fp_mul(&v, &b, p), p);
        assert_eq!(lhs, g);
    }

    #[test]
    fn fp_squarefree_detects_square() {
        let p = 5;
        // (x+1)^2 = x^2 + 2x + 1
        assert!(!fp_is_squarefree(&vec![1, 2, 1], p));
        // x^2 + 1 is square-free mod 5 (= (x+2)(x+3))
        assert!(fp_is_squarefree(&vec![1, 0, 1], p));
    }

    #[test]
    fn fp_factor_x4_minus_1_mod_5_splits_fully() {
        let p = 5;
        let f = vec![4, 0, 0, 0, 1]; // x^4 − 1 ≡ x^4 + 4
        let fs = fp_factor_squarefree_monic(&f, p);
        assert_eq!(fs.len(), 4);
        for g in &fs {
            assert_eq!(fp_degree(g), Some(1));
        }
    }

    #[test]
    fn fp_squarefree_factorization_with_pth_power() {
        let p = 3;
        // (x+1)^3 (x+2) mod 3: (x+1)^3 = x^3 + 1 (mod 3).  Times (x+2): x^4 + 2x^3 + x + 2
        let f = vec![2, 1, 0, 2, 1];
        let sqf = fp_squarefree_factorization(&f, p);
        // Reconstruct.
        let mut acc = vec![1u64];
        for (g, m) in &sqf {
            for _ in 0..*m {
                acc = fp_mul(&acc, g, p);
            }
        }
        assert_eq!(acc, f);
        assert!(sqf.iter().any(|(_, m)| *m == 3));
    }

    #[test]
    fn factor_mod_p_public_api() {
        let f = qp(&[1, 0, 1]);
        let (lc, fs) = factor_mod_p(&f, 5).unwrap();
        assert_eq!(lc, 1);
        assert_eq!(fs, vec![(vec![2, 1], 1), (vec![3, 1], 1)]);
        assert!(factor_mod_p(&f, 4).is_none());
        assert!(factor_mod_p(&f, 2).is_none());
        // Denominator divisible by p → None.
        let half = Poly::from_coeffs(vec![
            Ratio::new(BigInt::from(1), BigInt::from(5)),
            Ratio::one(),
        ]);
        assert!(factor_mod_p(&half, 5).is_none());
        assert!(factor_mod_p(&half, 7).is_some());
    }

    #[test]
    fn factor_mod_p_with_multiplicity() {
        // (x+1)^2 (x+2) = x^3 + 4x^2 + 5x + 2 mod 7
        let f = qp(&[2, 5, 4, 1]);
        let (_, fs) = factor_mod_p(&f, 7).unwrap();
        assert_eq!(fs, vec![(vec![1, 1], 2), (vec![2, 1], 1)]);
    }

    // ── ℤ[x] helpers ────────────────────────────────────────────────

    #[test]
    fn z_div_exact_works_and_rejects() {
        let f = z_mul(&zp(&[1, 1]), &zp(&[-2, 3]));
        assert_eq!(z_div_exact(&f, &zp(&[1, 1])), Some(zp(&[-2, 3])));
        assert_eq!(z_div_exact(&f, &zp(&[-2, 3])), Some(zp(&[1, 1])));
        assert_eq!(z_div_exact(&f, &zp(&[1, 2])), None);
        assert_eq!(z_div_exact(&f, &zp(&[5, 1])), None);
    }

    #[test]
    fn next_combination_enumerates_all() {
        let mut idx = vec![0, 1];
        let mut count = 1;
        while next_combination(&mut idx, 5) {
            count += 1;
        }
        assert_eq!(count, 10);
    }

    // ── Hensel lifting ──────────────────────────────────────────────

    #[test]
    fn hensel_lift_reconstructs_mod_pk() {
        // f = (x − 1)(x + 2)(2x + 3) = 2x^3 + 5x^2 − 4x − 6 ... compute
        let f = product(&[zp(&[-1, 1]), zp(&[2, 1]), zp(&[3, 2])]);
        let p = 7;
        let fp = fp_monic(&z_to_fp(&f, p), p);
        let factors = fp_factor_squarefree_monic(&fp, p);
        assert_eq!(factors.len(), 3);
        let k = 6;
        let lifted = hensel_lift(&f, p, &factors, k);
        let pk = BigInt::from(p).pow(k);
        let n = z_degree(&f).unwrap();
        let mut prod = vec![f[n].mod_floor(&pk)];
        for g in &lifted {
            prod = z_mul_mod(&prod, g, &pk);
        }
        let f_mod: ZPoly = f.iter().map(|c| c.mod_floor(&pk)).collect();
        assert_eq!(prod, f_mod);
    }

    // ── Full factorization ──────────────────────────────────────────

    fn check_factorization(f: &ZPoly, expected_degrees: &[usize]) -> Vec<ZPoly> {
        let fs = factor_squarefree_z(f);
        let mut degs: Vec<usize> = fs.iter().map(|g| z_degree(g).unwrap()).collect();
        degs.sort_unstable();
        let mut exp = expected_degrees.to_vec();
        exp.sort_unstable();
        assert_eq!(degs, exp, "degrees for {f:?}: got {fs:?}");
        let back = product(&fs);
        let mut f_pos = f.clone();
        if f_pos.last().unwrap().is_negative() {
            for c in &mut f_pos {
                *c = -std::mem::take(c);
            }
        }
        assert_eq!(back, f_pos, "product of factors must equal input");
        fs
    }

    #[test]
    fn factor_x2_minus_1() {
        check_factorization(&zp(&[-1, 0, 1]), &[1, 1]);
    }

    #[test]
    fn factor_x2_plus_1_irreducible() {
        check_factorization(&zp(&[1, 0, 1]), &[2]);
    }

    #[test]
    fn factor_x4_plus_1_irreducible() {
        // x^4 + 1 is reducible mod every prime but irreducible over ℤ.
        check_factorization(&zp(&[1, 0, 0, 0, 1]), &[4]);
    }

    #[test]
    fn factor_x4_plus_4_sophie_germain() {
        let fs = check_factorization(&zp(&[4, 0, 0, 0, 1]), &[2, 2]);
        assert!(fs.contains(&zp(&[2, -2, 1])));
        assert!(fs.contains(&zp(&[2, 2, 1])));
    }

    #[test]
    fn factor_6x4_minus_7x3_minus_8x2_plus_7x_plus_2() {
        // 6x^4 − 7x^3 − 8x^2 + 7x + 2 = (x − 1)(x + 1)(6x² − 7x − 2)
        let f = zp(&[2, 7, -8, -7, 6]);
        let fs = factor_squarefree_z(&f);
        assert_eq!(product(&fs), f);
        assert_eq!(fs, vec![zp(&[-1, 1]), zp(&[1, 1]), zp(&[-2, -7, 6])]);
    }

    #[test]
    fn factor_x8_plus_x4_plus_1() {
        // = (x^2+x+1)(x^2−x+1)(x^4−x^2+1)
        check_factorization(&zp(&[1, 0, 0, 0, 1, 0, 0, 0, 1]), &[2, 2, 4]);
    }

    #[test]
    fn factor_x12_minus_1() {
        // Φ1 Φ2 Φ3 Φ4 Φ6 Φ12 → degrees 1,1,2,2,2,4
        check_factorization(&xn_minus_1(12), &[1, 1, 2, 2, 2, 4]);
    }

    #[test]
    fn factor_x_n_minus_1_for_n_up_to_30() {
        for n in 2..=30usize {
            let fs = factor_squarefree_z(&xn_minus_1(n));
            assert_eq!(product(&fs), xn_minus_1(n), "x^{n} - 1");
            // Number of factors = number of divisors of n.
            let divisors = (1..=n).filter(|d| n % d == 0).count();
            assert_eq!(fs.len(), divisors, "x^{n} - 1 should have τ(n) factors");
        }
    }

    #[test]
    fn factor_x60_minus_1() {
        let n = 60;
        let fs = factor_squarefree_z(&xn_minus_1(n));
        assert_eq!(product(&fs), xn_minus_1(n));
        assert_eq!(fs.len(), 12); // τ(60) = 12
    }

    #[test]
    fn factor_x105_minus_1() {
        let n = 105;
        let start = std::time::Instant::now();
        let fs = factor_squarefree_z(&xn_minus_1(n));
        let elapsed = start.elapsed();
        assert_eq!(product(&fs), xn_minus_1(n));
        assert_eq!(fs.len(), 8); // τ(105) = 8
        // Φ_105 has degree 48 and is famous for having a coefficient −2.
        let phi105 = fs.iter().find(|g| z_degree(g) == Some(48)).unwrap();
        assert!(phi105.iter().any(|c| *c == BigInt::from(-2)));
        eprintln!("x^105 - 1 factored in {elapsed:?}");
    }

    #[test]
    fn factor_swinnerton_dyer_degree_8_irreducible() {
        // ∏ (x ± √2 ± √3 ± √5) = x^8 − 40x^6 + 352x^4 − 960x^2 + 576
        let f = zp(&[576, 0, -960, 0, 352, 0, -40, 0, 1]);
        check_factorization(&f, &[8]);
    }

    #[test]
    fn factor_swinnerton_dyer_degree_4_irreducible() {
        // ∏ (x ± √2 ± √3) = x^4 − 10x^2 + 1
        check_factorization(&zp(&[1, 0, -10, 0, 1]), &[4]);
    }

    #[test]
    fn factor_random_products_of_irreducibles() {
        // Products of 3–4 irreducibles of degree 2–5 with non-unit leading coefficients.
        let irreducibles: Vec<ZPoly> = vec![
            zp(&[1, 0, 1]),            // x^2 + 1
            zp(&[3, 1, 2]),            // 2x^2 + x + 3
            zp(&[-2, 0, 1]),           // x^2 − 2
            zp(&[1, 1, 0, 1]),         // x^3 + x + 1
            zp(&[-3, 0, 0, 2]),        // 2x^3 − 3
            zp(&[1, 0, 0, 0, 1]),      // x^4 + 1
            zp(&[2, -1, 0, 1, 3]),     // 3x^4 + x^3 − x + 2
            zp(&[-1, -1, 0, 0, 0, 1]), // x^5 − x − 1
            zp(&[7, 0, 0, 1, 0, 5]),   // 5x^5 + x^3 + 7
        ];
        let combos: [&[usize]; 5] = [
            &[0, 1, 3],
            &[2, 4, 5],
            &[1, 3, 6, 7],
            &[0, 2, 5, 8],
            &[3, 4, 6, 8],
        ];
        for combo in combos {
            let parts: Vec<ZPoly> = combo.iter().map(|&i| irreducibles[i].clone()).collect();
            let f = product(&parts);
            let fs = factor_squarefree_z(&f);
            assert_eq!(fs.len(), combo.len(), "combo {combo:?}: got {fs:?}");
            assert_eq!(product(&fs), f);
            for part in &parts {
                assert!(fs.contains(part), "missing factor {part:?} in {fs:?}");
            }
        }
    }

    #[test]
    fn factor_with_x_factor_and_content_via_poly_api() {
        // 4x^3 − 4x = 4·x·(x−1)(x+1)
        let f = qp(&[0, -4, 0, 4]);
        let (content, fs) = factor_zassenhaus_with_content(&f);
        assert_eq!(content, Ratio::from_integer(BigInt::from(4)));
        assert_eq!(fs.len(), 3);
        assert!(fs.iter().all(|(_, m)| *m == 1));
    }

    #[test]
    fn factor_repeated_factors_multiplicities() {
        // (x+1)^3 (x^2+1)^2
        let f =
            &(&(&qp(&[1, 1]) * &qp(&[1, 1])) * &qp(&[1, 1])) * &(&qp(&[1, 0, 1]) * &qp(&[1, 0, 1]));
        let fs = factor_zassenhaus(&f);
        assert_eq!(fs, vec![(qp(&[1, 1]), 3), (qp(&[1, 0, 1]), 2)]);
    }

    #[test]
    fn factor_rational_coefficients() {
        // (1/2)x^2 − 1/2 = (1/2)(x−1)(x+1)
        let half = Ratio::new(BigInt::from(1), BigInt::from(2));
        let f = Poly::from_coeffs(vec![-half.clone(), Ratio::zero(), half.clone()]);
        let (content, fs) = factor_zassenhaus_with_content(&f);
        assert_eq!(content, half);
        assert_eq!(fs.len(), 2);
    }

    #[test]
    fn irreducibility_predicate() {
        assert_eq!(is_irreducible_z(&qp(&[1, 0, 1])), Some(true));
        assert_eq!(is_irreducible_z(&qp(&[-1, 0, 1])), Some(false));
        assert_eq!(is_irreducible_z(&qp(&[1, 2, 1])), Some(false));
        assert_eq!(is_irreducible_z(&qp(&[2, 2])), Some(true));
        assert_eq!(is_irreducible_z(&qp(&[5])), None);
        assert_eq!(is_irreducible_z(&Poly::zero()), None);
    }

    #[test]
    fn factor_large_coefficients() {
        // (10^12 x + 1)(x − 10^12)
        let big = BigInt::from(1_000_000_000_000i64);
        let a = vec![BigInt::one(), big.clone()];
        let b = vec![-big.clone(), BigInt::one()];
        let f = z_mul(&a, &b);
        let fs = factor_squarefree_z(&f);
        assert_eq!(fs.len(), 2);
        assert_eq!(product(&fs), f);
    }

    #[test]
    fn factor_squarefree_z_handles_repeated_factors() {
        // (x + 1)^2 (x^2 + 1) passed directly to the "square-free" entry point.
        let f = product(&[zp(&[1, 1]), zp(&[1, 1]), zp(&[1, 0, 1])]);
        let fs = factor_squarefree_z(&f);
        assert_eq!(fs, vec![zp(&[1, 1]), zp(&[1, 1]), zp(&[1, 0, 1])]);
        assert_eq!(product(&fs), f);
    }

    #[test]
    fn factor_negative_leading_coefficient() {
        // −(x^2 − 1) → factors have positive lc
        let fs = factor_squarefree_z(&zp(&[1, 0, -1]));
        assert_eq!(fs, vec![zp(&[-1, 1]), zp(&[1, 1])]);
    }

    #[test]
    fn factor_constant_and_zero() {
        assert!(factor_squarefree_z(&zp(&[5])).is_empty());
        assert!(factor_squarefree_z(&[]).is_empty());
        assert_eq!(
            factor_zassenhaus_with_content(&Poly::zero()).0,
            Ratio::zero()
        );
        assert!(factor_zassenhaus(&qp(&[3])).is_empty());
    }

    // ── Multivariate ──────────────────────────────────────────────────────

    type MP = MultiPoly<super::super::multipoly::GrevLex>;

    fn mv_vars(n: usize) -> Vec<MP> {
        (0..n).map(|i| MultiPoly::var(n, i)).collect()
    }

    fn mv_check(f: &MP, expected_count: usize) -> Vec<(MP, u32)> {
        let (content, fs) = factor_multivariate(f).expect("factorable");
        let mut back = MultiPoly::constant(f.num_vars(), content);
        for (g, m) in &fs {
            for _ in 0..*m {
                back = back.mul(g);
            }
        }
        assert_eq!(&back, f, "product must reconstruct input");
        assert_eq!(fs.len(), expected_count, "factors: {fs:?}");
        fs
    }

    #[test]
    fn mv_x2_minus_y2() {
        let v = mv_vars(2);
        let f = v[0].mul(&v[0]).sub(&v[1].mul(&v[1]));
        let fs = mv_check(&f, 2);
        assert!(fs.contains(&(v[0].sub(&v[1]), 1)));
        assert!(fs.contains(&(v[0].add(&v[1]), 1)));
    }

    #[test]
    fn mv_x3_minus_y3() {
        let v = mv_vars(2);
        let x3 = v[0].mul(&v[0]).mul(&v[0]);
        let y3 = v[1].mul(&v[1]).mul(&v[1]);
        let fs = mv_check(&x3.sub(&y3), 2);
        assert!(fs.iter().any(|(g, _)| g.total_degree() == Some(1)));
        assert!(fs.iter().any(|(g, _)| g.total_degree() == Some(2)));
    }

    #[test]
    fn mv_perfect_square() {
        let v = mv_vars(2);
        // x² + 2xy + y² = (x + y)²
        let s = v[0].add(&v[1]);
        let f = s.mul(&s);
        let fs = mv_check(&f, 1);
        assert_eq!(fs[0], (s, 2));
    }

    #[test]
    fn mv_monomial_content() {
        let v = mv_vars(2);
        // x²y − y = y (x − 1)(x + 1)
        let f = v[0].mul(&v[0]).mul(&v[1]).sub(&v[1]);
        let fs = mv_check(&f, 3);
        assert!(fs.contains(&(v[1].clone(), 1)));
    }

    #[test]
    fn mv_polynomial_content_in_other_variable() {
        let v = mv_vars(2);
        // (y + 1) x² − (y + 1) = (y + 1)(x − 1)(x + 1)
        let one = MP::from_int(2, 1);
        let yp1 = v[1].add(&one);
        let f = yp1.mul(&v[0]).mul(&v[0]).sub(&yp1);
        let fs = mv_check(&f, 3);
        assert!(fs.contains(&(yp1, 1)));
        assert!(fs.contains(&(v[0].sub(&one), 1)));
        assert!(fs.contains(&(v[0].add(&one), 1)));
    }

    #[test]
    fn mv_irreducible_stays_whole() {
        let v = mv_vars(2);
        // x² + y² is irreducible over ℚ.
        let f = v[0].mul(&v[0]).add(&v[1].mul(&v[1]));
        let fs = mv_check(&f, 1);
        assert_eq!(fs[0], (f, 1));
    }

    #[test]
    fn mv_three_variables() {
        let v = mv_vars(3);
        // (x + y + z)(x − y)(y + z)
        let a = v[0].add(&v[1]).add(&v[2]);
        let b = v[0].sub(&v[1]);
        let c = v[1].add(&v[2]);
        let f = a.mul(&b).mul(&c);
        let fs = mv_check(&f, 3);
        assert!(fs.contains(&(a, 1)));
        assert!(fs.contains(&(b, 1)));
        assert!(fs.contains(&(c, 1)));
    }

    #[test]
    fn mv_rational_content_and_sign() {
        let v = mv_vars(2);
        // -(3/2) (x - y)(x + y)
        let f = v[0]
            .mul(&v[0])
            .sub(&v[1].mul(&v[1]))
            .scale(&Ratio::new(BigInt::from(-3), BigInt::from(2)));
        let (content, fs) = factor_multivariate(&f).unwrap();
        assert_eq!(content, Ratio::new(BigInt::from(-3), BigInt::from(2)));
        assert_eq!(fs.len(), 2);
    }

    #[test]
    fn mv_univariate_input_delegates() {
        let v = mv_vars(2);
        // y² − 1 as a 2-variable polynomial.
        let f = v[1].mul(&v[1]).sub(&MP::from_int(2, 1));
        let fs = mv_check(&f, 2);
        assert!(fs.iter().all(|(g, _)| g.degree_in(0) == 0));
    }
}
