//! Dense univariate polynomials over ℚ — specialization of [`GenPoly`].
//!
//! This module defines [`Poly`] as a type alias for
//! `GenPoly<Ratio<BigInt>>`, inheriting all generic polynomial arithmetic
//! from [`GenPoly`].  It adds ℚ-specific
//! operations that depend on integer structure:
//!
//! - [`has_integer_coeffs`](GenPoly::has_integer_coeffs) — check all denominators are 1
//! - [`content`](GenPoly::content) / [`primitive_part`](GenPoly::primitive_part) — integer GCD of coefficients
//! - [`factor_over_z`](GenPoly::factor_over_z) — full factorization over ℤ
//!   (Berlekamp–Zassenhaus, see [`super::factor_zassenhaus`])
//! - [`derivative`](GenPoly::derivative) — specialized O(1)-per-coefficient version
//!
//! The generic operations (add, mul, div_rem, gcd, extended_gcd,
//! squarefree_factors, resultant, Display, etc.) are all provided by
//! `GenPoly<C>` in [`super::generic`].
//!
//! # Representation
//!
//! Coefficients are stored as `Vec<Ratio<BigInt>>` in ascending degree
//! order: `coeffs[i]` is the coefficient of `x^i`.  The zero polynomial
//! has an empty coefficient vector.  Non-zero polynomials are normalized
//! (leading coefficient is nonzero).

use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::Ratio;
use num_traits::{One, Signed, Zero};

use super::generic::GenPoly;

/// A dense univariate polynomial over ℚ.
///
/// This is a type alias for `GenPoly<Ratio<BigInt>>`.  All generic
/// polynomial operations (arithmetic, GCD, factorization, Display)
/// are inherited.  ℚ-specific methods like [`content`](GenPoly::content),
/// [`primitive_part`](GenPoly::primitive_part), and
/// [`factor_over_z`](GenPoly::factor_over_z) are added via a specialized
/// `impl` block in this module.
pub type Poly = GenPoly<Ratio<BigInt>>;

// ═══════════════════════════════════════════════════════════════════════════
// ℚ-specific methods (specialized impl on GenPoly<Ratio<BigInt>>)
// ═══════════════════════════════════════════════════════════════════════════

impl GenPoly<Ratio<BigInt>> {
    /// Returns `true` if every coefficient is an integer (denominator 1).
    pub fn has_integer_coeffs(&self) -> bool {
        self.coeffs.iter().all(|c| c.denom().is_one())
    }

    /// Compute the content (GCD of all coefficients) of the polynomial.
    ///
    /// Returns 0 for the zero polynomial.
    pub fn content(&self) -> Ratio<BigInt> {
        if self.is_zero() {
            return Ratio::zero();
        }
        let nonzero: Vec<_> = self.coeffs().iter().filter(|c| !c.is_zero()).collect();
        if nonzero.is_empty() {
            return Ratio::one();
        }
        let mut result = (*nonzero[0]).clone();
        for c in &nonzero[1..] {
            result = rational_gcd(&result, c);
        }
        if result.is_negative() {
            -result
        } else {
            result
        }
    }

    /// Compute the primitive part: `self / content`.
    #[must_use]
    pub fn primitive_part(&self) -> Self {
        let c = self.content();
        if c.is_one() || c.is_zero() {
            return self.clone();
        }
        self.scale(&(Ratio::one() / c))
    }

    /// Is this polynomial square-free over ℚ (no repeated roots)?
    ///
    /// Constants and the zero polynomial return `None`.
    pub fn is_squarefree(&self) -> Option<bool> {
        let d = self.degree()?;
        if d == 0 {
            return None;
        }
        let g = Self::gcd(self, &self.derivative());
        Some(g.degree() == Some(0))
    }

    /// Square-free decomposition over ℤ: `(content, [(a₁, 1), (a₂, 2), …])`
    /// with `self = content · ∏ aᵢ^i`, each `aᵢ` primitive, square-free,
    /// pairwise coprime, with positive leading coefficient.
    ///
    /// Constants return `(c, [])`; zero returns `(0, [])`.
    pub fn sqf_list(&self) -> (Ratio<BigInt>, Vec<(Poly, u32)>) {
        if self.is_zero() {
            return (Ratio::zero(), vec![]);
        }
        if self.is_constant() {
            return (self.coeff(0), vec![]);
        }
        let mut content = self.content();
        let mut prim = self.primitive_part();
        if prim.leading_coeff().is_some_and(|lc| lc.is_negative()) {
            content = -content;
            prim = -&prim;
        }
        (content, square_free_decomposition(&prim))
    }

    /// Factor this polynomial over ℤ.
    ///
    /// Returns `(content, factors)` where:
    /// - `content` is the rational GCD of all coefficients, with sign chosen
    ///   so that each factor has a positive leading coefficient.
    /// - `factors` is a list of `(irreducible_factor, multiplicity)` pairs,
    ///   sorted by degree then coefficients.
    ///
    /// The original polynomial equals `content * ∏ factor^multiplicity`.
    ///
    /// This is a thin wrapper around
    /// [`factor_zassenhaus_with_content`](super::factor_zassenhaus::factor_zassenhaus_with_content):
    /// content extraction, Yun's square-free decomposition, then
    /// Berlekamp–Zassenhaus on each square-free part.
    pub fn factor_over_z(&self) -> (Ratio<BigInt>, Vec<(Poly, u32)>) {
        super::factor_zassenhaus::factor_zassenhaus_with_content(self)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Interpolation adapters (ℚ-specific point shapes)
// ═══════════════════════════════════════════════════════════════════════════

/// Interpolation through rational-valued points at integer abscissae.
///
/// Given `(x₀, y₀), …, (xₙ, yₙ)` with integer `xᵢ` and rational `yᵢ`,
/// returns the unique polynomial of degree ≤ n passing through all points
/// (see [`interp::interpolate`](super::interp::interpolate)).
pub(crate) fn lagrange_interpolate_rational(points: &[(i64, Ratio<BigInt>)]) -> Poly {
    super::interp::interpolate_at_integers(points)
}

/// Interpolation through points with rational abscissae.
///
/// Returns the unique polynomial of degree `< points.len()` passing through
/// every `(xᵢ, yᵢ)`, or `None` if two abscissae coincide.  An empty input
/// yields the zero polynomial.
pub(crate) fn lagrange_interpolate_points(
    points: &[(Ratio<BigInt>, Ratio<BigInt>)],
) -> Option<Poly> {
    let xs: Vec<Ratio<BigInt>> = points.iter().map(|(x, _)| x.clone()).collect();
    let ys: Vec<Ratio<BigInt>> = points.iter().map(|(_, y)| y.clone()).collect();
    super::interp::interpolate(&xs, &ys)
}

// ═══════════════════════════════════════════════════════════════════════════
// Private helpers: rational GCD
// ═══════════════════════════════════════════════════════════════════════════

/// GCD of two positive rationals: gcd(a/b, c/d) = gcd(a,c) / lcm(b,d).
fn rational_gcd(a: &Ratio<BigInt>, b: &Ratio<BigInt>) -> Ratio<BigInt> {
    let numer_gcd = a.numer().gcd(b.numer());
    let denom_lcm = a.denom().lcm(b.denom());
    Ratio::new(numer_gcd, denom_lcm)
}

// ═══════════════════════════════════════════════════════════════════════════
// Private helpers: square-free decomposition (Yun's algorithm)
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the square-free decomposition of a primitive polynomial with
/// positive leading coefficient.
///
/// Returns `[(a₁, 1), (a₂, 2), …]` where `f = ∏ aᵢ^i` (up to a unit)
/// and each `aᵢ` is square-free, primitive with positive leading
/// coefficient, and pairwise coprime.
pub(crate) fn square_free_decomposition(f: &Poly) -> Vec<(Poly, u32)> {
    let deg = match f.degree() {
        Some(d) if d >= 1 => d,
        _ => return vec![],
    };

    let df = f.derivative();
    if df.is_zero() {
        // Shouldn't happen for degree ≥ 1 in characteristic 0.
        return vec![(ensure_positive_lc(f), 1)];
    }

    let g = Poly::gcd(f, &df);

    // If gcd is trivial (constant), f is already square-free.
    if g.degree().unwrap_or(0) == 0 {
        return vec![(ensure_positive_lc(f), 1)];
    }

    // Yun's iterative decomposition.
    let mut w = f.div(&g); // product of all distinct irreducible factors
    let mut c = g; //          ∏ pᵢ^(eᵢ−1)
    let mut result: Vec<(Poly, u32)> = Vec::new();
    let mut i = 1u32;

    loop {
        if w.degree().unwrap_or(0) == 0 {
            break;
        }

        let y = Poly::gcd(&w, &c); // factors with multiplicity > i
        let z = w.div(&y); //         factors with multiplicity exactly i

        if z.degree().unwrap_or(0) > 0 {
            // Normalize to primitive with positive leading coefficient.
            let z_norm = ensure_positive_lc(&z.primitive_part());
            result.push((z_norm, i));
        }

        w = y;
        if c.degree().unwrap_or(0) > 0 && w.degree().unwrap_or(0) > 0 {
            c = c.div(&w);
        } else {
            c = Poly::from_int(1);
        }
        i += 1;

        // Safety bound.
        if i > deg as u32 + 1 {
            break;
        }
    }

    if result.is_empty() {
        result.push((ensure_positive_lc(f), 1));
    }

    result
}

/// Return a copy of `p` with positive leading coefficient.
pub(crate) fn ensure_positive_lc(p: &Poly) -> Poly {
    match p.leading_coeff() {
        Some(lc) if lc.is_negative() => -p,
        _ => p.clone(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Crate-private helpers: Rational Root Theorem
// ═══════════════════════════════════════════════════════════════════════════

/// Extract all linear factors using the Rational Root Theorem.
///
/// Returns `(remaining, linear_factors)`.  Used by
/// [`factor_squarefree_z`](super::factor_zassenhaus::factor_squarefree_z)
/// as a cheap pre-pass before Berlekamp–Zassenhaus.  The search is capped
/// (see [`MAX_DIVISOR_COMBINATIONS`](super::MAX_DIVISOR_COMBINATIONS)), so
/// `remaining` may still have rational roots when the caps are hit.
pub(crate) fn extract_rational_roots(f: &Poly) -> (Poly, Vec<Poly>) {
    let mut remaining = f.clone();
    let mut factors: Vec<Poly> = Vec::new();

    loop {
        let deg = match remaining.degree() {
            Some(d) if d >= 1 => d,
            _ => break,
        };

        let a0 = remaining.coeff(0);

        // Handle root at x = 0.
        if a0.is_zero() {
            remaining = remaining.div(&Poly::x());
            factors.push(Poly::x());
            continue;
        }

        // Need integer coefficients.
        if !remaining.has_integer_coeffs() {
            break;
        }

        let an = remaining.coeff(deg);
        let a0_abs = a0.numer().abs();
        let an_abs = an.numer().abs();

        let p_divs = positive_divisors(&a0_abs);
        let q_divs = positive_divisors(&an_abs);

        // Safety cap.
        if p_divs.is_empty() || q_divs.is_empty() {
            break;
        }
        if p_divs.len() * q_divs.len() > super::MAX_DIVISOR_COMBINATIONS {
            tracing::warn!(
                "factoring: rational root search truncated — {} candidate combinations exceeds cap {}",
                p_divs.len() * q_divs.len(),
                super::MAX_DIVISOR_COMBINATIONS
            );
            break;
        }

        let mut found = false;

        'search: for p in &p_divs {
            for q in &q_divs {
                // Only consider coprime (p, q) to avoid redundant/non-primitive factors.
                if p.gcd(q) != BigInt::one() {
                    continue;
                }
                for &sign in &[1i64, -1i64] {
                    let candidate = Ratio::new(p * BigInt::from(sign), q.clone());
                    if remaining.eval(&candidate).is_zero() {
                        // Build integer linear factor (q·x − sign·p).
                        let int_factor = Poly::from_coeffs(vec![
                            Ratio::from_integer(-(p * BigInt::from(sign))),
                            Ratio::from_integer(q.clone()),
                        ]);
                        let prim_factor = int_factor.primitive_part();
                        let prim_factor = ensure_positive_lc(&prim_factor);
                        let (quot, rem) = remaining.div_rem(&prim_factor);
                        if rem.is_zero() {
                            remaining = if quot.has_integer_coeffs() {
                                quot
                            } else {
                                // Normalize to integer coefficients.
                                ensure_positive_lc(&quot.primitive_part())
                            };
                            factors.push(prim_factor);
                            found = true;
                            break 'search;
                        }
                    }
                }
            }
        }

        if !found {
            break;
        }
    }

    (remaining, factors)
}

// ═══════════════════════════════════════════════════════════════════════════
// Crate-private helpers: Kronecker's method
// ═══════════════════════════════════════════════════════════════════════════

/// Try to find a non-trivial factor of `f` with degree exactly
/// `trial_deg` using Kronecker's method.
///
/// Returns `Some((factor, quotient))` on success.  Kept as a fallback for
/// the (rare) case where Berlekamp–Zassenhaus exhausts its recombination
/// budget; see [`super::factor_zassenhaus`].
pub(crate) fn kronecker_find_factor(f: &Poly, trial_deg: usize) -> Option<(Poly, Poly)> {
    let f_deg = f.degree()?;
    if trial_deg == 0 || trial_deg * 2 > f_deg {
        return None;
    }

    let num_points = trial_deg + 1;

    // Choose small integer evaluation points: 0, 1, −1, 2, −2, …
    let eval_pts = small_integer_points(num_points);

    // Evaluate f at each point (all results are integers).
    let vals: Vec<BigInt> = eval_pts
        .iter()
        .map(|&x| {
            let v = f.eval(&Ratio::from_integer(BigInt::from(x)));
            // Integer-coeff poly at integer arg ⇒ integer.
            v.numer().clone()
        })
        .collect();

    // If any evaluation is zero, rational root extraction should have
    // caught it already.  Skip to avoid infinite divisor enumeration.
    if vals.iter().any(|v| v.is_zero()) {
        return None;
    }

    // Signed divisors for each evaluation value.
    let div_lists: Vec<Vec<BigInt>> = vals.iter().map(signed_divisors).collect();

    // Bail out if any divisor list is empty (value too large) or
    // total combinations exceed a practical limit.
    if div_lists.iter().any(Vec::is_empty) {
        return None;
    }
    let total: usize = div_lists
        .iter()
        .map(|d| d.len())
        .try_fold(1usize, |acc, n| acc.checked_mul(n))
        .unwrap_or(usize::MAX);
    if total > super::MAX_KRONECKER_COMBINATIONS {
        tracing::debug!(
            "factoring: Kronecker search abandoned — {} evaluation combinations exceeds cap {}",
            total,
            super::MAX_KRONECKER_COMBINATIONS
        );
        return None;
    }

    let mut indices = vec![0usize; num_points];
    let mut count = 0usize;

    loop {
        // Build the current divisor combination.
        let points: Vec<(i64, BigInt)> = (0..num_points)
            .map(|i| (eval_pts[i], div_lists[i][indices[i]].clone()))
            .collect();

        if let Some(candidate) = super::interp::interpolate_integer_points(&points)
            && candidate.degree() == Some(trial_deg)
            && candidate.has_integer_coeffs()
        {
            let prim = ensure_positive_lc(&candidate.primitive_part());
            if prim.degree() == Some(trial_deg) {
                let (quot, rem) = f.div_rem(&prim);
                if rem.is_zero() && quot.has_integer_coeffs() {
                    return Some((prim, quot));
                }
            }
        }

        // Advance odometer.
        count += 1;
        if count >= total {
            break;
        }
        let mut carry = true;
        for k in (0..num_points).rev() {
            if carry {
                indices[k] += 1;
                if indices[k] >= div_lists[k].len() {
                    indices[k] = 0;
                } else {
                    carry = false;
                    break;
                }
            }
        }
        if carry {
            break;
        }
    }

    None
}

// ═══════════════════════════════════════════════════════════════════════════
// Private helpers: integer point generation and divisors
// ═══════════════════════════════════════════════════════════════════════════

/// Generate `n` distinct small integer points: 0, 1, −1, 2, −2, …
fn small_integer_points(n: usize) -> Vec<i64> {
    let mut pts = Vec::with_capacity(n);
    pts.push(0);
    let mut k = 1i64;
    while pts.len() < n {
        pts.push(k);
        if pts.len() < n {
            pts.push(-k);
        }
        k += 1;
    }
    pts
}

/// All positive divisors of `|n|` in ascending order.
///
/// Returns an empty list if `n` is zero or `|n|` exceeds an internal
/// threshold (to keep Kronecker practical).
fn positive_divisors(n: &BigInt) -> Vec<BigInt> {
    if n.is_zero() {
        return vec![];
    }
    let n_abs = n.abs();
    // Bail out for very large values.
    if n_abs > BigInt::from(super::MAX_DIVISOR_COEFFICIENT) {
        tracing::debug!(
            "factoring: coefficient {} exceeds divisor magnitude cap {}; skipping divisor enumeration",
            n_abs,
            super::MAX_DIVISOR_COEFFICIENT
        );
        return vec![];
    }

    let mut small = Vec::new();
    let mut large = Vec::new();
    let mut i = BigInt::one();
    while &i * &i <= n_abs {
        if (&n_abs % &i).is_zero() {
            let complement = &n_abs / &i;
            if complement != i {
                large.push(complement);
            }
            small.push(i.clone());
        }
        i += 1;
    }

    large.reverse();
    small.extend(large);
    small
}

/// All signed (positive and negative) divisors of `|n|`.
fn signed_divisors(n: &BigInt) -> Vec<BigInt> {
    let pos = positive_divisors(n);
    let mut result = Vec::with_capacity(pos.len() * 2);
    for d in pos {
        result.push(d.clone());
        result.push(-d);
    }
    result
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn r(n: i64, d: i64) -> Ratio<BigInt> {
        Ratio::new(BigInt::from(n), BigInt::from(d))
    }

    fn ri(n: i64) -> Ratio<BigInt> {
        Ratio::from_integer(BigInt::from(n))
    }

    // ── gcd through ℤ[x] ───────────────────────────────────────────────────────────

    /// Small pseudo-random polynomial (xorshift), degree `deg`, with
    /// integer coefficients in `-9..=9` or, when `frac`, denominators too.
    fn pseudo_random(seed: u64, deg: usize, frac: bool) -> Poly {
        let mut s = seed;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let mut coeffs: Vec<Ratio<BigInt>> = (0..=deg)
            .map(|_| {
                let n = (next() % 19) as i64 - 9;
                let d = if frac { (next() % 5) as i64 + 1 } else { 1 };
                r(n, d)
            })
            .collect();
        if coeffs[deg].is_zero() {
            coeffs[deg] = ri(1);
        }
        Poly::from_coeffs(coeffs)
    }

    /// `Poly::gcd` (the ℤ[x] primitive-PRS fast path) returns exactly what
    /// Euclid over ℚ returns, including for common factors, coprime
    /// inputs, rational coefficients and the zero conventions.
    #[test]
    fn gcd_via_z_matches_generic_gcd() {
        for seed in 1..=15u64 {
            let a = pseudo_random(seed, 3 + seed as usize % 4, seed % 2 == 0);
            let b = pseudo_random(seed + 50, 2 + seed as usize % 3, seed % 3 == 0);
            let g = pseudo_random(seed + 100, 1 + seed as usize % 3, false);
            let ag = &a * &g;
            let bg = &b * &g;
            assert_eq!(
                Poly::gcd(&ag, &bg),
                Poly::gcd_euclid(&ag, &bg),
                "seed {seed}"
            );
            assert_eq!(
                Poly::gcd(&a, &b),
                Poly::gcd_euclid(&a, &b),
                "seed {seed} coprime-ish"
            );
            assert_eq!(
                Poly::gcd(&ag, &ag.derivative()),
                Poly::gcd_euclid(&ag, &ag.derivative())
            );
        }
        let a = pseudo_random(7, 5, true);
        assert_eq!(
            Poly::gcd(&a, &Poly::zero()),
            Poly::gcd_euclid(&a, &Poly::zero())
        );
        assert_eq!(
            Poly::gcd(&Poly::zero(), &a),
            Poly::gcd_euclid(&Poly::zero(), &a)
        );
        assert_eq!(Poly::gcd(&Poly::zero(), &Poly::zero()), Poly::zero());
        assert_eq!(Poly::gcd(&Poly::from_int(6), &a), Poly::from_int(1));
        assert_eq!(
            Poly::gcd(&Poly::from_int(6), &Poly::from_int(-4)),
            Poly::from_int(1)
        );
    }

    /// The positive pseudo-remainder is a positive multiple of the
    /// rational remainder.
    #[test]
    fn pseudo_rem_pos_is_positive_multiple_of_rem() {
        use crate::poly::zpoly::{integer_scaled, pseudo_rem_pos};
        for seed in 1..=10u64 {
            let a = pseudo_random(seed, 7, false);
            let b = pseudo_random(seed + 7, 3, false);
            let rem = a.rem(&b);
            let prem = pseudo_rem_pos(&integer_scaled(a.coeffs()), &integer_scaled(b.coeffs()));
            let prem = Poly::from_coeffs(prem.into_iter().map(Ratio::from_integer).collect());
            if rem.is_zero() {
                assert!(prem.is_zero());
                continue;
            }
            let scale = prem.leading_coeff().cloned().unwrap_or_default()
                / rem.leading_coeff().cloned().unwrap_or_default();
            assert!(scale.is_positive(), "seed {seed}: scale {scale}");
            assert_eq!(rem.scale(&scale), prem, "seed {seed}");
        }
    }

    // ── Resultant ────────────────────────────────────────────────────────────────

    #[test]
    fn resultant_coprime() {
        // res(x+1, x+2) = lc(x+1)^1 · (x+2)(root of x+1)
        //               = 1 · ((-1)+2) = 1
        let a = Poly::from_coeffs(vec![ri(1), ri(1)]); // x + 1
        let b = Poly::from_coeffs(vec![ri(2), ri(1)]); // x + 2
        let res = Poly::resultant(&a, &b);
        assert_eq!(res, ri(1));
    }

    #[test]
    fn resultant_common_root() {
        // x^2-1 and x-1 share root x=1 → resultant = 0
        let a = Poly::from_coeffs(vec![ri(-1), ri(0), ri(1)]); // x^2 - 1
        let b = Poly::from_coeffs(vec![ri(-1), ri(1)]); // x - 1
        assert_eq!(Poly::resultant(&a, &b), ri(0));
    }

    #[test]
    fn resultant_x2_plus_1_x_minus_1() {
        // res(x^2+1, x-1) = (x^2+1) at x=1 = 2
        let a = Poly::from_coeffs(vec![ri(1), ri(0), ri(1)]); // x^2 + 1
        let b = Poly::from_coeffs(vec![ri(-1), ri(1)]); // x - 1
        let res = Poly::resultant(&a, &b);
        assert_eq!(res, ri(2));
    }

    #[test]
    fn resultant_two_quadratics() {
        // res(x^2+1, x^2-1): roots of x^2-1 are ±1.
        // res = (1+1)((-1)^2+1) = 2*2 = 4  (lc(a)^deg(b) * ∏ a(root_of_b))
        let a = Poly::from_coeffs(vec![ri(1), ri(0), ri(1)]); // x^2+1
        let b = Poly::from_coeffs(vec![ri(-1), ri(0), ri(1)]); // x^2-1
        assert_eq!(Poly::resultant(&a, &b), ri(4));
    }

    #[test]
    fn resultant_with_zero() {
        let a = Poly::from_coeffs(vec![ri(1), ri(1)]); // x+1
        assert_eq!(Poly::resultant(&a, &Poly::zero()), ri(0));
        assert_eq!(Poly::resultant(&Poly::zero(), &a), ri(0));
    }

    #[test]
    fn resultant_2x_plus_3_x_plus_1() {
        // Sylvester: det [[2,3],[1,1]] = 2-3 = -1
        let a = Poly::from_coeffs(vec![ri(3), ri(2)]); // 2x+3
        let b = Poly::from_coeffs(vec![ri(1), ri(1)]); // x+1
        assert_eq!(Poly::resultant(&a, &b), ri(-1));
    }

    #[test]
    fn resultant_constant() {
        let a = Poly::from_int(5);
        let b = Poly::from_coeffs(vec![ri(1), ri(0), ri(1)]); // x^2+1
        // res(5, x^2+1) = 5^2 = 25
        assert_eq!(Poly::resultant(&a, &b), ri(25));
    }

    // ── Resultant polynomial (eval-interpolation) ───────────────────

    #[test]
    fn resultant_poly_linear_param() {
        // f = x^2 + 1, g = x, h = 1  →  res_x(x^2+1, x - t)
        // = (x^2+1) evaluated at x=t = t^2 + 1
        let f = Poly::from_coeffs(vec![ri(1), ri(0), ri(1)]); // x^2+1
        let g = Poly::x(); // x
        let h = Poly::from_int(1); // 1
        let r = Poly::resultant_poly(&f, &g, &h);
        let expected = Poly::from_coeffs(vec![ri(1), ri(0), ri(1)]); // t^2+1
        assert_eq!(r, expected, "res_x(x^2+1, x-t) should be t^2+1, got {r}");
    }

    #[test]
    fn resultant_poly_x5_plus_1() {
        // f = x^5+1, g = 1, h = 5x^4  →  R(t) = res_x(x^5+1, 1-5t·x^4)
        // By calculation: R(t) = 1 - 3125·t^5
        let mut f_coeffs = vec![ri(0); 6];
        f_coeffs[0] = ri(1);
        f_coeffs[5] = ri(1);
        let f = Poly::from_coeffs(f_coeffs); // x^5+1

        let g = Poly::from_int(1);

        let mut h_coeffs = vec![ri(0); 5];
        h_coeffs[4] = ri(5);
        let h = Poly::from_coeffs(h_coeffs); // 5x^4

        let r = Poly::resultant_poly(&f, &g, &h);
        // R(t) should be 1 - 3125 t^5
        assert_eq!(
            r.degree(),
            Some(5),
            "R(t) should have degree 5, got {:?}",
            r.degree()
        );
        assert_eq!(r.coeff(0), ri(1), "constant term should be 1");
        assert_eq!(
            r.coeff(5),
            ri(-3125),
            "t^5 coeff should be -3125, got {}",
            r.coeff(5)
        );
        // Middle terms should be 0
        for k in 1..5 {
            assert_eq!(r.coeff(k), ri(0), "coeff of t^{k} should be 0");
        }
    }

    // ── Squarefree factors ──────────────────────────────────────────

    #[test]
    fn squarefree_factors_square() {
        // (x+1)^2 = x^2+2x+1
        let p = Poly::from_coeffs(vec![ri(1), ri(2), ri(1)]);
        let factors = p.squarefree_factors();
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0].1, 2);
        assert_eq!(factors[0].0.degree(), Some(1));
    }

    #[test]
    fn squarefree_factors_distinct() {
        // (x-1)(x-2) = x^2-3x+2 — already squarefree
        let p = Poly::from_coeffs(vec![ri(2), ri(-3), ri(1)]);
        let factors = p.squarefree_factors();
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0].1, 1);
    }

    #[test]
    fn squarefree_factors_mixed() {
        // (x-1)^2*(x-2) = x^3-4x^2+5x-2
        let x_minus = |n: i64| Poly::from_coeffs(vec![ri(-n), ri(1)]);
        let p = &(&x_minus(1) * &x_minus(1)) * &x_minus(2);
        let factors = p.squarefree_factors();
        // Should have two factors: (x-1) with mult 2, (x-2) with mult 1
        // or equivalently, something that reconstructs correctly
        let mut product = Poly::from_int(1);
        for (f, m) in &factors {
            for _ in 0..*m {
                product = &product * f;
            }
        }
        let product_monic = product.make_monic();
        let p_monic = p.make_monic();
        assert_eq!(
            product_monic, p_monic,
            "product of squarefree factors should reconstruct original"
        );
    }

    // ── Lagrange interpolation (rational) ───────────────────────────

    #[test]
    fn lagrange_rational_constant() {
        // f(x) = 5 → interpolate from (0,5), (1,5)
        let pts = vec![(0i64, ri(5)), (1, ri(5))];
        let p = lagrange_interpolate_rational(&pts);
        assert_eq!(p, Poly::from_int(5));
    }

    #[test]
    fn lagrange_rational_linear() {
        // f(x) = 2x + 1 → (0,1), (1,3)
        let pts = vec![(0i64, ri(1)), (1, ri(3))];
        let p = lagrange_interpolate_rational(&pts);
        assert_eq!(p, Poly::from_coeffs(vec![ri(1), ri(2)]));
    }

    #[test]
    fn lagrange_rational_quadratic() {
        // f(x) = x^2 → (0,0), (1,1), (2,4)
        let pts = vec![(0i64, ri(0)), (1, ri(1)), (2, ri(4))];
        let p = lagrange_interpolate_rational(&pts);
        assert_eq!(p, Poly::from_coeffs(vec![ri(0), ri(0), ri(1)]));
    }

    // ── Construction ────────────────────────────────────────────────

    #[test]
    fn zero_polynomial() {
        let p = Poly::zero();
        assert!(p.is_zero());
        assert_eq!(p.degree(), None);
        assert_eq!(format!("{p}"), "0");
    }

    #[test]
    fn constant_polynomial() {
        let p = Poly::from_int(5);
        assert!(!p.is_zero());
        assert_eq!(p.degree(), Some(0));
        assert_eq!(format!("{p}"), "5");
    }

    #[test]
    fn monomial_x() {
        let p = Poly::x();
        assert_eq!(p.degree(), Some(1));
        assert_eq!(format!("{p}"), "θ");
    }

    #[test]
    fn from_coeffs_strips_trailing_zeros() {
        let p = Poly::from_coeffs(vec![ri(1), ri(2), ri(0), ri(0)]);
        assert_eq!(p.degree(), Some(1));
    }

    #[test]
    fn display_polynomial() {
        // x^2 + 2x + 1
        let p = Poly::from_coeffs(vec![ri(1), ri(2), ri(1)]);
        assert_eq!(format!("{p}"), "θ^2 + 2*θ + 1");
    }

    #[test]
    fn display_negative_terms() {
        // x^2 - 3x + 2
        let p = Poly::from_coeffs(vec![ri(2), ri(-3), ri(1)]);
        assert_eq!(format!("{p}"), "θ^2 - 3*θ + 2");
    }

    #[test]
    fn display_rational_coeffs() {
        // (1/2)x + 1/3
        let p = Poly::from_coeffs(vec![r(1, 3), r(1, 2)]);
        assert_eq!(format!("{p}"), "(1/2)*θ + 1/3");
    }

    // ── Evaluation ──────────────────────────────────────────────────

    #[test]
    fn eval_polynomial() {
        // x^2 + 2x + 1 at x=3 → 9 + 6 + 1 = 16
        let p = Poly::from_coeffs(vec![ri(1), ri(2), ri(1)]);
        assert_eq!(p.eval(&ri(3)), ri(16));
    }

    #[test]
    fn eval_zero_polynomial() {
        let p = Poly::zero();
        assert_eq!(p.eval(&ri(42)), ri(0));
    }

    // ── Add ─────────────────────────────────────────────────────────

    #[test]
    fn add_polynomials() {
        // (x + 1) + (x^2 + 2) = x^2 + x + 3
        let a = Poly::from_coeffs(vec![ri(1), ri(1)]);
        let b = Poly::from_coeffs(vec![ri(2), ri(0), ri(1)]);
        let c = &a + &b;
        assert_eq!(format!("{c}"), "θ^2 + θ + 3");
    }

    #[test]
    fn add_cancellation() {
        // (x + 1) + (-x + 2) = 3
        let a = Poly::from_coeffs(vec![ri(1), ri(1)]);
        let b = Poly::from_coeffs(vec![ri(2), ri(-1)]);
        let c = &a + &b;
        assert_eq!(format!("{c}"), "3");
    }

    #[test]
    fn add_to_zero() {
        let a = Poly::from_coeffs(vec![ri(1), ri(2)]);
        let b = Poly::from_coeffs(vec![ri(-1), ri(-2)]);
        let c = &a + &b;
        assert!(c.is_zero());
    }

    // ── Sub ─────────────────────────────────────────────────────────

    #[test]
    fn sub_polynomials() {
        // (x^2 + x) - (x - 1) = x^2 + 1
        let a = Poly::from_coeffs(vec![ri(0), ri(1), ri(1)]);
        let b = Poly::from_coeffs(vec![ri(-1), ri(1)]);
        let c = &a - &b;
        assert_eq!(format!("{c}"), "θ^2 + 1");
    }

    #[test]
    fn sub_self_is_zero() {
        let a = Poly::from_coeffs(vec![ri(3), ri(2), ri(1)]);
        let c = &a - &a;
        assert!(c.is_zero());
    }

    // ── Neg ─────────────────────────────────────────────────────────

    #[test]
    fn neg_polynomial() {
        let a = Poly::from_coeffs(vec![ri(1), ri(-2), ri(3)]);
        let b = -&a;
        assert_eq!(format!("{b}"), "-3*θ^2 + 2*θ - 1");
    }

    // ── Mul ─────────────────────────────────────────────────────────

    #[test]
    fn mul_polynomials() {
        // (x + 1) * (x - 1) = x^2 - 1
        let a = Poly::from_coeffs(vec![ri(1), ri(1)]);
        let b = Poly::from_coeffs(vec![ri(-1), ri(1)]);
        let c = &a * &b;
        assert_eq!(format!("{c}"), "θ^2 - 1");
    }

    #[test]
    fn mul_by_zero() {
        let a = Poly::from_coeffs(vec![ri(1), ri(2), ri(3)]);
        let b = Poly::zero();
        assert!((&a * &b).is_zero());
    }

    #[test]
    fn mul_by_constant() {
        let a = Poly::from_coeffs(vec![ri(1), ri(2)]);
        let b = Poly::from_int(3);
        let c = &a * &b;
        assert_eq!(format!("{c}"), "6*θ + 3");
    }

    #[test]
    fn mul_larger() {
        // (x + 1)^2 = x^2 + 2x + 1
        let a = Poly::from_coeffs(vec![ri(1), ri(1)]);
        let c = &a * &a;
        assert_eq!(format!("{c}"), "θ^2 + 2*θ + 1");
    }

    #[test]
    fn scale_polynomial() {
        let a = Poly::from_coeffs(vec![ri(2), ri(4)]);
        let b = a.scale(&r(1, 2));
        assert_eq!(format!("{b}"), "2*θ + 1");
    }

    // ── Div / Rem ───────────────────────────────────────────────────

    #[test]
    fn div_rem_exact() {
        // (x^2 - 1) / (x - 1) = (x + 1), remainder 0
        let dividend = Poly::from_coeffs(vec![ri(-1), ri(0), ri(1)]);
        let divisor = Poly::from_coeffs(vec![ri(-1), ri(1)]);
        let (q, r) = dividend.div_rem(&divisor);
        assert_eq!(format!("{q}"), "θ + 1");
        assert!(r.is_zero(), "remainder should be zero, got: {r}");
    }

    #[test]
    fn div_rem_with_remainder() {
        // (x^2 + 1) / (x - 1) = (x + 1), remainder 2
        let dividend = Poly::from_coeffs(vec![ri(1), ri(0), ri(1)]);
        let divisor = Poly::from_coeffs(vec![ri(-1), ri(1)]);
        let (q, r) = dividend.div_rem(&divisor);
        assert_eq!(format!("{q}"), "θ + 1");
        assert_eq!(format!("{r}"), "2");
    }

    #[test]
    fn div_rem_lower_degree() {
        // (x + 1) / (x^2 + 1) = 0, remainder (x + 1)
        let dividend = Poly::from_coeffs(vec![ri(1), ri(1)]);
        let divisor = Poly::from_coeffs(vec![ri(1), ri(0), ri(1)]);
        let (q, r) = dividend.div_rem(&divisor);
        assert!(q.is_zero());
        assert_eq!(format!("{r}"), "θ + 1");
    }

    #[test]
    fn div_rem_constant_divisor() {
        // (2x + 4) / 2 = (x + 2), remainder 0
        let dividend = Poly::from_coeffs(vec![ri(4), ri(2)]);
        let divisor = Poly::from_int(2);
        let (q, r) = dividend.div_rem(&divisor);
        assert_eq!(format!("{q}"), "θ + 2");
        assert!(r.is_zero());
    }

    #[test]
    #[should_panic(expected = "division by zero polynomial")]
    fn div_by_zero_panics() {
        let a = Poly::from_int(1);
        let _ = a.div_rem(&Poly::zero());
    }

    // ── GCD ─────────────────────────────────────────────────────────

    #[test]
    fn gcd_coprime() {
        // gcd(x + 1, x + 2) = 1  (coprime)
        let a = Poly::from_coeffs(vec![ri(1), ri(1)]);
        let b = Poly::from_coeffs(vec![ri(2), ri(1)]);
        let g = Poly::gcd(&a, &b);
        assert_eq!(format!("{g}"), "1");
    }

    #[test]
    fn gcd_common_factor() {
        // gcd(x^2 - 1, x^2 - 2x + 1) = gcd((x-1)(x+1), (x-1)^2) = x - 1
        let a = Poly::from_coeffs(vec![ri(-1), ri(0), ri(1)]); // x^2 - 1
        let b = Poly::from_coeffs(vec![ri(1), ri(-2), ri(1)]); // x^2 - 2x + 1
        let g = Poly::gcd(&a, &b);
        // Should be monic: x - 1 (or equivalent)
        assert_eq!(g.degree(), Some(1), "GCD should be degree 1");
        // Verify by checking that g divides both a and b.
        assert!(a.rem(&g).is_zero(), "g should divide a");
        assert!(b.rem(&g).is_zero(), "g should divide b");
    }

    #[test]
    fn gcd_identical() {
        let a = Poly::from_coeffs(vec![ri(1), ri(2), ri(1)]);
        let g = Poly::gcd(&a, &a);
        // GCD of a with itself is the monic version of a.
        assert_eq!(g.degree(), a.degree());
        assert!(a.rem(&g).is_zero());
    }

    #[test]
    fn gcd_with_zero() {
        let a = Poly::from_coeffs(vec![ri(4), ri(2)]);
        let g = Poly::gcd(&a, &Poly::zero());
        // GCD(a, 0) = monic(a): 2x + 4 → monic → x + 2.
        assert_eq!(g.degree(), Some(1));
        assert_eq!(g.leading_coeff(), Some(&ri(1)));
    }

    #[test]
    fn gcd_both_zero() {
        let g = Poly::gcd(&Poly::zero(), &Poly::zero());
        assert!(g.is_zero());
    }

    #[test]
    fn gcd_quadratic_factors() {
        // a = (x - 1)(x - 2)(x - 3) = x^3 - 6x^2 + 11x - 6
        // b = (x - 1)(x - 2)(x - 4) = x^3 - 7x^2 + 14x - 8
        // gcd = (x - 1)(x - 2) = x^2 - 3x + 2
        let x_minus = |n: i64| Poly::from_coeffs(vec![ri(-n), ri(1)]);
        let a = &(&x_minus(1) * &x_minus(2)) * &x_minus(3);
        let b = &(&x_minus(1) * &x_minus(2)) * &x_minus(4);
        let g = Poly::gcd(&a, &b);
        assert_eq!(g.degree(), Some(2));
        // Verify it divides both.
        assert!(a.rem(&g).is_zero(), "g should divide a");
        assert!(b.rem(&g).is_zero(), "g should divide b");
        // Check it's the right polynomial (x^2 - 3x + 2 monic).
        assert_eq!(g.eval(&ri(1)), ri(0), "g(1) should be 0");
        assert_eq!(g.eval(&ri(2)), ri(0), "g(2) should be 0");
    }

    // ── Monic / Primitive ───────────────────────────────────────────

    #[test]
    fn make_monic() {
        let p = Poly::from_coeffs(vec![ri(2), ri(4)]);
        let m = p.make_monic();
        assert_eq!(m.leading_coeff(), Some(&ri(1)));
        assert_eq!(format!("{m}"), "θ + 1/2");
    }

    #[test]
    fn make_monic_already_monic() {
        let p = Poly::from_coeffs(vec![ri(1), ri(1)]);
        let m = p.make_monic();
        assert_eq!(p, m);
    }

    #[test]
    fn make_monic_zero() {
        let p = Poly::zero();
        let m = p.make_monic();
        assert!(m.is_zero());
    }

    // ── Correctness via evaluation ──────────────────────────────────

    #[test]
    fn mul_and_div_roundtrip() {
        // (x + 1) * (x + 2) = x^2 + 3x + 2
        // divided by (x + 1) should give (x + 2)
        let a = Poly::from_coeffs(vec![ri(1), ri(1)]);
        let b = Poly::from_coeffs(vec![ri(2), ri(1)]);
        let product = &a * &b;
        let (q, r) = product.div_rem(&a);
        assert_eq!(q, b);
        assert!(r.is_zero());
    }

    #[test]
    fn div_rem_identity() {
        // For any a, b (b ≠ 0): a = q*b + r
        let a = Poly::from_coeffs(vec![ri(1), ri(-3), ri(0), ri(2), ri(1)]);
        let b = Poly::from_coeffs(vec![ri(1), ri(1)]);
        let (q, r) = a.div_rem(&b);
        let reconstructed = &(&q * &b) + &r;
        assert_eq!(a, reconstructed, "a should equal q*b + r");
    }

    // ── Deep nesting (verify no stack issues) ───────────────────────

    #[test]
    fn multiply_many_linear_factors() {
        // (x-1)(x-2)...(x-10) — degree 10 polynomial
        let mut result = Poly::from_int(1);
        for i in 1..=10 {
            let factor = Poly::from_coeffs(vec![ri(-i), ri(1)]);
            result = &result * &factor;
        }
        assert_eq!(result.degree(), Some(10));
        // All roots should evaluate to 0.
        for i in 1..=10 {
            assert_eq!(result.eval(&ri(i)), ri(0), "should be zero at x={i}");
        }
    }

    #[test]
    fn gcd_of_products() {
        // gcd((x-1)(x-2)(x-3)(x-4), (x-2)(x-4)(x-6)) = (x-2)(x-4)
        let x_minus = |n: i64| Poly::from_coeffs(vec![ri(-n), ri(1)]);
        let a = &(&(&x_minus(1) * &x_minus(2)) * &x_minus(3)) * &x_minus(4);
        let b = &(&x_minus(2) * &x_minus(4)) * &x_minus(6);
        let g = Poly::gcd(&a, &b);
        assert_eq!(g.degree(), Some(2));
        assert_eq!(g.eval(&ri(2)), ri(0));
        assert_eq!(g.eval(&ri(4)), ri(0));
        assert!(g.eval(&ri(1)) != ri(0), "x=1 should not be a root of gcd");
    }

    // ── Content / Primitive part ────────────────────────────────────

    #[test]
    fn content_of_2x_plus_4() {
        // 2x + 4 → content = 2
        let p = Poly::from_coeffs(vec![ri(4), ri(2)]);
        assert_eq!(p.content(), ri(2));
    }

    #[test]
    fn content_of_6x2_4x_2() {
        // 6x² + 4x + 2 → content = 2
        let p = Poly::from_coeffs(vec![ri(2), ri(4), ri(6)]);
        assert_eq!(p.content(), ri(2));
    }

    #[test]
    fn content_of_x_plus_1() {
        // x + 1 → content = 1
        let p = Poly::from_coeffs(vec![ri(1), ri(1)]);
        assert_eq!(p.content(), ri(1));
    }

    #[test]
    fn primitive_part_of_2x_plus_4() {
        // 2x + 4 → primitive part = x + 2
        let p = Poly::from_coeffs(vec![ri(4), ri(2)]);
        let pp = p.primitive_part();
        let expected = Poly::from_coeffs(vec![ri(2), ri(1)]);
        assert_eq!(pp, expected);
    }

    // ── Factoring over ℤ ────────────────────────────────────────────

    #[test]
    fn has_integer_coeffs_true() {
        let p = Poly::from_coeffs(vec![ri(1), ri(2), ri(3)]);
        assert!(p.has_integer_coeffs());
    }

    #[test]
    fn has_integer_coeffs_false() {
        let p = Poly::from_coeffs(vec![r(1, 2), ri(1)]);
        assert!(!p.has_integer_coeffs());
    }

    #[test]
    fn factor_z_x2_minus_1() {
        // x^2 - 1 = (x - 1)(x + 1)
        let p = Poly::from_coeffs(vec![ri(-1), ri(0), ri(1)]);
        let (content, factors) = p.factor_over_z();
        assert_eq!(content, ri(1));
        assert_eq!(factors.len(), 2, "should have 2 factors: {factors:?}");
        // Verify product.
        let mut product = Poly::from_int(1);
        for (f, m) in &factors {
            for _ in 0..*m {
                product = &product * f;
            }
        }
        product = product.scale(&content);
        assert_eq!(product, p, "product of factors should equal original");
    }

    #[test]
    fn factor_z_x2_plus_1_irreducible() {
        // x^2 + 1 is irreducible over ℤ
        let p = Poly::from_coeffs(vec![ri(1), ri(0), ri(1)]);
        let (content, factors) = p.factor_over_z();
        assert_eq!(content, ri(1));
        assert_eq!(factors.len(), 1, "x^2+1 is irreducible: {factors:?}");
        assert_eq!(factors[0].1, 1);
    }

    #[test]
    fn factor_z_x4_minus_1() {
        // x^4 - 1 = (x - 1)(x + 1)(x^2 + 1)
        let p = Poly::from_coeffs(vec![ri(-1), ri(0), ri(0), ri(0), ri(1)]);
        let (content, factors) = p.factor_over_z();
        assert_eq!(content, ri(1));
        assert!(
            factors.len() >= 3,
            "x^4-1 should have >= 3 factors: {factors:?}"
        );
        // Verify product.
        let mut product = Poly::from_int(1);
        for (f, m) in &factors {
            for _ in 0..*m {
                product = &product * f;
            }
        }
        product = product.scale(&content);
        assert_eq!(product, p, "product mismatch");
    }

    #[test]
    fn factor_z_x4_plus_5x2_plus_6() {
        // x^4 + 5x^2 + 6 = (x^2 + 2)(x^2 + 3)
        let p = Poly::from_coeffs(vec![ri(6), ri(0), ri(5), ri(0), ri(1)]);
        let (content, factors) = p.factor_over_z();
        assert_eq!(content, ri(1));
        assert_eq!(
            factors.len(),
            2,
            "should factor into 2 quadratics: {factors:?}"
        );
        // Verify product.
        let mut product = Poly::from_int(1);
        for (f, m) in &factors {
            for _ in 0..*m {
                product = &product * f;
            }
        }
        product = product.scale(&content);
        assert_eq!(product, p, "product mismatch");
    }

    #[test]
    fn factor_z_perfect_square() {
        // x^2 + 2x + 1 = (x + 1)^2
        let p = Poly::from_coeffs(vec![ri(1), ri(2), ri(1)]);
        let (content, factors) = p.factor_over_z();
        assert_eq!(content, ri(1));
        assert_eq!(factors.len(), 1, "perfect square has 1 unique factor");
        assert_eq!(factors[0].1, 2, "multiplicity should be 2");
    }

    #[test]
    fn factor_z_6x2_plus_12x_plus_6() {
        // 6x^2 + 12x + 6 = 6·(x + 1)^2
        let p = Poly::from_coeffs(vec![ri(6), ri(12), ri(6)]);
        let (content, factors) = p.factor_over_z();
        assert_eq!(content, ri(6));
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0].1, 2);
        assert_eq!(factors[0].0.degree(), Some(1));
    }

    #[test]
    fn factor_z_cubic_three_roots() {
        // x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)
        let p = Poly::from_coeffs(vec![ri(-6), ri(11), ri(-6), ri(1)]);
        let (content, factors) = p.factor_over_z();
        assert_eq!(content, ri(1));
        assert_eq!(
            factors.len(),
            3,
            "should have 3 linear factors: {factors:?}"
        );
        // Verify product.
        let mut product = Poly::from_int(1);
        for (f, m) in &factors {
            for _ in 0..*m {
                product = &product * f;
            }
        }
        product = product.scale(&content);
        assert_eq!(product, p, "product mismatch");
    }

    #[test]
    fn factor_z_x6_minus_1() {
        // x^6 - 1 = (x-1)(x+1)(x^2-x+1)(x^2+x+1)
        let mut coeffs = vec![ri(0); 7];
        coeffs[0] = ri(-1);
        coeffs[6] = ri(1);
        let p = Poly::from_coeffs(coeffs);
        let (content, factors) = p.factor_over_z();
        assert_eq!(content, ri(1));
        assert!(
            factors.len() >= 4,
            "x^6-1 should have >= 4 factors: {factors:?}"
        );
        // Verify product.
        let mut product = Poly::from_int(1);
        for (f, m) in &factors {
            for _ in 0..*m {
                product = &product * f;
            }
        }
        product = product.scale(&content);
        assert_eq!(product, p, "product mismatch");
    }

    #[test]
    fn factor_z_constant() {
        let p = Poly::from_int(42);
        let (content, factors) = p.factor_over_z();
        assert_eq!(content, ri(42));
        assert!(factors.is_empty());
    }

    #[test]
    fn factor_z_zero() {
        let p = Poly::zero();
        let (content, factors) = p.factor_over_z();
        assert!(content.is_zero());
        assert!(factors.is_empty());
    }

    #[test]
    fn factor_z_linear() {
        let p = Poly::from_coeffs(vec![ri(4), ri(2)]); // 2x + 4
        let (content, factors) = p.factor_over_z();
        assert_eq!(content, ri(2));
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0].1, 1);
        // Factor should be (x + 2).
        let f = &factors[0].0;
        assert_eq!(f.eval(&ri(-2)), ri(0));
    }

    #[test]
    fn factor_z_preserves_value_at_points() {
        // x^4 + 5x^2 + 6
        let p = Poly::from_coeffs(vec![ri(6), ri(0), ri(5), ri(0), ri(1)]);
        let (content, factors) = p.factor_over_z();
        for &x in &[-3i64, -2, -1, 0, 1, 2, 3] {
            let xr = ri(x);
            let original = p.eval(&xr);
            let mut factored = content.clone();
            for (f, m) in &factors {
                let val = f.eval(&xr);
                for _ in 0..*m {
                    factored = &factored * &val;
                }
            }
            assert_eq!(
                original, factored,
                "value mismatch at x={x}: orig={original}, factored={factored}"
            );
        }
    }

    // ── Specialized derivative ───────────────────────────────────────

    #[test]
    fn derivative_specialized_matches_generic() {
        // Verify the specialized derivative matches manual calculation
        // p = 3x^3 + 2x^2 + x + 5  →  p' = 9x^2 + 4x + 1
        let p = Poly::from_coeffs(vec![ri(5), ri(1), ri(2), ri(3)]);
        let dp = p.derivative();
        let expected = Poly::from_coeffs(vec![ri(1), ri(4), ri(9)]);
        assert_eq!(dp, expected);
    }

    #[test]
    fn derivative_constant_is_zero() {
        let p = Poly::from_int(42);
        assert!(p.derivative().is_zero());
    }

    #[test]
    fn derivative_zero_is_zero() {
        assert!(Poly::zero().derivative().is_zero());
    }

    #[test]
    fn derivative_linear() {
        // 3x + 7 → 3
        let p = Poly::from_coeffs(vec![ri(7), ri(3)]);
        assert_eq!(p.derivative(), Poly::from_int(3));
    }

    #[test]
    fn derivative_high_degree_exact() {
        // p = x^10 + x^5 + 1
        // p' = 10x^9 + 5x^4
        let mut coeffs = vec![ri(0); 11];
        coeffs[0] = ri(1);
        coeffs[5] = ri(1);
        coeffs[10] = ri(1);
        let p = Poly::from_coeffs(coeffs);
        let dp = p.derivative();

        assert_eq!(dp.degree(), Some(9));
        assert_eq!(dp.coeff(9), ri(10), "coeff of x^9 should be 10");
        assert_eq!(dp.coeff(4), ri(5), "coeff of x^4 should be 5");
        // All other coefficients should be zero
        for i in [0, 1, 2, 3, 5, 6, 7, 8] {
            assert_eq!(dp.coeff(i), ri(0), "coeff of x^{i} should be 0");
        }
    }

    #[test]
    fn derivative_degree_20_coefficient_check() {
        // p = Σ_{k=0}^{20} x^k  (all coefficients = 1)
        // p' = Σ_{k=1}^{20} k * x^{k-1} = Σ_{j=0}^{19} (j+1) * x^j
        let coeffs: Vec<_> = (0..=20).map(|_| ri(1)).collect();
        let p = Poly::from_coeffs(coeffs);
        let dp = p.derivative();
        assert_eq!(dp.degree(), Some(19));
        for j in 0..=19usize {
            assert_eq!(
                dp.coeff(j),
                ri(j as i64 + 1),
                "derivative coeff at x^{j} should be {}",
                j + 1
            );
        }
    }

    #[test]
    fn derivative_evaluation_cross_check() {
        // For p(x) = x^7 - 3x^4 + 2x^2 + 5x - 1
        // verify p'(x) by evaluating at multiple rational points
        // and comparing against (p(x+h) - p(x-h)) / (2h) for small h
        let p = Poly::from_coeffs(vec![
            ri(-1),
            ri(5),
            ri(2),
            ri(0),
            ri(-3),
            ri(0),
            ri(0),
            ri(1),
        ]);
        let dp = p.derivative();

        // Check exact derivative at x=0: p'(0) = 5
        assert_eq!(dp.eval(&ri(0)), ri(5));

        // Check exact derivative at x=1: 7(1)^6 - 12(1)^3 + 4(1) + 5 = 7-12+4+5 = 4
        assert_eq!(dp.eval(&ri(1)), ri(4));

        // Check exact derivative at x=-1: 7(-1)^6 - 12(-1)^3 + 4(-1) + 5 = 7+12-4+5 = 20
        assert_eq!(dp.eval(&ri(-1)), ri(20));

        // Numerical finite-difference cross-check at x=2
        let h = r(1, 1000);
        let x = ri(2);
        let x_plus_h = &x + &h;
        let x_minus_h = &x - &h;
        let fd =
            (p.eval(&x_plus_h) - p.eval(&x_minus_h)) / (Ratio::from_integer(BigInt::from(2)) * &h);
        let exact = dp.eval(&x);
        // Finite difference should be close to the exact derivative
        let diff = (&fd - &exact).abs();
        assert!(
            diff < r(1, 100),
            "finite diff at x=2: fd={fd}, exact={exact}, diff={diff}"
        );
    }

    #[test]
    fn derivative_product_rule_cross_check() {
        // Verify (fg)' = f'g + fg' for two polynomials
        let f = Poly::from_coeffs(vec![ri(1), ri(2), ri(3)]); // 3x^2 + 2x + 1
        let g = Poly::from_coeffs(vec![ri(-1), ri(1)]); // x - 1
        let fg = &f * &g;
        let fg_prime = fg.derivative();
        let f_prime_g = &f.derivative() * &g;
        let f_g_prime = &f * &g.derivative();
        let leibniz = &f_prime_g + &f_g_prime;
        assert_eq!(
            fg_prime, leibniz,
            "product rule: (fg)' should equal f'g + fg'"
        );
    }

    #[test]
    fn derivative_chain_rule_power_cross_check() {
        // Verify d/dx[p(x)^2] = 2 p(x) p'(x) via evaluation
        let p = Poly::from_coeffs(vec![ri(1), ri(-1), ri(1)]); // x^2 - x + 1
        let p_sq = &p * &p;
        let p_sq_prime = p_sq.derivative();
        let two_p_pprime = &(&p * &p.derivative()).scale(&ri(2));
        // Check at several points
        for x_val in [0i64, 1, 2, -1, -2, 3] {
            let xr = ri(x_val);
            let lhs = p_sq_prime.eval(&xr);
            let rhs = two_p_pprime.eval(&xr);
            assert_eq!(
                lhs, rhs,
                "chain rule failed at x={x_val}: d/dx[p²]={lhs}, 2pp'={rhs}"
            );
        }
    }

    // ── Cross-validation: GenPoly operations match expected math ─────

    #[test]
    fn extended_gcd_bezout_identity() {
        // For a = x^3 - 1, b = x^2 - 1
        // verify s*a + t*b = gcd(a, b) (Bézout's identity)
        let a = Poly::from_coeffs(vec![ri(-1), ri(0), ri(0), ri(1)]); // x^3 - 1
        let b = Poly::from_coeffs(vec![ri(-1), ri(0), ri(1)]); // x^2 - 1
        let num_integer::ExtendedGcd { gcd: g, x: s, y: t } = Poly::extended_gcd(&a, &b);

        // Check that g divides both a and b
        assert!(a.rem(&g).is_zero(), "gcd should divide a");
        assert!(b.rem(&g).is_zero(), "gcd should divide b");

        // Check Bézout: s*a + t*b == g at several points
        for x_val in [-3i64, -1, 0, 1, 2, 5] {
            let xr = ri(x_val);
            let lhs = s.eval(&xr) * a.eval(&xr) + t.eval(&xr) * b.eval(&xr);
            let rhs = g.eval(&xr);
            assert_eq!(
                lhs, rhs,
                "Bézout identity failed at x={x_val}: sa+tb={lhs}, g={rhs}"
            );
        }
    }

    #[test]
    fn extended_gcd_bezout_identity_coprime() {
        // For coprime polynomials, gcd should be 1 and s*a + t*b = 1
        let a = Poly::from_coeffs(vec![ri(1), ri(0), ri(1)]); // x^2 + 1
        let b = Poly::from_coeffs(vec![ri(1), ri(1)]); // x + 1
        let num_integer::ExtendedGcd { gcd: g, x: s, y: t } = Poly::extended_gcd(&a, &b);

        assert!(g.is_constant(), "gcd of coprime polys should be constant");
        assert_eq!(g.coeff(0), ri(1), "gcd should be 1");

        for x_val in [-2i64, 0, 1, 3, 7] {
            let xr = ri(x_val);
            let lhs = s.eval(&xr) * a.eval(&xr) + t.eval(&xr) * b.eval(&xr);
            assert_eq!(lhs, ri(1), "Bézout s*a+t*b should be 1 at x={x_val}");
        }
    }

    #[test]
    fn div_rem_identity_multipoint() {
        // For several (a, b) pairs, verify a = q*b + r at multiple points
        let pairs = [
            (
                Poly::from_coeffs(vec![ri(1), ri(-3), ri(0), ri(2), ri(1)]),
                Poly::from_coeffs(vec![ri(1), ri(1)]),
            ),
            (
                Poly::from_coeffs(vec![ri(6), ri(-5), ri(1)]), // x^2 - 5x + 6
                Poly::from_coeffs(vec![ri(-2), ri(1)]),        // x - 2
            ),
            (
                Poly::from_coeffs(vec![ri(1), ri(0), ri(0), ri(0), ri(0), ri(1)]), // x^5 + 1
                Poly::from_coeffs(vec![ri(1), ri(1)]),                             // x + 1
            ),
        ];
        for (a, b) in &pairs {
            let (q, r) = a.div_rem(b);
            // Structural check
            let reconstructed = &(&q * b) + &r;
            assert_eq!(a, &reconstructed, "a != q*b+r structurally");
            // Point-wise check
            for x_val in [-5i64, -1, 0, 1, 2, 7] {
                let xr = ri(x_val);
                let lhs = a.eval(&xr);
                let rhs = q.eval(&xr) * b.eval(&xr) + r.eval(&xr);
                assert_eq!(
                    lhs, rhs,
                    "div_rem identity failed at x={x_val}: a={lhs}, qb+r={rhs}"
                );
            }
        }
    }

    #[test]
    fn square_free_part_removes_repeated_roots() {
        // p = (x-1)^3 * (x-2) = x^4 - 5x^3 + 9x^2 - 7x + 2
        let x_minus = |n: i64| Poly::from_coeffs(vec![ri(-n), ri(1)]);
        let p = &(&(&x_minus(1) * &x_minus(1)) * &x_minus(1)) * &x_minus(2);

        let sfp = p.square_free_part();
        // Square-free part should have roots at x=1 and x=2, each with multiplicity 1
        assert_eq!(sfp.eval(&ri(1)), ri(0), "x=1 should be root of sfp");
        assert_eq!(sfp.eval(&ri(2)), ri(0), "x=2 should be root of sfp");
        // Degree should be 2 (two distinct roots)
        assert_eq!(sfp.degree(), Some(2), "sfp should be degree 2");
    }

    #[test]
    fn gcd_value_divides_both_inputs_at_all_points() {
        // a = (x-1)(x-2)(x-3), b = (x-2)(x-4)(x-6)
        // gcd should divide both at all evaluation points
        let x_minus = |n: i64| Poly::from_coeffs(vec![ri(-n), ri(1)]);
        let a = &(&x_minus(1) * &x_minus(2)) * &x_minus(3);
        let b = &(&x_minus(2) * &x_minus(4)) * &x_minus(6);
        let g = Poly::gcd(&a, &b);

        // g should be monic degree 1 (just x-2)
        assert_eq!(g.degree(), Some(1), "gcd degree");
        assert_eq!(g.eval(&ri(2)), ri(0), "x=2 should be root of gcd");
        assert!(g.eval(&ri(4)) != ri(0), "x=4 should NOT be root of gcd");

        // Verify division is exact
        assert!(a.rem(&g).is_zero(), "g should divide a");
        assert!(b.rem(&g).is_zero(), "g should divide b");
    }

    #[test]
    fn resultant_matches_evaluation_at_roots() {
        // res(f, g) = lc(g)^deg(f) * ∏ f(root_of_g)
        // For g = (x-2)(x-3) = x^2-5x+6, roots are 2 and 3
        // f = x^2 + 1
        // res = 1^2 * f(2) * f(3) = 5 * 10 = 50
        let f = Poly::from_coeffs(vec![ri(1), ri(0), ri(1)]); // x^2 + 1
        let g = Poly::from_coeffs(vec![ri(6), ri(-5), ri(1)]); // x^2-5x+6
        let res = Poly::resultant(&f, &g);
        let expected = f.eval(&ri(2)) * f.eval(&ri(3)); // 5 * 10 = 50
        assert_eq!(res, expected, "resultant should be 50, got {res}");
    }

    #[test]
    fn squarefree_factors_reconstruct_to_original() {
        // (x-1)^2 * (x+1)^3 * (x-5)
        let x_minus = |n: i64| Poly::from_coeffs(vec![ri(-n), ri(1)]);
        let p = &(&(&(&x_minus(1) * &x_minus(1))
            * &(&(&x_minus(-1) * &x_minus(-1)) * &x_minus(-1)))
            * &x_minus(5));

        let factors = p.squarefree_factors();

        // Reconstruct
        let mut product = Poly::from_int(1);
        for (f, m) in &factors {
            for _ in 0..*m {
                product = &product * f;
            }
        }
        // Monic versions should match
        let p_monic = p.make_monic();
        let prod_monic = product.make_monic();
        assert_eq!(
            p_monic, prod_monic,
            "squarefree factor product should reconstruct original"
        );

        // Verify at evaluation points
        for x_val in [-2i64, -1, 0, 1, 2, 5, 7] {
            let xr = ri(x_val);
            let orig = p_monic.eval(&xr);
            let recon = prod_monic.eval(&xr);
            assert_eq!(
                orig, recon,
                "squarefree reconstruction mismatch at x={x_val}"
            );
        }
    }

    // ── Eq and Hash ─────────────────────────────────────────────────

    #[test]
    fn eq_and_hash_consistent() {
        use std::collections::HashSet;
        let a = Poly::from_coeffs(vec![ri(1), ri(2), ri(1)]);
        let b = Poly::from_coeffs(vec![ri(1), ri(2), ri(1)]);
        let c = Poly::from_coeffs(vec![ri(1), ri(3), ri(1)]);
        assert_eq!(a, b);
        assert_ne!(a, c);
        let mut set = HashSet::new();
        set.insert(a.clone());
        assert!(set.contains(&b));
        assert!(!set.contains(&c));
    }

    // ── Operator overloads (owned + mixed) ──────────────────────────

    #[test]
    fn owned_add() {
        let a = Poly::from_coeffs(vec![ri(1), ri(1)]);
        let b = Poly::from_coeffs(vec![ri(2), ri(0), ri(1)]);
        let c = a + b;
        assert_eq!(c, Poly::from_coeffs(vec![ri(3), ri(1), ri(1)]));
    }

    #[test]
    fn mixed_ref_owned_add() {
        let a = Poly::from_coeffs(vec![ri(1), ri(1)]);
        let b = Poly::from_coeffs(vec![ri(2), ri(0), ri(1)]);
        let c = &a + b;
        assert_eq!(c, Poly::from_coeffs(vec![ri(3), ri(1), ri(1)]));
    }

    #[test]
    fn mixed_owned_ref_sub() {
        let a = Poly::from_coeffs(vec![ri(3), ri(2)]);
        let b = Poly::from_coeffs(vec![ri(1), ri(1)]);
        let c = a - &b;
        assert_eq!(c, Poly::from_coeffs(vec![ri(2), ri(1)]));
    }

    #[test]
    fn owned_mul() {
        let a = Poly::from_coeffs(vec![ri(1), ri(1)]);
        let b = Poly::from_coeffs(vec![ri(-1), ri(1)]);
        let c = a * b;
        assert_eq!(c, Poly::from_coeffs(vec![ri(-1), ri(0), ri(1)]));
    }

    #[test]
    fn owned_neg() {
        let a = Poly::from_coeffs(vec![ri(1), ri(-2)]);
        let b = -a;
        assert_eq!(b, Poly::from_coeffs(vec![ri(-1), ri(2)]));
    }
}
