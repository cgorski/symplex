//! Lazard-Rioboo-Trager `log_to_real` conversion.
//!
//! Converts algebraic logarithmic terms from the Rothstein-Trager algorithm
//! into real-valued `ln + atan` expressions.
//!
//! Given an irreducible factor `q(t)` of the Rothstein-Trager resultant
//! (degree ≥ 2) and the degree-1 PRS member `h(t, x)`, this module computes:
//!
//! ```text
//! Σ_{α: q(α)=0} α · ln(h(α, x))
//! ```
//!
//! as a real-valued sum of `ln` and `atan` terms, using the identity:
//!
//! ```text
//! α·ln(h(α,x)) + ᾱ·ln(h(ᾱ,x)) = u·ln(A²+B²) + v·log_to_atan(A, B)
//! ```
//!
//! where `α = u + iv`, `h(α,x) = A(x) + iB(x)`, and `log_to_atan`
//! converts the imaginary part to a sum of arctangents.
//!
//! # Algorithm
//!
//! 1. Solve `q(t) = 0` for complex roots (exact radicals via Cardano/Ferrari
//!    for degree ≤ 4).
//! 2. Decompose each root into `(Re, Im)` via [`as_real_imag`](crate::base::complex::as_real_imag).
//! 3. For each conjugate pair `(u, v)` with `v > 0`:
//!    a. Evaluate `h(u+iv, x)` and separate `A(x) + iB(x)`.
//!    b. Emit `u · ln(A² + B²) + v · log_to_atan(A, B)`.
//!
//! # References
//!
//! - Lazard & Rioboo, "Integration of rational functions: rational computation
//!   of the logarithmic part", 1990
//! - Bronstein, *Symbolic Integration I*, §2.5
//! - SymPy `integrals/rationaltools.py`, `log_to_real` and `log_to_atan`

use num_bigint::BigInt;
use num_rational::Ratio;

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::Q;
use crate::poly::dense::Poly;
use crate::poly::generic::GenPoly;
use crate::poly::ratfn::RationalFn;
use crate::poly::traits::Ring;

// ═══════════════════════════════════════════════════════════════════════════
// Helper: Poly → GenPoly<RationalFn>
// ═══════════════════════════════════════════════════════════════════════════

/// Convert a `Poly` (polynomial in `x` with rational coefficients) to a
/// `GenPoly<RationalFn>` where each coefficient is a constant rational
/// function (denominator = 1, independent of `t`).
pub(crate) fn poly_to_genpoly_rf(p: &Poly) -> GenPoly<RationalFn> {
    let coeffs: Vec<RationalFn> = p
        .coeffs()
        .iter()
        .map(|c| RationalFn::from_rational(c.clone()))
        .collect();
    GenPoly::from_coeffs(coeffs)
}

/// Convert a `Poly` (polynomial in `x` with rational coefficients) to a
/// `GenPoly<RationalFn>` where each coefficient is multiplied by `t`.
///
/// If `p(x) = Σ c_k · x^k`, this returns `Σ (c_k · t) · x^k` where
/// `t` is the variable of the `RationalFn` coefficient type.  Only the tests
/// build `A − t·D'` this way after 0.28.0 (the Lazard–Rioboo–Trager sequence
/// works in `ℤ[t][x]`).
#[cfg(test)]
pub(crate) fn poly_to_genpoly_rf_times_t(p: &Poly) -> GenPoly<RationalFn> {
    let t_poly = Poly::from_coeffs(vec![
        Ratio::from_integer(BigInt::from(0)),
        Ratio::from_integer(BigInt::from(1)),
    ]); // t
    let coeffs: Vec<RationalFn> = p
        .coeffs()
        .iter()
        .map(|c| {
            if c.is_zero() {
                RationalFn::zero()
            } else {
                let c_poly = Poly::constant(c.clone());
                let c_times_t = &c_poly * &t_poly;
                RationalFn::from_poly(c_times_t)
            }
        })
        .collect();
    GenPoly::from_coeffs(coeffs)
}

// ═══════════════════════════════════════════════════════════════════════════
// log_to_atan — convert imaginary log part to arctangent sum
// ═══════════════════════════════════════════════════════════════════════════

/// Convert the imaginary part of a complex logarithm to a sum of arctangents.
///
/// Given `A(x) = a₁·x + a₀` and `B(x) = b₁·x + b₀`, computes an
/// expression `F(x)` such that:
///
/// ```text
/// dF/dx = d/dx [I · ln((A + I·B) / (A - I·B))]
/// ```
///
/// For the common case where `B` is constant (`b₁ ≈ 0`), this is
/// simply `2 · atan(A / B)`.
///
/// For the general case where both `A` and `B` are linear, the result
/// is a sum of two `atan` terms computed via the Bézout identity.
///
/// Returns `None` if the inputs are degenerate (e.g., `B ≈ 0`).
pub(crate) fn log_to_atan_deg1(
    arena: &mut Arena,
    var: ExprId,
    a1: ExprId,
    a0: ExprId,
    b1: ExprId,
    b0: ExprId,
) -> Option<ExprId> {
    let two = arena.int(2);

    // Check if b1 = 0 (B is constant — the common case).
    let b1_is_zero = crate::poly::algebraic::is_zero_checked(arena, b1).unwrap_or(false);

    if b1_is_zero {
        tracing::debug!("log_to_atan_deg1: b1 ≈ 0, B is constant → single atan");

        // Check b0 ≠ 0 (bail unless we can confirm nonzero, since we divide by b0)
        if crate::poly::algebraic::is_zero_checked(arena, b0) != Some(false) {
            tracing::debug!("log_to_atan_deg1: b0 ≈ 0, degenerate → None");
            return None;
        }

        // F = 2 · atan((a₁·x + a₀) / b₀)
        let a_x = arena.mul(&[a1, var]);
        let a_poly = arena.add(&[a_x, a0]);
        let ratio = arena.div(a_poly, b0);
        let atan_val = arena.atan(ratio);
        let result = arena.mul(&[two, atan_val]);
        return Some(result);
    }

    tracing::debug!("log_to_atan_deg1: both A and B are degree 1 → Bézout path");

    // Both A and B are linear: A = a₁x + a₀, B = b₁x + b₀.
    //
    // Polynomial division: A / B gives quotient q = a₁/b₁, remainder r = a₀ - q·b₀.
    let q = arena.div(a1, b1);
    let q_b0 = arena.mul(&[q, b0]);
    let r = arena.sub(a0, q_b0);
    let r = crate::transforms::eval::eval(arena, r);

    let r_is_zero = crate::poly::algebraic::is_zero_checked(arena, r).unwrap_or(false);

    if r_is_zero {
        // Division is exact: F = 2 · atan(q)
        tracing::trace!("log_to_atan_deg1: remainder is zero → single constant atan");
        let atan_q = arena.atan(q);
        return Some(arena.mul(&[two, atan_q]));
    }

    // General case: compute via explicit Bézout identity.
    //
    // For coprime degree-1 polynomials A = a₁x+a₀ and B = b₁x+b₀,
    // the extended GCD gives s, t such that s·B + t·(-A) = 1.
    // Solving the 2×2 linear system:
    //   s·b₁ - t·a₁ = 0  (coefficient of x)
    //   s·b₀ - t·a₀ = 1  (constant term)
    // yields s = a₁/D, t = b₁/D where D = a₁·b₀ - b₁·a₀.
    //
    // Then u = (A·s + B·t) / gcd = (a₁·s + b₁·t)·x + (a₀·s + b₀·t)
    // = ((a₁² + b₁²)/D)·x + ((a₀·a₁ + b₀·b₁)/D).
    //
    // The result is: F = 2·atan(u) + log_to_atan(s, t)
    // where s = a₁/D and t = b₁/D are both constants (degree 0),
    // so log_to_atan(s, t) = 2·atan(s/t) = 2·atan(a₁/b₁).

    let a1_b0 = arena.mul(&[a1, b0]);
    let b1_a0 = arena.mul(&[b1, a0]);
    let det = arena.sub(a1_b0, b1_a0);
    let det = crate::transforms::eval::eval(arena, det);

    if crate::poly::algebraic::is_zero_checked(arena, det).unwrap_or(false) {
        tracing::debug!("log_to_atan_deg1: determinant ≈ 0 (A and B proportional) → None");
        return None;
    }

    // u(x) = ((a₁² + b₁²) / D) · x + ((a₀·a₁ + b₀·b₁) / D)
    let a1_sq = arena.mul(&[a1, a1]);
    let b1_sq = arena.mul(&[b1, b1]);
    let u_x_numer = arena.add(&[a1_sq, b1_sq]);
    let u_x_coeff = arena.div(u_x_numer, det);

    let a0_a1 = arena.mul(&[a0, a1]);
    let b0_b1 = arena.mul(&[b0, b1]);
    let u_c_numer = arena.add(&[a0_a1, b0_b1]);
    let u_const = arena.div(u_c_numer, det);

    let u_x_term = arena.mul(&[u_x_coeff, var]);
    let u_expr = arena.add(&[u_x_term, u_const]);
    let u_expr = crate::transforms::eval::eval(arena, u_expr);
    let atan_u = arena.atan(u_expr);
    let first_atan = arena.mul(&[two, atan_u]);

    // Second term: 2·atan(a₁/b₁)
    let a1_over_b1 = arena.div(a1, b1);
    let a1_over_b1 = crate::transforms::eval::eval(arena, a1_over_b1);
    let atan_a1b1 = arena.atan(a1_over_b1);
    let second_atan = arena.mul(&[two, atan_a1b1]);

    tracing::trace!("log_to_atan_deg1: Bézout path produced 2 atan terms");
    Some(arena.add(&[first_atan, second_atan]))
}

// ═══════════════════════════════════════════════════════════════════════════
// Quadratic factors: exact conversion over ℚ[x]
// ═══════════════════════════════════════════════════════════════════════════

/// `Σ_{q(α)=0} α·ln(S(α, x))` for an irreducible quadratic `q`, as a real
/// expression, with every polynomial computed exactly over `ℚ`.
///
/// Write `q = a·t² + b·t + c`, `u = −b/2a` and `w = √|b² − 4ac| / 2a`.  The
/// coefficients of `S` lie in `ℚ[t]/(q)`, so `S(t, x) = Σ (p_k + s_k·t)·x^k`
/// and at a root `α = u ± w` (real roots) or `u ± i·w` (complex roots)
/// `S(α, x) = P(x) ± w·Q(x)` resp. `P ± i·w·Q`, with
/// `P = Σ (p_k + s_k·u)·x^k` and `Q = Σ s_k·x^k` in `ℚ[x]`.  Then
///
/// - real roots: `(u + w)·ln|P + w·Q| + (u − w)·ln|P − w·Q|`;
/// - complex roots: `u·ln(P² + w²·Q²) + w·LogToAtan(P, w·Q)`
///   ([`log_to_atan_surd`]), where `P² + w²Q² ∈ ℚ[x]` has no real zero.
///
/// `S` is the log argument of the Lazard–Rioboo–Trager algorithm, monic
/// in `x`, of any degree; `None` if `q` is not quadratic or a coefficient
/// of `S` is not a polynomial in `t`.  (Bronstein, *Symbolic Integration I*,
/// §2.8; SymPy's `log_to_real` does the same with the roots found by
/// `roots`.)
pub(crate) fn quadratic_log_to_real(
    arena: &mut Arena,
    var: ExprId,
    q: &Poly,
    s: &GenPoly<RationalFn>,
) -> Option<Vec<ExprId>> {
    if q.degree() != Some(2) {
        return None;
    }
    let (c, b, a) = (q.coeff(0), q.coeff(1), q.coeff(2));
    let two = Q::from_integer(BigInt::from(2));
    let u = -&b / (&two * &a);
    let disc = &b * &b - Q::from_integer(BigInt::from(4)) * &a * &c;
    if disc.is_zero() {
        return None;
    }
    let w2 = num_traits::Signed::abs(&disc) / (Q::from_integer(BigInt::from(4)) * &a * &a);

    // P and Q from the coefficients of S, reduced modulo q.
    let mut p_coeffs = Vec::with_capacity(s.coeffs.len());
    let mut q_coeffs = Vec::with_capacity(s.coeffs.len());
    for coeff in &s.coeffs {
        if !coeff.denom().is_constant() {
            return None;
        }
        let lin = coeff.numer().rem(q).scale(&coeff.denom().coeff(0).recip());
        let (p0, p1) = (lin.coeff(0), lin.coeff(1));
        p_coeffs.push(&p0 + &p1 * &u);
        q_coeffs.push(p1);
    }
    let big_p = Poly::from_coeffs(p_coeffs);
    let big_q = Poly::from_coeffs(q_coeffs);
    if big_q.is_zero() {
        // S(α, x) would not depend on the root: impossible for the log
        // argument of distinct residues.
        return None;
    }

    let u_id = arena.num_ratio(u.clone());
    let w = surd(arena, &w2);
    let p_expr = crate::poly::polybridge::poly_to_expr(arena, &big_p, var);
    let q_expr = crate::poly::polybridge::poly_to_expr(arena, &big_q, var);
    let mut terms = Vec::new();

    if num_traits::Signed::is_positive(&disc) {
        let w_q = arena.mul(&[w, q_expr]);
        let neg_w = arena.neg(w);
        for (sign_w, sign_wq) in [(w, w_q), (neg_w, arena.neg(w_q))] {
            let alpha = arena.add(&[u_id, sign_w]);
            let arg = arena.add(&[p_expr, sign_wq]);
            let abs_arg = arena.abs(arg);
            let ln = arena.ln(abs_arg);
            terms.push(arena.mul(&[alpha, ln]));
        }
    } else {
        if !u.is_zero() {
            let norm = &(&big_p * &big_p) + &(&big_q * &big_q).scale(&w2);
            let norm_expr = crate::poly::polybridge::poly_to_expr(arena, &norm, var);
            let ln = arena.ln(norm_expr);
            terms.push(arena.mul(&[u_id, ln]));
        }
        let atans = log_to_atan_surd(arena, var, &big_p, &big_q, &w2)?;
        terms.push(arena.mul(&[w, atans]));
    }
    Some(terms)
}

/// `√r` for a positive rational `r`, canonicalised by the arena (`√(1/64)`
/// is `1/8`, `√(3/4)` is `√3/2`).
fn surd(arena: &mut Arena, r: &Q) -> ExprId {
    let r_id = arena.num_ratio(r.clone());
    let half = arena.rational(1, 2);
    arena.pow(r_id, half)
}

/// `2·atan(u(x)/√r2)`, written `±2·atan((c/√r2)·p(x))` with `c > 0` and
/// `p` primitive over ℤ with a positive leading coefficient (atan is odd):
/// `2·atan(√3/3·(2x² − 1))` rather than `2·atan(4√3·(x²/6 − 1/12))`.
fn two_atan_poly_over_surd(arena: &mut Arena, u: &Poly, var: ExprId, r2: &Q) -> ExprId {
    let c = u.content();
    if c.is_zero() {
        return arena.zero;
    }
    let negative = u
        .leading_coeff()
        .is_some_and(num_traits::Signed::is_negative);
    let p = u.scale(&if negative { -c.recip() } else { c.recip() });
    let p_expr = crate::poly::polybridge::poly_to_expr(arena, &p, var);
    let c_id = arena.num_ratio(c);
    let inv_r = surd(arena, &r2.recip());
    let arg = arena.mul(&[c_id, inv_r, p_expr]);
    let atan = arena.atan(arg);
    let two = arena.int(if negative { -2 } else { 2 });
    arena.mul(&[two, atan])
}

/// Bronstein's `LogToAtan(A, r·B)` for `A, B ∈ ℚ[x]` and a real constant
/// `r = √r2` (`r2 ∈ ℚ`, positive): a sum `F` of arctangents of real
/// polynomials with `F' = (i·ln((A + i·rB)/(A − i·rB)))'`.
///
/// Bronstein's recursion (*Symbolic Integration I*, §2.8) stays in `ℚ[x]`
/// because only `r²` enters the polynomial arithmetic: with
/// `s·B − t·A = h` over `ℚ` (so `(s/r)·(rB) − t·A = h`),
///
/// - `rB | A`: `2·atan(A/(rB))`;
/// - `deg A < deg B`: `LogToAtan(−rB, A)`, which is `LogToAtan(−B, A/r)` up
///   to the common real factor `r` (the value is invariant under scaling
///   both arguments), i.e. `B ← A`, `A ← −B`, `r2 ← 1/r2`;
/// - otherwise `2·atan((A·s + r2·B·t)/(h·r)) + LogToAtan(s/r, t)`, the
///   last again `LogToAtan(s, r·t)` up to the factor `r`.
///
/// `None` if `B` is zero.
pub(crate) fn log_to_atan_surd(
    arena: &mut Arena,
    var: ExprId,
    a: &Poly,
    b: &Poly,
    r2: &Q,
) -> Option<ExprId> {
    if b.is_zero() {
        return None;
    }
    let (mut a, mut b, mut r2) = (a.clone(), b.clone(), r2.clone());
    let mut atans = Vec::new();
    // Each step lowers deg A + deg B, as in Euclid's algorithm.
    let max_steps = 2 * (a.degree().unwrap_or(0) + b.degree().unwrap_or(0)) + 4;
    for _ in 0..max_steps {
        if b.is_zero() {
            break;
        }
        if a.is_zero() {
            // i·ln(i·rB / (−i·rB)) is constant.
            break;
        }
        if a.degree() < b.degree() {
            (a, b) = (-&b, a);
            r2 = r2.recip();
        }
        let (quot, rem) = a.div_rem(&b);
        if rem.is_zero() {
            atans.push(two_atan_poly_over_surd(arena, &quot, var, &r2));
            break;
        }
        let eg = Poly::extended_gcd(&b, &(-&a));
        let (s, t, h) = (eg.x, eg.y, eg.gcd);
        let u_num = &(&a * &s) + &(&b * &t).scale(&r2);
        let (u, u_rem) = u_num.div_rem(&h);
        if !u_rem.is_zero() {
            return None;
        }
        atans.push(two_atan_poly_over_surd(arena, &u, var, &r2));
        (a, b) = (s, t);
    }
    Some(match atans.len() {
        0 => arena.zero,
        1 => atans[0],
        _ => arena.add(&atans),
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// log_to_real — main conversion function
// ═══════════════════════════════════════════════════════════════════════════

/// Convert an algebraic logarithmic term to real `ln + atan` expressions.
///
/// Given an irreducible factor `q(t)` of the Rothstein-Trager resultant
/// and the degree-1 PRS member `h(t, x)` (as a `GenPoly<RationalFn>`),
/// computes the real-valued contribution:
///
/// ```text
/// Σ_{α: q(α)=0} α · ln(h(α, x))
/// = Σ_j [ u_j · ln(A_j² + B_j²) + v_j · log_to_atan(A_j, B_j) ]
/// ```
///
/// where `(u_j, v_j)` are the real/imaginary parts of the roots of `q`
/// with `v_j > 0` (one from each conjugate pair).
///
/// Returns `None` if the roots of `q` cannot be found (degree ≥ 5 with
/// non-solvable Galois group) or if any intermediate computation fails.
pub(crate) fn log_to_real(
    arena: &mut Arena,
    var: ExprId,
    q: &Poly,
    h_prs: &GenPoly<RationalFn>,
) -> Option<Vec<ExprId>> {
    let q_deg = q.degree().unwrap_or(0);
    tracing::debug!(q_degree = q_deg, "log_to_real: starting conversion");

    if q_deg < 2 {
        tracing::debug!("log_to_real: q has degree < 2, should be handled as rational term");
        return None;
    }

    // ── Step 1: Solve q(t) = 0 for complex roots ──────────────────
    let t_sym = arena.symbol("__lrt_t");
    let q_expr = crate::poly::polybridge::poly_to_expr(arena, q, t_sym);
    let roots = crate::transforms::solve::solve(arena, q_expr, t_sym);

    tracing::debug!(n_roots = roots.len(), "log_to_real: solved q(t) = 0");

    if roots.is_empty() {
        tracing::debug!("log_to_real: no roots found (degree ≥ 5 non-solvable?)");
        return None;
    }

    // ── Step 2: Decompose roots into (u, v) = (Re, Im) pairs ──────
    let i_unit = arena.i_unit;
    let mut pairs: Vec<(ExprId, ExprId)> = Vec::new();
    let mut reals: Vec<ExprId> = Vec::new();
    let mut used = vec![false; roots.len()];

    for (idx, root) in roots.iter().enumerate() {
        if used[idx] {
            continue;
        }

        // The decomposition must be exact (no opaque `re(…)`/`im(…)` nodes):
        // nested Cardano/Ferrari radicals of complex numbers cannot be split
        // into real algebraic parts, and using them would only produce an
        // unusable `ln`/`atan` form.  Bail out so the caller can fall back.
        let parts = crate::base::complex::decompose(arena, root.value);
        if !parts.exact {
            tracing::debug!(idx, "log_to_real: root has no exact Re/Im decomposition");
            return None;
        }
        let (re_raw, im_raw) = (parts.re, parts.im);
        let u_val = crate::transforms::eval::eval(arena, re_raw);
        let v_val = crate::transforms::eval::eval(arena, im_raw);

        // Check imaginary part: cross-checked zero test.
        let v_is_zero = crate::poly::algebraic::is_zero_checked(arena, v_val).unwrap_or(false);

        if v_is_zero {
            // Real (irrational) root α: contributes α·ln|h(α, x)|.  Before
            // 0.25 it was skipped, so `∫ 1/(x³ − 2) dx` lost its
            // `ln|x − ∛2|` term and was rejected.
            tracing::trace!(idx, "log_to_real: real root");
            reals.push(u_val);
            used[idx] = true;
            continue;
        }

        // Sign test: keep roots with positive imaginary part.
        // A root listed before its conjugate may have the negative
        // imaginary part; the pair is represented by the conjugate.
        match crate::poly::algebraic::sign_checked(arena, v_val) {
            Some(1) => {
                tracing::trace!(idx, "log_to_real: found root with positive Im");
                pairs.push((u_val, v_val));
            }
            Some(-1) => {
                let neg_v = arena.neg(v_val);
                let neg_v = crate::transforms::eval::eval(arena, neg_v);
                pairs.push((u_val, neg_v));
            }
            _ => return None,
        }

        // Mark this root as used.
        used[idx] = true;

        // Find and mark the conjugate root (same Re, opposite Im).
        // Conjugate matching uses f64 sum-to-zero heuristic.
        let v_f64 = crate::transforms::evalf::eval_const_f64(arena, v_val)?;
        for j in (idx + 1)..roots.len() {
            if used[j] {
                continue;
            }
            let (_, im_j_raw) = crate::base::complex::as_real_imag(arena, roots[j].value);
            let v_j = crate::transforms::eval::eval(arena, im_j_raw);
            let v_j_f64 = crate::transforms::evalf::eval_const_f64(arena, v_j);
            if let Some(vj) = v_j_f64
                && (vj + v_f64).abs() < 1e-10
            {
                // Conjugate found.
                tracing::trace!(j, "log_to_real: conjugate root found and marked");
                used[j] = true;
                break;
            }
        }
    }

    tracing::debug!(
        n_pairs = pairs.len(),
        "log_to_real: conjugate pairs identified"
    );

    if pairs.is_empty() && reals.is_empty() {
        tracing::debug!("log_to_real: no roots to convert");
        return None;
    }
    if pairs.len() * 2 + reals.len() != q_deg {
        tracing::debug!(
            n_pairs = pairs.len(),
            n_reals = reals.len(),
            "log_to_real: roots not all classified"
        );
        return None;
    }

    // ── Step 3: Extract h(t,x) coefficients ────────────────────────
    // h(t,x) is degree 1 in x (monic): h = 1·x + h₀(t)
    // where h₀(t) is a RationalFn.
    if h_prs.degree() != Some(1) {
        tracing::debug!(
            h_degree = ?h_prs.degree(),
            "log_to_real: h(t,x) is not degree 1 in x, cannot proceed"
        );
        return None;
    }
    let h_coeff_1: RationalFn = h_prs.coeff(1);
    let h_coeff_0: RationalFn = h_prs.coeff(0);

    // Convert to arena expressions in t_sym.
    let h1_expr = crate::poly::polybridge::ratfn_to_expr(arena, &h_coeff_1, t_sym);
    let h0_expr = crate::poly::polybridge::ratfn_to_expr(arena, &h_coeff_0, t_sym);

    tracing::trace!("log_to_real: h(t,x) coefficients converted to arena");

    // ── Step 4: For each (u_j, v_j), build ln + atan terms ────────
    let mut result_terms: Vec<ExprId> = Vec::new();

    for &alpha in &reals {
        let h1_at = crate::transforms::subs::subs(arena, h1_expr, t_sym, alpha);
        let h1_at = crate::transforms::eval::eval(arena, h1_at);
        let h0_at = crate::transforms::subs::subs(arena, h0_expr, t_sym, alpha);
        let h0_at = crate::transforms::eval::eval(arena, h0_at);
        let h1_x = arena.mul(&[h1_at, var]);
        let h_at = arena.add(&[h1_x, h0_at]);
        let abs_h = arena.abs(h_at);
        let ln_h = arena.ln(abs_h);
        result_terms.push(arena.mul(&[alpha, ln_h]));
    }

    for (pair_idx, (u_j, v_j)) in pairs.iter().enumerate() {
        tracing::debug!(pair_idx, "log_to_real: processing conjugate pair");

        // Construct t_val = u_j + I·v_j
        let i_v = arena.mul(&[i_unit, *v_j]);
        let t_val = arena.add(&[*u_j, i_v]);
        let t_val = crate::transforms::eval::eval(arena, t_val);

        // Evaluate h coefficients at t = t_val
        let h1_at = crate::transforms::subs::subs(arena, h1_expr, t_sym, t_val);
        let h1_at = crate::transforms::eval::eval(arena, h1_at);
        let h0_at = crate::transforms::subs::subs(arena, h0_expr, t_sym, t_val);
        let h0_at = crate::transforms::eval::eval(arena, h0_at);

        // h(α, x) = h1_at · x + h0_at
        // Separate Re/Im for each coefficient.
        let (re_h1, im_h1) = crate::base::complex::as_real_imag(arena, h1_at);
        let re_h1 = crate::transforms::eval::eval(arena, re_h1);
        let im_h1 = crate::transforms::eval::eval(arena, im_h1);

        let (re_h0, im_h0) = crate::base::complex::as_real_imag(arena, h0_at);
        let re_h0 = crate::transforms::eval::eval(arena, re_h0);
        let im_h0 = crate::transforms::eval::eval(arena, im_h0);

        // A(x) = Re(h1)·x + Re(h0)
        // B(x) = Im(h1)·x + Im(h0)
        tracing::trace!(
            pair_idx,
            "log_to_real: A(x) = Re(h1)·x + Re(h0), B(x) = Im(h1)·x + Im(h0)"
        );

        // ── Build ln(A² + B²) ──────────────────────────────────────
        //
        // The norm |h(α,x)|² = h(α,x)·h(ᾱ,x) is always a polynomial in x
        // with rational coefficients (the norm over the extension field).
        //
        // For h(t,x) = x + c(t):  |h|² = (x + c(α))(x + c(ᾱ))
        //   = x² + (c(α)+c(ᾱ))·x + c(α)·c(ᾱ)
        //   = x² + 2·Re(c(α))·x + |c(α)|²
        //
        // We compute this symbolically: A² + B² where A and B are arena
        // expressions, then expand + eval to simplify.
        let re_h1_x = arena.mul(&[re_h1, var]);
        let a_expr = arena.add(&[re_h1_x, re_h0]);
        let im_h1_x = arena.mul(&[im_h1, var]);
        let b_expr = arena.add(&[im_h1_x, im_h0]);

        let a_sq = arena.mul(&[a_expr, a_expr]);
        let b_sq = arena.mul(&[b_expr, b_expr]);
        let norm_sq = arena.add(&[a_sq, b_sq]);

        // Expand and eval to simplify: (x+1/2)² + (√3/2)² → x²+x+1
        let norm_sq = crate::transforms::expand::expand(arena, norm_sq);
        let norm_sq = crate::transforms::eval::eval(arena, norm_sq);

        // Apply radical simplification (powdenest + powsimp_base) to
        // clean up terms like (√3)² → 3.
        let norm_sq = crate::simplify::powsimp::powdenest(arena, norm_sq);
        let norm_sq = crate::simplify::powsimp::powsimp_base(arena, norm_sq);
        let norm_sq = crate::transforms::eval::eval(arena, norm_sq);

        tracing::trace!(pair_idx, "log_to_real: norm |h|² computed and simplified");

        // ln term: u_j · ln(|h|²)
        // ── Check if u_j = 0 (pure imaginary root) ────────────────
        let u_is_zero = crate::poly::algebraic::is_zero_checked(arena, *u_j).unwrap_or(false);

        // ── Check if B(x) = 0 (imaginary part vanishes) ───────────
        let im_h1_is_zero = crate::poly::algebraic::is_zero_checked(arena, im_h1).unwrap_or(false);
        let im_h0_is_zero = crate::poly::algebraic::is_zero_checked(arena, im_h0).unwrap_or(false);
        let b_is_zero = im_h1_is_zero && im_h0_is_zero;

        if u_is_zero && b_is_zero {
            // Both u ≈ 0 and B ≈ 0: this pair contributes nothing.
            // This happens for integrands like x/(x⁴+x²+1) where the
            // Rothstein-Trager roots are pure imaginary and h evaluates
            // to a real expression.  Return None to let the algebraic
            // remainder fallback handle this case (via apart).
            tracing::debug!(
                pair_idx,
                "log_to_real: u ≈ 0 and B ≈ 0 — pair contributes nothing, bailing"
            );
            return None;
        }

        // Build ln term: u_j · ln(|h|²)  (skip if u_j ≈ 0)
        if !u_is_zero {
            let ln_norm = arena.ln(norm_sq);
            let ln_term = arena.mul(&[*u_j, ln_norm]);
            result_terms.push(ln_term);
        }

        if b_is_zero {
            // B ≈ 0 but u ≠ 0: only the ln term contributes (already pushed above).
            tracing::debug!(pair_idx, "log_to_real: B ≈ 0, ln-only term (no atan)");
        } else {
            // ── Build atan via log_to_atan_deg1 ────────────────────────
            let atan_result = log_to_atan_deg1(arena, var, re_h1, re_h0, im_h1, im_h0);

            match atan_result {
                Some(atan_expr) => {
                    // atan term: v_j · log_to_atan(A, B)
                    // (log_to_atan already includes the factor of 2)
                    let atan_term = arena.mul(&[*v_j, atan_expr]);
                    result_terms.push(atan_term);
                    tracing::debug!(pair_idx, "log_to_real: emitted atan term");
                }
                None => {
                    tracing::debug!(
                        pair_idx,
                        "log_to_real: log_to_atan_deg1 failed, skipping pair"
                    );
                    return None;
                }
            }
        }
    }

    if result_terms.is_empty() {
        None
    } else {
        tracing::debug!(
            n_terms = result_terms.len(),
            "log_to_real: conversion complete"
        );
        Some(result_terms)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Vieta's formulas — symmetric functions of roots without root-finding
// ═══════════════════════════════════════════════════════════════════════════

/// Extract the elementary symmetric polynomials `e_k` from a polynomial's
/// coefficients using Vieta's formulas.
///
/// For a monic polynomial `t^n + a_{n-1}·t^{n-1} + ... + a_1·t + a_0`,
/// the elementary symmetric polynomials of its roots are:
///
/// ```text
/// e_1 = -a_{n-1}          (sum of roots)
/// e_2 =  a_{n-2}          (sum of products of pairs)
/// e_k = (-1)^k · a_{n-k}  (k-th elementary symmetric polynomial)
/// ```
///
/// Returns `e_1, e_2, ..., e_n` as `Ratio<BigInt>` values.
/// The input polynomial must be monic (leading coefficient = 1).
///
/// Returns `None` if the polynomial is zero, constant, or not monic.
pub(crate) fn vieta_elementary_symmetric(poly: &Poly) -> Option<Vec<Q>> {
    let n = poly.degree()?;
    if n == 0 {
        return None;
    }

    // Check monic (leading coefficient = 1).
    let lc = poly.leading_coeff()?;
    if !num_traits::One::is_one(lc) {
        tracing::trace!("vieta_elementary_symmetric: polynomial is not monic");
        return None;
    }

    let one: Q = Ratio::from_integer(BigInt::from(1));
    let neg_one: Q = Ratio::from_integer(BigInt::from(-1));

    let mut e = Vec::with_capacity(n);
    for k in 1..=n {
        // e_k = (-1)^k · a_{n-k}
        let coeff = poly.coeff(n - k);
        let sign = if k % 2 == 0 { &one } else { &neg_one };
        e.push(sign * &coeff);
    }

    Some(e)
}

/// Compute the power sum `p_k = Σ_{i=1}^{n} α_i^k` using Newton's identities,
/// given the elementary symmetric polynomials `e_1, ..., e_n`.
///
/// Newton's identities:
/// ```text
/// p_1 = e_1
/// p_2 = e_1·p_1 - 2·e_2
/// p_k = Σ_{i=1}^{k-1} (-1)^{i-1}·e_i·p_{k-i} + (-1)^{k-1}·k·e_k   (k ≤ n)
/// p_k = Σ_{i=1}^{n}   (-1)^{i-1}·e_i·p_{k-i}                        (k > n)
/// ```
///
/// Returns `p_1, p_2, ..., p_max_k`.
pub(crate) fn vieta_power_sums(elementary: &[Q], max_k: usize) -> Vec<Q> {
    let n = elementary.len(); // degree of the polynomial
    let one: Q = Ratio::from_integer(BigInt::from(1));
    let neg_one: Q = Ratio::from_integer(BigInt::from(-1));
    let mut p: Vec<Q> = Vec::with_capacity(max_k);

    for k in 1..=max_k {
        let mut pk = Ratio::from_integer(BigInt::from(0));

        let upper = if k <= n { k - 1 } else { n };
        for i in 1..=upper {
            // (-1)^{i-1} · e_i · p_{k-i}
            let sign = if (i - 1) % 2 == 0 { &one } else { &neg_one };
            let e_i = &elementary[i - 1];
            let p_km = if k - i >= 1 {
                &p[k - i - 1] // p_{k-i} (0-indexed)
            } else {
                // k - i == 0 → this shouldn't happen since i ≤ k-1
                continue;
            };
            pk += sign * e_i * p_km;
        }

        // For k ≤ n: add the (-1)^{k-1} · k · e_k term
        if k <= n {
            let sign = if (k - 1) % 2 == 0 { &one } else { &neg_one };
            let k_rat = Ratio::from_integer(BigInt::from(k));
            pk += sign * &k_rat * &elementary[k - 1];
        }

        p.push(pk);
    }

    p
}

/// Try to evaluate `RootSum(poly, body, sumvar)` when the body is a
/// polynomial in `sumvar` (no other variables), using Vieta's formulas
/// and Newton's identities.
///
/// For `body = c_m·t^m + ... + c_1·t + c_0`, the sum over all roots is:
/// ```text
/// Σ body(α_i) = c_m·p_m + ... + c_1·p_1 + n·c_0
/// ```
/// where `p_k = Σ α_i^k` (power sums) and `n` is the polynomial degree.
///
/// Returns `Some(rational_value)` if the body is a polynomial in sumvar
/// with rational coefficients and no other free variables.
/// Returns `None` otherwise.
pub(crate) fn vieta_rootsum_poly_body(
    arena: &Arena,
    poly_id: ExprId,
    body_id: ExprId,
    sumvar_id: ExprId,
) -> Option<Q> {
    // Extract the polynomial as Poly.
    let poly = crate::poly::polybridge::expr_to_poly(arena, poly_id, sumvar_id)?;
    let n = poly.degree()?;

    // Make monic for Vieta's formulas.
    let monic = poly.make_monic();

    // Extract body as Poly in sumvar.
    let body_poly = crate::poly::polybridge::expr_to_poly(arena, body_id, sumvar_id)?;
    let body_deg = body_poly.degree().unwrap_or(0);

    tracing::debug!(
        poly_degree = n,
        body_degree = body_deg,
        "vieta_rootsum_poly_body: attempting Vieta evaluation"
    );

    // Get elementary symmetric polynomials.
    let elementary = vieta_elementary_symmetric(&monic)?;

    // Compute power sums up to the body degree.
    let power_sums = vieta_power_sums(&elementary, body_deg);

    // Evaluate: Σ body(α_i) = Σ_k c_k · p_k + n · c_0
    let n_rat = Ratio::from_integer(BigInt::from(n));
    let mut result = &n_rat * body_poly.coeff(0); // n · c_0

    for k in 1..=body_deg {
        let c_k = body_poly.coeff(k);
        if !c_k.is_zero() && k <= power_sums.len() {
            result += &c_k * &power_sums[k - 1];
        }
    }

    tracing::debug!(%result, "vieta_rootsum_poly_body: computed via Vieta");
    Some(result)
}

// ═══════════════════════════════════════════════════════════════════════════
// rootsum_doit — expand RootSum when the polynomial is solvable
// ═══════════════════════════════════════════════════════════════════════════

/// Try to expand a `RootSum(poly, body, sumvar)` into an explicit sum
/// by finding the roots of the polynomial via [`solve`](crate::transforms::solve::solve).
///
/// For degree ≤ 4 polynomials, `solve` produces exact radical roots
/// (quadratic formula, Cardano, Ferrari).  Each root is substituted
/// into the body, and the results are summed.
///
/// Returns `Some(expanded_sum)` if all roots are found, `None` if
/// `solve` can't find them (degree ≥ 5 non-solvable).
///
/// # Example
///
/// ```text
/// RootSum(9t²+3t+1, t -> t·ln(x-3t))
///   → (-1/6+i√3/6)·ln(x-3(-1/6+i√3/6)) + (-1/6-i√3/6)·ln(x-3(-1/6-i√3/6))
/// ```
///
/// The result can then be simplified via `eval` / `as_real_imag` / etc.
pub(crate) fn rootsum_doit(
    arena: &mut Arena,
    poly_id: ExprId,
    body_id: ExprId,
    sumvar_id: ExprId,
) -> Option<ExprId> {
    // Solve the polynomial for roots.
    let roots = crate::transforms::solve::solve(arena, poly_id, sumvar_id);

    if roots.is_empty() {
        tracing::debug!("rootsum_doit: solve returned no roots, cannot expand");
        return None;
    }
    // Expanding over `RootOf` placeholders gains nothing and multiplies the
    // size by the degree: since 0.22 `solve` returns one `RootOf` per root of
    // an irreducible factor of degree ≥ 5, and `∫ atan(√x − x⁹) dx` became a
    // 486 KB sum of twelve copies of the body.  Keep the compact `RootSum`.
    let has_rootof = roots.iter().any(|r| {
        crate::base::walk::post_order_ids(arena, r.value)
            .into_iter()
            .any(|id| matches!(arena.node(id), ExprNode::RootOf(..)))
    });
    if has_rootof {
        tracing::debug!("rootsum_doit: roots are RootOf placeholders, keeping RootSum");
        return None;
    }

    // Check that we got the expected number of roots (= degree of poly).
    let poly_obj = crate::poly::polybridge::expr_to_poly(arena, poly_id, sumvar_id);
    if let Some(ref p) = poly_obj
        && let Some(deg) = p.degree()
        && roots.len() != deg
    {
        tracing::debug!(
            expected = deg,
            found = roots.len(),
            "rootsum_doit: solve found fewer roots than polynomial degree, cannot expand fully"
        );
        return None;
    }

    tracing::debug!(
        n_roots = roots.len(),
        "rootsum_doit: expanding RootSum by substituting each root"
    );

    // For each root α_k: substitute sumvar = α_k into body, eval.
    let mut terms: Vec<ExprId> = Vec::with_capacity(roots.len());
    for root in &roots {
        let substituted = crate::transforms::subs::subs(arena, body_id, sumvar_id, root.value);
        let evaluated = crate::transforms::eval::eval(arena, substituted);
        terms.push(evaluated);
    }

    // Sum all terms.
    let result = if terms.len() == 1 {
        terms[0]
    } else {
        arena.add(&terms)
    };

    tracing::debug!("rootsum_doit: expansion complete");
    Some(result)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Vieta tests ────────────────────────────────────────────────

    #[test]
    fn vieta_elementary_symmetric_quadratic() {
        // q(t) = t² + 3t + 2 = (t+1)(t+2)
        // Roots: -1, -2
        // e_1 = -(-1-2) = 3  (but Vieta: e_1 = -a_{n-1} = -3)
        // Wait: for t² + 3t + 2, a_1=3, a_0=2
        // e_1 = -a_1 = -3  (sum of roots = -1 + -2 = -3) ✓
        // e_2 = a_0 = 2   (product of roots = (-1)(-2) = 2) ✓
        let q = Poly::from_coeffs(vec![r(2, 1), r(3, 1), r(1, 1)]);
        let e = vieta_elementary_symmetric(&q).unwrap();
        assert_eq!(e.len(), 2);
        assert_eq!(e[0], r(-3, 1), "e_1 = sum of roots = -3");
        assert_eq!(e[1], r(2, 1), "e_2 = product of roots = 2");
    }

    #[test]
    fn vieta_power_sums_quadratic() {
        // Roots: -1, -2
        // p_1 = (-1) + (-2) = -3
        // p_2 = (-1)² + (-2)² = 1 + 4 = 5
        // p_3 = (-1)³ + (-2)³ = -1 + (-8) = -9
        let e = vec![r(-3, 1), r(2, 1)];
        let p = vieta_power_sums(&e, 3);
        assert_eq!(p.len(), 3);
        assert_eq!(p[0], r(-3, 1), "p_1 = -3");
        assert_eq!(p[1], r(5, 1), "p_2 = 5");
        assert_eq!(p[2], r(-9, 1), "p_3 = -9");
    }

    #[test]
    fn vieta_power_sums_cubic() {
        // q(t) = t³ - 6t² + 11t - 6 = (t-1)(t-2)(t-3)
        // Roots: 1, 2, 3
        // e_1 = 6, e_2 = 11, e_3 = 6
        let q = Poly::from_coeffs(vec![r(-6, 1), r(11, 1), r(-6, 1), r(1, 1)]);
        let e = vieta_elementary_symmetric(&q).unwrap();
        assert_eq!(e[0], r(6, 1), "e_1 = 1+2+3 = 6");
        assert_eq!(e[1], r(11, 1), "e_2 = 1·2+1·3+2·3 = 11");
        assert_eq!(e[2], r(6, 1), "e_3 = 1·2·3 = 6");

        // p_1 = 6, p_2 = 1+4+9 = 14, p_3 = 1+8+27 = 36
        let p = vieta_power_sums(&e, 3);
        assert_eq!(p[0], r(6, 1), "p_1 = 6");
        assert_eq!(p[1], r(14, 1), "p_2 = 14");
        assert_eq!(p[2], r(36, 1), "p_3 = 36");
    }

    #[test]
    fn vieta_rootsum_poly_body_sum_of_roots() {
        // RootSum(t²+3t+2, t -> t) = sum of roots = -3
        let mut arena = Arena::new();
        let t = arena.symbol("t");
        let two = arena.int(2);
        let three = arena.int(3);
        let t_sq = arena.pow(t, two);
        let poly_expr = {
            let three_t = arena.mul(&[three, t]);
            arena.add(&[t_sq, three_t, two])
        };

        let result = vieta_rootsum_poly_body(&arena, poly_expr, t, t);
        assert_eq!(
            result,
            Some(r(-3, 1)),
            "sum of roots of t²+3t+2 should be -3"
        );
    }

    #[test]
    fn vieta_rootsum_poly_body_sum_of_squares() {
        // RootSum(t²+3t+2, t -> t²) = sum of squares of roots = 5
        let mut arena = Arena::new();
        let t = arena.symbol("t");
        let two = arena.int(2);
        let three = arena.int(3);
        let t_sq = arena.pow(t, two);
        let poly_expr = {
            let three_t = arena.mul(&[three, t]);
            arena.add(&[t_sq, three_t, two])
        };
        let body = arena.pow(t, two); // t²

        let result = vieta_rootsum_poly_body(&arena, poly_expr, body, t);
        assert_eq!(
            result,
            Some(r(5, 1)),
            "sum of squares of roots of t²+3t+2 should be 5"
        );
    }

    #[test]
    fn vieta_rootsum_constant_body() {
        // RootSum(t²+3t+2, t -> 7) = 2 * 7 = 14 (n roots, each contributing 7)
        let mut arena = Arena::new();
        let t = arena.symbol("t");
        let two = arena.int(2);
        let three = arena.int(3);
        let seven = arena.int(7);
        let t_sq = arena.pow(t, two);
        let poly_expr = {
            let three_t = arena.mul(&[three, t]);
            arena.add(&[t_sq, three_t, two])
        };

        let result = vieta_rootsum_poly_body(&arena, poly_expr, seven, t);
        assert_eq!(
            result,
            Some(r(14, 1)),
            "RootSum with constant body 7 over degree-2 poly = 14"
        );
    }

    fn r(n: i64, d: i64) -> Q {
        Ratio::new(BigInt::from(n), BigInt::from(d))
    }

    fn sym(arena: &mut Arena, name: &str) -> ExprId {
        arena.symbol(name)
    }

    // ── poly_to_genpoly_rf tests ───────────────────────────────────

    #[test]
    fn poly_to_genpoly_rf_constant() {
        let p = Poly::from_int(5);
        let gp = poly_to_genpoly_rf(&p);
        assert_eq!(gp.degree(), Some(0));
        let c = gp.coeff(0);
        assert_eq!(c.to_rational(), Some(r(5, 1)));
    }

    #[test]
    fn poly_to_genpoly_rf_linear() {
        // p(x) = 3x + 2
        let p = Poly::from_coeffs(vec![r(2, 1), r(3, 1)]);
        let gp = poly_to_genpoly_rf(&p);
        assert_eq!(gp.degree(), Some(1));
        assert_eq!(gp.coeff(0).to_rational(), Some(r(2, 1)));
        assert_eq!(gp.coeff(1).to_rational(), Some(r(3, 1)));
    }

    #[test]
    fn poly_to_genpoly_rf_times_t_linear() {
        // p(x) = 3x + 2 → 3t·x + 2t
        let p = Poly::from_coeffs(vec![r(2, 1), r(3, 1)]);
        let gp = poly_to_genpoly_rf_times_t(&p);
        assert_eq!(gp.degree(), Some(1));
        // coeff(0) = 2t → numer should be 2t polynomial
        let c0 = gp.coeff(0);
        assert_eq!(c0.numer().degree(), Some(1));
        assert_eq!(c0.numer().coeff(0), r(0, 1));
        assert_eq!(c0.numer().coeff(1), r(2, 1));
    }

    // ── log_to_atan_deg1 tests ─────────────────────────────────────

    #[test]
    fn log_to_atan_constant_b() {
        // A(x) = x + 1/2, B(x) = -√3/2
        // → 2·atan((x + 1/2) / (-√3/2)) = 2·atan(-(2x+1)/√3)
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let a1 = arena.one;
        let a0 = arena.rational(1, 2);
        let b1 = arena.zero;
        let three = arena.int(3);
        let half = arena.rational(1, 2);
        let sqrt3 = arena.pow(three, half);
        let half2 = arena.rational(1, 2);
        let half_sqrt3 = arena.mul(&[half2, sqrt3]);
        let b0 = arena.neg(half_sqrt3);

        let result = log_to_atan_deg1(&mut arena, x, a1, a0, b1, b0);
        assert!(result.is_some(), "should produce atan for constant B");

        // Verify numerically at x = 1
        let one = arena.int(1);
        let at_1 = crate::transforms::subs::subs(&mut arena, result.unwrap(), x, one);
        let at_1 = crate::transforms::eval::eval(&mut arena, at_1);
        let val = crate::transforms::evalf::eval_const_f64(&mut arena, at_1);
        assert!(val.is_some(), "should evaluate to f64");

        // Expected: 2·atan((1 + 0.5) / (-√3/2)) = 2·atan(1.5 / (-0.866)) = 2·atan(-1.7321)
        let expected = 2.0 * (1.5_f64 / (-(3.0_f64.sqrt()) / 2.0)).atan();
        let v = val.unwrap();
        assert!(
            (v - expected).abs() < 1e-8,
            "log_to_atan at x=1: got {v}, expected {expected}"
        );
    }

    #[test]
    fn log_to_atan_degenerate_b_zero() {
        // B(x) = 0 → should return None
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let one = arena.one;
        let zero = arena.zero;
        let result = log_to_atan_deg1(&mut arena, x, one, zero, zero, zero);
        assert!(result.is_none(), "B = 0 should be degenerate");
    }

    // ── log_to_real tests ──────────────────────────────────────────

    #[test]
    fn log_to_real_x3_minus_1_quadratic_factor() {
        // For 1/(x³-1), the quadratic factor of R(t) is q(t) = 9t²+3t+1.
        // The PRS degree-1 member is h(t,x) = x - 3t.
        //
        // log_to_real should produce:
        //   u·ln(x²+x+1) + v·(2·atan(...))
        // where u = -1/6 and v = √3/6.
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");

        // q(t) = 9t² + 3t + 1
        let q = Poly::from_coeffs(vec![r(1, 1), r(3, 1), r(9, 1)]);

        // h(t,x) = x - 3t as GenPoly<RationalFn>
        let neg_3t = RationalFn::from_poly(Poly::from_coeffs(vec![r(0, 1), r(-3, 1)]));
        let h_prs: GenPoly<RationalFn> =
            GenPoly::from_coeffs(vec![neg_3t, RationalFn::from_rational(r(1, 1))]);

        let terms = log_to_real(&mut arena, x, &q, &h_prs);
        assert!(
            terms.is_some(),
            "log_to_real should succeed for x³-1 quadratic factor"
        );

        let terms = terms.unwrap();
        assert!(
            terms.len() >= 2,
            "should produce at least 2 terms (ln + atan)"
        );

        // Sum all terms and evaluate the definite integral [2, 3].
        let sum = arena.add(&terms);
        let val_3 = arena.int(3);
        let f3 = crate::transforms::subs::subs(&mut arena, sum, x, val_3);
        let f3 = crate::transforms::eval::eval(&mut arena, f3);
        let val_2 = arena.int(2);
        let f2 = crate::transforms::subs::subs(&mut arena, sum, x, val_2);
        let f2 = crate::transforms::eval::eval(&mut arena, f2);

        let f3_f64 = crate::transforms::evalf::eval_const_f64(&mut arena, f3);
        let f2_f64 = crate::transforms::evalf::eval_const_f64(&mut arena, f2);

        if let (Some(f3v), Some(f2v)) = (f3_f64, f2_f64) {
            let integral = f3v - f2v;
            // This is ONLY the quadratic-factor contribution.
            // Full answer = 1/3·ln|x-1| + (this).
            // Ground truth for the quadratic part on [2,3]: ≈ -0.1557
            let expected = {
                let gt_full = 0.07539_f64;
                let gt_linear = (1.0 / 3.0) * (2.0_f64.ln()); // 1/3 · ln(2)
                gt_full - gt_linear
            };
            assert!(
                (integral - expected).abs() < 0.01,
                "quadratic factor ∫₂³: got {integral}, expected ≈ {expected}"
            );
        } else {
            panic!("could not evaluate log_to_real result to f64");
        }
    }

    /// `f(x0)` as an `f64`.
    fn value_at(arena: &mut Arena, f: ExprId, x: ExprId, x0: Q) -> f64 {
        let x0 = arena.num_ratio(x0);
        let v = crate::transforms::subs::subs(arena, f, x, x0);
        let v = crate::transforms::eval::eval(arena, v);
        crate::transforms::evalf::eval_const_f64(arena, v).expect("numeric value")
    }

    fn poly_q(coeffs: &[(i64, i64)]) -> Poly {
        Poly::from_coeffs(coeffs.iter().map(|&(n, d)| r(n, d)).collect())
    }

    fn ratfn_poly(coeffs: &[(i64, i64)]) -> RationalFn {
        RationalFn::from_poly(poly_q(coeffs))
    }

    #[test]
    fn quadratic_factor_with_a_degree_4_log_argument() {
        // Σ_{64α²+1=0} α·ln(x⁴ + 8α) = atan(x⁴)/4 (the log part of
        // ∫ x³/(x⁸ + 1) dx; SymPy 1.14 `integrate(x**3/(x**8+1), x)`).
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let q = poly_q(&[(1, 1), (0, 1), (64, 1)]);
        let zero = ratfn_poly(&[]);
        let s: GenPoly<RationalFn> = GenPoly::from_coeffs(vec![
            ratfn_poly(&[(0, 1), (8, 1)]),
            zero.clone(),
            zero.clone(),
            zero,
            ratfn_poly(&[(1, 1)]),
        ]);
        let terms = quadratic_log_to_real(&mut arena, x, &q, &s).expect("real form");
        let sum = arena.add(&terms);
        assert_eq!(arena.display(sum).to_string(), "1/4*atan(x^4)");
    }

    #[test]
    fn quadratic_factor_with_real_roots() {
        // Σ_{8α²−1=0} α·ln(x − 4α) = (√2/4)·(ln|x − √2| − ln|x + √2|), the
        // log part of ∫ 1/(x² − 2) dx (SymPy 1.14 `integrate(1/(x**2-2), x)`).
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let q = poly_q(&[(-1, 1), (0, 1), (8, 1)]);
        let s: GenPoly<RationalFn> =
            GenPoly::from_coeffs(vec![ratfn_poly(&[(0, 1), (-4, 1)]), ratfn_poly(&[(1, 1)])]);
        let terms = quadratic_log_to_real(&mut arena, x, &q, &s).expect("real form");
        let sum = arena.add(&terms);
        for x0 in [r(3, 1), r(1, 3), r(-7, 2)] {
            let got = value_at(&mut arena, sum, x, x0.clone());
            let xf = num_traits::ToPrimitive::to_f64(&x0).expect("f64");
            let s2 = 2f64.sqrt();
            let want = s2 / 4.0 * ((xf - s2).abs().ln() - (xf + s2).abs().ln());
            assert!((got - want).abs() < 1e-12, "x = {xf}: {got} vs {want}");
        }
    }

    #[test]
    fn log_to_atan_surd_satisfies_its_defining_derivative() {
        // F' = 2(A'·rB − A·rB')/(A² + r²B²) for F = LogToAtan(A, rB), here with
        // A = x³ + 2x, B = x² − 3, r = √3: B ∤ A, so the Bézout step and the
        // recursion both run.
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let a = poly_q(&[(0, 1), (2, 1), (0, 1), (1, 1)]);
        let b = poly_q(&[(-3, 1), (0, 1), (1, 1)]);
        let r2 = r(3, 1);
        let f = log_to_atan_surd(&mut arena, x, &a, &b, &r2).expect("atan sum");
        let df = crate::transforms::diff::diff(&mut arena, f, x);
        for x0 in [r(1, 2), r(5, 3), r(-9, 4)] {
            let got = value_at(&mut arena, df, x, x0.clone());
            let xf = num_traits::ToPrimitive::to_f64(&x0).expect("f64");
            let rt = 3f64.sqrt();
            let (av, dav) = (xf.powi(3) + 2.0 * xf, 3.0 * xf * xf + 2.0);
            let (bv, dbv) = (rt * (xf * xf - 3.0), rt * 2.0 * xf);
            let want = 2.0 * (dav * bv - av * dbv) / (av * av + bv * bv);
            assert!((got - want).abs() < 1e-10, "x = {xf}: {got} vs {want}");
        }
    }

    #[test]
    fn log_to_real_keeps_the_real_roots_of_a_cubic() {
        // ∫ 1/(x³ − 2) dx: q = 108t³ − 1 has the real root ∛2/6, whose term
        // (∛2/6)·ln|x − ∛2| was dropped before 0.25.  Checked through the
        // whole integrator: F' = f at three points.
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let three = arena.int(3);
        let x3 = arena.pow(x, three);
        let m2 = arena.int(-2);
        let den = arena.add(&[x3, m2]);
        let m1 = arena.int(-1);
        let f = arena.pow(den, m1);
        let big_f =
            crate::calculus::risch::try_risch_rational(&mut arena, f, x).expect("closed form");
        assert!(!crate::base::walk::has_unevaluated(&arena, big_f));
        let df = crate::transforms::diff::diff(&mut arena, big_f, x);
        for x0 in [r(3, 1), r(1, 2), r(-5, 2)] {
            let got = value_at(&mut arena, df, x, x0.clone());
            let xf = num_traits::ToPrimitive::to_f64(&x0).expect("f64");
            let want = 1.0 / (xf.powi(3) - 2.0);
            assert!((got - want).abs() < 1e-10, "x = {xf}: {got} vs {want}");
        }
    }

    #[test]
    fn log_to_real_degree_less_than_2_returns_none() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");

        // Linear q(t) = 3t - 1 → should return None
        let q = Poly::from_coeffs(vec![r(-1, 1), r(3, 1)]);
        let h_prs: GenPoly<RationalFn> = GenPoly::from_coeffs(vec![
            RationalFn::from_rational(r(0, 1)),
            RationalFn::from_rational(r(1, 1)),
        ]);

        let result = log_to_real(&mut a, x, &q, &h_prs);
        assert!(result.is_none(), "degree-1 q should return None");
    }
}
