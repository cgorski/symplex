//! Aberth's method for finding all roots (real and complex) of a polynomial.
//!
//! Given a polynomial `p(x) = a_n x^n + ... + a_1 x + a_0` with rational
//! coefficients, Aberth's method simultaneously converges to all `n` roots
//! using cubic-order iteration.
//!
//! # Algorithm
//!
//! Starting from `n` initial guesses distributed on a circle, the method
//! iterates:
//!
//! ```text
//! w_k = p(z_k) / (p'(z_k) - p(z_k) * Σ_{j≠k} 1/(z_k - z_j))
//! z_k ← z_k - w_k
//! ```
//!
//! until all corrections `|w_k|` are below the desired precision.

use astro_float::{BigFloat, Consts, RoundingMode};
use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Signed, Zero};

use super::dense::Poly;
use crate::transforms::evalf::{c_add, c_div, c_from_real, c_mul, c_one, c_sub, c_zero};

/// A complex number as `(real, imaginary)` pair of arbitrary-precision floats.
type Complex = (BigFloat, BigFloat);

/// Evaluate a polynomial with rational coefficients at a complex point
/// using Horner's method.
///
/// Coefficients are in ascending degree order: `[a_0, a_1, ..., a_n]`.
fn poly_eval_complex(
    coeffs: &[Ratio<BigInt>],
    z: &Complex,
    prec: usize,
    rm: RoundingMode,
) -> Complex {
    if coeffs.is_empty() {
        return c_zero(prec);
    }
    let mut result = c_zero(prec);
    for c in coeffs.iter().rev() {
        result = c_mul(&result, z, prec, rm);
        let c_re = ratio_to_bigfloat(c, prec);
        let c_complex = c_from_real(c_re, prec);
        result = c_add(&result, &c_complex, prec, rm);
    }
    result
}

/// Convert a `Ratio<BigInt>` to a `BigFloat` at the given precision.
fn ratio_to_bigfloat(r: &Ratio<BigInt>, prec: usize) -> BigFloat {
    let numer_f = bigint_to_bigfloat(r.numer(), prec);
    let denom_f = bigint_to_bigfloat(r.denom(), prec);
    if denom_f.is_zero() {
        return BigFloat::new(prec);
    }
    numer_f.div(&denom_f, prec, RoundingMode::None)
}

/// Convert a `BigInt` to a `BigFloat`.
fn bigint_to_bigfloat(n: &BigInt, prec: usize) -> BigFloat {
    // For small integers, use direct conversion
    if let Ok(small) = i64::try_from(n) {
        return BigFloat::from_i64(small, prec);
    }
    // For large integers, convert via i128 or fall back to f64
    if let Ok(medium) = i128::try_from(n) {
        return BigFloat::from_i128(medium, prec);
    }
    // Last resort: lose precision via f64 (only for very large BigInts)
    let f = n.to_string().parse::<f64>().unwrap_or(f64::NAN);
    BigFloat::from_f64(f, prec)
}

/// Compute Cauchy's upper bound on the absolute value of all roots.
///
/// For `p(x) = a_n x^n + ... + a_0`, all roots satisfy
/// `|z| ≤ 1 + max(|a_i / a_n|)` for `i = 0..n-1`.
fn cauchy_bound(poly: &Poly, prec: usize) -> BigFloat {
    let rm = RoundingMode::None;
    let lc = poly.leading_coeff().cloned().unwrap_or_else(Ratio::one);
    if lc.is_zero() {
        return BigFloat::from_i32(1, prec);
    }
    let mut max_ratio = BigFloat::from_i32(0, prec);
    for c in poly
        .coeffs()
        .iter()
        .take(poly.coeffs().len().saturating_sub(1))
    {
        let ratio = c / &lc;
        let abs_ratio = if ratio.is_negative() { -ratio } else { ratio };
        let bf = ratio_to_bigfloat(&abs_ratio, prec);
        if bf.sub(&max_ratio, prec, rm).is_positive() {
            max_ratio = bf;
        }
    }
    max_ratio.add(&BigFloat::from_i32(1, prec), prec, rm)
}

/// Generate initial root approximations distributed on a circle.
///
/// Uses the classic Aberth initialization
/// `z_k = center + radius · exp(i(2πk/n + θ))` where
/// `center = -a_{n-1} / (n · a_n)` and `radius` is the Cauchy bound.
///
/// The phase offset `θ = π/(2n) + 0.4` is deliberately *not* a rational
/// multiple of `π/n`: the textbook choice `θ = π/(2n)` places the guesses
/// mirror-symmetrically about the imaginary axis for odd `n`, and the
/// Aberth iteration preserves that symmetry for polynomials with real
/// coefficients that are odd or even functions (`x³ + x`).  The paired
/// guesses can then never split to the distinct self-symmetric roots `0`
/// and `i`, and the iteration stalls without ever converging.  The
/// irrational-looking offset breaks every such symmetry (this is the
/// same trick MPSolve uses).
fn initial_guesses(poly: &Poly, n: usize, prec: usize, cc: &mut Consts) -> Vec<Complex> {
    let rm = RoundingMode::None;
    let radius = cauchy_bound(poly, prec);

    // Center: -a_{n-1} / (n * a_n) — shifts initial guesses toward the centroid of roots
    let center = if n >= 2 && poly.coeffs().len() > n {
        let an = &poly.coeffs()[n];
        let an1 = &poly.coeffs()[n - 1];
        if !an.is_zero() {
            let ratio = -(an1 / an) / Ratio::from_integer(BigInt::from(n));
            ratio_to_bigfloat(&ratio, prec)
        } else {
            BigFloat::new(prec)
        }
    } else {
        BigFloat::new(prec)
    };

    let two_pi = cc.pi(prec, rm).mul(&BigFloat::from_i32(2, prec), prec, rm);
    let n_bf = BigFloat::from_i64(n as i64, prec);
    let quarter = BigFloat::from_f64(0.25, prec);
    let offset = BigFloat::from_f64(0.4, prec);

    (0..n)
        .map(|k| {
            // angle = 2π * (k + 1/4) / n + 0.4
            let k_bf = BigFloat::from_i64(k as i64, prec);
            let frac = k_bf.add(&quarter, prec, rm).div(&n_bf, prec, rm);
            let angle = two_pi.mul(&frac, prec, rm).add(&offset, prec, rm);

            let cos_a = angle.cos(prec, rm, cc);
            let sin_a = angle.sin(prec, rm, cc);

            let re = center.add(&radius.mul(&cos_a, prec, rm), prec, rm);
            let im = radius.mul(&sin_a, prec, rm);

            (re, im)
        })
        .collect()
}

/// Find all roots of a polynomial using Aberth's method.
///
/// Returns `n` complex roots sorted by (real part, imaginary part) for
/// deterministic, stable indexing.
///
/// # Arguments
///
/// * `poly` — The polynomial (coefficients in ascending degree order).
/// * `prec` — Working precision in bits.
/// * `max_iter` — Maximum number of Aberth iterations.
///
/// # Returns
///
/// A vector of `n` complex roots as `(BigFloat, BigFloat)` pairs, sorted
/// by real part (then imaginary part for ties).  Should astro-float fail to
/// allocate its constant caches, only the exact zero roots are returned;
/// callers already treat a short vector as "fewer roots than expected".
pub(crate) fn aberth_roots(poly: &Poly, prec: usize, max_iter: usize) -> Vec<Complex> {
    if poly.degree().is_none_or(|d| d == 0) {
        return vec![];
    }

    let rm = RoundingMode::None;
    let wp = prec + 64; // working precision with guard bits

    // Roots at exactly zero are read off the coefficients: `x^k · q(x)` with
    // `q(0) ≠ 0`.  They are returned as exact zeros and the iteration only
    // sees `q`, whose roots are all nonzero.
    let zero_mult = poly.coeffs().iter().take_while(|c| c.is_zero()).count();
    let mut roots: Vec<Complex> = (0..zero_mult).map(|_| c_zero(wp)).collect();
    let reduced = if zero_mult > 0 {
        Poly::from_coeffs(poly.coeffs()[zero_mult..].to_vec())
    } else {
        poly.clone()
    };
    let n = match reduced.degree() {
        Some(d) if d >= 1 => d,
        _ => return roots,
    };

    // Normalize to monic
    let monic = reduced.make_monic();
    let deriv = monic.derivative();

    let mut cc = match Consts::new() {
        Ok(cc) => cc,
        Err(e) => {
            tracing::warn!(error = ?e, "aberth_roots: astro-float constants init failed");
            return roots;
        }
    };

    let mut nonzero = aberth_iterate(&monic, &deriv, n, wp, prec, max_iter, rm, &mut cc);
    roots.append(&mut nonzero);

    // Sort by (real part, imaginary part) for stable indexing
    roots.sort_by(|a, b| {
        let re_cmp = a.0.cmp(&b.0).unwrap_or(0);
        if re_cmp < 0 {
            std::cmp::Ordering::Less
        } else if re_cmp > 0 {
            std::cmp::Ordering::Greater
        } else {
            let im_cmp = a.1.cmp(&b.1).unwrap_or(0);
            if im_cmp < 0 {
                std::cmp::Ordering::Less
            } else if im_cmp > 0 {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        }
    });

    roots
}

/// The Aberth–Ehrlich iteration proper, for a monic polynomial of degree
/// `n ≥ 1` with `p(0) ≠ 0`.  Returns the `n` (unsorted) roots.
#[allow(clippy::too_many_arguments)]
fn aberth_iterate(
    monic: &Poly,
    deriv: &Poly,
    n: usize,
    wp: usize,
    prec: usize,
    max_iter: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Vec<Complex> {
    let mut roots = initial_guesses(monic, n, wp, cc);

    // Convergence threshold: ~10^{-30} (good enough for f64 output)
    let threshold = BigFloat::from_f64(1e-30, wp);

    for _iter in 0..max_iter {
        let mut max_correction = BigFloat::new(wp);
        let mut corrections: Vec<Complex> = Vec::with_capacity(n);

        for i in 0..n {
            // p(z_i)
            let p_zi = poly_eval_complex(monic.coeffs(), &roots[i], wp, rm);

            // p'(z_i)
            let pp_zi = poly_eval_complex(deriv.coeffs(), &roots[i], wp, rm);

            // Σ_{j≠i} 1/(z_i - z_j)
            let mut sum_recip = c_zero(wp);
            for j in 0..n {
                if j != i {
                    let diff = c_sub(&roots[i], &roots[j], wp, rm);
                    // Avoid division by zero for near-coincident roots
                    let diff_abs_sq =
                        diff.0
                            .mul(&diff.0, wp, rm)
                            .add(&diff.1.mul(&diff.1, wp, rm), wp, rm);
                    if diff_abs_sq.is_zero() {
                        continue;
                    }
                    let recip = c_div(&c_one(wp), &diff, wp, rm);
                    sum_recip = c_add(&sum_recip, &recip, wp, rm);
                }
            }

            // denom = p'(z_i) - p(z_i) * Σ 1/(z_i - z_j)
            let pz_sum = c_mul(&p_zi, &sum_recip, wp, rm);
            let denom = c_sub(&pp_zi, &pz_sum, wp, rm);

            // w_i = p(z_i) / denom
            let denom_abs_sq =
                denom
                    .0
                    .mul(&denom.0, wp, rm)
                    .add(&denom.1.mul(&denom.1, wp, rm), wp, rm);
            let correction = if denom_abs_sq.is_zero() {
                // Degenerate: skip this root
                c_zero(wp)
            } else {
                c_div(&p_zi, &denom, wp, rm)
            };

            // Track maximum correction magnitude
            let corr_abs_sq = correction.0.mul(&correction.0, wp, rm).add(
                &correction.1.mul(&correction.1, wp, rm),
                wp,
                rm,
            );
            if corr_abs_sq.sub(&max_correction, prec, rm).is_positive() {
                max_correction = corr_abs_sq;
            }

            corrections.push(correction);
        }

        // Apply corrections simultaneously
        for i in 0..n {
            roots[i] = c_sub(&roots[i], &corrections[i], wp, rm);
        }

        // Check convergence: max |correction|^2 < threshold^2
        let threshold_sq = threshold.mul(&threshold, wp, rm);
        if max_correction.is_zero() || !max_correction.sub(&threshold_sq, prec, rm).is_positive() {
            break;
        }
    }

    roots
}

// ── `RootOf` indexing ───────────────────────────────────────────────────────────────

/// Binary precision (before the guard bits [`aberth_roots`] adds) at which
/// a `RootOf` node is evaluated by the default numeric path: `evalf` at
/// 16 digits, i.e. `eval_f64` / `eval_complex64`.
pub(crate) const ROOTOF_DEFAULT_PREC: usize = 128;

/// Every complex root of `poly` as the `RootOf` evaluator computes them:
/// [`aberth_roots`] at `prec + 64` bits with 200 iterations, sorted by
/// (real part, imaginary part).  `RootOf(poly, k)` *means* the `k`-th
/// entry of this list; [`real_root_index`] derives `k` from the same
/// call so that the constructor of a `RootOf` node and its evaluator can
/// never disagree on the order.
pub(crate) fn rootof_roots(poly: &Poly, prec: usize) -> Vec<Complex> {
    aberth_roots(poly, prec + 64, 200)
}

/// Relative distance below which two computed real parts are treated as
/// tied — their (re, im) order would then depend on rounding — and below
/// which a computed imaginary part counts as noise on a real root.  Aberth
/// stops at corrections around `10⁻³⁰`, so `2⁻⁶⁰ ≈ 10⁻¹⁸` is far above the
/// noise and far below any separation the sort could meaningfully resolve.
const ROOTOF_TIE_BITS: usize = 60;

/// The index `k` for which `RootOf(g, k)` denotes the real root of the
/// square-free polynomial `g` that lies in the (Sturm) isolating interval
/// `[lo, hi]` — its position in [`rootof_roots`]`(g, ROOTOF_DEFAULT_PREC)`.
///
/// The root is verified, not assumed: exactly one computed root must have
/// its real part in `[lo, hi]` and a negligible imaginary part, and no
/// other root may have a real part within `2⁻⁶⁰` (relative) of it, since
/// the (re, im) sort would then order the two by rounding noise and the
/// index would not be stable across evaluation precisions.  `None` when
/// any of these fails; the caller then has no reliable name for the root.
pub(crate) fn real_root_index(g: &Poly, lo: &Ratio<BigInt>, hi: &Ratio<BigInt>) -> Option<usize> {
    let roots = rootof_roots(g, ROOTOF_DEFAULT_PREC);
    let wp = ROOTOF_DEFAULT_PREC + 128;
    let rm = RoundingMode::None;
    let one = BigFloat::from_i32(1, wp);
    let tie = BigFloat::from_i32(2, wp)
        .powi(ROOTOF_TIE_BITS, wp, rm)
        .reciprocal(wp, rm);
    // `[lo, hi]` widened by the rounding of its endpoints to `wp` bits.
    let lo_bf = ratio_to_bigfloat(lo, wp);
    let hi_bf = ratio_to_bigfloat(hi, wp);
    let slack = hi_bf.abs().max(&lo_bf.abs()).max(&one).mul(
        &BigFloat::from_i32(2, wp)
            .powi(wp - 16, wp, rm)
            .reciprocal(wp, rm),
        wp,
        rm,
    );
    let lo_bf = lo_bf.sub(&slack, wp, rm);
    let hi_bf = hi_bf.add(&slack, wp, rm);

    let mut found: Option<usize> = None;
    for (k, (re, im)) in roots.iter().enumerate() {
        if re.is_nan() || im.is_nan() {
            return None;
        }
        if re.cmp(&lo_bf).unwrap_or(0) < 0 || re.cmp(&hi_bf).unwrap_or(0) > 0 {
            continue;
        }
        let scale = re.abs().max(&one);
        let noise = scale.mul(&tie, wp, rm);
        if im.abs().cmp(&noise).unwrap_or(1) > 0 {
            // A complex root whose real part happens to fall in the interval.
            continue;
        }
        if found.is_some() {
            return None;
        }
        found = Some(k);
    }
    let k = found?;
    let re_k = &roots[k].0;
    let noise = re_k.abs().max(&one).mul(&tie, wp, rm);
    let tied = roots
        .iter()
        .enumerate()
        .any(|(j, (re, _))| j != k && re.sub(re_k, wp, rm).abs().cmp(&noise).unwrap_or(-1) <= 0);
    if tied {
        tracing::debug!(
            index = k,
            "real_root_index: another root shares the real part; RootOf index not stable"
        );
        return None;
    }
    Some(k)
}

/// Convenience: evaluate the `index`-th root of a polynomial as a complex
/// `(f64, f64)` pair.
///
/// Returns `None` if the index is out of range.
#[allow(dead_code)] // Used by tests; will be wired to evalf in a future PR
pub(crate) fn rootof_eval_f64(poly: &Poly, index: usize) -> Option<(f64, f64)> {
    let n = poly.degree()?;
    if index >= n {
        return None;
    }

    // Use 128 bits of working precision for f64 output
    let roots = aberth_roots(poly, 128, 100);
    if index >= roots.len() {
        return None;
    }

    let (re, im) = &roots[index];
    let re_f64 = bigfloat_to_f64(re);
    let im_f64 = bigfloat_to_f64(im);
    Some((re_f64, im_f64))
}

/// Relative size below which a computed imaginary part is treated as a
/// *candidate* for being numerical noise on a real root.  The decision
/// itself is made exactly (see [`is_real_root_near`]); this only avoids
/// building a Sturm chain for clearly complex roots.
const REAL_AXIS_NOISE: f64 = 1e-6;

/// Does the square-free polynomial `part` have a real root within a tiny
/// interval around `z.0`?  Decided exactly with a Sturm count over
/// `[re − ε, re + ε]`, `ε = 2⁻³⁰ · max(1, |re|)`, on exact rationals.
///
/// Returns `false` immediately when `|im|` is not small relative to the
/// root, so the chain is only built when a root actually looks real.  The
/// chain is built lazily and cached in `sturm` across calls.
fn is_real_root_near(
    part: &Poly,
    z: (f64, f64),
    sturm: &mut Option<super::sturm::SturmChain>,
) -> bool {
    let (re, im) = z;
    if !re.is_finite() || !im.is_finite() {
        return false;
    }
    let scale = re.abs().max(1.0);
    if im.abs() > REAL_AXIS_NOISE * scale {
        return false;
    }
    let Some(center) = crate::base::numeric::f64_to_ratio_exact(re) else {
        return false;
    };
    let Some(eps) = crate::base::numeric::f64_to_ratio_exact(scale * 2f64.powi(-30)) else {
        return false;
    };
    let chain = sturm.get_or_insert_with(|| super::sturm::SturmChain::new(part));
    let lo = &center - &eps;
    let hi = &center + &eps;
    chain.count_roots_in_closed(&lo, &hi) >= 1
}

/// Convert a `BigFloat` to `f64` (best-effort).
fn bigfloat_to_f64(bf: &BigFloat) -> f64 {
    // Try direct conversion via the Display trait
    let s = format!("{}", bf);
    s.parse::<f64>().unwrap_or(f64::NAN)
}

/// All complex roots of `poly` as `(re, im)` pairs in `f64`, with
/// multiplicities (a `k`-fold root appears `k` times).
///
/// The polynomial is first split into square-free parts (Yun) so that
/// Aberth's method only ever sees simple roots, where it converges
/// cubically; each root is then replicated according to its multiplicity.
/// `prec_bits` is the working precision handed to [`aberth_roots`]
/// (at least 128 is recommended for full `f64` accuracy).
///
/// Real roots are returned with `im == 0.0` *exactly*.  A root whose
/// computed imaginary part is at the noise floor is snapped onto the real
/// axis only after an exact check: the square-free part must have a real
/// root in a tiny rational interval around the computed real part (Sturm
/// count).  Genuinely complex roots with a small imaginary part therefore
/// keep it, and the number of returned real roots always agrees with
/// [`SturmChain::count_real_roots`](super::sturm::SturmChain::count_real_roots).
///
/// Roots are sorted by real part, then imaginary part.  Constants and the
/// zero polynomial produce an empty vector.
pub(crate) fn nroots_f64(poly: &Poly, prec_bits: usize) -> Vec<(f64, f64)> {
    let mut out: Vec<(f64, f64)> = Vec::new();
    if poly.degree().unwrap_or(0) == 0 {
        return out;
    }
    let (_content, parts) = poly.sqf_list();
    for (part, mult) in parts {
        if part.degree().unwrap_or(0) == 0 {
            continue;
        }
        let max_iter = 100 + 20 * part.degree().unwrap_or(0);
        let roots = aberth_roots(&part, prec_bits, max_iter);
        let mut sturm: Option<super::sturm::SturmChain> = None;
        for (re, im) in roots {
            let mut pair = (bigfloat_to_f64(&re), bigfloat_to_f64(&im));
            if pair.1 != 0.0 && is_real_root_near(&part, pair, &mut sturm) {
                pair.1 = 0.0;
            }
            for _ in 0..mult {
                out.push(pair);
            }
        }
    }
    out.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn poly_from_coeffs(coeffs: &[i64]) -> Poly {
        let rat_coeffs: Vec<Ratio<BigInt>> = coeffs
            .iter()
            .map(|&c| Ratio::from_integer(BigInt::from(c)))
            .collect();
        Poly::from_coeffs(rat_coeffs)
    }

    #[test]
    fn aberth_quadratic_real_roots() {
        // x^2 - 5x + 6 = (x-2)(x-3)
        let poly = poly_from_coeffs(&[6, -5, 1]);
        let roots = aberth_roots(&poly, 128, 100);
        assert_eq!(roots.len(), 2);

        let mut real_parts: Vec<f64> = roots.iter().map(|r| bigfloat_to_f64(&r.0)).collect();
        real_parts.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert!(
            (real_parts[0] - 2.0).abs() < 1e-10,
            "root 0: {}",
            real_parts[0]
        );
        assert!(
            (real_parts[1] - 3.0).abs() < 1e-10,
            "root 1: {}",
            real_parts[1]
        );
    }

    #[test]
    fn aberth_quadratic_complex_roots() {
        // x^2 + 1 = 0  → roots ±i
        let poly = poly_from_coeffs(&[1, 0, 1]);
        let roots = aberth_roots(&poly, 128, 100);
        assert_eq!(roots.len(), 2);

        // Both should have real part ≈ 0 and imaginary parts ≈ ±1
        for root in &roots {
            let re = bigfloat_to_f64(&root.0);
            let im = bigfloat_to_f64(&root.1);
            assert!(re.abs() < 1e-10, "real part should be ~0: {re}");
            assert!((im.abs() - 1.0).abs() < 1e-10, "|im| should be ~1: {im}");
        }
    }

    #[test]
    fn aberth_quintic() {
        // x^5 - x - 1 = 0 — one real root ≈ 1.1673, four complex
        let poly = poly_from_coeffs(&[-1, -1, 0, 0, 0, 1]);
        let roots = aberth_roots(&poly, 128, 200);
        assert_eq!(roots.len(), 5);

        // Verify each root satisfies the polynomial
        for (i, root) in roots.iter().enumerate() {
            let val = poly_eval_complex(poly.coeffs(), root, 128, RoundingMode::None);
            let mag_sq = bigfloat_to_f64(&val.0).powi(2) + bigfloat_to_f64(&val.1).powi(2);
            assert!(
                mag_sq < 1e-15,
                "root {i} residual too large: |p(z)|^2 = {mag_sq}"
            );
        }

        // Check we have exactly one real root (im ≈ 0)
        let real_roots: Vec<_> = roots
            .iter()
            .filter(|r| bigfloat_to_f64(&r.1).abs() < 1e-8)
            .collect();
        assert_eq!(real_roots.len(), 1, "should have exactly 1 real root");
        let real_val = bigfloat_to_f64(&real_roots[0].0);
        assert!(
            (real_val - 1.1673).abs() < 0.001,
            "real root ≈ 1.1673, got {real_val}"
        );
    }

    #[test]
    fn rootof_eval_f64_basic() {
        // x^2 - 4 = (x-2)(x+2)
        let poly = poly_from_coeffs(&[-4, 0, 1]);
        let r0 = rootof_eval_f64(&poly, 0).unwrap();
        let r1 = rootof_eval_f64(&poly, 1).unwrap();

        // Both imaginary parts should be ~0
        assert!(r0.1.abs() < 1e-10);
        assert!(r1.1.abs() < 1e-10);

        // Real parts should be -2 and 2 (sorted)
        let mut reals = [r0.0, r1.0];
        reals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((reals[0] - (-2.0)).abs() < 1e-10);
        assert!((reals[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn rootof_out_of_range() {
        let poly = poly_from_coeffs(&[-1, 0, 1]); // degree 2
        assert!(rootof_eval_f64(&poly, 2).is_none());
        assert!(rootof_eval_f64(&poly, 100).is_none());
    }

    #[test]
    fn nroots_with_multiplicity() {
        // (x - 1)^2 (x + 2) = x^3 - 3x + 2
        let poly = poly_from_coeffs(&[2, -3, 0, 1]);
        let roots = nroots_f64(&poly, 128);
        assert_eq!(roots.len(), 3);
        assert!((roots[0].0 + 2.0).abs() < 1e-12, "{roots:?}");
        assert!((roots[1].0 - 1.0).abs() < 1e-12, "{roots:?}");
        assert!((roots[2].0 - 1.0).abs() < 1e-12, "{roots:?}");
        assert!(roots.iter().all(|r| r.1.abs() < 1e-12));
    }

    #[test]
    fn nroots_wilkinson_like_degree_10() {
        // ∏ (x - k) for k = 1..10
        let mut poly = poly_from_coeffs(&[1]);
        for k in 1..=10 {
            poly = &poly * &poly_from_coeffs(&[-k, 1]);
        }
        let roots = nroots_f64(&poly, 192);
        assert_eq!(roots.len(), 10);
        for (i, r) in roots.iter().enumerate() {
            assert!((r.0 - (i as f64 + 1.0)).abs() < 1e-8, "root {i}: {r:?}");
            assert!(r.1.abs() < 1e-8);
        }
    }
}
