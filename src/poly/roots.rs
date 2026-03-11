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
/// Uses the classic Aberth initialization: `z_k = center + radius * exp(2πi(k + 1/4)/n)`
/// where `center = -a_{n-1} / (n * a_n)` and `radius` is the Cauchy bound.
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

    (0..n)
        .map(|k| {
            // angle = 2π * (k + 1/4) / n
            let k_bf = BigFloat::from_i64(k as i64, prec);
            let frac = k_bf.add(&quarter, prec, rm).div(&n_bf, prec, rm);
            let angle = two_pi.mul(&frac, prec, rm);

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
/// by real part (then imaginary part for ties).
pub(crate) fn aberth_roots(poly: &Poly, prec: usize, max_iter: usize) -> Vec<Complex> {
    let n = match poly.degree() {
        Some(d) if d >= 1 => d,
        _ => return vec![],
    };

    // Normalize to monic
    let monic = poly.make_monic();
    let deriv = monic.derivative();

    let rm = RoundingMode::None;
    let wp = prec + 64; // working precision with guard bits
    let mut cc = Consts::new().expect("Consts::new");

    let mut roots = initial_guesses(&monic, n, wp, &mut cc);

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

/// Convert a `BigFloat` to `f64` (best-effort).
#[allow(dead_code)]
fn bigfloat_to_f64(bf: &BigFloat) -> f64 {
    // Try direct conversion via the Display trait
    let s = format!("{}", bf);
    s.parse::<f64>().unwrap_or(f64::NAN)
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
}
