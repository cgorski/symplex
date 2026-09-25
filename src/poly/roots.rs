//! Aberth's method for finding all roots (real and complex) of a polynomial.
//!
//! Given a polynomial `p(x) = a_n x^n + ... + a_1 x + a_0` with rational
//! coefficients, Aberth's method simultaneously converges to all `n` roots
//! using cubic-order iteration.
//!
//! # Algorithm
//!
//! Starting from `n` initial guesses on the circles given by the Newton
//! polygon of the coefficients (one circle per scale of root moduli, see
//! [`initial_guesses`]), the method iterates:
//!
//! ```text
//! w_k = p(z_k) / (p'(z_k) - p(z_k) * Σ_{j≠k} 1/(z_k - z_j))
//! z_k ← z_k - w_k
//! ```
//!
//! until all corrections `|w_k|` are below the desired precision.

use astro_float::{BigFloat, Consts, RoundingMode};
use num_bigint::BigInt;
use num_complex::Complex64;
use num_rational::Ratio;
use num_traits::{ToPrimitive, Zero};

use super::dense::Poly;
use crate::base::bigcomplex::{c_add, c_div, c_mul, c_one, c_sub, c_zero};
use crate::base::numeric;

/// A complex number as `(real, imaginary)` pair of arbitrary-precision
/// floats (the working type; results are handed out as [`Complex64`]).
pub(crate) use crate::base::bigcomplex::Complex;

/// Evaluate a polynomial with rational coefficients at a complex point
/// using Horner's method.
///
/// Coefficients are in ascending degree order: `[a_0, a_1, ..., a_n]`.
#[cfg(test)]
fn poly_eval_complex(
    coeffs: &[Ratio<BigInt>],
    z: &Complex,
    prec: usize,
    rm: RoundingMode,
) -> Complex {
    let bf: Vec<BigFloat> = coeffs.iter().map(|c| ratio_to_bigfloat(c, prec)).collect();
    poly_eval_complex_bf(&bf, z, prec, rm)
}

/// `poly_eval_complex` on coefficients already converted to `BigFloat`
/// (the Aberth loop evaluates the same polynomial `n` times per iteration
/// and converts once).  The coefficients are real, so each Horner step is
/// `result · z + c` with a real `c`.
fn poly_eval_complex_bf(
    coeffs: &[BigFloat],
    z: &Complex,
    prec: usize,
    rm: RoundingMode,
) -> Complex {
    let mut result = c_zero(prec);
    for c in coeffs.iter().rev() {
        result = c_mul(&result, z, prec, rm);
        result.0 = result.0.add(c, prec, rm);
    }
    result
}

/// Convert a `Ratio<BigInt>` to a `BigFloat` at the given precision
/// (see [`numeric::ratio_to_bigfloat`]).
fn ratio_to_bigfloat(r: &Ratio<BigInt>, prec: usize) -> BigFloat {
    numeric::ratio_to_bigfloat(r, prec, RoundingMode::None)
}

/// Cauchy's upper bound on the absolute value of all roots,
/// `1 + maxᵢ |aᵢ / aₙ|` (see [`super::sturm::cauchy_bound`]), converted
/// once to a `BigFloat`.
fn cauchy_bound(poly: &Poly, prec: usize) -> BigFloat {
    ratio_to_bigfloat(&super::sturm::cauchy_bound(poly), prec)
}

/// `log₂ |r|` for a non-zero rational, from the bit lengths when the value
/// does not fit an `f64`.  Only used to size the starting circle of the
/// iteration, so a few bits of error are irrelevant.
fn log2_abs(r: &Ratio<BigInt>) -> f64 {
    match r.to_f64() {
        Some(v) if v.is_finite() && v != 0.0 => v.abs().log2(),
        _ => r.numer().bits() as f64 - r.denom().bits() as f64,
    }
}

/// Fujiwara's bound on the modulus of the roots of
/// `p(x) = a_n x^n + … + a_0`:
///
/// ```text
/// |z| ≤ 2 · max( |a_{n−1}/a_n|, |a_{n−2}/a_n|^{1/2}, …, |a_1/a_n|^{1/(n−1)}, |a_0/(2a_n)|^{1/n} )
/// ```
///
/// Unlike the Cauchy bound `1 + max |a_i/a_n|` this is within a factor of
/// two of the largest root modulus even when a middle coefficient is huge
/// (a binomial-tail polynomial of degree 40 has all roots in `|z| < 1.5`
/// but a Cauchy bound above `6·10⁷`; Aberth started that far out shrinks
/// the circle by only `≈ (1 − 1/n)` per step and needs thousands of
/// iterations).  Computed in `f64` from `log₂` of the coefficient ratios;
/// `None` when that fails (a non-finite result), in which case the caller
/// falls back to the Cauchy bound.
fn fujiwara_bound(poly: &Poly) -> Option<f64> {
    let n = poly.degree()?;
    if n == 0 {
        return None;
    }
    let lc = poly.coeff(n);
    if lc.is_zero() {
        return None;
    }
    let mut max_log = f64::NEG_INFINITY;
    for i in 1..=n {
        let c = poly.coeff(n - i);
        if c.is_zero() {
            continue;
        }
        let mut ratio = &c / &lc;
        if i == n {
            ratio /= Ratio::from_integer(BigInt::from(2));
        }
        let l = log2_abs(&ratio) / i as f64;
        if l > max_log {
            max_log = l;
        }
    }
    if !max_log.is_finite() {
        return None;
    }
    let bound = 2.0 * max_log.exp2();
    (bound.is_finite() && bound > 0.0).then_some(bound)
}

/// Radius of the starting circle for the Aberth iteration: the Fujiwara
/// root bound, widened by 10% so that the guesses do not sit on a root,
/// or the Cauchy bound when the former cannot be computed.
fn start_radius(poly: &Poly, prec: usize) -> BigFloat {
    match fujiwara_bound(poly) {
        Some(b) => BigFloat::from_f64(b * 1.1, prec),
        None => cauchy_bound(poly, prec),
    }
}

/// A group of starting points on one circle: `count` guesses at modulus
/// `radius`, the circle belonging to the Newton-polygon edge that starts
/// at coefficient index `start`.
struct StartCircle {
    start: usize,
    count: usize,
    radius: f64,
}

/// Bini's starting circles from the Newton polygon of `p`: the upper
/// convex hull of the points `(i, log₂|aᵢ|)`.  An edge from `(k, yₖ)` to
/// `(l, yₗ)` says that `p` has `l − k` roots of modulus close to
/// `2^{(yₖ − yₗ)/(l − k)}` (Bini, *Numer. Algorithms* 13 (1996); the
/// initialisation MPSolve uses).  Where all roots have comparable modulus
/// the hull is a single edge and this reduces to one circle at the
/// geometric mean of the root moduli; where they do not — the
/// binomial-tail polynomials have a dozen roots at modulus `≈ 0.1` and
/// the rest near `1` — one circle per scale lets Aberth converge in a
/// few dozen iterations instead of a few hundred.
///
/// `p(0) ≠ 0` is required (zero roots are split off by the caller).
/// `None` when a logarithm is not finite, in which case the caller falls
/// back to a single circle.
fn newton_polygon_circles(poly: &Poly) -> Option<Vec<StartCircle>> {
    let n = poly.degree()?;
    let pts: Vec<(usize, f64)> = poly
        .coeffs()
        .iter()
        .enumerate()
        .filter(|(_, c)| !c.is_zero())
        .map(|(i, c)| (i, log2_abs(c)))
        .collect();
    if pts.len() < 2 || pts.iter().any(|(_, y)| !y.is_finite()) {
        return None;
    }
    // Upper hull, left to right: keep only clockwise turns.
    let mut hull: Vec<(usize, f64)> = Vec::with_capacity(pts.len());
    for &p in &pts {
        while hull.len() >= 2 {
            let o = hull[hull.len() - 2];
            let a = hull[hull.len() - 1];
            let cross =
                (a.0 as f64 - o.0 as f64) * (p.1 - o.1) - (a.1 - o.1) * (p.0 as f64 - o.0 as f64);
            if cross >= 0.0 {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(p);
    }
    let mut circles: Vec<StartCircle> = Vec::with_capacity(hull.len() - 1);
    for w in hull.windows(2) {
        let (k, yk) = w[0];
        let (l, yl) = w[1];
        let count = l - k;
        let radius = ((yk - yl) / count as f64).exp2();
        if !radius.is_finite() || radius <= 0.0 {
            return None;
        }
        circles.push(StartCircle {
            start: k,
            count,
            radius,
        });
    }
    (circles.iter().map(|c| c.count).sum::<usize>() == n).then_some(circles)
}

/// Generate initial root approximations.
///
/// The guesses lie on the circles of [`newton_polygon_circles`]: for the
/// edge starting at coefficient `k` with `m` roots of modulus `r`,
/// `z_j = r · exp(iθ_j)` with `θ_j = 2π(j + 1/4)/m + 2πk/n + 0.4`.  The
/// `2πk/n` term rotates each circle differently so guesses on different
/// circles are not radially aligned.  If the polygon cannot be computed
/// the classic single circle is used instead: `z_k = center + radius ·
/// exp(i(2π(k + 1/4)/n + 0.4))` with `center = -a_{n-1}/(n · a_n)` and
/// `radius` from [`start_radius`].
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
    let two_pi = cc.pi(prec, rm).mul(&BigFloat::from_i32(2, prec), prec, rm);
    let n_bf = BigFloat::from_i64(n as i64, prec);
    let quarter = BigFloat::from_f64(0.25, prec);
    let offset = BigFloat::from_f64(0.4, prec);

    // `center + radius · exp(i · (2π · frac + rotation + 0.4))`.
    let mut point = |center: &BigFloat, radius: &BigFloat, frac: &BigFloat, rotation: &BigFloat| {
        let angle = two_pi
            .mul(frac, prec, rm)
            .add(rotation, prec, rm)
            .add(&offset, prec, rm);
        let cos_a = angle.cos(prec, rm, cc);
        let sin_a = angle.sin(prec, rm, cc);
        let re = center.add(&radius.mul(&cos_a, prec, rm), prec, rm);
        let im = radius.mul(&sin_a, prec, rm);
        (re, im)
    };

    if let Some(circles) = newton_polygon_circles(poly) {
        let zero = BigFloat::new(prec);
        let mut guesses = Vec::with_capacity(n);
        for circle in circles {
            let radius = BigFloat::from_f64(circle.radius, prec);
            let m_bf = BigFloat::from_i64(circle.count as i64, prec);
            let rotation = two_pi
                .mul(&BigFloat::from_i64(circle.start as i64, prec), prec, rm)
                .div(&n_bf, prec, rm);
            for j in 0..circle.count {
                let frac = BigFloat::from_i64(j as i64, prec)
                    .add(&quarter, prec, rm)
                    .div(&m_bf, prec, rm);
                guesses.push(point(&zero, &radius, &frac, &rotation));
            }
        }
        return guesses;
    }

    let radius = start_radius(poly, prec);
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
    let zero = BigFloat::new(prec);
    (0..n)
        .map(|k| {
            // angle = 2π * (k + 1/4) / n + 0.4
            let frac = BigFloat::from_i64(k as i64, prec)
                .add(&quarter, prec, rm)
                .div(&n_bf, prec, rm);
            point(&center, &radius, &frac, &zero)
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
    if let Some(roots) = ABERTH_MEMO.with(|m| m.borrow_mut().get(poly, prec, max_iter)) {
        return roots;
    }
    let roots = aberth_roots_uncached(poly, prec, max_iter);
    ABERTH_MEMO.with(|m| m.borrow_mut().put(poly, prec, max_iter, &roots));
    roots
}

/// How many `(polynomial, precision)` root sets [`ABERTH_MEMO`] keeps.
const ABERTH_MEMO_CAPACITY: usize = 8;

/// The last few results of [`aberth_roots`], most recent first.
///
/// The same polynomial is solved over and over: every `RootOf(p, k)` node
/// is evaluated by computing all roots of `p` and picking the `k`-th, and
/// a `RootSum` is re-solved at every point it is evaluated at.  The two
/// self-checks of `∫ atan(√x − x⁹) dx` evaluate its degree-36 `RootSum`
/// at each sample point: 13.0 s of 13.5 s in a debug build without the
/// memo, 1.8 s with it.  The result is a pure function of the key, so a
/// hit returns exactly what a recomputation would.
struct AberthMemo {
    entries: Vec<(Poly, usize, usize, Vec<Complex>)>,
}

impl AberthMemo {
    fn get(&mut self, poly: &Poly, prec: usize, max_iter: usize) -> Option<Vec<Complex>> {
        let pos = self
            .entries
            .iter()
            .position(|(p, pr, it, _)| *pr == prec && *it == max_iter && p == poly)?;
        let entry = self.entries.remove(pos);
        let roots = entry.3.clone();
        self.entries.insert(0, entry);
        Some(roots)
    }

    fn put(&mut self, poly: &Poly, prec: usize, max_iter: usize, roots: &[Complex]) {
        self.entries.truncate(ABERTH_MEMO_CAPACITY - 1);
        self.entries
            .insert(0, (poly.clone(), prec, max_iter, roots.to_vec()));
    }
}

thread_local! {
    /// Per-thread memo of [`aberth_roots`] (see [`AberthMemo`]).
    static ABERTH_MEMO: std::cell::RefCell<AberthMemo> =
        const { std::cell::RefCell::new(AberthMemo { entries: Vec::new() }) };
}

fn aberth_roots_uncached(poly: &Poly, prec: usize, max_iter: usize) -> Vec<Complex> {
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

/// Convergence tolerance of the Aberth iteration: it stops once every
/// correction `|w_k|` is below this (absolute) value, which is far beyond
/// what an `f64` result can show.  A computed root component smaller than
/// this (relative to `max(1, |z|)`) is therefore not distinguishable from
/// zero by the method; [`nroots_f64`] uses that to clean up noise such as
/// `re ≈ 10⁻⁹³` on `±i`.
pub(crate) const ABERTH_TOLERANCE: f64 = 1e-30;

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

    // Coefficients converted once; the loop evaluates `p` and `p'` at `n`
    // points per iteration.
    let monic_bf: Vec<BigFloat> = monic
        .coeffs()
        .iter()
        .map(|c| ratio_to_bigfloat(c, wp))
        .collect();
    let deriv_bf: Vec<BigFloat> = deriv
        .coeffs()
        .iter()
        .map(|c| ratio_to_bigfloat(c, wp))
        .collect();

    let threshold = BigFloat::from_f64(ABERTH_TOLERANCE, wp);

    for _iter in 0..max_iter {
        let mut max_correction = BigFloat::new(wp);
        let mut corrections: Vec<Complex> = Vec::with_capacity(n);

        for i in 0..n {
            // p(z_i)
            let p_zi = poly_eval_complex_bf(&monic_bf, &roots[i], wp, rm);

            // p'(z_i)
            let pp_zi = poly_eval_complex_bf(&deriv_bf, &roots[i], wp, rm);

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
/// `[lo, hi]` — its position in `roots = `[`rootof_roots`]`(g,
/// ROOTOF_DEFAULT_PREC)`, which the caller computes once per `g` (it is
/// the expensive step) and reuses for every real root of `g`.
///
/// The root is verified, not assumed: exactly one computed root must have
/// its real part in `[lo, hi]` and a negligible imaginary part, and no
/// other root may have a real part within `2⁻⁶⁰` (relative) of it, since
/// the (re, im) sort would then order the two by rounding noise and the
/// index would not be stable across evaluation precisions.  `None` when
/// any of these fails (including when the iteration behind `roots` did
/// not converge); the caller then has no reliable name for the root.
pub(crate) fn real_root_index(
    roots: &[Complex],
    lo: &Ratio<BigInt>,
    hi: &Ratio<BigInt>,
) -> Option<usize> {
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

/// Convenience: evaluate the `index`-th root of a polynomial as a
/// [`Complex64`].
///
/// Returns `None` if the index is out of range.
#[allow(dead_code)] // Used by tests; will be wired to evalf in a future PR
pub(crate) fn rootof_eval_f64(poly: &Poly, index: usize) -> Option<Complex64> {
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
    Some(Complex64::new(bigfloat_to_f64(re), bigfloat_to_f64(im)))
}

/// Relative size below which a computed imaginary part is treated as a
/// *candidate* for being numerical noise on a real root.  The decision
/// itself is made exactly (see [`is_real_root_near`]); this only avoids
/// building a Sturm chain for clearly complex roots.
const REAL_AXIS_NOISE: f64 = 1e-6;

/// Does the square-free polynomial `part` have a real root within a tiny
/// interval around `z.re`?  Decided exactly with a Sturm count over
/// `[re − ε, re + ε]`, `ε = 2⁻³⁰ · max(1, |re|)`, on exact rationals.
///
/// Returns `false` immediately when `|im|` is not small relative to the
/// root, so the chain is only built when a root actually looks real.  The
/// chain is built lazily and cached in `sturm` across calls.
fn is_real_root_near(
    part: &Poly,
    z: Complex64,
    sturm: &mut Option<super::sturm::SturmChain>,
) -> bool {
    let Complex64 { re, im } = z;
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

// ── Inclusion disks ──────────────────────────────────────────────────────────────────────────

/// A computed root with a certified error bound (see [`root_balls`]).
#[derive(Clone, Debug)]
pub(crate) struct RootBall {
    /// The root.  A root certified real has an exactly zero imaginary part.
    pub(crate) value: Complex,
    /// An upper bound on the distance from `value` to the root, a 64-bit
    /// float; `None` when no disk isolating the root was certified.
    pub(crate) radius: Option<BigFloat>,
}

/// Precision of the radius arithmetic in [`root_balls`].
const BALL_PREC: usize = 64;

/// `x · (1 + 2⁻⁴⁰)`: absorbs the round-to-nearest errors of a few
/// [`BALL_PREC`]-bit operations in an upper bound.
fn inflate(x: &BigFloat) -> BigFloat {
    let rm = RoundingMode::ToEven;
    let tiny = pow2(-40, BALL_PREC);
    x.add(&x.abs().mul(&tiny, BALL_PREC, rm), BALL_PREC, rm)
}

/// `x · (1 − 2⁻⁴⁰)`, the lower-bound counterpart of [`inflate`].
fn deflate(x: &BigFloat) -> BigFloat {
    let rm = RoundingMode::ToEven;
    let tiny = pow2(-40, BALL_PREC);
    x.sub(&x.abs().mul(&tiny, BALL_PREC, rm), BALL_PREC, rm)
}

/// `2^e` at precision `p` (clamped to astro-float's exponent range).
fn pow2(e: i64, p: usize) -> BigFloat {
    let mut x = BigFloat::from_i32(1, p);
    let e = e.saturating_add(1).clamp(
        i64::from(astro_float::EXPONENT_MIN),
        i64::from(astro_float::EXPONENT_MAX),
    );
    x.set_exponent(astro_float::Exponent::try_from(e).unwrap_or(0));
    x
}

/// `|z|` at [`BALL_PREC`] bits.
fn modulus(z: &Complex) -> BigFloat {
    crate::base::bigcomplex::c_abs(z, BALL_PREC, RoundingMode::ToEven)
}

/// `p(z)` at `wp` bits by Horner's scheme, and `Σ |aᵢ|·|z|ⁱ` at
/// [`BALL_PREC`] bits, which bounds the rounding error of the scheme.
fn horner_with_scale(
    coeffs: &[BigFloat],
    abs_coeffs: &[BigFloat],
    z: &Complex,
    wp: usize,
) -> (Complex, BigFloat) {
    let rm = RoundingMode::ToEven;
    let value = poly_eval_complex_bf(coeffs, z, wp, rm);
    let r = modulus(z);
    let mut scale = BigFloat::new(BALL_PREC);
    for c in abs_coeffs.iter().rev() {
        scale = scale.mul(&r, BALL_PREC, rm).add(c, BALL_PREC, rm);
    }
    (value, inflate(&scale))
}

/// Certified inclusion disks for the computed roots `roots` (all `deg p`
/// of them, in their order) of `p`.
///
/// For any point `z`, `p′(z)/p(z) = Σⱼ 1/(z − rⱼ)` over the roots, so some
/// root lies within `n·|p(z)/p′(z)|` of `z` (the classical Newton inclusion
/// radius), with `|p(z)|` enlarged by the rounding error of its evaluation
/// (Higham's bound for Horner's scheme, `≈ 2n·u·Σ|aᵢ||z|ⁱ`, taken
/// generously) and `|p′(z)|` reduced by its own.  Each root is first
/// polished by Newton steps at `prec + 64` bits: the Aberth iteration stops
/// at corrections of [`ABERTH_TOLERANCE`], far above what a high precision
/// asks for.
///
/// When the `n` disks are pairwise disjoint, each holds exactly one root:
/// every disk holds at least one, and there are `n` roots (so `p` is then
/// square-free as well).  A disk whose mirror image in the real axis meets
/// no other disk holds a *real* root, since `p` has real coefficients: the
/// conjugate of its root is a root in the mirror image, hence in no other
/// disk, hence in this one, which holds only one.  Such a root is returned
/// with an exactly zero imaginary part, and its real part is within the
/// radius as well.  When some disks overlap (close or multiple roots) no
/// radius is certified and the computed roots are returned unchanged.
///
/// Before 0.29 a `RootOf` was trusted to its working precision whatever
/// the iteration left: `im(RootOf(y⁵ − y + 1, 0))` came out `−4·10⁻¹⁶¹`
/// with 30 certified digits, for a real root.
pub(crate) fn root_balls(poly: &Poly, roots: &[Complex], prec: usize) -> Vec<RootBall> {
    let uncertified = || {
        roots
            .iter()
            .map(|z| RootBall {
                value: z.clone(),
                radius: None,
            })
            .collect::<Vec<_>>()
    };
    let n = match poly.degree() {
        Some(d) if d >= 1 && d == roots.len() => d,
        _ => return uncertified(),
    };
    if roots.iter().any(|z| !is_finite(z)) {
        return uncertified();
    }
    let rm = RoundingMode::ToEven;
    let wp = prec + 64;
    let monic = poly.make_monic();
    let deriv = monic.derivative();
    let to_bf = |p: &Poly, bits: usize| -> Vec<BigFloat> {
        p.coeffs()
            .iter()
            .map(|c| ratio_to_bigfloat(c, bits))
            .collect()
    };
    let p_bf = to_bf(&monic, wp);
    let d_bf = to_bf(&deriv, wp);
    let p_abs: Vec<BigFloat> = to_bf(&monic, BALL_PREC).iter().map(BigFloat::abs).collect();
    let d_abs: Vec<BigFloat> = to_bf(&deriv, BALL_PREC).iter().map(BigFloat::abs).collect();
    // Relative rounding of a Horner evaluation, generously: (8n + 8)·u.
    let n_bits = i64::from(usize::BITS - (8 * n + 8).leading_zeros());
    let unit = pow2(
        n_bits + 1 - i64::try_from(wp).unwrap_or(i64::MAX / 4),
        BALL_PREC,
    );
    let n_bf = BigFloat::from_u64(n as u64, BALL_PREC);

    let mut polished: Vec<Complex> = Vec::with_capacity(n);
    let mut radii: Vec<BigFloat> = Vec::with_capacity(n);
    for z0 in roots {
        let z = newton_polish(&p_bf, &d_bf, z0, wp);
        let (pz, p_scale) = horner_with_scale(&p_bf, &p_abs, &z, wp);
        let (dz, d_scale) = horner_with_scale(&d_bf, &d_abs, &z, wp);
        let p_err = p_scale.mul(&unit, BALL_PREC, rm);
        let d_err = d_scale.mul(&unit, BALL_PREC, rm);
        let numer = inflate(&modulus(&pz).add(&p_err, BALL_PREC, rm));
        let denom = deflate(&deflate(&modulus(&dz)).sub(&p_err.max(&d_err), BALL_PREC, rm));
        if !denom.is_positive() {
            return uncertified();
        }
        let r = inflate(&n_bf.mul(&numer, BALL_PREC, rm).div(&denom, BALL_PREC, rm));
        if r.is_inf() || r.is_nan() {
            return uncertified();
        }
        polished.push(z);
        radii.push(r);
    }

    // Is |a − b| > reach?  Compared squared (no square root), with the
    // difference taken at `wp` bits and the rest rounded outwards.
    let beyond = |a: &Complex, b: &Complex, reach: &BigFloat| -> bool {
        let d = c_sub(a, b, wp, rm);
        let re = d.0.mul(&d.0, BALL_PREC, rm);
        let im = d.1.mul(&d.1, BALL_PREC, rm);
        let d2 = deflate(&deflate(&re.add(&im, BALL_PREC, rm)));
        let reach = inflate(reach);
        d2.cmp(&inflate(&reach.mul(&reach, BALL_PREC, rm)))
            .unwrap_or(-1)
            > 0
    };
    // Pairwise disjoint: |zⱼ − zₖ| > rⱼ + rₖ.
    for k in 0..n {
        for j in (k + 1)..n {
            let reach = radii[j].add(&radii[k], BALL_PREC, rm);
            if !beyond(&polished[j], &polished[k], &reach) {
                return uncertified();
            }
        }
    }

    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        let z = &polished[k];
        // A disk that misses the real axis holds a non-real root; one that
        // meets it holds a real root when the mirror-symmetric disk around
        // Re zₖ of radius |Im zₖ| + rₖ meets no other disk.
        let centre = (z.0.clone(), BigFloat::new(wp));
        let im = modulus(&(z.1.clone(), BigFloat::new(wp)));
        let meets_axis = deflate(&im).cmp(&radii[k]).unwrap_or(-1) <= 0;
        let mirror = z.1.abs().add(&radii[k], BALL_PREC, rm);
        let real = meets_axis
            && (0..n).filter(|&j| j != k).all(|j| {
                let reach = mirror.add(&radii[j], BALL_PREC, rm);
                beyond(&polished[j], &centre, &reach)
            });
        out.push(RootBall {
            value: if real { centre } else { z.clone() },
            radius: Some(radii[k].clone()),
        });
    }
    out
}

fn is_finite(z: &Complex) -> bool {
    !(z.0.is_nan() || z.0.is_inf() || z.1.is_nan() || z.1.is_inf())
}

/// Newton's iteration `z ← z − p(z)/p′(z)` at `wp` bits from `z0`, while the
/// corrections shrink (at most 8 steps): an Aberth root is already in the
/// region of quadratic convergence when it is isolated.  A step that does
/// not shrink is not taken.
fn newton_polish(p: &[BigFloat], d: &[BigFloat], z0: &Complex, wp: usize) -> Complex {
    let rm = RoundingMode::ToEven;
    let mag = |z: &Complex| -> Option<i64> {
        let part = |x: &BigFloat| {
            (!x.is_zero())
                .then(|| x.exponent().map(i64::from))
                .flatten()
        };
        match (part(&z.0), part(&z.1)) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        }
    };
    let mut z = z0.clone();
    let mut previous: Option<i64> = None;
    for _ in 0..8 {
        let pz = poly_eval_complex_bf(p, &z, wp, rm);
        let dz = poly_eval_complex_bf(d, &z, wp, rm);
        if mag(&dz).is_none() {
            break;
        }
        let w = c_div(&pz, &dz, wp, rm);
        let Some(wm) = mag(&w) else { break };
        if previous.is_some_and(|pm| wm >= pm) || !is_finite(&w) {
            break;
        }
        z = c_sub(&z, &w, wp, rm);
        previous = Some(wm);
        if mag(&z).is_some_and(|zm| wm < zm - i64::try_from(wp).unwrap_or(i64::MAX / 4) + 2) {
            break;
        }
    }
    z
}

/// Convert a `BigFloat` to `f64` (best-effort).
fn bigfloat_to_f64(bf: &BigFloat) -> f64 {
    // Try direct conversion via the Display trait
    let s = format!("{}", bf);
    s.parse::<f64>().unwrap_or(f64::NAN)
}

/// All complex roots of `poly` as [`Complex64`] values, with
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
/// A real part below the iteration's convergence tolerance
/// ([`ABERTH_TOLERANCE`]` · max(1, |z|)`, i.e. `10⁻³⁰` for roots of
/// modulus at most one) is numerical noise on a purely imaginary root and
/// is returned as exactly `0.0`, so `x² + 1` gives `±i` and not
/// `-7.7·10⁻⁹³ ± i`.  Only a component the iteration cannot distinguish
/// from zero is touched (the tolerance is some fourteen orders of
/// magnitude below `f64` resolution at that scale); a genuinely tiny root
/// such as `±10⁻²⁰` (from `x² − 10⁻⁴⁰`) is far above it and kept.
/// The imaginary part is never snapped this way — whether a root is real
/// is decided exactly, as above.
///
/// Roots are sorted by real part, then imaginary part.  Constants and the
/// zero polynomial produce an empty vector.
pub(crate) fn nroots_f64(poly: &Poly, prec_bits: usize) -> Vec<Complex64> {
    let mut out: Vec<Complex64> = Vec::new();
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
            let mut z = Complex64::new(bigfloat_to_f64(&re), bigfloat_to_f64(&im));
            if z.im != 0.0 && is_real_root_near(&part, z, &mut sturm) {
                z.im = 0.0;
            }
            if z.re != 0.0 && z.re.abs() < ABERTH_TOLERANCE * z.norm().max(1.0) {
                z.re = 0.0;
            }
            for _ in 0..mult {
                out.push(z);
            }
        }
    }
    out.sort_by(|a, b| {
        a.re.partial_cmp(&b.re)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.im.partial_cmp(&b.im).unwrap_or(std::cmp::Ordering::Equal))
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::One;

    fn poly_from_coeffs(coeffs: &[i64]) -> Poly {
        let rat_coeffs: Vec<Ratio<BigInt>> = coeffs
            .iter()
            .map(|&c| Ratio::from_integer(BigInt::from(c)))
            .collect();
        Poly::from_coeffs(rat_coeffs)
    }

    /// Aberth started from the Newton-polygon circles converges (residual
    /// below `10⁻²⁰`) on polynomials whose roots span several scales: a
    /// cluster far from the origin, Wilkinson's, and root moduli from
    /// `10⁻³` to `10³`.
    #[test]
    fn aberth_converges_from_newton_polygon_start() {
        let cases: Vec<(&str, Poly)> = vec![
            ("x^2+1", poly_from_coeffs(&[1, 0, 1])),
            ("x^3-1", poly_from_coeffs(&[-1, 0, 0, 1])),
            ("x^5-x-1", poly_from_coeffs(&[-1, -1, 0, 0, 0, 1])),
            ("(x-10)^5+1", {
                let mut p = poly_from_coeffs(&[1]);
                for _ in 0..5 {
                    p = &p * &poly_from_coeffs(&[-10, 1]);
                }
                &p + &poly_from_coeffs(&[1])
            }),
            ("wilkinson10", {
                let mut p = poly_from_coeffs(&[1]);
                for k in 1..=10 {
                    p = &p * &poly_from_coeffs(&[-k, 1]);
                }
                p
            }),
            ("wilkinson20", {
                let mut p = poly_from_coeffs(&[1]);
                for k in 1..=20 {
                    p = &p * &poly_from_coeffs(&[-k, 1]);
                }
                p
            }),
            ("(x^2-2)(x^2-3)(x-1000)(x+1/1000)", {
                let p = &poly_from_coeffs(&[-2, 0, 1]) * &poly_from_coeffs(&[-3, 0, 1]);
                let p = &p * &poly_from_coeffs(&[-1000, 1]);
                &p * &Poly::from_coeffs(vec![
                    Ratio::new(BigInt::from(1), BigInt::from(1000)),
                    Ratio::from_integer(BigInt::from(1)),
                ])
            }),
            ("x^30 + 3x^7 - 2x + 5", {
                let mut c = vec![0i64; 31];
                c[0] = 5;
                c[1] = -2;
                c[7] = 3;
                c[30] = 1;
                poly_from_coeffs(&c)
            }),
        ];
        for (name, p) in cases {
            // Well below the 200 iterations `rootof_roots` allows.
            let roots = aberth_roots(&p, 192, 60);
            assert_eq!(roots.len(), p.degree().unwrap_or(0), "{name}");
            for (re, im) in &roots {
                let v = poly_eval_complex(
                    p.coeffs(),
                    &(re.clone(), im.clone()),
                    256,
                    RoundingMode::None,
                );
                let mag = bigfloat_to_f64(&v.0).abs() + bigfloat_to_f64(&v.1).abs();
                assert!(
                    mag < 1e-20,
                    "{name}: residual {mag} at {} {}",
                    bigfloat_to_f64(re),
                    bigfloat_to_f64(im)
                );
            }
        }
    }

    /// The Newton polygon of the degree-40 binomial-tail polynomial has
    /// two edges: a dozen roots at modulus ≈ 0.1 and the rest near 1.
    #[test]
    fn newton_polygon_separates_scales() {
        // (x² + 10⁻⁴)(x² + 10⁴): two roots at 10⁻², two at 10².
        let p = &poly_from_coeffs(&[1, 0, 10_000]) * &poly_from_coeffs(&[10_000, 0, 1]);
        let circles = newton_polygon_circles(&p).expect("finite logs");
        assert_eq!(circles.len(), 2);
        assert_eq!((circles[0].start, circles[0].count), (0, 2));
        assert!(
            (circles[0].radius - 0.01).abs() < 1e-9,
            "{}",
            circles[0].radius
        );
        assert_eq!((circles[1].start, circles[1].count), (2, 2));
        assert!(
            (circles[1].radius - 100.0).abs() < 1e-6,
            "{}",
            circles[1].radius
        );
        // x⁵ − x − 1: one edge, geometric-mean radius 1.
        let q = poly_from_coeffs(&[-1, -1, 0, 0, 0, 1]);
        let circles = newton_polygon_circles(&q).expect("finite logs");
        assert_eq!(circles.len(), 1);
        assert_eq!(circles[0].count, 5);
        assert!((circles[0].radius - 1.0).abs() < 1e-12);
    }

    /// `nroots_f64` returns exactly `0.0` for the real part of `±i` and
    /// keeps the genuinely tiny roots `±10⁻²⁰` of `x² − 10⁻⁴⁰`.
    #[test]
    fn nroots_snaps_noise_but_not_tiny_roots() {
        let roots = nroots_f64(&poly_from_coeffs(&[1, 0, 1]), 128);
        assert_eq!(roots.len(), 2);
        assert!(roots.iter().all(|z| z.re == 0.0), "{roots:?}");
        assert!(
            roots.iter().all(|z| (z.im.abs() - 1.0).abs() < 1e-15),
            "{roots:?}"
        );

        let tiny = Ratio::new(BigInt::from(-1), BigInt::from(10).pow(40));
        let p = Poly::from_coeffs(vec![tiny, Ratio::zero(), Ratio::one()]);
        let roots = nroots_f64(&p, 128);
        assert_eq!(roots.len(), 2);
        assert!(
            (roots[0].re + 1e-20).abs() < 1e-33 && roots[0].im == 0.0,
            "{roots:?}"
        );
        assert!(
            (roots[1].re - 1e-20).abs() < 1e-33 && roots[1].im == 0.0,
            "{roots:?}"
        );
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
        assert!(r0.im.abs() < 1e-10);
        assert!(r1.im.abs() < 1e-10);

        // Real parts should be -2 and 2 (sorted)
        let mut reals = [r0.re, r1.re];
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
        assert!((roots[0].re + 2.0).abs() < 1e-12, "{roots:?}");
        assert!((roots[1].re - 1.0).abs() < 1e-12, "{roots:?}");
        assert!((roots[2].re - 1.0).abs() < 1e-12, "{roots:?}");
        assert!(roots.iter().all(|r| r.im.abs() < 1e-12));
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
            assert!((r.re - (i as f64 + 1.0)).abs() < 1e-8, "root {i}: {r:?}");
            assert!(r.im.abs() < 1e-8);
        }
    }

    /// Inclusion disks isolate the roots of `y⁵ − y + 1` (one real, two
    /// conjugate pairs) at a radius far below the working precision, and
    /// the real root comes out with an exactly zero imaginary part (Aberth
    /// leaves noise of about `10⁻¹⁶¹` there).
    #[test]
    fn root_balls_certify_radii_and_realness() {
        let p = poly_from_coeffs(&[1, -1, 0, 0, 0, 1]);
        let prec = 166;
        let roots = rootof_roots(&p, prec);
        let balls = root_balls(&p, &roots, prec);
        assert_eq!(balls.len(), 5);
        let reals: Vec<bool> = balls.iter().map(|b| b.value.1.is_zero()).collect();
        assert_eq!(reals, [true, false, false, false, false], "{balls:?}");
        for b in &balls {
            let r = b.radius.as_ref().expect("certified radius");
            assert!(
                r.is_zero() || r.exponent().is_some_and(|e| i64::from(e) < -150),
                "radius {r} for {:?}",
                b.value
            );
        }
    }
}
