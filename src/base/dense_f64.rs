//! Dense `f64` linear algebra on small row-major matrices: Cholesky
//! (factor, solve, inverse, congruence, regularised solve), the cyclic
//! Jacobi symmetric eigen-decomposition, Gaussian elimination with partial
//! pivoting and Householder least squares.  One implementation behind the
//! SOS interior point, the maximum-likelihood fits in `stats`, the numeric
//! system solver, `polyfit` and heurisch.
//!
//! A matrix is a flat `&[f64]` in row-major order with its dimensions
//! passed alongside (`a[i * n_cols + j]`); [`flatten`] and [`to_rows`]
//! convert to and from the `Vec<Vec<f64>>` shape some callers store.
//! Depends on `std` only.
//!
//! The loop bodies are the SOS solver's, whose printed certificates are
//! byte-compared against baselines: callers that differ only in a
//! tolerance pass it as a parameter, and nothing here reorders a floating
//! point operation.

// ═══════════════════════════════════════════════════════════════════════════
// Layout
// ═══════════════════════════════════════════════════════════════════════════

/// Row-major flat copy of a matrix given as rows.
pub(crate) fn flatten(rows: &[Vec<f64>]) -> Vec<f64> {
    rows.iter().flatten().copied().collect()
}

/// The rows of an `n_rows × n_cols` flat matrix.
pub(crate) fn to_rows(a: &[f64], n_rows: usize, n_cols: usize) -> Vec<Vec<f64>> {
    (0..n_rows)
        .map(|i| a[i * n_cols..(i + 1) * n_cols].to_vec())
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// Products
// ═══════════════════════════════════════════════════════════════════════════

/// `Σ aᵢ bᵢ`, accumulated left to right from `0`.
pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `A x` for an `n_rows × n_cols` matrix.
pub(crate) fn matvec(a: &[f64], n_rows: usize, n_cols: usize, x: &[f64]) -> Vec<f64> {
    (0..n_rows)
        .map(|i| dot(&a[i * n_cols..(i + 1) * n_cols], x))
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// Cholesky
// ═══════════════════════════════════════════════════════════════════════════

/// Lower Cholesky factor `L` with `A = L Lᵀ` of a symmetric `n×n` matrix,
/// or `None` when a pivot `d = a_jj − Σ_k l_jk²` is not finite or
/// `d <= rel_tol · |a_jj|`.  `rel_tol = 0` is the plain positive-definite
/// test; the statistics fits use `1e-12` to reject numerically singular
/// information matrices.
pub(crate) fn cholesky(a: &[f64], n: usize, rel_tol: f64) -> Option<Vec<f64>> {
    let mut l = vec![0.0; n * n];
    for j in 0..n {
        let mut s = a[j * n + j];
        for k in 0..j {
            s -= l[j * n + k] * l[j * n + k];
        }
        if !s.is_finite() || s <= rel_tol * a[j * n + j].abs() {
            return None;
        }
        let d = s.sqrt();
        l[j * n + j] = d;
        for i in (j + 1)..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = s / d;
        }
    }
    Some(l)
}

/// Solve `L Lᵀ x = b` for a lower-triangular `L` (forward then back
/// substitution).
pub(crate) fn cholesky_solve(l: &[f64], n: usize, b: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * n + k] * y[k];
        }
        y[i] = s / l[i * n + i];
    }
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in (i + 1)..n {
            s -= l[k * n + i] * y[k];
        }
        y[i] = s / l[i * n + i];
    }
    y
}

/// `(L Lᵀ)⁻¹`, one column per unit right-hand side.
pub(crate) fn spd_inverse(l: &[f64], n: usize) -> Vec<f64> {
    let mut inv = vec![0.0; n * n];
    let mut e = vec![0.0; n];
    for c in 0..n {
        e[c] = 1.0;
        let col = cholesky_solve(l, n, &e);
        e[c] = 0.0;
        for i in 0..n {
            inv[i * n + c] = col[i];
        }
    }
    inv
}

/// `xᵀ (L Lᵀ)⁻¹ x`.
pub(crate) fn quadratic_form(l: &[f64], n: usize, x: &[f64]) -> f64 {
    dot(x, &cholesky_solve(l, n, x))
}

/// `L⁻¹ A L⁻ᵀ` for a lower-triangular `L` and symmetric `A`, symmetrised
/// against rounding.
pub(crate) fn congruence_inverse(l: &[f64], a: &[f64], n: usize) -> Vec<f64> {
    // Solve L Y = A (column by column), then W = Y L⁻ᵀ, i.e. L Wᵀ = Yᵀ.
    let mut y = vec![0.0; n * n];
    for c in 0..n {
        for i in 0..n {
            let mut s = a[i * n + c];
            for k in 0..i {
                s -= l[i * n + k] * y[k * n + c];
            }
            y[i * n + c] = s / l[i * n + i];
        }
    }
    let mut w = vec![0.0; n * n];
    for r in 0..n {
        // Row r of W: solve L wᵀ = (row r of Y)ᵀ.
        for i in 0..n {
            let mut s = y[r * n + i];
            for k in 0..i {
                s -= l[i * n + k] * w[r * n + k];
            }
            w[r * n + i] = s / l[i * n + i];
        }
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let v = 0.5 * (w[i * n + j] + w[j * n + i]);
            w[i * n + j] = v;
            w[j * n + i] = v;
        }
    }
    w
}

/// The diagonal regularisation [`solve_spd_regularised`] falls back on:
/// the first attempt adds nothing; the next adds `initial_rel · max|a_ii|`
/// (at least `initial_rel · 1e-300`) to the diagonal, and each further
/// attempt multiplies the shift by `growth`, for `rounds` attempts in all.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RegSchedule {
    pub rounds: usize,
    pub initial_rel: f64,
    pub growth: f64,
}

/// Solve the symmetric positive definite system `a x = b` by Cholesky
/// (pivot test `d <= 0`), retrying with a growing diagonal shift per
/// `schedule` when the factorisation fails; `None` when every attempt
/// fails.
pub(crate) fn solve_spd_regularised(
    a: &[f64],
    n: usize,
    b: &[f64],
    schedule: &RegSchedule,
) -> Option<Vec<f64>> {
    let mut reg = 0.0;
    for _ in 0..schedule.rounds {
        let mut m = a.to_vec();
        if reg > 0.0 {
            for i in 0..n {
                m[i * n + i] += reg;
            }
        }
        if let Some(l) = cholesky(&m, n, 0.0) {
            return Some(cholesky_solve(&l, n, b));
        }
        let scale = (0..n)
            .map(|i| a[i * n + i].abs())
            .fold(0.0, f64::max)
            .max(1e-300);
        reg = if reg == 0.0 {
            scale * schedule.initial_rel
        } else {
            reg * schedule.growth
        };
    }
    None
}

// ═══════════════════════════════════════════════════════════════════════════
// Symmetric eigen-decomposition (cyclic Jacobi)
// ═══════════════════════════════════════════════════════════════════════════

/// When the Jacobi sweeps stop.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum EigenTol {
    /// `‖off-diagonal‖_F ≤ rel · ‖A‖_F` (both triangles, `‖A‖_F` floored at
    /// `f64::MIN_POSITIVE`).
    RelativeFrobenius(f64),
    /// `Σ_{i<j} a_ij² < eps` (upper triangle, no square root).
    Absolute(f64),
}

/// What to do when `max_sweeps` pass without meeting the tolerance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OnExhaust {
    /// Return [`DenseError::NoConvergence`].
    Error,
    /// Return the current diagonal and rotations as the decomposition.
    Accept,
}

/// Options of [`sym_eigen`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct EigenOpts {
    pub tol: EigenTol,
    pub max_sweeps: usize,
    pub on_exhaust: OnExhaust,
}

/// Eigen-decomposition of a symmetric `n×n` matrix.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SymEigen {
    /// The `n` eigenvalues, in the order the Jacobi sweeps leave them on
    /// the diagonal (not sorted).
    pub values: Vec<f64>,
    /// The eigenvectors as the **columns** of a row-major `n×n` matrix:
    /// component `i` of the eigenvector for `values[c]` is
    /// `vectors[i * n + c]`.
    pub vectors: Vec<f64>,
}

/// Failure of a dense routine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DenseError {
    /// The Jacobi sweeps did not meet the tolerance in `sweeps` sweeps.
    NoConvergence { sweeps: usize },
}

impl std::fmt::Display for DenseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DenseError::NoConvergence { sweeps } => {
                write!(
                    f,
                    "the Jacobi sweeps did not converge in {sweeps} iterations"
                )
            }
        }
    }
}

/// Eigen-decomposition of a symmetric matrix by cyclic Jacobi rotations
/// (Golub & Van Loan, *Matrix Computations*, Algorithm 8.5.1): each sweep
/// visits every pair `p < q` and rotates it to zero `a_pq`; the tolerance
/// is tested before each sweep.
///
/// Under a [`EigenTol::RelativeFrobenius`] tolerance a matrix whose largest
/// entry lies outside `[2⁻⁴⁵⁰, 2⁴⁵⁰]` is first scaled by a power of two
/// (exact) to largest entry near 1, and the eigenvalues scaled back, as
/// LAPACK's `dsyev` scales its input: the test forms sums of squares,
/// which underflow to 0 (or overflow to ∞) beyond about `10±¹⁵⁴`, and
/// 0.28 then declared an unrotated matrix converged — `[[2, 1], [1, 2]]·10⁻¹⁷⁰`
/// had "eigenvalues" `2·10⁻¹⁷⁰, 2·10⁻¹⁷⁰` for `3·10⁻¹⁷⁰, 10⁻¹⁷⁰`.  An
/// [`EigenTol::Absolute`] tolerance is the caller's own scale and is used
/// as given.
pub(crate) fn sym_eigen(a: &[f64], n: usize, opts: &EigenOpts) -> Result<SymEigen, DenseError> {
    const SAFE: i32 = 450;
    let amax = a.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    // 2^e ≤ amax < 2^(e+1), subnormals included (|e| ≤ 1075).
    let e = if amax.is_finite() && amax > 0.0 {
        amax.log2().floor() as i32
    } else {
        0
    };
    if !matches!(opts.tol, EigenTol::RelativeFrobenius(_)) || (-SAFE..=SAFE).contains(&e) {
        return sym_eigen_unscaled(a, n, opts);
    }
    // Two factors, so that neither 2^±e alone under- or overflows.
    let (h1, h2) = (e / 2, e - e / 2);
    let down = |v: f64| v * 2f64.powi(-h1) * 2f64.powi(-h2);
    let scaled: Vec<f64> = a.iter().map(|&v| down(v)).collect();
    let mut r = sym_eigen_unscaled(&scaled, n, opts)?;
    for v in &mut r.values {
        *v = *v * 2f64.powi(h1) * 2f64.powi(h2);
    }
    Ok(r)
}

/// [`sym_eigen`] on the matrix as given.
fn sym_eigen_unscaled(a: &[f64], n: usize, opts: &EigenOpts) -> Result<SymEigen, DenseError> {
    let mut m = a.to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    let threshold = match opts.tol {
        EigenTol::RelativeFrobenius(rel) => {
            let frob: f64 = m.iter().map(|x| x * x).sum::<f64>().sqrt();
            rel * frob.max(f64::MIN_POSITIVE)
        }
        EigenTol::Absolute(eps) => eps,
    };
    let mut converged = false;
    for _sweep in 0..opts.max_sweeps {
        let done = match opts.tol {
            EigenTol::RelativeFrobenius(_) => {
                let off: f64 = (0..n)
                    .flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
                    .map(|(i, j)| m[i * n + j] * m[i * n + j])
                    .sum::<f64>()
                    .sqrt();
                off <= threshold
            }
            EigenTol::Absolute(_) => {
                let mut off = 0.0;
                for i in 0..n {
                    for j in (i + 1)..n {
                        off += m[i * n + j] * m[i * n + j];
                    }
                }
                off < threshold
            }
        };
        if done {
            converged = true;
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = m[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let app = m[p * n + p];
                let aqq = m[q * n + q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // A ← A·J (columns p, q), then A ← Jᵀ·A (rows p, q), V ← V·J.
                for k in 0..n {
                    let mkp = m[k * n + p];
                    let mkq = m[k * n + q];
                    m[k * n + p] = c * mkp - s * mkq;
                    m[k * n + q] = s * mkp + c * mkq;
                }
                for k in 0..n {
                    let mpk = m[p * n + k];
                    let mqk = m[q * n + k];
                    m[p * n + k] = c * mpk - s * mqk;
                    m[q * n + k] = s * mpk + c * mqk;
                }
                for k in 0..n {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    if !converged && opts.on_exhaust == OnExhaust::Error {
        return Err(DenseError::NoConvergence {
            sweeps: opts.max_sweeps,
        });
    }
    Ok(SymEigen {
        values: (0..n).map(|i| m[i * n + i]).collect(),
        vectors: v,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// General solvers
// ═══════════════════════════════════════════════════════════════════════════

/// Solve the `n×n` system `a x = b` by Gaussian elimination with partial
/// pivoting; `None` when a pivot column has no entry of magnitude
/// `≥ 1e-300` (singular).  Ties in the pivot search go to the last row.
pub(crate) fn solve_partial_pivot(a: &[f64], n: usize, b: &[f64]) -> Option<Vec<f64>> {
    let mut m: Vec<f64> = a.to_vec();
    let mut r = b.to_vec();
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| {
            m[i * n + c]
                .abs()
                .partial_cmp(&m[j * n + c].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
        if m[p * n + c].abs() < 1e-300 {
            return None;
        }
        if p != c {
            for k in 0..n {
                m.swap(c * n + k, p * n + k);
            }
            r.swap(c, p);
        }
        for i in (c + 1)..n {
            let f = m[i * n + c] / m[c * n + c];
            if f != 0.0 {
                for k in c..n {
                    m[i * n + k] -= f * m[c * n + k];
                }
                r[i] -= f * r[c];
            }
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sacc = r[i];
        for k in (i + 1)..n {
            sacc -= m[i * n + k] * x[k];
        }
        x[i] = sacc / m[i * n + i];
    }
    Some(x)
}

/// Least-squares solution of the overdetermined system `a x = b` (`a` is
/// `n_rows × n_cols` with `n_rows ≥ n_cols`) by Householder QR.  `None`
/// if the shapes disagree or `a` is numerically rank deficient (a
/// diagonal of `R` within `n_rows · ε` of zero relative to the largest).
pub(crate) fn lstsq_householder(
    a: &[f64],
    n_rows: usize,
    n_cols: usize,
    b: &[f64],
) -> Option<Vec<f64>> {
    let (m, n) = (n_rows, n_cols);
    if m < n || b.len() != m || a.len() != m * n {
        return None;
    }
    let mut a = a.to_vec();
    let mut b = b.to_vec();
    for k in 0..n {
        let norm = (k..m)
            .map(|i| a[i * n + k] * a[i * n + k])
            .sum::<f64>()
            .sqrt();
        if norm == 0.0 || !norm.is_finite() {
            return None;
        }
        // Householder vector v = x − α·e₁ with α chosen to avoid cancellation.
        let alpha = if a[k * n + k] > 0.0 { -norm } else { norm };
        let mut v: Vec<f64> = (k..m).map(|i| a[i * n + k]).collect();
        v[0] -= alpha;
        let vnorm2: f64 = v.iter().map(|x| x * x).sum();
        if vnorm2 == 0.0 {
            continue;
        }
        // Apply H = I − 2vvᵀ/‖v‖² to the trailing block of `a` and to `b`:
        // A ← A − (2/‖v‖²)·v·(vᵀA).
        let scale = 2.0 / vnorm2;
        let w: Vec<f64> = (k..n)
            .map(|j| {
                v.iter()
                    .zip(k..m)
                    .map(|(vi, i)| vi * a[i * n + j])
                    .sum::<f64>()
            })
            .collect();
        for (vi, i) in v.iter().zip(k..m) {
            for (wj, j) in w.iter().zip(k..n) {
                a[i * n + j] -= scale * vi * wj;
            }
        }
        let s: f64 = v.iter().zip(k..m).map(|(vi, i)| vi * b[i]).sum();
        let factor = scale * s;
        for (vi, i) in v.iter().zip(k..m) {
            b[i] -= factor * vi;
        }
    }
    // Back-substitution on the leading n × n block (R).
    let r_max = (0..n).map(|k| a[k * n + k].abs()).fold(0.0_f64, f64::max);
    let threshold = r_max * f64::EPSILON * m as f64;
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let diag = a[r * n + r];
        if !diag.is_finite() || diag.abs() <= threshold {
            return None;
        }
        let s = b[r] - ((r + 1)..n).map(|c| a[r * n + c] * x[c]).sum::<f64>();
        x[r] = s / diag;
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    //! The kernel is `pub(crate)`, so its direct tests live here; the
    //! callers are exercised end to end in `tests/v20/v20_dense.rs`.
    //! Reference values: `numpy.linalg.{cholesky,inv,solve,eigh,lstsq}`
    //! (`symplex/.venv/bin/python`).

    use super::*;

    fn close(actual: f64, expected: f64) -> bool {
        (actual - expected).abs() < 1e-12 * expected.abs().max(1.0)
    }

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!(close(*a, *e), "{actual:?} vs {expected:?}");
        }
    }

    /// `[[4, 12, -16], [12, 37, -43], [-16, -43, 98]]`, whose Cholesky
    /// factor is the integer matrix `[[2, 0, 0], [6, 1, 0], [-8, 5, 3]]`.
    const SPD3: [f64; 9] = [4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0];

    /// The 4×4 second-difference matrix `2I − (shift + shiftᵀ)`, with
    /// eigenvalues `2 − 2cos(kπ/5)`, k = 1..4.
    const TRIDIAG4: [f64; 16] = [
        2.0, -1.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, -1.0, 2.0,
    ];

    const EIGEN_STRICT: EigenOpts = EigenOpts {
        tol: EigenTol::RelativeFrobenius(1e-15),
        max_sweeps: 100,
        on_exhaust: OnExhaust::Error,
    };

    #[test]
    fn cholesky_3x3_is_the_integer_factor() {
        // numpy.linalg.cholesky(SPD3) = [[2, 0, 0], [6, 1, 0], [-8, 5, 3]]
        let l = cholesky(&SPD3, 3, 0.0).unwrap();
        assert_eq!(l, [2.0, 0.0, 0.0, 6.0, 1.0, 0.0, -8.0, 5.0, 3.0]);
        // numpy.linalg.solve(SPD3, [1, 2, 3]) = [28.583333333333268, -7.666666666666649, 1.3333333333333306]
        let x = cholesky_solve(&l, 3, &[1.0, 2.0, 3.0]);
        assert_close(&x, &[343.0 / 12.0, -23.0 / 3.0, 4.0 / 3.0]);
        // numpy.linalg.inv(SPD3)[0] = [49.36111111111101, -13.555555555555525, 2.1111111111111054]
        let inv = spd_inverse(&l, 3);
        assert_close(&inv[..3], &[1777.0 / 36.0, -122.0 / 9.0, 19.0 / 9.0]);
        assert_close(&inv[3..6], &[-122.0 / 9.0, 34.0 / 9.0, -5.0 / 9.0]);
        assert_close(&inv[6..], &[19.0 / 9.0, -5.0 / 9.0, 1.0 / 9.0]);
        // xᵀ A⁻¹ x for x = [1, 2, 3] is x · solve = 28.583 − 15.333 + 4 = 17.25
        assert!(close(quadratic_form(&l, 3, &[1.0, 2.0, 3.0]), 17.25));
        // L⁻¹ A L⁻ᵀ = I for the matrix's own factor.
        let w = congruence_inverse(&l, &SPD3, 3);
        assert_close(&w, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn non_spd_matrices_return_none() {
        // Indefinite: [[1, 2], [2, 1]] has eigenvalues 3 and −1.
        assert!(cholesky(&[1.0, 2.0, 2.0, 1.0], 2, 0.0).is_none());
        // Singular: rank one.
        assert!(cholesky(&[1.0, 2.0, 2.0, 4.0], 2, 0.0).is_none());
        // Non-finite.
        assert!(cholesky(&[1.0, 0.0, 0.0, f64::NAN], 2, 0.0).is_none());
        assert!(cholesky(&[f64::INFINITY, 0.0, 0.0, 1.0], 2, 0.0).is_none());
        // Zero diagonal with rel_tol = 0: the pivot test `d <= 0` fires.
        assert!(cholesky(&[0.0, 0.0, 0.0, 1.0], 2, 0.0).is_none());
    }

    #[test]
    fn relative_pivot_tolerance() {
        // Second pivot is 1e-13 = 1e-13 · a_11: rejected at rel_tol = 1e-12,
        // accepted at rel_tol = 0 (and at 1e-14).
        let a = [1.0, 1.0, 1.0, 1.0 + 1e-13];
        assert!(cholesky(&a, 2, 1e-12).is_none());
        assert!(cholesky(&a, 2, 1e-14).is_some());
        assert!(cholesky(&a, 2, 0.0).is_some());
        // The tolerance is relative to the diagonal entry, not absolute:
        // scaling the matrix by 1e-20 changes nothing.
        let tiny: Vec<f64> = SPD3.iter().map(|v| v * 1e-20).collect();
        assert!(cholesky(&tiny, 3, 1e-12).is_some());
    }

    #[test]
    fn regularised_solve_recovers_from_a_singular_system() {
        let schedule = RegSchedule {
            rounds: 6,
            initial_rel: 1e-12,
            growth: 100.0,
        };
        // Positive definite: the first attempt succeeds and equals the plain solve.
        let x = solve_spd_regularised(&SPD3, 3, &[1.0, 2.0, 3.0], &schedule).unwrap();
        assert_close(&x, &[343.0 / 12.0, -23.0 / 3.0, 4.0 / 3.0]);
        // Singular but consistent: the shifted system is solved instead
        // (a least-norm-like answer, finite and close to a solution).
        let sing = [1.0, 1.0, 1.0, 1.0];
        let x = solve_spd_regularised(&sing, 2, &[2.0, 2.0], &schedule).unwrap();
        assert!(x.iter().all(|v| v.is_finite()));
        assert!((x[0] + x[1] - 2.0).abs() < 1e-6, "{x:?}");
        // Indefinite beyond the reach of six shifts: `None`.
        assert!(
            solve_spd_regularised(&[-1.0, 0.0, 0.0, -1.0], 2, &[1.0, 1.0], &schedule).is_none()
        );
    }

    #[test]
    fn jacobi_eigen_4x4_matches_eigh() {
        // numpy.linalg.eigh(TRIDIAG4)[0] = [0.3819660112501053, 1.3819660112501055,
        //                                   2.618033988749895, 3.6180339887498945]
        let e = sym_eigen(&TRIDIAG4, 4, &EIGEN_STRICT).unwrap();
        let mut order: Vec<usize> = (0..4).collect();
        order.sort_by(|&a, &b| e.values[a].total_cmp(&e.values[b]));
        let sorted: Vec<f64> = order.iter().map(|&i| e.values[i]).collect();
        assert_close(
            &sorted,
            &[
                0.3819660112501053,
                1.3819660112501055,
                2.618033988749895,
                3.6180339887498945,
            ],
        );
        // eigh eigenvector of the smallest eigenvalue: [0.3717480344601845,
        // 0.6015009550075455, 0.6015009550075455, 0.3717480344601844]
        let c = order[0];
        let mut v: Vec<f64> = (0..4).map(|i| e.vectors[i * 4 + c]).collect();
        if v[0] < 0.0 {
            v.iter_mut().for_each(|x| *x = -*x);
        }
        assert_close(
            &v,
            &[
                0.3717480344601845,
                0.6015009550075455,
                0.6015009550075455,
                0.3717480344601844,
            ],
        );
        // A v = λ v for every column.
        for c in 0..4 {
            let v: Vec<f64> = (0..4).map(|i| e.vectors[i * 4 + c]).collect();
            let av = matvec(&TRIDIAG4, 4, 4, &v);
            for i in 0..4 {
                assert!((av[i] - e.values[c] * v[i]).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn jacobi_eigen_3x3_matches_eigh() {
        // numpy.linalg.eigh([[2, 1, 0.5], [1, 3, 1], [0.5, 1, 4]])[0] =
        //   [1.3770948783558645, 2.6824555614434478, 4.940449560200685]
        let a = [2.0, 1.0, 0.5, 1.0, 3.0, 1.0, 0.5, 1.0, 4.0];
        let mut values = sym_eigen(&a, 3, &EIGEN_STRICT).unwrap().values;
        values.sort_by(f64::total_cmp);
        assert_close(
            &values,
            &[1.3770948783558645, 2.6824555614434478, 4.940449560200685],
        );
    }

    /// Before: the relative test's sums of squares underflowed (overflowed)
    /// at `10∓¹⁷⁰`, and the unrotated diagonal came back as the eigenvalues.
    #[test]
    fn jacobi_eigen_at_extreme_scales() {
        for scale in [1e-170, 1e-300, 1e170, 1e300] {
            let a = [2.0 * scale, scale, scale, 2.0 * scale];
            let mut values = sym_eigen(&a, 2, &EIGEN_STRICT).unwrap().values;
            values.sort_by(f64::total_cmp);
            // Exactly 1 and 3 times the scale (numpy.linalg.eigh agrees).
            assert!((values[0] / scale - 1.0).abs() < 1e-14, "{values:?}");
            assert!((values[1] / scale - 3.0).abs() < 1e-14, "{values:?}");
        }
    }

    #[test]
    fn eigen_tolerances_and_exhaustion() {
        let absolute = EigenOpts {
            tol: EigenTol::Absolute(1e-30),
            max_sweeps: 100,
            on_exhaust: OnExhaust::Accept,
        };
        let a = sym_eigen(&TRIDIAG4, 4, &absolute).unwrap();
        let r = sym_eigen(&TRIDIAG4, 4, &EIGEN_STRICT).unwrap();
        let (mut av, mut rv) = (a.values.clone(), r.values.clone());
        av.sort_by(f64::total_cmp);
        rv.sort_by(f64::total_cmp);
        assert_close(&av, &rv);
        // The relative tolerance scales with the matrix; the absolute one
        // does not: a 1e-16-scaled copy has Σ_{i<j} a_ij² = 6e-32 < 1e-30
        // and is "converged" before any sweep (the raw diagonal comes
        // back), while 1e-15·‖A‖_F still demands the real eigenvalues.
        let tiny: Vec<f64> = TRIDIAG4.iter().map(|v| v * 1e-16).collect();
        assert_eq!(
            sym_eigen(&tiny, 4, &absolute).unwrap().values,
            vec![2e-16; 4]
        );
        let mut scaled = sym_eigen(&tiny, 4, &EIGEN_STRICT).unwrap().values;
        scaled.sort_by(f64::total_cmp);
        let expected: Vec<f64> = rv.iter().map(|v| v * 1e-16).collect();
        assert_close(&scaled, &expected);
        // An unmeetable absolute tolerance (`off < 0` never holds) runs all
        // 100 sweeps: an error, or the (by then converged) diagonal.
        let never = EigenOpts {
            tol: EigenTol::Absolute(0.0),
            max_sweeps: 100,
            on_exhaust: OnExhaust::Error,
        };
        assert_eq!(
            sym_eigen(&TRIDIAG4, 4, &never),
            Err(DenseError::NoConvergence { sweeps: 100 })
        );
        let mut accepted = sym_eigen(
            &TRIDIAG4,
            4,
            &EigenOpts {
                on_exhaust: OnExhaust::Accept,
                ..never
            },
        )
        .unwrap()
        .values;
        accepted.sort_by(f64::total_cmp);
        assert_close(&accepted, &rv);
        // Zero sweeps: the diagonal as is, or an error.
        let none = EigenOpts {
            max_sweeps: 0,
            ..EIGEN_STRICT
        };
        assert_eq!(
            sym_eigen(&TRIDIAG4, 4, &none),
            Err(DenseError::NoConvergence { sweeps: 0 })
        );
        let accept_none = EigenOpts {
            on_exhaust: OnExhaust::Accept,
            ..none
        };
        assert_eq!(
            sym_eigen(&TRIDIAG4, 4, &accept_none).unwrap().values,
            vec![2.0; 4]
        );
        // The tolerance is tested at the top of each sweep, so a diagonal
        // matrix converges at the start of the first one under both tests
        // (and `max_sweeps = 0` never tests it).
        let d = [3.0, 0.0, 0.0, 1.0];
        assert!(sym_eigen(&d, 2, &none).is_err());
        let one = EigenOpts {
            max_sweeps: 1,
            ..EIGEN_STRICT
        };
        assert_eq!(sym_eigen(&d, 2, &one).unwrap().values, vec![3.0, 1.0]);
        let e = sym_eigen(
            &d,
            2,
            &EigenOpts {
                max_sweeps: 1,
                on_exhaust: OnExhaust::Error,
                ..absolute
            },
        )
        .unwrap();
        assert_eq!(e.vectors, vec![1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn partial_pivot_solves_and_detects_singularity() {
        // solve([[0, 1], [1, 0]], [3, 4]) = [4, 3] — needs the row swap.
        let x = solve_partial_pivot(&[0.0, 1.0, 1.0, 0.0], 2, &[3.0, 4.0]).unwrap();
        assert_close(&x, &[4.0, 3.0]);
        let x = solve_partial_pivot(&SPD3, 3, &[1.0, 2.0, 3.0]).unwrap();
        assert_close(&x, &[343.0 / 12.0, -23.0 / 3.0, 4.0 / 3.0]);
        assert!(solve_partial_pivot(&[1.0, 2.0, 2.0, 4.0], 2, &[1.0, 2.0]).is_none());
        assert!(solve_partial_pivot(&[0.0, 0.0, 0.0, 0.0], 2, &[1.0, 2.0]).is_none());
        assert_eq!(solve_partial_pivot(&[], 0, &[]), Some(vec![]));
    }

    #[test]
    fn householder_least_squares_overdetermined() {
        // numpy.linalg.lstsq([[1, 0], [1, 1], [1, 2], [1, 3]], [1, 3, 2, 5])[0] = [1.1, 1.1]
        let a = [1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0];
        let x = lstsq_householder(&a, 4, 2, &[1.0, 3.0, 2.0, 5.0]).unwrap();
        assert_close(&x, &[1.1, 1.1]);
        // Square, consistent: the exact solution.
        let x = lstsq_householder(&SPD3, 3, 3, &[1.0, 2.0, 3.0]).unwrap();
        assert!((x[0] - 343.0 / 12.0).abs() < 1e-9, "{x:?}");
        // Rank deficient (second column = 2 × first): `None`.
        assert!(
            lstsq_householder(&[1.0, 2.0, 2.0, 4.0, 3.0, 6.0], 3, 2, &[1.0, 2.0, 3.0]).is_none()
        );
        // Shape errors: more unknowns than rows, or a mis-sized right-hand side.
        assert!(lstsq_householder(&[1.0, 2.0], 1, 2, &[1.0]).is_none());
        assert!(lstsq_householder(&a, 4, 2, &[1.0, 2.0]).is_none());
    }

    #[test]
    fn layout_and_products() {
        let rows = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let flat = flatten(&rows);
        assert_eq!(flat, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(to_rows(&flat, 2, 3), rows);
        assert_eq!(matvec(&flat, 2, 3, &[1.0, 1.0, 1.0]), [6.0, 15.0]);
        assert_eq!(dot(&[1.0, 2.0], &[3.0, 4.0]), 11.0);
        assert_eq!(
            dot(&[1.0, 2.0, 3.0], &[3.0, 4.0]),
            11.0,
            "zip stops at the shorter"
        );
    }
}
