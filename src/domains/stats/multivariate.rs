//! The multivariate normal distribution with symbolic parameters, and
//! exact multivariate descriptive statistics on data (covariance and
//! correlation matrices, principal components).
//!
//! [`Distribution`] is a distribution *on the line*, so the multivariate
//! normal is its own type: [`MultivariateNormal`] holds a mean vector of
//! expressions and a symbolic covariance [`Matrix`], and answers the
//! closed-form questions exactly — density, Mahalanobis distance,
//! marginals (which are again multivariate normals, or a univariate
//! [`Normal`](super::Normal) through [`marginal_1d`](MultivariateNormal::marginal_1d)),
//! conditionals by the Schur complement, and affine images.  SymPy:
//! `sympy.stats.MultivariateNormal`.
//!
//! ```
//! use symplex::prelude::*;
//! use symplex::stats::multivariate::MultivariateNormal;
//!
//! let ctx = Context::new();
//! let mvn = MultivariateNormal::try_new(
//!     vec![ctx.int(1), ctx.int(2)],
//!     matrix![ctx, [2, 1], [1, 3]],
//! )?;
//! // X₀ | X₁ = x₁ ~ Normal(1 + (x₁ − 2)/3, √(5/3))
//! let x1 = ctx.symbol("x1");
//! let cond = mvn.conditional(&[(1, x1.clone())])?;
//! assert_eq!(cond.mean[0].equals(&(ctx.int(1) + (&x1 - 2) / 3)), Some(true));
//! assert_eq!(cond.cov[(0, 0)], ctx.rational(5, 3));
//! # Ok::<(), SymplexError>(())
//! ```
//!
//! # Conventions
//!
//! * Data matrices are `&[Vec<Q>]` with **rows = observations** and
//!   **columns = variables** (`numpy.cov(data, rowvar=False)`).
//! * [`covariance_matrix`] takes a [`Ddof`]: `Sample` divides by `n − 1`
//!   (`numpy.cov` default, `ddof=1`), `Population` by `n`.
//! * Principal components are unit vectors; each is signed so that its
//!   first non-zero coordinate is positive (`numpy.linalg.eigh` returns an
//!   arbitrary sign, so compare up to sign).  Components are listed in
//!   order of decreasing eigenvalue.

use num_traits::{Signed, Zero};

use crate::api::context::Context;
use crate::api::expr::Ex;
use crate::base::dense_f64::{self, EigenOpts, EigenTol, OnExhaust};
use crate::base::errors::SymplexError;
use crate::domains::exact_matrix::QMatrix;
use crate::domains::matrix::Matrix;

use super::data::{self, Ddof, Q};
use super::family::Distribution;
use super::sample::Rng;

use super::common::invalid;

fn failed(op: &'static str, reason: impl Into<String>) -> SymplexError {
    SymplexError::computation_failed(op, reason)
}

/// What [`MultivariateNormal::check_parameters`] does with a singular
/// positive-semidefinite covariance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Singular {
    /// Refuse it (`try_new`: the distribution has no density).
    Refuse,
    /// Let it through for the caller's inverse, determinant or Cholesky
    /// factor to report (`ComputationFailed`), or to give the limit (the
    /// entropy `−∞`).
    Defer,
}

// ═══════════════════════════════════════════════════════════════════════════
// MultivariateNormal
// ═══════════════════════════════════════════════════════════════════════════

/// `N_k(μ, Σ)`: the multivariate normal distribution with density
///
/// `f(x) = exp(−½ (x−μ)ᵀ Σ⁻¹ (x−μ)) / √((2π)ᵏ det Σ)`
///
/// for a symmetric positive-definite covariance `Σ`.  Parameters are
/// expressions, so a symbolic `μ` or `Σ` gives symbolic answers; a
/// rational `Σ` is checked for positive-definiteness exactly by
/// [`try_new`](Self::try_new).
#[derive(Clone, Debug, PartialEq)]
pub struct MultivariateNormal {
    /// Mean vector `μ` (length `k`).
    pub mean: Vec<Ex>,
    /// Covariance matrix `Σ` (`k × k`, symmetric positive definite).
    pub cov: Matrix,
}

impl MultivariateNormal {
    /// `N_k(μ, Σ)` after validating the parameters.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `mean` is empty, `cov` is not
    /// `k × k` for `k = mean.len()`, `cov` is provably not symmetric, or
    /// `cov` is not positive definite — decided exactly (`L·D·Lᵀ`) for a
    /// rational `cov`, and by Sylvester's criterion for a symbolic one
    /// (an undecidable sign is the caller's promise and is accepted).
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::stats::multivariate::MultivariateNormal;
    ///
    /// let ctx = Context::new();
    /// let ok = MultivariateNormal::try_new(vec![ctx.int(0), ctx.int(0)], matrix![ctx, [2, 1], [1, 3]]);
    /// assert!(ok.is_ok());
    /// let indefinite = MultivariateNormal::try_new(vec![ctx.int(0), ctx.int(0)], matrix![ctx, [1, 2], [2, 1]]);
    /// assert!(indefinite.is_err());
    /// ```
    pub fn try_new(mean: Vec<Ex>, cov: Matrix) -> Result<Self, SymplexError> {
        const OP: &str = "MultivariateNormal::try_new";
        let mvn = MultivariateNormal { mean, cov };
        mvn.check_parameters(OP, Singular::Refuse)?;
        Ok(mvn)
    }

    /// `N_k(μ, Σ)` without validation (see [`try_new`](Self::try_new)).
    /// The accessors that need a valid covariance —
    /// [`precision`](Self::precision), [`mahalanobis`](Self::mahalanobis),
    /// [`density`](Self::density), [`entropy`](Self::entropy),
    /// [`sample`](Self::sample) — run `try_new`'s checks themselves and
    /// refuse an asymmetric or indefinite `Σ` with
    /// [`SymplexError::InvalidArgument`], rather than return, say, the
    /// imaginary "density" `exp(1/6)/√(−12π²)` of `Σ = [[1, 2], [2, 1]]`;
    /// a singular positive-semidefinite `Σ` (a degenerate [`affine`](Self::affine)
    /// image) fails them with [`SymplexError::ComputationFailed`] as before.
    pub fn new(mean: Vec<Ex>, cov: Matrix) -> Self {
        MultivariateNormal { mean, cov }
    }

    /// [`try_new`](Self::try_new)'s checks on the current parameters: a
    /// `k × k` covariance (`k ≥ 1`), not provably asymmetric, and positive
    /// definite — exactly (`L·D·Lᵀ`) when rational, by Sylvester's criterion
    /// otherwise (an undecidable sign is the caller's promise).  With
    /// [`Singular::Defer`] a singular `Σ` that is not provably indefinite
    /// passes, for the caller's inverse or factorisation to report.
    fn check_parameters(&self, op: &'static str, singular: Singular) -> Result<(), SymplexError> {
        let (k, cov) = (self.dim(), &self.cov);
        if k == 0 {
            return Err(invalid(
                op,
                "the mean vector must have at least one coordinate",
            ));
        }
        if cov.shape() != (k, k) {
            return Err(invalid(
                op,
                format!(
                    "the covariance must be {k}×{k} for a mean of length {k}, got {}×{}",
                    cov.nrows(),
                    cov.ncols()
                ),
            ));
        }
        if cov.is_symmetric() == Some(false) {
            return Err(invalid(op, "the covariance matrix must be symmetric"));
        }
        let defer = singular == Singular::Defer;
        match QMatrix::try_from(cov) {
            Ok(q) => match q.ldl_psd() {
                Some((_, d)) if defer || d.iter().all(|v| v.is_positive()) => Ok(()),
                _ => Err(invalid(
                    op,
                    "the covariance matrix must be positive definite (exact L·D·Lᵀ test failed)",
                )),
            },
            Err(_) => {
                if cov.is_positive_definite() == Some(false)
                    && !(defer
                        && cov
                            .det()
                            .is_ok_and(|d| d.simplify().is_zero() == Some(true)))
                {
                    return Err(invalid(
                        op,
                        "the covariance matrix must be positive definite (a leading minor is not positive)",
                    ));
                }
                Ok(())
            }
        }
    }

    /// The dimension `k`.
    pub fn dim(&self) -> usize {
        self.mean.len()
    }

    /// The context the parameters live in.
    pub fn context(&self) -> Context {
        self.cov.context()
    }

    fn check_point(&self, op: &'static str, x: &[Ex]) -> Result<(), SymplexError> {
        if x.len() != self.dim() {
            return Err(invalid(
                op,
                format!(
                    "expected a point of dimension {}, got {}",
                    self.dim(),
                    x.len()
                ),
            ));
        }
        Ok(())
    }

    /// The precision matrix `Σ⁻¹`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if the parameters fail
    /// [`try_new`](Self::try_new)'s checks (an unchecked
    /// [`new`](Self::new) accepted them); [`SymplexError::ComputationFailed`]
    /// if `Σ` is singular.
    pub fn precision(&self) -> Result<Matrix, SymplexError> {
        self.precision_for("MultivariateNormal::precision")
    }

    /// [`precision`](Self::precision) with errors naming `op`.
    fn precision_for(&self, op: &'static str) -> Result<Matrix, SymplexError> {
        self.check_parameters(op, Singular::Defer)?;
        self.cov
            .inv()
            .map_err(|e| failed(op, format!("the covariance matrix is not invertible: {e}")))
    }

    /// The quadratic form `(x−μ)ᵀ Σ⁻¹ (x−μ)`, simplified.
    fn quadratic_form(&self, op: &'static str, x: &[Ex]) -> Result<Ex, SymplexError> {
        let prec = self.precision_for(op)?;
        let d: Vec<Ex> = x.iter().zip(&self.mean).map(|(xi, mi)| xi - mi).collect();
        let mut acc = self.context().zero();
        for (i, di) in d.iter().enumerate() {
            for (j, dj) in d.iter().enumerate() {
                acc += di * prec.get(i, j) * dj;
            }
        }
        Ok(acc.simplify())
    }

    /// The squared Mahalanobis distance `(x−μ)ᵀ Σ⁻¹ (x−μ)` of a point.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `x` has the wrong length or the
    /// parameters fail [`try_new`](Self::try_new)'s checks;
    /// [`SymplexError::ComputationFailed`] if `Σ` is singular.
    pub fn mahalanobis_squared(&self, x: &[Ex]) -> Result<Ex, SymplexError> {
        const OP: &str = "MultivariateNormal::mahalanobis_squared";
        self.check_point(OP, x)?;
        self.quadratic_form(OP, x)
    }

    /// The Mahalanobis distance `√((x−μ)ᵀ Σ⁻¹ (x−μ))` of a point.
    /// `scipy.spatial.distance.mahalanobis(x, μ, Σ⁻¹)`.
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::stats::multivariate::MultivariateNormal;
    ///
    /// let ctx = Context::new();
    /// let mvn = MultivariateNormal::try_new(vec![ctx.int(1), ctx.int(2)], matrix![ctx, [2, 1], [1, 3]])?;
    /// // Σ⁻¹ = [[3, −1], [−1, 2]]/5 and x − μ = (1, 1): d² = 3/5
    /// assert_eq!(mvn.mahalanobis_squared(&[ctx.int(2), ctx.int(3)])?, ctx.rational(3, 5));
    /// # Ok::<(), SymplexError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// As [`mahalanobis_squared`](Self::mahalanobis_squared).
    pub fn mahalanobis(&self, x: &[Ex]) -> Result<Ex, SymplexError> {
        Ok(self.mahalanobis_squared(x)?.sqrt())
    }

    /// The density `exp(−½ (x−μ)ᵀ Σ⁻¹ (x−μ)) / √((2π)ᵏ det Σ)` at a point
    /// (coordinates may be symbols).  SymPy: `density(X)(x₁, …, xₖ)`;
    /// scipy: `multivariate_normal(mean, cov).pdf(x)`.
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::stats::multivariate::MultivariateNormal;
    ///
    /// let ctx = Context::new();
    /// let mvn = MultivariateNormal::try_new(vec![ctx.int(1), ctx.int(2)], matrix![ctx, [2, 1], [1, 3]])?;
    /// // At the mean: 1/√((2π)² · 5) = 1/(2π√5)
    /// let at_mean = mvn.density(&[ctx.int(1), ctx.int(2)])?;
    /// assert_eq!(at_mean.equals(&(ctx.one() / (2 * ctx.pi() * ctx.int(5).sqrt()))), Some(true));
    /// # Ok::<(), SymplexError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `x` has the wrong length or the
    /// parameters fail [`try_new`](Self::try_new)'s checks (an unchecked
    /// [`new`](Self::new) accepted an indefinite or asymmetric `Σ`);
    /// [`SymplexError::ComputationFailed`] if `Σ` is singular.
    pub fn density(&self, x: &[Ex]) -> Result<Ex, SymplexError> {
        const OP: &str = "MultivariateNormal::density";
        self.check_point(OP, x)?;
        let ctx = self.context();
        let quad = self.quadratic_form(OP, x)?;
        let det = self
            .cov
            .det()
            .map_err(|e| failed(OP, format!("determinant of the covariance: {e}")))?
            .simplify();
        let k = i64::try_from(self.dim()).map_err(|_| invalid(OP, "dimension too large"))?;
        let two_pi_k = (ctx.int(2) * ctx.pi()).powi(k);
        Ok((-quad / 2).exp() / (two_pi_k * det).sqrt())
    }

    /// The differential entropy `½ ln((2πe)ᵏ det Σ)` in nats.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if the parameters fail
    /// [`try_new`](Self::try_new)'s checks; [`SymplexError::ComputationFailed`]
    /// if the determinant cannot be computed.
    pub fn entropy(&self) -> Result<Ex, SymplexError> {
        const OP: &str = "MultivariateNormal::entropy";
        self.check_parameters(OP, Singular::Defer)?;
        let ctx = self.context();
        let det = self
            .cov
            .det()
            .map_err(|e| failed(OP, format!("determinant of the covariance: {e}")))?
            .simplify();
        let k = i64::try_from(self.dim()).map_err(|_| invalid(OP, "dimension too large"))?;
        Ok(ctx.rational(1, 2) * ((ctx.int(2) * ctx.pi() * ctx.e()).powi(k) * det).ln())
    }

    fn check_indices(&self, op: &'static str, indices: &[usize]) -> Result<(), SymplexError> {
        if indices.is_empty() {
            return Err(invalid(op, "at least one coordinate is needed"));
        }
        for (pos, &i) in indices.iter().enumerate() {
            if i >= self.dim() {
                return Err(invalid(
                    op,
                    format!(
                        "coordinate index {i} out of range for dimension {}",
                        self.dim()
                    ),
                ));
            }
            if indices[..pos].contains(&i) {
                return Err(invalid(op, format!("coordinate index {i} listed twice")));
            }
        }
        Ok(())
    }

    /// The marginal distribution of the coordinates `indices`, in the
    /// order given: `N(μ_I, Σ_II)`.  SymPy: `marginal_distribution(X, …)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `indices` is empty, repeats an
    /// index, or names a coordinate out of range.
    pub fn marginal(&self, indices: &[usize]) -> Result<MultivariateNormal, SymplexError> {
        const OP: &str = "MultivariateNormal::marginal";
        self.check_indices(OP, indices)?;
        let mean = indices.iter().map(|&i| self.mean[i].clone()).collect();
        let cov = self
            .cov
            .extract(indices, indices)
            .map_err(|e| failed(OP, e.to_string()))?;
        Ok(MultivariateNormal { mean, cov })
    }

    /// The marginal of one coordinate as a univariate
    /// [`Normal`](super::Normal)`(μᵢ, √Σᵢᵢ)`.
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::stats::multivariate::MultivariateNormal;
    ///
    /// let ctx = Context::new();
    /// let mvn = MultivariateNormal::try_new(vec![ctx.int(1), ctx.int(2)], matrix![ctx, [2, 1], [1, 3]])?;
    /// let x1 = mvn.marginal_1d(1)?;
    /// assert_eq!(x1.mean(), ctx.int(2));
    /// assert_eq!(x1.variance(), ctx.int(3));
    /// # Ok::<(), SymplexError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `i` is out of range.
    pub fn marginal_1d(&self, i: usize) -> Result<Distribution, SymplexError> {
        self.check_indices("MultivariateNormal::marginal_1d", &[i])?;
        let std = self.cov.get(i, i).sqrt().simplify();
        Ok(Distribution::normal(self.mean[i].clone(), std))
    }

    /// The conditional distribution of the remaining coordinates given
    /// `X_B = x_B` for the listed `(index, value)` pairs, by the Schur
    /// complement:
    ///
    /// `μ_{A|B} = μ_A + Σ_AB Σ_BB⁻¹ (x_B − μ_B)`,
    /// `Σ_{A|B} = Σ_AA − Σ_AB Σ_BB⁻¹ Σ_BA`.
    ///
    /// The result's coordinates are the *unconditioned* indices in
    /// ascending order (conditioning on `X₁` in a 3-D normal leaves
    /// `(X₀, X₂)`).
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `given` is empty, repeats an
    /// index, names a coordinate out of range, or conditions on every
    /// coordinate; [`SymplexError::ComputationFailed`] if `Σ_BB` is
    /// singular.
    pub fn conditional(&self, given: &[(usize, Ex)]) -> Result<MultivariateNormal, SymplexError> {
        const OP: &str = "MultivariateNormal::conditional";
        let b: Vec<usize> = given.iter().map(|(i, _)| *i).collect();
        self.check_indices(OP, &b)?;
        let a: Vec<usize> = (0..self.dim()).filter(|i| !b.contains(i)).collect();
        if a.is_empty() {
            return Err(invalid(
                OP,
                "conditioning on every coordinate leaves nothing to distribute",
            ));
        }
        let wrap = |e: SymplexError| failed(OP, e.to_string());
        let s_aa = self.cov.extract(&a, &a).map_err(wrap)?;
        let s_ab = self.cov.extract(&a, &b).map_err(wrap)?;
        let s_ba = self.cov.extract(&b, &a).map_err(wrap)?;
        let s_bb = self.cov.extract(&b, &b).map_err(wrap)?;
        // K = Σ_AB Σ_BB⁻¹
        let k = s_ab.matmul(&s_bb.inv().map_err(wrap)?).map_err(wrap)?;
        // x_B − μ_B as a column
        let shift = Matrix::col_vector(
            given
                .iter()
                .map(|(i, v)| (v - &self.mean[*i]).simplify())
                .collect(),
        )
        .map_err(wrap)?;
        let mean_shift = k.matmul(&shift).map_err(wrap)?;
        let mean = a
            .iter()
            .enumerate()
            .map(|(pos, &i)| (&self.mean[i] + mean_shift.get(pos, 0)).simplify())
            .collect();
        let cov = s_aa
            .sub(&k.matmul(&s_ba).map_err(wrap)?)
            .map_err(wrap)?
            .simplify();
        Ok(MultivariateNormal { mean, cov })
    }

    /// The image `Y = A X + b` for an `m × k` matrix `A` and a shift `b`
    /// of length `m`: `N_m(Aμ + b, A Σ Aᵀ)`.  The image is degenerate
    /// (singular covariance) when `A` does not have full row rank; it is
    /// returned unchecked and its [`density`](Self::density) then fails.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `A` has `≠ k` columns or `b`
    /// has `≠ m` entries.
    pub fn affine(&self, a: &Matrix, b: &[Ex]) -> Result<MultivariateNormal, SymplexError> {
        const OP: &str = "MultivariateNormal::affine";
        if a.ncols() != self.dim() {
            return Err(invalid(
                OP,
                format!(
                    "A has {} columns but the distribution has dimension {}",
                    a.ncols(),
                    self.dim()
                ),
            ));
        }
        if b.len() != a.nrows() {
            return Err(invalid(
                OP,
                format!("b has {} entries but A has {} rows", b.len(), a.nrows()),
            ));
        }
        let wrap = |e: SymplexError| failed(OP, e.to_string());
        let mu = Matrix::col_vector(self.mean.clone()).map_err(wrap)?;
        let a_mu = a.matmul(&mu).map_err(wrap)?;
        let mean = b
            .iter()
            .enumerate()
            .map(|(i, bi)| (a_mu.get(i, 0) + bi).simplify())
            .collect();
        let cov = a
            .matmul(&self.cov)
            .map_err(wrap)?
            .matmul(&a.transpose())
            .map_err(wrap)?
            .simplify();
        Ok(MultivariateNormal { mean, cov })
    }

    /// `n` samples as rows, by `x = μ + L z` with `Σ = L Lᵀ` the Cholesky
    /// factor (in `f64`) and `z` standard normal (through the crate's
    /// [`Normal`](super::Normal) sampler).  Deterministic for a given
    /// [`Rng`] seed.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if the parameters fail
    /// [`try_new`](Self::try_new)'s checks, or `n` rows cannot be stored
    /// (more memory than the allocator grants);
    /// [`SymplexError::Unevaluable`] if a parameter is symbolic;
    /// [`SymplexError::ComputationFailed`] if `Σ` is not numerically
    /// positive definite.
    pub fn sample(&self, n: usize, rng: &mut Rng) -> Result<Vec<Vec<f64>>, SymplexError> {
        const OP: &str = "MultivariateNormal::sample";
        self.check_parameters(OP, Singular::Defer)?;
        let k = self.dim();
        let mean: Vec<f64> = self
            .mean
            .iter()
            .map(Ex::eval_f64)
            .collect::<Result<_, _>>()?;
        let cov = self.cov.eval_f64()?;
        let l = dense_f64::cholesky(&dense_f64::flatten(&cov), k, 0.0)
            .ok_or_else(|| failed(OP, "the covariance is not numerically positive definite"))?;
        let ctx = self.context();
        let mut z = Distribution::normal(ctx.zero(), ctx.one()).sampler()?;
        let mut out = Vec::new();
        out.try_reserve_exact(n)
            .map_err(|e| invalid(OP, format!("cannot store {n} samples: {e}")))?;
        for _ in 0..n {
            let zs: Vec<f64> = (0..k).map(|_| z(rng)).collect();
            let x: Vec<f64> = (0..k)
                .map(|i| mean[i] + dense_f64::dot(&l[i * k..i * k + i + 1], &zs))
                .collect();
            out.push(x);
        }
        Ok(out)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Covariance / correlation matrices from data
// ═══════════════════════════════════════════════════════════════════════════

/// Validate a data matrix (rows = observations) and return `(n, p)`.
fn check_data(op: &'static str, data: &[Vec<Q>]) -> Result<(usize, usize), SymplexError> {
    let n = data.len();
    if n == 0 {
        return Err(invalid(op, "no observations"));
    }
    let p = data[0].len();
    if p == 0 {
        return Err(invalid(op, "observations have no variables"));
    }
    if let Some((i, row)) = data.iter().enumerate().find(|(_, r)| r.len() != p) {
        return Err(invalid(
            op,
            format!("observation {i} has {} variables, expected {p}", row.len()),
        ));
    }
    Ok((n, p))
}

fn column(data: &[Vec<Q>], j: usize) -> Vec<Q> {
    data.iter().map(|row| row[j].clone()).collect()
}

/// The exact `p × p` covariance matrix of `p` variables observed `n`
/// times (rows of `data` are observations).  `numpy.cov(data,
/// rowvar=False, ddof=1)` for [`Ddof::Sample`], `ddof=0` for
/// [`Ddof::Population`].
///
/// ```
/// use symplex::linprog::{q, qi};
/// use symplex::stats::data::Ddof;
/// use symplex::stats::multivariate::covariance_matrix;
///
/// // numpy.cov([[1, 2], [2, 4], [3, 7]], rowvar=False) = [[1, 2.5], [2.5, 6.333…]]
/// let data = vec![vec![qi(1), qi(2)], vec![qi(2), qi(4)], vec![qi(3), qi(7)]];
/// let c = covariance_matrix(&data, Ddof::Sample)?;
/// assert_eq!(c[(0, 1)], q(5, 2));
/// assert_eq!(c[(1, 1)], q(19, 3));
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] if `data` is empty or jagged, or has
/// fewer than two observations for [`Ddof::Sample`].
pub fn covariance_matrix(data: &[Vec<Q>], ddof: Ddof) -> Result<QMatrix, SymplexError> {
    let (_, p) = check_data("covariance_matrix", data)?;
    let cols: Vec<Vec<Q>> = (0..p).map(|j| column(data, j)).collect();
    let mut rows = vec![vec![Q::zero(); p]; p];
    for i in 0..p {
        for j in i..p {
            let c = data::covariance(&cols[i], &cols[j], ddof)?;
            rows[i][j] = c.clone();
            rows[j][i] = c;
        }
    }
    QMatrix::new(rows)
}

/// The exact `p × p` Pearson correlation matrix of the variables in
/// `data` (rows = observations): `rᵢⱼ = cov(xᵢ, xⱼ) / (σᵢ σⱼ)`, with the
/// square roots kept symbolic and the diagonal exactly `1`.
/// `numpy.corrcoef(data, rowvar=False)`.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] if `data` is empty or jagged, or a
/// variable is constant (its correlation is undefined — even with itself,
/// so a single constant variable is refused too rather than given the
/// diagonal `1`; `numpy.corrcoef` returns `nan`).
pub fn correlation_matrix(ctx: &Context, data: &[Vec<Q>]) -> Result<Matrix, SymplexError> {
    const OP: &str = "correlation_matrix";
    let (_, p) = check_data(OP, data)?;
    let cols: Vec<Vec<Q>> = (0..p).map(|j| column(data, j)).collect();
    if let Some(j) = cols.iter().position(|c| c.iter().all(|v| v == &c[0])) {
        return Err(invalid(
            OP,
            format!("variable {j} is constant, so its correlations are undefined"),
        ));
    }
    let mut rows = vec![vec![ctx.one(); p]; p];
    for i in 0..p {
        for j in (i + 1)..p {
            let r = data::pearson(ctx, &cols[i], &cols[j])?;
            rows[i][j] = r.clone();
            rows[j][i] = r;
        }
    }
    Matrix::new(rows)
}

// ═══════════════════════════════════════════════════════════════════════════
// Principal components
// ═══════════════════════════════════════════════════════════════════════════

/// Exact principal components of a covariance matrix, from [`pca`].
#[derive(Clone, Debug, PartialEq)]
pub struct Pca {
    /// Eigenvalues of the covariance in decreasing order (the variances
    /// along the components).
    pub eigenvalues: Vec<Ex>,
    /// The unit eigenvectors, one per eigenvalue, each signed so that its
    /// first non-zero coordinate is positive.
    pub components: Vec<Vec<Ex>>,
    /// `λᵢ / Σⱼ λⱼ = λᵢ / tr Σ`.
    pub explained_variance_ratio: Vec<Ex>,
}

/// Exact principal component analysis of a rational covariance matrix:
/// the eigen-decomposition of `Σ` by [`Matrix::eigenvects`] on the
/// characteristic polynomial, with eigenvectors orthonormalised exactly
/// (Gram–Schmidt inside each eigenspace) and sorted by decreasing
/// eigenvalue.  `numpy.linalg.eigh(cov)` (which lists eigenvalues in
/// *increasing* order and with arbitrary signs).
///
/// Exact for small dimensions: a `2 × 2` covariance has eigenvalues
/// `(tr ± √(tr² − 4 det))/2`, and any matrix whose characteristic
/// polynomial factors over ℚ gives rational or quadratic-surd eigenvalues.
/// When the characteristic polynomial has an irreducible factor of degree
/// `≥ 3` (the generic `3 × 3` case is the *casus irreducibilis*: three
/// real roots not expressible in real radicals; degree `≥ 5` is not
/// solvable at all) an eigenvalue comes back as a `RootOf` — still exact,
/// still ordered and evaluable, but not a radical.  Use [`pca_f64`] for
/// plain floats there.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::multivariate::pca;
///
/// let ctx = Context::new();
/// // eigh([[2, 1], [1, 2]]) = eigenvalues (1, 3), components (1, ±1)/√2
/// let cov = QMatrix::from_i64(&[&[2, 1], &[1, 2]])?;
/// let p = pca(&ctx, &cov)?;
/// assert_eq!(p.eigenvalues, vec![ctx.int(3), ctx.int(1)]);
/// assert_eq!(p.explained_variance_ratio, vec![ctx.rational(3, 4), ctx.rational(1, 4)]);
/// assert_eq!(p.components[0][0].equals(&(ctx.one() / ctx.int(2).sqrt())), Some(true));
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] if `cov` is not symmetric positive
/// semidefinite; [`SymplexError::ComputationFailed`] if the eigenvalues
/// cannot be found, ordered numerically, or an eigenspace is defective.
pub fn pca(ctx: &Context, cov: &QMatrix) -> Result<Pca, SymplexError> {
    const OP: &str = "pca";
    if !cov.is_symmetric() {
        return Err(invalid(OP, "the covariance matrix must be symmetric"));
    }
    if !cov.is_positive_semidefinite() {
        return Err(invalid(
            OP,
            "the covariance matrix must be positive semidefinite",
        ));
    }
    let n = cov.nrows();
    let trace = cov.diagonal().iter().fold(Q::zero(), |acc, v| acc + v);
    if trace.is_zero() {
        return Err(invalid(OP, "the covariance matrix is zero"));
    }
    let m = cov.to_matrix(ctx);
    let eigen = m
        .eigenvects()
        .map_err(|e| failed(OP, format!("eigen-decomposition failed: {e}")))?;
    // Decreasing eigenvalue, decided exactly by insertion: `f64`
    // approximations would tie `1` and `1 − 10⁻²⁰` and leave them in
    // `eigenvects`' order (ascending, for that pair).
    let mut items: Vec<(Ex, Vec<Ex>)> = Vec::with_capacity(n);
    for (value, mult, vecs) in eigen {
        if vecs.len() != mult {
            return Err(failed(
                OP,
                format!(
                    "eigenvalue {value} has geometric multiplicity {} < algebraic {mult}",
                    vecs.len()
                ),
            ));
        }
        value
            .eval_f64()
            .map_err(|e| failed(OP, format!("cannot order eigenvalue {value}: {e}")))?;
        let raw: Vec<Vec<Ex>> = vecs.iter().map(|v| v.col(0)).collect();
        for v in gram_schmidt(ctx, &raw) {
            let mut pos = items.len();
            while pos > 0 {
                let before = &items[pos - 1].0;
                match before.compare_numeric(&value) {
                    Some(std::cmp::Ordering::Less) => pos -= 1,
                    Some(_) => break,
                    None => {
                        return Err(failed(
                            OP,
                            format!("cannot order the eigenvalues {before} and {value}"),
                        ));
                    }
                }
            }
            items.insert(pos, (value.clone(), v));
        }
    }
    if items.len() != n {
        return Err(failed(
            OP,
            format!("found {} eigenvectors for dimension {n}", items.len()),
        ));
    }
    let trace_ex = ctx.from_ratio(trace);
    Ok(Pca {
        explained_variance_ratio: items
            .iter()
            .map(|(v, _)| (v / &trace_ex).simplify())
            .collect(),
        eigenvalues: items.iter().map(|(v, _)| v.clone()).collect(),
        components: items.into_iter().map(|(_, c)| c).collect(),
    })
}

/// Exact Gram–Schmidt: orthonormal vectors spanning the same space, each
/// signed so its first non-zero coordinate is positive.
fn gram_schmidt(ctx: &Context, vecs: &[Vec<Ex>]) -> Vec<Vec<Ex>> {
    let dot = |a: &[Ex], b: &[Ex]| -> Ex {
        a.iter()
            .zip(b)
            .fold(ctx.zero(), |acc, (x, y)| acc + x * y)
            .simplify()
    };
    // Surds from the nullspace come as `1/(a + b√d)`: rationalise first so
    // the norm and the unit vector stay readable.
    let tidy = |e: &Ex| e.rationalize_denom().simplify();
    let mut basis: Vec<Vec<Ex>> = Vec::with_capacity(vecs.len());
    for v in vecs {
        let mut w: Vec<Ex> = v.iter().map(tidy).collect();
        for u in &basis {
            let proj = dot(&w, u);
            w = w
                .iter()
                .zip(u)
                .map(|(wi, ui)| tidy(&(wi - &proj * ui)))
                .collect();
        }
        let norm = dot(&w, &w).sqrt().simplify();
        if norm.is_zero() == Some(true) {
            continue;
        }
        let mut unit: Vec<Ex> = w.iter().map(|wi| tidy(&(wi / &norm))).collect();
        let negative_lead = unit
            .iter()
            .find(|c| c.is_zero() != Some(true))
            .is_some_and(is_negative_constant);
        if negative_lead {
            unit = unit.iter().map(|c| (-c).simplify()).collect();
        }
        basis.push(unit);
    }
    basis
}

/// Is a constant expression negative?  The assumption system first, a
/// numeric evaluation when it cannot decide a nest of radicals.
fn is_negative_constant(e: &Ex) -> bool {
    match e.is_negative() {
        Some(b) => b,
        None => e.eval_f64().is_ok_and(|v| v < 0.0),
    }
}

/// Floating-point principal components, from [`pca_f64`].
#[derive(Clone, Debug, PartialEq)]
pub struct PcaF64 {
    /// Eigenvalues in decreasing order.
    pub eigenvalues: Vec<f64>,
    /// Unit eigenvectors, one per eigenvalue, first non-zero coordinate
    /// positive.
    pub components: Vec<Vec<f64>>,
    /// `λᵢ / Σⱼ λⱼ`.
    pub explained_variance_ratio: Vec<f64>,
}

/// Principal components of a symmetric matrix in `f64` by the cyclic
/// Jacobi eigenvalue algorithm (Golub & Van Loan, *Matrix Computations*,
/// §8.5): deterministic, accurate to roughly machine precision, and
/// suited to the dimensions where the exact [`pca`] no longer closes.
/// `numpy.linalg.eigh(cov)` up to ordering and signs.
///
/// The matrix is first scaled by a power of two (exact) so that its
/// largest entry is near `1`, as LAPACK's `dsyev` scales outside a safe
/// range: the sweeps' Frobenius-norm test squares the entries, which
/// underflowed to `0` for a covariance of order `10⁻¹⁷⁰` (or overflowed
/// for `10¹⁷⁰`) and returned the unrotated diagonal as the eigenvalues.
/// The explained variance ratios are `λᵢ / tr Σ`; an indefinite matrix
/// (a covariance assembled from rounded pieces) is not projected, so its
/// negative eigenvalues give negative ratios.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] if `cov` is empty, not square, has a
/// non-finite entry, is not symmetric (to `1e-12` relative to its largest
/// entry), or has a trace `≤ 0` (no total variance to take ratios of —
/// the zero matrix, as for [`pca`]); [`SymplexError::ComputationFailed`]
/// if the sweeps do not converge or an eigenvalue overflows `f64` when
/// scaled back.
pub fn pca_f64(cov: &[Vec<f64>]) -> Result<PcaF64, SymplexError> {
    const OP: &str = "pca_f64";
    let n = cov.len();
    if n == 0 {
        return Err(invalid(OP, "empty matrix"));
    }
    if cov.iter().any(|r| r.len() != n) {
        return Err(invalid(OP, "the matrix must be square"));
    }
    if let Some(v) = cov.iter().flatten().find(|v| !v.is_finite()) {
        return Err(invalid(OP, format!("non-finite entry {v}")));
    }
    let scale = cov.iter().flatten().fold(0.0_f64, |m, v| m.max(v.abs()));
    for (i, row) in cov.iter().enumerate() {
        for (j, &below) in row.iter().enumerate().take(i) {
            let above = cov[j][i];
            if (below - above).abs() > 1e-12 * scale {
                return Err(invalid(
                    OP,
                    format!("not symmetric at ({i}, {j}): {below} vs {above}"),
                ));
            }
        }
    }
    // 2^-e with 2^(e-1) ≤ scale < 2^e; the zero matrix fails the trace test.
    let e = if scale > 0.0 {
        scale.log2().floor() as i32 + 1
    } else {
        0
    };
    let to_unit = |v: f64| v * 2f64.powi(-e / 2) * 2f64.powi(-(e - e / 2));
    let scaled: Vec<f64> = cov.iter().flatten().map(|&v| to_unit(v)).collect();
    let trace: f64 = (0..n).map(|i| scaled[i * n + i]).sum();
    if trace <= 0.0 {
        return Err(invalid(
            OP,
            format!(
                "the trace (total variance) must be positive, got {}",
                trace * 2f64.powi(e)
            ),
        ));
    }
    let dense_f64::SymEigen { values, vectors } =
        dense_f64::sym_eigen(&scaled, n, &PCA_EIGEN).map_err(|e| failed(OP, e.to_string()))?;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| values[b].total_cmp(&values[a]));
    let from_unit = |v: f64| v * 2f64.powi(e / 2) * 2f64.powi(e - e / 2);
    let mut eigenvalues = Vec::with_capacity(n);
    let mut components = Vec::with_capacity(n);
    let mut ratio = Vec::with_capacity(n);
    for &idx in &order {
        let mut v: Vec<f64> = (0..n).map(|i| vectors[i * n + idx]).collect();
        if v.iter()
            .find(|c| c.abs() > 1e-12)
            .is_some_and(|lead| *lead < 0.0)
        {
            for c in &mut v {
                *c = -*c;
            }
        }
        let value = from_unit(values[idx]);
        if !value.is_finite() {
            return Err(failed(
                OP,
                format!("an eigenvalue ({} · 2^{e}) overflows f64", values[idx]),
            ));
        }
        eigenvalues.push(value);
        components.push(v);
        ratio.push(values[idx] / trace);
    }
    Ok(PcaF64 {
        eigenvalues,
        components,
        explained_variance_ratio: ratio,
    })
}

/// The Jacobi settings of [`pca_f64`]: stop when the off-diagonal norm is
/// below `1e-15 · ‖A‖_F`, and fail rather than return a partial result
/// after 100 sweeps.
const PCA_EIGEN: EigenOpts = EigenOpts {
    tol: EigenTol::RelativeFrobenius(1e-15),
    max_sweeps: 100,
    on_exhaust: OnExhaust::Error,
};

impl Pca {
    /// The explained variance ratios evaluated to `f64`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::Unevaluable`] if a ratio does not evaluate.
    pub fn explained_variance_ratio_f64(&self) -> Result<Vec<f64>, SymplexError> {
        self.explained_variance_ratio
            .iter()
            .map(Ex::eval_f64)
            .collect()
    }
}
