//! Regression: exact ordinary and weighted least squares over ℚ, and
//! logistic regression by Newton–Raphson in `f64` (statsmodels' `OLS`,
//! `WLS`, `Logit`; numpy's `polyfit`; scipy's `linregress`).
//!
//! # Design matrices
//!
//! A design is passed as **rows of observations**: `x: &[Vec<Q>]` with one
//! inner vector per observation, whose entries are the regressors
//! (**columns** of the design matrix `X`).  `add_intercept = true`
//! prepends a column of ones, so the first coefficient is the intercept
//! (`sm.add_constant(x)`).  [`Design`] is a column-wise builder for the
//! same thing: `Design::new().intercept().column(&x1).column(&x2)`.
//!
//! # Exactness
//!
//! Least squares is solved exactly through the normal equations with
//! [`QMatrix`]: `β̂ = (XᵀWX)⁻¹XᵀWy` (`W = I` for OLS).  Every quantity that
//! is a rational function of the data — coefficients, fitted values,
//! residuals, sums of squares, `R²`, `σ̂²`, the covariance matrix, `F`,
//! leverages, Cook's distances, Durbin–Watson, VIFs — is a [`Q`].
//! Standard errors, `t` statistics and p-values are exact expressions
//! ([`Ex`]: a rational times a square root, and the Student-t / F tails
//! through `betainc_regularized`); critical values and confidence limits
//! are `f64` (a Brent root of the exact CDF).  Logistic regression has no
//! closed form and is fitted numerically in `f64`.
//!
//! The maximum-likelihood fits — [`Logit`], [`MnLogit`], [`OrderedLogit`]
//! and [`CoxModel`](super::cox::CoxModel) — share two traits:
//! [`LikelihoodFit`] (information criteria, pseudo-`R²`, the
//! likelihood-ratio test) and [`WaldFit`] (`z`, p-values and Wald
//! intervals from coefficients and standard errors).  Each fit also offers
//! the same methods inherently.
//!
//! ```
//! use symplex::prelude::*;
//! use symplex::linprog::q;
//! use symplex::stats::data::from_i64;
//! use symplex::stats::regression::ols;
//!
//! let ctx = Context::new();
//! let x: Vec<Vec<Q>> = from_i64(&[1, 2, 3, 4, 5, 6, 7]).into_iter().map(|v| vec![v]).collect();
//! let y = from_i64(&[2, 3, 5, 4, 6, 8, 9]);
//! // statsmodels: OLS(y, add_constant(x)).fit().params = [0.7142857142857169, 1.1428571428571432]
//! let fit = ols(&y, &x, true)?;
//! assert_eq!(fit.coefficients, vec![q(5, 7), q(8, 7)]);
//! assert_eq!(fit.r_squared, q(64, 69));           // rsquared 0.927536231884058
//! assert_eq!(fit.f_statistic()?, q(64, 1));         // fvalue 64.0
//! assert!((fit.p_values(&ctx)?[1].eval_f64()? - 0.000_492_906_660_572_44).abs() < 1e-12);
//! # Ok::<(), SymplexError>(())
//! ```

use num_traits::{One, Signed, Zero};

use super::common::{
    WaldSummary, check_confidence, chi_squared_sf, ex, ex_usize, f_sf, information_cholesky,
    invalid, normal_two_sided, qu, t_two_sided, z_two_sided,
};
use super::data::Q;
use super::hypothesis::{self, Alternative, TestResult};
use crate::api::context::Context;
use crate::api::expr::Ex;
use crate::base::dense_f64::{self, dot as dot_f64};
use crate::base::errors::SymplexError;
use crate::base::interval::Interval;
use crate::base::numeric::ratio_to_f64;
use crate::domains::exact_matrix::QMatrix;

// ═══════════════════════════════════════════════════════════════════════════
// Small helpers
// ═══════════════════════════════════════════════════════════════════════════

fn failed(op: &'static str, reason: impl Into<String>) -> SymplexError {
    SymplexError::computation_failed(op, reason)
}

fn to_f64(op: &'static str, q: &Q) -> Result<f64, SymplexError> {
    ratio_to_f64(q).ok_or_else(|| failed(op, format!("{q} does not fit in an f64")))
}

fn dot(a: &[Q], b: &[Q]) -> Q {
    a.iter().zip(b).fold(Q::zero(), |acc, (x, y)| acc + x * y)
}

/// `vᵀ M v` for a square `M` whose dimension equals `v.len()`.
fn quadratic_form(m: &QMatrix, v: &[Q]) -> Q {
    let mut acc = Q::zero();
    for (a, va) in v.iter().enumerate() {
        for (b, vb) in v.iter().enumerate() {
            if let Some(mab) = m.try_get(a, b) {
                acc += va * mab * vb;
            }
        }
    }
    acc
}

fn column_vector(op: &'static str, v: &[Q]) -> Result<QMatrix, SymplexError> {
    QMatrix::new(v.iter().map(|q| vec![q.clone()]).collect())
        .map_err(|e| invalid(op, e.to_string()))
}

/// Two-sided Student-t tail `P(|T_ν| ≥ |t|) = I_{ν/(t²+ν)}(ν/2, ½)` for a
/// rational `t²`.
fn student_two_sided(ctx: &Context, df: usize, t_squared: &Q) -> Ex {
    if t_squared.is_zero() {
        return ctx.one();
    }
    let nu = qu(df);
    let z = &nu / (t_squared + &nu);
    ex(ctx, &z).betainc_regularized(&ex(ctx, &(nu / qu(2))), &ctx.rational(1, 2), &ctx.zero())
}

// ═══════════════════════════════════════════════════════════════════════════
// Design matrices
// ═══════════════════════════════════════════════════════════════════════════

/// Column-wise builder of a design matrix: an optional intercept followed
/// by regressor columns, all of the same length.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::linprog::q;
/// use symplex::stats::data::from_i64;
/// use symplex::stats::regression::Design;
///
/// let x = from_i64(&[1, 2, 3, 4, 5, 6, 7]);
/// let y = from_i64(&[2, 3, 5, 4, 6, 8, 9]);
/// let fit = Design::new().intercept().column(&x).fit(&y)?;
/// assert_eq!(fit.coefficients, vec![q(5, 7), q(8, 7)]);
/// # Ok::<(), SymplexError>(())
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Design {
    intercept: bool,
    columns: Vec<Vec<Q>>,
}

impl Design {
    /// An empty design (no intercept, no columns).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add the constant column (`sm.add_constant`).
    #[must_use]
    pub fn intercept(mut self) -> Self {
        self.intercept = true;
        self
    }

    /// Append a regressor column.
    #[must_use]
    pub fn column(mut self, values: &[Q]) -> Self {
        self.columns.push(values.to_vec());
        self
    }

    /// Whether the intercept column is included.
    #[must_use]
    pub fn has_intercept(&self) -> bool {
        self.intercept
    }

    /// Number of regressor columns (without the intercept).
    #[must_use]
    pub fn n_columns(&self) -> usize {
        self.columns.len()
    }

    /// The observations as rows (without the intercept), the shape
    /// [`ols`] takes.  With no columns, `n` empty rows are produced.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if the columns have different
    /// lengths.
    pub fn rows(&self, n: usize) -> Result<Vec<Vec<Q>>, SymplexError> {
        const OP: &str = "Design::rows";
        for (j, c) in self.columns.iter().enumerate() {
            if c.len() != n {
                return Err(invalid(
                    OP,
                    format!("column {j} has {} entries, expected {n}", c.len()),
                ));
            }
        }
        Ok((0..n)
            .map(|i| self.columns.iter().map(|c| c[i].clone()).collect())
            .collect())
    }

    /// Fit `y` on this design by ordinary least squares ([`ols`]).
    ///
    /// # Errors
    ///
    /// As [`ols`], plus a column-length mismatch.
    pub fn fit(&self, y: &[Q]) -> Result<Ols, SymplexError> {
        let rows = self.rows(y.len())?;
        ols(y, &rows, self.intercept)
    }

    /// Fit `y` on this design by weighted least squares ([`wls`]).
    ///
    /// # Errors
    ///
    /// As [`wls`], plus a column-length mismatch.
    pub fn fit_weighted(&self, y: &[Q], weights: &[Q]) -> Result<Ols, SymplexError> {
        let rows = self.rows(y.len())?;
        wls(y, &rows, weights, self.intercept)
    }
}

/// Build the `n × p` design matrix from observation rows, prepending the
/// column of ones when `add_intercept`.
fn build_design(
    op: &'static str,
    x: &[Vec<Q>],
    n: usize,
    add_intercept: bool,
) -> Result<QMatrix, SymplexError> {
    if x.len() != n {
        return Err(invalid(
            op,
            format!("y has {n} observations but x has {} rows", x.len()),
        ));
    }
    let k = x.first().map_or(0, Vec::len);
    if let Some((i, r)) = x.iter().enumerate().find(|(_, r)| r.len() != k) {
        return Err(invalid(
            op,
            format!("row {i} of x has {} entries, expected {k}", r.len()),
        ));
    }
    if k + usize::from(add_intercept) == 0 {
        return Err(invalid(
            op,
            "the design has no columns: pass at least one regressor or add_intercept = true",
        ));
    }
    let rows = x
        .iter()
        .map(|r| {
            let mut row = Vec::with_capacity(k + 1);
            if add_intercept {
                row.push(Q::one());
            }
            row.extend(r.iter().cloned());
            row
        })
        .collect();
    QMatrix::new(rows).map_err(|e| invalid(op, e.to_string()))
}

/// statsmodels' `k_constant` detection: a nonzero constant column, or the
/// vector of ones lying in the column space (an implicit constant, e.g. a
/// full set of dummies).
fn has_constant_column(x: &QMatrix) -> bool {
    let explicit = (0..x.ncols()).any(|j| {
        let c = x.col(j);
        c.first()
            .is_some_and(|c0| !c0.is_zero() && c.iter().all(|v| v == c0))
    });
    if explicit {
        return true;
    }
    let ones = QMatrix::new(vec![vec![Q::one()]; x.nrows()]);
    match ones.and_then(|o| QMatrix::hstack(&[&o, x])) {
        Ok(aug) => aug.rank() == x.rank(),
        Err(_) => false,
    }
}

/// Solve the (weighted) normal equations exactly: `β̂ = (XᵀWX)⁻¹XᵀWy`,
/// returning `(β̂, (XᵀWX)⁻¹)`.
fn normal_equations(
    op: &'static str,
    x: &QMatrix,
    y: &[Q],
    weights: Option<&[Q]>,
) -> Result<(Vec<Q>, QMatrix), SymplexError> {
    let p = x.ncols();
    let weight = |i: usize| {
        weights
            .and_then(|w| w.get(i).cloned())
            .unwrap_or_else(Q::one)
    };
    let wx = QMatrix::new(
        x.rows()
            .enumerate()
            .map(|(i, r)| {
                let wi = weight(i);
                r.iter().map(|v| v * &wi).collect()
            })
            .collect(),
    )
    .map_err(|e| invalid(op, e.to_string()))?;
    let wy: Vec<Q> = y.iter().enumerate().map(|(i, v)| v * weight(i)).collect();
    let xt = x.transpose();
    let xtwx = xt.matmul(&wx)?;
    let xtwy = xt.matmul(&column_vector(op, &wy)?)?;
    let xtx_inv = xtwx.inv().map_err(|_| {
        invalid(
            op,
            format!(
                "the design matrix is rank deficient (rank {} of {p} columns): drop a collinear regressor",
                x.rank()
            ),
        )
    })?;
    let beta = xtx_inv.matmul(&xtwy)?.col(0);
    Ok((beta, xtx_inv))
}

// ═══════════════════════════════════════════════════════════════════════════
// Least squares
// ═══════════════════════════════════════════════════════════════════════════

/// One row of an ANOVA-style decomposition of a least-squares fit; see
/// [`Ols::anova_table`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AnovaTable {
    /// Explained (model) sum of squares, `ESS` (`ess`).
    pub ss_model: Q,
    /// Model degrees of freedom, `p − k_constant` (`df_model`).
    pub df_model: usize,
    /// `ESS / df_model` (`mse_model`).
    pub ms_model: Q,
    /// Residual sum of squares `Σ wᵢ eᵢ²` (`ssr`).
    pub ss_resid: Q,
    /// Residual degrees of freedom `n − p` (`df_resid`).
    pub df_resid: usize,
    /// `SSR / df_resid` (`mse_resid`, the estimate `σ̂²`).
    pub ms_resid: Q,
    /// Total sum of squares (`centered_tss` with a constant, `uncentered_tss`
    /// without).
    pub ss_total: Q,
    /// `df_model + df_resid`.
    pub df_total: usize,
    /// `F = ms_model / ms_resid` (`fvalue`).
    pub f: Q,
}

/// An exact least-squares fit (statsmodels `RegressionResults` of `OLS` /
/// `WLS`).  Produced by [`ols`], [`wls`], [`simple_linear_regression`] and
/// [`Design::fit`].
///
/// The public fields are the exact rational statistics; the methods derive
/// standard errors, tests, intervals and diagnostics from them.  For a
/// weighted fit every "sum of squares" is weighted (`Σ wᵢ(·)²`), as in
/// statsmodels' `WLS`, while `fitted` and `residuals` are the plain
/// `Xβ̂` and `y − Xβ̂` (`fittedvalues`, `resid`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ols {
    /// `β̂ = (XᵀWX)⁻¹XᵀWy` (`params`), intercept first when one was added.
    pub coefficients: Vec<Q>,
    /// `ŷ = Xβ̂` (`fittedvalues`).
    pub fitted: Vec<Q>,
    /// `e = y − ŷ` (`resid`).
    pub residuals: Vec<Q>,
    /// Residual sum of squares `Σ wᵢ eᵢ²` (`ssr`).
    pub ssr: Q,
    /// Explained sum of squares `TSS − SSR` (`ess`).
    pub ess: Q,
    /// Total sum of squares: `Σ wᵢ(yᵢ − ȳ_w)²` when the design has a
    /// constant (`centered_tss`), `Σ wᵢ yᵢ²` otherwise (`uncentered_tss`).
    pub tss: Q,
    /// `R² = 1 − SSR/TSS` (`rsquared`).
    pub r_squared: Q,
    /// `1 − (n − k_constant)/(n − p) · (1 − R²)` (`rsquared_adj`).
    pub adjusted_r_squared: Q,
    /// `p − k_constant` (`df_model`).
    pub df_model: usize,
    /// `n − p` (`df_resid`).
    pub df_resid: usize,
    /// `σ̂² = SSR / (n − p)` (`mse_resid`, `scale`).
    pub mse_resid: Q,
    /// `σ̂² (XᵀWX)⁻¹` (`cov_params()`), exact.
    pub cov_params: QMatrix,
    design: QMatrix,
    y: Vec<Q>,
    weights: Option<Vec<Q>>,
    xtx_inv: QMatrix,
    has_constant: bool,
    added_intercept: bool,
}

fn fit_least_squares(
    op: &'static str,
    y: &[Q],
    x: &[Vec<Q>],
    weights: Option<&[Q]>,
    add_intercept: bool,
) -> Result<Ols, SymplexError> {
    let n = y.len();
    if n == 0 {
        return Err(invalid(op, "y is empty"));
    }
    let design = build_design(op, x, n, add_intercept)?;
    let p = design.ncols();
    if n <= p {
        return Err(invalid(
            op,
            format!("need more observations than parameters: n = {n}, p = {p}"),
        ));
    }
    if let Some(w) = weights {
        if w.len() != n {
            return Err(invalid(
                op,
                format!("weights has {} entries, expected {n}", w.len()),
            ));
        }
        if let Some((i, wi)) = w.iter().enumerate().find(|(_, wi)| !wi.is_positive()) {
            return Err(invalid(
                op,
                format!("weights must be positive, got {wi} at index {i}"),
            ));
        }
    }
    let (coefficients, xtx_inv) = normal_equations(op, &design, y, weights)?;
    let fitted: Vec<Q> = design.rows().map(|r| dot(r, &coefficients)).collect();
    let residuals: Vec<Q> = y.iter().zip(&fitted).map(|(a, b)| a - b).collect();
    let weight = |i: usize| {
        weights
            .and_then(|w| w.get(i).cloned())
            .unwrap_or_else(Q::one)
    };
    let ssr = residuals
        .iter()
        .enumerate()
        .fold(Q::zero(), |acc, (i, e)| acc + weight(i) * e * e);
    let has_constant = add_intercept || has_constant_column(&design);
    let tss = if has_constant {
        let sum_w = (0..n).fold(Q::zero(), |acc, i| acc + weight(i));
        let ybar = y
            .iter()
            .enumerate()
            .fold(Q::zero(), |acc, (i, v)| acc + weight(i) * v)
            / sum_w;
        y.iter().enumerate().fold(Q::zero(), |acc, (i, v)| {
            let d = v - &ybar;
            acc + weight(i) * &d * &d
        })
    } else {
        y.iter()
            .enumerate()
            .fold(Q::zero(), |acc, (i, v)| acc + weight(i) * v * v)
    };
    if tss.is_zero() {
        return Err(invalid(
            op,
            "the response is constant (zero total sum of squares): R² is undefined",
        ));
    }
    let ess = &tss - &ssr;
    let r_squared = Q::one() - &ssr / &tss;
    let k_constant = usize::from(has_constant);
    let df_model = p - k_constant;
    let df_resid = n - p;
    let adjusted_r_squared = Q::one() - qu(n - k_constant) / qu(df_resid) * (Q::one() - &r_squared);
    let mse_resid = &ssr / qu(df_resid);
    let cov_params = xtx_inv.scale(&mse_resid);
    Ok(Ols {
        coefficients,
        fitted,
        residuals,
        ssr,
        ess,
        tss,
        r_squared,
        adjusted_r_squared,
        df_model,
        df_resid,
        mse_resid,
        cov_params,
        design,
        y: y.to_vec(),
        weights: weights.map(<[Q]>::to_vec),
        xtx_inv,
        has_constant,
        added_intercept: add_intercept,
    })
}

/// Ordinary least squares, exactly: `β̂ = (XᵀX)⁻¹Xᵀy` by the normal
/// equations over ℚ.  `x` holds one row per observation whose entries are
/// the regressors (columns of `X`); `add_intercept` prepends the column of
/// ones.  `statsmodels.api.OLS(y, add_constant(x)).fit()`.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::linprog::q;
/// use symplex::stats::data::from_i64;
/// use symplex::stats::regression::ols;
///
/// let x = vec![from_i64(&[1, 5]), from_i64(&[2, 3]), from_i64(&[3, 8]), from_i64(&[4, 1])];
/// let y = from_i64(&[6, 5, 10, 4]);
/// let fit = ols(&y, &x, true)?;
/// assert_eq!(fit.coefficients.len(), 3);
/// assert_eq!(fit.df_resid, 1);
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] if `y` is empty, `x` has a different
/// number of rows or ragged rows, the design has no columns, `n ≤ p`,
/// `XᵀX` is singular (collinear regressors), or `y` is constant.
pub fn ols(y: &[Q], x: &[Vec<Q>], add_intercept: bool) -> Result<Ols, SymplexError> {
    fit_least_squares("ols", y, x, None, add_intercept)
}

/// Weighted least squares, exactly: `β̂ = (XᵀWX)⁻¹XᵀWy` with
/// `W = diag(weights)`.  Sums of squares, `R²`, `σ̂²` and the covariance
/// matrix are the weighted ones, as in
/// `statsmodels.api.WLS(y, add_constant(x), weights=w).fit()`.
///
/// # Errors
///
/// As [`ols`]; additionally if `weights` has the wrong length or a
/// non-positive entry.
pub fn wls(y: &[Q], x: &[Vec<Q>], weights: &[Q], add_intercept: bool) -> Result<Ols, SymplexError> {
    fit_least_squares("wls", y, x, Some(weights), add_intercept)
}

/// Simple linear regression `y = a + b·x`: `coefficients = [a, b]`.
/// `scipy.stats.linregress(x, y)` (`intercept`, `slope`, `stderr`,
/// `intercept_stderr`, `rvalue² = r_squared`, `pvalue = p_values()[1]`).
///
/// # Errors
///
/// As [`ols`] (fewer than three observations, constant `x` or `y`).
pub fn simple_linear_regression(x: &[Q], y: &[Q]) -> Result<Ols, SymplexError> {
    const OP: &str = "simple_linear_regression";
    if x.len() != y.len() {
        return Err(invalid(
            OP,
            format!(
                "x and y must have the same length ({} and {})",
                x.len(),
                y.len()
            ),
        ));
    }
    let rows: Vec<Vec<Q>> = x.iter().map(|v| vec![v.clone()]).collect();
    fit_least_squares(OP, y, &rows, None, true)
}

/// Exact polynomial least squares of `degree`: the coefficients
/// `[c₀, c₁, …, c_d]` of `c₀ + c₁x + … + c_d xᵈ`, **ascending** (index =
/// power), the crate-wide convention shared with `optimize::poly_fit`,
/// `optimize::eval_poly` and `Ex::coeffs` — `numpy.polyfit` returns the
/// same numbers highest power first.  With `n = degree + 1` distinct
/// abscissae this is the interpolating polynomial.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for mismatched lengths, `n < degree +
/// 1`, or a singular Vandermonde system (too few distinct `x`).
pub fn polyfit(x: &[Q], y: &[Q], degree: usize) -> Result<Vec<Q>, SymplexError> {
    const OP: &str = "polyfit";
    if x.len() != y.len() {
        return Err(invalid(
            OP,
            format!(
                "x and y must have the same length ({} and {})",
                x.len(),
                y.len()
            ),
        ));
    }
    let n = x.len();
    if degree >= n {
        return Err(invalid(
            OP,
            format!("degree {degree} needs more than {degree} points, got {n}"),
        ));
    }
    let rows: Vec<Vec<Q>> = x
        .iter()
        .map(|v| {
            let mut row = Vec::with_capacity(degree + 1);
            let mut power = Q::one();
            row.push(power.clone());
            for _ in 0..degree {
                power *= v;
                row.push(power.clone());
            }
            row
        })
        .collect();
    let design = QMatrix::new(rows).map_err(|e| invalid(OP, e.to_string()))?;
    let (beta, _) = normal_equations(OP, &design, y, None)?;
    Ok(beta)
}

/// The hat (projection) matrix `H = X(XᵀX)⁻¹Xᵀ` of a design matrix, exact:
/// `ŷ = Hy`, `H² = H = Hᵀ`, `tr H = p`.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] if `XᵀX` is singular.
pub fn hat_matrix(design: &QMatrix) -> Result<QMatrix, SymplexError> {
    const OP: &str = "hat_matrix";
    let xt = design.transpose();
    let xtx_inv = xt.matmul(design)?.inv().map_err(|_| {
        invalid(
            OP,
            format!(
                "the design matrix is rank deficient (rank {} of {} columns)",
                design.rank(),
                design.ncols()
            ),
        )
    })?;
    design.matmul(&xtx_inv)?.matmul(&xt)
}

/// Variance inflation factors `VIF_j = 1 / (1 − R²_j)`, where `R²_j` is the
/// `R²` of regressing column `j` of `x` on the other columns **and an
/// intercept**.  `statsmodels.stats.outliers_influence.
/// variance_inflation_factor(add_constant(x), j + 1)`.  A lone column has
/// `VIF = 1`.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for ragged rows, too few observations,
/// a constant column, or exactly collinear columns (infinite VIF).
pub fn vif(x: &[Vec<Q>]) -> Result<Vec<Q>, SymplexError> {
    const OP: &str = "vif";
    let n = x.len();
    if n == 0 {
        return Err(invalid(OP, "x is empty"));
    }
    let k = x[0].len();
    if let Some((i, r)) = x.iter().enumerate().find(|(_, r)| r.len() != k) {
        return Err(invalid(
            OP,
            format!("row {i} of x has {} entries, expected {k}", r.len()),
        ));
    }
    (0..k)
        .map(|j| {
            let target: Vec<Q> = x.iter().map(|r| r[j].clone()).collect();
            let others: Vec<Vec<Q>> = x
                .iter()
                .map(|r| {
                    r.iter()
                        .enumerate()
                        .filter(|&(c, _)| c != j)
                        .map(|(_, v)| v.clone())
                        .collect()
                })
                .collect();
            let fit = fit_least_squares(OP, &target, &others, None, true)
                .map_err(|e| invalid(OP, format!("column {j}: {e}")))?;
            if fit.r_squared.is_one() {
                return Err(invalid(
                    OP,
                    format!(
                        "column {j} is an exact linear combination of the others (infinite VIF)"
                    ),
                ));
            }
            Ok((Q::one() - fit.r_squared).recip())
        })
        .collect()
}

/// `R² = r²`: the coefficient of determination of the simple regression
/// with Pearson correlation `r` (either regression direction).
#[must_use]
pub fn r_squared_from_correlation(r: &Ex) -> Ex {
    r.powi(2).simplify()
}

/// The slope of the simple regression of `y` on `x` from the Pearson
/// correlation and the two standard deviations: `b = r · s_y / s_x` (with
/// the same `ddof` for both).
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] if `sd_x` is zero.
pub fn slope_from_correlation(r: &Ex, sd_x: &Ex, sd_y: &Ex) -> Result<Ex, SymplexError> {
    if sd_x.as_rational().is_some_and(|q| q.is_zero()) {
        return Err(invalid(
            "slope_from_correlation",
            "the standard deviation of x is zero",
        ));
    }
    Ok((r * sd_y / sd_x).simplify())
}

impl Ols {
    /// The `n × p` design matrix `X` (with the intercept column if one was
    /// added).
    #[must_use]
    pub fn design(&self) -> &QMatrix {
        &self.design
    }

    /// Number of observations `n` (`nobs`).
    #[must_use]
    pub fn nobs(&self) -> usize {
        self.design.nrows()
    }

    /// Number of parameters `p` (columns of `X`).
    #[must_use]
    pub fn n_params(&self) -> usize {
        self.design.ncols()
    }

    /// `(XᵀWX)⁻¹`, the unscaled covariance (`normalized_cov_params`).
    #[must_use]
    pub fn normalized_cov_params(&self) -> &QMatrix {
        &self.xtx_inv
    }

    /// Whether the design has a constant (added, explicit, or implicit) —
    /// statsmodels' `k_constant == 1`.  Decides centred vs uncentred `TSS`.
    #[must_use]
    pub fn has_constant(&self) -> bool {
        self.has_constant
    }

    /// The weights of a [`wls`] fit, `None` for [`ols`].
    #[must_use]
    pub fn weights(&self) -> Option<&[Q]> {
        self.weights.as_deref()
    }

    fn weight(&self, i: usize) -> Q {
        self.weights
            .as_ref()
            .and_then(|w| w.get(i).cloned())
            .unwrap_or_else(Q::one)
    }

    /// `σ̂ = √(SSR/(n − p))` as an exact expression (`np.sqrt(scale)`).
    #[must_use]
    pub fn residual_standard_error(&self, ctx: &Context) -> Ex {
        ex(ctx, &self.mse_resid).sqrt().simplify()
    }

    /// Standard errors `√(σ̂² [(XᵀWX)⁻¹]ⱼⱼ)` as exact expressions (`bse`).
    #[must_use]
    pub fn standard_errors(&self, ctx: &Context) -> Vec<Ex> {
        self.cov_params
            .diagonal()
            .iter()
            .map(|v| ex(ctx, v).sqrt().simplify())
            .collect()
    }

    fn require_residual_variance(&self, op: &'static str) -> Result<(), SymplexError> {
        if self.ssr.is_zero() {
            return Err(invalid(
                op,
                "the fit is perfect (SSR = 0): σ̂² = 0 and the statistic is undefined",
            ));
        }
        Ok(())
    }

    /// `t_j = β̂_j / se_j`, exact (`tvalues`).
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] for a perfect fit (`SSR = 0`).
    pub fn t_statistics(&self, ctx: &Context) -> Result<Vec<Ex>, SymplexError> {
        self.require_residual_variance("t_statistics")?;
        Ok(self
            .coefficients
            .iter()
            .zip(self.cov_params.diagonal())
            .map(|(b, v)| (ex(ctx, b) / ex(ctx, &v).sqrt()).simplify())
            .collect())
    }

    /// Two-sided p-values `P(|T_{n−p}| ≥ |t_j|)` as exact expressions
    /// (`pvalues`); evaluate with `eval_f64`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] for a perfect fit (`SSR = 0`).
    pub fn p_values(&self, ctx: &Context) -> Result<Vec<Ex>, SymplexError> {
        self.require_residual_variance("p_values")?;
        Ok(self
            .coefficients
            .iter()
            .zip(self.cov_params.diagonal())
            .map(|(b, v)| student_two_sided(ctx, self.df_resid, &(b * b / v)))
            .collect())
    }

    /// `log10` of each two-sided p-value of [`p_values`](Self::p_values),
    /// evaluated as expressions so the values stay finite where `eval_f64`
    /// underflows to `0.0` (below about `1e-308`); see
    /// [`PValue`](super::hypothesis::PValue).
    ///
    /// # Errors
    ///
    /// As [`p_values`](Self::p_values), plus the evaluation error of an
    /// expression (not expected).
    pub fn p_values_log10(&self, ctx: &Context) -> Result<Vec<f64>, SymplexError> {
        self.p_values(ctx)?
            .iter()
            .map(hypothesis::p_value_log10_of)
            .collect()
    }

    /// One [`TestResult`] per coefficient: the Student-t test of `β_j = 0`
    /// with `df = n − p`, two-sided (`summary()` rows).
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] for a perfect fit (`SSR = 0`).
    pub fn coefficient_tests(&self, ctx: &Context) -> Result<Vec<TestResult>, SymplexError> {
        let stats = self.t_statistics(ctx)?;
        let ps = self.p_values(ctx)?;
        Ok(stats
            .into_iter()
            .zip(ps)
            .map(|(statistic, p_value)| TestResult {
                statistic,
                p_value,
                df: Some(ex_usize(ctx, self.df_resid)),
                alternative: Alternative::TwoSided,
            })
            .collect())
    }

    /// The overall `F = (ESS/df_model) / (SSR/df_resid)`, exact (`fvalue`).
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] for an intercept-only model
    /// (`df_model = 0`) or a perfect fit.
    pub fn f_statistic(&self) -> Result<Q, SymplexError> {
        const OP: &str = "f_statistic";
        if self.df_model == 0 {
            return Err(invalid(
                OP,
                "the model has no regressors besides the constant (df_model = 0)",
            ));
        }
        self.require_residual_variance(OP)?;
        Ok((&self.ess / qu(self.df_model)) / &self.mse_resid)
    }

    /// The overall F-test of `β = 0` for every non-constant coefficient:
    /// exact statistic, `P(F_{df_model, df_resid} ≥ F)` as an exact
    /// expression (`fvalue`, `f_pvalue`).  `df` holds the denominator
    /// degrees of freedom `n − p`; the numerator is `df_model`.
    ///
    /// # Errors
    ///
    /// As [`f_statistic`](Self::f_statistic).
    pub fn f_test(&self, ctx: &Context) -> Result<TestResult, SymplexError> {
        let f = self.f_statistic()?;
        Ok(TestResult {
            statistic: ex(ctx, &f),
            p_value: f_sf(ctx, self.df_model, self.df_resid, &f),
            df: Some(ex_usize(ctx, self.df_resid)),
            alternative: Alternative::Greater,
        })
    }

    /// The ANOVA decomposition `TSS = ESS + SSR` with mean squares and `F`.
    ///
    /// # Errors
    ///
    /// As [`f_statistic`](Self::f_statistic).
    pub fn anova_table(&self) -> Result<AnovaTable, SymplexError> {
        let f = self.f_statistic()?;
        Ok(AnovaTable {
            ss_model: self.ess.clone(),
            df_model: self.df_model,
            ms_model: &self.ess / qu(self.df_model),
            ss_resid: self.ssr.clone(),
            df_resid: self.df_resid,
            ms_resid: self.mse_resid.clone(),
            ss_total: self.tss.clone(),
            df_total: self.df_model + self.df_resid,
            f,
        })
    }

    /// `β̂_j ± t_{(1+c)/2, n−p} · se_j` for every coefficient
    /// (`conf_int(alpha = 1 − c)`).  The limits are `f64`, so no context is
    /// needed.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] for `confidence ∉ (0, 1)`; the
    /// quantile's error if it does not converge.
    pub fn conf_int(&self, confidence: f64) -> Result<Vec<Interval<f64>>, SymplexError> {
        const OP: &str = "conf_int";
        check_confidence(OP, confidence)?;
        let t = t_two_sided(OP, self.df_resid as f64, confidence)?;
        self.coefficients
            .iter()
            .zip(self.cov_params.diagonal())
            .map(|(b, v)| {
                let b = to_f64(OP, b)?;
                let se = to_f64(OP, &v)?.sqrt();
                Ok(Interval::closed(b - t * se, b + t * se))
            })
            .collect()
    }

    /// The design row for new regressor values: `x_row` lists the
    /// regressors exactly as the rows of `x` given to the fit (without the
    /// added intercept).
    fn design_row(&self, op: &'static str, x_row: &[Q]) -> Result<Vec<Q>, SymplexError> {
        let k = self.n_params() - usize::from(self.added_intercept);
        if x_row.len() != k {
            return Err(invalid(
                op,
                format!(
                    "x_row has {} entries, expected {k} (the regressors without the intercept)",
                    x_row.len()
                ),
            ));
        }
        let mut row = Vec::with_capacity(k + 1);
        if self.added_intercept {
            row.push(Q::one());
        }
        row.extend(x_row.iter().cloned());
        Ok(row)
    }

    /// `ŷ₀ = x₀ᵀβ̂` for new regressor values (`predict`).
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `x_row` has the wrong length.
    pub fn predict(&self, x_row: &[Q]) -> Result<Q, SymplexError> {
        let row = self.design_row("predict", x_row)?;
        Ok(dot(&row, &self.coefficients))
    }

    /// `(ŷ₀, x₀ᵀ(XᵀWX)⁻¹x₀, t)` shared by the two intervals.
    fn interval_parts(
        &self,
        op: &'static str,
        x_row: &[Q],
        confidence: f64,
    ) -> Result<(f64, f64, f64), SymplexError> {
        check_confidence(op, confidence)?;
        let row = self.design_row(op, x_row)?;
        let yhat = to_f64(op, &dot(&row, &self.coefficients))?;
        let factor = to_f64(op, &quadratic_form(&self.xtx_inv, &row))?;
        let t = t_two_sided(op, self.df_resid as f64, confidence)?;
        Ok((yhat, factor, t))
    }

    /// Confidence interval for the mean response at `x_row`:
    /// `ŷ₀ ± t_{(1+c)/2, n−p} · √(σ̂² x₀ᵀ(XᵀWX)⁻¹x₀)`
    /// (`get_prediction(x).summary_frame()['mean_ci_lower' / 'mean_ci_upper']`).
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] for a wrong `x_row` length or
    /// `confidence ∉ (0, 1)`.
    pub fn confidence_interval_mean_response(
        &self,
        x_row: &[Q],
        confidence: f64,
    ) -> Result<Interval<f64>, SymplexError> {
        const OP: &str = "confidence_interval_mean_response";
        let (yhat, factor, t) = self.interval_parts(OP, x_row, confidence)?;
        let se = (to_f64(OP, &self.mse_resid)? * factor).sqrt();
        Ok(Interval::closed(yhat - t * se, yhat + t * se))
    }

    /// Prediction interval for a new observation at `x_row`:
    /// `ŷ₀ ± t_{(1+c)/2, n−p} · √(σ̂² (1 + x₀ᵀ(XᵀWX)⁻¹x₀))`
    /// (`get_prediction(x).summary_frame()['obs_ci_lower' / 'obs_ci_upper']`,
    /// with statsmodels' default unit weight for the new observation).
    ///
    /// # Errors
    ///
    /// As [`confidence_interval_mean_response`](Self::confidence_interval_mean_response).
    pub fn prediction_interval(
        &self,
        x_row: &[Q],
        confidence: f64,
    ) -> Result<Interval<f64>, SymplexError> {
        const OP: &str = "prediction_interval";
        let (yhat, factor, t) = self.interval_parts(OP, x_row, confidence)?;
        let se = (to_f64(OP, &self.mse_resid)? * (1.0 + factor)).sqrt();
        Ok(Interval::closed(yhat - t * se, yhat + t * se))
    }

    /// The hat matrix `H = X(XᵀWX)⁻¹XᵀW` with `ŷ = Hy`, exact (`W = I` for
    /// OLS, where `H` is the symmetric projection `X(XᵀX)⁻¹Xᵀ`).
    ///
    /// # Errors
    ///
    /// Propagates a shape error from the matrix products (not expected).
    pub fn hat_matrix(&self) -> Result<QMatrix, SymplexError> {
        const OP: &str = "hat_matrix";
        let xtw = QMatrix::new(
            self.design
                .rows()
                .enumerate()
                .map(|(i, r)| {
                    let w = self.weight(i);
                    r.iter().map(|v| v * &w).collect()
                })
                .collect(),
        )
        .map_err(|e| failed(OP, e.to_string()))?
        .transpose();
        self.design.matmul(&self.xtx_inv)?.matmul(&xtw)
    }

    /// Leverages `hᵢᵢ = wᵢ xᵢᵀ(XᵀWX)⁻¹xᵢ`, the diagonal of the hat matrix
    /// (`get_influence().hat_matrix_diag`); `Σ hᵢᵢ = p`.
    #[must_use]
    pub fn leverage(&self) -> Vec<Q> {
        self.design
            .rows()
            .enumerate()
            .map(|(i, r)| self.weight(i) * quadratic_form(&self.xtx_inv, r))
            .collect()
    }

    /// Cook's distances `Dᵢ = wᵢeᵢ² hᵢᵢ / (p σ̂² (1 − hᵢᵢ)²)`
    /// (`get_influence().cooks_distance[0]`).
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] for a perfect fit or an
    /// observation with leverage `1`.
    pub fn cooks_distance(&self) -> Result<Vec<Q>, SymplexError> {
        const OP: &str = "cooks_distance";
        self.require_residual_variance(OP)?;
        let p = qu(self.n_params());
        self.leverage()
            .iter()
            .zip(&self.residuals)
            .enumerate()
            .map(|(i, (h, e))| {
                let one_minus = Q::one() - h;
                if one_minus.is_zero() {
                    return Err(invalid(
                        OP,
                        format!("observation {i} has leverage 1: Cook's distance is undefined"),
                    ));
                }
                Ok(self.weight(i) * e * e * h / (&p * &self.mse_resid * &one_minus * &one_minus))
            })
            .collect()
    }

    /// Durbin–Watson statistic `Σₜ (eₜ − eₜ₋₁)² / Σ eₜ²` on the residuals
    /// in observation order (`statsmodels.stats.stattools.durbin_watson(resid)`).
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if every residual is zero.
    pub fn durbin_watson(&self) -> Result<Q, SymplexError> {
        let denom = self.residuals.iter().fold(Q::zero(), |acc, e| acc + e * e);
        if denom.is_zero() {
            return Err(invalid(
                "durbin_watson",
                "every residual is zero: the statistic is undefined",
            ));
        }
        let num = self.residuals.windows(2).fold(Q::zero(), |acc, w| {
            let d = &w[1] - &w[0];
            acc + &d * &d
        });
        Ok(num / denom)
    }

    /// Gaussian log-likelihood at the fit, concentrated over `σ²`:
    /// `ℓ = −n/2 · (ln 2π + ln(SSR/n) + 1)`, plus `½ Σ ln wᵢ` for weighted
    /// fits (`llf`).
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] for a perfect fit (`SSR = 0`).
    pub fn log_likelihood(&self, ctx: &Context) -> Result<Ex, SymplexError> {
        const OP: &str = "log_likelihood";
        if self.ssr.is_zero() {
            return Err(invalid(
                OP,
                "the fit is perfect (SSR = 0): the Gaussian log-likelihood is unbounded",
            ));
        }
        let n = self.nobs();
        let half_n = ex(ctx, &(qu(n) / qu(2)));
        let two_pi = ctx.int(2) * ctx.pi();
        let mut llf = -half_n * (two_pi.ln() + ex(ctx, &(&self.ssr / qu(n))).ln() + ctx.one());
        if let Some(w) = &self.weights {
            let sum_ln = w.iter().fold(ctx.zero(), |acc, wi| acc + ex(ctx, wi).ln());
            llf += ctx.rational(1, 2) * sum_ln;
        }
        Ok(llf)
    }

    /// `AIC = −2ℓ + 2p` (`aic`).
    ///
    /// # Errors
    ///
    /// As [`log_likelihood`](Self::log_likelihood).
    pub fn aic(&self, ctx: &Context) -> Result<Ex, SymplexError> {
        let llf = self.log_likelihood(ctx)?;
        Ok(ctx.int(2) * ex_usize(ctx, self.n_params()) - ctx.int(2) * llf)
    }

    /// `BIC = −2ℓ + p ln n` (`bic`).
    ///
    /// # Errors
    ///
    /// As [`log_likelihood`](Self::log_likelihood).
    pub fn bic(&self, ctx: &Context) -> Result<Ex, SymplexError> {
        let llf = self.log_likelihood(ctx)?;
        Ok(ex_usize(ctx, self.n_params()) * ex_usize(ctx, self.nobs()).ln() - ctx.int(2) * llf)
    }

    /// The response `y` the model was fitted to.
    #[must_use]
    pub fn response(&self) -> &[Q] {
        &self.y
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Logistic regression
// ═══════════════════════════════════════════════════════════════════════════

/// A binary response value: `bool`, or `0`/`1` as `u8`, `i64` or `f64`.
pub trait BinaryOutcome: Copy {
    /// `Some(true)` for a success, `Some(false)` for a failure, `None` for
    /// anything that is not a 0/1 outcome.
    fn as_outcome(self) -> Option<bool>;
}

impl BinaryOutcome for bool {
    fn as_outcome(self) -> Option<bool> {
        Some(self)
    }
}

impl BinaryOutcome for u8 {
    fn as_outcome(self) -> Option<bool> {
        match self {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        }
    }
}

impl BinaryOutcome for i64 {
    fn as_outcome(self) -> Option<bool> {
        match self {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        }
    }
}

impl BinaryOutcome for f64 {
    fn as_outcome(self) -> Option<bool> {
        if self == 0.0 {
            Some(false)
        } else if self == 1.0 {
            Some(true)
        } else {
            None
        }
    }
}

/// Options of the Newton–Raphson iteration of [`logit`], [`mnlogit`] and
/// [`ologit`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogitOpts {
    /// Maximum number of Newton steps (default `100`; statsmodels `maxiter`).
    pub max_iter: usize,
    /// Convergence when an accepted full Newton step satisfies
    /// `max_j |Δγ_j| ≤ tol · max(1, max_j |γ_j|)` (default `1e-10`), where
    /// `γ` are the coefficients of the standardised design the iteration
    /// runs on (see [`logit`]) — so the criterion does not depend on the
    /// units of the regressors.  Must be finite and positive.
    pub tol: f64,
}

impl Default for LogitOpts {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tol: 1e-10,
        }
    }
}

/// A fitted logistic regression (statsmodels `Logit(y, X).fit()`), in `f64`.
///
/// Like statsmodels, the null-model summaries assume the design contains a
/// constant: `null_log_likelihood` is the intercept-only model and
/// `df_model = p − 1` whatever the design.  Without a constant column the
/// intercept-only model is not nested in the fit, so `llr` can be
/// negative and [`llr_test`](Self::llr_test) is not a valid
/// likelihood-ratio test (statsmodels reports it all the same).
#[derive(Clone, Debug, PartialEq)]
pub struct Logit {
    /// `β̂`, the maximum-likelihood coefficients (`params`), intercept first
    /// when one was added.
    pub coefficients: Vec<f64>,
    /// `√diag((XᵀŴX)⁻¹)` with `Ŵ = diag(p̂ᵢ(1 − p̂ᵢ))` (`bse`).
    pub standard_errors: Vec<f64>,
    /// `z_j = β̂_j / se_j` (`tvalues`).
    pub z_values: Vec<f64>,
    /// Two-sided normal p-values `erfc(|z_j|/√2)` (`pvalues`).
    pub p_values: Vec<f64>,
    /// `ℓ(β̂) = Σ [yᵢ ln p̂ᵢ + (1 − yᵢ) ln(1 − p̂ᵢ)]` (`llf`).
    pub log_likelihood: f64,
    /// Log-likelihood of the intercept-only model, `s ln(s/n) + (n − s) ln((n − s)/n)`
    /// for `s` successes (`llnull`, which statsmodels obtains by an
    /// iterative fit of that model: it agrees to about `1e-10`).
    pub null_log_likelihood: f64,
    /// McFadden's `1 − ℓ/ℓ₀` (`prsquared`).
    pub pseudo_r_squared: f64,
    /// `−2ℓ` (the binomial GLM `deviance`; the saturated log-likelihood is 0).
    pub deviance: f64,
    /// Newton steps taken.
    pub iterations: usize,
    /// Whether an accepted full Newton step met the step criterion within
    /// `max_iter` ([`LogitOpts::tol`]).
    pub converged: bool,
    /// `p̂ᵢ = σ(xᵢᵀβ̂)` (`predict()`).
    pub fitted_probabilities: Vec<f64>,
    /// `(XᵀŴX)⁻¹` (`cov_params()`).
    pub cov_params: Vec<Vec<f64>>,
    /// Number of observations (`nobs`).
    pub nobs: usize,
    /// `p − 1` (`df_model`; statsmodels' convention, which assumes a
    /// constant column).
    pub df_model: usize,
    /// `n − p` (`df_resid`).
    pub df_resid: usize,
    added_intercept: bool,
}

/// `σ(η) = 1/(1 + e^{−η})`, accurate to a few ulps relative everywhere
/// (`σ(−η)` is the accurate complement `1 − σ(η)`).
fn sigmoid(eta: f64) -> f64 {
    if eta >= 0.0 {
        1.0 / (1.0 + (-eta).exp())
    } else {
        let e = eta.exp();
        e / (1.0 + e)
    }
}

/// `ln(1 + eˣ)`, stable for large `|x|`.
fn softplus(x: f64) -> f64 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

/// Logistic regression `P(y = 1 | x) = σ(xᵀβ)` by maximum likelihood.
/// `statsmodels.api.Logit(y, add_constant(x)).fit()`.
///
/// `y` is a slice of `bool`, or of `0`/`1` as `u8`, `i64` or `f64`
/// ([`BinaryOutcome`]); `x` holds one row of regressors per observation.
///
/// The fit is [`mnlogit`]'s with two categories: Newton–Raphson with step
/// halving (a step that lowers `ℓ` is halved, up to 40 times) from
/// `β = 0`, `β ← β + t (XᵀWX)⁻¹Xᵀ(y − p)`, `W = diag(pᵢ(1 − pᵢ))`.  The
/// iteration runs on a standardised design — every non-constant regressor
/// centred on its mean (when the design has a constant column to absorb
/// the shift) and divided by its root-mean-square deviation — and the
/// estimate and its covariance are mapped back exactly
/// (`β̂ = Aγ̂`, `cov β̂ = A cov γ̂ Aᵀ`): the maximum-likelihood fit is
/// equivariant under that affine change, while the information matrix of
/// a regressor with a large offset (a calendar year, a timestamp) is
/// otherwise so ill-conditioned that the standard errors lose about
/// `2 log₁₀(mean/sd)` digits (and statsmodels' Newton fit fails outright).
/// The covariance is the inverse of the information accumulated in
/// square-root form (Givens rotations of the rows `√wᵢ xᵢ`), never the
/// inverse of the formed `XᵀWX`, so it carries the conditioning of the
/// (standardised) design rather than its square.
///
/// Perfect (complete or quasi-complete) separation — where the likelihood
/// has no finite maximiser and statsmodels emits `PerfectSeparationWarning`
/// / `ConvergenceWarning` with huge coefficients — is reported as
/// [`SymplexError::ComputationFailed`]: either every observation is
/// predicted to within `1e-8` (statsmodels' perfect-prediction check), or
/// the iteration fails to converge while fitted probabilities reach `0`
/// or `1`, or the Hessian becomes singular.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::regression::{logit, LogitOpts};
///
/// // Ten controls with 3 successes, ten treated with 7.
/// let y: Vec<bool> = (0..20).map(|i| matches!(i, 7..=9 | 13..=19)).collect();
/// let x: Vec<Vec<f64>> = (0..20).map(|i| vec![if i < 10 { 0.0 } else { 1.0 }]).collect();
/// // statsmodels: Logit(y, add_constant(x)).fit().params = [-0.8472978603872037, 1.6945957207744073]
/// let fit = logit(&y, &x, true, &LogitOpts::default())?;
/// assert!((fit.coefficients[0] - (3.0f64 / 7.0).ln()).abs() < 1e-9);
/// assert!((fit.coefficients[1] - (49.0f64 / 9.0).ln()).abs() < 1e-9);
/// assert!((fit.predict_proba(&[1.0])? - 0.7).abs() < 1e-9);
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// - [`SymplexError::InvalidArgument`] for an empty or non-binary `y`, a
///   constant `y`, mismatched or ragged `x`, non-finite entries, `n ≤ p`,
///   collinear regressors (numerically: the information matrix of the
///   standardised design is singular to `1e-12` relative), `max_iter = 0`,
///   or a `tol` that is not finite and positive.
/// - [`SymplexError::ComputationFailed`] for perfect separation (see above).
pub fn logit<B: BinaryOutcome>(
    y: &[B],
    x: &[Vec<f64>],
    add_intercept: bool,
    opts: &LogitOpts,
) -> Result<Logit, SymplexError> {
    const OP: &str = "logit";
    let n = y.len();
    if n == 0 {
        return Err(invalid(OP, "y is empty"));
    }
    check_logit_opts(OP, opts)?;
    let classes: Vec<usize> = y
        .iter()
        .enumerate()
        .map(|(i, v)| {
            v.as_outcome()
                .map(usize::from)
                .ok_or_else(|| invalid(OP, format!("y[{i}] is not a 0/1 outcome")))
        })
        .collect::<Result<_, _>>()?;
    let successes = classes.iter().filter(|&&c| c == 1).count();
    if successes == 0 || successes == n {
        return Err(invalid(
            OP,
            "y is constant (all successes or all failures): the coefficients are not identified",
        ));
    }
    let data = categorical_data(OP, &classes, x, add_intercept, opts)?;
    let p = data.design.first().map_or(0, Vec::len);
    if n <= p {
        return Err(invalid(
            OP,
            format!("need more observations than parameters: n = {n}, p = {p}"),
        ));
    }
    let fit = fit_multinomial(OP, &data, add_intercept, opts, false)?;
    let log_likelihood = fit.eval.ll;
    let null_log_likelihood = categorical_null_log_likelihood(&data.counts);
    // From the reported coefficients on the original rows, so that they
    // are exactly `predict_proba` of those rows.
    let fitted_probabilities = data
        .design
        .iter()
        .map(|row| sigmoid(dot_f64(row, &fit.params)))
        .collect();
    Ok(Logit {
        pseudo_r_squared: 1.0 - log_likelihood / null_log_likelihood,
        deviance: -2.0 * log_likelihood,
        coefficients: fit.params,
        standard_errors: fit.wald.se,
        z_values: fit.wald.z,
        p_values: fit.wald.p,
        log_likelihood,
        null_log_likelihood,
        iterations: fit.iterations,
        converged: fit.converged,
        fitted_probabilities,
        cov_params: fit.wald.cov,
        nobs: n,
        df_model: p - 1,
        df_resid: n - p,
        added_intercept: add_intercept,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Maximum-likelihood fits: the shared summaries
// ═══════════════════════════════════════════════════════════════════════════

/// A `χ²_df` test of an `f64` statistic as a [`TestResult`]: the statistic
/// exactly, `P(χ²_df ≥ statistic)` through [`chi_squared_sf`] (`1` for a
/// non-positive statistic), the degrees of freedom, and the alternative
/// the caller reports.  Behind [`LikelihoodFit::llr_test`] and the Cox
/// model's likelihood-ratio / Wald / score trio.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for a `NaN` statistic.
pub(crate) fn chi_squared_test_result(
    ctx: &Context,
    statistic: f64,
    df: usize,
    alternative: Alternative,
) -> Result<TestResult, SymplexError> {
    let positive = statistic > 0.0;
    let statistic = ctx.from_f64(statistic)?;
    let p_value = if positive {
        chi_squared_sf(ctx, df, &statistic)
    } else {
        ctx.one()
    };
    Ok(TestResult {
        statistic,
        p_value,
        df: Some(ex_usize(ctx, df)),
        alternative,
    })
}

/// A model fitted by maximum likelihood, read through its log-likelihoods.
/// The information criteria, McFadden's pseudo-`R²` and the
/// likelihood-ratio test against the null model all follow from five
/// numbers, so a fit supplies those and gets the rest here.
///
/// Implemented by [`Logit`], [`MnLogit`], [`OrderedLogit`] and
/// [`CoxModel`](super::cox::CoxModel); each also offers `llr`, `aic`,
/// `bic` and `llr_test` inherently (the log-likelihoods are fields), so
/// the trait need not be imported to call them on a concrete fit — import
/// it to write code generic over fits:
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::LikelihoodFit;
/// use symplex::stats::regression::{logit, LogitOpts};
///
/// fn summary<F: LikelihoodFit>(fit: &F) -> String {
///     format!("AIC {:.3}, pseudo-R² {:.3}", fit.aic(), fit.pseudo_r_squared())
/// }
///
/// // Ten controls with 3 successes, ten treated with 7.
/// let y: Vec<bool> = (0..20).map(|i| matches!(i, 7..=9 | 13..=19)).collect();
/// let x: Vec<Vec<f64>> = (0..20).map(|i| vec![if i < 10 { 0.0 } else { 1.0 }]).collect();
/// let fit = logit(&y, &x, true, &LogitOpts::default())?;
/// // statsmodels: aic 28.434572082195736, prsquared 0.11870910076930752
/// assert_eq!(summary(&fit), "AIC 28.435, pseudo-R² 0.119");
/// # Ok::<(), SymplexError>(())
/// ```
pub trait LikelihoodFit {
    /// `ℓ`, the maximised log-likelihood (`llf`).
    fn log_likelihood(&self) -> f64;

    /// `ℓ₀`, the log-likelihood of the null model the fit is compared
    /// against (`llnull`): intercept-only for [`Logit`] and [`MnLogit`],
    /// thresholds-only for [`OrderedLogit`], no covariate effect for
    /// [`CoxModel`](super::cox::CoxModel).
    fn null_log_likelihood(&self) -> f64;

    /// The number of free parameters `k` that [`aic`](Self::aic) and
    /// [`bic`](Self::bic) charge for.
    fn n_params(&self) -> usize;

    /// The number of observations `n` (`nobs`).
    fn nobs(&self) -> usize;

    /// The degrees of freedom of [`llr_test`](Self::llr_test): how many
    /// parameters the null model fixes (`df_model`).
    fn df_model(&self) -> usize;

    /// The likelihood-ratio statistic `2(ℓ − ℓ₀)` against the null model
    /// (`llr`), asymptotically `χ²_{df_model}`.
    #[must_use]
    fn llr(&self) -> f64 {
        2.0 * (self.log_likelihood() - self.null_log_likelihood())
    }

    /// McFadden's pseudo-`R²`, `1 − ℓ/ℓ₀` (`prsquared`).
    #[must_use]
    fn pseudo_r_squared(&self) -> f64 {
        1.0 - self.log_likelihood() / self.null_log_likelihood()
    }

    /// `AIC = −2ℓ + 2k` (`aic`).
    #[must_use]
    fn aic(&self) -> f64 {
        -2.0 * self.log_likelihood() + 2.0 * self.n_params() as f64
    }

    /// `BIC = −2ℓ + k ln n` with `n = nobs()` (`bic`).  A fit whose
    /// effective sample size is not its observation count overrides this:
    /// [`CoxModel`](super::cox::CoxModel) counts events, as R's
    /// `BIC(coxph)` does.
    #[must_use]
    fn bic(&self) -> f64 {
        -2.0 * self.log_likelihood() + self.n_params() as f64 * (self.nobs() as f64).ln()
    }

    /// The likelihood-ratio test of the null model: [`llr`](Self::llr)
    /// referred to `χ²_{df_model}` (`llr_pvalue`), reported with
    /// [`Alternative::Greater`] — the upper tail of `χ²`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] when the null model has as many
    /// parameters as the model (`df_model = 0`: an intercept-only fit, or a
    /// single-regressor [`Logit`] without a constant under statsmodels'
    /// `p − 1` convention; statsmodels reports `nan`), or for a `NaN`
    /// statistic.
    fn llr_test(&self, ctx: &Context) -> Result<TestResult, SymplexError> {
        let df = self.df_model();
        if df == 0 {
            return Err(invalid(
                "llr_test",
                "df_model = 0: the null model has as many parameters as the fit, there is nothing to test",
            ));
        }
        chi_squared_test_result(ctx, self.llr(), df, Alternative::Greater)
    }
}

/// A fit with coefficients and their standard errors from an information
/// matrix: the Wald `z` statistics, two-sided normal p-values and Wald
/// intervals follow.  Implemented by [`Logit`], [`OrderedLogit`] (over the
/// slopes `β`) and [`CoxModel`](super::cox::CoxModel), and by [`MnLogit`]
/// over its parameters flattened in the order of its `cov_params`; each
/// offers [`conf_int`](Self::conf_int) inherently as well.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::WaldFit;
/// use symplex::stats::regression::{logit, LogitOpts};
///
/// let y: Vec<bool> = (0..20).map(|i| matches!(i, 7..=9 | 13..=19)).collect();
/// let x: Vec<Vec<f64>> = (0..20).map(|i| vec![if i < 10 { 0.0 } else { 1.0 }]).collect();
/// let fit = logit(&y, &x, true, &LogitOpts::default())?;
/// // statsmodels: tvalues [-1.2278512511111188, 1.736443891898117], pvalues[1] 0.08248537711586468
/// assert!((fit.z_values()[1] - 1.736_443_891_898_117).abs() < 1e-8);
/// assert!((fit.p_values()[1] - 0.082_485_377_115_864_68).abs() < 1e-9);
/// # Ok::<(), SymplexError>(())
/// ```
pub trait WaldFit {
    /// The estimates `β̂` (`params`).
    fn coefficients(&self) -> &[f64];

    /// The standard errors `√diag(I⁻¹)`, one per coefficient (`bse`).
    fn standard_errors(&self) -> &[f64];

    /// `z_j = β̂_j / se_j` (`tvalues`).
    #[must_use]
    fn z_values(&self) -> Vec<f64> {
        self.coefficients()
            .iter()
            .zip(self.standard_errors())
            .map(|(b, s)| b / s)
            .collect()
    }

    /// Two-sided normal p-values `P(|Z| ≥ |z_j|) = erfc(|z_j|/√2)`
    /// (`pvalues`).
    #[must_use]
    fn p_values(&self) -> Vec<f64> {
        self.z_values().into_iter().map(normal_two_sided).collect()
    }

    /// Wald intervals `β̂_j ± z_{(1+c)/2} · se_j` (`conf_int(alpha = 1 − c)`).
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] for `confidence ∉ (0, 1)`.
    fn conf_int(&self, confidence: f64) -> Result<Vec<Interval<f64>>, SymplexError> {
        check_confidence("conf_int", confidence)?;
        let z = z_two_sided(confidence);
        Ok(self
            .coefficients()
            .iter()
            .zip(self.standard_errors())
            .map(|(b, se)| Interval::closed(b - z * se, b + z * se))
            .collect())
    }
}

impl LikelihoodFit for Logit {
    fn log_likelihood(&self) -> f64 {
        self.log_likelihood
    }

    fn null_log_likelihood(&self) -> f64 {
        self.null_log_likelihood
    }

    fn n_params(&self) -> usize {
        self.coefficients.len()
    }

    fn nobs(&self) -> usize {
        self.nobs
    }

    fn df_model(&self) -> usize {
        self.df_model
    }
}

impl WaldFit for Logit {
    fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    fn standard_errors(&self) -> &[f64] {
        &self.standard_errors
    }
}

impl Logit {
    /// Number of parameters `p`.
    #[must_use]
    pub fn n_params(&self) -> usize {
        <Self as LikelihoodFit>::n_params(self)
    }

    fn design_row(&self, op: &'static str, x_row: &[f64]) -> Result<Vec<f64>, SymplexError> {
        let k = self.n_params() - usize::from(self.added_intercept);
        if x_row.len() != k {
            return Err(invalid(
                op,
                format!(
                    "x_row has {} entries, expected {k} (the regressors without the intercept)",
                    x_row.len()
                ),
            ));
        }
        if let Some(j) = x_row.iter().position(|v| !v.is_finite()) {
            return Err(invalid(op, format!("x_row[{j}] is not finite")));
        }
        let mut row = Vec::with_capacity(k + 1);
        if self.added_intercept {
            row.push(1.0);
        }
        row.extend_from_slice(x_row);
        Ok(row)
    }

    /// `P(y = 1 | x₀) = σ(x₀ᵀβ̂)` (`predict(x)`).
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] for a wrong length or a non-finite
    /// entry.
    pub fn predict_proba(&self, x_row: &[f64]) -> Result<f64, SymplexError> {
        let row = self.design_row("predict_proba", x_row)?;
        Ok(sigmoid(dot_f64(&row, &self.coefficients)))
    }

    /// The linear predictor `x₀ᵀβ̂` (the log-odds).
    ///
    /// # Errors
    ///
    /// As [`predict_proba`](Self::predict_proba).
    pub fn predict_log_odds(&self, x_row: &[f64]) -> Result<f64, SymplexError> {
        let row = self.design_row("predict_log_odds", x_row)?;
        Ok(dot_f64(&row, &self.coefficients))
    }

    /// `exp(β̂_j)`: the multiplicative change in the odds per unit of
    /// regressor `j` (`np.exp(params)`).
    #[must_use]
    pub fn odds_ratios(&self) -> Vec<f64> {
        self.coefficients.iter().map(|b| b.exp()).collect()
    }

    /// Wald intervals `β̂_j ± z_{(1+c)/2} · se_j` (`conf_int(alpha = 1 − c)`);
    /// [`WaldFit::conf_int`].
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] for `confidence ∉ (0, 1)`.
    pub fn conf_int(&self, confidence: f64) -> Result<Vec<Interval<f64>>, SymplexError> {
        <Self as WaldFit>::conf_int(self, confidence)
    }

    /// Likelihood-ratio statistic `2(ℓ − ℓ₀)` against the intercept-only
    /// model (`llr`), asymptotically `χ²_{df_model}`; [`LikelihoodFit::llr`].
    #[must_use]
    pub fn llr(&self) -> f64 {
        <Self as LikelihoodFit>::llr(self)
    }

    /// The likelihood-ratio test of every slope being zero:
    /// [`llr`](Self::llr) referred to `χ²_{p−1}` (`llr_pvalue`);
    /// [`LikelihoodFit::llr_test`].
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] for an intercept-only model
    /// (`df_model = 0`) or a non-finite statistic.
    pub fn llr_test(&self, ctx: &Context) -> Result<TestResult, SymplexError> {
        <Self as LikelihoodFit>::llr_test(self, ctx)
    }

    /// `AIC = −2ℓ + 2p` (`aic`); [`LikelihoodFit::aic`].
    #[must_use]
    pub fn aic(&self) -> f64 {
        <Self as LikelihoodFit>::aic(self)
    }

    /// `BIC = −2ℓ + p ln n` (`bic`); [`LikelihoodFit::bic`].
    #[must_use]
    pub fn bic(&self) -> f64 {
        <Self as LikelihoodFit>::bic(self)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Multinomial and ordinal logistic regression
// ═══════════════════════════════════════════════════════════════════════════

/// A validated categorical response with its design matrix, shared by
/// [`mnlogit`] and [`ologit`].
struct CategoricalData {
    /// `y[i] ∈ 0..k`.
    y: Vec<usize>,
    /// Observations per category; `k = counts.len() ≥ 2`, every entry `≥ 1`.
    counts: Vec<usize>,
    /// `n × p`, intercept column first when one was added.
    design: Vec<Vec<f64>>,
    /// Root-mean-square of each design column (`1` for the intercept): the
    /// scale-free unit in which a diverging coefficient is reported.
    rms: Vec<f64>,
}

fn categorical_data(
    op: &'static str,
    y: &[usize],
    x: &[Vec<f64>],
    add_intercept: bool,
    opts: &LogitOpts,
) -> Result<CategoricalData, SymplexError> {
    let n = y.len();
    if n == 0 {
        return Err(invalid(op, "y is empty"));
    }
    check_logit_opts(op, opts)?;
    if x.len() != n {
        return Err(invalid(
            op,
            format!("y has {n} observations but x has {} rows", x.len()),
        ));
    }
    let max_label = y.iter().copied().max().unwrap_or(0);
    if max_label == 0 {
        return Err(invalid(
            op,
            "y is constant (every observation is in category 0): need at least two categories",
        ));
    }
    if max_label >= n {
        // n observations cannot cover the max_label + 1 > n labels
        // 0..=max_label, so some label below n is unused: name it without
        // allocating (or overflowing) max_label + 1 counters.
        let mut seen = vec![false; n];
        for &c in y {
            if c < n {
                seen[c] = true;
            }
        }
        let j = seen.iter().position(|s| !s).unwrap_or(n);
        return Err(invalid(
            op,
            format!(
                "category {j} has no observations: y must take every value in 0..={max_label} (statsmodels relabels the observed values with np.unique; an explicitly declared empty level fails there too)"
            ),
        ));
    }
    let k = max_label + 1;
    let mut counts = vec![0usize; k];
    for &c in y {
        counts[c] += 1;
    }
    if let Some(j) = counts.iter().position(|&c| c == 0) {
        return Err(invalid(
            op,
            format!(
                "category {j} has no observations: y must take every value in 0..{k} (statsmodels relabels the observed values with np.unique; an explicitly declared empty level fails there too)"
            ),
        ));
    }
    let cols = x.first().map_or(0, Vec::len);
    if let Some((i, r)) = x.iter().enumerate().find(|(_, r)| r.len() != cols) {
        return Err(invalid(
            op,
            format!("row {i} of x has {} entries, expected {cols}", r.len()),
        ));
    }
    if let Some((i, j)) = x
        .iter()
        .enumerate()
        .find_map(|(i, r)| r.iter().position(|v| !v.is_finite()).map(|j| (i, j)))
    {
        return Err(invalid(op, format!("x[{i}][{j}] is not finite")));
    }
    let p = cols + usize::from(add_intercept);
    if p == 0 {
        return Err(invalid(
            op,
            "the design has no columns: pass at least one regressor",
        ));
    }
    let design: Vec<Vec<f64>> = x
        .iter()
        .map(|r| {
            let mut row = Vec::with_capacity(p);
            if add_intercept {
                row.push(1.0);
            }
            row.extend_from_slice(r);
            row
        })
        .collect();
    let rms: Vec<f64> = (0..p)
        .map(|a| {
            let s: f64 = design.iter().map(|r| r[a] * r[a]).sum();
            let v = (s / n as f64).sqrt();
            if v > 0.0 { v } else { 1.0 }
        })
        .collect();
    Ok(CategoricalData {
        y: y.to_vec(),
        counts,
        design,
        rms,
    })
}

/// `max_iter ≥ 1` and a finite positive `tol` (an infinite one declared
/// convergence after the first step).
fn check_logit_opts(op: &'static str, opts: &LogitOpts) -> Result<(), SymplexError> {
    if opts.max_iter == 0 {
        return Err(invalid(op, "max_iter must be positive"));
    }
    if !(opts.tol > 0.0 && opts.tol.is_finite()) {
        return Err(invalid(
            op,
            format!("tol must be finite and positive, got {}", opts.tol),
        ));
    }
    Ok(())
}

/// The affine change of regressor columns the Newton iterations run in:
/// column `a` becomes `(x_a − m_a)/s_a`.  A non-constant column is scaled
/// by its root-mean-square deviation `s_a` and centred on its mean when
/// the shift can be absorbed — by a constant column (an intercept) for
/// [`logit`] / [`mnlogit`], by the thresholds for [`ologit`]; constant
/// columns are left alone.  The likelihood is equivariant under the
/// change, so the fit is exactly the same model, while the information
/// matrix of the new columns is well conditioned whatever the regressors'
/// offsets and units (an offset `m ≫ s` makes `XᵀWX` ill-conditioned like
/// `(m/s)²`, and its inverse — the covariance — loses that many digits).
struct Standardization {
    center: Vec<f64>,
    scale: Vec<f64>,
    /// Index and value of the constant column that absorbs the centring.
    constant: Option<(usize, f64)>,
}

impl Standardization {
    /// The change for `design` (`n × p`, `n ≥ 1`); `thresholds_absorb_shift`
    /// centres every non-constant column.
    fn new(design: &[Vec<f64>], thresholds_absorb_shift: bool) -> Self {
        let p = design.first().map_or(0, Vec::len);
        let n = design.len() as f64;
        let is_constant = |a: usize| design.iter().all(|r| r[a] == design[0][a]);
        let constant = (0..p)
            .find(|&a| design[0][a] != 0.0 && is_constant(a))
            .map(|a| (a, design[0][a]));
        let centre = thresholds_absorb_shift || constant.is_some();
        let mut center = vec![0.0; p];
        let mut scale = vec![1.0; p];
        for a in 0..p {
            if is_constant(a) {
                continue;
            }
            // A running mean and a deviation scaled by its largest entry,
            // so neither overflows nor underflows for entries near the
            // ends of the float range.
            let m = if centre {
                design
                    .iter()
                    .enumerate()
                    .fold(0.0, |m, (i, r)| m + (r[a] - m) / (i + 1) as f64)
            } else {
                0.0
            };
            let top = design.iter().fold(0.0_f64, |t, r| t.max((r[a] - m).abs()));
            let s = top
                * (design
                    .iter()
                    .map(|r| ((r[a] - m) / top).powi(2))
                    .sum::<f64>()
                    / n)
                    .sqrt();
            // A column this cannot rescale keeps its raw scale (the
            // Cholesky factorisation then judges it).
            if m.is_finite() && s > 0.0 && s.is_finite() && (1.0 / s).is_finite() {
                center[a] = m;
                scale[a] = s;
            }
        }
        Self {
            center,
            scale,
            constant,
        }
    }

    fn apply(&self, design: &[Vec<f64>]) -> Vec<Vec<f64>> {
        design
            .iter()
            .map(|r| {
                r.iter()
                    .zip(self.center.iter().zip(&self.scale))
                    .map(|(v, (m, s))| (v - m) / s)
                    .collect()
            })
            .collect()
    }

    /// `A` with `β = Aγ` for one linear predictor `η = xᵀβ = x̃ᵀγ` over the
    /// `p` columns: `β_a = γ_a/s_a`, and the constant column `c` (value
    /// `v`) absorbs the centring, `β_c = γ_c − Σ_{a≠c} γ_a m_a/(s_a v)`.
    fn linear_predictor_map(&self) -> Vec<Vec<f64>> {
        let p = self.scale.len();
        let mut a_map = vec![vec![0.0; p]; p];
        for (a, (row, s)) in a_map.iter_mut().zip(&self.scale).enumerate() {
            row[a] = 1.0 / s;
        }
        if let Some((c, v)) = self.constant {
            let shifts = self.center.iter().zip(&self.scale).enumerate();
            for (a, (m, s)) in shifts.filter(|&(a, _)| a != c) {
                a_map[c][a] = -m / (s * v);
            }
        }
        a_map
    }

    /// `A` with `(β, θ) = A(γ, τ)` for the ordered logit, whose cut points
    /// absorb the centring: `θ_j − xᵀβ = τ_j − x̃ᵀγ` gives `β_a = γ_a/s_a`
    /// and `θ_j = τ_j + Σ_a γ_a m_a/s_a`.
    fn ordered_map(&self, n_thresholds: usize) -> Vec<Vec<f64>> {
        let p = self.scale.len();
        let q = p + n_thresholds;
        let shifts: Vec<f64> = self
            .center
            .iter()
            .zip(&self.scale)
            .map(|(m, s)| m / s)
            .collect();
        let mut a_map = vec![vec![0.0; q]; q];
        for (j, row) in a_map.iter_mut().enumerate() {
            if j < p {
                row[j] = 1.0 / self.scale[j];
            } else {
                row[j] = 1.0;
                row[..p].copy_from_slice(&shifts);
            }
        }
        a_map
    }
}

/// `k` copies of the square `block` down the diagonal.
fn block_diagonal(block: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
    let p = block.len();
    let mut out = vec![vec![0.0; p * k]; p * k];
    for e in 0..k {
        for (a, row) in block.iter().enumerate() {
            out[e * p + a][e * p..(e + 1) * p].copy_from_slice(row);
        }
    }
    out
}

/// `A v`.
fn map_vector(a_map: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    a_map.iter().map(|row| dot_f64(row, v)).collect()
}

/// The upper-triangular square root `R` (`RᵀR = ZᵀZ`) of an information
/// matrix `I = ZᵀZ` given by rows `z` of `Z`, accumulated by Givens
/// rotations (Golub & Van Loan, *Matrix Computations*, 4th ed., §5.1.8 and
/// §6.5.3, updating a QR factorisation by a row).  The covariance
/// `I⁻¹ = R⁻¹R⁻ᵀ` then carries the conditioning of `Z`, where inverting
/// the formed `ZᵀZ` squared it: the standard errors of a polynomial logit
/// (`x, …, x⁸` on `[0, 1]`) kept only 6–7 correct digits.
struct RootInformation {
    r: Vec<Vec<f64>>,
    /// `‖Z e_j‖²`, the diagonal of `I`, for the singularity test.
    diag: Vec<f64>,
}

/// `R_jj ≤ this · √I_jj` marks `I` numerically singular — the same
/// criterion as the Newton iteration's Cholesky pivot `R_jj² ≤ 10⁻¹² I_jj`
/// ([`information_cholesky`]).
const ROOT_PIVOT_REL_TOL: f64 = 1e-6;

impl RootInformation {
    fn new(q: usize) -> Self {
        Self {
            r: vec![vec![0.0; q]; q],
            diag: vec![0.0; q],
        }
    }

    /// Rotate the row `z` (consumed) into `R`.
    fn add_row(&mut self, z: &mut [f64]) {
        let q = self.diag.len();
        for (d, v) in self.diag.iter_mut().zip(z.iter()) {
            *d += v * v;
        }
        for j in 0..q {
            if z[j] == 0.0 {
                continue;
            }
            let row = &mut self.r[j];
            let h = row[j].hypot(z[j]);
            let (c, s) = (row[j] / h, z[j] / h);
            row[j] = h;
            for (rt, zt) in row[j + 1..].iter_mut().zip(&mut z[j + 1..]) {
                let (r0, z0) = (*rt, *zt);
                *rt = c * r0 + s * z0;
                *zt = c * z0 - s * r0;
            }
        }
    }

    /// `B = A R⁻¹`, so that `A I⁻¹ Aᵀ = B Bᵀ`; `None` when `I` is
    /// numerically singular.
    fn covariance_factor(&self, a_map: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
        let q = self.diag.len();
        if (0..q).any(|j| {
            let rjj = self.r[j][j];
            !(rjj.is_finite() && rjj > ROOT_PIVOT_REL_TOL * self.diag[j].sqrt())
        }) {
            return None;
        }
        // Columns of R⁻¹ by back substitution on R x = e_c, stored as the
        // rows of R⁻ᵀ (x_i = 0 for i > c).
        let r_inv_t: Vec<Vec<f64>> = (0..q)
            .map(|c| {
                let mut x = vec![0.0; q];
                for i in (0..=c).rev() {
                    let e = if i == c { 1.0 } else { 0.0 };
                    let s = e - dot_f64(&self.r[i][i + 1..=c], &x[i + 1..=c]);
                    x[i] = s / self.r[i][i];
                }
                x
            })
            .collect();
        Some(
            a_map
                .iter()
                .map(|row| r_inv_t.iter().map(|col| dot_f64(row, col)).collect())
                .collect(),
        )
    }
}

/// The Wald summary of `params = Aγ̂` from the square root `R` of the
/// information of the standardised parameters `γ`: `cov = (AR⁻¹)(AR⁻¹)ᵀ`
/// (symmetric, positive semi-definite by construction; each standard error
/// is the norm of a row of `AR⁻¹`), then `z` and the two-sided normal
/// p-values of `params`.
fn mapped_wald_summary(
    op: &'static str,
    root: &RootInformation,
    a_map: &[Vec<f64>],
    params: &[f64],
) -> Result<WaldSummary, SymplexError> {
    let b = root.covariance_factor(a_map).ok_or_else(|| {
        failed(
            op,
            "the Hessian at the estimate is singular: the standard errors are undefined",
        )
    })?;
    let q = b.len();
    let mut cov = vec![vec![0.0; q]; q];
    for i in 0..q {
        for j in 0..=i {
            let v = dot_f64(&b[i], &b[j]);
            cov[i][j] = v;
            cov[j][i] = v;
        }
    }
    // ‖row‖₂ scaled by its largest entry: a standard error of 1e-302 (or
    // 1e299) is representable although its variance is not.
    let se: Vec<f64> = b
        .iter()
        .map(|row| {
            let top = row.iter().fold(0.0_f64, |t, v| t.max(v.abs()));
            if top == 0.0 || !top.is_finite() {
                return top;
            }
            top * row.iter().map(|v| (v / top).powi(2)).sum::<f64>().sqrt()
        })
        .collect();
    let z: Vec<f64> = params.iter().zip(&se).map(|(b, s)| b / s).collect();
    let p = z.iter().map(|z| normal_two_sided(*z)).collect();
    Ok(WaldSummary { cov, se, z, p })
}

/// The square-root information of the multinomial logit at the flat
/// (standardised) `beta`: per observation, the rows `L e_m ⊗ x` for the
/// exact Cholesky factor `L` of `W = diag(π) − ππᵀ` over the `k − 1`
/// non-reference categories (K. Tanabe and M. Sagae, "An exact Cholesky
/// decomposition and the generalized inverse of the variance-covariance
/// matrix of the multinomial distribution", *J. R. Stat. Soc. B* 54 (1992)
/// 211–219; derived here from the Schur complements, which keep the form
/// `diag(π') − π'π'ᵀ/t`): with the tail masses
/// `t_j = π_0 + Σ_{m≥j} π_m` (sums of positives, no cancellation),
/// `L_jj = √(π_j t_{j+1}/t_j)` and `L_ij = −π_i √(π_j/(t_j t_{j+1}))` for
/// `i > j`, so that `Σ_m (L e_m)(L e_m)ᵀ ⊗ xxᵀ = W ⊗ xxᵀ` is the
/// observation's information.
fn mnlogit_root_information(design: &[Vec<f64>], k: usize, beta: &[f64]) -> RootInformation {
    let p = design.first().map_or(0, Vec::len);
    let m = (k - 1) * p;
    let mut root = RootInformation::new(m);
    let coefficients: Vec<Vec<f64>> = beta.chunks(p.max(1)).map(<[f64]>::to_vec).collect();
    let mut z = vec![0.0; m];
    let mut tail = vec![0.0; k + 1];
    for row in design {
        let pr = softmax_probabilities(&coefficients, row);
        // tail[j] = π_0 + Σ_{m ≥ j} π_m for j = 1..k (tail[k] = π_0).
        tail[k] = pr[0];
        for j in (1..k).rev() {
            tail[j] = tail[j + 1] + pr[j];
        }
        for col in 1..k {
            if tail[col + 1] == 0.0 {
                // Every later category (and the reference) has π = 0.
                continue;
            }
            let root_share = (pr[col] / tail[col]).sqrt();
            let root_next = tail[col + 1].sqrt();
            z.fill(0.0);
            for j in col..k {
                let l_jc = if j == col {
                    root_share * root_next
                } else {
                    -(pr[j] / root_next) * root_share
                };
                for (a, xa) in row.iter().enumerate() {
                    z[(j - 1) * p + a] = l_jc * xa;
                }
            }
            root.add_row(&mut z);
        }
    }
    root
}

/// The square-root information of the ordered logit at `par = (β, θ)`:
/// per observation ([`OrderedTerm`]) the rows `√(σ(u)σ(−u))·∂u`,
/// `√(σ(l)σ(−l))·∂l` and `√q·(∂u − ∂l)` with `∂u = (−x, e_c)` and
/// `∂l = (−x, e_{c−1})`, whose outer products sum to the observation's
/// information `Jᵀ[[s_u + q, −q], [−q, s_l + q]]J`.
fn ologit_root_information(
    design: &[Vec<f64>],
    y: &[usize],
    k: usize,
    par: &[f64],
) -> RootInformation {
    let p = design.first().map_or(0, Vec::len);
    let q = p + k - 1;
    let (beta, theta) = par.split_at(p);
    let mut root = RootInformation::new(q);
    let mut z = vec![0.0; q];
    for (row, &c) in design.iter().zip(y) {
        let eta = dot_f64(row, beta);
        let has_upp = c < k - 1;
        let has_low = c > 0;
        let t = ordered_term(
            has_low.then(|| theta[c - 1] - eta),
            has_upp.then(|| theta[c] - eta),
        );
        for (weight, threshold) in [
            (t.s_upp, has_upp.then_some(c)),
            (t.s_low, has_low.then(|| c - 1)),
        ] {
            if let Some(j) = threshold {
                let w = weight.sqrt();
                z.fill(0.0);
                for (a, xa) in row.iter().enumerate() {
                    z[a] = -w * xa;
                }
                z[p + j] = w;
                root.add_row(&mut z);
            }
        }
        if has_upp && has_low && t.q > 0.0 {
            let w = t.q.sqrt();
            z.fill(0.0);
            z[p + c] = w;
            z[p + c - 1] = -w;
            root.add_row(&mut z);
        }
    }
    root
}

/// `Σ_j n_j ln(n_j / n)`: the log-likelihood of the model with no
/// regressors (`llnull` of `MNLogit` and `OrderedModel`).
fn categorical_null_log_likelihood(counts: &[usize]) -> f64 {
    let n = counts.iter().sum::<usize>() as f64;
    counts.iter().map(|&c| c as f64 * (c as f64 / n).ln()).sum()
}

/// `max_i (1 − π_{i, y_i})`: statsmodels' `_check_perfect_pred` distance.
fn max_own_category_miss(probs: &[Vec<f64>], y: &[usize]) -> f64 {
    probs
        .iter()
        .zip(y)
        .fold(0.0_f64, |m, (pr, &c)| m.max(1.0 - pr[c]))
}

/// Whether some fitted probability has reached `0` or `1` numerically.
fn has_degenerate_probability(probs: &[Vec<f64>]) -> bool {
    probs
        .iter()
        .any(|pr| pr.iter().any(|&v| !(1e-10..=1.0 - 1e-10).contains(&v)))
}

/// Which coefficient is diverging, in scale-free units `|β| · rms(x)`:
/// `"the intercept of category 2"`, `"covariate 1 of category 2"` (for
/// `k − 1` equations of `p` coefficients) or `"covariate 1"` (one equation).
fn diverging_coefficient(
    beta: &[f64],
    p: usize,
    rms: &[f64],
    added_intercept: bool,
    per_category: bool,
) -> String {
    let mut best = 0;
    let mut best_scaled = -1.0;
    for (idx, b) in beta.iter().enumerate() {
        let scaled = b.abs() * rms[idx % p];
        if scaled > best_scaled {
            best_scaled = scaled;
            best = idx;
        }
    }
    let a = best % p;
    let column = if added_intercept {
        if a == 0 {
            "the intercept".to_string()
        } else {
            format!("covariate {}", a - 1)
        }
    } else {
        format!("covariate {a}")
    };
    if per_category {
        format!("{column} of category {}", best / p + 1)
    } else {
        column
    }
}

/// One evaluation of a categorical log-likelihood: value, score, observed
/// information (`−∂²ℓ`) and the fitted probabilities (`n × k`).
struct CategoricalEval {
    ll: f64,
    score: Vec<f64>,
    info: Vec<Vec<f64>>,
    probs: Vec<Vec<f64>>,
}

/// Newton–Raphson with step halving on a concave log-likelihood, shared by
/// [`mnlogit`] and [`ologit`].  `evaluate` returns `None` at an infeasible
/// point (unordered thresholds).  Returns the final evaluation, the
/// parameters and the iteration count; the separation checks mirror
/// [`logit`]'s and name the diverging coefficient through `culprit`.
struct NewtonOutcome {
    params: Vec<f64>,
    eval: CategoricalEval,
    iterations: usize,
    converged: bool,
}

fn newton_categorical(
    op: &'static str,
    opts: &LogitOpts,
    y: &[usize],
    start: Vec<f64>,
    evaluate: &dyn Fn(&[f64]) -> Option<CategoricalEval>,
    culprit: &dyn Fn(&[f64]) -> String,
) -> Result<NewtonOutcome, SymplexError> {
    let mut params = start;
    let mut cur = evaluate(&params)
        .ok_or_else(|| failed(op, "the starting point is infeasible (internal invariant)"))?;
    let mut converged = false;
    let mut iterations = 0;
    for iter in 1..=opts.max_iter {
        iterations = iter;
        let Some(l) = information_cholesky(&cur.info) else {
            return Err(if iter == 1 {
                invalid(
                    op,
                    "the design matrix is rank deficient (numerically: collinear or nearly collinear regressors): drop a collinear regressor",
                )
            } else {
                failed(
                    op,
                    format!(
                        "the Hessian became singular: complete or quasi-complete separation, the maximum-likelihood estimate does not exist ({} is diverging)",
                        culprit(&params)
                    ),
                )
            });
        };
        let dir = dense_f64::cholesky_solve(&l, cur.info.len(), &cur.score);
        let max_step = dir.iter().fold(0.0_f64, |m, s| m.max(s.abs()));
        let scale = params.iter().fold(1.0_f64, |m, b| m.max(b.abs()));
        // Step halving: accept the first fraction of the Newton step that
        // does not decrease ℓ (up to rounding); a rejected full step is a
        // sign the quadratic model is poor, so convergence is only declared
        // on an accepted full step.
        let mut t = 1.0;
        let mut accepted = None;
        for _ in 0..40 {
            let trial: Vec<f64> = params.iter().zip(&dir).map(|(b, d)| b + t * d).collect();
            if let Some(next) = evaluate(&trial)
                && next.ll.is_finite()
                && next.ll >= cur.ll - 1e-10 * (1.0 + cur.ll.abs())
            {
                accepted = Some((trial, next));
                break;
            }
            t *= 0.5;
        }
        let Some((trial, next)) = accepted else {
            return Err(failed(
                op,
                "no step along the Newton direction increases the log-likelihood (the likelihood is not locally concave here)",
            ));
        };
        params = trial;
        cur = next;
        if params.iter().any(|b| !b.is_finite()) {
            return Err(failed(
                op,
                format!(
                    "the coefficients diverged: perfect separation, the maximum-likelihood estimate does not exist ({} is diverging)",
                    culprit(&params)
                ),
            ));
        }
        // statsmodels' `_check_perfect_pred`: every observation predicted
        // to within 1e-8 means the likelihood is maximised only at infinity.
        if max_own_category_miss(&cur.probs, y) <= 1e-8 {
            return Err(failed(
                op,
                format!(
                    "perfect separation: every observation is predicted exactly (1 − π̂ ≤ 1e-8), the maximum-likelihood estimate does not exist ({} is diverging)",
                    culprit(&params)
                ),
            ));
        }
        if t == 1.0 && max_step <= opts.tol * scale {
            converged = true;
            break;
        }
    }
    if !converged && has_degenerate_probability(&cur.probs) {
        return Err(failed(
            op,
            format!(
                "no convergence in {} iterations while fitted probabilities reached 0 or 1: complete or quasi-complete separation, the maximum-likelihood estimate does not exist ({} is diverging)",
                opts.max_iter,
                culprit(&params)
            ),
        ));
    }
    Ok(NewtonOutcome {
        params,
        eval: cur,
        iterations,
        converged,
    })
}

/// Checks a new regressor row against the fitted design and prepends the
/// intercept when the fit added one.
fn categorical_design_row(
    op: &'static str,
    x_row: &[f64],
    p: usize,
    added_intercept: bool,
) -> Result<Vec<f64>, SymplexError> {
    let k = p - usize::from(added_intercept);
    if x_row.len() != k {
        return Err(invalid(
            op,
            format!(
                "x_row has {} entries, expected {k} (the regressors without the intercept)",
                x_row.len()
            ),
        ));
    }
    if let Some(j) = x_row.iter().position(|v| !v.is_finite()) {
        return Err(invalid(op, format!("x_row[{j}] is not finite")));
    }
    let mut row = Vec::with_capacity(p);
    if added_intercept {
        row.push(1.0);
    }
    row.extend_from_slice(x_row);
    Ok(row)
}

/// `argmax` of a probability vector (the first maximum on ties).
fn modal_category(probs: &[f64]) -> usize {
    let mut best = 0;
    for (j, &v) in probs.iter().enumerate() {
        if v > probs[best] {
            best = j;
        }
    }
    best
}

// ── Multinomial logit ───────────────────────────────────────────────────

/// A fitted multinomial logistic regression (statsmodels
/// `MNLogit(y, X).fit()`), in `f64`.  Category `0` is the reference:
/// `P(y = j | x) / P(y = 0 | x) = exp(xᵀβ_j)` for `j = 1, …, k − 1`.
///
/// Per-coefficient fields are indexed `[j − 1][a]`: the equation of
/// category `j` first, then the parameter (intercept first when one was
/// added) — statsmodels' `params.T`.  [`cov_params`](Self::cov_params) is
/// `(k − 1)p × (k − 1)p` over the flat index `(j − 1)·p + a`, statsmodels'
/// Fortran-order flattening (`cov_params()`); the [`WaldFit`] view of the
/// fit uses that same flat index, while the inherent
/// [`conf_int`](Self::conf_int) keeps the nested shape.
///
/// [`LikelihoodFit::n_params`] is the total `(k − 1)p` the information
/// criteria charge for; the [`n_params`](Self::n_params) *field* is the
/// per-equation `p`.
#[derive(Clone, Debug, PartialEq)]
pub struct MnLogit {
    /// `β̂_j` for each non-reference category (`params.T`).
    pub coefficients: Vec<Vec<f64>>,
    /// `√diag(I(β̂)⁻¹)`, same shape (`bse.T`).
    pub standard_errors: Vec<Vec<f64>>,
    /// `β̂ / se` (`tvalues.T`).
    pub z_values: Vec<Vec<f64>>,
    /// Two-sided normal p-values `erfc(|z|/√2)` (`pvalues.T`).
    pub p_values: Vec<Vec<f64>>,
    /// `ℓ(β̂) = Σᵢ ln π̂_{i, yᵢ}` (`llf`).
    pub log_likelihood: f64,
    /// `Σ_j n_j ln(n_j / n)`, the intercept-only model (`llnull`).
    pub null_log_likelihood: f64,
    /// McFadden's `1 − ℓ/ℓ₀` (`prsquared`).
    pub pseudo_r_squared: f64,
    /// Newton steps taken.
    pub iterations: usize,
    /// Whether an accepted full Newton step met the step criterion within
    /// `max_iter`.
    pub converged: bool,
    /// `I(β̂)⁻¹` over the flat index `(j − 1)·p + a` (`cov_params()`).
    pub cov_params: Vec<Vec<f64>>,
    /// `k`, the number of categories (`J`).
    pub n_categories: usize,
    /// `p`, the number of parameters per equation, intercept included (`K`).
    pub n_params: usize,
    /// `π̂_{ij}` for every observation and category, `n × k` (`predict()`).
    pub fitted_probabilities: Vec<Vec<f64>>,
    /// Number of observations (`nobs`).
    pub nobs: usize,
    /// `(k − 1)(p − 1)`: the slope count, statsmodels' `df_model` (which
    /// assumes an intercept column: without one, the intercept-only null
    /// model behind `null_log_likelihood`, `llr` and `llr_test` is not
    /// nested in the fit, and the likelihood-ratio test is not valid).
    pub df_model: usize,
    /// `n − (k − 1)p` (`df_resid`).
    pub df_resid: usize,
    added_intercept: bool,
    /// `coefficients` over the flat index `(j − 1)·p + a`, for [`WaldFit`].
    flat_coefficients: Vec<f64>,
    /// `standard_errors` over the same flat index.
    flat_standard_errors: Vec<f64>,
}

/// Log-likelihood, score and observed information of the multinomial logit
/// at the flat `beta` (index `(j − 1)·p + a`).  `1 − π_j` enters the
/// score and the weights as `Σ_{m≠j} π_m`, not by subtraction: at a
/// fitted probability near 1 the difference kept only its leading digits
/// (`π = 1` exactly past `η ≈ 37`, where the weight became 0).
fn mnlogit_evaluate(design: &[Vec<f64>], y: &[usize], k: usize, beta: &[f64]) -> CategoricalEval {
    let p = design.first().map_or(0, Vec::len);
    let m = (k - 1) * p;
    let mut ll = 0.0;
    let mut score = vec![0.0; m];
    let mut info = vec![vec![0.0; m]; m];
    let mut probs = Vec::with_capacity(design.len());
    let mut eta = vec![0.0; k];
    let mut rest = vec![0.0; k];
    for (row, &c) in design.iter().zip(y) {
        eta[0] = 0.0;
        for j in 1..k {
            eta[j] = dot_f64(row, &beta[(j - 1) * p..j * p]);
        }
        let mx = eta.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut pr: Vec<f64> = eta.iter().map(|e| (e - mx).exp()).collect();
        let s: f64 = pr.iter().sum();
        for v in &mut pr {
            *v /= s;
        }
        ll += eta[c] - mx - s.ln();
        // rest[j] = Σ_{m≠j} π_m from prefix and suffix sums of positives.
        let mut prefix = 0.0;
        for j in 0..k {
            rest[j] = prefix;
            prefix += pr[j];
        }
        let mut suffix = 0.0;
        for j in (0..k).rev() {
            rest[j] += suffix;
            suffix += pr[j];
        }
        for j in 1..k {
            let r = if c == j { rest[j] } else { -pr[j] };
            let base = (j - 1) * p;
            for (a, xa) in row.iter().enumerate() {
                score[base + a] += xa * r;
            }
            for l in 1..k {
                let w = if j == l {
                    pr[j] * rest[j]
                } else {
                    -pr[j] * pr[l]
                };
                let base_l = (l - 1) * p;
                for (a, xa) in row.iter().enumerate() {
                    for (b, xb) in row.iter().enumerate() {
                        info[base + a][base_l + b] += w * xa * xb;
                    }
                }
            }
        }
        probs.push(pr);
    }
    CategoricalEval {
        ll,
        score,
        info,
        probs,
    }
}

/// Multinomial logistic regression `P(y = j | x) ∝ exp(xᵀβ_j)` with
/// `β_0 = 0` (category `0` is the reference), fitted by Newton–Raphson with
/// step halving on the full `(k − 1)p` parameter vector and the exact
/// block Hessian `Xᵀ(diag(π_j) − π_j π_lᵀ)X`, from `β = 0`, on the
/// standardised design and mapped back exactly as described for [`logit`]
/// (whose fit this is with `k = 2`).  The covariance is the inverse of the
/// information accumulated in square-root form (never forming and
/// inverting `XᵀWX`), so it carries the conditioning of the design rather
/// than its square.
/// `statsmodels.api.MNLogit(y, add_constant(x)).fit(method='newton')`.
///
/// `y[i] ∈ 0..k` with every category observed at least once; `x` holds one
/// row of regressors per observation.  Convergence is declared when an
/// accepted full Newton step satisfies `max |Δβ| ≤ tol · max(1, max |β|)`
/// ([`LogitOpts`]).  With `k = 2` the fit is [`logit`]'s: the single
/// equation is the log-odds of category `1` against `0`.
///
/// Perfect (complete or quasi-complete) separation — no finite maximiser;
/// statsmodels returns enormous coefficients with a `ConvergenceWarning` —
/// is reported as [`SymplexError::ComputationFailed`] naming the diverging
/// coefficient: either every observation is predicted to within `1e-8`,
/// or the iteration fails to converge while a fitted probability reaches
/// `0` or `1`, or the Hessian becomes singular.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::regression::{mnlogit, LogitOpts};
///
/// let y = [0, 0, 1, 0, 0, 1, 2, 0, 1, 1, 2, 0, 1, 2, 1, 2, 1, 2, 2, 1, 2, 0, 2, 2];
/// let x: Vec<Vec<f64>> = (1..=24).map(|i| vec![f64::from(i)]).collect();
/// // statsmodels: MNLogit(y, add_constant(x)).fit(method='newton').params.T =
/// //   [[-0.9175964754951935, 0.10985702371911016], [-2.875029245270399, 0.2536273262970427]]
/// let fit = mnlogit(&y, &x, true, &LogitOpts::default())?;
/// assert!(fit.converged);
/// assert!((fit.coefficients[1][1] - 0.2536273262970427).abs() < 1e-8);
/// assert!((fit.log_likelihood - -22.117379463121267).abs() < 1e-9);
/// let probs = fit.predict_proba(&[12.0])?;             // predict([1, 12])
/// assert!((probs[1] - 0.4060657594944313).abs() < 1e-8);
/// assert_eq!(fit.predict(&[12.0])?, 1);
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// - [`SymplexError::InvalidArgument`] for an empty `y`, fewer than two
///   categories, a category in `0..k` with no observations, mismatched or
///   ragged `x`, non-finite entries, `n ≤ (k − 1)p`, collinear regressors,
///   or `max_iter = 0` / a `tol` that is not finite and positive.
/// - [`SymplexError::ComputationFailed`] for perfect separation (see above).
pub fn mnlogit(
    y: &[usize],
    x: &[Vec<f64>],
    add_intercept: bool,
    opts: &LogitOpts,
) -> Result<MnLogit, SymplexError> {
    const OP: &str = "mnlogit";
    let data = categorical_data(OP, y, x, add_intercept, opts)?;
    let n = data.y.len();
    let k = data.counts.len();
    let p = data.design.first().map_or(0, Vec::len);
    let m = (k - 1) * p;
    if n <= m {
        return Err(invalid(
            OP,
            format!("need more observations than parameters: n = {n}, (k − 1)·p = {m}"),
        ));
    }
    let fit = fit_multinomial(OP, &data, add_intercept, opts, true)?;
    let by_category = |v: &[f64]| -> Vec<Vec<f64>> { v.chunks(p).map(<[f64]>::to_vec).collect() };
    let log_likelihood = fit.eval.ll;
    let null_log_likelihood = categorical_null_log_likelihood(&data.counts);
    let coefficients = by_category(&fit.params);
    // From the reported coefficients on the original rows, so that they
    // are exactly `predict_proba` of those rows.
    let fitted_probabilities = data
        .design
        .iter()
        .map(|row| softmax_probabilities(&coefficients, row))
        .collect();
    Ok(MnLogit {
        coefficients,
        standard_errors: by_category(&fit.wald.se),
        z_values: by_category(&fit.wald.z),
        p_values: by_category(&fit.wald.p),
        log_likelihood,
        null_log_likelihood,
        pseudo_r_squared: 1.0 - log_likelihood / null_log_likelihood,
        iterations: fit.iterations,
        converged: fit.converged,
        cov_params: fit.wald.cov,
        n_categories: k,
        n_params: p,
        fitted_probabilities,
        nobs: n,
        df_model: (k - 1) * (p - 1),
        df_resid: n - m,
        added_intercept: add_intercept,
        flat_coefficients: fit.params,
        flat_standard_errors: fit.wald.se,
    })
}

/// A multinomial (or, with `k = 2`, binary) logit fit before it is
/// dressed as [`MnLogit`] or [`Logit`]: the flat coefficients in the
/// original units and their Wald summary, the final evaluation (fitted
/// probabilities, `ℓ`), the iteration count and the convergence flag.
struct MultinomialFit {
    params: Vec<f64>,
    wald: WaldSummary,
    eval: CategoricalEval,
    iterations: usize,
    converged: bool,
}

/// The shared fit of [`logit`] and [`mnlogit`]: Newton–Raphson with step
/// halving on the standardised design ([`Standardization`]), the estimate
/// and its covariance mapped back to the original columns.  The caller has
/// checked `n > (k − 1)p`.
fn fit_multinomial(
    op: &'static str,
    data: &CategoricalData,
    add_intercept: bool,
    opts: &LogitOpts,
    per_category: bool,
) -> Result<MultinomialFit, SymplexError> {
    let k = data.counts.len();
    let p = data.design.first().map_or(0, Vec::len);
    let change = Standardization::new(&data.design, false);
    let design = change.apply(&data.design);
    let a_map = block_diagonal(&change.linear_predictor_map(), k - 1);
    let evaluate = |gamma: &[f64]| Some(mnlogit_evaluate(&design, &data.y, k, gamma));
    let culprit = |gamma: &[f64]| {
        diverging_coefficient(
            &map_vector(&a_map, gamma),
            p,
            &data.rms,
            add_intercept,
            per_category,
        )
    };
    let out = newton_categorical(
        op,
        opts,
        &data.y,
        vec![0.0; (k - 1) * p],
        &evaluate,
        &culprit,
    )?;
    let params = map_vector(&a_map, &out.params);
    let root = mnlogit_root_information(&design, k, &out.params);
    let wald = mapped_wald_summary(op, &root, &a_map, &params)?;
    Ok(MultinomialFit {
        params,
        wald,
        eval: out.eval,
        iterations: out.iterations,
        converged: out.converged,
    })
}

impl LikelihoodFit for MnLogit {
    fn log_likelihood(&self) -> f64 {
        self.log_likelihood
    }

    fn null_log_likelihood(&self) -> f64 {
        self.null_log_likelihood
    }

    /// `(k − 1)p`, every equation's parameters together (the
    /// [`n_params`](MnLogit::n_params) field is the per-equation `p`).
    fn n_params(&self) -> usize {
        (self.n_categories - 1) * self.n_params
    }

    fn nobs(&self) -> usize {
        self.nobs
    }

    fn df_model(&self) -> usize {
        self.df_model
    }
}

/// Over the flat index `(j − 1)·p + a` of [`cov_params`](MnLogit::cov_params).
impl WaldFit for MnLogit {
    fn coefficients(&self) -> &[f64] {
        &self.flat_coefficients
    }

    fn standard_errors(&self) -> &[f64] {
        &self.flat_standard_errors
    }
}

/// `P(y = j | x)` of a multinomial logit with reference category `0` and
/// one coefficient vector per other category (a max-shifted softmax).
fn softmax_probabilities(coefficients: &[Vec<f64>], row: &[f64]) -> Vec<f64> {
    let mut eta: Vec<f64> = std::iter::once(0.0)
        .chain(coefficients.iter().map(|b| dot_f64(row, b)))
        .collect();
    let mx = eta.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    for e in &mut eta {
        *e = (*e - mx).exp();
    }
    let s: f64 = eta.iter().sum();
    eta.into_iter().map(|e| e / s).collect()
}

impl MnLogit {
    fn probabilities(&self, row: &[f64]) -> Vec<f64> {
        softmax_probabilities(&self.coefficients, row)
    }

    /// `P(y = j | x₀)` for `j = 0, …, k − 1` (`predict(x)`).
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] for a wrong length or a non-finite
    /// entry.
    pub fn predict_proba(&self, x_row: &[f64]) -> Result<Vec<f64>, SymplexError> {
        let row =
            categorical_design_row("predict_proba", x_row, self.n_params, self.added_intercept)?;
        Ok(self.probabilities(&row))
    }

    /// The most probable category at `x₀` (`predict(x).argmax()`; the
    /// lowest on ties).
    ///
    /// # Errors
    ///
    /// As [`predict_proba`](Self::predict_proba).
    pub fn predict(&self, x_row: &[f64]) -> Result<usize, SymplexError> {
        let row = categorical_design_row("predict", x_row, self.n_params, self.added_intercept)?;
        Ok(modal_category(&self.probabilities(&row)))
    }

    /// `exp(β̂_{j,a})`: the multiplicative change in `P(y = j)/P(y = 0)` per
    /// unit of regressor `a` (`np.exp(params.T)`), same shape as
    /// [`coefficients`](Self::coefficients).
    #[must_use]
    pub fn relative_risk_ratios(&self) -> Vec<Vec<f64>> {
        self.coefficients
            .iter()
            .map(|b| b.iter().map(|v| v.exp()).collect())
            .collect()
    }

    /// Wald intervals `β̂ ± z_{(1+c)/2} · se` for every coefficient, same
    /// shape as [`coefficients`](Self::coefficients) (`conf_int(alpha = 1 − c)`).
    /// [`WaldFit::conf_int`] gives the same intervals over the flat index
    /// of [`cov_params`](Self::cov_params).
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] for `confidence ∉ (0, 1)`.
    pub fn conf_int(&self, confidence: f64) -> Result<Vec<Vec<Interval<f64>>>, SymplexError> {
        let flat = <Self as WaldFit>::conf_int(self, confidence)?;
        Ok(flat
            .chunks(self.n_params.max(1))
            .map(<[Interval<f64>]>::to_vec)
            .collect())
    }

    /// Likelihood-ratio statistic `2(ℓ − ℓ₀)` against the intercept-only
    /// model (`llr`), asymptotically `χ²_{df_model}`; [`LikelihoodFit::llr`].
    #[must_use]
    pub fn llr(&self) -> f64 {
        <Self as LikelihoodFit>::llr(self)
    }

    /// The likelihood-ratio test of every slope being zero:
    /// [`llr`](Self::llr) referred to `χ²_{(k−1)(p−1)}` (`llr_pvalue`, whose
    /// `df_model` is `(J − 1)(K − 1)`); [`LikelihoodFit::llr_test`].
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] for an intercept-only model
    /// (`df_model = 0`) or a non-finite statistic.
    pub fn llr_test(&self, ctx: &Context) -> Result<TestResult, SymplexError> {
        <Self as LikelihoodFit>::llr_test(self, ctx)
    }

    /// `AIC = −2ℓ + 2(k − 1)p` (`aic`); [`LikelihoodFit::aic`].
    #[must_use]
    pub fn aic(&self) -> f64 {
        <Self as LikelihoodFit>::aic(self)
    }

    /// `BIC = −2ℓ + (k − 1)p ln n` (`bic`); [`LikelihoodFit::bic`].
    #[must_use]
    pub fn bic(&self) -> f64 {
        <Self as LikelihoodFit>::bic(self)
    }
}

// ── Ordinal (proportional-odds) logit ───────────────────────────────────

/// A fitted proportional-odds (cumulative-link) logistic regression
/// (statsmodels `OrderedModel(y, X, distr='logit').fit()`), in `f64`:
/// `P(y ≤ j | x) = σ(θ_j − xᵀβ)` for `j = 0, …, k − 2`, with thresholds
/// `θ_0 < θ_1 < … < θ_{k−2}` and no intercept.
///
/// # Parametrisation
///
/// The thresholds are reported **as thresholds**.  statsmodels reports
/// `params = (β, θ_0, ln(θ_1 − θ_0), …, ln(θ_{k−2} − θ_{k−3}))` and
/// recovers `θ` with `transform_threshold_params`; its `bse` for the
/// log-differences are therefore not comparable to
/// [`standard_errors`](Self::standard_errors) beyond `β` and `θ_0` (the
/// delta method `J · cov · Jᵀ` with `∂θ_j/∂α_m = exp(α_m)` for `1 ≤ m ≤ j`
/// converts them).  `standard_errors`, `z_values`, `p_values` and
/// `cov_params` are over the vector `(β_0, …, β_{p−1}, θ_0, …, θ_{k−2})`.
#[derive(Clone, Debug, PartialEq)]
pub struct OrderedLogit {
    /// `θ̂_0 < … < θ̂_{k−2}` (`transform_threshold_params(params)[1:-1]`).
    pub thresholds: Vec<f64>,
    /// `β̂`, the slopes (`params[:p]`).
    pub coefficients: Vec<f64>,
    /// `√diag(I⁻¹)` over `(β, θ)`, slopes first.
    pub standard_errors: Vec<f64>,
    /// `(β̂, θ̂) / se`.
    pub z_values: Vec<f64>,
    /// Two-sided normal p-values `erfc(|z|/√2)`.
    pub p_values: Vec<f64>,
    /// `ℓ = Σᵢ ln [σ(θ_{yᵢ} − xᵢᵀβ) − σ(θ_{yᵢ−1} − xᵢᵀβ)]` (`llf`).
    pub log_likelihood: f64,
    /// `Σ_j n_j ln(n_j / n)`, the thresholds-only model (`llnull`).
    pub null_log_likelihood: f64,
    /// McFadden's `1 − ℓ/ℓ₀` (`prsquared`).
    pub pseudo_r_squared: f64,
    /// Newton steps taken.
    pub iterations: usize,
    /// Whether an accepted full Newton step met the step criterion within
    /// `max_iter`.
    pub converged: bool,
    /// `I(β̂, θ̂)⁻¹` over `(β, θ)`, slopes first.
    pub cov_params: Vec<Vec<f64>>,
    /// `π̂_{ij}` for every observation and category, `n × k` (`predict()`).
    pub fitted_probabilities: Vec<Vec<f64>>,
    /// Number of observations (`nobs`).
    pub nobs: usize,
    /// `k`, the number of ordered categories (`k_levels`).
    pub n_categories: usize,
    /// `p`, the number of slopes (`df_model = k_vars`).
    pub df_model: usize,
    /// `n − p − (k − 1)` (`df_resid`).
    pub df_resid: usize,
}

/// `ln σ(z)`, stable for large `|z|`: `ln P(Y ≤ j)`.
fn log_sigmoid(z: f64) -> f64 {
    -softplus(-z)
}

/// `ln(1 − e^{−a})` for `a > 0`: `ln(−expm1(−a))` for `a ≤ ln 2`, where
/// `1 − e^{−a}` is small and `ln_1p(−e^{−a})` would take the logarithm of
/// a cancelled difference, and `ln_1p(−e^{−a})` beyond (the switch point
/// of M. Mächler, "Accurately computing log(1 − exp(−|a|))", 2012).
fn log1mexp(a: f64) -> f64 {
    if a <= std::f64::consts::LN_2 {
        (-(-a).exp_m1()).ln()
    } else {
        (-(-a).exp()).ln_1p()
    }
}

/// One observation of the ordered logit in category `c` with linear
/// predictor `η`: bounds `u = θ_c − η` (absent for the top category) and
/// `l = θ_{c−1} − η` (absent for the bottom one), probability
/// `P = σ(u) − σ(l) = σ(u) σ(−l) (1 − e^{l−u})`.
///
/// With `r = 1/expm1(u − l)` (`0` unless both bounds exist),
/// `ln P = ln σ(u) + ln σ(−l) + ln(1 − e^{−(u−l)})` has the derivatives
/// `∂ln P/∂u = σ(−u) + r`, `∂ln P/∂l = −σ(l) − r` and the information
/// `−∂²ln P = [[σ(u)σ(−u) + q, −q], [−q, σ(l)σ(−l) + q]]`, `q = r(1 + r)`:
/// sums of same-signed terms, so neither cancels, and none divides by `P`
/// (the old `∂P/P` forms were `0/0` once `P` underflowed, and lost digits
/// when `l ≈ u`).
struct OrderedTerm {
    log_p: f64,
    /// `∂ln P/∂u` (unused without `u`).
    g_upp: f64,
    /// `∂ln P/∂l` (unused without `l`).
    g_low: f64,
    /// `σ(−u) − σ(l) = −∂ln P/∂η` (the `r` terms cancel exactly).
    d_eta: f64,
    /// `σ(u)σ(−u)`, `0` without `u`.
    s_upp: f64,
    /// `σ(l)σ(−l)`, `0` without `l`.
    s_low: f64,
    /// `r(1 + r)`.
    q: f64,
}

fn ordered_term(low: Option<f64>, upp: Option<f64>) -> OrderedTerm {
    let (sig_neg_u, s_upp, log_upp) = upp.map_or((0.0, 0.0, 0.0), |u| {
        let neg = sigmoid(-u);
        (neg, sigmoid(u) * neg, log_sigmoid(u))
    });
    let (sig_l, s_low, log_low) = low.map_or((0.0, 0.0, 0.0), |l| {
        let pos = sigmoid(l);
        (pos, pos * sigmoid(-l), log_sigmoid(-l))
    });
    let (r, log_gap) = match (low, upp) {
        (Some(l), Some(u)) => {
            let a = u - l;
            (1.0 / a.exp_m1(), log1mexp(a))
        }
        _ => (0.0, 0.0),
    };
    OrderedTerm {
        log_p: log_upp + log_low + log_gap,
        g_upp: sig_neg_u + r,
        g_low: -(sig_l + r),
        d_eta: sig_neg_u - sig_l,
        s_upp,
        s_low,
        q: r * (1.0 + r),
    }
}

/// `P = σ(u) σ(−l) (−expm1(l − u))` of one ordered category (a missing bound
/// contributes the factor `1`): every factor is accurate to a few ulps and
/// at most `1`, so the product is too — no `σ(u) − σ(l)` or `1 − σ(l)`
/// subtraction, which returned `0` for a true `1e-20`.
fn ordered_probability(low: Option<f64>, upp: Option<f64>) -> f64 {
    let below_upp = upp.map_or(1.0, sigmoid);
    let above_low = low.map_or(1.0, |l| sigmoid(-l));
    let gap = match (low, upp) {
        (Some(l), Some(u)) => -(l - u).exp_m1(),
        _ => 1.0,
    };
    below_upp * above_low * gap
}

/// Log-likelihood, score and observed information of the ordered logit at
/// `par = (β, θ)` ([`OrderedTerm`] per observation, chained through
/// `u = θ_c − xᵀβ`, `l = θ_{c−1} − xᵀβ`); `None` when the thresholds are
/// not strictly increasing.
fn ologit_evaluate(
    design: &[Vec<f64>],
    y: &[usize],
    k: usize,
    par: &[f64],
) -> Option<CategoricalEval> {
    let p = design.first().map_or(0, Vec::len);
    let q = p + k - 1;
    let (beta, theta) = par.split_at(p);
    // `partial_cmp` rejects a NaN threshold too.
    if theta
        .windows(2)
        .any(|w| w[1].partial_cmp(&w[0]) != Some(std::cmp::Ordering::Greater))
    {
        return None;
    }
    let mut ll = 0.0;
    let mut score = vec![0.0; q];
    let mut info = vec![vec![0.0; q]; q];
    let mut probs = Vec::with_capacity(design.len());
    for (row, &c) in design.iter().zip(y) {
        let eta = dot_f64(row, beta);
        let has_upp = c < k - 1;
        let has_low = c > 0;
        let t = ordered_term(
            has_low.then(|| theta[c - 1] - eta),
            has_upp.then(|| theta[c] - eta),
        );
        ll += t.log_p;
        let s_eta = t.s_upp + t.s_low;
        for (a, xa) in row.iter().enumerate() {
            score[a] -= xa * t.d_eta;
            for (b, xb) in row.iter().enumerate() {
                info[a][b] += xa * xb * s_eta;
            }
            if has_upp {
                info[a][p + c] -= xa * t.s_upp;
                info[p + c][a] -= xa * t.s_upp;
            }
            if has_low {
                info[a][p + c - 1] -= xa * t.s_low;
                info[p + c - 1][a] -= xa * t.s_low;
            }
        }
        if has_upp {
            score[p + c] += t.g_upp;
            info[p + c][p + c] += t.s_upp + t.q;
        }
        if has_low {
            score[p + c - 1] += t.g_low;
            info[p + c - 1][p + c - 1] += t.s_low + t.q;
        }
        if has_upp && has_low {
            info[p + c][p + c - 1] -= t.q;
            info[p + c - 1][p + c] -= t.q;
        }
        probs.push(ordered_probabilities(theta, eta));
    }
    Some(CategoricalEval {
        ll,
        score,
        info,
        probs,
    })
}

/// `P(y = j | η)` for `j = 0, …, k − 1` ([`ordered_probability`]).
fn ordered_probabilities(theta: &[f64], eta: f64) -> Vec<f64> {
    let k = theta.len() + 1;
    (0..k)
        .map(|j| {
            ordered_probability(
                (j > 0).then(|| theta[j - 1] - eta),
                (j < k - 1).then(|| theta[j] - eta),
            )
        })
        .collect()
}

/// Ordinal (proportional-odds, cumulative-link) logistic regression
/// `P(y ≤ j | x) = σ(θ_j − xᵀβ)`, `j = 0, …, k − 2`, fitted by
/// Newton–Raphson with step halving on the cut points and slopes of the
/// standardised design (every column centred and scaled to unit
/// root-mean-square deviation, the centring absorbed by the cut points;
/// see [`logit`]) — an affine reparametrisation of `(β, θ)`, so the
/// log-likelihood stays concave (Pratt 1981).  A step that would disorder
/// the thresholds is rejected and halved, and the starting point `β = 0`,
/// `θ_j = logit(cumulative frequency of y ≤ j)` is feasible.  Each
/// observation's log-likelihood `ln[σ(u) − σ(l)]` and its derivatives are
/// evaluated without cancellation or division by the probability (see
/// the source's `OrderedTerm`).  Standard errors come from the observed
/// information at the estimate, mapped back to `(β, θ)` (see
/// [`OrderedLogit`]).
/// `statsmodels.miscmodels.ordinal_model.OrderedModel(y, x, distr='logit').fit()`.
///
/// `y[i] ∈ 0..k` (ordered, every category observed at least once); `x`
/// holds one row of regressors per observation and **must not contain a
/// constant column** — the thresholds play the intercept's role
/// (statsmodels: "There should not be a constant in the model").  With
/// `k = 2` the fit is [`logit`]'s with intercept `−θ_0`.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::regression::{ologit, LogitOpts};
///
/// let y = [0, 0, 1, 0, 1, 1, 2, 1, 2, 2, 0, 1, 2, 2, 1, 0, 0, 1, 2, 2];
/// let x: Vec<Vec<f64>> = (0..20).map(|i| vec![f64::from(i)]).collect();
/// // statsmodels: OrderedModel(y, x, distr='logit').fit(method='newton').params =
/// //   [0.11402351915116841, 0.12118510043763872, 0.47652318845685754]  (β, θ₀, ln(θ₁ − θ₀))
/// //   transform_threshold_params(params)[1:-1] = [0.12118510043763872, 1.731650472914306]
/// let fit = ologit(&y, &x, &LogitOpts::default())?;
/// assert!(fit.converged);
/// assert!((fit.coefficients[0] - 0.11402351915116841).abs() < 1e-7);
/// assert!((fit.thresholds[1] - 1.731650472914306).abs() < 1e-7);
/// assert!((fit.log_likelihood - -20.751965417580625).abs() < 1e-9);
/// let probs = fit.predict_proba(&[7.0])?;              // predict(exog=[[7.0]])
/// assert!((probs[1] - 0.38084617872902227).abs() < 1e-7);
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// - [`SymplexError::InvalidArgument`] for an empty `y`, fewer than two
///   categories, a category in `0..k` with no observations (statsmodels
///   silently relabels a numpy `y` with `np.unique`, and fails on an
///   explicitly declared empty level: its start value `ln 0 = −∞`),
///   mismatched or ragged `x`, non-finite entries, a constant column,
///   `n ≤ p + k − 1`, collinear regressors, or `max_iter = 0` / a `tol` that is not finite and positive.
/// - [`SymplexError::ComputationFailed`] for perfect separation, as
///   [`mnlogit`].
pub fn ologit(y: &[usize], x: &[Vec<f64>], opts: &LogitOpts) -> Result<OrderedLogit, SymplexError> {
    const OP: &str = "ologit";
    let data = categorical_data(OP, y, x, false, opts)?;
    let n = data.y.len();
    let k = data.counts.len();
    let p = data.design.first().map_or(0, Vec::len);
    let q = p + k - 1;
    if let Some(a) = (0..p).find(|&a| data.design.iter().all(|r| r[a] == data.design[0][a])) {
        return Err(invalid(
            OP,
            format!(
                "column {a} of x is constant: the thresholds play the intercept's role, drop it"
            ),
        ));
    }
    if n <= q {
        return Err(invalid(
            OP,
            format!("need more observations than parameters: n = {n}, p + k − 1 = {q}"),
        ));
    }
    // statsmodels' start_params: β = 0, θ_j = logit(F̂(j)) = ln(N_j/(n − N_j))
    // for the cumulative count N_j; the standardised start (γ, τ) = (0, θ)
    // is the same point.
    let mut start = vec![0.0; q];
    let mut cum = 0usize;
    for (j, &c) in data.counts.iter().take(k - 1).enumerate() {
        cum += c;
        start[p + j] = (cum as f64 / (n - cum) as f64).ln();
    }
    let change = Standardization::new(&data.design, true);
    let design = change.apply(&data.design);
    let a_map = change.ordered_map(k - 1);
    let evaluate = |par: &[f64]| ologit_evaluate(&design, &data.y, k, par);
    let culprit = |par: &[f64]| {
        diverging_coefficient(&map_vector(&a_map, par)[..p], p, &data.rms, false, false)
    };
    let out = newton_categorical(OP, opts, &data.y, start, &evaluate, &culprit)?;
    let params = map_vector(&a_map, &out.params);
    let root = ologit_root_information(&design, &data.y, k, &out.params);
    let wald = mapped_wald_summary(OP, &root, &a_map, &params)?;
    let log_likelihood = out.eval.ll;
    let null_log_likelihood = categorical_null_log_likelihood(&data.counts);
    let (beta, theta) = params.split_at(p);
    // From the reported (β, θ) on the original rows, so that they are
    // exactly `predict_proba` of those rows.
    let fitted_probabilities = data
        .design
        .iter()
        .map(|row| ordered_probabilities(theta, dot_f64(row, beta)))
        .collect();
    Ok(OrderedLogit {
        thresholds: theta.to_vec(),
        coefficients: beta.to_vec(),
        standard_errors: wald.se,
        z_values: wald.z,
        p_values: wald.p,
        log_likelihood,
        null_log_likelihood,
        pseudo_r_squared: 1.0 - log_likelihood / null_log_likelihood,
        iterations: out.iterations,
        converged: out.converged,
        cov_params: wald.cov,
        fitted_probabilities,
        nobs: n,
        n_categories: k,
        df_model: p,
        df_resid: n - q,
    })
}

impl LikelihoodFit for OrderedLogit {
    fn log_likelihood(&self) -> f64 {
        self.log_likelihood
    }

    fn null_log_likelihood(&self) -> f64 {
        self.null_log_likelihood
    }

    /// `p + k − 1`: slopes and thresholds together.
    fn n_params(&self) -> usize {
        self.coefficients.len() + self.thresholds.len()
    }

    fn nobs(&self) -> usize {
        self.nobs
    }

    fn df_model(&self) -> usize {
        self.df_model
    }
}

/// Over the slopes `β` only: the thresholds' Wald quantities are the tail
/// of the [`standard_errors`](OrderedLogit::standard_errors),
/// [`z_values`](OrderedLogit::z_values) and [`p_values`](OrderedLogit::p_values)
/// fields.
impl WaldFit for OrderedLogit {
    fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    fn standard_errors(&self) -> &[f64] {
        let p = self.coefficients.len().min(self.standard_errors.len());
        &self.standard_errors[..p]
    }
}

impl OrderedLogit {
    /// `p + k − 1`: slopes and thresholds together.
    #[must_use]
    pub fn n_params(&self) -> usize {
        <Self as LikelihoodFit>::n_params(self)
    }

    fn linear_predictor(&self, op: &'static str, x_row: &[f64]) -> Result<f64, SymplexError> {
        let row = categorical_design_row(op, x_row, self.coefficients.len(), false)?;
        Ok(dot_f64(&row, &self.coefficients))
    }

    /// `P(y = j | x₀)` for `j = 0, …, k − 1` (`predict(exog)`).
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] for a wrong length or a non-finite
    /// entry.
    pub fn predict_proba(&self, x_row: &[f64]) -> Result<Vec<f64>, SymplexError> {
        let eta = self.linear_predictor("predict_proba", x_row)?;
        Ok(ordered_probabilities(&self.thresholds, eta))
    }

    /// `P(y ≤ j | x₀) = σ(θ̂_j − x₀ᵀβ̂)` for `j = 0, …, k − 1` (the last entry
    /// is `1`; `predict(exog, which='cumprob')`).
    ///
    /// # Errors
    ///
    /// As [`predict_proba`](Self::predict_proba).
    pub fn cumulative_proba(&self, x_row: &[f64]) -> Result<Vec<f64>, SymplexError> {
        let eta = self.linear_predictor("cumulative_proba", x_row)?;
        Ok(self
            .thresholds
            .iter()
            .map(|t| sigmoid(t - eta))
            .chain(std::iter::once(1.0))
            .collect())
    }

    /// The most probable category at `x₀` (the lowest on ties).
    ///
    /// # Errors
    ///
    /// As [`predict_proba`](Self::predict_proba).
    pub fn predict(&self, x_row: &[f64]) -> Result<usize, SymplexError> {
        let eta = self.linear_predictor("predict", x_row)?;
        Ok(modal_category(&ordered_probabilities(
            &self.thresholds,
            eta,
        )))
    }

    /// `exp(β̂_a)`: the multiplicative change in the odds of `y > j` (any
    /// `j`) per unit of regressor `a` (`np.exp(params[:p])`).
    #[must_use]
    pub fn odds_ratios(&self) -> Vec<f64> {
        self.coefficients.iter().map(|b| b.exp()).collect()
    }

    /// Wald intervals `β̂_a ± z_{(1+c)/2} · se_a` for the slopes
    /// (`conf_int(alpha = 1 − c)[:p]`); [`WaldFit::conf_int`].
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] for `confidence ∉ (0, 1)`.
    pub fn conf_int(&self, confidence: f64) -> Result<Vec<Interval<f64>>, SymplexError> {
        <Self as WaldFit>::conf_int(self, confidence)
    }

    /// Likelihood-ratio statistic `2(ℓ − ℓ₀)` against the thresholds-only
    /// model (`llr`), asymptotically `χ²_p`; [`LikelihoodFit::llr`].
    #[must_use]
    pub fn llr(&self) -> f64 {
        <Self as LikelihoodFit>::llr(self)
    }

    /// The likelihood-ratio test of `β = 0`: [`llr`](Self::llr) referred to
    /// `χ²_p` (`llr_pvalue`, `df_model = k_vars`); [`LikelihoodFit::llr_test`].
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] for a non-finite statistic.
    pub fn llr_test(&self, ctx: &Context) -> Result<TestResult, SymplexError> {
        <Self as LikelihoodFit>::llr_test(self, ctx)
    }

    /// `AIC = −2ℓ + 2(p + k − 1)` (`aic`); [`LikelihoodFit::aic`].
    #[must_use]
    pub fn aic(&self) -> f64 {
        <Self as LikelihoodFit>::aic(self)
    }

    /// `BIC = −2ℓ + (p + k − 1) ln n` (`bic`); [`LikelihoodFit::bic`].
    #[must_use]
    pub fn bic(&self) -> f64 {
        <Self as LikelihoodFit>::bic(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn o1() -> (Vec<usize>, Vec<Vec<f64>>) {
        let y = vec![0, 0, 1, 0, 1, 1, 2, 1, 2, 2, 0, 1, 2, 2, 1, 0, 0, 1, 2, 2];
        let x = (0..20).map(|i| vec![f64::from(i) / 10.0]).collect();
        (y, x)
    }

    fn m2() -> (Vec<usize>, Vec<Vec<f64>>) {
        let y = vec![
            0, 3, 1, 1, 2, 1, 2, 0, 3, 3, 0, 3, 0, 1, 1, 2, 2, 3, 0, 1, 2, 3, 3, 3, 0, 3, 2, 3, 2,
            2,
        ];
        let x1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x2 = [
            3.0, 1.0, 4.0, 1.0, 5.0, 2.0, 6.0, 5.0, 3.0, 5.0, 8.0, 9.0, 7.0, 9.0, 3.0, 2.0, 3.0,
            8.0, 4.0, 6.0, 2.0, 6.0, 4.0, 3.0, 3.0, 8.0, 3.0, 2.0, 7.0, 9.0,
        ];
        let x = (0..30).map(|i| vec![1.0, x1[i % 6], x2[i]]).collect();
        (y, x)
    }

    /// Central differences of `f` at `at`, one coordinate at a time.
    fn numeric_gradient(f: &dyn Fn(&[f64]) -> f64, at: &[f64], h: f64) -> Vec<f64> {
        (0..at.len())
            .map(|j| {
                let mut up = at.to_vec();
                let mut dn = at.to_vec();
                up[j] += h;
                dn[j] -= h;
                (f(&up) - f(&dn)) / (2.0 * h)
            })
            .collect()
    }

    #[test]
    fn ologit_score_is_the_gradient_of_the_log_likelihood() {
        let (y, x) = o1();
        let at = [0.8, -0.2, 1.1];
        let e = ologit_evaluate(&x, &y, 3, &at).unwrap();
        let ll = |par: &[f64]| ologit_evaluate(&x, &y, 3, par).unwrap().ll;
        for (s, n) in e.score.iter().zip(numeric_gradient(&ll, &at, 1e-6)) {
            assert!((s - n).abs() < 1e-6, "score {s} vs numeric {n}");
        }
    }

    #[test]
    fn ologit_information_is_minus_the_hessian() {
        let (y, x) = o1();
        let at = [0.8, -0.2, 1.1];
        let e = ologit_evaluate(&x, &y, 3, &at).unwrap();
        for r in 0..3 {
            let score_r = |par: &[f64]| ologit_evaluate(&x, &y, 3, par).unwrap().score[r];
            let row = numeric_gradient(&score_r, &at, 1e-6);
            for (s, numeric) in row.iter().enumerate() {
                assert!(
                    (e.info[r][s] + numeric).abs() < 1e-5,
                    "info[{r}][{s}] = {} vs −∂²ℓ = {}",
                    e.info[r][s],
                    -numeric
                );
                assert!((e.info[r][s] - e.info[s][r]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn ologit_rejects_disordered_thresholds() {
        let (y, x) = o1();
        assert!(ologit_evaluate(&x, &y, 3, &[0.1, 1.0, 1.0]).is_none());
        assert!(ologit_evaluate(&x, &y, 3, &[0.1, 1.0, 0.5]).is_none());
        assert!(ologit_evaluate(&x, &y, 3, &[0.1, 0.5, 1.0]).is_some());
    }

    #[test]
    fn mnlogit_score_and_information_match_finite_differences() {
        let (y, x) = m2();
        let at = [-0.5, 0.3, -0.1, -1.0, 0.6, 0.0, -1.2, 0.7, -0.05];
        let e = mnlogit_evaluate(&x, &y, 4, &at);
        let ll = |b: &[f64]| mnlogit_evaluate(&x, &y, 4, b).ll;
        for (s, n) in e.score.iter().zip(numeric_gradient(&ll, &at, 1e-6)) {
            assert!((s - n).abs() < 1e-5, "score {s} vs numeric {n}");
        }
        for r in 0..9 {
            let score_r = |b: &[f64]| mnlogit_evaluate(&x, &y, 4, b).score[r];
            let row = numeric_gradient(&score_r, &at, 1e-6);
            for (s, numeric) in row.iter().enumerate() {
                assert!(
                    (e.info[r][s] + numeric).abs() < 1e-4,
                    "info[{r}][{s}] = {} vs −∂²ℓ = {}",
                    e.info[r][s],
                    -numeric
                );
            }
        }
    }

    /// Between cut points `1` and `1 + 2⁻⁴⁰` the old score `f(u)/P` divided
    /// a difference of densities by a cancelled `σ(u) − σ(l)`.
    #[test]
    fn ordered_term_between_nearly_tied_thresholds() {
        let t = ordered_term(Some(1.0), Some(1.0 + 2f64.powi(-40)));
        // mpmath (60 digits): u = 1 + 2^-40, l = 1: log(sigma(u) - sigma(l)) = -29.35241059743446819134034,
        //   d/du = 1099511627775.76894142137, d/dl = -1099511627776.23105857863
        let rel = |a: f64, b: f64| ((a - b) / b).abs();
        assert!(rel(t.log_p, -29.352_410_597_434_468) < 1e-15, "{}", t.log_p);
        assert!(rel(t.g_upp, 1_099_511_627_775.768_9) < 1e-12, "{}", t.g_upp);
        assert!(rel(t.g_low, -1_099_511_627_776.231) < 1e-12, "{}", t.g_low);
    }

    /// An observation whose probability underflows (`P = e^{−800}`): the
    /// old `∂P/P` and `∂P∂Pᵀ/P²` were `0/0`.
    #[test]
    fn ologit_derivatives_stay_finite_when_a_probability_underflows() {
        let x = vec![vec![0.0], vec![1.0], vec![-800.0]];
        let y = [0, 1, 1];
        // Category 1 of 2 at η = −8000: l = θ₀ + 8000, P = σ(−l) = e^{−8000}.
        let e = ologit_evaluate(&x, &y, 2, &[10.0, 0.0]).unwrap();
        assert!(e.ll.is_finite() && e.ll < -7_999.0, "{}", e.ll);
        assert!(e.score.iter().all(|v| v.is_finite()), "{:?}", e.score);
        assert!(
            e.info.iter().flatten().all(|v| v.is_finite()),
            "{:?}",
            e.info
        );
        // ∂ln P/∂θ₀ = −σ(l) = −1 for the underflowing observation.
        let lone = ologit_evaluate(&x[2..], &y[2..], 2, &[10.0, 0.0]).unwrap();
        assert_eq!(lone.score[1], -1.0);
    }

    #[test]
    fn ordered_probabilities_sum_to_one() {
        let pr = ordered_probabilities(&[-1.0, 0.5, 2.0], 0.3);
        assert_eq!(pr.len(), 4);
        assert!((pr.iter().sum::<f64>() - 1.0).abs() < 1e-15);
        assert!(pr.iter().all(|v| *v > 0.0));
    }
}
