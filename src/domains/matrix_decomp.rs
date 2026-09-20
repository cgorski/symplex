//! Additional matrix decompositions and structure tests.
//!
//! Everything here is a method on [`Matrix`] (or a free function taking
//! matrices); the module split is purely organisational.
//!
//! * **Decompositions:** [`Matrix::qr`] (Gram–Schmidt, exact with
//!   radicals), [`Matrix::cholesky`], [`Matrix::ldl`], the free function
//!   [`gram_schmidt`].
//! * **Structure tests** (`Option<bool>`, `None` = undecidable):
//!   [`Matrix::is_symmetric`], [`Matrix::is_hermitian`],
//!   [`Matrix::is_orthogonal`], [`Matrix::is_unitary`],
//!   [`Matrix::is_positive_definite`], [`Matrix::is_positive_semidefinite`],
//!   [`Matrix::is_upper_triangular`], [`Matrix::is_lower_triangular`],
//!   [`Matrix::is_diagonal`], [`Matrix::is_identity`], [`Matrix::is_zero`],
//!   [`Matrix::is_skew_symmetric`], [`Matrix::is_nilpotent`].
//! * **Norms:** [`Matrix::norm_1`], [`Matrix::norm_inf`],
//!   [`Matrix::norm_p`] (vectors), plus
//!   [`norm_frobenius`](Matrix::norm_frobenius) in the core module.
//! * **Functions of diagonalizable matrices:**
//!   [`Matrix::matrix_pow_symbolic`], [`Matrix::matrix_sqrt`],
//!   [`Matrix::matrix_log`] (any Jordan form with non-zero eigenvalues).
//! * **Calculus:** [`hessian`], [`wronskian`], [`Matrix::casoratian`].

use crate::api::expr::{Ex, ExprType};
use crate::base::errors::SymplexError;
use crate::domains::matrix::{
    Matrix, all3, budget_check, ex_is_nonnegative, ex_is_positive, ex_is_zero, reop,
};

/// Orthogonal basis vectors and the upper-triangular coefficient matrix
/// produced by [`gram_schmidt_cols`], both as raw row-major `Vec`s.
type GramSchmidtParts = (Vec<Vec<Ex>>, Vec<Vec<Ex>>);

fn invalid(operation: &'static str, reason: impl Into<String>) -> SymplexError {
    SymplexError::invalid_argument(operation, reason)
}

fn failed(operation: &'static str, reason: impl Into<String>) -> SymplexError {
    SymplexError::computation_failed(operation, reason)
}

/// Dot product of two equal-length column vectors given as `Vec<Ex>`.
fn dot_vec(a: &[Ex], b: &[Ex]) -> Ex {
    let mut acc = &a[0] * &b[0];
    for k in 1..a.len() {
        acc += &a[k] * &b[k];
    }
    acc
}

// ═══════════════════════════════════════════════════════════════════════════
// Gram–Schmidt / QR
// ═══════════════════════════════════════════════════════════════════════════

/// Gram–Schmidt orthogonalisation of a list of column vectors.
///
/// Returns vectors spanning the same space, pairwise orthogonal (and of
/// unit length if `normalize` is set).  Radicals are kept exact: a
/// normalised entry is `uᵢ · ‖u‖⁻¹` with the norm left as `√(‖u‖²)`, so
/// that dot products of the results cancel structurally (`qᵢ·qⱼ` is
/// `0`, `qᵢ·qᵢ` is `1` without further simplification).  Symbolic entries
/// are simplified; constant entries are only constant-folded.
///
/// # Errors
///
/// - [`SymplexError::InvalidArgument`] if `vectors` is empty, any vector
///   is not a column vector, or lengths differ.
/// - [`SymplexError::ComputationFailed`] if the vectors are linearly
///   dependent (a residual vector is provably zero), or if symbolic
///   entries swell beyond
///   [`EXPRESSION_BUDGET`](crate::domains::matrix::EXPRESSION_BUDGET).
///   Symbolic residuals whose zero-ness cannot be decided are assumed
///   non-zero.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::matrix_decomp::gram_schmidt;
///
/// let ctx = Context::new();
/// let v1 = matrix![ctx, [1], [1]];
/// let v2 = matrix![ctx, [1], [0]];
/// let q = gram_schmidt(&[v1, v2], true).unwrap();
/// // q0 = (1/√2, 1/√2), q1 = (1/√2, −1/√2)
/// assert_eq!(q[0][(0, 0)].powi(2), ctx.rational(1, 2));
/// assert_eq!(symplex::matrix::dot(&q[0], &q[1]), ctx.int(0));
/// assert_eq!(symplex::matrix::dot(&q[1], &q[1]), ctx.int(1));
/// ```
pub fn gram_schmidt(vectors: &[Matrix], normalize: bool) -> Result<Vec<Matrix>, SymplexError> {
    if vectors.is_empty() {
        return Err(invalid("gram_schmidt", "need at least one vector"));
    }
    let n = vectors[0].nrows();
    for (idx, v) in vectors.iter().enumerate() {
        if v.ncols() != 1 {
            return Err(invalid(
                "gram_schmidt",
                format!(
                    "vector {idx} is {}×{}, expected a column vector",
                    v.nrows(),
                    v.ncols()
                ),
            ));
        }
        if v.nrows() != n {
            return Err(invalid(
                "gram_schmidt",
                format!("vector {idx} has length {}, expected {n}", v.nrows()),
            ));
        }
    }
    let cols: Vec<Vec<Ex>> = vectors.iter().map(|v| v.col(0)).collect();
    let (basis, _) = gram_schmidt_cols(&cols, normalize, "gram_schmidt")?;
    Ok(basis.into_iter().map(Matrix::col_vector).collect())
}

/// Core Gram–Schmidt on raw columns.
///
/// Returns `(orthogonal vectors, R)` where `R` is the `k×k` upper
/// triangular coefficient matrix (`R[i][j] = q_i · a_j` for `i < j`,
/// `R[j][j] = ‖u_j‖`) when `normalize` is true.
fn gram_schmidt_cols(
    cols: &[Vec<Ex>],
    normalize: bool,
    op: &'static str,
) -> Result<GramSchmidtParts, SymplexError> {
    let k = cols.len();
    let zero = cols[0][0].context().zero();
    let mut q: Vec<Vec<Ex>> = Vec::with_capacity(k);
    let mut r: Vec<Vec<Ex>> = vec![vec![zero.clone(); k]; k];
    // Constant entries only need folding; `simplify` would rewrite the
    // radicals inconsistently (`√(2/3)` vs `2^(-1/2)·√3`) and break the
    // structural cancellation of dot products.
    let tidy = |e: Ex| {
        if e.is_constant() {
            e.eval()
        } else {
            e.simplify()
        }
    };

    for j in 0..k {
        let mut u = cols[j].clone();
        for i in 0..j {
            // r_ij = q_i · a_j  (q_i normalised) or (q_i · a_j)/(q_i · q_i) otherwise
            let proj = if normalize {
                tidy(dot_vec(&q[i], &cols[j]))
            } else {
                let qq = dot_vec(&q[i], &q[i]);
                tidy(&dot_vec(&q[i], &cols[j]) / &qq)
            };
            for (u_e, q_e) in u.iter_mut().zip(q[i].iter()) {
                *u_e = &*u_e - &(&proj * q_e);
            }
            r[i][j] = proj;
        }
        budget_check(u.iter(), op)?;
        let u: Vec<Ex> = u.into_iter().map(tidy).collect();
        let norm_sq = tidy(dot_vec(&u, &u));
        if ex_is_zero(&norm_sq) == Some(true) {
            return Err(failed(
                op,
                format!(
                    "vectors are linearly dependent (vector {j} lies in the span of the previous ones)"
                ),
            ));
        }
        if normalize {
            let norm = norm_sq.sqrt();
            let one = cols[0][0].context().one();
            // `√(1/n)` displays nicely for rational `n` and cancels against
            // `√n` structurally (canonical numeric radicals); for symbolic
            // `n` only `n^(-1/2)` is guaranteed to cancel.
            let inv_norm = if norm_sq.expr_type() == ExprType::Number {
                (&one / &norm_sq).sqrt()
            } else {
                &one / &norm
            };
            r[j][j] = norm;
            q.push(u.iter().map(|e| tidy(e * &inv_norm)).collect());
        } else {
            r[j][j] = cols[0][0].context().one();
            q.push(u);
        }
    }
    Ok((q, r))
}

impl Matrix {
    /// QR decomposition via Gram–Schmidt: `A = Q·R` with `Q` (m×n) having
    /// orthonormal columns and `R` (n×n) upper triangular.
    ///
    /// Works exactly with radicals (entries like `1/√2`).  Requires the
    /// columns of `A` to be linearly independent (`rank == ncols`).
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::ComputationFailed`] if the columns are
    /// linearly dependent or symbolic entries swell beyond
    /// [`EXPRESSION_BUDGET`](crate::domains::matrix::EXPRESSION_BUDGET).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let a = matrix![ctx, [1, 1], [0, 1]];
    /// let (q, r) = a.qr().unwrap();
    /// assert_eq!((&q * &r).simplify(), a);
    /// assert_eq!((&q.transpose() * &q).simplify(), Matrix::identity(&ctx, 2));
    /// assert!(r[(1, 0)].is_zero_structural());
    /// ```
    pub fn qr(&self) -> Result<(Matrix, Matrix), SymplexError> {
        budget_check(self.iter(), "qr")?;
        if self.rank() < self.ncols() {
            return Err(failed(
                "qr",
                format!(
                    "columns are linearly dependent (rank {} < {} columns)",
                    self.rank(),
                    self.ncols()
                ),
            ));
        }
        let cols: Vec<Vec<Ex>> = (0..self.ncols()).map(|j| self.col(j)).collect();
        let (q_cols, r_rows) = gram_schmidt_cols(&cols, true, "qr")?;
        let q_mats: Vec<Matrix> = q_cols.into_iter().map(Matrix::col_vector).collect();
        let q_refs: Vec<&Matrix> = q_mats.iter().collect();
        let q = Matrix::hstack(&q_refs)?;
        let r = Matrix::new(r_rows)?;
        Ok((q, r))
    }

    // ── Cholesky / LDLᵀ ────────────────────────────────────────────────

    /// Cholesky decomposition `A = L·Lᵀ` for a symmetric positive-definite
    /// matrix (`L` lower triangular with positive diagonal).
    ///
    /// Positive-definiteness is decided pivot by pivot: each diagonal
    /// pivot must be provably positive (assumption system, or numeric
    /// evaluation for constant entries).
    ///
    /// # Errors
    ///
    /// - [`SymplexError::InvalidArgument`] if the matrix is not square or
    ///   provably not symmetric.
    /// - [`SymplexError::ComputationFailed`] if a pivot is provably
    ///   non-positive (not positive definite) **or** its sign cannot be
    ///   decided symbolically.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let a = matrix![ctx, [4, 2], [2, 3]];
    /// let l = a.cholesky().unwrap();
    /// assert_eq!((&l * &l.transpose()).simplify(), a);
    /// assert!(matrix![ctx, [1, 2], [2, 1]].cholesky().is_err()); // indefinite
    /// ```
    pub fn cholesky(&self) -> Result<Matrix, SymplexError> {
        if !self.is_square() {
            return Err(invalid(
                "cholesky",
                format!(
                    "requires a square matrix, got {}×{}",
                    self.nrows(),
                    self.ncols()
                ),
            ));
        }
        if self.is_symmetric() == Some(false) {
            return Err(invalid("cholesky", "matrix is not symmetric"));
        }
        let n = self.nrows();
        let zero = self.context().zero();
        let mut l: Vec<Vec<Ex>> = vec![vec![zero.clone(); n]; n];

        for j in 0..n {
            let mut sum_sq = zero.clone();
            for item in l[j].iter().take(j) {
                sum_sq += item.powi(2);
            }
            let diag = (self.get(j, j) - &sum_sq).simplify();
            match ex_is_positive(&diag) {
                Some(true) => {}
                Some(false) => {
                    return Err(failed(
                        "cholesky",
                        format!("matrix is not positive definite (pivot {j} is {diag})"),
                    ));
                }
                None => {
                    return Err(failed(
                        "cholesky",
                        format!(
                            "cannot decide the sign of pivot {j} = {diag}; add assumptions or use ldl()"
                        ),
                    ));
                }
            }
            l[j][j] = diag.sqrt().simplify();
            for i in (j + 1)..n {
                let mut sum_prod = zero.clone();
                for (l_ik, l_jk) in l[i].iter().zip(l[j].iter()).take(j) {
                    sum_prod += &(l_ik * l_jk);
                }
                let num = self.get(i, j) - &sum_prod;
                l[i][j] = (&num / &l[j][j]).simplify();
            }
        }
        Matrix::new(l)
    }

    /// LDLᵀ decomposition `A = L·D·Lᵀ` for a symmetric matrix (`L` unit
    /// lower triangular, `D` diagonal).
    ///
    /// Unlike [`cholesky`](Self::cholesky) this needs no square roots and
    /// no sign information, so it works for symbolic symmetric matrices
    /// and for indefinite ones.
    ///
    /// # Errors
    ///
    /// - [`SymplexError::InvalidArgument`] if the matrix is not square or
    ///   provably not symmetric.
    /// - [`SymplexError::ComputationFailed`] if a zero pivot is
    ///   encountered (no pivoting is performed).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let a = matrix![ctx, [4, 2], [2, 3]];
    /// let (l, d) = a.ldl().unwrap();
    /// assert_eq!((&(&l * &d) * &l.transpose()).eval(), a);
    /// assert_eq!(d, matrix![ctx, [4, 0], [0, 2]]);
    /// ```
    pub fn ldl(&self) -> Result<(Matrix, Matrix), SymplexError> {
        if !self.is_square() {
            return Err(invalid(
                "ldl",
                format!(
                    "requires a square matrix, got {}×{}",
                    self.nrows(),
                    self.ncols()
                ),
            ));
        }
        if self.is_symmetric() == Some(false) {
            return Err(invalid("ldl", "matrix is not symmetric"));
        }
        let n = self.nrows();
        let ctx = self.context();
        let zero = ctx.zero();
        let one = ctx.one();
        let mut l: Vec<Vec<Ex>> = vec![vec![zero.clone(); n]; n];
        let mut d: Vec<Ex> = vec![zero.clone(); n];

        for j in 0..n {
            l[j][j] = one.clone();
            // d_j = a_jj − Σ_{k<j} l_jk² d_k
            let mut acc = zero.clone();
            for k in 0..j {
                acc += &l[j][k].powi(2) * &d[k];
            }
            let dj = (self.get(j, j) - &acc).simplify();
            if ex_is_zero(&dj) == Some(true) {
                return Err(failed(
                    "ldl",
                    format!("zero pivot at position {j}; matrix needs pivoting or is singular"),
                ));
            }
            d[j] = dj;
            // l_ij = (a_ij − Σ_{k<j} l_ik l_jk d_k) / d_j
            for i in (j + 1)..n {
                let mut acc = zero.clone();
                for k in 0..j {
                    acc += &(&l[i][k] * &l[j][k]) * &d[k];
                }
                l[i][j] = (&(self.get(i, j) - &acc) / &d[j]).simplify();
            }
        }
        Ok((Matrix::new(l)?, Matrix::diag(&d)))
    }

    // ── Structure tests ────────────────────────────────────────────────

    /// Is `A = Aᵀ`?  Three-valued; `Some(false)` for non-square.
    pub fn is_symmetric(&self) -> Option<bool> {
        if !self.is_square() {
            return Some(false);
        }
        let n = self.nrows();
        all3((0..n).flat_map(|i| {
            ((i + 1)..n).map(move |j| ex_is_zero(&(self.get(i, j) - self.get(j, i))))
        }))
    }

    /// Is `A = −Aᵀ`?  Three-valued; `Some(false)` for non-square.
    pub fn is_skew_symmetric(&self) -> Option<bool> {
        if !self.is_square() {
            return Some(false);
        }
        let n = self.nrows();
        all3(
            (0..n)
                .flat_map(|i| (i..n).map(move |j| ex_is_zero(&(self.get(i, j) + self.get(j, i))))),
        )
    }

    /// Is `A = Aᴴ` (conjugate transpose)?  Three-valued.
    pub fn is_hermitian(&self) -> Option<bool> {
        if !self.is_square() {
            return Some(false);
        }
        let adj = self.adjoint();
        self.equals(&adj)
    }

    /// Is `AᵀA = I`?  Three-valued; `Some(false)` for non-square.
    pub fn is_orthogonal(&self) -> Option<bool> {
        if !self.is_square() {
            return Some(false);
        }
        let prod = self.transpose().matmul(self).ok()?;
        prod.is_identity()
    }

    /// Is `AᴴA = I`?  Three-valued; `Some(false)` for non-square.
    pub fn is_unitary(&self) -> Option<bool> {
        if !self.is_square() {
            return Some(false);
        }
        let prod = self.adjoint().matmul(self).ok()?;
        prod.is_identity()
    }

    /// Are all entries below the main diagonal zero?  Three-valued.
    pub fn is_upper_triangular(&self) -> Option<bool> {
        all3(
            (0..self.nrows())
                .flat_map(|i| (0..i.min(self.ncols())).map(move |j| ex_is_zero(self.get(i, j)))),
        )
    }

    /// Are all entries above the main diagonal zero?  Three-valued.
    pub fn is_lower_triangular(&self) -> Option<bool> {
        all3(
            (0..self.nrows())
                .flat_map(|i| ((i + 1)..self.ncols()).map(move |j| ex_is_zero(self.get(i, j)))),
        )
    }

    /// Are all off-diagonal entries zero?  Three-valued (works for
    /// rectangular matrices too).
    pub fn is_diagonal(&self) -> Option<bool> {
        all3((0..self.nrows()).flat_map(|i| {
            (0..self.ncols())
                .filter(move |&j| j != i)
                .map(move |j| ex_is_zero(self.get(i, j)))
        }))
    }

    /// Is this the identity matrix?  Three-valued; `Some(false)` for
    /// non-square.
    pub fn is_identity(&self) -> Option<bool> {
        if !self.is_square() {
            return Some(false);
        }
        let one = self.context().one();
        let one = &one;
        all3((0..self.nrows()).flat_map(|i| {
            (0..self.ncols()).map(move |j| {
                if i == j {
                    ex_is_zero(&(self.get(i, j) - one))
                } else {
                    ex_is_zero(self.get(i, j))
                }
            })
        }))
    }

    /// Are all entries zero?  Three-valued.
    pub fn is_zero(&self) -> Option<bool> {
        all3(self.iter().map(ex_is_zero))
    }

    /// Is `Aⁿ = 0` (n = dimension)?  Three-valued; `Some(false)` for
    /// non-square.
    ///
    /// A square matrix is nilpotent iff `Aⁿ = 0`, so exactly one matrix
    /// power is examined.
    pub fn is_nilpotent(&self) -> Option<bool> {
        if !self.is_square() {
            return Some(false);
        }
        let p = self.powi(self.nrows() as u32).ok()?;
        p.expand().is_zero()
    }

    /// Positive-definiteness via Sylvester's criterion: symmetric and all
    /// leading principal minors strictly positive.  Three-valued.
    ///
    /// Returns `Some(false)` for non-square or provably non-symmetric
    /// matrices and `None` when a minor's sign cannot be decided
    /// (symbolic entries without assumptions).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// assert_eq!(matrix![ctx, [2, -1], [-1, 2]].is_positive_definite(), Some(true));
    /// assert_eq!(matrix![ctx, [1, 2], [2, 1]].is_positive_definite(), Some(false));
    /// let x = ctx.symbol("x");
    /// let m = Matrix::new(vec![vec![x.clone(), ctx.int(0)], vec![ctx.int(0), ctx.int(1)]]).unwrap();
    /// assert_eq!(m.is_positive_definite(), None);
    /// ```
    pub fn is_positive_definite(&self) -> Option<bool> {
        match self.is_symmetric() {
            Some(true) => {}
            Some(false) => return Some(false),
            None => return None,
        }
        let n = self.nrows();
        all3((1..=n).map(|k| {
            let minor = self.submatrix(0..k, 0..k).det().ok()?;
            ex_is_positive(&minor)
        }))
    }

    /// Positive-semidefiniteness: symmetric and **all** principal minors
    /// non-negative (leading minors alone are not sufficient).
    /// Three-valued.
    ///
    /// Examines `2ⁿ − 1` minors, so this is intended for small matrices.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// assert_eq!(matrix![ctx, [1, 1], [1, 1]].is_positive_semidefinite(), Some(true));
    /// assert_eq!(matrix![ctx, [0, 0], [0, -1]].is_positive_semidefinite(), Some(false));
    /// ```
    pub fn is_positive_semidefinite(&self) -> Option<bool> {
        match self.is_symmetric() {
            Some(true) => {}
            Some(false) => return Some(false),
            None => return None,
        }
        let n = self.nrows();
        all3((1u32..(1u32 << n)).map(|mask| {
            let idx: Vec<usize> = (0..n).filter(|&i| mask & (1 << i) != 0).collect();
            let rows: Vec<Vec<Ex>> = idx
                .iter()
                .map(|&i| idx.iter().map(|&j| self.get(i, j).clone()).collect())
                .collect();
            let minor = Matrix::new(rows).ok()?.det().ok()?;
            ex_is_nonnegative(&minor)
        }))
    }

    // ── Norms ──────────────────────────────────────────────────────────

    /// Induced 1-norm: maximum absolute column sum.
    ///
    /// For symbolic entries the result is a `max(…)` of `abs(…)` sums;
    /// call `.eval()` to fold constants.
    pub fn norm_1(&self) -> Ex {
        let ctx = self.context();
        let sums = (0..self.ncols()).map(|j| {
            let mut acc = self.get(0, j).abs();
            for i in 1..self.nrows() {
                acc += self.get(i, j).abs();
            }
            acc
        });
        Ex::max_of(&ctx, sums).eval()
    }

    /// Induced ∞-norm: maximum absolute row sum.
    pub fn norm_inf(&self) -> Ex {
        let ctx = self.context();
        let sums = (0..self.nrows()).map(|i| {
            let mut acc = self.get(i, 0).abs();
            for j in 1..self.ncols() {
                acc += self.get(i, j).abs();
            }
            acc
        });
        Ex::max_of(&ctx, sums).eval()
    }

    /// Vector p-norm `(Σ |xᵢ|ᵖ)^(1/p)` for a row or column vector.
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::InvalidArgument`] if the matrix is not a
    /// row or column vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let v = matrix![ctx, [3], [-4]];
    /// assert_eq!(v.norm_p(&ctx.int(2)).unwrap().eval(), ctx.int(5));
    /// assert_eq!(v.norm_p(&ctx.int(1)).unwrap().eval(), ctx.int(7));
    /// ```
    pub fn norm_p(&self, p: &Ex) -> Result<Ex, SymplexError> {
        if self.nrows() != 1 && self.ncols() != 1 {
            return Err(invalid(
                "norm_p",
                format!(
                    "requires a row or column vector, got {}×{}",
                    self.nrows(),
                    self.ncols()
                ),
            ));
        }
        let ctx = self.context();
        let mut acc = ctx.zero();
        for e in self.iter() {
            acc += e.abs().pow(p);
        }
        let inv_p = &ctx.one() / p;
        Ok(acc.pow(&inv_p))
    }

    // ── Functions of diagonalizable matrices ───────────────────────────

    /// Symbolic power `Aⁿ` for a diagonalizable matrix via `P·Dⁿ·P⁻¹`.
    ///
    /// `n` may be any expression (a symbol, a rational, …); each diagonal
    /// entry becomes `λᵢⁿ`.  Entries are simplified.
    ///
    /// # Errors
    ///
    /// Same conditions as [`diagonalize`](Matrix::diagonalize).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let n = ctx.symbol("n");
    /// let a = matrix![ctx, [2, 0], [0, 3]];
    /// let an = a.matrix_pow_symbolic(&n).unwrap();
    /// assert_eq!(an[(0, 0)], ctx.int(2).pow(&n));
    /// assert_eq!(an[(1, 1)], ctx.int(3).pow(&n));
    /// ```
    pub fn matrix_pow_symbolic(&self, n: &Ex) -> Result<Matrix, SymplexError> {
        let (p, d) = self.diagonalize().map_err(|e| {
            failed(
                "matrix_pow_symbolic",
                format!("requires a diagonalizable matrix: {e}"),
            )
        })?;
        let dn = d.map_indexed(|i, j, e| if i == j { e.pow(n) } else { e.clone() });
        let p_inv = p.inv()?;
        Ok(p.matmul(&dn)?.matmul(&p_inv)?.simplify())
    }

    /// Principal square root `√A` of a diagonalizable matrix via
    /// `P·√D·P⁻¹` (principal branch on each eigenvalue).
    ///
    /// # Errors
    ///
    /// Same conditions as [`diagonalize`](Matrix::diagonalize).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let a = matrix![ctx, [4, 0], [0, 9]];
    /// let s = a.matrix_sqrt().unwrap();
    /// assert_eq!((&s * &s).simplify(), a);
    /// ```
    pub fn matrix_sqrt(&self) -> Result<Matrix, SymplexError> {
        let (p, d) = self.diagonalize().map_err(|e| {
            failed(
                "matrix_sqrt",
                format!("requires a diagonalizable matrix: {e}"),
            )
        })?;
        let sd = d.map_indexed(|i, j, e| if i == j { e.sqrt() } else { e.clone() });
        let p_inv = p.inv()?;
        Ok(p.matmul(&sd)?.matmul(&p_inv)?.simplify())
    }

    /// Principal matrix logarithm `log A` via the Jordan decomposition
    /// `A = P·J·P⁻¹`: `log A = P·log(J)·P⁻¹`.  SymPy: `Matrix.log()`.
    ///
    /// For a Jordan block `J_k(λ) = λI + N`,
    /// `log J = ln(λ)·I + Σ_{d=1}^{k−1} (−1)^{d+1} Nᵈ / (d·λᵈ)`, i.e. the
    /// entry `d` places above the diagonal is `(−1)^{d+1} / (d·λᵈ)`.  For
    /// diagonalizable matrices this is `P·diag(ln λᵢ)·P⁻¹`.  Each
    /// eigenvalue uses the principal branch of `ln`, so `matrix_exp` of
    /// the result reproduces `A` but `matrix_log(matrix_exp(A))` need not
    /// reproduce `A` when eigenvalues of `A` have imaginary part outside
    /// `(−π, π]`.  Entries are simplified.
    ///
    /// # Errors
    ///
    /// - [`SymplexError::InvalidArgument`] if the matrix is not square.
    /// - [`SymplexError::ComputationFailed`] if the matrix is singular
    ///   (an eigenvalue is provably zero: the logarithm does not exist),
    ///   or the Jordan form cannot be computed (see
    ///   [`jordan_form`](Matrix::jordan_form)).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// // SymPy: Matrix([[2, 0], [0, 3]]).log() == [[log(2), 0], [0, log(3)]]
    /// let l = matrix![ctx, [2, 0], [0, 3]].matrix_log().unwrap();
    /// assert_eq!(l, Matrix::diag(&[ctx.int(2).ln(), ctx.int(3).ln()]));
    /// // Defective: log [[1, 1], [0, 1]] = [[0, 1], [0, 0]]
    /// let n = matrix![ctx, [1, 1], [0, 1]];
    /// assert_eq!(n.matrix_log().unwrap(), matrix![ctx, [0, 1], [0, 0]]);
    /// assert!(matrix![ctx, [1, 2], [2, 4]].matrix_log().is_err());   // singular
    /// ```
    pub fn matrix_log(&self) -> Result<Matrix, SymplexError> {
        if !self.is_square() {
            return Err(invalid(
                "matrix_log",
                format!(
                    "requires a square matrix, got {}×{}",
                    self.nrows(),
                    self.ncols()
                ),
            ));
        }
        let n = self.nrows();
        let ctx = self.context();
        let (p, j) = self.jordan_form().map_err(|e| match e {
            SymplexError::ComputationFailed { reason, .. }
                if reason.starts_with("expression swell") =>
            {
                failed("matrix_log", reason)
            }
            e => failed("matrix_log", format!("Jordan form unavailable ({e})")),
        })?;

        let zero = ctx.zero();
        let mut log_j: Vec<Vec<Ex>> = vec![vec![zero; n]; n];
        let mut col = 0;
        while col < n {
            let lambda = j[(col, col)].clone();
            let mut block_size = 1;
            while col + block_size < n
                && j[(col + block_size - 1, col + block_size)].is_one_structural()
                && j[(col + block_size, col + block_size)] == lambda
            {
                block_size += 1;
            }
            if ex_is_zero(&lambda) == Some(true) {
                return Err(failed(
                    "matrix_log",
                    "matrix is singular (eigenvalue 0); the logarithm does not exist",
                ));
            }
            let ln_lambda = lambda.ln();
            for i in 0..block_size {
                log_j[col + i][col + i] = ln_lambda.clone();
                for d in 1..(block_size - i) {
                    // (−1)^{d+1} / (d · λ^d)
                    let sign = if d % 2 == 1 { 1 } else { -1 };
                    let coef = ctx.rational(sign, d as i64);
                    log_j[col + i][col + i + d] = &coef / &lambda.powi(d as i64);
                }
            }
            col += block_size;
        }
        let log_j = Matrix::new(log_j)?;
        let p_inv = p.inv().map_err(|e| match e {
            SymplexError::ComputationFailed { reason, .. }
                if reason.starts_with("expression swell") =>
            {
                failed("matrix_log", reason)
            }
            _ => failed(
                "matrix_log",
                "eigenvector matrix is singular (internal inconsistency)",
            ),
        })?;
        let result = p.matmul(&log_j)?.matmul(&p_inv)?;
        budget_check(result.iter(), "matrix_log")?;
        Ok(result.simplify())
    }

    /// Casoratian (discrete Wronskian) of the sequences `f₁(n), …, fₖ(n)`:
    /// `det[ fⱼ(n + i) ]` for `i, j = 0..k`.  A non-zero Casoratian
    /// proves the sequences linearly independent.  SymPy:
    /// `casoratian(seqs, n, zero=False)`; SymPy's default `zero=True`
    /// evaluates at `n = 0`, i.e. `Matrix::casoratian(seqs, n)?.subs(n, 0)`.
    ///
    /// # Errors
    ///
    /// - [`SymplexError::InvalidArgument`] if `seqs` is empty.
    /// - [`SymplexError::ComputationFailed`] if the determinant exceeds
    ///   [`EXPRESSION_BUDGET`](crate::matrix::EXPRESSION_BUDGET).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let n = ctx.symbol("n");
    /// // SymPy: casoratian([2**n, 3**n], n, zero=False) == 6**n, and == 1 at n = 0
    /// let w = Matrix::casoratian(&[ctx.int(2).pow(&n), ctx.int(3).pow(&n)], &n).unwrap();
    /// assert_eq!(w.simplify(), ctx.int(6).pow(&n));
    /// assert_eq!(w.subs(&n, &ctx.int(0)).eval(), ctx.int(1));
    /// // SymPy: casoratian([1, n, n**2], n) == 2
    /// let w2 = Matrix::casoratian(&[ctx.one(), n.clone(), n.powi(2)], &n).unwrap();
    /// assert_eq!(w2.expand(), ctx.int(2));
    /// ```
    pub fn casoratian(seqs: &[Ex], n: &Ex) -> Result<Ex, SymplexError> {
        if seqs.is_empty() {
            return Err(invalid("casoratian", "seqs must be non-empty"));
        }
        let k = seqs.len();
        let ctx = n.context();
        let rows: Vec<Vec<Ex>> = (0..k)
            .map(|i| {
                let shifted = n + &ctx.int(i as i64);
                seqs.iter().map(|f| f.subs(n, &shifted)).collect()
            })
            .collect();
        // k rows of k entries, k ≥ 1.
        Matrix::from_rows_unchecked(rows)
            .det()
            .map_err(|e| reop(e, "casoratian"))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Calculus: Hessian & Wronskian
// ═══════════════════════════════════════════════════════════════════════════

/// Hessian matrix `H[i][j] = ∂²f / ∂vars[i] ∂vars[j]`.
///
/// # Panics
///
/// Panics if `vars` is empty.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::matrix_decomp::hessian;
///
/// let ctx = Context::new();
/// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
/// let f = &x.powi(2) * &y + &y.powi(3);
/// let h = hessian(&f, &[&x, &y]);
/// assert_eq!(h[(0, 0)], &y * 2);
/// assert_eq!(h[(0, 1)], &x * 2);
/// assert_eq!(h[(1, 1)], &y * 6);
/// assert_eq!(h.is_symmetric(), Some(true));
/// ```
pub fn hessian(f: &Ex, vars: &[&Ex]) -> Matrix {
    assert!(!vars.is_empty(), "hessian: vars must be non-empty");
    let firsts: Vec<Ex> = vars.iter().map(|v| f.diff(v)).collect();
    Matrix::from_fn(vars.len(), vars.len(), |i, j| firsts[i].diff(vars[j]))
}

/// Wronskian `W(f₁, …, fₙ)(x) = det[ fⱼ^(i) ]` of a list of functions.
///
/// Row `i` holds the `i`-th derivatives.  A non-zero Wronskian proves
/// linear independence of the functions.
///
/// Returns NaN when the determinant cannot be formed: `funcs` is empty, or
/// the derivatives exceed [`EXPRESSION_BUDGET`](crate::matrix::EXPRESSION_BUDGET).
/// Use [`try_wronskian`] to get the error instead.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::matrix_decomp::wronskian;
///
/// let ctx = Context::new();
/// let x = ctx.symbol("x");
/// // W(sin x, cos x) = −sin² − cos² = −1
/// let w = wronskian(&[&x.sin(), &x.cos()], &x);
/// assert_eq!(w.simplify(), ctx.int(-1));
/// // W(x, x²) = x²
/// let w2 = wronskian(&[&x, &x.powi(2)], &x);
/// assert_eq!(w2.simplify(), x.powi(2));
/// ```
pub fn wronskian(funcs: &[&Ex], var: &Ex) -> Ex {
    try_wronskian(funcs, var).unwrap_or_else(|_| var.context().nan())
}

/// Wronskian `W(f₁, …, fₙ)(x)` of a list of functions; see [`wronskian`].
///
/// # Errors
///
/// - [`SymplexError::InvalidArgument`] if `funcs` is empty.
/// - [`SymplexError::ComputationFailed`] if the derivatives exceed
///   [`EXPRESSION_BUDGET`](crate::matrix::EXPRESSION_BUDGET).
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::matrix_decomp::try_wronskian;
///
/// let ctx = Context::new();
/// let x = ctx.symbol("x");
/// let w = try_wronskian(&[&x.exp(), &(&x * 2).exp()], &x).unwrap();
/// assert_eq!(w.simplify(), (&x * 3).exp());
/// assert!(try_wronskian(&[], &x).is_err());
/// ```
pub fn try_wronskian(funcs: &[&Ex], var: &Ex) -> Result<Ex, SymplexError> {
    if funcs.is_empty() {
        return Err(invalid("wronskian", "funcs must be non-empty"));
    }
    let n = funcs.len();
    let mut rows: Vec<Vec<Ex>> = Vec::with_capacity(n);
    let mut current: Vec<Ex> = funcs.iter().map(|f| (*f).clone()).collect();
    for i in 0..n {
        if i > 0 {
            current = current.iter().map(|f| f.diff(var)).collect();
        }
        rows.push(current.clone());
    }
    // n rows of n entries, n ≥ 1.
    Matrix::from_rows_unchecked(rows).det()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::context::Context;
    use crate::base::assumptions::Assumption;

    fn ctxi(ctx: &Context, rows: &[&[i64]]) -> Matrix {
        Matrix::from_i64(ctx, rows).unwrap()
    }

    #[test]
    fn qr_reconstructs_and_is_orthonormal() {
        let ctx = Context::new();
        let a = ctxi(&ctx, &[&[1, 2], &[3, 4], &[5, 6]]);
        let (q, r) = a.qr().unwrap();
        assert_eq!(q.shape(), (3, 2));
        assert_eq!(r.shape(), (2, 2));
        assert_eq!((&q * &r).simplify(), a);
        assert_eq!((&q.transpose() * &q).simplify(), Matrix::identity(&ctx, 2));
        assert_eq!(r.is_upper_triangular(), Some(true));
        assert_eq!(ex_is_positive(r.get(0, 0)), Some(true));
    }

    #[test]
    fn qr_rejects_dependent_columns() {
        let ctx = Context::new();
        let a = ctxi(&ctx, &[&[1, 2], &[2, 4]]);
        assert!(a.qr().is_err());
    }

    #[test]
    fn gram_schmidt_unnormalized() {
        let ctx = Context::new();
        let v1 = ctxi(&ctx, &[&[1], &[1], &[0]]);
        let v2 = ctxi(&ctx, &[&[1], &[0], &[1]]);
        let g = gram_schmidt(&[v1.clone(), v2], false).unwrap();
        assert_eq!(g[0], v1);
        assert_eq!(crate::domains::matrix::dot(&g[0], &g[1]).eval(), ctx.int(0));
        assert!(gram_schmidt(&[v1.clone(), &v1 * 2], false).is_err());
        assert!(gram_schmidt(&[], false).is_err());
    }

    #[test]
    fn cholesky_and_ldl() {
        let ctx = Context::new();
        let a = ctxi(&ctx, &[&[4, 12, -16], &[12, 37, -43], &[-16, -43, 98]]);
        let l = a.cholesky().unwrap();
        assert_eq!(l, ctxi(&ctx, &[&[2, 0, 0], &[6, 1, 0], &[-8, 5, 3]]));
        let (l2, d) = a.ldl().unwrap();
        assert_eq!((&(&l2 * &d) * &l2.transpose()).eval(), a);
        assert_eq!(d.diagonal(), vec![ctx.int(4), ctx.int(1), ctx.int(9)]);
    }

    #[test]
    fn cholesky_errors() {
        let ctx = Context::new();
        assert!(ctxi(&ctx, &[&[-1, 0], &[0, 1]]).cholesky().is_err());
        assert!(ctxi(&ctx, &[&[1, 2], &[3, 4]]).cholesky().is_err()); // not symmetric
        assert!(ctxi(&ctx, &[&[1, 2, 3]]).cholesky().is_err());
        let x = ctx.symbol("x");
        let sym = Matrix::new(vec![
            vec![x.clone(), ctx.int(0)],
            vec![ctx.int(0), ctx.int(1)],
        ])
        .unwrap();
        assert!(sym.cholesky().is_err(), "undecidable pivot must be Err");
        // With a positivity assumption the symbolic factorization works.
        let p = ctx.symbol_with("p", &[Assumption::Positive]);
        let sym_p = Matrix::new(vec![
            vec![p.clone(), ctx.int(0)],
            vec![ctx.int(0), ctx.int(1)],
        ])
        .unwrap();
        let l = sym_p.cholesky().unwrap();
        assert_eq!(l.get(0, 0), &p.sqrt());
    }

    #[test]
    fn ldl_symbolic_indefinite() {
        let ctx = Context::new();
        let (a, b, c) = (ctx.symbol("a"), ctx.symbol("b"), ctx.symbol("c"));
        let m = Matrix::new(vec![vec![a.clone(), b.clone()], vec![b.clone(), c.clone()]]).unwrap();
        let (l, d) = m.ldl().unwrap();
        let back = (&(&l * &d) * &l.transpose()).simplify();
        assert_eq!(back.equals(&m), Some(true));
        assert!(ctxi(&ctx, &[&[0, 1], &[1, 0]]).ldl().is_err());
    }

    #[test]
    fn structure_predicates() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let sym = ctxi(&ctx, &[&[1, 2], &[2, 1]]);
        assert_eq!(sym.is_symmetric(), Some(true));
        assert_eq!(ctxi(&ctx, &[&[1, 2], &[3, 4]]).is_symmetric(), Some(false));
        assert_eq!(ctxi(&ctx, &[&[1, 2, 3]]).is_symmetric(), Some(false));
        let trig = Matrix::new(vec![
            vec![ctx.int(1), &x.sin().powi(2) + &x.cos().powi(2)],
            vec![ctx.int(1), ctx.int(0)],
        ])
        .unwrap();
        assert_eq!(trig.is_symmetric(), Some(true));
        let unknown = Matrix::new(vec![
            vec![ctx.int(1), x.clone()],
            vec![ctx.int(1), ctx.int(0)],
        ])
        .unwrap();
        assert_eq!(unknown.is_symmetric(), None);

        assert_eq!(
            ctxi(&ctx, &[&[0, 1], &[-1, 0]]).is_skew_symmetric(),
            Some(true)
        );
        assert_eq!(sym.is_skew_symmetric(), Some(false));
        assert_eq!(sym.is_hermitian(), Some(true));
        let i = ctx.i_unit();
        let herm = Matrix::new(vec![vec![ctx.int(1), i.clone()], vec![-&i, ctx.int(2)]]).unwrap();
        assert_eq!(herm.is_hermitian(), Some(true));
        assert_eq!(herm.is_symmetric(), Some(false));

        let rot = Matrix::new(vec![vec![x.cos(), -&x.sin()], vec![x.sin(), x.cos()]]).unwrap();
        assert_eq!(rot.is_orthogonal(), Some(true));
        assert_eq!(ctxi(&ctx, &[&[1, 1], &[0, 1]]).is_orthogonal(), Some(false));
        let u = Matrix::new(vec![
            vec![ctx.int(0), i.clone()],
            vec![i.clone(), ctx.int(0)],
        ])
        .unwrap();
        assert_eq!(u.is_unitary(), Some(true));

        let up = ctxi(&ctx, &[&[1, 2], &[0, 3]]);
        assert_eq!(up.is_upper_triangular(), Some(true));
        assert_eq!(up.is_lower_triangular(), Some(false));
        assert_eq!(up.transpose().is_lower_triangular(), Some(true));
        assert_eq!(Matrix::identity(&ctx, 3).is_diagonal(), Some(true));
        assert_eq!(Matrix::identity(&ctx, 3).is_identity(), Some(true));
        assert_eq!(up.is_identity(), Some(false));
        assert_eq!(Matrix::zeros(&ctx, 2, 3).is_zero(), Some(true));
        assert_eq!(up.is_zero(), Some(false));
        assert_eq!(ctxi(&ctx, &[&[0, 1], &[0, 0]]).is_nilpotent(), Some(true));
        assert_eq!(up.is_nilpotent(), Some(false));
        assert_eq!(ctxi(&ctx, &[&[1, 2, 3]]).is_nilpotent(), Some(false));
    }

    #[test]
    fn definiteness() {
        let ctx = Context::new();
        assert_eq!(
            ctxi(&ctx, &[&[2, -1, 0], &[-1, 2, -1], &[0, -1, 2]]).is_positive_definite(),
            Some(true)
        );
        assert_eq!(
            ctxi(&ctx, &[&[1, 2], &[2, 1]]).is_positive_definite(),
            Some(false)
        );
        assert_eq!(
            ctxi(&ctx, &[&[1, 2], &[3, 4]]).is_positive_definite(),
            Some(false)
        );
        assert_eq!(
            ctxi(&ctx, &[&[1, 1], &[1, 1]]).is_positive_definite(),
            Some(false)
        );
        assert_eq!(
            ctxi(&ctx, &[&[1, 1], &[1, 1]]).is_positive_semidefinite(),
            Some(true)
        );
        // Leading minors are all zero but the matrix is not PSD:
        assert_eq!(
            ctxi(&ctx, &[&[0, 0], &[0, -1]]).is_positive_semidefinite(),
            Some(false)
        );
        let p = ctx.symbol_with("p", &[Assumption::Positive]);
        let m = Matrix::new(vec![
            vec![p.clone(), ctx.int(0)],
            vec![ctx.int(0), p.clone()],
        ])
        .unwrap();
        assert_eq!(m.is_positive_definite(), Some(true));
    }

    #[test]
    fn norms() {
        let ctx = Context::new();
        let a = ctxi(&ctx, &[&[1, -2], &[3, 4]]);
        assert_eq!(a.norm_1(), ctx.int(6));
        assert_eq!(a.norm_inf(), ctx.int(7));
        assert_eq!(a.norm_frobenius().eval(), ctx.int(30).sqrt().eval());
        assert_eq!(a.norm(), a.norm_frobenius());
        let v = ctxi(&ctx, &[&[3], &[-4]]);
        assert_eq!(v.norm_p(&ctx.int(2)).unwrap().eval(), ctx.int(5));
        assert!(a.norm_p(&ctx.int(2)).is_err());
    }

    #[test]
    fn symbolic_power_and_sqrt() {
        let ctx = Context::new();
        let n = ctx.symbol("n");
        let a = ctxi(&ctx, &[&[2, 1], &[0, 3]]);
        let an = a.matrix_pow_symbolic(&n).unwrap();
        for k in 0..4u32 {
            let direct = a.powi(k).unwrap();
            let via = an.subs(&n, &ctx.int(k as i64)).eval().simplify();
            assert_eq!(via, direct, "A^{k}");
        }
        let s = ctxi(&ctx, &[&[4, 0], &[0, 9]]).matrix_sqrt().unwrap();
        assert_eq!(s, ctxi(&ctx, &[&[2, 0], &[0, 3]]));
        let spd = ctxi(&ctx, &[&[2, 1], &[1, 2]]);
        let r = spd.matrix_sqrt().unwrap();
        assert_eq!((&r * &r).simplify(), spd);
        assert!(ctxi(&ctx, &[&[1, 1], &[0, 1]]).matrix_sqrt().is_err());
    }

    #[test]
    fn hessian_and_wronskian() {
        let ctx = Context::new();
        let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
        let f = &x.powi(3) + &(&x * &y.powi(2));
        let h = hessian(&f, &[&x, &y]);
        assert_eq!(h.get(0, 0), &(&x * 6));
        assert_eq!(h.get(0, 1), &(&y * 2));
        assert_eq!(h.get(1, 0), &(&y * 2));
        assert_eq!(h.get(1, 1), &(&x * 2));
        let w = wronskian(&[&x.exp(), &(&x * 2).exp()], &x).simplify();
        // W(e^x, e^{2x}) = 2e^{3x} − e^{3x} = e^{3x}
        assert_eq!(w, (&x * 3).exp());
        // Dependent functions → zero Wronskian
        let w0 = wronskian(&[&x, &(&x * 2)], &x).simplify();
        assert!(w0.is_zero_structural());
    }
}
