//! Symbolic matrix type.
//!
//! Provides [`Matrix`], a dense matrix of symbolic expressions with
//! exact arithmetic: construction, arithmetic operators, determinant
//! (Bareiss / Berkowitz) and permanent, inverse and pseudo-inverse (any
//! rank), linear solving, characteristic polynomial, eigenvalues /
//! eigenvectors, singular values, diagonalization, Jordan and Hessenberg
//! forms, matrix exponential, LU/RREF, rank factorisation, subspaces and
//! code generation.
//!
//! Additional decompositions (QR, LDLᵀ, Gram–Schmidt), structure tests
//! (`is_symmetric`, `is_positive_definite`, …) and norms live in
//! [`matrix_decomp`](crate::domains::matrix_decomp) but are all methods
//! on `Matrix`.
//!
//! [`QMatrix`] and [`ZMatrix`] are plain exact matrices over ℚ and ℤ
//! (`Ratio<BigInt>` / `BigInt` entries, no expression arena).  The exact
//! linear algebra of `Matrix` — `rref`, `rank`, `nullspace`, `det`, `inv`,
//! `solve`, `linsolve_matrix`, the normal forms — routes through them
//! automatically whenever every entry is a rational literal, using
//! fraction-free (Bareiss) elimination.
//!
//! # API conventions
//!
//! * **Shape preconditions** (non-square input, mismatched dimensions,
//!   empty data) return [`SymplexError::InvalidArgument`].
//! * **Mathematical failure** (singular matrix, not diagonalizable,
//!   eigenvalue solver gave up, …) returns
//!   [`SymplexError::ComputationFailed`].
//! * **Indexing** (`get`, `row`, `col`, `m[(i, j)]`, `submatrix`) panics on
//!   out-of-bounds indices, exactly like slice indexing; use
//!   [`Matrix::try_get`] for a checked variant.
//! * **Sized constructors** (`zeros`, `identity`, `from_fn`) panic on a
//!   zero dimension — that is a programming error, not a data error.
//!   Data constructors (`new`, `TryFrom<Vec<Vec<Ex>>>`, `from_i64`) return
//!   `Result`.
//! * **Structural queries** (`is_symmetric`, `is_diagonalizable`, …) are
//!   three-valued `Option<bool>`: `None` means "cannot decide symbolically".
//! * The eigen-family (`eigenvals`, `eigenvects`, `diagonalize`,
//!   `jordan_form`, `matrix_exp`) creates its own internal bound variable;
//!   the reserved name never appears in results.  Only
//!   [`Matrix::char_poly`] takes a caller-supplied variable because the
//!   result is a polynomial in it.

use crate::api::context::Context;
use crate::api::expr::{Ex, ExprType};
use crate::base::combinatorics::factorial;
use crate::base::errors::SymplexError;
use crate::base::numeric::Q;
use crate::base::truth::Truth;
use crate::domains::decompositions::{
    Diagonalization, Hessenberg, JordanForm, Lu, RankDecomposition,
};
use num_bigint::BigInt;
use num_rational::Rational64;
use std::fmt;
use std::sync::{Arc, OnceLock};
use tracing::{debug, trace};

// Re-export codegen option types so users can access them from the public
// `symplex::matrix` module (the `codegen` module itself is pub(crate)).
pub use crate::domains::exact_matrix::{
    ExactMatrix, ExactScalar, LLL_DEFAULT_DELTA, QMatrix, ZMatrix,
};
pub use crate::output::codegen::{CodegenOptions, MathBackend, Precision};

use crate::domains::exact_matrix::PERMANENT_MAX_DIM;

mod eigen;
mod zero_test;
use eigen::*;
pub(crate) use eigen::{PivotWork, Spectrum, pivot_truth, undecided_pivot};
pub(crate) use zero_test::*;

/// A dense matrix of symbolic expressions.
///
/// Elements are stored in row-major order as `Vec<Vec<Ex>>`.
///
/// Equality (`==`) is *structural*: two matrices are equal when they have
/// the same shape and every pair of entries is the same canonical
/// expression in the same [`Context`].  Use
/// [`equals`](Self::equals) for a mathematical (simplifying) comparison.
///
/// Whether every entry is a rational literal — the condition for the exact
/// [`QMatrix`] fast paths of `det`, `inv`, `rref`, `char_poly_coeffs`,
/// `matmul`, `lu`, … — is decided once per matrix, on the first such call,
/// and cached together with the converted entries; the cache travels with
/// [`clone`](Clone::clone), is dropped by [`get_mut`](Self::get_mut) /
/// [`set`](Self::set), and is invisible to `==` and `Debug`.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
///
/// let ctx = Context::new();
/// let a = matrix![ctx, [1, 2], [3, 4]];
/// let b = matrix![ctx, [0, 1], [1, 0]];
/// let c = &a * &b;                      // matrix product
/// assert_eq!(c, matrix![ctx, [2, 1], [4, 3]]);
/// assert_eq!(format!("{}", a.det().unwrap()), "-2");
/// assert_eq!(a[(1, 0)], ctx.int(3));
/// ```
#[derive(Clone)]
pub struct Matrix {
    rows: Vec<Vec<Ex>>,
    nrows: usize,
    ncols: usize,
    /// `Some` when every entry is a rational literal: the entries as a
    /// [`QMatrix`], converted on the first request (see
    /// [`as_qmatrix`](Self::as_qmatrix)).  Shared, so a clone is a pointer
    /// copy.
    exact: OnceLock<Option<Arc<QMatrix>>>,
}

impl PartialEq for Matrix {
    fn eq(&self, other: &Self) -> bool {
        self.nrows == other.nrows && self.ncols == other.ncols && self.rows == other.rows
    }
}

impl Eq for Matrix {}

// ═══════════════════════════════════════════════════════════════════════════
// Error helpers
// ═══════════════════════════════════════════════════════════════════════════

fn invalid(operation: &'static str, reason: impl Into<String>) -> SymplexError {
    SymplexError::invalid_argument(operation, reason)
}

fn failed(operation: &'static str, reason: impl Into<String>) -> SymplexError {
    SymplexError::computation_failed(operation, reason)
}

/// Re-attribute an error raised by a helper (e.g. `det` inside `inv`) to
/// the public operation the caller invoked, keeping the reason.
pub(crate) fn reop(e: SymplexError, operation: &'static str) -> SymplexError {
    match e {
        SymplexError::ComputationFailed { reason, .. } => failed(operation, reason),
        SymplexError::InvalidArgument { reason, .. } => invalid(operation, reason),
        SymplexError::ExpressionSwell { nodes, .. } => {
            SymplexError::ExpressionSwell { operation, nodes }
        }
        SymplexError::NotDiagonalizable { reason, .. } => {
            SymplexError::NotDiagonalizable { operation, reason }
        }
        other => other,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Expression-swell budget
// ═══════════════════════════════════════════════════════════════════════════

// The swell budget is a crate-wide knob (`rsolve` uses a quarter of it);
// this is its historical public path.
pub use crate::base::config::EXPRESSION_BUDGET;

/// Tree size of `e` (nodes, without sharing), saturating at `cap + 1` so
/// that a caller comparing against `cap` sees the overflow.  O(1) from the
/// node's facts.
pub(crate) fn tree_size_capped(e: &Ex, cap: usize) -> usize {
    let inner = e.inner.read();
    inner
        .arena
        .tree_size_capped(e.raw_id(), cap.saturating_add(1))
}

/// Return `Err(ExpressionSwell)` if the combined tree size of `entries`
/// exceeds [`EXPRESSION_BUDGET`].
pub(crate) fn budget_check<'a>(
    entries: impl IntoIterator<Item = &'a Ex>,
    operation: &'static str,
) -> Result<(), SymplexError> {
    let mut total = 0usize;
    for e in entries {
        total += tree_size_capped(e, EXPRESSION_BUDGET - total);
        if total > EXPRESSION_BUDGET {
            return Err(SymplexError::ExpressionSwell {
                operation,
                nodes: EXPRESSION_BUDGET,
            });
        }
    }
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// Constructors
// ═══════════════════════════════════════════════════════════════════════════

impl Matrix {
    /// Create a matrix from nested `Vec`s (row-major).
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::InvalidArgument`] if `rows` is empty,
    /// any row is empty, or rows have inconsistent lengths.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let m = Matrix::new(vec![vec![ctx.int(1), ctx.int(2)], vec![ctx.int(3), ctx.int(4)]]).unwrap();
    /// assert_eq!(m.shape(), (2, 2));
    /// assert!(Matrix::new(vec![vec![ctx.int(1)], vec![]]).is_err());
    /// ```
    pub fn new(rows: Vec<Vec<Ex>>) -> Result<Self, SymplexError> {
        if rows.is_empty() {
            return Err(invalid("Matrix::new", "must have at least one row"));
        }
        let ncols = rows[0].len();
        if ncols == 0 {
            return Err(invalid("Matrix::new", "must have at least one column"));
        }
        for (i, row) in rows.iter().enumerate() {
            if row.len() != ncols {
                return Err(invalid(
                    "Matrix::new",
                    format!("row {i} has length {} but expected {ncols}", row.len()),
                ));
            }
        }
        let nrows = rows.len();
        Ok(Matrix::with_shape(rows, nrows, ncols))
    }

    /// The one place a `Matrix` is assembled from its parts (no
    /// validation; the exact cache starts empty).
    #[inline]
    fn with_shape(rows: Vec<Vec<Ex>>, nrows: usize, ncols: usize) -> Self {
        Matrix {
            rows,
            nrows,
            ncols,
            exact: OnceLock::new(),
        }
    }

    /// Build a matrix from validated parts without re-checking.
    #[inline]
    pub(crate) fn from_rows_unchecked(rows: Vec<Vec<Ex>>) -> Self {
        let nrows = rows.len();
        let ncols = rows.first().map_or(0, Vec::len);
        debug_assert!(nrows > 0 && ncols > 0);
        debug_assert!(rows.iter().all(|r| r.len() == ncols));
        Matrix::with_shape(rows, nrows, ncols)
    }

    /// Build the matrix written by the `matrix!` macro.  Not public API.
    ///
    /// The macro's parser rejects an empty literal or an empty row at
    /// compile time and the array type makes the literal rectangular, so
    /// this cannot fail; it exists so that the expansion needs no fallible
    /// call.
    #[doc(hidden)]
    pub fn __from_literal<const R: usize, const C: usize>(rows: [[Ex; C]; R]) -> Matrix {
        Matrix::from_rows_unchecked(rows.into_iter().map(Vec::from).collect())
    }

    /// Create an `n × m` matrix of zeros.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `n == 0` or `m == 0` (a matrix
    /// has at least one row and one column).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// assert_eq!(Matrix::zeros(&ctx, 2, 3).unwrap().shape(), (2, 3));
    /// assert!(Matrix::zeros(&ctx, 0, 3).is_err());
    /// ```
    pub fn zeros(ctx: &Context, n: usize, m: usize) -> Result<Matrix, SymplexError> {
        positive_shape("Matrix::zeros", n, m)?;
        Ok(Matrix::zeros_unchecked(ctx, n, m))
    }

    /// [`zeros`](Self::zeros) for a shape known to be positive.
    pub(crate) fn zeros_unchecked(ctx: &Context, n: usize, m: usize) -> Matrix {
        debug_assert!(n > 0 && m > 0);
        let zero = ctx.zero();
        let rows = (0..n)
            .map(|_| (0..m).map(|_| zero.clone()).collect())
            .collect();
        Matrix::with_shape(rows, n, m)
    }

    /// Create an `n × n` identity matrix.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `n == 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// assert_eq!(Matrix::identity(&ctx, 2).unwrap(), matrix![ctx, [1, 0], [0, 1]]);
    /// assert!(Matrix::identity(&ctx, 0).is_err());
    /// ```
    pub fn identity(ctx: &Context, n: usize) -> Result<Matrix, SymplexError> {
        positive_shape("Matrix::identity", n, n)?;
        Ok(Matrix::identity_unchecked(ctx, n))
    }

    /// [`identity`](Self::identity) for a size known to be positive.
    pub(crate) fn identity_unchecked(ctx: &Context, n: usize) -> Matrix {
        debug_assert!(n > 0);
        let one = ctx.one();
        let zero = ctx.zero();
        let rows = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| if i == j { one.clone() } else { zero.clone() })
                    .collect()
            })
            .collect();
        Matrix::with_shape(rows, n, n)
    }

    /// Create an `n × m` matrix from a closure `f(i, j)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `n == 0` or `m == 0` (`f` is
    /// then not called).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let m = Matrix::from_fn(2, 2, |i, j| ctx.int((i + j) as i64)).unwrap();
    /// assert_eq!(m, matrix![ctx, [0, 1], [1, 2]]);
    /// ```
    pub fn from_fn(
        n: usize,
        m: usize,
        f: impl FnMut(usize, usize) -> Ex,
    ) -> Result<Matrix, SymplexError> {
        positive_shape("Matrix::from_fn", n, m)?;
        Ok(Matrix::from_fn_unchecked(n, m, f))
    }

    /// [`from_fn`](Self::from_fn) for a shape known to be positive.
    pub(crate) fn from_fn_unchecked(
        n: usize,
        m: usize,
        mut f: impl FnMut(usize, usize) -> Ex,
    ) -> Matrix {
        debug_assert!(n > 0 && m > 0);
        let rows = (0..n).map(|i| (0..m).map(|j| f(i, j)).collect()).collect();
        Matrix::with_shape(rows, n, m)
    }

    /// Create a 1×n row vector from a list of elements.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `elems` is empty.
    pub fn row_vector(elems: Vec<Ex>) -> Result<Matrix, SymplexError> {
        if elems.is_empty() {
            return Err(invalid("Matrix::row_vector", "need at least one element"));
        }
        Ok(Matrix::row_vector_unchecked(elems))
    }

    /// [`row_vector`](Self::row_vector) for elements known to be non-empty.
    pub(crate) fn row_vector_unchecked(elems: Vec<Ex>) -> Matrix {
        debug_assert!(!elems.is_empty());
        let ncols = elems.len();
        Matrix::with_shape(vec![elems], 1, ncols)
    }

    /// Create an n×1 column vector from a list of elements.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `elems` is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let v = Matrix::col_vector(vec![ctx.int(1), ctx.int(2)]).unwrap();
    /// assert_eq!(v.shape(), (2, 1));
    /// assert!(Matrix::col_vector(vec![]).is_err());
    /// ```
    pub fn col_vector(elems: Vec<Ex>) -> Result<Matrix, SymplexError> {
        if elems.is_empty() {
            return Err(invalid("Matrix::col_vector", "need at least one element"));
        }
        Ok(Matrix::col_vector_unchecked(elems))
    }

    /// [`col_vector`](Self::col_vector) for elements known to be non-empty.
    pub(crate) fn col_vector_unchecked(elems: Vec<Ex>) -> Matrix {
        debug_assert!(!elems.is_empty());
        let nrows = elems.len();
        let rows = elems.into_iter().map(|e| vec![e]).collect();
        Matrix::with_shape(rows, nrows, 1)
    }

    /// Create a square diagonal matrix from a slice of diagonal entries.
    ///
    /// The accessor returning the diagonal of an existing matrix is
    /// [`diagonal`](Self::diagonal).
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `entries` is empty.
    pub fn diag(entries: &[Ex]) -> Result<Matrix, SymplexError> {
        let Some(first) = entries.first() else {
            return Err(invalid("Matrix::diag", "entries must be non-empty"));
        };
        let n = entries.len();
        let zero = first.context().zero();
        let rows: Vec<Vec<Ex>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        if i == j {
                            entries[i].clone()
                        } else {
                            zero.clone()
                        }
                    })
                    .collect()
            })
            .collect();
        Ok(Matrix::from_rows_unchecked(rows))
    }

    /// Create a matrix of integer literals in the given context.
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::InvalidArgument`] for empty or jagged input.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let m = Matrix::from_i64(&ctx, &[&[1, 2], &[3, 4]]).unwrap();
    /// assert_eq!(m, matrix![ctx, [1, 2], [3, 4]]);
    /// ```
    pub fn from_i64(ctx: &Context, rows: &[&[i64]]) -> Result<Matrix, SymplexError> {
        let data: Vec<Vec<Ex>> = rows
            .iter()
            .map(|row| row.iter().map(|&v| ctx.int(v)).collect())
            .collect();
        Matrix::new(data)
    }

    /// Block-diagonal matrix built from the given blocks.
    ///
    /// Blocks need not be square; the result is
    /// `(Σ rows) × (Σ cols)` with zeros off the block diagonal.
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::InvalidArgument`] if `blocks` is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let a = matrix![ctx, [1]];
    /// let b = matrix![ctx, [2, 3], [4, 5]];
    /// let bd = Matrix::block_diag(&[&a, &b]).unwrap();
    /// assert_eq!(bd, matrix![ctx, [1, 0, 0], [0, 2, 3], [0, 4, 5]]);
    /// ```
    pub fn block_diag(blocks: &[&Matrix]) -> Result<Matrix, SymplexError> {
        if blocks.is_empty() {
            return Err(invalid("block_diag", "need at least one block"));
        }
        let nrows: usize = blocks.iter().map(|b| b.nrows).sum();
        let ncols: usize = blocks.iter().map(|b| b.ncols).sum();
        let zero = blocks[0].ctx_zero();
        let mut rows: Vec<Vec<Ex>> = vec![vec![zero; ncols]; nrows];
        let (mut r0, mut c0) = (0, 0);
        for b in blocks {
            for i in 0..b.nrows {
                for j in 0..b.ncols {
                    rows[r0 + i][c0 + j] = b.rows[i][j].clone();
                }
            }
            r0 += b.nrows;
            c0 += b.ncols;
        }
        Ok(Matrix::with_shape(rows, nrows, ncols))
    }

    /// Companion matrix of the monic polynomial
    /// `xⁿ + c_{n−1}xⁿ⁻¹ + … + c₁x + c₀`, whose non-leading coefficients
    /// are given in **ascending** order `[c₀, c₁, …, c_{n−1}]` (the same
    /// order as [`char_poly_coeffs`](Self::char_poly_coeffs)).  SymPy:
    /// `Matrix.companion(Poly)`.
    ///
    /// The result is `n × n` with ones on the sub-diagonal and `−cᵢ` in
    /// row `i` of the last column, so `det(xI − C)` is the given
    /// polynomial (and [`char_poly`](Self::char_poly), which is
    /// `det(C − xI)`, is `(−1)ⁿ` times it).
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `coeffs` is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// // x³ + 2x² + 3x + 4
    /// let c = Matrix::companion(&[ctx.int(4), ctx.int(3), ctx.int(2)]).unwrap();
    /// assert_eq!(c, matrix![ctx, [0, 0, -4], [1, 0, -3], [0, 1, -2]]);
    /// let x = ctx.symbol("x");
    /// let p = &x.powi(3) + &x.powi(2) * 2 + &x * 3 + 4;
    /// assert_eq!(-c.char_poly(&x).unwrap(), p);   // det(C − xI) = −p for odd n
    /// ```
    pub fn companion(coeffs: &[Ex]) -> Result<Matrix, SymplexError> {
        let Some(first) = coeffs.first() else {
            return Err(invalid(
                "companion",
                "need at least one coefficient (a monic polynomial of degree ≥ 1)",
            ));
        };
        let n = coeffs.len();
        let ctx = first.context();
        let zero = ctx.zero();
        let one = ctx.one();
        let mut rows: Vec<Vec<Ex>> = vec![vec![zero; n]; n];
        for (i, c) in coeffs.iter().enumerate() {
            rows[i][n - 1] = -c;
            if i + 1 < n {
                rows[i + 1][i] = one.clone();
            }
        }
        Ok(Matrix::from_rows_unchecked(rows))
    }

    /// Jordan block `J_size(λ)`: `λ` on the diagonal, ones on the
    /// super-diagonal, zeros elsewhere.  SymPy:
    /// `Matrix.jordan_block(size, eigenvalue)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `size == 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let j = Matrix::jordan_block(&ctx.int(2), 3).unwrap();
    /// assert_eq!(j, matrix![ctx, [2, 1, 0], [0, 2, 1], [0, 0, 2]]);
    /// assert!(Matrix::jordan_block(&ctx.int(2), 0).is_err());
    /// ```
    pub fn jordan_block(eigenvalue: &Ex, size: usize) -> Result<Matrix, SymplexError> {
        if size == 0 {
            return Err(invalid("jordan_block", "size must be positive"));
        }
        let ctx = eigenvalue.context();
        let zero = ctx.zero();
        let one = ctx.one();
        Ok(Matrix::from_fn_unchecked(size, size, |i, j| {
            if i == j {
                eigenvalue.clone()
            } else if j == i + 1 {
                one.clone()
            } else {
                zero.clone()
            }
        }))
    }
}

/// `Err` unless `n × m` is a valid matrix shape (both positive).
fn positive_shape(operation: &'static str, n: usize, m: usize) -> Result<(), SymplexError> {
    if n == 0 || m == 0 {
        return Err(invalid(
            operation,
            format!("dimensions must be positive, got {n}×{m}"),
        ));
    }
    Ok(())
}

impl TryFrom<Vec<Vec<Ex>>> for Matrix {
    type Error = SymplexError;

    /// Same validation as [`Matrix::new`].
    fn try_from(rows: Vec<Vec<Ex>>) -> Result<Self, Self::Error> {
        Matrix::new(rows)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Accessors
// ═══════════════════════════════════════════════════════════════════════════

impl Matrix {
    /// Number of rows.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of columns.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Shape as `(nrows, ncols)`.
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    /// Is this matrix square?
    #[inline]
    pub fn is_square(&self) -> bool {
        self.nrows == self.ncols
    }

    /// Immutable reference to element `(i, j)`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= nrows` or `j >= ncols`.
    /// For a non-panicking alternative, see [`try_get`](Self::try_get).
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> &Ex {
        assert!(
            i < self.nrows && j < self.ncols,
            "Index ({i}, {j}) out of bounds for {}×{} matrix",
            self.nrows,
            self.ncols
        );
        &self.rows[i][j]
    }

    /// Checked immutable reference to element `(i, j)`.
    ///
    /// Returns `None` if `i >= nrows` or `j >= ncols`.
    #[inline]
    pub fn try_get(&self, i: usize, j: usize) -> Option<&Ex> {
        if i < self.nrows && j < self.ncols {
            Some(&self.rows[i][j])
        } else {
            None
        }
    }

    /// Mutable reference to element `(i, j)`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= nrows` or `j >= ncols`, like slice indexing (this is
    /// the body of `m[(i, j)] = …`).  For a non-panicking alternative, see
    /// [`try_get_mut`](Self::try_get_mut).
    #[inline]
    pub fn get_mut(&mut self, i: usize, j: usize) -> &mut Ex {
        assert!(
            i < self.nrows && j < self.ncols,
            "Index ({i}, {j}) out of bounds for {}×{} matrix",
            self.nrows,
            self.ncols
        );
        // The entry may stop (or start) being a rational literal.
        self.exact.take();
        &mut self.rows[i][j]
    }

    /// Checked mutable reference to element `(i, j)`.
    ///
    /// Returns `None` if `i >= nrows` or `j >= ncols`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let mut m = matrix![ctx, [1, 2]];
    /// *m.try_get_mut(0, 1).unwrap() = ctx.int(5);
    /// assert_eq!(m, matrix![ctx, [1, 5]]);
    /// assert!(m.try_get_mut(1, 0).is_none());
    /// ```
    #[inline]
    pub fn try_get_mut(&mut self, i: usize, j: usize) -> Option<&mut Ex> {
        if i < self.nrows && j < self.ncols {
            self.exact.take();
            Some(&mut self.rows[i][j])
        } else {
            None
        }
    }

    /// Overwrite element `(i, j)`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= nrows` or `j >= ncols`; see
    /// [`try_get_mut`](Self::try_get_mut).
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, value: Ex) {
        *self.get_mut(i, j) = value;
    }

    /// Immutable reference to row `i` as a slice.
    ///
    /// # Panics
    ///
    /// Panics if `i >= nrows`; see [`try_row`](Self::try_row).
    #[inline]
    pub fn row(&self, i: usize) -> &[Ex] {
        assert!(
            i < self.nrows,
            "Row index {i} out of bounds for {}×{} matrix",
            self.nrows,
            self.ncols
        );
        &self.rows[i]
    }

    /// Checked row `i`: `None` if `i >= nrows`.
    #[inline]
    pub fn try_row(&self, i: usize) -> Option<&[Ex]> {
        self.rows.get(i).map(Vec::as_slice)
    }

    /// Column `j` as an owned `Vec<Ex>`.
    ///
    /// # Panics
    ///
    /// Panics if `j >= ncols`; see [`try_col`](Self::try_col).
    pub fn col(&self, j: usize) -> Vec<Ex> {
        assert!(
            j < self.ncols,
            "Column index {j} out of bounds for {}×{} matrix",
            self.nrows,
            self.ncols
        );
        self.rows.iter().map(|r| r[j].clone()).collect()
    }

    /// Checked column `j`: `None` if `j >= ncols`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let m = matrix![ctx, [1, 2], [3, 4]];
    /// assert_eq!(m.try_col(1), Some(vec![ctx.int(2), ctx.int(4)]));
    /// assert_eq!(m.try_row(0), Some(&[ctx.int(1), ctx.int(2)][..]));
    /// assert!(m.try_col(2).is_none() && m.try_row(2).is_none());
    /// ```
    pub fn try_col(&self, j: usize) -> Option<Vec<Ex>> {
        (j < self.ncols).then(|| self.rows.iter().map(|r| r[j].clone()).collect())
    }

    /// The main diagonal `[a₀₀, a₁₁, …]` (length `min(nrows, ncols)`).
    ///
    /// The constructor building a diagonal matrix is [`diag`](Self::diag).
    pub fn diagonal(&self) -> Vec<Ex> {
        (0..self.nrows.min(self.ncols))
            .map(|i| self.rows[i][i].clone())
            .collect()
    }

    /// The sub-block with the given row and column ranges.
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::InvalidArgument`] if either range is empty
    /// (a matrix has at least one row and column) or extends past the
    /// matrix bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let m = matrix![ctx, [1, 2, 3], [4, 5, 6], [7, 8, 9]];
    /// assert_eq!(m.submatrix(1..3, 0..2).unwrap(), matrix![ctx, [4, 5], [7, 8]]);
    /// assert!(m.submatrix(1..1, 0..2).is_err());
    /// assert!(m.submatrix(0..4, 0..2).is_err());
    /// ```
    pub fn submatrix(
        &self,
        rows: std::ops::Range<usize>,
        cols: std::ops::Range<usize>,
    ) -> Result<Matrix, SymplexError> {
        if rows.start >= rows.end || rows.end > self.nrows {
            return Err(SymplexError::invalid_argument(
                "Matrix::submatrix",
                format!("row range {rows:?} invalid for {} rows", self.nrows),
            ));
        }
        if cols.start >= cols.end || cols.end > self.ncols {
            return Err(SymplexError::invalid_argument(
                "Matrix::submatrix",
                format!("column range {cols:?} invalid for {} columns", self.ncols),
            ));
        }
        let data: Vec<Vec<Ex>> = rows.map(|i| self.rows[i][cols.clone()].to_vec()).collect();
        Ok(Matrix::from_rows_unchecked(data))
    }

    /// Iterate over all entries in row-major order.
    pub fn iter(&self) -> impl Iterator<Item = &Ex> + '_ {
        self.rows.iter().flatten()
    }

    /// Clone the entries into nested `Vec`s (row-major).
    pub fn to_vec(&self) -> Vec<Vec<Ex>> {
        self.rows.clone()
    }

    /// Evaluate every entry to `f64`.
    ///
    /// # Errors
    ///
    /// Propagates the first entry that cannot be evaluated (free symbols,
    /// non-real values, …).
    pub fn eval_f64(&self) -> Result<Vec<Vec<f64>>, SymplexError> {
        self.rows
            .iter()
            .map(|r| r.iter().map(Ex::eval_f64).collect())
            .collect()
    }

    /// Mathematical equality of two matrices (three-valued).
    ///
    /// Returns `Some(false)` for different shapes, `Some(true)` when every
    /// entry difference simplifies to zero, and `None` if some entry cannot
    /// be decided.
    pub fn equals(&self, other: &Matrix) -> Option<bool> {
        if self.shape() != other.shape() {
            return Some(false);
        }
        Truth::all(
            self.iter()
                .zip(other.iter())
                .map(|(a, b)| ex_is_zero(&(a - b))),
        )
        .to_option()
    }

    // ── Context-aware helpers ──────────────────────────────────────────
    //
    // These produce 0 and 1 in the *same* context as the matrix's
    // elements, avoiding "cannot mix expressions from different contexts"
    // panics when the matrix was built from a non-default Context.

    /// Zero expression in the matrix's own context.
    pub(crate) fn ctx_zero(&self) -> Ex {
        let elem = &self.rows[0][0];
        let zero_id = elem.inner.read().arena.zero();
        elem.wrap(zero_id)
    }

    /// One expression in the matrix's own context.
    pub(crate) fn ctx_one(&self) -> Ex {
        let elem = &self.rows[0][0];
        let one_id = elem.inner.read().arena.one();
        elem.wrap(one_id)
    }

    /// Returns a [`Context`] handle for this matrix's elements.
    pub fn context(&self) -> Context {
        self.rows[0][0].context()
    }

    /// Alias kept for internal call sites.
    #[inline]
    pub(crate) fn ctx(&self) -> Context {
        self.context()
    }

    /// Return `Err` unless the matrix is square.
    fn require_square(&self, operation: &'static str) -> Result<(), SymplexError> {
        if self.nrows != self.ncols {
            return Err(invalid(
                operation,
                format!(
                    "requires a square matrix, got {}×{}",
                    self.nrows, self.ncols
                ),
            ));
        }
        Ok(())
    }

    /// Return `Err(ExpressionSwell)` if the entries together exceed
    /// [`EXPRESSION_BUDGET`] tree nodes.
    fn check_budget(&self, operation: &'static str) -> Result<(), SymplexError> {
        budget_check(self.iter(), operation)
    }

    /// The matrix as a [`QMatrix`] if every entry is a rational literal
    /// (the exact fast path of `rref`, `det`, `inv`, `solve`,
    /// `char_poly_coeffs`, `matmul`, `lu`, …).
    ///
    /// The entries are scanned and converted on the first call and the
    /// answer — either way — is cached, so the later exact operations on the
    /// same matrix pay nothing for the decision.
    pub(crate) fn as_qmatrix(&self) -> Option<&QMatrix> {
        self.exact
            .get_or_init(|| {
                let rows = self.to_rational_rows()?;
                QMatrix::new(rows).ok().map(Arc::new)
            })
            .as_deref()
    }

    /// Has the rationality decision of [`as_qmatrix`](Self::as_qmatrix)
    /// been made (and cached) for this matrix?  A probe for the tests of
    /// the cache's lifetime; not part of the supported API.
    #[doc(hidden)]
    pub fn is_exact_cached(&self) -> bool {
        self.exact.get().is_some()
    }

    /// Does any entry contain `sym` as a sub-expression?
    fn contains(&self, sym: &Ex) -> bool {
        self.iter().any(|e| e.contains(sym))
    }

    /// Create an internal bound variable that does not occur in any entry.
    ///
    /// The name is reserved-looking (`__lambda`, `__lambda_1`, …) and is
    /// never allowed to leak into user-visible results — see
    /// [`hide_dummy`](Self::hide_dummy).
    pub(crate) fn fresh_symbol(&self, base: &str) -> Ex {
        let ctx = self.ctx();
        let mut k = 0usize;
        loop {
            let name = if k == 0 {
                format!("__{base}")
            } else {
                format!("__{base}_{k}")
            };
            let sym = ctx.symbol(&name);
            if !self.contains(&sym) {
                return sym;
            }
            k += 1;
        }
    }

    /// Replace the internal dummy `dummy` (if it survived, e.g. inside a
    /// `RootOf` for an unsolvable characteristic polynomial) by the
    /// display-friendly bound variable `λ`.  If the matrix itself mentions
    /// `λ`, the expression is returned unchanged to avoid capture.
    ///
    /// The dummy is bound (the variable of a `RootOf`), so this is a
    /// renaming of every occurrence — the structural
    /// [`replace`](Ex::replace) — not a substitution, which leaves bound
    /// occurrences alone.
    fn hide_dummy(&self, e: &Ex, dummy: &Ex) -> Ex {
        if !e.contains(dummy) {
            return e.clone();
        }
        let lam = self.ctx().symbol("λ");
        if self.contains(&lam) {
            return e.clone();
        }
        e.replace(|v| (v == dummy).then(|| lam.clone()))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Matrix arithmetic
// ═══════════════════════════════════════════════════════════════════════════

impl Matrix {
    /// Transpose: swap rows and columns.
    pub fn transpose(&self) -> Matrix {
        let rows: Vec<Vec<Ex>> = (0..self.ncols)
            .map(|j| (0..self.nrows).map(|i| self.rows[i][j].clone()).collect())
            .collect();
        Matrix::with_shape(rows, self.ncols, self.nrows)
    }

    /// Conjugate transpose `Aᴴ = conj(A)ᵀ`.
    ///
    /// Uses [`Ex::conjugate`] on every entry; for real-valued entries this
    /// coincides with [`transpose`](Self::transpose).
    pub fn adjoint(&self) -> Matrix {
        self.transpose().map(|e| e.conjugate())
    }

    /// Element-wise addition (method form of operator `+`).
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::InvalidArgument`] if shapes differ.
    pub fn add(&self, other: &Matrix) -> Result<Matrix, SymplexError> {
        if self.shape() != other.shape() {
            return Err(invalid(
                "add",
                format!(
                    "cannot add matrices with shapes {:?} and {:?}",
                    self.shape(),
                    other.shape()
                ),
            ));
        }
        Ok(self.zip_with(other, |a, b| a + b))
    }

    /// Element-wise subtraction (method form of operator `-`).
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::InvalidArgument`] if shapes differ.
    pub fn sub(&self, other: &Matrix) -> Result<Matrix, SymplexError> {
        if self.shape() != other.shape() {
            return Err(invalid(
                "sub",
                format!(
                    "cannot subtract matrices with shapes {:?} and {:?}",
                    self.shape(),
                    other.shape()
                ),
            ));
        }
        Ok(self.zip_with(other, |a, b| a - b))
    }

    /// Element-wise (Hadamard) product.
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::InvalidArgument`] if shapes differ.
    pub fn hadamard(&self, other: &Matrix) -> Result<Matrix, SymplexError> {
        if self.shape() != other.shape() {
            return Err(invalid(
                "hadamard",
                format!(
                    "cannot multiply element-wise matrices with shapes {:?} and {:?}",
                    self.shape(),
                    other.shape()
                ),
            ));
        }
        Ok(self.zip_with(other, |a, b| a * b))
    }

    /// Combine two same-shape matrices entry by entry (unchecked).
    fn zip_with(&self, other: &Matrix, f: impl Fn(&Ex, &Ex) -> Ex) -> Matrix {
        let rows = (0..self.nrows)
            .map(|i| {
                (0..self.ncols)
                    .map(|j| f(&self.rows[i][j], &other.rows[i][j]))
                    .collect()
            })
            .collect();
        Matrix::with_shape(rows, self.nrows, self.ncols)
    }

    /// Scalar multiplication: multiply every element by `scalar`.
    pub fn scale(&self, scalar: &Ex) -> Matrix {
        self.map(|elem| scalar * elem)
    }

    /// Matrix multiplication (`self * other`).
    ///
    /// `self` is `n × p` and `other` is `p × m`; the result is `n × m`.
    /// Each element is computed symbolically:
    /// `result[i][j] = Σ_k self[i][k] * other[k][j]`.  When both operands
    /// are rational the product is formed on [`QMatrix`] and converted
    /// back — the same entries, without an arena operation per term.
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::InvalidArgument`] if `self.ncols != other.nrows`.
    pub fn matmul(&self, other: &Matrix) -> Result<Matrix, SymplexError> {
        if self.ncols != other.nrows {
            return Err(invalid(
                "matmul",
                format!(
                    "cannot multiply {}×{} by {}×{} matrices",
                    self.nrows, self.ncols, other.nrows, other.ncols
                ),
            ));
        }
        // (Operands from different contexts fall through to the symbolic
        // loop, which raises the standard cross-context panic.)
        if self.rows[0][0].ctx_id() == other.rows[0][0].ctx_id()
            && let (Some(qa), Some(qb)) = (self.as_qmatrix(), other.as_qmatrix())
        {
            let product = qa.matmul(qb).map_err(|e| reop(e, "matmul"))?;
            return Ok(product.to_matrix(&self.ctx()));
        }
        let p = self.ncols;
        let rows: Vec<Vec<Ex>> = (0..self.nrows)
            .map(|i| {
                (0..other.ncols)
                    .map(|j| {
                        let mut acc: Ex = &self.rows[i][0] * &other.rows[0][j];
                        for k in 1..p {
                            let term = &self.rows[i][k] * &other.rows[k][j];
                            acc += term;
                        }
                        acc
                    })
                    .collect()
            })
            .collect();
        Ok(Matrix::with_shape(rows, self.nrows, other.ncols))
    }

    /// Trace: sum of the diagonal elements.
    ///
    /// A rational diagonal is summed exactly and interned once (one arena
    /// operation instead of `n − 1`); this needs only the diagonal to be
    /// rational, not the whole matrix.
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::InvalidArgument`] if the matrix is not square.
    pub fn trace(&self) -> Result<Ex, SymplexError> {
        self.require_square("trace")?;
        let rational: Option<Q> = (0..self.nrows)
            .try_fold(<Q as num_traits::Zero>::zero(), |acc, i| {
                self.rows[i][i].as_rational().map(|v| acc + v)
            });
        if let Some(sum) = rational {
            return Ok(self.ctx().from_ratio(sum));
        }
        let mut acc = self.rows[0][0].clone();
        for (i, row) in self.rows.iter().enumerate().take(self.nrows).skip(1) {
            acc += &row[i];
        }
        Ok(acc)
    }

    /// Determinant of a square matrix.
    ///
    /// Dispatch strategy:
    /// - 1×1 → element
    /// - 2×2 → `ad − bc`
    /// - 3×3 → cofactor expansion (hard-coded, fast)
    /// - n ≥ 4, all entries rational numbers → Bareiss fraction-free
    ///   elimination (O(n³), exact)
    /// - n ≥ 4, symbolic entries → Berkowitz's division-free algorithm
    ///   (O(n⁴)), which yields a fully expanded polynomial in the entries
    ///   instead of an unsimplified rational function.
    ///
    /// # Errors
    ///
    /// - [`SymplexError::InvalidArgument`] if the matrix is not square.
    /// - [`SymplexError::ComputationFailed`] if the entries or an
    ///   intermediate result exceed [`EXPRESSION_BUDGET`].
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (a, b, c, d) = (ctx.symbol("a"), ctx.symbol("b"), ctx.symbol("c"), ctx.symbol("d"));
    /// let m = Matrix::new(vec![vec![a.clone(), b.clone()], vec![c.clone(), d.clone()]]).unwrap();
    /// assert_eq!(m.det().unwrap(), &a * &d - &b * &c);
    /// ```
    pub fn det(&self) -> Result<Ex, SymplexError> {
        self.require_square("det")?;
        self.check_budget("det")?;
        let n = self.nrows;
        if n > 3
            && let Some(q) = self.as_qmatrix()
        {
            let d = q.det().map_err(|e| reop(e, "det"))?;
            return Ok(self.ctx().from_ratio(d));
        }
        Ok(match n {
            1 => self.rows[0][0].clone(),
            2 => {
                let a = &self.rows[0][0];
                let b = &self.rows[0][1];
                let c = &self.rows[1][0];
                let d = &self.rows[1][1];
                &(a * d) - &(b * c)
            }
            3 => det_cofactor_inner(&self.rows),
            _ => {
                // det(A) = (−1)ⁿ · [constant coefficient of det(λI − A)]
                let coeffs = self.berkowitz_monic("det")?;
                let c0 = coeffs[n].clone();
                if n % 2 == 1 { -c0 } else { c0 }
            }
        })
    }

    /// Permanent `per(A) = Σ_σ ∏ᵢ a_{i,σ(i)}` — the determinant without the
    /// signs.  SymPy: `Matrix.per()`.
    ///
    /// Rational matrices use Ryser's inclusion–exclusion formula on
    /// [`QMatrix`] (`O(2ⁿ·n)`, exact); symbolic matrices use a
    /// subset-dynamic-programming Laplace expansion (`O(2ⁿ·n)` expression
    /// operations) and return the expanded polynomial in the entries.
    /// Either way the cost is **exponential** in `n`: matrices larger than
    /// 20×20 are rejected, and symbolic intermediate results are subject to
    /// [`EXPRESSION_BUDGET`].
    ///
    /// # Errors
    ///
    /// - [`SymplexError::InvalidArgument`] if the matrix is not square.
    /// - [`SymplexError::ComputationFailed`] if `n > 20` or the symbolic
    ///   result exceeds [`EXPRESSION_BUDGET`].
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// assert_eq!(matrix![ctx, [1, 2], [3, 4]].permanent().unwrap(), ctx.int(10));
    /// assert_eq!(matrix![ctx, [1, 2, 3], [4, 5, 6], [7, 8, 9]].permanent().unwrap(), ctx.int(450));
    /// let (a, b, c, d) = (ctx.symbol("a"), ctx.symbol("b"), ctx.symbol("c"), ctx.symbol("d"));
    /// let m = Matrix::new(vec![vec![a.clone(), b.clone()], vec![c.clone(), d.clone()]]).unwrap();
    /// assert_eq!(m.permanent().unwrap(), &a * &d + &b * &c);
    /// assert!(matrix![ctx, [1, 2, 3]].permanent().is_err());
    /// ```
    pub fn permanent(&self) -> Result<Ex, SymplexError> {
        self.require_square("permanent")?;
        let n = self.nrows;
        if n > PERMANENT_MAX_DIM {
            return Err(failed(
                "permanent",
                format!(
                    "needs 2^{n} terms for a {n}×{n} matrix; the limit is \
                     {PERMANENT_MAX_DIM}×{PERMANENT_MAX_DIM}"
                ),
            ));
        }
        if let Some(q) = self.as_qmatrix() {
            let p = q.permanent().map_err(|e| reop(e, "permanent"))?;
            return Ok(self.ctx().from_ratio(p));
        }
        self.check_budget("permanent")?;
        // f(S) = permanent of rows 0..|S| restricted to the column set S:
        // f(S) = Σ_{j∈S} a_{|S|−1, j} · f(S ∖ {j}), f(∅) = 1.
        let zero = self.ctx_zero();
        let size = 1usize << n;
        let mut memo: Vec<Ex> = vec![zero.clone(); size];
        memo[0] = self.ctx_one();
        for s in 1..size {
            let row = s.count_ones() as usize - 1;
            let mut acc: Option<Ex> = None;
            for j in 0..n {
                if (s >> j) & 1 == 0 {
                    continue;
                }
                let a = &self.rows[row][j];
                let sub = &memo[s & !(1 << j)];
                if a.is_zero_structural() || sub.is_zero_structural() {
                    continue;
                }
                let term = a * sub;
                acc = Some(match acc {
                    None => term,
                    Some(x) => x + term,
                });
            }
            let val = acc.unwrap_or_else(|| zero.clone());
            budget_check(std::iter::once(&val), "permanent")?;
            memo[s] = val;
        }
        let result = memo[size - 1].expand();
        budget_check(std::iter::once(&result), "permanent")?;
        Ok(result)
    }

    /// Berkowitz's division-free characteristic polynomial.
    ///
    /// Returns the coefficients of `det(λI − A)`, **highest degree first**:
    /// `[1, c_{n−1}, …, c_0]` (length `n + 1`).  Every coefficient is a
    /// fully expanded polynomial in the matrix entries.  Requires a square
    /// matrix (checked by callers).
    ///
    /// Returns `Err(ComputationFailed)` (reported under `operation`) if the
    /// expanded coefficients exceed [`EXPRESSION_BUDGET`] at any stage.
    ///
    /// A rational matrix runs the same algorithm on [`QMatrix`] (integer
    /// cells, no `expand`) and interns the coefficients afterwards: every
    /// intermediate of the symbolic route would have been a rational
    /// literal, so the results are the same nodes.
    fn berkowitz_monic(&self, operation: &'static str) -> Result<Vec<Ex>, SymplexError> {
        if let Some(q) = self.as_qmatrix() {
            let ctx = self.ctx();
            let coeffs = q.berkowitz_monic().map_err(|e| reop(e, operation))?;
            return Ok(coeffs.into_iter().map(|c| ctx.from_ratio(c)).collect());
        }
        let n = self.nrows;
        let one = self.ctx_one();
        let zero = self.ctx_zero();

        // Char poly of the trailing 1×1 block.
        let mut vec: Vec<Ex> = vec![one.clone(), -&self.rows[n - 1][n - 1]];

        for k in 2..=n {
            let s = n - k; // top-left index of the trailing k×k block
            let a = &self.rows[s][s];
            // R = row s (cols s+1..n), C = column s (rows s+1..n), A' = trailing (k−1)×(k−1).
            let r: Vec<&Ex> = (s + 1..n).map(|j| &self.rows[s][j]).collect();
            let mut c: Vec<Ex> = (s + 1..n).map(|i| self.rows[i][s].clone()).collect();

            // diags = [1, −a, −R·C, −R·A'·C, …, −R·A'^{k−2}·C]
            let mut diags: Vec<Ex> = Vec::with_capacity(k + 1);
            diags.push(one.clone());
            diags.push(-a);
            for step in 0..(k - 1) {
                if step > 0 {
                    // c ← A'·c
                    let next: Vec<Ex> = (s + 1..n)
                        .map(|i| {
                            let mut acc = zero.clone();
                            for (idx, j) in (s + 1..n).enumerate() {
                                acc += &self.rows[i][j] * &c[idx];
                            }
                            acc.expand()
                        })
                        .collect();
                    c = next;
                }
                let mut rc = zero.clone();
                for (ri, ci) in r.iter().zip(c.iter()) {
                    rc += *ri * ci;
                }
                diags.push((-rc).expand());
            }

            // vec ← T · vec, where T is the (k+1)×k lower-triangular Toeplitz
            // matrix with T[i][j] = diags[i − j].
            let mut next_vec: Vec<Ex> = Vec::with_capacity(k + 1);
            for i in 0..=k {
                let mut acc = zero.clone();
                for (j, v) in vec.iter().enumerate().take(k) {
                    if j <= i {
                        acc += &diags[i - j] * v;
                    }
                }
                next_vec.push(acc.expand());
            }
            vec = next_vec;
            budget_check(vec.iter().chain(c.iter()), operation)?;
        }
        Ok(vec)
    }

    /// Apply a function to every element, producing a new matrix.
    pub fn map(&self, mut f: impl FnMut(&Ex) -> Ex) -> Matrix {
        let rows = self
            .rows
            .iter()
            .map(|row| row.iter().map(&mut f).collect())
            .collect();
        Matrix::with_shape(rows, self.nrows, self.ncols)
    }

    /// Apply `f(i, j, &a_ij)` to every element, producing a new matrix.
    pub fn map_indexed(&self, mut f: impl FnMut(usize, usize, &Ex) -> Ex) -> Matrix {
        let rows = self
            .rows
            .iter()
            .enumerate()
            .map(|(i, row)| row.iter().enumerate().map(|(j, e)| f(i, j, e)).collect())
            .collect();
        Matrix::with_shape(rows, self.nrows, self.ncols)
    }

    // ── Linear-algebra: minor, cofactor, adjugate, inverse ─────────────

    /// The `(n−1) × (n−1)` matrix obtained by deleting row `row` and
    /// column `col`.
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::InvalidArgument`] if the matrix is not
    /// square, has dimension ≤ 1, or the indices are out of range.
    pub fn minor_matrix(&self, row: usize, col: usize) -> Result<Matrix, SymplexError> {
        self.require_square("minor_matrix")?;
        if self.nrows <= 1 {
            return Err(invalid("minor_matrix", "requires matrix dimension > 1"));
        }
        if row >= self.nrows || col >= self.ncols {
            return Err(invalid(
                "minor_matrix",
                format!(
                    "index ({row}, {col}) out of range for {}×{} matrix",
                    self.nrows, self.ncols
                ),
            ));
        }
        let rows: Vec<Vec<Ex>> = self
            .rows
            .iter()
            .enumerate()
            .filter(|(r, _)| *r != row)
            .map(|(_, row_data)| {
                row_data
                    .iter()
                    .enumerate()
                    .filter(|(c, _)| *c != col)
                    .map(|(_, v)| v.clone())
                    .collect()
            })
            .collect();
        Ok(Matrix::from_rows_unchecked(rows))
    }

    /// The minor `M_ij = det(minor_matrix(i, j))`.
    ///
    /// # Errors
    ///
    /// Same conditions as [`minor_matrix`](Self::minor_matrix).
    pub fn minor(&self, row: usize, col: usize) -> Result<Ex, SymplexError> {
        self.minor_matrix(row, col)?.det()
    }

    /// Cofactor `C_ij = (−1)^(i+j) · M_ij`.
    ///
    /// # Errors
    ///
    /// Same conditions as [`minor_matrix`](Self::minor_matrix).
    pub fn cofactor(&self, row: usize, col: usize) -> Result<Ex, SymplexError> {
        let minor_det = self.minor(row, col)?;
        Ok(if (row + col).is_multiple_of(2) {
            minor_det
        } else {
            -minor_det
        })
    }

    /// Adjugate matrix (transpose of the cofactor matrix):
    /// `adj(A)[i][j] = cofactor(A, j, i)`.
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::InvalidArgument`] if the matrix is not square.
    pub fn adjugate(&self) -> Result<Matrix, SymplexError> {
        self.require_square("adjugate")?;
        let n = self.nrows;
        if n == 1 {
            return Ok(Matrix::from_rows_unchecked(vec![vec![self.ctx_one()]]));
        }
        let mut rows = Vec::with_capacity(n);
        for j in 0..n {
            let mut row = Vec::with_capacity(n);
            for i in 0..n {
                row.push(self.cofactor(i, j)?);
            }
            rows.push(row);
        }
        Ok(Matrix::from_rows_unchecked(rows))
    }

    /// Matrix inverse.
    ///
    /// Uses Gauss–Jordan elimination on `[A | I]` for matrices whose
    /// entries are all rational numbers (exact, O(n³)) and the adjugate
    /// formula `A⁻¹ = adj(A) / det(A)` for symbolic matrices (keeps
    /// entries as `polynomial / det`).
    ///
    /// # Errors
    ///
    /// - [`SymplexError::InvalidArgument`] if the matrix is not square.
    /// - [`SymplexError::ComputationFailed`] if the determinant is zero, or
    ///   if neither it nor the rank decides whether the matrix is singular
    ///   (an entry neither proved zero nor proved nonzero; see
    ///   [`rref`](Self::rref));
    ///   [`SymplexError::ExpressionSwell`] if the entries / intermediate
    ///   results exceed [`EXPRESSION_BUDGET`].
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let a = matrix![ctx, [2, 1], [1, 1]];
    /// let inv = a.inv().unwrap();
    /// assert_eq!(&a * &inv, Matrix::identity(&ctx, 2).unwrap());
    /// assert!(matrix![ctx, [1, 2], [2, 4]].inv().is_err());
    /// ```
    pub fn inv(&self) -> Result<Matrix, SymplexError> {
        self.require_square("inv")?;
        if let Some(q) = self.as_qmatrix() {
            return match q.inv() {
                Ok(inv) => Ok(inv.to_matrix(&self.ctx())),
                Err(SymplexError::ComputationFailed { .. }) => {
                    Err(failed("inv", "matrix is singular (determinant is zero)"))
                }
                Err(e) => Err(reop(e, "inv")),
            };
        }
        self.check_budget("inv")?;
        let _work = PivotWork::enter();
        let d = self.det().map_err(|e| reop(e, "inv"))?;
        // The determinant decides (three-valued, as a pivot): proved zero is
        // singular, proved not identically zero is invertible (for generic
        // values of the symbols).  An undecided one leaves it to the rank,
        // whose entries are smaller than their determinant; if that is not
        // decided either, the call is refused — up to 0.60.0 the matrix was
        // then inverted, with `1/d` for a `d` that may be 0 (`[[Γ(x + 1) −
        // x·Γ(x)]]`).
        let singular = match pivot_truth(&d) {
            Truth::True => true,
            Truth::False => false,
            Truth::Unknown => self.rank_decided("inv")? < self.nrows,
        };
        if singular {
            return Err(failed("inv", "matrix is singular (determinant is zero)"));
        }
        let n = self.nrows;
        if n == 1 {
            let one_over_det = &self.ctx_one() / &d;
            return Ok(Matrix::from_rows_unchecked(vec![vec![one_over_det]]));
        }
        let adj = self.adjugate().map_err(|e| reop(e, "inv"))?;
        budget_check(adj.iter().chain(std::iter::once(&d)), "inv")?;
        let one_over_det = &self.ctx_one() / &d;
        Ok(adj.scale(&one_over_det))
    }

    // ── Linear system solving ──────────────────────────────────────────

    /// Solve the linear system `Ax = b` where `self` is `A`.
    ///
    /// Uses Gauss–Jordan elimination on the augmented matrix `[A | b]`.
    /// Returns the solution as a `Matrix` (column vector or multi-column
    /// for multiple right-hand sides).  Each entry is passed through
    /// `eval()` to fold constants.
    ///
    /// # Errors
    ///
    /// - [`SymplexError::InvalidArgument`] if `A` is not square or `A` and
    ///   `b` have different row counts.
    /// - [`SymplexError::ComputationFailed`] if the system is singular
    ///   (no unique solution), if a pivot of the elimination is neither
    ///   proved zero nor proved nonzero (see [`rref`](Self::rref)), or the
    ///   entries / solution exceed [`EXPRESSION_BUDGET`].
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let a = matrix![ctx, [2, 1], [1, 3]];
    /// let b = matrix![ctx, [5], [10]];
    /// let x = a.solve(&b).unwrap();
    /// assert_eq!(x, matrix![ctx, [1], [3]]);
    /// ```
    pub fn solve(&self, b: &Matrix) -> Result<Matrix, SymplexError> {
        let n = self.nrows;
        self.require_square("solve")?;
        if b.nrows != n {
            return Err(invalid(
                "solve",
                format!(
                    "row count mismatch: A is {}×{}, b has {} rows",
                    n, self.ncols, b.nrows
                ),
            ));
        }

        if let (Some(qa), Some(qb)) = (self.as_qmatrix(), b.as_qmatrix()) {
            return qa.solve(qb).map(|x| x.to_matrix(&self.ctx()));
        }

        let augmented = Matrix::hstack(&[self, b])?;
        augmented.check_budget("solve")?;
        let (rref_mat, pivots) = augmented.rref_decided("solve")?;
        rref_mat.check_budget("solve")?;

        if pivots.len() != n || pivots.iter().any(|&p| p >= n) {
            return Err(failed(
                "solve",
                "matrix is singular; no unique solution exists",
            ));
        }

        let b_cols = b.ncols;
        let sol_rows: Vec<Vec<Ex>> = (0..n)
            .map(|i| {
                (n..(n + b_cols))
                    .map(|j| rref_mat.rows[i][j].eval())
                    .collect()
            })
            .collect();
        Ok(Matrix::from_rows_unchecked(sol_rows))
    }

    /// Least-squares solution of `Ax ≈ b` via the normal equations
    /// `AᴴA x = Aᴴb` (`Aᴴ` the conjugate transpose, as SymPy's
    /// `solve_least_squares`; `Aᵀ` when every entry of `A` is provably real).
    /// Before 0.32 `Aᵀ` was used for complex `A` too, which does not
    /// minimise `‖Ax − b‖` (the same inner product as [`qr`](Self::qr)).
    ///
    /// Requires `A` to have full column rank (so that `AᴴA` is invertible).
    ///
    /// # Errors
    ///
    /// - [`SymplexError::InvalidArgument`] if `b` has a different number of
    ///   rows than `A`.
    /// - [`SymplexError::ComputationFailed`] if `AᵀA` is singular.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// // Fit y = c0 + c1·x through (0,1), (1,2), (2,4): least squares gives c1 = 3/2, c0 = 5/6
    /// let a = matrix![ctx, [1, 0], [1, 1], [1, 2]];
    /// let b = matrix![ctx, [1], [2], [4]];
    /// let x = a.solve_least_squares(&b).unwrap();
    /// assert_eq!(x[(0, 0)], ctx.rational(5, 6));
    /// assert_eq!(x[(1, 0)], ctx.rational(3, 2));
    /// ```
    pub fn solve_least_squares(&self, b: &Matrix) -> Result<Matrix, SymplexError> {
        if b.nrows != self.nrows {
            return Err(invalid(
                "solve_least_squares",
                format!(
                    "row count mismatch: A is {}×{}, b has {} rows",
                    self.nrows, self.ncols, b.nrows
                ),
            ));
        }
        // The rank of `A` is decided exactly for rational functions over
        // `ℚ(radicals, i)`; that of `AᴴA` is not (`conjugate(x)` entries),
        // and before 0.38 a symbolic rank-deficient `A` (`[[x − 2, 2], [0, 0]]`)
        // had `AᴴA` "solved" by dividing by an unrecognised zero.
        if self.as_qmatrix().is_none() && self.rank_decided("solve_least_squares")? < self.ncols {
            return Err(failed(
                "solve_least_squares",
                "AᵀA is singular (A does not have full column rank)",
            ));
        }
        let real = self.iter().all(|e| Truth::from(e.is_real()).is_true());
        let at = if real {
            self.transpose()
        } else {
            self.adjoint()
        };
        let ata = at.matmul(self)?;
        let atb = at.matmul(b)?;
        let x = ata.solve(&atb).map_err(|_| {
            failed(
                "solve_least_squares",
                "AᵀA is singular (A does not have full column rank)",
            )
        })?;
        if real {
            return Ok(x);
        }
        // Constants of ℚ(√p…, i) in normal form (`3/2 − i/8`, not the
        // quotients of the elimination).
        Ok(x.map(|e| {
            crate::domains::linalg::algebraic_constant_normal_form(e).unwrap_or_else(|| e.clone())
        }))
    }

    // ── Characteristic polynomial & eigenvalues ────────────────────────

    /// Coefficients of the characteristic polynomial `det(A − λI)` in
    /// **ascending** degree order: `[c_0, c_1, …, c_n]` with `c_0 = det(A)`
    /// and `c_n = (−1)ⁿ`.
    ///
    /// Computed with Berkowitz's division-free algorithm, so the
    /// coefficients are expanded polynomials in the entries even for fully
    /// symbolic matrices.
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::InvalidArgument`] if the matrix is not square.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let a = matrix![ctx, [1, 2], [3, 4]];
    /// // det(A − λI) = λ² − 5λ − 2
    /// let c = a.char_poly_coeffs().unwrap();
    /// assert_eq!(c, vec![ctx.int(-2), ctx.int(-5), ctx.int(1)]);
    /// ```
    pub fn char_poly_coeffs(&self) -> Result<Vec<Ex>, SymplexError> {
        self.require_square("char_poly_coeffs")?;
        self.check_budget("char_poly_coeffs")?;
        let n = self.nrows;
        let monic = self.berkowitz_monic("char_poly_coeffs")?; // det(λI − A), highest first
        let sign_flip = n % 2 == 1;
        Ok((0..=n)
            .map(|k| {
                let c = monic[n - k].clone();
                if sign_flip { -c } else { c }
            })
            .collect())
    }

    /// Characteristic polynomial `det(A − λI)` as a polynomial in `var`.
    ///
    /// This is the one eigen-related method that takes a caller-supplied
    /// variable, because the result is a polynomial *in* that variable.
    /// `char_poly(λ)` evaluated at `λ = 0` is `det(A)`.
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::InvalidArgument`] if the matrix is not square.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let lam = ctx.symbol("lambda");
    /// let a = matrix![ctx, [2, 1], [1, 2]];
    /// let p = a.char_poly(&lam).unwrap();
    /// assert_eq!(p, &lam.powi(2) - &lam * 4 + 3);
    /// ```
    pub fn char_poly(&self, var: &Ex) -> Result<Ex, SymplexError> {
        let coeffs = self.char_poly_coeffs()?;
        let mut acc = coeffs[0].clone();
        for (k, c) in coeffs.iter().enumerate().skip(1) {
            if c.is_zero_structural() {
                continue;
            }
            acc += c * &var.powi(k as i64);
        }
        Ok(acc.expand())
    }

    /// Eigenvalues with algebraic multiplicities: `[(λ, multiplicity), …]`.
    ///
    /// The characteristic polynomial is factored over ℤ (exact
    /// multiplicities); each irreducible factor is then solved.  Rational
    /// and quadratic roots are returned in closed form.  Irreducible
    /// cubic/quartic factors are solved in radicals only when the result is
    /// compact (binomial-like after depressing, e.g. `λ³ − 2` or
    /// `λ⁴ − 10λ² + 1`); otherwise — and always for degree ≥ 5 — the roots
    /// are exact `RootOf` expressions whose bound variable displays as `λ`
    /// and which evaluate numerically via `eval_f64`/`eval_complex64`.
    /// (The general Cardano/Ferrari formulas produce nested complex cube
    /// roots that make eigenvectors and `P⁻¹` swell exponentially.)  For
    /// 1×1 and 2×2 matrices with symbolic entries the closed-form
    /// (quadratic) formula is used.  A larger symbolic matrix has its
    /// characteristic polynomial factored over `ℚ[symbols]`; an irreducible
    /// cubic factor is solved by Cardano's formulas as SymPy's `roots_cubic`
    /// (a generic formula, as SymPy's: `0/0` at values of the symbols where
    /// the depressed cubic's `p` vanishes and `q` has one sign), a quartic
    /// one as SymPy's `roots_quartic` (Ferrari's formulas in general, also
    /// generic: `0/0` where the depressed quartic's linear coefficient
    /// vanishes; refused up to 0.44); higher degrees are refused.  A characteristic polynomial with
    /// coefficients in `ℚ(√p₁, …, i)` (algebraic entries) is split into
    /// square-free factors over that field and by the factors of their norms
    /// over `ℚ`; the pieces of degree ≤ 3 are solved by the quadratic
    /// formula and by Cardano's formulas as SymPy's `roots_cubic` (before
    /// 0.32 such 3×3 matrices were refused unless the polynomial was
    /// `λ³ − c`), and their eigenvectors are computed over `K[t]/(g)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// // Symmetric with an irreducible cubic characteristic polynomial.
    /// let m = matrix![ctx, [4, 1, 2], [1, 3, 1], [2, 1, 5]];
    /// let ev = m.eigenvals_with_multiplicity().unwrap();
    /// assert_eq!(ev.len(), 3);
    /// assert!(ev.iter().all(|(v, m)| *m == 1 && v.to_string().starts_with("RootOf")));
    /// let sum: f64 = ev.iter().map(|(v, _)| v.eval_f64().unwrap()).sum();
    /// assert!((sum - 12.0).abs() < 1e-9); // Σλ = tr(A)
    /// ```
    ///
    /// The multiplicities sum to `n`: the list is every eigenvalue or the
    /// call is an error.  (Up to 0.59 the roots of a factor that was not
    /// solved were left out and a warning was logged: a 6×6 matrix with the
    /// characteristic polynomial `(λ − 1)·(λ⁵ − a·λ − 1)` had the
    /// eigenvalues `[(1, 1)]`.)
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let a = ctx.symbol("a");
    /// let i = |k: i64| ctx.int(k);
    /// // One block with the characteristic polynomial (λ − 2)·(λ⁵ − a·λ − 1);
    /// // the quintic has no roots in closed form.
    /// let m = Matrix::new(vec![
    ///     vec![i(0), i(-1), i(0), i(0), i(0), i(2)],
    ///     vec![i(0), i(0), i(1), i(0), i(0), i(0)],
    ///     vec![i(0), i(0), i(0), i(1), i(0), i(0)],
    ///     vec![i(0), i(0), i(0), i(0), i(1), i(0)],
    ///     vec![i(1), &a + 1, i(0), i(0), i(0), i(-1)],
    ///     vec![i(0), i(-2), i(1), i(0), i(0), i(2)],
    /// ])
    /// .unwrap();
    /// assert!(matches!(
    ///     m.eigenvals_with_multiplicity(),
    ///     Err(SymplexError::ComputationFailed { .. })
    /// ));
    /// // With a number for `a` every eigenvalue has an exact form.
    /// let ev = m.subs(&a, &ctx.int(3)).eigenvals_with_multiplicity().unwrap();
    /// assert_eq!(ev.iter().map(|(_, k)| *k).sum::<usize>(), 6);
    /// ```
    ///
    /// # Errors
    ///
    /// - [`SymplexError::InvalidArgument`] if the matrix is not square.
    /// - [`SymplexError::ComputationFailed`] if the eigenvalues are not all
    ///   found: the characteristic polynomial has a factor without roots in
    ///   closed form (an irreducible one of degree above 4 whose
    ///   coefficients are not rational numbers), or a multiplicity was not
    ///   established.  The message names the eigenvalues that were found.
    pub fn eigenvals_with_multiplicity(&self) -> Result<Vec<(Ex, usize)>, SymplexError> {
        self.spectrum()?.all("eigenvals")
    }

    /// The eigenvalues found, with their algebraic multiplicities, and
    /// whether they are all of them ([`Spectrum`]): for a caller inside the
    /// crate that can use a part of the spectrum (one eigenvalue proved in
    /// the closed right half-plane decides that a system is not stable).
    /// Everything else takes [`eigenvals_with_multiplicity`](Self::eigenvals_with_multiplicity).
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if the matrix is not square, and
    /// whatever the characteristic polynomial reports (expression swell).
    pub(crate) fn spectrum(&self) -> Result<Spectrum, SymplexError> {
        self.require_square("eigenvals")?;
        let lam = self.fresh_symbol("lambda");
        Ok(self.eigen_pairs(&lam)?.map(|v| self.hide_dummy(v, &lam)))
    }

    /// Every eigenvalue with its multiplicity, written with the dummy
    /// `lam`, or the error of an incomplete spectrum (which names the
    /// eigenvalues found, the dummy hidden).
    fn complete_eigen_pairs(&self, lam: &Ex) -> Result<Vec<(Ex, usize)>, SymplexError> {
        let spectrum = self.eigen_pairs(lam)?;
        if spectrum.is_complete() {
            return spectrum.all("eigenvals");
        }
        spectrum.map(|v| self.hide_dummy(v, lam)).all("eigenvals")
    }

    /// Internal: eigenvalue/multiplicity pairs expressed with the dummy `lam`.
    ///
    /// A matrix with symbolic or algebraic entries is split first into the
    /// diagonal blocks of its block-triangular form (the strongly connected
    /// components of the graph of its nonzero off-diagonal entries, as
    /// SymPy's `eigenvals`): the characteristic polynomial is the product of
    /// theirs, and a 1×1 block is its own eigenvalue.  Before 0.31 the
    /// eigenvalues of `diag(1, 2i)` came from the quadratic formula as
    /// `√(−3 − 4i)/2 + 1/2 + i` and its conjugate, whose `A − λI` had no
    /// decidable zero pivot (no eigenvectors).
    ///
    /// The result is a [`Spectrum`]: complete when the multiplicities sum to
    /// `n` (checked here once more for the whole matrix), partial when a
    /// block's characteristic polynomial has a factor that was not solved.
    ///
    /// The split reads the entries with the cheap test of 0.60.0
    /// ([`pivot_is_zero`]): what it finds is right whatever an entry is.
    /// When that spectrum is complete and an off-diagonal entry is zero
    /// only by the certified test ([`pivot_truth`]: `sin²x + cos²x − 1`),
    /// the eigenvalues are taken from the matrix with those entries
    /// written `0` instead — its blocks are parts of the blocks solved
    /// before.  Up to 0.60.0 the hidden zero stayed in the characteristic
    /// polynomial: `[[2, 0, x], [0, x, 1], [z, 0, x]]` with `z = atan(1/2) +
    /// atan(1/3) − π/4 + cos 2x − 2cos²x + 1` had the three "eigenvalues"
    /// `x` and `(x + 2 ± √((x − 2)² + 4xz))/2` with one eigenvector each (the
    /// matrix is triangular, `x` is a double eigenvalue with one
    /// eigenvector).  A spectrum that the cheap split does not complete is
    /// returned as it is, so that a matrix that was refused at once still is:
    /// the finer split can leave a cubic block with unsimplified
    /// coefficients, seconds of Cardano's formula (measured: `eigenvals` of a
    /// 4×4 matrix of the hunt, 0.14 s → 38 s).
    fn eigen_pairs(&self, lam: &Ex) -> Result<Spectrum, SymplexError> {
        let n = self.nrows;
        let found = self.eigen_pairs_by_blocks(lam)?;
        if n < 2 || self.as_qmatrix().is_some() || !found.is_complete() {
            return Ok(found);
        }
        // Eigenvalues that fill pages (Cardano's formula over unsimplified
        // coefficients) took seconds to find: the second pass is not
        // made for them (it cost as much again: 8 s → 14 s for a matrix of
        // the hunt), and `eigenvects` refuses what it cannot tell apart.
        const FINER_SPLIT_NODES: usize = 2_000;
        let Spectrum::Complete(pairs) = &found else {
            return Ok(found);
        };
        let size: usize = pairs
            .iter()
            .map(|(v, _)| tree_size_capped(v, FINER_SPLIT_NODES))
            .sum();
        if size > FINER_SPLIT_NODES {
            return Ok(found);
        }
        // One work budget for the certified tests of the entries.
        let _work = PivotWork::enter();
        let mut hidden = false;
        let cleaned = self.map_indexed(|i, j, e| {
            if i != j && !pivot_is_zero(e) && proved_zero(e) {
                hidden = true;
                e.context().zero()
            } else {
                e.clone()
            }
        });
        if !hidden {
            return Ok(found);
        }
        let finer = cleaned.eigen_pairs_by_blocks(lam)?;
        Ok(if finer.is_complete() { finer } else { found })
    }

    /// [`eigen_pairs`](Self::eigen_pairs) of the matrix as it is written:
    /// the split by the cheap zero test.
    fn eigen_pairs_by_blocks(&self, lam: &Ex) -> Result<Spectrum, SymplexError> {
        let n = self.nrows;
        if n > 1 && self.as_qmatrix().is_none() {
            let adj: Vec<Vec<usize>> = (0..n)
                .map(|i| {
                    (0..n)
                        .filter(|&j| j != i && !pivot_is_zero(&self.rows[i][j]))
                        .collect()
                })
                .collect();
            let comps = crate::base::graph::strongly_connected_components(&adj);
            if comps.len() > 1 {
                let mut out = Spectrum::of(Vec::new(), 0);
                for c in &comps {
                    let block = if let [k] = c.as_slice() {
                        Spectrum::of(vec![(self.rows[*k][*k].eval(), 1)], 1)
                    } else {
                        self.extract(c, c)?.eigen_pairs_irreducible(lam)?
                    };
                    out = out.joined(block, |w, v| {
                        w == v || pivot_is_zero(&(w.clone() - v.clone()))
                    });
                }
                return Ok(out.checked_for(n));
            }
        }
        self.eigen_pairs_irreducible(lam)
    }

    /// [`eigen_pairs`](Self::eigen_pairs) from the whole characteristic
    /// polynomial.
    fn eigen_pairs_irreducible(&self, lam: &Ex) -> Result<Spectrum, SymplexError> {
        let n = self.nrows;
        let coeffs = self.char_poly_coeffs()?;
        let cp = {
            let mut acc = coeffs[0].clone();
            for (k, c) in coeffs.iter().enumerate().skip(1) {
                if !c.is_zero_structural() {
                    acc += c * &lam.powi(k as i64);
                }
            }
            acc.expand()
        };

        // 1×1 / 2×2 with symbolic coefficients: the explicit linear /
        // quadratic formula (with a friendly square root of the
        // discriminant) beats the generic solver, whose `√(−ω²)` forms
        // stop `A − λI` pivots from cancelling.
        let symbolic = coeffs.iter().any(|c| c.expr_type() != ExprType::Number);
        let mut spectrum = if n <= 2 && symbolic {
            Spectrum::of(low_degree_roots(&coeffs), n)
        } else if let Some(split) = symbolic.then(|| symbolic_factor_roots(&cp, lam)).flatten() {
            split.checked_for(n)
        } else {
            eigvals_with_multiplicity(&cp, lam, n)
        };

        // Each fallback below is taken when the spectrum so far is not
        // complete, and only when its own is.

        // Closed-form fallback for low degree with symbolic coefficients.
        if !spectrum.is_complete() && n <= 2 {
            let closed = Spectrum::of(low_degree_roots(&coeffs), n);
            if closed.is_complete() {
                spectrum = closed;
            }
        }

        // Coefficients in ℚ(√p…, i): square-free factors and the factors
        // of their norms over ℚ, then the quadratic formula and Cardano as
        // SymPy's `roots` (`roots_cubic`).  Before 0.32 the solver took only
        // binomials here, so every 3×3 matrix with algebraic entries whose
        // characteristic polynomial was not `λ³ − c` was refused, even one
        // with `det A = 0` (`λ·(λ² + bλ + c)`).
        if !spectrum.is_complete()
            && let Some(alg) = crate::domains::linalg::algebraic_poly_roots(&coeffs)
        {
            let alg = Spectrum::of(alg, n);
            if alg.is_complete() {
                spectrum = alg;
            }
        }

        // An irreducible cubic or quartic with symbolic coefficients: in
        // radicals, as SymPy's `roots(…, cubics=True, quartics=True)`
        // (generic formulas, see `parametric_radical_roots`).  Before 0.44
        // every symbolic 3×3 or 4×4 matrix whose characteristic polynomial
        // did not factor was refused ("could not solve the characteristic
        // polynomial"), and so were its poles, eigenvectors and `exp`.
        if !spectrum.is_complete()
            && symbolic
            && (3..=4).contains(&n)
            && let Some(roots) = squarefree_parametric_roots(&cp, &coeffs, lam)
        {
            let radical = Spectrum::of(roots.into_iter().map(|r| (r, 1)).collect(), n);
            if radical.is_complete() {
                spectrum = radical;
            }
        }

        // Complete, or partial: the caller decides what a part is worth
        // (`Spectrum::all` is the error of the public functions).  Up to
        // 0.59 a short list was returned as it was, with a log line.
        Ok(spectrum)
    }

    /// Eigenvalues, repeated according to algebraic multiplicity.
    ///
    /// For an `n × n` matrix this list has exactly `n` entries: when the
    /// eigenvalues are not all found the call is an error (up to 0.59 a
    /// shorter list came back).  See
    /// [`eigenvals_with_multiplicity`](Self::eigenvals_with_multiplicity)
    /// for the grouped form and for the forms of the eigenvalues.
    ///
    /// # Errors
    ///
    /// - [`SymplexError::InvalidArgument`] if the matrix is not square.
    /// - [`SymplexError::ComputationFailed`] if the eigenvalues are not all
    ///   found (see
    ///   [`eigenvals_with_multiplicity`](Self::eigenvals_with_multiplicity)).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let a = matrix![ctx, [2, 1], [1, 2]];
    /// let mut ev: Vec<String> = a.eigenvals().unwrap().iter().map(|e| e.to_string()).collect();
    /// ev.sort();
    /// assert_eq!(ev, ["1", "3"]);
    ///
    /// // Symbolic 2×2: closed form via the quadratic formula
    /// let (a, b) = (ctx.symbol("a"), ctx.symbol("b"));
    /// let m = Matrix::new(vec![vec![a.clone(), b.clone()], vec![b.clone(), a.clone()]]).unwrap();
    /// let ev = m.eigenvals().unwrap();
    /// assert_eq!(ev.len(), 2);
    /// ```
    pub fn eigenvals(&self) -> Result<Vec<Ex>, SymplexError> {
        let pairs = self.eigenvals_with_multiplicity()?;
        let mut out = Vec::new();
        for (v, m) in pairs {
            for _ in 0..m {
                out.push(v.clone());
            }
        }
        Ok(out)
    }

    /// Eigenvectors: for each eigenvalue, a basis for its eigenspace.
    ///
    /// Returns a list of `(eigenvalue, algebraic_multiplicity, eigenvectors)`
    /// tuples.  Each eigenvector is a column-vector [`Matrix`]; the number
    /// of vectors is the geometric multiplicity.  The algebraic
    /// multiplicities sum to `n` (up to 0.59 the eigenvalues of a factor
    /// that was not solved were left out, with their vectors).
    ///
    /// The eigenvectors are those for generic values of the symbols (the
    /// reading of [`rref`](Self::rref)), and every zero decision behind
    /// them is a proof: where the solve of `A − λI` meets an entry that is
    /// neither proved zero nor proved nonzero, the call is an error.  (Up
    /// to 0.60.0 such an entry counted as nonzero: the identity matrix
    /// written `[[1, Γ(x + 1) − x·Γ(x)], [0, 1]]` had one eigenvector for
    /// its eigenvalue 1 of multiplicity 2.)  Two eigenvalues that are
    /// proved equal are one (`1` and `sin²x + cos²x`).
    ///
    /// # Errors
    ///
    /// - [`SymplexError::InvalidArgument`] if the matrix is not square.
    /// - [`SymplexError::ComputationFailed`] if the eigenvalues are not all
    ///   found (see
    ///   [`eigenvals_with_multiplicity`](Self::eigenvals_with_multiplicity)),
    ///   or if an entry of an eigenvector solve, or the equality of two
    ///   eigenvalues, is neither proved nor refuted (the message names
    ///   it).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let m = matrix![ctx, [2, 1], [0, 3]];
    /// for (val, mult, vecs) in m.eigenvects().unwrap() {
    ///     assert_eq!(mult, 1);
    ///     assert_eq!(vecs.len(), 1);
    ///     // A·v = λ·v
    ///     assert_eq!((&m * &vecs[0]).simplify(), vecs[0].scale(&val).simplify());
    /// }
    /// ```
    pub fn eigenvects(&self) -> Result<Vec<(Ex, usize, Vec<Matrix>)>, SymplexError> {
        self.require_square("eigenvects")?;
        let n = self.nrows;
        debug!(n, "eigenvects: computing for {}×{} matrix", n, n);
        let lam = self.fresh_symbol("lambda");
        let eigen_pairs = self.complete_eigen_pairs(&lam)?;
        // Entries that are not rational functions over `ℚ(radicals, i)` (a
        // function, a constant such as `sin 1`): two roots of the
        // characteristic polynomial can be one eigenvalue written twice,
        // which the factorization does not see.
        let eigen_pairs = if self.iter().all(|e| rational_function_is_zero(e).is_known()) {
            eigen_pairs
        } else {
            self.distinct_eigen_pairs(eigen_pairs, "eigenvects")?
        };
        let eye = Matrix::identity(&self.ctx(), n)?;

        let mut result = Vec::new();
        let mut charpoly: Option<Vec<Ex>> = None;
        let over_tower = self.as_qmatrix().is_none()
            && self.iter().all(Ex::is_constant)
            && crate::domains::linalg::in_square_root_tower(
                &self.iter().cloned().collect::<Vec<_>>(),
            );
        for (eigenval, alg_mult) in &eigen_pairs {
            let a_minus_lambda_i = self.sub(&eye.scale(eigenval))?;
            let field = self
                .as_qmatrix()
                .and_then(|q| Some((q, eigenvalue_minimal_polynomial(eigenval)?)));
            let vecs = match field {
                Some((q, g)) => a_minus_lambda_i.nullspace_in_field(q, &g, eigenval),
                None => {
                    // An eigenvalue outside the square-root tower (Cardano's
                    // cube roots) of a matrix over ℚ(√p…, i): exactly over
                    // K[t]/(g) (see `algebraic_eigenvectors`).
                    let exact = if !over_tower
                        || crate::domains::linalg::in_square_root_tower(std::slice::from_ref(
                            eigenval,
                        )) {
                        None
                    } else {
                        if charpoly.is_none() {
                            charpoly = self.char_poly_coeffs().ok();
                        }
                        charpoly.as_deref().and_then(|cp| {
                            crate::domains::linalg::algebraic_eigenvectors(&self.rows, cp, eigenval)
                        })
                    };
                    // A simple eigenvalue in radicals of a symbolic
                    // cubic/quartic: a column of adj(λI − A).
                    // (The elimination below cannot decide its pivots in
                    // such a `λ` and takes minutes trying: no answer then.)
                    let exact = match exact {
                        None if n >= 3 && *alg_mult == 1 && is_parametric_radical(eigenval) => {
                            if charpoly.is_none() {
                                charpoly = self.char_poly_coeffs().ok();
                            }
                            let v = charpoly
                                .as_deref()
                                .and_then(|cp| self.adjugate_eigenvector(cp, eigenval));
                            match v {
                                Some(v) => Some(vec![v]),
                                None => {
                                    return Err(failed(
                                        "eigenvects",
                                        "no column of adj(λI − A) was proven nonzero for an \
                                         eigenvalue in radicals of the symbols",
                                    ));
                                }
                            }
                        }
                        other => other,
                    };
                    match exact {
                        Some(vs) => vs
                            .into_iter()
                            .map(Matrix::col_vector)
                            .collect::<Result<Vec<_>, _>>()?,
                        None => {
                            // A simple eigenvalue has exactly one eigenvector:
                            // `n − 1` pivots proved nonzero are enough, when
                            // the elimination meets one it cannot decide (or
                            // finds none: an eigenvalue in radicals that is
                            // one only on a part of the symbols' domain).
                            match a_minus_lambda_i.nullspace_decided("eigenvects") {
                                Ok(vs) if *alg_mult != 1 || vs.len() == 1 => vs,
                                found if *alg_mult == 1 => {
                                    match (simple_eigenvector(&a_minus_lambda_i), found) {
                                        (Some(v), _) => vec![v],
                                        (None, Err(e)) => return Err(e),
                                        (None, Ok(vs)) => {
                                            return Err(failed(
                                                "eigenvects",
                                                format!(
                                                    "found {} eigenvectors for a simple \
                                                     eigenvalue",
                                                    vs.len()
                                                ),
                                            ));
                                        }
                                    }
                                }
                                // An eigenvalue of multiplicity above 1: the
                                // dimension of its eigenspace is not known
                                // when a pivot is undecided.  (Up to 0.60.0
                                // the undecided entry was a pivot: one
                                // eigenvector for the eigenvalue 1 of the
                                // identity matrix written `[[1, Γ(x + 1) −
                                // x·Γ(x)], [0, 1]]`.)
                                other => other?,
                            }
                        }
                    }
                }
            };
            trace!(
                alg_mult,
                geom_mult = vecs.len(),
                "eigenvects: eigenvalue has alg_mult={}, geom_mult={}",
                alg_mult,
                vecs.len()
            );
            let vecs = vecs
                .into_iter()
                .map(|v| v.map(|e| tidy_symbolic_entry(&self.hide_dummy(e, &lam))))
                .collect();
            result.push((self.hide_dummy(eigenval, &lam), *alg_mult, vecs));
        }
        Ok(result)
    }

    /// The eigenvector of a simple eigenvalue `lam` as a column of
    /// `adj(λI − A)`: `(λI − A)·adj(λI − A) = det(λI − A)·I = 0`, and at a
    /// simple eigenvalue the adjugate has rank 1, so every nonzero column is
    /// an eigenvector.  The adjugate is `Σⱼ Nⱼ·λ^(n−1−j)` with `N₀ = I`,
    /// `Nⱼ = A·Nⱼ₋₁ + cₙ₋ⱼ·I` (Faddeev–LeVerrier, as
    /// [`sylvester_function`](Self::sylvester_function)), so each entry is a
    /// polynomial in `λ` of degree below `n` with coefficients rational in
    /// the entries — no zero test on an expression in `λ` is needed, only
    /// one that some entry of the chosen column is not the zero function
    /// ([`radical_nonzero_by_value`]: certified values at rational points).
    ///
    /// This is what an eigenvalue in Cardano/Ferrari radicals of a symbolic
    /// matrix needs: the pivots of `A − λI` in such a `λ` are undecidable by
    /// the elimination's zero tests.  `charpoly` is
    /// [`char_poly_coeffs`](Self::char_poly_coeffs).  `None` when no column
    /// is proven nonzero.
    fn adjugate_eigenvector(&self, charpoly: &[Ex], lam: &Ex) -> Option<Vec<Ex>> {
        let n = self.nrows;
        if charpoly.len() != n + 1 {
            return None;
        }
        let ctx = self.ctx();
        // Monic det(tI − A) = (−1)ⁿ·det(A − tI), ascending.
        let sign = ctx.int(if n.is_multiple_of(2) { 1 } else { -1 });
        let c: Vec<Ex> = charpoly
            .iter()
            .map(|x| parametric_normal(&(x * &sign)))
            .collect();
        let eye = Matrix::identity_unchecked(&ctx, n);
        let mut ns: Vec<Matrix> = vec![eye.clone()];
        for j in 1..n {
            let next = self
                .matmul(&ns[j - 1])
                .ok()?
                .add(&eye.scale(&c[n - j]))
                .ok()?;
            budget_check(next.iter(), "eigenvects").ok()?;
            ns.push(next.map(parametric_normal));
        }
        let lam_pows: Vec<Ex> = (0..n).map(|k| lam.powi(k as i64)).collect();
        let mut best: Option<(usize, Vec<Ex>)> = None;
        for s in 0..n {
            // Entry r: Σⱼ Nⱼ[r][s]·λ^(n−1−j).
            let col: Vec<Ex> = (0..n)
                .map(|r| {
                    let terms = ns.iter().enumerate().filter_map(|(j, nj)| {
                        let coef = &nj.rows[r][s];
                        (!coef.is_zero_structural()).then(|| {
                            let k = n - 1 - j;
                            if k == 0 {
                                coef.clone()
                            } else {
                                coef * &lam_pows[k]
                            }
                        })
                    });
                    Ex::sum_of(&ctx, terms)
                })
                .collect();
            let size: usize = col.iter().map(Ex::count_ops).sum();
            if best.as_ref().is_some_and(|(b, _)| *b <= size) {
                continue;
            }
            if col.iter().any(radical_nonzero_by_value) {
                best = Some((size, col));
            }
        }
        best.map(|(_, v)| v)
    }

    /// Is the matrix diagonalizable (three-valued)?
    ///
    /// - `Some(true)` — the matrix is diagonalizable **for every value of
    ///   its symbols**: every eigenvalue's geometric multiplicity equals its
    ///   algebraic multiplicity, and — when there are symbols — one of the
    ///   following is proved: the matrix is Hermitian; the discriminant of
    ///   the characteristic polynomial vanishes for no value of the symbols
    ///   (the eigenvalues are pairwise different everywhere); or the matrix
    ///   `P` of eigenvectors is finite and invertible for every value (`A·P =
    ///   P·D` holds identically, so `A = P·D·P⁻¹` everywhere: `[[x, y], [y,
    ///   x]]`).  Up to 0.60.0 the answer for generic values was given:
    ///   `[[1, 1], [0, x]]`, a Jordan block at `x = 1`, was `Some(true)`.
    /// - `Some(false)` — non-square, or some eigenvalue is defective.  With
    ///   symbols this is the answer for generic values of them (as SymPy's
    ///   `is_diagonalizable`): every zero test of the eigenvector solves
    ///   was a proof, in the reading of [`rref`](Self::rref).  `[[sin x, 1],
    ///   [0, sin x]]` is a Jordan block for every `x` (`None` up to 0.60.0,
    ///   which answered only for rational functions of the symbols: `[[x,
    ///   1], [0, x]]` was `None` before 0.38).
    /// - `None` — the eigenvalue solver could not find all eigenvalues (a
    ///   part of the spectrum decides nothing here); an entry of an
    ///   eigenvector solve, or the equality of two eigenvalues, is neither
    ///   proved nor refuted; or the matrix is diagonalizable for generic
    ///   values of its symbols and none of the proofs above holds.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// assert_eq!(matrix![ctx, [1, 0], [0, 2]].is_diagonalizable(), Some(true));
    /// assert_eq!(matrix![ctx, [1, 1], [0, 1]].is_diagonalizable(), Some(false));
    /// // Diagonalizable unless x = 1:
    /// let x = ctx.symbol("x");
    /// let m = Matrix::new(vec![vec![ctx.int(1), ctx.int(1)], vec![ctx.int(0), x.clone()]]).unwrap();
    /// assert_eq!(m.is_diagonalizable(), None);
    /// assert_eq!(m.subs(&x, &ctx.int(2)).is_diagonalizable(), Some(true));
    /// assert_eq!(m.subs(&x, &ctx.int(1)).is_diagonalizable(), Some(false));
    /// ```
    pub fn is_diagonalizable(&self) -> Option<bool> {
        if !self.is_square() {
            return Some(false);
        }
        // An error: the spectrum is incomplete, or a zero test of the
        // eigenvector solves (or the equality of two eigenvalues) is
        // undecided.
        let eigvs = self.eigenvects().ok()?;
        let total_alg: usize = eigvs.iter().map(|(_, m, _)| *m).sum();
        if total_alg != self.nrows {
            return None;
        }
        if eigvs.iter().any(|(_, m, v)| v.len() != *m) {
            // Every decision behind the missing eigenvector was a proof.
            return Some(false);
        }
        // Diagonalizable for generic values of the symbols.  For every
        // value?
        let small = || {
            self.iter()
                .map(|e| tree_size_capped(e, EVERYWHERE_PROOF_NODES))
                .sum::<usize>()
                <= EVERYWHERE_PROOF_NODES
        };
        let holds_everywhere = self.iter().all(|e| e.free_symbols().is_empty())
            || eigenvectors_independent_everywhere(&eigvs)
            || self.eigenvalues_differ_everywhere()
            || (small() && Truth::from(self.is_hermitian()).is_true());
        holds_everywhere.then_some(true)
    }

    /// The eigenvalues `pairs` with those proved equal merged
    /// ([`merged_eigen_pairs`]), when every two of the result are proved
    /// different: each difference by a certified value at a sample point
    /// where it is meromorphic in the symbols, or — every eigenvalue being
    /// simple — the discriminant of the characteristic polynomial proved
    /// not identically zero (a difference of eigenvalues in radicals is
    /// analytic only piece by piece: `x − (x + 2 − √((x − 2)²))/2` is 0 for
    /// `x < 2` and not for `x > 2`, and one sample point says nothing of the
    /// other piece).  An error (under `operation`) otherwise: the
    /// multiplicities are not known, and neither is the dimension an
    /// eigenspace should have.
    fn distinct_eigen_pairs(
        &self,
        pairs: Vec<(Ex, usize)>,
        operation: &'static str,
    ) -> Result<Vec<(Ex, usize)>, SymplexError> {
        // One work budget for the tests of all the differences.
        let _work = PivotWork::enter();
        let (merged, open) = merged_eigen_pairs(pairs);
        let Some((w, v)) = open else {
            return Ok(merged);
        };
        let simple = merged.iter().all(|(_, m)| *m == 1);
        if simple
            && self
                .char_poly_discriminant(EVERYWHERE_PROOF_NODES)
                .is_some_and(|d| pivot_truth(&d).is_false())
        {
            return Ok(merged);
        }
        Err(failed(
            operation,
            format!(
                "cannot decide whether the eigenvalues {} and {} are equal, so their \
                 multiplicities are not known",
                shortened(&w),
                shortened(&v)
            ),
        ))
    }

    /// The discriminant of the characteristic polynomial, when its
    /// coefficients have at most `max_nodes` nodes together (the
    /// discriminant has degree `2n − 2` in them) and it can be formed.
    fn char_poly_discriminant(&self, max_nodes: usize) -> Option<Ex> {
        let lam = self.fresh_symbol("lambda");
        let coeffs = self.char_poly_coeffs().ok()?;
        let size: usize = coeffs.iter().map(|c| tree_size_capped(c, max_nodes)).sum();
        if size > max_nodes {
            return None;
        }
        let mut cp = self.ctx_zero();
        for (k, c) in coeffs.iter().enumerate() {
            if !c.is_zero_structural() {
                cp += c * &lam.powi(k as i64);
            }
        }
        cp.discriminant_symbolic(&lam)
    }

    /// Is the discriminant of the characteristic polynomial proved nonzero
    /// for every value of the symbols?  Then the `n` eigenvalues are
    /// pairwise different at every point, whatever form they were found
    /// in.  `false` proves nothing (also: not tried, for a characteristic
    /// polynomial whose coefficients are large).
    fn eigenvalues_differ_everywhere(&self) -> bool {
        self.char_poly_discriminant(EVERYWHERE_PROOF_NODES / 2)
            .is_some_and(|d| {
                tree_size_capped(&d, 8 * EVERYWHERE_PROOF_NODES) <= 8 * EVERYWHERE_PROOF_NODES
                    && Truth::from(d.is_nonzero()).is_true()
            })
    }

    /// Diagonalize: find invertible `P` and diagonal `D` with `A = P D P⁻¹`.
    ///
    /// `P` has the eigenvectors as columns; `D` carries the eigenvalues in
    /// the same order.
    ///
    /// # Errors
    ///
    /// - [`SymplexError::InvalidArgument`] if the matrix is not square.
    /// - [`SymplexError::NotDiagonalizable`] if an eigenvalue has fewer
    ///   eigenvectors than its algebraic multiplicity.  This is a proof (for
    ///   generic values of the symbols, the reading of
    ///   [`rref`](Self::rref)): every eigenvalue is known when it is
    ///   reported (up to 0.59 a matrix whose spectrum was only partly found
    ///   was called not diagonalizable: "insufficient eigenvectors found"),
    ///   and every zero test of the eigenvector solves was decided (up to
    ///   0.60.0 an undecided entry counted as nonzero: the identity matrix
    ///   written `[[1, Γ(x + 1) − x·Γ(x)], [0, 1]]` was "not
    ///   diagonalizable").
    /// - [`SymplexError::ComputationFailed`] if the eigenvalue solver could
    ///   not find all eigenvalues, or if an entry of an eigenvector solve (or
    ///   the equality of two eigenvalues) is neither proved nor refuted: the
    ///   message names it.
    /// - [`SymplexError::ExpressionSwell`] if the eigenvectors exceed
    ///   [`EXPRESSION_BUDGET`].
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let m = matrix![ctx, [2, 1], [0, 3]];
    /// let Diagonalization { p, d } = m.diagonalize().unwrap();
    /// let back = (&(&p * &d) * &p.inv().unwrap()).simplify();
    /// assert_eq!(back, m);
    /// ```
    pub fn diagonalize(&self) -> Result<Diagonalization<Matrix>, SymplexError> {
        self.require_square("diagonalize")?;
        self.check_budget("diagonalize")?;
        debug!(
            "diagonalize: attempting for {}×{} matrix",
            self.nrows, self.ncols
        );
        let eigvs = self.eigenvects().map_err(|e| reop(e, "diagonalize"))?;
        let n = self.nrows;
        let mut total_vecs = 0usize;
        for (_, alg_mult, vecs) in &eigvs {
            if vecs.len() != *alg_mult {
                return Err(SymplexError::NotDiagonalizable {
                    operation: "diagonalize",
                    reason: "geometric multiplicity does not equal algebraic multiplicity for \
                             all eigenvalues"
                        .into(),
                });
            }
            total_vecs += vecs.len();
        }
        if total_vecs != n {
            return Err(SymplexError::NotDiagonalizable {
                operation: "diagonalize",
                reason: "insufficient eigenvectors found".into(),
            });
        }

        let mut p_cols: Vec<&Matrix> = Vec::new();
        let mut diag_entries: Vec<Ex> = Vec::new();
        for (eigenval, _, vecs) in &eigvs {
            for v in vecs {
                p_cols.push(v);
                diag_entries.push(eigenval.clone());
            }
        }
        let p = Matrix::hstack(&p_cols)?;
        p.check_budget("diagonalize")?;
        let d = Matrix::diag(&diag_entries)?;
        Ok(Diagonalization { p, d })
    }

    /// Jordan normal form: `P` and block-diagonal `J` such that `A = P J P⁻¹`.
    ///
    /// `J` consists of Jordan blocks `J_k(λ)` (eigenvalue on the diagonal,
    /// ones on the superdiagonal); `P` holds the (generalized)
    /// eigenvectors.  For diagonalizable matrices this equals
    /// [`diagonalize`](Self::diagonalize).
    ///
    /// # Errors
    ///
    /// - [`SymplexError::InvalidArgument`] if the matrix is not square.
    /// - [`SymplexError::ComputationFailed`] if the eigenvalue solver cannot
    ///   find all eigenvalues or an intermediate result exceeds
    ///   [`EXPRESSION_BUDGET`].
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// // Defective: eigenvalue 2 with algebraic mult 2, geometric mult 1
    /// let m = matrix![ctx, [2, 1, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]];
    /// let JordanForm { p, j } = m.jordan_form().unwrap();
    /// assert_eq!((&(&p * &j) * &p.inv().unwrap()).simplify(), m);
    /// ```
    pub fn jordan_form(&self) -> Result<JordanForm<Matrix>, SymplexError> {
        self.require_square("jordan_form")?;
        self.check_budget("jordan_form")?;
        let n = self.nrows;
        debug!(n, "jordan_form: computing for {}×{} matrix", n, n);
        let eye = Matrix::identity(&self.ctx(), n)?;

        let eigvs = self.eigenvects().map_err(|e| reop(e, "jordan_form"))?;

        // Fast path: diagonalizable.
        let total_vecs: usize = eigvs.iter().map(|(_, _, v)| v.len()).sum();
        let all_match = eigvs.iter().all(|(_, m, v)| v.len() == *m);
        if all_match && total_vecs == n {
            let Diagonalization { p, d } = self.diagonalize()?;
            return Ok(JordanForm { p, j: d });
        }

        let total_alg: usize = eigvs.iter().map(|(_, m, _)| *m).sum();
        if total_alg != n {
            return Err(failed(
                "jordan_form",
                format!(
                    "eigenvalue solver found algebraic multiplicity sum {} but matrix is {}×{}",
                    total_alg, n, n
                ),
            ));
        }

        let mut jordan_rows: Vec<Vec<Ex>> = Vec::new();
        let mut basis_cols: Vec<Matrix> = Vec::new();

        for (eigenval, alg_mult, _) in &eigvs {
            let a_minus_lambda = self.sub(&eye.scale(eigenval))?;

            // Nullity chain: [0, nullity(E), nullity(E²), …] where E = A − λI.
            let mut chain: Vec<usize> = vec![0];
            let mut power = a_minus_lambda.clone();
            loop {
                let nullity = n - power.rank_decided("jordan_form")?;
                let last = *chain.last().unwrap_or(&0);
                if nullity == last || nullity >= *alg_mult {
                    if nullity > last {
                        chain.push(nullity);
                    }
                    break;
                }
                chain.push(nullity);
                power = power.matmul(&a_minus_lambda)?;
                power.check_budget("jordan_form")?;
            }

            // block_counts[i] = number of Jordan blocks of size (i+1).
            let max_block_size = chain.len() - 1;
            let mut block_counts: Vec<usize> = Vec::new();
            for i in 0..max_block_size {
                let diff_curr = chain[i + 1] - chain[i];
                let diff_next = if i + 2 < chain.len() {
                    chain[i + 2] - chain[i + 1]
                } else {
                    0
                };
                block_counts.push(diff_curr - diff_next);
            }
            let mut blocks: Vec<(usize, usize)> = block_counts
                .iter()
                .enumerate()
                .filter(|(_, c)| **c > 0)
                .map(|(i, c)| (i + 1, *c))
                .collect();
            blocks.sort_by_key(|b| std::cmp::Reverse(b.0));

            let mut eig_basis: Vec<Matrix> = Vec::new();
            for (block_size, count) in &blocks {
                for _ in 0..*count {
                    let null_big = jordan_null_power(&a_minus_lambda, *block_size)?;
                    let null_small = if *block_size > 1 {
                        jordan_null_power(&a_minus_lambda, block_size - 1)?
                    } else {
                        Vec::new()
                    };
                    let exclude: Vec<&Matrix> = null_small.iter().chain(eig_basis.iter()).collect();
                    let Some(vec) = pick_independent_vec(&null_big, &exclude)? else {
                        return Err(failed(
                            "jordan_form",
                            "could not find independent generalized eigenvector",
                        ));
                    };

                    // Jordan chain [E^(k−1)·v, …, E·v, v]: eigenvector first.
                    let mut chain_vecs: Vec<Matrix> = Vec::with_capacity(*block_size);
                    for i in (0..*block_size).rev() {
                        chain_vecs.push(if i == 0 {
                            vec.clone()
                        } else {
                            matrix_pow_vec(&a_minus_lambda, &vec, i)?
                        });
                    }
                    eig_basis.extend(chain_vecs.iter().cloned());
                    basis_cols.extend(chain_vecs);

                    let bs = *block_size;
                    let col_offset = jordan_rows.len();
                    for row_idx in 0..bs {
                        let mut row = vec![self.ctx_zero(); n];
                        row[col_offset + row_idx] = eigenval.clone();
                        if row_idx + 1 < bs {
                            row[col_offset + row_idx + 1] = self.ctx_one();
                        }
                        jordan_rows.push(row);
                    }
                }
            }
        }

        if jordan_rows.len() != n || basis_cols.len() != n {
            return Err(failed(
                "jordan_form",
                format!(
                    "internal error: expected {} basis vectors, got {}",
                    n,
                    basis_cols.len()
                ),
            ));
        }

        let j = Matrix::from_rows_unchecked(jordan_rows);
        let col_refs: Vec<&Matrix> = basis_cols.iter().collect();
        let p = Matrix::hstack(&col_refs)?;
        p.check_budget("jordan_form")?;
        Ok(JordanForm { p, j })
    }

    /// Integer power of a square matrix via repeated squaring.
    ///
    /// `powi(0)` is the identity.  Negative powers are not supported —
    /// use `inv()?.powi(k)`.
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::InvalidArgument`] if the matrix is not square.
    pub fn powi(&self, n: u32) -> Result<Matrix, SymplexError> {
        self.require_square("powi")?;
        if n == 0 {
            return Matrix::identity(&self.ctx(), self.nrows);
        }
        if n == 1 {
            return Ok(self.clone());
        }
        let mut result = Matrix::identity(&self.ctx(), self.nrows)?;
        let mut base = self.clone();
        let mut exp = n;
        while exp > 0 {
            if exp % 2 == 1 {
                result = result.matmul(&base)?;
            }
            exp /= 2;
            if exp > 0 {
                base = base.matmul(&base)?;
            }
        }
        Ok(result)
    }

    /// Kronecker (tensor) product `A ⊗ B`.
    ///
    /// For `A` (m×n) and `B` (p×q), produces an (mp×nq) matrix.
    pub fn kronecker(&self, other: &Matrix) -> Matrix {
        let mut rows = Vec::with_capacity(self.nrows * other.nrows);
        for i in 0..self.nrows {
            for k in 0..other.nrows {
                let mut row = Vec::with_capacity(self.ncols * other.ncols);
                for a in self.rows[i].iter().take(self.ncols) {
                    for b in other.rows[k].iter().take(other.ncols) {
                        row.push(a * b);
                    }
                }
                rows.push(row);
            }
        }
        Matrix::from_rows_unchecked(rows)
    }

    /// Matrix exponential via truncated Taylor series `eᴬ ≈ Σₖ₌₀ⁿ Aᵏ/k!`.
    ///
    /// This is an *approximation*.  For exact results use
    /// [`matrix_exp`](Self::matrix_exp).
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::InvalidArgument`] if the matrix is not square.
    pub fn exp_series(&self, order: usize) -> Result<Matrix, SymplexError> {
        self.require_square("exp_series")?;
        let n = self.nrows;
        let ctx = self.ctx();
        let mut result = Matrix::identity(&ctx, n)?;
        let mut term = result.clone();
        for k in 1..=order {
            term = term.matmul(self)?;
            let inv_k = ctx.rational(1, k as i64);
            term = term.scale(&inv_k);
            result = result.add(&term)?;
        }
        Ok(result)
    }

    /// Exact matrix exponential `eᴬ` via the Jordan decomposition
    /// `eᴬ = P · e^J · P⁻¹`.
    ///
    /// For each Jordan block `J_k(λ)`:
    ///
    /// ```text
    /// e^{J_k(λ)} = e^λ · [ 1,  1,  1/2!, …, 1/(k−1)! ]
    ///                    [ 0,  1,  1,    …, 1/(k−2)! ]
    ///                    [ …                        ]
    /// ```
    ///
    /// When `P·e^J·P⁻¹` exceeds [`EXPRESSION_BUDGET`] (typically `RootOf`
    /// eigenvalues) and the eigenvalues are distinct, Sylvester's formula
    /// `eᴬ = Σᵢ e^λᵢ·adj(λᵢI − A)/p′(λᵢ)` is used instead, which inverts
    /// nothing.
    ///
    /// Unlike a numerical library this never falls back to a series
    /// approximation; use [`exp_series`](Self::exp_series) explicitly if
    /// an approximation is acceptable.
    ///
    /// # Errors
    ///
    /// - [`SymplexError::InvalidArgument`] if the matrix is not square.
    /// - [`SymplexError::ComputationFailed`] if the Jordan form cannot be
    ///   computed (eigenvalues not found in closed form) or the result
    ///   exceeds [`EXPRESSION_BUDGET`].
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let m = matrix![ctx, [0, 1], [-1, 0]];
    /// let e = m.matrix_exp().unwrap();
    /// // e^A = [[cos 1, sin 1], [−sin 1, cos 1]]: complex-conjugate eigenvalue
    /// // pairs are rewritten with Euler's formula, so the entries are real trig.
    /// assert_eq!(e[(0, 0)], ctx.one().cos());
    /// assert_eq!(e[(0, 1)], ctx.one().sin());
    /// ```
    pub fn matrix_exp(&self) -> Result<Matrix, SymplexError> {
        self.matrix_exp_impl(None)
    }

    /// Exact matrix exponential `e^{At}` for a scalar `t`.
    ///
    /// Equivalent to `self.scale(t).matrix_exp()` but keeps `t` out of the
    /// eigenvalue computation, so it works for any `t` (including symbols)
    /// whenever [`matrix_exp`](Self::matrix_exp) works for `A`.  For each
    /// Jordan block the entries are `e^{λt} · t^{j−i} / (j−i)!`.
    ///
    /// # Errors
    ///
    /// Same conditions as [`matrix_exp`](Self::matrix_exp).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let t = ctx.symbol("t");
    /// let a = matrix![ctx, [0, 1], [0, 0]];
    /// let e = a.matrix_exp_t(&t).unwrap();
    /// assert_eq!(e, Matrix::new(vec![vec![ctx.int(1), t.clone()], vec![ctx.int(0), ctx.int(1)]]).unwrap());
    /// ```
    pub fn matrix_exp_t(&self, t: &Ex) -> Result<Matrix, SymplexError> {
        self.matrix_exp_impl(Some(t))
    }

    fn matrix_exp_impl(&self, t: Option<&Ex>) -> Result<Matrix, SymplexError> {
        self.require_square("matrix_exp")?;
        let f = |l: &Ex| match t {
            Some(t) => (l * t).exp(),
            None => l.exp(),
        };
        if let Some(m) = self.radical_spectrum_function("matrix_exp", &f, false, false)? {
            return Ok(match t {
                Some(t) => m.map(|e| collect_exponents(e, t)),
                None => m,
            });
        }
        match self.matrix_exp_jordan(t) {
            Err(e) if is_swell(&e) => self.sylvester_function("matrix_exp", &f)?.ok_or(e),
            r => r,
        }
    }

    /// `f(A)` by Sylvester's formula for a matrix with `n` distinct
    /// eigenvalues: `f(A) = Σᵢ f(λᵢ)·adj(λᵢI − A)/p′(λᵢ)` with
    /// `p(t) = det(tI − A)` and `adj(tI − A) = Σⱼ Nⱼ·t^(n−1−j)`,
    /// `N₀ = I`, `Nⱼ = A·Nⱼ₋₁ + cₙ₋ⱼ·I` (Faddeev–LeVerrier; Higham,
    /// *Functions of Matrices*, §1.2).  No eigenvector matrix is inverted,
    /// which is where `P·f(J)·P⁻¹` swells when the eigenvalues are
    /// `RootOf` values: before 0.38 `matrix_exp` of a 4×4 integer matrix
    /// with an irreducible quartic characteristic polynomial was refused
    /// ("expression swell").  `Ok(None)` when the eigenvalues are not
    /// distinct; the error of
    /// [`eigenvals_with_multiplicity`](Self::eigenvals_with_multiplicity)
    /// when they are not all found.
    pub(crate) fn sylvester_function(
        &self,
        op: &'static str,
        f: &dyn Fn(&Ex) -> Ex,
    ) -> Result<Option<Matrix>, SymplexError> {
        let pairs = self.eigenvals_with_multiplicity()?;
        self.sylvester_with(op, f, &pairs)
    }

    /// `f(A)` by Sylvester's formula straight away when the eigenvalues are
    /// distinct and some are Cardano/Ferrari radicals (a root of index ≥ 3
    /// of a non-rational radicand, or a nested radical of the symbols): the
    /// eigenvector route `P·f(J)·P⁻¹` then needs `det P`, whose expansion in
    /// such radicals is exponential — before 0.44 `matrix_exp` of a 4×4
    /// Hermitian matrix over `ℚ(√2, √3, i)` with an irreducible quartic
    /// characteristic polynomial ran for minutes.  `Ok(None)` otherwise (also
    /// when the eigenvalues are not all found: the caller's route reports
    /// that).
    /// With `nonsingular`, also `Ok(None)` when an eigenvalue is zero (the
    /// logarithm: the caller's route reports the singular matrix).
    ///
    /// With `branch_cut` (`√`, `ln`) an eigenvalue with Cardano's cube root
    /// of the symbols ([`has_parametric_cardano_root`]) is refused: at a real
    /// value of the symbols where such an eigenvalue is a negative real
    /// number (three real roots, *casus irreducibilis*) its imaginary part
    /// cancels only exactly, so `√λ` sits on the cut and no evaluation
    /// decides its branch — every entry of the result would be unevaluable
    /// there.
    pub(crate) fn radical_spectrum_function(
        &self,
        op: &'static str,
        f: &dyn Fn(&Ex) -> Ex,
        nonsingular: bool,
        branch_cut: bool,
    ) -> Result<Option<Matrix>, SymplexError> {
        if !self.is_square() || self.nrows < 3 || self.as_qmatrix().is_some() {
            return Ok(None);
        }
        let Ok(pairs) = self.eigenvals_with_multiplicity() else {
            return Ok(None);
        };
        if branch_cut && pairs.iter().any(|(l, _)| has_parametric_cardano_root(l)) {
            return Err(failed(
                op,
                "the eigenvalues are roots of an irreducible symbolic cubic or quartic \
                 (Cardano/Ferrari radicals), whose principal branch is not decidable where \
                 they are real; substitute values for the symbols first",
            ));
        }
        if !pairs
            .iter()
            .any(|(l, _)| has_nested_higher_root(l) || is_parametric_radical(l))
            || (nonsingular && !pairs.iter().all(|(l, _)| radical_nonzero_by_value(l)))
        {
            return Ok(None);
        }
        self.sylvester_with(op, f, &pairs)
    }

    /// [`sylvester_function`](Self::sylvester_function) with the eigenvalues
    /// already computed.
    fn sylvester_with(
        &self,
        op: &'static str,
        f: &dyn Fn(&Ex) -> Ex,
        pairs: &[(Ex, usize)],
    ) -> Result<Option<Matrix>, SymplexError> {
        let n = self.nrows;
        if pairs.len() != n || pairs.iter().any(|(_, m)| *m != 1) {
            return Ok(None);
        }
        let ctx = self.ctx();
        let tidy = |e: Ex| {
            if self.as_qmatrix().is_some() {
                e.eval()
            } else {
                crate::domains::linalg::algebraic_constant_normal_form(&e)
                    .unwrap_or_else(|| e.expand())
            }
        };
        // Monic `p(t) = (−1)ⁿ·det(A − tI)`, ascending.
        let sign = ctx.int(if n.is_multiple_of(2) { 1 } else { -1 });
        let c: Vec<Ex> = self
            .char_poly_coeffs()?
            .iter()
            .map(|x| tidy(x * &sign))
            .collect();
        let eye = Matrix::identity(&ctx, n)?;
        let mut ns: Vec<Matrix> = vec![eye.clone()];
        for j in 1..n {
            let prev = &ns[j - 1];
            let next = self.matmul(prev)?.add(&eye.scale(&c[n - j]))?;
            ns.push(next.map(|e| tidy(e.clone())));
        }
        let mut out: Vec<Vec<Ex>> = vec![vec![ctx.zero(); n]; n];
        for (l, _) in pairs {
            // Horner: p′(λ) = Σ k·c_k·λ^(k−1), and Σⱼ Nⱼ[r][s]·λ^(n−1−j).
            let mut dp = &c[n] * &ctx.int(n as i64);
            for k in (1..n).rev() {
                dp = &(&dp * l) + &(&c[k] * &ctx.int(k as i64));
            }
            let w = (&f(l) / &dp).eval();
            for (r, row) in out.iter_mut().enumerate() {
                for (s, cell) in row.iter_mut().enumerate() {
                    let mut q = ns[0].rows[r][s].clone();
                    for nj in &ns[1..] {
                        q = &(&q * l) + &nj.rows[r][s];
                    }
                    *cell = &*cell + &(&w * &q);
                }
            }
        }
        let result = Matrix::from_rows_unchecked(out);
        result.check_budget(op)?;
        Ok(Some(result.map(simplify_function_entry)))
    }

    /// [`matrix_exp_impl`](Self::matrix_exp_impl) through the Jordan form.
    fn matrix_exp_jordan(&self, t: Option<&Ex>) -> Result<Matrix, SymplexError> {
        debug!(
            "matrix_exp: computing for {}×{} matrix",
            self.nrows, self.ncols
        );
        let n = self.nrows;
        let ctx = self.ctx();

        let JordanForm { p, j } = self.jordan_form().map_err(|e| {
            swell_or("matrix_exp", e, |e| {
                failed(
                    "matrix_exp",
                    format!(
                        "Jordan form unavailable ({e}); use exp_series(order) for a truncated approximation"
                    ),
                )
            })
        })?;

        let mut exp_j_rows: Vec<Vec<Ex>> = vec![vec![self.ctx_zero(); n]; n];
        let mut col = 0;
        while col < n {
            let lambda = j.rows[col][col].clone();
            let mut block_size = 1;
            while col + block_size < n {
                let superdiag = &j.rows[col + block_size - 1][col + block_size];
                let diag_next = &j.rows[col + block_size][col + block_size];
                if !superdiag.is_one_structural() || diag_next != &lambda {
                    break;
                }
                block_size += 1;
            }
            trace!(col, block_size, "matrix_exp: processing Jordan block");

            let exponent = match t {
                Some(t) => &lambda * t,
                None => lambda.clone(),
            };
            let exp_lambda = exponent.exp();

            for i in 0..block_size {
                for jj in i..block_size {
                    let d = jj - i;
                    let factorial_val = ctx.from_bigint(factorial(d as u64));
                    let mut entry = &exp_lambda / &factorial_val;
                    if d > 0
                        && let Some(t) = t
                    {
                        entry *= t.powi(d as i64);
                    }
                    exp_j_rows[col + i][col + jj] = entry;
                }
            }
            col += block_size;
        }
        let exp_j = Matrix::from_rows_unchecked(exp_j_rows);

        let p_inv = p.inv().map_err(|e| {
            swell_or("matrix_exp", e, |_| {
                failed(
                    "matrix_exp",
                    "eigenvector matrix is singular (internal inconsistency)",
                )
            })
        })?;
        let result = p.matmul(&exp_j)?.matmul(&p_inv)?;
        result.check_budget("matrix_exp")?;
        let i_unit = ctx.i_unit();
        Ok(result.map(|e| {
            // `simplify` cannot do anything useful with `RootOf` values or
            // Cardano cube roots but is very slow on them; constant folding
            // is all they need.
            if has_root_of(e) || has_nested_higher_root(e) {
                return e.eval();
            }
            let s = e.simplify();
            // Complex-conjugate eigenvalue pairs: Euler's formula turns
            // `½e^{iωt} + ½e^{−iωt}` into `cos(ωt)` (valid for any complex
            // argument, so no realness assumption is needed).  Only when
            // there is an exponential to rewrite: before 0.38 the `expand`
            // ran on every entry with an `i`, and a 3×3 Hermitian matrix over
            // `ℚ(√2, √3, i)` took 15 s for a 179 MB result.
            let s = if s.contains(&i_unit) {
                let w = s.rewrite_as_trig();
                if w == s {
                    s
                } else {
                    fix_trig_parity(&w).expand().eval().simplify()
                }
            } else {
                s
            };
            match t {
                // Products distributed over sums (not inside functions) and
                // exponents collected in `t`: every entry a sum of
                // `c·tᵏ·e^(λt)` (times `sin`/`cos`), each term of which the
                // integrator takes (`solve_ode_system_nonhomogeneous`
                // integrates `e^(−At)·b`; `t·(sin t + cos t)·eᵗ` stayed
                // an unevaluated `Integral`).
                Some(t) => {
                    let mut opts = crate::transforms::expand::ExpandOpts::none().with_mul(true);
                    opts.deep = false;
                    collect_exponents(&s.expand_with(&opts), t)
                }
                None => s,
            }
        }))
    }

    // ── Pseudo-inverse, rank factorisation, singular values, Hessenberg ──

    /// Moore–Penrose pseudo-inverse `A⁺` (`n × m` for an `m × n` matrix),
    /// for **any** rank.  SymPy: `Matrix.pinv()`.
    ///
    /// * Full column rank: `A⁺ = (AᵀA)⁻¹Aᵀ`.
    /// * Rank-deficient: via the full-rank factorisation `A = C·F` of
    ///   [`rank_decomposition`](Self::rank_decomposition),
    ///   `A⁺ = Fᵀ (F Fᵀ)⁻¹ (Cᵀ C)⁻¹ Cᵀ`; the zero matrix maps to the zero
    ///   `n × m` matrix.
    ///
    /// Rational matrices are computed exactly on [`QMatrix`].  For symbolic
    /// matrices the rank is the rank of [`rref`](Self::rref) (generic
    /// rank for symbolic pivots).  The adjoints are conjugate transposes
    /// (`Aᴴ`, as SymPy's), plain transposes when every entry is provably
    /// real; before 0.38 `Aᵀ` was used for complex entries too, which is
    /// not the Moore–Penrose inverse.
    ///
    /// # Errors
    ///
    /// [`SymplexError::ComputationFailed`] if a symbolic `AᵀA` is singular
    /// although the structural pivot search found full column rank
    /// (simplify the entries first); [`SymplexError::ExpressionSwell`] on
    /// expression swell beyond [`EXPRESSION_BUDGET`].
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// // Rank 1: A⁺ = (1/25)·Aᵀ
    /// let a = matrix![ctx, [1, 2], [2, 4]];
    /// let p = a.pinv().unwrap();
    /// assert_eq!(p, a.transpose().scale(&ctx.rational(1, 25)));
    /// assert_eq!(&(&a * &p) * &a, a);            // A A⁺ A = A
    /// // Full column rank: the classical formula
    /// let b = matrix![ctx, [1, 0], [0, 1], [1, 1]];
    /// assert_eq!(&b.pinv().unwrap() * &b, Matrix::identity(&ctx, 2).unwrap());
    /// ```
    pub fn pinv(&self) -> Result<Matrix, SymplexError> {
        let ctx = self.ctx();
        if let Some(q) = self.as_qmatrix() {
            return q
                .pinv()
                .map(|p| p.to_matrix(&ctx))
                .map_err(|e| reop(e, "pinv"));
        }
        self.check_budget("pinv")?;
        let inv_err = |what: &'static str| {
            move |e: SymplexError| {
                swell_or("pinv", e, |_| {
                    failed(
                        "pinv",
                        format!(
                            "{what} is singular: the columns are linearly dependent in a way the \
                             structural pivot search did not detect; simplify the entries first"
                        ),
                    )
                })
            }
        };
        let (r, pivots) = self.rref_decided("pinv")?;
        if pivots.is_empty() {
            return Matrix::zeros(&ctx, self.ncols, self.nrows);
        }
        // `Aᴴ` (SymPy's `pinv` uses `.H`); `Aᵀ` only when every entry is
        // provably real.  Before 0.38 `Aᵀ` was used for complex `A` too, so
        // `(A·A⁺)ᴴ ≠ A·A⁺` for `[[0, 1, √2·i], [0, −√2, √3], [0, 0, 2]]`.
        let real = self.iter().all(|e| Truth::from(e.is_real()).is_true());
        let adj = |m: &Matrix| if real { m.transpose() } else { m.adjoint() };
        let at = adj(self);
        if pivots.len() == self.ncols {
            let ata = at.matmul(self)?;
            let ata_inv = ata
                .inv()
                .map_err(inv_err(if real { "AᵀA" } else { "AᴴA" }))?;
            return ata_inv.matmul(&at);
        }
        let c = self.select_cols(&pivots)?;
        let f = r.select_rows(&(0..pivots.len()).collect::<Vec<_>>())?;
        let ft = adj(&f);
        let ct = adj(&c);
        let fft_inv = f.matmul(&ft)?.inv().map_err(inv_err("FFᵀ"))?;
        let ctc_inv = ct.matmul(&c)?.inv().map_err(inv_err("CᵀC"))?;
        let result = ft.matmul(&fft_inv)?.matmul(&ctc_inv)?.matmul(&ct)?;
        result.check_budget("pinv")?;
        Ok(result)
    }

    /// Full-rank factorisation `A = C·F`: `C` (`m × r`) is made of the
    /// pivot columns of `A`, `F` (`r × n`) of the nonzero rows of
    /// [`rref`](Self::rref), with `r = rank A`.  SymPy:
    /// `Matrix.rank_decomposition()`.
    ///
    /// Rational matrices route through [`QMatrix`]; for symbolic matrices
    /// the rank is the structural rank of `rref` (symbolic pivots are
    /// treated as non-zero).
    ///
    /// # Errors
    ///
    /// [`SymplexError::ComputationFailed`] for the (structurally) zero
    /// matrix: rank 0 would make `C` and `F` empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let a = matrix![ctx, [1, 2, 3], [4, 5, 6], [7, 8, 9]];
    /// let RankDecomposition { c, f } = a.rank_decomposition().unwrap();
    /// assert_eq!(c, matrix![ctx, [1, 2], [4, 5], [7, 8]]);
    /// assert_eq!(f, matrix![ctx, [1, 0, -1], [0, 1, 2]]);
    /// assert_eq!(&c * &f, a);
    /// ```
    pub fn rank_decomposition(&self) -> Result<RankDecomposition<Matrix>, SymplexError> {
        let ctx = self.ctx();
        if let Some(q) = self.as_qmatrix() {
            let RankDecomposition { c, f } = q
                .rank_decomposition()
                .map_err(|e| reop(e, "rank_decomposition"))?;
            return Ok(RankDecomposition {
                c: c.to_matrix(&ctx),
                f: f.to_matrix(&ctx),
            });
        }
        let (r, pivots) = self.rref_decided("rank_decomposition")?;
        if pivots.is_empty() {
            return Err(failed(
                "rank_decomposition",
                "matrix is zero (rank 0); the factors C (m×0) and F (0×n) would be empty",
            ));
        }
        let c = self.select_cols(&pivots)?;
        let f = r.select_rows(&(0..pivots.len()).collect::<Vec<_>>())?;
        Ok(RankDecomposition { c, f })
    }

    /// Singular values: the square roots of the eigenvalues of `AᵀA`,
    /// `ncols` of them, sorted in descending order when they can be
    /// compared numerically.  SymPy: `Matrix.singular_values()`.
    ///
    /// The eigenvalues come from [`eigenvals`](Self::eigenvals) of the
    /// smaller Gram matrix (`AᵀA` or `AAᵀ`; their nonzero eigenvalues
    /// coincide, and the list is padded with zeros to `ncols` entries).
    /// The result is therefore exact — rationals, radicals, or `RootOf`
    /// values — exactly when `eigenvals` can solve the characteristic
    /// polynomial of the Gram matrix: always for rational matrices
    /// (radicals when its factors have degree ≤ 2 or a compact radical
    /// form, `RootOf` otherwise), and for symbolic matrices whose Gram
    /// matrix is at most 2×2 or has a factorable characteristic
    /// polynomial.  The Gram matrix is `AᴴA` (`AᵀA` when every entry is
    /// provably real), as SymPy's.
    ///
    /// # Errors
    ///
    /// [`SymplexError::ComputationFailed`] if `eigenvals` cannot find every
    /// eigenvalue of the Gram matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// assert_eq!(matrix![ctx, [2, 0], [0, 3]].singular_values().unwrap(), vec![ctx.int(3), ctx.int(2)]);
    /// // SymPy: Matrix([[1, 2], [3, 4]]).singular_values()
    /// //   == [sqrt(sqrt(221) + 15), sqrt(15 - sqrt(221))]
    /// let sv = matrix![ctx, [1, 2], [3, 4]].singular_values().unwrap();
    /// let s221 = ctx.int(221).sqrt();
    /// assert_eq!(sv[0], (&s221 + 15).sqrt());
    /// assert_eq!(sv[1], (-&s221 + 15).sqrt());
    /// // 2×3: two nonzero singular values, padded with a zero
    /// let sv = matrix![ctx, [3, 0, 0], [0, 4, 0]].singular_values().unwrap();
    /// assert_eq!(sv, vec![ctx.int(4), ctx.int(3), ctx.int(0)]);
    /// ```
    pub fn singular_values(&self) -> Result<Vec<Ex>, SymplexError> {
        let ctx = self.ctx();
        let (m, n) = self.shape();
        let gram = if let Some(q) = self.as_qmatrix() {
            let qt = q.transpose();
            let g = if m >= n { qt.matmul(q) } else { q.matmul(&qt) };
            g.map_err(|e| reop(e, "singular_values"))?.to_matrix(&ctx)
        } else {
            // `AᴴA` (SymPy's `_singular_values`: `M.H.multiply(M)`); `AᵀA`
            // only when every entry is provably real.  Before 0.32 `AᵀA` was
            // used for complex `A` too: `[[i, 0], [0, 2]]` had the "singular
            // value" `i`.
            let at = if self.iter().all(|e| Truth::from(e.is_real()).is_true()) {
                self.transpose()
            } else {
                self.adjoint()
            };
            if m >= n {
                at.matmul(self)?
            } else {
                self.matmul(&at)?
            }
        };
        let dim = gram.nrows;
        // A structurally diagonal Gram matrix (orthogonal columns) has its
        // eigenvalues on the diagonal; skip the characteristic polynomial,
        // whose quadratic-formula roots would hide `a²` inside `√((a²−b²)²)`.
        let eig = if Truth::from(gram.is_diagonal()).is_true() {
            gram.diagonal()
        } else {
            gram.eigenvals().map_err(|e| reop(e, "singular_values"))?
        };
        if eig.len() < dim {
            return Err(failed(
                "singular_values",
                format!(
                    "found only {} of the {dim} eigenvalues of the Gram matrix AᵀA",
                    eig.len()
                ),
            ));
        }
        let mut vals: Vec<Ex> = eig.iter().map(|l| l.sqrt().eval()).collect();
        vals.resize_with(vals.len().max(n), || ctx.zero());
        sort_descending_numeric(&mut vals);
        Ok(vals)
    }

    /// Condition number in the 2-norm, `σ_max / σ_min`, from
    /// [`singular_values`](Self::singular_values).  SymPy:
    /// `Matrix.condition_number()`.
    ///
    /// When the singular values cannot be ordered numerically (symbolic
    /// entries) the result is `Max(σ₁, …) / Min(σ₁, …)`.
    ///
    /// # Errors
    ///
    /// - [`SymplexError::ComputationFailed`] with reason `"matrix is
    ///   singular …"` if the smallest singular value is provably zero
    ///   (SymPy returns `zoo` here).
    /// - Anything [`singular_values`](Self::singular_values) returns.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// assert_eq!(matrix![ctx, [2, 0], [0, 3]].condition_number().unwrap(), ctx.rational(3, 2));
    /// // SymPy: sqrt(sqrt(221) + 15)/sqrt(15 - sqrt(221)) ≈ 14.933
    /// let k = matrix![ctx, [1, 2], [3, 4]].condition_number().unwrap().eval_f64().unwrap();
    /// assert!((k - 14.933034373659268).abs() < 1e-9);
    /// assert!(matrix![ctx, [1, 2], [2, 4]].condition_number().is_err());
    /// ```
    pub fn condition_number(&self) -> Result<Ex, SymplexError> {
        // Rank first, decided exactly: a zero singular value of a symbolic
        // matrix is often not recognisable as such (`√(a − √(a²))`), and
        // before 0.38 `[[x − 2, 2], [0, 0]]` had `Max(…)/Min(…)` with an
        // identically zero `Min` (SymPy 1.14 gives `Max(0, σ)/Min(0, σ)`
        // too, without deciding the zero).
        if self.as_qmatrix().is_none()
            && self.rank_decided("condition_number")? < self.nrows.min(self.ncols)
        {
            return Err(failed(
                "condition_number",
                "matrix is singular (smallest singular value is 0), condition number is infinite",
            ));
        }
        let sv = self
            .singular_values()
            .map_err(|e| reop(e, "condition_number"))?;
        let ctx = self.ctx();
        let numeric = sv.iter().all(|v| v.eval_f64().is_ok());
        // A singular value in Cardano radicals (a symbolic 3×3 or 4×4 Gram
        // matrix) is real, but its imaginary part cancels only exactly, so
        // `max`/`min` would not evaluate ("min requires real arguments");
        // `re(σ) = σ` makes them evaluable.
        let real = |v: &Ex| {
            if has_nested_higher_root(v) {
                v.re()
            } else {
                v.clone()
            }
        };
        let (max, min) = match (sv.first(), sv.last()) {
            (Some(first), Some(last)) if numeric => (first.clone(), last.clone()),
            _ => (
                Ex::max_of(&ctx, sv.iter().map(real)),
                Ex::min_of(&ctx, sv.iter().map(real)),
            ),
        };
        if ex_is_zero(&min).is_true() {
            return Err(failed(
                "condition_number",
                "matrix is singular (smallest singular value is 0), condition number is infinite",
            ));
        }
        Ok(&max / &min)
    }

    /// Upper Hessenberg form by Gaussian similarity transforms:
    /// [`Hessenberg`]`{ h, p }` with `H = P⁻¹ A P` and `h_ij = 0` for
    /// `i > j + 1`.  SymPy: `Matrix.upper_hessenberg_decomposition()`
    /// (Householder reflections, hence radicals; this variant uses
    /// eliminations and stays in the field of the entries).
    ///
    /// Column `k` is cleared below the sub-diagonal with the first usable
    /// entry as pivot — moved into row `k + 1` by a symmetric row/column
    /// swap when needed — followed by `row_j −= f·row_{k+1}` and the
    /// compensating `col_{k+1} += f·col_j`.  Rational matrices run exactly
    /// on [`QMatrix`].  For symbolic matrices a pivot is chosen among the
    /// entries that are non-zero for every value of the symbols, falling
    /// back to one that is proved not identically zero — so the identity
    /// `A·P = P·H` holds generically (wherever those pivots do not vanish),
    /// exactly as for [`lu`](Self::lu) and [`rref`](Self::rref).  Symbolic
    /// entries are simplified as they are produced.
    ///
    /// # Errors
    ///
    /// - [`SymplexError::InvalidArgument`] if the matrix is not square.
    /// - [`SymplexError::ComputationFailed`] if a column has several entries
    ///   below the diagonal that are not proved zero and none of them is
    ///   proved nonzero (up to 0.60.0 the first of them was the pivot,
    ///   *assumed* non-zero).
    /// - [`SymplexError::ExpressionSwell`] on expression swell beyond
    ///   [`EXPRESSION_BUDGET`].
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let a = matrix![ctx, [1, 2, 3], [4, 5, 6], [7, 8, 10]];
    /// let Hessenberg { h, p } = a.hessenberg().unwrap();
    /// assert!(h[(2, 0)].is_zero_structural());
    /// assert_eq!(&a * &p, &p * &h);
    /// // One elimination with f = 7/4: row₂ −= f·row₁, col₁ += f·col₂.
    /// assert_eq!(p, Matrix::new(vec![
    ///     vec![ctx.int(1), ctx.int(0), ctx.int(0)],
    ///     vec![ctx.int(0), ctx.int(1), ctx.int(0)],
    ///     vec![ctx.int(0), ctx.rational(7, 4), ctx.int(1)],
    /// ]).unwrap());
    /// assert_eq!(h[(2, 1)], ctx.rational(-13, 8));
    /// ```
    pub fn hessenberg(&self) -> Result<Hessenberg<Matrix>, SymplexError> {
        self.require_square("hessenberg")?;
        let ctx = self.ctx();
        if let Some(q) = self.as_qmatrix() {
            let Hessenberg { h, p } = q.hessenberg().map_err(|e| reop(e, "hessenberg"))?;
            return Ok(Hessenberg {
                h: h.to_matrix(&ctx),
                p: p.to_matrix(&ctx),
            });
        }
        self.check_budget("hessenberg")?;
        let _work = PivotWork::enter();
        let n = self.nrows;
        let zero = self.ctx_zero();
        let one = self.ctx_one();
        let mut h: Vec<Vec<Ex>> = self.rows.clone();
        let mut p: Vec<Vec<Ex>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| if i == j { one.clone() } else { zero.clone() })
                    .collect()
            })
            .collect();
        let tidy = |e: Ex| {
            if e.is_constant() {
                e.eval()
            } else {
                e.simplify()
            }
        };

        for k in 0..n.saturating_sub(2) {
            // The entries of the column that are not proved zero.  One alone
            // goes below the diagonal as it is (nothing is divided by it);
            // of several, the pivot is one that is non-zero for every value
            // of the symbols, else one proved not identically zero.
            let open: Vec<usize> = ((k + 1)..n).filter(|&i| !proved_zero(&h[i][k])).collect();
            let pivot = match open.as_slice() {
                [] => None,
                [only] => Some(*only),
                several => {
                    let proved = several
                        .iter()
                        .copied()
                        .find(|&i| ex_is_zero(&h[i][k]).is_false())
                        .or_else(|| {
                            several
                                .iter()
                                .copied()
                                .find(|&i| pivot_truth(&h[i][k]).is_false())
                        });
                    match (proved, several.first()) {
                        (Some(i), _) => Some(i),
                        (None, Some(&i)) => {
                            return Err(undecided_pivot("hessenberg", &h[i][k]));
                        }
                        (None, None) => None,
                    }
                }
            };
            let Some(piv) = pivot else {
                for row in h.iter_mut().skip(k + 1) {
                    row[k] = zero.clone();
                }
                continue;
            };
            if piv != k + 1 {
                h.swap(k + 1, piv);
                for row in h.iter_mut() {
                    row.swap(k + 1, piv);
                }
                for row in p.iter_mut() {
                    row.swap(k + 1, piv);
                }
            }
            let pv = h[k + 1][k].clone();
            for j in (k + 2)..n {
                if proved_zero(&h[j][k]) {
                    h[j][k] = zero.clone();
                    continue;
                }
                let f = tidy(&h[j][k] / &pv);
                let pivot_row = h[k + 1].clone();
                for (c, pr) in pivot_row.iter().enumerate() {
                    if c == k {
                        h[j][c] = zero.clone();
                    } else if !pr.is_zero_structural() {
                        let v = tidy(&h[j][c] - &(&f * pr));
                        h[j][c] = v;
                    }
                }
                for r in 0..n {
                    if !h[r][j].is_zero_structural() {
                        let v = tidy(&h[r][k + 1] + &(&f * &h[r][j]));
                        h[r][k + 1] = v;
                    }
                    if !p[r][j].is_zero_structural() {
                        let v = tidy(&p[r][k + 1] + &(&f * &p[r][j]));
                        p[r][k + 1] = v;
                    }
                }
            }
            budget_check(h.iter().flatten().chain(p.iter().flatten()), "hessenberg")?;
        }
        Ok(Hessenberg {
            h: Matrix::from_rows_unchecked(h),
            p: Matrix::from_rows_unchecked(p),
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Calculus helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Build the Jacobian matrix of `funcs` with respect to `vars`:
/// `J[i][j] = ∂funcs[i] / ∂vars[j]`.
///
/// # Errors
///
/// Returns [`SymplexError::InvalidArgument`] if `funcs` or `vars` is empty.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::matrix::jacobian;
///
/// let ctx = Context::new();
/// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
/// let j = jacobian(&[&(&x * &y), &(&x + &y)], &[&x, &y]).unwrap();
/// assert_eq!(j, Matrix::new(vec![vec![y.clone(), x.clone()], vec![ctx.int(1), ctx.int(1)]]).unwrap());
/// assert!(jacobian(&[], &[&x]).is_err());
/// ```
pub fn jacobian(funcs: &[&Ex], vars: &[&Ex]) -> Result<Matrix, SymplexError> {
    if funcs.is_empty() || vars.is_empty() {
        return Err(SymplexError::invalid_argument(
            "jacobian",
            format!(
                "funcs and vars must be non-empty (got {} and {})",
                funcs.len(),
                vars.len()
            ),
        ));
    }
    let rows: Vec<Vec<Ex>> = funcs
        .iter()
        .map(|fi| vars.iter().map(|vj| fi.diff(vj)).collect())
        .collect();
    Ok(Matrix::from_rows_unchecked(rows))
}

impl Matrix {
    /// Differentiate every element with respect to `var`.
    pub fn diff(&self, var: &Ex) -> Matrix {
        self.map(|elem| elem.diff(var))
    }

    /// Integrate every element with respect to `var` (indefinite).
    pub fn integrate(&self, var: &Ex) -> Matrix {
        self.map(|elem| elem.integrate(var))
    }

    /// Substitute `old` → `new` in every element.
    pub fn subs(&self, old: &Ex, new: &Ex) -> Matrix {
        self.map(|elem| elem.subs(old, new))
    }

    /// Evaluate every element (constant-fold where possible).
    pub fn eval(&self) -> Matrix {
        self.map(|elem| elem.eval())
    }

    /// Expand every element (distribute products over sums).
    pub fn expand(&self) -> Matrix {
        self.map(|elem| elem.expand())
    }

    /// Simplify every element via built-in rewrite rules.
    pub fn simplify(&self) -> Matrix {
        self.map(|elem| elem.simplify())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Code generation
// ═══════════════════════════════════════════════════════════════════════════

impl Matrix {
    /// Generate a Rust function that computes this matrix and returns a flat array.
    ///
    /// Uses cross-entry common subexpression elimination for optimal performance.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::matrix::Matrix;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let m = Matrix::new(vec![
    ///     vec![x.sin(), x.cos()],
    ///     vec![-x.cos(), x.sin()],
    /// ]).unwrap();
    /// let code = m.to_rust_fn("rotation", &["x"]).unwrap();
    /// assert!(code.contains("pub fn rotation"));
    /// assert!(code.contains("[f64; 4]"));
    /// ```
    pub fn to_rust_fn(&self, name: &str, params: &[&str]) -> Result<String, SymplexError> {
        self.to_rust_fn_with_options(name, params, &CodegenOptions::default())
    }

    /// Generate a Rust function with custom code generation options.
    ///
    /// See [`CodegenOptions`] for available settings (precision, math backend,
    /// annotations, CSE toggle).
    pub fn to_rust_fn_with_options(
        &self,
        name: &str,
        params: &[&str],
        options: &CodegenOptions,
    ) -> Result<String, SymplexError> {
        let first = &self.rows[0][0];
        let mut guard = first.inner.write();
        let arena = &mut guard.arena;
        let entry_ids: Vec<crate::base::node::ExprId> = self
            .rows
            .iter()
            .flat_map(|row| row.iter().map(|e| e.raw_id()))
            .collect();
        crate::output::codegen::matrix_to_rust_fn(
            arena, &entry_ids, self.nrows, self.ncols, name, params, options,
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// LaTeX rendering
// ═══════════════════════════════════════════════════════════════════════════

impl Matrix {
    /// Render this matrix as a LaTeX `bmatrix`.
    ///
    /// # Example
    /// ```
    /// use symplex::prelude::*;
    /// let ctx = Context::new();
    /// let m = matrix![ctx, [1, 2], [3, 4]];
    /// assert!(m.to_latex().contains(r"\begin{bmatrix}"));
    /// ```
    pub fn to_latex(&self) -> String {
        let mut s = String::from(r"\begin{bmatrix} ");
        for i in 0..self.nrows {
            if i > 0 {
                s.push_str(r" \\ ");
            }
            for (j, entry) in self.rows[i].iter().enumerate().take(self.ncols) {
                if j > 0 {
                    s.push_str(" & ");
                }
                s.push_str(&entry.to_latex());
            }
        }
        s.push_str(r" \end{bmatrix}");
        s
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Decompositions and subspaces
//
// Matrix algorithms naturally use index-based loops for row/column access.
// ═══════════════════════════════════════════════════════════════════════════
#[allow(clippy::needless_range_loop)]
impl Matrix {
    /// LU decomposition with partial pivoting: `P·A = L·U`.
    ///
    /// Returns [`Lu`]`{ l, u, perm }` where `L` is unit lower triangular,
    /// `U` is upper triangular and `perm` is the row permutation (`perm[i]`
    /// is the original index of row `i` of `P·A`).  The pivot of a column
    /// is its first entry proved nonzero by the test of
    /// [`rref`](Self::rref) (for an entry with symbols: not identically
    /// zero, the generic reading).
    ///
    /// # Errors
    ///
    /// - [`SymplexError::InvalidArgument`] if the matrix is not square.
    /// - [`SymplexError::ComputationFailed`] if the matrix is singular, or
    ///   if a column has no entry proved nonzero and one that is not proved
    ///   zero either (the message names it).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let a = matrix![ctx, [4, 3], [6, 3]];
    /// let Lu { l, u, perm } = a.lu().unwrap();
    /// // Rebuild P·A from perm and compare with L·U
    /// let pa = Matrix::new(perm.iter().map(|&i| a.row(i).to_vec()).collect()).unwrap();
    /// assert_eq!((&l * &u).eval(), pa);
    /// assert!(matrix![ctx, [1, 2], [2, 4]].lu().is_err());
    /// ```
    pub fn lu(&self) -> Result<Lu<Matrix>, SymplexError> {
        self.require_square("lu")?;
        let n = self.nrows;
        if let Some(q) = self.as_qmatrix() {
            // Same pivot rule (first non-zero entry), so the same factors.
            let ctx = self.ctx();
            let Lu { l, u, perm } = q.lu().map_err(|e| reop(e, "lu"))?;
            return Ok(Lu {
                l: l.to_matrix(&ctx),
                u: u.to_matrix(&ctx),
                perm,
            });
        }

        let _work = PivotWork::enter();
        let mut perm: Vec<usize> = (0..n).collect();
        let mut u: Vec<Vec<Ex>> = self.rows.clone();
        let one = self.ctx_one();
        let zero = self.ctx_zero();
        let mut l: Vec<Vec<Ex>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| if i == j { one.clone() } else { zero.clone() })
                    .collect()
            })
            .collect();

        // Pivots are decided by the test of `rref` (before 0.31
        // structurally: the zero `1/(√2 − 1) − √2 − 1` was a pivot and `L`
        // had the entry `5/0`): the first entry of the column proved
        // nonzero.  A column whose entries are proved zero is singular; one
        // with an undecided entry and none proved nonzero is refused (up to
        // 0.60.0 the undecided entry was the pivot).
        for k in 0..n {
            let mut pivot_row = None;
            let mut open = None;
            for i in k..n {
                match pivot_truth(&u[i][k]) {
                    Truth::False => {
                        pivot_row = Some(i);
                        break;
                    }
                    Truth::Unknown => {
                        if open.is_none() {
                            open = Some(i);
                        }
                    }
                    Truth::True => u[i][k] = zero.clone(),
                }
            }
            let Some(pivot_row) = pivot_row else {
                return Err(match open {
                    Some(i) => undecided_pivot("lu", &u[i][k]),
                    None => failed("lu", "matrix is singular (zero pivot column)"),
                });
            };

            if pivot_row != k {
                u.swap(k, pivot_row);
                perm.swap(k, pivot_row);
                for j in 0..k {
                    let tmp = l[k][j].clone();
                    l[k][j] = l[pivot_row][j].clone();
                    l[pivot_row][j] = tmp;
                }
            }

            for i in (k + 1)..n {
                if proved_zero(&u[i][k]) {
                    u[i][k] = zero.clone();
                    continue;
                }
                let factor = &u[i][k] / &u[k][k];
                l[i][k] = factor.clone();
                u[i][k] = zero.clone();
                for j in (k + 1)..n {
                    let term = &factor * &u[k][j];
                    u[i][j] = &u[i][j] - &term;
                }
            }
        }

        Ok(Lu {
            l: Matrix::from_rows_unchecked(l),
            u: Matrix::from_rows_unchecked(u),
            perm,
        })
    }

    /// Row-reduced echelon form via Gauss–Jordan elimination.
    ///
    /// Returns `(rref_matrix, pivot_columns)`.  Uses exact arithmetic.
    ///
    /// **Which entries are pivots.**  The matrix is read as one over the
    /// functions of its symbols (the generic reading): an entry is a pivot
    /// candidate iff it is not identically zero.  That is decided by a
    /// three-valued test: exactly for rational functions of the symbols
    /// over `ℚ(radicals of rationals, i)` (since 0.31 also with `√2`, `√3`,
    /// `i`, … — `[[√2 − 3, −2], [3√3 − √6, 2√3]]` had rank 2); for any other
    /// constant by the certified zero test of expressions (what
    /// [`Ex::is_zero`] decides); for an entry with symbols and functions by
    /// a certified nonzero value at a sample point of the symbols (not
    /// identically zero) or by an identity `simplify` proves (zero).  A
    /// column's pivot is its first entry proved nonzero.
    ///
    /// **An entry that is neither proved zero nor proved nonzero** (`Γ(x + 1)
    /// − x·Γ(x)`, which is 0 by an identity the zero test does not know)
    /// is taken for a pivot when its column has no entry proved nonzero:
    /// this function has no way to refuse, so the result is then the one
    /// for a nonzero entry, and the rank may be too large.  The functions
    /// that return a `Result` ([`solve`](Self::solve), [`lu`](Self::lu),
    /// [`inv`](Self::inv), [`eigenvects`](Self::eigenvects), …) refuse
    /// instead.
    ///
    /// The rank is the generic one: `[[x, 1], [1, x]]` has rank 2, although
    /// it has rank 1 at `x = ±1`.  Before 0.30 every structurally non-zero
    /// entry was a pivot, so `[[2t − 2, 3t − 3], [6, 9]]` had rank 2; up to
    /// 0.60.0 every entry the first two tests did not decide was one, so
    /// `[[sin(1)² + cos(1)² − 1]]` and `[[sin²x + cos²x − 1]]` had rank 1.
    pub fn rref(&self) -> (Matrix, Vec<usize>) {
        let (r, pivots, _) = self.rref_generic();
        (r, pivots)
    }

    /// [`rref`](Self::rref) with the first pivot that was taken without a
    /// proof (an entry neither proved zero nor proved nonzero), if any.
    fn rref_generic(&self) -> (Matrix, Vec<usize>, Option<Ex>) {
        // Rational literals: every zero test agrees, so the fraction-free
        // exact core gives the same (unique) RREF and pivots.
        if let Some(q) = self.as_qmatrix() {
            let (r, pivots) = q.rref();
            return (r.to_matrix(&self.ctx()), pivots, None);
        }
        self.rref_core(&mut ClosureOracle::new(&pivot_truth, true))
    }

    /// [`rref`](Self::rref) for a caller that can refuse: an error (under
    /// `operation`) when a pivot was neither proved zero nor proved nonzero.
    pub(crate) fn rref_decided(
        &self,
        operation: &'static str,
    ) -> Result<(Matrix, Vec<usize>), SymplexError> {
        match self.rref_generic() {
            (r, pivots, None) => Ok((r, pivots)),
            (_, _, Some(entry)) => Err(undecided_pivot(operation, &entry)),
        }
    }

    /// Gauss–Jordan elimination with the zero decisions taken by `oracle`,
    /// which is told every row operation (see [`PivotOracle`]).
    ///
    /// The pivot of a column is its first entry (from the pivot row down)
    /// that the oracle proves nonzero.  When there is none and an entry is
    /// undecided, the first undecided entry is the pivot — the generic
    /// reading — and the third component of the result is the first such
    /// entry of the elimination: `None` means every pivot decision was a
    /// proof.  A row is cleared whenever its entry is not proved zero, which
    /// is right whatever the entry is.
    fn rref_core(&self, oracle: &mut dyn PivotOracle) -> (Matrix, Vec<usize>, Option<Ex>) {
        let nrows = self.nrows;
        let ncols = self.ncols;
        let mut rows: Vec<Vec<Ex>> = self.rows.clone();
        let mut pivots = Vec::new();
        let mut pivot_row = 0;
        let mut undecided: Option<Ex> = None;
        // One work budget for the zero tests of the elimination.
        let _work = PivotWork::enter();
        // The public `rref` writes an entry found zero as `0` (it can be
        // zero without being structurally so: `∛(−2) − ∛2·(1/2 + √3·i/2)`).
        let clean = oracle.clean_zero_rows();

        for col in 0..ncols {
            if pivot_row >= nrows {
                break;
            }
            let mut found = None;
            let mut open = None;
            for i in pivot_row..nrows {
                match oracle.is_zero(i, col, &rows[i][col]) {
                    Truth::False => {
                        found = Some(i);
                        break;
                    }
                    Truth::Unknown => {
                        if open.is_none() {
                            open = Some(i);
                        }
                    }
                    Truth::True => {
                        if clean && !rows[i][col].is_zero_structural() {
                            rows[i][col] = rows[i][col].context().zero();
                        }
                    }
                }
            }
            let found = match (found, open) {
                (Some(i), _) => i,
                (None, Some(i)) => {
                    if undecided.is_none() {
                        undecided = Some(rows[i][col].clone());
                    }
                    i
                }
                (None, None) => continue,
            };

            if found != pivot_row {
                rows.swap(pivot_row, found);
                oracle.swap(pivot_row, found);
            }
            let pivot_val = rows[pivot_row][col].clone();
            for j in 0..ncols {
                rows[pivot_row][j] = &rows[pivot_row][j] / &pivot_val;
            }
            oracle.normalize(pivot_row, col);
            for i in 0..nrows {
                if i == pivot_row {
                    continue;
                }
                if !oracle.is_zero(i, col, &rows[i][col]).is_true() {
                    let factor = rows[i][col].clone();
                    for j in 0..ncols {
                        let term = &factor * &rows[pivot_row][j];
                        rows[i][j] = &rows[i][j] - &term;
                    }
                    oracle.eliminate(i, pivot_row, col);
                } else if clean && !rows[i][col].is_zero_structural() {
                    rows[i][col] = rows[i][col].context().zero();
                }
            }
            pivots.push(col);
            pivot_row += 1;
        }
        // The rows below the rank were found zero entry by entry; show them
        // as zeros (an entry can be zero without being structurally so).
        if oracle.clean_zero_rows() {
            for (i, row) in rows.iter_mut().enumerate().skip(pivot_row) {
                for (j, e) in row.iter_mut().enumerate() {
                    if !e.is_zero_structural() && oracle.is_zero(i, j, e).is_true() {
                        *e = e.context().zero();
                    }
                }
            }
        }

        (Matrix::with_shape(rows, nrows, ncols), pivots, undecided)
    }

    /// Rank of the matrix (number of pivot columns in RREF).
    ///
    /// The generic rank of [`rref`](Self::rref): `[[x, 1], [1, x]]` has
    /// rank 2.  An entry that is neither proved zero nor proved nonzero
    /// counts as nonzero (see `rref`; this function cannot refuse), so the
    /// result is then an upper bound of the rank only if that entry does
    /// not vanish.
    pub fn rank(&self) -> usize {
        let (_, pivots) = self.rref();
        pivots.len()
    }

    /// [`rank`](Self::rank) for a caller that can refuse: an error (under
    /// `operation`) when a pivot was neither proved zero nor proved nonzero.
    pub(crate) fn rank_decided(&self, operation: &'static str) -> Result<usize, SymplexError> {
        Ok(self.rref_decided(operation)?.1.len())
    }

    /// Null space (kernel): basis vectors for `Ax = 0`, as column vectors.
    ///
    /// Empty for a full-column-rank matrix.  Uses the pivoting of
    /// [`rref`](Self::rref), with its generic reading: an entry that is
    /// neither proved zero nor proved nonzero counts as nonzero.
    pub fn nullspace(&self) -> Vec<Matrix> {
        self.nullspace_generic().0
    }

    /// [`nullspace`](Self::nullspace) with the first pivot that was taken
    /// without a proof, if any.  The entries are those of the elimination
    /// (not replaced by `0` where they were found zero).
    fn nullspace_generic(&self) -> (Vec<Matrix>, Option<Ex>) {
        if let Some(q) = self.as_qmatrix() {
            let (r, pivots) = q.rref();
            let basis = self.nullspace_from_rref(&r.to_matrix(&self.ctx()), &pivots);
            return (basis, None);
        }
        let (r, pivots, undecided) = self.rref_core(&mut ClosureOracle::new(&pivot_truth, false));
        (self.nullspace_from_rref(&r, &pivots), undecided)
    }

    /// [`nullspace`](Self::nullspace) for a caller that can refuse (the
    /// eigenvector solves): an error (under `operation`) when a pivot was
    /// neither proved zero nor proved nonzero.
    fn nullspace_decided(&self, operation: &'static str) -> Result<Vec<Matrix>, SymplexError> {
        match self.nullspace_generic() {
            (basis, None) => Ok(basis),
            (_, Some(entry)) => Err(undecided_pivot(operation, &entry)),
        }
    }

    /// Null space of `self = A − λI` for a rational `A` (`a`) and an
    /// eigenvalue `λ` with minimal polynomial `g` over ℚ (degree ≥ 2):
    /// the elimination on the expressions is shadowed by the same
    /// elimination on `A − tI` over the number field `ℚ[t]/(g)`, which
    /// takes every zero decision exactly — whatever form `λ` has
    /// (radicals or `RootOf`), and with one minimal polynomial per
    /// eigenvalue instead of one exact zero test per entry.
    ///
    /// The vectors are read off the shadow: each entry is the residue
    /// `r(t)` (degree `< deg g`) evaluated at `λ`, i.e. `c₀ + c₁λ + …`
    /// (SymPy's form).  Before 0.31 they were the expressions of the
    /// elimination on `A − λI`, nested quotients in `λ` in which, for
    /// `λ ≈ −10⁻⁴⁰` (`[[3, −3, 3], [6 + 10⁻³⁹, −6, 6], [5, 4, −3]]`), a
    /// difference cancelled 40 digits and `eval_complex64` failed.
    fn nullspace_in_field(
        &self,
        a: &QMatrix,
        g: &crate::poly::dense::Poly,
        lam: &Ex,
    ) -> Vec<Matrix> {
        use crate::poly::dense::Poly;
        use num_traits::Zero;

        let mut shadow = FieldShadow::new(a, g);
        let n = self.ncols;
        let nr = self.nrows;
        let mut pivots: Vec<usize> = Vec::new();
        let mut pr = 0usize;
        for col in 0..n {
            if pr >= nr {
                break;
            }
            let Some(f) = (pr..nr).find(|&i| !shadow.rows[i][col].is_zero()) else {
                continue;
            };
            shadow.swap(pr, f);
            shadow.normalize(pr, col);
            for i in 0..nr {
                if i != pr && !shadow.rows[i][col].is_zero() {
                    shadow.eliminate(i, pr, col);
                }
            }
            pivots.push(col);
            pr += 1;
        }
        let ctx = self.ctx();
        let lam_pows: Vec<Ex> = (0..g.degree().unwrap_or(1))
            .map(|k| lam.powi(k as i64))
            .collect();
        let to_ex = |r: &Poly| -> Ex {
            let terms: Vec<Ex> = (0..lam_pows.len())
                .filter_map(|k| {
                    let c = r.coeff(k);
                    (!c.is_zero()).then(|| &ctx.from_ratio(c) * &lam_pows[k])
                })
                .collect();
            Ex::sum_of(&ctx, terms).expand().eval()
        };
        let pivot_set: std::collections::HashSet<usize> = pivots.iter().copied().collect();
        let mut basis = Vec::new();
        for free_col in (0..n).filter(|c| !pivot_set.contains(c)) {
            let mut entries = vec![self.ctx_zero(); n];
            entries[free_col] = self.ctx_one();
            for (pivot_idx, &pivot_col) in pivots.iter().enumerate() {
                entries[pivot_col] = to_ex(&shadow.rows[pivot_idx][free_col].neg());
            }
            basis.push(Matrix::col_vector_unchecked(entries));
        }
        basis
    }

    fn nullspace_from_rref(&self, rref_mat: &Matrix, pivots: &[usize]) -> Vec<Matrix> {
        let n = self.ncols;

        let pivot_set: std::collections::HashSet<usize> = pivots.iter().copied().collect();
        let free_vars: Vec<usize> = (0..n).filter(|c| !pivot_set.contains(c)).collect();

        let mut basis = Vec::new();
        for &free_col in &free_vars {
            let mut entries = vec![self.ctx_zero(); n];
            entries[free_col] = self.ctx_one();
            for (pivot_idx, &pivot_col) in pivots.iter().enumerate() {
                entries[pivot_col] = -&rref_mat.rows[pivot_idx][free_col];
            }
            // `n` = `self.ncols` ≥ 1 (a matrix has a column).
            basis.push(Matrix::col_vector_unchecked(entries));
        }
        basis
    }

    /// Column space basis: the pivot columns of the original matrix (the
    /// pivots of [`rref`](Self::rref), with its generic reading).
    pub fn columnspace(&self) -> Vec<Matrix> {
        let (_, pivots) = self.rref();
        pivots
            .iter()
            .map(|&col| Matrix::col_vector_unchecked(self.col(col)))
            .collect()
    }

    /// Row space basis: the non-zero rows of the RREF, as row vectors (the
    /// rows of [`rref`](Self::rref), with its generic reading).
    pub fn rowspace(&self) -> Vec<Matrix> {
        let (rref_mat, pivots) = self.rref();
        (0..pivots.len())
            .map(|i| Matrix::row_vector_unchecked(rref_mat.rows[i].clone()))
            .collect()
    }

    /// Left null space: basis of `{ y : yᵀA = 0 }` = `nullspace(Aᵀ)`.
    pub fn left_nullspace(&self) -> Vec<Matrix> {
        self.transpose().nullspace()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Utilities
// ═══════════════════════════════════════════════════════════════════════════

/// `|e|²` as a term of the Frobenius norm: `e·e` for an `e` that is
/// provably real (the form real matrices always had), `re² + im²` when
/// [`decompose`](crate::base::complex::decompose) determines both parts in
/// real terms (`|1 + 2i|² = 5`), and `abs(e)²` otherwise (SymPy's
/// `Abs(a)**2` for a symbol `a` that may be complex).  A factor `e^z`
/// contributes `e^(2·Re z)` (`|e^z| = e^(Re z)`) rather than its cosine and
/// sine: before 0.35 the norm of `[[exp(i)]]` was `√(sin²1 + cos²1)`
/// (SymPy: `1`).
fn abs_squared_entry(e: &Ex) -> Ex {
    if Truth::from(e.is_real()).is_true() {
        return e * e;
    }
    if let Some(sq) = abs_squared_through_exp(e) {
        return sq;
    }
    let id = {
        let mut inner = e.inner.write();
        let arena = &mut inner.arena;
        let parts = crate::base::complex::decompose(arena, e.raw_id());
        if arena.is_zero_structural(parts.im) {
            None
        } else {
            let two = arena.int(2);
            Some(if parts.exact {
                let re2 = arena.pow(parts.re, two);
                let im2 = arena.pow(parts.im, two);
                arena.add(&[re2, im2])
            } else {
                let abs = arena.abs(e.raw_id());
                arena.pow(abs, two)
            })
        }
    };
    id.map_or_else(|| e * e, |id| e.wrap(id))
}

/// `|e|²` for an entry `c·e^z` (`e^z` alone, or a product with such
/// factors) whose exponents have an exact real part: `|c|²·e^(2·Re z)`,
/// with `|c|²` by [`abs_squared_entry`].  `None` when there is no such
/// factor.
fn abs_squared_through_exp(e: &Ex) -> Option<Ex> {
    use crate::base::node::{ExprId, ExprNode};
    let (exp_re, rest, rest_is_one) = {
        let mut inner = e.inner.write();
        let arena = &mut inner.arena;
        let factors: Vec<ExprId> = match arena.node(e.raw_id()) {
            ExprNode::Mul(children) => children.to_vec(),
            _ => vec![e.raw_id()],
        };
        let mut res = Vec::new();
        let mut others = Vec::new();
        for f in factors {
            match *arena.node(f) {
                ExprNode::Exp(z) => {
                    let parts = crate::base::complex::decompose(arena, z);
                    if parts.exact {
                        res.push(parts.re);
                    } else {
                        others.push(f);
                    }
                }
                _ => others.push(f),
            }
        }
        if res.is_empty() {
            return None;
        }
        let re = arena.add(&res);
        let two = arena.int(2);
        let twice = arena.mul(&[two, re]);
        let exp = arena.exp(twice);
        let rest = arena.mul(&others);
        (exp, rest, rest == arena.one)
    };
    let exp_re = e.wrap(exp_re);
    if rest_is_one {
        return Some(exp_re);
    }
    Some(&abs_squared_entry(&e.wrap(rest)) * &exp_re)
}

impl Matrix {
    /// Frobenius norm `‖A‖_F = √(Σ |a_ij|²)` (for a row or column vector,
    /// its Euclidean 2-norm).  SymPy: `Matrix.norm()`, `sqrt(Add(*(abs(i)
    /// ** 2 for i in vals)))`.
    ///
    /// Each `|a|²` is `a²` for an entry that is provably real, `re² + im²`
    /// when the real and imaginary parts of `a` are known (`|1 + 2i|² = 5`,
    /// `|x + i·y|² = x² + y²` for real `x`, `y`), and `abs(a)²` otherwise —
    /// SymPy's form for a symbol that may be complex, `[[a, b]]` ↦
    /// `sqrt(Abs(a)**2 + Abs(b)**2)`.  Before 0.33 every entry was squared:
    /// `[[i, 0], [0, 2]]` had the norm `√3` (it is `√5`) and `[[a, b]]` the
    /// norm `√(a² + b²)`, which is not a norm for complex `a`, `b`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let i = ctx.i_unit();
    /// // SymPy: Matrix([[I, 0], [0, 2]]).norm() == sqrt(5)
    /// let m = Matrix::new(vec![vec![i.clone(), ctx.int(0)], vec![ctx.int(0), ctx.int(2)]]).unwrap();
    /// assert_eq!(m.norm(), ctx.int(5).sqrt());
    /// assert_eq!(matrix![ctx, [3, 4]].norm(), ctx.int(5));
    /// // SymPy: Matrix([[a, b]]).norm() == sqrt(Abs(a)**2 + Abs(b)**2)
    /// let (a, b) = (ctx.symbol("a"), ctx.symbol("b"));
    /// let v = Matrix::new(vec![vec![a.clone(), b.clone()]]).unwrap();
    /// assert_eq!(v.norm(), (&a.abs().powi(2) + &b.abs().powi(2)).sqrt());
    /// // ... and sqrt(a**2 + b**2) for real a, b
    /// let x = ctx.symbol_with("x", &[Assumption::Real]).unwrap();
    /// let y = ctx.symbol_with("y", &[Assumption::Real]).unwrap();
    /// let v = Matrix::new(vec![vec![x.clone(), y.clone()]]).unwrap();
    /// assert_eq!(v.norm(), (&x.powi(2) + &y.powi(2)).sqrt());
    /// ```
    pub fn norm_frobenius(&self) -> Ex {
        let mut sum = self.ctx_zero();
        for elem in self.iter() {
            sum += &abs_squared_entry(elem);
        }
        sum.sqrt()
    }

    /// Default norm — the Frobenius norm ([`norm_frobenius`](Self::norm_frobenius)),
    /// as SymPy's `Matrix.norm()` (`ord=None`).
    pub fn norm(&self) -> Ex {
        self.norm_frobenius()
    }

    /// Stack matrices horizontally (side by side).
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::InvalidArgument`] if `matrices` is empty
    /// or row counts differ.
    pub fn hstack(matrices: &[&Matrix]) -> Result<Matrix, SymplexError> {
        if matrices.is_empty() {
            return Err(invalid("hstack", "need at least one matrix"));
        }
        let nrows = matrices[0].nrows;
        for (idx, m) in matrices.iter().enumerate() {
            if m.nrows != nrows {
                return Err(invalid(
                    "hstack",
                    format!("matrix {idx} has {} rows, expected {nrows}", m.nrows),
                ));
            }
        }
        let ncols: usize = matrices.iter().map(|m| m.ncols).sum();
        let rows: Vec<Vec<Ex>> = (0..nrows)
            .map(|i| {
                matrices
                    .iter()
                    .flat_map(|m| m.rows[i].iter().cloned())
                    .collect()
            })
            .collect();
        Ok(Matrix::with_shape(rows, nrows, ncols))
    }

    /// Stack matrices vertically (on top of each other).
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::InvalidArgument`] if `matrices` is empty
    /// or column counts differ.
    pub fn vstack(matrices: &[&Matrix]) -> Result<Matrix, SymplexError> {
        if matrices.is_empty() {
            return Err(invalid("vstack", "need at least one matrix"));
        }
        let ncols = matrices[0].ncols;
        for (idx, m) in matrices.iter().enumerate() {
            if m.ncols != ncols {
                return Err(invalid(
                    "vstack",
                    format!("matrix {idx} has {} cols, expected {ncols}", m.ncols),
                ));
            }
        }
        let nrows: usize = matrices.iter().map(|m| m.nrows).sum();
        let rows: Vec<Vec<Ex>> = matrices
            .iter()
            .flat_map(|m| m.rows.iter().cloned())
            .collect();
        Ok(Matrix::with_shape(rows, nrows, ncols))
    }

    /// Vectorization `vec(A)`: stack the columns into a single `(mn)×1`
    /// column vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let a = matrix![ctx, [1, 2], [3, 4]];
    /// assert_eq!(a.vec(), matrix![ctx, [1], [3], [2], [4]]);
    /// ```
    pub fn vec(&self) -> Matrix {
        let mut entries = Vec::with_capacity(self.nrows * self.ncols);
        for j in 0..self.ncols {
            for row in self.rows.iter().take(self.nrows) {
                entries.push(row[j].clone());
            }
        }
        // `nrows · ncols` ≥ 1 entries.
        Matrix::col_vector_unchecked(entries)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Selection, deletion and numeric conversion (0.3 ergonomics)
// ═══════════════════════════════════════════════════════════════════════════

impl Matrix {
    /// Check that every index in `idx` is below `bound`, naming the axis
    /// in the error.
    fn check_indices(
        operation: &'static str,
        axis: &str,
        idx: &[usize],
        bound: usize,
    ) -> Result<(), SymplexError> {
        if idx.is_empty() {
            return Err(invalid(
                operation,
                format!("{axis} selection must contain at least one index"),
            ));
        }
        if let Some(&bad) = idx.iter().find(|&&k| k >= bound) {
            return Err(invalid(
                operation,
                format!("{axis} index {bad} out of range for {bound} {axis}s"),
            ));
        }
        Ok(())
    }

    /// The sub-matrix formed by the given rows and columns, in the order
    /// listed.  Indices may be repeated or reordered.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if either list is empty or contains
    /// an out-of-range index.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let m = matrix![ctx, [1, 2, 3], [4, 5, 6], [7, 8, 9]];
    /// assert_eq!(m.extract(&[2, 0], &[0, 2, 2]).unwrap(), matrix![ctx, [7, 9, 9], [1, 3, 3]]);
    /// assert!(m.extract(&[3], &[0]).is_err());
    /// assert!(m.extract(&[], &[0]).is_err());
    /// ```
    pub fn extract(&self, rows: &[usize], cols: &[usize]) -> Result<Matrix, SymplexError> {
        Self::check_indices("extract", "row", rows, self.nrows)?;
        Self::check_indices("extract", "column", cols, self.ncols)?;
        let data: Vec<Vec<Ex>> = rows
            .iter()
            .map(|&i| cols.iter().map(|&j| self.rows[i][j].clone()).collect())
            .collect();
        Ok(Matrix::from_rows_unchecked(data))
    }

    /// The rows with the given indices (all columns), in the order listed.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `rows` is empty or contains an
    /// out-of-range index.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let m = matrix![ctx, [1, 2], [3, 4], [5, 6]];
    /// assert_eq!(m.select_rows(&[2, 0]).unwrap(), matrix![ctx, [5, 6], [1, 2]]);
    /// ```
    pub fn select_rows(&self, rows: &[usize]) -> Result<Matrix, SymplexError> {
        Self::check_indices("select_rows", "row", rows, self.nrows)?;
        let data: Vec<Vec<Ex>> = rows.iter().map(|&i| self.rows[i].clone()).collect();
        Ok(Matrix::from_rows_unchecked(data))
    }

    /// The columns with the given indices (all rows), in the order listed.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `cols` is empty or contains an
    /// out-of-range index.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let m = matrix![ctx, [1, 2, 3], [4, 5, 6]];
    /// assert_eq!(m.select_cols(&[2, 2]).unwrap(), matrix![ctx, [3, 3], [6, 6]]);
    /// ```
    pub fn select_cols(&self, cols: &[usize]) -> Result<Matrix, SymplexError> {
        Self::check_indices("select_cols", "column", cols, self.ncols)?;
        let data: Vec<Vec<Ex>> = self
            .rows
            .iter()
            .map(|r| cols.iter().map(|&j| r[j].clone()).collect())
            .collect();
        Ok(Matrix::from_rows_unchecked(data))
    }

    /// The matrix with row `i` removed.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `i` is out of range or the
    /// matrix has a single row (the result would be empty).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let m = matrix![ctx, [1, 2], [3, 4], [5, 6]];
    /// assert_eq!(m.delete_row(1).unwrap(), matrix![ctx, [1, 2], [5, 6]]);
    /// assert!(matrix![ctx, [1, 2]].delete_row(0).is_err());
    /// ```
    pub fn delete_row(&self, i: usize) -> Result<Matrix, SymplexError> {
        if i >= self.nrows {
            return Err(invalid(
                "delete_row",
                format!("row index {i} out of range for {} rows", self.nrows),
            ));
        }
        if self.nrows == 1 {
            return Err(invalid(
                "delete_row",
                "cannot delete the only row of a matrix",
            ));
        }
        let data: Vec<Vec<Ex>> = self
            .rows
            .iter()
            .enumerate()
            .filter(|&(k, _)| k != i)
            .map(|(_, r)| r.clone())
            .collect();
        Ok(Matrix::from_rows_unchecked(data))
    }

    /// The matrix with column `j` removed.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `j` is out of range or the
    /// matrix has a single column (the result would be empty).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let m = matrix![ctx, [1, 2, 3], [4, 5, 6]];
    /// assert_eq!(m.delete_col(0).unwrap(), matrix![ctx, [2, 3], [5, 6]]);
    /// assert!(matrix![ctx, [1], [2]].delete_col(0).is_err());
    /// ```
    pub fn delete_col(&self, j: usize) -> Result<Matrix, SymplexError> {
        if j >= self.ncols {
            return Err(invalid(
                "delete_col",
                format!("column index {j} out of range for {} columns", self.ncols),
            ));
        }
        if self.ncols == 1 {
            return Err(invalid(
                "delete_col",
                "cannot delete the only column of a matrix",
            ));
        }
        let data: Vec<Vec<Ex>> = self
            .rows
            .iter()
            .map(|r| {
                r.iter()
                    .enumerate()
                    .filter(|&(k, _)| k != j)
                    .map(|(_, e)| e.clone())
                    .collect()
            })
            .collect();
        Ok(Matrix::from_rows_unchecked(data))
    }

    /// The matrix with row `i` removed.  SymPy name for
    /// [`delete_row`](Self::delete_row).
    ///
    /// # Errors
    ///
    /// Same as [`delete_row`](Self::delete_row).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let m = matrix![ctx, [1, 2], [3, 4], [5, 6]];
    /// assert_eq!(m.row_del(0).unwrap(), matrix![ctx, [3, 4], [5, 6]]);
    /// assert!(m.row_del(3).is_err());
    /// ```
    pub fn row_del(&self, i: usize) -> Result<Matrix, SymplexError> {
        self.delete_row(i).map_err(|e| reop(e, "row_del"))
    }

    /// The matrix with column `j` removed.  SymPy name for
    /// [`delete_col`](Self::delete_col).
    ///
    /// # Errors
    ///
    /// Same as [`delete_col`](Self::delete_col).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let m = matrix![ctx, [1, 2, 3], [4, 5, 6]];
    /// assert_eq!(m.col_del(2).unwrap(), matrix![ctx, [1, 2], [4, 5]]);
    /// assert!(m.col_del(3).is_err());
    /// ```
    pub fn col_del(&self, j: usize) -> Result<Matrix, SymplexError> {
        self.delete_col(j).map_err(|e| reop(e, "col_del"))
    }

    /// Insert the rows of `rows` before row `pos` (`pos == nrows` appends).
    /// SymPy: `Matrix.row_insert(pos, other)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `pos > nrows` or `rows` has a
    /// different number of columns.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let m = matrix![ctx, [1, 2], [3, 4]];
    /// let r = matrix![ctx, [5, 6]];
    /// assert_eq!(m.row_insert(1, &r).unwrap(), matrix![ctx, [1, 2], [5, 6], [3, 4]]);
    /// assert_eq!(m.row_insert(2, &r).unwrap(), matrix![ctx, [1, 2], [3, 4], [5, 6]]);
    /// assert!(m.row_insert(3, &r).is_err());
    /// assert!(m.row_insert(0, &matrix![ctx, [5, 6, 7]]).is_err());
    /// ```
    pub fn row_insert(&self, pos: usize, rows: &Matrix) -> Result<Matrix, SymplexError> {
        if pos > self.nrows {
            return Err(invalid(
                "row_insert",
                format!(
                    "position {pos} out of range for {} rows (use {} to append)",
                    self.nrows, self.nrows
                ),
            ));
        }
        if rows.ncols != self.ncols {
            return Err(invalid(
                "row_insert",
                format!(
                    "inserted rows have {} columns, expected {}",
                    rows.ncols, self.ncols
                ),
            ));
        }
        let mut data: Vec<Vec<Ex>> = Vec::with_capacity(self.nrows + rows.nrows);
        data.extend(self.rows[..pos].iter().cloned());
        data.extend(rows.rows.iter().cloned());
        data.extend(self.rows[pos..].iter().cloned());
        Ok(Matrix::with_shape(
            data,
            self.nrows + rows.nrows,
            self.ncols,
        ))
    }

    /// Insert the columns of `cols` before column `pos` (`pos == ncols`
    /// appends).  SymPy: `Matrix.col_insert(pos, other)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `pos > ncols` or `cols` has a
    /// different number of rows.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let m = matrix![ctx, [1, 2], [3, 4]];
    /// let c = matrix![ctx, [5], [6]];
    /// assert_eq!(m.col_insert(1, &c).unwrap(), matrix![ctx, [1, 5, 2], [3, 6, 4]]);
    /// assert!(m.col_insert(0, &matrix![ctx, [5]]).is_err());
    /// ```
    pub fn col_insert(&self, pos: usize, cols: &Matrix) -> Result<Matrix, SymplexError> {
        if pos > self.ncols {
            return Err(invalid(
                "col_insert",
                format!(
                    "position {pos} out of range for {} columns (use {} to append)",
                    self.ncols, self.ncols
                ),
            ));
        }
        if cols.nrows != self.nrows {
            return Err(invalid(
                "col_insert",
                format!(
                    "inserted columns have {} rows, expected {}",
                    cols.nrows, self.nrows
                ),
            ));
        }
        let ncols = self.ncols + cols.ncols;
        let data: Vec<Vec<Ex>> = self
            .rows
            .iter()
            .zip(&cols.rows)
            .map(|(r, c)| {
                let mut row = Vec::with_capacity(ncols);
                row.extend(r[..pos].iter().cloned());
                row.extend(c.iter().cloned());
                row.extend(r[pos..].iter().cloned());
                row
            })
            .collect();
        Ok(Matrix::with_shape(data, self.nrows, ncols))
    }

    /// Check that `perm` is a permutation of `0..bound`.
    fn check_permutation(
        operation: &'static str,
        axis: &str,
        perm: &[usize],
        bound: usize,
    ) -> Result<(), SymplexError> {
        if perm.len() != bound {
            return Err(invalid(
                operation,
                format!(
                    "permutation has {} entries, expected one per {axis} ({bound})",
                    perm.len()
                ),
            ));
        }
        let mut seen = vec![false; bound];
        for &k in perm {
            if k >= bound {
                return Err(invalid(
                    operation,
                    format!("{axis} index {k} out of range for {bound} {axis}s"),
                ));
            }
            if seen[k] {
                return Err(invalid(
                    operation,
                    format!("{axis} index {k} appears twice; not a permutation"),
                ));
            }
            seen[k] = true;
        }
        Ok(())
    }

    /// Reorder the rows: row `i` of the result is row `perm[i]` of `self`.
    /// SymPy: `Matrix.permute_rows(perm)` for a permutation given in array
    /// form.
    ///
    /// Unlike [`select_rows`](Self::select_rows), `perm` must be a genuine
    /// permutation of `0..nrows`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `perm` is not a permutation of
    /// `0..nrows` (wrong length, out-of-range or repeated index).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let m = matrix![ctx, [1], [2], [3]];
    /// assert_eq!(m.permute_rows(&[2, 0, 1]).unwrap(), matrix![ctx, [3], [1], [2]]);
    /// assert!(m.permute_rows(&[0, 0, 1]).is_err());
    /// ```
    pub fn permute_rows(&self, perm: &[usize]) -> Result<Matrix, SymplexError> {
        Self::check_permutation("permute_rows", "row", perm, self.nrows)?;
        self.select_rows(perm).map_err(|e| reop(e, "permute_rows"))
    }

    /// Reorder the columns: column `j` of the result is column `perm[j]`
    /// of `self`.  SymPy: `Matrix.permute_cols(perm)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `perm` is not a permutation of
    /// `0..ncols`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let m = matrix![ctx, [1, 2, 3]];
    /// assert_eq!(m.permute_cols(&[2, 0, 1]).unwrap(), matrix![ctx, [3, 1, 2]]);
    /// assert!(m.permute_cols(&[0, 1]).is_err());
    /// ```
    pub fn permute_cols(&self, perm: &[usize]) -> Result<Matrix, SymplexError> {
        Self::check_permutation("permute_cols", "column", perm, self.ncols)?;
        self.select_cols(perm).map_err(|e| reop(e, "permute_cols"))
    }

    /// Is every entry an integer literal?  Three-valued: `Some(true)` when
    /// every entry is an integer literal, `Some(false)` when some entry is a
    /// non-integer *numeric* literal (`1/2`), `None` when some entry is
    /// symbolic (a symbol, `pi`, an unevaluated sum, …).
    ///
    /// Entries are inspected as they are — call [`eval`](Self::eval) first
    /// to constant-fold `2 + 3` into `5`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// assert_eq!(matrix![ctx, [1, -2], [3, 0]].is_integer_matrix(), Some(true));
    /// let half = Matrix::new(vec![vec![ctx.rational(1, 2)]]).unwrap();
    /// assert_eq!(half.is_integer_matrix(), Some(false));
    /// let sym = Matrix::new(vec![vec![ctx.symbol("x")]]).unwrap();
    /// assert_eq!(sym.is_integer_matrix(), None);
    /// ```
    pub fn is_integer_matrix(&self) -> Option<bool> {
        let mut unknown = false;
        for e in self.iter() {
            match e.as_rational() {
                Some(r) if r.is_integer() => {}
                Some(_) => return Some(false),
                None => unknown = true,
            }
        }
        if unknown { None } else { Some(true) }
    }

    /// The entries as exact rationals, row-major.
    ///
    /// Returns `None` if any entry is not a numeric literal.  Entries are
    /// inspected as they are — call [`eval`](Self::eval) first to fold
    /// constant expressions such as `1/2 + 1/3`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use num_bigint::BigInt;
    /// use num_rational::Ratio;
    ///
    /// let ctx = Context::new();
    /// let m = Matrix::new(vec![vec![ctx.rational(1, 2), ctx.int(3)]]).unwrap();
    /// let rows = m.to_rational_rows().unwrap();
    /// assert_eq!(rows[0][0], Ratio::new(BigInt::from(1), BigInt::from(2)));
    /// assert_eq!(rows[0][1], Ratio::from_integer(BigInt::from(3)));
    /// assert!(Matrix::new(vec![vec![ctx.symbol("x")]]).unwrap().to_rational_rows().is_none());
    /// ```
    pub fn to_rational_rows(&self) -> Option<Vec<Vec<Q>>> {
        self.rows
            .iter()
            .map(|r| r.iter().map(Ex::as_rational).collect())
            .collect()
    }

    /// The entries as big integers, row-major.
    ///
    /// Returns `None` if any entry is not an integer literal (a fraction,
    /// a symbol, an unevaluated expression, …).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use num_bigint::BigInt;
    ///
    /// let ctx = Context::new();
    /// let rows = matrix![ctx, [1, -2], [3, 4]].to_bigint_rows().unwrap();
    /// assert_eq!(rows[0][1], BigInt::from(-2));
    /// let half = Matrix::new(vec![vec![ctx.rational(1, 2)]]).unwrap();
    /// assert!(half.to_bigint_rows().is_none());
    /// ```
    pub fn to_bigint_rows(&self) -> Option<Vec<Vec<BigInt>>> {
        self.rows
            .iter()
            .map(|r| r.iter().map(Ex::as_bigint).collect())
            .collect()
    }

    /// Create a matrix of exact rational literals.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] for empty or jagged input.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use num_bigint::BigInt;
    /// use num_rational::Ratio;
    ///
    /// let ctx = Context::new();
    /// let q = |n: i64, d: i64| Ratio::new(BigInt::from(n), BigInt::from(d));
    /// let m = Matrix::from_ratio(&ctx, &[vec![q(1, 2), q(3, 1)]]).unwrap();
    /// assert_eq!(m.get(0, 0), &ctx.rational(1, 2));
    /// assert_eq!(m.to_rational_rows().unwrap(), vec![vec![q(1, 2), q(3, 1)]]);
    /// ```
    pub fn from_ratio(ctx: &Context, rows: &[Vec<Q>]) -> Result<Matrix, SymplexError> {
        let data: Vec<Vec<Ex>> = rows
            .iter()
            .map(|r| r.iter().map(|q| ctx.from_ratio(q.clone())).collect())
            .collect();
        Matrix::new(data)
    }

    /// Create a matrix of integer literals from big integers.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] for empty or jagged input.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use num_bigint::BigInt;
    ///
    /// let ctx = Context::new();
    /// let rows = vec![vec![BigInt::from(1), BigInt::from(2)], vec![BigInt::from(3), BigInt::from(4)]];
    /// let m = Matrix::from_bigint(&ctx, &rows).unwrap();
    /// assert_eq!(m, matrix![ctx, [1, 2], [3, 4]]);
    /// ```
    pub fn from_bigint(ctx: &Context, rows: &[Vec<BigInt>]) -> Result<Matrix, SymplexError> {
        let data: Vec<Vec<Ex>> = rows
            .iter()
            .map(|r| r.iter().map(|n| ctx.from_bigint(n.clone())).collect())
            .collect();
        Matrix::new(data)
    }

    /// Create a matrix from `f64` values, converting each **exactly** to
    /// the dyadic rational it represents (via [`Context::from_f64`]): `0.5`
    /// becomes `1/2`, but `0.1` becomes `3602879701896397/36028797018963968`,
    /// not `1/10`.  Use [`Context::from_f64_nice`] entry-wise if you want
    /// the "human" rational reading of a float.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] for empty or jagged input or a
    /// `NaN` entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let m = Matrix::from_f64_rows(&ctx, &[vec![0.5, -3.0], vec![0.25, 2.0]]).unwrap();
    /// assert_eq!(m.get(0, 0), &ctx.rational(1, 2));
    /// assert_eq!(m.get(1, 0), &ctx.rational(1, 4));
    /// assert!(Matrix::from_f64_rows(&ctx, &[vec![f64::NAN]]).is_err());
    /// ```
    pub fn from_f64_rows(ctx: &Context, rows: &[Vec<f64>]) -> Result<Matrix, SymplexError> {
        let mut data: Vec<Vec<Ex>> = Vec::with_capacity(rows.len());
        for r in rows {
            let mut out = Vec::with_capacity(r.len());
            for &v in r {
                out.push(ctx.from_f64(v).map_err(|e| reop(e, "from_f64_rows"))?);
            }
            data.push(out);
        }
        Matrix::new(data)
    }

    /// Simultaneous substitution of several `(old, new)` pairs in every
    /// entry (see [`Ex::subs_map`]).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// let m = Matrix::new(vec![vec![x.clone(), y.clone()]]).unwrap();
    /// // Swap x and y in one step — sequential `subs` would collapse both to y.
    /// let swapped = m.subs_map(&[(&x, &y), (&y, &x)]);
    /// assert_eq!(swapped, Matrix::new(vec![vec![y, x]]).unwrap());
    /// ```
    pub fn subs_map(&self, replacements: &[(&Ex, &Ex)]) -> Matrix {
        self.map(|elem| elem.subs_map(replacements))
    }

    /// Number of structurally non-zero entries (entries that are not the
    /// literal `0`).  Symbolic entries count as non-zero even if they
    /// would simplify to zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// assert_eq!(matrix![ctx, [1, 0, 2], [0, 0, 3]].nnz(), 3);
    /// assert_eq!(Matrix::identity(&ctx, 4).unwrap().nnz(), 4);
    /// ```
    pub fn nnz(&self) -> usize {
        self.iter().filter(|e| !e.is_zero_structural()).count()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Integer normal forms (delegating to `normalforms`)
// ═══════════════════════════════════════════════════════════════════════════

impl Matrix {
    /// Row-style Hermite normal form `H = U·A` of an integer matrix.
    ///
    /// See [`normalforms::hermite_normal_form`](crate::normalforms::hermite_normal_form)
    /// for the exact normalisation and
    /// [`normalforms::hermite_normal_form_with_transform`](crate::normalforms::hermite_normal_form_with_transform)
    /// to obtain `U` as well.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if any entry is not an integer
    /// literal.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let a = matrix![ctx, [2, 4, 4], [-6, 6, 12], [10, -4, -16]];
    /// let h = a.hermite_normal_form().unwrap();
    /// assert_eq!(h, matrix![ctx, [2, 4, 4], [0, 6, 0], [0, 0, 12]]);
    /// ```
    pub fn hermite_normal_form(&self) -> Result<Matrix, SymplexError> {
        crate::domains::normalforms::hermite_normal_form(self)
    }

    /// Smith normal form `S = U·A·V` of an integer matrix: a diagonal
    /// matrix `diag(d₁, …, dᵣ, 0, …)` with `dᵢ > 0` and `dᵢ | dᵢ₊₁`.
    ///
    /// See [`normalforms::smith_normal_form`](crate::normalforms::smith_normal_form)
    /// and
    /// [`normalforms::smith_normal_form_with_transforms`](crate::normalforms::smith_normal_form_with_transforms).
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if any entry is not an integer
    /// literal.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let a = matrix![ctx, [2, 4, 4], [-6, 6, 12], [10, -4, -16]];
    /// assert_eq!(a.smith_normal_form().unwrap(), matrix![ctx, [2, 0, 0], [0, 6, 0], [0, 0, 12]]);
    /// ```
    pub fn smith_normal_form(&self) -> Result<Matrix, SymplexError> {
        crate::domains::normalforms::smith_normal_form(self)
    }

    /// A ℤ-basis of the integer kernel `{x ∈ ℤⁿ : A·x = 0}`, as column
    /// vectors (empty when the kernel is trivial).
    ///
    /// See [`normalforms::integer_nullspace`](crate::normalforms::integer_nullspace).
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if any entry is not an integer
    /// literal.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let a = matrix![ctx, [2, 4, 6]];
    /// let basis = a.integer_nullspace().unwrap();
    /// assert_eq!(basis.len(), 2);
    /// for k in &basis {
    ///     assert_eq!((&a * k).eval(), matrix![ctx, [0]]);
    /// }
    /// ```
    pub fn integer_nullspace(&self) -> Result<Vec<Matrix>, SymplexError> {
        crate::domains::normalforms::integer_nullspace(self)
    }

    /// Inverse modulo `m` of an integer matrix: entries in `[0, m)` with
    /// `A·A⁻¹ ≡ I (mod m)`, computed as `adj(A)·det(A)⁻¹ mod m` (see
    /// [`ZMatrix::inv_mod`]).  SymPy: `Matrix.inv_mod(m)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if the matrix is not square, an
    /// entry is not an integer literal, `m < 2`, or `gcd(det A, m) ≠ 1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let a = matrix![ctx, [1, 2], [3, 4]];
    /// assert_eq!(a.inv_mod(5).unwrap(), matrix![ctx, [3, 1], [4, 2]]);
    /// assert!(a.inv_mod(2).is_err());   // det = −2
    /// ```
    pub fn inv_mod(&self, m: u64) -> Result<Matrix, SymplexError> {
        let z = ZMatrix::try_from(self).map_err(|e| reop(e, "inv_mod"))?;
        let r = z
            .inv_mod(&BigInt::from(m))
            .map_err(|e| reop(e, "inv_mod"))?;
        Ok(r.to_matrix(&self.ctx()))
    }

    /// LLL-reduced basis of the lattice spanned by the rows of an integer
    /// matrix, Lovász parameter `δ` (a [`Rational64`]).  See
    /// [`ZMatrix::lll`] for the guarantees and
    /// [`normalforms::lll_with_transform`](crate::normalforms::lll_with_transform)
    /// for the unimodular transform.  SymPy: `Matrix.lll(delta)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if an entry is not an integer
    /// literal, `δ ∉ (1/4, 1)`, or the rows are linearly dependent.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::num_rational::Ratio;
    ///
    /// let ctx = Context::new();
    /// let b = matrix![ctx, [1, 1, 1], [-1, 0, 2], [3, 5, 6]];
    /// assert_eq!(b.lll(Ratio::new(3, 4)).unwrap(), matrix![ctx, [0, 1, 0], [1, 0, 1], [-1, 0, 2]]);
    /// ```
    pub fn lll(&self, delta: Rational64) -> Result<Matrix, SymplexError> {
        crate::domains::normalforms::lll(self, delta)
    }

    /// [`lll`](Self::lll) with the standard `δ = 3/4`.
    ///
    /// # Errors
    ///
    /// Same as [`lll`](Self::lll).
    pub fn lll_default(&self) -> Result<Matrix, SymplexError> {
        crate::domains::normalforms::lll(self, LLL_DEFAULT_DELTA)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Free-standing vector operations
// ═══════════════════════════════════════════════════════════════════════════

/// Cross product of two 3×1 column vectors.
///
/// # Errors
///
/// Returns [`SymplexError::InvalidArgument`] if either argument is not a
/// 3×1 matrix.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::matrix::cross;
///
/// let ctx = Context::new();
/// let (i, j) = (matrix![ctx, [1], [0], [0]], matrix![ctx, [0], [1], [0]]);
/// assert_eq!(cross(&i, &j).unwrap(), matrix![ctx, [0], [0], [1]]);
/// assert!(cross(&i, &matrix![ctx, [0], [1]]).is_err());
/// ```
pub fn cross(a: &Matrix, b: &Matrix) -> Result<Matrix, SymplexError> {
    for (which, v) in [("first", a), ("second", b)] {
        if v.shape() != (3, 1) {
            return Err(SymplexError::invalid_argument(
                "cross",
                format!(
                    "{which} argument must be a 3×1 column vector, got {}×{}",
                    v.nrows(),
                    v.ncols()
                ),
            ));
        }
    }
    let (a0, a1, a2) = (a.get(0, 0), a.get(1, 0), a.get(2, 0));
    let (b0, b1, b2) = (b.get(0, 0), b.get(1, 0), b.get(2, 0));
    Matrix::col_vector(vec![
        a1 * b2 - a2 * b1,
        a2 * b0 - a0 * b2,
        a0 * b1 - a1 * b0,
    ])
}

/// Dot product `Σ aᵢ·bᵢ` of two column vectors (n×1 matrices).
///
/// The product is bilinear, as SymPy's `Matrix.dot` by default
/// (`hermitian=False`): nothing is conjugated, so for complex vectors
/// `dot(v, v)` is not `‖v‖²` (`dot([1, i], [1, i]) = 0`).  Use
/// [`Matrix::norm`] for the length, or `dot(&v.map(Ex::conjugate), &w)`
/// for the Hermitian product `⟨v, w⟩`.
///
/// # Errors
///
/// Returns [`SymplexError::InvalidArgument`] if either argument is not a
/// column vector or the lengths differ.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::matrix::dot;
///
/// let ctx = Context::new();
/// assert_eq!(dot(&matrix![ctx, [1], [2]], &matrix![ctx, [3], [4]]).unwrap(), ctx.int(11));
/// assert!(dot(&matrix![ctx, [1], [2]], &matrix![ctx, [3]]).is_err());
/// assert!(dot(&matrix![ctx, [1, 2]], &matrix![ctx, [3, 4]]).is_err());
/// // Bilinear (SymPy: Matrix([1, I]).dot(Matrix([1, I])) == 0)
/// let v = Matrix::col_vector(vec![ctx.int(1), ctx.i_unit()]).unwrap();
/// assert_eq!(dot(&v, &v).unwrap(), ctx.int(0));
/// assert_eq!(dot(&v.map(Ex::conjugate), &v).unwrap(), ctx.int(2));
/// ```
pub fn dot(a: &Matrix, b: &Matrix) -> Result<Ex, SymplexError> {
    if a.ncols() != 1 || b.ncols() != 1 || a.nrows() != b.nrows() {
        return Err(SymplexError::invalid_argument(
            "dot",
            format!(
                "arguments must be column vectors of equal length, got {}×{} and {}×{}",
                a.nrows(),
                a.ncols(),
                b.nrows(),
                b.ncols()
            ),
        ));
    }
    let mut sum = a.get(0, 0) * b.get(0, 0);
    for i in 1..a.nrows() {
        sum += &(a.get(i, 0) * b.get(i, 0));
    }
    Ok(sum)
}

// ═══════════════════════════════════════════════════════════════════════════
// Display / Debug
// ═══════════════════════════════════════════════════════════════════════════

impl fmt::Display for Matrix {
    /// Single-row matrices print inline as `[[a, b, c]]`; larger matrices
    /// print one row per line with columns right-aligned:
    ///
    /// ```text
    /// [
    ///   [ 1, -2],
    ///   [30,  4]
    /// ]
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.nrows == 1 {
            write!(f, "[[")?;
            for (j, elem) in self.rows[0].iter().enumerate() {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{elem}")?;
            }
            return write!(f, "]]");
        }
        let cells: Vec<Vec<String>> = self
            .rows
            .iter()
            .map(|r| r.iter().map(ToString::to_string).collect())
            .collect();
        let widths: Vec<usize> = (0..self.ncols)
            .map(|j| {
                cells
                    .iter()
                    .map(|r| r[j].chars().count())
                    .max()
                    .unwrap_or(0)
            })
            .collect();
        writeln!(f, "[")?;
        for (i, row) in cells.iter().enumerate() {
            write!(f, "  [")?;
            for (j, cell) in row.iter().enumerate() {
                if j > 0 {
                    write!(f, ", ")?;
                }
                let pad = widths[j] - cell.chars().count();
                write!(f, "{}{}", " ".repeat(pad), cell)?;
            }
            write!(f, "]")?;
            if i + 1 < self.nrows {
                writeln!(f, ",")?;
            } else {
                writeln!(f)?;
            }
        }
        write!(f, "]")
    }
}

impl fmt::Debug for Matrix {
    /// `Matrix(2×2, [[1, 2], [3, 4]])` — shape followed by the rows inline.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Matrix({}×{}, [", self.nrows, self.ncols)?;
        for (i, row) in self.rows.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "[")?;
            for (j, elem) in row.iter().enumerate() {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{elem}")?;
            }
            write!(f, "]")?;
        }
        write!(f, "])")
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Indexing
// ═══════════════════════════════════════════════════════════════════════════

impl std::ops::Index<(usize, usize)> for Matrix {
    type Output = Ex;

    /// `m[(i, j)]` — panics on out-of-bounds like slice indexing.
    #[inline]
    fn index(&self, (i, j): (usize, usize)) -> &Ex {
        self.get(i, j)
    }
}

impl std::ops::IndexMut<(usize, usize)> for Matrix {
    #[inline]
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Ex {
        self.get_mut(i, j)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Operator overloads
//
// Operators panic on shape mismatch (like `Vec` indexing); use the
// `add` / `sub` / `matmul` methods for a `Result`.
// ═══════════════════════════════════════════════════════════════════════════

macro_rules! matrix_binop {
    ($trait:ident, $method:ident, $inner:ident, $msg:literal) => {
        impl std::ops::$trait<&Matrix> for &Matrix {
            type Output = Matrix;
            fn $method(self, rhs: &Matrix) -> Matrix {
                match self.$inner(rhs) {
                    Ok(m) => m,
                    Err(e) => panic!(concat!($msg, ": {}"), e),
                }
            }
        }
        impl std::ops::$trait<Matrix> for Matrix {
            type Output = Matrix;
            fn $method(self, rhs: Matrix) -> Matrix {
                std::ops::$trait::$method(&self, &rhs)
            }
        }
        impl std::ops::$trait<&Matrix> for Matrix {
            type Output = Matrix;
            fn $method(self, rhs: &Matrix) -> Matrix {
                std::ops::$trait::$method(&self, rhs)
            }
        }
        impl std::ops::$trait<Matrix> for &Matrix {
            type Output = Matrix;
            fn $method(self, rhs: Matrix) -> Matrix {
                std::ops::$trait::$method(self, &rhs)
            }
        }
    };
}

matrix_binop!(Add, add, add, "Matrix + Matrix");
matrix_binop!(Sub, sub, sub, "Matrix - Matrix");
matrix_binop!(Mul, mul, matmul, "Matrix * Matrix");

// -Matrix
impl std::ops::Neg for &Matrix {
    type Output = Matrix;
    fn neg(self) -> Matrix {
        self.map(|e| -e)
    }
}
impl std::ops::Neg for Matrix {
    type Output = Matrix;
    fn neg(self) -> Matrix {
        -&self
    }
}

// Matrix * scalar, scalar * Matrix, Matrix / scalar
impl std::ops::Mul<&Ex> for &Matrix {
    type Output = Matrix;
    fn mul(self, rhs: &Ex) -> Matrix {
        self.scale(rhs)
    }
}
impl std::ops::Mul<Ex> for &Matrix {
    type Output = Matrix;
    fn mul(self, rhs: Ex) -> Matrix {
        self.scale(&rhs)
    }
}
impl std::ops::Mul<&Ex> for Matrix {
    type Output = Matrix;
    fn mul(self, rhs: &Ex) -> Matrix {
        self.scale(rhs)
    }
}
impl std::ops::Mul<Ex> for Matrix {
    type Output = Matrix;
    fn mul(self, rhs: Ex) -> Matrix {
        self.scale(&rhs)
    }
}
impl std::ops::Mul<&Matrix> for &Ex {
    type Output = Matrix;
    fn mul(self, rhs: &Matrix) -> Matrix {
        rhs.scale(self)
    }
}
impl std::ops::Mul<Matrix> for &Ex {
    type Output = Matrix;
    fn mul(self, rhs: Matrix) -> Matrix {
        rhs.scale(self)
    }
}
impl std::ops::Mul<&Matrix> for Ex {
    type Output = Matrix;
    fn mul(self, rhs: &Matrix) -> Matrix {
        rhs.scale(&self)
    }
}
impl std::ops::Mul<Matrix> for Ex {
    type Output = Matrix;
    fn mul(self, rhs: Matrix) -> Matrix {
        rhs.scale(&self)
    }
}
impl std::ops::Div<&Ex> for &Matrix {
    type Output = Matrix;
    fn div(self, rhs: &Ex) -> Matrix {
        self.map(|e| e / rhs)
    }
}
impl std::ops::Div<Ex> for &Matrix {
    type Output = Matrix;
    fn div(self, rhs: Ex) -> Matrix {
        self / &rhs
    }
}
impl std::ops::Div<&Ex> for Matrix {
    type Output = Matrix;
    fn div(self, rhs: &Ex) -> Matrix {
        &self / rhs
    }
}
impl std::ops::Div<Ex> for Matrix {
    type Output = Matrix;
    fn div(self, rhs: Ex) -> Matrix {
        &self / &rhs
    }
}

// Matrix * i64, i64 * Matrix, Matrix / i64
impl std::ops::Mul<i64> for &Matrix {
    type Output = Matrix;
    fn mul(self, rhs: i64) -> Matrix {
        let s = self.ctx().int(rhs);
        self.scale(&s)
    }
}
impl std::ops::Mul<i64> for Matrix {
    type Output = Matrix;
    fn mul(self, rhs: i64) -> Matrix {
        &self * rhs
    }
}
impl std::ops::Mul<&Matrix> for i64 {
    type Output = Matrix;
    fn mul(self, rhs: &Matrix) -> Matrix {
        rhs * self
    }
}
impl std::ops::Mul<Matrix> for i64 {
    type Output = Matrix;
    fn mul(self, rhs: Matrix) -> Matrix {
        &rhs * self
    }
}
impl std::ops::Div<i64> for &Matrix {
    type Output = Matrix;
    fn div(self, rhs: i64) -> Matrix {
        let s = self.ctx().int(rhs);
        self / &s
    }
}
impl std::ops::Div<i64> for Matrix {
    type Output = Matrix;
    fn div(self, rhs: i64) -> Matrix {
        &self / rhs
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn tctx() -> Context {
        Context::new()
    }

    fn assert_zero_matrix(m: &Matrix, label: &str) {
        for i in 0..m.nrows() {
            for j in 0..m.ncols() {
                let d = m.get(i, j).expand().eval().simplify();
                assert!(d.is_zero_structural(), "{label}: entry ({i},{j}) = {d} ≠ 0");
            }
        }
    }

    // ── Constructor tests ──────────────────────────────────────────────

    #[test]
    fn identity_2x2() {
        let ctx = tctx();
        let m = Matrix::identity(&ctx, 2).unwrap();
        assert_eq!(m.nrows(), 2);
        assert_eq!(m.ncols(), 2);
        assert_eq!(format!("{}", m.get(0, 0)), "1");
        assert_eq!(format!("{}", m.get(0, 1)), "0");
        assert_eq!(format!("{}", m.get(1, 0)), "0");
        assert_eq!(format!("{}", m.get(1, 1)), "1");
    }

    #[test]
    fn identity_3x3() {
        let ctx = tctx();
        let m = Matrix::identity(&ctx, 3).unwrap();
        assert_eq!(m.shape(), (3, 3));
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { "1" } else { "0" };
                assert_eq!(format!("{}", m.get(i, j)), expected);
            }
        }
    }

    #[test]
    fn zeros_and_from_fn() {
        let ctx = tctx();
        let z = Matrix::zeros(&ctx, 3, 3).unwrap();
        assert_eq!(format!("{}", z.get(1, 1)), "0");

        let m = Matrix::from_fn(2, 2, |i, j| ctx.int((i * 2 + j + 1) as i64)).unwrap();
        assert_eq!(format!("{}", m.get(0, 0)), "1");
        assert_eq!(format!("{}", m.get(0, 1)), "2");
        assert_eq!(format!("{}", m.get(1, 0)), "3");
        assert_eq!(format!("{}", m.get(1, 1)), "4");
    }

    #[test]
    fn row_and_col_vectors() {
        let ctx = tctx();
        let rv = Matrix::row_vector(vec![ctx.int(1), ctx.int(2), ctx.int(3)]).unwrap();
        assert_eq!(rv.shape(), (1, 3));
        assert_eq!(format!("{}", rv.get(0, 1)), "2");

        let cv = Matrix::col_vector(vec![ctx.int(10), ctx.int(20)]).unwrap();
        assert_eq!(cv.shape(), (2, 1));
        assert_eq!(format!("{}", cv.get(1, 0)), "20");
    }

    #[test]
    fn from_i64_and_try_from() {
        let ctx = tctx();
        let m = Matrix::from_i64(&ctx, &[&[1, 2], &[3, 4]]).unwrap();
        assert_eq!(m.shape(), (2, 2));
        assert_eq!(m.get(1, 0), &ctx.int(3));
        assert!(Matrix::from_i64(&ctx, &[&[1, 2], &[3]]).is_err());
        let t: Matrix = vec![vec![ctx.int(7)]].try_into().unwrap();
        assert_eq!(t.shape(), (1, 1));
        let bad: Result<Matrix, _> = Vec::<Vec<Ex>>::new().try_into();
        assert!(bad.is_err());
    }

    #[test]
    fn block_diag_shapes() {
        let ctx = tctx();
        let a = Matrix::identity(&ctx, 2).unwrap();
        let b = Matrix::row_vector(vec![ctx.int(5), ctx.int(6), ctx.int(7)]).unwrap();
        let bd = Matrix::block_diag(&[&a, &b]).unwrap();
        assert_eq!(bd.shape(), (3, 5));
        assert_eq!(bd.get(2, 4), &ctx.int(7));
        assert!(bd.get(0, 2).is_zero_structural());
        assert!(Matrix::block_diag(&[]).is_err());
    }

    // ── Accessor tests ─────────────────────────────────────────────────

    #[test]
    fn get_mut_and_set_work() {
        let ctx = tctx();
        let mut m = Matrix::zeros(&ctx, 2, 2).unwrap();
        *m.get_mut(0, 1) = ctx.int(42);
        m.set(1, 0, ctx.int(7));
        m[(1, 1)] = ctx.int(9);
        assert_eq!(format!("{}", m.get(0, 1)), "42");
        assert_eq!(m[(1, 0)], ctx.int(7));
        assert_eq!(m[(1, 1)], ctx.int(9));
    }

    #[test]
    fn row_col_diagonal_accessors() {
        let ctx = tctx();
        let m = Matrix::from_i64(&ctx, &[&[1, 2, 3], &[4, 5, 6]]).unwrap();
        assert_eq!(m.row(0).len(), 3);
        assert_eq!(m.col(2), vec![ctx.int(3), ctx.int(6)]);
        assert_eq!(m.diagonal(), vec![ctx.int(1), ctx.int(5)]);
        assert_eq!(m.iter().count(), 6);
        assert_eq!(m.to_vec()[1][2], ctx.int(6));
        assert_eq!(m.eval_f64().unwrap()[1], vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn submatrix_and_minor_matrix() {
        let ctx = tctx();
        let m = Matrix::from_i64(&ctx, &[&[1, 2, 3], &[4, 5, 6], &[7, 8, 9]]).unwrap();
        let s = m.submatrix(0..2, 1..3).unwrap();
        assert_eq!(s, Matrix::from_i64(&ctx, &[&[2, 3], &[5, 6]]).unwrap());
        let mm = m.minor_matrix(1, 1).unwrap();
        assert_eq!(mm, Matrix::from_i64(&ctx, &[&[1, 3], &[7, 9]]).unwrap());
        assert_eq!(m.minor(1, 1).unwrap(), ctx.int(-12));
        assert!(m.minor_matrix(3, 0).is_err());
    }

    #[test]
    fn try_get_works() {
        let ctx = tctx();
        let m = Matrix::zeros(&ctx, 2, 2).unwrap();
        assert!(m.try_get(0, 0).is_some());
        assert!(m.try_get(1, 1).is_some());
        assert!(m.try_get(2, 0).is_none());
        assert!(m.try_get(0, 2).is_none());
    }

    #[test]
    fn structural_equality() {
        let ctx = tctx();
        let a = Matrix::from_i64(&ctx, &[&[1, 2], &[3, 4]]).unwrap();
        let b = Matrix::from_i64(&ctx, &[&[1, 2], &[3, 4]]).unwrap();
        let c = Matrix::from_i64(&ctx, &[&[1, 2], &[3, 5]]).unwrap();
        let d = Matrix::from_i64(&ctx, &[&[1, 2, 3, 4]]).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);
        assert_eq!(a.equals(&b), Some(true));
        assert_eq!(a.equals(&c), Some(false));
        assert_eq!(a.equals(&d), Some(false));
    }

    // ── Transpose ──────────────────────────────────────────────────────

    #[test]
    fn transpose() {
        let ctx = tctx();
        let a = ctx.symbol("a");
        let b = ctx.symbol("b");
        let c = ctx.symbol("c");
        let d = ctx.symbol("d");
        let m = Matrix::new(vec![vec![a.clone(), b.clone()], vec![c.clone(), d.clone()]]).unwrap();
        let t = m.transpose();
        assert_eq!(t.get(0, 0), &a);
        assert_eq!(t.get(0, 1), &c);
        assert_eq!(t.get(1, 0), &b);
        assert_eq!(t.get(1, 1), &d);
    }

    #[test]
    fn transpose_non_square() {
        let ctx = tctx();
        let m = Matrix::from_i64(&ctx, &[&[1, 2, 3], &[4, 5, 6]]).unwrap();
        let t = m.transpose();
        assert_eq!(t.shape(), (3, 2));
        assert_eq!(format!("{}", t.get(2, 0)), "3");
        assert_eq!(format!("{}", t.get(2, 1)), "6");
    }

    #[test]
    fn adjoint_conjugates() {
        let ctx = tctx();
        let i = ctx.i_unit();
        let m = Matrix::new(vec![
            vec![ctx.int(1), &ctx.int(2) * &i],
            vec![ctx.int(3), ctx.int(4)],
        ])
        .unwrap();
        let h = m.adjoint();
        assert_eq!(h.get(1, 0), &(&ctx.int(-2) * &i));
        assert_eq!(h.get(0, 1), &ctx.int(3));
    }

    // ── Addition / Subtraction ─────────────────────────────────────────

    #[test]
    fn matrix_add() {
        let ctx = tctx();
        let m1 = Matrix::from_i64(&ctx, &[&[1, 2], &[3, 4]]).unwrap();
        let m2 = Matrix::from_i64(&ctx, &[&[5, 6], &[7, 8]]).unwrap();
        let sum = m1.add(&m2).unwrap();
        assert_eq!(format!("{}", sum.get(0, 0)), "6");
        assert_eq!(format!("{}", sum.get(1, 1)), "12");
        assert_eq!(&m1 + &m2, sum);
    }

    #[test]
    fn matrix_sub() {
        let ctx = tctx();
        let m1 = Matrix::new(vec![vec![ctx.int(10), ctx.int(20)]]).unwrap();
        let m2 = Matrix::new(vec![vec![ctx.int(3), ctx.int(7)]]).unwrap();
        let diff = m1.sub(&m2).unwrap();
        assert_eq!(format!("{}", diff.get(0, 0)), "7");
        assert_eq!(format!("{}", diff.get(0, 1)), "13");
    }

    #[test]
    fn hadamard_product() {
        let ctx = tctx();
        let m1 = Matrix::from_i64(&ctx, &[&[1, 2], &[3, 4]]).unwrap();
        let m2 = Matrix::from_i64(&ctx, &[&[5, 6], &[7, 8]]).unwrap();
        let h = m1.hadamard(&m2).unwrap();
        assert_eq!(h, Matrix::from_i64(&ctx, &[&[5, 12], &[21, 32]]).unwrap());
    }

    // ── Scalar multiplication ──────────────────────────────────────────

    #[test]
    fn scale() {
        let ctx = tctx();
        let m = Matrix::identity(&ctx, 2).unwrap();
        let two = ctx.int(2);
        let scaled = m.scale(&two);
        assert_eq!(format!("{}", scaled.get(0, 0)), "2");
        assert_eq!(format!("{}", scaled.get(0, 1)), "0");
    }

    #[test]
    fn scalar_operators_both_sides() {
        let ctx = tctx();
        let x = ctx.symbol("x");
        let m = Matrix::from_i64(&ctx, &[&[1, 2]]).unwrap();
        let a = &m * &x;
        let b = &x * &m;
        let c = m.clone() * x.clone();
        let d = x.clone() * m.clone();
        assert_eq!(a, b);
        assert_eq!(a, c);
        assert_eq!(a, d);
        assert_eq!(a.get(0, 1), &(&ctx.int(2) * &x));
        let e = &m * 3;
        let f = 3 * &m;
        assert_eq!(e, f);
        assert_eq!(e.get(0, 1), &ctx.int(6));
        let g = &m / 2;
        assert_eq!(g.get(0, 0), &ctx.rational(1, 2));
        let h = &m / &ctx.int(2);
        assert_eq!(g, h);
        let n = -&m;
        assert_eq!(n.get(0, 1), &ctx.int(-2));
    }

    // ── Matrix multiplication ──────────────────────────────────────────

    #[test]
    fn matmul_2x2() {
        let ctx = tctx();
        let m = Matrix::identity(&ctx, 2).unwrap();
        let a = ctx.symbol("a");
        let b = ctx.symbol("b");
        let c = ctx.symbol("c");
        let d = ctx.symbol("d");
        let n = Matrix::new(vec![vec![a.clone(), b.clone()], vec![c.clone(), d.clone()]]).unwrap();
        let result = m.matmul(&n).unwrap();
        assert_eq!(result.get(0, 0), &a);
        assert_eq!(result.get(1, 1), &d);
    }

    #[test]
    fn matmul_non_square() {
        let ctx = tctx();
        let rv = Matrix::row_vector(vec![ctx.int(2), ctx.int(3)]).unwrap();
        let cv = Matrix::col_vector(vec![ctx.int(4), ctx.int(5)]).unwrap();
        let result = rv.matmul(&cv).unwrap();
        assert_eq!(result.shape(), (1, 1));
        assert_eq!(format!("{}", result.get(0, 0)), "23");
    }

    #[test]
    fn matmul_numeric() {
        let ctx = tctx();
        let a = Matrix::from_i64(&ctx, &[&[1, 2], &[3, 4]]).unwrap();
        let b = Matrix::from_i64(&ctx, &[&[5, 6], &[7, 8]]).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c, Matrix::from_i64(&ctx, &[&[19, 22], &[43, 50]]).unwrap());
    }

    // ── Trace ──────────────────────────────────────────────────────────

    #[test]
    fn trace_2x2() {
        let ctx = tctx();
        let a = ctx.symbol("a");
        let d = ctx.symbol("d");
        let m = Matrix::new(vec![
            vec![a.clone(), ctx.symbol("b")],
            vec![ctx.symbol("c"), d.clone()],
        ])
        .unwrap();
        assert_eq!(m.trace().unwrap(), &a + &d);
    }

    #[test]
    fn trace_numeric() {
        let ctx = tctx();
        let m = Matrix::from_i64(&ctx, &[&[1, 2], &[3, 4]]).unwrap();
        assert_eq!(format!("{}", m.trace().unwrap()), "5");
    }

    // ── Determinant ────────────────────────────────────────────────────

    #[test]
    fn det_1x1() {
        let ctx = tctx();
        let a = ctx.symbol("a");
        let m = Matrix::new(vec![vec![a.clone()]]).unwrap();
        assert_eq!(m.det().unwrap(), a);
    }

    #[test]
    fn det_2x2() {
        let ctx = tctx();
        let a = ctx.symbol("a");
        let b = ctx.symbol("b");
        let c = ctx.symbol("c");
        let d = ctx.symbol("d");
        let m = Matrix::new(vec![vec![a.clone(), b.clone()], vec![c.clone(), d.clone()]]).unwrap();
        assert_eq!(m.det().unwrap(), &(&a * &d) - &(&b * &c));
    }

    #[test]
    fn det_2x2_numeric() {
        let ctx = tctx();
        let m = Matrix::from_i64(&ctx, &[&[3, 8], &[4, 6]]).unwrap();
        assert_eq!(format!("{}", m.det().unwrap()), "-14");
    }

    #[test]
    fn det_3x3_numeric() {
        let ctx = tctx();
        let m = Matrix::from_i64(&ctx, &[&[1, 2, 3], &[4, 5, 6], &[7, 8, 9]]).unwrap();
        assert_eq!(format!("{}", m.det().unwrap()), "0");
    }

    #[test]
    fn det_3x3_nonsingular() {
        let ctx = tctx();
        let m = Matrix::from_i64(&ctx, &[&[1, 0, 2], &[0, 1, 0], &[3, 0, 1]]).unwrap();
        assert_eq!(format!("{}", m.det().unwrap()), "-5");
    }

    #[test]
    fn det_4x4_symbolic_is_polynomial() {
        // Berkowitz path: result must be an expanded polynomial, equal to
        // the cofactor expansion.
        let ctx = tctx();
        let syms: Vec<Vec<Ex>> = (0..4)
            .map(|i| (0..4).map(|j| ctx.symbol(&format!("a{i}{j}"))).collect())
            .collect();
        let m = Matrix::new(syms).unwrap();
        let d = m.det().unwrap();
        assert_eq!(d.term_count(), 24, "4×4 symbolic det has 24 terms: {d}");
        // Cross-check against Laplace expansion along the first row.
        let mut expected = ctx.zero();
        for j in 0..4 {
            expected += m.get(0, j) * &m.cofactor(0, j).unwrap();
        }
        assert!((&d - &expected).expand().is_zero_structural());
    }

    #[test]
    fn det_4x4_mixed_symbolic_matches_bareiss_numeric_specialization() {
        let ctx = tctx();
        let x = ctx.symbol("x");
        let m = Matrix::new(vec![
            vec![x.clone(), ctx.int(2), ctx.int(0), ctx.int(1)],
            vec![ctx.int(1), x.clone(), ctx.int(3), ctx.int(0)],
            vec![ctx.int(0), ctx.int(1), x.clone(), ctx.int(2)],
            vec![ctx.int(2), ctx.int(0), ctx.int(1), x.clone()],
        ])
        .unwrap();
        let d = m.det().unwrap();
        assert!(d.is_polynomial(&x), "symbolic det must be polynomial: {d}");
        for v in [-2i64, 0, 1, 3, 7] {
            let numeric = m.subs(&x, &ctx.int(v)).det().unwrap().eval();
            let via_sym = d.subs(&x, &ctx.int(v)).eval();
            assert_eq!(numeric, via_sym, "det mismatch at x={v}");
        }
    }

    // ── Map ────────────────────────────────────────────────────────────

    #[test]
    fn map_doubles() {
        let ctx = tctx();
        let m = Matrix::from_i64(&ctx, &[&[1, 2], &[3, 4]]).unwrap();
        let mut calls = 0;
        let doubled = m.map(|e| {
            calls += 1;
            e * 2
        });
        assert_eq!(calls, 4);
        assert_eq!(format!("{}", doubled.get(0, 0)), "2");
        assert_eq!(format!("{}", doubled.get(1, 1)), "8");
        let idx = m.map_indexed(|i, j, e| e + ctx.int((10 * i + j) as i64));
        assert_eq!(idx.get(1, 1), &ctx.int(15));
    }

    // ── Calculus helpers ───────────────────────────────────────────────

    #[test]
    fn jacobian_test() {
        let ctx = tctx();
        let x = ctx.symbol("x");
        let y = ctx.symbol("y");
        let f1 = &x.powi(2) + &y;
        let f2 = &x * &y;
        let j = jacobian(&[&f1, &f2], &[&x, &y]).unwrap();
        assert_eq!(j.shape(), (2, 2));
        assert_eq!(j.get(0, 0), &(&x * 2));
        assert_eq!(j.get(0, 1), &ctx.int(1));
        assert_eq!(j.get(1, 0), &y);
        assert_eq!(j.get(1, 1), &x);
    }

    #[test]
    fn matrix_diff_and_integrate() {
        let ctx = tctx();
        let x = ctx.symbol("x");
        let m = Matrix::new(vec![vec![x.powi(2), x.sin()]]).unwrap();
        let dm = m.diff(&x);
        assert_eq!(dm.get(0, 0), &(&x * 2));
        assert_eq!(dm.get(0, 1), &x.cos());
        let im = m.integrate(&x);
        assert_eq!(im.get(0, 0), &(&x.powi(3) / 3));
    }

    #[test]
    fn matrix_subs() {
        let ctx = tctx();
        let x = ctx.symbol("x");
        let m = Matrix::new(vec![vec![x.powi(2), x.clone()]]).unwrap();
        let result = m.subs(&x, &ctx.int(3));
        assert_eq!(format!("{}", result.get(0, 0)), "9");
        assert_eq!(format!("{}", result.get(0, 1)), "3");
    }

    #[test]
    fn matrix_eval() {
        let ctx = tctx();
        let m = Matrix::new(vec![vec![ctx.pi().cos(), ctx.int(2) + ctx.int(3)]]).unwrap();
        let evaled = m.eval();
        assert_eq!(format!("{}", evaled.get(0, 0)), "-1");
        assert_eq!(format!("{}", evaled.get(0, 1)), "5");
    }

    #[test]
    fn matrix_expand() {
        let ctx = tctx();
        let x = ctx.symbol("x");
        let expr = (&x + ctx.int(1)).powi(2);
        let m = Matrix::new(vec![vec![expr]]).unwrap();
        let expanded = m.expand();
        assert_eq!(expanded.get(0, 0), &(&x.powi(2) + &x * 2 + 1));
    }

    // ── Characteristic polynomial ──────────────────────────────────────

    #[test]
    fn char_poly_coeffs_2x2_and_3x3() {
        let ctx = tctx();
        let a = Matrix::from_i64(&ctx, &[&[1, 2], &[3, 4]]).unwrap();
        // det(A − λI) = λ² − 5λ − 2
        assert_eq!(
            a.char_poly_coeffs().unwrap(),
            vec![ctx.int(-2), ctx.int(-5), ctx.int(1)]
        );
        let b = Matrix::diag(&[ctx.int(2), ctx.int(3), ctx.int(4)]).unwrap();
        // (2−λ)(3−λ)(4−λ) = −λ³ + 9λ² − 26λ + 24
        assert_eq!(
            b.char_poly_coeffs().unwrap(),
            vec![ctx.int(24), ctx.int(-26), ctx.int(9), ctx.int(-1)]
        );
    }

    #[test]
    fn char_poly_symbolic_2x2() {
        let ctx = tctx();
        let (a, b, c, d) = (
            ctx.symbol("a"),
            ctx.symbol("b"),
            ctx.symbol("c"),
            ctx.symbol("d"),
        );
        let m = Matrix::new(vec![vec![a.clone(), b.clone()], vec![c.clone(), d.clone()]]).unwrap();
        let coeffs = m.char_poly_coeffs().unwrap();
        assert_eq!(coeffs.len(), 3);
        assert!(
            (&coeffs[0] - &(&a * &d - &b * &c))
                .expand()
                .is_zero_structural()
        );
        assert!((&coeffs[1] + &(&a + &d)).expand().is_zero_structural());
        assert!(coeffs[2].is_one_structural());
    }

    #[test]
    fn char_poly_5x5_numeric_is_polynomial_and_fast() {
        let ctx = tctx();
        let lam = ctx.symbol("lambda");
        let m = Matrix::from_fn(5, 5, |i, j| {
            ctx.int(((i * 7 + j * 3) % 5) as i64 + if i == j { 2 } else { 0 })
        })
        .unwrap();
        let timer = crate::base::perf_assert::CpuTimer::start();
        let cp = m.char_poly(&lam).unwrap();
        timer.assert_within(std::time::Duration::from_secs(5), "char_poly");
        assert_eq!(
            cp.degree(&lam),
            Some(5),
            "cp must be a degree-5 polynomial: {cp}"
        );
        // p(0) = det(A)
        let p0 = cp.subs(&lam, &ctx.int(0)).eval();
        assert_eq!(p0, m.det().unwrap().eval());
        // Cayley–Hamilton: p(A) = 0
        let coeffs = m.char_poly_coeffs().unwrap();
        let mut acc = Matrix::zeros(&ctx, 5, 5).unwrap();
        for (k, c) in coeffs.iter().enumerate() {
            acc = &acc + &(&m.powi(k as u32).unwrap() * c);
        }
        assert_zero_matrix(&acc.eval(), "Cayley–Hamilton 5×5");
    }

    #[test]
    fn char_poly_5x5_symbolic_terminates_quickly() {
        let ctx = tctx();
        let syms: Vec<Vec<Ex>> = (0..5)
            .map(|i| (0..5).map(|j| ctx.symbol(&format!("a{i}{j}"))).collect())
            .collect();
        let m = Matrix::new(syms).unwrap();
        let timer = crate::base::perf_assert::CpuTimer::start();
        let coeffs = m.char_poly_coeffs().unwrap();
        timer.assert_within(std::time::Duration::from_secs(20), "5×5 symbolic char poly");
        assert_eq!(coeffs.len(), 6);
        // det(A − λI) = −λ⁵ + tr·λ⁴ − … + det, so c_5 = −1, c_4 = +trace, c_0 = det (120 terms)
        assert_eq!(coeffs[5], ctx.int(-1));
        assert!(
            (&coeffs[4] - &m.trace().unwrap())
                .expand()
                .is_zero_structural()
        );
        assert_eq!(coeffs[0].term_count(), 120);
        assert!(
            (&coeffs[0] - &m.det().unwrap())
                .expand()
                .is_zero_structural()
        );
    }

    // ── Eigenvector / Diagonalization tests ─────────────────────────

    #[test]
    fn eigenvects_2x2_distinct() {
        let ctx = tctx();
        let m = Matrix::from_i64(&ctx, &[&[2, 1], &[0, 3]]).unwrap();
        let evs = m.eigenvects().expect("eigenvects should succeed");
        assert_eq!(evs.len(), 2);
        for (eigenval, mult, vecs) in &evs {
            assert_eq!(*mult, 1);
            assert_eq!(vecs.len(), 1);
            let av = m.matmul(&vecs[0]).unwrap();
            let lambda_v = vecs[0].scale(eigenval);
            assert_zero_matrix(&(&av - &lambda_v), "A·v = λ·v");
        }
    }

    #[test]
    fn eigenvects_non_square_returns_error() {
        let ctx = tctx();
        let m = Matrix::from_i64(&ctx, &[&[1, 2, 3], &[4, 5, 6]]).unwrap();
        let result = m.eigenvects();
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("square"), "{err_msg}");
    }

    #[test]
    fn eigenvects_diagonal_matrix() {
        let ctx = tctx();
        let m = Matrix::diag(&[ctx.int(5), ctx.int(-3)]).unwrap();
        let evs = m.eigenvects().unwrap();
        assert_eq!(evs.len(), 2);
        let vals: Vec<_> = evs.iter().map(|(v, _, _)| v.clone()).collect();
        assert!(vals.contains(&ctx.int(5)) && vals.contains(&ctx.int(-3)));
    }

    #[test]
    fn eigenvals_repeat_with_multiplicity() {
        let ctx = tctx();
        let m = Matrix::from_i64(&ctx, &[&[1, 1], &[0, 1]]).unwrap();
        assert_eq!(m.eigenvals().unwrap(), vec![ctx.int(1), ctx.int(1)]);
        assert_eq!(
            m.eigenvals_with_multiplicity().unwrap(),
            vec![(ctx.int(1), 2)]
        );
    }

    #[test]
    fn eigenvals_symbolic_2x2_quadratic_formula() {
        let ctx = tctx();
        let (a, b) = (ctx.symbol("a"), ctx.symbol("b"));
        let m = Matrix::new(vec![vec![a.clone(), b.clone()], vec![b.clone(), a.clone()]]).unwrap();
        let evs = m.eigenvals().unwrap();
        assert_eq!(evs.len(), 2);
        // Eigenvalues a ± √(b²): check they satisfy the char poly at sample points
        let lam = ctx.symbol("lambda");
        let cp = m.char_poly(&lam).unwrap();
        for ev in &evs {
            assert!(!ev.to_string().contains("__lambda"), "dummy leaked: {ev}");
            for (av, bv) in [(3i64, 2i64), (-1, 5), (7, -4)] {
                let residual = cp
                    .subs(&lam, ev)
                    .subs(&a, &ctx.int(av))
                    .subs(&b, &ctx.int(bv))
                    .eval_f64()
                    .unwrap();
                assert!(
                    residual.abs() < 1e-9,
                    "eigenvalue {ev} does not satisfy char poly at a={av}, b={bv}: {residual}"
                );
            }
        }
        // Sum of eigenvalues is the trace
        let sum = (&evs[0] + &evs[1]).expand();
        assert!((&sum - &m.trace().unwrap()).expand().is_zero_structural());
    }

    #[test]
    fn eigen_dummy_never_leaks_even_for_rootof() {
        // Companion matrix of x^5 − x − 1 (irreducible, unsolvable in radicals).
        let ctx = tctx();
        let m = Matrix::new(vec![
            vec![ctx.int(0), ctx.int(0), ctx.int(0), ctx.int(0), ctx.int(1)],
            vec![ctx.int(1), ctx.int(0), ctx.int(0), ctx.int(0), ctx.int(1)],
            vec![ctx.int(0), ctx.int(1), ctx.int(0), ctx.int(0), ctx.int(0)],
            vec![ctx.int(0), ctx.int(0), ctx.int(1), ctx.int(0), ctx.int(0)],
            vec![ctx.int(0), ctx.int(0), ctx.int(0), ctx.int(1), ctx.int(0)],
        ])
        .unwrap();
        let evs = m.eigenvals().unwrap();
        assert_eq!(evs.len(), 5);
        for ev in &evs {
            let s = ev.to_string();
            assert!(!s.contains("__"), "reserved dummy leaked into result: {s}");
            assert!(s.contains("RootOf"), "expected RootOf, got {s}");
        }
    }

    #[test]
    fn eigen_dummy_avoids_user_symbol_named_lambda() {
        let ctx = tctx();
        let user = ctx.symbol("__lambda");
        let m = Matrix::new(vec![
            vec![user.clone(), ctx.int(1)],
            vec![ctx.int(0), ctx.int(2)],
        ])
        .unwrap();
        // The internal dummy must dodge the user's symbol.
        let dummy = m.fresh_symbol("lambda");
        assert_ne!(dummy, user);
        assert_eq!(dummy.to_string(), "__lambda_1");
        // 1×1: eigenvalue is exactly the user's symbol.
        let one_by_one = Matrix::new(vec![vec![user.clone()]]).unwrap();
        assert_eq!(one_by_one.eigenvals().unwrap(), vec![user.clone()]);
        // 2×2 upper triangular: eigenvalues {__lambda, 2}, verified numerically.
        let evs = m.eigenvals().unwrap();
        assert_eq!(evs.len(), 2);
        let mut nums: Vec<f64> = evs
            .iter()
            .map(|e| e.subs(&user, &ctx.int(5)).eval_f64().unwrap())
            .collect();
        nums.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(
            (nums[0] - 2.0).abs() < 1e-12 && (nums[1] - 5.0).abs() < 1e-12,
            "{nums:?}"
        );
    }

    #[test]
    fn diagonalize_upper_triangular() {
        let ctx = tctx();
        let m = Matrix::from_i64(&ctx, &[&[2, 1], &[0, 3]]).unwrap();
        let Diagonalization { p, d } = m.diagonalize().unwrap();
        assert_eq!(p.nrows(), 2);
        assert_eq!(d.nrows(), 2);
        let p_inv = p.inv().unwrap();
        let reconstructed = p.matmul(&d).unwrap().matmul(&p_inv).unwrap();
        assert_zero_matrix(&(&reconstructed - &m), "P·D·P⁻¹ = M");
    }

    #[test]
    fn diagonalize_non_diagonalizable() {
        let ctx = tctx();
        let m = Matrix::from_i64(&ctx, &[&[1, 1], &[0, 1]]).unwrap();
        let result = m.diagonalize();
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("not diagonalizable"), "{err_msg}");
        assert_eq!(m.is_diagonalizable(), Some(false));
    }

    #[test]
    fn is_diagonalizable_yes() {
        let ctx = tctx();
        let m = Matrix::diag(&[ctx.int(1), ctx.int(2), ctx.int(3)]).unwrap();
        assert_eq!(m.is_diagonalizable(), Some(true));
    }

    // ── Jordan form tests ──────────────────────────────────────────

    #[test]
    fn jordan_form_diagonal() {
        let ctx = tctx();
        let m = Matrix::diag(&[ctx.int(1), ctx.int(2), ctx.int(3)]).unwrap();
        let JordanForm { p, j } = m.jordan_form().unwrap();
        assert_eq!(j.nrows(), 3);
        let p_inv = p.inv().unwrap();
        let reconstructed = p.matmul(&j).unwrap().matmul(&p_inv).unwrap();
        assert_zero_matrix(&(&reconstructed - &m), "P·J·P⁻¹ = M");
    }

    #[test]
    fn jordan_form_defective_2x2() {
        let ctx = tctx();
        let m = Matrix::from_i64(&ctx, &[&[1, 1], &[0, 1]]).unwrap();
        let JordanForm { p, j } = m.jordan_form().unwrap();
        assert_eq!(j.nrows(), 2);
        assert!(j.get(0, 1).is_one_structural(), "J[0,1] should be 1");
        let reconstructed = p.matmul(&j).unwrap().matmul(&p.inv().unwrap()).unwrap();
        assert_zero_matrix(&(&reconstructed - &m), "P·J·P⁻¹ = M");
    }

    #[test]
    fn jordan_form_upper_triangular_distinct() {
        let ctx = tctx();
        let m = Matrix::from_i64(&ctx, &[&[2, 1], &[0, 3]]).unwrap();
        let JordanForm { p, j } = m.jordan_form().unwrap();
        assert_eq!(j.nrows(), 2);
        let reconstructed = p.matmul(&j).unwrap().matmul(&p.inv().unwrap()).unwrap();
        assert_zero_matrix(&(&reconstructed - &m), "P·J·P⁻¹ = M");
    }

    // ── Matrix exponential tests ───────────────────────────────────

    #[test]
    fn matrix_exp_identity() {
        let ctx = tctx();
        let m = Matrix::identity(&ctx, 2).unwrap();
        let result = m.matrix_exp().unwrap();
        assert_eq!(result.get(0, 0), &ctx.e());
        assert!(result.get(0, 1).is_zero_structural());
    }

    #[test]
    fn matrix_exp_zero() {
        let ctx = tctx();
        let m = Matrix::zeros(&ctx, 2, 2).unwrap();
        let result = m.matrix_exp().unwrap();
        assert_eq!(result, Matrix::identity(&ctx, 2).unwrap());
    }

    #[test]
    fn matrix_exp_t_nilpotent_and_rotation() {
        let ctx = tctx();
        let t = ctx.symbol("t");
        let n = Matrix::from_i64(&ctx, &[&[0, 1, 0], &[0, 0, 1], &[0, 0, 0]]).unwrap();
        let e = n.matrix_exp_t(&t).unwrap();
        // e^{Nt} = I + Nt + N²t²/2
        assert_eq!(e.get(0, 1), &t);
        assert_eq!(e.get(0, 2), &(&t.powi(2) / 2));
        assert!(e.get(1, 0).is_zero_structural());

        let rot = Matrix::from_i64(&ctx, &[&[0, 1], &[-1, 0]]).unwrap();
        let et = rot.matrix_exp_t(&t).unwrap();
        // Exactly [[cos t, sin t], [−sin t, cos t]] (Euler rewrite of e^{±it}).
        assert_eq!(et.get(0, 0), &t.cos());
        assert_eq!(et.get(0, 1), &t.sin());
        assert_eq!(et.get(1, 0), &(-&t.sin()));
        assert_eq!(et.get(1, 1), &t.cos());
        // …and numerically at t = 0.7
        let at = et.subs(&t, &ctx.rational(7, 10));
        let expect = [[0.7f64.cos(), 0.7f64.sin()], [-0.7f64.sin(), 0.7f64.cos()]];
        for (i, row) in expect.iter().enumerate() {
            for (j, want) in row.iter().enumerate() {
                let z = at.get(i, j).eval_complex64().unwrap();
                assert!((z.re - want).abs() < 1e-12, "({i},{j}) re={}", z.re);
                assert!(z.im.abs() < 1e-12, "({i},{j}) im={}", z.im);
            }
        }
    }

    #[test]
    fn matrix_exp_non_square_returns_error() {
        let ctx = tctx();
        let m = Matrix::from_i64(&ctx, &[&[1, 2, 3], &[4, 5, 6]]).unwrap();
        assert!(m.matrix_exp().is_err());
    }

    #[test]
    fn jordan_form_non_square_returns_error() {
        let ctx = tctx();
        let m = Matrix::from_i64(&ctx, &[&[1, 2, 3], &[4, 5, 6]]).unwrap();
        assert!(m.jordan_form().is_err());
    }

    #[test]
    fn diagonalize_non_square_returns_error() {
        let ctx = tctx();
        let m = Matrix::from_i64(&ctx, &[&[1, 2, 3], &[4, 5, 6]]).unwrap();
        assert!(m.diagonalize().is_err());
        assert_eq!(m.is_diagonalizable(), Some(false));
    }

    // ── Decompositions ─────────────────────────────────────────────────

    #[test]
    fn lu_returns_result() {
        let ctx = tctx();
        let a = Matrix::from_i64(&ctx, &[&[4, 3], &[6, 3]]).unwrap();
        let Lu { l, u, perm } = a.lu().unwrap();
        assert_eq!(perm.len(), 2);
        let pa = Matrix::new(perm.iter().map(|&i| a.row(i).to_vec()).collect()).unwrap();
        assert_zero_matrix(&(&(&l * &u) - &pa), "L·U = P·A");
        assert!(l.get(0, 1).is_zero_structural());
        assert!(u.get(1, 0).is_zero_structural());
        assert!(
            Matrix::from_i64(&ctx, &[&[1, 2], &[2, 4]])
                .unwrap()
                .lu()
                .is_err()
        );
        assert!(Matrix::from_i64(&ctx, &[&[1, 2, 3]]).unwrap().lu().is_err());
    }

    #[test]
    fn inverse_numeric_via_gauss_jordan() {
        let ctx = tctx();
        let a = Matrix::from_i64(&ctx, &[&[2, 1, 0], &[1, 3, 1], &[0, 1, 4]]).unwrap();
        let inv = a.inv().unwrap();
        assert_eq!((&a * &inv).eval(), Matrix::identity(&ctx, 3).unwrap());
    }

    #[test]
    fn rowspace_and_left_nullspace() {
        let ctx = tctx();
        let a = Matrix::from_i64(&ctx, &[&[1, 2], &[2, 4], &[3, 6]]).unwrap();
        assert_eq!(a.rowspace().len(), 1);
        let ln = a.left_nullspace();
        assert_eq!(ln.len(), 2);
        for y in &ln {
            let yt_a = y.transpose().matmul(&a).unwrap();
            assert_zero_matrix(&yt_a.eval(), "yᵀA = 0");
        }
    }

    #[test]
    fn solve_least_squares_matches_normal_equations() {
        let ctx = tctx();
        let a = Matrix::from_i64(&ctx, &[&[1, 0], &[1, 1], &[1, 2]]).unwrap();
        let b = Matrix::from_i64(&ctx, &[&[1], &[2], &[4]]).unwrap();
        let x = a.solve_least_squares(&b).unwrap();
        assert_eq!(x.get(0, 0), &ctx.rational(5, 6));
        assert_eq!(x.get(1, 0), &ctx.rational(3, 2));
        let rank_deficient = Matrix::from_i64(&ctx, &[&[1, 2], &[2, 4]]).unwrap();
        assert!(
            rank_deficient
                .solve_least_squares(&Matrix::from_i64(&ctx, &[&[1], &[1]]).unwrap())
                .is_err()
        );
    }

    #[test]
    fn vec_column_stacking() {
        let ctx = tctx();
        let a = Matrix::from_i64(&ctx, &[&[1, 2], &[3, 4]]).unwrap();
        assert_eq!(
            a.vec(),
            Matrix::from_i64(&ctx, &[&[1], &[3], &[2], &[4]]).unwrap()
        );
    }

    // ── Display tests ──────────────────────────────────────────────

    #[test]
    fn matrix_display_aligned() {
        let ctx = tctx();
        let m = Matrix::from_i64(&ctx, &[&[1, -2], &[30, 4]]).unwrap();
        assert_eq!(format!("{m}"), "[\n  [ 1, -2],\n  [30,  4]\n]");
        assert_eq!(format!("{m:?}"), "Matrix(2×2, [[1, -2], [30, 4]])");
    }

    #[test]
    fn row_vector_display() {
        let ctx = tctx();
        let m = Matrix::row_vector(vec![ctx.int(1), ctx.int(2), ctx.int(3)]).unwrap();
        assert_eq!(format!("{m}"), "[[1, 2, 3]]");
    }

    // ── Error tests (non-square / shape mismatch) ──────────────────────

    #[test]
    fn add_mismatched_shapes_returns_err() {
        let ctx = tctx();
        let a = Matrix::zeros(&ctx, 2, 3).unwrap();
        let b = Matrix::zeros(&ctx, 3, 2).unwrap();
        let result = a.add(&b);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("add"), "{err_msg}");
    }

    #[test]
    fn matmul_incompatible_returns_err() {
        let ctx = tctx();
        let a = Matrix::zeros(&ctx, 2, 3).unwrap();
        let b = Matrix::zeros(&ctx, 2, 3).unwrap();
        let result = a.matmul(&b);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("multiply"), "{err_msg}");
    }

    #[test]
    fn trace_non_square_returns_err() {
        let ctx = tctx();
        let m = Matrix::zeros(&ctx, 2, 3).unwrap();
        let result = m.trace();
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("square"));
    }

    #[test]
    fn det_non_square_returns_err() {
        let ctx = tctx();
        let m = Matrix::zeros(&ctx, 2, 3).unwrap();
        let result = m.det();
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("square"));
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn get_out_of_bounds_panics() {
        let ctx = Context::new();
        let m = Matrix::zeros(&ctx, 2, 2).unwrap();
        let _ = m.get(2, 0);
    }

    #[test]
    #[should_panic(expected = "Matrix + Matrix")]
    fn operator_add_mismatch_panics() {
        let ctx = Context::new();
        let _ = &Matrix::zeros(&ctx, 2, 2).unwrap() + &Matrix::zeros(&ctx, 3, 3).unwrap();
    }

    // ── constructor validation tests ───────────────────────────────────

    #[test]
    fn new_empty_returns_error() {
        let result = Matrix::new(vec![]);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("at least one row"), "{err_msg}");
    }

    #[test]
    fn new_jagged_returns_error() {
        let ctx = tctx();
        let result = Matrix::new(vec![vec![ctx.int(1), ctx.int(2)], vec![ctx.int(3)]]);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("row 1 has length 1 but expected 2"),
            "{err_msg}"
        );
    }

    #[test]
    fn new_valid_succeeds() {
        let ctx = tctx();
        let m = Matrix::from_i64(&ctx, &[&[1, 2], &[3, 4]]).unwrap();
        assert_eq!(m.shape(), (2, 2));
        assert_eq!(format!("{}", m.get(0, 0)), "1");
        assert_eq!(format!("{}", m.get(1, 1)), "4");
    }

    // ── scalar helpers ─────────────────────────────────────────────────

    #[test]
    fn three_valued_helpers() {
        let ctx = tctx();
        let x = ctx.symbol("x");
        assert_eq!(ex_is_zero(&ctx.int(0)), Truth::True);
        assert_eq!(ex_is_zero(&ctx.int(2)), Truth::False);
        assert_eq!(ex_is_zero(&(ctx.int(2).sqrt() - 1)), Truth::False);
        assert_eq!(
            ex_is_zero(&(&x.sin().powi(2) + &x.cos().powi(2) - 1)),
            Truth::True
        );
        assert_eq!(ex_is_zero(&x), Truth::Unknown);
        assert_eq!(ex_is_positive(&(ctx.int(2).sqrt() - 1)), Truth::True);
        assert_eq!(
            ex_is_positive(&(ctx.int(1) - ctx.int(2).sqrt())),
            Truth::False
        );
        assert_eq!(ex_is_positive(&x), Truth::Unknown);
        assert_eq!(ex_is_nonnegative(&ctx.int(0)), Truth::True);
    }

    /// `P = (x − p₀)(x − p₁)(x − p₂)·(1 + y)³¹·(1 + z)³¹` with `pₖ` the
    /// `x`-coordinates of the three sample points vanishes at all of them,
    /// and its numerator (4,096 terms) is past the expansion budget.  Up to
    /// 0.49 the zero samples were the answer (Schwartz–Zippel):
    /// `Some(true)`, so `P` was "identically zero".  The grid test refutes
    /// it exactly.
    ///
    /// Oracle: `P(0, 0, 0) = −p₀p₁p₂ ≠ 0` (the `pₖ` are nonzero rationals).
    #[test]
    fn zero_samples_do_not_prove_a_rational_function_zero() {
        let ctx = tctx();
        let build = |i: usize| {
            let p: Vec<String> = (0..RF_POINTS)
                .map(|k| rf_sample_point(i, k).to_string())
                .collect();
            ctx.parse(&format!(
                "(x - ({}))*(x - ({}))*(x - ({}))*(1 + y)^31*(1 + z)^31",
                p[0], p[1], p[2]
            ))
            .unwrap()
        };
        // The coordinate of `x` is its position among the free symbols.
        let x = ctx.symbol("x");
        let mut i = 0;
        let mut e = build(i);
        for _ in 0..3 {
            let at = {
                let inner = e.inner.read();
                crate::base::walk::free_symbols(&inner.arena, e.raw_id())
                    .iter()
                    .position(|&s| s == x.raw_id())
                    .unwrap()
            };
            if at == i {
                break;
            }
            i = at;
            e = build(i);
        }
        {
            let inner = e.inner.read();
            let syms = crate::base::walk::free_symbols(&inner.arena, e.raw_id());
            for k in 0..RF_POINTS {
                let v = rf_value_at(&inner.arena, e.raw_id(), &syms, k).unwrap();
                assert_eq!(v.map(|v| num_traits::Zero::is_zero(&v)), Some(true));
            }
            assert_eq!(
                rf_numerator_is_zero(&inner.arena, e.raw_id(), &syms),
                Truth::Unknown
            );
        }
        assert_eq!(rational_function_is_zero_q(&e), Truth::False);
    }

    /// `(1 + x)⁶³·(1 + y)⁶³ − (1 + x + y + xy)⁶³` is identically zero, and its
    /// numerator (4,096 terms) is past the expansion budget: up to 0.49 the
    /// answer `Some(true)` came from three samples; now from the exact grid
    /// test (64 × 64 points).
    ///
    /// Oracle (exact identity): `1 + x + y + xy = (1 + x)(1 + y)`.
    #[test]
    fn a_large_identically_zero_rational_function_is_proved_zero() {
        let ctx = tctx();
        let e = ctx
            .parse("(1 + x)^63*(1 + y)^63 - (1 + x + y + x*y)^63")
            .unwrap();
        assert!(!e.is_zero_structural());
        {
            let inner = e.inner.read();
            let syms = crate::base::walk::free_symbols(&inner.arena, e.raw_id());
            assert_eq!(
                rf_numerator_is_zero(&inner.arena, e.raw_id(), &syms),
                Truth::Unknown
            );
            assert_eq!(
                rf_numerator_vanishes_on_grid(&inner.arena, e.raw_id(), &syms),
                Truth::True
            );
        }
        assert_eq!(rational_function_is_zero_q(&e), Truth::True);
    }
}
