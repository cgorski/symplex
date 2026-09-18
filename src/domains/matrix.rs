//! Symbolic matrix type.
//!
//! Provides [`Matrix`], a dense matrix of symbolic expressions with
//! exact arithmetic: construction, arithmetic operators, determinant
//! (Bareiss / Berkowitz), inverse, linear solving, characteristic
//! polynomial, eigenvalues/eigenvectors, diagonalization, Jordan form,
//! matrix exponential, LU/RREF, subspaces and code generation.
//!
//! Additional decompositions (QR, LDLᵀ, Gram–Schmidt), structure tests
//! (`is_symmetric`, `is_positive_definite`, …) and norms live in
//! [`matrix_decomp`](crate::domains::matrix_decomp) but are all methods
//! on `Matrix`.
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
use crate::base::errors::SymplexError;
use std::fmt;
use tracing::{debug, trace, warn};

// Re-export codegen option types so users can access them from the public
// `symplex::matrix` module (the `codegen` module itself is pub(crate)).
pub use crate::output::codegen::{CodegenOptions, MathBackend, Precision};

/// A dense matrix of symbolic expressions.
///
/// Elements are stored in row-major order as `Vec<Vec<Ex>>`.
///
/// Equality (`==`) is *structural*: two matrices are equal when they have
/// the same shape and every pair of entries is the same canonical
/// expression in the same [`Context`].  Use
/// [`equals`](Self::equals) for a mathematical (simplifying) comparison.
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
#[derive(Clone, PartialEq, Eq)]
pub struct Matrix {
    rows: Vec<Vec<Ex>>,
    nrows: usize,
    ncols: usize,
}

// ═══════════════════════════════════════════════════════════════════════════
// Error helpers
// ═══════════════════════════════════════════════════════════════════════════

fn invalid(operation: &'static str, reason: impl Into<String>) -> SymplexError {
    SymplexError::InvalidArgument {
        operation,
        reason: reason.into(),
    }
}

fn failed(operation: &'static str, reason: impl Into<String>) -> SymplexError {
    SymplexError::ComputationFailed {
        operation,
        reason: reason.into(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Three-valued scalar helpers (shared with matrix_decomp / quaternion)
// ═══════════════════════════════════════════════════════════════════════════

/// Three-valued zero test for a scalar expression.
///
/// Layers: structural zero → assumption system → `eval().simplify()` →
/// numeric evaluation for constants.  Returns `None` when the sign cannot
/// be decided symbolically (e.g. a free symbol without assumptions).
pub(crate) fn ex_is_zero(e: &Ex) -> Option<bool> {
    if e.is_zero_structural() {
        return Some(true);
    }
    if let Some(b) = e.is_zero() {
        return Some(b);
    }
    let s = e.eval().simplify();
    if s.is_zero_structural() {
        return Some(true);
    }
    if let Some(b) = s.is_zero() {
        return Some(b);
    }
    if s.is_constant() {
        if let Ok((re, im)) = s.eval_complex64() {
            if re == 0.0 && im == 0.0 {
                return Some(true);
            }
            if re.abs() > 1e-12 || im.abs() > 1e-12 {
                return Some(false);
            }
        }
    }
    None
}

/// Three-valued "is strictly positive" test for a scalar expression.
pub(crate) fn ex_is_positive(e: &Ex) -> Option<bool> {
    if let Some(b) = e.is_positive() {
        return Some(b);
    }
    let s = e.eval().simplify();
    if let Some(b) = s.is_positive() {
        return Some(b);
    }
    if s.is_constant()
        && let Ok(v) = s.eval_f64()
    {
        return Some(v > 0.0);
    }
    None
}

/// Three-valued "is non-negative" test for a scalar expression.
pub(crate) fn ex_is_nonnegative(e: &Ex) -> Option<bool> {
    if let Some(b) = e.is_nonnegative() {
        return Some(b);
    }
    let s = e.eval().simplify();
    if let Some(b) = s.is_nonnegative() {
        return Some(b);
    }
    if s.is_constant()
        && let Ok(v) = s.eval_f64()
    {
        return Some(v >= 0.0);
    }
    None
}

/// Combine three-valued results with logical AND (short-circuit on `false`).
pub(crate) fn all3(iter: impl IntoIterator<Item = Option<bool>>) -> Option<bool> {
    let mut unknown = false;
    for v in iter {
        match v {
            Some(false) => return Some(false),
            None => unknown = true,
            Some(true) => {}
        }
    }
    if unknown { None } else { Some(true) }
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
        Ok(Matrix { rows, nrows, ncols })
    }

    /// Build a matrix from validated parts without re-checking.
    #[inline]
    fn from_rows_unchecked(rows: Vec<Vec<Ex>>) -> Self {
        let nrows = rows.len();
        let ncols = rows.first().map_or(0, Vec::len);
        debug_assert!(nrows > 0 && ncols > 0);
        debug_assert!(rows.iter().all(|r| r.len() == ncols));
        Matrix { rows, nrows, ncols }
    }

    /// Create an `n × m` matrix of zeros.
    ///
    /// # Panics
    ///
    /// Panics if `n == 0` or `m == 0`.
    pub fn zeros(ctx: &Context, n: usize, m: usize) -> Self {
        assert!(n > 0 && m > 0, "Matrix::zeros: dimensions must be positive");
        let zero = ctx.zero();
        let rows = (0..n)
            .map(|_| (0..m).map(|_| zero.clone()).collect())
            .collect();
        Matrix {
            rows,
            nrows: n,
            ncols: m,
        }
    }

    /// Create an `n × n` identity matrix.
    ///
    /// # Panics
    ///
    /// Panics if `n == 0`.
    pub fn identity(ctx: &Context, n: usize) -> Self {
        assert!(n > 0, "Matrix::identity: dimension must be positive");
        let one = ctx.one();
        let zero = ctx.zero();
        let rows = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| if i == j { one.clone() } else { zero.clone() })
                    .collect()
            })
            .collect();
        Matrix {
            rows,
            nrows: n,
            ncols: n,
        }
    }

    /// Create an `n × m` matrix from a closure `f(i, j)`.
    ///
    /// # Panics
    ///
    /// Panics if `n == 0` or `m == 0`.
    pub fn from_fn(n: usize, m: usize, mut f: impl FnMut(usize, usize) -> Ex) -> Self {
        assert!(
            n > 0 && m > 0,
            "Matrix::from_fn: dimensions must be positive"
        );
        let rows = (0..n).map(|i| (0..m).map(|j| f(i, j)).collect()).collect();
        Matrix {
            rows,
            nrows: n,
            ncols: m,
        }
    }

    /// Create a 1×n row vector from a list of elements.
    ///
    /// # Panics
    ///
    /// Panics if `elems` is empty.
    pub fn row_vector(elems: Vec<Ex>) -> Self {
        assert!(
            !elems.is_empty(),
            "Matrix::row_vector: need at least one element"
        );
        let ncols = elems.len();
        Matrix {
            rows: vec![elems],
            nrows: 1,
            ncols,
        }
    }

    /// Create an n×1 column vector from a list of elements.
    ///
    /// # Panics
    ///
    /// Panics if `elems` is empty.
    pub fn col_vector(elems: Vec<Ex>) -> Self {
        assert!(
            !elems.is_empty(),
            "Matrix::col_vector: need at least one element"
        );
        let nrows = elems.len();
        let rows = elems.into_iter().map(|e| vec![e]).collect();
        Matrix {
            rows,
            nrows,
            ncols: 1,
        }
    }

    /// Create a square diagonal matrix from a slice of diagonal entries.
    ///
    /// The accessor returning the diagonal of an existing matrix is
    /// [`diagonal`](Self::diagonal).
    ///
    /// # Panics
    ///
    /// Panics if `entries` is empty.
    pub fn diag(entries: &[Ex]) -> Matrix {
        let n = entries.len();
        assert!(n > 0, "Matrix::diag: entries must be non-empty");
        let zero = entries[0].context().zero();
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
        Matrix::from_rows_unchecked(rows)
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
        Ok(Matrix { rows, nrows, ncols })
    }
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
    /// Panics if `i >= nrows` or `j >= ncols`.
    #[inline]
    pub fn get_mut(&mut self, i: usize, j: usize) -> &mut Ex {
        assert!(
            i < self.nrows && j < self.ncols,
            "Index ({i}, {j}) out of bounds for {}×{} matrix",
            self.nrows,
            self.ncols
        );
        &mut self.rows[i][j]
    }

    /// Overwrite element `(i, j)`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= nrows` or `j >= ncols`.
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, value: Ex) {
        *self.get_mut(i, j) = value;
    }

    /// Immutable reference to row `i` as a slice.
    ///
    /// # Panics
    ///
    /// Panics if `i >= nrows`.
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

    /// Column `j` as an owned `Vec<Ex>`.
    ///
    /// # Panics
    ///
    /// Panics if `j >= ncols`.
    pub fn col(&self, j: usize) -> Vec<Ex> {
        assert!(
            j < self.ncols,
            "Column index {j} out of bounds for {}×{} matrix",
            self.nrows,
            self.ncols
        );
        self.rows.iter().map(|r| r[j].clone()).collect()
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
    /// # Panics
    ///
    /// Panics if either range is empty or extends past the matrix bounds,
    /// mirroring slice indexing.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let m = matrix![ctx, [1, 2, 3], [4, 5, 6], [7, 8, 9]];
    /// assert_eq!(m.submatrix(1..3, 0..2), matrix![ctx, [4, 5], [7, 8]]);
    /// ```
    pub fn submatrix(&self, rows: std::ops::Range<usize>, cols: std::ops::Range<usize>) -> Matrix {
        assert!(
            rows.start < rows.end && rows.end <= self.nrows,
            "submatrix: row range {rows:?} invalid for {} rows",
            self.nrows
        );
        assert!(
            cols.start < cols.end && cols.end <= self.ncols,
            "submatrix: column range {cols:?} invalid for {} columns",
            self.ncols
        );
        let data: Vec<Vec<Ex>> = rows.map(|i| self.rows[i][cols.clone()].to_vec()).collect();
        Matrix::from_rows_unchecked(data)
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
        all3(
            self.iter()
                .zip(other.iter())
                .map(|(a, b)| ex_is_zero(&(a - b))),
        )
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

    /// `true` if every entry is a rational number literal.
    fn all_numeric(&self) -> bool {
        self.iter().all(|e| e.expr_type() == ExprType::Number)
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
    fn hide_dummy(&self, e: &Ex, dummy: &Ex) -> Ex {
        if !e.contains(dummy) {
            return e.clone();
        }
        let lam = self.ctx().symbol("λ");
        if self.contains(&lam) {
            return e.clone();
        }
        e.subs(dummy, &lam)
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
        Matrix {
            nrows: self.ncols,
            ncols: self.nrows,
            rows,
        }
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
        Matrix {
            nrows: self.nrows,
            ncols: self.ncols,
            rows,
        }
    }

    /// Scalar multiplication: multiply every element by `scalar`.
    pub fn scale(&self, scalar: &Ex) -> Matrix {
        self.map(|elem| scalar * elem)
    }

    /// Matrix multiplication (`self * other`).
    ///
    /// `self` is `n × p` and `other` is `p × m`; the result is `n × m`.
    /// Each element is computed symbolically:
    /// `result[i][j] = Σ_k self[i][k] * other[k][j]`.
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
        Ok(Matrix {
            nrows: self.nrows,
            ncols: other.ncols,
            rows,
        })
    }

    /// Trace: sum of the diagonal elements.
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::InvalidArgument`] if the matrix is not square.
    pub fn trace(&self) -> Result<Ex, SymplexError> {
        self.require_square("trace")?;
        let mut acc = self.rows[0][0].clone();
        for i in 1..self.nrows {
            acc += &self.rows[i][i];
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
    /// Returns [`SymplexError::InvalidArgument`] if the matrix is not square.
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
        let n = self.nrows;
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
            _ if self.all_numeric() => self.det_bareiss(),
            _ => {
                // det(A) = (−1)ⁿ · [constant coefficient of det(λI − A)]
                let coeffs = self.berkowitz_monic();
                let c0 = coeffs[n].clone();
                if n % 2 == 1 { -c0 } else { c0 }
            }
        })
    }

    /// Determinant via Bareiss fraction-free elimination.
    ///
    /// O(n³) element operations.  Every division is exact by Sylvester's
    /// identity, which for rational-number entries means no fractions
    /// ever appear in intermediate results.
    fn det_bareiss(&self) -> Ex {
        let n = self.nrows();
        let mut m: Vec<Vec<Ex>> = self.rows.clone();
        let mut sign = 1i64;
        let mut prev_pivot = self.ctx_one();

        for k in 0..n - 1 {
            let pivot_row = Self::find_bareiss_pivot(&m, k, n);
            match pivot_row {
                None => return self.ctx_zero(),
                Some(pr) if pr != k => {
                    m.swap(k, pr);
                    sign = -sign;
                }
                _ => {}
            }
            let pivot = m[k][k].clone();
            for i in (k + 1)..n {
                for j in (k + 1)..n {
                    let numer = &(&pivot * &m[i][j]) - &(&m[i][k] * &m[k][j]);
                    m[i][j] = (&numer / &prev_pivot).eval();
                }
                m[i][k] = self.ctx_zero();
            }
            prev_pivot = pivot;
        }

        let det = m[n - 1][n - 1].clone();
        if sign < 0 { -det } else { det }
    }

    /// Find a non-zero pivot in column k, rows k..n (structural, then evaluated).
    #[allow(clippy::needless_range_loop)]
    fn find_bareiss_pivot(m: &[Vec<Ex>], k: usize, n: usize) -> Option<usize> {
        for i in k..n {
            if !m[i][k].is_zero_structural() {
                return Some(i);
            }
        }
        for i in k..n {
            if !m[i][k].eval().is_zero_structural() {
                return Some(i);
            }
        }
        None
    }

    /// Berkowitz's division-free characteristic polynomial.
    ///
    /// Returns the coefficients of `det(λI − A)`, **highest degree first**:
    /// `[1, c_{n−1}, …, c_0]` (length `n + 1`).  Every coefficient is a
    /// fully expanded polynomial in the matrix entries.  Requires a square
    /// matrix (checked by callers).
    fn berkowitz_monic(&self) -> Vec<Ex> {
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
                                acc = acc + &self.rows[i][j] * &c[idx];
                            }
                            acc.expand()
                        })
                        .collect();
                    c = next;
                }
                let mut rc = zero.clone();
                for (ri, ci) in r.iter().zip(c.iter()) {
                    rc = rc + *ri * ci;
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
                        acc = acc + &diags[i - j] * v;
                    }
                }
                next_vec.push(acc.expand());
            }
            vec = next_vec;
        }
        vec
    }

    /// Apply a function to every element, producing a new matrix.
    pub fn map(&self, mut f: impl FnMut(&Ex) -> Ex) -> Matrix {
        let rows = self
            .rows
            .iter()
            .map(|row| row.iter().map(&mut f).collect())
            .collect();
        Matrix {
            nrows: self.nrows,
            ncols: self.ncols,
            rows,
        }
    }

    /// Apply `f(i, j, &a_ij)` to every element, producing a new matrix.
    pub fn map_indexed(&self, mut f: impl FnMut(usize, usize, &Ex) -> Ex) -> Matrix {
        let rows = self
            .rows
            .iter()
            .enumerate()
            .map(|(i, row)| row.iter().enumerate().map(|(j, e)| f(i, j, e)).collect())
            .collect();
        Matrix {
            nrows: self.nrows,
            ncols: self.ncols,
            rows,
        }
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
    /// - [`SymplexError::ComputationFailed`] if the determinant is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let a = matrix![ctx, [2, 1], [1, 1]];
    /// let inv = a.inv().unwrap();
    /// assert_eq!(&a * &inv, Matrix::identity(&ctx, 2));
    /// assert!(matrix![ctx, [1, 2], [2, 4]].inv().is_err());
    /// ```
    pub fn inv(&self) -> Result<Matrix, SymplexError> {
        self.require_square("inv")?;
        let d = self.det()?;
        if ex_is_zero(&d) == Some(true) {
            return Err(failed("inv", "matrix is singular (determinant is zero)"));
        }
        let n = self.nrows;
        if n == 1 {
            let one_over_det = &self.ctx_one() / &d;
            return Ok(Matrix::from_rows_unchecked(vec![vec![one_over_det]]));
        }
        if self.all_numeric() {
            let eye = Matrix::identity(&self.ctx(), n);
            return self.solve(&eye);
        }
        let adj = self.adjugate()?;
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
    ///   (no unique solution).
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

        let augmented = Matrix::hstack(&[self, b])?;
        let (rref_mat, pivots) = augmented.rref();

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
    /// `AᵀA x = Aᵀb`.
    ///
    /// Requires `A` to have full column rank (so that `AᵀA` is invertible).
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
        let at = self.transpose();
        let ata = at.matmul(self)?;
        let atb = at.matmul(b)?;
        ata.solve(&atb).map_err(|_| {
            failed(
                "solve_least_squares",
                "AᵀA is singular (A does not have full column rank)",
            )
        })
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
        let n = self.nrows;
        let monic = self.berkowitz_monic(); // det(λI − A), highest first
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
            acc = acc + c * &var.powi(k as i64);
        }
        Ok(acc.expand())
    }

    /// Eigenvalues with algebraic multiplicities: `[(λ, multiplicity), …]`.
    ///
    /// The characteristic polynomial is factored over ℤ (exact
    /// multiplicities); each irreducible factor is then solved.  Roots of
    /// irreducible factors of degree ≥ 5 are returned as `RootOf`
    /// expressions whose bound variable displays as `λ`.  For 1×1 and 2×2
    /// matrices with symbolic entries the closed-form (quadratic) formula
    /// is used.
    ///
    /// If the solver cannot find every root, the multiplicities sum to less
    /// than `n` and a warning is logged.
    ///
    /// # Errors
    ///
    /// - [`SymplexError::InvalidArgument`] if the matrix is not square.
    /// - [`SymplexError::ComputationFailed`] if no eigenvalue could be found.
    pub fn eigenvals_with_multiplicity(&self) -> Result<Vec<(Ex, usize)>, SymplexError> {
        self.require_square("eigenvals")?;
        let lam = self.fresh_symbol("lambda");
        let pairs = self.eigen_pairs(&lam)?;
        Ok(pairs
            .into_iter()
            .map(|(v, m)| (self.hide_dummy(&v, &lam), m))
            .collect())
    }

    /// Internal: eigenvalue/multiplicity pairs expressed with the dummy `lam`.
    fn eigen_pairs(&self, lam: &Ex) -> Result<Vec<(Ex, usize)>, SymplexError> {
        let n = self.nrows;
        let coeffs = self.char_poly_coeffs()?;
        let cp = {
            let mut acc = coeffs[0].clone();
            for (k, c) in coeffs.iter().enumerate().skip(1) {
                if !c.is_zero_structural() {
                    acc = acc + c * &lam.powi(k as i64);
                }
            }
            acc.expand()
        };

        let mut pairs = eigvals_with_multiplicity(&cp, lam);
        let mut total: usize = pairs.iter().map(|(_, m)| *m).sum();

        // Closed-form fallback for low degree with symbolic coefficients.
        if total < n && n <= 2 {
            pairs = low_degree_roots(&coeffs);
            total = pairs.iter().map(|(_, m)| *m).sum();
        }

        if pairs.is_empty() {
            return Err(failed(
                "eigenvals",
                "could not solve the characteristic polynomial; \
                 symbolic matrices larger than 2×2 need a factorable characteristic polynomial",
            ));
        }
        if total < n {
            warn!(
                "eigenvals: found algebraic multiplicity {} for a {}×{} matrix — \
                 characteristic polynomial may have factors beyond solver capability",
                total, n, n
            );
        }
        Ok(pairs)
    }

    /// Eigenvalues, repeated according to algebraic multiplicity.
    ///
    /// For an `n × n` matrix whose characteristic polynomial the solver
    /// can fully handle this list has exactly `n` entries.  See
    /// [`eigenvals_with_multiplicity`](Self::eigenvals_with_multiplicity)
    /// for the grouped form and for the handling of unsolvable factors.
    ///
    /// # Errors
    ///
    /// - [`SymplexError::InvalidArgument`] if the matrix is not square.
    /// - [`SymplexError::ComputationFailed`] if no eigenvalue could be found.
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
    /// of vectors is the geometric multiplicity.
    ///
    /// # Errors
    ///
    /// - [`SymplexError::InvalidArgument`] if the matrix is not square.
    /// - [`SymplexError::ComputationFailed`] if no eigenvalue could be found.
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
        let eigen_pairs = self.eigen_pairs(&lam)?;
        let eye = Matrix::identity(&self.ctx(), n);

        let mut result = Vec::new();
        for (eigenval, alg_mult) in &eigen_pairs {
            let a_minus_lambda_i = self.sub(&eye.scale(eigenval))?;
            let vecs = a_minus_lambda_i.nullspace_semantic();
            trace!(
                alg_mult,
                geom_mult = vecs.len(),
                "eigenvects: eigenvalue has alg_mult={}, geom_mult={}",
                alg_mult,
                vecs.len()
            );
            let vecs = vecs
                .into_iter()
                .map(|v| v.map(|e| self.hide_dummy(e, &lam)))
                .collect();
            result.push((self.hide_dummy(eigenval, &lam), *alg_mult, vecs));
        }
        Ok(result)
    }

    /// Is the matrix diagonalizable (three-valued)?
    ///
    /// - `Some(true)` — every eigenvalue's geometric multiplicity equals
    ///   its algebraic multiplicity.
    /// - `Some(false)` — non-square, or some eigenvalue is defective
    ///   (decided only for matrices without free symbols).
    /// - `None` — the eigenvalue solver could not find all eigenvalues, or
    ///   the entries are symbolic and an eigenspace dimension could not be
    ///   established.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// assert_eq!(matrix![ctx, [1, 0], [0, 2]].is_diagonalizable(), Some(true));
    /// assert_eq!(matrix![ctx, [1, 1], [0, 1]].is_diagonalizable(), Some(false));
    /// ```
    pub fn is_diagonalizable(&self) -> Option<bool> {
        if !self.is_square() {
            return Some(false);
        }
        let eigvs = self.eigenvects().ok()?;
        let total_alg: usize = eigvs.iter().map(|(_, m, _)| *m).sum();
        if total_alg != self.nrows {
            return None;
        }
        if eigvs.iter().all(|(_, m, v)| v.len() == *m) {
            return Some(true);
        }
        // A missing eigenvector is conclusive only when zero-tests were
        // decidable, i.e. for constant matrices.
        if self.iter().all(Ex::is_constant) {
            Some(false)
        } else {
            None
        }
    }

    /// Diagonalize: find invertible `P` and diagonal `D` with `A = P D P⁻¹`.
    ///
    /// `P` has the eigenvectors as columns; `D` carries the eigenvalues in
    /// the same order.
    ///
    /// # Errors
    ///
    /// - [`SymplexError::InvalidArgument`] if the matrix is not square.
    /// - [`SymplexError::ComputationFailed`] if the matrix is not
    ///   diagonalizable or the eigenvalue solver could not find all
    ///   eigenvalues.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let m = matrix![ctx, [2, 1], [0, 3]];
    /// let (p, d) = m.diagonalize().unwrap();
    /// let back = (&(&p * &d) * &p.inv().unwrap()).simplify();
    /// assert_eq!(back, m);
    /// ```
    pub fn diagonalize(&self) -> Result<(Matrix, Matrix), SymplexError> {
        self.require_square("diagonalize")?;
        debug!(
            "diagonalize: attempting for {}×{} matrix",
            self.nrows, self.ncols
        );
        let eigvs = self.eigenvects()?;
        let n = self.nrows;
        let mut total_vecs = 0usize;
        for (_, alg_mult, vecs) in &eigvs {
            if vecs.len() != *alg_mult {
                return Err(failed(
                    "diagonalize",
                    "matrix is not diagonalizable: geometric multiplicity \
                     does not equal algebraic multiplicity for all eigenvalues",
                ));
            }
            total_vecs += vecs.len();
        }
        if total_vecs != n {
            return Err(failed(
                "diagonalize",
                "matrix is not diagonalizable: insufficient eigenvectors found",
            ));
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
        let d = Matrix::diag(&diag_entries);
        Ok((p, d))
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
    ///   find all eigenvalues.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// // Defective: eigenvalue 2 with algebraic mult 2, geometric mult 1
    /// let m = matrix![ctx, [2, 1, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]];
    /// let (p, j) = m.jordan_form().unwrap();
    /// assert_eq!((&(&p * &j) * &p.inv().unwrap()).simplify(), m);
    /// ```
    pub fn jordan_form(&self) -> Result<(Matrix, Matrix), SymplexError> {
        self.require_square("jordan_form")?;
        let n = self.nrows;
        debug!(n, "jordan_form: computing for {}×{} matrix", n, n);
        let eye = Matrix::identity(&self.ctx(), n);

        let eigvs = self.eigenvects()?;

        // Fast path: diagonalizable.
        let total_vecs: usize = eigvs.iter().map(|(_, _, v)| v.len()).sum();
        let all_match = eigvs.iter().all(|(_, m, v)| v.len() == *m);
        if all_match && total_vecs == n {
            return self.diagonalize();
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
                let nullity = n - power.rank_semantic();
                let last = *chain.last().unwrap_or(&0);
                if nullity == last || nullity >= *alg_mult {
                    if nullity > last {
                        chain.push(nullity);
                    }
                    break;
                }
                chain.push(nullity);
                power = power.matmul(&a_minus_lambda)?;
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
        Ok((p, j))
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
            return Ok(Matrix::identity(&self.ctx(), self.nrows));
        }
        if n == 1 {
            return Ok(self.clone());
        }
        let mut result = Matrix::identity(&self.ctx(), self.nrows);
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
                for j in 0..self.ncols {
                    for l in 0..other.ncols {
                        row.push(&self.rows[i][j] * &other.rows[k][l]);
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
        let mut result = Matrix::identity(&ctx, n);
        let mut term = Matrix::identity(&ctx, n);
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
    /// Unlike a numerical library this never falls back to a series
    /// approximation; use [`exp_series`](Self::exp_series) explicitly if
    /// an approximation is acceptable.
    ///
    /// # Errors
    ///
    /// - [`SymplexError::InvalidArgument`] if the matrix is not square.
    /// - [`SymplexError::ComputationFailed`] if the Jordan form cannot be
    ///   computed (eigenvalues not found in closed form).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let m = matrix![ctx, [0, 1], [-1, 0]];
    /// let e = m.matrix_exp().unwrap();
    /// // e^A = [[cos 1, sin 1], [−sin 1, cos 1]]  (entries may be in exponential form)
    /// let (re, im) = e[(0, 1)].eval_complex64().unwrap();
    /// assert!((re - 1f64.sin()).abs() < 1e-12 && im.abs() < 1e-12);
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
        debug!(
            "matrix_exp: computing for {}×{} matrix",
            self.nrows, self.ncols
        );
        let n = self.nrows;
        let ctx = self.ctx();

        let (p, j) = self.jordan_form().map_err(|e| {
            failed(
                "matrix_exp",
                format!(
                    "Jordan form unavailable ({e}); use exp_series(order) for a truncated approximation"
                ),
            )
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
                    let factorial_val = ctx.int(factorial_usize(d) as i64);
                    let mut entry = &exp_lambda / &factorial_val;
                    if d > 0 {
                        if let Some(t) = t {
                            entry = entry * t.powi(d as i64);
                        }
                    }
                    exp_j_rows[col + i][col + jj] = entry;
                }
            }
            col += block_size;
        }
        let exp_j = Matrix::from_rows_unchecked(exp_j_rows);

        let p_inv = p.inv().map_err(|_| {
            failed(
                "matrix_exp",
                "eigenvector matrix is singular (internal inconsistency)",
            )
        })?;
        let result = p.matmul(&exp_j)?.matmul(&p_inv)?;
        Ok(result.map(|e| e.simplify()))
    }

    // ── Pseudo-inverse ─────────────────────────────────────────────────

    /// Moore–Penrose pseudo-inverse via `A⁺ = (AᵀA)⁻¹Aᵀ`.
    ///
    /// Valid for full-column-rank matrices.
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::ComputationFailed`] if `AᵀA` is singular
    /// (the matrix does not have full column rank).
    pub fn pinv(&self) -> Result<Matrix, SymplexError> {
        let at = self.transpose();
        let ata = at.matmul(self)?;
        let ata_inv = ata
            .inv()
            .map_err(|_| failed("pinv", "AᵀA is singular (A does not have full column rank)"))?;
        ata_inv.matmul(&at)
    }
}

/// Recursive cofactor expansion (used for the hard-coded 3×3 path).
fn det_cofactor_inner(m: &[Vec<Ex>]) -> Ex {
    let n = m.len();
    if n == 1 {
        return m[0][0].clone();
    }
    if n == 2 {
        let ad = &m[0][0] * &m[1][1];
        let bc = &m[0][1] * &m[1][0];
        return ad - bc;
    }
    let mut result: Option<Ex> = None;
    for j in 0..n {
        let minor: Vec<Vec<Ex>> = (1..n)
            .map(|row| {
                (0..n)
                    .filter(|&col| col != j)
                    .map(|col| m[row][col].clone())
                    .collect()
            })
            .collect();
        let cofactor = det_cofactor_inner(&minor);
        let term = &m[0][j] * &cofactor;
        result = Some(match result {
            None => {
                if j % 2 == 0 {
                    term
                } else {
                    -term
                }
            }
            Some(acc) => {
                if j % 2 == 0 {
                    acc + term
                } else {
                    acc - term
                }
            }
        });
    }
    result.unwrap_or_else(|| m[0][0].clone())
}

/// Closed-form roots for `c_0 + c_1 λ + c_2 λ²` (degree ≤ 2), with
/// multiplicities.  Used when the generic solver cannot handle symbolic
/// coefficients.
fn low_degree_roots(coeffs: &[Ex]) -> Vec<(Ex, usize)> {
    match coeffs.len() {
        2 => {
            // c0 + c1 λ = 0 → λ = −c0/c1
            let root = (-&coeffs[0] / &coeffs[1]).eval().simplify();
            vec![(root, 1)]
        }
        3 => {
            let (c0, c1, c2) = (&coeffs[0], &coeffs[1], &coeffs[2]);
            let ctx = c0.context();
            let two = ctx.int(2);
            let disc = (&c1.powi(2) - &(&(&ctx.int(4) * c2) * c0)).expand();
            let denom = &two * c2;
            if ex_is_zero(&disc) == Some(true) {
                let root = (-c1 / &denom).eval();
                return vec![(root, 2)];
            }
            // Keep `sqrt(disc)` rather than simplifying to `abs(…)`: as a
            // *set* the two roots are the same either way, and the radical
            // form is friendlier for downstream algebra.
            let sq = disc.sqrt();
            let r1 = (&(-c1 + &sq) / &denom).eval();
            let r2 = (&(-c1 - &sq) / &denom).eval();
            vec![(r1, 1), (r2, 1)]
        }
        _ => Vec::new(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Calculus helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Build the Jacobian matrix of `funcs` with respect to `vars`:
/// `J[i][j] = ∂funcs[i] / ∂vars[j]`.
///
/// # Panics
///
/// Panics if `funcs` or `vars` is empty.
pub fn jacobian(funcs: &[&Ex], vars: &[&Ex]) -> Matrix {
    assert!(!funcs.is_empty(), "jacobian: funcs must be non-empty");
    assert!(!vars.is_empty(), "jacobian: vars must be non-empty");
    let rows: Vec<Vec<Ex>> = funcs
        .iter()
        .map(|fi| vars.iter().map(|vj| fi.diff(vj)).collect())
        .collect();
    Matrix::from_rows_unchecked(rows)
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
            for j in 0..self.ncols {
                if j > 0 {
                    s.push_str(" & ");
                }
                s.push_str(&self.rows[i][j].to_latex());
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
    /// Returns `(L, U, perm)` where `L` is unit lower triangular, `U` is
    /// upper triangular and `perm` is the row permutation (`perm[i]` is
    /// the original index of row `i` of `P·A`).  Pivots are chosen as the
    /// first structurally non-zero entry, so symbolic entries are treated
    /// as non-zero.
    ///
    /// # Errors
    ///
    /// - [`SymplexError::InvalidArgument`] if the matrix is not square.
    /// - [`SymplexError::ComputationFailed`] if the matrix is singular.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let a = matrix![ctx, [4, 3], [6, 3]];
    /// let (l, u, perm) = a.lu().unwrap();
    /// // Rebuild P·A from perm and compare with L·U
    /// let pa = Matrix::new(perm.iter().map(|&i| a.row(i).to_vec()).collect()).unwrap();
    /// assert_eq!((&l * &u).eval(), pa);
    /// assert!(matrix![ctx, [1, 2], [2, 4]].lu().is_err());
    /// ```
    pub fn lu(&self) -> Result<(Matrix, Matrix, Vec<usize>), SymplexError> {
        self.require_square("lu")?;
        let n = self.nrows;

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

        for k in 0..n {
            let mut pivot_row = None;
            for i in k..n {
                if !u[i][k].is_zero_structural() {
                    pivot_row = Some(i);
                    break;
                }
            }
            let Some(pivot_row) = pivot_row else {
                return Err(failed("lu", "matrix is singular (zero pivot column)"));
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
                if u[i][k].is_zero_structural() {
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

        Ok((
            Matrix::from_rows_unchecked(l),
            Matrix::from_rows_unchecked(u),
            perm,
        ))
    }

    /// Row-reduced echelon form via Gauss–Jordan elimination.
    ///
    /// Returns `(rref_matrix, pivot_columns)`.  Uses exact arithmetic;
    /// pivots are the first *structurally* non-zero entries, so symbolic
    /// entries are always treated as non-zero.  (The eigen-family uses a
    /// simplifying zero test internally so that irrational eigenvalues
    /// still yield eigenvectors.)
    pub fn rref(&self) -> (Matrix, Vec<usize>) {
        self.rref_by(&|e: &Ex| e.is_zero_structural())
    }

    /// RREF with a caller-supplied zero test for pivot selection.
    fn rref_by(&self, is_zero: &dyn Fn(&Ex) -> bool) -> (Matrix, Vec<usize>) {
        let nrows = self.nrows;
        let ncols = self.ncols;
        let mut rows: Vec<Vec<Ex>> = self.rows.clone();
        let mut pivots = Vec::new();
        let mut pivot_row = 0;

        for col in 0..ncols {
            if pivot_row >= nrows {
                break;
            }
            let mut found = None;
            for i in pivot_row..nrows {
                if !is_zero(&rows[i][col]) {
                    found = Some(i);
                    break;
                }
            }
            let Some(found) = found else { continue };

            if found != pivot_row {
                rows.swap(pivot_row, found);
            }
            let pivot_val = rows[pivot_row][col].clone();
            for j in 0..ncols {
                rows[pivot_row][j] = &rows[pivot_row][j] / &pivot_val;
            }
            for i in 0..nrows {
                if i == pivot_row {
                    continue;
                }
                if !is_zero(&rows[i][col]) {
                    let factor = rows[i][col].clone();
                    for j in 0..ncols {
                        let term = &factor * &rows[pivot_row][j];
                        rows[i][j] = &rows[i][j] - &term;
                    }
                }
            }
            pivots.push(col);
            pivot_row += 1;
        }

        (Matrix { rows, nrows, ncols }, pivots)
    }

    /// Rank of the matrix (number of pivot columns in RREF).
    pub fn rank(&self) -> usize {
        let (_, pivots) = self.rref();
        pivots.len()
    }

    /// Rank using the simplifying zero test (for eigen computations).
    fn rank_semantic(&self) -> usize {
        self.rref_by(&eigen_zero_test).1.len()
    }

    /// Null space (kernel): basis vectors for `Ax = 0`, as column vectors.
    ///
    /// Empty for a full-column-rank matrix.  Uses structural pivoting like
    /// [`rref`](Self::rref).
    pub fn nullspace(&self) -> Vec<Matrix> {
        self.nullspace_by(&|e: &Ex| e.is_zero_structural())
    }

    /// Null space using the simplifying zero test (for eigen computations).
    fn nullspace_semantic(&self) -> Vec<Matrix> {
        self.nullspace_by(&eigen_zero_test)
    }

    fn nullspace_by(&self, is_zero: &dyn Fn(&Ex) -> bool) -> Vec<Matrix> {
        let (rref_mat, pivots) = self.rref_by(is_zero);
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
            basis.push(Matrix::col_vector(entries));
        }
        basis
    }

    /// Column space basis: the pivot columns of the original matrix.
    pub fn columnspace(&self) -> Vec<Matrix> {
        let (_, pivots) = self.rref();
        pivots
            .iter()
            .map(|&col| Matrix::col_vector(self.col(col)))
            .collect()
    }

    /// Row space basis: the non-zero rows of the RREF, as row vectors.
    pub fn rowspace(&self) -> Vec<Matrix> {
        let (rref_mat, pivots) = self.rref();
        (0..pivots.len())
            .map(|i| Matrix::row_vector(rref_mat.rows[i].clone()))
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

impl Matrix {
    /// Frobenius norm `‖A‖_F = √(Σ |a_ij|²)`.
    ///
    /// Entries are treated as real (squared, not `|·|²`); for complex
    /// entries apply [`adjoint`](Self::adjoint) manually.
    pub fn norm_frobenius(&self) -> Ex {
        let mut sum = self.ctx_zero();
        for elem in self.iter() {
            sum += &(elem * elem);
        }
        sum.sqrt()
    }

    /// Default norm — the Frobenius norm ([`norm_frobenius`](Self::norm_frobenius)).
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
        Ok(Matrix { rows, nrows, ncols })
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
        Ok(Matrix { rows, nrows, ncols })
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
            for i in 0..self.nrows {
                entries.push(self.rows[i][j].clone());
            }
        }
        Matrix::col_vector(entries)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Eigenvalue multiplicity via polynomial factoring
// ═══════════════════════════════════════════════════════════════════════════

/// Extract eigenvalues with algebraic multiplicities from a characteristic
/// polynomial.
///
/// **Primary path:** convert to `Poly`, call `factor_over_z()` (Yun's
/// square-free decomposition) for exact `(factor, multiplicity)` pairs;
/// each factor is solved for its roots, which inherit the multiplicity.
///
/// **Fallback:** if the polynomial has non-rational coefficients,
/// `expr_to_poly` returns `None`; then derivative-based multiplicity
/// detection is used on whatever roots the generic solver finds.
fn eigvals_with_multiplicity(char_poly: &Ex, var: &Ex) -> Vec<(Ex, usize)> {
    if let Some(pairs) = eigvals_via_poly_factor(char_poly, var)
        && !pairs.is_empty()
    {
        debug!(
            count = pairs.len(),
            "eigvals_with_multiplicity: used Poly::factor_over_z path"
        );
        return pairs;
    }
    debug!("eigvals_with_multiplicity: falling back to derivative-based detection");
    eigvals_via_derivative(char_poly, var)
}

/// Primary multiplicity path: convert to Poly, factor, solve each factor.
fn eigvals_via_poly_factor(char_poly: &Ex, var: &Ex) -> Option<Vec<(Ex, usize)>> {
    let inner = char_poly.inner.read();
    let arena = &inner.arena;
    let poly = crate::poly::polybridge::expr_to_poly(arena, char_poly.raw_id(), var.raw_id())?;
    let (_content, factors) = poly.factor_over_z();
    if factors.is_empty() {
        return None;
    }
    drop(inner);

    let mut eigen_pairs: Vec<(Ex, usize)> = Vec::new();
    for (factor, mult) in &factors {
        let degree = factor.degree().unwrap_or(0);
        if degree == 0 {
            continue;
        }
        if degree == 1 {
            let coeffs = factor.coeffs();
            let a = &coeffs[1];
            let b = &coeffs[0];
            let root_val = -(b / a);
            let root_numer: Result<i64, _> = root_val.numer().clone().try_into();
            let root_denom: Result<i64, _> = root_val.denom().clone().try_into();
            if let (Ok(n), Ok(d)) = (root_numer, root_denom) {
                eigen_pairs.push((char_poly.context().rational(n, d), *mult as usize));
                continue;
            }
        }
        // Higher degree (or huge rational root): solve the factor.
        let mut write_inner = char_poly.inner.write();
        let factor_expr =
            crate::poly::polybridge::poly_to_expr(&mut write_inner.arena, factor, var.raw_id());
        drop(write_inner);
        let factor_ex = char_poly.wrap(factor_expr);
        let roots = factor_ex.solve_or_empty(var);
        if roots.is_empty() {
            warn!(
                "eigenvals: irreducible factor of degree {} yielded no roots — \
                 eigenvalues from this factor are missing",
                degree
            );
        }
        trace!(
            degree,
            root_count = roots.len(),
            mult,
            "eigvals_via_poly_factor: degree-{} factor, {} roots, mult {}",
            degree,
            roots.len(),
            mult
        );
        for r in roots {
            eigen_pairs.push((r, *mult as usize));
        }
    }
    Some(eigen_pairs)
}

/// Fallback: derivative-based multiplicity detection.
///
/// For each root `r`, finds the smallest `k` such that `p^(k)(r) ≠ 0`.
fn eigvals_via_derivative(char_poly: &Ex, var: &Ex) -> Vec<(Ex, usize)> {
    let all_roots = char_poly.solve_or_empty(var);
    if all_roots.is_empty() {
        return Vec::new();
    }
    let mut unique_roots: Vec<Ex> = Vec::new();
    for root in &all_roots {
        if !unique_roots.iter().any(|r| r == root) {
            unique_roots.push(root.clone());
        }
    }

    let mut eigen_pairs: Vec<(Ex, usize)> = Vec::new();
    for root in &unique_roots {
        let mut mult = 0usize;
        let mut current = char_poly.clone();
        for k in 0..20 {
            let val = current.subs(var, root).eval().simplify();
            if !val.is_zero_structural() {
                mult = k;
                break;
            }
            if k < 19 {
                current = current.diff(var);
            }
        }
        let mult = if mult == 0 { 1 } else { mult };
        eigen_pairs.push((root.clone(), mult));
    }
    eigen_pairs
}

// ═══════════════════════════════════════════════════════════════════════════
// Jordan form helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Zero test used inside the eigen-family (pivot selection in
/// `A − λI`): structural, then the simplifying three-valued test, then —
/// for constant expressions such as nested radicals that `simplify` cannot
/// collapse — numeric evaluation with a tight tolerance.
///
/// The numeric fallback is sound here because `λ` is an exact eigenvalue:
/// `A − λI` *is* singular, so a pivot candidate that evaluates to ~1e-15
/// is a radical expression for zero, not a genuinely tiny constant.
fn eigen_zero_test(e: &Ex) -> bool {
    if e.is_zero_structural() {
        return true;
    }
    if e.expr_type() == ExprType::Number {
        return false;
    }
    match ex_is_zero(e) {
        Some(b) => b,
        None => {
            e.is_constant()
                && matches!(e.eval_complex64(), Ok((re, im)) if re.abs() < 1e-10 && im.abs() < 1e-10)
        }
    }
}

/// Compute nullspace of `(a_minus_lambda)^power`.
fn jordan_null_power(a_minus_lambda: &Matrix, power: usize) -> Result<Vec<Matrix>, SymplexError> {
    if power == 0 {
        return Ok(Vec::new());
    }
    let mut m = a_minus_lambda.clone();
    for _ in 1..power {
        m = m.matmul(a_minus_lambda)?;
    }
    Ok(m.nullspace_semantic())
}

/// Compute `(a_minus_lambda)^power * vec` where vec is a column vector.
fn matrix_pow_vec(
    a_minus_lambda: &Matrix,
    vec: &Matrix,
    power: usize,
) -> Result<Matrix, SymplexError> {
    let mut result = vec.clone();
    for _ in 0..power {
        result = a_minus_lambda.matmul(&result)?;
    }
    Ok(result)
}

/// Compute n! for small n (used by the matrix-exponential Jordan block formula).
fn factorial_usize(n: usize) -> usize {
    (1..=n).product::<usize>().max(1)
}

/// Pick a vector from `candidates` that is linearly independent from all
/// vectors in `exclude`.  Uses RREF to check independence.
fn pick_independent_vec(
    candidates: &[Matrix],
    exclude: &[&Matrix],
) -> Result<Option<Matrix>, SymplexError> {
    if candidates.is_empty() {
        return Ok(None);
    }
    if exclude.is_empty() {
        return Ok(Some(candidates[0].clone()));
    }
    for candidate in candidates {
        let mut cols: Vec<&Matrix> = exclude.to_vec();
        cols.push(candidate);
        let combined = Matrix::hstack(&cols)?;
        if combined.rank_semantic() == cols.len() {
            return Ok(Some(candidate.clone()));
        }
    }
    Ok(None)
}

// ═══════════════════════════════════════════════════════════════════════════
// Free-standing vector operations
// ═══════════════════════════════════════════════════════════════════════════

/// Cross product of two 3×1 column vectors.
///
/// # Panics
///
/// Panics if either argument is not a 3×1 matrix.
pub fn cross(a: &Matrix, b: &Matrix) -> Matrix {
    assert!(
        a.nrows() == 3 && a.ncols() == 1,
        "cross: first argument must be a 3×1 column vector, got {}×{}",
        a.nrows(),
        a.ncols()
    );
    assert!(
        b.nrows() == 3 && b.ncols() == 1,
        "cross: second argument must be a 3×1 column vector, got {}×{}",
        b.nrows(),
        b.ncols()
    );
    let (a0, a1, a2) = (a.get(0, 0), a.get(1, 0), a.get(2, 0));
    let (b0, b1, b2) = (b.get(0, 0), b.get(1, 0), b.get(2, 0));
    Matrix::col_vector(vec![
        a1 * b2 - a2 * b1,
        a2 * b0 - a0 * b2,
        a0 * b1 - a1 * b0,
    ])
}

/// Dot product of two column vectors (n×1 matrices).
///
/// # Panics
///
/// Panics if either argument is not a column vector or lengths differ.
pub fn dot(a: &Matrix, b: &Matrix) -> Ex {
    assert_eq!(a.ncols(), 1, "dot: first argument must be a column vector");
    assert_eq!(b.ncols(), 1, "dot: second argument must be a column vector");
    assert_eq!(
        a.nrows(),
        b.nrows(),
        "dot: vectors must have the same length ({} vs {})",
        a.nrows(),
        b.nrows()
    );
    let mut sum = a.get(0, 0) * b.get(0, 0);
    for i in 1..a.nrows() {
        sum += &(a.get(i, 0) * b.get(i, 0));
    }
    sum
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
        let m = Matrix::identity(&ctx, 2);
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
        let m = Matrix::identity(&ctx, 3);
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
        let z = Matrix::zeros(&ctx, 3, 3);
        assert_eq!(format!("{}", z.get(1, 1)), "0");

        let m = Matrix::from_fn(2, 2, |i, j| ctx.int((i * 2 + j + 1) as i64));
        assert_eq!(format!("{}", m.get(0, 0)), "1");
        assert_eq!(format!("{}", m.get(0, 1)), "2");
        assert_eq!(format!("{}", m.get(1, 0)), "3");
        assert_eq!(format!("{}", m.get(1, 1)), "4");
    }

    #[test]
    fn row_and_col_vectors() {
        let ctx = tctx();
        let rv = Matrix::row_vector(vec![ctx.int(1), ctx.int(2), ctx.int(3)]);
        assert_eq!(rv.shape(), (1, 3));
        assert_eq!(format!("{}", rv.get(0, 1)), "2");

        let cv = Matrix::col_vector(vec![ctx.int(10), ctx.int(20)]);
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
        let a = Matrix::identity(&ctx, 2);
        let b = Matrix::row_vector(vec![ctx.int(5), ctx.int(6), ctx.int(7)]);
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
        let mut m = Matrix::zeros(&ctx, 2, 2);
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
        let s = m.submatrix(0..2, 1..3);
        assert_eq!(s, Matrix::from_i64(&ctx, &[&[2, 3], &[5, 6]]).unwrap());
        let mm = m.minor_matrix(1, 1).unwrap();
        assert_eq!(mm, Matrix::from_i64(&ctx, &[&[1, 3], &[7, 9]]).unwrap());
        assert_eq!(m.minor(1, 1).unwrap(), ctx.int(-12));
        assert!(m.minor_matrix(3, 0).is_err());
    }

    #[test]
    fn try_get_works() {
        let ctx = tctx();
        let m = Matrix::zeros(&ctx, 2, 2);
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
        let m = Matrix::identity(&ctx, 2);
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
        let m = Matrix::identity(&ctx, 2);
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
        let rv = Matrix::row_vector(vec![ctx.int(2), ctx.int(3)]);
        let cv = Matrix::col_vector(vec![ctx.int(4), ctx.int(5)]);
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
            expected = expected + m.get(0, j) * &m.cofactor(0, j).unwrap();
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
        let j = jacobian(&[&f1, &f2], &[&x, &y]);
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
        let b = Matrix::diag(&[ctx.int(2), ctx.int(3), ctx.int(4)]);
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
        });
        let start = std::time::Instant::now();
        let cp = m.char_poly(&lam).unwrap();
        assert!(start.elapsed().as_secs() < 5, "char_poly too slow");
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
        let mut acc = Matrix::zeros(&ctx, 5, 5);
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
        let start = std::time::Instant::now();
        let coeffs = m.char_poly_coeffs().unwrap();
        assert!(
            start.elapsed().as_secs() < 20,
            "5×5 symbolic char poly took {:?}",
            start.elapsed()
        );
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
        let m = Matrix::diag(&[ctx.int(5), ctx.int(-3)]);
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
        let (p, d) = m.diagonalize().unwrap();
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
        let m = Matrix::diag(&[ctx.int(1), ctx.int(2), ctx.int(3)]);
        assert_eq!(m.is_diagonalizable(), Some(true));
    }

    // ── Jordan form tests ──────────────────────────────────────────

    #[test]
    fn jordan_form_diagonal() {
        let ctx = tctx();
        let m = Matrix::diag(&[ctx.int(1), ctx.int(2), ctx.int(3)]);
        let (p, j) = m.jordan_form().unwrap();
        assert_eq!(j.nrows(), 3);
        let p_inv = p.inv().unwrap();
        let reconstructed = p.matmul(&j).unwrap().matmul(&p_inv).unwrap();
        assert_zero_matrix(&(&reconstructed - &m), "P·J·P⁻¹ = M");
    }

    #[test]
    fn jordan_form_defective_2x2() {
        let ctx = tctx();
        let m = Matrix::from_i64(&ctx, &[&[1, 1], &[0, 1]]).unwrap();
        let (p, j) = m.jordan_form().unwrap();
        assert_eq!(j.nrows(), 2);
        assert!(j.get(0, 1).is_one_structural(), "J[0,1] should be 1");
        let reconstructed = p.matmul(&j).unwrap().matmul(&p.inv().unwrap()).unwrap();
        assert_zero_matrix(&(&reconstructed - &m), "P·J·P⁻¹ = M");
    }

    #[test]
    fn jordan_form_upper_triangular_distinct() {
        let ctx = tctx();
        let m = Matrix::from_i64(&ctx, &[&[2, 1], &[0, 3]]).unwrap();
        let (p, j) = m.jordan_form().unwrap();
        assert_eq!(j.nrows(), 2);
        let reconstructed = p.matmul(&j).unwrap().matmul(&p.inv().unwrap()).unwrap();
        assert_zero_matrix(&(&reconstructed - &m), "P·J·P⁻¹ = M");
    }

    // ── Matrix exponential tests ───────────────────────────────────

    #[test]
    fn matrix_exp_identity() {
        let ctx = tctx();
        let m = Matrix::identity(&ctx, 2);
        let result = m.matrix_exp().unwrap();
        assert_eq!(result.get(0, 0), &ctx.e());
        assert!(result.get(0, 1).is_zero_structural());
    }

    #[test]
    fn matrix_exp_zero() {
        let ctx = tctx();
        let m = Matrix::zeros(&ctx, 2, 2);
        let result = m.matrix_exp().unwrap();
        assert_eq!(result, Matrix::identity(&ctx, 2));
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
        // Compare with [[cos t, sin t], [−sin t, cos t]] numerically at t = 0.7
        let at = et.subs(&t, &ctx.rational(7, 10));
        let expect = [[0.7f64.cos(), 0.7f64.sin()], [-0.7f64.sin(), 0.7f64.cos()]];
        for i in 0..2 {
            for j in 0..2 {
                let (re, im) = at.get(i, j).eval_complex64().unwrap();
                assert!((re - expect[i][j]).abs() < 1e-12, "({i},{j}) re={re}");
                assert!(im.abs() < 1e-12, "({i},{j}) im={im}");
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
        let (l, u, perm) = a.lu().unwrap();
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
        assert_eq!((&a * &inv).eval(), Matrix::identity(&ctx, 3));
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
        let m = Matrix::row_vector(vec![ctx.int(1), ctx.int(2), ctx.int(3)]);
        assert_eq!(format!("{m}"), "[[1, 2, 3]]");
    }

    // ── Error tests (non-square / shape mismatch) ──────────────────────

    #[test]
    fn add_mismatched_shapes_returns_err() {
        let ctx = tctx();
        let a = Matrix::zeros(&ctx, 2, 3);
        let b = Matrix::zeros(&ctx, 3, 2);
        let result = a.add(&b);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("add"), "{err_msg}");
    }

    #[test]
    fn matmul_incompatible_returns_err() {
        let ctx = tctx();
        let a = Matrix::zeros(&ctx, 2, 3);
        let b = Matrix::zeros(&ctx, 2, 3);
        let result = a.matmul(&b);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("multiply"), "{err_msg}");
    }

    #[test]
    fn trace_non_square_returns_err() {
        let ctx = tctx();
        let m = Matrix::zeros(&ctx, 2, 3);
        let result = m.trace();
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("square"));
    }

    #[test]
    fn det_non_square_returns_err() {
        let ctx = tctx();
        let m = Matrix::zeros(&ctx, 2, 3);
        let result = m.det();
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("square"));
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn get_out_of_bounds_panics() {
        let ctx = Context::new();
        let m = Matrix::zeros(&ctx, 2, 2);
        let _ = m.get(2, 0);
    }

    #[test]
    #[should_panic(expected = "Matrix + Matrix")]
    fn operator_add_mismatch_panics() {
        let ctx = Context::new();
        let _ = &Matrix::zeros(&ctx, 2, 2) + &Matrix::zeros(&ctx, 3, 3);
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
        assert_eq!(ex_is_zero(&ctx.int(0)), Some(true));
        assert_eq!(ex_is_zero(&ctx.int(2)), Some(false));
        assert_eq!(ex_is_zero(&(ctx.int(2).sqrt() - 1)), Some(false));
        assert_eq!(
            ex_is_zero(&(&x.sin().powi(2) + &x.cos().powi(2) - 1)),
            Some(true)
        );
        assert_eq!(ex_is_zero(&x), None);
        assert_eq!(ex_is_positive(&(ctx.int(2).sqrt() - 1)), Some(true));
        assert_eq!(
            ex_is_positive(&(ctx.int(1) - ctx.int(2).sqrt())),
            Some(false)
        );
        assert_eq!(ex_is_positive(&x), None);
        assert_eq!(ex_is_nonnegative(&ctx.int(0)), Some(true));
        assert_eq!(all3([Some(true), None]), None);
        assert_eq!(all3([Some(true), None, Some(false)]), Some(false));
        assert_eq!(all3([Some(true), Some(true)]), Some(true));
    }
}
