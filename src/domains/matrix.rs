//! Symbolic matrix type.
//!
//! Provides [`Matrix`], a dense matrix of symbolic expressions with
//! operations: construction, display, transpose, addition, scalar
//! multiplication, matrix multiplication, determinant, and trace.

use crate::base::errors::SymplexError;
use crate::api::expr::Ex;
use std::fmt;
use tracing::{debug, trace, warn};

// Re-export codegen option types so users can access them from the public
// `symplex::matrix` module (the `codegen` module itself is pub(crate)).
pub use crate::output::codegen::{CodegenOptions, MathBackend, Precision};

/// A dense matrix of symbolic expressions.
///
/// Elements are stored in row-major order as `Vec<Vec<Ex>>`.
#[derive(Clone)]
pub struct Matrix {
    rows: Vec<Vec<Ex>>,
    nrows: usize,
    ncols: usize,
}

// ═══════════════════════════════════════════════════════════════════════════
// Constructors
// ═══════════════════════════════════════════════════════════════════════════

impl Matrix {
    /// Create a matrix from nested `Vec`s (row-major).
    ///
    /// # Panics
    ///
    /// Panics if `rows` is empty, any row is empty, or rows have
    /// inconsistent lengths.
    pub fn new(rows: Vec<Vec<Ex>>) -> Self {
        assert!(!rows.is_empty(), "Matrix must have at least one row");
        let ncols = rows[0].len();
        assert!(ncols > 0, "Matrix must have at least one column");
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(
                row.len(),
                ncols,
                "Row {i} has length {} but expected {ncols}",
                row.len()
            );
        }
        let nrows = rows.len();
        Matrix { rows, nrows, ncols }
    }

    /// Create an `n × m` matrix of zeros.
    pub fn zeros(n: usize, m: usize) -> Self {
        assert!(n > 0 && m > 0, "Matrix dimensions must be positive");
        let rows = (0..n)
            .map(|_| (0..m).map(|_| Ex::zero()).collect())
            .collect();
        Matrix {
            rows,
            nrows: n,
            ncols: m,
        }
    }

    /// Create an `n × n` identity matrix.
    pub fn identity(n: usize) -> Self {
        assert!(n > 0, "Identity matrix dimension must be positive");
        let rows = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| if i == j { Ex::one() } else { Ex::zero() })
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
    pub fn from_fn(n: usize, m: usize, f: impl Fn(usize, usize) -> Ex) -> Self {
        assert!(n > 0 && m > 0, "Matrix dimensions must be positive");
        let rows = (0..n).map(|i| (0..m).map(|j| f(i, j)).collect()).collect();
        Matrix {
            rows,
            nrows: n,
            ncols: m,
        }
    }

    /// Create a 1×n row vector from a list of elements.
    pub fn row_vector(elems: Vec<Ex>) -> Self {
        assert!(
            !elems.is_empty(),
            "Row vector must have at least one element"
        );
        let ncols = elems.len();
        Matrix {
            rows: vec![elems],
            nrows: 1,
            ncols,
        }
    }

    /// Create an n×1 column vector from a list of elements.
    pub fn col_vector(elems: Vec<Ex>) -> Self {
        assert!(
            !elems.is_empty(),
            "Column vector must have at least one element"
        );
        let nrows = elems.len();
        let rows = elems.into_iter().map(|e| vec![e]).collect();
        Matrix {
            rows,
            nrows,
            ncols: 1,
        }
    }

    /// Create a diagonal matrix from a slice of expressions.
    pub fn diag(entries: &[Ex]) -> Matrix {
        let n = entries.len();
        assert!(n > 0, "diag: entries must be non-empty");
        let zero = crate::int(0);
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
        Matrix::new(rows)
    }

    /// Create a matrix from a 2D slice of i64 values.
    pub fn from_i64(rows: &[&[i64]]) -> Matrix {
        let data: Vec<Vec<Ex>> = rows
            .iter()
            .map(|row| row.iter().map(|&v| crate::int(v)).collect())
            .collect();
        Matrix::new(data)
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

    /// Safe immutable reference to element `(i, j)`.
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

    // ── Context-aware zero / one ───────────────────────────────────────
    //
    // These produce 0 and 1 in the *same* context as the matrix's
    // elements, avoiding "cannot mix expressions from different contexts"
    // panics when the matrix was built from a non-default Context.

    /// Zero expression in the matrix's own context.
    ///
    /// Falls back to `Ex::zero()` (global context) for empty matrices.
    fn ctx_zero(&self) -> Ex {
        if let Some(elem) = self.rows.first().and_then(|r| r.first()) {
            let zero_id = elem.inner.read().arena.zero();
            elem.wrap(zero_id)
        } else {
            Ex::zero()
        }
    }

    /// One expression in the matrix's own context.
    ///
    /// Falls back to `Ex::one()` (global context) for empty matrices.
    fn ctx_one(&self) -> Ex {
        if let Some(elem) = self.rows.first().and_then(|r| r.first()) {
            let one_id = elem.inner.read().arena.one();
            elem.wrap(one_id)
        } else {
            Ex::one()
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Matrix operations
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

    /// Element-wise addition. Both matrices must have the same shape.
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::ComputationFailed`] if shapes differ.
    pub fn add_elementwise(&self, other: &Matrix) -> Result<Matrix, SymplexError> {
        if self.shape() != other.shape() {
            return Err(SymplexError::ComputationFailed {
                operation: "add_elementwise",
                reason: format!(
                    "cannot add matrices with shapes {:?} and {:?}",
                    self.shape(),
                    other.shape()
                ),
            });
        }
        let rows = (0..self.nrows)
            .map(|i| {
                (0..self.ncols)
                    .map(|j| &self.rows[i][j] + &other.rows[i][j])
                    .collect()
            })
            .collect();
        Ok(Matrix {
            nrows: self.nrows,
            ncols: self.ncols,
            rows,
        })
    }

    /// Element-wise subtraction. Both matrices must have the same shape.
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::ComputationFailed`] if shapes differ.
    pub fn sub_elementwise(&self, other: &Matrix) -> Result<Matrix, SymplexError> {
        if self.shape() != other.shape() {
            return Err(SymplexError::ComputationFailed {
                operation: "sub_elementwise",
                reason: format!(
                    "cannot subtract matrices with shapes {:?} and {:?}",
                    self.shape(),
                    other.shape()
                ),
            });
        }
        let rows = (0..self.nrows)
            .map(|i| {
                (0..self.ncols)
                    .map(|j| &self.rows[i][j] - &other.rows[i][j])
                    .collect()
            })
            .collect();
        Ok(Matrix {
            nrows: self.nrows,
            ncols: self.ncols,
            rows,
        })
    }

    /// Element-wise addition (convenience alias for operator `+`).
    ///
    /// Prefer using `m1 + m2` via the `Add` trait. This method is retained
    /// for backward-compatibility with code written before operator
    /// overloads were available.
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::ComputationFailed`] if shapes differ.
    pub fn add(&self, other: &Matrix) -> Result<Matrix, SymplexError> {
        self.add_elementwise(other)
    }

    /// Element-wise subtraction (convenience alias for operator `-`).
    ///
    /// Prefer using `m1 - m2` via the `Sub` trait. This method is retained
    /// for backward-compatibility with code written before operator
    /// overloads were available.
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::ComputationFailed`] if shapes differ.
    pub fn sub(&self, other: &Matrix) -> Result<Matrix, SymplexError> {
        self.sub_elementwise(other)
    }

    /// Scalar multiplication: multiply every element by `scalar`.
    pub fn scale(&self, scalar: &Ex) -> Matrix {
        self.map(|elem| scalar * elem)
    }

    /// Matrix multiplication (`self * other`).
    ///
    /// `self` is `n × p` and `other` is `p × m`; the result is `n × m`.
    ///
    /// Each element is computed symbolically:
    /// `result[i][j] = Σ_k self[i][k] * other[k][j]`.
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::ComputationFailed`] if `self.ncols != other.nrows`.
    pub fn matmul(&self, other: &Matrix) -> Result<Matrix, SymplexError> {
        if self.ncols != other.nrows {
            return Err(SymplexError::ComputationFailed {
                operation: "matmul",
                reason: format!(
                    "cannot multiply {}×{} by {}×{} matrices",
                    self.nrows, self.ncols, other.nrows, other.ncols
                ),
            });
        }
        let p = self.ncols;
        let rows: Vec<Vec<Ex>> = (0..self.nrows)
            .map(|i| {
                (0..other.ncols)
                    .map(|j| {
                        // Build the sum: a[i][0]*b[0][j] + a[i][1]*b[1][j] + ...
                        let mut acc: Ex = &self.rows[i][0] * &other.rows[0][j];
                        for k in 1..p {
                            let term = &self.rows[i][k] * &other.rows[k][j];
                            acc = acc + term;
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
    /// Returns [`SymplexError::ComputationFailed`] if the matrix is not square.
    pub fn trace(&self) -> Result<Ex, SymplexError> {
        if self.nrows != self.ncols {
            return Err(SymplexError::ComputationFailed {
                operation: "trace",
                reason: format!(
                    "requires a square matrix, got {}×{}",
                    self.nrows, self.ncols
                ),
            });
        }
        let mut acc = self.rows[0][0].clone();
        for i in 1..self.nrows {
            acc = acc + &self.rows[i][i];
        }
        Ok(acc)
    }

    /// Determinant of a square matrix.
    ///
    /// Dispatch strategy:
    /// - 0×0 → 1 (empty product)
    /// - 1×1 → element
    /// - 2×2 → ad − bc
    /// - 3×3 → cofactor expansion (hard-coded, fast)
    /// - n ≥ 4 → Bareiss fraction-free elimination (O(n³))
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::ComputationFailed`] if the matrix is not square.
    pub fn det(&self) -> Result<Ex, SymplexError> {
        if self.nrows != self.ncols {
            return Err(SymplexError::ComputationFailed {
                operation: "det",
                reason: format!(
                    "requires a square matrix, got {}×{}",
                    self.nrows, self.ncols
                ),
            });
        }
        let n = self.nrows;
        Ok(match n {
            0 => Ex::one(),
            1 => self.get(0, 0).clone(),
            2 => {
                // ad - bc
                let a = self.get(0, 0);
                let b = self.get(0, 1);
                let c = self.get(1, 0);
                let d = self.get(1, 1);
                &(a * d) - &(b * c)
            }
            3 => self.det_cofactor(), // hard-coded 3×3 is fast
            _ => self.det_bareiss(),  // Bareiss for n ≥ 4
        })
    }

    /// Determinant via cofactor expansion (O(n!) — only for small matrices).
    fn det_cofactor(&self) -> Ex {
        self.det_cofactor_inner(&self.rows)
    }

    /// Determinant via LU decomposition (O(n³)).
    #[allow(dead_code)]
    fn det_lu(&self) -> Ex {
        match self.lu() {
            Some((_, u, perm)) => {
                // det = product of U diagonal × sign of permutation
                let mut det = u.get(0, 0).clone();
                for i in 1..self.nrows() {
                    det = &det * u.get(i, i);
                }
                // Count inversions in permutation to determine sign
                let mut inversions = 0;
                for i in 0..perm.len() {
                    for j in (i + 1)..perm.len() {
                        if perm[i] > perm[j] {
                            inversions += 1;
                        }
                    }
                }
                if inversions % 2 == 1 { -det } else { det }
            }
            None => Ex::zero(), // Singular matrix
        }
    }

    /// Determinant via Bareiss fraction-free elimination.
    ///
    /// O(n³) element operations. Uses exact division (Sylvester's identity)
    /// to avoid introducing symbolic fractions. Superior to LU for symbolic
    /// matrices because it doesn't accumulate denominators.
    fn det_bareiss(&self) -> Ex {
        let n = self.nrows();

        // Work on a copy of the matrix entries
        let mut m: Vec<Vec<Ex>> = (0..n)
            .map(|i| (0..n).map(|j| self.get(i, j).clone()).collect())
            .collect();

        let mut sign = 1i64; // track row swaps
        let mut prev_pivot = self.ctx_one();

        for k in 0..n - 1 {
            // Find pivot: first non-zero in column k, rows k..n
            let pivot_row = Self::find_bareiss_pivot(&m, k, n);

            match pivot_row {
                None => return self.ctx_zero(), // singular matrix
                Some(pr) if pr != k => {
                    // Swap rows
                    m.swap(k, pr);
                    sign = -sign;
                }
                _ => {} // pivot is already in row k
            }

            let pivot = m[k][k].clone();

            // Bareiss elimination
            for i in (k + 1)..n {
                for j in (k + 1)..n {
                    // new[i][j] = (pivot * m[i][j] - m[i][k] * m[k][j]) / prev_pivot
                    let numer = &(&pivot * &m[i][j]) - &(&m[i][k] * &m[k][j]);
                    // Exact division by prev_pivot (guaranteed by Sylvester's identity)
                    m[i][j] = (&numer / &prev_pivot).eval();
                }
                m[i][k] = self.ctx_zero(); // zero below pivot (optional but clean)
            }

            prev_pivot = pivot;
        }

        // Determinant is bottom-right element × sign
        let det = m[n - 1][n - 1].clone();
        if sign < 0 { -det } else { det }
    }

    /// Find a non-zero pivot in column k, rows k..n.
    /// Uses 2-pass strategy: structural zero → eval zero.
    #[allow(clippy::needless_range_loop)]
    fn find_bareiss_pivot(m: &[Vec<Ex>], k: usize, n: usize) -> Option<usize> {
        // Pass 1: find structurally non-zero entry
        for i in k..n {
            if !m[i][k].is_zero_structural() {
                return Some(i);
            }
        }
        // Pass 2: try eval().is_zero_structural()
        for i in k..n {
            let evaled = m[i][k].eval();
            if !evaled.is_zero_structural() {
                return Some(i);
            }
        }
        // All entries appear to be zero
        None
    }

    /// Recursive cofactor expansion helper.
    fn det_cofactor_inner(&self, m: &[Vec<Ex>]) -> Ex {
        let n = m.len();
        if n == 1 {
            return m[0][0].clone();
        }
        if n == 2 {
            // ad - bc
            let ad = &m[0][0] * &m[1][1];
            let bc = &m[0][1] * &m[1][0];
            return ad - bc;
        }

        // General cofactor expansion along the first row.
        let mut result: Option<Ex> = None;
        for j in 0..n {
            // Build the (n-1)×(n-1) minor by removing row 0 and column j.
            let minor: Vec<Vec<Ex>> = (1..n)
                .map(|row| {
                    (0..n)
                        .filter(|&col| col != j)
                        .map(|col| m[row][col].clone())
                        .collect()
                })
                .collect();
            let cofactor = self.det_cofactor_inner(&minor);
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

        result.unwrap()
    }

    /// Apply a function to every element, producing a new matrix.
    pub fn map(&self, f: impl Fn(&Ex) -> Ex) -> Matrix {
        let rows = self
            .rows
            .iter()
            .map(|row| row.iter().map(&f).collect())
            .collect();
        Matrix {
            nrows: self.nrows,
            ncols: self.ncols,
            rows,
        }
    }

    // ── Linear-algebra: minor, cofactor, adjugate, inverse ─────────────

    /// Extract the minor matrix (row `row`, col `col` removed).
    ///
    /// The result is an `(n-1) × (n-1)` matrix formed by deleting the
    /// specified row and column.
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::ComputationFailed`] if the matrix is not
    /// square or has dimension ≤ 1.
    pub fn minor(&self, row: usize, col: usize) -> Result<Matrix, SymplexError> {
        if self.nrows != self.ncols {
            return Err(SymplexError::ComputationFailed {
                operation: "minor",
                reason: format!(
                    "requires a square matrix, got {}×{}",
                    self.nrows, self.ncols
                ),
            });
        }
        if self.nrows <= 1 {
            return Err(SymplexError::ComputationFailed {
                operation: "minor",
                reason: "requires matrix dimension > 1".into(),
            });
        }
        let mut rows = Vec::new();
        for (r, row_data) in self.rows.iter().enumerate() {
            if r == row {
                continue;
            }
            let mut new_row = Vec::new();
            for (c, val) in row_data.iter().enumerate() {
                if c == col {
                    continue;
                }
                new_row.push(val.clone());
            }
            rows.push(new_row);
        }
        Ok(Matrix::new(rows))
    }

    /// Cofactor C(i, j) = (-1)^(i+j) * det(minor(i, j)).
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::ComputationFailed`] if the matrix is not
    /// square or has dimension ≤ 1.
    pub fn cofactor(&self, row: usize, col: usize) -> Result<Ex, SymplexError> {
        let minor_mat = self.minor(row, col)?;
        let minor_det = minor_mat.det()?;
        Ok(if (row + col).is_multiple_of(2) {
            minor_det
        } else {
            -minor_det
        })
    }

    /// Adjugate matrix (transpose of the cofactor matrix).
    ///
    /// `adj(A)[i][j] = cofactor(A, j, i)`.
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::ComputationFailed`] if the matrix is not square.
    pub fn adjugate(&self) -> Result<Matrix, SymplexError> {
        let n = self.nrows();
        if n != self.ncols() {
            return Err(SymplexError::ComputationFailed {
                operation: "adjugate",
                reason: format!(
                    "requires a square matrix, got {}×{}",
                    self.nrows(),
                    self.ncols()
                ),
            });
        }
        let mut rows = Vec::new();
        for j in 0..n {
            let mut row = Vec::new();
            for i in 0..n {
                row.push(self.cofactor(i, j)?); // note: transposed
            }
            rows.push(row);
        }
        Ok(Matrix::new(rows))
    }

    /// Matrix inverse: A⁻¹ = adj(A) / det(A).
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::ComputationFailed`] if the matrix is not
    /// square or if the matrix is singular (determinant is structurally
    /// zero).
    pub fn inv(&self) -> Result<Matrix, SymplexError> {
        if self.nrows() != self.ncols() {
            return Err(SymplexError::ComputationFailed {
                operation: "inv",
                reason: format!(
                    "requires a square matrix, got {}×{}",
                    self.nrows(),
                    self.ncols()
                ),
            });
        }
        let d = self.det()?;
        if d.is_zero_structural() {
            return Err(SymplexError::ComputationFailed {
                operation: "inv",
                reason: "matrix is singular (determinant is zero)".into(),
            });
        }
        // 1×1 special case: inverse is just [[1/a]]
        if self.nrows() == 1 {
            let one_over_det = &self.ctx_one() / &d;
            return Ok(Matrix::new(vec![vec![one_over_det]]));
        }
        let adj = self.adjugate()?;
        let one_over_det = &self.ctx_one() / &d;
        Ok(adj.scale(&one_over_det))
    }

    // ── Characteristic polynomial & eigenvalues ────────────────────────

    /// Characteristic polynomial: det(A − λI).
    ///
    /// Returns the polynomial as an [`Ex`] in the given variable `var`
    /// (which plays the role of λ).
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::ComputationFailed`] if the matrix is not square.
    pub fn char_poly(&self, var: &Ex) -> Result<Ex, SymplexError> {
        let n = self.nrows();
        if n != self.ncols() {
            return Err(SymplexError::ComputationFailed {
                operation: "char_poly",
                reason: format!(
                    "requires a square matrix, got {}×{}",
                    self.nrows(),
                    self.ncols()
                ),
            });
        }
        // Build A - λI
        let mut rows = Vec::new();
        for i in 0..n {
            let mut row = Vec::new();
            for j in 0..n {
                let entry = self.rows[i][j].clone();
                if i == j {
                    row.push(&entry - var);
                } else {
                    row.push(entry);
                }
            }
            rows.push(row);
        }
        let m = Matrix::new(rows);
        Ok(m.det()?.expand())
    }

    /// Eigenvalues: solve `char_poly(var) = 0` for `var`.
    ///
    /// Returns a list of eigenvalues. If the solver cannot factor the
    /// characteristic polynomial, the returned list may be empty.
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::ComputationFailed`] if the matrix is not square.
    pub fn eigenvals(&self, var: &Ex) -> Result<Vec<Ex>, SymplexError> {
        let cp = self.char_poly(var)?;
        Ok(cp.solve_or_empty(var))
    }

    /// Eigenvectors: for each eigenvalue, compute a basis for its eigenspace.
    ///
    /// Returns a list of `(eigenvalue, algebraic_multiplicity, eigenvectors)`
    /// tuples.  Each eigenvector is returned as a column-vector [`Matrix`].
    ///
    /// Algebraic multiplicity is estimated by counting duplicate roots
    /// returned by the solver.  For matrices with symbolic entries or
    /// characteristic polynomials of degree ≥ 5, the solver may not
    /// return all roots and the list can be incomplete.
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::ComputationFailed`] if the matrix is not square.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// let var = symplex::var("λ");
    /// let m = symplex::matrix![[2, 1], [0, 3]];
    /// let evs = m.eigenvects(&var).unwrap();
    /// for (val, mult, vecs) in &evs {
    ///     assert!(!vecs.is_empty());
    /// }
    /// ```
    pub fn eigenvects(&self, var: &Ex) -> Result<Vec<(Ex, usize, Vec<Matrix>)>, SymplexError> {
        if !self.is_square() {
            return Err(SymplexError::ComputationFailed {
                operation: "eigenvects",
                reason: format!(
                    "requires a square matrix, got {}×{}",
                    self.nrows, self.ncols
                ),
            });
        }
        let n = self.nrows;
        debug!(n, "eigenvects: computing for {}×{} matrix", n, n);
        let eye = Matrix::identity(n);

        // Compute characteristic polynomial and try to factor it for
        // proper algebraic multiplicities via Poly::factor_over_z().
        let cp = self.char_poly(var)?;
        let eigen_pairs = eigvals_with_multiplicity(&cp, var);
        trace!(
            eigenvalue_count = eigen_pairs.len(),
            "eigenvects: found eigenvalues with multiplicities"
        );

        // For each unique eigenvalue, compute eigenvectors via nullspace(A − λI).
        let mut result = Vec::new();
        for (eigenval, alg_mult) in &eigen_pairs {
            let a_minus_lambda_i = self.sub_elementwise(&eye.scale(eigenval))?;
            let vecs = a_minus_lambda_i.nullspace();
            trace!(
                alg_mult,
                geom_mult = vecs.len(),
                "eigenvects: eigenvalue has alg_mult={}, geom_mult={}",
                alg_mult,
                vecs.len()
            );
            result.push((eigenval.clone(), *alg_mult, vecs));
        }

        Ok(result)
    }

    /// Check whether the matrix is diagonalizable.
    ///
    /// A matrix is diagonalizable iff for every eigenvalue the geometric
    /// multiplicity (dimension of eigenspace) equals the algebraic
    /// multiplicity.  Returns `Ok(false)` if the eigenvalue solver cannot
    /// find all roots.
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::ComputationFailed`] if the matrix is not square.
    pub fn is_diagonalizable(&self, var: &Ex) -> Result<bool, SymplexError> {
        let eigvs = self.eigenvects(var)?;
        let mut total = 0usize;
        for (_, alg_mult, vecs) in &eigvs {
            if vecs.len() != *alg_mult {
                return Ok(false);
            }
            total += vecs.len();
        }
        Ok(total == self.nrows)
    }

    /// Diagonalize: find invertible `P` and diagonal `D` such that
    /// `D = P⁻¹ A P`.
    ///
    /// `P` is the matrix whose columns are eigenvectors and `D` is
    /// the diagonal matrix of eigenvalues.
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::ComputationFailed`] if the matrix is not
    /// square or is not diagonalizable (geometric multiplicity ≠ algebraic
    /// multiplicity for some eigenvalue).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// let var = symplex::var("λ");
    /// let m = symplex::matrix![[2, 1], [0, 3]];
    /// let (p, d) = m.diagonalize(&var).unwrap();
    /// assert_eq!(d.nrows(), 2);
    /// ```
    pub fn diagonalize(&self, var: &Ex) -> Result<(Matrix, Matrix), SymplexError> {
        debug!("diagonalize: attempting for {}×{} matrix", self.nrows, self.ncols);
        let eigvs = self.eigenvects(var)?;

        // Verify diagonalizability: need n linearly independent eigenvectors.
        let n = self.nrows;
        let mut total_vecs = 0usize;
        for (_, alg_mult, vecs) in &eigvs {
            if vecs.len() != *alg_mult {
                return Err(SymplexError::ComputationFailed {
                    operation: "diagonalize",
                    reason: "matrix is not diagonalizable: geometric multiplicity \
                             does not equal algebraic multiplicity for all eigenvalues"
                        .into(),
                });
            }
            total_vecs += vecs.len();
        }
        if total_vecs != n {
            return Err(SymplexError::ComputationFailed {
                operation: "diagonalize",
                reason: "matrix is not diagonalizable: insufficient eigenvectors found"
                    .into(),
            });
        }

        // Build P (eigenvector columns) and D (diagonal eigenvalues).
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

    /// Jordan normal form: find `P` and block-diagonal `J` such that
    /// `A = P J P⁻¹`.
    ///
    /// `J` is a block-diagonal matrix of Jordan blocks `J_k(λ)`, where
    /// each block has eigenvalue `λ` on the diagonal and 1s on the
    /// superdiagonal.  `P` is the matrix of (generalized) eigenvectors.
    ///
    /// For diagonalizable matrices, the Jordan form equals the diagonal
    /// form and this method delegates to [`diagonalize`](Self::diagonalize).
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::ComputationFailed`] if the matrix is not
    /// square or if the eigenvalue solver cannot find all roots.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// let var = symplex::var("λ");
    /// // Defective matrix: eigenvalue 2 with algebraic mult 2, geometric mult 1
    /// let m = symplex::matrix![[2, 1, 0, 0],
    ///                          [0, 2, 0, 0],
    ///                          [0, 0, 3, 0],
    ///                          [0, 0, 0, 4]];
    /// let (p, j) = m.jordan_form(&var).unwrap();
    /// assert_eq!(j.nrows(), 4);
    /// ```
    pub fn jordan_form(&self, var: &Ex) -> Result<(Matrix, Matrix), SymplexError> {
        if !self.is_square() {
            return Err(SymplexError::ComputationFailed {
                operation: "jordan_form",
                reason: format!(
                    "requires a square matrix, got {}×{}",
                    self.nrows, self.ncols
                ),
            });
        }
        let n = self.nrows;
        debug!(n, "jordan_form: computing for {}×{} matrix", n, n);
        let eye = Matrix::identity(n);

        // Get eigenvalues with multiplicities.
        let eigvs = self.eigenvects(var)?;

        // Fast path: if diagonalizable, delegate.
        let total_vecs: usize = eigvs.iter().map(|(_, _, v)| v.len()).sum();
        let all_match = eigvs.iter().all(|(_, m, v)| v.len() == *m);
        if all_match && total_vecs == n {
            return self.diagonalize(var);
        }

        // Check that we found all eigenvalues.
        let total_alg: usize = eigvs.iter().map(|(_, m, _)| *m).sum();
        if total_alg != n {
            return Err(SymplexError::ComputationFailed {
                operation: "jordan_form",
                reason: format!(
                    "eigenvalue solver found algebraic multiplicity sum {} but matrix is {}×{}",
                    total_alg, n, n
                ),
            });
        }

        // For each eigenvalue, compute the Jordan block structure via nullity chain,
        // then build generalized eigenvectors.
        let mut jordan_blocks: Vec<Vec<Ex>> = Vec::new(); // rows of J
        let mut basis_cols: Vec<Matrix> = Vec::new();

        for (eigenval, alg_mult, _) in &eigvs {
            let a_minus_lambda = self.sub_elementwise(&eye.scale(&eigenval))?;

            // Nullity chain: [0, nullity(E), nullity(E²), ...] where E = A - λI
            trace!("jordan_form: computing nullity chain for eigenvalue");
            let mut chain: Vec<usize> = vec![0];
            let mut power = a_minus_lambda.clone();
            loop {
                let nullity = n - power.rank();
                if nullity == *chain.last().unwrap() || nullity >= *alg_mult {
                    if nullity > *chain.last().unwrap() {
                        chain.push(nullity);
                    }
                    break;
                }
                chain.push(nullity);
                power = power.matmul(&a_minus_lambda)?;
            }

            // Derive block sizes from nullity chain differences.
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

            // Collect (block_size, count) pairs, largest first.
            let mut blocks: Vec<(usize, usize)> = block_counts
                .iter()
                .enumerate()
                .filter(|(_, c)| **c > 0)
                .map(|(i, c)| (i + 1, *c))
                .collect();
            blocks.sort_by(|a, b| b.0.cmp(&a.0));

            // Build generalized eigenvectors for each block.
            let mut eig_basis: Vec<Matrix> = Vec::new();

            for (block_size, count) in &blocks {
                for _ in 0..*count {
                    // Compute ker(E^block_size) and ker(E^(block_size-1))
                    let null_big = jordan_null_power(&a_minus_lambda, *block_size, n)?;
                    let null_small = if *block_size > 1 {
                        jordan_null_power(&a_minus_lambda, block_size - 1, n)?
                    } else {
                        Vec::new()
                    };

                    // Pick a vector in null_big but not in span(null_small ∪ eig_basis)
                    let exclude: Vec<&Matrix> = null_small
                        .iter()
                        .chain(eig_basis.iter())
                        .collect();
                    let vec = match pick_independent_vec(&null_big, &exclude, n)? {
                        Some(v) => v,
                        None => {
                            return Err(SymplexError::ComputationFailed {
                                operation: "jordan_form",
                                reason: "could not find independent generalized eigenvector"
                                    .into(),
                            });
                        }
                    };

                    // Build Jordan chain: [E^(k-1)·v, E^(k-2)·v, ..., E·v, v]
                    let mut chain_vecs: Vec<Matrix> = Vec::new();
                    for i in (0..*block_size).rev() {
                        if i == 0 {
                            chain_vecs.push(vec.clone());
                        } else {
                            let powered = matrix_pow_vec(&a_minus_lambda, &vec, i)?;
                            chain_vecs.push(powered);
                        }
                    }
                    // chain_vecs is [E^(k-1)·v, ..., v] — eigenvector first
                    chain_vecs.reverse();
                    // Now [v, E·v, ..., E^(k-1)·v] — but we want columns ordered
                    // so eigenvector is last in the block. Reverse again:
                    // Actually the standard convention is eigenvector FIRST in block.
                    // [E^(k-1)·v, E^(k-2)·v, ..., v] — this is correct.
                    chain_vecs.reverse();

                    eig_basis.extend(chain_vecs.iter().cloned());
                    basis_cols.extend(chain_vecs);

                    // Add Jordan block to J: λ on diagonal, 1 on superdiagonal.
                    let bs = *block_size;
                    let col_offset = jordan_blocks.len(); // fixed: compute once before loop
                    for row_idx in 0..bs {
                        let mut row = vec![Ex::zero(); n];
                        row[col_offset + row_idx] = eigenval.clone();
                        if row_idx + 1 < bs {
                            row[col_offset + row_idx + 1] = Ex::one();
                        }
                        jordan_blocks.push(row);
                    }
                }
            }
        }

        if jordan_blocks.len() != n || basis_cols.len() != n {
            return Err(SymplexError::ComputationFailed {
                operation: "jordan_form",
                reason: format!(
                    "internal error: expected {} basis vectors, got {}",
                    n,
                    basis_cols.len()
                ),
            });
        }

        let j = Matrix::new(jordan_blocks);
        let col_refs: Vec<&Matrix> = basis_cols.iter().collect();
        let p = Matrix::hstack(&col_refs)?;

        Ok((p, j))
    }

    /// Compute the integer power of a square matrix via repeated squaring.
    ///
    /// - `powi(0)` returns the identity matrix
    /// - `powi(1)` returns a clone
    /// - `powi(n)` for n ≥ 2 uses binary exponentiation
    /// - Negative powers are not supported (use `inv()` + `powi()`)
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::ComputationFailed`] if the matrix is not square.
    pub fn powi(&self, n: u32) -> Result<Matrix, SymplexError> {
        if !self.is_square() {
            return Err(SymplexError::ComputationFailed {
                operation: "powi",
                reason: format!(
                    "requires a square matrix, got {}×{}",
                    self.nrows, self.ncols
                ),
            });
        }
        if n == 0 {
            return Ok(Matrix::identity(self.nrows));
        }
        if n == 1 {
            return Ok(self.clone());
        }
        // Binary exponentiation
        let mut result = Matrix::identity(self.nrows);
        let mut base = self.clone();
        let mut exp = n;
        while exp > 0 {
            if exp % 2 == 1 {
                result = result.matmul(&base)?;
            }
            base = base.matmul(&base)?;
            exp /= 2;
        }
        Ok(result)
    }

    /// Kronecker (tensor) product: A ⊗ B.
    ///
    /// For A (m×n) and B (p×q), produces an (mp×nq) matrix.
    pub fn kronecker(&self, other: &Matrix) -> Matrix {
        let mut rows = Vec::new();
        for i in 0..self.nrows {
            for k in 0..other.nrows {
                let mut row = Vec::new();
                for j in 0..self.ncols {
                    for l in 0..other.ncols {
                        row.push(&self.rows[i][j] * &other.rows[k][l]);
                    }
                }
                rows.push(row);
            }
        }
        Matrix::new(rows)
    }

    /// Matrix exponential via truncated Taylor series: eᴬ ≈ Σₖ₌₀ⁿ Aᵏ/k!.
    ///
    /// This computes a symbolic approximation. For exact results, prefer
    /// [`matrix_exp`](Self::matrix_exp) which uses Jordan decomposition.
    ///
    /// `order` controls the number of terms (default: 10 is good for most cases).
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::ComputationFailed`] if the matrix is not square.
    pub fn exp_series(&self, order: usize) -> Result<Matrix, SymplexError> {
        if !self.is_square() {
            return Err(SymplexError::ComputationFailed {
                operation: "exp_series",
                reason: format!(
                    "requires a square matrix, got {}×{}",
                    self.nrows, self.ncols
                ),
            });
        }
        let n = self.nrows;
        let mut result = Matrix::identity(n);
        let mut a_power_over_factorial = Matrix::identity(n);
        for k in 1..=order {
            a_power_over_factorial = a_power_over_factorial.matmul(self)?;
            let inv_k = crate::rational(1, k as i64);
            a_power_over_factorial = a_power_over_factorial.scale(&inv_k);
            result = result.add_elementwise(&a_power_over_factorial)?;
        }
        Ok(result)
    }

    /// Exact symbolic matrix exponential via Jordan decomposition.
    ///
    /// Computes `eᴬ = P · e^J · P⁻¹` where `J` is the Jordan normal form.
    /// For each Jordan block `J_k(λ)`, the matrix exponential is:
    ///
    /// ```text
    /// e^{J_k(λ)} = e^λ · [ 1,    1,    1/2!, ..., 1/(k-1)! ]
    ///                     [ 0,    1,    1,    ..., 1/(k-2)! ]
    ///                     [ 0,    0,    1,    ..., ...       ]
    ///                     [ ...                   1         ]
    /// ```
    ///
    /// Falls back to [`exp_series`](Self::exp_series) with 10 terms if
    /// the Jordan form cannot be computed (e.g., eigenvalues not found).
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::ComputationFailed`] if the matrix is not square.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// let var = symplex::var("λ");
    /// let m = symplex::matrix![[0, 1], [-1, 0]];
    /// // e^[[0,1],[-1,0]] involves sin and cos
    /// let result = m.matrix_exp(&var);
    /// assert!(result.is_ok());
    /// ```
    pub fn matrix_exp(&self, var: &Ex) -> Result<Matrix, SymplexError> {
        if !self.is_square() {
            return Err(SymplexError::ComputationFailed {
                operation: "matrix_exp",
                reason: format!(
                    "requires a square matrix, got {}×{}",
                    self.nrows, self.ncols
                ),
            });
        }
        debug!("matrix_exp: computing for {}×{} matrix", self.nrows, self.ncols);

        let n = self.nrows;

        // Try Jordan decomposition.
        let (p, j) = match self.jordan_form(var) {
            Ok(pj) => pj,
            Err(_) => {
                // Fallback to Taylor series.
                debug!("matrix_exp: Jordan form failed, falling back to exp_series(10)");
                return self.exp_series(10);
            }
        };

        // Compute e^J block by block.
        // J is block-diagonal with Jordan blocks. We compute e^J by
        // exponentiating each block independently.
        //
        // For a Jordan block J_k(λ):
        //   e^{J_k(λ)}[i][j] = e^λ / (j-i)!   if j >= i
        //                     = 0                if j < i
        let mut exp_j_rows: Vec<Vec<Ex>> = vec![vec![Ex::zero(); n]; n];

        // Walk along the diagonal of J to identify blocks.
        let mut col = 0;
        while col < n {
            // Determine block size: count consecutive 1s on the superdiagonal.
            let lambda = j.get(col, col).clone();
            let mut block_size = 1;
            while col + block_size < n {
                let superdiag = j.get(col + block_size - 1, col + block_size).simplify();
                let diag_next = j.get(col + block_size, col + block_size).clone();
                let lambda_diff = (&diag_next - &lambda).simplify();
                if superdiag.is_zero_structural() || !lambda_diff.is_zero_structural() {
                    break;
                }
                // Check superdiag is 1
                let one_diff = (&superdiag - &Ex::one()).simplify();
                if !one_diff.is_zero_structural() {
                    break;
                }
                block_size += 1;
            }

            trace!(col, block_size, "matrix_exp: processing Jordan block");

            // Compute e^λ
            let exp_lambda = lambda.exp();

            // Fill in the block: e^{J_k}[i][j] = e^λ / (j-i)! for j >= i
            for i in 0..block_size {
                for jj in i..block_size {
                    let diff = jj - i;
                    let factorial_val = crate::int(factorial_usize(diff) as i64);
                    let entry = &exp_lambda / &factorial_val;
                    exp_j_rows[col + i][col + jj] = entry;
                }
            }

            col += block_size;
        }

        let exp_j = Matrix::new(exp_j_rows);

        // e^A = P · e^J · P⁻¹
        let p_inv = match p.inv() {
            Ok(pi) => pi,
            Err(_) => {
                warn!("matrix_exp: P is singular, falling back to exp_series(10)");
                return self.exp_series(10);
            }
        };

        let result = p.matmul(&exp_j)?.matmul(&p_inv)?;

        // Simplify each entry.
        let result = result.map(|e| e.simplify());

        Ok(result)
    }

    // ── Cholesky decomposition & pseudo-inverse ────────────────────────

    /// Cholesky decomposition for symmetric positive-definite matrices.
    ///
    /// Returns `Ok(Some(L))` such that `A = LLᵀ`, where `L` is lower triangular.
    /// Returns `Ok(None)` if a diagonal element becomes non-positive during
    /// factorization (the matrix is not positive definite).
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::ComputationFailed`] if the matrix is not square.
    pub fn cholesky(&self) -> Result<Option<Matrix>, SymplexError> {
        if !self.is_square() {
            return Err(SymplexError::ComputationFailed {
                operation: "cholesky",
                reason: format!(
                    "requires a square matrix, got {}×{}",
                    self.nrows, self.ncols
                ),
            });
        }
        let n = self.nrows;
        let zero = Ex::zero();
        let mut l_rows: Vec<Vec<Ex>> = (0..n)
            .map(|_| (0..n).map(|_| zero.clone()).collect())
            .collect();

        for j in 0..n {
            // L[j][j] = sqrt(A[j][j] - sum(L[j][k]^2 for k < j))
            let mut sum_sq = zero.clone();
            for item in l_rows[j].iter().take(j) {
                sum_sq = sum_sq + item.powi(2);
            }
            let diag = self.get(j, j) - &sum_sq;
            let diag_simplified = diag.simplify();
            // For numeric matrices, check positive-definiteness
            if let Ok(v) = diag_simplified.eval_f64()
                && v <= 0.0 {
                    return Ok(None);
                }
            l_rows[j][j] = diag_simplified.sqrt();

            // L[i][j] = (A[i][j] - sum(L[i][k]*L[j][k] for k < j)) / L[j][j]
            for i in (j + 1)..n {
                let mut sum_prod = zero.clone();
                for (l_ik, l_jk) in l_rows[i].iter().zip(l_rows[j].iter()).take(j) {
                    sum_prod = sum_prod + &(l_ik * l_jk);
                }
                let num = self.get(i, j) - &sum_prod;
                l_rows[i][j] = &num / &l_rows[j][j];
            }
        }

        Ok(Some(Matrix::new(l_rows)))
    }

    /// Moore–Penrose pseudo-inverse via `A⁺ = (AᵀA)⁻¹Aᵀ`.
    ///
    /// This formula is valid for full-column-rank matrices.
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::ComputationFailed`] if `AᵀA` is singular
    /// (the matrix is not full column rank).
    pub fn pinv(&self) -> Result<Matrix, SymplexError> {
        let at = self.transpose();
        let ata = at.matmul(self)?;
        let ata_inv = ata.inv()?;
        ata_inv.matmul(&at)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Calculus helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Build the Jacobian matrix of `funcs` with respect to `vars`.
///
/// `J[i][j] = d(funcs[i]) / d(vars[j])`.
///
/// # Panics
///
/// Panics if `funcs` or `vars` is empty.
pub fn jacobian(funcs: &[&Ex], vars: &[&Ex]) -> Matrix {
    assert!(!funcs.is_empty(), "jacobian: funcs must be non-empty");
    assert!(!vars.is_empty(), "jacobian: vars must be non-empty");
    let nrows = funcs.len();
    let ncols = vars.len();
    let rows: Vec<Vec<Ex>> = funcs
        .iter()
        .map(|fi| vars.iter().map(|vj| fi.diff(vj)).collect())
        .collect();
    Matrix { rows, nrows, ncols }
}

impl Matrix {
    /// Differentiate every element with respect to `var`.
    pub fn diff(&self, var: &Ex) -> Matrix {
        self.map(|elem| elem.diff(var))
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
    /// let x = symplex::var("x");
    /// let m = Matrix::new(vec![
    ///     vec![x.sin(), x.cos()],
    ///     vec![-x.cos(), x.sin()],
    /// ]);
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
        assert!(
            !self.rows.is_empty() && !self.rows[0].is_empty(),
            "cannot generate code for an empty matrix"
        );
        // Get the context from the first entry.
        let first = &self.rows[0][0];
        let mut guard = first.inner.write();
        let arena = &mut guard.arena;
        // Collect all ExprIds in row-major order.
        let entry_ids: Vec<crate::base::node::ExprId> = self
            .rows
            .iter()
            .flat_map(|row| row.iter().map(|e| e.id))
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
    /// Render this matrix as a LaTeX bmatrix.
    ///
    /// # Example
    /// ```
    /// use symplex::prelude::*;
    /// let m = matrix![[1, 2], [3, 4]];
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
                s.push_str(&self.get(i, j).to_latex());
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
    /// LU decomposition with partial pivoting over exact rationals.
    /// Returns (L, U, perm) where perm is the row permutation vector.
    /// L is lower triangular with 1s on diagonal, U is upper triangular.
    /// P*A = L*U where P is the permutation matrix.
    ///
    /// Returns `None` if the matrix is not square or is singular.
    pub fn lu(&self) -> Option<(Matrix, Matrix, Vec<usize>)> {
        let n = self.nrows;
        if n != self.ncols {
            return None;
        }

        let mut perm: Vec<usize> = (0..n).collect();
        let mut u: Vec<Vec<Ex>> = self.rows.clone();
        let mut l: Vec<Vec<Ex>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| if i == j { Ex::one() } else { Ex::zero() })
                    .collect()
            })
            .collect();

        for k in 0..n {
            // Partial pivoting: find first non-zero in column k, rows k..n
            let mut pivot_row = None;
            for i in k..n {
                if !u[i][k].is_zero_structural() {
                    pivot_row = Some(i);
                    break;
                }
            }
            let pivot_row = pivot_row?;

            if pivot_row != k {
                u.swap(k, pivot_row);
                perm.swap(k, pivot_row);
                // Swap the already-computed L multipliers (columns 0..k)
                for j in 0..k {
                    let tmp = l[k][j].clone();
                    l[k][j] = l[pivot_row][j].clone();
                    l[pivot_row][j] = tmp;
                }
            }

            // Eliminate below pivot
            for i in (k + 1)..n {
                if u[i][k].is_zero_structural() {
                    continue;
                }
                let factor = &u[i][k] / &u[k][k];
                l[i][k] = factor.clone();
                u[i][k] = Ex::zero();
                for j in (k + 1)..n {
                    let term = &factor * &u[k][j];
                    u[i][j] = &u[i][j] - &term;
                }
            }
        }

        let l_mat = Matrix {
            rows: l,
            nrows: n,
            ncols: n,
        };
        let u_mat = Matrix {
            rows: u,
            nrows: n,
            ncols: n,
        };
        Some((l_mat, u_mat, perm))
    }

    /// Row-reduced echelon form via Gauss-Jordan elimination.
    /// Returns (rref_matrix, pivot_columns).
    /// Uses exact rational arithmetic.
    pub fn rref(&self) -> (Matrix, Vec<usize>) {
        let nrows = self.nrows;
        let ncols = self.ncols;
        let mut rows: Vec<Vec<Ex>> = self.rows.clone();
        let mut pivots = Vec::new();
        let mut pivot_row = 0;

        for col in 0..ncols {
            if pivot_row >= nrows {
                break;
            }

            // Find a non-zero entry in this column at or below pivot_row
            let mut found = None;
            for i in pivot_row..nrows {
                if !rows[i][col].is_zero_structural() {
                    found = Some(i);
                    break;
                }
            }

            let found = match found {
                Some(r) => r,
                None => continue, // no pivot in this column
            };

            // Swap to pivot position
            if found != pivot_row {
                rows.swap(pivot_row, found);
            }

            // Scale pivot row so the pivot entry becomes 1
            let pivot_val = rows[pivot_row][col].clone();
            for j in 0..ncols {
                rows[pivot_row][j] = &rows[pivot_row][j] / &pivot_val;
            }

            // Eliminate all other entries in this column
            for i in 0..nrows {
                if i == pivot_row {
                    continue;
                }
                if !rows[i][col].is_zero_structural() {
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

        let mat = Matrix { rows, nrows, ncols };
        (mat, pivots)
    }

    /// Rank of the matrix (number of pivot columns in RREF).
    pub fn rank(&self) -> usize {
        let (_, pivots) = self.rref();
        pivots.len()
    }

    /// Null space (kernel): basis vectors for Ax = 0.
    /// Returns column vectors as 1-column matrices.
    pub fn nullspace(&self) -> Vec<Matrix> {
        let (rref_mat, pivots) = self.rref();
        let n = self.ncols;

        let pivot_set: std::collections::HashSet<usize> = pivots.iter().copied().collect();
        let free_vars: Vec<usize> = (0..n).filter(|c| !pivot_set.contains(c)).collect();

        let mut basis = Vec::new();
        for &free_col in &free_vars {
            let mut entries = vec![Ex::zero(); n];
            entries[free_col] = Ex::one();

            // Back-substitute: for each pivot row, the pivot column gets
            // the negation of the RREF entry in the free column.
            for (pivot_idx, &pivot_col) in pivots.iter().enumerate() {
                entries[pivot_col] = -rref_mat.get(pivot_idx, free_col);
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
            .map(|&col| {
                let elems: Vec<Ex> = (0..self.nrows).map(|i| self.rows[i][col].clone()).collect();
                Matrix::col_vector(elems)
            })
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Utilities
// ═══════════════════════════════════════════════════════════════════════════

impl Matrix {
    /// Frobenius norm: sqrt(sum of squares of all entries).
    pub fn norm(&self) -> Ex {
        let mut sum = Ex::zero();
        for row in &self.rows {
            for elem in row {
                sum = sum + &(elem * elem);
            }
        }
        sum.sqrt()
    }

    /// Is this matrix square?
    pub fn is_square(&self) -> bool {
        self.nrows == self.ncols
    }

    /// Is this matrix symmetric? (A = Aᵀ, checked structurally)
    pub fn is_symmetric(&self) -> bool {
        if self.nrows != self.ncols {
            return false;
        }
        for i in 0..self.nrows {
            for j in (i + 1)..self.ncols {
                let diff = &self.rows[i][j] - &self.rows[j][i];
                if !diff.is_zero_structural() {
                    return false;
                }
            }
        }
        true
    }

    /// Stack matrices horizontally (side by side).
    /// All matrices must have the same number of rows.
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::ComputationFailed`] if `matrices` is empty
    /// or row counts differ.
    pub fn hstack(matrices: &[&Matrix]) -> Result<Matrix, SymplexError> {
        if matrices.is_empty() {
            return Err(SymplexError::ComputationFailed {
                operation: "hstack",
                reason: "need at least one matrix".into(),
            });
        }
        let nrows = matrices[0].nrows;
        for (idx, m) in matrices.iter().enumerate() {
            if m.nrows != nrows {
                return Err(SymplexError::ComputationFailed {
                    operation: "hstack",
                    reason: format!(
                        "matrix {idx} has {} rows, expected {nrows}",
                        m.nrows
                    ),
                });
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
    /// All matrices must have the same number of columns.
    ///
    /// # Errors
    ///
    /// Returns [`SymplexError::ComputationFailed`] if `matrices` is empty
    /// or column counts differ.
    pub fn vstack(matrices: &[&Matrix]) -> Result<Matrix, SymplexError> {
        if matrices.is_empty() {
            return Err(SymplexError::ComputationFailed {
                operation: "vstack",
                reason: "need at least one matrix".into(),
            });
        }
        let ncols = matrices[0].ncols;
        for (idx, m) in matrices.iter().enumerate() {
            if m.ncols != ncols {
                return Err(SymplexError::ComputationFailed {
                    operation: "vstack",
                    reason: format!(
                        "matrix {idx} has {} cols, expected {ncols}",
                        m.ncols
                    ),
                });
            }
        }
        let nrows: usize = matrices.iter().map(|m| m.nrows).sum();
        let rows: Vec<Vec<Ex>> = matrices
            .iter()
            .flat_map(|m| m.rows.iter().cloned())
            .collect();
        Ok(Matrix { rows, nrows, ncols })
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Eigenvalue multiplicity via polynomial factoring
// ═══════════════════════════════════════════════════════════════════════════

/// Extract eigenvalues with correct algebraic multiplicities from a
/// characteristic polynomial.
///
/// Attempts to use `Poly::factor_over_z()` (Yun's square-free decomposition)
/// for proper `(factor, multiplicity)` pairs.  Falls back to derivative-based
/// multiplicity detection if the polynomial layer can't handle the expression.
/// Extract eigenvalues with correct algebraic multiplicities from a
/// characteristic polynomial.
///
/// **Primary path:** Convert to `Poly`, call `factor_over_z()` (Yun's
/// square-free decomposition) for exact `(factor, multiplicity)` pairs.
/// Each factor is solved for roots, which inherit the factor's multiplicity.
///
/// **Fallback:** If the polynomial has symbolic (non-rational) coefficients,
/// `expr_to_poly` returns `None`. In that case, use derivative-based
/// multiplicity detection: for each root `r`, find the smallest `k` such
/// that `p^(k)(r) ≠ 0`.
fn eigvals_with_multiplicity(char_poly: &Ex, var: &Ex) -> Vec<(Ex, usize)> {
    // ── Primary path: Poly::factor_over_z() ────────────────────────
    // This gives exact multiplicities via Yun's square-free decomposition.
    if let Some(pairs) = eigvals_via_poly_factor(char_poly, var) {
        if !pairs.is_empty() {
            debug!(
                count = pairs.len(),
                "eigvals_with_multiplicity: used Poly::factor_over_z path"
            );
            return pairs;
        }
    }

    // ── Fallback: derivative-based multiplicity detection ──────────
    // Used when char poly has symbolic coefficients or factor_over_z
    // can't find roots.
    debug!("eigvals_with_multiplicity: falling back to derivative-based detection");
    eigvals_via_derivative(char_poly, var)
}

/// Primary multiplicity path: convert to Poly, factor, solve each factor.
fn eigvals_via_poly_factor(char_poly: &Ex, var: &Ex) -> Option<Vec<(Ex, usize)>> {
    let inner = char_poly.inner.read();
    let arena = &inner.arena;

    // Try to convert the characteristic polynomial expression to a dense Poly.
    let poly = crate::poly::polybridge::expr_to_poly(arena, char_poly.id, var.id)?;

    // Factor: returns (content, [(factor_poly, multiplicity), ...]).
    let (_content, factors) = poly.factor_over_z();
    if factors.is_empty() {
        return None;
    }

    drop(inner); // Release read lock before solving (needs write lock).

    let mut eigen_pairs: Vec<(Ex, usize)> = Vec::new();

    for (factor, mult) in &factors {
        let degree = factor.degree().unwrap_or(0);
        if degree == 0 {
            // Constant factor — not an eigenvalue.
            continue;
        }
        if degree == 1 {
            // Linear factor: ax + b → root = -b/a.
            let coeffs = factor.coeffs();
            let a = &coeffs[1]; // coefficient of x
            let b = &coeffs[0]; // constant term
            let root_val = -(b / a);
            // Build the root as an Ex.
            let root_ex = crate::rational(
                root_val.numer().clone().try_into().unwrap_or(0i64),
                root_val.denom().clone().try_into().unwrap_or(1i64),
            );
            // For large BigInt roots that don't fit i64, use the general path.
            let root_check: Result<i64, _> = root_val.numer().clone().try_into();
            let denom_check: Result<i64, _> = root_val.denom().clone().try_into();
            let root_ex = if root_check.is_ok() && denom_check.is_ok() {
                crate::rational(root_check.unwrap(), denom_check.unwrap())
            } else {
                // Root doesn't fit i64 — fall back to constructing from BigInt.
                // Use the flat solver as a workaround.
                let mut write_inner = char_poly.inner.write();
                let factor_expr = crate::poly::polybridge::poly_to_expr(
                    &mut write_inner.arena,
                    factor,
                    var.id,
                );
                drop(write_inner);
                let factor_ex = char_poly.wrap(factor_expr);
                let roots = factor_ex.solve_or_empty(var);
                for r in roots {
                    eigen_pairs.push((r, *mult as usize));
                }
                continue;
            };
            trace!(
                mult,
                "eigvals_via_poly_factor: linear factor, root = {}, mult = {}",
                root_ex,
                mult
            );
            eigen_pairs.push((root_ex, *mult as usize));
        } else {
            // Higher-degree factor: solve it for roots.
            let mut write_inner = char_poly.inner.write();
            let factor_expr = crate::poly::polybridge::poly_to_expr(
                &mut write_inner.arena,
                factor,
                var.id,
            );
            drop(write_inner);
            let factor_ex = char_poly.wrap(factor_expr);
            let roots = factor_ex.solve_or_empty(var);
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
    }

    Some(eigen_pairs)
}

/// Fallback: derivative-based multiplicity detection.
///
/// For each root `r`, finds the smallest `k` such that `p^(k)(r) ≠ 0`.
/// That `k` is the algebraic multiplicity.
fn eigvals_via_derivative(char_poly: &Ex, var: &Ex) -> Vec<(Ex, usize)> {
    let all_roots = char_poly.solve_or_empty(var);
    if all_roots.is_empty() {
        return Vec::new();
    }

    // Deduplicate structurally.
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
        // mult is the order of the zero — that's the algebraic multiplicity.
        // If the loop never found a nonzero value, assume mult = 1.
        let mult = if mult == 0 { 1 } else { mult };
        trace!(
            mult,
            "eigvals_via_derivative: root has algebraic multiplicity {}",
            mult
        );
        eigen_pairs.push((root.clone(), mult));
    }

    // Sanity: if total is 0, fall back to dedup counting.
    let total: usize = eigen_pairs.iter().map(|(_, m)| *m).sum();
    if total == 0 {
        warn!("eigvals_via_derivative: multiplicity detection failed, using dedup fallback");
        eigen_pairs.clear();
        for root in &all_roots {
            if let Some(entry) = eigen_pairs.iter_mut().find(|(e, _)| e == root) {
                entry.1 += 1;
            } else {
                eigen_pairs.push((root.clone(), 1));
            }
        }
    }

    eigen_pairs
}

// ═══════════════════════════════════════════════════════════════════════════
// Jordan form helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Compute nullspace of `(a_minus_lambda)^power`.
fn jordan_null_power(a_minus_lambda: &Matrix, power: usize, _n: usize) -> Result<Vec<Matrix>, SymplexError> {
    if power == 0 {
        return Ok(Vec::new());
    }
    let mut m = a_minus_lambda.clone();
    for _ in 1..power {
        m = m.matmul(a_minus_lambda)?;
    }
    Ok(m.nullspace())
}

/// Compute `(a_minus_lambda)^power * vec` where vec is a column vector.
fn matrix_pow_vec(a_minus_lambda: &Matrix, vec: &Matrix, power: usize) -> Result<Matrix, SymplexError> {
    let mut result = vec.clone();
    for _ in 0..power {
        result = a_minus_lambda.matmul(&result)?;
    }
    Ok(result)
}

/// Compute n! for small n (used by matrix_exp Jordan block formula).
fn factorial_usize(n: usize) -> usize {
    (1..=n).product::<usize>().max(1)
}

/// Pick a vector from `candidates` that is linearly independent from all
/// vectors in `exclude`.  Uses RREF to check independence.
fn pick_independent_vec(
    candidates: &[Matrix],
    exclude: &[&Matrix],
    _n: usize,
) -> Result<Option<Matrix>, SymplexError> {
    if candidates.is_empty() {
        return Ok(None);
    }
    if exclude.is_empty() {
        return Ok(Some(candidates[0].clone()));
    }
    for candidate in candidates {
        // Stack exclude vectors + candidate, check if rank increases.
        let mut cols: Vec<&Matrix> = exclude.to_vec();
        cols.push(candidate);
        let combined = Matrix::hstack(&cols)?;
        let rank = combined.rank();
        if rank == cols.len() {
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
        sum = sum + &(a.get(i, 0) * b.get(i, 0));
    }
    sum
}

// ═══════════════════════════════════════════════════════════════════════════
// Display
// ═══════════════════════════════════════════════════════════════════════════

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.nrows == 1 {
            // Single row — inline format: [[a, b, c]]
            write!(f, "[[")?;
            for (j, elem) in self.rows[0].iter().enumerate() {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{elem}")?;
            }
            write!(f, "]]")
        } else {
            // Multi-row — one row per line for readability.
            writeln!(f, "[")?;
            for (i, row) in self.rows.iter().enumerate() {
                write!(f, "  [")?;
                for (j, elem) in row.iter().enumerate() {
                    if j > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{elem}")?;
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
}

impl fmt::Debug for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Matrix({}×{}, ", self.nrows, self.ncols)?;
        fmt::Display::fmt(self, f)?;
        write!(f, ")")
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Operator overloads
// ═══════════════════════════════════════════════════════════════════════════

// Matrix + Matrix (element-wise addition)
impl std::ops::Add for &Matrix {
    type Output = Matrix;
    fn add(self, rhs: &Matrix) -> Matrix {
        self.add_elementwise(rhs).expect("operator +: dimensions validated by type")
    }
}
impl std::ops::Add for Matrix {
    type Output = Matrix;
    fn add(self, rhs: Matrix) -> Matrix {
        self.add_elementwise(&rhs).expect("operator +: dimensions validated by type")
    }
}
impl std::ops::Add<&Matrix> for Matrix {
    type Output = Matrix;
    fn add(self, rhs: &Matrix) -> Matrix {
        self.add_elementwise(rhs).expect("operator +: dimensions validated by type")
    }
}
impl std::ops::Add<Matrix> for &Matrix {
    type Output = Matrix;
    fn add(self, rhs: Matrix) -> Matrix {
        self.add_elementwise(&rhs).expect("operator +: dimensions validated by type")
    }
}

// Matrix - Matrix (element-wise subtraction)
impl std::ops::Sub for &Matrix {
    type Output = Matrix;
    fn sub(self, rhs: &Matrix) -> Matrix {
        self.sub_elementwise(rhs).expect("operator -: dimensions validated by type")
    }
}
impl std::ops::Sub for Matrix {
    type Output = Matrix;
    fn sub(self, rhs: Matrix) -> Matrix {
        self.sub_elementwise(&rhs).expect("operator -: dimensions validated by type")
    }
}
impl std::ops::Sub<&Matrix> for Matrix {
    type Output = Matrix;
    fn sub(self, rhs: &Matrix) -> Matrix {
        self.sub_elementwise(rhs).expect("operator -: dimensions validated by type")
    }
}
impl std::ops::Sub<Matrix> for &Matrix {
    type Output = Matrix;
    fn sub(self, rhs: Matrix) -> Matrix {
        self.sub_elementwise(&rhs).expect("operator -: dimensions validated by type")
    }
}

// Matrix * Matrix (matrix multiplication, NOT element-wise)
impl std::ops::Mul for &Matrix {
    type Output = Matrix;
    fn mul(self, rhs: &Matrix) -> Matrix {
        self.matmul(rhs).expect("operator *: dimensions validated by type")
    }
}
impl std::ops::Mul for Matrix {
    type Output = Matrix;
    fn mul(self, rhs: Matrix) -> Matrix {
        self.matmul(&rhs).expect("operator *: dimensions validated by type")
    }
}
impl std::ops::Mul<&Matrix> for Matrix {
    type Output = Matrix;
    fn mul(self, rhs: &Matrix) -> Matrix {
        self.matmul(rhs).expect("operator *: dimensions validated by type")
    }
}
impl std::ops::Mul<Matrix> for &Matrix {
    type Output = Matrix;
    fn mul(self, rhs: Matrix) -> Matrix {
        self.matmul(&rhs).expect("operator *: dimensions validated by type")
    }
}

// -Matrix (negation)
impl std::ops::Neg for &Matrix {
    type Output = Matrix;
    fn neg(self) -> Matrix {
        let neg_one = crate::int(-1);
        self.scale(&neg_one)
    }
}
impl std::ops::Neg for Matrix {
    type Output = Matrix;
    fn neg(self) -> Matrix {
        -&self
    }
}

// Matrix * &Ex (scalar multiplication)
impl std::ops::Mul<&Ex> for &Matrix {
    type Output = Matrix;
    fn mul(self, rhs: &Ex) -> Matrix {
        self.scale(rhs)
    }
}
impl std::ops::Mul<&Ex> for Matrix {
    type Output = Matrix;
    fn mul(self, rhs: &Ex) -> Matrix {
        self.scale(rhs)
    }
}

// Matrix * i64 (scalar multiplication by integer)
impl std::ops::Mul<i64> for &Matrix {
    type Output = Matrix;
    fn mul(self, rhs: i64) -> Matrix {
        let s = crate::int(rhs);
        self.scale(&s)
    }
}
impl std::ops::Mul<i64> for Matrix {
    type Output = Matrix;
    fn mul(self, rhs: i64) -> Matrix {
        (&self) * rhs
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Constructor tests ──────────────────────────────────────────────

    #[test]
    fn identity_2x2() {
        let m = Matrix::identity(2);
        assert_eq!(m.nrows(), 2);
        assert_eq!(m.ncols(), 2);
        assert_eq!(format!("{}", m.get(0, 0)), "1");
        assert_eq!(format!("{}", m.get(0, 1)), "0");
        assert_eq!(format!("{}", m.get(1, 0)), "0");
        assert_eq!(format!("{}", m.get(1, 1)), "1");
    }

    #[test]
    fn identity_3x3() {
        let m = Matrix::identity(3);
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
        let z = Matrix::zeros(3, 3);
        assert_eq!(format!("{}", z.get(1, 1)), "0");

        let m = Matrix::from_fn(2, 2, |i, j| crate::int((i * 2 + j + 1) as i64));
        assert_eq!(format!("{}", m.get(0, 0)), "1");
        assert_eq!(format!("{}", m.get(0, 1)), "2");
        assert_eq!(format!("{}", m.get(1, 0)), "3");
        assert_eq!(format!("{}", m.get(1, 1)), "4");
    }

    #[test]
    fn row_and_col_vectors() {
        let rv = Matrix::row_vector(vec![crate::int(1), crate::int(2), crate::int(3)]);
        assert_eq!(rv.shape(), (1, 3));
        assert_eq!(format!("{}", rv.get(0, 1)), "2");

        let cv = Matrix::col_vector(vec![crate::int(10), crate::int(20)]);
        assert_eq!(cv.shape(), (2, 1));
        assert_eq!(format!("{}", cv.get(1, 0)), "20");
    }

    // ── Accessor tests ─────────────────────────────────────────────────

    #[test]
    fn get_mut_works() {
        let mut m = Matrix::zeros(2, 2);
        *m.get_mut(0, 1) = crate::int(42);
        assert_eq!(format!("{}", m.get(0, 1)), "42");
    }

    #[test]
    fn row_accessor() {
        let m = Matrix::new(vec![
            vec![crate::int(1), crate::int(2)],
            vec![crate::int(3), crate::int(4)],
        ]);
        let r = m.row(0);
        assert_eq!(r.len(), 2);
        assert_eq!(format!("{}", r[0]), "1");
        assert_eq!(format!("{}", r[1]), "2");
    }

    #[test]
    fn try_get_works() {
        let m = Matrix::zeros(2, 2);
        assert!(m.try_get(0, 0).is_some());
        assert!(m.try_get(1, 1).is_some());
        assert!(m.try_get(2, 0).is_none());
        assert!(m.try_get(0, 2).is_none());
    }

    // ── Transpose ──────────────────────────────────────────────────────

    #[test]
    fn transpose() {
        let a = crate::var("a");
        let b = crate::var("b");
        let c = crate::var("c");
        let d = crate::var("d");
        let m = Matrix::new(vec![vec![a.clone(), b.clone()], vec![c.clone(), d.clone()]]);
        let t = m.transpose();
        assert_eq!(format!("{}", t.get(0, 0)), "a");
        assert_eq!(format!("{}", t.get(0, 1)), "c");
        assert_eq!(format!("{}", t.get(1, 0)), "b");
        assert_eq!(format!("{}", t.get(1, 1)), "d");
    }

    #[test]
    fn transpose_non_square() {
        let m = Matrix::new(vec![
            vec![crate::int(1), crate::int(2), crate::int(3)],
            vec![crate::int(4), crate::int(5), crate::int(6)],
        ]);
        let t = m.transpose();
        assert_eq!(t.shape(), (3, 2));
        assert_eq!(format!("{}", t.get(2, 0)), "3");
        assert_eq!(format!("{}", t.get(2, 1)), "6");
    }

    // ── Addition / Subtraction ─────────────────────────────────────────

    #[test]
    fn matrix_add() {
        let m1 = Matrix::new(vec![
            vec![crate::int(1), crate::int(2)],
            vec![crate::int(3), crate::int(4)],
        ]);
        let m2 = Matrix::new(vec![
            vec![crate::int(5), crate::int(6)],
            vec![crate::int(7), crate::int(8)],
        ]);
        let sum = m1.add(&m2).unwrap();
        assert_eq!(format!("{}", sum.get(0, 0)), "6");
        assert_eq!(format!("{}", sum.get(1, 1)), "12");
    }

    #[test]
    fn matrix_sub() {
        let m1 = Matrix::new(vec![vec![crate::int(10), crate::int(20)]]);
        let m2 = Matrix::new(vec![vec![crate::int(3), crate::int(7)]]);
        let diff = m1.sub(&m2).unwrap();
        assert_eq!(format!("{}", diff.get(0, 0)), "7");
        assert_eq!(format!("{}", diff.get(0, 1)), "13");
    }

    // ── Scalar multiplication ──────────────────────────────────────────

    #[test]
    fn scale() {
        let m = Matrix::identity(2);
        let two = crate::int(2);
        let scaled = m.scale(&two);
        assert_eq!(format!("{}", scaled.get(0, 0)), "2");
        assert_eq!(format!("{}", scaled.get(0, 1)), "0");
    }

    #[test]
    fn scale_symbolic() {
        let x = crate::var("x");
        let m = Matrix::new(vec![vec![crate::int(1), crate::int(2)]]);
        let scaled = m.scale(&x);
        let s0 = format!("{}", scaled.get(0, 0));
        let s1 = format!("{}", scaled.get(0, 1));
        assert!(s0.contains("x"), "scaled[0,0] should contain x: {s0}");
        assert!(s1.contains("x"), "scaled[0,1] should contain x: {s1}");
    }

    // ── Matrix multiplication ──────────────────────────────────────────

    #[test]
    fn matmul_2x2() {
        let m = Matrix::identity(2);
        let a = crate::var("a");
        let b = crate::var("b");
        let c = crate::var("c");
        let d = crate::var("d");
        let n = Matrix::new(vec![vec![a.clone(), b.clone()], vec![c.clone(), d.clone()]]);
        let result = m.matmul(&n).unwrap();
        // I * N = N
        assert_eq!(format!("{}", result.get(0, 0)), "a");
        assert_eq!(format!("{}", result.get(1, 1)), "d");
    }

    #[test]
    fn matmul_non_square() {
        // (1×2) * (2×1) → (1×1)
        let rv = Matrix::row_vector(vec![crate::int(2), crate::int(3)]);
        let cv = Matrix::col_vector(vec![crate::int(4), crate::int(5)]);
        let result = rv.matmul(&cv).unwrap();
        assert_eq!(result.shape(), (1, 1));
        // 2*4 + 3*5 = 8 + 15 = 23
        assert_eq!(format!("{}", result.get(0, 0)), "23");
    }

    #[test]
    fn matmul_numeric() {
        let a = Matrix::new(vec![
            vec![crate::int(1), crate::int(2)],
            vec![crate::int(3), crate::int(4)],
        ]);
        let b = Matrix::new(vec![
            vec![crate::int(5), crate::int(6)],
            vec![crate::int(7), crate::int(8)],
        ]);
        let c = a.matmul(&b).unwrap();
        // [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        assert_eq!(format!("{}", c.get(0, 0)), "19");
        assert_eq!(format!("{}", c.get(0, 1)), "22");
        assert_eq!(format!("{}", c.get(1, 0)), "43");
        assert_eq!(format!("{}", c.get(1, 1)), "50");
    }

    // ── Trace ──────────────────────────────────────────────────────────

    #[test]
    fn trace_2x2() {
        let a = crate::var("a");
        let b = crate::var("b");
        let c = crate::var("c");
        let d = crate::var("d");
        let m = Matrix::new(vec![vec![a.clone(), b.clone()], vec![c.clone(), d.clone()]]);
        let tr = m.trace().unwrap();
        let s = format!("{tr}");
        assert!(
            s.contains("a") && s.contains("d"),
            "trace should be a+d: {s}"
        );
    }

    #[test]
    fn trace_numeric() {
        let m = Matrix::new(vec![
            vec![crate::int(1), crate::int(2)],
            vec![crate::int(3), crate::int(4)],
        ]);
        let tr = m.trace().unwrap();
        assert_eq!(format!("{tr}"), "5");
    }

    // ── Determinant ────────────────────────────────────────────────────

    #[test]
    fn det_1x1() {
        let a = crate::var("a");
        let m = Matrix::new(vec![vec![a.clone()]]);
        let det = m.det().unwrap();
        assert_eq!(format!("{det}"), "a");
    }

    #[test]
    fn det_2x2() {
        let a = crate::var("a");
        let b = crate::var("b");
        let c = crate::var("c");
        let d = crate::var("d");
        let m = Matrix::new(vec![vec![a.clone(), b.clone()], vec![c.clone(), d.clone()]]);
        let det = m.det().unwrap();
        // det = ad - bc
        let s = format!("{det}");
        assert!(
            s.contains("a") && s.contains("d"),
            "det should contain ad: {s}"
        );
    }

    #[test]
    fn det_2x2_numeric() {
        let m = Matrix::new(vec![
            vec![crate::int(3), crate::int(8)],
            vec![crate::int(4), crate::int(6)],
        ]);
        let det = m.det().unwrap();
        // 3*6 - 8*4 = 18 - 32 = -14
        assert_eq!(format!("{det}"), "-14");
    }

    #[test]
    fn det_3x3_numeric() {
        let m = Matrix::new(vec![
            vec![crate::int(1), crate::int(2), crate::int(3)],
            vec![crate::int(4), crate::int(5), crate::int(6)],
            vec![crate::int(7), crate::int(8), crate::int(9)],
        ]);
        let det = m.det().unwrap();
        // This matrix is singular: det = 0
        assert_eq!(format!("{det}"), "0");
    }

    #[test]
    fn det_3x3_nonsingular() {
        let m = Matrix::new(vec![
            vec![crate::int(1), crate::int(0), crate::int(2)],
            vec![crate::int(0), crate::int(1), crate::int(0)],
            vec![crate::int(3), crate::int(0), crate::int(1)],
        ]);
        let det = m.det().unwrap();
        // det = 1*(1*1 - 0*0) - 0 + 2*(0*0 - 1*3) = 1 + 2*(-3) = -5
        assert_eq!(format!("{det}"), "-5");
    }

    // ── Map ────────────────────────────────────────────────────────────

    #[test]
    fn map_doubles() {
        let m = Matrix::new(vec![
            vec![crate::int(1), crate::int(2)],
            vec![crate::int(3), crate::int(4)],
        ]);
        let doubled = m.map(|e| {
            let two = crate::int(2);
            &two * e
        });
        assert_eq!(format!("{}", doubled.get(0, 0)), "2");
        assert_eq!(format!("{}", doubled.get(1, 1)), "8");
    }

    // ── Calculus helpers ───────────────────────────────────────────────

    #[test]
    fn jacobian_test() {
        let x = crate::var("x");
        let y = crate::var("y");
        let f1 = &x.powi(2) + &y; // f1 = x² + y
        let f2 = &x * &y; // f2 = x*y
        let j = jacobian(&[&f1, &f2], &[&x, &y]);
        // J = [[2x, 1], [y, x]]
        assert_eq!(j.nrows(), 2);
        assert_eq!(j.ncols(), 2);
        let s00 = format!("{}", j.get(0, 0));
        assert!(s00.contains("x"), "J[0,0] should be 2x: {s00}");
    }

    #[test]
    fn matrix_diff() {
        let x = crate::var("x");
        let m = Matrix::new(vec![vec![x.powi(2), x.sin()]]);
        let dm = m.diff(&x);
        let s00 = format!("{}", dm.get(0, 0));
        assert!(s00.contains("x"), "d/dx(x²): {s00}");
    }

    #[test]
    fn matrix_subs() {
        let x = crate::var("x");
        let m = Matrix::new(vec![vec![x.powi(2), x.clone()]]);
        let result = m.subs(&x, &crate::int(3));
        assert_eq!(format!("{}", result.get(0, 0)), "9");
        assert_eq!(format!("{}", result.get(0, 1)), "3");
    }

    #[test]
    fn matrix_eval() {
        let m = Matrix::new(vec![vec![crate::pi().cos(), crate::int(2) + crate::int(3)]]);
        let evaled = m.eval();
        assert_eq!(format!("{}", evaled.get(0, 0)), "-1");
        assert_eq!(format!("{}", evaled.get(0, 1)), "5");
    }

    #[test]
    fn matrix_expand() {
        let x = crate::var("x");
        let expr = (&x + crate::int(1)).powi(2);
        let m = Matrix::new(vec![vec![expr]]);
        let expanded = m.expand();
        let s = format!("{}", expanded.get(0, 0));
        // Should be expanded: 1 + x^2 + 2*x (or similar)
        assert!(s.contains("x"), "expanded should contain x: {s}");
    }

    // ── Display ────────────────────────────────────────────────────────

    // ── Eigenvector / Diagonalization tests ─────────────────────────

    #[test]
    fn eigenvects_2x2_distinct() {
        let var = crate::var("lam_ev1");
        // Upper-triangular: eigenvalues are 2 and 3 on the diagonal.
        let m = Matrix::new(vec![
            vec![crate::int(2), crate::int(1)],
            vec![crate::int(0), crate::int(3)],
        ]);
        let evs = m.eigenvects(&var).expect("eigenvects should succeed for square matrix");
        assert_eq!(evs.len(), 2, "should have 2 distinct eigenvalues");
        for (eigenval, mult, vecs) in &evs {
            assert_eq!(*mult, 1, "each mult should be 1");
            assert_eq!(vecs.len(), 1, "each eigenspace should be 1-dimensional");
            // Verify A·v = λ·v
            let av = m.matmul(&vecs[0]).unwrap();
            let lambda_v = vecs[0].scale(eigenval);
            for i in 0..2 {
                let diff = (av.get(i, 0) - lambda_v.get(i, 0)).expand().eval();
                assert!(
                    diff.is_zero_structural(),
                    "A·v ≠ λ·v at row {i}, got {diff}"
                );
            }
        }
    }

    #[test]
    fn eigenvects_non_square_returns_error() {
        let var = crate::var("lam_nonsq");
        let m = Matrix::new(vec![
            vec![crate::int(1), crate::int(2), crate::int(3)],
            vec![crate::int(4), crate::int(5), crate::int(6)],
        ]);
        let result = m.eigenvects(&var);
        assert!(result.is_err(), "non-square matrix should return Err");
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("square"),
            "error should mention 'square': {err_msg}"
        );
    }

    #[test]
    fn eigenvects_diagonal_matrix() {
        let var = crate::var("lam_ev2");
        let m = Matrix::diag(&[crate::int(5), crate::int(-3)]);
        let evs = m.eigenvects(&var).expect("eigenvects should succeed");
        assert_eq!(evs.len(), 2, "diagonal matrix has 2 eigenvalues");
        // Eigenvalues should be 5 and -3.
        let vals: Vec<_> = evs.iter().map(|(v, _, _)| v.clone()).collect();
        assert!(
            vals.contains(&crate::int(5)) && vals.contains(&crate::int(-3)),
            "eigenvalues should be 5 and -3, got {:?}",
            vals
        );
    }

    #[test]
    fn diagonalize_upper_triangular() {
        let var = crate::var("lam_diag1");
        let m = Matrix::new(vec![
            vec![crate::int(2), crate::int(1)],
            vec![crate::int(0), crate::int(3)],
        ]);
        let (p, d) = m
            .diagonalize(&var)
            .expect("upper triangular should be diagonalizable");
        assert_eq!(p.nrows(), 2);
        assert_eq!(d.nrows(), 2);
        // Verify: P * D * P^{-1} ≈ M
        if let Ok(p_inv) = p.inv() {
            let reconstructed = p.matmul(&d).unwrap().matmul(&p_inv).unwrap();
            for i in 0..2 {
                for j in 0..2 {
                    let diff =
                        (reconstructed.get(i, j) - m.get(i, j)).expand().eval();
                    assert!(
                        diff.simplify().is_zero_structural(),
                        "P·D·P⁻¹ ≠ M at ({i},{j})"
                    );
                }
            }
        }
    }

    #[test]
    fn diagonalize_non_diagonalizable() {
        let var = crate::var("lam_nd");
        // [[1,1],[0,1]] — defective: eigenvalue 1 alg-mult 2, geom-mult 1
        let m = Matrix::new(vec![
            vec![crate::int(1), crate::int(1)],
            vec![crate::int(0), crate::int(1)],
        ]);
        let result = m.diagonalize(&var);
        assert!(
            result.is_err(),
            "defective matrix should return Err from diagonalize"
        );
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("not diagonalizable"),
            "error should mention 'not diagonalizable': {err_msg}"
        );
    }

    #[test]
    fn is_diagonalizable_yes() {
        let var = crate::var("lam_diag_y");
        let m = Matrix::diag(&[crate::int(1), crate::int(2), crate::int(3)]);
        assert_eq!(m.is_diagonalizable(&var).unwrap(), true);
    }

    // ── Jordan form tests ──────────────────────────────────────────

    #[test]
    fn jordan_form_diagonal() {
        // A diagonal matrix has trivial Jordan form = itself.
        let var = crate::var("lam_jf1");
        let m = Matrix::diag(&[crate::int(1), crate::int(2), crate::int(3)]);
        let (p, j) = m.jordan_form(&var).expect("should succeed");
        assert_eq!(j.nrows(), 3);
        // J should be diagonal (same as D from diagonalize).
        // Verify P * J * P^{-1} = M
        if let Ok(p_inv) = p.inv() {
            let reconstructed = p.matmul(&j).unwrap().matmul(&p_inv).unwrap();
            for i in 0..3 {
                for k in 0..3 {
                    let diff = (reconstructed.get(i, k) - m.get(i, k)).expand().eval();
                    assert!(
                        diff.simplify().is_zero_structural(),
                        "P·J·P⁻¹ ≠ M at ({i},{k}): {diff}"
                    );
                }
            }
        }
    }

    #[test]
    fn jordan_form_defective_2x2() {
        // [[1,1],[0,1]] — eigenvalue 1, alg mult 2, geom mult 1
        // Jordan form should be [[1,1],[0,1]] (single 2×2 block)
        let var = crate::var("lam_jf2");
        let m = Matrix::new(vec![
            vec![crate::int(1), crate::int(1)],
            vec![crate::int(0), crate::int(1)],
        ]);
        let (_p, j) = m.jordan_form(&var).expect("should succeed for defective matrix");
        assert_eq!(j.nrows(), 2);
        // J should have 1 on diagonal and 1 on superdiagonal
        let j_01 = j.get(0, 1).eval();
        assert!(
            (&j_01 - &crate::int(1)).eval().is_zero_structural(),
            "J[0,1] should be 1 (superdiagonal of Jordan block)"
        );
    }

    #[test]
    fn jordan_form_upper_triangular_distinct() {
        // [[2,1],[0,3]] — distinct eigenvalues, so Jordan = diagonal form
        let var = crate::var("lam_jf3");
        let m = Matrix::new(vec![
            vec![crate::int(2), crate::int(1)],
            vec![crate::int(0), crate::int(3)],
        ]);
        let (p, j) = m.jordan_form(&var).expect("distinct eigenvalues should succeed");
        assert_eq!(j.nrows(), 2);
        // Since eigenvalues are distinct, Jordan form = diagonal form.
        // Verify P * J * P^{-1} = M
        if let Ok(p_inv) = p.inv() {
            let reconstructed = p.matmul(&j).unwrap().matmul(&p_inv).unwrap();
            for i in 0..2 {
                for k in 0..2 {
                    let diff = (reconstructed.get(i, k) - m.get(i, k)).expand().eval();
                    assert!(
                        diff.simplify().is_zero_structural(),
                        "P·J·P⁻¹ ≠ M at ({i},{k}): {diff}"
                    );
                }
            }
        }
    }

    // ── Matrix exponential tests ───────────────────────────────────

    #[test]
    fn matrix_exp_identity() {
        let var = crate::var("lam_mexp1");
        let m = Matrix::identity(2);
        let result = m.matrix_exp(&var).expect("identity should succeed");
        // e^0 = I, so e^I should have e on diagonal (but I = [[1,0],[0,1]])
        // Actually e^I = e * I for I = identity (since I is diagonal with 1s)
        // The diagonal entries should be e^1 = e.
        assert_eq!(result.nrows(), 2);
    }

    #[test]
    fn matrix_exp_zero() {
        let var = crate::var("lam_mexp0");
        let m = Matrix::zeros(2, 2);
        let result = m.matrix_exp(&var).expect("zero matrix should succeed");
        // e^0 = I
        let diag_00 = result.get(0, 0).simplify().eval();
        let diag_11 = result.get(1, 1).simplify().eval();
        let off_01 = result.get(0, 1).simplify().eval();
        assert!(
            (&diag_00 - &crate::int(1)).eval().is_zero_structural(),
            "e^0 [0,0] should be 1, got {diag_00}"
        );
        assert!(
            (&diag_11 - &crate::int(1)).eval().is_zero_structural(),
            "e^0 [1,1] should be 1, got {diag_11}"
        );
        assert!(
            off_01.is_zero_structural(),
            "e^0 [0,1] should be 0, got {off_01}"
        );
    }

    #[test]
    fn matrix_exp_non_square_returns_error() {
        let var = crate::var("lam_mexp_ns");
        let m = Matrix::new(vec![
            vec![crate::int(1), crate::int(2), crate::int(3)],
            vec![crate::int(4), crate::int(5), crate::int(6)],
        ]);
        assert!(m.matrix_exp(&var).is_err());
    }

    #[test]
    fn jordan_form_non_square_returns_error() {
        let var = crate::var("lam_jf_ns");
        let m = Matrix::new(vec![
            vec![crate::int(1), crate::int(2), crate::int(3)],
            vec![crate::int(4), crate::int(5), crate::int(6)],
        ]);
        assert!(m.jordan_form(&var).is_err());
    }

    #[test]
    fn diagonalize_non_square_returns_error() {
        let var = crate::var("lam_diag_nonsq");
        let m = Matrix::new(vec![
            vec![crate::int(1), crate::int(2), crate::int(3)],
            vec![crate::int(4), crate::int(5), crate::int(6)],
        ]);
        assert!(
            m.diagonalize(&var).is_err(),
            "non-square should return Err"
        );
        assert!(
            m.is_diagonalizable(&var).is_err(),
            "non-square should return Err"
        );
    }

    // ── Display tests ──────────────────────────────────────────────

    #[test]
    fn matrix_display() {
        let m = Matrix::new(vec![
            vec![crate::int(1), crate::int(2)],
            vec![crate::int(3), crate::int(4)],
        ]);
        let s = format!("{m}");
        assert!(s.contains("1") && s.contains("4"), "display: {s}");
    }

    #[test]
    fn row_vector_display() {
        let m = Matrix::row_vector(vec![crate::int(1), crate::int(2), crate::int(3)]);
        let s = format!("{m}");
        // Single-row matrices use inline format
        assert!(s.starts_with("[["), "should start with [[: {s}");
        assert!(s.ends_with("]]"), "should end with ]]: {s}");
    }

    // ── Error tests (non-square / shape mismatch) ──────────────────────

    #[test]
    fn add_mismatched_shapes_returns_err() {
        let a = Matrix::zeros(2, 3);
        let b = Matrix::zeros(3, 2);
        let result = a.add(&b);
        assert!(result.is_err(), "mismatched shapes should return Err");
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("add"),
            "error should mention add: {err_msg}"
        );
    }

    #[test]
    fn matmul_incompatible_returns_err() {
        let a = Matrix::zeros(2, 3);
        let b = Matrix::zeros(2, 3);
        let result = a.matmul(&b);
        assert!(result.is_err(), "incompatible dimensions should return Err");
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("multiply"),
            "error should mention multiply: {err_msg}"
        );
    }

    #[test]
    fn trace_non_square_returns_err() {
        let m = Matrix::zeros(2, 3);
        let result = m.trace();
        assert!(result.is_err(), "non-square should return Err");
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("square"),
            "error should mention square: {err_msg}"
        );
    }

    #[test]
    fn det_non_square_returns_err() {
        let m = Matrix::zeros(2, 3);
        let result = m.det();
        assert!(result.is_err(), "non-square should return Err");
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("square"),
            "error should mention square: {err_msg}"
        );
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn get_out_of_bounds_panics() {
        let m = Matrix::zeros(2, 2);
        let _ = m.get(2, 0);
    }
}
