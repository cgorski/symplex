//! Symbolic matrix type.
//!
//! Provides [`Matrix`], a dense matrix of symbolic expressions with
//! operations: construction, display, transpose, addition, scalar
//! multiplication, matrix multiplication, determinant, and trace.

use crate::expr::Ex;
use std::fmt;

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
    /// # Panics
    ///
    /// Panics if shapes differ.
    pub fn add(&self, other: &Matrix) -> Matrix {
        assert_eq!(
            self.shape(),
            other.shape(),
            "Cannot add matrices with shapes {:?} and {:?}",
            self.shape(),
            other.shape()
        );
        let rows = (0..self.nrows)
            .map(|i| {
                (0..self.ncols)
                    .map(|j| &self.rows[i][j] + &other.rows[i][j])
                    .collect()
            })
            .collect();
        Matrix {
            nrows: self.nrows,
            ncols: self.ncols,
            rows,
        }
    }

    /// Element-wise subtraction. Both matrices must have the same shape.
    ///
    /// # Panics
    ///
    /// Panics if shapes differ.
    pub fn sub(&self, other: &Matrix) -> Matrix {
        assert_eq!(
            self.shape(),
            other.shape(),
            "Cannot subtract matrices with shapes {:?} and {:?}",
            self.shape(),
            other.shape()
        );
        let rows = (0..self.nrows)
            .map(|i| {
                (0..self.ncols)
                    .map(|j| &self.rows[i][j] - &other.rows[i][j])
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
    ///
    /// Each element is computed symbolically:
    /// `result[i][j] = Σ_k self[i][k] * other[k][j]`.
    ///
    /// # Panics
    ///
    /// Panics if `self.ncols != other.nrows`.
    pub fn matmul(&self, other: &Matrix) -> Matrix {
        assert_eq!(
            self.ncols, other.nrows,
            "Cannot multiply {}×{} by {}×{} matrices",
            self.nrows, self.ncols, other.nrows, other.ncols
        );
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
        Matrix {
            nrows: self.nrows,
            ncols: other.ncols,
            rows,
        }
    }

    /// Trace: sum of the diagonal elements.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    pub fn trace(&self) -> Ex {
        assert_eq!(
            self.nrows, self.ncols,
            "Trace requires a square matrix, got {}×{}",
            self.nrows, self.ncols
        );
        let mut acc = self.rows[0][0].clone();
        for i in 1..self.nrows {
            acc = acc + &self.rows[i][i];
        }
        acc
    }

    /// Determinant via cofactor expansion along the first row.
    ///
    /// Determinant of a square matrix.
    ///
    /// For small matrices (≤ 4×4) this uses cofactor expansion.
    /// For larger matrices it switches to LU decomposition (O(n³)).
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    pub fn det(&self) -> Ex {
        assert_eq!(
            self.nrows, self.ncols,
            "Determinant requires a square matrix, got {}×{}",
            self.nrows, self.ncols
        );

        // For small matrices, cofactor expansion is fine
        if self.nrows <= 4 {
            return self.det_cofactor();
        }

        // For larger matrices, use LU decomposition (O(n³))
        self.det_lu()
    }

    /// Determinant via cofactor expansion (O(n!) — only for small matrices).
    fn det_cofactor(&self) -> Ex {
        self.det_cofactor_inner(&self.rows)
    }

    /// Determinant via LU decomposition (O(n³)).
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
    /// # Panics
    ///
    /// Panics if the matrix is not square or has dimension 1.
    pub fn minor(&self, row: usize, col: usize) -> Matrix {
        assert_eq!(
            self.nrows, self.ncols,
            "minor requires a square matrix, got {}×{}",
            self.nrows, self.ncols
        );
        assert!(self.nrows > 1, "minor requires matrix dimension > 1");
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
        Matrix::new(rows)
    }

    /// Cofactor C(i, j) = (-1)^(i+j) * det(minor(i, j)).
    pub fn cofactor(&self, row: usize, col: usize) -> Ex {
        let minor_det = self.minor(row, col).det();
        if (row + col).is_multiple_of(2) {
            minor_det
        } else {
            -minor_det
        }
    }

    /// Adjugate matrix (transpose of the cofactor matrix).
    ///
    /// `adj(A)[i][j] = cofactor(A, j, i)`.
    pub fn adjugate(&self) -> Matrix {
        let n = self.nrows();
        assert_eq!(
            n,
            self.ncols(),
            "adjugate requires a square matrix, got {}×{}",
            self.nrows(),
            self.ncols()
        );
        let mut rows = Vec::new();
        for j in 0..n {
            let mut row = Vec::new();
            for i in 0..n {
                row.push(self.cofactor(i, j)); // note: transposed
            }
            rows.push(row);
        }
        Matrix::new(rows)
    }

    /// Matrix inverse: A⁻¹ = adj(A) / det(A).
    ///
    /// Returns `None` if the matrix is singular (determinant is
    /// structurally zero).
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    pub fn inv(&self) -> Option<Matrix> {
        assert_eq!(
            self.nrows(),
            self.ncols(),
            "inverse requires a square matrix, got {}×{}",
            self.nrows(),
            self.ncols()
        );
        let d = self.det();
        if d.is_zero_structural() {
            return None;
        }
        // 1×1 special case: inverse is just [[1/a]]
        if self.nrows() == 1 {
            let one_over_det = &Ex::one() / &d;
            return Some(Matrix::new(vec![vec![one_over_det]]));
        }
        let adj = self.adjugate();
        let one_over_det = &Ex::one() / &d;
        Some(adj.scale(&one_over_det))
    }

    // ── Characteristic polynomial & eigenvalues ────────────────────────

    /// Characteristic polynomial: det(A − λI).
    ///
    /// Returns the polynomial as an [`Ex`] in the given variable `var`
    /// (which plays the role of λ).
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    pub fn char_poly(&self, var: &Ex) -> Ex {
        let n = self.nrows();
        assert_eq!(
            n,
            self.ncols(),
            "char_poly requires a square matrix, got {}×{}",
            self.nrows(),
            self.ncols()
        );
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
        m.det().expand()
    }

    /// Eigenvalues: solve `char_poly(var) = 0` for `var`.
    ///
    /// Returns a list of eigenvalues. If the solver cannot factor the
    /// characteristic polynomial, the returned list may be empty.
    pub fn eigenvals(&self, var: &Ex) -> Vec<Ex> {
        let cp = self.char_poly(var);
        cp.solve_or_empty(var)
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
pub fn jacobian(funcs: &[Ex], vars: &[Ex]) -> Matrix {
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
    /// # Panics
    ///
    /// Panics if `matrices` is empty or row counts differ.
    pub fn hstack(matrices: &[&Matrix]) -> Matrix {
        assert!(!matrices.is_empty(), "hstack: need at least one matrix");
        let nrows = matrices[0].nrows;
        for (idx, m) in matrices.iter().enumerate() {
            assert_eq!(
                m.nrows, nrows,
                "hstack: matrix {idx} has {} rows, expected {nrows}",
                m.nrows
            );
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
        Matrix { rows, nrows, ncols }
    }

    /// Stack matrices vertically (on top of each other).
    /// All matrices must have the same number of columns.
    ///
    /// # Panics
    ///
    /// Panics if `matrices` is empty or column counts differ.
    pub fn vstack(matrices: &[&Matrix]) -> Matrix {
        assert!(!matrices.is_empty(), "vstack: need at least one matrix");
        let ncols = matrices[0].ncols;
        for (idx, m) in matrices.iter().enumerate() {
            assert_eq!(
                m.ncols, ncols,
                "vstack: matrix {idx} has {} cols, expected {ncols}",
                m.ncols
            );
        }
        let nrows: usize = matrices.iter().map(|m| m.nrows).sum();
        let rows: Vec<Vec<Ex>> = matrices
            .iter()
            .flat_map(|m| m.rows.iter().cloned())
            .collect();
        Matrix { rows, nrows, ncols }
    }
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
        let sum = m1.add(&m2);
        assert_eq!(format!("{}", sum.get(0, 0)), "6");
        assert_eq!(format!("{}", sum.get(1, 1)), "12");
    }

    #[test]
    fn matrix_sub() {
        let m1 = Matrix::new(vec![vec![crate::int(10), crate::int(20)]]);
        let m2 = Matrix::new(vec![vec![crate::int(3), crate::int(7)]]);
        let diff = m1.sub(&m2);
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
        let result = m.matmul(&n);
        // I * N = N
        assert_eq!(format!("{}", result.get(0, 0)), "a");
        assert_eq!(format!("{}", result.get(1, 1)), "d");
    }

    #[test]
    fn matmul_non_square() {
        // (1×2) * (2×1) → (1×1)
        let rv = Matrix::row_vector(vec![crate::int(2), crate::int(3)]);
        let cv = Matrix::col_vector(vec![crate::int(4), crate::int(5)]);
        let result = rv.matmul(&cv);
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
        let c = a.matmul(&b);
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
        let tr = m.trace();
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
        let tr = m.trace();
        assert_eq!(format!("{tr}"), "5");
    }

    // ── Determinant ────────────────────────────────────────────────────

    #[test]
    fn det_1x1() {
        let a = crate::var("a");
        let m = Matrix::new(vec![vec![a.clone()]]);
        let det = m.det();
        assert_eq!(format!("{det}"), "a");
    }

    #[test]
    fn det_2x2() {
        let a = crate::var("a");
        let b = crate::var("b");
        let c = crate::var("c");
        let d = crate::var("d");
        let m = Matrix::new(vec![vec![a.clone(), b.clone()], vec![c.clone(), d.clone()]]);
        let det = m.det();
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
        let det = m.det();
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
        let det = m.det();
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
        let det = m.det();
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
        let j = jacobian(&[f1, f2], &[x.clone(), y.clone()]);
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

    // ── Panics ─────────────────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "Cannot add matrices")]
    fn add_mismatched_shapes_panics() {
        let a = Matrix::zeros(2, 3);
        let b = Matrix::zeros(3, 2);
        let _ = a.add(&b);
    }

    #[test]
    #[should_panic(expected = "Cannot multiply")]
    fn matmul_incompatible_panics() {
        let a = Matrix::zeros(2, 3);
        let b = Matrix::zeros(2, 3);
        let _ = a.matmul(&b);
    }

    #[test]
    #[should_panic(expected = "Trace requires a square matrix")]
    fn trace_non_square_panics() {
        let m = Matrix::zeros(2, 3);
        let _ = m.trace();
    }

    #[test]
    #[should_panic(expected = "Determinant requires a square matrix")]
    fn det_non_square_panics() {
        let m = Matrix::zeros(2, 3);
        let _ = m.det();
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn get_out_of_bounds_panics() {
        let m = Matrix::zeros(2, 2);
        let _ = m.get(2, 0);
    }
}
