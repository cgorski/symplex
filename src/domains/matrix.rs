//! Symbolic matrix type.
//!
//! Provides [`Matrix`], a dense matrix of symbolic expressions with
//! operations: construction, display, transpose, addition, scalar
//! multiplication, matrix multiplication, determinant, and trace.

use crate::base::errors::SymplexError;
use crate::api::expr::Ex;
use std::fmt;

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
    /// # Panics
    ///
    /// Panics if shapes differ.
    pub fn add_elementwise(&self, other: &Matrix) -> Matrix {
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
    pub fn sub_elementwise(&self, other: &Matrix) -> Matrix {
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

    /// Element-wise addition (convenience alias for operator `+`).
    ///
    /// Prefer using `m1 + m2` via the `Add` trait. This method is retained
    /// for backward-compatibility with code written before operator
    /// overloads were available.
    pub fn add(&self, other: &Matrix) -> Matrix {
        self.add_elementwise(other)
    }

    /// Element-wise subtraction (convenience alias for operator `-`).
    ///
    /// Prefer using `m1 - m2` via the `Sub` trait. This method is retained
    /// for backward-compatibility with code written before operator
    /// overloads were available.
    pub fn sub(&self, other: &Matrix) -> Matrix {
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

    /// Determinant of a square matrix.
    ///
    /// Dispatch strategy:
    /// - 0×0 → 1 (empty product)
    /// - 1×1 → element
    /// - 2×2 → ad − bc
    /// - 3×3 → cofactor expansion (hard-coded, fast)
    /// - n ≥ 4 → Bareiss fraction-free elimination (O(n³))
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
        let n = self.nrows;
        match n {
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
        }
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
            let one_over_det = &self.ctx_one() / &d;
            return Some(Matrix::new(vec![vec![one_over_det]]));
        }
        let adj = self.adjugate();
        let one_over_det = &self.ctx_one() / &d;
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
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// let var = symplex::var("λ");
    /// let m = symplex::matrix![[2, 1], [0, 3]];
    /// let evs = m.eigenvects(&var);
    /// for (val, mult, vecs) in &evs {
    ///     assert!(!vecs.is_empty());
    /// }
    /// ```
    pub fn eigenvects(&self, var: &Ex) -> Vec<(Ex, usize, Vec<Matrix>)> {
        assert!(
            self.is_square(),
            "eigenvects requires a square matrix, got {}×{}",
            self.nrows,
            self.ncols
        );
        let n = self.nrows;
        let eye = Matrix::identity(n);

        // Get eigenvalues (flat list, may contain duplicates).
        let all_roots = self.eigenvals(var);

        // Deduplicate and count algebraic multiplicities.
        let mut eigen_pairs: Vec<(Ex, usize)> = Vec::new();
        for root in &all_roots {
            if let Some(entry) = eigen_pairs.iter_mut().find(|(e, _)| e == root) {
                entry.1 += 1;
            } else {
                eigen_pairs.push((root.clone(), 1));
            }
        }

        // For each unique eigenvalue, compute eigenvectors via nullspace(A − λI).
        let mut result = Vec::new();
        for (eigenval, alg_mult) in eigen_pairs {
            let a_minus_lambda_i = self.sub(&eye.scale(&eigenval));
            let vecs = a_minus_lambda_i.nullspace();
            result.push((eigenval, alg_mult, vecs));
        }

        result
    }

    /// Check whether the matrix is diagonalizable.
    ///
    /// A matrix is diagonalizable iff for every eigenvalue the geometric
    /// multiplicity (dimension of eigenspace) equals the algebraic
    /// multiplicity.  Returns `false` if the eigenvalue solver cannot
    /// find all roots.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    pub fn is_diagonalizable(&self, var: &Ex) -> bool {
        assert!(
            self.is_square(),
            "is_diagonalizable requires a square matrix"
        );
        let eigvs = self.eigenvects(var);
        let mut total = 0usize;
        for (_, alg_mult, vecs) in &eigvs {
            if vecs.len() != *alg_mult {
                return false;
            }
            total += vecs.len();
        }
        total == self.nrows
    }

    /// Diagonalize: find invertible `P` and diagonal `D` such that
    /// `D = P⁻¹ A P`.
    ///
    /// `P` is the matrix whose columns are eigenvectors and `D` is
    /// the diagonal matrix of eigenvalues.
    ///
    /// Returns `None` if the matrix is not diagonalizable or the
    /// eigenvalue solver cannot find all roots.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// let var = symplex::var("λ");
    /// let m = symplex::matrix![[2, 1], [0, 3]];
    /// if let Some((p, d)) = m.diagonalize(&var) {
    ///     // D is diagonal, P columns are eigenvectors
    ///     assert_eq!(d.nrows(), 2);
    /// }
    /// ```
    pub fn diagonalize(&self, var: &Ex) -> Option<(Matrix, Matrix)> {
        assert!(
            self.is_square(),
            "diagonalize requires a square matrix"
        );
        let eigvs = self.eigenvects(var);

        // Verify diagonalizability: need n linearly independent eigenvectors.
        let n = self.nrows;
        let mut total_vecs = 0usize;
        for (_, alg_mult, vecs) in &eigvs {
            if vecs.len() != *alg_mult {
                return None;
            }
            total_vecs += vecs.len();
        }
        if total_vecs != n {
            return None;
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

        let p = Matrix::hstack(&p_cols);
        let d = Matrix::diag(&diag_entries);

        Some((p, d))
    }

    /// Compute the integer power of a square matrix via repeated squaring.
    ///
    /// - `powi(0)` returns the identity matrix
    /// - `powi(1)` returns a clone
    /// - `powi(n)` for n ≥ 2 uses binary exponentiation
    /// - Negative powers are not supported (use `inv()` + `powi()`)
    pub fn powi(&self, n: u32) -> Matrix {
        assert!(self.is_square(), "powi requires a square matrix");
        if n == 0 {
            return Matrix::identity(self.nrows);
        }
        if n == 1 {
            return self.clone();
        }
        // Binary exponentiation
        let mut result = Matrix::identity(self.nrows);
        let mut base = self.clone();
        let mut exp = n;
        while exp > 0 {
            if exp % 2 == 1 {
                result = result.matmul(&base);
            }
            base = base.matmul(&base);
            exp /= 2;
        }
        result
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
    /// This computes a symbolic approximation. For exact results on
    /// diagonalizable matrices, use eigendecomposition externally.
    ///
    /// `order` controls the number of terms (default: 10 is good for most cases).
    pub fn exp_series(&self, order: usize) -> Matrix {
        assert!(self.is_square(), "matrix exp requires square matrix");
        let n = self.nrows;
        let mut result = Matrix::identity(n);
        let mut a_power_over_factorial = Matrix::identity(n);
        for k in 1..=order {
            a_power_over_factorial = a_power_over_factorial.matmul(self);
            let inv_k = crate::rational(1, k as i64);
            a_power_over_factorial = a_power_over_factorial.scale(&inv_k);
            result = result.add(&a_power_over_factorial);
        }
        result
    }

    // ── Cholesky decomposition & pseudo-inverse ────────────────────────

    /// Cholesky decomposition for symmetric positive-definite matrices.
    ///
    /// Returns `L` such that `A = LLᵀ`, where `L` is lower triangular.
    /// Returns `None` if a diagonal element becomes non-positive during
    /// factorization (the matrix is not positive definite).
    pub fn cholesky(&self) -> Option<Matrix> {
        assert!(self.is_square(), "Cholesky requires a square matrix");
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
                    return None;
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

        Some(Matrix::new(l_rows))
    }

    /// Moore–Penrose pseudo-inverse via `A⁺ = (AᵀA)⁻¹Aᵀ`.
    ///
    /// This formula is valid for full-column-rank matrices.
    /// Returns `None` if `AᵀA` is singular.
    pub fn pinv(&self) -> Option<Matrix> {
        let at = self.transpose();
        let ata = at.matmul(self);
        let ata_inv = ata.inv()?;
        Some(ata_inv.matmul(&at))
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
// Operator overloads
// ═══════════════════════════════════════════════════════════════════════════

// Matrix + Matrix (element-wise addition)
impl std::ops::Add for &Matrix {
    type Output = Matrix;
    fn add(self, rhs: &Matrix) -> Matrix {
        self.add_elementwise(rhs)
    }
}
impl std::ops::Add for Matrix {
    type Output = Matrix;
    fn add(self, rhs: Matrix) -> Matrix {
        self.add_elementwise(&rhs)
    }
}
impl std::ops::Add<&Matrix> for Matrix {
    type Output = Matrix;
    fn add(self, rhs: &Matrix) -> Matrix {
        self.add_elementwise(rhs)
    }
}
impl std::ops::Add<Matrix> for &Matrix {
    type Output = Matrix;
    fn add(self, rhs: Matrix) -> Matrix {
        self.add_elementwise(&rhs)
    }
}

// Matrix - Matrix (element-wise subtraction)
impl std::ops::Sub for &Matrix {
    type Output = Matrix;
    fn sub(self, rhs: &Matrix) -> Matrix {
        self.sub_elementwise(rhs)
    }
}
impl std::ops::Sub for Matrix {
    type Output = Matrix;
    fn sub(self, rhs: Matrix) -> Matrix {
        self.sub_elementwise(&rhs)
    }
}
impl std::ops::Sub<&Matrix> for Matrix {
    type Output = Matrix;
    fn sub(self, rhs: &Matrix) -> Matrix {
        self.sub_elementwise(rhs)
    }
}
impl std::ops::Sub<Matrix> for &Matrix {
    type Output = Matrix;
    fn sub(self, rhs: Matrix) -> Matrix {
        self.sub_elementwise(&rhs)
    }
}

// Matrix * Matrix (matrix multiplication, NOT element-wise)
impl std::ops::Mul for &Matrix {
    type Output = Matrix;
    fn mul(self, rhs: &Matrix) -> Matrix {
        self.matmul(rhs)
    }
}
impl std::ops::Mul for Matrix {
    type Output = Matrix;
    fn mul(self, rhs: Matrix) -> Matrix {
        self.matmul(&rhs)
    }
}
impl std::ops::Mul<&Matrix> for Matrix {
    type Output = Matrix;
    fn mul(self, rhs: &Matrix) -> Matrix {
        self.matmul(rhs)
    }
}
impl std::ops::Mul<Matrix> for &Matrix {
    type Output = Matrix;
    fn mul(self, rhs: Matrix) -> Matrix {
        self.matmul(&rhs)
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
        let evs = m.eigenvects(&var);
        assert_eq!(evs.len(), 2, "should have 2 distinct eigenvalues");
        for (eigenval, mult, vecs) in &evs {
            assert_eq!(*mult, 1, "each mult should be 1");
            assert_eq!(vecs.len(), 1, "each eigenspace should be 1-dimensional");
            // Verify A·v = λ·v
            let av = m.matmul(&vecs[0]);
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
    fn eigenvects_diagonal_matrix() {
        let var = crate::var("lam_ev2");
        let m = Matrix::diag(&[crate::int(5), crate::int(-3)]);
        let evs = m.eigenvects(&var);
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
        let result = m.diagonalize(&var);
        assert!(result.is_some(), "upper triangular should be diagonalizable");
        let (p, d) = result.unwrap();
        assert_eq!(p.nrows(), 2);
        assert_eq!(d.nrows(), 2);
        // Verify: P * D * P^{-1} ≈ M
        if let Some(p_inv) = p.inv() {
            let reconstructed = p.matmul(&d).matmul(&p_inv);
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
            result.is_none(),
            "defective matrix should not be diagonalizable"
        );
    }

    #[test]
    fn is_diagonalizable_yes() {
        let var = crate::var("lam_diag_y");
        let m = Matrix::diag(&[crate::int(1), crate::int(2), crate::int(3)]);
        assert!(m.is_diagonalizable(&var));
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
