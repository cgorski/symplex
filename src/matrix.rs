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
    /// This is a recursive algorithm suitable for small matrices
    /// (roughly up to 8×8). For larger matrices, consider LU
    /// decomposition or other factorisation approaches.
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
        self.det_inner(&self.rows)
    }

    /// Recursive cofactor expansion helper.
    fn det_inner(&self, m: &[Vec<Ex>]) -> Ex {
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
            let cofactor = self.det_inner(&minor);
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
