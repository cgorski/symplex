//! Exact dense matrices over ℤ ([`ZMatrix`]) and ℚ ([`QMatrix`]).
//!
//! [`Matrix`] stores expressions: every entry operation goes through the
//! expression arena (hash-consing, canonicalisation, a write lock).  That
//! is the right tool for symbolic matrices and the wrong one for exact
//! numeric linear algebra, where the entries are plain integers or
//! fractions and the work is dominated by big-number arithmetic.
//! `ExactMatrix<T>` is a plain row-major `Vec<T>`; `ZMatrix` and `QMatrix`
//! are its instances for [`BigInt`] and [`Ratio<BigInt>`](Ratio).
//!
//! Everything computed here is exact.  All rational eliminations are
//! **fraction-free**: a `QMatrix` is scaled row-wise to integers and
//! reduced with Bareiss's fraction-free Gauss–Jordan algorithm, whose
//! intermediate entries are minors of the input (so every division is
//! exact and no gcd normalisation happens inside the inner loop).  On a
//! 60×72 integer matrix this is about 40× faster than Gauss–Jordan over
//! `Ratio<BigInt>` and two orders of magnitude faster than the same
//! elimination over expressions.
//!
//! `Matrix` methods (`rref`, `rank`, `nullspace`, `det`, `inv`, `solve`,
//! `linsolve_matrix`, …) detect all-rational input and route through these
//! types automatically, so most users never construct one; use them
//! directly when the data is numeric from the start (LP formulations,
//! lattices, coefficient matrices) to skip the expression layer entirely.
//!
//! # Examples
//!
//! ```
//! use symplex::matrix::{QMatrix, ZMatrix};
//! use symplex::linprog::q;
//!
//! let a = QMatrix::from_i64(&[&[2, 1], &[1, 3]]).unwrap();
//! let b = QMatrix::from_i64(&[&[5], &[10]]).unwrap();
//! assert_eq!(a.solve(&b).unwrap(), QMatrix::from_i64(&[&[1], &[3]]).unwrap());
//! assert_eq!(a.det().unwrap(), q(5, 1));
//! assert_eq!(a.inv().unwrap()[(0, 1)], q(-1, 5));
//!
//! let z = ZMatrix::from_i64(&[&[2, 4, 4], &[-6, 6, 12], &[10, -4, -16]]).unwrap();
//! let (h, u) = z.hermite_normal_form_with_transform();
//! assert_eq!(h, ZMatrix::from_i64(&[&[2, 4, 4], &[0, 6, 0], &[0, 0, 12]]).unwrap());
//! assert_eq!(&u * &z, h);
//! let s = z.smith_normal_form();
//! assert_eq!(s.diagonal(), ZMatrix::from_i64(&[&[2, 6, 12]]).unwrap().row(0).to_vec());
//! ```

use std::fmt;
use std::hash::Hash;

use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::Ratio;
use num_traits::{One, Signed, Zero};

use crate::api::context::Context;
use crate::base::errors::SymplexError;
use crate::domains::linprog::Q;
use crate::domains::matrix::Matrix;
use crate::domains::ntheory::gcdex;

// ═══════════════════════════════════════════════════════════════════════════
// Scalar trait
// ═══════════════════════════════════════════════════════════════════════════

mod sealed {
    pub trait Sealed {}
    impl Sealed for num_bigint::BigInt {}
    impl Sealed for num_rational::Ratio<num_bigint::BigInt> {}
}

/// Entry type of an [`ExactMatrix`]: [`BigInt`] or [`Ratio<BigInt>`](Ratio).
///
/// Sealed — the two implementations are [`ZMatrix`] and [`QMatrix`].
pub trait ExactScalar:
    sealed::Sealed + Clone + Eq + Ord + Hash + fmt::Display + fmt::Debug + Zero + One + Signed
{
    /// Name used by `Debug` output (`ZMatrix` / `QMatrix`).
    fn matrix_name() -> &'static str;
    /// `self + other`, by reference.
    fn add_ref(&self, other: &Self) -> Self;
    /// `self − other`, by reference.
    fn sub_ref(&self, other: &Self) -> Self;
    /// `self · other`, by reference.
    fn mul_ref(&self, other: &Self) -> Self;
    /// The scalar `n`.
    fn from_i64(n: i64) -> Self;
}

impl ExactScalar for BigInt {
    fn matrix_name() -> &'static str {
        "ZMatrix"
    }
    #[inline]
    fn add_ref(&self, other: &Self) -> Self {
        self + other
    }
    #[inline]
    fn sub_ref(&self, other: &Self) -> Self {
        self - other
    }
    #[inline]
    fn mul_ref(&self, other: &Self) -> Self {
        self * other
    }
    fn from_i64(n: i64) -> Self {
        BigInt::from(n)
    }
}

impl ExactScalar for Q {
    fn matrix_name() -> &'static str {
        "QMatrix"
    }
    #[inline]
    fn add_ref(&self, other: &Self) -> Self {
        self + other
    }
    #[inline]
    fn sub_ref(&self, other: &Self) -> Self {
        self - other
    }
    #[inline]
    fn mul_ref(&self, other: &Self) -> Self {
        self * other
    }
    fn from_i64(n: i64) -> Self {
        Ratio::from_integer(BigInt::from(n))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Type
// ═══════════════════════════════════════════════════════════════════════════

/// Dense row-major matrix with exact numeric entries.
///
/// Use the aliases [`ZMatrix`] (`BigInt` entries) and [`QMatrix`]
/// (`Ratio<BigInt>` entries).  Shapes are always non-empty
/// (`nrows, ncols ≥ 1`), like [`Matrix`].
///
/// Element access follows `Matrix`: [`get`](Self::get) / indexing panic on
/// out-of-range indices (a logic error, like slice indexing);
/// [`try_get`](Self::try_get) returns `Option`.  Shape-checked arithmetic
/// ([`add`](Self::add), [`sub`](Self::sub), [`matmul`](Self::matmul)) returns
/// `Result`; the operators `+ − *` panic on a shape mismatch.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ExactMatrix<T: ExactScalar> {
    data: Vec<T>,
    nrows: usize,
    ncols: usize,
}

/// Dense matrix over ℤ with [`BigInt`] entries.
///
/// Integer-specific operations: Bareiss [`det`](ExactMatrix::det),
/// [`hermite_normal_form`](ExactMatrix::hermite_normal_form),
/// [`smith_normal_form`](ExactMatrix::smith_normal_form),
/// [`integer_nullspace`](ExactMatrix::integer_nullspace),
/// [`is_unimodular`](ExactMatrix::is_unimodular),
/// [`lattice_determinant`](ExactMatrix::lattice_determinant).
pub type ZMatrix = ExactMatrix<BigInt>;

/// Dense matrix over ℚ with [`Ratio<BigInt>`](Ratio) entries.
///
/// Rational-specific operations (all fraction-free internally):
/// [`rref`](ExactMatrix::rref), [`rank`](ExactMatrix::rank),
/// [`nullspace`](ExactMatrix::nullspace), [`det`](ExactMatrix::det),
/// [`inv`](ExactMatrix::inv), [`solve`](ExactMatrix::solve).
pub type QMatrix = ExactMatrix<Q>;

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
// Construction and access (generic)
// ═══════════════════════════════════════════════════════════════════════════

impl<T: ExactScalar> ExactMatrix<T> {
    /// Build from rows.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if there are no rows, a row is
    /// empty, or the rows have different lengths.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::matrix::ZMatrix;
    /// use symplex::num_bigint::BigInt;
    ///
    /// let m = ZMatrix::new(vec![vec![BigInt::from(1), BigInt::from(2)]]).unwrap();
    /// assert_eq!(m.shape(), (1, 2));
    /// assert!(ZMatrix::new(vec![vec![BigInt::from(1)], vec![]]).is_err());
    /// ```
    pub fn new(rows: Vec<Vec<T>>) -> Result<Self, SymplexError> {
        let nrows = rows.len();
        let ncols = rows.first().map_or(0, Vec::len);
        if nrows == 0 || ncols == 0 {
            return Err(invalid(
                "ExactMatrix::new",
                "matrix must have at least one row and one column",
            ));
        }
        if let Some((i, r)) = rows.iter().enumerate().find(|(_, r)| r.len() != ncols) {
            return Err(invalid(
                "ExactMatrix::new",
                format!("row {i} has {} entries, expected {ncols}", r.len()),
            ));
        }
        let data: Vec<T> = rows.into_iter().flatten().collect();
        Ok(Self { data, nrows, ncols })
    }

    /// Build from a flat row-major buffer of length `nrows · ncols`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if a dimension is zero or the
    /// buffer length does not match.
    pub fn from_flat(nrows: usize, ncols: usize, data: Vec<T>) -> Result<Self, SymplexError> {
        if nrows == 0 || ncols == 0 {
            return Err(invalid(
                "ExactMatrix::from_flat",
                "matrix must have at least one row and one column",
            ));
        }
        if data.len() != nrows * ncols {
            return Err(invalid(
                "ExactMatrix::from_flat",
                format!(
                    "buffer has {} entries, expected {nrows}×{ncols} = {}",
                    data.len(),
                    nrows * ncols
                ),
            ));
        }
        Ok(Self { data, nrows, ncols })
    }

    /// Build from machine-integer rows.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] for empty or jagged input.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::matrix::QMatrix;
    /// use symplex::linprog::q;
    ///
    /// let m = QMatrix::from_i64(&[&[1, 2], &[3, 4]]).unwrap();
    /// assert_eq!(m[(1, 0)], q(3, 1));
    /// ```
    pub fn from_i64(rows: &[&[i64]]) -> Result<Self, SymplexError> {
        Self::new(
            rows.iter()
                .map(|r| r.iter().map(|&v| T::from_i64(v)).collect())
                .collect(),
        )
    }

    /// Build an `nrows × ncols` matrix from `f(i, j)`.
    ///
    /// # Panics
    ///
    /// Panics if `nrows == 0` or `ncols == 0`.
    pub fn from_fn(nrows: usize, ncols: usize, mut f: impl FnMut(usize, usize) -> T) -> Self {
        assert!(
            nrows > 0 && ncols > 0,
            "ExactMatrix::from_fn: dimensions must be positive"
        );
        let mut data = Vec::with_capacity(nrows * ncols);
        for i in 0..nrows {
            for j in 0..ncols {
                data.push(f(i, j));
            }
        }
        Self { data, nrows, ncols }
    }

    /// The `nrows × ncols` zero matrix.
    ///
    /// # Panics
    ///
    /// Panics if `nrows == 0` or `ncols == 0`.
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        assert!(
            nrows > 0 && ncols > 0,
            "ExactMatrix::zeros: dimensions must be positive"
        );
        Self {
            data: vec![T::zero(); nrows * ncols],
            nrows,
            ncols,
        }
    }

    /// The `n × n` identity matrix.
    ///
    /// # Panics
    ///
    /// Panics if `n == 0`.
    pub fn identity(n: usize) -> Self {
        assert!(n > 0, "ExactMatrix::identity: dimension must be positive");
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.data[i * n + i] = T::one();
        }
        m
    }

    /// Square matrix with `entries` on the diagonal.
    ///
    /// # Panics
    ///
    /// Panics if `entries` is empty.
    pub fn diag(entries: &[T]) -> Self {
        assert!(
            !entries.is_empty(),
            "ExactMatrix::diag: need at least one entry"
        );
        let n = entries.len();
        let mut m = Self::zeros(n, n);
        for (i, e) in entries.iter().enumerate() {
            m.data[i * n + i] = e.clone();
        }
        m
    }

    /// A `1 × n` matrix.
    ///
    /// # Panics
    ///
    /// Panics if `entries` is empty.
    pub fn row_vector(entries: Vec<T>) -> Self {
        assert!(
            !entries.is_empty(),
            "ExactMatrix::row_vector: need at least one entry"
        );
        let n = entries.len();
        Self {
            data: entries,
            nrows: 1,
            ncols: n,
        }
    }

    /// An `n × 1` matrix.
    ///
    /// # Panics
    ///
    /// Panics if `entries` is empty.
    pub fn col_vector(entries: Vec<T>) -> Self {
        assert!(
            !entries.is_empty(),
            "ExactMatrix::col_vector: need at least one entry"
        );
        let n = entries.len();
        Self {
            data: entries,
            nrows: n,
            ncols: 1,
        }
    }

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

    /// `(nrows, ncols)`.
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    /// `true` if `nrows == ncols`.
    #[inline]
    pub fn is_square(&self) -> bool {
        self.nrows == self.ncols
    }

    /// Reference to entry `(i, j)`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= nrows` or `j >= ncols`; see [`try_get`](Self::try_get).
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> &T {
        assert!(
            i < self.nrows && j < self.ncols,
            "Index ({i}, {j}) out of bounds for {}×{} matrix",
            self.nrows,
            self.ncols
        );
        &self.data[i * self.ncols + j]
    }

    /// Checked reference to entry `(i, j)`.
    #[inline]
    pub fn try_get(&self, i: usize, j: usize) -> Option<&T> {
        if i < self.nrows && j < self.ncols {
            Some(&self.data[i * self.ncols + j])
        } else {
            None
        }
    }

    /// Mutable reference to entry `(i, j)`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= nrows` or `j >= ncols`.
    #[inline]
    pub fn get_mut(&mut self, i: usize, j: usize) -> &mut T {
        assert!(
            i < self.nrows && j < self.ncols,
            "Index ({i}, {j}) out of bounds for {}×{} matrix",
            self.nrows,
            self.ncols
        );
        &mut self.data[i * self.ncols + j]
    }

    /// Overwrite entry `(i, j)`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= nrows` or `j >= ncols`.
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, value: T) {
        *self.get_mut(i, j) = value;
    }

    /// Row `i` as a slice.
    ///
    /// # Panics
    ///
    /// Panics if `i >= nrows`.
    #[inline]
    pub fn row(&self, i: usize) -> &[T] {
        assert!(
            i < self.nrows,
            "Row {i} out of bounds for {} rows",
            self.nrows
        );
        &self.data[i * self.ncols..(i + 1) * self.ncols]
    }

    /// Column `j` as an owned vector.
    ///
    /// # Panics
    ///
    /// Panics if `j >= ncols`.
    pub fn col(&self, j: usize) -> Vec<T> {
        assert!(
            j < self.ncols,
            "Column {j} out of bounds for {} columns",
            self.ncols
        );
        (0..self.nrows)
            .map(|i| self.data[i * self.ncols + j].clone())
            .collect()
    }

    /// The diagonal entries `(0,0), (1,1), …` (length `min(nrows, ncols)`).
    pub fn diagonal(&self) -> Vec<T> {
        (0..self.nrows.min(self.ncols))
            .map(|i| self.data[i * self.ncols + i].clone())
            .collect()
    }

    /// Iterator over the rows as slices.
    pub fn rows(&self) -> impl Iterator<Item = &[T]> + '_ {
        self.data.chunks(self.ncols)
    }

    /// Iterator over all entries, row-major.
    pub fn iter(&self) -> impl Iterator<Item = &T> + '_ {
        self.data.iter()
    }

    /// The entries as a flat row-major slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// The entries as nested vectors.
    pub fn to_rows(&self) -> Vec<Vec<T>> {
        self.rows().map(<[T]>::to_vec).collect()
    }

    /// Consume into the flat row-major buffer.
    pub fn into_flat(self) -> Vec<T> {
        self.data
    }

    /// `true` if every entry is zero.
    pub fn is_zero(&self) -> bool {
        self.data.iter().all(Zero::is_zero)
    }

    /// `true` if this is a square identity matrix.
    pub fn is_identity(&self) -> bool {
        self.is_square()
            && self.data.iter().enumerate().all(|(k, v)| {
                if k / self.ncols == k % self.ncols {
                    v.is_one()
                } else {
                    v.is_zero()
                }
            })
    }

    /// Transpose.
    pub fn transpose(&self) -> Self {
        let mut data = Vec::with_capacity(self.data.len());
        for j in 0..self.ncols {
            for i in 0..self.nrows {
                data.push(self.data[i * self.ncols + j].clone());
            }
        }
        Self {
            data,
            nrows: self.ncols,
            ncols: self.nrows,
        }
    }

    /// The sub-block with rows in `rows` and columns in `cols`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if a range is empty or out of
    /// bounds.
    pub fn submatrix(
        &self,
        rows: std::ops::Range<usize>,
        cols: std::ops::Range<usize>,
    ) -> Result<Self, SymplexError> {
        if rows.is_empty() || cols.is_empty() || rows.end > self.nrows || cols.end > self.ncols {
            return Err(invalid(
                "submatrix",
                format!(
                    "ranges {rows:?} × {cols:?} are empty or exceed the {}×{} shape",
                    self.nrows, self.ncols
                ),
            ));
        }
        let (r0, c0) = (rows.start, cols.start);
        Ok(Self::from_fn(rows.len(), cols.len(), |i, j| {
            self.data[(r0 + i) * self.ncols + c0 + j].clone()
        }))
    }

    /// Horizontal concatenation `[A | B | …]`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if the list is empty or the row
    /// counts differ.
    pub fn hstack(parts: &[&Self]) -> Result<Self, SymplexError> {
        let Some(first) = parts.first() else {
            return Err(invalid("hstack", "need at least one matrix"));
        };
        let nrows = first.nrows;
        if let Some(p) = parts.iter().find(|p| p.nrows != nrows) {
            return Err(invalid(
                "hstack",
                format!("row count mismatch: {} vs {}", nrows, p.nrows),
            ));
        }
        let ncols: usize = parts.iter().map(|p| p.ncols).sum();
        let mut data = Vec::with_capacity(nrows * ncols);
        for i in 0..nrows {
            for p in parts {
                data.extend_from_slice(p.row(i));
            }
        }
        Ok(Self { data, nrows, ncols })
    }

    /// Vertical concatenation.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if the list is empty or the column
    /// counts differ.
    pub fn vstack(parts: &[&Self]) -> Result<Self, SymplexError> {
        let Some(first) = parts.first() else {
            return Err(invalid("vstack", "need at least one matrix"));
        };
        let ncols = first.ncols;
        if let Some(p) = parts.iter().find(|p| p.ncols != ncols) {
            return Err(invalid(
                "vstack",
                format!("column count mismatch: {} vs {}", ncols, p.ncols),
            ));
        }
        let nrows: usize = parts.iter().map(|p| p.nrows).sum();
        let mut data = Vec::with_capacity(nrows * ncols);
        for p in parts {
            data.extend_from_slice(&p.data);
        }
        Ok(Self { data, nrows, ncols })
    }

    /// Apply `f` to every entry.
    pub fn map<U: ExactScalar>(&self, f: impl FnMut(&T) -> U) -> ExactMatrix<U> {
        ExactMatrix {
            data: self.data.iter().map(f).collect(),
            nrows: self.nrows,
            ncols: self.ncols,
        }
    }

    fn require_same_shape(
        &self,
        other: &Self,
        operation: &'static str,
    ) -> Result<(), SymplexError> {
        if self.shape() != other.shape() {
            return Err(invalid(
                operation,
                format!(
                    "shape mismatch: {}×{} vs {}×{}",
                    self.nrows, self.ncols, other.nrows, other.ncols
                ),
            ));
        }
        Ok(())
    }

    fn require_square(&self, operation: &'static str) -> Result<(), SymplexError> {
        if !self.is_square() {
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

    /// Entry-wise sum.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] on a shape mismatch.
    pub fn add(&self, other: &Self) -> Result<Self, SymplexError> {
        self.require_same_shape(other, "add")?;
        Ok(Self {
            data: self
                .data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| a.add_ref(b))
                .collect(),
            nrows: self.nrows,
            ncols: self.ncols,
        })
    }

    /// Entry-wise difference.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] on a shape mismatch.
    pub fn sub(&self, other: &Self) -> Result<Self, SymplexError> {
        self.require_same_shape(other, "sub")?;
        Ok(Self {
            data: self
                .data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| a.sub_ref(b))
                .collect(),
            nrows: self.nrows,
            ncols: self.ncols,
        })
    }

    /// Entry-wise negation.
    pub fn neg(&self) -> Self {
        self.map(|v| -v.clone())
    }

    /// Multiply every entry by `k`.
    pub fn scale(&self, k: &T) -> Self {
        self.map(|v| v.mul_ref(k))
    }

    /// Matrix product.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `self.ncols != other.nrows`.
    pub fn matmul(&self, other: &Self) -> Result<Self, SymplexError> {
        if self.ncols != other.nrows {
            return Err(invalid(
                "matmul",
                format!(
                    "dimension mismatch: {}×{} times {}×{}",
                    self.nrows, self.ncols, other.nrows, other.ncols
                ),
            ));
        }
        let (n, k, m) = (self.nrows, self.ncols, other.ncols);
        let mut data = vec![T::zero(); n * m];
        for i in 0..n {
            for p in 0..k {
                let a = &self.data[i * k + p];
                if a.is_zero() {
                    continue;
                }
                let brow = &other.data[p * m..(p + 1) * m];
                let crow = &mut data[i * m..(i + 1) * m];
                for (c, b) in crow.iter_mut().zip(brow) {
                    if !b.is_zero() {
                        *c = c.add_ref(&a.mul_ref(b));
                    }
                }
            }
        }
        Ok(Self {
            data,
            nrows: n,
            ncols: m,
        })
    }

    /// Sum of the diagonal entries.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if the matrix is not square.
    pub fn trace(&self) -> Result<T, SymplexError> {
        self.require_square("trace")?;
        Ok(self
            .diagonal()
            .iter()
            .fold(T::zero(), |acc, v| acc.add_ref(v)))
    }

    /// Swap rows `a` and `b`.
    fn swap_rows(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        let n = self.ncols;
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        let (head, tail) = self.data.split_at_mut(hi * n);
        head[lo * n..(lo + 1) * n].swap_with_slice(&mut tail[..n]);
    }

    /// Disjoint mutable borrows of rows `r` and `i` (`r != i`), in that order.
    fn two_rows_mut(&mut self, r: usize, i: usize) -> (&mut [T], &mut [T]) {
        debug_assert!(r != i);
        let n = self.ncols;
        if r < i {
            let (lo, hi) = self.data.split_at_mut(i * n);
            (&mut lo[r * n..(r + 1) * n], &mut hi[..n])
        } else {
            let (lo, hi) = self.data.split_at_mut(r * n);
            (&mut hi[..n], &mut lo[i * n..(i + 1) * n])
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Fraction-free Gauss–Jordan kernel (shared by ℤ rank and every ℚ operation)
// ═══════════════════════════════════════════════════════════════════════════

/// Bareiss's fraction-free Gauss–Jordan elimination on a row-major integer
/// matrix.  Pivots are taken from columns `0..pivot_limit` only.
///
/// On return every row equals `d · (row of the rational reduced row-echelon
/// form)` for the returned `d` (the last pivot; it may be negative), so
/// the RREF entry `(i, j)` is `a[i][j] / d`.  Rows without a pivot are zero
/// in columns `< pivot_limit`; in the remaining columns they hold `d` times
/// the reduced right-hand side (the inconsistency residual of an augmented
/// system).  Every division is exact (Sylvester's identity: all
/// intermediate entries are minors of the input).
///
/// Returns the pivot columns (their count is the rank of the first
/// `pivot_limit` columns) and `d`.
fn fraction_free_gauss_jordan(
    a: &mut [BigInt],
    nrows: usize,
    ncols: usize,
    pivot_limit: usize,
) -> (Vec<usize>, BigInt) {
    let mut d = BigInt::one();
    let mut pivots: Vec<usize> = Vec::new();
    let mut pr = 0usize;
    for c in 0..pivot_limit.min(ncols) {
        if pr >= nrows {
            break;
        }
        let Some(p) = (pr..nrows).find(|&i| !a[i * ncols + c].is_zero()) else {
            continue;
        };
        if p != pr {
            let (head, tail) = a.split_at_mut(p * ncols);
            head[pr * ncols..(pr + 1) * ncols].swap_with_slice(&mut tail[..ncols]);
        }
        // Take the pivot row out so the other rows can be updated against it.
        let prow: Vec<BigInt> = a[pr * ncols..(pr + 1) * ncols]
            .iter_mut()
            .map(std::mem::take)
            .collect();
        let pv = prow[c].clone();
        for i in 0..nrows {
            if i == pr {
                continue;
            }
            let row = &mut a[i * ncols..(i + 1) * ncols];
            let f = std::mem::take(&mut row[c]);
            if f.is_zero() {
                // Row keeps its rational value: rescale d_old → d_new = pv.
                for v in row.iter_mut() {
                    if !v.is_zero() {
                        let t = &*v * &pv;
                        debug_assert!(
                            (&t % &d).is_zero(),
                            "fraction-free elimination: inexact division"
                        );
                        *v = t / &d;
                    }
                }
                continue;
            }
            for (j, (v, p)) in row.iter_mut().zip(prow.iter()).enumerate() {
                if j == c {
                    continue;
                }
                let t = &*v * &pv - &f * p;
                if t.is_zero() {
                    *v = t;
                } else {
                    debug_assert!(
                        (&t % &d).is_zero(),
                        "fraction-free elimination: inexact division"
                    );
                    *v = t / &d;
                }
            }
        }
        a[pr * ncols..(pr + 1) * ncols]
            .iter_mut()
            .zip(prow)
            .for_each(|(slot, v)| *slot = v);
        d = pv;
        pivots.push(c);
        pr += 1;
    }
    (pivots, d)
}

/// Bareiss fraction-free determinant of a square row-major integer matrix
/// (consumes the buffer).
fn bareiss_det(mut a: Vec<BigInt>, n: usize) -> BigInt {
    if n == 0 {
        return BigInt::one();
    }
    let mut sign = false;
    let mut prev = BigInt::one();
    for k in 0..n - 1 {
        if a[k * n + k].is_zero() {
            let Some(p) = ((k + 1)..n).find(|&i| !a[i * n + k].is_zero()) else {
                return BigInt::zero();
            };
            let (head, tail) = a.split_at_mut(p * n);
            head[k * n..(k + 1) * n].swap_with_slice(&mut tail[..n]);
            sign = !sign;
        }
        let pivot = a[k * n + k].clone();
        for i in (k + 1)..n {
            let aik = std::mem::take(&mut a[i * n + k]);
            for j in (k + 1)..n {
                let t = if aik.is_zero() {
                    &a[i * n + j] * &pivot
                } else {
                    &a[i * n + j] * &pivot - &aik * &a[k * n + j]
                };
                debug_assert!((&t % &prev).is_zero(), "Bareiss: inexact division");
                a[i * n + j] = if prev.is_one() { t } else { t / &prev };
            }
        }
        prev = pivot;
    }
    let d = std::mem::take(&mut a[n * n - 1]);
    if sign { -d } else { d }
}

// ═══════════════════════════════════════════════════════════════════════════
// ℤ-specific operations
// ═══════════════════════════════════════════════════════════════════════════

impl ZMatrix {
    /// Convert to a [`QMatrix`] (every entry becomes `n/1`).
    pub fn to_qmatrix(&self) -> QMatrix {
        self.map(|v| Ratio::from_integer(v.clone()))
    }

    /// Convert to a symbolic [`Matrix`] of integer literals in `ctx`.
    pub fn to_matrix(&self, ctx: &Context) -> Matrix {
        Matrix::from_rows_unchecked(
            self.rows()
                .map(|r| r.iter().map(|v| ctx.from_bigint(v.clone())).collect())
                .collect(),
        )
    }

    /// Determinant by Bareiss fraction-free elimination (`O(n³)` integer
    /// operations, no fractions at any stage).
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if the matrix is not square.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::matrix::ZMatrix;
    /// use symplex::num_bigint::BigInt;
    ///
    /// let m = ZMatrix::from_i64(&[&[2, 1, 0], &[1, 3, 1], &[0, 1, 4]]).unwrap();
    /// assert_eq!(m.det().unwrap(), BigInt::from(18));
    /// assert!(ZMatrix::from_i64(&[&[1, 2, 3]]).unwrap().det().is_err());
    /// ```
    pub fn det(&self) -> Result<BigInt, SymplexError> {
        self.require_square("det")?;
        Ok(bareiss_det(self.data.clone(), self.nrows))
    }

    /// Rank over ℚ (equivalently over ℤ as a lattice).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::matrix::ZMatrix;
    ///
    /// assert_eq!(ZMatrix::from_i64(&[&[1, 2], &[2, 4]]).unwrap().rank(), 1);
    /// ```
    pub fn rank(&self) -> usize {
        let mut a = self.data.clone();
        fraction_free_gauss_jordan(&mut a, self.nrows, self.ncols, self.ncols)
            .0
            .len()
    }

    /// Greatest common divisor of all entries (`0` for the zero matrix).
    pub fn content(&self) -> BigInt {
        self.data.iter().fold(BigInt::zero(), |g, v| g.gcd(v))
    }

    // ── Row / column operation helpers for the normal forms ────────────

    /// `row_i -= q · row_r`.
    fn sub_scaled_row(&mut self, i: usize, r: usize, q: &BigInt) {
        if r == i {
            return;
        }
        let (row_r, row_i) = self.two_rows_mut(r, i);
        for (ar, ai) in row_r.iter().zip(row_i.iter_mut()) {
            if !ar.is_zero() {
                *ai -= q * ar;
            }
        }
    }

    /// Replace rows `r` and `i` by `(x·row_r + y·row_i, p·row_r + q·row_i)`;
    /// the caller guarantees `x·q − y·p = ±1`.
    fn combine_rows(&mut self, r: usize, i: usize, x: &BigInt, y: &BigInt, p: &BigInt, q: &BigInt) {
        if r == i {
            return;
        }
        let (row_r, row_i) = self.two_rows_mut(r, i);
        for (ar, ai) in row_r.iter_mut().zip(row_i.iter_mut()) {
            let old_r = std::mem::take(ar);
            let old_i = std::mem::take(ai);
            *ar = x * &old_r + y * &old_i;
            *ai = p * &old_r + q * &old_i;
        }
    }

    fn negate_row(&mut self, r: usize) {
        let n = self.ncols;
        for v in &mut self.data[r * n..(r + 1) * n] {
            *v = -std::mem::take(v);
        }
    }

    fn swap_cols(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        let n = self.ncols;
        for i in 0..self.nrows {
            self.data.swap(i * n + a, i * n + b);
        }
    }

    /// `col_j -= q · col_c`.
    fn sub_scaled_col(&mut self, j: usize, c: usize, q: &BigInt) {
        let n = self.ncols;
        for i in 0..self.nrows {
            if !self.data[i * n + c].is_zero() {
                let t = q * &self.data[i * n + c];
                self.data[i * n + j] -= t;
            }
        }
    }

    /// Column analogue of [`combine_rows`](Self::combine_rows).
    fn combine_cols(&mut self, c: usize, j: usize, x: &BigInt, y: &BigInt, p: &BigInt, q: &BigInt) {
        let n = self.ncols;
        for i in 0..self.nrows {
            let ac = std::mem::take(&mut self.data[i * n + c]);
            let aj = std::mem::take(&mut self.data[i * n + j]);
            self.data[i * n + c] = x * &ac + y * &aj;
            self.data[i * n + j] = p * &ac + q * &aj;
        }
    }

    /// Zero `self[i][col]` against `self[r][col]` by a unimodular row
    /// combination, mirrored on `u`.
    fn eliminate_row_pair(&mut self, u: &mut ZMatrix, r: usize, i: usize, col: usize) {
        let n = self.ncols;
        if self.data[i * n + col].is_zero() {
            return;
        }
        if self.data[r * n + col].is_zero() {
            self.swap_rows(r, i);
            u.swap_rows(r, i);
            return;
        }
        let pivot = self.data[r * n + col].clone();
        let other = self.data[i * n + col].clone();
        let (quot, rem) = other.div_rem(&pivot);
        if rem.is_zero() {
            self.sub_scaled_row(i, r, &quot);
            u.sub_scaled_row(i, r, &quot);
            return;
        }
        let (g, x, y) = gcdex(pivot.clone(), other.clone());
        let p = -(&other / &g);
        let q = &pivot / &g;
        self.combine_rows(r, i, &x, &y, &p, &q);
        u.combine_rows(r, i, &x, &y, &p, &q);
    }

    /// Zero `self[t][j]` against `self[t][t]` by a unimodular column
    /// combination, mirrored on `v`.
    fn eliminate_col_pair(&mut self, v: &mut ZMatrix, t: usize, j: usize) {
        let n = self.ncols;
        if self.data[t * n + j].is_zero() {
            return;
        }
        let pivot = self.data[t * n + t].clone();
        let other = self.data[t * n + j].clone();
        if pivot.is_zero() {
            self.swap_cols(t, j);
            v.swap_cols(t, j);
            return;
        }
        let (quot, rem) = other.div_rem(&pivot);
        if rem.is_zero() {
            self.sub_scaled_col(j, t, &quot);
            v.sub_scaled_col(j, t, &quot);
            return;
        }
        let (g, x, y) = gcdex(pivot.clone(), other.clone());
        let p = -(&other / &g);
        let q = &pivot / &g;
        self.combine_cols(t, j, &x, &y, &p, &q);
        v.combine_cols(t, j, &x, &y, &p, &q);
    }

    /// Row-HNF core: `(H, U, pivots)` with `H = U·A`.
    fn row_hnf(&self) -> (ZMatrix, ZMatrix, Vec<usize>) {
        let (m, n) = (self.nrows, self.ncols);
        let mut h = self.clone();
        let mut u = ZMatrix::identity(m);
        let mut pivots = Vec::new();
        let mut r = 0usize;
        for col in 0..n {
            if r >= m {
                break;
            }
            // Smallest non-zero |entry| first keeps the quotients small.
            if let Some(best) = (r..m)
                .filter(|&i| !h.data[i * n + col].is_zero())
                .min_by(|&i, &j| h.data[i * n + col].abs().cmp(&h.data[j * n + col].abs()))
                && best != r
            {
                h.swap_rows(r, best);
                u.swap_rows(r, best);
            }
            for i in (r + 1)..m {
                h.eliminate_row_pair(&mut u, r, i, col);
            }
            if h.data[r * n + col].is_zero() {
                continue;
            }
            if h.data[r * n + col].is_negative() {
                h.negate_row(r);
                u.negate_row(r);
            }
            let pivot = h.data[r * n + col].clone();
            for i in 0..r {
                let q = h.data[i * n + col].div_floor(&pivot);
                if !q.is_zero() {
                    h.sub_scaled_row(i, r, &q);
                    u.sub_scaled_row(i, r, &q);
                }
            }
            pivots.push(col);
            r += 1;
        }
        (h, u, pivots)
    }

    /// Row-style Hermite normal form `H = U·A` (`U` unimodular).
    ///
    /// `H` is in row echelon form with positive pivots, every entry above a
    /// pivot reduced into `[0, pivot)`, zero rows at the bottom.  This form
    /// is unique.  See [`normalforms`](crate::normalforms) for the
    /// conventions in detail.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::matrix::ZMatrix;
    ///
    /// let a = ZMatrix::from_i64(&[&[1, 2], &[2, 4]]).unwrap();
    /// assert_eq!(a.hermite_normal_form(), ZMatrix::from_i64(&[&[1, 2], &[0, 0]]).unwrap());
    /// ```
    pub fn hermite_normal_form(&self) -> ZMatrix {
        self.row_hnf().0
    }

    /// Row-style Hermite normal form with its transform: `(H, U)`,
    /// `H = U·A`, `det U = ±1`.
    pub fn hermite_normal_form_with_transform(&self) -> (ZMatrix, ZMatrix) {
        let (h, u, _) = self.row_hnf();
        (h, u)
    }

    /// Column-style Hermite normal form `H = A·V` (Cohen's Algorithm 2.4.5,
    /// SymPy's convention): zero columns first, each pivot the lowest
    /// nonzero entry of its column, pivots positive and strictly descending
    /// from left to right, entries to the right of a pivot in `[0, pivot)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::matrix::ZMatrix;
    ///
    /// let a = ZMatrix::from_i64(&[&[12, 6, 4], &[3, 9, 6], &[2, 16, 14]]).unwrap();
    /// assert_eq!(
    ///     a.column_hermite_normal_form(),
    ///     ZMatrix::from_i64(&[&[10, 0, 2], &[0, 15, 3], &[0, 0, 2]]).unwrap()
    /// );
    /// ```
    pub fn column_hermite_normal_form(&self) -> ZMatrix {
        // Reverse rows, row-HNF the transpose, transpose back, reverse both.
        let m = self.nrows;
        let reversed = ZMatrix::from_fn(m, self.ncols, |i, j| {
            self.data[(m - 1 - i) * self.ncols + j].clone()
        });
        let (h, _, _) = reversed.transpose().row_hnf();
        let ht = h.transpose();
        let (r, c) = (ht.nrows, ht.ncols);
        ZMatrix::from_fn(r, c, |i, j| ht.data[(r - 1 - i) * c + (c - 1 - j)].clone())
    }

    /// Smith normal form core: `(S, U, V)` with `S = U·A·V`.
    fn smith(&self) -> (ZMatrix, ZMatrix, ZMatrix) {
        let (m, n) = (self.nrows, self.ncols);
        let mut s = self.clone();
        let mut u = ZMatrix::identity(m);
        let mut v = ZMatrix::identity(n);

        for t in 0..m.min(n) {
            // Move the smallest non-zero |entry| of the trailing block to (t, t).
            let mut best: Option<(usize, usize)> = None;
            for i in t..m {
                for j in t..n {
                    if s.data[i * n + j].is_zero() {
                        continue;
                    }
                    match best {
                        Some((bi, bj)) if s.data[bi * n + bj].abs() <= s.data[i * n + j].abs() => {}
                        _ => best = Some((i, j)),
                    }
                }
            }
            let Some((bi, bj)) = best else {
                break;
            };
            if bi != t {
                s.swap_rows(t, bi);
                u.swap_rows(t, bi);
            }
            if bj != t {
                s.swap_cols(t, bj);
                v.swap_cols(t, bj);
            }

            loop {
                for i in (t + 1)..m {
                    s.eliminate_row_pair(&mut u, t, i, t);
                }
                for j in (t + 1)..n {
                    s.eliminate_col_pair(&mut v, t, j);
                }
                let col_clear = ((t + 1)..m).all(|i| s.data[i * n + t].is_zero());
                let row_clear = ((t + 1)..n).all(|j| s.data[t * n + j].is_zero());
                if !(col_clear && row_clear) {
                    continue;
                }
                // Divisibility: every remaining entry must be a multiple of
                // the pivot.  If not, fold the offending row into row `t`;
                // the next sweep replaces the pivot by a proper divisor.
                let d = s.data[t * n + t].clone();
                let offender = ((t + 1)..m)
                    .find(|&i| ((t + 1)..n).any(|j| !(&s.data[i * n + j] % &d).is_zero()));
                match offender {
                    Some(i) => {
                        let one = BigInt::one();
                        let zero = BigInt::zero();
                        s.combine_rows(t, i, &one, &one, &zero, &one);
                        u.combine_rows(t, i, &one, &one, &zero, &one);
                    }
                    None => break,
                }
            }
            if s.data[t * n + t].is_negative() {
                s.negate_row(t);
                u.negate_row(t);
            }
        }
        (s, u, v)
    }

    /// Smith normal form `diag(d₁, …, dᵣ, 0, …)` with `dᵢ > 0`, `dᵢ | dᵢ₊₁`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::matrix::ZMatrix;
    ///
    /// let a = ZMatrix::from_i64(&[&[2, 0], &[0, 3]]).unwrap();
    /// assert_eq!(a.smith_normal_form(), ZMatrix::from_i64(&[&[1, 0], &[0, 6]]).unwrap());
    /// ```
    pub fn smith_normal_form(&self) -> ZMatrix {
        self.smith().0
    }

    /// Smith normal form with transforms `(S, U, V)`, `S = U·A·V`,
    /// `det U = det V = ±1`.
    pub fn smith_normal_form_with_transforms(&self) -> (ZMatrix, ZMatrix, ZMatrix) {
        self.smith()
    }

    /// A ℤ-basis of the integer kernel `{x ∈ ℤⁿ : A·x = 0}`, as `n × 1`
    /// column vectors (`n − rank` of them; empty for full column rank).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::matrix::ZMatrix;
    ///
    /// let a = ZMatrix::from_i64(&[&[2, 4, 6]]).unwrap();
    /// let basis = a.integer_nullspace();
    /// assert_eq!(basis.len(), 2);
    /// for k in &basis {
    ///     assert!((&a * k).is_zero());
    /// }
    /// ```
    pub fn integer_nullspace(&self) -> Vec<ZMatrix> {
        let (m, n) = (self.nrows, self.ncols);
        // [Aᵀ | I]  (n × (m + n)); kernel rows are those whose Aᵀ part vanishes.
        let at = self.transpose();
        let eye = ZMatrix::identity(n);
        let Ok(aug) = ZMatrix::hstack(&[&at, &eye]) else {
            return Vec::new();
        };
        let (h, _, pivots) = aug.row_hnf();
        let rank = pivots.iter().filter(|&&c| c < m).count();
        (rank..n)
            .map(|i| ZMatrix::col_vector(h.row(i)[m..].to_vec()))
            .collect()
    }

    /// Is this a square matrix with `det = ±1` (invertible over ℤ)?
    /// Non-square matrices give `false`.
    pub fn is_unimodular(&self) -> bool {
        if !self.is_square() {
            return false;
        }
        let (h, _, pivots) = self.row_hnf();
        pivots.len() == self.nrows && (0..self.nrows).all(|i| h.data[i * self.ncols + i].is_one())
    }

    /// Index `[ℤᵐ : A·ℤⁿ]` of the lattice spanned by the columns
    /// (`|det A|` for a square nonsingular matrix).
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `A` does not have full row rank
    /// (the index would be infinite).
    pub fn lattice_determinant(&self) -> Result<BigInt, SymplexError> {
        let (h, _, pivots) = self.transpose().row_hnf();
        if pivots.len() != self.nrows {
            return Err(invalid(
                "lattice_determinant",
                format!(
                    "matrix must have full row rank (rank {} of {} rows); the column \
                     lattice has infinite index otherwise",
                    pivots.len(),
                    self.nrows
                ),
            ));
        }
        Ok(pivots
            .iter()
            .enumerate()
            .map(|(i, &c)| h.data[i * h.ncols + c].clone())
            .product())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ℚ-specific operations
// ═══════════════════════════════════════════════════════════════════════════

impl QMatrix {
    /// Convert to a [`ZMatrix`] if every entry is an integer.
    pub fn to_zmatrix(&self) -> Option<ZMatrix> {
        if self.data.iter().all(Ratio::is_integer) {
            Some(self.map(|q| q.numer().clone()))
        } else {
            None
        }
    }

    /// `true` if every entry is an integer.
    pub fn is_integer(&self) -> bool {
        self.data.iter().all(Ratio::is_integer)
    }

    /// Convert to a symbolic [`Matrix`] of rational literals in `ctx`.
    pub fn to_matrix(&self, ctx: &Context) -> Matrix {
        Matrix::from_rows_unchecked(
            self.rows()
                .map(|r| r.iter().map(|v| ctx.from_ratio(v.clone())).collect())
                .collect(),
        )
    }

    /// Multiply every row by the least common multiple of its denominators.
    /// Returns the integer matrix and the per-row multipliers `sᵢ > 0`.
    fn integer_rows(&self) -> (Vec<BigInt>, Vec<BigInt>) {
        let n = self.ncols;
        let mut data = Vec::with_capacity(self.data.len());
        let mut scales = Vec::with_capacity(self.nrows);
        for i in 0..self.nrows {
            let row = &self.data[i * n..(i + 1) * n];
            let s = row.iter().fold(BigInt::one(), |l, q| l.lcm(q.denom()));
            for q in row {
                if s.is_one() {
                    data.push(q.numer().clone());
                } else {
                    data.push(q.numer() * (&s / q.denom()));
                }
            }
            scales.push(s);
        }
        (data, scales)
    }

    /// Clear denominators: `(Z, s)` with `Z = s · self` integral and `s`
    /// the least common multiple of all denominators.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::matrix::{QMatrix, ZMatrix};
    /// use symplex::linprog::q;
    /// use symplex::num_bigint::BigInt;
    ///
    /// let m = QMatrix::new(vec![vec![q(1, 2), q(1, 3)], vec![q(2, 1), q(-1, 6)]]).unwrap();
    /// let (z, s) = m.clear_denominators();
    /// assert_eq!(s, BigInt::from(6));
    /// assert_eq!(z, ZMatrix::from_i64(&[&[3, 2], &[12, -1]]).unwrap());
    /// ```
    pub fn clear_denominators(&self) -> (ZMatrix, BigInt) {
        let s = self
            .data
            .iter()
            .fold(BigInt::one(), |l, q| l.lcm(q.denom()));
        let z = self.map(|q| q.numer() * (&s / q.denom()));
        (z, s)
    }

    /// RREF restricted to pivots in columns `0..pivot_limit`.  Rows without
    /// a pivot are zero in those columns and carry the reduced residual in
    /// the rest (see [`fraction_free_gauss_jordan`]).
    pub(crate) fn rref_limited(&self, pivot_limit: usize) -> (QMatrix, Vec<usize>) {
        let (mut a, _) = self.integer_rows();
        let (pivots, d) = fraction_free_gauss_jordan(&mut a, self.nrows, self.ncols, pivot_limit);
        let data: Vec<Q> = a
            .into_iter()
            .map(|v| {
                if v.is_zero() {
                    Q::zero()
                } else {
                    Ratio::new(v, d.clone())
                }
            })
            .collect();
        (
            QMatrix {
                data,
                nrows: self.nrows,
                ncols: self.ncols,
            },
            pivots,
        )
    }

    /// Reduced row-echelon form and the pivot columns.
    ///
    /// Computed fraction-free (Bareiss Gauss–Jordan on the row-wise
    /// integerised matrix); the result is the unique RREF over ℚ.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::matrix::QMatrix;
    /// use symplex::linprog::q;
    ///
    /// let a = QMatrix::from_i64(&[&[1, 2, 3], &[4, 5, 6], &[7, 8, 9]]).unwrap();
    /// let (r, pivots) = a.rref();
    /// assert_eq!(pivots, vec![0, 1]);
    /// assert_eq!(r, QMatrix::from_i64(&[&[1, 0, -1], &[0, 1, 2], &[0, 0, 0]]).unwrap());
    /// assert_eq!(QMatrix::from_i64(&[&[2, 4], &[1, 3]]).unwrap().rref().0[(0, 0)], q(1, 1));
    /// ```
    pub fn rref(&self) -> (QMatrix, Vec<usize>) {
        self.rref_limited(self.ncols)
    }

    /// Rank.
    pub fn rank(&self) -> usize {
        let (mut a, _) = self.integer_rows();
        fraction_free_gauss_jordan(&mut a, self.nrows, self.ncols, self.ncols)
            .0
            .len()
    }

    /// Basis of `{x : A·x = 0}` as `n × 1` column vectors (empty for full
    /// column rank).  Each basis vector has a `1` in one free column and
    /// zeros in the other free columns (the standard RREF construction).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::matrix::QMatrix;
    ///
    /// let a = QMatrix::from_i64(&[&[1, 2, 3], &[4, 5, 6]]).unwrap();
    /// let ns = a.nullspace();
    /// assert_eq!(ns.len(), 1);
    /// assert!((&a * &ns[0]).is_zero());
    /// ```
    pub fn nullspace(&self) -> Vec<QMatrix> {
        let (r, pivots) = self.rref();
        let n = self.ncols;
        let mut is_pivot = vec![false; n];
        for &c in &pivots {
            is_pivot[c] = true;
        }
        (0..n)
            .filter(|&c| !is_pivot[c])
            .map(|free| {
                let mut v = vec![Q::zero(); n];
                v[free] = Q::one();
                for (k, &pc) in pivots.iter().enumerate() {
                    let e = &r.data[k * n + free];
                    if !e.is_zero() {
                        v[pc] = -e.clone();
                    }
                }
                QMatrix::col_vector(v)
            })
            .collect()
    }

    /// Determinant: the rows are scaled to integers, Bareiss's fraction-free
    /// elimination runs over ℤ, and the row scales are divided back out.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if the matrix is not square.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::matrix::QMatrix;
    /// use symplex::linprog::q;
    ///
    /// let m = QMatrix::new(vec![vec![q(1, 2), q(1, 3)], vec![q(1, 4), q(1, 5)]]).unwrap();
    /// assert_eq!(m.det().unwrap(), q(1, 60));
    /// ```
    pub fn det(&self) -> Result<Q, SymplexError> {
        self.require_square("det")?;
        let (a, scales) = self.integer_rows();
        let dz = bareiss_det(a, self.nrows);
        let denom: BigInt = scales.into_iter().product();
        Ok(Ratio::new(dz, denom))
    }

    /// Solve `A·X = B` for square nonsingular `A` (`B` may have several
    /// columns).
    ///
    /// # Errors
    ///
    /// - [`SymplexError::InvalidArgument`] if `A` is not square or the row
    ///   counts differ.
    /// - [`SymplexError::ComputationFailed`] if `A` is singular.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::matrix::QMatrix;
    /// use symplex::linprog::q;
    ///
    /// let a = QMatrix::from_i64(&[&[2, 1], &[1, 3]]).unwrap();
    /// let b = QMatrix::from_i64(&[&[1], &[1]]).unwrap();
    /// assert_eq!(a.solve(&b).unwrap().col(0), vec![q(2, 5), q(1, 5)]);
    /// assert!(QMatrix::from_i64(&[&[1, 2], &[2, 4]]).unwrap().solve(&b).is_err());
    /// ```
    pub fn solve(&self, b: &QMatrix) -> Result<QMatrix, SymplexError> {
        self.require_square("solve")?;
        let n = self.nrows;
        if b.nrows != n {
            return Err(invalid(
                "solve",
                format!("row count mismatch: A is {n}×{n}, b has {} rows", b.nrows),
            ));
        }
        let aug = QMatrix::hstack(&[self, b])?;
        let (r, pivots) = aug.rref_limited(n);
        if pivots.len() != n {
            return Err(failed(
                "solve",
                "matrix is singular; no unique solution exists",
            ));
        }
        r.submatrix(0..n, n..n + b.ncols)
    }

    /// Inverse.
    ///
    /// # Errors
    ///
    /// - [`SymplexError::InvalidArgument`] if not square.
    /// - [`SymplexError::ComputationFailed`] if singular.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::matrix::QMatrix;
    ///
    /// let a = QMatrix::from_i64(&[&[2, 1], &[1, 1]]).unwrap();
    /// let inv = a.inv().unwrap();
    /// assert!((&a * &inv).is_identity());
    /// ```
    pub fn inv(&self) -> Result<QMatrix, SymplexError> {
        self.require_square("inv")?;
        self.solve(&QMatrix::identity(self.nrows))
            .map_err(|e| match e {
                SymplexError::ComputationFailed { reason, .. } => failed("inv", reason),
                other => other,
            })
    }

    /// Basis of the column space: the pivot columns of `self`.
    pub fn columnspace(&self) -> Vec<QMatrix> {
        let (_, pivots) = self.rref();
        pivots
            .into_iter()
            .map(|c| QMatrix::col_vector(self.col(c)))
            .collect()
    }

    /// Basis of the row space: the nonzero rows of the RREF.
    pub fn rowspace(&self) -> Vec<QMatrix> {
        let (r, pivots) = self.rref();
        (0..pivots.len())
            .map(|i| QMatrix::row_vector(r.row(i).to_vec()))
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Conversions
// ═══════════════════════════════════════════════════════════════════════════

impl From<ZMatrix> for QMatrix {
    fn from(z: ZMatrix) -> Self {
        z.to_qmatrix()
    }
}

impl From<&ZMatrix> for QMatrix {
    fn from(z: &ZMatrix) -> Self {
        z.to_qmatrix()
    }
}

impl TryFrom<&Matrix> for QMatrix {
    type Error = SymplexError;

    /// Every entry must be a rational literal (after a constant-folding
    /// `eval` if the raw entries are not).
    fn try_from(m: &Matrix) -> Result<Self, SymplexError> {
        let rows = m
            .to_rational_rows()
            .or_else(|| m.eval().to_rational_rows())
            .ok_or_else(|| {
                invalid(
                    "QMatrix::try_from",
                    "every entry must be a rational literal (symbolic entries are not allowed)",
                )
            })?;
        QMatrix::new(rows)
    }
}

impl TryFrom<&Matrix> for ZMatrix {
    type Error = SymplexError;

    /// Every entry must be an integer literal (after a constant-folding
    /// `eval` if the raw entries are not).
    fn try_from(m: &Matrix) -> Result<Self, SymplexError> {
        let rows = m
            .to_bigint_rows()
            .or_else(|| m.eval().to_bigint_rows())
            .ok_or_else(|| {
                invalid(
                    "ZMatrix::try_from",
                    "every entry must be an integer literal (fractions and symbolic \
                     entries are not allowed)",
                )
            })?;
        ZMatrix::new(rows)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Display / Debug / indexing / operators
// ═══════════════════════════════════════════════════════════════════════════

impl<T: ExactScalar> fmt::Display for ExactMatrix<T> {
    /// Same layout as [`Matrix`]: single rows inline as `[[a, b]]`, larger
    /// matrices one row per line with right-aligned columns.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.nrows == 1 {
            write!(f, "[[")?;
            for (j, v) in self.data.iter().enumerate() {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{v}")?;
            }
            return write!(f, "]]");
        }
        let cells: Vec<String> = self.data.iter().map(ToString::to_string).collect();
        let widths: Vec<usize> = (0..self.ncols)
            .map(|j| {
                (0..self.nrows)
                    .map(|i| cells[i * self.ncols + j].chars().count())
                    .max()
                    .unwrap_or(0)
            })
            .collect();
        writeln!(f, "[")?;
        for i in 0..self.nrows {
            write!(f, "  [")?;
            for j in 0..self.ncols {
                if j > 0 {
                    write!(f, ", ")?;
                }
                let cell = &cells[i * self.ncols + j];
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

impl<T: ExactScalar> fmt::Debug for ExactMatrix<T> {
    /// `QMatrix(2×2, [[1/2, 3], [0, 1]])`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({}×{}, [", T::matrix_name(), self.nrows, self.ncols)?;
        for (i, row) in self.rows().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "[")?;
            for (j, v) in row.iter().enumerate() {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{v}")?;
            }
            write!(f, "]")?;
        }
        write!(f, "])")
    }
}

impl<T: ExactScalar> std::ops::Index<(usize, usize)> for ExactMatrix<T> {
    type Output = T;
    #[inline]
    fn index(&self, (i, j): (usize, usize)) -> &T {
        self.get(i, j)
    }
}

impl<T: ExactScalar> std::ops::IndexMut<(usize, usize)> for ExactMatrix<T> {
    #[inline]
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut T {
        self.get_mut(i, j)
    }
}

macro_rules! exact_binop {
    ($trait:ident, $method:ident, $inner:ident, $msg:literal) => {
        impl<T: ExactScalar> std::ops::$trait<&ExactMatrix<T>> for &ExactMatrix<T> {
            type Output = ExactMatrix<T>;
            fn $method(self, rhs: &ExactMatrix<T>) -> ExactMatrix<T> {
                match self.$inner(rhs) {
                    Ok(m) => m,
                    Err(e) => panic!(concat!($msg, ": {}"), e),
                }
            }
        }
        impl<T: ExactScalar> std::ops::$trait<ExactMatrix<T>> for ExactMatrix<T> {
            type Output = ExactMatrix<T>;
            fn $method(self, rhs: ExactMatrix<T>) -> ExactMatrix<T> {
                std::ops::$trait::$method(&self, &rhs)
            }
        }
        impl<T: ExactScalar> std::ops::$trait<&ExactMatrix<T>> for ExactMatrix<T> {
            type Output = ExactMatrix<T>;
            fn $method(self, rhs: &ExactMatrix<T>) -> ExactMatrix<T> {
                std::ops::$trait::$method(&self, rhs)
            }
        }
        impl<T: ExactScalar> std::ops::$trait<ExactMatrix<T>> for &ExactMatrix<T> {
            type Output = ExactMatrix<T>;
            fn $method(self, rhs: ExactMatrix<T>) -> ExactMatrix<T> {
                std::ops::$trait::$method(self, &rhs)
            }
        }
    };
}

exact_binop!(Add, add, add, "ExactMatrix + ExactMatrix");
exact_binop!(Sub, sub, sub, "ExactMatrix - ExactMatrix");
exact_binop!(Mul, mul, matmul, "ExactMatrix * ExactMatrix");

impl<T: ExactScalar> std::ops::Neg for &ExactMatrix<T> {
    type Output = ExactMatrix<T>;
    fn neg(self) -> ExactMatrix<T> {
        ExactMatrix::neg(self)
    }
}

impl<T: ExactScalar> std::ops::Neg for ExactMatrix<T> {
    type Output = ExactMatrix<T>;
    fn neg(self) -> ExactMatrix<T> {
        ExactMatrix::neg(&self)
    }
}

impl<T: ExactScalar> std::ops::Mul<&T> for &ExactMatrix<T> {
    type Output = ExactMatrix<T>;
    fn mul(self, k: &T) -> ExactMatrix<T> {
        self.scale(k)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn z(rows: &[&[i64]]) -> ZMatrix {
        ZMatrix::from_i64(rows).unwrap()
    }

    fn qm(rows: &[&[i64]]) -> QMatrix {
        QMatrix::from_i64(rows).unwrap()
    }

    fn q(n: i64, d: i64) -> Q {
        Ratio::new(BigInt::from(n), BigInt::from(d))
    }

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> i64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) % 19) as i64 - 9
        }
    }

    fn random_z(n: usize, m: usize, seed: u64) -> ZMatrix {
        let mut g = Lcg(seed);
        ZMatrix::from_fn(n, m, |_, _| BigInt::from(g.next()))
    }

    /// Reference RREF over `Ratio<BigInt>` (plain Gauss–Jordan).
    fn naive_rref(a: &QMatrix) -> (QMatrix, Vec<usize>) {
        let mut rows = a.to_rows();
        let (n, m) = a.shape();
        let mut pivots = Vec::new();
        let mut pr = 0;
        for c in 0..m {
            if pr >= n {
                break;
            }
            let Some(p) = (pr..n).find(|&i| !rows[i][c].is_zero()) else {
                continue;
            };
            rows.swap(pr, p);
            let pv = rows[pr][c].clone();
            for v in rows[pr].iter_mut() {
                *v = &*v / &pv;
            }
            let prow = rows[pr].clone();
            for (i, row) in rows.iter_mut().enumerate() {
                if i == pr || row[c].is_zero() {
                    continue;
                }
                let f = row[c].clone();
                for (v, p) in row.iter_mut().zip(&prow) {
                    *v = &*v - &f * p;
                }
            }
            pivots.push(c);
            pr += 1;
        }
        (QMatrix::new(rows).unwrap(), pivots)
    }

    #[test]
    fn construction_and_access() {
        let m = z(&[&[1, 2, 3], &[4, 5, 6]]);
        assert_eq!(m.shape(), (2, 3));
        assert_eq!(m[(1, 2)], BigInt::from(6));
        assert_eq!(
            m.row(1),
            &[BigInt::from(4), BigInt::from(5), BigInt::from(6)]
        );
        assert_eq!(m.col(0), vec![BigInt::from(1), BigInt::from(4)]);
        assert!(m.try_get(2, 0).is_none());
        assert_eq!(m.transpose().shape(), (3, 2));
        assert_eq!(m.transpose().transpose(), m);
        assert!(ZMatrix::new(vec![]).is_err());
        assert!(ZMatrix::from_flat(2, 2, vec![BigInt::zero(); 3]).is_err());
        assert!(ZMatrix::identity(3).is_identity());
        assert!(!z(&[&[1, 1], &[0, 1]]).is_identity());
        assert_eq!(
            format!("{:?}", z(&[&[1, 2], &[3, 4]])),
            "ZMatrix(2×2, [[1, 2], [3, 4]])"
        );
        assert_eq!(format!("{}", qm(&[&[1, 2]])), "[[1, 2]]");
        assert_eq!(
            format!("{}", z(&[&[1, -20], &[3, 4]])),
            "[\n  [1, -20],\n  [3,   4]\n]"
        );
    }

    #[test]
    fn arithmetic() {
        let a = z(&[&[1, 2], &[3, 4]]);
        let b = z(&[&[0, 1], &[1, 0]]);
        assert_eq!(&a * &b, z(&[&[2, 1], &[4, 3]]));
        assert_eq!(&a + &b, z(&[&[1, 3], &[4, 4]]));
        assert_eq!(&a - &a, ZMatrix::zeros(2, 2));
        assert_eq!(-&a, z(&[&[-1, -2], &[-3, -4]]));
        assert_eq!(a.scale(&BigInt::from(2)), z(&[&[2, 4], &[6, 8]]));
        assert_eq!(a.trace().unwrap(), BigInt::from(5));
        assert!(a.matmul(&z(&[&[1, 2, 3]])).is_err());
        assert!(a.add(&z(&[&[1, 2, 3]])).is_err());
        let h = ZMatrix::hstack(&[&a, &b]).unwrap();
        assert_eq!(h.shape(), (2, 4));
        assert_eq!(h.submatrix(0..2, 2..4).unwrap(), b);
        let v = ZMatrix::vstack(&[&a, &b]).unwrap();
        assert_eq!(v.shape(), (4, 2));
        assert_eq!(v.submatrix(2..4, 0..2).unwrap(), b);
    }

    #[test]
    fn fraction_free_rref_matches_naive_on_random_matrices() {
        for seed in 1..=12u64 {
            for &(n, m) in &[(3usize, 3usize), (4, 6), (6, 4), (7, 9), (5, 5)] {
                let a = random_z(n, m, seed * 31 + (n * m) as u64).to_qmatrix();
                let (r1, p1) = a.rref();
                let (r2, p2) = naive_rref(&a);
                assert_eq!(p1, p2, "pivots differ for seed {seed} {n}x{m}");
                assert_eq!(r1, r2, "rref differs for seed {seed} {n}x{m}");
                assert_eq!(a.rank(), p1.len());
            }
        }
    }

    #[test]
    fn rref_handles_rank_deficiency_and_fractions() {
        let a = QMatrix::new(vec![
            vec![q(1, 2), q(1, 3), q(1, 6)],
            vec![q(1, 1), q(2, 3), q(1, 3)],
            vec![q(0, 1), q(1, 1), q(1, 1)],
        ])
        .unwrap();
        let (r, p) = a.rref();
        assert_eq!(p, vec![0, 1]);
        let (r2, _) = naive_rref(&a);
        assert_eq!(r, r2);
        assert!(r.row(2).iter().all(Zero::is_zero));
        let zero = QMatrix::zeros(2, 3);
        assert_eq!(zero.rref().1, Vec::<usize>::new());
        assert_eq!(zero.rank(), 0);
        assert_eq!(zero.nullspace().len(), 3);
    }

    #[test]
    fn nullspace_vectors_are_in_the_kernel() {
        for seed in 1..=6u64 {
            let a = random_z(4, 7, seed).to_qmatrix();
            let ns = a.nullspace();
            assert_eq!(ns.len(), 7 - a.rank());
            for v in &ns {
                assert!((&a * v).is_zero());
            }
        }
        assert!(qm(&[&[1, 0], &[0, 1]]).nullspace().is_empty());
    }

    #[test]
    fn det_and_inverse() {
        let a = qm(&[&[2, 1, 0], &[1, 3, 1], &[0, 1, 4]]);
        assert_eq!(a.det().unwrap(), q(18, 1));
        assert_eq!(a.to_zmatrix().unwrap().det().unwrap(), BigInt::from(18));
        let inv = a.inv().unwrap();
        assert!((&a * &inv).is_identity());
        assert!((&inv * &a).is_identity());
        let singular = qm(&[&[1, 2], &[2, 4]]);
        assert_eq!(singular.det().unwrap(), Q::zero());
        assert!(singular.inv().is_err());
        assert!(qm(&[&[1, 2, 3]]).det().is_err());
        // Fractions: det of [[1/2, 1/3], [1/4, 1/5]] = 1/10 - 1/12 = 1/60.
        let f = QMatrix::new(vec![vec![q(1, 2), q(1, 3)], vec![q(1, 4), q(1, 5)]]).unwrap();
        assert_eq!(f.det().unwrap(), q(1, 60));
        assert!((&f * &f.inv().unwrap()).is_identity());
        // Random: det(A) · det(A⁻¹) = 1 and det(A·B) = det(A)·det(B).
        for seed in 1..=5u64 {
            let a = random_z(5, 5, seed).to_qmatrix();
            let b = random_z(5, 5, seed + 100).to_qmatrix();
            let (da, db) = (a.det().unwrap(), b.det().unwrap());
            assert_eq!((&a * &b).det().unwrap(), &da * &db);
            if !da.is_zero() {
                assert_eq!(&da * &a.inv().unwrap().det().unwrap(), Q::one());
            }
        }
    }

    #[test]
    fn det_zero_pivot_column_needs_row_swap() {
        let a = z(&[&[0, 1], &[1, 0]]);
        assert_eq!(a.det().unwrap(), BigInt::from(-1));
        let b = z(&[&[0, 0, 1], &[0, 1, 0], &[1, 0, 0]]);
        assert_eq!(b.det().unwrap(), BigInt::from(-1));
        let c = z(&[&[0, 2, 3], &[0, 0, 5], &[0, 1, 1]]);
        assert_eq!(c.det().unwrap(), BigInt::zero());
        assert_eq!(z(&[&[7]]).det().unwrap(), BigInt::from(7));
        // Zero in the pivot column of a row that is not the pivot row.
        let d = z(&[&[2, 3, 1], &[0, 5, 7], &[4, 1, 9]]);
        assert_eq!(
            d.det().unwrap(),
            BigInt::from(2 * (45 - 7) - 3 * (0 - 28) + (0 - 20))
        );
    }

    #[test]
    fn solve_multiple_rhs() {
        let a = qm(&[&[2, 1], &[1, 3]]);
        let b = qm(&[&[5, 1], &[10, 1]]);
        let x = a.solve(&b).unwrap();
        assert_eq!(&a * &x, b);
        assert_eq!(x.col(0), vec![q(1, 1), q(3, 1)]);
        assert!(a.solve(&qm(&[&[1, 2, 3]])).is_err());
    }

    #[test]
    fn hnf_and_snf_match_normalforms_conventions() {
        let a = z(&[&[2, 4, 4], &[-6, 6, 12], &[10, -4, -16]]);
        let (h, u) = a.hermite_normal_form_with_transform();
        assert_eq!(h, z(&[&[2, 4, 4], &[0, 6, 0], &[0, 0, 12]]));
        assert_eq!(&u * &a, h);
        assert_eq!(u.det().unwrap().abs(), BigInt::one());
        assert_eq!(
            a.smith_normal_form(),
            z(&[&[2, 0, 0], &[0, 6, 0], &[0, 0, 12]])
        );
        let (s, u, v) = a.smith_normal_form_with_transforms();
        assert_eq!(&(&u * &a) * &v, s);
        assert!(u.is_unimodular() && v.is_unimodular());

        let b = z(&[&[12, 6, 4], &[3, 9, 6], &[2, 16, 14]]);
        assert_eq!(
            b.smith_normal_form(),
            z(&[&[1, 0, 0], &[0, 10, 0], &[0, 0, 30]])
        );
        assert_eq!(
            b.column_hermite_normal_form(),
            z(&[&[10, 0, 2], &[0, 15, 3], &[0, 0, 2]])
        );

        let c = z(&[&[1, 2, 3], &[-2, -4, -6], &[0, 1, 1]]);
        let (h, _, p) = c.row_hnf();
        assert_eq!(p, vec![0, 1]);
        assert!(h.row(2).iter().all(Zero::is_zero));
        assert_eq!(
            z(&[&[2, 0], &[0, 3]]).smith_normal_form(),
            z(&[&[1, 0], &[0, 6]])
        );
    }

    #[test]
    fn hnf_is_idempotent_and_unimodular_invariant() {
        for seed in 1..=6u64 {
            let a = random_z(4, 5, seed);
            let h = a.hermite_normal_form();
            assert_eq!(h.hermite_normal_form(), h);
            let (_, u) = random_z(4, 4, seed + 7).hermite_normal_form_with_transform();
            assert!(u.is_unimodular());
            assert_eq!((&u * &a).hermite_normal_form(), h);
        }
    }

    #[test]
    fn integer_kernel_unimodular_lattice() {
        let a = z(&[&[2, 4, 6]]);
        let ns = a.integer_nullspace();
        assert_eq!(ns.len(), 2);
        for k in &ns {
            assert!((&a * k).is_zero());
        }
        assert!(
            z(&[&[1, 0], &[0, 1], &[1, 1]])
                .integer_nullspace()
                .is_empty()
        );
        assert!(z(&[&[2, 1], &[1, 1]]).is_unimodular());
        assert!(!z(&[&[2, 0], &[0, 1]]).is_unimodular());
        assert!(!z(&[&[1, 2, 3]]).is_unimodular());
        assert_eq!(
            z(&[&[2, 0], &[0, 3]]).lattice_determinant().unwrap(),
            BigInt::from(6)
        );
        assert_eq!(
            z(&[&[2, 0, 1], &[0, 3, 1]]).lattice_determinant().unwrap(),
            BigInt::one()
        );
        assert!(z(&[&[1, 2], &[2, 4]]).lattice_determinant().is_err());
        assert_eq!(z(&[&[4, 6], &[8, 10]]).content(), BigInt::from(2));
    }

    #[test]
    fn conversions() {
        let ctx = Context::new();
        let m = crate::matrix![ctx, [1, 2], [3, 4]];
        let zm = ZMatrix::try_from(&m).unwrap();
        assert_eq!(zm, z(&[&[1, 2], &[3, 4]]));
        assert_eq!(zm.to_matrix(&ctx), m);
        let qmx = QMatrix::try_from(&m).unwrap();
        assert_eq!(qmx, zm.to_qmatrix());
        assert_eq!(qmx.to_matrix(&ctx), m);
        assert_eq!(QMatrix::from(&zm), qmx);
        let half = Matrix::new(vec![vec![ctx.rational(1, 2)]]).unwrap();
        assert!(ZMatrix::try_from(&half).is_err());
        assert_eq!(QMatrix::try_from(&half).unwrap()[(0, 0)], q(1, 2));
        let sym = Matrix::new(vec![vec![ctx.symbol("x")]]).unwrap();
        assert!(QMatrix::try_from(&sym).is_err());
        // Unevaluated constant arithmetic is folded before conversion.
        let folded = Matrix::new(vec![vec![ctx.rational(1, 3) + ctx.rational(1, 6)]]).unwrap();
        assert_eq!(QMatrix::try_from(&folded).unwrap()[(0, 0)], q(1, 2));
        assert!(qm(&[&[1, 2]]).to_zmatrix().is_some());
        assert!(
            QMatrix::new(vec![vec![q(1, 2)]])
                .unwrap()
                .to_zmatrix()
                .is_none()
        );
        let (zc, s) = QMatrix::new(vec![vec![q(1, 2), q(1, 3)], vec![q(2, 1), q(-1, 6)]])
            .unwrap()
            .clear_denominators();
        assert_eq!(s, BigInt::from(6));
        assert_eq!(zc, z(&[&[3, 2], &[12, -1]]));
    }
}
