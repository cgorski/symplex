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
//! let hnf = z.hermite_normal_form_with_transform();
//! assert_eq!(hnf.h, ZMatrix::from_i64(&[&[2, 4, 4], &[0, 6, 0], &[0, 0, 12]]).unwrap());
//! assert_eq!(&hnf.u * &z, hnf.h);
//! let s = z.smith_normal_form();
//! assert_eq!(s.diagonal(), ZMatrix::from_i64(&[&[2, 6, 12]]).unwrap().row(0).to_vec());
//! ```

use std::fmt;
use std::hash::Hash;

use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::{Ratio, Rational64};
use num_traits::{One, Signed, Zero};

use crate::api::context::Context;
use crate::base::errors::SymplexError;
use crate::base::numeric::Q;
use crate::domains::decompositions::{
    HermiteNormalForm, Hessenberg, LllReduction, Lu, RankDecomposition, SmithNormalForm,
};
use crate::domains::exact_kernel::{
    KernelError, LuParts, scaled_rref, try_char_poly, try_det, try_lu, try_scaled_rref,
};
use crate::domains::matrix::Matrix;
use crate::domains::ntheory::{gcdex, mod_inverse};
// Named through the module on purpose: importing `Ring` itself would make
// every `BigInt::zero()` / `x.is_zero()` on the concrete scalars below
// ambiguous with `num_traits::{Zero, One}`.
use crate::poly::traits as coeff;

// ═══════════════════════════════════════════════════════════════════════════
// Scalar trait
// ═══════════════════════════════════════════════════════════════════════════

mod sealed {
    pub trait Sealed {}
    impl Sealed for num_bigint::BigInt {}
    impl Sealed for crate::base::numeric::Q {}
}

/// Entry type of an [`ExactMatrix`]: [`BigInt`] or [`Q`].
///
/// Sealed — the two implementations are [`ZMatrix`] and [`QMatrix`].  The
/// arithmetic is the polynomial coefficient hierarchy's:
/// [`Ring`](crate::factor_zassenhaus::traits::Ring) supplies
/// `add`/`sub`/`mul`/`neg` by reference and `zero`/`one`,
/// [`IntegralCoeff`](crate::factor_zassenhaus::traits::IntegralCoeff)
/// supplies `from_i64`; `Signed` adds `abs`/`is_negative`.  (`Zero`/`One`
/// arrive through `Signed`, so generic code names `zero`/`is_zero` through
/// one trait explicitly.)
pub trait ExactScalar:
    sealed::Sealed
    + crate::poly::traits::Ring
    + crate::poly::traits::IntegralCoeff
    + Ord
    + Hash
    + fmt::Display
    + Signed
{
    /// Name used by `Debug` output (`ZMatrix` / `QMatrix`).
    fn matrix_name() -> &'static str;
    /// The product of `a` (`n × k`) and `b` (`k × m`); the shapes are
    /// checked by [`ExactMatrix::matmul`].  `Q` clears denominators and
    /// multiplies over ℤ (one normalised fraction per result entry instead
    /// of one per operation).
    #[doc(hidden)]
    fn matmul(a: &ExactMatrix<Self>, b: &ExactMatrix<Self>) -> ExactMatrix<Self> {
        schoolbook_matmul(a, b)
    }
}

/// `a · b` by the schoolbook loop, skipping zero entries.
fn schoolbook_matmul<T: ExactScalar>(a: &ExactMatrix<T>, b: &ExactMatrix<T>) -> ExactMatrix<T> {
    let (n, k, m) = (a.nrows, a.ncols, b.ncols);
    let mut data = vec![<T as Zero>::zero(); n * m];
    for i in 0..n {
        for p in 0..k {
            let x = &a.data[i * k + p];
            if Zero::is_zero(x) {
                continue;
            }
            let brow = &b.data[p * m..(p + 1) * m];
            let crow = &mut data[i * m..(i + 1) * m];
            for (c, y) in crow.iter_mut().zip(brow) {
                if !Zero::is_zero(y) {
                    *c = coeff::Ring::add(c, &coeff::Ring::mul(x, y));
                }
            }
        }
    }
    ExactMatrix {
        data,
        nrows: n,
        ncols: m,
    }
}

impl ExactScalar for BigInt {
    fn matrix_name() -> &'static str {
        "ZMatrix"
    }
}

impl ExactScalar for Q {
    fn matrix_name() -> &'static str {
        "QMatrix"
    }
    fn matmul(a: &QMatrix, b: &QMatrix) -> QMatrix {
        // (Dₐ·A)(B·D_b) = Dₐ·(AB)·D_b with Dₐ the row scales of A and D_b
        // the column scales of B, so cᵢⱼ = zᵢⱼ / (sₐᵢ · s_bⱼ).
        let (za, sa) = a.integer_rows();
        let (zb, sb) = b.integer_cols();
        let zc = schoolbook_matmul(
            &ZMatrix {
                data: za,
                nrows: a.nrows,
                ncols: a.ncols,
            },
            &ZMatrix {
                data: zb,
                nrows: b.nrows,
                ncols: b.ncols,
            },
        );
        let m = b.ncols;
        let data = zc
            .data
            .into_iter()
            .enumerate()
            .map(|(idx, z)| {
                let d = &sa[idx / m] * &sb[idx % m];
                if d.is_one() {
                    Ratio::from_integer(z)
                } else {
                    Ratio::new(z, d)
                }
            })
            .collect();
        QMatrix {
            data,
            nrows: a.nrows,
            ncols: m,
        }
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
    SymplexError::invalid_argument(operation, reason)
}

fn failed(operation: &'static str, reason: impl Into<String>) -> SymplexError {
    SymplexError::computation_failed(operation, reason)
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
            data: vec![<T as Zero>::zero(); nrows * ncols],
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
            m.data[i * n + i] = <T as One>::one();
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
                    One::is_one(v)
                } else {
                    Zero::is_zero(v)
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
                .map(|(a, b)| coeff::Ring::add(a, b))
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
                .map(|(a, b)| coeff::Ring::sub(a, b))
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
        self.map(|v| coeff::Ring::mul(v, k))
    }

    /// Matrix product.  A `QMatrix` product is formed over ℤ (row scales
    /// of `self`, column scales of `other`) and normalised once per entry.
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
        Ok(T::matmul(self, other))
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
            .fold(<T as Zero>::zero(), |acc, v| coeff::Ring::add(&acc, v)))
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
// Permanent (0.9)
// ═══════════════════════════════════════════════════════════════════════════

/// Largest dimension accepted by [`ExactMatrix::permanent`] and
/// [`Matrix::permanent`]: Ryser's formula sums `2ⁿ` terms, so a 20×20
/// permanent is about a million products and anything larger would run for
/// minutes to years.
pub(crate) const PERMANENT_MAX_DIM: usize = 20;

impl<T: ExactScalar> ExactMatrix<T> {
    /// Permanent `per(A) = Σ_σ ∏ᵢ a_{i,σ(i)}` (the determinant without the
    /// signs), by Ryser's inclusion–exclusion formula walked in Gray-code
    /// order: `O(2ⁿ · n)` ring operations and `O(n)` memory.  SymPy:
    /// `Matrix.per()`.
    ///
    /// The cost is exponential in `n`; matrices larger than 20×20 are
    /// rejected rather than left to run for an unbounded time.
    ///
    /// # Errors
    ///
    /// - [`SymplexError::InvalidArgument`] if the matrix is not square.
    /// - [`SymplexError::ComputationFailed`] if `n > 20`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::matrix::ZMatrix;
    /// use symplex::num_bigint::BigInt;
    ///
    /// let m = ZMatrix::from_i64(&[&[1, 2], &[3, 4]]).unwrap();
    /// assert_eq!(m.permanent().unwrap(), BigInt::from(10)); // 1·4 + 2·3
    /// let m3 = ZMatrix::from_i64(&[&[1, 2, 3], &[4, 5, 6], &[7, 8, 9]]).unwrap();
    /// assert_eq!(m3.permanent().unwrap(), BigInt::from(450));
    /// ```
    pub fn permanent(&self) -> Result<T, SymplexError> {
        self.require_square("permanent")?;
        let n = self.nrows;
        if n > PERMANENT_MAX_DIM {
            return Err(failed(
                "permanent",
                format!(
                    "Ryser's formula needs 2^{n} terms for a {n}×{n} matrix; the limit is \
                     {PERMANENT_MAX_DIM}×{PERMANENT_MAX_DIM}"
                ),
            ));
        }
        Ok(ryser_permanent(&self.data, n))
    }
}

/// Ryser's formula `per(A) = (−1)ⁿ Σ_{S⊆[n]} (−1)^{|S|} ∏ᵢ Σ_{j∈S} a_ij`
/// over the non-empty subsets in Gray-code order, so that each step
/// updates the `n` row sums by a single addition or subtraction.
/// `data` is a row-major `n × n` buffer with `1 ≤ n ≤ 63`.
fn ryser_permanent<T: ExactScalar>(data: &[T], n: usize) -> T {
    let mut row_sums = vec![<T as Zero>::zero(); n];
    let mut total = <T as Zero>::zero();
    let mut size = 0usize; // |S|
    for k in 1u64..(1u64 << n) {
        // gray(k) and gray(k − 1) differ exactly in bit `trailing_zeros(k)`.
        let bit = k.trailing_zeros() as usize;
        let added = ((k ^ (k >> 1)) >> bit) & 1 == 1;
        for (i, sum) in row_sums.iter_mut().enumerate() {
            let a = &data[i * n + bit];
            if Zero::is_zero(a) {
                continue;
            }
            *sum = if added {
                coeff::Ring::add(sum, a)
            } else {
                coeff::Ring::sub(sum, a)
            };
        }
        if added {
            size += 1;
        } else {
            size -= 1;
        }
        if row_sums.iter().any(Zero::is_zero) {
            continue;
        }
        let prod = row_sums
            .iter()
            .fold(<T as One>::one(), |acc, r| coeff::Ring::mul(&acc, r));
        total = if size % 2 == 1 {
            coeff::Ring::sub(&total, &prod)
        } else {
            coeff::Ring::add(&total, &prod)
        };
    }
    if n % 2 == 1 { -total } else { total }
}

// ═══════════════════════════════════════════════════════════════════════════
// Fraction-free kernels (shared with the polytopes and the simplex tableau)
// ═══════════════════════════════════════════════════════════════════════════
//
// Every ℤ rank and every ℚ elimination below runs on the `Cell`-generic
// kernel in `exact_kernel`: Bareiss's fraction-free Gauss–Jordan
// (`scaled_rref` / `try_scaled_rref`) and the Bareiss determinant
// (`try_det`), both starting on `i64` cells and widening to `i128`,
// 256-bit and finally `BigInt` cells as the minors grow, with every exact
// division verified in release builds.  The fallible operations map a
// verification failure to `ComputationFailed`; the infallible ones
// (`rank`, `rref`) fall back to plain Gauss–Jordan over ℚ inside the kernel.

/// `ComputationFailed` for a kernel that found a fraction-free division
/// inexact (a violated internal invariant).
fn kernel_failed(operation: &'static str, err: KernelError) -> SymplexError {
    failed(
        operation,
        match err {
            KernelError::Inexact => {
                "internal invariant violated: fraction-free elimination met an inexact division"
            }
            KernelError::Overflow => "internal invariant violated: fixed-width cells overflowed",
        },
    )
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
        try_det(self.data.clone(), self.nrows).map_err(|e| kernel_failed("det", e))
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
        scaled_rref(&mut a, self.nrows, self.ncols, self.ncols)
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
        let num_integer::ExtendedGcd { gcd: g, x, y } = gcdex(pivot.clone(), other.clone());
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
        let num_integer::ExtendedGcd { gcd: g, x, y } = gcdex(pivot.clone(), other.clone());
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

    /// Row-style Hermite normal form with its transform:
    /// [`HermiteNormalForm`]`{ h, u }` with `H = U·A`, `det U = ±1`.
    pub fn hermite_normal_form_with_transform(&self) -> HermiteNormalForm<ZMatrix> {
        let (h, u, _) = self.row_hnf();
        HermiteNormalForm { h, u }
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

    /// Smith normal form with transforms [`SmithNormalForm`]`{ s, u, v }`,
    /// `S = U·A·V`, `det U = det V = ±1`.
    pub fn smith_normal_form_with_transforms(&self) -> SmithNormalForm<ZMatrix> {
        let (s, u, v) = self.smith();
        SmithNormalForm { s, u, v }
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
// ℤ-specific operations (0.9): modular inverse, LLL lattice reduction
// ═══════════════════════════════════════════════════════════════════════════

/// Lovász parameter of [`ZMatrix::lll_default`]: `δ = 3/4`.
pub const LLL_DEFAULT_DELTA: Rational64 = Ratio::new_raw(3, 4);

impl ZMatrix {
    /// Inverse modulo `m`: the integer matrix `B` with entries in `[0, m)`
    /// and `A·B ≡ I (mod m)`, computed as `adj(A) · det(A)⁻¹ mod m`.  SymPy:
    /// `Matrix.inv_mod(m)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if the matrix is not square, `m < 2`,
    /// or `gcd(det A, m) ≠ 1` (which includes every singular matrix).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::matrix::ZMatrix;
    /// use symplex::num_bigint::BigInt;
    ///
    /// let a = ZMatrix::from_i64(&[&[1, 2], &[3, 4]]).unwrap();
    /// let b = a.inv_mod(&BigInt::from(5)).unwrap();
    /// assert_eq!(b, ZMatrix::from_i64(&[&[3, 1], &[4, 2]]).unwrap());
    /// // det = −2 shares the factor 2 with the modulus 4.
    /// assert!(a.inv_mod(&BigInt::from(4)).is_err());
    /// ```
    pub fn inv_mod(&self, m: &BigInt) -> Result<ZMatrix, SymplexError> {
        self.require_square("inv_mod")?;
        if *m < BigInt::from(2) {
            return Err(invalid(
                "inv_mod",
                format!("modulus must be at least 2, got {m}"),
            ));
        }
        let det =
            try_det(self.data.clone(), self.nrows).map_err(|e| kernel_failed("inv_mod", e))?;
        let Some(det_inv) = mod_inverse(det.mod_floor(m), m.clone()) else {
            return Err(invalid(
                "inv_mod",
                format!("determinant {det} is not invertible modulo {m} (gcd ≠ 1)"),
            ));
        };
        // det ≠ 0 here (gcd(0, m) = m ≥ 2), so the rational inverse exists and
        // adj(A) = det(A) · A⁻¹ is integral.
        let inv = self.to_qmatrix().inv().map_err(|e| match e {
            SymplexError::ComputationFailed { reason, .. } => failed("inv_mod", reason),
            other => other,
        })?;
        let mut data = Vec::with_capacity(self.data.len());
        for q in inv.iter() {
            let adj = q * &det;
            if !adj.is_integer() {
                return Err(failed(
                    "inv_mod",
                    "internal invariant violated: adjugate entry is not an integer",
                ));
            }
            data.push((adj.numer() * &det_inv).mod_floor(m));
        }
        Ok(ZMatrix {
            data,
            nrows: self.nrows,
            ncols: self.ncols,
        })
    }

    /// LLL-reduced basis of the lattice spanned by the rows, with the
    /// standard Lovász parameter `δ = 3/4`.  See [`lll`](Self::lll).
    ///
    /// # Errors
    ///
    /// Same as [`lll`](Self::lll).
    pub fn lll_default(&self) -> Result<ZMatrix, SymplexError> {
        self.lll(LLL_DEFAULT_DELTA)
    }

    /// Lenstra–Lenstra–Lovász reduction of the lattice basis formed by the
    /// **rows**, with Lovász parameter `δ` (a [`Rational64`]).  SymPy:
    /// `Matrix.lll(delta)`.
    ///
    /// The Gram–Schmidt data is kept as exact rationals, so the result is
    /// exactly LLL-reduced: with `μ_ij = ⟨b_i, b*_j⟩ / ⟨b*_j, b*_j⟩`,
    /// every `|μ_ij| ≤ 1/2` (size condition) and
    /// `‖b*_k‖² ≥ (δ − μ²_{k,k−1}) ‖b*_{k−1}‖²` (Lovász condition).  The
    /// reduced rows span the same lattice as the input (they differ by a
    /// unimodular transform, see
    /// [`lll_with_transform`](Self::lll_with_transform)), and the first row
    /// is within a factor `2^{(n−1)/2}` of a shortest lattice vector.  The
    /// reduction order and rounding follow SymPy's `DomainMatrix.lll`, so
    /// the output coincides with SymPy's for the same input.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `δ` is not in the open interval
    /// `(1/4, 1)`, or the rows are linearly dependent (this includes having
    /// more rows than columns).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::matrix::ZMatrix;
    /// use symplex::num_rational::Ratio;
    ///
    /// let b = ZMatrix::from_i64(&[&[1, 1, 1], &[-1, 0, 2], &[3, 5, 6]]).unwrap();
    /// let r = b.lll(Ratio::new(3, 4)).unwrap();
    /// assert_eq!(r, ZMatrix::from_i64(&[&[0, 1, 0], &[1, 0, 1], &[-1, 0, 2]]).unwrap());
    /// // Same lattice: identical Hermite normal forms.
    /// assert_eq!(r.hermite_normal_form(), b.hermite_normal_form());
    /// assert!(b.lll(Ratio::new(1, 4)).is_err());
    /// ```
    pub fn lll(&self, delta: Rational64) -> Result<ZMatrix, SymplexError> {
        self.lll_impl(delta, false).map(|(y, _)| y)
    }

    /// LLL reduction together with the unimodular transform:
    /// [`LllReduction`]`{ reduced, transform }` with `reduced = T·A`
    /// (`det T = ±1`).  SymPy: `Matrix.lll_transform(delta)`.
    ///
    /// # Errors
    ///
    /// Same as [`lll`](Self::lll).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::matrix::ZMatrix;
    /// use symplex::num_rational::Ratio;
    ///
    /// let b = ZMatrix::from_i64(&[&[1, 1, 1], &[-1, 0, 2], &[3, 5, 6]]).unwrap();
    /// let lll = b.lll_with_transform(Ratio::new(3, 4)).unwrap();
    /// assert_eq!(&lll.transform * &b, lll.reduced);
    /// assert!(lll.transform.is_unimodular());
    /// ```
    pub fn lll_with_transform(
        &self,
        delta: Rational64,
    ) -> Result<LllReduction<ZMatrix>, SymplexError> {
        let (reduced, t) = self.lll_impl(delta, true)?;
        let transform = t.ok_or_else(|| {
            failed(
                "lll_with_transform",
                "internal invariant violated: transform was not tracked",
            )
        })?;
        Ok(LllReduction { reduced, transform })
    }

    /// LLL core (SymPy's `_ddm_lll`): exact rational Gram–Schmidt with the
    /// incremental `μ` / `‖b*‖²` updates on a swap.
    fn lll_impl(
        &self,
        delta: Rational64,
        track: bool,
    ) -> Result<(ZMatrix, Option<ZMatrix>), SymplexError> {
        let (num, den) = (*delta.numer(), *delta.denom());
        if den == 0 {
            return Err(invalid("lll", "delta denominator must be non-zero"));
        }
        let delta = Ratio::new(BigInt::from(num), BigInt::from(den));
        let quarter = Ratio::new(BigInt::from(1), BigInt::from(4));
        if delta <= quarter || delta >= Q::one() {
            return Err(invalid(
                "lll",
                format!("delta must lie in the open interval (1/4, 1), got {delta}"),
            ));
        }
        let (m, n) = (self.nrows, self.ncols);
        let dependent = || invalid("lll", "rows must be linearly independent (a lattice basis)");
        if m > n {
            return Err(dependent());
        }

        let mut y: Vec<Vec<BigInt>> = self.to_rows();
        let mut t: Option<Vec<Vec<BigInt>>> = track.then(|| {
            (0..m)
                .map(|i| {
                    (0..m)
                        .map(|j| {
                            if i == j {
                                BigInt::one()
                            } else {
                                BigInt::zero()
                            }
                        })
                        .collect()
                })
                .collect()
        });

        // Gram–Schmidt: b*_i = b_i − Σ_{j<i} μ_ij b*_j,  g_star[i] = ‖b*_i‖².
        let mut mu: Vec<Vec<Q>> = vec![vec![Q::zero(); m]; m];
        let mut g_star: Vec<Q> = vec![Q::zero(); m];
        let mut y_star: Vec<Vec<Q>> = Vec::with_capacity(m);
        for i in 0..m {
            let mut yi: Vec<Q> = y[i]
                .iter()
                .map(|v| Ratio::from_integer(v.clone()))
                .collect();
            for j in 0..i {
                if g_star[j].is_zero() {
                    return Err(dependent());
                }
                let dot = y[i]
                    .iter()
                    .zip(&y_star[j])
                    .fold(Q::zero(), |acc, (a, b)| acc + b * a);
                let mu_ij = dot / &g_star[j];
                for (v, s) in yi.iter_mut().zip(&y_star[j]) {
                    *v -= &mu_ij * s;
                }
                mu[i][j] = mu_ij;
            }
            g_star[i] = yi.iter().fold(Q::zero(), |acc, v| acc + v * v);
            if g_star[i].is_zero() {
                return Err(dependent());
            }
            y_star.push(yi);
        }

        let half = Ratio::new(BigInt::from(1), BigInt::from(2));
        // Size-reduce row k against row l: b_k ← b_k − ⌊μ_kl + ½⌋ b_l.
        let reduce_row = |y: &mut [Vec<BigInt>],
                          mu: &mut [Vec<Q>],
                          t: &mut Option<Vec<Vec<BigInt>>>,
                          k: usize,
                          l: usize| {
            let r = (&mu[k][l] + &half).floor().to_integer();
            if r.is_zero() {
                return;
            }
            let yl = y[l].clone();
            for (a, b) in y[k].iter_mut().zip(&yl) {
                *a -= &r * b;
            }
            let mul: Vec<Q> = mu[l][..l].to_vec();
            let rq = Ratio::from_integer(r.clone());
            for (a, b) in mu[k].iter_mut().zip(&mul) {
                *a -= &rq * b;
            }
            mu[k][l] -= &rq;
            if let Some(t) = t {
                let tl = t[l].clone();
                for (a, b) in t[k].iter_mut().zip(&tl) {
                    *a -= &r * b;
                }
            }
        };

        let mut k = 1usize;
        while k < m {
            if mu[k][k - 1].abs() > half {
                reduce_row(&mut y, &mut mu, &mut t, k, k - 1);
            }
            let lovasz = g_star[k] >= (&delta - &mu[k][k - 1] * &mu[k][k - 1]) * &g_star[k - 1];
            if lovasz {
                for l in (0..k.saturating_sub(1)).rev() {
                    if mu[k][l].abs() > half {
                        reduce_row(&mut y, &mut mu, &mut t, k, l);
                    }
                }
                k += 1;
            } else {
                let nu = mu[k][k - 1].clone();
                let alpha = &g_star[k] + &nu * &nu * &g_star[k - 1];
                if alpha.is_zero() {
                    return Err(dependent());
                }
                let beta = &g_star[k - 1] / &alpha;
                mu[k][k - 1] = &nu * &beta;
                g_star[k] = &g_star[k] * &beta;
                g_star[k - 1] = alpha;
                y.swap(k, k - 1);
                if let Some(t) = &mut t {
                    t.swap(k, k - 1);
                }
                let (lo, hi) = mu.split_at_mut(k);
                lo[k - 1][..k - 1].swap_with_slice(&mut hi[0][..k - 1]);
                let mu_kk1 = mu[k][k - 1].clone();
                for row in mu.iter_mut().skip(k + 1) {
                    let xi = row[k].clone();
                    row[k] = &row[k - 1] - &nu * &xi;
                    row[k - 1] = &mu_kk1 * &row[k] + &xi;
                }
                k = (k - 1).max(1);
            }
        }

        let reduced = ZMatrix::new(y)?;
        let transform = match t {
            Some(rows) => Some(ZMatrix::new(rows)?),
            None => None,
        };
        Ok((reduced, transform))
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

    /// Multiply every column by the least common multiple of its
    /// denominators.  Returns the integer data and the per-column
    /// multipliers `sⱼ > 0`.
    fn integer_cols(&self) -> (Vec<BigInt>, Vec<BigInt>) {
        let (n, m) = (self.nrows, self.ncols);
        let scales: Vec<BigInt> = (0..m)
            .map(|j| (0..n).fold(BigInt::one(), |l, i| l.lcm(self.data[i * m + j].denom())))
            .collect();
        let data = self
            .data
            .iter()
            .enumerate()
            .map(|(idx, q)| {
                let s = &scales[idx % m];
                if s.is_one() {
                    q.numer().clone()
                } else {
                    q.numer() * (s / q.denom())
                }
            })
            .collect();
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
    /// the rest (see [`Elimination`](crate::domains::exact_kernel::Elimination)).
    pub(crate) fn rref_limited(&self, pivot_limit: usize) -> (QMatrix, Vec<usize>) {
        let (mut a, _) = self.integer_rows();
        let (pivots, d) = scaled_rref(&mut a, self.nrows, self.ncols, pivot_limit);
        (self.with_scaled(a, &d), pivots)
    }

    /// [`rref_limited`](Self::rref_limited) that reports a violated
    /// exactness invariant instead of falling back — for the fallible
    /// operations (`solve`, `inv`).
    fn try_rref_limited(
        &self,
        operation: &'static str,
        pivot_limit: usize,
    ) -> Result<(QMatrix, Vec<usize>), SymplexError> {
        let (mut a, _) = self.integer_rows();
        let (pivots, d) = try_scaled_rref(&mut a, self.nrows, self.ncols, pivot_limit)
            .map_err(|e| kernel_failed(operation, e))?;
        Ok((self.with_scaled(a, &d), pivots))
    }

    /// The matrix of the same shape whose entries are `a[k] / d`.
    fn with_scaled(&self, a: Vec<BigInt>, d: &BigInt) -> QMatrix {
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
        QMatrix {
            data,
            nrows: self.nrows,
            ncols: self.ncols,
        }
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
        scaled_rref(&mut a, self.nrows, self.ncols, self.ncols)
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
        let dz = try_det(a, self.nrows).map_err(|e| kernel_failed("det", e))?;
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
        let (r, pivots) = aug.try_rref_limited("solve", n)?;
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

    /// Berkowitz's division-free characteristic polynomial: the
    /// coefficients of `det(λI − A)`, **highest degree first**
    /// (`[1, c_{n−1}, …, c_0]`, length `n + 1`).  Requires a square matrix
    /// (checked by the callers).
    ///
    /// The denominators are cleared once (`Z = s·A`), the integer
    /// characteristic polynomial runs on the escalating fraction-free
    /// kernel, and `det(λI − sA) = sⁿ·det((λ/s)I − A)` divides the scale
    /// back out: the coefficient of `λ^{n−i}` is the integer one over `sⁱ`.
    pub(crate) fn berkowitz_monic(&self) -> Result<Vec<Q>, SymplexError> {
        let n = self.nrows;
        let (z, s) = self.clear_denominators();
        let coeffs =
            try_char_poly(z.into_flat(), n).map_err(|e| kernel_failed("char_poly_coeffs", e))?;
        if s.is_one() {
            return Ok(coeffs.into_iter().map(Ratio::from_integer).collect());
        }
        let mut pow = BigInt::one();
        Ok(coeffs
            .into_iter()
            .enumerate()
            .map(|(i, c)| {
                if i > 0 {
                    pow *= &s;
                }
                Ratio::new(c, pow.clone())
            })
            .collect())
    }

    /// Coefficients of the characteristic polynomial `det(A − λI)` in
    /// **ascending** degree order: `[c_0, c_1, …, c_n]` with `c_0 = det(A)`
    /// and `c_n = (−1)ⁿ` — the convention of
    /// [`Matrix::char_poly_coeffs`], which routes here for rational
    /// matrices.  Berkowitz's division-free algorithm (`O(n⁴)` ring
    /// operations, no gcd normalisation inside the loops).
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if the matrix is not square.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::matrix::QMatrix;
    /// use symplex::linprog::{q, qi};
    ///
    /// // det(A − λI) = λ² − 5λ − 2
    /// let a = QMatrix::from_i64(&[&[1, 2], &[3, 4]]).unwrap();
    /// assert_eq!(a.char_poly_coeffs().unwrap(), vec![qi(-2), qi(-5), qi(1)]);
    /// // det(B − λI) = −λ³ + 3λ²/2 + 3λ − 1/4 for a rational 3×3
    /// let b = QMatrix::new(vec![
    ///     vec![q(1, 2), qi(1), qi(0)],
    ///     vec![qi(2), qi(1), q(1, 2)],
    ///     vec![qi(1), qi(3), qi(0)],
    /// ]).unwrap();
    /// assert_eq!(b.char_poly_coeffs().unwrap(), vec![q(-1, 4), qi(3), q(3, 2), qi(-1)]);
    /// ```
    pub fn char_poly_coeffs(&self) -> Result<Vec<Q>, SymplexError> {
        self.require_square("char_poly_coeffs")?;
        let n = self.nrows;
        let monic = self.berkowitz_monic()?;
        let sign_flip = n % 2 == 1;
        Ok((0..=n)
            .map(|k| {
                let c = monic[n - k].clone();
                if sign_flip { -c } else { c }
            })
            .collect())
    }

    /// LU decomposition with partial pivoting: `P·A = L·U`, returned as
    /// [`Lu`]`{ l, u, perm }` with `L` unit lower triangular, `U` upper
    /// triangular and `perm[i]` the original index of row `i` of `P·A`.
    /// The pivot of each column is the first non-zero entry at or below
    /// the diagonal — the rule of [`Matrix::lu`], which routes here for
    /// rational matrices and therefore returns the same factors entry for
    /// entry.
    ///
    /// # Errors
    ///
    /// - [`SymplexError::InvalidArgument`] if the matrix is not square.
    /// - [`SymplexError::ComputationFailed`] if the matrix is singular.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::decompositions::Lu;
    /// use symplex::matrix::QMatrix;
    /// use symplex::linprog::{q, qi};
    ///
    /// let a = QMatrix::from_i64(&[&[4, 3], &[6, 3]]).unwrap();
    /// let Lu { l, u, perm } = a.lu().unwrap();
    /// assert_eq!(perm, vec![0, 1]);
    /// assert_eq!(l[(1, 0)], q(3, 2));
    /// assert_eq!(u.row(1), &[qi(0), q(-3, 2)]);
    /// assert_eq!(&l * &u, a);
    /// assert!(QMatrix::from_i64(&[&[1, 2], &[2, 4]]).unwrap().lu().is_err());
    /// ```
    pub fn lu(&self) -> Result<Lu<QMatrix>, SymplexError> {
        self.require_square("lu")?;
        let n = self.nrows;
        // P·(sA) = L·U_Z  ⇒  P·A = L·(U_Z / s): fraction-free over ℤ (Bareiss,
        // whose intermediates are the minors that Gaussian elimination's
        // entries are ratios of), one exact division per factor entry.
        let (z, s) = self.clear_denominators();
        let LuParts {
            rows,
            col,
            pivots,
            perm,
        } = try_lu(z.into_flat(), n)
            .map_err(|e| kernel_failed("lu", e))?
            .ok_or_else(|| failed("lu", "matrix is singular (zero pivot column)"))?;
        let mut l = QMatrix::identity(n);
        let mut u = QMatrix::zeros(n, n);
        for k in 0..n {
            // Row k of the buffer is pivot[k−1] · (row k of U_Z).
            let denom = match k {
                0 => s.clone(),
                _ => &pivots[k - 1] * &s,
            };
            for j in k..n {
                let v = &rows[k * n + j];
                if !v.is_zero() {
                    u.data[k * n + j] = Ratio::new(v.clone(), denom.clone());
                }
            }
            // Entry (i, k) before step k cleared it is pivot[k] · L[i][k].
            for i in (k + 1)..n {
                let v = &col[i * n + k];
                if !v.is_zero() {
                    l.data[i * n + k] = Ratio::new(v.clone(), pivots[k].clone());
                }
            }
        }
        Ok(Lu { l, u, perm })
    }

    /// `true` if `self` equals its transpose.
    pub fn is_symmetric(&self) -> bool {
        self.is_square()
            && (0..self.nrows).all(|i| {
                (0..i).all(|j| self.data[i * self.ncols + j] == self.data[j * self.ncols + i])
            })
    }

    /// Exact `L·D·Lᵀ` factorisation of a symmetric **positive semidefinite**
    /// matrix: `L` unit lower triangular, `D = diag(d)` with every `dₖ ≥ 0`.
    /// Returns `None` if the matrix is not square, not symmetric, or not
    /// PSD — this is an exact PSD test.  When a pivot `dₖ` is zero the
    /// remaining entries of its column must vanish (as they do for a PSD
    /// matrix), and `L`'s column is left zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::matrix::QMatrix;
    /// use symplex::linprog::{q, qi};
    ///
    /// let a = QMatrix::from_i64(&[&[4, 2], &[2, 1]]).unwrap();       // rank 1, PSD
    /// let (l, d) = a.ldl_psd().unwrap();
    /// assert_eq!(d, vec![qi(4), qi(0)]);
    /// assert_eq!(l[(1, 0)], q(1, 2));
    /// assert!(QMatrix::from_i64(&[&[1, 2], &[2, 1]]).unwrap().ldl_psd().is_none());   // indefinite
    /// ```
    pub fn ldl_psd(&self) -> Option<(QMatrix, Vec<Q>)> {
        if !self.is_symmetric() {
            return None;
        }
        let n = self.nrows;
        let mut l = QMatrix::identity(n);
        let mut d: Vec<Q> = Vec::with_capacity(n);
        for k in 0..n {
            let mut dk = self.data[k * n + k].clone();
            for (j, dj) in d.iter().enumerate() {
                let ljk = &l.data[k * n + j];
                if !ljk.is_zero() {
                    dk -= ljk * ljk * dj;
                }
            }
            if dk.is_negative() {
                return None;
            }
            for i in (k + 1)..n {
                let mut c = self.data[i * n + k].clone();
                for (j, dj) in d.iter().enumerate() {
                    let (lij, lkj) = (&l.data[i * n + j], &l.data[k * n + j]);
                    if !lij.is_zero() && !lkj.is_zero() {
                        c -= lij * dj * lkj;
                    }
                }
                if dk.is_zero() {
                    if !c.is_zero() {
                        return None;
                    }
                    l.data[i * n + k] = Q::zero();
                } else {
                    l.data[i * n + k] = c / &dk;
                }
            }
            d.push(dk);
        }
        Some((l, d))
    }

    /// Exact positive-semidefiniteness test (symmetric and every
    /// `xᵀAx ≥ 0`), via [`ldl_psd`](Self::ldl_psd).
    pub fn is_positive_semidefinite(&self) -> bool {
        self.ldl_psd().is_some()
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
// ℚ-specific operations (0.9): rank factorisation, pseudo-inverse, Hessenberg
// ═══════════════════════════════════════════════════════════════════════════

impl QMatrix {
    /// `(C, F)` of the full-rank factorisation `A = C·F`, or `None` for the
    /// zero matrix (rank 0, whose factors would be empty).
    fn rank_factors(&self) -> Option<(QMatrix, QMatrix)> {
        let (r, pivots) = self.rref();
        let rank = pivots.len();
        if rank == 0 {
            return None;
        }
        let c = QMatrix::from_fn(self.nrows, rank, |i, k| {
            self.data[i * self.ncols + pivots[k]].clone()
        });
        let f = r.submatrix(0..rank, 0..self.ncols).ok()?;
        Some((c, f))
    }

    /// Full-rank factorisation `A = C·F`: `C` (`m × r`) holds the pivot
    /// columns of `A`, `F` (`r × n`) the nonzero rows of `rref(A)`, where
    /// `r = rank A`.  SymPy: `Matrix.rank_decomposition()`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::ComputationFailed`] for the zero matrix (rank 0; the
    /// factors would have an empty dimension).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::matrix::QMatrix;
    ///
    /// let a = QMatrix::from_i64(&[&[1, 2, 3], &[4, 5, 6], &[7, 8, 9]]).unwrap();
    /// let rd = a.rank_decomposition().unwrap();
    /// assert_eq!(rd.c, QMatrix::from_i64(&[&[1, 2], &[4, 5], &[7, 8]]).unwrap());
    /// assert_eq!(rd.f, QMatrix::from_i64(&[&[1, 0, -1], &[0, 1, 2]]).unwrap());
    /// assert_eq!(&rd.c * &rd.f, a);
    /// ```
    pub fn rank_decomposition(&self) -> Result<RankDecomposition<QMatrix>, SymplexError> {
        self.rank_factors()
            .map(|(c, f)| RankDecomposition { c, f })
            .ok_or_else(|| {
                failed(
                    "rank_decomposition",
                    "matrix is zero (rank 0); the factors C (m×0) and F (0×n) would be empty",
                )
            })
    }

    /// Moore–Penrose pseudo-inverse `A⁺` (`n × m`), exact for any rank.
    /// SymPy: `Matrix.pinv()`.
    ///
    /// Uses the full-rank factorisation `A = C·F` of
    /// [`rank_decomposition`](Self::rank_decomposition):
    /// `A⁺ = Fᵀ (F Fᵀ)⁻¹ (Cᵀ C)⁻¹ Cᵀ`.  The zero matrix maps to the zero
    /// `n × m` matrix.
    ///
    /// # Errors
    ///
    /// [`SymplexError::ComputationFailed`] only if an internal invariant is
    /// violated (`CᵀC` and `FFᵀ` are invertible by construction).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::matrix::QMatrix;
    /// use symplex::linprog::q;
    ///
    /// let a = QMatrix::from_i64(&[&[1, 2], &[2, 4]]).unwrap();   // rank 1
    /// let p = a.pinv().unwrap();
    /// assert_eq!(p[(0, 0)], q(1, 25));
    /// assert_eq!(p[(1, 1)], q(4, 25));
    /// assert_eq!(&(&a * &p) * &a, a);    // A A⁺ A = A
    /// ```
    pub fn pinv(&self) -> Result<QMatrix, SymplexError> {
        let Some((c, f)) = self.rank_factors() else {
            return Ok(QMatrix::zeros(self.ncols, self.nrows));
        };
        let internal = |e: SymplexError| match e {
            SymplexError::ComputationFailed { reason, .. }
            | SymplexError::InvalidArgument { reason, .. } => failed(
                "pinv",
                format!("internal invariant violated in the rank factorisation: {reason}"),
            ),
            other => other,
        };
        let ft = f.transpose();
        let ct = c.transpose();
        let fft_inv = f.matmul(&ft).and_then(|m| m.inv()).map_err(internal)?;
        let ctc_inv = ct.matmul(&c).and_then(|m| m.inv()).map_err(internal)?;
        ft.matmul(&fft_inv)
            .and_then(|m| m.matmul(&ctc_inv))
            .and_then(|m| m.matmul(&ct))
            .map_err(internal)
    }

    /// Upper Hessenberg form by Gaussian similarity transforms:
    /// [`Hessenberg`]`{ h, p }` with `H = P⁻¹ A P` and `h_ij = 0` for
    /// `i > j + 1`.  SymPy: `Matrix.upper_hessenberg_decomposition()` (which
    /// uses Householder reflections and therefore radicals; this variant
    /// stays in ℚ).
    ///
    /// Column `k` is cleared below the sub-diagonal with the first nonzero
    /// candidate as pivot (a symmetric row/column swap when it is not
    /// already in row `k + 1`), followed by the row operations
    /// `row_j −= f·row_{k+1}` and the compensating column operations
    /// `col_{k+1} += f·col_j`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if the matrix is not square.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::decompositions::Hessenberg;
    /// use symplex::matrix::QMatrix;
    /// use symplex::linprog::q;
    ///
    /// let a = QMatrix::from_i64(&[&[1, 2, 3], &[4, 5, 6], &[7, 8, 10]]).unwrap();
    /// let Hessenberg { h, p } = a.hessenberg().unwrap();
    /// assert_eq!(h[(2, 0)], q(0, 1));
    /// assert_eq!(&a * &p, &p * &h);          // A P = P H
    /// assert_eq!(p.inv().unwrap() * &a * &p, h);
    /// ```
    pub fn hessenberg(&self) -> Result<Hessenberg<QMatrix>, SymplexError> {
        self.require_square("hessenberg")?;
        let n = self.nrows;
        let mut h = self.clone();
        let mut p = QMatrix::identity(n);
        let swap_cols = |m: &mut QMatrix, a: usize, b: usize| {
            for i in 0..n {
                m.data.swap(i * n + a, i * n + b);
            }
        };
        for k in 0..n.saturating_sub(2) {
            let Some(piv) = ((k + 1)..n).find(|&i| !h.data[i * n + k].is_zero()) else {
                continue;
            };
            if piv != k + 1 {
                h.swap_rows(k + 1, piv);
                swap_cols(&mut h, k + 1, piv);
                swap_cols(&mut p, k + 1, piv);
            }
            let pv = h.data[(k + 1) * n + k].clone();
            for j in (k + 2)..n {
                let f = &h.data[j * n + k] / &pv;
                if f.is_zero() {
                    continue;
                }
                // row_j −= f · row_{k+1}  (entry (j, k) becomes exactly zero)
                let pivot_row: Vec<Q> = h.data[(k + 1) * n..(k + 2) * n].to_vec();
                for (c, pr) in pivot_row.iter().enumerate() {
                    if c == k {
                        h.data[j * n + c] = Q::zero();
                    } else if !pr.is_zero() {
                        let t = pr * &f;
                        h.data[j * n + c] -= t;
                    }
                }
                // col_{k+1} += f · col_j  (in H and in P)
                for r in 0..n {
                    let t = &h.data[r * n + j] * &f;
                    h.data[r * n + k + 1] += t;
                    let t = &p.data[r * n + j] * &f;
                    p.data[r * n + k + 1] += t;
                }
            }
        }
        Ok(Hessenberg { h, p })
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
        let HermiteNormalForm { h, u } = a.hermite_normal_form_with_transform();
        assert_eq!(h, z(&[&[2, 4, 4], &[0, 6, 0], &[0, 0, 12]]));
        assert_eq!(&u * &a, h);
        assert_eq!(u.det().unwrap().abs(), BigInt::one());
        assert_eq!(
            a.smith_normal_form(),
            z(&[&[2, 0, 0], &[0, 6, 0], &[0, 0, 12]])
        );
        let SmithNormalForm { s, u, v } = a.smith_normal_form_with_transforms();
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
            let u = random_z(4, 4, seed + 7)
                .hermite_normal_form_with_transform()
                .u;
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
    fn ldl_psd_is_an_exact_psd_test() {
        let psd = qm(&[&[4, 2, 0], &[2, 2, 1], &[0, 1, 1]]);
        let (l, d) = psd.ldl_psd().unwrap();
        assert!(d.iter().all(|v| !v.is_negative()));
        let dm = QMatrix::diag(&d);
        assert_eq!(&(&l * &dm) * &l.transpose(), psd);
        assert!(psd.is_positive_semidefinite());
        // Rank-deficient PSD: Gram matrix of (1, 1, 1).
        let ones = qm(&[&[1, 1, 1], &[1, 1, 1], &[1, 1, 1]]);
        let (l, d) = ones.ldl_psd().unwrap();
        assert_eq!(d, vec![q(1, 1), Q::zero(), Q::zero()]);
        assert_eq!(&(&l * &QMatrix::diag(&d)) * &l.transpose(), ones);
        // Zero pivot with a nonzero column entry: not PSD.
        assert!(qm(&[&[0, 1], &[1, 0]]).ldl_psd().is_none());
        assert!(qm(&[&[1, 2], &[2, 1]]).ldl_psd().is_none());
        assert!(qm(&[&[-1]]).ldl_psd().is_none());
        assert!(qm(&[&[1, 2], &[3, 4]]).ldl_psd().is_none()); // not symmetric
        assert!(qm(&[&[1, 2, 3]]).ldl_psd().is_none());
        assert!(QMatrix::zeros(3, 3).is_positive_semidefinite());
        // Random Gram matrices BᵀB are PSD; BᵀB − εI is not for ε above λ_min.
        for seed in 1..=6u64 {
            let b = random_z(3, 5, seed).to_qmatrix();
            let g = &b.transpose() * &b;
            assert!(g.is_positive_semidefinite(), "seed {seed}");
            let big = &g - &QMatrix::identity(5).scale(&q(1000, 1));
            assert!(!big.is_positive_semidefinite());
        }
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
