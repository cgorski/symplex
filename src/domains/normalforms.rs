//! Integer matrix normal forms: Hermite normal form, Smith normal form,
//! unimodular transforms and integer kernels.
//!
//! Everything here is exact over `BigInt`.  Inputs are [`Matrix`] values
//! whose entries must be integer literals (a fraction, a symbol or an
//! unevaluated constant expression gives
//! [`SymplexError::InvalidArgument`]); results are returned as integer
//! matrices in the same [`Context`](crate::context::Context).  The
//! algorithms live on [`ZMatrix`] — use it directly when the data is
//! already integer to skip the expression layer.
//!
//! # Conventions
//!
//! * [`hermite_normal_form`] is **row-style**: `H = U·A` with `U`
//!   unimodular (`det U = ±1`).  `H` is in row echelon form with positive
//!   pivots, every entry above a pivot reduced into `[0, pivot)`, zero rows
//!   at the bottom.  This form is unique, so it is idempotent and invariant
//!   under left-multiplication by any unimodular matrix.
//! * [`column_hermite_normal_form`] is **column-style**: `H = A·V`, the
//!   convention used by SymPy's `hermite_normal_form` (Cohen, *A Course in
//!   Computational Algebraic Number Theory*, Algorithm 2.4.5): each nonzero
//!   column's pivot is its *lowest* nonzero entry, pivots move strictly
//!   downwards from left to right, pivots are positive, entries to the
//!   right of a pivot in its row lie in `[0, pivot)`, zero columns come
//!   first.  For a square nonsingular matrix this is upper triangular.
//! * [`smith_normal_form`] gives `S = U·A·V = diag(d₁, …, dᵣ, 0, …)` with
//!   `dᵢ > 0` and `dᵢ | dᵢ₊₁`.
//!
//! # Examples
//!
//! ```
//! use symplex::prelude::*;
//! use symplex::normalforms::{hermite_normal_form_with_transform, smith_normal_form};
//!
//! let ctx = Context::new();
//! let a = matrix![ctx, [2, 4, 4], [-6, 6, 12], [10, -4, -16]];
//! let (h, u) = hermite_normal_form_with_transform(&a).unwrap();
//! assert_eq!(h, matrix![ctx, [2, 4, 4], [0, 6, 0], [0, 0, 12]]);
//! assert_eq!((&u * &a).eval(), h);
//! assert_eq!(smith_normal_form(&a).unwrap(), matrix![ctx, [2, 0, 0], [0, 6, 0], [0, 0, 12]]);
//! ```

use num_bigint::BigInt;

use crate::base::errors::SymplexError;
use crate::domains::exact_matrix::ZMatrix;
use crate::domains::matrix::Matrix;

/// Integer entries of `m` as a [`ZMatrix`], trying a constant-folding
/// `eval()` if the raw entries are not literals.
fn integer_matrix(m: &Matrix, operation: &'static str) -> Result<ZMatrix, SymplexError> {
    ZMatrix::try_from(m).map_err(|e| match e {
        SymplexError::InvalidArgument { reason, .. } => {
            SymplexError::InvalidArgument { operation, reason }
        }
        other => other,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Public API: Hermite normal form
// ═══════════════════════════════════════════════════════════════════════════

/// Row-style Hermite normal form `H` of an integer matrix `A`.
///
/// There is a unimodular matrix `U` (`det U = ±1`) with `H = U·A`, and `H`
/// satisfies:
///
/// * **echelon:** the nonzero rows come first and the pivot (first nonzero
///   entry) of each nonzero row is strictly to the right of the pivot of
///   the row above; zero rows are at the bottom;
/// * **positive pivots:** every pivot is `> 0`;
/// * **reduced:** every entry above a pivot, in the pivot's column, lies in
///   `[0, pivot)`.
///
/// This `H` is unique, so `hermite_normal_form` is idempotent and
/// `hermite_normal_form(V·A) == hermite_normal_form(A)` for every
/// unimodular `V`.  The number of nonzero rows is the rank of `A`, and the
/// rows of `H` are a canonical basis of the row lattice of `A`.  Use
/// [`hermite_normal_form_with_transform`] to obtain `U`.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] if any entry is not an integer literal.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::normalforms::hermite_normal_form;
///
/// let ctx = Context::new();
/// let a = matrix![ctx, [2, 4, 4], [-6, 6, 12], [10, -4, -16]];
/// assert_eq!(hermite_normal_form(&a).unwrap(), matrix![ctx, [2, 4, 4], [0, 6, 0], [0, 0, 12]]);
///
/// // Rank-deficient: the zero row moves to the bottom.
/// let b = matrix![ctx, [1, 2], [2, 4]];
/// assert_eq!(hermite_normal_form(&b).unwrap(), matrix![ctx, [1, 2], [0, 0]]);
/// ```
pub fn hermite_normal_form(m: &Matrix) -> Result<Matrix, SymplexError> {
    let z = integer_matrix(m, "hermite_normal_form")?;
    Ok(z.hermite_normal_form().to_matrix(&m.context()))
}

/// Row-style Hermite normal form together with its transform: `(H, U)`
/// with `H = U·A` and `det U = ±1`.  See [`hermite_normal_form`] for the
/// normalisation of `H`.
///
/// `U` is not unique when `A` is rank-deficient (any row of `U` mapping to
/// a zero row of `H` can be adjusted by kernel vectors); the returned `U` is
/// the one produced by the elimination.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] if any entry is not an integer literal.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::normalforms::hermite_normal_form_with_transform;
///
/// let ctx = Context::new();
/// let a = matrix![ctx, [3, 1], [1, 2]];
/// let (h, u) = hermite_normal_form_with_transform(&a).unwrap();
/// assert_eq!((&u * &a).eval(), h);
/// assert_eq!(h, matrix![ctx, [1, 2], [0, 5]]);
/// assert_eq!(u.det().unwrap().as_i64().unwrap().abs(), 1);
/// ```
pub fn hermite_normal_form_with_transform(m: &Matrix) -> Result<(Matrix, Matrix), SymplexError> {
    let z = integer_matrix(m, "hermite_normal_form_with_transform")?;
    let (h, u) = z.hermite_normal_form_with_transform();
    let ctx = m.context();
    Ok((h.to_matrix(&ctx), u.to_matrix(&ctx)))
}

/// Column-style Hermite normal form `H = A·V` (column operations), the
/// convention of SymPy's `hermite_normal_form` and of Cohen's
/// Algorithm 2.4.5.
///
/// `H` has the same shape as `A` and there is a unimodular `V` with
/// `H = A·V`.  Normalisation:
///
/// * zero columns come first, followed by the nonzero columns;
/// * the pivot of a nonzero column is its **lowest** nonzero entry, and the
///   pivot rows strictly increase from left to right (so a square
///   nonsingular `A` gives an upper-triangular `H`);
/// * pivots are positive;
/// * in a pivot's row, every entry to the *right* of the pivot lies in
///   `[0, pivot)`.
///
/// SymPy drops the leading zero columns; here they are kept so that
/// `H = A·V` holds with `V` square.
///
/// Relation to the row form: reverse the rows of `A`, take
/// [`hermite_normal_form`] of the transpose, transpose back, and reverse
/// both rows and columns.  (Transposing alone would give a *lower*
/// triangular form with pivots at the top of each column.)
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] if any entry is not an integer literal.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::normalforms::column_hermite_normal_form;
///
/// let ctx = Context::new();
/// // SymPy: hermite_normal_form(Matrix([[12, 6, 4], [3, 9, 6], [2, 16, 14]]))
/// let a = matrix![ctx, [12, 6, 4], [3, 9, 6], [2, 16, 14]];
/// let h = column_hermite_normal_form(&a).unwrap();
/// assert_eq!(h, matrix![ctx, [10, 0, 2], [0, 15, 3], [0, 0, 2]]);
/// ```
pub fn column_hermite_normal_form(m: &Matrix) -> Result<Matrix, SymplexError> {
    let z = integer_matrix(m, "column_hermite_normal_form")?;
    Ok(z.column_hermite_normal_form().to_matrix(&m.context()))
}

// ═══════════════════════════════════════════════════════════════════════════
// Smith normal form
// ═══════════════════════════════════════════════════════════════════════════

/// Smith normal form `S = diag(d₁, …, dᵣ, 0, …, 0)` of an integer matrix,
/// with `dᵢ > 0` and `dᵢ | dᵢ₊₁`.
///
/// There are unimodular `U`, `V` with `S = U·A·V` (see
/// [`smith_normal_form_with_transforms`]).  `r` is the rank of `A`, and
/// `d₁⋯dₖ` equals the gcd of the `k×k` minors of `A`, so the `dᵢ`
/// (the *invariant factors*) are unique.  For a square nonsingular `A`,
/// `d₁⋯dₙ = |det A|`.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] if any entry is not an integer literal.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::normalforms::smith_normal_form;
///
/// let ctx = Context::new();
/// // SymPy: smith_normal_form(Matrix([[12, 6, 4], [3, 9, 6], [2, 16, 14]]))
/// let a = matrix![ctx, [12, 6, 4], [3, 9, 6], [2, 16, 14]];
/// assert_eq!(smith_normal_form(&a).unwrap(), matrix![ctx, [1, 0, 0], [0, 10, 0], [0, 0, 30]]);
/// ```
pub fn smith_normal_form(m: &Matrix) -> Result<Matrix, SymplexError> {
    let z = integer_matrix(m, "smith_normal_form")?;
    Ok(z.smith_normal_form().to_matrix(&m.context()))
}

/// Smith normal form with transforms: `(S, U, V)` such that `S = U·A·V`,
/// `det U = ±1`, `det V = ±1`.  See [`smith_normal_form`] for the form of
/// `S`.  `U` and `V` are not unique; the returned pair is the one produced
/// by the elimination.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] if any entry is not an integer literal.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::normalforms::smith_normal_form_with_transforms;
///
/// let ctx = Context::new();
/// let a = matrix![ctx, [2, 4, 4], [-6, 6, 12], [10, -4, -16]];
/// let (s, u, v) = smith_normal_form_with_transforms(&a).unwrap();
/// assert_eq!(s, matrix![ctx, [2, 0, 0], [0, 6, 0], [0, 0, 12]]);
/// assert_eq!((&(&u * &a) * &v).eval(), s);
/// ```
pub fn smith_normal_form_with_transforms(
    m: &Matrix,
) -> Result<(Matrix, Matrix, Matrix), SymplexError> {
    let z = integer_matrix(m, "smith_normal_form_with_transforms")?;
    let (s, u, v) = z.smith_normal_form_with_transforms();
    let ctx = m.context();
    Ok((s.to_matrix(&ctx), u.to_matrix(&ctx), v.to_matrix(&ctx)))
}

// ═══════════════════════════════════════════════════════════════════════════
// Integer kernel, unimodularity, lattice determinant
// ═══════════════════════════════════════════════════════════════════════════

/// A ℤ-basis of the integer kernel `{x ∈ ℤⁿ : A·x = 0}` of an `m×n`
/// integer matrix, as `n×1` column vectors.
///
/// The basis has `n − rank(A)` vectors (empty for full column rank) and
/// generates *every* integer solution — unlike the rational
/// [`Matrix::nullspace`], whose basis vectors may only span the kernel over
/// ℚ.  Computed from the row Hermite normal form of `[Aᵀ | I]`: the rows
/// whose `Aᵀ` part vanishes are exactly a basis of the kernel lattice.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] if any entry is not an integer literal.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::normalforms::integer_nullspace;
///
/// let ctx = Context::new();
/// let a = matrix![ctx, [2, 4, 6]];
/// let basis = integer_nullspace(&a).unwrap();
/// assert_eq!(basis.len(), 2);
/// for k in &basis {
///     assert_eq!((&a * k).eval(), matrix![ctx, [0]]);
/// }
/// // Full column rank: trivial kernel.
/// assert!(integer_nullspace(&matrix![ctx, [1, 0], [0, 1], [1, 1]]).unwrap().is_empty());
/// ```
pub fn integer_nullspace(m: &Matrix) -> Result<Vec<Matrix>, SymplexError> {
    let z = integer_matrix(m, "integer_nullspace")?;
    let ctx = m.context();
    Ok(z.integer_nullspace()
        .iter()
        .map(|k| k.to_matrix(&ctx))
        .collect())
}

/// Is `A` a square integer matrix with `det A = ±1` (invertible over ℤ)?
///
/// Non-square matrices give `Ok(false)`.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] if any entry is not an integer literal.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::normalforms::is_unimodular;
///
/// let ctx = Context::new();
/// assert!(is_unimodular(&matrix![ctx, [2, 1], [1, 1]]).unwrap());
/// assert!(!is_unimodular(&matrix![ctx, [2, 0], [0, 1]]).unwrap());
/// assert!(!is_unimodular(&matrix![ctx, [1, 2, 3]]).unwrap());
/// ```
pub fn is_unimodular(m: &Matrix) -> Result<bool, SymplexError> {
    let z = integer_matrix(m, "is_unimodular")?;
    Ok(z.is_unimodular())
}

/// Determinant (index) of the lattice spanned by the **columns** of `A`.
///
/// For an `m×n` integer matrix of full row rank (`rank A = m ≤ n`) the
/// column lattice `L = A·ℤⁿ` is a full-rank sublattice of `ℤᵐ`; this
/// returns its index `[ℤᵐ : L]`, a positive integer equal to the product of
/// the pivots of the [column Hermite normal form](column_hermite_normal_form)
/// and to the gcd of all `m×m` minors of `A`.  For a square nonsingular
/// matrix it is `|det A|`.  (The covolume `√det(AᵀA)` of a full-column-rank
/// lattice is generally irrational and is not what this function computes.)
///
/// # Errors
///
/// * [`SymplexError::InvalidArgument`] if any entry is not an integer
///   literal, or if `A` does not have full row rank (the index would be
///   infinite).
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::normalforms::lattice_determinant;
/// use num_bigint::BigInt;
///
/// let ctx = Context::new();
/// assert_eq!(lattice_determinant(&matrix![ctx, [2, 0], [0, 3]]).unwrap(), BigInt::from(6));
/// // The columns (2, 0), (0, 3), (1, 1) generate a lattice of index 1 in ℤ².
/// assert_eq!(lattice_determinant(&matrix![ctx, [2, 0, 1], [0, 3, 1]]).unwrap(), BigInt::from(1));
/// assert!(lattice_determinant(&matrix![ctx, [1, 2], [2, 4]]).is_err());
/// ```
pub fn lattice_determinant(m: &Matrix) -> Result<BigInt, SymplexError> {
    let z = integer_matrix(m, "lattice_determinant")?;
    z.lattice_determinant()
}

// ═══════════════════════════════════════════════════════════════════════════
// Public API: LLL lattice reduction (0.9)
// ═══════════════════════════════════════════════════════════════════════════

/// LLL-reduced basis of the lattice spanned by the **rows** of the integer
/// matrix `A`, with Lovász parameter `δ = num/den` (the standard choice
/// is `(3, 4)`).  SymPy: `Matrix.lll(delta)`.
///
/// The Gram–Schmidt data is exact (rational), so the result satisfies the
/// size condition `|μ_ij| ≤ 1/2` and the Lovász condition
/// `‖b*_k‖² ≥ (δ − μ²_{k,k−1})‖b*_{k−1}‖²` exactly, and spans the same
/// lattice as `A` (same [`hermite_normal_form`]).  See [`ZMatrix::lll`]
/// for the algorithm.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] if any entry is not an integer
/// literal, `δ` is not in the open interval `(1/4, 1)`, or the rows are
/// linearly dependent.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::normalforms::{hermite_normal_form, lll};
///
/// let ctx = Context::new();
/// let b = matrix![ctx, [1, 1, 1], [-1, 0, 2], [3, 5, 6]];
/// let r = lll(&b, (3, 4)).unwrap();
/// // SymPy 1.14: Matrix([[1,1,1],[-1,0,2],[3,5,6]]).lll() == [[0,1,0],[1,0,1],[-1,0,2]]
/// assert_eq!(r, matrix![ctx, [0, 1, 0], [1, 0, 1], [-1, 0, 2]]);
/// assert_eq!(hermite_normal_form(&r).unwrap(), hermite_normal_form(&b).unwrap());
/// ```
pub fn lll(m: &Matrix, delta: (i64, i64)) -> Result<Matrix, SymplexError> {
    let z = integer_matrix(m, "lll")?;
    Ok(z.lll(delta)?.to_matrix(&m.context()))
}

/// LLL reduction with its unimodular transform: `(R, T)` with `R = T·A`
/// and `det T = ±1`.  SymPy: `Matrix.lll_transform(delta)`.
///
/// # Errors
///
/// Same as [`lll`].
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::normalforms::{is_unimodular, lll_with_transform};
///
/// let ctx = Context::new();
/// let b = matrix![ctx, [1, 1, 1], [-1, 0, 2], [3, 5, 6]];
/// let (r, t) = lll_with_transform(&b, (3, 4)).unwrap();
/// assert_eq!((&t * &b).eval(), r);
/// assert!(is_unimodular(&t).unwrap());
/// ```
pub fn lll_with_transform(m: &Matrix, delta: (i64, i64)) -> Result<(Matrix, Matrix), SymplexError> {
    let z = integer_matrix(m, "lll_with_transform")?;
    let (r, t) = z.lll_with_transform(delta)?;
    let ctx = m.context();
    Ok((r.to_matrix(&ctx), t.to_matrix(&ctx)))
}
