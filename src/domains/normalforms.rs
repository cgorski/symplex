//! Integer matrix normal forms: Hermite normal form, Smith normal form,
//! unimodular transforms and integer kernels.
//!
//! Everything here is exact over `BigInt`.  Inputs are [`Matrix`] values
//! whose entries must be integer literals (a fraction, a symbol or an
//! unevaluated constant expression gives
//! [`SymplexError::InvalidArgument`]); results are returned as integer
//! matrices in the same [`Context`].
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
use num_integer::Integer;
use num_traits::{One, Signed, Zero};

use crate::api::context::Context;
use crate::base::errors::SymplexError;
use crate::domains::matrix::Matrix;
use crate::domains::ntheory::gcdex;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

fn invalid(operation: &'static str, reason: impl Into<String>) -> SymplexError {
    SymplexError::InvalidArgument {
        operation,
        reason: reason.into(),
    }
}

/// Integer entries of `m`, trying a constant-folding `eval()` if the raw
/// entries are not literals.
fn integer_rows(m: &Matrix, operation: &'static str) -> Result<Vec<Vec<BigInt>>, SymplexError> {
    if let Some(rows) = m.to_bigint_rows() {
        return Ok(rows);
    }
    m.eval().to_bigint_rows().ok_or_else(|| {
        invalid(
            operation,
            "every entry must be an integer literal (fractions and symbolic \
             entries are not allowed)",
        )
    })
}

fn identity_rows(n: usize) -> Vec<Vec<BigInt>> {
    (0..n)
        .map(|i| {
            (0..n)
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
}

fn transpose_rows(a: &[Vec<BigInt>]) -> Vec<Vec<BigInt>> {
    let m = a.len();
    let n = a.first().map_or(0, Vec::len);
    (0..n)
        .map(|j| (0..m).map(|i| a[i][j].clone()).collect())
        .collect()
}

fn to_matrix(ctx: &Context, rows: &[Vec<BigInt>]) -> Result<Matrix, SymplexError> {
    Matrix::from_bigint(ctx, rows)
}

/// Replace rows `r` and `i` by the unimodular combination
/// `(x·row_r + y·row_i, p·row_r + q·row_i)`; the caller guarantees
/// `x·q − y·p = ±1`.
fn combine_rows(
    a: &mut [Vec<BigInt>],
    r: usize,
    i: usize,
    x: &BigInt,
    y: &BigInt,
    p: &BigInt,
    q: &BigInt,
) {
    if r == i {
        return;
    }
    let (row_r, row_i) = two_rows_mut(a, r, i);
    for (ar, ai) in row_r.iter_mut().zip(row_i.iter_mut()) {
        let old_r = std::mem::take(ar);
        let old_i = std::mem::take(ai);
        *ar = x * &old_r + y * &old_i;
        *ai = p * &old_r + q * &old_i;
    }
}

/// `row_i -= q · row_r`.
fn sub_scaled_row(a: &mut [Vec<BigInt>], i: usize, r: usize, q: &BigInt) {
    if r == i {
        return;
    }
    let (row_r, row_i) = two_rows_mut(a, r, i);
    for (ar, ai) in row_r.iter().zip(row_i.iter_mut()) {
        if !ar.is_zero() {
            *ai -= q * ar;
        }
    }
}

/// Disjoint mutable borrows of rows `r` and `i` (`r != i`), returned in
/// that order.
fn two_rows_mut(a: &mut [Vec<BigInt>], r: usize, i: usize) -> (&mut [BigInt], &mut [BigInt]) {
    debug_assert!(r != i);
    if r < i {
        let (lo, hi) = a.split_at_mut(i);
        (&mut lo[r], &mut hi[0])
    } else {
        let (lo, hi) = a.split_at_mut(r);
        (&mut hi[0], &mut lo[i])
    }
}

fn negate_row(a: &mut [Vec<BigInt>], r: usize) {
    for v in a[r].iter_mut() {
        *v = -std::mem::take(v);
    }
}

/// Use the pair `(a[r][col], a[i][col])` to zero `a[i][col]` with a
/// unimodular row combination applied to both `a` and `u`.
fn eliminate_row_pair(
    a: &mut [Vec<BigInt>],
    u: &mut [Vec<BigInt>],
    r: usize,
    i: usize,
    col: usize,
) {
    if a[i][col].is_zero() {
        return;
    }
    if a[r][col].is_zero() {
        a.swap(r, i);
        u.swap(r, i);
        return;
    }
    let pivot = a[r][col].clone();
    let other = a[i][col].clone();
    let (rem, quot) = (&other % &pivot, &other / &pivot);
    if rem.is_zero() {
        sub_scaled_row(a, i, r, &quot);
        sub_scaled_row(u, i, r, &quot);
        return;
    }
    let (g, x, y) = gcdex(pivot.clone(), other.clone());
    let p = -(&other / &g);
    let q = &pivot / &g;
    combine_rows(a, r, i, &x, &y, &p, &q);
    combine_rows(u, r, i, &x, &y, &p, &q);
}

// ═══════════════════════════════════════════════════════════════════════════
// Row-style Hermite normal form (core)
// ═══════════════════════════════════════════════════════════════════════════

/// Result of the row-HNF core: `h = u · a`, plus the pivot column of each
/// nonzero row of `h` (so `pivots.len()` is the rank).
struct RowHnf {
    h: Vec<Vec<BigInt>>,
    u: Vec<Vec<BigInt>>,
    pivots: Vec<usize>,
}

fn row_hnf(a: &[Vec<BigInt>]) -> RowHnf {
    let m = a.len();
    let n = a.first().map_or(0, Vec::len);
    let mut h: Vec<Vec<BigInt>> = a.to_vec();
    let mut u = identity_rows(m);
    let mut pivots = Vec::new();
    let mut r = 0usize;

    for col in 0..n {
        if r >= m {
            break;
        }
        // Bring the smallest non-zero |entry| of the column into row `r`
        // first: it keeps the intermediate quotients small.
        if let Some(best) = (r..m)
            .filter(|&i| !h[i][col].is_zero())
            .min_by(|&i, &j| h[i][col].abs().cmp(&h[j][col].abs()))
            && best != r
        {
            h.swap(r, best);
            u.swap(r, best);
        }
        for i in (r + 1)..m {
            eliminate_row_pair(&mut h, &mut u, r, i, col);
        }
        if h[r][col].is_zero() {
            continue;
        }
        if h[r][col].is_negative() {
            negate_row(&mut h, r);
            negate_row(&mut u, r);
        }
        let pivot = h[r][col].clone();
        for i in 0..r {
            let q = h[i][col].div_floor(&pivot);
            if !q.is_zero() {
                sub_scaled_row(&mut h, i, r, &q);
                sub_scaled_row(&mut u, i, r, &q);
            }
        }
        pivots.push(col);
        r += 1;
    }
    RowHnf { h, u, pivots }
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
    let rows = integer_rows(m, "hermite_normal_form")?;
    let hnf = row_hnf(&rows);
    to_matrix(&m.context(), &hnf.h)
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
    let rows = integer_rows(m, "hermite_normal_form_with_transform")?;
    let hnf = row_hnf(&rows);
    let ctx = m.context();
    Ok((to_matrix(&ctx, &hnf.h)?, to_matrix(&ctx, &hnf.u)?))
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
    let rows = integer_rows(m, "column_hermite_normal_form")?;
    let h = column_hnf_rows(&rows);
    to_matrix(&m.context(), &h)
}

/// Column HNF core (Cohen convention) via the reversed row HNF.
fn column_hnf_rows(a: &[Vec<BigInt>]) -> Vec<Vec<BigInt>> {
    let mut reversed: Vec<Vec<BigInt>> = a.to_vec();
    reversed.reverse();
    let hnf = row_hnf(&transpose_rows(&reversed));
    let mut h = transpose_rows(&hnf.h);
    h.reverse();
    for row in &mut h {
        row.reverse();
    }
    h
}

// ═══════════════════════════════════════════════════════════════════════════
// Smith normal form
// ═══════════════════════════════════════════════════════════════════════════

/// Column analogue of [`combine_rows`]: replaces columns `c` and `j`.
fn combine_cols(
    a: &mut [Vec<BigInt>],
    c: usize,
    j: usize,
    x: &BigInt,
    y: &BigInt,
    p: &BigInt,
    q: &BigInt,
) {
    for row in a.iter_mut() {
        let ac = row[c].clone();
        let aj = row[j].clone();
        row[c] = x * &ac + y * &aj;
        row[j] = p * &ac + q * &aj;
    }
}

/// `col_j -= q · col_c`.
fn sub_scaled_col(a: &mut [Vec<BigInt>], j: usize, c: usize, q: &BigInt) {
    for row in a.iter_mut() {
        if !row[c].is_zero() {
            let t = q * &row[c];
            row[j] -= t;
        }
    }
}

/// Zero `s[t][j]` using columns `t` and `j` (applied to `s` and `v`).
fn eliminate_col_pair(s: &mut [Vec<BigInt>], v: &mut [Vec<BigInt>], t: usize, j: usize) {
    if s[t][j].is_zero() {
        return;
    }
    let pivot = s[t][t].clone();
    let other = s[t][j].clone();
    if pivot.is_zero() {
        for row in s.iter_mut() {
            row.swap(t, j);
        }
        for row in v.iter_mut() {
            row.swap(t, j);
        }
        return;
    }
    let (rem, quot) = (&other % &pivot, &other / &pivot);
    if rem.is_zero() {
        sub_scaled_col(s, j, t, &quot);
        sub_scaled_col(v, j, t, &quot);
        return;
    }
    let (g, x, y) = gcdex(pivot.clone(), other.clone());
    let p = -(&other / &g);
    let q = &pivot / &g;
    combine_cols(s, t, j, &x, &y, &p, &q);
    combine_cols(v, t, j, &x, &y, &p, &q);
}

/// Row-major integer matrix used by the numeric cores.
type IntRows = Vec<Vec<BigInt>>;

/// Smith normal form core: `(s, u, v)` with `s = u · a · v`.
fn smith(a: &[Vec<BigInt>]) -> (IntRows, IntRows, IntRows) {
    let m = a.len();
    let n = a.first().map_or(0, Vec::len);
    let mut s: Vec<Vec<BigInt>> = a.to_vec();
    let mut u = identity_rows(m);
    let mut v = identity_rows(n);

    for t in 0..m.min(n) {
        // Move the smallest non-zero |entry| of the trailing block to (t, t).
        let mut best: Option<(usize, usize)> = None;
        for i in t..m {
            for j in t..n {
                if s[i][j].is_zero() {
                    continue;
                }
                match best {
                    Some((bi, bj)) if s[bi][bj].abs() <= s[i][j].abs() => {}
                    _ => best = Some((i, j)),
                }
            }
        }
        let Some((bi, bj)) = best else {
            break;
        };
        if bi != t {
            s.swap(t, bi);
            u.swap(t, bi);
        }
        if bj != t {
            for row in s.iter_mut() {
                row.swap(t, bj);
            }
            for row in v.iter_mut() {
                row.swap(t, bj);
            }
        }

        loop {
            for i in (t + 1)..m {
                eliminate_row_pair(&mut s, &mut u, t, i, t);
            }
            for j in (t + 1)..n {
                eliminate_col_pair(&mut s, &mut v, t, j);
            }
            let col_clear = ((t + 1)..m).all(|i| s[i][t].is_zero());
            let row_clear = ((t + 1)..n).all(|j| s[t][j].is_zero());
            if !(col_clear && row_clear) {
                continue;
            }
            // Divisibility: every remaining entry must be a multiple of the
            // pivot.  If not, fold the offending row into row `t`; the next
            // column sweep then replaces the pivot by a proper divisor.
            let d = s[t][t].clone();
            let offender = ((t + 1)..m).find(|&i| ((t + 1)..n).any(|j| !(&s[i][j] % &d).is_zero()));
            match offender {
                Some(i) => {
                    let one = BigInt::one();
                    let zero = BigInt::zero();
                    // row_t += row_i  (unimodular: [[1, 1], [0, 1]])
                    combine_rows(&mut s, t, i, &one, &one, &zero, &one);
                    combine_rows(&mut u, t, i, &one, &one, &zero, &one);
                }
                None => break,
            }
        }
        if s[t][t].is_negative() {
            negate_row(&mut s, t);
            negate_row(&mut u, t);
        }
    }
    (s, u, v)
}

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
    let rows = integer_rows(m, "smith_normal_form")?;
    let (s, _, _) = smith(&rows);
    to_matrix(&m.context(), &s)
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
    let rows = integer_rows(m, "smith_normal_form_with_transforms")?;
    let (s, u, v) = smith(&rows);
    let ctx = m.context();
    Ok((
        to_matrix(&ctx, &s)?,
        to_matrix(&ctx, &u)?,
        to_matrix(&ctx, &v)?,
    ))
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
    let rows = integer_rows(m, "integer_nullspace")?;
    let nrows = rows.len();
    let ncols = rows.first().map_or(0, Vec::len);
    // Build [Aᵀ | I] (ncols × (nrows + ncols)).
    let at = transpose_rows(&rows);
    let augmented: Vec<Vec<BigInt>> = at
        .into_iter()
        .enumerate()
        .map(|(i, mut row)| {
            row.extend((0..ncols).map(|j| {
                if i == j {
                    BigInt::one()
                } else {
                    BigInt::zero()
                }
            }));
            row
        })
        .collect();
    let hnf = row_hnf(&augmented);
    let rank = hnf.pivots.iter().filter(|&&c| c < nrows).count();
    let ctx = m.context();
    let mut basis = Vec::with_capacity(ncols - rank);
    for row in hnf.h.iter().skip(rank) {
        debug_assert!(row[..nrows].iter().all(Zero::is_zero));
        let col: Vec<Vec<BigInt>> = row[nrows..].iter().map(|v| vec![v.clone()]).collect();
        basis.push(to_matrix(&ctx, &col)?);
    }
    Ok(basis)
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
    let rows = integer_rows(m, "is_unimodular")?;
    if m.nrows() != m.ncols() {
        return Ok(false);
    }
    let hnf = row_hnf(&rows);
    if hnf.pivots.len() != m.nrows() {
        return Ok(false);
    }
    Ok((0..m.nrows()).all(|i| hnf.h[i][i].is_one()))
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
    let rows = integer_rows(m, "lattice_determinant")?;
    let nrows = m.nrows();
    let hnf = row_hnf(&transpose_rows(&rows));
    if hnf.pivots.len() != nrows {
        return Err(invalid(
            "lattice_determinant",
            format!(
                "matrix must have full row rank (rank {} of {} rows); the column \
                 lattice has infinite index otherwise",
                hnf.pivots.len(),
                nrows
            ),
        ));
    }
    Ok(hnf
        .pivots
        .iter()
        .enumerate()
        .map(|(i, &c)| hnf.h[i][c].clone())
        .product())
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn bi(n: i64) -> BigInt {
        BigInt::from(n)
    }

    fn rows(data: &[&[i64]]) -> Vec<Vec<BigInt>> {
        data.iter()
            .map(|r| r.iter().map(|&v| bi(v)).collect())
            .collect()
    }

    fn matmul(a: &[Vec<BigInt>], b: &[Vec<BigInt>]) -> Vec<Vec<BigInt>> {
        let n = b[0].len();
        a.iter()
            .map(|row| {
                (0..n)
                    .map(|j| row.iter().zip(b).map(|(x, brow)| x * &brow[j]).sum())
                    .collect()
            })
            .collect()
    }

    fn det(a: &[Vec<BigInt>]) -> BigInt {
        // Fraction-free Bareiss on a copy.
        let n = a.len();
        let mut m: Vec<Vec<BigInt>> = a.to_vec();
        let mut sign = BigInt::one();
        let mut prev = BigInt::one();
        for k in 0..n {
            if m[k][k].is_zero() {
                let Some(p) = (k + 1..n).find(|&i| !m[i][k].is_zero()) else {
                    return BigInt::zero();
                };
                m.swap(k, p);
                sign = -sign;
            }
            for i in k + 1..n {
                for j in k + 1..n {
                    let v = (&m[i][j] * &m[k][k] - &m[i][k] * &m[k][j]) / &prev;
                    m[i][j] = v;
                }
            }
            prev = m[k][k].clone();
        }
        sign * m[n - 1][n - 1].clone()
    }

    #[test]
    fn row_hnf_known_answer() {
        let a = rows(&[&[2, 4, 4], &[-6, 6, 12], &[10, -4, -16]]);
        let r = row_hnf(&a);
        assert_eq!(r.h, rows(&[&[2, 4, 4], &[0, 6, 0], &[0, 0, 12]]));
        assert_eq!(matmul(&r.u, &a), r.h);
        assert_eq!(det(&r.u).abs(), bi(1));
    }

    #[test]
    fn row_hnf_rank_deficient_and_negative() {
        let a = rows(&[&[1, 2, 3], &[-2, -4, -6], &[0, 1, 1]]);
        let r = row_hnf(&a);
        assert_eq!(r.pivots, vec![0, 1]);
        assert_eq!(r.h[2], vec![bi(0), bi(0), bi(0)]);
        assert_eq!(matmul(&r.u, &a), r.h);
    }

    #[test]
    fn smith_known_answer() {
        let a = rows(&[&[12, 6, 4], &[3, 9, 6], &[2, 16, 14]]);
        let (s, u, v) = smith(&a);
        assert_eq!(s, rows(&[&[1, 0, 0], &[0, 10, 0], &[0, 0, 30]]));
        assert_eq!(matmul(&matmul(&u, &a), &v), s);
        assert_eq!(det(&u).abs(), bi(1));
        assert_eq!(det(&v).abs(), bi(1));
    }

    #[test]
    fn smith_needs_divisibility_fix() {
        // diag(2, 3) is not in SNF: gcd = 1, so S must be diag(1, 6).
        let a = rows(&[&[2, 0], &[0, 3]]);
        let (s, u, v) = smith(&a);
        assert_eq!(s, rows(&[&[1, 0], &[0, 6]]));
        assert_eq!(matmul(&matmul(&u, &a), &v), s);
    }

    #[test]
    fn column_hnf_matches_sympy() {
        let a = rows(&[&[12, 6, 4], &[3, 9, 6], &[2, 16, 14]]);
        assert_eq!(
            column_hnf_rows(&a),
            rows(&[&[10, 0, 2], &[0, 15, 3], &[0, 0, 2]])
        );
    }
}
