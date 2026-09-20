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
use crate::base::errors::SymplexError;
use num_bigint::BigInt;
use num_rational::Ratio;
use std::fmt;
use tracing::{debug, trace, warn};

// Re-export codegen option types so users can access them from the public
// `symplex::matrix` module (the `codegen` module itself is pub(crate)).
pub use crate::domains::exact_matrix::{
    ExactMatrix, ExactScalar, LLL_DEFAULT_DELTA, QMatrix, ZMatrix,
};
pub use crate::output::codegen::{CodegenOptions, MathBackend, Precision};

use crate::domains::exact_matrix::PERMANENT_MAX_DIM;

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
        other => other,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Three-valued scalar helpers (shared with matrix_decomp / quaternion)
// ═══════════════════════════════════════════════════════════════════════════

/// Three-valued zero test for a scalar expression.
///
/// Layers: structural zero → assumption system → `eval().simplify()` →
/// numeric evaluation for constants → `together().simplify()` (common
/// denominator, catches rational-function identities such as
/// `1 − 2sin²θ/(2cosθ + 2) − cosθ`).  Returns `None` when the value cannot
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
    // Numeric constants (including `RootOf`, whose bound variable
    // `is_constant` would report as free): decide by evaluation.
    if let Ok((re, im)) = s.eval_complex64() {
        if re == 0.0 && im == 0.0 {
            return Some(true);
        }
        if re.abs() > 1e-12 || im.abs() > 1e-12 {
            return Some(false);
        }
        return None;
    }
    // Symbolic with fractions: bring over a common denominator and retry.
    // Skipped when radicals are present: `together` then routes through a
    // canonicalisation path that is not yet robust for products of radicals.
    if !has_radical(&s) {
        let t = s.together();
        if t != s {
            let t = t.simplify();
            if t.is_zero_structural() {
                return Some(true);
            }
            if let Some(b) = t.is_zero() {
                return Some(b);
            }
        }
    }
    None
}

/// Does `e` contain a power with a numeric non-integer exponent (a radical)?
fn has_radical(e: &Ex) -> bool {
    use crate::base::node::ExprNode;
    let inner = e.inner.read();
    let arena = &inner.arena;
    let mut stack = vec![e.raw_id()];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        if let ExprNode::Pow(_, exp) = arena.node(id)
            && let Some(r) = arena.as_num(*exp)
            && !r.is_integer()
        {
            return true;
        }
        stack.extend(arena.node(id).children());
    }
    false
}

/// Rewrite `sin(−c)` → `−sin(c)` and `cos(−c)` → `cos(c)` for *numeric*
/// literals `c > 0`.  The canonicaliser applies these parity rules to
/// symbolic negations (`sin(−x)`) but not to negative number literals, so
/// `½e^{i} + ½e^{−i}` would otherwise stay `½cos(1) + ½cos(−1) + …`.
fn fix_trig_parity(e: &Ex) -> Ex {
    use crate::base::node::ExprNode;
    use num_traits::Signed;
    // Pass 1 (read lock): find sin/cos nodes with a negative numeric argument.
    let targets: Vec<(
        crate::base::node::ExprId,
        bool,
        num_rational::Ratio<num_bigint::BigInt>,
    )> = {
        let inner = e.inner.read();
        let arena = &inner.arena;
        let mut out = Vec::new();
        let mut stack = vec![e.raw_id()];
        let mut seen = rustc_hash::FxHashSet::default();
        while let Some(id) = stack.pop() {
            if !seen.insert(id) {
                continue;
            }
            let (is_sin, arg) = match arena.node(id) {
                ExprNode::Sin(a) => (true, *a),
                ExprNode::Cos(a) => (false, *a),
                _ => {
                    stack.extend(arena.node(id).children());
                    continue;
                }
            };
            if let Some(r) = arena.as_num(arg)
                && r.is_negative()
            {
                out.push((id, is_sin, r.clone()));
            }
        }
        out
    };
    if targets.is_empty() {
        return e.clone();
    }
    // Pass 2 (no lock held): build the replacements, then substitute by id.
    let ctx = e.context();
    let map: rustc_hash::FxHashMap<_, Ex> = targets
        .into_iter()
        .map(|(id, is_sin, r)| {
            let pos = ctx.from_ratio(-r);
            (id, if is_sin { -pos.sin() } else { pos.cos() })
        })
        .collect();
    e.replace(|v| map.get(&v.id()).cloned())
}

/// Does `e` contain a `RootOf` node?
fn has_root_of(e: &Ex) -> bool {
    use crate::base::node::ExprNode;
    let inner = e.inner.read();
    let arena = &inner.arena;
    let mut stack = vec![e.raw_id()];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        if matches!(arena.node(id), ExprNode::RootOf(..)) {
            return true;
        }
        stack.extend(arena.node(id).children());
    }
    false
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
    if let Ok(v) = s.eval_f64() {
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
    if let Ok(v) = s.eval_f64() {
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

/// Square root of `e` with rational radicands rationalised: for a positive
/// rational `p/q` returns `√(p·q) / q` (so `√(1/2)` becomes `√2/2`, matching
/// the form the evaluator produces for `cos(π/4)`), otherwise `e.sqrt()`.
///
/// Used wherever a norm is divided out (Gram–Schmidt, quaternion axis
/// extraction) so that products of the resulting radicals cancel
/// structurally instead of needing `simplify()`.
pub(crate) fn sqrt_rationalized(e: &Ex) -> Ex {
    if let Some(r) = e.as_rational()
        && r.numer().sign() == num_bigint::Sign::Plus
    {
        let ctx = e.context();
        let p = ctx.from_bigint(r.numer().clone());
        let q = ctx.from_bigint(r.denom().clone());
        return &(&p * &q).sqrt() / &q;
    }
    e.sqrt()
}

// ═══════════════════════════════════════════════════════════════════════════
// Expression-swell budget
// ═══════════════════════════════════════════════════════════════════════════

/// Upper bound on the total expression-tree size (nodes, counted without
/// sharing) of the operands and intermediate results of the symbolic
/// algorithms that can swell: [`Matrix::det`], [`Matrix::inv`],
/// [`Matrix::solve`], [`Matrix::diagonalize`], [`Matrix::jordan_form`],
/// [`Matrix::matrix_exp`] and [`Matrix::qr`].
///
/// When the budget is exceeded these return
/// [`SymplexError::ComputationFailed`] whose reason starts with
/// `"expression swell"` instead of running for an unbounded time.  The
/// value is calibrated so that everything below it finishes in well under
/// a minute: a fully symbolic 7×7 determinant (5040 terms, ≈ 40 000 nodes)
/// passes, an 8×8 one (≈ 360 000 nodes, many minutes) is rejected.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
///
/// let ctx = Context::new();
/// // A tiny DAG whose *tree* is enormous: eₙ₊₁ = sin(eₙ) + cos(eₙ).
/// let mut e = ctx.symbol("x");
/// for _ in 0..16 {
///     e = &e.sin() + &e.cos();
/// }
/// let m = Matrix::new(vec![vec![e.clone(), ctx.int(1)], vec![ctx.int(1), e]]).unwrap();
/// let err = m.inv().unwrap_err();
/// assert!(err.to_string().contains("expression swell"), "{err}");
/// ```
pub const EXPRESSION_BUDGET: usize = 100_000;

/// Tree size of `e` (nodes, without sharing), stopping as soon as `cap` is
/// exceeded.  Iterative, so deep expressions cannot overflow the stack.
pub(crate) fn tree_size_capped(e: &Ex, cap: usize) -> usize {
    let inner = e.inner.read();
    let arena = &inner.arena;
    let mut stack = vec![e.raw_id()];
    let mut count = 0usize;
    while let Some(id) = stack.pop() {
        count += 1;
        if count > cap {
            break;
        }
        stack.extend(arena.node(id).children());
    }
    count
}

/// Return `Err(ComputationFailed)` if the combined tree size of `entries`
/// exceeds [`EXPRESSION_BUDGET`].
pub(crate) fn budget_check<'a>(
    entries: impl IntoIterator<Item = &'a Ex>,
    operation: &'static str,
) -> Result<(), SymplexError> {
    let mut total = 0usize;
    for e in entries {
        total += tree_size_capped(e, EXPRESSION_BUDGET - total);
        if total > EXPRESSION_BUDGET {
            return Err(failed(
                operation,
                format!(
                    "expression swell: intermediate result exceeds the budget of \
                     {EXPRESSION_BUDGET} expression nodes; simplify the input or \
                     substitute numeric values first"
                ),
            ));
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
        Ok(Matrix { rows, nrows, ncols })
    }

    /// Build a matrix from validated parts without re-checking.
    #[inline]
    pub(crate) fn from_rows_unchecked(rows: Vec<Vec<Ex>>) -> Self {
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
        Ok(Matrix::from_fn(size, size, |i, j| {
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

    /// Return `Err(ComputationFailed)` if the entries together exceed
    /// [`EXPRESSION_BUDGET`] tree nodes.
    fn check_budget(&self, operation: &'static str) -> Result<(), SymplexError> {
        budget_check(self.iter(), operation)
    }

    /// The matrix as a [`QMatrix`] if every entry is a rational literal
    /// (the exact fast path of `rref`, `det`, `inv`, `solve`, …).
    pub(crate) fn as_qmatrix(&self) -> Option<QMatrix> {
        let rows = self.to_rational_rows()?;
        QMatrix::new(rows).ok()
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
    fn berkowitz_monic(&self, operation: &'static str) -> Result<Vec<Ex>, SymplexError> {
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
    /// - [`SymplexError::ComputationFailed`] if the determinant is zero or
    ///   the entries / intermediate results exceed [`EXPRESSION_BUDGET`]
    ///   ("expression swell").
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
        let d = self.det().map_err(|e| reop(e, "inv"))?;
        if ex_is_zero(&d) == Some(true) {
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
    ///   (no unique solution) or the entries / solution exceed
    ///   [`EXPRESSION_BUDGET`].
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
            return qa.solve(&qb).map(|x| x.to_matrix(&self.ctx()));
        }

        let augmented = Matrix::hstack(&[self, b])?;
        augmented.check_budget("solve")?;
        let (rref_mat, pivots) = augmented.rref();
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
    /// (quadratic) formula is used.
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
        let mut pairs = if n <= 2 && symbolic {
            low_degree_roots(&coeffs)
        } else {
            eigvals_with_multiplicity(&cp, lam)
        };
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
    ///   diagonalizable, the eigenvalue solver could not find all
    ///   eigenvalues, or the eigenvectors exceed [`EXPRESSION_BUDGET`].
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
        p.check_budget("diagonalize")?;
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
    /// let (p, j) = m.jordan_form().unwrap();
    /// assert_eq!((&(&p * &j) * &p.inv().unwrap()).simplify(), m);
    /// ```
    pub fn jordan_form(&self) -> Result<(Matrix, Matrix), SymplexError> {
        self.require_square("jordan_form")?;
        self.check_budget("jordan_form")?;
        let n = self.nrows;
        debug!(n, "jordan_form: computing for {}×{} matrix", n, n);
        let eye = Matrix::identity(&self.ctx(), n);

        let eigvs = self.eigenvects().map_err(|e| reop(e, "jordan_form"))?;

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
        debug!(
            "matrix_exp: computing for {}×{} matrix",
            self.nrows, self.ncols
        );
        let n = self.nrows;
        let ctx = self.ctx();

        let (p, j) = self.jordan_form().map_err(|e| match e {
            SymplexError::ComputationFailed { reason, .. }
                if reason.starts_with("expression swell") =>
            {
                failed("matrix_exp", reason)
            }
            e => failed(
                "matrix_exp",
                format!(
                    "Jordan form unavailable ({e}); use exp_series(order) for a truncated approximation"
                ),
            ),
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

        let p_inv = p.inv().map_err(|e| match e {
            SymplexError::ComputationFailed { reason, .. }
                if reason.starts_with("expression swell") =>
            {
                failed("matrix_exp", reason)
            }
            _ => failed(
                "matrix_exp",
                "eigenvector matrix is singular (internal inconsistency)",
            ),
        })?;
        let result = p.matmul(&exp_j)?.matmul(&p_inv)?;
        result.check_budget("matrix_exp")?;
        let i_unit = ctx.i_unit();
        Ok(result.map(|e| {
            // `simplify` cannot do anything useful with `RootOf` values but
            // is very slow on them; constant folding is all they need.
            if has_root_of(e) {
                return e.eval();
            }
            let s = e.simplify();
            // Complex-conjugate eigenvalue pairs: Euler's formula turns
            // `½e^{iωt} + ½e^{−iωt}` into `cos(ωt)` (valid for any complex
            // argument, so no realness assumption is needed).
            if s.contains(&i_unit) {
                fix_trig_parity(&s.rewrite_as_trig())
                    .expand()
                    .eval()
                    .simplify()
            } else {
                s
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
    /// matrices the rank is the *structural* rank of [`rref`](Self::rref)
    /// (symbolic pivots are treated as non-zero) and entries are treated
    /// as real (`Aᵀ`, not `Aᴴ`).
    ///
    /// # Errors
    ///
    /// [`SymplexError::ComputationFailed`] if a symbolic `AᵀA` is singular
    /// although the structural pivot search found full column rank
    /// (simplify the entries first), or on expression swell beyond
    /// [`EXPRESSION_BUDGET`].
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
    /// assert_eq!(&b.pinv().unwrap() * &b, Matrix::identity(&ctx, 2));
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
            move |e: SymplexError| match e {
                SymplexError::ComputationFailed { reason, .. }
                    if reason.starts_with("expression swell") =>
                {
                    failed("pinv", reason)
                }
                _ => failed(
                    "pinv",
                    format!(
                        "{what} is singular: the columns are linearly dependent in a way the \
                         structural pivot search did not detect; simplify the entries first"
                    ),
                ),
            }
        };
        let (r, pivots) = self.rref();
        if pivots.is_empty() {
            return Ok(Matrix::zeros(&ctx, self.ncols, self.nrows));
        }
        let at = self.transpose();
        if pivots.len() == self.ncols {
            let ata = at.matmul(self)?;
            let ata_inv = ata.inv().map_err(inv_err("AᵀA"))?;
            return ata_inv.matmul(&at);
        }
        let c = self.select_cols(&pivots)?;
        let f = r.select_rows(&(0..pivots.len()).collect::<Vec<_>>())?;
        let ft = f.transpose();
        let ct = c.transpose();
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
    /// let (c, f) = a.rank_decomposition().unwrap();
    /// assert_eq!(c, matrix![ctx, [1, 2], [4, 5], [7, 8]]);
    /// assert_eq!(f, matrix![ctx, [1, 0, -1], [0, 1, 2]]);
    /// assert_eq!(&c * &f, a);
    /// ```
    pub fn rank_decomposition(&self) -> Result<(Matrix, Matrix), SymplexError> {
        let ctx = self.ctx();
        if let Some(q) = self.as_qmatrix() {
            let (c, f) = q
                .rank_decomposition()
                .map_err(|e| reop(e, "rank_decomposition"))?;
            return Ok((c.to_matrix(&ctx), f.to_matrix(&ctx)));
        }
        let (r, pivots) = self.rref();
        if pivots.is_empty() {
            return Err(failed(
                "rank_decomposition",
                "matrix is zero (rank 0); the factors C (m×0) and F (0×n) would be empty",
            ));
        }
        let c = self.select_cols(&pivots)?;
        let f = r.select_rows(&(0..pivots.len()).collect::<Vec<_>>())?;
        Ok((c, f))
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
    /// polynomial.  Entries are treated as real (`Aᵀ`, not `Aᴴ`).
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
            let g = if m >= n { qt.matmul(&q) } else { q.matmul(&qt) };
            g.map_err(|e| reop(e, "singular_values"))?.to_matrix(&ctx)
        } else {
            let at = self.transpose();
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
        let eig = if gram.is_diagonal() == Some(true) {
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
        let sv = self
            .singular_values()
            .map_err(|e| reop(e, "condition_number"))?;
        let ctx = self.ctx();
        let numeric = sv.iter().all(|v| v.eval_f64().is_ok());
        let (max, min) = match (sv.first(), sv.last()) {
            (Some(first), Some(last)) if numeric => (first.clone(), last.clone()),
            _ => (
                Ex::max_of(&ctx, sv.iter().cloned()),
                Ex::min_of(&ctx, sv.iter().cloned()),
            ),
        };
        if ex_is_zero(&min) == Some(true) {
            return Err(failed(
                "condition_number",
                "matrix is singular (smallest singular value is 0), condition number is infinite",
            ));
        }
        Ok(&max / &min)
    }

    /// Upper Hessenberg form by Gaussian similarity transforms: `(H, P)`
    /// with `H = P⁻¹ A P` and `h_ij = 0` for `i > j + 1`.  SymPy:
    /// `Matrix.upper_hessenberg_decomposition()` (Householder reflections,
    /// hence radicals; this variant uses eliminations and stays in the
    /// field of the entries).
    ///
    /// Column `k` is cleared below the sub-diagonal with the first usable
    /// entry as pivot — moved into row `k + 1` by a symmetric row/column
    /// swap when needed — followed by `row_j −= f·row_{k+1}` and the
    /// compensating `col_{k+1} += f·col_j`.  Rational matrices run exactly
    /// on [`QMatrix`].  For symbolic matrices a pivot is chosen among the
    /// entries that are provably non-zero, falling back to one whose value
    /// cannot be decided — such a pivot is *assumed* non-zero, so the
    /// identity `A·P = P·H` holds generically (wherever those pivots do
    /// not vanish), exactly as for [`lu`](Self::lu) and
    /// [`rref`](Self::rref).  Symbolic entries are simplified as they are
    /// produced.
    ///
    /// # Errors
    ///
    /// - [`SymplexError::InvalidArgument`] if the matrix is not square.
    /// - [`SymplexError::ComputationFailed`] on expression swell beyond
    ///   [`EXPRESSION_BUDGET`].
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let a = matrix![ctx, [1, 2, 3], [4, 5, 6], [7, 8, 10]];
    /// let (h, p) = a.hessenberg().unwrap();
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
    pub fn hessenberg(&self) -> Result<(Matrix, Matrix), SymplexError> {
        self.require_square("hessenberg")?;
        let ctx = self.ctx();
        if let Some(q) = self.as_qmatrix() {
            let (h, p) = q.hessenberg().map_err(|e| reop(e, "hessenberg"))?;
            return Ok((h.to_matrix(&ctx), p.to_matrix(&ctx)));
        }
        self.check_budget("hessenberg")?;
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
            // Prefer a provably non-zero pivot; accept an undecidable one.
            let mut pivot = None;
            let mut fallback = None;
            for (i, row) in h.iter().enumerate().skip(k + 1) {
                match ex_is_zero(&row[k]) {
                    Some(false) => {
                        pivot = Some(i);
                        break;
                    }
                    None if fallback.is_none() => fallback = Some(i),
                    _ => {}
                }
            }
            let Some(piv) = pivot.or(fallback) else {
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
                if ex_is_zero(&h[j][k]) == Some(true) {
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
        Ok((
            Matrix::from_rows_unchecked(h),
            Matrix::from_rows_unchecked(p),
        ))
    }
}

/// Sort `vals` in descending numeric order if *every* entry evaluates to
/// an `f64`; otherwise leave the order unchanged (a partial comparator
/// would not be a total order).
fn sort_descending_numeric(vals: &mut [Ex]) {
    let keys: Option<Vec<f64>> = vals.iter().map(|v| v.eval_f64().ok()).collect();
    let Some(keys) = keys else { return };
    let mut order: Vec<usize> = (0..vals.len()).collect();
    order.sort_by(|&a, &b| keys[b].total_cmp(&keys[a]));
    let sorted: Vec<Ex> = order.iter().map(|&i| vals[i].clone()).collect();
    vals.clone_from_slice(&sorted);
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
            let sq = quadratic_sqrt(&disc);
            let r1 = (&(-c1 + &sq) / &denom).eval();
            let r2 = (&(-c1 - &sq) / &denom).eval();
            vec![(r1, 1), (r2, 1)]
        }
        _ => Vec::new(),
    }
}

/// A square root of `disc` for the quadratic formula.
///
/// Any `r` with `r² = disc` yields the same *set* `{(−b ± r)/2a}`, so we
/// are free to pick the friendliest one.  When `±disc` is a perfect square
/// (`4ω²`, `(a−d)²`, …) the evaluator returns `2·|ω|`; dropping the
/// absolute value (the other sign is also a square root) gives `2ω`, and
/// for negative discriminants `2·i·ω` instead of `√(−4ω²)`.  The nice form
/// matters downstream: `A − λI` pivots then cancel structurally, which is
/// what makes `eigenvects` / `matrix_exp` of e.g. `[[0, −ω], [ω, 0]]` work
/// without sign assumptions on `ω`.  Falls back to `disc.sqrt()`.
fn quadratic_sqrt(disc: &Ex) -> Ex {
    use crate::base::node::ExprNode;
    let ctx = disc.context();
    let strip_abs = |e: &Ex| {
        e.replace(|v| match v.node() {
            ExprNode::Abs(inner) => Some(e.wrap(*inner)),
            _ => None,
        })
    };
    for (base, factor) in [(disc.clone(), ctx.one()), ((-disc).eval(), ctx.i_unit())] {
        let root = strip_abs(&base.sqrt().simplify());
        if has_radical(&root) {
            continue;
        }
        let check = (&root.powi(2).expand() - &base.expand()).expand();
        if check.is_zero_structural() {
            return (&factor * &root).eval();
        }
    }
    disc.sqrt()
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
        // Rational literals: every zero test agrees, so the fraction-free
        // exact core gives the same (unique) RREF and pivots.
        if let Some(q) = self.as_qmatrix() {
            let (r, pivots) = q.rref();
            return (r.to_matrix(&self.ctx()), pivots);
        }
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
        Ok(Matrix {
            rows: data,
            nrows: self.nrows + rows.nrows,
            ncols: self.ncols,
        })
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
        Ok(Matrix {
            rows: data,
            nrows: self.nrows,
            ncols,
        })
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
    pub fn to_rational_rows(&self) -> Option<Vec<Vec<Ratio<BigInt>>>> {
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
    pub fn from_ratio(ctx: &Context, rows: &[Vec<Ratio<BigInt>>]) -> Result<Matrix, SymplexError> {
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
    /// assert_eq!(Matrix::identity(&ctx, 4).nnz(), 4);
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
    /// matrix, Lovász parameter `δ = num/den`.  See [`ZMatrix::lll`] for
    /// the guarantees and
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
    ///
    /// let ctx = Context::new();
    /// let b = matrix![ctx, [1, 1, 1], [-1, 0, 2], [3, 5, 6]];
    /// assert_eq!(b.lll((3, 4)).unwrap(), matrix![ctx, [0, 1, 0], [1, 0, 1], [-1, 0, 2]]);
    /// ```
    pub fn lll(&self, delta: (i64, i64)) -> Result<Matrix, SymplexError> {
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
        let roots = if (3..=4).contains(&degree) && !radical_form_is_compact(factor) {
            // Irreducible cubic/quartic without a compact radical form: the
            // Cardano/Ferrari expressions (nested complex cube roots) make
            // every downstream step — eigenvectors, P⁻¹, simplify — swell
            // exponentially.  `RootOf` is exact, small and numerically
            // evaluable, so it is the better closed form here.
            debug!(
                degree,
                "eigvals_via_poly_factor: irreducible factor without compact radicals → RootOf"
            );
            (0..degree).map(|i| root_of(&factor_ex, i)).collect()
        } else {
            factor_ex.solve_or_empty(var)
        };
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

/// `RootOf(poly, index)` — the `index`-th root (Aberth ordering: by real
/// part, then imaginary part) of the univariate polynomial `poly`.
fn root_of(poly: &Ex, index: usize) -> Ex {
    let mut inner = poly.inner.write();
    let idx = inner.arena.int(index as i64);
    let id = inner
        .arena
        .intern(crate::base::node::ExprNode::RootOf(poly.raw_id(), idx));
    drop(inner);
    poly.wrap(id)
}

/// Does an irreducible cubic or quartic over ℤ have a *compact* radical
/// form worth returning instead of `RootOf`?
///
/// After the Tschirnhaus shift `x = y − b/(n·a)` the polynomial is
/// "depressed"; if the remaining odd-degree coefficient vanishes the
/// roots are a rational shift plus a single cube root (cubic `y³ + q`)
/// or nested square roots (biquadratic `y⁴ + p·y² + r`).  Everything else
/// needs the full Cardano/Ferrari formulas, whose casus-irreducibilis
/// complex cube roots are enormous and unsimplifiable.
fn radical_form_is_compact(f: &crate::poly::dense::Poly) -> bool {
    use num_bigint::BigInt;
    let Some(n) = f.degree() else { return true };
    let c = |i: usize| f.coeff(i);
    let k = |v: i64| num_rational::Ratio::from_integer(BigInt::from(v));
    match n {
        // y³ + p y + q with p = (3ac − b²)/(3a²): compact iff p = 0.
        3 => &(&c(3) * &c(1)) * &k(3) == &c(2) * &c(2),
        // Linear coefficient of the depressed quartic ∝ b³ − 4abc + 8a²d.
        4 => {
            let (a, b, cc, d) = (c(4), c(3), c(2), c(1));
            let b3 = &(&b * &b) * &b;
            let abc = &(&(&a * &b) * &cc) * &k(4);
            let aad = &(&(&a * &a) * &d) * &k(8);
            &(&b3 - &abc) + &aad == k(0)
        }
        _ => true,
    }
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
/// for constant expressions such as nested radicals or `RootOf` values
/// that `simplify` cannot collapse — numeric evaluation with a tight
/// tolerance.
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
            matches!(e.eval_complex64(), Ok((re, im)) if re.abs() < 1e-10 && im.abs() < 1e-10)
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

/// Pick a vector from `candidates` that lies outside the span of
/// `exclude` (which may itself be linearly dependent or contain
/// duplicates).  A candidate qualifies when appending it raises the rank.
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
    let base_rank = Matrix::hstack(exclude)?.rank_semantic();
    for candidate in candidates {
        let mut cols: Vec<&Matrix> = exclude.to_vec();
        cols.push(candidate);
        let combined = Matrix::hstack(&cols)?;
        if combined.rank_semantic() > base_rank {
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
                let (re, im) = at.get(i, j).eval_complex64().unwrap();
                assert!((re - want).abs() < 1e-12, "({i},{j}) re={re}");
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
