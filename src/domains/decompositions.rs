//! Named results of matrix decompositions and normal forms.
//!
//! Every factorisation here used to come back as a tuple of two or three
//! matrices — `(Q, R)`, `(P, D)`, `(H, U)`, `(S, U, V)` — where nothing but
//! memory said which position was which, and a transposition compiled
//! silently.  These structs name each factor and document the identity it
//! satisfies, at zero runtime cost.  See CONTRIBUTING.md, "Tuples versus
//! structs".
//!
//! The structs are generic over the matrix type so that the symbolic
//! [`Matrix`](crate::matrix::Matrix) and the exact
//! [`ZMatrix`](crate::matrix::ZMatrix) / [`QMatrix`](crate::matrix::QMatrix)
//! share one vocabulary: `Qr<Matrix>`, `HermiteNormalForm<ZMatrix>`, ….
//!
//! # Examples
//!
//! ```
//! use symplex::prelude::*;
//!
//! let ctx = Context::new();
//! let a = matrix![ctx, [4, 2], [2, 3]];
//! let ldl = a.ldl().unwrap();
//! assert_eq!((&(&ldl.l * &ldl.d) * &ldl.l.transpose()).eval(), a);
//!
//! let Qr { q, r } = matrix![ctx, [1, 1], [0, 1]].qr().unwrap();
//! assert_eq!((&q * &r).simplify(), matrix![ctx, [1, 1], [0, 1]]);
//! ```

/// QR decomposition `A = Q·R`: `Q` has orthonormal columns, `R` is upper
/// triangular.
#[derive(Clone, Debug, PartialEq)]
pub struct Qr<M> {
    /// `Q` (`m × n`), orthonormal columns.
    pub q: M,
    /// `R` (`n × n`), upper triangular.
    pub r: M,
}

/// LDLᵀ decomposition `A = L·D·Lᵀ` of a symmetric matrix: `L` unit lower
/// triangular, `D` diagonal.
#[derive(Clone, Debug, PartialEq)]
pub struct Ldl<M> {
    /// `L`, unit lower triangular.
    pub l: M,
    /// `D`, diagonal.
    pub d: M,
}

/// LU decomposition with partial pivoting `P·A = L·U`: `L` unit lower
/// triangular, `U` upper triangular, and `perm` the row permutation —
/// row `i` of `P·A` is row `perm[i]` of `A`.
#[derive(Clone, Debug, PartialEq)]
pub struct Lu<M> {
    /// `L`, unit lower triangular.
    pub l: M,
    /// `U`, upper triangular.
    pub u: M,
    /// Row permutation: `perm[i]` is the original index of row `i` of `P·A`.
    pub perm: Vec<usize>,
}

/// Eigendecomposition `A = P·D·P⁻¹`: the columns of `P` are eigenvectors,
/// `D` is diagonal with the eigenvalues in the same order.
#[derive(Clone, Debug, PartialEq)]
pub struct Diagonalization<M> {
    /// `P`, invertible, eigenvectors as columns.
    pub p: M,
    /// `D`, diagonal of eigenvalues.
    pub d: M,
}

/// Jordan normal form `A = P·J·P⁻¹`: `J` is block diagonal with Jordan
/// blocks `J_k(λ)` (eigenvalue on the diagonal, ones on the superdiagonal),
/// `P` holds the (generalized) eigenvectors.
#[derive(Clone, Debug, PartialEq)]
pub struct JordanForm<M> {
    /// `P`, invertible, (generalized) eigenvectors as columns.
    pub p: M,
    /// `J`, block diagonal of Jordan blocks.
    pub j: M,
}

/// Upper Hessenberg form by similarity: `H = P⁻¹·A·P` (equivalently
/// `A·P = P·H`) with `h_ij = 0` for `i > j + 1`.
#[derive(Clone, Debug, PartialEq)]
pub struct Hessenberg<M> {
    /// `H`, upper Hessenberg.
    pub h: M,
    /// `P`, the invertible similarity transform.
    pub p: M,
}

/// Full-rank factorisation `A = C·F` with `r = rank A`: `C` (`m × r`) is
/// made of the pivot columns of `A`, `F` (`r × n`) of the nonzero rows of
/// `rref(A)`.
#[derive(Clone, Debug, PartialEq)]
pub struct RankDecomposition<M> {
    /// `C` (`m × r`), the pivot columns of `A`.
    pub c: M,
    /// `F` (`r × n`), the nonzero rows of `rref(A)`.
    pub f: M,
}

/// Row-style Hermite normal form `H = U·A` with `U` unimodular
/// (`det U = ±1`).
#[derive(Clone, Debug, PartialEq)]
pub struct HermiteNormalForm<M> {
    /// `H`, the Hermite normal form of `A`.
    pub h: M,
    /// `U`, unimodular, with `H = U·A`.
    pub u: M,
}

/// Smith normal form `S = U·A·V` with `U`, `V` unimodular
/// (`det U = det V = ±1`) and `S = diag(d₁, …, dᵣ, 0, …)`, `dᵢ | dᵢ₊₁`.
#[derive(Clone, Debug, PartialEq)]
pub struct SmithNormalForm<M> {
    /// `S`, the diagonal Smith normal form of `A`.
    pub s: M,
    /// `U`, unimodular row transform.
    pub u: M,
    /// `V`, unimodular column transform.
    pub v: M,
}

/// LLL reduction of the lattice basis formed by the rows of `A`:
/// `reduced = transform·A` with `transform` unimodular (`det = ±1`), so
/// both span the same lattice.
#[derive(Clone, Debug, PartialEq)]
pub struct LllReduction<M> {
    /// The LLL-reduced basis, one lattice vector per row.
    pub reduced: M,
    /// `T`, unimodular, with `reduced = T·A`.
    pub transform: M,
}
