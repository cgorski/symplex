pub(crate) mod algebraic;
pub(crate) mod dense;
pub mod generic;
pub mod groebner;
pub mod multipoly;
pub(crate) mod polybridge;
pub mod polysys;
pub mod ratfn;
pub(crate) mod roots;
pub(crate) mod sturm;
pub mod traits;

// Re-export the type alias and ℚ-specific free function.
// Poly is now `type Poly = GenPoly<Ratio<BigInt>>` defined in dense.rs.
pub(crate) use dense::Poly;
pub(crate) use dense::lagrange_interpolate_rational;

/// Maximum coefficient magnitude for rational root divisor enumeration.
pub(crate) const MAX_DIVISOR_COEFFICIENT: u64 = 1_000_000_000;
/// Maximum number of candidate rational roots to test (product of divisor counts).
pub(crate) const MAX_DIVISOR_COMBINATIONS: usize = 500;
/// Maximum trial factor degree for Kronecker's method.
pub(crate) const MAX_KRONECKER_DEGREE: usize = 6;
/// Maximum number of Kronecker evaluation-point combinations.
pub(crate) const MAX_KRONECKER_COMBINATIONS: usize = 100_000;
