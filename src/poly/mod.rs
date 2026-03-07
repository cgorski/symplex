pub(crate) mod dense;
pub mod multipoly;
pub mod groebner;
pub(crate) mod polybridge;
pub mod polysys;
pub(crate) mod sturm;

// Re-export commonly used items so crate::poly::Poly still works
pub(crate) use dense::Poly;
pub(crate) use dense::lagrange_interpolate_rational;
