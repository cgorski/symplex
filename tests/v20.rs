//! Feature tests written alongside the 0.21 APIs (`tests/v20/*.rs`).

#[path = "v20/v20_base.rs"]
mod v20_base;
#[path = "v20/v20_dense.rs"]
mod v20_dense;
#[path = "v20/v20_evalf.rs"]
mod v20_evalf;
#[path = "v20/v20_fits.rs"]
mod v20_fits;
#[path = "v20/v20_libfn.rs"]
mod v20_libfn;
#[path = "v20/v20_poly_kernel.rs"]
mod v20_poly_kernel;
#[path = "v20/v20_matrix_tier.rs"]
mod v20_matrix_tier;
#[path = "v20/v20_fp_poly.rs"]
mod v20_fp_poly;
#[path = "v20/v20_numdist.rs"]
mod v20_numdist;
