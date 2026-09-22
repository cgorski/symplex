//! Tests written alongside the 0.22 fixes (`tests/v21/*.rs`): the
//! regressions found by the 0.21 differential audit (simplify, series,
//! apart, integrate, numdist, solve) and the coverage gaps it exposed.

#[path = "v21/v21_apart.rs"]
mod v21_apart;
#[path = "v21/v21_coverage.rs"]
mod v21_coverage;
#[path = "v21/v21_integrate.rs"]
mod v21_integrate;
#[path = "v21/v21_numdist.rs"]
mod v21_numdist;
#[path = "v21/v21_series.rs"]
mod v21_series;
#[path = "v21/v21_simplify.rs"]
mod v21_simplify;
#[path = "v21/v21_solve.rs"]
mod v21_solve;
