//! Tests written alongside the 0.28 changes (`tests/v27/*.rs`): the far
//! tails of the symbolic distributions keep their digits (`Family::sf`,
//! `Family::cdf_lower`, `Distribution::sf`), each value checked against an
//! oracle (scipy, mpmath; each value cites its call).

#[path = "v27/v27_tails.rs"]
mod v27_tails;
