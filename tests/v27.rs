//! Tests written alongside the 0.28 changes (`tests/v27/*.rs`): the
//! statistics bug hunt continued — every bug found in `symplex::stats`
//! pinned against an oracle (scipy, statsmodels, mpmath; each value cites
//! its call).

#[path = "v27/v27_tails.rs"]
mod v27_tails;
