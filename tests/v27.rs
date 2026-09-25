//! Tests written alongside the 0.28 changes (`tests/v27/*.rs`): the
//! statistics bug hunt continued — every bug found in `symplex::stats`
//! pinned against an oracle (scipy, statsmodels, mpmath; each value cites
//! its call) — and the known bugs outside statistics (the integrator's
//! self-check, the logarithmic part's remainder sequence, radical
//! canonicalisation, load-robust timing tests).

#[path = "v27/v27_integrate.rs"]
mod v27_integrate;
#[path = "v27/v27_perf.rs"]
mod v27_perf;
#[path = "v27/v27_tails.rs"]
mod v27_tails;
