//! Tests written alongside the 0.28 changes (`tests/v27/*.rs`): the
//! statistics bug hunt continued — every bug found in `symplex::stats`
//! pinned against an oracle (scipy, statsmodels, mpmath; each value cites
//! its call) — and the known bugs outside statistics (the integrator's
//! self-check, the logarithmic part's remainder sequence, radical
//! canonicalisation, load-robust timing tests, panics on caller input,
//! binder-aware substitution, the evaluator's error bounds).

#[path = "v27/v27_api.rs"]
mod v27_api;
#[path = "v27/v27_binders.rs"]
mod v27_binders;
#[path = "v27/v27_evalf.rs"]
mod v27_evalf;
#[path = "v27/v27_integrate.rs"]
mod v27_integrate;
#[path = "v27/v27_perf.rs"]
mod v27_perf;
#[path = "v27/v27_stats.rs"]
mod v27_stats;
#[path = "v27/v27_subs_matrix.rs"]
mod v27_subs_matrix;
#[path = "v27/v27_tails.rs"]
mod v27_tails;
