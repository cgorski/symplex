//! Tests written alongside the 0.29 changes (`tests/v28/*.rs`): the known
//! bugs resolved after 0.28 — the statistics bug hunt continued (every bug
//! found in `symplex::stats` pinned against an oracle: scipy, statsmodels,
//! mpmath; each value cites its call), the integrator's self-check, the
//! logarithmic part's remainder sequence, radical canonicalisation,
//! load-robust timing tests, panics on caller input, binder-aware
//! substitution and the evaluator's error bounds.

#[path = "v28/v28_api.rs"]
mod v28_api;
#[path = "v28/v28_audit_2a.rs"]
mod v28_audit_2a;
#[path = "v28/v28_audit_2b.rs"]
mod v28_audit_2b;
#[path = "v28/v28_audit_3a.rs"]
mod v28_audit_3a;
#[path = "v28/v28_audit_3b1.rs"]
mod v28_audit_3b1;
#[path = "v28/v28_audit_3b2.rs"]
mod v28_audit_3b2;
#[path = "v28/v28_audit_4.rs"]
mod v28_audit_4;
#[path = "v28/v28_binders.rs"]
mod v28_binders;
#[path = "v28/v28_evalf.rs"]
mod v28_evalf;
#[path = "v28/v28_evalf_cuts.rs"]
mod v28_evalf_cuts;
#[path = "v28/v28_integrate.rs"]
mod v28_integrate;
#[path = "v28/v28_perf.rs"]
mod v28_perf;
#[path = "v28/v28_quantiles.rs"]
mod v28_quantiles;
#[path = "v28/v28_semantics.rs"]
mod v28_semantics;
#[path = "v28/v28_stats.rs"]
mod v28_stats;
#[path = "v28/v28_subs_matrix.rs"]
mod v28_subs_matrix;
