//! Feature tests written alongside the 0.17 APIs (`tests/v17/*.rs`, one
//! module per track). Run one module with e.g.
//! `cargo test --test v17 v17_anova::`.

#[path = "v17/v17_anova.rs"]
mod v17_anova;
#[path = "v17/v17_audit_a.rs"]
mod v17_audit_a;
#[path = "v17/v17_audit_b.rs"]
mod v17_audit_b;
#[path = "v17/v17_audit_c.rs"]
mod v17_audit_c;
#[path = "v17/v17_exact_intervals.rs"]
mod v17_exact_intervals;
#[path = "v17/v17_pvalues.rs"]
mod v17_pvalues;
