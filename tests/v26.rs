//! Tests written alongside the 0.27 changes (`tests/v26/*.rs`): the first
//! round of the statistics bug hunt — `stats::hypothesis` and
//! `stats::numdist` differentially tested against scipy / statsmodels /
//! mpmath, every bug found pinned against its oracle (each value cites the
//! call that produced it).

#[path = "v26/v26_hypothesis.rs"]
mod v26_hypothesis;
#[path = "v26/v26_numdist.rs"]
mod v26_numdist;
