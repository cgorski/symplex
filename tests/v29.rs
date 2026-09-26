//! Tests written alongside the 0.30 changes (`tests/v29/*.rs`): the bug hunt
//! after 0.29 — differential hunters (evalf against itself at two
//! precisions, the calculus routines against numerical oracles), the local
//! fuzz campaign, and the open items of the 0.29 hand-off.  Every reference
//! value cites the oracle call that produced it.

#[path = "v29/v29_calculus.rs"]
mod v29_calculus;
#[path = "v29/v29_evalf.rs"]
mod v29_evalf;
