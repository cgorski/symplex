//! Tests written alongside the 0.25 changes (`tests/v24/*.rs`): the rational
//! integrator's logarithmic part (Lazard–Rioboo–Trager with log arguments of
//! every degree, exact real forms of quadratic factors), the memory budgets
//! that keep any integrand or expansion from exhausting memory, panics
//! reachable from user input, and `(a^b)^c = a^(bc)` for integer `c`.

#[path = "v24/v24_canon.rs"]
mod v24_canon;
#[path = "v24/v24_integrate.rs"]
mod v24_integrate;
#[path = "v24/v24_panics.rs"]
mod v24_panics;
