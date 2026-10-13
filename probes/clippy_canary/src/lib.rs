//! Clippy canary for the root `clippy.toml` (found by clippy's upward
//! search from this directory; there is deliberately no `clippy.toml` here).
//!
//! Every call below is listed under `disallowed-methods`.  CI runs
//! `cargo clippy --manifest-path probes/clippy_canary/Cargo.toml -- -D
//! clippy::disallowed_methods` and expects it to FAIL, with one error per
//! method.  If an astro-float release moves `BigFloat` so that a configured
//! path stops resolving, clippy stops reporting that method, in symplex as
//! here, and this crate starts passing: the CI step then fails instead.
//!
//! The function is never executed (`2.5²` through `BigFloat::pow` would not
//! return); it only has to be linted.

use astro_float::{BigFloat, Consts, Radix, RoundingMode};

/// One call of each disallowed method.
pub fn disallowed_calls(cc: &mut Consts) -> usize {
    let p = 128;
    let rm = RoundingMode::ToEven;
    let x = BigFloat::from_f64(2.5, p);
    let two = BigFloat::from_f64(2.0, p);
    let power = x.pow(&two, p, rm, cc);
    let log = x.log(&two, p, rm, cc);
    let digits = x.convert_to_radix(Radix::Dec, rm, cc);
    usize::from(power.is_nan()) + usize::from(log.is_nan()) + usize::from(digits.is_err())
}
