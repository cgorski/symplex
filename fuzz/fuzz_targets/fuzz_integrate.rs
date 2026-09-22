//! Antiderivatives are checked by differentiation.
//!
//! For an elementary expression `f` (see `common::expr`), when
//! `integrate(f)` returns a closed form `F` (no unevaluated node), `F′`
//! must equal `f` at every sample point where both are finite reals
//! (30-digit evaluation, relative 1e-9).  Differentiation is the simple,
//! trusted direction.  The 0.21 audit found sixteen wrong antiderivatives
//! this way.  The integrator self-verifies its riskiest routes since 0.22;
//! this target keeps every route honest.
#![no_main]

#[path = "common/mod.rs"]
mod common;

use libfuzzer_sys::fuzz_target;
use symplex::prelude::*;

fuzz_target!(|data: &[u8]| {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let mut b = common::Bytes::new(data);
    // Depth 3: integration of depth-4 trees is dominated by slow Risch
    // refusals and adds little.
    let f = common::expr(&ctx, &x, &mut b, 3);
    let big_f = f.integrate(&x);
    if big_f.has_unevaluated() {
        return;
    }
    let df = big_f.diff(&x);
    if let Some((v, fa, fb)) = common::close_at(&ctx, &f, &df, &x) {
        panic!(
            "wrong antiderivative: ∫ {f} dx = {big_f}, but F′ = {df}; at x = {v}: f = {fa:e}, F′ = {fb:e}"
        );
    }
});
