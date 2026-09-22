//! Not a fuzz target: prints the expression a fuzz input decodes to, so a
//! crash or timeout artifact can be reproduced as a regression test.
//!
//! ```text
//! cargo +nightly fuzz run print_expr -s none fuzz/artifacts/fuzz_integrate/timeout-…
//! ```
//!
//! prints `depth 4: …` (the `fuzz_simplify` tree) and `depth 3: …` (the
//! `fuzz_integrate` tree).
#![no_main]

#[path = "common/mod.rs"]
mod common;

use libfuzzer_sys::fuzz_target;
use symplex::prelude::*;

fuzz_target!(|data: &[u8]| {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for depth in [4, 3] {
        let mut b = common::Bytes::new(data);
        let f = common::expr(&ctx, &x, &mut b, depth);
        println!("depth {depth}: {f}");
    }
});
