//! Not a fuzz target: prints the expression a fuzz input decodes to, so a
//! crash or timeout artifact can be reproduced as a regression test.
//!
//! ```text
//! cargo +nightly fuzz run print_expr -s none fuzz/artifacts/fuzz_integrate/timeout-…
//! ```
//!
//! prints `depth 4: …` (the `fuzz_simplify` tree, full grammar) and
//! `depth 3: …` (the `fuzz_integrate` tree, elementary grammar).
#![no_main]

#[path = "common/mod.rs"]
mod common;

use libfuzzer_sys::fuzz_target;
use symplex::prelude::*;

fuzz_target!(|data: &[u8]| {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for (depth, grammar) in [(4, common::Grammar::Full), (3, common::Grammar::Elementary)] {
        let mut b = common::Bytes::new(data);
        let f = common::expr_in(&ctx, &x, &mut b, depth, grammar);
        println!("depth {depth}: {f}");
    }
});
