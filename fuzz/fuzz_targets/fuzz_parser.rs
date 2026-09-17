#![no_main]

use libfuzzer_sys::fuzz_target;
use symplex::prelude::*;

fuzz_target!(|data: &[u8]| {
    // Convert bytes to a string (skip invalid UTF-8)
    if let Ok(input) = std::str::from_utf8(data) {
        let ctx = Context::new();
        // parse should never panic, only return Err
        if let Ok(expr) = symplex::parse::parse(&ctx, input) {
            // Anything that parses must also display and re-parse without panicking.
            let shown = format!("{expr}");
            let _ = symplex::parse::parse(&ctx, &shown);
            let _ = expr.to_latex();
        }
    }
});
