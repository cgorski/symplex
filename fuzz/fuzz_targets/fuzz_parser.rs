#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Convert bytes to a string (skip invalid UTF-8)
    if let Ok(input) = std::str::from_utf8(data) {
        let ctx = symplex::Context::new();
        // parse should never panic, only return Err
        let _ = symplex::parse::parse(&ctx, input);
    }
});
