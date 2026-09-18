//! Scratch exploration (will be deleted).
use symplex::prelude::*;

#[test]
#[ignore]
fn trace_one() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("symplex=debug")
        .with_test_writer()
        .try_init();
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &x * (-&x).exp();
    let r = e.try_limit(&x, &ctx.infinity());
    println!("RESULT: {r:?}");
}
