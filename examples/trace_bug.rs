use symplex::prelude::*;
fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("symplex::transforms::integrate=debug,symplex::calculus::risch=debug")
        .with_target(false)
        .init();
    
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &x / (&x.powi(4) + &x.powi(2) + 1);
    let anti = e.integrate(&x);
    println!("\nRESULT: {anti}");
    println!("unevaluated: {}", anti.has_unevaluated());
}
