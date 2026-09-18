use symplex::prelude::*;
#[test]
fn probe8() {
    let ctx = Context::new();
    for s in [
        "-x^2",
        "x^2^3",
        "E^(I*pi) + 1",
        "E^(I*pi)",
        "sin(pi)",
        "2*pi",
        "E^x",
        "pi",
        "E",
        "I",
    ] {
        let e = symplex::parse::parse(&ctx, s).unwrap();
        println!(
            "{s} -> {e} | eval {} | f64 {:?}",
            e.eval(),
            e.eval_complex64()
        );
    }
    let x = ctx.symbol("x");
    println!(
        "-x^2 at 3 = {:?}",
        symplex::parse::parse(&ctx, "-x^2")
            .unwrap()
            .subs_i64(&x, 3)
            .eval()
    );
    println!(
        "x^2^3 at 2 = {:?}",
        symplex::parse::parse(&ctx, "x^2^3")
            .unwrap()
            .subs_i64(&x, 2)
            .eval()
    );
}
