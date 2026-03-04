use symplex::prelude::*;
fn main() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let inf = ctx.infinity();
    let one = ctx.int(1);

    let cases: Vec<(&str, Ex, &str)> = vec![
        ("x/(x+1)", &x / &(&x + &one), "1"),
        ("1/x", &one / &x, "0"),
        ("(2x+1)/(x+1)", &(&(&x * 2) + &one) / &(&x + &one), "2"),
        ("(3x²+1)/(x²+1)", &(&(&x.powi(2) * 3) + &one) / &(&x.powi(2) + &one), "3"),
        ("x/(x²+1)", &x / &(&x.powi(2) + &one), "0"),
        ("exp(-x)", (-&x).exp(), "0"),
        ("ln(x)/x", &x.ln() / &x, "0"),
        ("x*exp(-x)", &x * &(-&x).exp(), "0"),
    ];

    println!("=== LIMITS AT INFINITY ===\n");
    let mut pass = 0;
    let mut fail = 0;
    for (desc, expr, expected) in &cases {
        match expr.limit(&x, &inf) {
            Ok(r) => {
                let got = format!("{r}");
                let ok = got == *expected;
                if ok { pass += 1; } else { fail += 1; }
                println!("{} lim({}, x→∞) = {}  (expected: {})",
                    if ok { "✅" } else { "❌" }, desc, got, expected);
            }
            Err(e) => {
                fail += 1;
                println!("❌ lim({}, x→∞) = ERROR: {}  (expected: {})", desc, e, expected);
            }
        }
    }
    println!("\nResults: {pass} passed, {fail} failed out of {}", cases.len());
}
