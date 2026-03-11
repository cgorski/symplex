use symplex::prelude::*;
fn main() {
    let ctx = Context::new();
    let sqrt5 = ctx.int(5).sqrt();
    println!("(√5)² = {}", sqrt5.powi(2));
    println!("(√5)² eval = {}", sqrt5.powi(2).eval());
    println!("(2^(1/3))^3 = {}", ctx.int(2).cbrt().powi(3));
    println!("(3^(1/4))^2 = {}", ctx.int(3).nthroot(4).powi(2));
    // φ² - φ - 1
    let phi = (ctx.int(1) + &sqrt5) / 2;
    let test = (phi.powi(2).expand().eval() - &phi - 1).eval();
    println!("φ²-φ-1 = {}", test);
    println!("simplified = {}", test.simplify());
}
