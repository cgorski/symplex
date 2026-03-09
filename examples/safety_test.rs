use symplex::prelude::*;
use symplex::units::*;

fn main() {
    let ctx = Context::new();
    let m = Mass::symbol("m");
    let g = Acceleration::symbol("g");
    let l = Length::symbol("l");
    let v = Velocity::symbol("v");

    // Type-safe: all orderings work, AND dimension is tracked
    let e1 = symplex::dim!(ctx, Energy: m * l * g);
    let e2 = symplex::dim!(ctx, Energy: l * g * m);
    let e3 = symplex::dim!(ctx, Energy: g * l * m);
    println!("m*l*g = {} (all orderings ✓)", e1);
    println!("l*g*m = {}", e2);
    println!("g*l*m = {}", e3);

    // 4 factors: KE = ½mv²
    let ke = symplex::dim!(ctx, Energy: 1/2 * m * v * v);
    println!("½mv² = {}", ke);

    // Division: velocity = length / time
    let t = Time::symbol("t");
    let vel = symplex::dim!(ctx, Velocity: l / t);
    println!("l/t = {}", vel);

    // Negation
    let f = symplex::dim!(ctx, Force: m * g);
    let neg_f = symplex::dim!(ctx, Force: -(m * g));
    println!("F = {}, -F = {}", f, neg_f);

    // Addition of same type
    let f2 = Force::symbol("F2");
    let f_total = symplex::dim!(ctx, Force: f + f2);
    println!("F + F2 = {}", f_total);

    // TYPE SAFETY: wrong dimension → COMPILE ERROR
    // Uncomment to see: "expected Velocity, found Force"
    // let _bad: Velocity = symplex::dim!(ctx, Force: m * g);

    println!("\n✓ dim! macro is order-independent AND type-safe!");
}
