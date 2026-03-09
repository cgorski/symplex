use symplex::prelude::*;
use symplex::units::*;

fn main() {
    let ctx = Context::new();
    let m = Mass::symbol(&ctx, "m");
    let a = Acceleration::symbol(&ctx, "a");
    let _bad: Velocity = symplex::dim!(ctx, Force: m * a);
}
