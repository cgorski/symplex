use symplex::prelude::*;
use symplex::units::*;

fn main() {
    let ctx = Context::new();
    let e = Energy::symbol(&ctx, "e");
    let t = Torque::symbol(&ctx, "t");
    let _bad = &e + &t;
}
