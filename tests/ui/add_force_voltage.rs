use symplex::prelude::*;
use symplex::units::*;

fn main() {
    let ctx = Context::new();
    let f = Force::symbol(&ctx, "f");
    let v = Voltage::symbol(&ctx, "v");
    let _bad = &f + &v;
}
