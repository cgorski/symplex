use symplex::prelude::*;
use symplex::units::*;

fn main() {
    let ctx = Context::new();
    let f = Force::symbol(&ctx, "f");
    let m = Mass::symbol(&ctx, "m");
    // AsRef<Ex> should NOT make this compile — still different types
    let _bad = &f + &m;
}
