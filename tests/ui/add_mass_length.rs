use symplex::prelude::*;
use symplex::units::*;

fn main() {
    let ctx = Context::new();
    let m = Mass::symbol(&ctx, "m");
    let l = Length::symbol(&ctx, "l");
    let _bad = &m + &l;
}
