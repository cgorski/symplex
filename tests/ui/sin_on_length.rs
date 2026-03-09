use symplex::prelude::*;
use symplex::units::*;

fn main() {
    let ctx = Context::new();
    let l = Length::symbol(&ctx, "l");
    let _bad = l.sin();
}
