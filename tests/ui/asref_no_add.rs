use symplex::prelude::*;
use symplex::units::*;

fn main() {
    let f = Force::symbol("f");
    let m = Mass::symbol("m");
    // AsRef<Ex> should NOT make this compile — still different types
    let _bad = &f + &m;
}
