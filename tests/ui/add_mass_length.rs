use symplex::prelude::*;
use symplex::units::*;

fn main() {
    let m = Mass::symbol("m");
    let l = Length::symbol("l");
    let _bad = &m + &l;
}
